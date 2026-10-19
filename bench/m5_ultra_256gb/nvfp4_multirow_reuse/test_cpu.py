"""Source/dispatch mechanism tests only: no MLX import, arithmetic or GPU proof."""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

PATH = Path(__file__).with_name('candidate.py')
spec = importlib.util.spec_from_file_location('nvfp4_multirow_reuse_cpu', PATH)
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)


class Forbidden:
    def __getattribute__(self, name):
        raise AssertionError('OFF touched injected backend')


class Fake:
    bfloat16, uint32, uint8, gpu = 'bf16', 'u32', 'u8', 'gpu'

    def __init__(self):
        self.device = self.gpu
        self.architecture, self.name = 'applegpu_g17d', 'Apple M5 Ultra'
        self.builds, self.calls = [], []
        self.error = None
        self.fast = SimpleNamespace(metal_kernel=self.metal_kernel)
        self.metal = SimpleNamespace(device_info=lambda: {'architecture': self.architecture})

    def default_device(self):
        return self.device

    def device_info(self):
        return {'device_name': self.name}

    def metal_kernel(self, **kw):
        self.builds.append(kw)
        def run(**args):
            self.calls.append(args)
            if self.error:
                raise self.error
            return ['candidate-output']
        return run


def arrays(rows=2, K=512, N=4):
    return [SimpleNamespace(shape=s, dtype=d) for s, d in
            (((rows, K), 'bf16'), ((N, K // 8), 'u32'), ((N, K // 16), 'u8'))]


class Tests(unittest.TestCase):
    def build(self, mx):
        with patch('importlib.metadata.version', return_value='0.32.3'):
            return c.build(mx, enabled=True)

    def test_import_and_off_have_no_backend_work(self):
        tree = ast.parse(PATH.read_text())
        self.assertFalse(any(isinstance(n, (ast.Import, ast.ImportFrom, ast.Expr))
                             for n in tree.body[1:]))
        self.assertIsNone(c.build(Forbidden()))
        self.assertIsNone(c.build(Forbidden(), enabled=False))
        for value in (1, None, 'yes'):
            with self.assertRaises(TypeError):
                c.build(Forbidden(), enabled=value)

    def test_unknown_runtime_device_and_architecture_do_not_build(self):
        mx = Fake()
        with patch('importlib.metadata.version', return_value='0.32.4'):
            self.assertIsNone(c.build(mx, enabled=True))
        for attr, bad in (('device', 'cpu'), ('architecture', 'unknown'), ('name', 'other')):
            old = getattr(mx, attr)
            setattr(mx, attr, bad)
            self.assertIsNone(self.build(mx))
            setattr(mx, attr, old)
        self.assertEqual(mx.builds, [])

    def test_all_widths_explicit_contiguous_staging_and_single_dispatch(self):
        mx = Fake()
        fn = self.build(mx)
        self.assertEqual(len(mx.builds), 1)
        options = mx.builds[0]
        self.assertIs(options['ensure_row_contiguous'], True)
        self.assertEqual(options['compile_options'], {'math_mode': 'safe'})
        for width in (2, 4, 8):
            args = arrays(width, 5120, 48)
            args[0].strides = (10240, 2)  # Fake metadata: API must stage, never raw-stride indexing.
            before = [vars(a).copy() for a in args]
            self.assertEqual(fn(*args), 'candidate-output')
            call = mx.calls[-1]
            self.assertEqual(call['template'], [('DT', 'bf16'), ('ROWS', width)])
            self.assertEqual(call['output_shapes'], [(width, 48)])
            self.assertEqual(call['grid'], (32, 48, 1))
            self.assertEqual(call['threadgroup'], (32, 2, 1))
            self.assertTrue(all(a is b for a, b in zip(call['inputs'], args)))
            self.assertEqual([vars(a) for a in args], before)
        self.assertEqual(len(mx.calls), 3)

    def test_rejected_metadata_never_dispatches_or_falls_back(self):
        mx = Fake()
        fn = self.build(mx)
        for rows, K, N in ((1, 512, 4), (3, 512, 4), (16, 512, 4),
                           (2, 256, 4), (2, 0, 4), (2, 512, 6), (2, 512, 0),
                           (8, 65536, 65536)):
            self.assertIsNone(fn(*arrays(rows, K, N)))
        for index, attr, value in ((0, 'dtype', 'f16'), (1, 'dtype', 'u8'),
                                   (2, 'dtype', 'bf16'), (1, 'shape', (4, 63)),
                                   (2, 'shape', (4, 31)), (0, 'shape', (2, 1, 512))):
            args = arrays()
            setattr(args[index], attr, value)
            self.assertIsNone(fn(*args))
        self.assertIsNone(fn(object(), *arrays()[1:]))
        mx.device = 'cpu'
        self.assertIsNone(fn(*arrays()))
        self.assertEqual(mx.calls, [])

    def test_admitted_kernel_error_propagates_without_retry(self):
        mx = Fake()
        fn = self.build(mx)
        mx.error = RuntimeError('injected Metal failure')
        with self.assertRaisesRegex(RuntimeError, 'injected Metal'):
            fn(*arrays())
        self.assertEqual(len(mx.calls), 1)


if __name__ == '__main__':
    unittest.main()
