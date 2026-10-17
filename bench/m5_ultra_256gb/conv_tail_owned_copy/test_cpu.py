"""CPU interface/source checks only; no real MLX, device, socket or model."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('owned_tail_candidate', HERE / 'candidate.py')
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)


class Array:
    def __init__(self, dtype='bf16', shape=(3, 10240), label='input', views=None):
        self.dtype, self.shape, self.ndim = dtype, shape, len(shape)
        self.label, self.views = label, [] if views is None else views

    def view(self, dtype):
        self.views.append((self.label, self.dtype, dtype))
        return Array(dtype, self.shape, self.label, self.views)


class FakeRuntime:
    bfloat16, uint16 = 'bf16', 'u16'

    def __init__(self):
        self.copies, self.evaluated = [], []

    def contiguous(self, value):
        self.copies.append(value)
        return Array(value.dtype, value.shape, 'compact', value.views)

    def eval(self, value):
        self.evaluated.append(value)


class TailReview(unittest.TestCase):
    def test_off_never_touches_runtime_or_array(self):
        value = object()
        self.assertIs(c.own_conv_tail(object(), value), value)

    def test_uint16_transfer_and_materialization_are_separate(self):
        for materialize in (False, True):
            mx, tail = FakeRuntime(), Array()
            with patch.object(c, 'supported_backend', return_value=True):
                result = c.own_conv_tail(mx, tail, enabled=True, materialize=materialize)
            self.assertEqual(tail.views, [('input', 'bf16', 'u16'), ('compact', 'u16', 'bf16')])
            self.assertEqual((result.dtype, result.shape), ('bf16', (3, 10240)))
            self.assertEqual(len(mx.copies), 1)
            self.assertEqual(mx.copies[0].dtype, 'u16')
            self.assertEqual(mx.evaluated, [result] if materialize else [])
            self.assertEqual(tail.label, 'input')

    def test_outside_domain_is_identity_without_copy_or_eval(self):
        for tail in (Array('f32'), Array(shape=(4, 10240)), Array(shape=(3, 1024)), Array(shape=(1, 3, 10240))):
            mx = FakeRuntime()
            with patch.object(c, 'supported_backend', side_effect=AssertionError('shape should reject first')):
                self.assertIs(c.own_conv_tail(mx, tail, enabled=True, materialize=True), tail)
            self.assertEqual((mx.copies, mx.evaluated), ([], []))
        mx, tail = FakeRuntime(), Array()
        with patch.object(c, 'supported_backend', return_value=False):
            self.assertIs(c.own_conv_tail(mx, tail, enabled=True, materialize=True), tail)
        self.assertEqual((mx.copies, mx.evaluated), ([], []))

    def test_explicit_flags(self):
        for options in ({'enabled': 1}, {'materialize': 'yes'}, {'materialize': True}):
            with self.assertRaises((TypeError, ValueError)):
                c.own_conv_tail(object(), object(), **options)

    def test_backend_guard(self):
        mx = NS(gpu='gpu', default_device=lambda: 'gpu',
                device_info=lambda: {'device_name': 'Apple M5 Ultra'},
                metal=NS(device_info=lambda: {'architecture': 'applegpu_g17d'}))
        with patch.object(c, 'version', return_value='0.32.3'):
            self.assertTrue(c.supported_backend(mx))
            mx.default_device = lambda: 'cpu'
            self.assertFalse(c.supported_backend(mx))
        with patch.object(c, 'version', return_value='0.32.4'):
            self.assertFalse(c.supported_backend(mx))

    def test_pinned_source_proves_excess_buffer_copy_not_donation(self):
        manifest = json.loads((HERE / 'source-pins.json').read_text())
        for name, item in manifest['sources'].items():
            self.assertEqual(hashlib.sha256((HERE / item['copy']).read_bytes()).hexdigest(), item['sha256'], name)
        source = (HERE / 'sources/mlx/backend/gpu/primitives.cpp').read_text()
        body = source.split('void Contiguous::eval_gpu', 1)[1].split('void Copy::eval_gpu', 1)[0]
        self.assertIn('extra_bytes = 16384', body)
        self.assertIn('in.buffer_size() <= out.nbytes() + extra_bytes', body)
        self.assertIn('copy_gpu(in, out, CopyType::General)', body)
        copy = (HERE / 'sources/mlx/backend/common/copy.h').read_text()
        self.assertIn('} else {\n    out.set_data(mallocfn(out.nbytes()));\n    return false;', copy)
        # Arithmetic from the fixed physical layout, not a fake device result.
        for length in (1, 19, 886, 935, 978, 1024):
            self.assertGreater((length + 3) * 10240 * 2, 3 * 10240 * 2 + 16384)

    def test_import_and_default_off_with_forbidden_imports_and_socket(self):
        code = '''
import importlib.abc, importlib.util, socket, sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self, name, *args):
  if name.split('.')[0] in {'mlx','numpy','torch','tutorial_code'}: raise AssertionError(name)
sys.meta_path.insert(0, Block())
socket.socket=lambda *a,**k: (_ for _ in ()).throw(AssertionError('socket'))
s=importlib.util.spec_from_file_location('candidate',sys.argv[1]);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
x=object();assert m.own_conv_tail(object(),x) is x
'''
        result = subprocess.run([sys.executable, '-I', '-B', '-c', code, str(HERE / 'candidate.py')],
                                capture_output=True, text=True, timeout=5)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
