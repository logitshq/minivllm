"""Independent CPU review of dispatch, source transformation and launch geometry.

These checks do not compile Metal, import MLX, run a model or establish numeric
equivalence. Actual partial BF16 bits and original-QMM reduction remain GPU gates.
"""
import hashlib
import ast
import importlib.util
import math
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
INCLUDE = ROOT / '.venv/lib/python3.13/site-packages/mlx/include'
spec = importlib.util.spec_from_file_location('same_split_review_candidate', HERE / 'candidate.py')
c = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = c
spec.loader.exec_module(c)


class IndependentDispatchReview(unittest.TestCase):
    def test_fixed_original_partitions_not_naive_target_groups(self):
        # Independent values derived from official v0.32.3 quantized.cpp.
        # MLP down starts with 3 groups but must reduce to 2 for K/32 divisibility.
        shapes = ((5120, 6144, 2, 2560), (17408, 5120, 2, 8704), (6144, 5120, 3, 2048))
        self.assertEqual(c.NORMAL_M, (19,))
        for m in (19,):
            for k, n, splits, partition in shapes:
                with self.subTest(M=m, K=k, N=n):
                    g = c.geometry(m, k, n)
                    self.assertEqual((g['M'], g['K'], g['N'], g['split_k'], g['K_eff']),
                                     (m, k, n, splits, partition))
                    self.assertEqual(splits * partition, k)
                    self.assertEqual(partition % 64, 0)

    def test_unqualified_shapes_and_dispatch_boundaries_fall_back(self):
        for m in (0, 1, 11, 12, 13, 18, 20, 31, 32, 33, 128, 512):
            for k, n in ((5120, 6144), (17408, 5120), (6144, 5120)):
                with self.subTest(M=m, K=k, N=n):
                    self.assertIsNone(c.geometry(m, k, n))
        for k, n in ((5120, 48), (5120, 1024), (4096, 5120), (5120, 6143), (5120, 6145), (6143, 5120)):
            with self.subTest(K=k, N=n):
                self.assertIsNone(c.geometry(19, k, n))

    def test_thread_grid_encodes_all_partitions_and_every_output_tile(self):
        for m in (19,):
            for k, n in ((5120, 6144), (17408, 5120), (6144, 5120)):
                g = c.geometry(m, k, n)
                for kind, expected_tile in (('reference', (32, 32, 32)), ('candidate', (32, 64, 64))):
                    with self.subTest(M=m, K=k, N=n, kind=kind):
                        launch = g[kind]
                        self.assertEqual((launch['bm'], launch['bn'], launch['bk']), expected_tile)
                        group = tuple(launch['threadgroup'])
                        self.assertEqual(math.prod(group), 128)
                        # fast.metal_kernel grid is threads, not threadgroups.
                        expected_groups = ((n + launch['bn'] - 1) // launch['bn'],
                                           (m + launch['bm'] - 1) // launch['bm'], g['split_k'])
                        self.assertEqual(tuple(launch['grid']), tuple(a * b for a, b in zip(expected_groups, group)))
                        # An independent tiling oracle proves no row/column is
                        # omitted or duplicated, including M19 partial rows.
                        rows = [row for tm in range(expected_groups[1])
                                for row in range(tm * launch['bm'], min(m, (tm + 1) * launch['bm']))]
                        cols = [col for tn in range(expected_groups[0])
                                for col in range(tn * launch['bn'], min(n, (tn + 1) * launch['bn']))]
                        self.assertEqual(rows, list(range(m)))
                        self.assertEqual(cols, list(range(n)))

    def test_nax_helper_only_changes_name_partition_argument_and_loop(self):
        path = INCLUDE / 'mlx/backend/metal/kernels/fp_quantized_nax.h'
        original = path.read_text()
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
                         '8a712870dc17493ddf05daa706481d52867b29e9141d2ecb76dccf47f04c8b42')
        prefix = c.source_prefix(original)
        changed = c.nax_partial_helper(prefix)
        self.assertIn('e025_fp_qmm_t_partial_nax', changed)
        self.assertEqual(changed.count('for (int k = 0; k < K_eff; k += BK)'), 1)
        reverted = changed.replace('e025_fp_qmm_t_partial_nax', 'fp_qmm_t_impl')
        reverted = re.sub(r'\n[ \t]*const constant int& K_eff,', '', reverted)
        reverted = reverted.replace('for (int k = 0; k < K_eff; k += BK)', 'for (int k = 0; k < K; k += BK)')
        self.assertEqual(reverted, prefix)
        for fragment in ('const int K_w = K * bytes_per_pack / pack_factor;',
                         'const int K_g = K / group_size;', 'x += tm * K;',
                         'Atile.load(x + kk1, K);', 'Atile.load_safe(x + kk1, K, short2(SK, sgp_sm));'):
            self.assertIn(fragment, changed)
        self.assertNotIn('[[kernel]]', prefix)

    def test_reference_helper_and_nax_libraries_stay_separate(self):
        g = c.geometry(19, 6144, 5120)
        ref, new = [c.build_source(kind, g, INCLUDE) for kind in ('reference', 'candidate')]
        self.assertIn('fp_qmm_t_impl', ref['header'])
        self.assertNotIn('e025_fp_qmm_t_partial_nax', ref['header'])
        self.assertIn('e025_fp_qmm_t_partial_nax', new['header'])
        self.assertNotIn('[[kernel]]', ref['header'])
        self.assertNotIn('[[kernel]]', new['header'])
        for source in (ref, new):
            self.assertNotRegex(source['header'], r'#\s*include\s*"mlx/')
            self.assertTrue(source['pins'])

    def test_rendered_partition_pointer_units_preserve_full_row_strides(self):
        def arithmetic(text, variables):
            text = text.strip().replace('tid.z', 'partition').replace('/', '//')
            tree = ast.parse(text, mode='eval')
            allowed = (ast.Expression, ast.BinOp, ast.Add, ast.Sub, ast.Mult, ast.FloorDiv,
                       ast.Name, ast.Constant, ast.Load)
            self.assertTrue(all(isinstance(n, allowed) for n in ast.walk(tree)))
            return eval(compile(tree, '<rendered-pointer-offset>', 'eval'), {'__builtins__': {}}, variables)
        for k, n, split, partition_size in ((5120, 6144, 2, 2560), (17408, 5120, 2, 8704), (6144, 5120, 3, 2048)):
            g = c.geometry(19, k, n)
            for kind in ('reference', 'candidate'):
                source = c.build_source(kind, g, INCLUDE)
                body, header = source['source'], source['header']
                constants = {name: int(value) for name, value in re.findall(r'constant int (E025_\w+) = (\d+);', header)}
                self.assertEqual(constants, {'E025_K': k, 'E025_N': n, 'E025_M': 19, 'E025_K_eff': partition_size})
                self.assertIn('E025_K, E025_N, E025_M, E025_K_eff, tid', body)
                starts = re.search(r'\bk_start\s*=\s*([^;]+);', body)[1]
                expressions = {name: re.search(r'\b' + name + r'\s*=\s*' + base + r'\s*\+\s*([^;]+);', body)[1]
                               for name, base in (('wp', 'weight'), ('sp', 'scales'), ('xp', 'activation'), ('yp', 'partial'))}
                for part in range(split):
                    variables = {**constants, 'partition': part}
                    variables['k_start'] = arithmetic(starts, variables)
                    offsets = {name: arithmetic(expr, variables) for name, expr in expressions.items()}
                    for row in (0, 1, 18):
                        for col in (0, 1, n - 1):
                            for within in (0, 15, 16, 31, 32, 63, partition_size - 1):
                                with self.subTest(K=k, N=n, kind=kind, partition=part, row=row, col=col, k=within):
                                    absolute_k = part * partition_size + within
                                    # Convert emitted uint32 weight offset back
                                    # to logical FP4 elements before comparing.
                                    self.assertEqual(offsets['wp'] * 8 + col * k + within, col * k + absolute_k)
                                    self.assertEqual(offsets['sp'] + col * (k // 16) + within // 16,
                                                     col * (k // 16) + absolute_k // 16)
                                    self.assertEqual(offsets['xp'] + row * k + within, row * k + absolute_k)
                                    self.assertEqual(offsets['yp'] + row * n + col, part * 19 * n + row * n + col)

    def test_fake_runtime_launch_reduction_and_array_replacement(self):
        class Array:
            def __init__(self, shape, dtype):
                self.shape, self.dtype, self.ndim = tuple(shape), dtype, len(shape)
        class Runtime:
            bfloat16, uint32, uint8, gpu = 'bf16', 'u32', 'u8', 'gpu'
            def __init__(self):
                self.compiles, self.launches, self.reductions, self.originals = [], [], [], []
                self.fast = NS(metal_kernel=self.compile)
                self.metal = NS(device_info=lambda: {'architecture': 'applegpu_g17d'})
            def default_device(self):
                return self.gpu
            def device_info(self):
                return {'device_name': 'Apple M5 Ultra'}
            def compile(self, **kw):
                self.compiles.append(kw)
                def launch(**args):
                    self.launches.append(args)
                    return [Array(args['output_shapes'][0], args['output_dtypes'][0])]
                return launch
            def sum(self, x, *, axis):
                self.reductions.append((x, axis))
                return Array(x.shape[1:], x.dtype)
            def quantized_matmul(self, *args, **kwargs):
                self.originals.append((args, kwargs))
                return 'original'
        with patch.object(c, 'version', return_value='0.32.3'):
            mx = Runtime()
            wrapper = c.SameSplitPartial(mx, enabled=True, include_root=INCLUDE)
            x, w, s = Array((19, 17408), mx.bfloat16), Array((5120, 2176), mx.uint32), Array((5120, 1088), mx.uint8)
            for _ in range(2):
                result = wrapper(x, w, s)
                self.assertEqual((result.shape, result.dtype), ((19, 5120), mx.bfloat16))
                self.assertEqual(mx.launches[-1]['inputs'], [x, w, s])
                self.assertEqual((mx.reductions[-1][0].shape, mx.reductions[-1][1]), ((2, 19, 5120), 0))
                w, s = Array(w.shape, w.dtype), Array(s.shape, s.dtype)
            self.assertEqual(len(mx.compiles), 1)  # values are explicit inputs, never captured
            self.assertEqual(mx.compiles[0]['compile_options'], {'math_mode': 'safe'})
            self.assertTrue(mx.compiles[0]['ensure_row_contiguous'])
            self.assertEqual(mx.compiles[0]['input_names'], ['activation', 'weight', 'scales'])
            self.assertEqual(mx.originals, [])
            ref = wrapper.partials(x, w, s, kind='reference')
            self.assertEqual((ref.shape, ref.dtype), ((2, 19, 5120), mx.bfloat16))
            self.assertEqual(len(mx.compiles), 2)
            for invalid in (Array((19, 17408), 'f32'), Array((1, 17408), mx.bfloat16), Array((32, 17408), mx.bfloat16)):
                self.assertEqual(wrapper(invalid, w, s), 'original')
            self.assertEqual(len(mx.compiles), 2)
            mx.device_info = lambda: {'device_name': 'unknown'}
            self.assertEqual(wrapper(x, w, s), 'original')
            with self.assertRaises(ValueError):
                wrapper.partials(x, w, s)

    def test_header_pin_corruption_fails_before_kernel_creation(self):
        import json
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            pins = Path(tmp) / 'bad-pins.json'
            record = json.loads(c.PINS.read_text())
            record['files']['mlx/backend/metal/kernels/steel/gemm/nax.h'] = '0' * 64
            pins.write_text(json.dumps(record))
            with patch.object(c, 'PINS', pins), self.assertRaisesRegex(ValueError, 'SHA mismatch'):
                c.build_source('candidate', c.geometry(19, 6144, 5120), INCLUDE)

    def test_import_and_default_off_require_no_mlx_or_network(self):
        program = '''
import importlib.abc, importlib.util, socket, sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self, name, *args):
  if name.split('.')[0] in {'mlx','numpy','tutorial_code'}: raise AssertionError('blocked import '+name)
sys.meta_path.insert(0, Block())
socket.socket=lambda *a,**k: (_ for _ in ()).throw(AssertionError('socket'))
spec=importlib.util.spec_from_file_location('safe_candidate',sys.argv[1])
c=importlib.util.module_from_spec(spec);sys.modules[spec.name]=c;spec.loader.exec_module(c)
class Fake:
 def quantized_matmul(self,*a,**k): self.args=(a,k);return 'unchanged'
mx=Fake(); wrapper=c.SameSplitPartial(mx)
x,w,s=object(),object(),object()
assert wrapper(x,w,s)=='unchanged'
assert mx.args[0][:3]==(x,w,s)
assert mx.args[1]==dict(transpose=True,group_size=16,bits=4,mode='nvfp4')
'''
        result = subprocess.run([sys.executable, '-I', '-B', '-c', program, str(HERE / 'candidate.py')],
                                capture_output=True, text=True, timeout=5)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_failed_reconstructed_oracle_never_reaches_candidate(self):
        """Exercise failure evidence and sequencing with a tiny non-MLX runtime."""
        import json
        import tempfile
        from types import ModuleType
        spec = importlib.util.spec_from_file_location('same_split_review_screen', HERE / 'screen.py')
        screen = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(screen)
        class Scalar:
            def __init__(self, value): self.value = value
            def item(self): return self.value
        class Array:
            def __init__(self, shape, dtype, tag='original'):
                self.shape, self.dtype, self.tag = tuple(shape), dtype, tag
            def __getitem__(self, sl): return Array((sl.stop, self.shape[1]), self.dtype, self.tag)
            def view(self, dtype): return self
            def __ne__(self, other): return self
            def reshape(self, *args): return self
            def astype(self, dtype): return self
        calls, rows = [], []
        mx = ModuleType('mlx.core')
        mx.bfloat16, mx.uint32, mx.uint8, mx.int32 = 'bf16', 'u32', 'u8', 'i32'
        mx.set_cache_limit = mx.eval = lambda *a: None
        mx.device_info = lambda: {'device_name': 'mock'}
        mx.metal = NS(device_info=lambda: {'architecture': 'mock'})
        mx.random = NS(seed=lambda seed: None)
        mx.load = lambda path: {'activation': Array((128, 16), 'bf16'),
                               'weight': Array((8, 2), 'u32'), 'scales': Array((8, 1), 'u8')}
        mx.isfinite = lambda value: Scalar(True)
        mx.all = lambda value: value
        mx.array_equal = lambda a, b: Scalar(a.tag == b.tag)
        mx.argmax = lambda value: Scalar(0)
        mx.sum = lambda value, axis: Array(value.shape[1:], value.dtype, value.tag)
        mx.save_safetensors = lambda path, arrays: Path(path).write_text(json.dumps(
            {name: [value.shape, value.dtype, value.tag] for name, value in arrays.items()}, sort_keys=True))
        class Wrapper:
            def __init__(self, runtime, enabled=False): self.receipts, self.kernels = [], {}
            def _hardware(self): return True
            def original(self, *args): return Array((19, 8), 'bf16')
            __call__ = original
            def partials(self, x, w, s, kind):
                calls.append(kind)
                if kind != 'reference': raise AssertionError('invalid oracle reached candidate')
                return Array((2, 19, 8), 'bf16', 'incorrect reference')
        mlx = ModuleType('mlx')
        mlx.core = mx
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, {'mlx': mlx, 'mlx.core': mx}), \
                patch.object(screen.c, 'SameSplitPartial', Wrapper), \
                patch.object(screen.c, 'SHAPES', (('mock', 16, 8),)):
            out = Path(tmp)
            with self.assertRaisesRegex(RuntimeError, 'reference-vs-original-QMM'):
                screen.execute(out, {'seed': 1}, {'cases': {'mock': {'path': 'mock-input'}}}, rows.append)
            self.assertEqual(calls, ['reference'])
            failure = json.loads((out / 'first-difference.json').read_text())
            self.assertEqual(failure['status'], 'FAIL')
            self.assertEqual(failure['label'], 'mock-M19/reference-vs-original-QMM')
            self.assertTrue((out / 'first-difference.safetensors').exists())
            self.assertEqual(rows[-1]['event'], 'input-unchanged-on-exit')
            self.assertTrue(rows[-1]['bytes_equal'])
            self.assertTrue((out / 'mock-input-after.safetensors').exists())
            self.assertEqual(json.loads((out / 'kernel-source-receipts.json').read_text()), [])


if __name__ == '__main__':
    unittest.main()
