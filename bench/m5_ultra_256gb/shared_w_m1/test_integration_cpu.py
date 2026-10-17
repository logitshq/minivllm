"""Structural bridge checks with a fake backend; no MLX or numerical claim."""
import ast
import gc
import hashlib
import importlib.util
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import weakref

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('shared_w_bridge_cpu', HERE / 'integration.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class Tensor:
    def __init__(self, shape, dtype, parent=None, row=None):
        self.shape, self.dtype, self.parent, self.row = shape, dtype, parent, row
    @property
    def ndim(self): return len(self.shape)
    def __getitem__(self, sl):
        assert type(sl) is slice and sl.step is None and sl.stop == sl.start + 1
        return Tensor((1, self.shape[1]), self.dtype, self, sl.start)


class Backend:
    gpu, bfloat16, uint32, uint8 = 'gpu', 'bf16', 'u32', 'u8'
    def __init__(self):
        self.qmm, self.cat, self.builds, self.launches = [], [], [], []
        self.device, self.qmm_error, self.kernel_error = self.gpu, None, None
        self.fast = NS(metal_kernel=self.build)
        self.metal = NS(device_info=lambda: {'architecture': 'applegpu_g17d'})
    def default_device(self): return self.device
    def device_info(self): return {'device_name': 'Apple M5 Ultra'}
    def quantized_matmul(self, x, w, s, b, **kwargs):
        self.qmm.append((x, w, s, b, kwargs))
        if self.qmm_error: raise self.qmm_error
        return Tensor((x.shape[0], w.shape[0]), self.bfloat16)
    def concatenate(self, values, *, axis):
        self.cat.append((values, axis))
        return Tensor((sum(v.shape[0] for v in values), values[0].shape[1]), values[0].dtype)
    def build(self, **kwargs):
        self.builds.append(kwargs)
        def call(**kw):
            self.launches.append(kw)
            if self.kernel_error: raise self.kernel_error
            return [Tensor(kw['output_shapes'][0], kw['output_dtypes'][0])]
        return call


def op(n=48, k=5120):
    return NS(weight=Tensor((n, k // 8), 'u32'), scales=Tensor((n, k // 16), 'u8'), q=dict(m.Q))


def prepare(mx, operator, **overrides):
    kwargs = dict(enabled=True, weight_strides=(640, 1), scale_strides=(320, 1),
                  runtime_version='0.32.3', device_name='Apple M5 Ultra', architecture='applegpu_g17d')
    kwargs.update(overrides)
    return m.prepare_qualification(mx, operator, **kwargs)


class BridgeTests(unittest.TestCase):
    def test_off_and_blocked_native_are_inert(self):
        class Poison:
            def __getattribute__(self, key): raise AssertionError(key)
        with patch.object(Path, 'read_bytes', side_effect=AssertionError('read')):
            self.assertIsNone(m.prepare_qualification(Poison(), Poison()))
            self.assertIsNone(m.prepare_native_route(Poison(), Poison()))
            with self.assertRaisesRegex(ValueError, 'unqualified'):
                m.prepare_native_route(Poison(), Poison(), enabled=True)
        for value in (0, 1, None, 'yes'):
            for f in (m.prepare_qualification, m.prepare_native_route):
                with self.subTest(value=value, fn=f), self.assertRaises(TypeError):
                    f(None, None, enabled=value)

    def test_only_previously_tested_shapes_runtime_and_strides(self):
        for n in (48, 1024, 10240):
            scope = prepare(Backend(), op(n)); scope.close()
        for n, k in ((6144, 5120), (5120, 6144), (17408, 5120), (248320, 5120)):
            with self.subTest(n=n, k=k), self.assertRaisesRegex(ValueError, 'three'):
                prepare(Backend(), op(n, k))
        for kwargs in ({'runtime_version': '0.32.4'}, {'device_name': 'Apple M4 Max'},
                       {'weight_strides': (641, 1)}, {'scale_strides': (320, True)}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                prepare(Backend(), op(), **kwargs)

    def test_native_uses_exact_multirow_expression_original_storage(self):
        mx, operator = Backend(), op()
        class_before = dict(vars(type(operator)))
        with prepare(mx, operator) as scope:
            x = Tensor((5, 5120), mx.bfloat16)
            y = scope.native_multirow(x, observed_x_strides=(5120, 1))
            self.assertEqual(y.shape, (5, 48))
            self.assertEqual(len(mx.qmm), 1)
            args = mx.qmm[0]
            self.assertIs(args[0], x); self.assertIs(args[1], operator.weight)
            self.assertIs(args[2], operator.scales); self.assertIsNone(args[3])
            self.assertEqual(args[4], dict(transpose=True, **m.Q))
            self.assertEqual(mx.cat + mx.builds + mx.launches, [])
        self.assertEqual(set(vars(operator)), {'weight', 'scales', 'q'})
        self.assertEqual(dict(vars(type(operator))), class_before)

    def test_row_oracle_explicit_slices_then_same_output_concat(self):
        mx, operator = Backend(), op(1024)
        with prepare(mx, operator) as scope:
            x = Tensor((4, 5120), mx.bfloat16)
            y = scope.row_m1(x, observed_x_strides=(5120, 1))
            self.assertEqual(y.shape, (4, 1024)); self.assertEqual(len(mx.qmm), 4)
            for row, args in enumerate(mx.qmm):
                self.assertIs(args[0].parent, x); self.assertEqual(args[0].row, row)
                self.assertEqual(args[0].shape, (1, 5120))
                self.assertIs(args[1], operator.weight); self.assertIs(args[2], operator.scales)
                self.assertEqual(args[4], dict(transpose=True, **m.Q))
            self.assertEqual(mx.cat[0][1], 0)
            self.assertEqual((scope.row_m1_calls, scope.row_m1_launches), (1, 4))
            self.assertEqual(mx.builds, [])

    def test_preparation_reuse_and_exit_no_operand_or_output_retention(self):
        mx, operator = Backend(), op(10240)
        with prepare(mx, operator) as scope:
            self.assertEqual(mx.builds, [])
            for t in m.ROWS:
                x = Tensor((t, 5120), mx.bfloat16)
                y = scope.shared(x, observed_x_strides=(5120, 1))
                self.assertEqual(y.shape, (t, 10240))
                self.assertIs(mx.launches[-1]['inputs'][0], operator.weight)
            self.assertEqual(len(mx.builds), 1)
            xr, yr = weakref.ref(x), weakref.ref(y)
            mx.launches.clear(); del x, y; gc.collect()
            self.assertIsNone(xr()); self.assertIsNone(yr())
        self.assertEqual(scope.counts(), dict(native_multirow=0, row_m1=0, row_m1_launches=0,
                                             shared=3, kernel_callable_builds=1))
        self.assertIsNone(scope.op); self.assertIsNone(scope.prepared); self.assertIsNone(scope.mx)
        scope.close()
        with self.assertRaises(RuntimeError): scope.__enter__()

    def test_metadata_change_noncanonical_or_unqualified_rows_fail_before_call(self):
        for mode in ('weight', 'scales', 'q', 'dtype', 'bias', 'device'):
            mx, operator = Backend(), op()
            with prepare(mx, operator) as scope:
                if mode == 'weight': operator.weight = Tensor((48, 640), 'u32')
                elif mode == 'scales': operator.scales = Tensor((48, 320), 'u8')
                elif mode == 'q': operator.q['bits'] = 8
                elif mode == 'dtype': operator.weight.dtype = 'f32'
                elif mode == 'bias': operator.biases = object()
                else: mx.device = 'cpu'
                for fn in (scope.native_multirow, scope.row_m1, scope.shared):
                    with self.subTest(mode=mode, fn=fn), self.assertRaises(RuntimeError):
                        fn(Tensor((2, 5120), 'bf16'), observed_x_strides=(5120, 1))
                self.assertEqual(mx.qmm + mx.builds, [])
        mx = Backend()
        with prepare(mx, op()) as scope:
            for shape, dtype, strides in (((3, 5120), 'bf16', (5120, 1)),
                    ((1, 5120), 'bf16', (5120, 1)), ((5, 5120), 'f32', (5120, 1)),
                    ((2, 5120), 'bf16', (5120, 2))):
                with self.subTest(shape=shape, strides=strides), self.assertRaises(ValueError):
                    scope.native_multirow(Tensor(shape, dtype), observed_x_strides=strides)

    def test_error_does_not_fallback_or_hide_and_scope_closes(self):
        mx = Backend(); scope = prepare(mx, op())
        error = RuntimeError('original native failure')
        mx.qmm_error = error
        try:
            with scope:
                scope.native_multirow(Tensor((2, 5120), 'bf16'), observed_x_strides=(5120, 1))
        except RuntimeError as caught:
            self.assertIs(caught, error)
        else: self.fail('error hidden')
        self.assertTrue(scope.closed); self.assertEqual(mx.builds, [])
        self.assertEqual(scope.counts()['native_multirow'], 0)

    def test_inspected_native_current_and_frozen_sources_match(self):
        root = HERE.parents[2]
        for prefix in ('tutorial_code/miniengine',
                '.artifacts/m5-ultra-256gb/external-spec-http-v1-inputs/frozen-native/tutorial_code/miniengine'):
            for relative, sha in m.INSPECTED_SOURCE_PINS.items():
                raw = (root / prefix / relative).read_bytes()
                self.assertEqual(hashlib.sha256(raw).hexdigest(), sha)
            tree = ast.parse((root / prefix / 'model/layers.py').read_text())
            linear = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Linear')
            call = next(n for n in linear.body if isinstance(n, ast.FunctionDef) and n.name == '__call__')
            qmm = call.body[-1].value
            self.assertEqual(ast.unparse(qmm.func), 'mx.quantized_matmul')
            self.assertEqual(ast.unparse(qmm.args[0]), 'x')
            self.assertFalse(any(isinstance(n, (ast.For, ast.ListComp)) for n in ast.walk(call)))

    def test_import_inert_no_runtime_network_or_adoption(self):
        code = '''
import importlib.abc,runpy,socket,sys
class Deny(importlib.abc.MetaPathFinder):
 def find_spec(self,name,*args):
  if name.split('.')[0] in {'mlx','numpy','torch'}: raise AssertionError(name)
sys.meta_path.insert(0,Deny());socket.socket=lambda *a,**kw:(_ for _ in ()).throw(AssertionError('socket'))
m=runpy.run_path(sys.argv[1]);assert m['prepare_qualification'](None,None) is None
assert m['prepare_native_route'](None,None) is None
assert m['contract']()['native_route_qualified'] is False
'''
        result = subprocess.run([sys.executable, '-I', '-B', '-c', code, str(HERE / 'integration.py')],
                                capture_output=True, text=True, timeout=5)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__': unittest.main()
