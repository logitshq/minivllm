"""CPU-only hook routing/lifetime/source tests; no real MLX, model or socket."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import threading
from types import MethodType, ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
FROZEN = REPO / '.artifacts/m5-ultra-256gb/external-spec-http-v1-inputs/frozen-native/tutorial_code/miniengine'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


i = load('e026_integration', HERE / 'integration.py')
c = load('e026_candidate', HERE / 'candidate.py')
VERIFY = i._verify


class Array:
    def __init__(self, shape=(3, 10240), dtype='bf16', tag='tail'):
        self.shape, self.dtype, self.tag, self.ndim = shape, dtype, tag, len(shape)

    def view(self, dtype):
        return Array(self.shape, dtype, self.tag)


class Runtime:
    bfloat16, uint16 = 'bf16', 'u16'

    def __init__(self):
        self.copies, self.evaluated = [], []

    def contiguous(self, value):
        self.copies.append(value)
        return Array(value.shape, value.dtype, 'copied')

    def eval(self, value):
        self.evaluated.append(value)


def fixture():
    module = ModuleType('fake_pinned_gdn')
    module.__file__ = str(FROZEN / 'model/gdn.py')
    module.mx = Runtime()
    module.NS, module.Array = NS, Array
    exec('''
class ForwardContext:
 def __init__(self, mode='prefill', record=False):
  self.mode, self.prefill_shape = mode, None
  self.caches, self.positions, self.lens, self.max_len = [], None, None, 0
  if record: self.record=True
class GatedDeltaNet:
 def __init__(self):
  self.w = Array((10240,4)); self.last_y = None
 def __call__(self,x,ctx,cache):
  if ctx.mode == 'decode': return self._decode(x,cache)
  if ctx.mode == 'prefill_batch': return self._prefill_batch(x,cache)
  return self._prefill(x,cache)
 def _prefill(self,x,cache):
  y,tail = causal_conv1d(x,self.w,None if cache is None else cache.conv_state)
  if cache is not None:
   cache.conv_state,cache.S,cache.offset = tail, 'S-original', cache.offset+x.shape[0]
  self.last_y=y
  return y
 def _decode(self,x,cache): return causal_conv1d(x,self.w)[1]
 def _prefill_batch(self,x,cache): return causal_conv1d(x,self.w)[1]
def causal_conv1d(x,w,state=None):
 calls.append((x,w,state))
 return object(), Array()
calls=[]
''', vars(module))
    layers = {2: module.GatedDeltaNet(), 6: module.GatedDeltaNet()}
    return module, layers


class IntegrationReview(unittest.TestCase):
    def setUp(self):
        self.module, self.layers = fixture()
        self.helper = NS(supported_backend=lambda mx: True, own_conv_tail=c.own_conv_tail)
        self.verify = patch.object(i, '_verify', return_value=self.helper)
        self.verify.start()
        self.backend = patch.object(c, 'supported_backend', return_value=True)
        self.backend.start()
        self.addCleanup(self.verify.stop)
        self.addCleanup(self.backend.stop)
        self.x = Array((19, 10240), tag='input')

    def test_default_off_exact_identity_and_scalar_per_layer_witness(self):
        rows = []
        original_conv = self.module.causal_conv1d
        original_method = self.module.GatedDeltaNet._prefill
        with i.PrefillTailScope(self.module, self.layers, on_call=rows.append) as scope:
            for n, layer in self.layers.items():
                cache = NS(offset=0, conv_state=None, record=False)
                result = layer(self.x, self.module.ForwardContext(), cache)
                self.assertIs(result, layer.last_y)
                self.assertEqual(cache.conv_state.tag, 'tail')
                self.assertEqual((cache.S, cache.offset), ('S-original', 19))
            self.assertIs(self.module.causal_conv1d, original_conv)
            self.assertIs(self.module.GatedDeltaNet._prefill, original_method)
        self.assertEqual([r['layer'] for r in rows], [2, 6])
        self.assertTrue(all(r['eligible'] and not r['enabled'] and not r['replaced'] for r in rows))
        self.assertEqual([r['route'] for r in rows], ['normal-prefill'] * 2)
        json.dumps(rows, allow_nan=False)
        self.assertEqual(self.module.mx.copies, [])
        self.assertEqual(scope.layers, {})
        self.assertTrue(all('_prefill' not in vars(layer) for layer in self.layers.values()))

    def test_copy_and_materialization_independent_y_and_other_state_unchanged(self):
        for materialize in (False, True):
            module, layers = fixture()
            cache, rows = NS(offset=7, conv_state=Array(), record=False), []
            with i.PrefillTailScope(module, layers, enabled=True, materialize=materialize,
                                    on_call=rows.append) as scope:
                result = scope.single_layer(2, self.x, cache)
                self.assertIs(result, layers[2].last_y)
            self.assertEqual(cache.conv_state.tag, 'copied')
            self.assertEqual((cache.S, cache.offset), ('S-original', 26))
            self.assertEqual(len(module.mx.copies), 1)
            self.assertEqual(module.mx.evaluated, [cache.conv_state] if materialize else [])
            self.assertEqual(len(module.calls), 1)
            self.assertTrue(rows[0]['state_present'] and rows[0]['replaced'])
            self.assertEqual(rows[0]['offset'], 7)
            self.assertEqual(rows[0]['route'], 'single-layer-eager')

    def test_decode_batch_nonprefill_record_and_unscoped_are_untouched(self):
        rows = []
        with i.PrefillTailScope(self.module, self.layers, enabled=True, on_call=rows.append):
            layer = self.layers[2]
            for mode in ('decode', 'prefill_batch', 'verify', 'unknown'):
                layer(self.x, self.module.ForwardContext(mode), None)
            layer(self.x, self.module.ForwardContext(record=True), None)
            layer(self.x, self.module.ForwardContext(), NS(record=True, offset=0, conv_state=None))
            layer._prefill(self.x, None)  # Arbitrary direct _prefill is not the diagnostic eager entry.
        layer(self.x, self.module.ForwardContext(), None)
        self.assertEqual((rows, self.module.mx.copies, self.module.mx.evaluated), ([], [], []))

    def test_ineligible_shape_reported_without_silent_coverage(self):
        rows = []
        with i.PrefillTailScope(self.module, self.layers, enabled=True, on_call=rows.append) as scope:
            scope.single_layer(2, Array((19, 512)), None)
        self.assertEqual(len(rows), 1)
        self.assertFalse(rows[0]['eligible'])
        self.assertFalse(rows[0]['replaced'])
        self.assertEqual(self.module.mx.copies, [])

    def test_actual_model_capture_argument_and_spec_runner_are_excluded(self):
        namespace = {}
        exec(compile('def __call__(layer,x,ctx,capture=()): return layer(x,ctx,None)',
                     str(FROZEN / 'model/qwen3_5.py'), 'exec'), namespace)
        model_call = namespace['__call__']
        exec(compile('def run(self,layer,x,ctx): return layer(x,ctx,None)',
                     str(FROZEN / 'engine/model_runner.py'), 'exec'), namespace)
        rows = []
        with i.PrefillTailScope(self.module, self.layers, enabled=True, on_call=rows.append):
            model_call(self.layers[2], self.x, self.module.ForwardContext(), capture=(5, 19))
            namespace['run'](NS(spec=object()), self.layers[2], self.x, self.module.ForwardContext())
            self.assertEqual(rows, [])
            model_call(self.layers[2], self.x, self.module.ForwardContext(), capture=())
        self.assertEqual(len(rows), 1)

    def test_callback_failure_restores_every_layer_and_preserves_existing_attribute(self):
        saved = MethodType(self.module.GatedDeltaNet._prefill, self.layers[6])
        object.__setattr__(self.layers[6], '_prefill', saved)
        def fail(row):
            raise ValueError('receipt failed')
        with self.assertRaisesRegex(ValueError, 'receipt failed'):
            with i.PrefillTailScope(self.module, self.layers, enabled=True, on_call=fail) as scope:
                scope.single_layer(2, self.x, None)
        self.assertNotIn('_prefill', vars(self.layers[2]))
        self.assertIs(vars(self.layers[6])['_prefill'], saved)
        self.assertEqual(scope.layers, {})

    def test_nested_cross_thread_and_stale_calls_fail(self):
        failures = []
        with i.PrefillTailScope(self.module, self.layers) as scope:
            stale = self.layers[2]._prefill
            with self.assertRaisesRegex(ValueError, 'untouched'):
                with i.PrefillTailScope(self.module, self.layers):
                    pass
            def other_thread():
                try:
                    stale(self.x, None)
                except RuntimeError as error:
                    failures.append(str(error))
            thread = threading.Thread(target=other_thread)
            thread.start(); thread.join(2)
            self.assertFalse(thread.is_alive())
        self.assertEqual(len(failures), 1)
        with self.assertRaisesRegex(RuntimeError, 'inactive'):
            stale(self.x, None)
        with self.assertRaisesRegex(RuntimeError, 'closed'):
            scope.__enter__()

    def test_flag_and_layer_validation(self):
        for kwargs in ({'enabled': 1}, {'materialize': 'yes'}, {'materialize': True}, {'on_call': 1}):
            with self.assertRaises((TypeError, ValueError)):
                i.PrefillTailScope(self.module, self.layers, **kwargs)
        for layers in ({}, {'2': self.layers[2]}, {2: self.layers[2], 6: self.layers[2]}):
            with self.assertRaises(ValueError):
                i.PrefillTailScope(self.module, layers)

    def test_scope_does_not_rewrite_prefill_code_or_other_globals(self):
        original = self.module.GatedDeltaNet._prefill
        with i.PrefillTailScope(self.module, self.layers):
            wrapper = self.layers[2]._prefill.__func__
            closure = dict(zip(wrapper.__code__.co_freevars, (v.cell_contents for v in wrapper.__closure__)))
            cloned = closure['cloned']
            self.assertIs(cloned.__code__, original.__code__)
            self.assertEqual(set(cloned.__globals__), set(original.__globals__))
            changed = [k for k in cloned.__globals__ if cloned.__globals__[k] is not original.__globals__[k]]
            self.assertEqual(changed, ['causal_conv1d'])

    def test_real_frozen_source_route_and_sha_cpu_only(self):
        for relative, sha in i.CORE_PINS.items():
            self.assertEqual(hashlib.sha256((FROZEN / relative).read_bytes()).hexdigest(), sha)
        tree = ast.parse((FROZEN / 'model/gdn.py').read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'GatedDeltaNet')
        methods = {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}
        def calls(node, name):
            return [n for n in ast.walk(node) if isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Name) and n.func.id == name]
        self.assertEqual(len(calls(methods['_prefill'], 'causal_conv1d')), 1)
        self.assertEqual(len(calls(methods['_prefill_batch'], 'causal_conv1d')), 1)
        self.assertEqual(len(calls(methods['_decode'], 'causal_conv1d')), 0)
        conv = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'causal_conv1d')
        self.assertEqual(ast.unparse(conv.body[-1]), 'return (y, xp[T:])')
        compiled = (FROZEN / 'engine/compiled_decode.py').read_text()
        self.assertIn('ctx.mode == "decode"', compiled)
        self.assertNotIn('ForwardContext("prefill"', compiled)

    def test_verify_real_source_functions_without_importing_model(self):
        tree = ast.parse((FROZEN / 'model/gdn.py').read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'GatedDeltaNet')
        cls.bases = []
        cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in ('_prefill', '__call__')]
        conv = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'causal_conv1d')
        tree.body = [conv, cls]
        for node in ast.walk(tree):
            if isinstance(node, ast.arg):
                node.annotation = None
            if isinstance(node, ast.FunctionDef):
                node.returns = None
        module = ModuleType('cpu_source_only')
        module.__file__ = str(FROZEN / 'model/gdn.py')
        exec(compile(tree, module.__file__, 'exec'), vars(module))  # Defines methods; executes no model code.
        helper = VERIFY(module)
        self.assertTrue(callable(helper.own_conv_tail))
        with patch.object(i, '_sha', return_value='0' * 64):
            with self.assertRaisesRegex(ValueError, 'source mismatch'):
                VERIFY(module)
        module.causal_conv1d = lambda *a: None
        with self.assertRaisesRegex(ValueError, 'loaded GDN function'):
            VERIFY(module)

    def test_import_forbids_runtime_model_and_socket(self):
        code = '''
import importlib.abc, importlib.util, socket, sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,name,*args):
  if name.split('.')[0] in {'mlx','numpy','torch','tutorial_code'}: raise AssertionError(name)
sys.meta_path.insert(0,Block())
socket.socket=lambda *a,**k: (_ for _ in ()).throw(AssertionError('socket'))
s=importlib.util.spec_from_file_location('integration',sys.argv[1]);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
assert m.PrefillTailScope
'''
        result = subprocess.run([sys.executable, '-I', '-B', '-c', code, str(HERE / 'integration.py')],
                                capture_output=True, text=True, timeout=5)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
