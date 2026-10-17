"""Execute the original runner AST with fake arrays only; never import MLX."""
import ast
import gc
import hashlib
import importlib.util
from pathlib import Path
import subprocess
import sys
import threading
from types import MethodType, ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch
import weakref

HERE = Path(__file__).resolve().parent
CORE = HERE.parents[2] / 'tutorial_code/miniengine'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


s = load('staged_candidate', HERE / 'staged_integration.py')
c = load('tail_candidate', HERE / 'candidate.py')
VERIFY = s._verify_runner


class Array:
    def __init__(self, shape, dtype='bf16', tag='original'):
        self.shape, self.ndim, self.dtype, self.tag = tuple(shape), len(shape), dtype, tag

    def view(self, dtype):
        return Array(self.shape, dtype, self.tag)


class Tokens(Array):
    def __init__(self, values):
        super().__init__((len(values),), 'int32', 'tokens'); self.values = values

    def __getitem__(self, key):
        return Tokens(self.values[key])

    def item(self):
        return self.values[0]

    def tolist(self):
        return self.values[:]


class Runtime:
    bfloat16, uint16, int32, gpu = 'bf16', 'u16', 'int32', 'gpu'
    metal = NS(device_info=lambda: {'architecture': 'applegpu_g17d'})

    def __init__(self):
        self.calls, self.copies = [], []
        self.fail_eval = False

    def default_device(self): return self.gpu
    def device_info(self): return {'device_name': 'Apple M5 Ultra'}

    def array(self, value):
        return Array((len(value), len(value[0])) if value and isinstance(value[0], list) else (len(value),), 'int32', 'input')

    def contiguous(self, value):
        self.copies.append((value.shape, value.dtype))
        return Array(value.shape, value.dtype, 'owned')

    def eval(self, *values, **kwargs):
        self.calls.append((values, kwargs))
        if self.fail_eval: raise ValueError('fake eval failure')

    def async_eval(self, *values):
        raise AssertionError('fixture intentionally avoids ahead submission')


class Cache:
    def __init__(self):
        self.conv_state = self.S = self.tape = self._snap = None
        self.offset, self.record = 0, False


class Context:
    def __init__(self, mode, caches, **kwargs):
        self.mode, self.caches = mode, caches
        self.__dict__.update(kwargs)


class Model:
    def __init__(self):
        self.cfg = NS(quantization={'group_size': 16, 'bits': 4, 'mode': 'nvfp4'})
        self.embed_tokens = NS(_output_dtype='bf16')
        self.fail = False
        self.tail_shape = (3, 10240)
        self.states = []

    def __call__(self, ids, ctx, **kwargs):
        if self.fail: raise ValueError('fake model failure')
        if ctx.mode == 'prefill_batch':
            caches = [cache for layer in ctx.caches for cache in layer]; n = ids.shape[1]
        else:
            caches = ctx.caches; n = ids.shape[0]
        if ctx.mode != 'decode':
            for cache in caches:
                cache.conv_state = Array(self.tail_shape)
                cache.S = object(); cache.offset += n
        output = Array((1, 5120), tag='unchanged-layer-product')
        return (output, Array((n, 10))) if kwargs.get('capture') else output


def original_class(path, class_name, method_names):
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    cls.bases = []; cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in method_names]
    tree.body = [cls]
    for node in ast.walk(tree):
        if isinstance(node, ast.arg): node.annotation = None
        if isinstance(node, ast.FunctionDef): node.returns = None
    return tree


def fixture():
    module = ModuleType('staged_fake_runner'); module.__file__ = str(CORE / 'engine/model_runner.py')
    mx = Runtime(); module.mx = mx
    module.ForwardContext = Context
    module.make_cache = lambda model: [Cache(), Cache()]
    module.prefill_batch_backend_supported = lambda shape=None: shape is None or shape[0] >= 2
    module.sample = lambda logits, sampling: Tokens([99] * len(sampling))
    exec(compile(original_class(Path(module.__file__), 'ModelRunner', {'run'}), module.__file__, 'exec'), vars(module))
    runner = module.ModelRunner(); runner.model = Model(); runner.decode_model = runner.model
    runner.spec = None; runner.staging = {}; runner.feats = {}; runner._ahead = None; runner.max_len = 2000
    runner.batch = NS(size=0, req_ids=[], layers=[], arrays=lambda: [], clear=lambda: None)
    runner._sync_batch = lambda requests: None
    deps = NS(original=type(runner).run, mx=mx, cache_cls=Cache,
        helper=NS(own_conv_tail=c.own_conv_tail))
    return module, runner, mx, deps


def request(rid, n, computed=0):
    return NS(req_id=rid, num_computed_tokens=computed, num_prompt_tokens=n,
        prompt_token_ids=[1] * n, all_token_ids=[1] * n, output_token_ids=[1], num_tokens=n,
        sampling=NS(temperature=0, top_p=1, top_k=0, max_tokens=1, ignore_eos=False, stop_token_ids=[]))


class StagedMechanisms(unittest.TestCase):
    def setUp(self):
        self.module, self.runner, self.mx, self.deps = fixture()
        self.verify = patch.object(s, '_verify_runner', return_value=self.deps); self.verify.start()
        self.backend = patch.object(c, 'supported_backend', return_value=True); self.backend.start()
        self.addCleanup(self.verify.stop); self.addCleanup(self.backend.stop)

    def out(self, *pairs, decode=()): return NS(prefill=list(pairs), decode=list(decode))

    def test_default_off_no_patch_no_backend_probe(self):
        original = self.runner.run.__func__
        with patch.object(s, '_verify_runner', side_effect=AssertionError('OFF probes backend')):
            with s.StagedPrefillTailScope(self.runner) as scope:
                self.assertNotIn('run', vars(self.runner))
                self.assertEqual(self.runner.run(self.out((request('a', 19), 19))), {'a': 99})
        self.assertIs(self.runner.run.__func__, original)
        self.assertEqual(self.mx.copies, [])
        self.assertIsNone(scope._runner)

    def test_original_runner_code_private_mx_and_one_shared_eval(self):
        rows = []; original = self.deps.original
        with s.StagedPrefillTailScope(self.runner, enabled=True, on_boundary=rows.append) as scope:
            self.assertIs(scope._cloned.__code__, original.__code__)
            self.assertEqual([k for k in original.__globals__ if scope._cloned.__globals__[k] is not original.__globals__[k]], ['mx'])
            self.assertIs(original.__globals__['mx'], self.mx)
            value = self.runner.run(self.out((request('a', 19), 19), (request('b', 20), 20)))
            self.assertEqual(value, {'a': 99, 'b': 99})
            tails = [x.conv_state for caches in self.runner.staging.values() for x in caches]
            self.assertEqual(len(self.mx.calls), 1)
            self.assertEqual(list(self.mx.calls[0][0][2:]), tails)
            self.assertTrue(all(t.tag == 'owned' for t in tails))
            self.assertEqual(len(self.mx.copies), 4)
        self.assertEqual(rows, [dict(ordinary_prefill_requests=2, grouped_requests_skipped=0,
            owned_tails=4, existing_eval_calls=1, per_layer_eval_calls=0)])
        self.assertNotIn('run', vars(self.runner))

    def test_continuing_chunks_stage_only_current_requests(self):
        idle = [Cache(), Cache()]
        for cache in idle: cache.conv_state = Array((3, 10240)); cache.offset = 900
        self.runner.staging['unrelated'] = idle
        req = request('a', 19)
        with s.StagedPrefillTailScope(self.runner, enabled=True):
            self.assertEqual(self.runner.run(self.out((req, 7))), {})
            self.assertEqual(len(self.mx.calls[0][0]), 3)  # original logits + two tails
            req.num_computed_tokens = 7
            self.assertEqual(self.runner.run(self.out((req, 12))), {'a': 99})
        self.assertEqual(len(self.mx.calls), 2)
        self.assertEqual([x.offset for x in self.runner.staging['a']], [19, 19])
        self.assertTrue(all(x.conv_state.tag == 'original' for x in idle))

    def test_actual_grouped_set_excluded_and_mixed_lengths_preserved(self):
        rows = []
        with s.StagedPrefillTailScope(self.runner, enabled=True, on_boundary=rows.append):
            self.runner.run(self.out((request('a', 19), 19), (request('b', 19), 19), (request('c', 20), 20)))
        self.assertEqual(rows[0]['grouped_requests_skipped'], 2)
        self.assertEqual(rows[0]['owned_tails'], 2)
        self.assertTrue(all(c.conv_state.tag == 'original' for rid in ('a', 'b') for c in self.runner.staging[rid]))
        self.assertTrue(all(c.conv_state.tag == 'owned' for c in self.runner.staging['c']))
        self.assertEqual(len(self.mx.calls), 1)

    def test_spec_capture_and_decode_only_use_original_globals(self):
        rows = []
        with s.StagedPrefillTailScope(self.runner, enabled=True, on_boundary=rows.append):
            self.runner.spec = NS(capture=(5, 19))
            self.assertEqual(self.runner.run(self.out((request('a', 19), 19))), {'a': 99})
            self.runner.spec = None
            self.assertEqual(self.runner.run(self.out(decode=[request('a', 19)])), {'a': 99})
        self.assertEqual(self.mx.copies, [])
        self.assertEqual(rows, [])
        self.assertEqual(len(self.mx.calls), 2)

    def test_record_snapshot_extra_cache_fields_and_unsupported_tail_skip(self):
        for key, value in (('record', True), ('tape', object()), ('_snap', object()), ('capture', True)):
            caches = [Cache(), Cache()]
            for cache in caches: setattr(cache, key, value)
            self.runner.staging['a'] = caches
            with s.StagedPrefillTailScope(self.runner, enabled=True):
                self.runner.run(self.out((request('a', 19), 19)))
            self.assertTrue(all(cache.conv_state.tag == 'original' for cache in caches))
        self.runner.model.tail_shape = (3, 100)
        with s.StagedPrefillTailScope(self.runner, enabled=True):
            self.runner.run(self.out((request('b', 19), 19)))
        self.assertEqual(self.mx.copies, [])

    def test_model_and_eval_failures_restore_scope_and_clear_instance_refs(self):
        for place in ('model', 'eval', 'callback'):
            self.module, self.runner, self.mx, self.deps = fixture()
            callback = lambda row: (_ for _ in ()).throw(ValueError('fake callback failure'))
            with patch.object(s, '_verify_runner', return_value=self.deps):
                self.runner.model.fail = place == 'model'; self.mx.fail_eval = place == 'eval'
                with self.assertRaisesRegex(ValueError, 'fake .* failure'):
                    with s.StagedPrefillTailScope(self.runner, enabled=True,
                            on_boundary=callback if place == 'callback' else None) as scope:
                        self.runner.run(self.out((request('a', 19), 19)))
                self.assertNotIn('run', vars(self.runner))
                self.assertTrue(all(getattr(scope, name) is None for name in ('_runner', '_model', '_out', '_cloned', '_deps', '_saved', 'on_boundary')))

    def test_saved_attribute_restoration_stale_cross_thread_and_reentry(self):
        saved = MethodType(type(self.runner).run, self.runner); self.runner.run = saved
        failures = []
        with s.StagedPrefillTailScope(self.runner, enabled=True) as scope:
            stale = self.runner.run
            def foreign():
                try: stale(self.out())
                except RuntimeError as error: failures.append(str(error))
            thread = threading.Thread(target=foreign); thread.start(); thread.join(2)
            self.assertFalse(thread.is_alive())
            scope._running = True
            with self.assertRaisesRegex(RuntimeError, 'reentrant'): stale(self.out())
            scope._running = False
        self.assertIs(self.runner.run, saved)
        self.assertEqual(len(failures), 1)
        with self.assertRaisesRegex(RuntimeError, 'inactive'): stale(self.out())
        with self.assertRaisesRegex(RuntimeError, 'closed'): scope.__enter__()

    def test_model_replacement_and_alias_fail_closed(self):
        with s.StagedPrefillTailScope(self.runner, enabled=True):
            self.runner.model = Model()
            with self.assertRaisesRegex(RuntimeError, 'replacement'): self.runner.run(self.out((request('a', 19), 19)))
        # A reused cache may be advanced twice by original run, so offset guard
        # rejects the alias before it could be counted as two valid copy paths.
        cache = Cache(); self.runner.staging['a'] = [cache, cache]
        with s.StagedPrefillTailScope(self.runner, enabled=True):
            with self.assertRaisesRegex(RuntimeError, 'offset|aliased'): self.runner.run(self.out((request('a', 19), 19)))

    def test_retained_closed_scope_does_not_keep_runner_or_model_alive(self):
        runner = self.runner; wr, wm = weakref.ref(runner), weakref.ref(runner.model)
        with s.StagedPrefillTailScope(runner, enabled=True) as scope:
            runner.run(self.out((request('a', 19), 19)))
        # unittest.mock stores every positional argument, including this runner.
        # Remove that test-only owner before testing the retained closed scope.
        s._verify_runner.reset_mock()
        self.runner = None; del runner
        gc.collect()
        self.assertIsNone(wr()); self.assertIsNone(wm()); self.assertIsNone(scope._runner)

    def test_source_hash_and_exact_one_eval_before_token_delivery(self):
        for name, expected in s.CORE_PINS.items(): self.assertEqual(hashlib.sha256((CORE/name).read_bytes()).hexdigest(), expected)
        tree = original_class(CORE/'engine/model_runner.py', 'ModelRunner', {'run'})
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                 and isinstance(n.func.value, ast.Name) and n.func.value.id == 'mx' and n.func.attr == 'eval']
        self.assertEqual(len(calls), 1)
        self.assertEqual(ast.unparse(calls[0]), 'mx.eval(*to_eval, *self.batch.arrays())')
        deliver = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in ('item', 'tolist')]
        self.assertTrue(all(n.lineno > calls[0].lineno for n in deliver))

    def test_actual_source_validator_with_ast_defined_functions_only(self):
        model_namespace = {}
        exec(compile(original_class(CORE/'model/qwen3_5.py', 'Qwen3_5ForCausalLM', {'__call__'}),
                     str(CORE/'model/qwen3_5.py'), 'exec'), model_namespace)
        model = model_namespace['Qwen3_5ForCausalLM']()
        model.cfg, model.embed_tokens = self.runner.model.cfg, self.runner.model.embed_tokens
        self.runner.model = model
        cache_tree = original_class(CORE/'cache.py', 'GDNCache', {'__init__'})
        cache_tree.body.append(next(n for n in ast.parse((CORE/'cache.py').read_text()).body
                                    if isinstance(n, ast.FunctionDef) and n.name == 'make_cache'))
        cache_namespace = {}
        exec(compile(cache_tree, str(CORE/'cache.py'), 'exec'), cache_namespace)
        self.module.make_cache = cache_namespace['make_cache']
        with patch.dict(sys.modules, {self.module.__name__: self.module}), patch('importlib.metadata.version', return_value='0.32.3'):
            checked = VERIFY(self.runner)
            self.assertIs(checked.original, type(self.runner).run)
            self.assertIs(checked.mx, self.mx)
            self.assertIs(checked.cache_cls, cache_namespace['GDNCache'])
            with patch.object(s, '_sha', return_value='0'*64):
                with self.assertRaisesRegex(ValueError, 'source mismatch'): VERIFY(self.runner)
            self.runner.model.cfg.quantization = {'mode': 'unsupported'}
            with self.assertRaisesRegex(ValueError, 'backend'): VERIFY(self.runner)
            self.runner.run = lambda out: {}
            with self.assertRaises((AttributeError, ValueError)): VERIFY(self.runner)

    def test_flag_validation_and_import_no_backend(self):
        for kwargs in ({'enabled': 1}, {'on_boundary': False}):
            with self.assertRaises(TypeError): s.StagedPrefillTailScope(self.runner, **kwargs)
        code = '''
import importlib.abc, importlib.util, sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,name,*args):
  if name.split('.')[0] in {'mlx','numpy','torch','tutorial_code'}: raise AssertionError(name)
sys.meta_path.insert(0,Block())
spec=importlib.util.spec_from_file_location('staged',sys.argv[1]);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
with module.StagedPrefillTailScope(object()): pass
'''
        done = subprocess.run([sys.executable, '-I', '-B', '-c', code, str(HERE/'staged_integration.py')], capture_output=True, text=True, timeout=5)
        self.assertEqual(done.returncode, 0, done.stderr)


if __name__ == '__main__':
    unittest.main()
