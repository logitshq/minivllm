"""E024 stdlib mocks only. No MLX, model load, listener or numerical claim."""
from dataclasses import dataclass
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


PATH = Path(__file__).with_name('candidate.py')
spec = importlib.util.spec_from_file_location('e024_cpu_candidate', PATH)
candidate = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = candidate
spec.loader.exec_module(candidate)


class Array:
    def __init__(self, shape, dtype='bf16', value=0):
        self.shape, self.dtype, self.value = tuple(shape), dtype, value

    @property
    def ndim(self):
        return len(self.shape)

    def __getitem__(self, index):
        if index != slice(None):
            raise AssertionError('mock only supports wrapper-preserving [:]')
        return Array(self.shape, self.dtype, self.value)


class BatchKV:
    def __init__(self, K, V):
        self.K, self.V = K, V


class BatchState:
    def __init__(self, conv, S):
        self.conv, self.S = conv, S


@dataclass
class ForwardContext:
    mode: str
    caches: list
    positions: object = None
    lens: object = None
    max_len: int = 0
    prefill_shape: object = None


def states(caches):
    return [(c.K, c.V) if type(c) is BatchKV else (c.conv, c.S) for c in caches]


class Model:
    def __init__(self):
        self.cfg = SimpleNamespace(hidden_size=5120,
            quantization={'group_size': 16, 'bits': 4, 'mode': 'nvfp4'})
        self.layers = [SimpleNamespace(is_attn=i % 4 == 3,
            linear_attn=SimpleNamespace(_gates_compile_enabled=True, _norm_reuse_enabled=False))
            for i in range(64)]
        self.embed_tokens = SimpleNamespace(_output_dtype='bf16')
        self.weight = Array((1,), value=100)
        self.calls, self.mutate = [], False

    def __call__(self, ids, ctx):
        self.calls.append((ids, ctx))
        if self.mutate:
            for c in ctx.caches:
                if type(c) is BatchKV:
                    c.K.value += 10  # Imitate mx.array handle __setitem__.
                    c.V.value += 20
                else:
                    c.conv = Array(c.conv.shape, c.conv.dtype, c.conv.value + 30)
                    c.S = Array(c.S.shape, c.S.dtype, c.S.value + 40)
        return (self.weight.value, ids.value, ctx.max_len)


class Runtime:
    def __init__(self):
        self.compiles, self.executions = [], []
        self.device, self.supported = 'gpu', True
        self.mx = SimpleNamespace(bfloat16='bf16', float32='fp32', int32='i32', int64='i64',
            gpu='gpu', default_device=lambda: self.device, compile=self.compile)
        self.BatchKV, self.BatchState = BatchKV, BatchState
        self.ForwardContext, self.states = ForwardContext, states
        self.identity = lambda: self.supported
        self.source_receipt = {'scope': 'mock only; no actual GPU proof'}

    def compile(self, fn, *, inputs):
        self.compiles.append((fn, inputs))
        def call(*args):
            self.executions.append(args)
            return fn(*args)
        return call


def setup(B=2, used=900, allocated=1536, value=0):
    model, runtime = Model(), Runtime()
    caches = [BatchKV(Array((B, 4, allocated, 256), value=value),
                       Array((B, 4, allocated, 256), value=value)) if l.is_attn
              else BatchState(Array((B, 3, 10240), value=value),
                              Array((B, 48, 128, 128), 'fp32', value)) for l in model.layers]
    ctx = ForwardContext('decode', caches, Array((B,), 'i32'), Array((B,), 'i32'), used)
    tokens = Array((B,), 'i32', 7)
    with patch.object(candidate, '_load_runtime', return_value=runtime):
        graph = candidate.LongContextDecode(model, enabled=True)
    return model, runtime, graph, tokens, ctx


class ContractTests(unittest.TestCase):
    def test_default_off_has_no_runtime_import_or_source_io(self):
        model = Model()
        with patch.object(candidate, '_load_runtime', side_effect=AssertionError('runtime forbidden')):
            graph = candidate.LongContextDecode(model)
            self.assertEqual(graph(Array((1,), 'i32', 3), SimpleNamespace(max_len=999)), (100, 3, 999))
        self.assertFalse(graph.enabled)
        self.assertFalse(graph._functions)
        self.assertIsNone(graph.source_receipt)

    def test_enablement_and_captured_model_are_read_only(self):
        for bad in (0, 1, None, 'on'):
            with self.subTest(bad=bad), self.assertRaises(TypeError):
                candidate.LongContextDecode(Model(), enabled=bad)
        model, r, graph, ids, ctx = setup()
        for key, value in (('enabled', False), ('model', Model()), ('bound', 2048)):
            with self.subTest(key=key), self.assertRaises(AttributeError):
                setattr(graph, key, value)
        with self.assertRaises(TypeError):
            candidate.LongContextDecode(model, enabled=True, fuse_decode=True)
        with self.assertRaises(TypeError):
            candidate.LongContextDecode(model, enabled=True, bound=2048)

    def test_used_length_boundary_and_both_backings_agree_with_closure(self):
        for used in (256, 257, 1536, 1537):
            with self.subTest(used=used):
                model, r, graph, ids, ctx = setup(used=used)
                actual = graph(ids, ctx)
                accepted = 257 <= used <= 1536
                self.assertEqual(len(r.compiles), int(accepted))
                self.assertEqual(actual[-1], 1536 if accepted else used)
                self.assertEqual(ctx.max_len, used)
        for allocated in (1535, 1536, 1792):
            model, r, graph, ids, ctx = setup(allocated=allocated)
            graph(ids, ctx)
            self.assertEqual(len(r.compiles), int(allocated >= 1536))
        model, r, graph, ids, ctx = setup()
        ctx.caches[3].V = Array((2, 4, 1535, 256))
        graph(ids, ctx)
        self.assertFalse(r.compiles)

    def test_batch_bounds(self):
        for B in (0, 1, 2, 3, 7, 8, 9, 64):
            with self.subTest(B=B):
                model, r, graph, ids, ctx = setup(B=B)
                graph(ids, ctx)
                self.assertEqual(bool(r.compiles), 1 <= B <= 8)

    def test_extra_context_and_mode_fail_before_compilation(self):
        for mutation in (
            lambda c: setattr(c, 'record', False),
            lambda c: setattr(c, 'capture', False),
            lambda c: setattr(c, 'mode', 'prefill'),
            lambda c: setattr(c, 'prefill_shape', (2, 900)),
            lambda c: setattr(c, 'max_len', 900.0),
            lambda c: setattr(c, 'positions', None),
            lambda c: setattr(c, 'positions', Array((3,), 'i32')),
            lambda c: setattr(c, 'lens', Array((2,), 'fp32')),
        ):
            model, r, graph, ids, ctx = setup()
            mutation(ctx)
            graph(ids, ctx)
            self.assertFalse(r.compiles)
            self.assertIs(model.calls[-1][1], ctx)
        class UnknownContext(ForwardContext):
            pass
        model, r, graph, ids, ctx = setup()
        ctx.__class__ = UnknownContext
        graph(ids, ctx)
        self.assertFalse(r.compiles)

    def test_bad_input_dtype_shape_and_cache_metadata(self):
        for mutation in (
            lambda t, c: setattr(t, 'dtype', 'fp32'),
            lambda t, c: setattr(t, 'shape', (2, 1)),
            lambda t, c: setattr(c.caches[0], 'record', False),
            lambda t, c: setattr(c.caches[0].S, 'dtype', 'bf16'),
            lambda t, c: setattr(c.caches[0].conv, 'shape', (2, 4, 10240)),
            lambda t, c: setattr(c.caches[3].K, 'dtype', 'fp32'),
            lambda t, c: setattr(c.caches[3].V, 'shape', (1, 4, 1536, 256)),
            lambda t, c: setattr(c, 'caches', c.caches[:-1]),
        ):
            model, r, graph, ids, ctx = setup()
            mutation(ids, ctx)
            graph(ids, ctx)
            self.assertFalse(r.compiles)

    def test_unsupported_backend_quant_model_shape_or_layer_replacement(self):
        for mutation in (
            lambda m, r: setattr(r, 'device', 'cpu'),
            lambda m, r: setattr(r, 'supported', False),
            lambda m, r: setattr(m.cfg, 'hidden_size', 4096),
            lambda m, r: setattr(m.cfg, 'quantization', {'group_size': 32, 'bits': 4, 'mode': 'affine'}),
            lambda m, r: setattr(m.embed_tokens, '_output_dtype', 'fp32'),
            lambda m, r: m.layers.pop(),
            lambda m, r: m.layers.__setitem__(0, SimpleNamespace(
                is_attn=False, linear_attn=m.layers[0].linear_attn)),
        ):
            model, r, graph, ids, ctx = setup()
            mutation(model, r)
            graph(ids, ctx)
            self.assertFalse(r.compiles)

    def test_parameters_are_capture_inputs_and_replacements_use_same_model(self):
        model, r, graph, ids, ctx = setup()
        first = graph(ids, ctx)
        self.assertIs(r.compiles[0][1], model)
        model.weight = Array((1,), value=1234)
        second = graph(ids, ctx)
        self.assertEqual((first[0], second[0]), (100, 1234))
        self.assertEqual(len(r.compiles), 1)
        # Mock semantics and inputs identity only: real MLX replacement is a GPU gate.
        nonlocals = inspect.getclosurevars(r.compiles[0][0]).nonlocals
        self.assertEqual(set(nonlocals), {'bound', 'layout', 'model', 'r'})
        self.assertIs(nonlocals['model'], model)
        self.assertNotIn('ctx', nonlocals)
        self.assertNotIn('states', nonlocals)

    def test_python_flags_select_separate_function_and_bound_is_in_key(self):
        model, r, graph, ids, ctx = setup()
        graph(ids, ctx)
        model.layers[0].linear_attn._gates_compile_enabled = False
        graph(ids, ctx)
        model.layers[0].linear_attn._norm_reuse_enabled = True
        graph(ids, ctx)
        self.assertEqual(len(r.compiles), 3)
        self.assertTrue(all(key[0] == 1536 for key in graph._functions))
        model.layers[0].linear_attn._gates_compile_enabled = True
        model.layers[0].linear_attn._norm_reuse_enabled = False
        graph(ids, ctx)
        self.assertEqual(len(r.compiles), 3)

    def test_explicit_states_do_not_capture_a_prior_request_or_mutate_input_wrappers(self):
        model, r, graph, ids, ctx = setup(value=11)
        model.mutate = True
        oldK, oldV = ctx.caches[3].K, ctx.caches[3].V
        oldconv, oldS = ctx.caches[0].conv, ctx.caches[0].S
        graph(ids, ctx)
        self.assertEqual((oldK.value, oldV.value, oldconv.value, oldS.value), (11, 11, 11, 11))
        self.assertEqual((ctx.caches[3].K.value, ctx.caches[3].V.value,
                          ctx.caches[0].conv.value, ctx.caches[0].S.value), (21, 31, 41, 51))
        _, _, _, next_ids, next_ctx = setup(value=1000)
        graph(next_ids, next_ctx)
        self.assertEqual(next_ctx.caches[3].K.value, 1010)
        self.assertEqual(ctx.caches[3].K.value, 21)
        self.assertEqual(len(r.compiles), 1)

    def test_incomplete_graph_state_fails_before_assigning_any_cache(self):
        model, r, graph, ids, ctx = setup()
        graph(ids, ctx)
        oldK = ctx.caches[3].K
        key = next(iter(graph._functions))
        graph._functions[key] = lambda *a: ('fake', [(Array((1,)), Array((1,)))])
        with self.assertRaisesRegex(RuntimeError, 'incomplete state'):
            graph(ids, ctx)
        self.assertIs(ctx.caches[3].K, oldK)


class SourceTests(unittest.TestCase):
    def test_pins_match_current_python_sources_without_importing_them(self):
        root = PATH.parents[3] / 'tutorial_code/miniengine'
        receipt = candidate.verify_model_sources(root)
        self.assertEqual(len(receipt['files']), 31)
        self.assertIn('engine/compiled_decode.py', receipt['files'])
        self.assertIn('model/qwen3_5.py', receipt['files'])

    def test_source_change_missing_or_extra_python_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / 'core'
            root.mkdir()
            (root / 'a.py').write_text('source=1\n')
            pins = Path(temp) / 'pins.json'
            pins.write_text(json.dumps({'files': {'a.py': hashlib.sha256((root / 'a.py').read_bytes()).hexdigest()}}))
            with patch.object(candidate, 'PINS', pins):
                candidate.verify_model_sources(root)
                (root / 'a.py').write_text('source=2\n')
                with self.assertRaises(RuntimeError):
                    candidate.verify_model_sources(root)
                (root / 'a.py').unlink()
                with self.assertRaises(RuntimeError):
                    candidate.verify_model_sources(root)
                (root / 'a.py').write_text('source=1\n')
                (root / 'unexpected.py').write_text('x=1\n')
                with self.assertRaises(RuntimeError):
                    candidate.verify_model_sources(root)

    def test_unknown_model_origin_is_rejected_before_runtime_import(self):
        with self.assertRaisesRegex(RuntimeError, 'pinned Qwen'):
            candidate._load_runtime(Model())

    def test_import_is_cpu_only_and_has_no_listener_side_effect(self):
        code = '''
import builtins, importlib.util, socket, sys
original = builtins.__import__
def checked(name, *args, **kwargs):
    if name.startswith(('mlx', 'tutorial_code', 'tokenizers', 'numpy')):
        raise AssertionError('GPU/model dependency import: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = checked
socket.socket = lambda *a, **k: (_ for _ in ()).throw(AssertionError('socket forbidden'))
s = importlib.util.spec_from_file_location('e024_no_gpu', sys.argv[1])
m = importlib.util.module_from_spec(s)
s.loader.exec_module(m)
assert m.ATTENTION_BOUND == 1536
'''
        result = subprocess.run([sys.executable, '-B', '-c', code, str(PATH)],
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
