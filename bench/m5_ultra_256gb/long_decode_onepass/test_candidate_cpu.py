"""Unexecuted CPU mechanism tests; real MLX arithmetic/throughput remains untested.

Use the existing stdlib E024 fixtures and actual frozen route implementation as
the behavioral baseline. The fake compiler records capture identity and executes
Python; it is NOT evidence about actual MLX tracing or parameter propagation.
"""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    # The existing fixtures contain a dataclass.
    import sys
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


c = load(HERE / 'candidate.py', '_onepass_candidate_cpu')
old = load(c.ORIGINAL, '_onepass_original_cpu')
f = load(HERE.parent / 'long_decode_graph/test_cpu.py', '_onepass_fixtures_cpu')
OnePass = c.make_decoder_class(old.RoutePreservingDecode)


def setup(kind=OnePass, **kwargs):
    model, runtime, _, tokens, ctx = f.setup(**kwargs)
    with patch.object(old, '_load_runtime', return_value=runtime):
        graph = kind(model, enabled=True)
    return model, runtime, graph, tokens, ctx


def summary(ctx):
    return [(a.shape, a.dtype, a.value) for pair in f.states(ctx.caches) for a in pair]


class Tests(unittest.TestCase):
    def test_off_does_not_read_source_or_load_runtime(self):
        model = f.Model()
        with patch.object(c, '_original_class', side_effect=AssertionError('source forbidden')):
            self.assertIs(c.build(model), model)
            for bad in (0, 1, None, 'on'):
                with self.subTest(bad=bad), self.assertRaises(TypeError):
                    c.build(model, enabled=bad)

    def test_all_used_boundaries_and_batch_domain_match_original(self):
        for B in (0, 1, 8, 9):
            for used in (256, 257, 1023, 1024, 1536, 1537, True, 257.0):
                with self.subTest(B=B, used=used):
                    records = []
                    for kind in (old.RoutePreservingDecode, OnePass):
                        m, r, graph, tokens, ctx = setup(kind, B=B, used=used)
                        records.append((graph(tokens, ctx), bool(r.compiles), summary(ctx)))
                    self.assertEqual(records[0], records[1])

    def test_rejected_metadata_keeps_original_ordinary_fallback(self):
        mutations = (
            lambda m,r,t,x: setattr(x, 'capture', True),
            lambda m,r,t,x: setattr(x, 'prefill_shape', (1, 1)),
            lambda m,r,t,x: setattr(x, 'mode', 'prefill'),
            lambda m,r,t,x: setattr(t, 'dtype', 'bf16'),
            lambda m,r,t,x: setattr(x.positions, 'shape', (99,)),
            lambda m,r,t,x: setattr(x.lens, 'dtype', 'fp32'),
            lambda m,r,t,x: setattr(m.cfg, 'hidden_size', 123),
            lambda m,r,t,x: setattr(m.cfg, 'quantization', {}),
            lambda m,r,t,x: setattr(m.embed_tokens, '_output_dtype', 'fp32'),
            lambda m,r,t,x: setattr(r, 'device', 'cpu'),
            lambda m,r,t,x: setattr(r, 'supported', False),
            lambda m,r,t,x: setattr(x.caches[0].S, 'dtype', 'bf16'),
            lambda m,r,t,x: setattr(x.caches[0].conv, 'shape', (2, 4, 10240)),
            lambda m,r,t,x: setattr(x.caches[3].K, 'shape', (2, 4, 1535, 256)),
            lambda m,r,t,x: setattr(x.caches[3].V, 'dtype', 'fp32'),
            lambda m,r,t,x: setattr(x.caches[-1], 'extra', True),
        )
        for mutate in mutations:
            records = []
            for kind in (old.RoutePreservingDecode, OnePass):
                m, r, graph, tokens, ctx = setup(kind)
                mutate(m, r, tokens, ctx)
                records.append(graph(tokens, ctx))
                self.assertFalse(r.compiles)
                self.assertIs(m.calls[-1][1], ctx)
            self.assertEqual(records[0], records[1])

    def test_short_backing_rejected_even_in_1023_route(self):
        for capacity in (1023, 1024, 1535, 1536, 1792):
            m, r, graph, tokens, ctx = setup(used=257, allocated=capacity)
            graph(tokens, ctx)
            self.assertEqual(bool(r.compiles), capacity >= 1536)

    def test_layer_replacement_and_reordering_do_not_use_cached_graph(self):
        for index in (0, 32, 63):
            m, r, graph, tokens, ctx = setup()
            graph(tokens, ctx); before = len(r.executions)
            m.layers[index] = SimpleNamespace(**vars(m.layers[index]))
            graph(tokens, ctx)
            self.assertEqual(len(r.executions), before)
            self.assertIs(m.calls[-1][1], ctx)
        m, r, graph, tokens, ctx = setup()
        m.layers[0], m.layers[4] = m.layers[4], m.layers[0]
        graph(tokens, ctx); self.assertFalse(r.compiles)

    def test_flags_change_selects_new_family_then_returns_to_old_family(self):
        m, r, graph, tokens, ctx = setup()
        graph(tokens, ctx)
        layer = m.layers[32].linear_attn
        layer._gates_compile_enabled = False
        graph(tokens, ctx)
        layer._norm_reuse_enabled = True
        graph(tokens, ctx)
        layer._gates_compile_enabled, layer._norm_reuse_enabled = True, False
        graph(tokens, ctx)
        self.assertEqual(len(r.compiles), 3)
        self.assertTrue(all(captured is m for _, captured in r.compiles))

    def test_parameter_capture_is_live_model_not_prebound_values(self):
        m, r, graph, tokens, ctx = setup()
        original = m.weight
        self.assertEqual(graph(tokens, ctx)[0], 100)
        m.weight = f.Array((1,), value=913)
        self.assertEqual(graph(tokens, ctx)[0], 913)
        m.weight = original
        self.assertEqual(graph(tokens, ctx)[0], 100)
        self.assertEqual(len(r.compiles), 1)
        self.assertIs(r.compiles[0][1], m)

    def test_kv_wrapper_protection_and_all_state_updates_match_original(self):
        records = []
        for kind in (old.RoutePreservingDecode, OnePass):
            m, r, graph, tokens, ctx = setup(kind, value=5)
            m.mutate = True
            saved = [a for pair in f.states(ctx.caches) for a in pair]
            graph(tokens, ctx)
            self.assertTrue(all(a.value == 5 for a in saved))
            self.assertEqual(len(r.executions[0][3]), 64)
            records.append(summary(ctx))
        self.assertEqual(records[0], records[1])

    def test_incomplete_return_is_rejected_before_any_cache_assignment(self):
        m, r, graph, tokens, ctx = setup()
        graph(tokens, ctx)
        before = [a for pair in f.states(ctx.caches) for a in pair]
        key = next(iter(graph._functions))
        bad = list(f.states(ctx.caches)); bad[-1] = (f.Array((1,)),)
        graph._functions[key] = lambda *args: ('unused', bad)
        with self.assertRaisesRegex(RuntimeError, 'incomplete state tree'):
            graph(tokens, ctx)
        after = [a for pair in f.states(ctx.caches) for a in pair]
        self.assertTrue(all(a is b for a, b in zip(before, after)))

    def test_compile_failure_propagates_without_eager_retry(self):
        m, r, graph, tokens, ctx = setup()
        def fail(*args, **kwargs): raise RuntimeError('injected compilation failure')
        r.mx.compile = fail
        with self.assertRaisesRegex(RuntimeError, 'injected compilation failure'):
            graph(tokens, ctx)
        self.assertEqual(m.calls, [])

    def test_trace_body_is_identical_to_frozen_original(self):
        def forward(path):
            return next(n for n in ast.walk(ast.parse(path.read_text()))
                        if isinstance(n, ast.FunctionDef) and n.name == 'forward')
        self.assertEqual(ast.dump(forward(c.ORIGINAL), include_attributes=False),
                         ast.dump(forward(HERE / 'candidate.py'), include_attributes=False))


if __name__ == '__main__': unittest.main()
