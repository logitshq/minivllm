"""CPU traces are deliberate fakes; frozen actual Observer/Witness are reused."""
import copy
import io
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from bench.m5_ultra_256gb.route_decode_graph import test_serving_qualification_cpu as f
from bench.m5_ultra_256gb.gdn_graph_combined import qualify as q


class Graph:
    def __init__(self, handles):
        self.handles = handles; self._functions = {}; self.seen = set()
        self.trace_passes = 1; self.mixed = False; self.silent = False; self.fail = False
        self.output = object()
    def __call__(self, tokens, ctx):
        bound = 1023 if ctx.max_len < 1024 else 1536
        if bound not in self._functions:
            def compiled(tokens, ctx):
                if self.fail: raise RuntimeError('actual callable failed')
                B = tokens.shape[0]; key = (bound, B)
                if key not in self.seen:
                    if not self.silent:
                        for i, handle in enumerate(self.handles):
                            if self.mixed and i == 47: continue
                            handle.calls['candidate' if B == 1 else 'fallback'] += self.trace_passes
                    self.seen.add(key)
                return self.output
            self._functions[bound] = compiled
        return self._functions[bound](tokens, ctx)


class Scope:
    def __init__(self, engine):
        self.runner = engine.runner; self.model = self.runner.model; self.previous = self.runner.decode_model
        self.gdn_handles = [SimpleNamespace(calls=dict(candidate=0, fallback=0)) for _ in range(48)]
        self.graph = Graph(self.gdn_handles)
        self.graph_handle = SimpleNamespace(decoder=f.install_module._Dispatch(self.previous, self.graph))
        self.runner.decode_model = self.graph_handle.decoder; self.closed = False
    def restore(self):
        if self.closed: return
        assert self.runner.decode_model is self.graph_handle.decoder
        assert self.runner.model is self.model and 'run' not in vars(self.runner)
        assert self.graph_handle.decoder.long is self.graph
        self.graph._functions.clear(); self.runner.decode_model = self.previous; self.closed = True


class Integration:
    def __init__(self): self.scopes = []
    def install(self, engine, *, gdn_prework, route_graph):
        assert gdn_prework is True and route_graph is True
        scope = Scope(engine); self.scopes.append(scope); return scope


def proxy_fixture():
    old = f.old_module(); events = []
    stats = dict(installs=0, restores=0, layer_installs=0, layer_restores=0,
        **{'compiled-tracing': 0, 'compiled-reuse': 0, 'eager-fallback': 0}, traced_batches=[])
    handles = [SimpleNamespace(calls=dict(candidate=0, fallback=0)) for _ in range(48)]
    graph = Graph(handles); proxy = q.TraceGraph(graph, handles, events.append, stats, 0)
    witness = f.q.RouteWitness(proxy, old, events.append, dict(route_hits={}, long_fallback_calls=0))
    proxy.witness = witness; witness.phase = 'D-C1'; witness.step = 1
    return graph, proxy, witness, stats, events


class Tests(unittest.TestCase):
    def test_default_description_and_partial_cli_never_load_runtime(self):
        with patch.object(q, 'load', side_effect=AssertionError('loaded helper')), patch('sys.stdout', new=io.StringIO()):
            self.assertEqual(q.main([]), 0)
            with patch('sys.stderr', new=io.StringIO()), self.assertRaises(SystemExit): q.main(['--plan', '/not-used'])

    def test_plan_exact_sources_deadline_and_frozen_baseline(self):
        path = Path('/fake/child.json'); inherited = Path('/fake/original.json')
        sources = {'self': 'x'}
        plan = dict(schema='GDN-graph-combined-serving-plan-v1', qualifier_sha256='self',
            qualification=dict(path=str(inherited), sha256=q.BASE_PLAN_SHA), sources=sources, deadline_s=1800)
        original = dict(deadline_s=1800, candidate=dict(compile_decode=True, fuse_decode=True, gates_compile=True, norm_reuse=False))
        def check(value, baseline=original):
            with patch.object(q, 'sha', side_effect=lambda p: 'plan' if Path(p) == path else q.BASE_PLAN_SHA if Path(p) == inherited else 'self'), \
                 patch.object(q, 'read', side_effect=lambda p: value if Path(p) == path else baseline), \
                 patch.object(q, 'source_pins', return_value=sources):
                return q.validate(path, 'plan')
        self.assertEqual(check(plan), plan)
        for key, value in (('deadline_s', 1800.0), ('schema', 'wrong'), ('sources', {}), ('unknown', True)):
            bad = copy.deepcopy(plan); bad[key] = value
            with self.assertRaises(ValueError): check(bad)
        bad = copy.deepcopy(original); bad['candidate']['gates_compile'] = False
        with self.assertRaises(ValueError): check(plan, bad)

    def test_real_route_observer_wraps_after_both_installs_and_restores_first(self):
        base = f.base_module(); integration = Integration(); e = f.engine(); old_model = e.runner.model
        original_observer, original_run = base.Observer, base.run_case
        events = []
        with q.adapt(base, f.q, integration, f.old_module(), [], events.append) as coverage:
            observer = base.Observer(e, object(), 'candidate')
            scope = integration.scopes[0]
            self.assertIsInstance(e.runner.decode_model, base.CallProxy)
            self.assertIs(e.runner.decode_model.target, scope.graph_handle.decoder)
            self.assertIs(observer.route_install.proxy.witness, observer.route_witness)
            observer.begin(); self.assertEqual(observer.route_witness.step, 1)
            observer.restore(); observer.restore()
            self.assertTrue(scope.closed); self.assertIs(e.runner.model, old_model)
            self.assertIs(e.runner.decode_model, old_model)
            self.assertEqual(coverage['combined']['layer_installs'], 48)
            self.assertEqual(coverage['combined']['layer_restores'], 48)
        self.assertIs(base.Observer, original_observer); self.assertIs(base.run_case, original_run)

    def test_reference_is_untouched_and_private_harness_run_is_retained(self):
        base = f.base_module(); original = base.run_case; integration = Integration()
        with q.adapt(base, f.q, integration, f.old_module(), [], lambda x: None):
            observer = base.Observer(f.engine(), object(), 'reference'); observer.restore()
            self.assertEqual(integration.scopes, [])
        self.assertIs(base.run_case, original)

    def test_constructor_failure_unwinds_real_observer_then_combined_scope(self):
        base = f.base_module(); original = base.Observer; e = f.engine(); before = dict(vars(e.runner))
        class Broken(original):
            def __init__(self, *args):
                super().__init__(*args); raise RuntimeError('partial original Observer')
        base.Observer = Broken; integration = Integration()
        with q.adapt(base, f.q, integration, f.old_module(), [], lambda x: None):
            with self.assertRaisesRegex(RuntimeError, 'partial original'): base.Observer(e, object(), 'candidate')
        self.assertTrue(integration.scopes[0].closed)
        for key, value in before.items(): self.assertIs(vars(e.runner)[key], value)
        self.assertIs(base.Observer, Broken)

    def test_actual_compiled_invocation_counts_trace_once_then_zero_python_reuse(self):
        graph, proxy, witness, stats, events = proxy_fixture()
        tokens, ctx = SimpleNamespace(shape=(1,)), f.context()
        self.assertIs(witness(tokens, ctx), graph.output); self.assertIs(witness(tokens, ctx), graph.output)
        rows = [r for r in events if r['event'] == 'combined-graph-call']
        self.assertEqual([r['kind'] for r in rows], ['compiled-tracing', 'compiled-reuse'])
        self.assertEqual([r['per_layer_candidate_delta'] for r in rows], [1, 0])
        self.assertEqual([r['actual_invocations'] for r in rows], [1, 1])
        self.assertEqual(proxy._functions.invocations, 2)

    def test_batch_retrace_is_original_multirow_fallback_not_batched_candidate(self):
        graph, proxy, witness, stats, events = proxy_fixture(); graph.trace_passes = 2
        for B, used in ((1, 1023), (8, 1023), (1, 1024), (8, 1024), (8, 1024)):
            witness(SimpleNamespace(shape=(B,)), f.context(B, used))
        rows = [r for r in events if r['event'] == 'combined-graph-call']
        self.assertEqual([r['per_layer_candidate_delta'] for r in rows], [2, 0, 2, 0, 0])
        self.assertEqual([r['per_layer_fallback_delta'] for r in rows], [0, 2, 0, 2, 0])
        self.assertEqual(proxy.seen, {(1023, 1), (1023, 8), (1536, 1), (1536, 8)})

    def test_missing_or_mixed_trace_is_rejected_with_failed_raw_event(self):
        for setting in ('mixed', 'silent'):
            graph, proxy, witness, stats, events = proxy_fixture(); setattr(graph, setting, True)
            with self.assertRaises(ValueError): witness(SimpleNamespace(shape=(1,)), f.context())
            rows = [r for r in events if r['event'] == 'combined-graph-call']
            self.assertEqual(rows[-1]['status'], 'FAILED'); self.assertEqual(stats['compiled-reuse'], 0)
            self.assertEqual(stats['compiled-tracing'], 0)

    def test_callable_failure_is_not_retried_or_promoted(self):
        graph, proxy, witness, stats, events = proxy_fixture(); graph.fail = True
        with self.assertRaisesRegex(RuntimeError, 'actual callable'): witness(SimpleNamespace(shape=(1,)), f.context())
        self.assertEqual(proxy._functions.invocations, 1); self.assertEqual(stats['compiled-tracing'], 0)
        self.assertEqual(events[0]['status'], 'FAILED')

    def test_original42_and37_gates_stay_required_in_addition_to_combined_traces(self):
        completed, route = f.coverage_fixture()
        stats = dict(installs=2, restores=2, layer_installs=96, layer_restores=96,
            traced_batches=[dict(B=1, bound=1023), dict(B=1, bound=1536), dict(B=8, bound=1023)],
            **{'compiled-reuse': 1})
        coverage = dict(route=route, combined=stats)
        q.require_complete(f.q, completed, coverage)
        for key, value in (('parameter_direct_calls', 41), ('installs', 1)):
            bad = copy.deepcopy(coverage); bad['route'][key] = value
            with self.assertRaises(ValueError): q.require_complete(f.q, completed, bad)
        for key, value in (('layer_restores', 95), ('compiled-reuse', 0), ('traced_batches', [])):
            bad = copy.deepcopy(coverage); bad['combined'][key] = value
            with self.assertRaises(ValueError): q.require_complete(f.q, completed, bad)


if __name__ == '__main__': unittest.main()
