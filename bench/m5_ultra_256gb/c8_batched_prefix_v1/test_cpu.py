"""CPU-only contracts: real frozen observer seam, synthetic engine ownership."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from bench.m5_ultra_256gb.c8_batched_prefix_v1 import bridge as b, prefix as p


class Handle:
    def __init__(self, layer):
        self.layer = layer; self.wrapper = object(); self.calls = dict(candidate=0, fallback=0); self.closed = False
        layer._decode = self.wrapper
    def restore(self):
        assert self.layer._decode is self.wrapper
        del self.layer._decode; self.layer = self.wrapper = None; self.closed = True


class Combined:
    def __init__(self): self.engines = []; self.scopes = []
    def install(self, engine, *, gdn_prework, route_graph):
        assert gdn_prework is route_graph is True
        self.engines.append(engine); runner = engine.runner; short = runner.decode_model
        handles = [Handle(x.linear_attn) for x in runner.model.layers if not x.is_attn]
        decoder = NS(long=NS(_functions={}), original=short)
        scope = NS(runner=runner, model=runner.model, short=short, closed=False,
                   gdn_handles=handles, graph_handle=NS(decoder=decoder))
        def restore():
            assert runner.model is scope.model and runner.decode_model is decoder and 'run' not in vars(runner)
            decoder.long._functions.clear(); runner.decode_model = short
            short._functions.clear(); short._packed = None
            for h in reversed(handles): h.restore()
            scope.gdn_handles = []; scope.graph_handle = None
            scope.runner = scope.model = scope.short = None; scope.closed = True
        scope.restore = restore; runner.decode_model = decoder; self.scopes.append(scope)
        return scope


def engine():
    layers = [NS(is_attn=(i % 4 == 3), linear_attn=NS()) for i in range(64)]
    model = NS(layers=layers)
    return NS(model=model, runner=NS(model=model, decode_model=NS(_functions={}, _packed=None)), scheduler=NS())


class BaseObserver:
    fail = False
    def __init__(self, engine, comparator, label):
        self.engine = engine; runner = engine.runner
        self.original_model, self.original_decode = runner.model, runner.decode_model
        self.original_run = self.original_schedule = None
        self.calls = {}; self.previous_future = self.postrun = None
        runner.model = object(); runner.decode_model = object(); runner.run = object(); engine.scheduler.schedule = object()
        if self.fail: raise RuntimeError('base constructor failure')
    def restore(self):
        self.engine.runner.model = self.original_model; self.engine.runner.decode_model = self.original_decode
        vars(self.engine.runner).pop('run', None); vars(self.engine.scheduler).pop('schedule', None)


class Witness:
    def __init__(self, graph, *unused):
        self.functions = {}; graph._functions = self.functions


class BridgeTests(unittest.TestCase):
    def setUp(self):
        self.q = p.load(p.B/'route_decode_graph/http_qualification_v2.py',
            'd3e2a4f76b7c384f6c6c986d0838b68227618502b0552e5cdf74f149029e39e7', '_e050_test_route')
        self.base = NS(Observer=BaseObserver, workloads=object(), run_case=object())
        self.original_observer = self.base.Observer; self.original_adapted = self.q.adapted
        self.combined = Combined(); self.events = []; self.bridge = b.Bridge(self.combined, self.events.append)
        self.bridge.integration_module(self.q)
        self.addCleanup(self.bridge.close_private_hook)
    def context(self):
        return self.q.adapted(self.base, None, self.bridge, NS(RouteWitness=Witness), None,
                              {'concurrency': 8}, [], NS(emit=lambda x: None))
    def test_off_is_inert_and_switch_requires_bool(self):
        self.assertIsNone(self.bridge.install(None))
        self.assertEqual(self.combined.engines, [])
        with self.assertRaises(TypeError): self.bridge.install(None, enabled=1)
    def test_unbound_or_wrong_runner_rejected(self):
        actual = engine()
        with self.assertRaisesRegex(ValueError, 'actual currently'): self.bridge.install(actual.runner, enabled=True)
        self.bridge._engine = actual
        try:
            with self.assertRaisesRegex(ValueError, 'actual currently'): self.bridge.install(engine().runner, enabled=True)
        finally: self.bridge._engine = None
        self.assertEqual(self.combined.engines, [])
    def test_real_outer_observer_passes_engine_before_install_and_restores(self):
        reference, candidate = engine(), engine()
        with self.context() as coverage:
            ref = self.base.Observer(reference, None, 'reference')
            self.assertEqual(self.combined.engines, [])
            obs = self.base.Observer(candidate, None, 'candidate')
            facade = obs.route_install
            self.assertIs(self.combined.engines[0], candidate)
            self.assertIs(facade.decoder, self.combined.scopes[0].graph_handle.decoder)
            self.assertIsNone(self.bridge._engine)
            self.assertEqual(len(facade.handles), 48)
            for handle in facade.handles: handle.calls['candidate'] = 2
            ref.restore(); obs.restore()
            self.assertEqual(coverage['installs'], 1); self.assertEqual(coverage['restores'], 2)
            result = self.bridge.checked()
            self.assertEqual(result['handle_count'], 48)
            rows = result['records'][0]['layers']
            self.assertEqual([r['layer'] for r in rows], [i for i in range(64) if i % 4 != 3])
            self.assertEqual(len({r['handle_id'] for r in rows}), 48)
            self.assertEqual(len({r['layer_id'] for r in rows}), 48)
            self.assertTrue(all(r['final_counts'] == {'candidate': 2, 'fallback': 0} for r in rows))
            self.assertIsNone(facade.decoder); self.assertIsNone(facade.engine)
            facade.restore()  # Idempotent after references are released.
        self.assertIs(self.base.Observer, self.original_observer)
        self.bridge.close_private_hook(); self.assertIs(self.q.adapted, self.original_adapted)
        self.assertEqual([e['event'] for e in self.events], ['combined-installed', 'combined-restored'])
    def test_constructor_failure_preserves_original_error_and_cleans_scope(self):
        with self.context(), patch.object(BaseObserver, 'fail', True):
            with self.assertRaisesRegex(RuntimeError, 'base constructor failure'):
                self.base.Observer(engine(), None, 'candidate')
        self.assertIsNone(self.bridge._engine)
        self.assertTrue(self.combined.scopes[0].closed)
        self.assertEqual(self.bridge.snapshot()['restores'], 1)
        with self.assertRaisesRegex(ValueError, 'every GDN layer'): self.bridge.checked()
        self.assertIs(self.base.Observer, self.original_observer)
    def test_fallback_does_not_prevent_cleanup_but_disqualifies(self):
        with self.context():
            obs = self.base.Observer(engine(), None, 'candidate')
            for h in obs.route_install.handles: h.calls['candidate'] = 1
            obs.route_install.handles[0].calls['fallback'] = 1
            obs.restore()
        self.assertTrue(self.combined.scopes[0].closed)
        with self.assertRaisesRegex(ValueError, 'without fallback'): self.bridge.checked()
    def test_foreign_ownership_failure_is_retained_and_retry_can_restore(self):
        with self.context():
            obs = self.base.Observer(engine(), None, 'candidate'); facade = obs.route_install
            # Original observer is restored before exercising the facade's owned decoder check.
            BaseObserver.restore(obs)
            facade.decoder.long = obs.graph; obs.graph._functions = obs.saved_functions
            facade.runner.decode_model = object()
            with self.assertRaises(AssertionError): facade.restore()
            self.assertEqual(facade.record['status'], 'RESTORE_FAILED')
            facade.runner.decode_model = facade.decoder
            for h in facade.handles: h.calls['candidate'] = 1
            facade.restore(); self.bridge.checked()
        self.assertIn('combined-restore-failed', [e['event'] for e in self.events])
    def test_reentrant_constructor_rejected(self):
        with self.context():
            self.bridge._engine = engine()
            try:
                with self.assertRaisesRegex(ValueError, 'reentry'): self.base.Observer(engine(), None, 'candidate')
            finally: self.bridge._engine = None
        self.assertEqual(self.combined.engines, [])


class PrefixTests(unittest.TestCase):
    def test_inherited_function_uses_private_globals_only(self):
        namespace = dict(integration_module=lambda q: 'old', validate=lambda p: None, SCOPE='old')
        exec('def run(plan, out):\n validate(plan)\n return integration_module(out), SCOPE\n', namespace)
        original = NS(**namespace); calls = []
        bridge = NS(integration_module=lambda q: ('new', q))
        fn = p.inherited_run(original, bridge, calls.append)
        self.assertIs(fn.__code__, original.run.__code__)
        self.assertEqual(fn('plan', 'q'), (('new', 'q'), p.SCOPE))
        self.assertEqual(calls, ['plan']); self.assertEqual(original.run('plan', 'q'), ('old', 'old'))
    def test_predecessor_rejects_incomplete_failed_or_different_plan(self):
        top = dict(schema='c8-native-graph-prefix-result-v1', status='DIAGNOSTIC_PREFIX_COMPLETE', plan_sha256='abc',
            source_input_pins_rechecked=True, completion_state='INCOMPLETE_BY_DESIGN', full_history_qualified=False,
            performance_qualified=False, base_result=dict(returncode=0, status='PASS'),
            prefix=dict(status='DIAGNOSTIC_PREFIX_COMPLETE', steps=49, completed=[],
                        active_request_ids=list(range(8)), future_unadmitted_request_ids=list(range(8))),
            coverage=dict(installs=1, restores=2), route_prefix=dict(full_route_stream_consumed=False))
        inner = dict(status='PASS', source_model_pins_rechecked=True, cases=[top['prefix']], total_requests=0)
        p.predecessor_ok(top, inner, 'abc')
        for key, value in [('status','FAIL'), ('plan_sha256','wrong'), ('full_history_qualified',True)]:
            changed = copy.deepcopy(top); changed[key] = value
            with self.assertRaises(ValueError): p.predecessor_ok(changed, inner, 'abc')
        changed = copy.deepcopy(top); changed['prefix']['steps'] = 50
        with self.assertRaises(ValueError): p.predecessor_ok(changed, inner, 'abc')
    def test_actual_sources_are_stdlib_and_combined_default_off(self):
        combined = p.load(p.COMBINED, p.COMBINED_SHA, '_e050_cpu_combined')
        self.assertIsNone(combined.install(None))
        pins = combined.source_pins()
        self.assertEqual(pins[str(p.B/'gdn_prework_batched/candidate.py')],
                         'acd2a23802296b9e420108203cc215c3170cd1eccd9a859e1a787432f98a5353')


if __name__ == '__main__': unittest.main()
