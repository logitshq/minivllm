"""Real frozen replay loop + CPU toy engine/observers, never MLX or services."""
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from bench.m5_ultra_256gb.c8_prefix_replay_v1 import prefix as p
from bench.m5_ultra_256gb.pareto_realistic import qualify_native as base, http_replay as replay
from bench.m5_ultra_256gb.pareto_realistic.test_qualify_cpu import CPU, Engine, SamplingParams
from bench.m5_ultra_256gb.pareto_realistic.test_http_replay_cpu import recording


class Prefix(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.cmp = base.Comparator(CPU(), self.temp.name); self.ledger = io.StringIO()
        self.engines = [Engine(8), Engine(8)]
        self.observers = [base.Observer(e, self.cmp, label) for e, label in zip(self.engines, ('reference', 'candidate'))]
        self.addCleanup(lambda: [o.restore() for o in self.observers])
        self.runner = SimpleNamespace(prefill_batch_backend_supported=lambda: True)
    def run_prefix(self, data, stop=2, deadline=lambda: None):
        return p.run_prefix(replay, data, stop, self.engines, self.observers, SamplingParams,
                            self.runner, self.cmp, self.ledger, deadline, base=base)
    def rows(self): return [json.loads(line) for line in self.ledger.getvalue().splitlines()]
    def test_boundary_follows_full_actual_checks_and_never_fakes_completion(self):
        data = recording(8); original = copy.deepcopy(data)
        result = self.run_prefix(data)
        self.assertEqual(data, original)
        self.assertEqual(result['status'], 'DIAGNOSTIC_PREFIX_COMPLETE')
        self.assertEqual(result['completion_state'], 'INCOMPLETE_BY_DESIGN')
        self.assertEqual(result['steps'], 2)
        self.assertEqual(result['completed'], [dict(rid='r1', raw_ids=[7, 7], committed_ids=[7, 7])])
        self.assertNotIn('r1', result['active_request_ids'])
        self.assertTrue(result['active_request_ids'])
        self.assertTrue(all(engine.has_work() for engine in self.engines))
        self.assertEqual([row['status'] for row in self.rows()], ['PASS', 'PASS'])
        labels = {a['check'] for row in self.rows() for a in row['arrays']}
        for label in ('before-step-state', 'before-retirement-state', 'after-retirement-state'):
            self.assertIn(label, labels)
        self.assertTrue(any(label.startswith('logits ') for label in labels))
        self.assertTrue(any(label.startswith('call-state ') for label in labels))
        self.assertTrue(all(o.calls == {} and o.postrun is None for o in self.observers))
        self.assertFalse(result['full_history_qualified'] or result['performance_qualified'])
    def test_recorded_token_mismatch_is_real_failure_not_boundary(self):
        data = recording(8); data['steps'][1]['results'][0][1] = 42
        with self.assertRaisesRegex(base.QualificationFailure, 'recorded delivered'): self.run_prefix(data)
        self.assertEqual(self.rows()[-1]['status'], 'RUNNING')
        self.assertEqual(self.engines[0].requests['r0'].output_token_ids, [7, 7])
    def test_actual_schedule_mismatch_precedes_candidate_model(self):
        data = recording(8); original = self.observers[1].original_schedule
        def reversed_schedule():
            out = original(); out.prefill.reverse(); return out
        self.engines[1].scheduler.schedule = reversed_schedule
        # Reinstall observer around the modified real scheduler method.
        self.observers[1].original_schedule = reversed_schedule
        self.engines[1].scheduler.schedule = self.observers[1].schedule_call
        with self.assertRaisesRegex(base.QualificationFailure, 'schedule-before-candidate-model'): self.run_prefix(data)
        self.assertEqual(self.observers[1].calls, {})
    def test_real_deadline_and_foreign_boundary_are_not_success(self):
        def stop(): raise TimeoutError('actual deadline')
        with self.assertRaisesRegex(TimeoutError, 'actual deadline'): self.run_prefix(recording(8), deadline=stop)
        foreign = p.PrefixBoundary('foreign')
        def other(): raise foreign
        with self.assertRaises(p.PrefixBoundary) as raised: self.run_prefix(recording(8), deadline=other)
        self.assertIs(raised.exception, foreign)
    def test_strict_prefix_and_original_finite_history_guard(self):
        data = recording(8)
        for stop in (0, len(data['steps']), True):
            with self.assertRaisesRegex(ValueError, 'incomplete prefix'): self.run_prefix(data, stop)
        data['steps'] *= 100
        with self.assertRaisesRegex(ValueError, 'finite progress'): self.run_prefix(data)
        self.assertEqual(self.rows(), [])


class ActualPlan(unittest.TestCase):
    def test_actual_warm49_plan_reconstructs_and_changed_boundary_is_rejected(self):
        child = p.A / 'c8-schedule-v1-screen/trials/c8-mb50-two-waves'
        analysis = p.A / 'c8-schedule-v1-analysis/analysis.json'
        plan = p.prepare(child, analysis); p.validate(plan)
        self.assertEqual(plan['stop_after_step'], 49)
        self.assertEqual(plan['recipe']['wave'], 'warm')
        self.assertEqual(plan['environment'], p.MLX_ENV)
        self.assertEqual(plan['recording'], p.ref(child / 'joined-input/recording.json'))
        plan['stop_after_step'] = 50
        with self.assertRaisesRegex(ValueError, 'changed prefix'): p.validate(plan)
    def test_default_integration_is_exact_existing_graph_only_source(self):
        q = p.load(p.ROUTE, p.ROUTE_SHA, '_prefix_cpu_contract')
        integration = p.integration_module(q)
        self.assertEqual(Path(integration.__file__), q.HERE / 'integration.py')
        self.assertFalse(integration.install(None, enabled=False))


if __name__ == '__main__': unittest.main()
