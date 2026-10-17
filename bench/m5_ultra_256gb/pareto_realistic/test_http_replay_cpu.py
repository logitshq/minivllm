"""Mechanism tests: real CPU Scheduler, toy NumPy one-layer model, no MLX/HTTP.

The recording fixture is produced independently by one toy engine. Replay uses
two new toy engines plus the unchanged qualifier Observer/Comparator. These
tests do not qualify any real model, compiled route or HTTP performance.
"""
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest

from bench.m5_ultra_256gb.pareto_realistic import qualify_native as base
from bench.m5_ultra_256gb.pareto_realistic.http_replay import run_recorded_case
from bench.m5_ultra_256gb.pareto_realistic.test_qualify_cpu import CPU, Engine, SamplingParams


def recording(concurrency=1, *, early_eos=False):
    """Three requests, including a later admission; tiny caps explicitly mocks."""
    engine = Engine(concurrency)
    if early_eos:
        engine.eos_ids = {7}
        engine.scheduler.eos_token_ids = {7}
    caps = [3, 2, 2]
    jobs = {f'r{i}': {'rid': f'r{i}', 'client_request_id': f'client-{i}',
        'fixture_id': f'D{i}', 'prompt_ids': [1, 2 + i], 'cap': caps[i],
        'raw_ids': [], 'committed_ids': []} for i in range(3)}
    result = {'schema': 'E023-recorded-http-v1', 'concurrency': concurrency,
              'eos_token_ids': sorted(engine.eos_ids), 'jobs': jobs, 'steps': [], 'histogram': {}}
    pending = list(jobs)
    def schedule():
        out = original_schedule()
        observed['schedule'] = {k: base.schedule_metadata(engine, out)[k]
                               for k in ('decode', 'prefill', 'finished_ids')}
        return out
    def run(out):
        sampled = original_run(out)
        observed['postrun_decode_rows'] = list(engine.runner.batch.req_ids)
        return sampled
    original_schedule, original_run = engine.scheduler.schedule, engine.runner.run
    engine.scheduler.schedule, engine.runner.run = schedule, run
    while pending or engine.has_work():
        observed = {'step': len(result['steps']) + 1, 'arrivals': []}
        # C8 starts two, then admits the third while old requests remain live.
        budget = (2 if not result['steps'] else 1) if concurrency == 8 else (not engine.has_work())
        for _ in range(min(int(budget), len(pending))):
            rid = pending.pop(0); job = jobs[rid]
            engine.add_request(job['prompt_ids'], SamplingParams(max_tokens=job['cap'],
                               temperature=0.0, ignore_eos=False), req_id=rid)
            observed['arrivals'].append(rid)
        observed['before'] = {'running': [r.req_id for r in engine.scheduler.running],
                              'waiting': [r.req_id for r in engine.scheduler.waiting]}
        delivered = engine.step()
        observed.update(results=base.plain(delivered), retired=[rid for rid, _, done in delivered if done],
                        surviving_decode_rows=list(engine.runner.batch.req_ids))
        B = str(len(observed['schedule']['decode']))
        result['histogram'][B] = result['histogram'].get(B, 0) + 1
        for rid, token, done in delivered:
            jobs[rid]['raw_ids'].append(token)
            if done:
                ids = jobs[rid]['raw_ids']
                jobs[rid]['committed_ids'] = ids[:-1] if token in engine.eos_ids else list(ids)
        result['steps'].append(observed)
    return result


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.cmp = base.Comparator(CPU(), self.temp.name)
        self.ledger = io.StringIO()
        self.runner_module = NS(prefill_batch_backend_supported=lambda: True)
        self.observers = []

    def tearDown(self):
        for observer in self.observers:
            observer.restore()
        self.temp.cleanup()

    def replay(self, data, *, engines=None, deadline=lambda: None):
        self.engines = engines or [Engine(data['concurrency']), Engine(data['concurrency'])]
        for engine in self.engines:
            engine.eos_ids = set(data['eos_token_ids'])
            engine.scheduler.eos_token_ids = set(data['eos_token_ids'])
        self.observers = [base.Observer(engine, self.cmp, arm)
                          for engine, arm in zip(self.engines, ('reference', 'candidate'))]
        return run_recorded_case(data, self.engines, self.observers, SamplingParams,
                                 self.runner_module, self.cmp, self.ledger, deadline, base=base)

    def rows(self):
        return [json.loads(line) for line in self.ledger.getvalue().splitlines()]

    def test_c1_and_c8_actual_admissions_schedules_and_full_comparisons(self):
        for C in (1, 8):
            with self.subTest(C=C):
                data = recording(C)
                self.ledger = io.StringIO()
                result = self.replay(data)
                self.assertEqual(result['status'], 'PASS')
                self.assertEqual(result['raw_output_count'], 7)
                self.assertEqual(result['decode_batch_histogram'], data['histogram'])
                self.assertFalse(result['performance_qualified'])
                self.assertEqual([x['before_step'] for x in result['admissions']],
                    [s['step'] for s in data['steps'] for _ in s['arrivals']])
                labels = {a['check'] for r in self.rows() for a in r['arrays']}
                self.assertTrue(any(x.startswith('logits ') for x in labels))
                self.assertTrue(any(x.startswith('call-state ') for x in labels))
                self.assertIn('before-retirement-state', labels)
                self.assertIn('after-retirement-state', labels)
                self.assertTrue(self.runner_module.prefill_batch_backend_supported())
                for observer in self.observers:
                    self.assertEqual(observer.calls, {})
                    self.assertIsNone(observer.postrun)
                    self.assertIsNone(observer.previous_future)
                    observer.restore()
                self.observers = []

    def test_early_eos_retains_raw_and_excludes_only_terminal_committed_token(self):
        result = self.replay(recording(early_eos=True))
        self.assertEqual(result['raw_output_count'], 3)
        self.assertEqual(result['committed_output_count'], 0)
        self.assertTrue(all(x['raw_ids'] == [7] and x['committed_ids'] == [] for x in result['completed']))

    def test_recorded_tokens_are_targets_never_engine_inputs(self):
        data = recording()
        data['steps'][0]['results'][0][1] = 42
        with self.assertRaisesRegex(base.QualificationFailure, 'recorded delivered'):
            self.replay(data)
        self.assertEqual(self.engines[0].requests['r0'].output_token_ids, [7])
        self.assertEqual(self.engines[1].requests['r0'].output_token_ids, [7])
        self.assertEqual(self.rows()[-1]['reference_delivered'][0][1], 7)

    def test_before_schedule_postrun_retirement_and_survivor_corruptions_fail(self):
        mutations = [
            lambda d: d['steps'][0]['before']['waiting'].reverse(),
            lambda d: d['steps'][0]['schedule']['prefill'].reverse(),
            lambda d: d['steps'][1]['postrun_decode_rows'].reverse(),
            lambda d: d['steps'][1]['surviving_decode_rows'].append('wrong-survivor'),
            lambda d: d['steps'][1]['retired'].append('r0'),
        ]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                self.cmp.first_failure = None; self.ledger = io.StringIO()
                data = recording(8); mutate(data)
                with self.assertRaises(base.QualificationFailure):
                    self.replay(data)
                self.assertNotEqual(self.rows()[-1]['status'], 'PASS')
                for observer in self.observers:
                    observer.restore()
                self.observers = []

    def test_candidate_independently_schedules_and_mismatch_precedes_model(self):
        data = recording(8)
        engines = [Engine(8), Engine(8)]
        original = engines[1].scheduler.schedule
        def reversed_schedule():
            out = original(); out.prefill.reverse(); return out
        engines[1].scheduler.schedule = reversed_schedule
        with self.assertRaisesRegex(base.QualificationFailure, 'schedule-before-candidate-model'):
            self.replay(data, engines=engines)
        self.assertEqual(self.observers[1].calls, {})
        self.assertTrue(self.observers[0].calls)

    def test_final_sse_raw_and_committed_id_mismatches_fail(self):
        for field in ('raw_ids', 'committed_ids'):
            with self.subTest(field=field):
                self.cmp.first_failure = None; self.ledger = io.StringIO()
                data = recording(); data['jobs']['r0'][field][-1] = 8
                with self.assertRaisesRegex(base.QualificationFailure, 'final HTTP'):
                    self.replay(data)
                for observer in self.observers:
                    observer.restore()
                self.observers = []

    def test_deadline_retains_successful_prefix_and_primary_exception(self):
        def deadline():
            if len(self.rows()) == 1:
                raise TimeoutError('fixture absolute deadline')
        with self.assertRaisesRegex(TimeoutError, 'fixture absolute deadline'):
            self.replay(recording(), deadline=deadline)
        rows = self.rows()
        self.assertEqual(rows[0]['status'], 'PASS')
        self.assertEqual(rows[1]['error']['type'], 'TimeoutError')
        self.assertEqual(rows[1]['status'], 'RUNNING')

    def test_truncated_recording_fails_drain_without_inventing_more_steps(self):
        data = recording(); data['steps'].pop()
        with self.assertRaisesRegex(base.QualificationFailure, 'engine failed to drain'):
            self.replay(data)
        self.assertEqual(len(self.rows()), len(data['steps']))

    def test_finite_step_bound_rejects_before_engine_work(self):
        data = recording(); data['steps'] *= 100
        with self.assertRaisesRegex(ValueError, 'finite progress step bound'):
            self.replay(data)
        self.assertEqual(self.rows(), [])
        self.assertFalse(self.engines[0].requests)

    def test_c8_requires_actual_multirow_not_concurrency_label(self):
        data = recording(); data['concurrency'] = 8
        with self.assertRaisesRegex(base.QualificationFailure, 'multi-row decode'):
            self.replay(data)

    def test_cpu_import_has_no_model_dependencies(self):
        path = Path(__file__).with_name('http_replay.py')
        script = '''import importlib.abc, importlib.util, sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self, fullname, *args):
  if fullname.split('.')[0] in {'mlx','numpy','tutorial_code'}:
   raise AssertionError('forbidden import '+fullname)
sys.meta_path.insert(0,Block())
spec=importlib.util.spec_from_file_location('replay',sys.argv[1])
module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
assert callable(module.run_recorded_case)
'''
        subprocess.run([sys.executable, '-I', '-B', '-c', script, str(path)], check=True,
                       capture_output=True, text=True, timeout=10)


if __name__ == '__main__':
    unittest.main()
