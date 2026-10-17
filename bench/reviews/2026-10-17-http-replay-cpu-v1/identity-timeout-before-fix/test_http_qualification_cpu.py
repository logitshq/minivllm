"""CPU plan/adapter routing tests, not model or HTTP qualification."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('_http_qualification_cpu_target', HERE / 'http_qualification.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
BASE_PLAN = HERE / 'prereg-qualification-v1/plan.json'


class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.group, self.trace = self.root / 'group.json', self.root / 'trace.jsonl'
        self.group.write_text('{}\n'); self.trace.write_text('')
        self.plan = {'schema': 'E023-recorded-http-plan-v1',
            'qualification': {'path': str(BASE_PLAN), 'sha256': m.BASE_PLAN_SHA},
            'concurrency': 1, 'transport': {'tcp_nodelay': False, 'io_timeout_seconds': 300},
            'inputs': {k: {'path': str(p), 'sha256': m.sha(p)} for k, p in [('group', self.group), ('trace', self.trace)]},
            'sources': {str(p): m.sha(p) for p in (*m.FIXED, *m.ADAPTERS)}}
        self.path = self.root / 'plan.json'; self.save()
        self.numeric = m.read(BASE_PLAN)
        self.fixtures = m.read(self.numeric['fixtures']['path'])['fixtures']

    def save(self):
        m.write(self.path, self.plan)
        return m.sha(self.path)

    def test_fixed_job_sequence_full_prompts_and_counts(self):
        for c, count in [(1, 8), (8, 16)]:
            jobs = m.jobs_for(self.fixtures, c)
            self.assertEqual(len(jobs), count)
            self.assertEqual(len({j['request_id'] for j in jobs}), count)
            for i, job in enumerate(jobs):
                self.assertEqual(job['fixture_id'], self.fixtures[i % 8]['fixture_id'])
                self.assertEqual(job['body']['prompt'], self.fixtures[i % 8]['prompt_text'])
                self.assertEqual(job['body']['max_tokens'], 512)
                self.assertEqual(job['body']['model'], 'native-target-only')
        with self.assertRaises(ValueError): m.jobs_for(self.fixtures, True)

    def test_real_source_and_base_plan_pins_without_runtime_imports(self):
        plan, numeric = m.validate(self.path, m.sha(self.path))
        self.assertEqual(plan, self.plan); self.assertEqual(numeric, self.numeric)
        self.assertEqual(numeric['deadline_s'], 1800)

    def test_changed_capture_and_source_rejected(self):
        self.group.write_text('{"changed": true}\n')
        with self.assertRaisesRegex(ValueError, 'input SHA'): m.validate(self.path, m.sha(self.path))
        self.group.write_text('{}\n')
        self.plan['sources'][str(HERE / 'http_join.py')] = '0' * 64
        digest = self.save()
        with self.assertRaisesRegex(ValueError, 'source changed'): m.validate(self.path, digest)

    def test_invalid_transport_or_unqualified_concurrency(self):
        original = copy.deepcopy(self.plan)
        for key, value in [('concurrency', 2), ('concurrency', True),
                           ('transport', {'tcp_nodelay': False, 'io_timeout_seconds': True}),
                           ('transport', {'tcp_nodelay': False, 'io_timeout_seconds': 0})]:
            self.plan = copy.deepcopy(original); self.plan[key] = value
            digest = self.save()
            with self.assertRaises(ValueError): m.validate(self.path, digest)

    def test_expected_identity_keeps_actual_server_and_case_configuration(self):
        source = m.read(self.numeric['source_manifest']['path'])
        model = m.read(self.numeric['model_manifest']['path'])
        identity = m.expected_identity(self.numeric, self.fixtures, source, model, 8, self.plan['transport'])
        self.assertEqual(identity['configuration']['max_seqs'], 8)
        self.assertEqual(identity['configuration']['max_batched_tokens'], 2048)
        self.assertEqual(identity['configuration']['compile_decode'], 'on')
        self.assertEqual(identity['configuration']['norm_reuse'], 'off')
        self.assertTrue(identity['step_observer']); self.assertFalse(identity['prefix_cache'])
        self.assertEqual(identity['server_source_sha256'], m.FIXED[HERE / 'native_server.py'])

    def test_prepare_only_cannot_run_numerical_loader_or_grant_qualification(self):
        recording = {'schema': 'CPU_MOCK_ONLY', 'concurrency': 1}
        output = self.root / 'prepared'
        with patch.object(m, 'prepare', return_value=(recording, self.fixtures)), patch.object(m, 'load', side_effect=AssertionError('must not load numerical code')):
            code = m.main(['--plan', str(self.path), '--plan-sha256', self.save(), '--out', str(output), '--prepare-only'])
        self.assertEqual(code, 0)
        result = m.read(output / 'complete.json')
        self.assertEqual(result['status'], 'PREPARED')
        self.assertFalse(result['numerical_qualified']); self.assertFalse(result['performance_qualified'])
        self.assertFalse((output / 'base').exists())

    def test_one_case_override_calls_actual_replay_interface_and_restores_hooks(self):
        called = []
        recording = {'concurrency': 8}
        old_workloads, old_run = object(), object()
        base = SimpleNamespace(workloads=old_workloads, run_case=old_run)
        arguments = [object() for _ in range(7)]
        def replay(recording_arg, *args, **kwargs):
            self.assertIs(recording_arg, recording)
            self.assertEqual(list(args), arguments)
            self.assertIs(kwargs['base'], base)
            called.append('replay')
            return {'name': 'D-C8'}
        def main(argv):
            cases = base.workloads(self.fixtures)
            self.assertEqual(cases, [{'name': 'D-C8', 'concurrency': 8}])
            self.assertEqual(base.run_case(cases[0], *arguments), {'name': 'D-C8'})
            self.assertEqual(argv[3], m.BASE_PLAN_SHA)
            return 0
        base.main = main
        result = m.run_adapter(base, SimpleNamespace(run_recorded_case=replay), recording, self.fixtures,
                               self.plan['qualification'], self.root / 'base')
        self.assertEqual(result, 0); self.assertEqual(called, ['replay'])
        self.assertIs(base.workloads, old_workloads); self.assertIs(base.run_case, old_run)
        def broken(argv):
            raise RuntimeError('simulated base failure')
        base.main = broken
        with self.assertRaisesRegex(RuntimeError, 'simulated base failure'):
            m.run_adapter(base, SimpleNamespace(), recording, self.fixtures, self.plan['qualification'], self.root / 'base')
        self.assertIs(base.workloads, old_workloads); self.assertIs(base.run_case, old_run)


if __name__ == '__main__':
    unittest.main()
