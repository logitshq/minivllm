"""Offline synthetic owner corruption tests; never claim a real E028 PASS."""
import copy
import importlib.util
import io
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


m = load(HERE / 'audit_model_owner.py', 'e028_owner_audit_tested')
fakes = load(HERE.parent / 'pareto_realistic/test_audit_absent_cpu.py', 'e028_existing_recovery_fixtures')


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fakes.AuditTests(methodName='runTest')
        self.fixture.setUp(); self.addCleanup(self.fixture.doCleanups)
        self.owner, self.journal = self.fixture.owner, self.fixture.journal
        self.a = m.load(m.SUPPORT, m.SUPPORT_SHA, '_e028_resource_fixture_audit')
        self.registered = m.registration(self.a)
        # All rewritten objects below are CPU SYNTHETIC fixtures in temp storage.
        # Production registration remains fixed to the real immutable plan hashes.
        self.registered = copy.deepcopy(self.registered)
        self.registered['wrapper']['cold_restart']['journal_path'] = str(self.journal)
        write(self.owner / 'owner-plan.json', self.registered['wrapper'])
        self.registered['wrapper_sha'] = m.sha(self.owner / 'owner-plan.json')
        (self.owner / 'owner-source/model_owner.py').write_bytes((HERE / 'model_owner.py').read_bytes())
        nested, child = self.owner / 'e023-input-preflight', self.owner / 'qualification'
        nested.mkdir(); (child / 'base').mkdir()
        (self.owner / 'plan.json').rename(nested / 'plan.json')
        (child / 'plan.json').rename(child / 'base/plan.json')
        (child / 'complete.json').rename(child / 'base/complete.json')
        (child / 'steps.jsonl').rename(child / 'base/steps.jsonl')
        raw = (m.PREREG / 'qualification-plan.json').read_bytes()
        (self.owner / 'qualification-plan.json').write_bytes(raw); (child / 'plan.json').write_bytes(raw)
        write(child / 'complete.json', {'status': 'PASS', 'CPU_SYNTHETIC_NUMERICAL_RESULT': True})
        self.owner_source = self.registered['wrapper']['execution']['owner']
        self.data = copy.deepcopy(self.fixture.data)
        self.data.update(owner_source=self.owner_source,
            owner_plan={'path': self.registered['wrapper_path'], 'sha256': self.registered['wrapper_sha']})
        stopped = self.a.read(self.owner / 'stopped-journal.json')
        stopped.update(owner_source=self.owner_source, owner_plan=self.data['owner_plan'])
        write(self.owner / 'stopped-journal.json', stopped)
        attempt = self.a.read(self.owner / 'attempt.json')
        attempt.update(owner_source=self.owner_source, owner_plan_sha256=self.registered['wrapper_sha'],
            stopped_journal_sha256=m.sha(self.owner / 'stopped-journal.json'))
        write(self.owner / 'attempt.json', attempt)
        intent = self.a.read(self.owner / 'spawn-intent.json')
        intent.update(owner_plan_sha256=self.registered['wrapper_sha'], attempt_sha256=m.sha(self.owner / 'attempt.json'))
        write(self.owner / 'spawn-intent.json', intent)
        complete = self.a.read(self.owner / 'complete.json')
        complete.update(owner_plan_sha256=self.registered['wrapper_sha'], numerical_plan_sha256=m.CHILD_PLAN_SHA)
        write(self.owner / 'complete.json', complete)
        numeric = self.a.read(nested / 'plan.json'); q = self.registered['wrapper']['qualification']
        write(self.owner / 'launch.json', {'command': [numeric['execution']['python'], '-I', '-B', str(HERE / 'model_qualification.py'),
            '--plan', q['path'], '--plan-sha256', m.CHILD_PLAN_SHA, '--out', str(child)],
            'environment': numeric['execution']['environment'], 'numerical_plan_sha256': m.CHILD_PLAN_SHA})
        self.data['reap']['receipts'] = [{'path': str(p), 'sha256': m.sha(p)} for p in
            [self.owner / n for n in ('attempt.json', 'stopped-journal.json', 'spawn-intent.json', 'child.json', 'release.json')]]
        write(self.journal / 'journal.json', self.data)
        self.numeric_calls = []
        def numerical(path):
            self.numeric_calls.append(path)
            return {'status': 'PASS', 'qualified': True, 'ledger_opened': True, 'CPU_MOCK_ONLY': True}
        for context in (patch.object(m, 'registration', return_value=self.registered),
                        patch.object(m, 'preflight_pins', return_value={'CPU_MOCK_ONLY': True}),
                        patch.object(m, 'numerical_auditor', return_value=NS(audit=numerical))):
            context.start(); self.addCleanup(context.stop)

    def mutate(self, path, fn):
        value = self.a.read(path); fn(value); write(path, value)

    def run_audit(self):
        return m.audit(self.owner, self.journal)

    def test_nested_routing_with_original_resource_and_recovery_validators(self):
        result = self.run_audit()
        self.assertEqual(result['status'], 'PASS', result)
        self.assertTrue(result['qualified'] and result['ledger_opened'])
        self.assertFalse(result['performance_qualified'])
        self.assertEqual(self.numeric_calls, [self.owner / 'qualification'])
        self.assertEqual(result['owner_resources']['pressure'], 1)

    def test_prepared_journal_never_opens_numerical_ledger(self):
        self.mutate(self.journal / 'journal.json', lambda x: x.update(state='PREPARED'))
        result = self.run_audit()
        self.assertEqual(result['status'], 'INCOMPLETE')
        self.assertFalse(result['ledger_opened']); self.assertEqual(self.numeric_calls, [])

    def test_missing_child_release_is_incomplete_without_numerical_read(self):
        (self.owner / 'release.json').unlink()
        result = self.run_audit()
        self.assertEqual(result['status'], 'INCOMPLETE')
        self.assertEqual(self.numeric_calls, [])

    def test_wrong_new_child_cli_fails_without_numerical_read(self):
        self.mutate(self.owner / 'launch.json', lambda x: x['command'].__setitem__(3, 'qualify_native.py'))
        result = self.run_audit()
        self.assertEqual(result['status'], 'FAIL')
        self.assertIn('CLI', result['first_failure']['message'])
        self.assertEqual(self.numeric_calls, [])

    def test_child_parent_pid_and_restoration_hash_cannot_be_forged(self):
        self.mutate(self.owner / 'child.json', lambda x: x.update(owner_pid=1))
        result = self.run_audit()
        self.assertEqual(result['status'], 'FAIL')
        self.assertIn('direct child', result['first_failure']['message'])
        self.assertEqual(self.numeric_calls, [])

    def test_unchanged_pressure_one_guard_rejects_pressure_two(self):
        self.mutate(self.owner / 'resource-002.json', lambda x: x.update(pressure_level_value=2))
        result = self.run_audit()
        self.assertEqual(result['status'], 'FAIL')
        self.assertIn('pressure', result['first_failure']['message'])
        self.assertEqual(self.numeric_calls, [])

    def test_runtime_pins_failure_precedes_numerical_read(self):
        with patch.object(m, 'preflight_pins', side_effect=ValueError('bad frozen source')):
            result = self.run_audit()
        self.assertEqual(result['status'], 'FAIL')
        self.assertEqual(self.numeric_calls, [])

    def test_changed_resource_auditor_is_failure_without_ledger(self):
        with patch.object(m, 'load', side_effect=ValueError('independent auditor source changed')):
            result = self.run_audit()
        self.assertEqual(result['status'], 'FAIL')
        self.assertFalse(result['ledger_opened']); self.assertEqual(self.numeric_calls, [])

    def test_restore_raw_log_corruption_fails_before_numerical_read(self):
        Path(self.data['log_window']['path']).write_text('forged load log')
        result = self.run_audit()
        self.assertEqual(result['status'], 'FAIL')
        self.assertEqual(self.numeric_calls, [])


class PreflightTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.owner = Path(self.tmp.name) / 'prepare'
        source = HERE.parents[2] / '.artifacts/m5-ultra-256gb/identity-refill-model-v1-cpu-prepare'
        shutil.copytree(source, self.owner)
        self.a = m.load(m.SUPPORT, m.SUPPORT_SHA, '_e028_pin_audit')
        self.numeric = self.a.read(self.owner / 'e023-input-preflight/plan.json')
        self.child = self.a.read(self.owner / 'qualification-plan.json')
        verified = self.a.read(self.owner / 'e023-input-preflight/verified-pins.json')
        runtime = {p: r['sha256'] for p, r in verified.items() if not r['source_copied']}
        real_sha = m.sha
        # Frozen source copies are really hashed. Runtime content is explicitly
        # mocked in CPU tests so 225+ MB binaries are not reread per corruption.
        context = patch.object(m, 'sha', side_effect=lambda p: runtime[str(p)] if str(p) in runtime else real_sha(p))
        context.start(); self.addCleanup(context.stop)

    def test_actual_nested_prepare_paths_and_all_source_copies(self):
        result = m.preflight_pins(self.owner, self.numeric, self.child, self.a)
        self.assertEqual(result['original_verified_entries'], 157)
        self.assertEqual(result['adapter_source_copies'], 5)
        self.assertGreater(result['runtime_bytes_rehashed'], 200 << 20)  # Content hash is mocked above.

    def test_missing_original_pin_is_not_accepted_as_prepared(self):
        path = self.owner / 'e023-input-preflight/verified-pins.json'
        rows = self.a.read(path); rows.pop(next(iter(rows))); write(path, rows)
        with self.assertRaisesRegex(ValueError, 'exact157'): m.preflight_pins(self.owner, self.numeric, self.child, self.a)

    def test_same_size_adapter_source_corruption_rejected(self):
        path = self.owner / 'frozen-source' / str(HERE / 'candidate.py').lstrip('/')
        raw = bytearray(path.read_bytes()); raw[-1] ^= 1; path.write_bytes(raw)
        with self.assertRaisesRegex(ValueError, 'adapter source copy'): m.preflight_pins(self.owner, self.numeric, self.child, self.a)


class SourceInterfaceTests(unittest.TestCase):
    def test_actual_preregistered_plan_hashes(self):
        a = m.load(m.SUPPORT, m.SUPPORT_SHA, '_e028_real_prereg_audit')
        registered = m.registration(a)
        self.assertEqual(registered['wrapper_sha'], m.WRAPPER_SHA)
        self.assertEqual(registered['child_sha'], m.CHILD_PLAN_SHA)

    def test_pinned_numerical_auditor_missing_terminal_does_not_read_ledger(self):
        a = m.numerical_auditor()
        with tempfile.TemporaryDirectory() as directory:
            result = a.audit(Path(directory))
        self.assertEqual(result['status'], 'INCOMPLETE')
        self.assertFalse(result['ledger_opened'])


if __name__ == '__main__':
    unittest.main()
