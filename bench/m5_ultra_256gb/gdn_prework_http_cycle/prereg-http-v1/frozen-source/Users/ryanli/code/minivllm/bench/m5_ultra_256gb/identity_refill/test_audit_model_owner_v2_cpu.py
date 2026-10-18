"""Focused v2 routing and unchanged-guard tests; all run evidence is synthetic."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from bench.m5_ultra_256gb.identity_refill import audit_model_owner_v2 as m
from bench.m5_ultra_256gb.identity_refill import test_audit_model_owner_cpu as old_tests


class SelectionTests(unittest.TestCase):
    def test_real_v2_selection_and_reported_sha_without_source_path_spoof(self):
        module = m.configured_owner()
        numerical = module.numerical_auditor()
        self.assertEqual(Path(module.__file__), m.FROZEN_OWNER)
        self.assertEqual(module.HERE, m.HERE)
        self.assertEqual(Path(numerical.__file__), m.NUMERICAL)
        self.assertEqual(module.MODEL_AUDIT_SHA, m.sha(m.NUMERICAL))
        self.assertEqual(module.MODEL_AUDIT_SHA, m.NUMERICAL_SHA)
        for name in ('audit', 'owner_binding', 'preflight_pins', 'registration'):
            self.assertEqual(Path(getattr(module, name).__code__.co_filename), m.FROZEN_OWNER)
        original = m.load(m.FROZEN_OWNER, m.FROZEN_OWNER_SHA, '_unchanged_owner_for_cpu_check')
        self.assertNotEqual(original.MODEL_AUDIT_SHA, m.NUMERICAL_SHA)
        self.assertEqual(Path(original.numerical_auditor().__file__), m.HERE / 'audit_model.py')

    def test_wrong_frozen_owner_hash_rejected_before_any_ledger(self):
        with patch.object(m, 'FROZEN_OWNER_SHA', '0' * 64):
            result = m.audit('/CPU-only-missing-owner', '/CPU-only-missing-journal')
        self.assertEqual(result['status'], 'FAIL')
        self.assertFalse(result['qualified'] or result['ledger_opened'])

    def test_cli_reports_actual_wrapper_sha_and_never_overwrites_old_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'new-audit.json'
            result = dict(status='INCOMPLETE', qualified=False, ledger_opened=False)
            with patch.object(m, 'audit', return_value=result):
                code = m.main(['--owner', directory, '--journal', directory, '--out', str(path)])
            self.assertEqual(code, 2)
            self.assertEqual(json.loads(path.read_text())['audit_source_sha256'], m.sha(m.__file__))
            original = path.read_bytes()
            with self.assertRaisesRegex(ValueError, 'preserve'):
                m.main(['--owner', directory, '--journal', directory, '--out', str(path)])
            self.assertEqual(path.read_bytes(), original)


class UnchangedGuardTests(unittest.TestCase):
    def setUp(self):
        # Reuse the existing small synthetic recovery evidence, including its
        # real frozen resource/recovery validators. No runtime/model hashing.
        self.fixture = old_tests.AuditTests(methodName='runTest')
        self.fixture.setUp(); self.addCleanup(self.fixture.doCleanups)
        self.module = m.configured_owner()
        self.calls = []
        def numerical(path):
            self.calls.append(path)
            return dict(status='PASS', qualified=True, ledger_opened=True, CPU_MOCK_ONLY=True)
        for context in (
            patch.object(m, 'configured_owner', return_value=self.module),
            patch.object(self.module, 'registration', return_value=self.fixture.registered),
            patch.object(self.module, 'preflight_pins', return_value={'CPU_MOCK_ONLY': True}),
            patch.object(self.module, 'numerical_auditor', return_value=NS(audit=numerical)),
        ):
            context.start(); self.addCleanup(context.stop)

    def audit(self):
        return m.audit(self.fixture.owner, self.fixture.journal)

    def test_pass_routes_one_child_directory_and_reports_selected_source(self):
        result = self.audit()
        self.assertEqual(result['status'], 'PASS', result)
        self.assertEqual(self.calls, [self.fixture.owner / 'qualification'])
        self.assertEqual(result['schema'], 'E028-model-owner-audit-v2')
        self.assertEqual(result['numerical_auditor_sha256'], m.NUMERICAL_SHA)
        self.assertEqual(result['frozen_owner_auditor_sha256'], m.FROZEN_OWNER_SHA)
        self.assertEqual(result['owner_resources']['pressure'], 1)
        self.assertFalse(result['performance_qualified'])

    def test_original_pressure_guard_still_rejects_before_numerical_audit(self):
        self.fixture.mutate(self.fixture.owner / 'resource-002.json', lambda x: x.update(pressure_level_value=2))
        result = self.audit()
        self.assertEqual(result['status'], 'FAIL')
        self.assertIn('pressure', result['first_failure']['message'])
        self.assertFalse(result['ledger_opened']); self.assertEqual(self.calls, [])

    def test_original_restore_raw_log_guard_still_rejects(self):
        Path(self.fixture.data['log_window']['path']).write_text('CPU corrupted raw restore log')
        result = self.audit()
        self.assertEqual(result['status'], 'FAIL')
        self.assertFalse(result['ledger_opened']); self.assertEqual(self.calls, [])


if __name__ == '__main__': unittest.main()
