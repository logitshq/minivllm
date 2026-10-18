"""Prepared CPU tests for the execution driver's ownership/restoration routing.

Only temporary files and an exact-command fake subprocess.run are used. The
real driver main/run functions execute when these tests are explicitly run;
no controller/owner subprocess, GPU, model, signal or service is launched.
Synthetic recovery results test routing, not real recovery or qualification.
"""
from contextlib import ExitStack, redirect_stdout
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


SOURCE = Path(__file__).with_name('execute.py').resolve()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def read(path):
    return json.loads(Path(path).read_bytes())


class ExecuteRouteHTTPTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.tmp = self.stack.enter_context(tempfile.TemporaryDirectory(prefix='gdn-http-driver-cpu-'))
        self.root = Path(self.tmp).resolve()
        self.calls = []
        self.stop_rc = self.owner_rc = self.restore_rc = 0
        self.stop_state = 'STOPPED'
        self.owner_exists = True
        self.owner_error = None
        self.stop_error = None
        # Install the deny-by-default subprocess boundary before importing the
        # driver. Even a newly introduced unknown subprocess command must fail.
        self.process_run = self.stack.enter_context(patch.object(subprocess, 'run', side_effect=self.fake_run))
        self.stack.enter_context(patch.object(subprocess, 'Popen', side_effect=AssertionError('no real child allowed')))
        spec = importlib.util.spec_from_file_location('_gdn_http_execute_cpu', SOURCE)
        self.driver = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.driver)
        self.p, self.d, self.j, self.o = (self.root / x for x in ('prereg', 'execution', 'journal', 'owner'))
        self.p.mkdir(); self.j.mkdir()
        self.controller = self.root / 'controller.py'
        self.owner = self.root / 'owner.py'
        self.controller.write_text('# CPU fixture; never executed\n')
        self.owner.write_text('# CPU fixture owner; never executed\n')
        self.python = str(self.root / 'not-a-real-python')
        self.approval = ['--journal', str(self.j), '--approved-contract', self.driver.A[3],
                         '--plan-sha256', self.driver.A[5]]
        replacements = dict(R=self.root, P=self.p, D=self.d, J=self.j, O=self.o,
                            PY=self.python, C=self.controller, OWNER=self.owner, A=self.approval)
        for name, value in replacements.items():
            self.stack.enter_context(patch.object(self.driver, name, value))
        write(self.p / 'owner-plan.json', {'CPU_only': True, 'distinct_plan_value': 37})
        write(self.j / 'journal.json', {'state': 'PREPARED', 'CPU_only': True})
        self.plan_sha = digest(self.p / 'owner-plan.json')
        self.prepared_raw = (self.j / 'journal.json').read_bytes()
        self.receipt = dict(source_pins={str(path): digest(path) for path in
            (SOURCE, self.controller, self.owner, self.p / 'owner-plan.json')},
            owner_plan_sha256=self.plan_sha,
            prepared_journal_sha256=hashlib.sha256(self.prepared_raw).hexdigest())
        write(self.p / 'receipt.json', self.receipt)
        self.receipt_raw = (self.p / 'receipt.json').read_bytes()

    def command(self, phase, *, no_experiment=False):
        if phase == 'owner':
            return [self.python, '-I', '-B', str(self.owner), '--run', '--plan', str(self.p / 'owner-plan.json'),
                    '--plan-sha256', self.plan_sha, '--out', str(self.o)]
        command = [self.python, '-I', '-B', str(self.controller), phase, *self.approval]
        if phase == 'recover':
            command += ['--watchdog-seconds', '3600']
            if no_experiment:
                command += ['--no-experiment']
        return command

    def fake_run(self, command, **kwargs):
        allowed = {tuple(self.command('stop')): 'stop', tuple(self.command('owner')): 'owner',
                   tuple(self.command('recover')): 'recover',
                   tuple(self.command('recover', no_experiment=True)): 'recover'}
        if tuple(command) not in allowed:
            raise AssertionError('unknown backend command refused: ' + repr(command))
        phase = allowed[tuple(command)]
        self.assertEqual(set(kwargs), {'cwd', 'stdout', 'stderr', 'check'})
        self.assertEqual(kwargs['cwd'], self.root)
        self.assertIs(kwargs['check'], False)
        for key in ('stdout', 'stderr'):
            self.assertEqual(Path(kwargs[key].name).parent, self.d)
        self.calls.append((phase, list(command)))
        kwargs['stdout'].write(('CPU fake ' + phase + '\n').encode())
        kwargs['stderr'].write(b'')
        if phase == 'stop':
            write(self.j / 'journal.json', {'state': self.stop_state, 'CPU_only': True})
            if self.stop_error is not None:
                raise self.stop_error
            code = self.stop_rc
        elif phase == 'owner':
            # The launch-attempt receipt must predate the potentially unknown
            # subprocess outcome; reading this file exercises the real driver.
            invocation = read(self.d / 'owner-invocation.json')
            self.assertIs(invocation['attempted'], True)
            self.assertIs(invocation['returned'], False)
            if self.owner_exists:
                self.o.mkdir()
                write(self.o / 'CPU-partial-owner.json', {'CPU_only': True})
            if self.owner_error is not None:
                raise self.owner_error
            code = self.owner_rc
        else:
            code = self.restore_rc
            # This fake deliberately models refusal of an unknown owner launch
            # with no durable directory. It never pretends that is NO_CHILD.
            if self.owner_error is not None and not self.o.exists() and '--no-experiment' not in command:
                code = 2
            if code == 0:
                write(self.j / 'journal.json', {'state': 'RESTORED', 'CPU_only': True})
        return subprocess.CompletedProcess(command, code)

    def invoke(self, *arguments):
        with patch.object(sys, 'argv', [str(SOURCE), *arguments]), redirect_stdout(io.StringIO()):
            return self.driver.main()

    def phases(self):
        return [phase for phase, _ in self.calls]

    def recovery(self):
        rows = [command for phase, command in self.calls if phase == 'recover']
        self.assertEqual(len(rows), 1)
        return rows[0]

    def test_default_and_nonexact_run_are_inert_without_receipt(self):
        (self.p / 'receipt.json').unlink()
        for arguments in ((), ('--help',), ('--run', 'extra')):
            with self.subTest(arguments=arguments):
                self.assertEqual(self.invoke(*arguments), 0)
        self.process_run.assert_not_called()
        self.assertFalse(self.d.exists() or self.o.exists())

    def test_sealed_sources_plan_and_prepared_bytes_reach_actual_driver(self):
        self.assertEqual(self.invoke('--run'), 0)
        self.assertEqual(self.phases(), ['stop', 'owner', 'recover'])
        self.assertEqual(self.calls[1][1], self.command('owner'))
        self.assertEqual((self.d / 'prepared-journal.json').read_bytes(), self.prepared_raw)
        self.assertEqual((self.d / 'execution-receipt.json').read_bytes(), self.receipt_raw)
        self.assertEqual((self.d / 'execute.py').read_bytes(), SOURCE.read_bytes())
        self.assertNotIn('--no-experiment', self.recovery())
        final = read(self.d / 'final.json')
        self.assertEqual(final['journal_state'], 'RESTORED')
        self.assertTrue(final['owner_invocation_attempted'] and final['owner_invocation_returned'])
        self.assertEqual([final[k] for k in ('stop_returncode', 'owner_returncode', 'recovery_returncode')], [0, 0, 0])

    def test_partial_stop_refusal_still_recovers_without_starting_owner(self):
        self.stop_rc = 1
        self.assertEqual(self.invoke('--run'), 1)
        self.assertEqual(self.phases(), ['stop', 'recover'])
        self.assertIn('--no-experiment', self.recovery())
        final = read(self.d / 'final.json')
        self.assertFalse(final['owner_invocation_attempted'] or final['owner_invocation_returned'])
        self.assertIsNone(final['owner_returncode'])

    def test_untouched_prepared_stop_refusal_does_not_launch_owner_or_recovery(self):
        self.stop_rc, self.stop_state = 1, 'PREPARED'
        self.assertEqual(self.invoke('--run'), 1)
        self.assertEqual(self.phases(), ['stop'])
        self.assertEqual(read(self.d / 'final.json')['journal_state'], 'PREPARED')

    def test_stop_exception_after_state_change_still_attempts_recovery(self):
        self.stop_error = RuntimeError('CPU unknown stop outcome')
        with self.assertRaisesRegex(RuntimeError, 'unknown stop'):
            self.invoke('--run')
        self.assertEqual(self.phases(), ['stop', 'recover'])
        self.assertIn('--no-experiment', self.recovery())

    def test_closed_failed_owner_is_recovered_and_remains_failure(self):
        self.owner_rc = 1
        self.assertEqual(self.invoke('--run'), 1)
        self.assertNotIn('--no-experiment', self.recovery())
        final = read(self.d / 'final.json')
        self.assertEqual(final['owner_returncode'], 1)
        self.assertEqual(final['recovery_returncode'], 0)
        self.assertEqual(read(self.d / 'owner-invocation.json')['returncode'], 1)

    def test_closed_owner_with_no_directory_can_use_no_experiment(self):
        self.owner_rc, self.owner_exists = 1, False
        self.assertEqual(self.invoke('--run'), 1)
        self.assertIn('--no-experiment', self.recovery())
        final = read(self.d / 'final.json')
        self.assertTrue(final['owner_invocation_attempted'] and final['owner_invocation_returned'])

    def test_unknown_owner_without_directory_never_becomes_no_experiment(self):
        self.owner_exists = False
        self.owner_error = RuntimeError('CPU unknown owner launch')
        with self.assertRaisesRegex(RuntimeError, 'unknown owner'):
            self.invoke('--run')
        self.assertNotIn('--no-experiment', self.recovery())
        final = read(self.d / 'final.json')
        self.assertTrue(final['owner_invocation_attempted'])
        self.assertFalse(final['owner_invocation_returned'])
        self.assertIsNone(final['owner_returncode'])
        self.assertEqual(final['recovery_returncode'], 2)
        self.assertEqual(final['journal_state'], 'STOPPED')
        self.assertIs(read(self.d / 'owner-invocation.json')['returned'], False)

    def test_unknown_owner_with_partial_directory_retains_exact_owner_recovery(self):
        self.owner_error = RuntimeError('CPU unknown owner launch')
        with self.assertRaisesRegex(RuntimeError, 'unknown owner'):
            self.invoke('--run')
        self.assertTrue(self.o.exists())
        self.assertNotIn('--no-experiment', self.recovery())
        self.assertFalse(read(self.d / 'final.json')['owner_invocation_returned'])

    def test_source_change_is_rejected_before_any_backend_or_execution_directory(self):
        self.controller.write_text('# changed CPU source\n')
        with self.assertRaisesRegex(ValueError, 'registered source changed'):
            self.invoke('--run')
        self.process_run.assert_not_called()
        self.assertFalse(self.d.exists())

    def test_prepared_journal_hash_change_is_rejected_before_backend(self):
        write(self.j / 'journal.json', {'state': 'PREPARED', 'CPU_only': 'changed bytes'})
        with self.assertRaisesRegex(ValueError, 'prepared journal changed'):
            self.invoke('--run')
        self.process_run.assert_not_called()
        self.assertFalse(self.d.exists())

    def test_recovery_failure_returns_nonzero_and_preserves_unrestored_state(self):
        self.restore_rc = 1
        self.assertEqual(self.invoke('--run'), 1)
        final = read(self.d / 'final.json')
        self.assertEqual(final['recovery_returncode'], 1)
        self.assertEqual(final['journal_state'], 'STOPPED')
        self.assertNotIn('--no-experiment', self.recovery())


if __name__ == '__main__':
    unittest.main()
