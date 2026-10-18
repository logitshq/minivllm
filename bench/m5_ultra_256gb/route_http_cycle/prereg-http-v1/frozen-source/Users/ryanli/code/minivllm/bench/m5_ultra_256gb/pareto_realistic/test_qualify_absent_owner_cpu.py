"""Stdlib fakes only: no real subprocess, signal, HTTP, service or MLX action."""
from contextlib import ExitStack
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('absent_owner_tested', HERE / 'qualify_absent_owner.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
REAL_LOAD_PREPARE = m.load_prepare
REAL_LOAD_RESOURCES = m.load_resources


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2) + '\n')


class Child:
    pid = 99002
    def __init__(self, finish=0):
        self.returncode, self.finish = None, finish
        self.terminated = self.killed = False
        self.waits = []
    def poll(self):
        return self.returncode
    def wait(self, timeout):
        self.waits.append(timeout)
        if self.returncode is None:
            self.returncode = self.finish
        return self.returncode
    def terminate(self):
        self.terminated = True
        self.returncode = -15
    def kill(self):
        self.killed = True
        self.returncode = -9


class FakeSystem:
    def __init__(self, fixture):
        self.fixture, self.reads, self.spawned = fixture, [], []
        self.child = Child()
        self.on_spawn = None
        self.present = set()
        self.cleanup_entered = self.deadline_restored = False
        self.on_read = None
    def identity(self, pid):
        return {'pid': pid, 'ppid': 1, 'start': 'Sat Oct 17 01:02:03 2026', 'comm': 'CPU fake owner'}
    def read(self, command, timeout=5):
        self.reads.append(command)
        if self.on_read:
            self.on_read(command)
        kind = 'job' if command[0].endswith('launchctl') else 'port' if command[0].endswith('lsof') else 'pid'
        present = kind in self.present or kind == 'pid' and command[2] == str(self.child.pid) and 'child' in self.present
        return {'command': command, 'returncode': 0 if present else (113 if kind == 'job' else 1),
                'stdout': 'present\n' if present else ''}
    def spawn(self, command, log, env):
        # Inspect fsynced intent BEFORE the fake child exists; no real spawn.
        intent = json.loads((self.fixture.out / 'spawn-intent.json').read_text())
        assert intent['attempt_sha256'] == m.sha(self.fixture.out / 'attempt.json')
        self.spawned.append(command)
        if self.on_spawn:
            self.on_spawn()
        q = self.fixture.out / 'qualification'
        q.mkdir()
        write(q / 'complete.json', {'status': 'PASS'})
        return self.child
    def start_deadline(self, seconds, stop):
        assert 0 < seconds <= 1830
    def cleanup_mode(self):
        self.cleanup_entered = True
    def postflight_deadline(self):
        pass
    def restore_deadline(self):
        self.deadline_restored = True


class Resources:
    def __init__(self):
        self.calls, self.on_snapshot = [], None
        self.pressure = 1
        self.foreign = False
        self.suspended = []
    def snapshot(self, label, pid, *, suspended_pids):
        assert suspended_pids == ()
        self.calls.append((label, pid))
        if self.on_snapshot:
            self.on_snapshot(label)
        rows = [] if pid is None else [{'pid': pid, 'owned': True, 'authorized_suspended': False}]
        if self.foreign:
            rows += [{'pid': 444, 'owned': False, 'authorized_suspended': False}]
        return {'label': label, 'native_process_inventory_ok': True, 'no_active_foreign_native_processes': not self.foreign,
                'native_inference_processes': rows, 'authorized_suspended_pids': self.suspended,
                'pressure_level_value': self.pressure, 'power_metadata': {'source': 'AC Power'},
                'processinfo_thermal_metadata': {'code': 0}}
    def guard(self, samples):
        if samples[-1]['pressure_level_value'] != 1 or not samples[-1]['no_active_foreign_native_processes']:
            raise RuntimeError('CPU fake original resource guard rejected')
    def deltas(self, samples):
        return samples


class OwnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name).resolve()
        self.out, self.journal = self.base / 'owner', self.base / 'journal'
        self.journal.mkdir()
        (self.journal / '.lock').touch()
        self.plan_path = self.base / 'wrapper.json'
        numeric_path = HERE.parents[2] / '.artifacts/m5-ultra-256gb/realistic-qualification-v1-inputs/plan.json'
        self.plan = {'schema': 'E023-service-absent-owner-plan-v1',
                     'execution': {'owner': {'path': m.__file__, 'sha256': m.sha(m.__file__)}},
                     'qualification': {'path': str(numeric_path), 'sha256': m.NUMERICAL_SHA},
                     'cold_restart': {'journal_path': str(self.journal),
                         'controller': {'path': str(HERE.parent / 'omlx_cold_restart/controller_v2.py'),
                                        'sha256': m.sha(HERE.parent / 'omlx_cold_restart/controller_v2.py')},
                         'approved_contract': m.CONTRACT, 'proposal_plan_sha256': m.PROPOSAL_SHA},
                     'deadline_seconds': 1830}
        self.commit_plan()
        self.system, self.resources = FakeSystem(self), Resources()
        self.prepare_error = None
        self.on_prepare = None
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(m, 'load_resources', return_value=self.resources))
        self.stack.enter_context(patch.object(m, 'load_prepare', return_value=self.prepare))
        # Any accidental real process/signal use fails a CPU test immediately.
        self.stack.enter_context(patch.object(m.subprocess, 'run', side_effect=AssertionError('real subprocess forbidden')))
        self.stack.enter_context(patch.object(m.subprocess, 'Popen', side_effect=AssertionError('real Popen forbidden')))
        self.stack.enter_context(patch.object(m.signal, 'signal', side_effect=AssertionError('real signal forbidden')))
        self.stack.enter_context(patch.object(m.signal, 'setitimer', side_effect=AssertionError('real alarm forbidden')))

    def commit_plan(self):
        write(self.plan_path, self.plan)
        self.args = NS(plan=self.plan_path, plan_sha256=m.sha(self.plan_path), out=self.out)
        self.j = {'schema': 'omlx-cold-restart-journal-v2', 'state': 'STOPPED', 'id': 'cpu-only-journal',
                  'removed_at': time.time() - 10, 'plan_sha256': m.PROPOSAL_SHA,
                  'controller_sha256': self.plan['cold_restart']['controller']['sha256'],
                  'approved_contract': m.CONTRACT, 'original': copy.deepcopy(m.ORIGINAL),
                  'owner_directory': str(self.out), 'owner_plan': {'path': str(self.plan_path), 'sha256': self.args.plan_sha256},
                  'owner_source': self.plan['execution']['owner'], 'events': []}
        write(self.journal / 'journal.json', self.j)

    def prepare(self, args, check):
        self.assertTrue((self.out / 'attempt.json').exists())
        self.assertFalse((self.out / 'spawn-intent.json').exists())
        self.assertEqual(args.plan_sha256, m.NUMERICAL_SHA)
        self.assertEqual(args.deadline_seconds, 1830)
        check()
        if self.on_prepare:
            self.on_prepare()
        if self.prepare_error:
            raise self.prepare_error
        # Never invoke the actual frozen prepare/hash/runtime or qualifier.
        return ['CPU_FAKE_PYTHON', 'qualify_native.py', '--plan-sha256', m.NUMERICAL_SHA], {'MLX_FAKE': 'cpu'}

    def run_owner(self):
        return m.run(self.args, system=self.system)

    def read(self, name):
        return json.loads((self.out / name).read_text())

    def no_child(self):
        self.assertEqual(self.system.spawned, [])
        terminal = self.read('terminal.json')
        self.assertEqual(terminal['status'], 'NO_CHILD_STARTED')
        self.assertFalse(terminal['spawn_intent_written'])
        self.assertFalse(terminal['child_started'])
        self.assertEqual(terminal['attempt_sha256'], m.sha(self.out / 'attempt.json'))
        self.assertEqual(terminal['owner_identity'], self.read('attempt.json')['owner_identity'])
        for name in ('spawn-intent.json', 'child.json', 'release.json'):
            self.assertFalse((self.out / name).exists())

    def test_success_same_child_plan_and_reap(self):
        self.assertEqual(self.run_owner(), 0)
        self.assertEqual(self.read('complete.json')['status'], 'PASS')
        self.assertEqual(self.read('release.json')['returncode'], 0)
        self.assertTrue(self.read('release.json')['waited_reaped'])
        self.assertFalse((self.out / 'terminal.json').exists())
        self.assertEqual(len(self.system.spawned), 1)
        self.assertEqual([x[0] for x in self.resources.calls], ['before-prepare', 'after-prepare-before-spawn', 'child-running', 'postflight'])
        self.assertTrue(self.system.deadline_restored)
        self.assertEqual(self.read('attempt.json')['stopped_journal_sha256'], m.sha(self.out / 'stopped-journal.json'))

    def test_preparation_error_is_durable_no_child(self):
        self.prepare_error = ValueError('pin mismatch fixture')
        self.assertEqual(self.run_owner(), 1)
        self.no_child()
        self.assertEqual(self.read('terminal.json')['phase'], 'prepare')

    def test_cancel_before_intent_is_no_child(self):
        self.prepare_error = KeyboardInterrupt('CPU fake cancellation')
        self.assertEqual(self.run_owner(), 1)
        self.no_child()

    def test_controller_lock_excludes_initial_admission(self):
        with m.journal_lock(self.journal):
            with self.assertRaises(BlockingIOError):
                self.run_owner()
        self.assertFalse(self.out.exists())
        self.assertEqual(self.system.spawned, [])

    def test_resource_preflight_rejects_pressure_two(self):
        self.resources.pressure = 2
        self.assertEqual(self.run_owner(), 1)
        self.no_child()
        self.assertTrue((self.out / 'resource-000.json').exists())

    def test_foreign_inference_prevents_spawn(self):
        self.resources.foreign = True
        self.assertEqual(self.run_owner(), 1)
        self.no_child()

    def test_no_suspended_service_exemption(self):
        self.resources.suspended = [74384]
        self.assertEqual(self.run_owner(), 1)
        self.no_child()

    def test_each_removed_service_component_is_required(self):
        for component in ('pid', 'job', 'port'):
            with self.subTest(component=component):
                self.system.present = {component}
                proof = m.absence(self.system)
                self.assertEqual(proof['status'], 'FAIL')
                self.assertEqual(sum(not x['absent'] for x in proof['checks']), 1)
                # No raw launchd output/arguments can leak into receipts.
                self.assertNotIn('present', json.dumps(proof))

    def test_absence_failure_keeps_first_and_postflight_receipts(self):
        self.system.present.add('port')
        self.assertEqual(self.run_owner(), 1)
        self.no_child()
        self.assertEqual(self.read('absence-000.json')['label'], 'before-prepare')
        self.assertEqual(self.read('absence-001.json')['label'], 'postflight')
        self.assertEqual(self.read('complete.json')['absence_probes'], 2)

    def test_initial_recovery_state_never_creates_directory(self):
        self.j['state'] = 'BOOTSTRAP_INTENT'
        write(self.journal / 'journal.json', self.j)
        with self.assertRaisesRegex(RuntimeError, 'STOPPED'):
            self.run_owner()
        self.assertFalse(self.out.exists())

    def test_journal_change_during_prepare_forbids_spawn(self):
        def change():
            self.j['state'] = 'LOAD_INTENT'
            write(self.journal / 'journal.json', self.j)
        self.on_prepare = change
        self.assertEqual(self.run_owner(), 1)
        self.no_child()

    def test_wait_events_can_grow_without_rebinding_stop(self):
        def event():
            self.j['events'].append({'event': 'owner_wait', 'time': time.time()})
            write(self.journal / 'journal.json', self.j)
        self.on_prepare = event
        self.assertEqual(self.run_owner(), 0)

    def test_stable_removed_time_change_forbids_spawn(self):
        def change():
            self.j['removed_at'] -= 1
            write(self.journal / 'journal.json', self.j)
        self.on_prepare = change
        self.assertEqual(self.run_owner(), 1)
        self.no_child()

    def test_spawn_exception_cannot_certify_no_child(self):
        def fail():
            raise OSError('unknown Popen result')
        self.system.on_spawn = fail
        self.assertEqual(self.run_owner(), 1)
        self.assertTrue((self.out / 'spawn-intent.json').exists())
        self.assertFalse((self.out / 'terminal.json').exists())
        self.assertFalse((self.out / 'release.json').exists())
        self.assertIsNone(self.read('complete.json')['child_started'])

    def test_intent_write_failure_cannot_certify_no_child(self):
        original = m.durable
        def broken(path, *args, **kwargs):
            if Path(path).name == 'spawn-intent.json':
                Path(path).write_text('partial')
                raise OSError('fsync failed')
            return original(path, *args, **kwargs)
        with patch.object(m, 'durable', side_effect=broken):
            self.assertEqual(self.run_owner(), 1)
        self.assertFalse((self.out / 'terminal.json').exists())
        self.assertEqual(self.system.spawned, [])

    def test_running_recovery_state_stops_and_reaps_child(self):
        def change(label):
            if label == 'child-running':
                self.j['state'] = 'BOOTSTRAP_INTENT'
                write(self.journal / 'journal.json', self.j)
        self.resources.on_snapshot = change
        self.assertEqual(self.run_owner(), 1)
        self.assertTrue(self.system.child.terminated)
        self.assertEqual(self.read('release.json')['returncode'], -15)

    def test_running_service_reappearance_stops_child(self):
        self.system.on_spawn = lambda: self.system.present.add('port')
        self.assertEqual(self.run_owner(), 1)
        self.assertTrue(self.system.child.terminated)
        self.assertTrue(self.read('release.json')['strict_absence'])

    def test_failed_child_still_reaped_but_no_qualification(self):
        self.system.child.finish = 2
        self.assertEqual(self.run_owner(), 1)
        self.assertEqual(self.read('release.json')['returncode'], 2)
        self.assertEqual(self.read('complete.json')['status'], 'FAIL')

    def test_pid_reuse_or_unknown_child_absence_never_release(self):
        self.system.present.add('child')
        self.assertEqual(self.run_owner(), 1)
        self.assertFalse((self.out / 'release.json').exists())
        self.assertFalse((self.out / 'terminal.json').exists())

    def test_unresponsive_child_gets_bounded_kill_reap(self):
        child = self.system.child
        def stubborn_terminate():
            child.terminated = True
        def wait(timeout):
            child.waits.append(timeout)
            if child.returncode is None:
                raise subprocess.TimeoutExpired('CPU fake child', timeout)
            return child.returncode
        child.terminate, child.wait = stubborn_terminate, wait
        self.system.on_spawn = lambda: self.system.present.add('port')
        self.assertEqual(self.run_owner(), 1)
        self.assertTrue(child.terminated and child.killed)
        self.assertEqual(child.waits, [15, 15])
        self.assertEqual(self.read('release.json')['returncode'], -9)

    def test_wrong_workload_plan_refused_before_attempt(self):
        self.plan['qualification']['sha256'] = '0' * 64
        self.commit_plan()
        with self.assertRaisesRegex(RuntimeError, 'unchanged original'):
            self.run_owner()
        self.assertFalse(self.out.exists())

    def test_wrong_wrapper_source_refused(self):
        self.plan['execution']['owner']['sha256'] = '0' * 64
        self.commit_plan()
        with self.assertRaisesRegex(RuntimeError, 'owner source'):
            self.run_owner()
        self.assertFalse(self.out.exists())

    def test_wrong_owner_directory_binding_refused(self):
        self.j['owner_directory'] = str(self.base / 'different')
        write(self.journal / 'journal.json', self.j)
        with self.assertRaisesRegex(RuntimeError, 'STOPPED'):
            self.run_owner()
        self.assertFalse(self.out.exists())

    def test_only_finite_1800_to_1830_deadlines(self):
        for seconds in (True, 1799, 1831, float('nan'), float('inf')):
            self.plan['deadline_seconds'] = seconds
            self.commit_plan()
            with self.subTest(seconds=seconds), self.assertRaises(RuntimeError):
                self.run_owner()
        self.assertFalse(self.out.exists())

    def test_existing_output_never_overwritten(self):
        self.out.mkdir()
        (self.out / 'sentinel').write_text('keep')
        with self.assertRaises(FileExistsError):
            self.run_owner()
        self.assertEqual(list(self.out.iterdir()), [self.out / 'sentinel'])

    def test_durable_intent_exclusive_and_data_dir_fsync(self):
        p = self.base / 'intent.json'
        calls = []
        real = os.fsync
        with patch.object(m.os, 'fsync', side_effect=lambda fd: (calls.append(fd), real(fd))):
            m.durable(p, {'intent': True}, exclusive=True)
        self.assertEqual(len(calls), 2)
        with self.assertRaises(FileExistsError):
            m.durable(p, {'intent': False}, exclusive=True)
        self.assertEqual(json.loads(p.read_text()), {'intent': True})

    def test_help_and_unknown_do_not_load_backend_or_mlx(self):
        with patch('sys.stdout', new=io.StringIO()), patch('sys.stderr', new=io.StringIO()):
            with self.assertRaises(SystemExit) as good:
                m.main(['--help'])
            self.assertEqual(good.exception.code, 0)
            with self.assertRaises(SystemExit) as bad:
                m.main(['--unknown'])
            self.assertEqual(bad.exception.code, 2)
        self.assertFalse(any(x == 'mlx' or x.startswith('mlx.') for x in sys.modules))

    def test_actual_frozen_stdlib_loaders_without_execution(self):
        prepare = REAL_LOAD_PREPARE()
        self.assertEqual(Path(prepare.__code__.co_filename), HERE / 'qualify_owner.py')
        numeric = json.loads(Path(self.plan['qualification']['path']).read_text())
        resources = REAL_LOAD_RESOURCES(numeric)
        self.assertEqual(Path(resources.__file__), HERE.parent / 'resources_v2.py')
        self.assertFalse(any(x == 'mlx' or x.startswith('mlx.') for x in sys.modules))

    def test_no_child_receipt_accepted_by_current_controller(self):
        self.prepare_error = RuntimeError('CPU fake preparation failure')
        self.assertEqual(self.run_owner(), 1)
        ctl = m.module_from(HERE.parent / 'omlx_cold_restart/controller_v2.py', 'cpu_test_controller')
        j = NS(directory=self.journal, data=self.j)
        proof = ctl.reaped(j, NS(process=lambda pid: None))
        self.assertTrue(proof['no_child_started'])
        with self.assertRaises(ctl.NotReady):
            ctl.reaped(j, NS(process=lambda pid: self.system.identity(pid)))

    def test_spawned_reap_receipt_accepted_by_current_controller(self):
        self.assertEqual(self.run_owner(), 0)
        ctl = m.module_from(HERE.parent / 'omlx_cold_restart/controller_v2.py', 'cpu_test_controller')
        proof = ctl.reaped(NS(directory=self.journal, data=self.j), NS(process=lambda pid: None))
        self.assertFalse(proof['no_child_started'])


if __name__ == '__main__':
    unittest.main()
