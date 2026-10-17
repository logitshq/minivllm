"""Owner protocol tests with explicit fake validation, preparation and OS backend.

These tests grant no device/runtime/request qualification. They exercise the real
owner run loop, durable files, journal lock and cancellation settlement, while
all processes, resources and trial responses are fake. Dependency-negative tests
use actual missing files; no placeholder controller or auditor is installed.
"""
import asyncio
from contextlib import ExitStack
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import threading
import time
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch


HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('omlx_owner_cpu', HERE / 'owner.py')
m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m)


def write(path, value):
    path.write_text(json.dumps(value) + '\n')


class Child:
    def __init__(self, pid, fixture):
        self.pid, self.f, self.returncode = pid, fixture, None
        self.terminated = self.killed = False
        self.stubborn = self.kill_fails = False

    def poll(self): return self.returncode

    def wait(self, timeout):
        if self.returncode is None:
            raise subprocess.TimeoutExpired('CPU fake server', timeout)
        return self.returncode

    def terminate(self):
        assert not self.f.probe_live.is_set(), 'termination raced an unsettled probe'
        self.terminated = True
        if not self.stubborn: self.returncode = -15

    def kill(self):
        self.killed = True
        if self.kill_fails: raise OSError('CPU kill failed')
        self.returncode = -9


class Resources:
    def __init__(self):
        self.calls, self.on_snapshot = [], None
        self.pressure = 1

    def snapshot(self, label, pid, *, suspended_pids):
        assert suspended_pids == ()
        if self.on_snapshot: self.on_snapshot(label)
        self.calls.append((label, pid))
        return dict(label=label, pressure_level_value=self.pressure,
            authorized_suspended_pids=[], native_inference_processes=[] if pid is None else [
                dict(pid=pid, owned=True, authorized_suspended=False)],
            power_metadata={'source': 'AC Power'}, processinfo_thermal_metadata={'code': 0})

    def guard(self, samples):
        if self.pressure != 1: raise RuntimeError('CPU pressure guard')

    def deltas(self, samples): return dict(count=len(samples))


class System:
    def __init__(self, fixture):
        self.f, self.children, self.spawns = fixture, [], []
        self.unknown = self.bad_identity = self.present_child = False
        self.deadline_restored = False

    def identity(self, pid):
        if pid != os.getpid() and self.bad_identity:
            raise RuntimeError('identity failed after actual returned Popen')
        return dict(pid=pid, ppid=1 if pid == os.getpid() else os.getpid(),
                    start='Sat Oct 17 18:00:00 2026', comm='CPU fake server')

    def read(self, command, timeout=5):
        job, port = command[0].endswith('launchctl'), command[0].endswith('lsof')
        child_probe = not job and not port and int(command[2]) != self.f.original['pid']
        if child_probe:
            assert not self.f.probe_live.is_set(), 'absence raced an unsettled probe'
        present = child_probe and self.present_child
        return dict(command=command, returncode=0 if present else 113 if job else 1,
                    stdout='present' if present else '')

    def start_deadline(self, seconds, stop):
        assert 0 < seconds <= self.f.plan['deadline_seconds']
        self.stop = stop

    def cleanup_mode(self): pass
    def postflight_deadline(self): pass
    def restore_deadline(self): self.deadline_restored = True

    def trial_backend(self, module):
        parent = self

        class Backend:
            def spawn(self, command, environment, stdout, stderr, *, cwd):
                stage = m.PHASES[len(parent.children)]
                assert (parent.f.out / 'spawn-intent.json').is_file()
                assert (parent.f.out / 'phases' / stage / 'intent.json').is_file()
                assert all(c.poll() is not None for c in parent.children), 'overlapping servers'
                parent.spawns.append(dict(command=command, environment=environment,
                    stdout=stdout, stderr=stderr, cwd=cwd, stage=stage))
                if parent.unknown: raise OSError('Popen outcome unknown')
                child = Child(99000 + len(parent.children), parent.f)
                parent.children.append(child)
                return child

        return Backend()


class Trial:
    """Explicit fake producer, not the real HTTP/observer qualification runner."""
    def __init__(self, fixture):
        self.f, self.calls, self.cancel_tasks = fixture, [], []

    async def run(self, spec, out, *, checkpoint, backend):
        f = self.f
        out.mkdir(); self.calls.append(spec['concurrency'])
        await checkpoint('pre-spawn', None)
        write(out / 'spawn-intent.json', dict(command=spec['command'], time=time.time()))
        child = backend.spawn(spec['command'], spec['environment'], spec['stdout'], spec['stderr'], cwd=spec['cwd'])
        write(out / 'child.json', dict(pid=child.pid + int(f.wrong_inner_pid)))
        if f.cancel_probe:
            target = asyncio.current_task()

            async def cancel_twice():
                while not f.probe_live.is_set(): await asyncio.sleep(0)
                target.cancel(); await asyncio.sleep(0); target.cancel()
                f.probe_release.set()

            self.cancel_tasks.append(asyncio.create_task(cancel_twice()))
        await checkpoint('spawned', child.pid)
        if f.cancel_live:
            child.stubborn = True; child.kill_fails = f.kill_fails
            raise asyncio.CancelledError('CPU cancellation with known live server')
        if f.exit_code == -15: child.terminate()
        else: child.returncode = f.exit_code
        child.wait(15)
        write(out / 'release.json', dict(pid=child.pid, returncode=child.returncode,
            term_sent=f.term_explained, waited_reaped=True, strict_absence=True, port_absent=True))
        await checkpoint('after-reap', None)
        seal = dict(kind='CPU fake request seal', count=8 if spec['concurrency'] == 1 else 16)
        write(out / 'request-seal.json', seal)
        write(out / 'raw-join.json', dict(schema='E023-omlx-D512-observed-join-v2', status='PASS',
            protocol_route_count_qualified=True, performance_qualified=False,
            full_state_numerical_qualified=False, request_sealed=True, seal=seal,
            concurrency=spec['concurrency'], requests=[dict(cpu=True)] * seal['count'],
            max_http_outstanding=spec['concurrency']))
        write(out / 'group.json', dict(cpu_only=True))
        (out / 'server.stdout').write_text('CPU stdout\n')
        (out / 'server.stderr').write_text('CPU stderr\n')
        (out / 'probe').mkdir(); (out / 'probe/observer.jsonl').write_text('{"cpu_only":true}\n')
        result = dict(schema='E023-omlx-D512-trial-result-v1', concurrency=spec['concurrency'], status='PASS',
            cleanup_complete=True, server_reaped=True, cleanup_task_resolved=True, client_task_resolved=True,
            route_count_text_qualified=True, performance_qualified=False, full_state_qualified=False,
            small_pins_rechecked=True, request_sealed=True)
        write(out / 'complete.json', result)
        return result


class OwnerTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve(); self.out = self.root / 'owner'
        self.journal = self.root / 'journal'; self.journal.mkdir(); (self.journal / '.lock').touch()
        # Real small stdlib receipt/lock helpers, never their OS backend.
        self.base = m.load('qualify_absent_owner')
        self.plan_path = self.root / 'plan.json'
        deadlines = dict(overall=900, close=10, term=15, kill=15)
        self.plan = dict(schema='E023-omlx-D512-owner-plan-v1',
            execution=dict(owner=dict(path=m.__file__, sha256=m.sha(m.__file__)), pins={}),
            qualification=dict(path='/CPU/not-validated', sha256='CPU'), verification={},
            trials=list(m.PHASES), deadline_seconds=m.work_seconds(list(m.PHASES), deadlines),
            http=dict(port=18765, api_key='CPU', environment={}, deadlines=deadlines),
            cold_restart=dict(journal_path=str(self.journal),
                controller=dict(path='/CPU/controller-NOT-CREATED.py', sha256='CPU'),
                predecessor=dict(path='/CPU/prior', sha256='CPU', audit=dict(path='/CPU/audit', sha256='CPU')),
                approved_contract=self.base.CONTRACT, proposal_plan_sha256=self.base.PROPOSAL_SHA))
        write(self.plan_path, self.plan)
        self.args = NS(plan=self.plan_path, plan_sha256=m.sha(self.plan_path), out=self.out)
        self.original = dict(pid=74456, ppid=1, start='Sat Oct 17 14:00:00 2026', comm='omlx-server')
        journal = dict(schema=m.JOURNAL_SCHEMA, state='STOPPED', plan_sha256=self.base.PROPOSAL_SHA,
            controller_sha256='CPU', approved_contract=self.base.CONTRACT, owner_directory=str(self.out),
            owner_plan=dict(path=str(self.plan_path), sha256=self.args.plan_sha256),
            owner_source=self.plan['execution']['owner'], predecessor=self.plan['cold_restart']['predecessor'],
            predecessor_copies={}, trials=list(m.PHASES), original=self.original,
            id='CPU-journal', removed_at=time.time() - 10)
        write(self.journal / 'journal.json', journal)
        self.specs = [dict(concurrency=c, command=['/CPU/omlx-python', '-m', 'serve_probe_v2'],
            environment={'CPU': str(c)}, cwd='/CPU/omlx-home', stdout='CPU-stdout', stderr='CPU-stderr') for c in (1, 8)]
        self.numeric = dict(execution=dict(python='/CPU/outer-python'))
        self.resources, self.system, self.trial = Resources(), System(self), Trial(self)
        self.wrong_inner_pid = self.cancel_probe = self.cancel_live = self.kill_fails = False
        self.exit_code, self.term_explained = -15, True
        self.probe_live, self.probe_release = threading.Event(), threading.Event()
        self.preparations = []
        stack = ExitStack(); self.addCleanup(stack.close)
        stack.enter_context(patch.object(m, 'validate_plan', return_value=(self.base, self.plan,
            self.plan_path.read_bytes(), self.numeric, self.specs)))
        stack.enter_context(patch.object(m, 'prior_replacement', return_value=self.original.copy()))
        stack.enter_context(patch.object(m, 'full_prepare', side_effect=self.prepare))
        stack.enter_context(patch.object(m.subprocess, 'Popen', side_effect=AssertionError('real Popen forbidden')))
        stack.enter_context(patch.object(m.subprocess, 'run', side_effect=AssertionError('real subprocess forbidden')))
        # asyncio.Runner's own SIGINT handler must not alter the test process.
        stack.enter_context(patch.object(m.signal, 'signal', return_value=None))

    def prepare(self, base, plan, numeric, out, check):
        check(); self.preparations.append(out.name)
        out.mkdir(); write(out / 'CPU-NOT-RUNTIME-VERIFIED.json', dict(fake_preparation=True))

    def run_owner(self):
        return m.run(self.args, system=self.system, resources=self.resources, trial_module=self.trial)

    def read(self, path): return json.loads((self.out / path).read_bytes())

    def test_direct_server_pid_cwd_and_both_layers_of_release_match(self):
        self.assertEqual(self.run_owner(), 0)
        self.assertEqual(self.preparations, ['preflight', 'postflight'])
        self.assertEqual(self.trial.calls, [1, 8])
        for i, phase in enumerate(m.PHASES):
            call = self.system.spawns[i]; spec = self.specs[i]
            self.assertEqual({k: call[k] for k in ('command', 'environment', 'cwd', 'stdout', 'stderr')},
                             {k: spec[k] for k in ('command', 'environment', 'cwd', 'stdout', 'stderr')})
            inner = self.read(f'phases/{phase}/trial/child.json')
            outer = self.read(f'phases/{phase}/child.json')
            self.assertEqual(inner['pid'], outer['pid'])
            self.assertEqual(outer['identity']['ppid'], os.getpid())
            self.assertNotEqual(outer['pid'], os.getpid())
            self.assertIn((phase + ':spawned', inner['pid']), self.resources.calls)
            self.assertEqual(self.read(f'phases/{phase}/trial/release.json')['returncode'], -15)
            self.assertEqual(self.read(f'phases/{phase}/release.json')['returncode'], -15)
        self.assertEqual(self.read('release.json')['children'], self.read('children.json')['children'])
        self.assertTrue(self.read('complete.json')['qualification_trials_complete'])
        for field in ('independently_qualified', 'performance_qualified', 'full_state_qualified',
                      'recursive_runtime_coverage_verified', 'closed_server_logs_independently_verified'):
            self.assertFalse(self.read('complete.json')[field])

    def test_wrong_inner_server_pid_is_rejected_even_after_actual_child_reap(self):
        self.wrong_inner_pid = True
        self.assertEqual(self.run_owner(), 1)
        self.assertEqual(len(self.system.children), 1)
        self.assertIn('actually owned/reaped', self.read('first-error.json')['error'])
        self.assertTrue(self.read('release.json')['all_children_reaped'])

    def test_identity_probe_failure_retains_and_reaps_known_popen(self):
        self.system.bad_identity = True
        self.assertEqual(self.run_owner(), 1)
        self.assertTrue(self.system.children[0].terminated)
        row = self.read('phases/omlx-c1/child.json')
        self.assertIsNone(row['identity'])
        self.assertEqual(row['ownership'], 'direct-Popen-returned')
        self.assertTrue(self.read('release.json')['all_children_reaped'])
        self.assertFalse((self.out / 'terminal.json').exists())

    def test_unknown_popen_retains_intent_and_never_certifies_no_child(self):
        self.system.unknown = True
        self.assertEqual(self.run_owner(), 1)
        self.assertTrue((self.out / 'spawn-intent.json').exists())
        self.assertTrue((self.out / 'phases/omlx-c1/intent.json').exists())
        self.assertFalse((self.out / 'release.json').exists())
        self.assertFalse((self.out / 'terminal.json').exists())
        self.assertIsNone(self.read('complete.json')['child_started'])

    def test_cancelled_probe_settles_thread_before_any_server_reap(self):
        self.cancel_probe = True

        def block(label):
            if label == 'omlx-c1:spawned':
                self.probe_live.set()
                try: self.assertTrue(self.probe_release.wait(2), 'CPU canceller failed to release probe')
                finally: self.probe_live.clear()

        self.resources.on_snapshot = block
        self.assertEqual(self.run_owner(), 1)
        self.assertFalse(self.probe_live.is_set())
        self.assertTrue(self.trial.cancel_tasks and all(t.done() for t in self.trial.cancel_tasks))
        self.assertIn(('omlx-c1:spawned', self.system.children[0].pid), self.resources.calls)
        self.assertTrue(self.read('release.json')['all_children_reaped'])
        self.assertTrue(self.system.deadline_restored)

    def test_cancelled_live_server_term_then_kill_is_reaped_but_never_qualified(self):
        self.cancel_live = True
        self.assertEqual(self.run_owner(), 1)
        self.assertTrue(self.system.children[0].terminated and self.system.children[0].killed)
        self.assertEqual(self.read('phases/omlx-c1/release.json')['returncode'], -9)
        self.assertTrue(self.read('release.json')['all_children_reaped'])
        self.assertFalse(self.read('complete.json')['qualification_trials_complete'])

    def test_kill_failure_leaves_unresolved_child_without_final_release(self):
        self.cancel_live = self.kill_fails = True
        self.assertEqual(self.run_owner(), 1)
        self.assertTrue(self.system.children[0].terminated and self.system.children[0].killed)
        self.assertIsNone(self.system.children[0].returncode)
        self.assertFalse((self.out / 'release.json').exists())
        self.assertFalse((self.out / 'terminal.json').exists())
        self.assertIn('CPU kill failed', str(self.read('errors.json')))

    def test_unexplained_term_is_rejected_despite_fake_trial_pass(self):
        self.term_explained = False
        self.assertEqual(self.run_owner(), 1)
        self.assertEqual(len(self.system.children), 1)
        self.assertIn('trial release does not match', self.read('first-error.json')['error'])
        self.assertTrue(self.read('release.json')['all_children_reaped'])

    def test_killed_trial_cannot_pass_qualification_even_if_fake_trial_claims_pass(self):
        self.exit_code = -9
        self.assertEqual(self.run_owner(), 1)
        self.assertEqual(len(self.system.children), 1)
        self.assertIn('trial release does not match', self.read('first-error.json')['error'])

    def test_live_pid_after_wait_forbids_final_release_and_next_phase(self):
        self.system.present_child = True
        self.assertEqual(self.run_owner(), 1)
        self.assertEqual(len(self.system.children), 1)
        self.assertFalse((self.out / 'release.json').exists())

    def test_pre_spawn_resource_rejection_has_real_no_child_terminal(self):
        self.resources.pressure = 2
        self.assertEqual(self.run_owner(), 1)
        self.assertEqual(self.system.children, [])
        self.assertEqual(self.read('terminal.json')['status'], 'NO_CHILD_STARTED')
        self.assertFalse((self.out / 'spawn-intent.json').exists())


class MissingDependenciesTests(unittest.TestCase):
    def test_each_missing_auditor_fails_on_real_filesystem_without_placeholder(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp).resolve()
            # Copy only real reviewed input modules. No controller/auditor is fabricated.
            for name in m.LOCAL: (folder / name).write_bytes((HERE / name).read_bytes())
            for name in m.REQUIRED_AUDIT_SOURCES:
                with self.subTest(name=name), patch.object(m, 'HERE', folder), \
                        patch.object(m, 'REQUIRED_AUDIT_SOURCES', (name,)):
                    self.assertFalse((folder / name).exists())
                    with self.assertRaisesRegex(RuntimeError, 'not yet supplied: ' + name):
                        m.source_pins()

    def test_validate_rejects_actual_missing_controller_before_any_prepare_or_spawn(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp).resolve(); plan_path = folder / 'plan.json'
            base = m.load('qualify_absent_owner')
            numeric_path = folder / 'numeric.json'; write(numeric_path, {'CPU_metadata': True})
            numeric_sha = m.sha(numeric_path)
            deadlines = dict(overall=900, close=10, term=15, kill=15)
            plan = dict(schema='E023-omlx-D512-owner-plan-v1', trials=list(m.PHASES),
                deadline_seconds=m.work_seconds(list(m.PHASES), deadlines),
                execution=dict(owner=dict(path=str(Path(m.__file__).resolve()), sha256=m.sha(m.__file__)), pins={}),
                qualification=dict(path=str(numeric_path), sha256=numeric_sha), verification={},
                http=dict(deadlines=deadlines), cold_restart=dict(journal_path=str(folder / 'journal'),
                    controller=dict(path=str(folder / 'controller.py'), sha256='0' * 64),
                    approved_contract=base.CONTRACT, proposal_plan_sha256=base.PROPOSAL_SHA, predecessor={}))
            write(plan_path, plan)
            args = NS(plan=plan_path, plan_sha256=m.sha(plan_path), out=folder / 'owner')
            # Bypass unrelated audit/numeric identity gates explicitly; retain the
            # real controller-path + file-read gate. No absent file is invented.
            with patch.object(m, 'HERE', folder), patch.object(m, 'load', return_value=base), \
                    patch.object(m, 'source_pins', return_value={}), patch.object(m, 'NUMERICAL_SHA', numeric_sha), \
                    patch.object(m, 'full_prepare', side_effect=AssertionError('prepare forbidden')):
                self.assertFalse((folder / 'controller.py').exists())
                with self.assertRaises(FileNotFoundError): m.validate_plan(args)
            self.assertFalse(args.out.exists())


if __name__ == '__main__':
    unittest.main()
