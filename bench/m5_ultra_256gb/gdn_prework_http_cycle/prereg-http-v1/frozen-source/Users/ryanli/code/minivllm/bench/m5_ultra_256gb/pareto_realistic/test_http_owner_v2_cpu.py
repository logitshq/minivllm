"""Focused direct-child ownership fakes; no socket, process, MLX or service calls."""
import asyncio
import ast
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
spec = importlib.util.spec_from_file_location('http_owner_cpu', HERE / 'http_owner_v2.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


def write(path, value):
    path.write_text(json.dumps(value) + '\n')


class Child:
    def __init__(self, pid):
        self.pid, self.returncode = pid, None
        self.terminated = self.killed = False
        self.stubborn = False
    def poll(self):
        return self.returncode
    def wait(self, timeout):
        if self.stubborn and not self.killed and self.returncode is None:
            raise subprocess.TimeoutExpired('CPU fake', timeout)
        if self.returncode is None:
            self.returncode = 0
        return self.returncode
    def terminate(self):
        self.terminated = True
        if not self.stubborn:
            self.returncode = -15
    def kill(self):
        self.killed = True; self.returncode = -9


class Resources:
    def __init__(self):
        self.pressure = 1; self.calls = []; self.foreign = False
        self.on_snapshot = None
    def snapshot(self, label, pid, *, suspended_pids):
        assert suspended_pids == ()
        if self.on_snapshot:
            self.on_snapshot(label)
        self.calls.append((label, pid))
        processes = [] if pid is None else [dict(pid=pid, owned=True, authorized_suspended=False)]
        if self.foreign:
            processes.append(dict(pid=8888, owned=False, authorized_suspended=False))
        return dict(label=label, pressure_level_value=self.pressure,
            authorized_suspended_pids=[], native_inference_processes=processes,
            power_metadata={'source': 'AC Power'}, processinfo_thermal_metadata={'code': 0})
    def guard(self, samples):
        if self.pressure != 1:
            raise RuntimeError('original pressure guard')
    def deltas(self, samples):
        return {'count': len(samples)}


class System:
    def __init__(self, fixture):
        self.f = fixture; self.children = []; self.events = []
        self.unknown = False; self.bad_identity = False; self.present_child = False
        self.cleanup = self.restored = False; self.stop = None
    def identity(self, pid):
        if pid != os.getpid() and self.bad_identity:
            raise RuntimeError('identity probe failed after returned Popen')
        return dict(pid=pid, ppid=os.getpid() if pid != os.getpid() else 1,
                    start='Sat Oct 17 15:00:00 2026', comm='CPU fake')
    def read(self, command, timeout=5):
        kind = 'job' if command[0].endswith('launchctl') else 'port' if command[0].endswith('lsof') else 'pid'
        present = self.present_child and kind == 'pid' and int(command[2]) != 74456
        if kind == 'pid' and int(command[2]) != 74456:
            assert not self.f.probe_in_progress.is_set(), 'reap raced unfinished resource probe'
        return dict(command=command, returncode=0 if present else 113 if kind == 'job' else 1,
                    stdout='present' if present else '')
    def start_deadline(self, seconds, stop):
        assert 0 < seconds <= 5700; self.stop = stop
    def cleanup_mode(self):
        self.cleanup = True
    def postflight_deadline(self):
        pass
    def restore_deadline(self):
        self.restored = True
    def new_child(self):
        phases = [p for p in m.PHASES if (self.f.out / 'phases' / p / 'intent.json').exists()]
        assert len(phases) == len(self.children) + 1
        assert (self.f.out / 'spawn-intent.json').exists()
        assert all(c.returncode is not None for c in self.children), 'simultaneous children'
        if self.unknown:
            raise OSError('Popen outcome unknown')
        child = Child(99000 + len(self.children)); self.children.append(child)
        self.events.append(('spawn', phases[-1]))
        return child
    def capture_backend(self, module):
        parent = self
        class Backend:
            def spawn(self, *unused):
                return parent.new_child()
        return Backend()
    def spawn(self, command, log, env):
        child = self.new_child()
        plan_path = Path(command[command.index('--plan') + 1])
        plan = json.loads(plan_path.read_text())
        output = Path(command[command.index('--out') + 1]); output.mkdir()
        write(output / 'complete.json', dict(status='PASS', numerical_qualified=True,
            concurrency=plan['concurrency'], total_requests=8 if plan['concurrency'] == 1 else 16,
            plan_sha256=m.sha(plan_path), input_source_pins_rechecked=True))
        if self.f.replay_stubborn:
            child.stubborn = True
            raise_after = self.f.resources
            old = raise_after.on_snapshot
            def fail(label):
                if label.endswith('replay:running'):
                    raise RuntimeError('running resource failure')
                if old: old(label)
            raise_after.on_snapshot = fail
        return child


class Capture:
    def __init__(self, fixture):
        self.f = fixture; self.calls = []
    async def capture(self, spec, out, *, checkpoint, backend):
        out.mkdir(); self.calls.append(spec['concurrency'])
        await checkpoint('pre-spawn', None)
        write(out / 'spawn-intent.json', {'command': ['CPU', 'native_server.py'], 'time': time.time()})
        if self.f.spawn_lock_failure:
            with self.f.base.journal_lock(self.f.journal):
                child = backend.spawn([], {}, None, None)
        else:
            child = backend.spawn([], {}, None, None)
        write(out / 'child.json', {'pid': child.pid})
        if self.f.cancel_probe:
            target = asyncio.current_task()
            async def cancel_twice():
                while not self.f.probe_in_progress.is_set():
                    await asyncio.sleep(0)
                target.cancel(); await asyncio.sleep(0); target.cancel()
                self.f.probe_release.set()
            canceller = asyncio.create_task(cancel_twice())
        await checkpoint('spawned', child.pid)
        if self.f.capture_exception:
            child.stubborn = True
            raise asyncio.CancelledError('CPU cancellation with live server')
        child.terminate(); child.wait(15)
        await checkpoint('after-reap', None)
        write(out / 'group.json', {'CPU': 'wire evidence'})
        (out / 'trace.jsonl').write_text('{"CPU":"trace"}\n')
        result = dict(status='FAIL' if self.f.capture_fail else 'PASS', cleanup_complete=not self.f.capture_fail,
                      server_reaped=True, cleanup_task_resolved=True, client_task_resolved=True)
        write(out / 'complete.json', result)
        return result


class OwnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve(); self.out = self.root / 'owner'
        self.journal = self.root / 'journal'; self.journal.mkdir(); (self.journal / '.lock').touch()
        self.base = m.load('qualify_absent_owner')
        self.plan_path = self.root / 'plan.json'
        self.plan = dict(schema='E023-http-service-absent-owner-plan-v1',
            execution={'owner': {'path': m.__file__, 'sha256': m.sha(m.__file__)}, 'pins': m.source_pins()},
            qualification={'path': str(self.root / 'numeric.json'), 'sha256': m.NUMERICAL_SHA},
            deadline_seconds=5700, cold_restart={'journal_path': str(self.journal),
                'controller': {'path': 'CPU/controller_v3.py', 'sha256': 'cpu-only'},
                'approved_contract': self.base.CONTRACT, 'proposal_plan_sha256': self.base.PROPOSAL_SHA,
                'predecessor': {'path': 'CPU/RESTORED', 'sha256': 'cpu-only', 'audit': {'path': 'CPU/audit', 'sha256': 'cpu-only'}}},
            http={'transport': {'port': 18765, 'tcp_nodelay': True, 'io_timeout_seconds': 300},
                  'environment': {'MLX_TEST': 'cpu-only'}, 'deadlines': {}})
        write(self.plan_path, self.plan)
        self.args = NS(plan=self.plan_path, plan_sha256=m.sha(self.plan_path), out=self.out)
        self.j = dict(schema='omlx-cold-restart-journal-v3', state='STOPPED',
            plan_sha256=self.base.PROPOSAL_SHA, controller_sha256='cpu-only', approved_contract=self.base.CONTRACT,
            owner_directory=str(self.out), owner_plan={'path': str(self.plan_path), 'sha256': self.args.plan_sha256},
            owner_source=self.plan['execution']['owner'], predecessor=self.plan['cold_restart']['predecessor'],
            predecessor_copies={},
            original={'pid': 74456, 'ppid': 1, 'start': 'Sat Oct 17 14:00:00 2026', 'comm': 'omlx-server'},
            id='CPU-journal', removed_at=time.time() - 10)
        write(self.journal / 'journal.json', self.j)
        self.resources, self.system, self.capture = Resources(), System(self), Capture(self)
        self.capture_fail = self.capture_exception = self.replay_stubborn = False
        self.cancel_probe = False
        self.spawn_lock_failure = False
        self.probe_in_progress, self.probe_release = threading.Event(), threading.Event()
        self.prepare_error = None; self.preparations = []
        self.numeric = {'execution': {'python': '/CPU/python'}}
        self.specs = [{'concurrency': c} for c in (1, 8)]
        self.q = NS(FIXED={HERE / 'qualify_native.py': 'cpu'}, ADAPTERS=(HERE / 'http_replay.py',),
                    validate=lambda *unused: None)
        original_load = m.load
        self.stack = ExitStack(); self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(m, 'validate_plan', return_value=(self.base, self.plan,
            self.plan_path.read_bytes(), self.numeric, self.specs)))
        self.stack.enter_context(patch.object(m, 'prior_replacement', return_value=self.j['original'].copy()))
        self.stack.enter_context(patch.object(m, 'full_prepare', side_effect=self.prepare))
        self.stack.enter_context(patch.object(m, 'load', side_effect=lambda name: self.q if name == 'http_qualification' else original_load(name)))
        self.stack.enter_context(patch.object(m.subprocess, 'Popen', side_effect=AssertionError('real Popen forbidden')))
        self.stack.enter_context(patch.object(m.subprocess, 'run', side_effect=AssertionError('real subprocess forbidden')))
        # asyncio.Runner itself installs/restores SIGINT; no real handler change.
        self.stack.enter_context(patch.object(m.signal, 'signal', return_value=None))

    def prepare(self, base, plan, numeric, out, check):
        check(); self.preparations.append(out.name)
        if self.prepare_error: raise self.prepare_error
        out.mkdir(); write(out / 'CPU-pins.json', {'CPU': True})

    def run_owner(self):
        return m.run(self.args, system=self.system, resources=self.resources, capture_module=self.capture)

    def read(self, name):
        return json.loads((self.out / name).read_text())

    def recovery_check(self, present=None):
        path = HERE.parent / 'omlx_cold_restart/controller_v3.py'
        spec = importlib.util.spec_from_file_location('http_owner_actual_v3_protocol', path)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        return module.reaped(NS(data=self.j, directory=self.journal),
            NS(process=lambda pid: {'CPU': 'still present'} if pid == present else None))

    def test_four_real_direct_children_never_overlap_and_replay_bound_to_closed_capture(self):
        self.assertEqual(self.run_owner(), 0)
        self.assertEqual(self.system.events, [('spawn', p) for p in m.PHASES])
        release = self.read('release.json'); index = self.read('children.json')
        self.assertEqual(release['children'], index['children'])
        self.assertEqual(release['children_sha256'], m.sha(self.out / 'children.json'))
        self.assertEqual(self.preparations, ['preflight', 'postflight'])
        self.assertTrue(self.read('complete.json')['numerical_replay_complete'])
        self.assertFalse(self.read('complete.json')['performance_qualified'])
        for c in (1, 8):
            plan = self.read(f'phases/c{c}-replay/plan.json')
            for item in plan['inputs'].values():
                self.assertEqual(item['sha256'], m.sha(item['path']))
            self.assertTrue(self.read(f'phases/c{c}-capture/release.json')['strict_absence'])
        self.assertFalse(self.recovery_check()['no_child_started'])

    def test_capture_failure_reaps_but_never_starts_replay(self):
        self.capture_fail = True
        self.assertEqual(self.run_owner(), 1)
        self.assertEqual(len(self.system.children), 1)
        self.assertEqual(len(self.read('release.json')['children']), 1)
        self.assertFalse((self.out / 'phases/c1-replay').exists())

    def test_cancelled_capture_with_live_child_term_kill_reaps(self):
        self.capture_exception = True
        self.assertEqual(self.run_owner(), 1)
        self.assertEqual(len(self.system.children), 1)
        self.assertTrue(self.system.children[0].terminated)
        self.assertTrue(self.system.children[0].killed)
        self.assertTrue(self.read('release.json')['all_children_reaped'])

    def test_replay_resource_failure_stops_following_C_and_kills(self):
        self.replay_stubborn = True
        self.assertEqual(self.run_owner(), 1)
        self.assertEqual(len(self.system.children), 2)
        self.assertTrue(self.system.children[1].killed)
        self.assertEqual(len(self.read('release.json')['children']), 2)

    def test_pre_intent_pressure_failure_is_no_child(self):
        self.resources.pressure = 2
        self.assertEqual(self.run_owner(), 1)
        self.assertEqual(self.system.children, [])
        self.assertEqual(self.read('terminal.json')['status'], 'NO_CHILD_STARTED')
        self.assertFalse((self.out / 'spawn-intent.json').exists())
        self.assertTrue(self.recovery_check()['no_child_started'])

    def test_unknown_spawn_cannot_claim_nochild_or_final_release(self):
        self.system.unknown = True
        self.assertEqual(self.run_owner(), 1)
        self.assertTrue((self.out / 'phases/c1-capture/intent.json').exists())
        self.assertFalse((self.out / 'terminal.json').exists())
        self.assertFalse((self.out / 'release.json').exists())
        self.assertIsNone(self.read('complete.json')['child_started'])

    def test_identity_error_after_Popen_reaps_explicit_known_child_without_forging_identity(self):
        self.system.bad_identity = True
        self.assertEqual(self.run_owner(), 1)
        self.assertTrue(self.system.children[0].terminated)
        self.assertTrue((self.out / 'release.json').exists())
        child = self.read('phases/c1-capture/child.json')
        self.assertIsNone(child['identity'])
        self.assertEqual(child['ownership'], 'direct-Popen-returned')
        self.assertIsNone(self.read('children.json')['children'][0]['identity'])
        self.assertFalse((self.out / 'terminal.json').exists())
        self.assertFalse(self.recovery_check()['no_child_started'])

    def test_strict_child_absence_failure_forbids_replay_and_release(self):
        self.system.present_child = True
        self.assertEqual(self.run_owner(), 1)
        self.assertEqual(len(self.system.children), 1)
        self.assertFalse((self.out / 'release.json').exists())

    def test_changed_journal_fails_before_attempt(self):
        self.j['state'] = 'RESTORED'; write(self.journal / 'journal.json', self.j)
        with self.assertRaisesRegex(RuntimeError, 'STOPPED'):
            self.run_owner()
        self.assertFalse(self.out.exists())

    def test_journal_lock_rejects_competing_owner(self):
        with self.base.journal_lock(self.journal):
            with self.assertRaises(BlockingIOError):
                self.run_owner()
        self.assertFalse(self.out.exists())

    def test_foreign_process_rejected_with_no_suspended_exemption(self):
        self.resources.foreign = True
        self.assertEqual(self.run_owner(), 1)
        self.assertEqual(self.system.children, [])
        self.assertEqual(self.read('terminal.json')['status'], 'NO_CHILD_STARTED')

    def test_repeated_capture_cancellation_waits_for_resource_thread_before_reap(self):
        self.cancel_probe = True
        def block(label):
            if label == 'c1-capture:spawned':
                self.probe_in_progress.set()
                try:
                    self.assertTrue(self.probe_release.wait(2))
                finally:
                    self.probe_in_progress.clear()
        self.resources.on_snapshot = block
        self.assertEqual(self.run_owner(), 1)
        self.assertEqual(len(self.system.children), 1)
        self.assertTrue(self.read('release.json')['all_children_reaped'])
        self.assertFalse(self.probe_in_progress.is_set())

    def test_replay_absolute_deadline_cancels_and_reaps(self):
        def deadline(label):
            if label == 'c1-replay:running':
                self.system.stop()
        self.resources.on_snapshot = deadline
        self.assertEqual(self.run_owner(), 1)
        self.assertEqual(len(self.system.children), 2)
        self.assertEqual(self.read('first-error.json')['type'], 'TimeoutError')
        self.assertTrue(self.read('release.json')['all_children_reaped'])

    def test_release_before_failed_index_write_is_reconciled_without_overwrite(self):
        real = self.base.durable
        original_release = []
        def write_with_once_failure(path, value, **kwargs):
            if path.name == 'children.json' and value['children'] and value['children'][-1]['release'] is not None and not original_release:
                original_release.append((self.out / value['children'][-1]['release']['path']).read_bytes())
                raise OSError('CPU simulated index write failure after durable release')
            return real(path, value, **kwargs)
        with patch.object(self.base, 'durable', side_effect=write_with_once_failure):
            self.assertEqual(self.run_owner(), 1)
        self.assertEqual(len(self.system.children), 1)
        self.assertTrue(self.read('release.json')['all_children_reaped'])
        self.assertEqual((self.out / 'phases/c1-capture/release.json').read_bytes(), original_release[0])

    def test_nested_spawn_receipt_forbids_nochild_when_outer_spawn_never_returned(self):
        async def nested(spec, out, **unused):
            out.mkdir(); write(out / 'spawn-intent.json', {'CPU': 'unknown'})
            raise RuntimeError('nested unknown outcome')
        self.capture.capture = nested
        self.assertEqual(self.run_owner(), 1)
        self.assertFalse((self.out / 'terminal.json').exists())
        self.assertFalse((self.out / 'release.json').exists())
        self.assertIsNone(self.read('complete.json')['child_started'])

    def test_inner_intent_then_tracked_lock_rejection_records_positive_no_popen(self):
        self.spawn_lock_failure = True
        self.assertEqual(self.run_owner(), 1)
        self.assertEqual(self.system.children, [])
        terminal = self.read('terminal.json')
        self.assertEqual(terminal['status'], 'NO_CHILD_STARTED')
        proof = self.read('phases/c1-capture/no-popen.json')
        self.assertFalse(proof['popen_called'])
        self.assertEqual(proof['ownership'], 'tracked-backend-rejected-before-Popen')
        self.assertEqual(proof['inner_spawn_intent']['sha256'], m.sha(self.out / proof['inner_spawn_intent']['path']))
        self.assertEqual(terminal['no_popen_receipt']['sha256'], m.sha(self.out / 'phases/c1-capture/no-popen.json'))
        self.assertFalse((self.out / 'spawn-intent.json').exists())
        self.assertTrue(self.recovery_check()['no_child_started'])

    def test_actual_controller_refuses_live_child_and_missing_wait_receipt(self):
        self.capture_fail = True
        self.assertEqual(self.run_owner(), 1)
        with self.assertRaises(Exception):
            self.recovery_check(present=self.system.children[0].pid)
        phase_release = self.out / 'phases/c1-capture/release.json'
        row = json.loads(phase_release.read_text()); row['waited_reaped'] = False; write(phase_release, row)
        index = self.read('children.json'); index['children'][0]['release']['sha256'] = m.sha(phase_release)
        write(self.out / 'children.json', index)
        final = self.read('release.json'); final['children'] = index['children']
        final['children_sha256'] = m.sha(self.out / 'children.json'); write(self.out / 'release.json', final)
        with self.assertRaises(Exception):
            self.recovery_check()

    def test_only_capture_version_binding_changed_and_actual_v2_paths_are_pinned(self):
        old_path = HERE / 'http_owner.py'
        self.assertEqual(m.sha(old_path), '8ee25e08f6028920e53d2f6d6287e91bfa318883e3becb4fe7a881624ff9dc62')
        pins = m.source_pins()
        self.assertEqual(pins[str(HERE / 'http_capture_v2.py')], m.sha(HERE / 'http_capture_v2.py'))
        self.assertEqual(pins[str(HERE / 'http_owner_v2.py')], m.sha(HERE / 'http_owner_v2.py'))
        self.assertNotIn(str(HERE / 'http_capture.py'), pins)
        self.assertNotIn(str(HERE / 'http_owner.py'), pins)
        text = Path(m.__file__).read_text().replace('http_capture_v2', 'http_capture')
        text = text.replace(m.FIXED['http_capture_v2.py'],
            '8f43c71b8f3fd99d655ba760aa344eb3026eef35450be9c5ed1f06171109a0fb')
        self.assertEqual(ast.dump(ast.parse(text)), ast.dump(ast.parse(old_path.read_text())))


if __name__ == '__main__':
    unittest.main()
