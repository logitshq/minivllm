"""Source-only direct Popen/unknown-spawn/async-reap regression fakes.

Not run during GPU work. Tests do not launch real children or grant performance.
"""
import asyncio
import ast
from contextlib import ExitStack
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
from bench.m5_ultra_256gb.route_http_formal import owner as m
from bench.m5_ultra_256gb.route_http_formal import controller as c


def write(path,value):path.write_text(json.dumps(value)+'\n')
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
        self.unknown = False; self.unknown_at = None; self.bad_identity = False; self.present_child = False
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
        assert 0 < seconds <= 40524; self.stop = stop
    def cleanup_mode(self):
        self.cleanup = True
    def postflight_deadline(self):
        pass
    def restore_deadline(self):
        self.restored = True
    def new_child(self):
        phases = [p for p in self.f.plan['trials'] if (self.f.out / 'phases' / p / 'intent.json').exists()]
        assert len(phases) == len(self.children) + 1
        assert (self.f.out / 'spawn-intent.json').exists()
        assert all(c.returncode is not None for c in self.children), 'simultaneous children'
        if self.unknown or self.unknown_at == len(self.children):
            raise OSError('Popen outcome unknown')
        child = Child(99000 + len(self.children)); self.children.append(child)
        self.events.append(('spawn', phases[-1]))
        return child
    def formal_backend(self, module):
        parent = self
        class Backend:
            def spawn(self, *unused):
                return parent.new_child()
        return Backend()


class Formal:
    def __init__(self, fixture):
        self.f = fixture; self.calls = []
    async def run(self, spec, out, *, checkpoint, backend):
        out.mkdir(); self.calls.append(spec['concurrency'])
        await checkpoint('pre-spawn', None)
        write(out / 'spawn-intent.json', {'command': ['CPU', 'native_server.py'], 'time': time.time()})
        if self.f.spawn_lock_failure == spec['phase']:
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
        if self.f.formal_exception:
            child.stubborn = True
            raise asyncio.CancelledError('CPU cancellation with live server')
        child.terminate(); child.wait(15)
        await checkpoint('after-reap', None)
        point=dict(concurrency=spec['concurrency'],repeat=spec['repeat'],requests=64,
                   committed_output_tokens=123,common_window_s=1.,aggregate_output_tok_s=123.,e2e_p50_s=1.,e2e_p95_s=1.)
        write(out/'release.json',dict(pid=child.pid,returncode=child.returncode,waited_reaped=True,strict_absence=True,port_absent=True,time=time.time()))
        write(out/'point.json',dict(**point,count_eligible=True,performance_qualified=False))
        result = dict(schema='E024-route-http-formal-result-v1',phase=spec['phase'],arm=spec['arm'],observer_enabled=False,status='FAIL' if self.f.formal_fail else 'PASS', cleanup_complete=not self.f.formal_fail,
                      server_reaped=True, cleanup_task_resolved=True, client_task_resolved=True,
                      count_eligible=True,performance_qualified=False,small_pins_rechecked=True,point=point)
        write(out / 'complete.json', result)
        return result


class OwnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve(); self.out = self.root / 'owner'
        self.journal = self.root / 'journal'; self.journal.mkdir(); (self.journal / '.lock').touch()
        self.base = m.load('qualify_absent_owner')
        self.plan_path = self.root / 'plan.json'
        self.plan = dict(schema=m.PLAN_SCHEMA,
            execution={'owner': {'path': m.__file__, 'sha256': m.sha(m.__file__)}, 'pins': {}},
            qualification={'path': str(self.root / 'numeric.json'), 'sha256': m.NUMERICAL_SHA},
            deadline_seconds=40524, trials=list(m.PHASES), qualifications={'1':{},'8':{}}, cold_restart={'journal_path': str(self.journal),
                'controller': {'path': str(Path(c.__file__)), 'sha256': m.sha(c.__file__)},
                'approved_contract': self.base.CONTRACT, 'proposal_plan_sha256': self.base.PROPOSAL_SHA,
                'predecessor': {'path': 'CPU/RESTORED', 'sha256': 'cpu-only', 'audit': {'path': 'CPU/audit', 'sha256': 'cpu-only'}}},
            http={'transport': {'port': 18765, 'tcp_nodelay': True, 'io_timeout_seconds': 300},
                  'environment': {'MLX_TEST': 'cpu-only'}, 'deadlines': dict(
                      ready=120,request=600,warm=720,formal=2400,close=10,term=15,kill=15,overall=3300,resource_interval=5)})
        write(self.plan_path, self.plan)
        self.args = NS(plan=self.plan_path, plan_sha256=m.sha(self.plan_path), out=self.out)
        self.j = dict(schema=m.JOURNAL_SCHEMA, state='STOPPED',
            plan_sha256=self.base.PROPOSAL_SHA, controller_sha256=m.sha(c.__file__), approved_contract=self.base.CONTRACT,
            owner_directory=str(self.out), owner_plan={'path': str(self.plan_path), 'sha256': self.args.plan_sha256},
            owner_source=self.plan['execution']['owner'], predecessor=self.plan['cold_restart']['predecessor'],
            predecessor_copies={}, trials=self.plan['trials'],
            original={'pid': 74456, 'ppid': 1, 'start': 'Sat Oct 17 14:00:00 2026', 'comm': 'omlx-server'},
            id='CPU-journal', removed_at=time.time() - 10)
        write(self.journal / 'journal.json', self.j)
        self.resources, self.system, self.formal = Resources(), System(self), Formal(self)
        self.formal_fail = self.formal_exception = False
        self.cancel_probe = False
        self.spawn_lock_failure = False
        self.probe_in_progress, self.probe_release = threading.Event(), threading.Event()
        self.prepare_error = None; self.preparations = []
        self.numeric = {'execution': {'python': '/CPU/python'}}
        self.specs = [dict(concurrency=int(p[4]),repeat=int(p[1]),arm=p.split('-')[2],phase=p,qualification={'audit':{'path':'CPU/audit','sha256':'cpu'}}) for p in m.PHASES]
        self.stack = ExitStack(); self.addCleanup(self.stack.close)
        self.validation = self.stack.enter_context(patch.object(m, 'validate_plan', return_value=(self.base, self.plan,
            self.plan_path.read_bytes(), self.numeric, self.specs)))
        self.stack.enter_context(patch.object(m, 'prior_replacement', return_value=self.j['original'].copy()))
        self.stack.enter_context(patch.object(m, 'full_prepare', side_effect=self.prepare))
        self.stack.enter_context(patch.object(m.subprocess, 'Popen', side_effect=AssertionError('real Popen forbidden')))
        self.stack.enter_context(patch.object(m.subprocess, 'run', side_effect=AssertionError('real subprocess forbidden')))
        # asyncio.Runner itself installs/restores SIGINT; no real handler change.
        self.stack.enter_context(patch.object(m.signal, 'signal', return_value=None))

    def recovery_check(self,present=None):
        with patch.object(c,'owner_plan',return_value=self.plan):
            return c.reaped(NS(data=self.j,directory=self.journal),
                NS(process=lambda pid:{'CPU':'present'} if pid==present else None))

    def prepare(self, base, plan, numeric, out, check):
        check(); self.preparations.append(out.name)
        if self.prepare_error: raise self.prepare_error
        out.mkdir(); write(out / 'CPU-pins.json', {'CPU': True})

    def run_owner(self):
        return m.run(self.args, system=self.system, resources=self.resources, formal_module=self.formal)

    def read(self, name):
        return json.loads((self.out / name).read_text())

    def test_formal_failure_reaps_and_stops_next_trial(self):
        self.formal_fail = True
        self.assertEqual(self.run_owner(), 1)
        self.assertEqual(len(self.system.children), 1)
        self.assertEqual(len(self.read('release.json')['children']), 1)
        self.assertFalse((self.out / 'phases/r1-c1-on').exists())

    def test_cancelled_trial_with_live_child_term_kill_reaps(self):
        self.formal_exception = True
        self.assertEqual(self.run_owner(), 1)
        self.assertEqual(len(self.system.children), 1)
        self.assertTrue(self.system.children[0].terminated)
        self.assertTrue(self.system.children[0].killed)
        self.assertTrue(self.read('release.json')['all_children_reaped'])

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
        self.assertTrue((self.out / 'phases/r1-c1-off/intent.json').exists())
        self.assertFalse((self.out / 'terminal.json').exists())
        self.assertFalse((self.out / 'release.json').exists())
        self.assertIsNone(self.read('complete.json')['child_started'])

    def test_identity_error_after_Popen_reaps_explicit_known_child_without_forging_identity(self):
        self.system.bad_identity = True
        self.assertEqual(self.run_owner(), 1)
        self.assertTrue(self.system.children[0].terminated)
        self.assertTrue((self.out / 'release.json').exists())
        child = self.read('phases/r1-c1-off/child.json')
        self.assertIsNone(child['identity'])
        self.assertEqual(child['ownership'], 'direct-Popen-returned')
        self.assertIsNone(self.read('children.json')['children'][0]['identity'])
        self.assertFalse((self.out / 'terminal.json').exists())
        self.assertFalse(self.recovery_check()['no_child_started'])

    def test_strict_child_absence_failure_forbids_next_trial_and_release(self):
        self.system.present_child = True
        self.assertEqual(self.run_owner(), 1)
        self.assertEqual(len(self.system.children), 1)
        self.assertFalse((self.out / 'release.json').exists())

    def test_changed_journal_fails_before_attempt(self):
        self.j['state'] = 'RESTORED'; write(self.journal / 'journal.json', self.j)
        with self.assertRaisesRegex(RuntimeError, 'STOPPED'):
            self.run_owner()
        self.assertFalse(self.out.exists())

    def test_repeated_trial_cancellation_waits_for_resource_thread_before_reap(self):
        self.cancel_probe = True
        def block(label):
            if label == 'r1-c1-off:spawned':
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
        self.assertEqual((self.out / 'phases/r1-c1-off/release.json').read_bytes(), original_release[0])

    def test_nested_spawn_receipt_forbids_nochild_when_outer_spawn_never_returned(self):
        async def nested(spec, out, **unused):
            out.mkdir(); write(out / 'spawn-intent.json', {'CPU': 'unknown'})
            raise RuntimeError('nested unknown outcome')
        self.formal.run = nested
        self.assertEqual(self.run_owner(), 1)
        self.assertFalse((self.out / 'terminal.json').exists())
        self.assertFalse((self.out / 'release.json').exists())
        self.assertIsNone(self.read('complete.json')['child_started'])

    def test_inner_intent_then_tracked_lock_rejection_records_positive_no_popen(self):
        self.spawn_lock_failure = 'r1-c1-off'
        self.assertEqual(self.run_owner(), 1)
        self.assertEqual(self.system.children, [])
        terminal = self.read('terminal.json')
        self.assertEqual(terminal['status'], 'NO_CHILD_STARTED')
        proof = self.read('phases/r1-c1-off/no-popen.json')
        self.assertFalse(proof['popen_called'])
        self.assertEqual(proof['ownership'], 'tracked-backend-rejected-before-Popen')
        self.assertEqual(proof['inner_spawn_intent']['sha256'], m.sha(self.out / proof['inner_spawn_intent']['path']))
        self.assertEqual(terminal['no_popen_receipt']['sha256'], m.sha(self.out / 'phases/r1-c1-off/no-popen.json'))
        self.assertFalse((self.out / 'spawn-intent.json').exists())
        self.assertTrue(self.recovery_check()['no_child_started'])

    def test_all_twelve_trials_preserve_order_points_and_no_overlap(self):
        self.assertEqual(self.run_owner(),0)
        self.assertEqual(self.system.events,[('spawn',p) for p in m.PHASES])
        self.assertEqual([r['phase'] for r in self.read('points.json')['trials']],list(m.PHASES))
        self.assertEqual([r['arm'] for r in self.read('points.json')['trials']],[p.split('-')[2] for p in m.PHASES])
        self.assertFalse(self.read('complete.json')['performance_qualified'])
        self.assertFalse(self.recovery_check()['no_child_started'])

    def test_exact_work_bound_and_no_donor_globals_mutation(self):
        self.assertEqual(m.work_seconds(list(m.PHASES),m.DEADLINES),40524)
        with self.assertRaises(RuntimeError):m.work_seconds(list(m.PHASES[:6]),m.DEADLINES)
        self.assertIs(c.stop.__code__,c._donor.stop.__code__)
        self.assertIsNot(c.stop.__globals__,c._donor.stop.__globals__)
        self.assertIs(c._donor.stop.__globals__['authorize'],c._donor.authorize)
        self.assertIs(c.stop.__globals__['authorize'],c.authorize)



class PreparationTests(unittest.TestCase):
    def test_original_preflight_and_collision_free_adapter_copies(self):
        import hashlib
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();out=root/'preflight';left=root/'old';right=root/'new'
            left.mkdir();right.mkdir();(left/'formal.py').write_bytes(b'old')
            (right/'formal.py').write_bytes(b'new')
            manifest=root/'manifest.json';manifest.write_text('{"files":{}}')
            sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (left/'formal.py',right/'formal.py')}
            calls=[]
            def durable(path,value,*,exclusive=False,raw=False):
                path.parent.mkdir(parents=True,exist_ok=True)
                with path.open('xb' if exclusive else 'wb') as f:
                    f.write(value if raw else json.dumps(value).encode())
            def original(args,check):
                check();calls.append((args.out,args.plan_sha256,args.deadline_seconds))
                durable(args.out/'original-preflight.json',{'original':True})
            base=NS(load_prepare=lambda:original,durable=durable,sha_bytes=lambda x:hashlib.sha256(x).hexdigest())
            plan={'qualification':{'path':str(root/'original-plan.json'),'sha256':'old-plan-sha'},'execution':{'pins':sources}}
            m.full_prepare(base,plan,{'model_manifest':{'path':str(manifest)},'model':str(root)},out,lambda:None)
            self.assertEqual(calls,[(out,'old-plan-sha',5700)])
            records=json.loads((out/'adapter-pins.json').read_text())
            self.assertEqual(set(records),set(sources))
            for source,h in sources.items():
                record=records[source];copy=out/'frozen-adapters'/Path(source).relative_to('/')
                self.assertEqual(copy.read_bytes(),Path(source).read_bytes())
                self.assertEqual(record,dict(sha256=h,bytes=3,copy=str(copy.relative_to(out))))

if __name__=='__main__':unittest.main()
