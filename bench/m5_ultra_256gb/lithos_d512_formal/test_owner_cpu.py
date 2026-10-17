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
spec = importlib.util.spec_from_file_location('lithos_owner_cpu', HERE / 'owner.py')
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
        assert 0 < seconds <= 22000; self.stop = stop
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
    def trial_backend(self, module):
        parent = self
        class Backend:
            def spawn(self, *unused, cwd):
                assert cwd == '/CPU/lithos-metal'
                parent.events.append(('cwd', cwd))
                return parent.new_child()
        return Backend()


class Trial:
    def __init__(self, fixture):
        self.f = fixture; self.calls = []
    async def run(self, spec, out, *, checkpoint, backend):
        out.mkdir(); self.calls.append(spec['concurrency'])
        await checkpoint('pre-spawn', None)
        write(out / 'spawn-intent.json', {'command': ['CPU', 'native_server.py'], 'time': time.time()})
        if self.f.spawn_lock_failure == spec['concurrency'] or self.f.spawn_lock_failure_phase==out.parent.name:
            with self.f.base.journal_lock(self.f.journal):
                child = backend.spawn([], {}, None, None, cwd='/CPU/lithos-metal')
        else:
            child = backend.spawn([], {}, None, None, cwd='/CPU/lithos-metal')
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
        write(out/'release.json',dict(pid=child.pid,waited_reaped=True,strict_absence=True,port_absent=True))
        await checkpoint('after-reap', None)
        point=dict(concurrency=spec['concurrency'],repeat=spec['repeat'],requests=64,group_window_s=100.0)
        write(out/'point.json',dict(**point,count_eligible=True,performance_qualified=False))
        result = dict(schema='E023-lithos-D512-formal-result-v1',concurrency=spec['concurrency'],repeat=spec['repeat'],point=point,logs_rechecked=True,
                      status='FAIL' if self.f.formal_fail else 'PASS', cleanup_complete=not self.f.formal_fail,
                      server_reaped=True, cleanup_task_resolved=True, client_task_resolved=True,
                      count_eligible=True,performance_qualified=False,full_state_qualified=False,small_pins_rechecked=True)
        write(out / 'complete.json', result)
        return result


class OwnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve(); self.out = self.root / 'owner'
        self.journal = self.root / 'journal'; self.journal.mkdir(); (self.journal / '.lock').touch()
        self.base = m.load('qualify_absent_owner')
        self.plan_path = self.root / 'plan.json'
        self.plan = dict(schema='E023-lithos-D512-formal-owner-plan-v1',
            execution={'owner': {'path': m.__file__, 'sha256': m.sha(m.__file__)}, 'pins': m.source_pins()},
            qualification={'path': str(self.root / 'numeric.json'), 'sha256': m.NUMERICAL_SHA},
            deadline_seconds=6012, trials=list(m.PHASES), verification={}, cold_restart={'journal_path': str(self.journal),
                'controller': {'path': 'CPU/lithos-controller.py', 'sha256': 'cpu-only'},
                'approved_contract': self.base.CONTRACT, 'proposal_plan_sha256': self.base.PROPOSAL_SHA,
                'predecessor': {'path': 'CPU/RESTORED', 'sha256': 'cpu-only', 'audit': {'path': 'CPU/audit', 'sha256': 'cpu-only'}}},
            http={'port':18765,'api_key':'CPU-ONLY', 'environment': {'MLX_TEST': 'cpu-only'}, 'deadlines':dict(
                ready=120,request=600,group=720,close=10,term=15,kill=15,overall=900,resource_interval=5)})
        write(self.plan_path, self.plan)
        self.args = NS(plan=self.plan_path, plan_sha256=m.sha(self.plan_path), out=self.out)
        self.j = dict(schema=m.JOURNAL_SCHEMA, state='STOPPED',
            plan_sha256=self.base.PROPOSAL_SHA, controller_sha256='cpu-only', approved_contract=self.base.CONTRACT,
            owner_directory=str(self.out), owner_plan={'path': str(self.plan_path), 'sha256': self.args.plan_sha256},
            owner_source=self.plan['execution']['owner'], predecessor=self.plan['cold_restart']['predecessor'],
            predecessor_copies={}, trials=self.plan['trials'],
            original={'pid': 74456, 'ppid': 1, 'start': 'Sat Oct 17 14:00:00 2026', 'comm': 'omlx-server'},
            id='CPU-journal', removed_at=time.time() - 10)
        write(self.journal / 'journal.json', self.j)
        self.resources, self.system, self.formal = Resources(), System(self), Trial(self)
        self.formal_fail = self.formal_exception = False
        self.cancel_probe = False
        self.spawn_lock_failure = False; self.spawn_lock_failure_phase=None
        self.probe_in_progress, self.probe_release = threading.Event(), threading.Event()
        self.prepare_error = None; self.preparations = []
        self.numeric = {'execution': {'python': '/CPU/python'}}
        self.specs = [dict(concurrency=c,repeat=r) for r,c in m.TRIALS]
        self.stack = ExitStack(); self.addCleanup(self.stack.close)
        self.validation = self.stack.enter_context(patch.object(m, 'validate_plan', return_value=(self.base, self.plan,
            self.plan_path.read_bytes(), self.numeric, self.specs)))
        self.stack.enter_context(patch.object(m, 'prior_replacement', return_value=self.j['original'].copy()))
        self.stack.enter_context(patch.object(m, 'full_prepare', side_effect=self.prepare))
        self.stack.enter_context(patch.object(m.subprocess, 'Popen', side_effect=AssertionError('real Popen forbidden')))
        self.stack.enter_context(patch.object(m.subprocess, 'run', side_effect=AssertionError('real subprocess forbidden')))
        # asyncio.Runner itself installs/restores SIGINT; no real handler change.
        self.stack.enter_context(patch.object(m.signal, 'signal', return_value=None))

    def prepare(self, base, plan, numeric, out, check):
        check(); self.preparations.append(out.name)
        if self.prepare_error: raise self.prepare_error
        out.mkdir(); write(out / 'CPU-pins.json', {'CPU': True})

    def run_owner(self):
        return m.run(self.args, system=self.system, resources=self.resources, trial_module=self.formal)

    def read(self, name):
        return json.loads((self.out / name).read_text())

    def test_six_real_named_trials_never_overlap_and_keep_each_closed_point(self):
        self.assertEqual(self.run_owner(), 0)
        self.assertEqual([e for e in self.system.events if e[0]=='spawn'], [('spawn', p) for p in self.plan['trials']])
        self.assertEqual([e for e in self.system.events if e[0]=='cwd'], [('cwd','/CPU/lithos-metal')]*6)
        release = self.read('release.json'); index = self.read('children.json')
        self.assertEqual(release['children'], index['children'])
        self.assertEqual(release['children_sha256'], m.sha(self.out / 'children.json'))
        self.assertEqual(self.preparations, ['preflight', 'postflight'])
        self.assertTrue(self.read('complete.json')['formal_trials_complete'])
        self.assertFalse(self.read('complete.json')['performance_qualified'])
        self.assertEqual([r['phase'] for r in self.read('points.json')['trials']], self.plan['trials'])
        for phase in self.plan['trials']:
            point=self.read('phases/'+phase+'/trial/complete.json')['point']
            self.assertNotIn('count_eligible',point)
            self.assertEqual(self.read('phases/'+phase+'/trial/point.json'),
                dict(**point,count_eligible=True,performance_qualified=False))

    def test_formal_failure_reaps_and_stops_next_trial(self):
        self.formal_fail = True
        self.assertEqual(self.run_owner(), 1)
        self.assertEqual(len(self.system.children), 1)
        self.assertEqual(len(self.read('release.json')['children']), 1)
        self.assertFalse((self.out / 'phases/r1-c8').exists())

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

    def test_unknown_spawn_cannot_claim_nochild_or_final_release(self):
        self.system.unknown = True
        self.assertEqual(self.run_owner(), 1)
        self.assertTrue((self.out / 'phases/r1-c1/intent.json').exists())
        self.assertFalse((self.out / 'terminal.json').exists())
        self.assertFalse((self.out / 'release.json').exists())
        self.assertIsNone(self.read('complete.json')['child_started'])

    def test_identity_error_after_Popen_reaps_explicit_known_child_without_forging_identity(self):
        self.system.bad_identity = True
        self.assertEqual(self.run_owner(), 1)
        self.assertTrue(self.system.children[0].terminated)
        self.assertTrue((self.out / 'release.json').exists())
        child = self.read('phases/r1-c1/child.json')
        self.assertIsNone(child['identity'])
        self.assertEqual(child['ownership'], 'direct-Popen-returned')
        self.assertIsNone(self.read('children.json')['children'][0]['identity'])
        self.assertFalse((self.out / 'terminal.json').exists())

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

    def test_repeated_trial_cancellation_waits_for_resource_thread_before_reap(self):
        self.cancel_probe = True
        def block(label):
            if label == 'r1-c1:spawned':
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
        self.assertEqual((self.out / 'phases/r1-c1/release.json').read_bytes(), original_release[0])

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
        self.spawn_lock_failure = 1
        self.assertEqual(self.run_owner(), 1)
        self.assertEqual(self.system.children, [])
        terminal = self.read('terminal.json')
        self.assertEqual(terminal['status'], 'NO_CHILD_STARTED')
        proof = self.read('phases/r1-c1/no-popen.json')
        self.assertFalse(proof['popen_called'])
        self.assertEqual(proof['ownership'], 'tracked-backend-rejected-before-Popen')
        self.assertEqual(proof['inner_spawn_intent']['sha256'], m.sha(self.out / proof['inner_spawn_intent']['path']))
        self.assertEqual(terminal['no_popen_receipt']['sha256'], m.sha(self.out / 'phases/r1-c1/no-popen.json'))
        self.assertFalse((self.out / 'spawn-intent.json').exists())

    def test_later_pre_popen_lock_rejection_recovers_only_closed_prefix(self):
        self.spawn_lock_failure=8
        self.assertEqual(self.run_owner(),1)
        self.assertEqual(len(self.system.children),1)
        final=self.read('release.json');proof=self.read('phases/r1-c8/no-popen.json')
        self.assertEqual([r['phase'] for r in final['children']],['r1-c1'])
        self.assertEqual(final['no_popen_receipt']['path'],'phases/r1-c8/no-popen.json')
        self.assertFalse(proof['popen_called'])
        self.assertFalse((self.out/'phases/r1-c8/intent.json').exists())
    def test_sixth_pre_popen_rejection_retains_five_closed_trials(self):
        self.spawn_lock_failure_phase='r3-c8'
        self.assertEqual(self.run_owner(),1)
        self.assertEqual(len(self.system.children),5)
        final=self.read('release.json')
        self.assertEqual([r['phase'] for r in final['children']],list(m.PHASES[:5]))
        self.assertEqual(final['no_popen_receipt']['path'],'phases/r3-c8/no-popen.json')
        self.assertFalse(self.read('phases/r3-c8/no-popen.json')['popen_called'])
        self.assertEqual(len(self.read('points.json')['trials']),5)
        self.assertFalse(self.read('complete.json')['performance_qualified'])
        self.assertFalse((self.out/'phases/r3-c8/intent.json').exists())

    def test_later_unknown_Popen_cannot_be_reported_as_pre_popen_rejection(self):
        self.system.unknown_at=1
        self.assertEqual(self.run_owner(),1)
        self.assertEqual(len(self.system.children),1)
        self.assertTrue((self.out/'phases/r1-c8/intent.json').exists())
        self.assertFalse((self.out/'phases/r1-c8/no-popen.json').exists())
        self.assertFalse((self.out/'release.json').exists())



class PreparationTests(unittest.TestCase):
    """Tiny real files exercise hashing; no runtime/weight files are read."""
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.base=m.load('qualify_absent_owner')
        groups={}
        for i,group in enumerate(m.GROUPS):
            path=self.root/(group+'.bin');path.write_bytes(b'' if i==0 else b'CPU fixture '+bytes([i]))
            groups[group]={str(path):dict(sha256=m.sha(path),bytes=path.stat().st_size,mtime_ns=path.stat().st_mtime_ns)}
        self.plan=dict(verification=dict(python='/CPU/python',groups=groups),execution=dict(pins={}))

    def test_all_five_groups_and_empty_file_are_actually_hashed(self):
        out=self.root/'prepared';calls=[]
        m.full_prepare(self.base,self.plan,{},out,lambda:calls.append(True))
        rows=json.loads((out/'verified-pins.json').read_text())
        self.assertEqual(set(rows),set(m.GROUPS))
        self.assertEqual(sum(map(len,rows.values())),5)
        self.assertEqual(next(iter(rows['lithos_runtime'].values()))['bytes'],0)
        self.assertGreaterEqual(len(calls),9)
        self.assertEqual(json.loads((out/'complete.json').read_text())['status'],'PASS')

    def test_same_size_mtime_but_changed_bytes_fail_hash(self):
        name=next(iter(self.plan['verification']['groups']['pack']))
        path=Path(name);stat=path.stat();path.write_bytes(b'X'*stat.st_size)
        os.utime(path,ns=(stat.st_atime_ns,stat.st_mtime_ns))
        out=self.root/'bad'
        with self.assertRaisesRegex(RuntimeError,'SHA differs'):
            m.full_prepare(self.base,self.plan,{},out,lambda:None)
        self.assertFalse((out/'complete.json').exists())
        self.assertTrue((out/'verified-pins.json').exists())

    def test_absolute_check_interrupts_inside_file_and_keeps_partial_receipt(self):
        name=next(iter(self.plan['verification']['groups']['lithos_runtime']))
        path=Path(name);path.write_bytes(b'x'*(2<<20))
        self.plan['verification']['groups']['lithos_runtime'][name]=dict(
            sha256=m.sha(path),bytes=path.stat().st_size,mtime_ns=path.stat().st_mtime_ns)
        calls=[]
        def check():
            calls.append(True)
            if len(calls)==3:raise TimeoutError('CPU fixed deadline')
        out=self.root/'deadline'
        with self.assertRaisesRegex(TimeoutError,'fixed deadline'):
            m.full_prepare(self.base,self.plan,{},out,check)
        self.assertEqual(len(calls),3)
        self.assertFalse((out/'complete.json').exists())
        self.assertEqual(json.loads((out/'verified-pins.json').read_text()),{'lithos_runtime':{}})

    def test_path_replacement_during_read_fails_same_fd_identity_check(self):
        name=next(iter(self.plan['verification']['groups']['lithos_runtime']))
        path=Path(name);path.write_bytes(b'x'*(2<<20));row=dict(
            sha256=m.sha(path),bytes=path.stat().st_size,mtime_ns=path.stat().st_mtime_ns)
        self.plan['verification']['groups']['lithos_runtime'][name]=row
        calls=[]
        def check():
            calls.append(True)
            if len(calls)==2:
                new=self.root/'replacement';new.write_bytes(path.read_bytes())
                os.utime(new,ns=(row['mtime_ns'],row['mtime_ns']));new.replace(path)
        with self.assertRaisesRegex(RuntimeError,'pin changed'):
            m.full_prepare(self.base,self.plan,{},self.root/'replaced',check)

    def test_registered_phases_and_budgets_are_exact(self):
        d=dict(overall=900,close=10,term=15,kill=15)
        self.assertEqual(m.work_seconds(list(m.PHASES),d),6012)
        self.assertEqual(m.work_seconds(list(m.PHASES),dict(d,overall=3600)),22212)
        for phases in ([],['r1-c1'],['r1-c8','r1-c1'],['r1-c1','r1-c8']):
            with self.subTest(phases=phases),self.assertRaises(RuntimeError):m.trial_names(phases)
        with self.assertRaises(RuntimeError):m.work_seconds(list(m.PHASES),dict(d,overall=float('inf')))

    def test_inherited_resource_and_spawn_guards_are_not_rewritten(self):
        original=ast.parse((HERE.parent/'pareto_realistic/native_formal_owner.py').read_text().replace(
            'formal owner absolute deadline or cancellation','Lithos owner absolute deadline or cancellation'))
        actual=ast.parse(Path(m.__file__).read_text())
        def nested(tree,name):
            run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
            return next(n for n in run.body if isinstance(n,ast.FunctionDef) and n.name==name)
        for name in ('stop','check','probe','spawn_impl'):
            self.assertEqual(ast.dump(nested(original,name)),ast.dump(nested(actual,name)),name)

    def test_failure_receipt_error_does_not_replace_primary_deadline(self):
        real=self.base.durable
        def save(path,value,**kwargs):
            if path.name=='verified-pins.json':raise OSError('secondary receipt failure')
            return real(path,value,**kwargs)
        with patch.object(self.base,'durable',side_effect=save):
            with self.assertRaisesRegex(TimeoutError,'primary deadline'):
                m.full_prepare(self.base,self.plan,{},self.root/'secondary',
                    lambda:(_ for _ in ()).throw(TimeoutError('primary deadline')))

    def test_formal_groups_must_equal_the_independently_qualified_plan(self):
        qualified=self.root/'qualified-owner-plan.json';write(qualified,dict(verification=self.plan['verification']))
        ref=dict(owner_plan=dict(path=str(qualified),sha256=m.sha(qualified)))
        bindings={'1':ref,'8':ref.copy()}
        m.qualified_verification(self.plan['verification'],bindings)
        bad=json.loads(json.dumps(self.plan['verification']));bad['python']='different'
        with self.assertRaisesRegex(RuntimeError,'must match qualification'):m.qualified_verification(bad,bindings)
        bindings['8']={**ref,'other':'different'}
        with self.assertRaisesRegex(RuntimeError,'one independent'):m.qualified_verification(self.plan['verification'],bindings)
        bindings['8']=ref.copy();qualified.write_text('{}')
        with self.assertRaisesRegex(RuntimeError,'source'):m.qualified_verification(self.plan['verification'],bindings)

    def test_frozen_argument_helpers_reused_without_module_global_patch(self):
        self.assertIs(m.full_prepare,m._REUSED.full_prepare)
        self.assertIs(m.System,m._REUSED.System)
        self.assertIs(m.prior_replacement,m._REUSED.prior_replacement)


if __name__ == '__main__':
    unittest.main()
