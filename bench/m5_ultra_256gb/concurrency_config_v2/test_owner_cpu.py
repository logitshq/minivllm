"""Fake ownership and threaded cancellation tests; no GPU/service operations."""
import asyncio
import contextlib
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from bench.m5_ultra_256gb.concurrency_config_v2 import execute as e
from bench.m5_ultra_256gb.concurrency_config_v2 import runner as r


def record(pid, ppid, start='known'):
    return dict(pid=pid, ppid=ppid, start=start, executable='/cpu/python')


class Lease:
    pid = 100
    def __init__(self, timeout=False): self.timeout=timeout; self.waits=[]
    def wait(self, timeout):
        self.waits.append(timeout)
        if self.timeout and len(self.waits)==1: raise subprocess.TimeoutExpired('fake lease',timeout)
        return 0


class Cleanup(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);(self.root/'lease').mkdir();(self.root/'screen/resources').mkdir(parents=True)
        self.bound=record(200,100);self.server=record(300,200);self.command=['/cpu/python','runner']
        self.plan=dict(outputs={'lease':str(self.root/'lease'),'screen':str(self.root/'screen')},screen_command=self.command)
        (self.root/'lease/lease.json').write_text(json.dumps(dict(owner_pid=100,child_pid=200,jobs=[self.command])))
        self.states={200:self.bound,300:self.server};self.signals=[]
        (self.root/'screen/resources/identity-300.json').write_text(json.dumps(self.server))
        self.stack=contextlib.ExitStack();self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(e.runner,'process',side_effect=lambda pid:self.states.get(pid)))
        self.stack.enter_context(patch.object(e.os,'kill',side_effect=self.kill))
    def kill(self,pid,sig):
        self.assertNotEqual(pid,100,'lease must never be signalled');self.signals.append((pid,sig))
        if sig==signal.SIGKILL:self.states.pop(pid,None)
    def test_natural_exit_keeps_binding_and_only_waits_for_lease(self):
        self.states.clear();lease=Lease()
        self.assertEqual(e.bind_runner(lease,self.plan,self.bound),self.bound)
        self.assertEqual(e.bounded_cleanup(lease,self.plan,self.bound),[])
        self.assertEqual(lease.waits,[90]);self.assertEqual(self.signals,[])
    def test_runner_TERM_can_finish_cleanup_naturally(self):
        lease=Lease();actions=e.bounded_cleanup(lease,self.plan,self.bound)
        self.assertEqual(self.signals,[(200,signal.SIGTERM)])
        self.assertEqual(actions,[dict(pid=200,signal='TERM')]);self.assertEqual(lease.waits,[90])
    def test_timeout_resolves_server_before_killing_runner_and_never_lease(self):
        clock=[0]
        def now():clock[0]+=20;return clock[0]
        with patch.object(e.time,'monotonic',side_effect=now),patch.object(e.time,'sleep'),patch.object(e.subprocess,'run',return_value=SimpleNamespace(stdout='',returncode=0)):
            lease=Lease(timeout=True);actions=e.bounded_cleanup(lease,self.plan,self.bound)
        self.assertEqual(self.signals,[(200,signal.SIGTERM),(300,signal.SIGTERM),(300,signal.SIGKILL),(200,signal.SIGKILL)])
        self.assertEqual(lease.waits,[90,20]);self.assertEqual(len(actions),4);self.assertEqual(self.states,{})
    def test_reused_runner_is_rejected_before_any_signal(self):
        self.states[200]=record(200,100,'reused')
        with self.assertRaisesRegex(AssertionError,'runner identity changed'):e.bounded_cleanup(Lease(),self.plan,self.bound)
        self.assertEqual(self.signals,[])
    def test_reused_server_is_rejected_without_signalling_that_pid(self):
        self.states[300]=record(300,200,'reused')
        with self.assertRaisesRegex(AssertionError,'server identity changed'):e.bounded_cleanup(Lease(timeout=True),self.plan,self.bound)
        self.assertEqual(self.signals,[(200,signal.SIGTERM)])
    def test_unrecorded_child_prevents_runner_KILL(self):
        self.states.pop(300)
        with patch.object(e.subprocess,'run',return_value=SimpleNamespace(stdout='999 200\n',returncode=0)):
            with self.assertRaisesRegex(AssertionError,'unresolved runner child'):e.bounded_cleanup(Lease(timeout=True),self.plan,self.bound)
        self.assertEqual(self.signals,[(200,signal.SIGTERM)])


class Checkpoint(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.service=record(111,1);self.child=record(222,os.getpid());self.events=[];self.calls=[]
        spec=self.root/'spec.json';spec.write_text(json.dumps(dict(label='cpu-only')))
        self.plan=dict(pins={},specs=[dict(path=str(spec),sha256=r.sha(spec))],service=dict(pid=111,process=self.service),
            physical_baseline=dict(swap_bytes=0,swapouts=0),screen_deadline_seconds=1800,max_observation_gap_seconds=12)
        self.sample=dict(power_metadata={'source':'AC Power'},processinfo_thermal_metadata={'code':0})
        self.stack=contextlib.ExitStack();self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(r,'process',side_effect=lambda pid:self.service if pid==111 else self.child))
        self.stack.enter_context(patch.object(r.subprocess,'run',side_effect=self.command))
        self.stack.enter_context(patch.object(r.physical,'checked',return_value=dict(swap_bytes=0,swapouts=0,
            top_bytes=123,aux_current_bytes=123,aux_peak_bytes=123)))
        self.stack.enter_context(patch.object(r.resources,'guard'))
    def command(self,argv,**kwargs):
        self.calls.append((argv,kwargs));self.assertEqual(kwargs['timeout'],5)
        if argv[0]=='/bin/ps':return SimpleNamespace(stdout='T\n',returncode=0)
        self.assertEqual(argv[1:4],['-I','-B','-c']);self.assertIn('resources_v2',argv[4])
        return SimpleNamespace(stdout=json.dumps(self.sample),returncode=0)
    def measure(self,path,pid):
        now=time.monotonic();self.events.append('probe-ended')
        return dict(started_monotonic=now,ended_monotonic=now)
    async def test_normal_checkpoint_persists_identity_and_result_interface(self):
        async def fake_trial(spec,out,*,checkpoint):
            await checkpoint('pre-spawn',None);await checkpoint('spawned',222)
            return dict(label=spec['label'],status='PASS',server_reaped=True,cleanup_complete=True,point={'diagnostic':True})
        with patch.object(r,'load',return_value=SimpleNamespace(run=fake_trial)),patch.object(r.physical,'measure',side_effect=self.measure),contextlib.redirect_stdout(io.StringIO()):
            code=await r.run(self.plan,self.root/'screen')
        self.assertEqual(code,0)
        result=json.loads((self.root/'screen/result.json').read_text())
        self.assertEqual(result['status'],'PASS_CAPTURE_ONLY');self.assertEqual(result['resource_samples'],2)
        self.assertEqual(result['physical_samples'],2);self.assertFalse(result['performance_qualified'])
        self.assertEqual(json.loads((self.root/'screen/resources/identity-222.json').read_text()),self.child)
        self.assertEqual(len([c for c in self.calls if c[0][0]!='/bin/ps']),2)
    async def test_repeated_cancel_waits_for_probe_before_cleanup(self):
        entered=threading.Event();release=threading.Event()
        def blocked(path,pid):
            entered.set()
            if not release.wait(3):raise TimeoutError('test release deadline')
            return self.measure(path,pid)
        async def fake_trial(spec,out,*,checkpoint):
            try:await checkpoint('spawned',222)
            finally:self.events.append('trial-cleanup')
        with patch.object(r,'load',return_value=SimpleNamespace(run=fake_trial)),patch.object(r.physical,'measure',side_effect=blocked),contextlib.redirect_stdout(io.StringIO()):
            task=asyncio.create_task(r.run(self.plan,self.root/'screen'))
            try:
                for _ in range(1000):
                    if entered.is_set():break
                    await asyncio.sleep(.001)
                self.assertTrue(entered.is_set());task.cancel();await asyncio.sleep(.005);task.cancel();await asyncio.sleep(.005)
                self.assertFalse(task.done());self.assertNotIn('trial-cleanup',self.events)
            finally:release.set()
            code=await task
        self.assertEqual(code,1);self.assertEqual(self.events,['probe-ended','trial-cleanup'])
        result=json.loads((self.root/'screen/result.json').read_text())
        self.assertEqual(result['status'],'FAIL');self.assertEqual(result['errors'][0]['type'],'CancelledError')
        self.assertEqual(result['resource_samples'],1)


if __name__=='__main__':unittest.main()
