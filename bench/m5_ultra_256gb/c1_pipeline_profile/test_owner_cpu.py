"""Unexecuted fake-only owner-glue contracts, not real cold/process evidence.

Capture's raw client/drain tests are separate. These fakes exercise callback
ownership and cancellation settlement only; no fake controller/auditor is made.
"""
import asyncio
import copy
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest

from bench.m5_ultra_256gb.c1_pipeline_profile import owner as m


def passing(mode):
    result=dict(schema='C1-pipeline-diagnostic-capture-result-v1',mode=mode,status='PASS',errors=[],
        count_verified=True,natural_exit=True,forced_stop=False,server_reaped=True,cleanup_complete=True,
        cleanup_task_resolved=True,client_task_resolved=True,drain_client_resolved=True,small_pins_rechecked=True,
        numerical_qualified=False,performance_qualified=False)
    result['closed_server']=dict(status='RECORDED',requests=16,numerical_qualified=False,performance_qualified=False)
    return result


class FakeCapture:
    def validate(self,spec):
        if spec['mode'] not in m.MODES: raise ValueError('fake malformed mode')
    async def run(self,spec,out,*,checkpoint,backend):
        out.mkdir();m.write(out/'spawn-intent.json',dict(command=['CPU-NO-EXEC']))
        child=backend.spawn(['CPU-NO-EXEC'],{},None,None)
        await checkpoint('spawned',child.pid)
        child.returncode=0
        m.write(out/'release.json',dict(pid=child.pid,returncode=0,waited_reaped=True,
            strict_absence=True,port_absent=True,natural_exit=True,forced_stop=False))
        return passing(spec['mode'])


class OwnerGlue(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.phase=Path(self.temp.name).resolve()/'off';self.phase.mkdir();self.out=self.phase/'trial'
        self.spec=dict(mode='off',output_directory=str(self.out))
        self.child=SimpleNamespace(pid=23456,returncode=None);self.events=[]
        self.capture=FakeCapture();self.ref=dict(path='CPU-FAKE-DEPENDENCY',sha256='no-device-proof')
        self.backend=SimpleNamespace(spawn=self.spawn)
    def spawn(self,*args): self.events.append(('Popen-returned',self.child));return self.child
    def registered(self,mode,thunk):
        self.events.append(('durable-outer-intent',mode));child=thunk()
        self.events.append(('tracked',child));return child
    def probe(self,label,pid,*,cleanup): self.events.append(('probe',label,pid,cleanup))
    def reap(self,mode):
        self.events.append(('reap',mode,self.child))
        if self.child.returncode is None: self.child.returncode=-15
        m.write(self.phase/'release.json',dict(phase=mode,pid=self.child.pid,returncode=self.child.returncode,
            waited_reaped=True,strict_absence=True))
    async def run_glue(self,**kwargs):
        options=dict(capture_source=self.ref,registered_spawn=self.registered,resource_probe=self.probe,
            strict_reap=self.reap,backend=self.backend,_capture=self.capture);options.update(kwargs)
        return await m.run_mode(self.spec,self.out,**options)
    async def test_same_actual_popen_is_registered_probed_reaped_and_not_performance(self):
        result=await self.run_glue()
        self.assertEqual(result['status'],'PASS',result['errors'])
        self.assertEqual(result['actual_server_pid'],23456);self.assertTrue(result['outer_reaped'])
        self.assertIs(self.events[1][1],self.child);self.assertIs(self.events[2][1],self.child)
        self.assertIn(('probe','off:spawned',23456,False),self.events)
        self.assertFalse(result['performance_qualified'] or result['numerical_qualified'] or result['restoration_verified'])
    async def test_outer_identity_failure_after_return_keeps_known_child_for_reap(self):
        def broken(mode,thunk): thunk();raise RuntimeError('identity/fsync failure after actual Popen return')
        result=await self.run_glue(registered_spawn=broken)
        self.assertEqual(result['status'],'FAIL');self.assertTrue(result['popen_returned'] and result['outer_reaped'])
        self.assertFalse(result['unknown_launch']);self.assertEqual(self.child.returncode,-15)
        self.assertTrue((self.out/'spawn-intent.json').exists());self.assertFalse((self.out/'release.json').exists())
    async def test_unknown_popen_never_becomes_no_child_or_reaped(self):
        def unknown(*args): raise OSError('unknown Popen outcome')
        self.backend.spawn=unknown
        result=await self.run_glue()
        self.assertEqual(result['status'],'FAIL');self.assertTrue(result['unknown_launch'])
        self.assertFalse(result['popen_returned'] or result['outer_reaped'])
        self.assertFalse(any(e[0]=='reap' for e in self.events))
        self.assertFalse((self.phase/'release.json').exists());self.assertFalse((self.phase/'terminal.json').exists())
    async def test_wrong_resource_pid_rejected_before_resource_callback(self):
        class WrongPID(FakeCapture):
            async def run(inner,spec,out,*,checkpoint,backend):
                out.mkdir();backend.spawn([],{},None,None);await checkpoint('spawned',99999)
        result=await self.run_glue(_capture=WrongPID())
        self.assertEqual(result['status'],'FAIL');self.assertTrue(result['outer_reaped'])
        self.assertFalse(any(e[0]=='probe' for e in self.events))
    async def test_repeated_cancel_settles_probe_before_reap(self):
        loop=asyncio.get_running_loop();started=asyncio.Event();finish=threading.Event();self.addCleanup(finish.set)
        def slow_probe(*args,**kwargs):
            loop.call_soon_threadsafe(started.set)
            if not finish.wait(2): raise TimeoutError('fake probe fixture deadline')
            self.events.append(('probe-settled',))
        task=asyncio.create_task(self.run_glue(resource_probe=slow_probe))
        await started.wait();task.cancel();await asyncio.sleep(0);task.cancel();await asyncio.sleep(0)
        self.assertFalse(any(e[0]=='reap' for e in self.events));finish.set()
        result=await task
        self.assertEqual(result['status'],'FAIL');self.assertTrue(result['outer_reaped'])
        labels=[e[0] for e in self.events];self.assertLess(labels.index('probe-settled'),labels.index('reap'))
    async def test_cancel_during_strict_reap_cannot_turn_clean_capture_into_pass(self):
        loop=asyncio.get_running_loop();started=asyncio.Event();finish=threading.Event();self.addCleanup(finish.set)
        def slow_reap(mode):
            loop.call_soon_threadsafe(started.set)
            if not finish.wait(2): raise TimeoutError('fake reap fixture deadline')
            self.reap(mode)
        task=asyncio.create_task(self.run_glue(strict_reap=slow_reap))
        await started.wait();task.cancel();await asyncio.sleep(0);task.cancel();finish.set()
        result=await task
        self.assertEqual(result['capture_status'],'PASS');self.assertEqual(result['status'],'FAIL')
        self.assertTrue(result['outer_reaped']);self.assertTrue(result['errors'])
    async def test_inner_outer_release_disagreement_remains_failure(self):
        def bad(mode):
            self.reap(mode);inner=json.loads((self.out/'release.json').read_bytes());inner['pid']=99999
            m.write(self.out/'release.json',inner)
        result=await self.run_glue(strict_reap=bad)
        self.assertEqual(result['status'],'FAIL');self.assertTrue(result['outer_reaped'])
        self.assertTrue(any('release mismatch' in e['message'] for e in result['errors']))
    async def test_missing_callback_fails_before_capture_or_spawn(self):
        with self.assertRaisesRegex(ValueError,'callbacks required'):
            await self.run_glue(strict_reap=None)
        self.assertEqual(self.events,[]);self.assertFalse(self.out.exists())


class Recipe(unittest.TestCase):
    def fake(self):
        def prepare(ref,mode,**kwargs):
            flags=dict(route_decode_graph_enabled=True,gdn_prework_enabled=True,identity_refill_enabled=False)
            return dict(mode=mode,formal_spec=copy.deepcopy(ref),concurrency=1,
                output_directory=str(kwargs['output_directory']),drain_token=kwargs['drain_token'],
                identity=dict(combined_entry=dict(combined_candidates=flags)),
                jobs={'warm':list(range(8)),'diagnostic':list(range(8))},environment={},transport={},
                deadlines={},fixtures=[],tokenizer={},qualification={})
        return SimpleNamespace(prepare=prepare)
    def test_order_and_modes_preserve_one_reference_and_workload(self):
        with tempfile.TemporaryDirectory() as temp:
            recipe=m.prepare_modes({'path':'same-C1ON','sha256':'fake'},Path(temp).resolve(),
                dict(off='nonce0',runner='nonce1',full='nonce2'),capture_source={},server_sha256='fake',
                order=('full','off','runner'),_capture=self.fake())
        self.assertEqual(recipe['order'],['full','off','runner'])
        self.assertFalse(recipe['execution_registered'] or recipe['performance_qualified'])
        self.assertEqual(len({s['output_directory'] for s in recipe['specs']}),3)
        self.assertTrue(all(s['formal_spec']==recipe['specs'][0]['formal_spec'] for s in recipe['specs']))
    def test_duplicate_mode_nonce_or_noncombined_configuration_is_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            args=({},Path(temp).resolve(),dict(off='a',runner='b',full='c'))
            with self.assertRaises(ValueError):
                m.prepare_modes(*args,capture_source={},server_sha256='fake',order=('off','off','full'),_capture=self.fake())
            with self.assertRaises(ValueError):
                m.prepare_modes({},Path(temp).resolve(),dict(off='a',runner='a',full='c'),capture_source={},server_sha256='fake',_capture=self.fake())
            fake=self.fake();original=fake.prepare
            def disabled(*a,**kw):
                spec=original(*a,**kw);spec['identity']['combined_entry']['combined_candidates']['gdn_prework_enabled']=False;return spec
            fake.prepare=disabled
            with self.assertRaisesRegex(ValueError,'combined-ON'):
                m.prepare_modes(*args,capture_source={},server_sha256='fake',_capture=fake)


if __name__=='__main__':unittest.main()
