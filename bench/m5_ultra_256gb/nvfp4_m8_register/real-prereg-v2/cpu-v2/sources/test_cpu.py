"""Pure wait proofs plus fake full-owner reproductions; no process or service I/O."""
from contextlib import ExitStack, redirect_stdout
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

P=Path(__file__).with_name('execute.py')
s=importlib.util.spec_from_file_location('m8_register_owner_v2',P);o=importlib.util.module_from_spec(s);s.loader.exec_module(o)
COMMAND=['fake-python','--run','--out','fake-output']
EXPECTED=dict(pid=88,ppid=77,start='known start',executable='fake-python')
def lease():return dict(child_pid=88,owner_pid=77,current_job=0,jobs=[COMMAND],closed=False)
def job(code=0):return dict(pid=88,command=COMMAND,waited=True,returncode=code)

class Scenario:
    def __init__(self,root,physical_failure=False,code=0,identity_mismatch=False,stall=False):
        self.root=root;self.physical_failure=physical_failure;self.code=code;self.identity_mismatch=identity_mismatch;self.stall=stall
        self.now=0;self.reaped=False;self.polls=0;self.identity_calls=0;self.measures=[];self.actions=[]
        self.pre=root/'prepared';self.pre.mkdir();self.lease_dir=root/'lease';self.lease_dir.mkdir()
        self.raw=root/'raw';self.raw.mkdir();self.command=['fake-python','--run','--out',str(self.raw)]
        self.leased=lease();self.leased['jobs']=[self.command];self.put(self.lease_dir/'lease.json',self.leased)
        for label in ('head','up'):
            case=self.raw/label/'T8';case.mkdir(parents=True)
            self.put(case/'numerical.json',dict(status='PASS'))
            (case/'timings.jsonl').write_text((''+json.dumps(dict(elapsed_ns=1))+'\n')*200)
        self.put(self.raw/'result.json',dict(status='PASS_LOCAL_PROJECTION_ONLY',cases=[{},{}]))
        source=root/'source.py';source.write_text('# fake pinned source\n')
        self.plan=dict(pins={str(source):o.sha(source)},environment={k:v for k,v in os.environ.items() if k.startswith('MLX_')},
                       child=self.command,outputs=dict(execution=str(root/'execution'),lease=str(self.lease_dir)),scope='fake lifecycle')
        self.put(self.pre/'plan.json',self.plan)
        self.resources=ModuleType('resources_v2');self.resources.snapshot=lambda *a,**kw:dict(pressure_level_value=1,swap_used_mb=0,power_metadata=dict(source='AC Power'),processinfo_thermal_metadata=dict(code=0))
        self.resources.guard=lambda *a:None
        self.physical=ModuleType('physical_guard');self.physical.policy=lambda **kw:dict(child_limit_bytes=4<<30)
        self.physical.measure=self.measure;self.physical.checked=lambda *a:dict(top_bytes=1,aux_current_bytes=1,aux_peak_bytes=1,swap_bytes=0,swapouts=0)
        scenario=self
        class Process:
            pid=77
            returncode=None
            def poll(self):
                if scenario.reaped and self.returncode is None:
                    scenario.polls+=1
                    if scenario.polls>=2 and not scenario.stall:scenario.close(self)
                return self.returncode
            def terminate(self):scenario.actions.append('TERM');raise AssertionError('must not interrupt waited-child cleanup')
            def kill(self):scenario.actions.append('KILL');raise AssertionError('must not kill cleanup')
            def wait(self,timeout):
                scenario.actions.append(('WAIT',timeout));scenario.close(self);return self.returncode
        self.process=Process()
    def put(self,p,v):p.write_text(json.dumps(v))
    def reap(self):
        self.reaped=True;j=job(self.code);j['command']=self.command;self.put(self.lease_dir/'job-0.json',j)
    def close(self,process):
        self.leased['closed']=True;self.put(self.lease_dir/'lease.json',self.leased)
        self.put(self.lease_dir/'resume.json',dict(identity_match=True,state_after='S'))
        process.returncode=0 if self.code==0 else 1
    def identity(self,pid):
        assert pid==88;self.identity_calls+=1
        if self.reaped:return None
        if self.identity_calls==2 and self.identity_mismatch:
            self.reap();return dict(EXPECTED,start='changed identity')
        if self.identity_calls==3:self.reap();return None
        return dict(EXPECTED)
    def measure(self,path,pid):
        self.measures.append(pid)
        if pid==88 and self.physical_failure:
            self.reap();raise ValueError('physical command failed/unknown')
        return dict(started_monotonic=self.now,ended_monotonic=self.now+.01)
    def sleep(self,seconds):self.now+=1
    def run(self):
        with ExitStack() as stack:
            for name,value in dict(HERE=self.pre,identity=lambda:o.IDENTITY,health=lambda:dict(healthy=True),process_record=self.identity).items():
                stack.enter_context(patch.object(o,name,value))
            stack.enter_context(patch.dict(sys.modules,dict(resources_v2=self.resources,physical_guard=self.physical)))
            stack.enter_context(patch.object(sys,'path',sys.path.copy()))
            stack.enter_context(patch.object(o.subprocess,'Popen',return_value=self.process))
            stack.enter_context(patch.object(o.subprocess,'run',return_value=SimpleNamespace(stdout='S\n',stderr='',returncode=0)))
            stack.enter_context(patch.object(o.signal,'signal'))
            stack.enter_context(patch.object(o.time,'monotonic',side_effect=lambda:self.now))
            stack.enter_context(patch.object(o.time,'sleep',side_effect=self.sleep))
            stack.enter_context(redirect_stdout(io.StringIO()))
            result=o.run(o.sha(self.pre/'plan.json'))
        return result,json.loads((self.root/'execution/final.json').read_text())

class Tests(unittest.TestCase):
    def test_proven_wait_accepts_nonzero_for_cleanup_only(self):
        for code in (0,7,-9):self.assertEqual(o.waited_job(lease(),job(code),EXPECTED,77,COMMAND),job(code))
    def test_unknown_or_mismatched_wait_is_rejected(self):
        changes=[('job','waited',False),('job','returncode',True),('job','returncode',None),('job','pid',89),('job','command',['other']),
                 ('lease','owner_pid',99),('lease','child_pid',89),('lease','current_job',1),('lease','jobs',[['other']]),('expected','ppid',99)]
        for target,key,value in changes:
            with self.subTest(target=target,key=key):
                l,j,e=lease(),job(),dict(EXPECTED);dict(lease=l,job=j,expected=e)[target][key]=value
                with self.assertRaises(AssertionError):o.waited_job(l,j,e,77,COMMAND)
        with self.assertRaises(AssertionError):o.waited_job(lease(),job(),None,77,COMMAND)
    def test_natural_reap_stops_sampling_and_allows_lease_finally(self):
        with tempfile.TemporaryDirectory() as tmp:
            s=Scenario(Path(tmp));code,result=s.run()
            self.assertEqual(code,0);self.assertEqual(result['status'],'PASS_LOCAL_PROJECTION_ONLY')
            self.assertTrue(result['lease_closed']);self.assertEqual(s.measures.count(88),1)
            self.assertEqual(s.actions,[]);self.assertTrue((s.root/'execution/numerical-child-waited.json').exists())
    def test_mid_probe_failure_stays_failed_but_does_not_term_cleanup(self):
        with tempfile.TemporaryDirectory() as tmp:
            s=Scenario(Path(tmp),physical_failure=True);code,result=s.run()
            self.assertEqual(code,1);self.assertEqual(result['status'],'FAIL');self.assertTrue(result['lease_closed'])
            self.assertTrue(any('physical command failed/unknown' in e['message'] for e in result['errors']))
            self.assertEqual(s.actions,[('WAIT',30)])
    def test_live_identity_mismatch_is_not_reclassified_as_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            s=Scenario(Path(tmp),identity_mismatch=True);code,result=s.run()
            self.assertEqual(code,1);self.assertEqual(result['status'],'FAIL');self.assertTrue(result['lease_closed'])
            self.assertTrue(any('child exited/changed during physical observation' in e['message'] for e in result['errors']))
            self.assertEqual(s.actions,[('WAIT',30)])
    def test_drain_remains_bounded_by_original_owner_deadline(self):
        with tempfile.TemporaryDirectory() as tmp:
            s=Scenario(Path(tmp),stall=True);code,result=s.run()
            self.assertEqual(code,1);self.assertEqual(result['status'],'FAIL')
            self.assertTrue(any('bounded owner deadline' in e['message'] for e in result['errors']))
            self.assertGreaterEqual(s.now,160);self.assertEqual(s.actions,[('WAIT',30)])
            self.assertEqual(s.measures.count(88),1)
    def test_nonzero_wait_finishes_cleanup_and_remains_failed(self):
        with tempfile.TemporaryDirectory() as tmp:
            s=Scenario(Path(tmp),code=7);code,result=s.run()
            self.assertEqual(code,1);self.assertEqual(result['status'],'FAIL');self.assertTrue(result['lease_closed'])
            self.assertNotIn('TERM',s.actions);self.assertNotIn('KILL',s.actions)

if __name__=='__main__':unittest.main()
