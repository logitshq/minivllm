"""CPU fakes only: no real child, signal, device, process probe or service."""
from contextlib import ExitStack
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
m=module(HERE/'cold_owner.py','e024_isolation_cold_cpu')
c=module(HERE.parent/'omlx_cold_restart/controller_v6.py','e024_controller_cpu')


def write(path,value):path.write_text(json.dumps(value)+'\n')


class Child:
    pid=99002
    def __init__(self):self.returncode=None;self.finish=0;self.terminated=False;self.killed=False;self.stubborn=False
    def poll(self):return self.returncode
    def wait(self,timeout):
        if self.stubborn and self.returncode is None:raise subprocess.TimeoutExpired('fake',timeout)
        if self.returncode is None:self.returncode=self.finish
        return self.returncode
    def terminate(self):
        self.terminated=True
        if not self.stubborn:self.returncode=-15
    def kill(self):self.killed=True;self.returncode=-9


class System:
    def __init__(self,f):
        self.f=f;self.child=Child();self.calls=[];self.spawned=False;self.unknown=False
        self.bad_identity=False;self.live_child=False;self.stop=None;self.cleaned=self.restored=False
        self.result=dict(schema='E024-dispatch-isolation-result-v1',status='PASS',diagnostic_complete=True,
                         numerical_equal=False,full_model_qualified=False,performance_qualified=False)
    def identity(self,pid):
        if pid!=os.getpid() and self.bad_identity:raise RuntimeError('CPU failed child identity')
        return dict(pid=pid,ppid=os.getpid() if pid!=os.getpid() else 1,start='CPU identity',comm='python')
    def read(self,command,timeout=5):
        self.calls.append(command);job=command[0].endswith('launchctl')
        live=self.live_child and command[:3]==['/bin/ps','-p',str(self.child.pid)]
        return dict(command=command,returncode=0 if live else 113 if job else 1,stdout='live' if live else '')
    def spawn(self,command,log,env):
        self.spawned=True
        a=json.loads((self.f.out/'attempt.json').read_bytes())
        intent=json.loads((self.f.folder/'intent.json').read_bytes())
        assert intent['attempt_sha256']==m.sha(self.f.out/'attempt.json') and intent['owner_pid']==a['owner_pid']
        assert (self.f.out/'spawn-intent.json').exists()
        if self.unknown:raise RuntimeError('CPU unknown Popen outcome')
        q=Path(command[-1]);q.mkdir()
        write(q/'complete.json',self.result)
        return self.child
    def start_deadline(self,seconds,stop):assert 0<seconds<=1830;self.stop=stop
    def cleanup_mode(self):self.cleaned=True
    def postflight_deadline(self):pass
    def restore_deadline(self):self.restored=True


class Resources:
    def __init__(self):self.pressure=1;self.callback=None;self.foreign=False
    def snapshot(self,label,pid,*,suspended_pids):
        assert suspended_pids==()
        if self.callback:self.callback(label)
        rows=[] if pid is None else [dict(pid=pid,owned=True,authorized_suspended=False)]
        if self.foreign:rows.append(dict(pid=88888,owned=False,authorized_suspended=False))
        return dict(label=label,pressure_level_value=self.pressure,authorized_suspended_pids=[],
            native_inference_processes=rows,power_metadata={'source':'AC Power'},processinfo_thermal_metadata={'code':0})
    def guard(self,samples):pass
    def deltas(self,samples):return dict(samples=len(samples))


class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name).resolve();self.out=self.root/'owner';self.folder=self.out/'phases'/m.PHASE
        self.journal=self.root/'journal';self.journal.mkdir();(self.journal/'.lock').touch()
        self.base=m.load('../pareto_realistic/qualify_absent_owner.py')
        self.plan_path=self.root/'wrapper.json'
        self.prior={'replacement':dict(pid=90204,ppid=1,start='CPU old service',comm='omlx-server')}
        self.plan=dict(schema='E024-isolation-cold-owner-plan-v1',execution=dict(owner=dict(path=str(HERE/'cold_owner.py'),sha256=m.sha(HERE/'cold_owner.py')),pins={}),
            trials=[m.PHASE],deadline_seconds=1830,diagnostic_owner_plan={},isolation_plan={'sha256':'CPU-isolation-plan'},cold_restart=dict(journal_path=str(self.journal),
            controller=dict(path=str(Path(c.__file__)),sha256=m.sha(c.__file__)),predecessor={'CPU':'archive'},
            approved_contract=c.CONTRACT,proposal_plan_sha256=c.PLAN_SHA))
        write(self.plan_path,self.plan);self.args=NS(plan=self.plan_path,plan_sha256=m.sha(self.plan_path),out=self.out)
        self.j=dict(schema='omlx-cold-restart-journal-v6',id='CPU-journal',state='STOPPED',removed_at=time.time()-10,
            controller_sha256=self.plan['cold_restart']['controller']['sha256'],plan_sha256=c.PLAN_SHA,
            approved_contract=c.CONTRACT,original=self.prior['replacement'],owner_directory=str(self.out),
            owner_plan=dict(path=str(self.plan_path),sha256=self.args.plan_sha256),owner_source=self.plan['execution']['owner'],
            predecessor=self.plan['cold_restart']['predecessor'],predecessor_copies={},trials=[m.PHASE])
        write(self.journal/'journal.json',self.j)
        self.system=System(self);self.resources=Resources();self.prepare_error=None;self.postflight_error=None
        self.stack=ExitStack();self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(m,'validate_plan',return_value=(self.base,self.plan,self.plan_path.read_bytes(),{},self.prior)))
        self.stack.enter_context(patch.object(m,'prepare',side_effect=self.prepare))
    def prepare(self,plan,out,check):
        check()
        if self.prepare_error:raise self.prepare_error
        if out.name=='postflight' and self.postflight_error:raise self.postflight_error
        out.mkdir()
        return ['python','-I','-B',str(HERE/'qualify.py'),'--plan','CPU-fixed-plan','--plan-sha256',m.QUALIFICATION_SHA,'--out',str(out/'qualification')],{'MLX_TEST':'fixed'}
    def run_owner(self):return m.run(self.args,system=self.system,resources=self.resources)
    def reap(self):
        s=NS(process=lambda pid:None)
        return c.reaped(NS(data=self.j,directory=self.journal),s)
    def test_success_actual_single_child_protocol(self):
        self.assertEqual(self.run_owner(),0)
        self.assertFalse(self.system.result['numerical_equal'])
        complete=json.loads((self.out/'complete.json').read_bytes())
        self.assertFalse(complete['full_model_workload_qualified'] or complete['performance_qualified'])
        self.assertFalse(self.reap()['no_child_started'])
        index=json.loads((self.out/'children.json').read_bytes())
        self.assertEqual([r['phase'] for r in index['children']],[m.PHASE])
        self.assertTrue(self.system.cleaned and self.system.restored)
        self.assertIn(['/bin/ps','-p','90204','-o','pid='],self.system.calls)
        self.assertFalse(any('74384' in cmd for cmd in self.system.calls))
    def test_incomplete_diagnostic_is_not_success_but_child_can_be_reaped(self):
        self.system.result['diagnostic_complete']=False
        self.assertEqual(self.run_owner(),1)
        self.assertFalse(self.reap()['no_child_started'])
        self.assertIn('diagnostic producer incomplete',json.loads((self.out/'first-error.json').read_bytes())['error'])
    def test_prespawn_failure_can_restore_without_child(self):
        self.prepare_error=RuntimeError('CPU preflight failure')
        self.assertEqual(self.run_owner(),1);self.assertFalse(self.system.spawned)
        self.assertTrue(self.reap()['no_child_started'])
    def test_pressure_foreign_and_cancel_before_spawn(self):
        self.resources.pressure=2;self.assertEqual(self.run_owner(),1)
        self.assertFalse(self.system.spawned);self.assertTrue(self.reap()['no_child_started'])
    def test_known_child_identity_failure_is_reaped(self):
        self.system.bad_identity=True;self.assertEqual(self.run_owner(),1)
        self.assertTrue(self.system.child.terminated);self.assertFalse(self.reap()['no_child_started'])
        self.assertIsNone(json.loads((self.folder/'child.json').read_bytes())['identity'])
    def test_unknown_spawn_has_no_terminal_or_false_release(self):
        self.system.unknown=True;self.assertEqual(self.run_owner(),1)
        self.assertFalse((self.out/'terminal.json').exists() or (self.out/'release.json').exists())
        with self.assertRaises(c.NotReady):self.reap()
    def test_numerical_failure_reaps_and_retains_primary_after_postflight_failure(self):
        self.system.child.finish=1;self.postflight_error=RuntimeError('CPU postflight error')
        self.assertEqual(self.run_owner(),1);self.assertFalse(self.reap()['no_child_started'])
        errors=json.loads((self.out/'errors.json').read_bytes());self.assertIn('diagnostic exit 1',errors[0]['error'])
        self.assertEqual(errors[-1]['phase'],'postflight')
    def test_cancellation_kills_stubborn_child_and_preserves_resources(self):
        self.system.child.stubborn=True
        def cancel(label):
            if label.endswith(':running'):self.system.stop()
        self.resources.callback=cancel;self.assertEqual(self.run_owner(),1)
        self.assertTrue(self.system.child.terminated and self.system.child.killed)
        self.assertFalse(self.reap()['no_child_started'])
        self.assertTrue((self.out/'first-error.json').exists())
    def test_surviving_pid_never_certified_reaped(self):
        self.system.live_child=True;self.assertEqual(self.run_owner(),1)
        self.assertFalse((self.out/'release.json').exists())
        with self.assertRaises(c.NotReady):self.reap()
    def test_real_validator_keeps_original_plans_without_runtime_hash(self):
        self.stack.close()
        plan=copy.deepcopy(self.plan);plan['execution']['pins']=m.source_pins()
        old=HERE.parent/'long_decode_graph/prereg-v1/owner-plan.json'
        plan['diagnostic_owner_plan']=dict(path=str(old),sha256=m.OWNER_PLAN_SHA)
        original=json.loads(old.read_bytes());isolation=self.root/'isolation-plan.json'
        write(isolation,dict(schema='E024-dispatch-isolation-plan-v1',qualifier_sha256=m.FIXED['qualify.py'],
            e024_plan=original['qualification_plan'],deadline_s=1800))
        plan['isolation_plan']=dict(path=str(isolation),sha256=m.sha(isolation))
        write(self.plan_path,plan);self.args.plan_sha256=m.sha(self.plan_path)
        with patch.object(m,'_predecessor',return_value=self.prior):
            _,parsed,_,numeric,_=m.validate_plan(self.args)
            self.assertEqual(parsed['diagnostic_owner_plan']['sha256'],m.OWNER_PLAN_SHA)
            self.assertEqual(numeric['deadline_s'],1800)
            for key,value in [('trials',['r1-c1']),('deadline_seconds',1831)]:
                bad=copy.deepcopy(plan);bad[key]=value;write(self.plan_path,bad);self.args.plan_sha256=m.sha(self.plan_path)
                with self.assertRaises(ValueError):m.validate_plan(self.args)


    def test_isolation_prepare_retains_original_plan_and_only_adapts_child_command(self):
        self.stack.close();out=self.root/'prepared';newplan=self.root/'isolation.json';write(newplan,{'CPU':'isolated'})
        old=self.root/'old-owner-plan.json';write(old,{})
        plan=dict(diagnostic_owner_plan=dict(path=str(old),sha256=m.sha(old)),
                  isolation_plan=dict(path=str(newplan),sha256=m.sha(newplan)))
        original=lambda *args:None;calls=[];environment={'MLX_TEST':'same'}
        def prepare(args,check,legacy):
            check();calls.append((args,legacy));(args.out/'e023-input-preflight').mkdir()
            return ['python','-I','-B','old-qualifier','--plan','old-plan','--out',str(args.out/'qualification')],environment
        adapter=NS(prepare=prepare,load_legacy=lambda:NS(prepare=original))
        real_load=m.load
        with patch.object(m,'load',side_effect=lambda name:adapter if name=='../long_decode_graph/owner.py' else real_load(name)):
            command,env=m.prepare(plan,out,lambda:None)
        self.assertEqual(calls[0][0].plan,old);self.assertIs(calls[0][1],original);self.assertIs(env,environment)
        self.assertEqual(command,['python','-I','-B',str(HERE/'qualify.py'),'--plan',str(newplan),
                                 '--plan-sha256',m.sha(newplan),'--out',str(out/'qualification')])
        self.assertEqual(json.loads((out/'inherited-launch.json').read_bytes())['command'][3],'old-qualifier')
        self.assertEqual((out/'isolation-plan.json').read_bytes(),newplan.read_bytes())



if __name__=='__main__':unittest.main()
