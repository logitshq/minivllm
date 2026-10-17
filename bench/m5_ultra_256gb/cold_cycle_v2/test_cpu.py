"""Files and fake processes/services only; never invokes a system backend."""
import ast
from contextlib import ExitStack, closing
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
c=module(HERE/'controller.py','stable_controller_cpu')
m=module(HERE/'owner.py','stable_owner_cpu')
a=module(HERE/'audit_restore.py','stable_restore_cpu')
def write(path,value):Path(path).write_text(json.dumps(value,indent=2)+'\n')
def ref(path):return dict(path=str(Path(path).resolve()),sha256=c.b.sha(path))
def migration():
    folder=c.ARCHIVE_ROOT/'E024-route-canary-v1'
    return dict(path=str(folder/'restored-journal.json'),sha256=c.MIGRATION_JOURNAL_SHA,
                audit=dict(path=str(folder/'restoration-audit.json'),sha256=c.MIGRATION_AUDIT_SHA))
def plan_at(root,name='run',predecessor=None):
    source=HERE.parent/'route_decode_graph'
    plan=dict(schema=c.RUN_SCHEMA,lifecycle=ref(HERE/'lifecycle.json'),predecessor=predecessor or migration(),
        experiment=dict(phase=name,qualifier=ref(source/'qualify.py'),plan=ref(source/'prereg-canary-v1/route-plan.json'),
                        source_pins={str(source/'qualify.py'):c.b.sha(source/'qualify.py'),str(source/'candidate.py'):c.b.sha(source/'candidate.py')}),
        inherited_owner_plan=ref(HERE.parent/'long_decode_graph/prereg-v1/owner-plan.json'),
        outputs=dict(journal=str(root/(name+'-journal')),out=str(root/(name+'-owner'))),
        budgets=dict(work_seconds=1830,child_seconds=1800,term_seconds=15,kill_seconds=15))
    path=root/(name+'-plan.json');write(path,plan);return path,plan


class Child:
    pid=99002
    def __init__(self):self.returncode=None;self.finish=0;self.stubborn=False;self.terminated=self.killed=False
    def poll(self):return self.returncode
    def wait(self,timeout):
        if self.stubborn and self.returncode is None:raise subprocess.TimeoutExpired('CPU fake',timeout)
        if self.returncode is None:self.returncode=self.finish
        return self.returncode
    def terminate(self):
        self.terminated=True
        if not self.stubborn:self.returncode=-15
    def kill(self):self.killed=True;self.returncode=-9


class OwnerSystem:
    def __init__(self,test):
        self.test=test;self.child=Child();self.spawned=False;self.unknown=False;self.bad_identity=False
        self.live_child=False;self.cleaned=self.restored=False;self.stop=None;self.command=None
    def identity(self,pid):
        if pid!=os.getpid() and self.bad_identity:raise RuntimeError('CPU identity failure')
        return dict(pid=pid,ppid=os.getpid() if pid!=os.getpid() else 1,start='CPU',comm='python')
    def read(self,command,timeout=5):
        live=self.live_child and command[:3]==['/bin/ps','-p',str(self.child.pid)]
        return dict(command=command,returncode=0 if live else 113 if command[0].endswith('launchctl') else 1,
                    stdout='live' if live else '',stderr='')
    def spawn(self,command,log,env):
        self.command=command;self.spawned=True
        assert (self.test.folder/'intent.json').exists() and (self.test.out/'spawn-intent.json').exists()
        if self.unknown:raise RuntimeError('CPU unknown Popen')
        target=Path(command[-1]);target.mkdir();write(target/'complete.json',{'status':'FAIL','CPU':'ignored numeric schema'})
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
        return dict(label=label,pressure_level_value=self.pressure,authorized_suspended_pids=[],native_inference_processes=rows,
                    power_metadata={'source':'AC Power'},processinfo_thermal_metadata={'code':0})
    def guard(self,samples):pass
    def deltas(self,samples):return dict(samples=len(samples))


class OwnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='cpu-stable-',dir=c.b.ROOT/'.artifacts/m5-ultra-256gb');self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.path,self.plan=plan_at(self.root)
        self.out=Path(self.plan['outputs']['out']);self.folder=self.out/'phases'/'run';self.journal=Path(self.plan['outputs']['journal'])
        self.journal.mkdir();(self.journal/'.lock').touch();self.args=NS(plan=self.path,plan_sha256=c.b.sha(self.path),out=self.out)
        self.prior,_=c.predecessor(self.plan['predecessor'],self.plan['lifecycle'])
        self.manifest=c.lifecycle(self.plan['lifecycle'])
        self.j=dict(schema=c.JOURNAL_SCHEMA,id='CPU-owner',state='STOPPED',removed_at=time.time()-10,
            plan_sha256=c.PLAN_SHA,approved_contract=c.CONTRACT,controller_sha256=self.manifest['sources']['controller']['sha256'],
            original=self.prior['replacement'],owner_directory=str(self.out),owner_plan=ref(self.path),owner_source=self.manifest['sources']['owner'],
            predecessor=self.plan['predecessor'],predecessor_copies={},trials=['run'],lifecycle=self.plan['lifecycle'])
        write(self.journal/'journal.json',self.j)
        self.system=OwnerSystem(self);self.resources=Resources();self.prepare_error=self.postflight_error=None
        self.stack=ExitStack();self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(m,'prepare',side_effect=self.prepare))
    def prepare(self,plan,out,check):
        check()
        if self.prepare_error:raise self.prepare_error
        if out.name=='postflight' and self.postflight_error:raise self.postflight_error
        out.mkdir();q=plan['experiment']['qualifier'];p=plan['experiment']['plan']
        return ['python','-I','-B',q['path'],'--plan',p['path'],'--plan-sha256',p['sha256'],'--out',str(out/'qualification')],{'MLX_CPU':'unchanged'}
    def run_owner(self):return m.run(self.args,system=self.system,resources=self.resources)
    def reap(self):return c.reaped(NS(data=self.j,directory=self.journal),NS(process=lambda pid:None))
    def test_exit0_owns_and_reaps_but_does_not_interpret_numeric_complete(self):
        self.assertEqual(self.run_owner(),0);self.assertFalse(self.reap()['no_child_started'])
        result=json.loads((self.out/'complete.json').read_bytes())
        self.assertEqual(result['planned_trials'],['run']);self.assertTrue(result['independent_numeric_audit_required'])
        self.assertFalse(result['full_model_workload_qualified'] or result['performance_qualified'])
        self.assertTrue(self.system.cleaned and self.system.restored)
    def test_prespawn_failure_is_durable_no_child(self):
        self.prepare_error=RuntimeError('CPU validation failure');self.assertEqual(self.run_owner(),1)
        self.assertFalse(self.system.spawned);self.assertTrue(self.reap()['no_child_started'])
    def test_pressure2_and_foreign_process_block_before_spawn(self):
        self.resources.pressure=2;self.resources.foreign=True
        self.assertEqual(self.run_owner(),1);self.assertFalse(self.system.spawned);self.assertTrue(self.reap()['no_child_started'])
    def test_known_returned_pid_identity_failure_is_reaped(self):
        self.system.bad_identity=True;self.assertEqual(self.run_owner(),1)
        self.assertTrue(self.system.child.terminated);self.assertFalse(self.reap()['no_child_started'])
        self.assertIsNone(json.loads((self.folder/'child.json').read_bytes())['identity'])
    def test_unknown_popen_never_becomes_no_child_or_released(self):
        self.system.unknown=True;self.assertEqual(self.run_owner(),1)
        self.assertFalse((self.out/'terminal.json').exists() or (self.out/'release.json').exists())
        with self.assertRaises(c.NotReady):self.reap()
    def test_cancel_stubborn_child_term_kill_wait_and_postflight(self):
        self.system.child.stubborn=True
        self.resources.callback=lambda label:self.system.stop() if label.endswith(':running') else None
        self.assertEqual(self.run_owner(),1);self.assertTrue(self.system.child.terminated and self.system.child.killed)
        self.assertFalse(self.reap()['no_child_started']);self.assertTrue(self.system.restored)
    def test_child_elapsed_budget_enforced_independently_of_plan_self_report(self):
        clock=[100.0]
        def advance(label):
            if label.endswith(':running'):clock[0]+=1800.1
        self.resources.callback=advance;self.system.child.stubborn=True
        with patch.object(m.time,'monotonic',side_effect=lambda:clock[0]):self.assertEqual(self.run_owner(),1)
        self.assertIn('child elapsed deadline',json.loads((self.out/'first-error.json').read_bytes())['error'])
        self.assertTrue(self.system.child.killed);self.assertFalse(self.reap()['no_child_started'])
    def test_child_nonzero_preserves_first_failure_even_postflight_fails(self):
        self.system.child.finish=1;self.postflight_error=RuntimeError('CPU postflight')
        self.assertEqual(self.run_owner(),1);self.assertFalse(self.reap()['no_child_started'])
        errors=json.loads((self.out/'errors.json').read_bytes())
        self.assertIn('experiment child exit 1',errors[0]['error']);self.assertEqual(errors[-1]['phase'],'postflight')
    def test_surviving_pid_never_certified_reaped(self):
        self.system.live_child=True;self.assertEqual(self.run_owner(),1)
        self.assertFalse((self.out/'release.json').exists())
        with self.assertRaises(c.NotReady):self.reap()
    def test_phase_source_plan_and_old_service_identity_changes_rejected(self):
        for key,value in [('trials',['different']),('original',dict(self.j['original'],pid=123))]:
            bad=copy.deepcopy(self.j);bad[key]=value;write(self.journal/'journal.json',bad)
            with self.subTest(key=key),self.assertRaises(ValueError):self.run_owner()
        write(self.journal/'journal.json',self.j)
        changed=copy.deepcopy(self.plan);changed['experiment']['phase']='different';write(self.path,changed)
        # owner loads its own private controller/support exception class.
        with self.assertRaises(RuntimeError):self.run_owner()
    def test_original_prepare_arguments_and_environment_are_preserved(self):
        self.stack.close();out=self.root/'prepared';calls=[];environment={'MLX_CPU':'exact'};original=lambda *args:None
        def prepare(args,check,legacy):
            check();calls.append((args,legacy));(args.out/'old-preparation').mkdir()
            return ['fixed-python','-I','-B','old-qualifier','--out',str(args.out/'qualification')],environment
        adapter=NS(prepare=prepare,load_legacy=lambda:NS(prepare=original));real=m.load
        with patch.object(m,'load',side_effect=lambda name:adapter if name=='../long_decode_graph/owner.py' else real(name)):
            command,env=m.prepare(self.plan,out,lambda:None)
        self.assertEqual(calls[0][0].plan_sha256,m.OWNER_PLAN_SHA);self.assertIs(calls[0][1],original)
        self.assertEqual(calls[0][0].deadline_seconds,1830);self.assertIs(env,environment)
        self.assertEqual(command[0:4],['fixed-python','-I','-B',self.plan['experiment']['qualifier']['path']])
        self.assertEqual(json.loads((out/'inherited-launch.json').read_bytes())['command'][3],'old-qualifier')
        self.assertEqual((out/'experiment-plan.json').read_bytes(),Path(self.plan['experiment']['plan']['path']).read_bytes())


class PlanTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='cpu-plan-',dir=c.b.ROOT/'.artifacts/m5-ultra-256gb');self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.path,self.plan=plan_at(self.root)
    def test_actual_three_source_manifest_and_exact_migration_without_backend(self):
        with patch.object(c,'System',side_effect=AssertionError('no OS backend')), \
             patch.object(c.b,'System',side_effect=AssertionError('no OS backend')):
            p,manifest=c.run_plan(self.path,c.b.sha(self.path));prior,_=c.predecessor(p['predecessor'],p['lifecycle'])
            self.assertEqual(prior['replacement']['pid'],13908)
            self.assertEqual(set(manifest['sources']),{'controller','owner','audit_restore'})
            actual,_=a.sources(p['lifecycle']);self.assertEqual(actual.JOURNAL_SCHEMA,c.JOURNAL_SCHEMA)
    def test_wrong_phase_budget_shell_and_missing_dependency_rejected(self):
        for change in ('phase','work','child','term','qualifier','dependency','source_collision','outputs'):
            bad=copy.deepcopy(self.plan)
            if change=='phase':bad['experiment']['phase']='../escape'
            elif change=='work':bad['budgets']['work_seconds']=1831
            elif change=='child':bad['budgets']['child_seconds']=True
            elif change=='term':bad['budgets']['term_seconds']=16
            elif change=='qualifier':bad['experiment']['qualifier']['path']='/bin/sh'
            elif change=='dependency':bad['experiment']['source_pins']={}
            elif change=='source_collision':bad['experiment']['source_pins'][str(HERE/'owner.py')]='0'*64
            else:bad['outputs']['out']=bad['outputs']['journal']
            write(self.path,bad)
            with self.subTest(change=change),self.assertRaises((c.Refuse,FileNotFoundError)):c.run_plan(self.path,c.b.sha(self.path))
    def test_manifest_wrong_digest_or_unknown_location_refused(self):
        bad=copy.deepcopy(self.plan['lifecycle']);bad['sha256']='0'*64
        with self.assertRaises(c.Refuse):c.lifecycle(bad)
        other=self.root/'unreviewed.json';other.write_bytes((HERE/'lifecycle.json').read_bytes())
        with self.assertRaises(c.Refuse):c.lifecycle(ref(other))
        with self.assertRaises(ValueError):a.sources(ref(other))
    def test_old_os_resources_stop_recover_and_worker_management_unchanged(self):
        old=ast.parse((HERE.parent/'omlx_cold_restart/controller_v7.py').read_text());new=ast.parse((HERE/'controller.py').read_text())
        for name in ('System','parse_resources','sample','check_resource','resource_gate','atomic_bytes','exclusive_json',
                     'fixed_identity','original_identity','absent_service','replacement','stop','monitored_load','recover','main'):
            left=next(n for n in old.body if getattr(n,'name',None)==name);right=next(n for n in new.body if getattr(n,'name',None)==name)
            self.assertEqual(ast.dump(left),ast.dump(right),name)


class Service:
    """Only state transitions; no process, socket, filesystem-service or signal."""
    def __init__(self,identity,status):
        self.identity=identity;self.status=dict(status);self.live=True;self.clock=time.time();self.calls=[];self.next_pid=identity['pid']+1
    def process(self,pid):return self.identity if self.live and pid==self.identity['pid'] else None
    def job(self):return dict(pid=self.identity['pid'],state='running',path=c.b.PLIST) if self.live else None
    def listener(self):return self.identity['pid'] if self.live else None
    def mutate(self,action):
        self.calls.append(action)
        if action=='stop':self.live=False
        elif action=='bootstrap':
            self.identity=dict(self.identity,pid=self.next_pid,start='CPU cycle '+str(self.next_pid));self.next_pid+=1
            self.live=True;self.status.update(loaded=[],loading=0,target_loading=False)
        else:raise AssertionError(action)
        return 0
    def sleep(self,seconds):self.clock+=seconds
    def resources(self):
        return dict(time=self.clock,system_footprint=(150 if self.live and self.status['loaded'] else 30)*c.GIB,
                    physical_ram=256*c.GIB,pressure=1,swap_bytes=0,raw={'CPU':'fake physical'})
    def log_identity(self):return dict(dev=1,ino=2,size=0,mtime_ns=3)
    def log_window(self,before):return ['CPU log'],dict(path='CPU-only',sha256='a'*64,bytes=7)


class CycleTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='cpu-cycles-',dir=c.b.ROOT/'.artifacts/m5-ultra-256gb');self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.archive=self.root/'archive';self.archive.mkdir()
        self.model=self.root/'model';self.model.mkdir();(self.model/'config.json').write_text('{}')
        self.settings=self.root/'settings.json';self.settings.write_text('{}')
        self.stack=ExitStack();self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(c,'ARCHIVE_ROOT',self.archive))
        self.stack.enter_context(patch.object(c.b,'MODEL',str(self.model)))
        self.stack.enter_context(patch.object(c.b,'GLOBAL_SETTINGS',self.settings))
        self.stack.enter_context(patch.object(c.b,'pins',return_value={}))
        self.stack.enter_context(patch.object(c.os,'getuid',return_value=501))
        self.stack.enter_context(patch.object(c.b,'service_status',side_effect=lambda s,*args:dict(s.status)))
        self.stack.enter_context(patch.object(c.b,'verify_logs',return_value={'CPU':'fake markers'}))
        self.stack.enter_context(patch.object(c,'monitored_load',side_effect=self.fake_post))
        self.lifecycle=ref(HERE/'lifecycle.json');self.manifest=c.lifecycle(self.lifecycle)
        identity=dict(pid=99010,ppid=1,start='CPU initial',comm='omlx-server')
        status=dict(version='0.7.0',default_model=c.b.TARGET,loaded=[c.b.TARGET],loading=0,target_loading=False,
                    path=c.b.MODEL,pinned=False,engine_type='vlm',config_model_type='qwen4_exp')
        self.service=Service(identity,status)
        prior=dict(schema=c.JOURNAL_SCHEMA,state='RESTORED',lifecycle=self.lifecycle,
            controller_sha256=c.b.sha(HERE/'controller.py'),support_sha256=c.BASE_SHA,approved_contract=c.CONTRACT,
            plan_sha256=c.PLAN_SHA,replacement=identity,final_status=status,owner_plan={'sha256':'1'*64},
            log_window=dict(sha256='a'*64,bytes=7),events=[],exact_runtime_signature_restored=False,warm_kv_restored=False,
            global_settings_stat=c.b.file_stat(self.settings),model_metadata={str(self.model/'config.json'):c.b.sha(self.model/'config.json')})
        self.initial=self.archive_receipt(prior,'initial')
    def archive_receipt(self,prior,name):
        folder=self.archive/name;folder.mkdir();path=folder/'restored-journal.json';write(path,prior)
        # Explicit synthetic independent-audit receipt fixture, not a real audit result.
        audit=dict(schema='cold-service-restoration-audit-v1',status='PASS',qualified=True,lifecycle=self.lifecycle,
            audit_source_sha256=self.manifest['sources']['audit_restore']['sha256'],controller_sha256=prior['controller_sha256'],
            journal_sha256=c.b.sha(path),owner_plan_sha256=prior['owner_plan']['sha256'],numerical_qualified=False,performance_qualified=False,
            owner_reap=dict(owner_status='FAIL',CPU_synthetic=True),
            service_restore=dict(status='PERSISTED_VISIBLE_RECEIPTS_CONSISTENT',replacement=prior['replacement'],
                raw_log_sha256=prior['log_window']['sha256'],raw_log_bytes=prior['log_window']['bytes'],http_worker_returncode=0))
        target=folder/'restoration-audit.json';write(target,audit)
        return dict(**ref(path),audit=ref(target))
    def fake_post(self,j,s,identity,baseline):
        folder=j.directory/'load-worker';folder.mkdir()
        j.save('load_intent_monitored',state='LOAD_INTENT',load_baseline_swap_bytes=baseline['swap_bytes'],
               load_worker={'nonce':'CPU','directory':str(folder)},post_worker_pid=99888)
        s.status.update(loaded=[c.b.TARGET],loading=0,target_loading=False)
    def cycle(self,name,predecessor):
        path,plan=plan_at(self.root,name,predecessor);journal=c.Journal(Path(plan['outputs']['journal']),create=True)
        self.addCleanup(journal.close)
        c.prepare(journal,self.service,plan['outputs']['out'],path,c.b.sha(path))
        c.stop(journal,self.service,c.CONTRACT,c.PLAN_SHA)
        self.assertEqual(journal.data['state'],'STOPPED');self.assertFalse(self.service.live)
        c.recover(journal,self.service,c.CONTRACT,c.PLAN_SHA,no_experiment=True)
        self.assertEqual(journal.data['state'],'RESTORED');return journal,plan
    def test_two_cycles_accept_exact_same_version_restored_predecessor(self):
        first,_=self.cycle('first',self.initial);first_pid=first.data['replacement']['pid']
        binding=self.archive_receipt(copy.deepcopy(first.data),'first-closed')
        second,_=self.cycle('second',binding)
        self.assertEqual(second.data['original'],first.data['replacement'])
        self.assertNotEqual(second.data['replacement']['pid'],first_pid)
        self.assertEqual(second.data['predecessor'],binding)
        self.assertEqual(self.service.calls,['stop','bootstrap','stop','bootstrap'])
    def test_same_version_mutated_receipt_source_old_pid_and_plan_rejected(self):
        first,_=self.cycle('first',self.initial);binding=self.archive_receipt(copy.deepcopy(first.data),'closed')
        saved=c.b.read_json(binding['audit']['path'])
        for key,value in [('owner_plan_sha256','bad'),('audit_source_sha256','0'*64),('journal_sha256','0'*64),('lifecycle',dict(self.lifecycle,sha256='0'*64))]:
            bad=dict(saved,**{key:value});write(binding['audit']['path'],bad)
            changed=copy.deepcopy(binding);changed['audit']=ref(binding['audit']['path'])
            with self.subTest(key=key),self.assertRaises(c.Refuse):c.predecessor(changed,self.lifecycle)
        write(binding['audit']['path'],saved)
        current=self.service.identity;self.service.identity=dict(current,start='CPU recycled same PID')
        path,plan=plan_at(self.root,'reused',binding)
        j=c.Journal(plan['outputs']['journal'],create=True);self.addCleanup(j.close)
        with self.assertRaises(c.Refuse):c.prepare(j,self.service,plan['outputs']['out'],path,c.b.sha(path))
    def test_physical_time_units_and_restore_admission_unchanged(self):
        with self.assertRaises(c.Refuse):c.parse_resources({'unit':'KiB','bytes per unit':1024},'1','used = 0M','1',1)
        s=self.service;s.status['loaded']=[]
        with closing(c.Journal(self.root/'resource-journal',create=True)) as j:
            j.data={'events':[]};c.resource_gate(j,s)
            with patch.object(s,'sleep',return_value=None),self.assertRaises(c.Refuse):c.resource_gate(j,s)


if __name__=='__main__':unittest.main()
