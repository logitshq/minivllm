"""Focused fake cold-owner tests. No GPU, process, service or model execution."""
import ast
import asyncio
from contextlib import ExitStack, redirect_stdout
import copy
import io
import json
import os
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
from bench.m5_ultra_256gb.c1_profile_cycle_v1 import owner as m, controller as c, preregister as reg, execute as driver, audit_closed as audit
from bench.m5_ultra_256gb.gdn_graph_combined_formal.test_owner_cpu import System, Resources


def write(path,value):path.write_text(json.dumps(value)+'\n')
def result(mode):
    return dict(schema='C1-pipeline-diagnostic-capture-result-v1',mode=mode,status='PASS',errors=[],count_verified=True,natural_exit=True,server_reaped=True,
        cleanup_complete=True,cleanup_task_resolved=True,client_task_resolved=True,drain_client_resolved=True,small_pins_rechecked=True,forced_stop=False,
        numerical_qualified=False,performance_qualified=False,closed_server=dict(status='RECORDED',requests=16,numerical_qualified=False,performance_qualified=False))


class Capture:
    def __init__(self,fixture):self.f=fixture
    async def run(self,spec,out,*,checkpoint,backend):
        out.mkdir();await checkpoint('pre-spawn',None)
        write(out/'spawn-intent.json',dict(command=['CPU','server'],time=time.time()))
        child=backend.spawn([],{},None,None);write(out/'child.json',dict(pid=child.pid));await checkpoint('spawned',child.pid)
        if self.f.cancel:
            child.stubborn=True;raise asyncio.CancelledError('fake live cancellation')
        child.returncode=self.f.rc;child.wait(15)
        write(out/'release.json',dict(pid=child.pid,returncode=child.returncode,waited_reaped=True,strict_absence=True,port_absent=True,natural_exit=self.f.rc==0,forced_stop=False,time=time.time()))
        await checkpoint('after-reap',None);value=result(spec['mode'])
        if self.f.bad_drain:value['closed_server']['requests']=15
        write(out/'complete.json',value);return value


class OwnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name).resolve();self.out=self.root/'owner'
        self.journal=self.root/'journal';self.journal.mkdir();(self.journal/'.lock').touch();self.base=m.load('qualify_absent_owner')
        self.plan_path=self.root/'plan.json';self.plan=dict(schema=m.PLAN_SCHEMA,trials=list(m.PHASES),deadline_seconds=10356,
            execution=dict(owner=dict(path=m.__file__,sha256=m.sha(m.__file__)),pins={}),qualification={},
            cold_restart=dict(journal_path=str(self.journal),controller=dict(path=c.__file__,sha256=m.sha(c.__file__)),approved_contract=self.base.CONTRACT,
                proposal_plan_sha256=self.base.PROPOSAL_SHA,predecessor={'CPU':'fake'}),http=dict(transport={'port':18765}))
        write(self.plan_path,self.plan);self.args=NS(plan=self.plan_path,plan_sha256=m.sha(self.plan_path),out=self.out)
        self.j=dict(schema=m.JOURNAL_SCHEMA,state='STOPPED',plan_sha256=self.base.PROPOSAL_SHA,controller_sha256=m.sha(c.__file__),approved_contract=self.base.CONTRACT,
            owner_directory=str(self.out),owner_plan=dict(path=str(self.plan_path),sha256=self.args.plan_sha256),owner_source=self.plan['execution']['owner'],
            predecessor={'CPU':'fake'},predecessor_copies={},trials=list(m.PHASES),original=dict(pid=74456,ppid=1,start='CPU',comm='omlx-server'),id='CPU-journal',removed_at=time.time()-10)
        write(self.journal/'journal.json',self.j);self.probe_in_progress=threading.Event();self.resources=Resources();self.system=System(self)
        self.rc=0;self.cancel=self.bad_drain=False;self.specs=[dict(mode=p,phase=p,concurrency=1) for p in m.PHASES]
        self.stack=ExitStack();self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(m,'validate_plan',return_value=(self.base,self.plan,self.plan_path.read_bytes(),{},self.specs)))
        self.stack.enter_context(patch.object(m,'prior_replacement',return_value=self.j['original']))
        self.stack.enter_context(patch.object(m,'full_prepare',side_effect=lambda base,plan,num,out,check:(check(),out.mkdir())))
        self.stack.enter_context(patch.object(m.subprocess,'Popen',side_effect=AssertionError('real Popen forbidden')))
        self.stack.enter_context(patch.object(m.subprocess,'run',side_effect=AssertionError('real command forbidden')))
        self.stack.enter_context(patch.object(m.signal,'signal',return_value=None))
    def run_owner(self):return m.run(self.args,system=self.system,resources=self.resources,formal_module=Capture(self))
    def read(self,name):return json.loads((self.out/name).read_bytes())
    def test_three_actual_modes_natural_exit_no_overlap_or_points(self):
        self.assertEqual(self.run_owner(),0);self.assertEqual(self.system.events,[('spawn',p) for p in m.PHASES])
        self.assertEqual([r['phase'] for r in self.read('diagnostics.json')['trials']],list(m.PHASES))
        self.assertTrue(self.read('complete.json')['diagnostic_modes_complete']);self.assertFalse((self.out/'points.json').exists())
        with patch.object(c,'owner_plan',return_value=self.plan):
            self.assertFalse(c.reaped(NS(data=self.j,directory=self.journal),NS(process=lambda pid:None))['no_child_started'])
    def test_unknown_spawn_never_gets_release_or_nochild(self):
        self.system.unknown=True;self.assertEqual(self.run_owner(),1)
        self.assertFalse((self.out/'release.json').exists());self.assertFalse((self.out/'terminal.json').exists());self.assertIsNone(self.read('complete.json')['child_started'])
    def test_second_unknown_spawn_cannot_certify_first_prefix(self):
        self.system.unknown_at=1;self.assertEqual(self.run_owner(),1);self.assertEqual(len(self.system.children),1)
        self.assertFalse((self.out/'release.json').exists());self.assertFalse((self.out/'terminal.json').exists())
    def test_cancelled_live_child_uses_outer_term_kill_reap(self):
        self.cancel=True;self.assertEqual(self.run_owner(),1);self.assertTrue(self.system.children[0].killed)
        self.assertTrue(self.read('release.json')['all_children_reaped']);self.assertEqual(len(self.system.children),1)
    def test_incomplete_drain_and_nonnatural_exit_forbid_next_mode(self):
        self.bad_drain=True;self.assertEqual(self.run_owner(),1);self.assertEqual(len(self.system.children),1)
        self.assertFalse(self.read('complete.json')['diagnostic_modes_complete'])
    def test_nonzero_natural_process_exit_is_failure_even_after_reap(self):
        self.rc=3;self.assertEqual(self.run_owner(),1);self.assertEqual(len(self.system.children),1);self.assertTrue(self.read('release.json')['all_children_reaped'])
    def test_prepare_pressure_failure_has_positive_nochild_proof(self):
        self.resources.pressure=2;self.assertEqual(self.run_owner(),1);self.assertEqual(self.read('terminal.json')['status'],'NO_CHILD_STARTED')
        self.assertFalse(self.system.children)
    def test_known_child_identity_failure_reaped_without_forging_identity(self):
        self.system.bad_identity=True;self.assertEqual(self.run_owner(),1);self.assertIsNone(self.read('children.json')['children'][0]['identity'])
        self.assertTrue(self.read('release.json')['all_children_reaped'])


class ContractTests(unittest.TestCase):
    def test_default_entries_are_inert_and_unsealed_refused(self):
        for module in (m,c,reg,driver,audit):
            with self.subTest(module=module.__name__),redirect_stdout(io.StringIO()):self.assertEqual(module.main([]),0)
        with patch.object(c,'BOUND_MANIFEST_SHA','UNSEALED'),self.assertRaisesRegex(Exception,'UNSEALED'):c.predecessor_manifest()
    def test_fixed_three_modes_and_unchanged_direct_owner_subfunctions(self):
        self.assertEqual(m.work_seconds(list(m.PHASES),m.DEADLINES),10356)
        for value in (['off','runner'],['off','full','runner'],['off','runner','full','full']):
            with self.assertRaises(RuntimeError):m.trial_names(value)
        original=ast.parse(m.reuse.inspect.getsource(m._original.run)).body[0];actual=ast.parse(m.ADAPTED_RUN).body[0]
        for name in ('spawn_impl','spawn','probe','release','trial_one'):
            a=next(n for n in original.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==name)
            b=next(n for n in actual.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==name)
            self.assertEqual(ast.dump(a,include_attributes=False),ast.dump(b,include_attributes=False),name)
        self.assertIs(c.stop.__code__,c._donor.stop.__code__);self.assertIsNot(c.stop.__globals__,c._donor.stop.__globals__)
    def test_natural_draining_absence_is_raw_pid_proof_only(self):
        exact=dict(status='unknown',pid=123,raw=dict(command=['/bin/ps','-p','123','-o','pid=,ppid=,rss=,comm='],returncode=1,stdout='',stderr=''))
        audit.drained_absence(exact,123,'draining')
        for sub in ('diagnostic','warm','after-reap'):
            with self.assertRaises(Exception):audit.drained_absence(exact,123,sub)
        exact['raw']['returncode']=None
        with self.assertRaises(Exception):audit.drained_absence(exact,123,'draining')
    def test_driver_unknown_owner_launch_still_recovers_without_noexperiment(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);journal=root/'journal';journal.mkdir();write(journal/'journal.json',{'state':'PREPARED'})
            context=dict(execution=root/'execution',owner=root/'owner',journal=journal,commands=dict(controller_stop=['stop'],owner=['owner'],controller_recover=['recover']),prepared_raw=b'{}',receipt_raw=b'{}',owner_plan_sha256='CPU')
            calls=[]
            def run(folder,name,command):
                calls.append((name,command))
                if name=='stop':write(journal/'journal.json',{'state':'STOPPED'});return 0
                if name=='diagnostic-owner':raise OSError('unknown owner Popen')
                write(journal/'journal.json',{'state':'RESTORED'});return 0
            with self.assertRaisesRegex(OSError,'unknown'):driver.execute(context,run=run)
            self.assertEqual(calls[-1],('recover',['recover']))
    def test_prepared_commands_use_real_owner_cli_and_two_offline_audits(self):
        path=Path('/CPU/registration/owner-plan.json');plan=dict(cold_restart=dict(controller={'path':str(c.HERE/'controller.py')},journal_path='/CPU/journal',proposal_plan_sha256='p',approved_contract='a'),execution=dict(owner={'path':str(m.HERE/'owner.py')}),outputs=dict(owner='/CPU/owner'))
        commands=reg.build_commands(path,'h',plan,Path('/CPU/analysis.json'),'/CPU/python')
        self.assertIn('--run',commands['owner']);self.assertIn('--prepare-only',commands['owner_prepare'])
        self.assertEqual(commands['owner_prepare'][3],str(m.HERE/'owner.py'))
        self.assertEqual(commands['closed_audit'][3],str(audit.HERE/'audit_closed.py'))
        self.assertEqual(commands['restoration_audit'][3],str(audit.HERE/'audit_restore.py'))


if __name__=='__main__':unittest.main()
