"""Prepared CPU-fake protocol tests; never service/device evidence. Not run yet."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent

def module(name):
    s=importlib.util.spec_from_file_location('glm_test_'+name,HERE/(name+'.py'));m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
c=module('controller')
a=module('audit_restore')
def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value)+'\n')
def reference(path):return dict(path=str(path),sha256=c.b.sha(path))


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve();self.owner=self.root/'owner';self.owner.mkdir()
        self.journal=self.root/'journal';self.journal.mkdir();self.t=time.time()-100
        source=self.root/'owner-source.py';source.write_text('# synthetic source identity\n')
        self.data=dict(schema=c.JOURNAL_SCHEMA,state='STOPPED',id='CPU',removed_at=self.t,
            controller_sha256=c.b.sha(HERE/'controller.py'),owner_directory=str(self.owner),
            owner_plan=dict(path=str(self.root/'owner-plan.json'),sha256='a'*64),owner_source=reference(source),
            original=dict(pid=20001,ppid=1,start='old',comm='omlx-server'),predecessor={},predecessor_copies={},trials=[c.PHASE])
        write(self.owner/'stopped-journal.json',self.data)
        self.attempt=dict(schema='cold-service-owner-attempt-v1',phase='PRESPAWN',owner_pid=20002,
            owner_identity=dict(pid=20002,ppid=2,start='owner',comm='python'),owner_source=self.data['owner_source'],
            owner_plan_sha256='a'*64,journal_id='CPU',journal_path=str(self.journal),controller_sha256=self.data['controller_sha256'],
            removed_at=self.t,created_at=self.t+1,stopped_journal_sha256=c.b.sha(self.owner/'stopped-journal.json'))
        write(self.owner/'attempt.json',self.attempt);self.j=SimpleNamespace(data=self.data,directory=self.journal)
        self.registration=patch.object(c,'registered_trials',return_value=[c.PHASE]);self.registration.start();self.addCleanup(self.registration.stop)
        self.system=SimpleNamespace(process=lambda pid:None)
        self.bound={k:self.attempt[k] for k in ('owner_pid','owner_identity','owner_source','owner_plan_sha256','journal_id','removed_at')}
        self.bound['attempt_sha256']=c.b.sha(self.owner/'attempt.json')

    def nochild(self):
        (self.owner/'phases'/c.PHASE).mkdir(parents=True)
        write(self.owner/'children.json',dict(schema='cold-service-owner-children-v1',children=[]))
        write(self.owner/'terminal.json',dict(self.bound,schema='cold-service-owner-terminal-v1',status='NO_CHILD_STARTED',
            terminal_at=self.t+2,spawn_intent_written=False,child_started=False))

    def child(self,identity=True):
        folder=self.owner/'phases'/c.PHASE
        pi=dict(self.bound,phase=c.PHASE,time=self.t+2,command=['original','--run'])
        write(self.owner/'spawn-intent.json',pi);write(folder/'intent.json',pi)
        ci=dict(pid=20003,ppid=20002,start='child',comm='python') if identity else None
        write(folder/'child.json',dict(pid=20003,owner_pid=20002,phase=c.PHASE,identity=ci,ownership='direct-Popen-returned'))
        raw=dict(command=['/bin/ps','-p','20003','-o','pid='],returncode=1,stdout='',stderr='')
        write(folder/'child-absence.json',raw)
        reap=dict(schema='GLM-owned-reap-v1',status='REAPED',ownership='direct-Popen-returned',pid=20003,owner_pid=20002,
            waited_reaped=True,strict_absence=True,returncode=-15,absence=raw,time=self.t+3,actions=['TERM'],errors=[],
            performance_qualified=False,physical_resource_safety_qualified=False)
        write(folder/'reap.json',reap)
        def rel(name):return dict(path=str((folder/name).relative_to(self.owner)),sha256=c.b.sha(folder/name))
        write(folder/'release.json',dict(pid=20003,phase=c.PHASE,returncode=-15,waited_reaped=True,strict_absence=True,
            time=self.t+4,absence=rel('child-absence.json'),reap=rel('reap.json')))
        row=dict(phase=c.PHASE,pid=20003,identity=ci,**{k:rel(k+'.json') for k in ('intent','child','release')})
        write(self.owner/'children.json',dict(schema='cold-service-owner-children-v1',children=[row]))
        write(self.owner/'release.json',dict(self.bound,schema='cold-service-owner-release-v2',children=[row],
            children_sha256=c.b.sha(self.owner/'children.json'),all_children_reaped=True,time=self.t+5))
        return folder

    def test_preintent_empty_index_and_directory_without_plan_copy(self):
        self.nochild();result=c.reaped(self.j,self.system)
        self.assertTrue(result['no_child_started']);self.assertFalse((self.owner/'owner-plan.json').exists())
        self.assertIn(str(self.owner/'children.json'),[r['path'] for r in result['receipts']])

    def test_unknown_popen_intent_never_becomes_nochild(self):
        self.nochild();write(self.owner/'spawn-intent.json',{})
        with self.assertRaises(Exception):c.reaped(self.j,self.system)
        (self.owner/'terminal.json').unlink()
        with self.assertRaises(Exception):c.reaped(self.j,self.system)

    def test_known_returned_pid_identity_failure_still_reaped(self):
        self.child(identity=False);result=c.reaped(self.j,self.system);self.assertFalse(result['no_child_started'])
        with self.assertRaises(c.NotReady):c.reaped(self.j,SimpleNamespace(process=lambda pid:dict(pid=pid)))

    def test_missing_release_and_extra_nested_intent_rejected(self):
        folder=self.child();write(folder/'unexpected/spawn-intent.json',{})
        with self.assertRaises(Exception):c.reaped(self.j,self.system)
        (folder/'unexpected/spawn-intent.json').unlink();(folder/'release.json').unlink()
        with self.assertRaises(Exception):c.reaped(self.j,self.system)

    def test_actual_four_field_absence_reap_no_formal_port(self):
        folder=self.child();release=json.loads((folder/'release.json').read_bytes())
        _,raw=a.sources();self.assertEqual(a.child_absence(self.owner,release,20003,20002,c.PHASE,raw)['status'],'REAPED')
        self.assertEqual(list(folder.glob('*port*')),[])
        bad=json.loads((folder/'child-absence.json').read_bytes());bad.pop('stderr');write(folder/'child-absence.json',bad)
        release['absence']['sha256']=c.b.sha(folder/'child-absence.json')
        with self.assertRaises(Exception):a.child_absence(self.owner,release,20003,20002,c.PHASE,raw)

    def test_owner_validator_gets_real_out_and_journal_binding(self):
        self.registration.stop();plan=dict(schema=c.OWNER_SCHEMA,outputs={'owner':str(self.owner)},
            cold_restart=dict(journal_path=str(self.journal),controller=reference(HERE/'controller.py')))
        path=self.root/'real-plan.json';write(path,plan);seen=[]
        def validate(args):seen.append(args);return plan,None,None,None
        with patch.object(c,'owner_module',return_value=SimpleNamespace(validate_plan=validate)),patch.object(c,'source_pins',return_value={}):
            c.owner_plan(path,c.b.sha(path),self.journal);self.assertEqual(seen[0].out,self.owner)
            with self.assertRaises(Exception):c.owner_plan(path,c.b.sha(path),self.root/'wrong-journal')
        self.registration.start()

    def test_source_pin_union_does_not_mutate_exact_execution_pins(self):
        pins={str(self.root/'synthetic'):'a'*64}
        plan={'execution':dict(owner=dict(path=str(HERE/'owner.py'),sha256=c.OWNER_SHA),pins=dict(pins))}
        fake=SimpleNamespace(source_pins=lambda p:dict(p['execution']['pins']))
        with patch.object(c,'owner_module',return_value=fake):out=c.source_pins(plan)
        self.assertEqual(plan['execution']['pins'],pins);self.assertGreater(len(out),len(pins))

    def test_explicit_future_predecessor_accepts_restoration_not_experiment_pass(self):
        archive=self.root/'archives';folder=archive/'future-closed';folder.mkdir(parents=True)
        replacement=dict(pid=20004,ppid=1,start='future actual',comm='omlx-server')
        prior=dict(schema='E024-route-http-formal-cold-journal-v1',controller_sha256=c.PRIOR_SHA,trials=list(c.PRIOR_PHASES),
            state='RESTORED',approved_contract=c.CONTRACT,plan_sha256=c.PLAN_SHA,support_sha256=c.BASE_SHA,events=[],replacement=replacement,
            owner_plan={'sha256':'b'*64},log_window=dict(sha256='c'*64,bytes=5),exact_runtime_signature_restored=False,warm_kv_restored=False,
            final_status=dict(version='0.7.0',default_model=c.b.TARGET,loaded=[c.b.TARGET],loading=0,target_loading=False,
                              path=c.b.MODEL,pinned=False,engine_type='vlm',config_model_type='qwen4_exp'))
        jp=folder/'restored-journal.json';ap=folder/'restoration-audit.json';write(jp,prior)
        proof=dict(schema='cold-service-restoration-audit-v1',status='PASS',qualified=True,audit_source_sha256=c.PRIOR_AUDIT_SHA,
            journal_sha256=c.b.sha(jp),controller_sha256=c.PRIOR_SHA,owner_plan_sha256='b'*64,
            service_restore=dict(status='PERSISTED_VISIBLE_RECEIPTS_CONSISTENT',replacement=replacement,raw_log_sha256='c'*64,raw_log_bytes=5,http_worker_returncode=0))
        write(ap,proof);binding=dict(reference(jp),audit=reference(ap))
        with patch.object(c,'ARCHIVE_ROOT',archive):
            self.assertEqual(c.predecessor(binding)[0]['replacement'],replacement)
            for bad in (dict(proof,status='FAIL'),dict(proof,audit_source_sha256='0'*64)):
                write(ap,bad);binding['audit']=reference(ap)
                with self.assertRaises(Exception):c.predecessor(binding)
            write(ap,proof);binding['audit']=reference(ap);write(jp,dict(prior,state='LOAD_INTENT'));binding.update(reference(jp))
            with self.assertRaises(Exception):c.predecessor(binding)

    def test_recovery_dependency_dictionary_does_not_mutate_donor(self):
        self.assertIsNot(c.stop.__globals__,c._donor.stop.__globals__)
        self.assertIs(c.stop.__globals__['authorize'],c.authorize)
        self.assertIs(c.recover.__globals__['monitored_load'],c.monitored_load)
        self.assertIsNot(c._donor.recover.__globals__['monitored_load'],c.monitored_load)
        self.assertEqual(Path(c.monitored_load.__globals__['__file__']).resolve(),HERE/'controller.py')


if __name__=='__main__':unittest.main()
