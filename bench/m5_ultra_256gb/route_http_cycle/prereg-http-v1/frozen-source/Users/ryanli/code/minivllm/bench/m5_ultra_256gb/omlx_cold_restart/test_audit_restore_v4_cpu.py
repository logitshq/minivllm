"""Simulated files only; tests do not certify any actual model or service run."""
import ast
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

a=module(HERE/'audit_restore_v4.py','restore_audit_cpu')
old=module(HERE.parent/'pareto_realistic/test_audit_absent_cpu.py','restore_raw_fixture_only')


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value)+'\n')


class Tests(unittest.TestCase):
    # Reuse only existing synthetic physical/log fixture construction, not its
    # setUp, numerical mocks, old owner protocol, or any real artifact contents.
    event=old.AuditTests.event
    identity=old.AuditTests.identity
    physical=old.AuditTests.physical

    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name).resolve();self.journal=self.root/'journal';self.owner=self.root/'owner'
        self.journal.mkdir();self.owner.mkdir();self.t=time.time()-1000
        self.owner_pid,self.child_pid,self.replacement_pid,self.worker_pid=20001,20002,30001,30002
        self.c,self.raw=a.sources()
        self.plan={'schema':'E023-native-formal-owner-plan-v1','trials':list(self.c.PHASES[:2]),
                   'http':{'transport':{'port':18023}},'execution':{'pins':{}}}
        write(self.owner/'owner-plan.json',self.plan);plan_sha=a.sha(self.owner/'owner-plan.json')
        source=self.root/'CPU-owner.py';source.write_text('# synthetic owner\n')
        status={'version':'0.7.0','default_model':self.raw.TARGET,'loaded':[self.raw.TARGET],
            'loading':0,'target_loading':False,'path':self.raw.MODEL,'pinned':False,'engine_type':'vlm','config_model_type':'qwen4_exp'}
        self.data=dict(schema='omlx-cold-restart-journal-v4',state='STOPPED',id='CPU-FIXTURE',
            controller_sha256=a.CONTROLLER_SHA,support_sha256=self.c.BASE_SHA,plan_sha256=self.c.PLAN_SHA,
            approved_contract=self.c.CONTRACT,original=dict(pid=74456,ppid=1,start='CPU-previous',comm='omlx-server'),
            original_status=status,global_settings_stat={},model_metadata={},log_start={},predecessor={},predecessor_copies={},
            owner_directory=str(self.owner),owner_source=dict(path=str(source),sha256=a.sha(source)),
            owner_plan=dict(path=str(self.owner/'owner-plan.json'),sha256=plan_sha),removed_at=self.t+3,
            trials=self.plan['trials'],events=[self.event('prepared_v4_read_only',0),self.event('bootout_intent',1),
                    self.event('bootout_returned',2),self.event('service_absent',3)],resource_samples=[])
        write(self.owner/'stopped-journal.json',self.data)
        attempt=dict(schema='cold-service-owner-attempt-v1',owner_pid=self.owner_pid,owner_identity=self.identity(self.owner_pid),
            owner_source=self.data['owner_source'],owner_plan_sha256=plan_sha,journal_id=self.data['id'],
            journal_path=str(self.journal),controller_sha256=a.CONTROLLER_SHA,removed_at=self.data['removed_at'],
            phase='PRESPAWN',created_at=self.t+4,stopped_journal_sha256=a.sha(self.owner/'stopped-journal.json'))
        write(self.owner/'attempt.json',attempt)
        bound={k:attempt[k] for k in ('owner_pid','owner_identity','owner_source','owner_plan_sha256','journal_id','removed_at')}
        bound['attempt_sha256']=a.sha(self.owner/'attempt.json')
        write(self.owner/'spawn-intent.json',dict(bound,time=self.t+9))
        folder=self.owner/'phases/r1-c1'
        identity=dict(pid=self.child_pid,ppid=self.owner_pid,start='CPU-child',comm='python')
        write(folder/'intent.json',dict(bound,phase='r1-c1',time=self.t+10))
        write(folder/'child.json',dict(pid=self.child_pid,owner_pid=self.owner_pid,phase='r1-c1',identity=identity))
        raw=folder/'child-absence-00.json';write(raw,dict(command=['/bin/ps','-p',str(self.child_pid),'-o','pid='],returncode=1,stdout=''))
        write(folder/'port-absence-00.json',dict(command=['/usr/sbin/lsof','-nP','-a','-iTCP:18023','-sTCP:LISTEN','-Fpn'],returncode=1,stdout=''))
        write(folder/'release.json',dict(pid=self.child_pid,returncode=-15,waited_reaped=True,strict_absence=True,
            absence={'path':str(raw.relative_to(self.owner)),'sha256':a.sha(raw)},time=self.t+20))
        row=dict(phase='r1-c1',pid=self.child_pid,identity=identity)
        for kind in ('intent','child','release'):
            path=folder/(kind+'.json');row[kind]=dict(path=str(path.relative_to(self.owner)),sha256=a.sha(path))
        write(self.owner/'children.json',dict(schema='cold-service-owner-children-v1',children=[row]))
        write(self.owner/'release.json',dict(bound,schema='cold-service-owner-release-v2',children=[row],
            children_sha256=a.sha(self.owner/'children.json'),all_children_reaped=True,time=self.t+21))
        write(self.owner/'complete.json',dict(schema='E023-native-formal-owner-result-v1',planned_trials=self.plan['trials'],status='FAIL',owner_pid=self.owner_pid,owner_plan_sha256=plan_sha))
        # The old fixture's recovery generator hashes five legacy filenames;
        # remove its placeholder immediately, before exercising the v3 protocol.
        write(self.owner/'child.json',{'CPU_fixture_placeholder':True})
        old.AuditTests.recovery(self,status);(self.owner/'child.json').unlink()
        files=[self.owner/'attempt.json',self.owner/'stopped-journal.json']
        files+=self.c.phase_reaps(type('J',(),{'data':self.data})(),self.owner,attempt)[0]
        self.data['reap']=dict(no_experiment=False,no_child_started=False,owner_pid=self.owner_pid,
            receipts=[dict(path=str(p),sha256=a.sha(p)) for p in files])
        write(self.journal/'journal.json',self.data)
        self.binding=patch.object(a,'binding',return_value=(self.plan,{}));self.binding.start();self.addCleanup(self.binding.stop)

    def run_audit(self):return a.audit(self.journal)

    def test_restored_failed_experiment_gets_explicit_journal_binding(self):
        result=self.run_audit();self.assertEqual(result['status'],'PASS',result)
        self.assertEqual(result['journal_sha256'],a.sha(self.journal/'journal.json'))
        self.assertEqual(result['owner_reap']['owner_status'],'FAIL')
        self.assertEqual(result['service_restore']['physical_samples'],8)
        self.assertFalse(result['numerical_qualified'] or result['performance_qualified'] or result['ledger_opened'])

    def test_not_restored_does_not_load_support_or_owner(self):
        write(self.journal/'journal.json',{'state':'LOAD_INTENT'})
        with patch.object(a,'sources',side_effect=AssertionError('no owner inspection')):
            self.assertEqual(self.run_audit()['status'],'INCOMPLETE')

    def test_raw_physical_tamper_and_pressure_and_missing_file_rejected(self):
        path=Path(self.data['resource_samples'][0]['raw']['path']);original=path.read_bytes();path.write_text('{}')
        self.assertEqual(self.run_audit()['status'],'FAIL');path.write_bytes(original);path.unlink()
        self.assertEqual(self.run_audit()['status'],'INCOMPLETE')

    def test_raw_log_cannot_fake_resident_completion(self):
        path=Path(self.data['log_window']['path']);path.write_bytes(path.read_bytes().replace(b': resident',b': mmap'))
        self.assertEqual(self.run_audit()['status'],'FAIL')

    def test_source_raw_unit_and_physical_gate_unchanged(self):
        row=self.data['resource_samples'][0];self.raw.physical(row,self.journal)
        bad=copy.deepcopy(row);bad['pressure']=2
        with self.assertRaises(ValueError):self.raw.physical(bad,self.journal)
        bad=copy.deepcopy(row);bad['system_footprint']=41*(1<<30)
        with self.assertRaises(ValueError):self.raw.physical_gate([bad,dict(bad,label='cold-stability-2',time=bad['time']+5)])

    def test_raw_child_absence_and_port_reject_live_or_wrong_pid(self):
        path=self.owner/'phases/r1-c1/port-absence-00.json';raw=json.loads(path.read_bytes());raw['returncode']=0;raw['stdout']='p20002\n'
        write(path,raw);self.assertEqual(self.run_audit()['status'],'FAIL')

    def test_reap_binding_and_missing_phase_release_rejected(self):
        self.data['reap']['receipts'][-1]['sha256']='0'*64;write(self.journal/'journal.json',self.data)
        self.assertEqual(self.run_audit()['status'],'FAIL')
        (self.owner/'release.json').unlink();self.assertEqual(self.run_audit()['status'],'INCOMPLETE')

    def test_worker_missing_or_unreaped_is_not_restored(self):
        path=self.journal/'load-worker/release.json';raw=path.read_bytes();d=json.loads(raw);d['waited_reaped']=False;write(path,d)
        self.assertEqual(self.run_audit()['status'],'FAIL');path.unlink()
        self.assertEqual(self.run_audit()['status'],'INCOMPLETE')

    def test_wrong_controller_pin_is_rejected_before_any_backend(self):
        self.binding.stop()
        self.data['controller_sha256']='0'*64;write(self.journal/'journal.json',self.data)
        self.assertEqual(self.run_audit()['status'],'FAIL')

    def test_exact_formal_owner_source_and_dependency_map(self):
        # Exercise real small Python source pins with a synthetic predecessor
        # and numerical-plan digest; no actual journal or runtime is audited.
        self.binding.stop();original_sha=a.sha
        q=self.root/'synthetic-numeric.json';write(q,{})
        prior=dict(replacement=self.data['original'],final_status=self.data['original_status'],
                   global_settings_stat={},model_metadata={})
        prior_path=self.journal/'predecessor.json';audit_path=self.journal/'predecessor-audit.json'
        write(prior_path,prior);write(audit_path,{})
        self.data['predecessor']=dict(path=str(prior_path),sha256=original_sha(prior_path),
            audit=dict(path=str(audit_path),sha256=original_sha(audit_path)))
        self.data['predecessor_copies']={p.name:original_sha(p) for p in (prior_path,audit_path)}
        with patch.object(self.c,'predecessor',return_value=(prior,{})), patch.object(a,'sha',
            side_effect=lambda p:a.NUMERIC_PLAN_SHA if Path(p)==q else original_sha(p)):
            for name,digest in a.OWNERS.items():
                source=HERE.parent/'pareto_realistic'/name
                fixed=next(ast.literal_eval(n.value) for n in ast.parse(source.read_text()).body
                    if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='FIXED' for t in n.targets))
                pins={str(source.parent/n):h for n,h in fixed.items()};pins[str(source)]=digest
                ref=dict(path=str(source),sha256=digest)
                plan=dict(schema='E023-native-formal-owner-plan-v1',trials=list(self.c.PHASES[:2]),deadline_seconds=1038,qualifications={'1':{},'8':{}},
                    execution=dict(owner=ref,pins=pins),qualification=dict(path=str(q),sha256=a.NUMERIC_PLAN_SHA),
                    http={'deadlines':{'overall':312,'close':15,'term':15,'kill':15,'resource_interval':5}},
                    cold_restart=dict(journal_path=str(self.journal),controller=dict(path=str(HERE/'controller_v4.py'),sha256=a.CONTROLLER_SHA),
                        approved_contract=self.c.CONTRACT,proposal_plan_sha256=self.c.PLAN_SHA,predecessor=self.data['predecessor']))
                write(self.owner/'owner-plan.json',plan)
                self.data.update(owner_source=ref,owner_plan=dict(path=str(self.owner/'owner-plan.json'),sha256=original_sha(self.owner/'owner-plan.json')))
                self.assertEqual(a.binding(self.data,self.journal,self.c,self.raw),(plan,pins))
                plan['execution']['owner']['sha256']='0'*64;write(self.owner/'owner-plan.json',plan)
                self.data['owner_plan']['sha256']=original_sha(self.owner/'owner-plan.json')
                with self.assertRaisesRegex(ValueError,'owner source identity'):
                    a.binding(self.data,self.journal,self.c,self.raw)

    def test_no_child_terminal_proof_is_separate_from_experiment_success(self):
        attempt=json.loads((self.owner/'attempt.json').read_bytes())
        # Unit-test the no-child branch on its own fresh directory.
        other=self.root/'no-child';other.mkdir();write(other/'attempt.json',attempt)
        shared={k:attempt[k] for k in ('owner_pid','owner_identity','owner_source','owner_plan_sha256','journal_id','removed_at')}
        write(other/'terminal.json',dict(shared,schema='cold-service-owner-terminal-v1',status='NO_CHILD_STARTED',
            attempt_sha256=a.sha(other/'attempt.json'),spawn_intent_written=False,child_started=False,terminal_at=self.t+22))
        files=[];self.assertEqual(a.no_child(self.data,other,attempt,files,self.c),self.t+22)
        write(other/'spawn-intent.json',{});
        with self.assertRaises(ValueError):a.no_child(self.data,other,attempt,[],self.c)

    def proof(self,owner,phase,at):
        attempt=json.loads((owner/'attempt.json').read_bytes())
        inner=owner/'phases'/phase/'trial/spawn-intent.json'
        write(inner,dict(command=['python','-I','-B','CPU-fixture'],time=at))
        path=inner.parent.parent/'no-popen.json'
        write(path,dict(schema='cold-service-no-popen-v1',phase=phase,owner_pid=attempt['owner_pid'],
            journal_id=attempt['journal_id'],owner_plan_sha256=attempt['owner_plan_sha256'],
            attempt_sha256=a.sha(owner/'attempt.json'),ownership='tracked-backend-rejected-before-Popen',
            popen_called=False,time=at+0.1,command=['python','-I','-B','CPU-fixture'],
            inner_spawn_intent=dict(path=str(inner.relative_to(owner)),sha256=a.sha(inner))))
        return dict(path=str(path.relative_to(owner)),sha256=a.sha(path))

    def test_no_child_positive_first_trial_proof_hash_and_unknown_intent(self):
        other=self.root/'no-child-proof';other.mkdir()
        attempt=json.loads((self.owner/'attempt.json').read_bytes());write(other/'attempt.json',attempt)
        shared={k:attempt[k] for k in ('owner_pid','owner_identity','owner_source','owner_plan_sha256','journal_id','removed_at')}
        proof=self.proof(other,'r1-c1',self.t+10)
        terminal=dict(shared,schema='cold-service-owner-terminal-v1',status='NO_CHILD_STARTED',
            attempt_sha256=a.sha(other/'attempt.json'),spawn_intent_written=False,child_started=False,
            terminal_at=self.t+22,no_popen_receipt=proof)
        write(other/'terminal.json',terminal)
        files=[];self.assertEqual(a.no_child(self.data,other,attempt,files,self.c),self.t+22)
        self.assertEqual([p.name for p in files],['terminal.json','spawn-intent.json','no-popen.json'])
        bad=copy.deepcopy(terminal);bad['no_popen_receipt']['sha256']='0'*64;write(other/'terminal.json',bad)
        with self.assertRaises(Exception):a.no_child(self.data,other,attempt,[],self.c)
        write(other/'terminal.json',terminal);write(other/'phases/r1-c8/trial/child.json',{'pid':123})
        with self.assertRaisesRegex(ValueError,'unresolved'):a.no_child(self.data,other,attempt,[],self.c)

    def test_later_positive_no_popen_receipts_retained_and_tampering_rejected(self):
        proof=self.proof(self.owner,'r1-c8',self.t+20.2)
        final=json.loads((self.owner/'release.json').read_bytes());final['no_popen_receipt']=proof
        write(self.owner/'release.json',final)
        attempt=json.loads((self.owner/'attempt.json').read_bytes())
        files=[self.owner/'attempt.json',self.owner/'stopped-journal.json']
        files+=self.c.phase_reaps(type('J',(),{'data':self.data})(),self.owner,attempt)[0]
        self.data['reap']['receipts']=[dict(path=str(p),sha256=a.sha(p)) for p in files]
        write(self.journal/'journal.json',self.data)
        result=self.run_audit();self.assertEqual(result['status'],'PASS',result)
        self.assertEqual(result['owner_reap']['children'],[dict(phase='r1-c1',pid=self.child_pid)])
        raw=json.loads((self.owner/proof['path']).read_bytes());raw['popen_called']=True
        write(self.owner/proof['path'],raw)
        final['no_popen_receipt']['sha256']=a.sha(self.owner/proof['path']);write(self.owner/'release.json',final)
        self.assertEqual(self.run_audit()['status'],'FAIL')

    def test_no_original_raw_validator_reimplementation_or_backend_calls(self):
        old=ast.parse((HERE/'audit_restore_v3.py').read_text())
        new=ast.parse((HERE/'audit_restore_v4.py').read_text())
        for name in ('need','sha','read','finite','load','main'):
            before=next(n for n in old.body if getattr(n,'name',None)==name)
            after=next(n for n in new.body if getattr(n,'name',None)==name)
            self.assertEqual(ast.dump(before),ast.dump(after),name)
        with patch.object(self.c.b,'System',side_effect=AssertionError('no backend')):
            self.assertEqual(self.run_audit()['status'],'PASS')

    def test_output_must_be_new_and_incomplete_receipt_is_preserved(self):
        out=self.root/'audit.json';write(self.journal/'journal.json',{'state':'PREPARED'})
        self.assertEqual(a.main(['--journal',str(self.journal),'--out',str(out)]),2)
        before=out.read_bytes()
        with self.assertRaises(ValueError):a.main(['--journal',str(self.journal),'--out',str(out)])
        self.assertEqual(out.read_bytes(),before)


if __name__=='__main__':unittest.main()
