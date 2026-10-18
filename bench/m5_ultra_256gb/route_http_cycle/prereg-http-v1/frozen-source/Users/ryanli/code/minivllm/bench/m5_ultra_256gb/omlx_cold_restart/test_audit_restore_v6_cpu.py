"""Synthetic CPU receipts only; no process/service/model/ledger operation."""
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

a=module(HERE/'audit_restore_v6.py','e024_isolation_restore_audit_cpu')
old=module(HERE.parent/'pareto_realistic/test_audit_absent_cpu.py','e024_restore_raw_fixture')


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value)+'\n')


class Tests(unittest.TestCase):
    event=old.AuditTests.event
    identity=old.AuditTests.identity
    physical=old.AuditTests.physical

    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name).resolve();self.journal=self.root/'journal';self.owner=self.root/'owner'
        self.journal.mkdir();self.owner.mkdir();self.t=time.time()-1000
        self.owner_pid,self.child_pid,self.replacement_pid,self.worker_pid=20001,20002,30001,30002
        self.c,self.raw=a.sources()
        self.phase='e024-isolation';self.folder=self.owner/'phases'/self.phase
        self.plan=dict(schema='E024-isolation-cold-owner-plan-v1',trials=[self.phase],execution={'pins':{}})
        write(self.owner/'owner-plan.json',self.plan);plan_sha=a.sha(self.owner/'owner-plan.json')
        source=self.root/'CPU-owner.py';source.write_text('# synthetic owner\n')
        status=dict(version='0.7.0',default_model=self.raw.TARGET,loaded=[self.raw.TARGET],loading=0,
            target_loading=False,path=self.raw.MODEL,pinned=False,engine_type='vlm',config_model_type='qwen4_exp')
        self.data=dict(schema='omlx-cold-restart-journal-v6',state='STOPPED',id='CPU-E024',
            controller_sha256=a.CONTROLLER_SHA,support_sha256=self.c.BASE_SHA,plan_sha256=self.c.PLAN_SHA,
            approved_contract=self.c.CONTRACT,original=dict(pid=74456,ppid=1,start='CPU-previous',comm='omlx-server'),
            original_status=status,global_settings_stat={},model_metadata={},log_start={},predecessor={},predecessor_copies={},
            owner_directory=str(self.owner),owner_source=dict(path=str(source),sha256=a.sha(source)),
            owner_plan=dict(path=str(self.owner/'owner-plan.json'),sha256=plan_sha),removed_at=self.t+3,
            trials=[self.phase],events=[self.event('prepared_v6_read_only',0),self.event('bootout_intent',1),
                self.event('bootout_returned',2),self.event('service_absent',3)],resource_samples=[])
        write(self.owner/'stopped-journal.json',self.data)
        self.attempt=dict(schema='cold-service-owner-attempt-v1',owner_pid=self.owner_pid,owner_identity=self.identity(self.owner_pid),
            owner_source=self.data['owner_source'],owner_plan_sha256=plan_sha,journal_id=self.data['id'],
            journal_path=str(self.journal),controller_sha256=a.CONTROLLER_SHA,removed_at=self.data['removed_at'],
            phase='PRESPAWN',created_at=self.t+4,stopped_journal_sha256=a.sha(self.owner/'stopped-journal.json'))
        write(self.owner/'attempt.json',self.attempt)
        bound={k:self.attempt[k] for k in ('owner_pid','owner_identity','owner_source','owner_plan_sha256','journal_id','removed_at')}
        bound['attempt_sha256']=a.sha(self.owner/'attempt.json')
        write(self.owner/'spawn-intent.json',dict(bound,time=self.t+9))
        identity=dict(pid=self.child_pid,ppid=self.owner_pid,start='CPU-child',comm='python')
        write(self.folder/'intent.json',dict(bound,phase=self.phase,time=self.t+10))
        write(self.folder/'child.json',dict(pid=self.child_pid,owner_pid=self.owner_pid,phase=self.phase,identity=identity,
            ownership='direct-Popen-returned',spawned_at=self.t+11))
        absence=self.folder/'child-absence.json'
        write(absence,dict(command=['/bin/ps','-p',str(self.child_pid),'-o','pid='],returncode=1,stdout=''))
        write(self.folder/'release.json',dict(phase=self.phase,pid=self.child_pid,returncode=1,waited_reaped=True,
            strict_absence=True,absence=dict(path=str(absence.relative_to(self.owner)),sha256=a.sha(absence)),time=self.t+20))
        row=dict(phase=self.phase,pid=self.child_pid,identity=identity)
        for kind in ('intent','child','release'):
            path=self.folder/(kind+'.json');row[kind]=dict(path=str(path.relative_to(self.owner)),sha256=a.sha(path))
        write(self.owner/'children.json',dict(schema='cold-service-owner-children-v1',children=[row]))
        write(self.owner/'release.json',dict(bound,schema='cold-service-owner-release-v2',children=[row],
            children_sha256=a.sha(self.owner/'children.json'),all_children_reaped=True,time=self.t+21))
        write(self.owner/'complete.json',dict(schema='E024-isolation-cold-owner-result-v1',planned_trials=[self.phase],status='FAIL',
            owner_pid=self.owner_pid,owner_plan_sha256=plan_sha))
        # Exact producer layout: absolute source-relative copies, no invented HTTP trial/port files.
        copies={str(HERE/'controller_v6.py'):a.CONTROLLER_SHA}
        for path in copies:
            target=self.owner/'owner-source'/path.lstrip('/');target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(Path(path).read_bytes())
        write(self.owner/'owner-source/manifest.json',copies)
        # Reuse only physical/log/load-worker fixture generation; its legacy hash
        # placeholder is removed before any E024 evidence is validated.
        write(self.owner/'child.json',{'fixture_only':True})
        old.AuditTests.recovery(self,status);(self.owner/'child.json').unlink()
        self.refresh_reap()
        self.binding=patch.object(a,'binding',return_value=(self.plan,{}));self.binding.start();self.addCleanup(self.binding.stop)

    def refresh_reap(self):
        files=[self.owner/'attempt.json',self.owner/'stopped-journal.json']
        files+=self.c.phase_reaps(type('J',(),{'data':self.data})(),self.owner,self.attempt)[0]
        self.data['reap']=dict(no_experiment=False,no_child_started=False,owner_pid=self.owner_pid,
            receipts=[dict(path=str(p),sha256=a.sha(p)) for p in files])
        write(self.journal/'journal.json',self.data)

    def rebind_child(self):
        index=json.loads((self.owner/'children.json').read_bytes())
        for kind in ('child','release'):index['children'][0][kind]['sha256']=a.sha(self.folder/(kind+'.json'))
        write(self.owner/'children.json',index)
        final=json.loads((self.owner/'release.json').read_bytes())
        final.update(children=index['children'],children_sha256=a.sha(self.owner/'children.json'))
        write(self.owner/'release.json',final);self.refresh_reap()

    def check(self):return a.audit(self.journal)

    def test_failed_diagnostic_successful_restoration_is_distinct_no_http_port(self):
        result=self.check();self.assertEqual(result['status'],'PASS',result)
        self.assertEqual(result['owner_reap']['owner_status'],'FAIL')
        self.assertEqual(result['owner_reap']['children'],[dict(phase=self.phase,pid=self.child_pid)])
        self.assertEqual(result['service_restore']['physical_samples'],8)
        self.assertFalse(result['numerical_qualified'] or result['performance_qualified'] or result['ledger_opened'])
        self.assertFalse(list(self.owner.rglob('port-*.json')))

    def test_known_popen_identity_probe_failure_still_requires_wait_and_absence(self):
        child=json.loads((self.folder/'child.json').read_bytes());child['identity']=None;write(self.folder/'child.json',child)
        index=json.loads((self.owner/'children.json').read_bytes());index['children'][0]['identity']=None
        write(self.owner/'children.json',index);self.rebind_child()
        self.assertEqual(self.check()['status'],'PASS')
        release=json.loads((self.folder/'release.json').read_bytes());release['waited_reaped']=False
        write(self.folder/'release.json',release)
        self.assertEqual(self.check()['status'],'FAIL')

    def test_actual_absence_bytes_not_claimed_flags_determine_reap(self):
        path=self.folder/'child-absence.json';write(path,dict(command=['/bin/ps','-p',str(self.child_pid),'-o','pid='],returncode=0,stdout=str(self.child_pid)))
        release=json.loads((self.folder/'release.json').read_bytes());release['absence']['sha256']=a.sha(path)
        write(self.folder/'release.json',release);self.rebind_child()
        result=self.check();self.assertEqual(result['status'],'FAIL');self.assertIn('raw child strict absence',result['error'])

    def test_no_child_rejects_any_nested_intent_and_unknown_spawn_is_incomplete(self):
        other=self.root/'no-child';other.mkdir();write(other/'attempt.json',self.attempt)
        shared={k:self.attempt[k] for k in ('owner_pid','owner_identity','owner_source','owner_plan_sha256','journal_id','removed_at')}
        write(other/'terminal.json',dict(shared,schema='cold-service-owner-terminal-v1',status='NO_CHILD_STARTED',
            attempt_sha256=a.sha(other/'attempt.json'),spawn_intent_written=False,child_started=False,terminal_at=self.t+22))
        self.assertEqual(a.no_child(self.data,other,self.attempt,[],self.c),self.t+22)
        write(other/'phases/e024-isolation/trial/spawn-intent.json',{})
        with self.assertRaisesRegex(ValueError,'nested spawn'):a.no_child(self.data,other,self.attempt,[],self.c)
        (self.owner/'release.json').unlink();self.assertEqual(self.check()['status'],'INCOMPLETE')

    def test_frozen_absolute_source_copy_manifest_and_payload_required(self):
        manifest=self.owner/'owner-source/manifest.json';write(manifest,{})
        self.assertEqual(self.check()['status'],'FAIL')
        write(manifest,{str(HERE/'controller_v6.py'):a.CONTROLLER_SHA})
        path=self.owner/'owner-source'/str(HERE/'controller_v6.py').lstrip('/');path.write_text('# changed\n')
        self.assertEqual(self.check()['status'],'FAIL')

    def test_original_physical_unit_and_recovery_time_gate_preserved(self):
        row=self.data['resource_samples'][0];path=Path(row['raw']['path'])
        raw=json.loads(path.read_bytes());raw['unit']='KiB';write(path,raw)
        row['raw']['sha256']=a.sha(path);write(self.journal/'journal.json',self.data)
        result=self.check();self.assertEqual(result['status'],'FAIL');self.assertIn('physical units',result['error'])
        path.unlink();self.assertEqual(self.check()['status'],'INCOMPLETE')

    def test_load_worker_and_log_missing_or_inconsistent_never_restore(self):
        path=self.journal/'load-worker/release.json';raw=json.loads(path.read_bytes());raw['waited_reaped']=False;write(path,raw)
        self.assertEqual(self.check()['status'],'FAIL')
        path.unlink();self.assertEqual(self.check()['status'],'INCOMPLETE')

    def test_missing_restoration_does_not_inspect_owner_or_support(self):
        write(self.journal/'journal.json',dict(state='LOAD_INTENT'))
        with patch.object(a,'sources',side_effect=AssertionError('no support before terminal')):
            self.assertEqual(self.check()['status'],'INCOMPLETE')

    def test_actual_fixed_plans_source_map_without_runtime_or_weight_execution(self):
        self.binding.stop();prior=dict(replacement=self.data['original'],final_status=self.data['original_status'],
            global_settings_stat={},model_metadata={})
        p=self.journal/'predecessor.json';r=self.journal/'predecessor-audit.json';write(p,prior);write(r,{})
        self.data['predecessor']=dict(path=str(p),sha256=a.sha(p),audit=dict(path=str(r),sha256=a.sha(r)))
        self.data['predecessor_copies']={p.name:a.sha(p),r.name:a.sha(r)}
        source=HERE.parent/'long_decode_isolation/cold_owner.py'
        fixed=next(ast.literal_eval(n.value) for n in ast.parse(source.read_text()).body
            if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='FIXED' for t in n.targets))
        pins={str((source.parent/n).resolve()):h for n,h in fixed.items()}
        pins.update({str(source):a.OWNER_SHA,str(source.with_name('test_cold_owner_cpu.py')):a.OWNER_TEST_SHA})
        owner_ref=dict(path=str(source),sha256=a.OWNER_SHA)
        original=json.loads((source.parent.parent/'long_decode_graph/prereg-v1/owner-plan.json').read_bytes())
        write(self.root/'isolation-plan.json',dict(schema='E024-dispatch-isolation-plan-v1',
            qualifier_sha256=fixed['qualify.py'],e024_plan=original['qualification_plan'],deadline_s=1800))
        plan=dict(schema='E024-isolation-cold-owner-plan-v1',execution=dict(owner=owner_ref,pins=pins),trials=[self.phase],deadline_seconds=1830,
            diagnostic_owner_plan=dict(path=str(source.parent.parent/'long_decode_graph/prereg-v1/owner-plan.json'),sha256=a.DIAGNOSTIC_OWNER_PLAN_SHA),
            isolation_plan=dict(path=str(self.root/'isolation-plan.json'),sha256=a.sha(self.root/'isolation-plan.json')),
            cold_restart=dict(journal_path=str(self.journal),controller=dict(path=str(HERE/'controller_v6.py'),sha256=a.CONTROLLER_SHA),
                approved_contract=self.c.CONTRACT,proposal_plan_sha256=self.c.PLAN_SHA,predecessor=self.data['predecessor']))
        write(self.owner/'owner-plan.json',plan)
        self.data.update(owner_plan=dict(path=str(self.owner/'owner-plan.json'),sha256=a.sha(self.owner/'owner-plan.json')),owner_source=owner_ref)
        with patch.object(self.c,'predecessor',return_value=(prior,[])):
            self.assertEqual(a.binding(self.data,self.journal,self.c,self.raw),(plan,pins))
            for key,value in [('deadline_seconds',1831),('trials',['r1-c1'])]:
                bad=copy.deepcopy(plan);bad[key]=value;write(self.owner/'owner-plan.json',bad)
                self.data['owner_plan']['sha256']=a.sha(self.owner/'owner-plan.json')
                with self.assertRaises(Exception):a.binding(self.data,self.journal,self.c,self.raw)

    def test_raw_restore_reused_no_backend_and_new_output_only(self):
        old_tree=ast.parse((HERE/'audit_restore_v5.py').read_text());new=ast.parse((HERE/'audit_restore_v6.py').read_text())
        for name in ('need','sha','read','finite','load','main'):
            before=next(n for n in old_tree.body if getattr(n,'name',None)==name);after=next(n for n in new.body if getattr(n,'name',None)==name)
            self.assertEqual(ast.dump(before),ast.dump(after),name)
        with patch.object(self.c.b,'System',side_effect=AssertionError('no backend')):
            self.assertEqual(self.check()['status'],'PASS')
        out=self.root/'audit.json';write(self.journal/'journal.json',dict(state='PREPARED'))
        self.assertEqual(a.main(['--journal',str(self.journal),'--out',str(out)]),2)
        with self.assertRaises(ValueError):a.main(['--journal',str(self.journal),'--out',str(out)])


if __name__=='__main__':unittest.main()
