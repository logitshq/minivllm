"""Synthetic receipt/AST tests only. Future route HTTP bytes do not exist yet."""
import copy
from contextlib import ExitStack
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent

def module(name):
    spec=importlib.util.spec_from_file_location('cold_v3_test_'+name,HERE/(name+'.py'))
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value
c=module('controller');a=module('audit_restore')
def ref(path):return dict(path=str(path.resolve()),sha256=c.b.sha(path))
def write(path,value):Path(path).write_text(json.dumps(value,indent=2)+'\n')


class PredecessorTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='cold-v3-receipts-');self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name).resolve();self.folder=self.root/'route-http-synthetic';self.folder.mkdir()
        source=c.ARCHIVE_ROOT/'E024-route-canary-v1'
        self.prior=json.loads((source/'restored-journal.json').read_bytes())
        self.audit=json.loads((source/'restoration-audit.json').read_bytes())
        self.manifest=ref(HERE/'lifecycle.json')
        self.stack=ExitStack();self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(c,'ARCHIVE_ROOT',self.root))
        self.stack.enter_context(patch.object(c,'System',side_effect=AssertionError('no OS backend')))
        self.stack.enter_context(patch.object(c.b,'System',side_effect=AssertionError('no OS backend')))

    def pair(self,kind='route'):
        prior,audit=copy.deepcopy(self.prior),copy.deepcopy(self.audit)
        if kind=='route':
            controller=c.b.sha(HERE.parent/'route_http_cycle/controller.py')
            auditor=c.b.sha(HERE.parent/'route_http_cycle/audit_restore.py')
            prior.update(schema=c.MIGRATION_SCHEMA,phases=['c1-capture','c1-replay','c8-capture','c8-replay'])
        else:
            controller=c.b.sha(HERE/'controller.py');auditor=c.b.sha(HERE/'audit_restore.py')
            prior.update(schema=c.JOURNAL_SCHEMA,lifecycle=copy.deepcopy(self.manifest))
            audit['lifecycle']=copy.deepcopy(self.manifest)
        prior['controller_sha256']=audit['controller_sha256']=controller
        audit['audit_source_sha256']=auditor
        # The previous experiment's failure is explicitly irrelevant to restore.
        audit['owner_reap']={'owner_status':'FAIL','CPU_synthetic':True}
        return prior,audit

    def save(self,prior,audit):
        journal=self.folder/'restored-journal.json';receipt=self.folder/'restoration-audit.json'
        write(journal,prior);audit=copy.deepcopy(audit);audit['journal_sha256']=c.b.sha(journal);write(receipt,audit)
        return dict(**ref(journal),audit=ref(receipt))

    def seal_synthetic(self,binding):
        stack=ExitStack()
        values=dict(MIGRATION_ARCHIVE=self.folder,MIGRATION_JOURNAL_SHA=binding['sha256'],
            MIGRATION_AUDIT_SHA=binding['audit']['sha256'],
            MIGRATION_CONTROLLER_SHA=c.b.sha(HERE.parent/'route_http_cycle/controller.py'),
            MIGRATION_AUDITOR_SHA=c.b.sha(HERE.parent/'route_http_cycle/audit_restore.py'))
        for key,value in values.items():stack.enter_context(patch.object(c,key,value))
        return stack

    def test_unsealed_actual_migration_never_authorizes_backend(self):
        prior,audit=self.pair();binding=self.save(prior,audit)
        with patch.object(c,'MIGRATION_JOURNAL_SHA','UNSEALED'),self.assertRaises(c.Refuse):
            c.predecessor(binding,self.manifest)

    def test_explicit_synthetic_route_migration_and_same_version_only(self):
        prior,audit=self.pair();binding=self.save(prior,audit)
        with self.seal_synthetic(binding):
            actual,_=c.predecessor(binding,self.manifest);self.assertEqual(actual,prior)
        prior,audit=self.pair('same');binding=self.save(prior,audit)
        actual,_=c.predecessor(binding,self.manifest);self.assertEqual(actual,prior)
        for schema in ('cold-single-child-journal-v2','E023-omlx-D512-formal-cold-journal-v1'):
            prior['schema']=schema
            with self.subTest(schema=schema),self.assertRaises(c.Refuse):
                c.predecessor(self.save(prior,audit),self.manifest)

    def test_route_hash_source_protocol_and_restore_failures_rejected(self):
        for change in ('source','auditor','phases','qualified','state','worker','replacement','visible','owner_plan','monitor'):
            prior,audit=self.pair()
            if change=='source':prior['controller_sha256']='0'*64
            elif change=='auditor':audit['audit_source_sha256']='0'*64
            elif change=='phases':prior['phases']=['c1-capture']
            elif change=='qualified':audit['qualified']=False
            elif change=='state':prior['state']='LOAD_INTENT'
            elif change=='worker':audit['service_restore']['http_worker_returncode']=1
            elif change=='replacement':audit['service_restore']['replacement']['pid']+=1
            elif change=='visible':prior['final_status']['loading']=1
            elif change=='owner_plan':audit['owner_plan_sha256']='0'*64
            else:prior['post_monitor_failed']=True
            binding=self.save(prior,audit)
            with self.subTest(change=change),self.seal_synthetic(binding),self.assertRaises(c.Refuse):
                c.predecessor(binding,self.manifest)
        prior,audit=self.pair();binding=self.save(prior,audit)
        with self.seal_synthetic(binding):
            other=self.root/'other';other.mkdir()
            for name in ('restored-journal.json','restoration-audit.json'):
                (other/name).write_bytes((self.folder/name).read_bytes())
            moved=dict(**ref(other/'restored-journal.json'),audit=ref(other/'restoration-audit.json'))
            with self.assertRaises(c.Refuse):c.predecessor(moved,self.manifest)
            (self.folder/'restored-journal.json').write_text('{}')
            with self.assertRaises(c.Refuse):c.predecessor(binding,self.manifest)

    def test_source_closure_contains_real_argument_helpers(self):
        manifest=c.lifecycle(self.manifest)
        plan={'predecessor':{'sha256':'0'*64},'experiment':{'source_pins':{},'plan':self.manifest},
              'lifecycle':self.manifest,'inherited_owner_plan':self.manifest}
        pins=c.source_pins(plan,manifest)
        for path,digest in c.ADAPTER_SUPPORT_PINS.items():
            self.assertEqual(pins[str((HERE/path).resolve())],digest)
        self.assertEqual(pins[str(HERE/'owner.py')],c.b.sha(HERE/'owner.py'))

    def test_restore_no_child_missing_copy_is_allowed_but_bad_copy_rejected(self):
        owner=self.root/'owner';owner.mkdir();journal=self.root/'journal';journal.mkdir()
        pid=99001;identity=dict(pid=pid,ppid=1,start='CPU-only',comm='python')
        source=ref(HERE/'owner.py');planref={'path':str(self.root/'registered-plan.json'),'sha256':'a'*64}
        data=dict(schema=c.JOURNAL_SCHEMA,state='RESTORED',id='CPU',removed_at=100.,events=[],
            controller_sha256=c.b.sha(HERE/'controller.py'),support_sha256=c.BASE_SHA,
            plan_sha256=c.PLAN_SHA,approved_contract=c.CONTRACT,original={},original_status={},
            global_settings_stat={},model_metadata={},log_start={},owner_directory=str(owner),
            owner_source=source,owner_plan=planref,predecessor={},predecessor_copies={},
            trials=['CPU-phase'],lifecycle=self.manifest)
        stopped=copy.deepcopy(data);stopped['state']='STOPPED';write(owner/'stopped-journal.json',stopped)
        attempt=dict(schema='cold-service-owner-attempt-v1',phase='PRESPAWN',owner_pid=pid,owner_identity=identity,
            owner_source=source,owner_plan_sha256=planref['sha256'],journal_id='CPU',journal_path=str(journal),
            controller_sha256=data['controller_sha256'],removed_at=100.,created_at=101.,
            stopped_journal_sha256=c.b.sha(owner/'stopped-journal.json'))
        write(owner/'attempt.json',attempt)
        shared={k:attempt[k] for k in ('owner_pid','owner_identity','owner_source','owner_plan_sha256','journal_id','removed_at')}
        write(owner/'terminal.json',dict(schema='cold-service-owner-terminal-v1',status='NO_CHILD_STARTED',**shared,
            attempt_sha256=c.b.sha(owner/'attempt.json'),spawn_intent_written=False,child_started=False,terminal_at=102.))
        write(owner/'children.json',dict(schema='cold-service-owner-children-v1',children=[]))
        write(owner/'complete.json',dict(schema='cold-single-child-owner-result-v1',status='FAIL',owner_pid=pid,
            owner_plan_sha256=planref['sha256'],planned_trials=data['trials'],lifecycle=self.manifest))
        files=[owner/name for name in ('attempt.json','stopped-journal.json','terminal.json','children.json')]
        data['reap']=dict(no_experiment=False,no_child_started=True,owner_pid=pid,receipts=[ref(p) for p in files])
        fake=NS(registered_trials=lambda j:list(data['trials']))
        raw=a.load(HERE.parent/'pareto_realistic/audit_absent_qualification.py',a.RAW_AUDITOR_SHA,'cpu_raw_identity')
        released,result=a.owner_reap(data,journal,{},fake,raw)
        self.assertEqual(released,102.);self.assertEqual(result['status'],'NO_CHILD_STARTED')
        (owner/'owner-plan.json').write_text('{}')
        with self.assertRaises(ValueError):a.owner_reap(data,journal,{},fake,raw)

if __name__=='__main__':unittest.main()
