"""Small real-file closure gates; no spec, runtime, process or service execution."""
from contextlib import ExitStack
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SOURCE=Path(__file__).with_name('prepare_specs.py')

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,sort_keys=True)+'\n')

class ClosedArchiveTests(unittest.TestCase):
    def setUp(self):
        self.stack=ExitStack();self.addCleanup(self.stack.close)
        self.root=Path(self.stack.enter_context(tempfile.TemporaryDirectory())).resolve()
        spec=importlib.util.spec_from_file_location('_gdn_formal_prepare_specs_cpu',SOURCE)
        self.m=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.m)
        self.archive=self.root/'archive';self.analysis=self.root/'analysis';self.reg=self.root/'reg'
        self.journal=self.root/'journal.json';self.final=self.root/'final.json'
        for name,value in dict(ARCHIVE=self.archive,ANALYSIS=self.analysis,ROUTE_REG=self.reg,
                               JOURNAL=self.journal,EXECUTION=self.final).items():
            self.stack.enter_context(patch.object(self.m,name,value))
        write(self.reg/'owner-plan.json',{'fixture':'registered plan'})
        self.plan_sha=self.m.sha(self.reg/'owner-plan.json')
        self.stack.enter_context(patch.object(self.m,'CYCLE_PLAN_SHA',self.plan_sha))
        write(self.journal,{'state':'RESTORED','cpu':'fixture'})
        write(self.final,{'journal_state':'RESTORED','recovery_returncode':0,'owner_returncode':0})
        restore=dict(status='PASS',qualified=True,journal_sha256=self.m.sha(self.journal),owner_plan_sha256=self.plan_sha)
        cycle=dict(status='PASS',qualified=True,restoration_verified=True,independent_resource_verified=True,
            http_history_qualified=True,journal_sha256=restore['journal_sha256'],owner_plan_sha256=self.plan_sha,restoration=restore)
        outputs={'restoration':('restoration-audit.json',restore),'cycle':('independent-cycle.json',cycle),
                 'c1':('independent-c1.json',{'status':'PASS'}),'c8':('independent-c8.json',{'status':'PASS'})}
        jobs={}
        for label,(name,value) in outputs.items():
            path=self.analysis/name;write(path,value)
            jobs[label]=dict(returncode=0,audit_status='PASS',ended_at='CPU-ended',
                            output=dict(path=str(path),sha256=self.m.sha(path),bytes=path.stat().st_size))
        summary=dict(schema='GDN-http-closed-audits-v1',status='PASS',restoration_qualified=True,numerical_qualified=True,
            http_history_qualified=True,performance_qualified=False,sources_unchanged=True,inputs_unchanged=True,
            ended_at='CPU-ended',owner_plan_sha256=self.plan_sha,jobs=jobs)
        write(self.analysis/'execution-summary.json',summary)
        self.archive.mkdir();files={}
        aliases={'restored-journal.json':self.journal,'audit-execution-summary.json':self.analysis/'execution-summary.json',
                 **{name:self.analysis/name for name,_ in outputs.values()}}
        for name,path in aliases.items():
            target=self.archive/name;target.write_bytes(path.read_bytes())
            files['archive-sidecars/'+name]=dict(source=str(target),sha256=self.m.sha(target),bytes=target.stat().st_size)
        write(self.archive/'summary.json',dict(schema='GDN-http-archive-summary-v1',owner_plan_sha256=self.plan_sha,
            owner_status='PASS',audit_status='PASS',restoration_qualified=True,http_history_qualified=True,performance_qualified=False))
        write(self.archive/'manifest.json',dict(files=files))
        archive=dict(sha256='a'*64,bytes=1)
        write(self.archive/'archive-stream.json',dict(archive=archive))
        write(self.archive/'archive-verification.json',dict(schema='GDN-http-archive-verification-v1',status='PASS',
            all_regular_members_decoded_verified=True,all_original_files_rehashed=True,all_tree_members_rechecked=True,
            archive=archive,members=len(files)))
        self.expected=cycle

    def test_actual_paths_closed_receipts_and_exact_sidecars_are_required(self):
        self.assertEqual(self.m.closed_gate(),self.expected)
        path=self.archive/'independent-c1.json';path.write_bytes(path.read_bytes()+b' ')
        with self.assertRaisesRegex(ValueError,'sidecar'):self.m.closed_gate()

    def test_restoration_and_count_qualification_cannot_be_substituted(self):
        write(self.final,{'journal_state':'RESTORED','recovery_returncode':1,'owner_returncode':0})
        with self.assertRaisesRegex(ValueError,'restored driver'):self.m.closed_gate()
        write(self.final,{'journal_state':'RESTORED','recovery_returncode':0,'owner_returncode':0})
        path=self.analysis/'execution-summary.json';value=json.loads(path.read_bytes());value['numerical_qualified']=False;write(path,value)
        with self.assertRaisesRegex(ValueError,'audit execution'):self.m.closed_gate()

    def test_archive_verification_failure_is_not_an_execution_prerequisite_pass(self):
        path=self.archive/'archive-verification.json';value=json.loads(path.read_bytes());value['all_regular_members_decoded_verified']=False;write(path,value)
        with self.assertRaisesRegex(ValueError,'archive verification'):self.m.closed_gate()

if __name__=='__main__':unittest.main()
