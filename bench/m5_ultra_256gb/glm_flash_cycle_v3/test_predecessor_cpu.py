"""Three narrow fake-archive predecessor tests; no actual service observation."""
import ast
import copy
import importlib.util
import inspect
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

path=Path(__file__).with_name('controller.py')
spec=importlib.util.spec_from_file_location('glm_v3_formal_predecessor_cpu',path)
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)


class PredecessorTests(unittest.TestCase):
    def fixtures(self):
        prior=dict(schema='GDN-graph-combined-http-formal-cold-journal-v1',state='RESTORED',
            controller_sha256=c.PRIOR_SHA,trials=list(c.PRIOR_PHASES),approved_contract=c.CONTRACT,
            plan_sha256=c.PLAN_SHA,support_sha256=c.BASE_SHA,events=[],post_monitor_failed=False,
            replacement=dict(pid=85002,ppid=1,start='CPU fake start',comm='omlx-server'),
            owner_source=dict(path=str(c.PRIOR_OWNER_PATH),sha256=c.PRIOR_OWNER_SHA),
            owner_plan=dict(path='/CPU/owner-plan.json',sha256='1'*64),
            log_window=dict(sha256='2'*64,bytes=100),
            final_status=dict(version='0.7.0',default_model=c.b.TARGET,loaded=[c.b.TARGET],loading=0,
                target_loading=False,path=c.b.MODEL,pinned=False,engine_type='vlm',config_model_type='qwen4_exp'),
            exact_runtime_signature_restored=False,warm_kv_restored=False)
        audit=dict(schema='cold-service-restoration-audit-v1',status='PASS',qualified=True,
            audit_source_sha256=c.PRIOR_AUDIT_SHA,controller_sha256=c.PRIOR_SHA,
            owner_plan_sha256='1'*64,owner_reap={'owner_status':'FAIL'},
            service_restore=dict(status='PERSISTED_VISIBLE_RECEIPTS_CONSISTENT',replacement=prior['replacement'],
                raw_log_sha256='2'*64,raw_log_bytes=100,http_worker_returncode=0))
        return prior,audit

    def check(self,prior,audit,*,wrong_path=False,bad_owner_source=False):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve();folder=root/'E032-gdn-graph-combined-formal-v1';folder.mkdir()
            jp=folder/'restored-journal.json';ap=folder/'restoration-audit.json'
            jp.write_text(json.dumps(prior));audit=copy.deepcopy(audit);audit['journal_sha256']=c.b.sha(jp)
            ap.write_text(json.dumps(audit));binding=dict(path=str(jp),sha256=c.b.sha(jp),audit=dict(path=str(ap),sha256=c.b.sha(ap)))
            manifest=dict(binding=binding,trials=list(c.PRIOR_PHASES))
            with patch.object(c,'predecessor_manifest',return_value=manifest),patch.object(c,'ARCHIVE_ROOT',root),\
                 patch.object(c,'PRIOR_ARCHIVE',folder if not wrong_path else root/'different'),\
                 patch.object(c,'PRIOR_OWNER_SHA','0'*64 if bad_owner_source else c.PRIOR_OWNER_SHA):
                return c.predecessor(binding)

    def test_only_schema_literal_changes_original_restoration_gates(self):
        restored=c._predecessor_source.replace(c._new_schema,c._old_schema,1)
        self.assertEqual(ast.dump(ast.parse(restored)),ast.dump(ast.parse(inspect.getsource(c._original.predecessor))))
        self.assertIs(c.stop.__code__,c._original.stop.__code__)
        self.assertIs(c.recover.__code__,c._original.recover.__code__)
        self.assertIs(c._original_phase_reaps.__code__,c._original.phase_reaps.__code__)

    def test_failed_experiment_restored_pass_allowed_but_incomplete_or_wrong_owner_rejected(self):
        prior,audit=self.fixtures();actual,raw=self.check(prior,audit)
        self.assertEqual(actual,prior);self.assertEqual(len(raw),2)
        changes=[('prior','state','LOAD_INTENT'),('prior','trials',list(c.PRIOR_PHASES)[:-1]),
                 ('prior','owner_source',dict(path=str(c.PRIOR_OWNER_PATH),sha256='0'*64)),
                 ('prior','schema','cold-single-child-journal-v3'),('prior','post_monitor_failed',True),
                 ('prior','events',[{'event':'post_worker_cleanup_failed'}]),
                 ('audit','status','FAIL'),('audit','audit_source_sha256','0'*64),
                 ('audit','owner_plan_sha256','0'*64)]
        for target,key,value in changes:
            with self.subTest(target=target,key=key):
                p,a=copy.deepcopy(prior),copy.deepcopy(audit);(p if target=='prior' else a)[key]=value
                with self.assertRaises(Exception):self.check(p,a)
        a=copy.deepcopy(audit);a['service_restore']['http_worker_returncode']=1
        with self.assertRaises(Exception):self.check(prior,a)
        a=copy.deepcopy(audit);a['service_restore']['raw_log_sha256']='3'*64
        with self.assertRaises(Exception):self.check(prior,a)

    def test_unsealed_real_binding_wrong_archive_and_source_change_fail_closed(self):
        with self.assertRaisesRegex(Exception,'not sealed'):c.predecessor({})
        prior,audit=self.fixtures()
        with self.assertRaisesRegex(Exception,'exact E032'):self.check(prior,audit,wrong_path=True)
        with self.assertRaisesRegex(Exception,'owner source changed'):self.check(prior,audit,bad_owner_source=True)


if __name__=='__main__':unittest.main()
