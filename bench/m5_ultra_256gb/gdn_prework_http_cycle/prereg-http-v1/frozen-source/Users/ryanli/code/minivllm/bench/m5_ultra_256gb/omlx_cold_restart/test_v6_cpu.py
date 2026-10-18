"""CPU files/fakes only; no actual controller prepare/stop/recover or process."""
import ast
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
c=module(HERE/'controller_v6.py','controller_v6_cpu')


class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        identity=dict(pid=12345,ppid=1,start='CPU prior replacement',comm='omlx-server')
        self.prior=dict(schema='omlx-cold-restart-journal-v5',controller_sha256=c.V5_SHA,state='RESTORED',
            approved_contract=c.CONTRACT,plan_sha256=c.PLAN_SHA,support_sha256=c.BASE_SHA,replacement=identity,
            final_status=dict(version='0.7.0',default_model=c.b.TARGET,loaded=[c.b.TARGET],loading=0,target_loading=False,
                path=c.b.MODEL,pinned=False,engine_type='vlm',config_model_type='qwen4_exp'),
            log_window=dict(sha256='a'*64,bytes=123),owner_plan=dict(sha256='b'*64),events=[],
            exact_runtime_signature_restored=False,warm_kv_restored=False)
        self.audit=dict(schema='cold-service-restoration-audit-v1',status='PASS',qualified=True,
            audit_source_sha256=c.RESTORE_AUDIT_SHA,controller_sha256=c.V5_SHA,owner_plan_sha256='b'*64,
            numerical_qualified=False,performance_qualified=False,
            service_restore=dict(status='PERSISTED_VISIBLE_RECEIPTS_CONSISTENT',replacement=identity,
                raw_log_sha256='a'*64,raw_log_bytes=123,http_worker_returncode=0))
    def binding(self):
        p=self.root/'restored.json';a=self.root/'audit.json';c.b.atomic_write(p,self.prior)
        self.audit['journal_sha256']=c.b.sha(p);c.b.atomic_write(a,self.audit)
        return dict(path=str(p),sha256=c.b.sha(p),audit=dict(path=str(a),sha256=c.b.sha(a)))
    def test_exact_v5_audit_binding_allows_failed_experiment_restore(self):
        self.audit['owner_reap']={'owner_status':'FAIL'}
        binding=self.binding()
        with patch.object(c,'ARCHIVE_ROOT',self.root),patch.object(c.b,'System',side_effect=AssertionError('no backend')):
            self.assertEqual(c.predecessor(binding)[0]['replacement']['pid'],12345)
    def test_rejects_nonarchive_unknown_controller_and_unreviewed_audit(self):
        with self.assertRaises(c.Refuse):c.predecessor(self.binding())
        saved=copy.deepcopy(self.audit)
        with patch.object(c,'ARCHIVE_ROOT',self.root):
            for field,value in [('status','FAIL'),('qualified',False),('audit_source_sha256','wrong'),
                                ('controller_sha256','wrong'),('owner_plan_sha256','wrong')]:
                self.audit=copy.deepcopy(saved);self.audit[field]=value
                with self.subTest(field=field),self.assertRaises(c.Refuse):c.predecessor(self.binding())
    def test_incomplete_restore_false_runtime_claim_and_worker_error_rejected(self):
        with patch.object(c,'ARCHIVE_ROOT',self.root):
            for field,value in [('state','LOAD_INTENT'),('schema','omlx-cold-restart-journal-v6'),
                                ('controller_sha256','wrong'),('warm_kv_restored',True),('post_monitor_failed',True)]:
                old=copy.deepcopy(self.prior);self.prior[field]=value
                with self.subTest(field=field),self.assertRaises(c.Refuse):c.predecessor(self.binding())
                self.prior=old
            self.audit['service_restore']['http_worker_returncode']=1
            with self.assertRaises(c.Refuse):c.predecessor(self.binding())
    def test_registered_phase_is_only_e024_isolation(self):
        self.assertEqual(c.trial_names(['e024-isolation']),['e024-isolation'])
        for value in [[],['r1-c1'],['e024-isolation','e024-isolation'],['r1-c1','r1-c8'],(c.PHASES)]:
            with self.subTest(value=value),self.assertRaises(c.Refuse):c.trial_names(value)
    def test_original_os_resource_stop_recovery_and_real_worker_cli_unchanged(self):
        old=ast.parse((HERE/'controller_v5.py').read_text());new=ast.parse((HERE/'controller_v6.py').read_text())
        for name in ('System','parse_resources','sample','check_resource','resource_gate','atomic_bytes','exclusive_json',
                     'fixed_identity','original_identity','absent_service','replacement','stop','monitored_load','recover','main'):
            a=next(n for n in old.body if getattr(n,'name',None)==name);b=next(n for n in new.body if getattr(n,'name',None)==name)
            self.assertEqual(ast.dump(a),ast.dump(b),name)
        before=copy.deepcopy(next(n for n in old.body if getattr(n,'name',None)=='load_worker'))
        after=next(n for n in new.body if getattr(n,'name',None)=='load_worker')
        for node in ast.walk(before):
            if isinstance(node,ast.Constant) and node.value=='omlx-cold-restart-journal-v5':node.value='omlx-cold-restart-journal-v6'
        self.assertEqual(ast.dump(before),ast.dump(after))
    def test_help_blocks_model_network_subprocess(self):
        script='''
import importlib.abc,importlib.util,http.client,socket,subprocess,sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,name,*args):
  if name.split('.')[0] in {'mlx','numpy','torch','tutorial_code'}:raise AssertionError(name)
sys.meta_path.insert(0,Block())
socket.socket=lambda *a,**k:(_ for _ in ()).throw(AssertionError('socket'))
subprocess.Popen=lambda *a,**k:(_ for _ in ()).throw(AssertionError('process'))
sp=importlib.util.spec_from_file_location('v6',sys.argv[1]);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
raise SystemExit(m.main(['--help']))
'''
        r=subprocess.run([sys.executable,'-I','-B','-c',script,str(HERE/'controller_v6.py')],capture_output=True,text=True,timeout=5)
        self.assertEqual(r.returncode,0,r.stderr)


if __name__=='__main__':unittest.main()
