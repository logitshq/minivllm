"""Prepared receipt-admission tests: files only, no OS/GPU/service backend."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('cold_v2_predecessor_tests',HERE/'controller.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)

def ref(path):return dict(path=str(path.resolve()),sha256=c.b.sha(path))

class PredecessorTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='cold-v2-receipts-')
        self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        folder=c.ARCHIVE_ROOT/'E024-route-canary-v1'
        self.prior=json.loads((folder/'restored-journal.json').read_bytes())
        self.audit=json.loads((folder/'restoration-audit.json').read_bytes())
        self.manifest=ref(HERE/'lifecycle.json')
        self.context=patch.object(c,'ARCHIVE_ROOT',self.root);self.context.start();self.addCleanup(self.context.stop)
    def pair(self,kind='formal'):
        prior,audit=copy.deepcopy(self.prior),copy.deepcopy(self.audit)
        if kind=='formal':
            prior['schema']='E023-lithos-D512-formal-cold-journal-v1'
            controller='5c13010c5645e93eb099592e32ee6192f8a2ec7ca81c5ab2ed67c06f7e4273c2'
            auditor='41d260e63ce13c5c5912217ae74169e5c3cd118af32f3667aefffc3637a2d66f'
        elif kind=='v1':
            prior['schema']='cold-single-child-journal-v1'
            controller='b5237c458afa8324ffa6d5e6a5c49167578daec7425ff036112ec8e108d61213'
            auditor='08f5a24dc9d100ca5e5b4ed469ec6587f7ba5b36b3910e6e580e7027f32b5d21'
            prior['lifecycle']=audit['lifecycle']=ref(HERE.parent/'cold_cycle/lifecycle.json')
        else:
            prior['schema']=c.JOURNAL_SCHEMA
            controller=c.b.sha(HERE/'controller.py');auditor=c.b.sha(HERE/'audit_restore.py')
            prior['lifecycle']=audit['lifecycle']=self.manifest
        prior['controller_sha256']=audit['controller_sha256']=controller
        audit['audit_source_sha256']=auditor
        return prior,audit
    def save(self,prior,audit):
        journal=self.root/'restored-journal.json';receipt=self.root/'restoration-audit.json'
        journal.write_text(json.dumps(prior));audit=dict(audit,journal_sha256=c.b.sha(journal))
        receipt.write_text(json.dumps(audit))
        return dict(**ref(journal),audit=ref(receipt))
    def test_reviewed_formal_v1_and_same_v2_receipts_accepted_without_backend(self):
        with patch.object(c,'System',side_effect=AssertionError('no service backend')), \
             patch.object(c.b,'System',side_effect=AssertionError('no service backend')):
            for kind in ('formal','v1','v2'):
                with self.subTest(kind=kind):
                    prior,audit=self.pair(kind);binding=self.save(prior,audit)
                    actual,_=c.predecessor(binding,self.manifest)
                    self.assertEqual(actual,prior)
    def test_source_qualification_and_unsettled_load_receipts_rejected(self):
        for change in ('controller','auditor','qualified','state','post_monitor','worker','replacement','visible','owner_plan'):
            prior,audit=self.pair()
            if change=='controller':prior['controller_sha256']='0'*64
            elif change=='auditor':audit['audit_source_sha256']='0'*64
            elif change=='qualified':audit['qualified']=False
            elif change=='state':prior['state']='LOAD_INTENT'
            elif change=='post_monitor':prior['post_monitor_failed']=True
            elif change=='worker':audit['service_restore']['http_worker_returncode']=1
            elif change=='replacement':audit['service_restore']['replacement']['pid']+=1
            elif change=='visible':prior['final_status']['loading']=1
            else:audit['owner_plan_sha256']='0'*64
            with self.subTest(change=change),self.assertRaises(c.Refuse):
                c.predecessor(self.save(prior,audit),self.manifest)
    def test_stale_hash_unknown_schema_and_wrong_v1_lifecycle_rejected(self):
        prior,audit=self.pair();binding=self.save(prior,audit)
        Path(binding['path']).write_text('{}')
        with self.assertRaises(c.Refuse):c.predecessor(binding,self.manifest)
        prior['schema']='unreviewed-cold-journal'
        with self.assertRaises(c.Refuse):c.predecessor(self.save(prior,audit),self.manifest)
        prior,audit=self.pair('v1');prior['lifecycle']=self.manifest
        with self.assertRaises(c.Refuse):c.predecessor(self.save(prior,audit),self.manifest)
    def test_monitored_post_and_reap_algorithms_remain_same_ast(self):
        import ast
        old=ast.parse((HERE.parent/'cold_cycle/controller.py').read_text())
        new=ast.parse((HERE/'controller.py').read_text())
        left={n.name:n for n in old.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
        right={n.name:n for n in new.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
        self.assertEqual(set(left),set(right))
        for name in left.keys()-{'predecessor'}:
            with self.subTest(name=name):self.assertEqual(ast.dump(left[name]),ast.dump(right[name]))
        self.assertEqual((HERE/'owner.py').read_bytes(),(HERE.parent/'cold_cycle/owner.py').read_bytes())

if __name__=='__main__':unittest.main()
