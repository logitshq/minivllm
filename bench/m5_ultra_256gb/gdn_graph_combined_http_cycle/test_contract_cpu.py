"""Source/closed-baseline contracts only; no service or model operations."""
import ast
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
def load(name):
    spec=importlib.util.spec_from_file_location('_combined_contract_'+name,HERE/(name+'.py'))
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value
owner=load('owner');controller=load('controller');restore=load('audit_restore')

class Contracts(unittest.TestCase):
    def test_run_control_flow_only_protocol_literals_changed(self):
        def tree(path):
            return next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='run')
        old=tree(HERE.parent/'gdn_prework_http_cycle/owner.py');new=tree(HERE/'owner.py')
        replacements={'GDN-prework-http':'GDN-graph-combined-http','GDN-recorded-http':'GDN-graph-combined-recorded-http',
                      'dispatch_events_retained':'route_events_retained','GDN-only':'Combined GDN+route'}
        for node in ast.walk(old):
            if isinstance(node,ast.Constant) and type(node.value) is str:
                for a,b in replacements.items():node.value=node.value.replace(a,b)
        self.assertEqual(ast.dump(old,include_attributes=False),ast.dump(new,include_attributes=False))

    def test_recovery_functions_use_actual_namespace_and_preserve_donor(self):
        for name in controller._REUSED:
            actual=getattr(controller,name);original=getattr(controller._donor,name)
            self.assertIs(actual.__code__,original.__code__)
            self.assertIs(actual.__globals__,vars(controller))
            self.assertIs(original.__globals__,vars(controller._donor))
        self.assertEqual(controller.PHASES,owner.PHASES)
        self.assertEqual(restore.CONTROLLER_SHA,owner.sha(HERE/'controller.py'))
        self.assertEqual(controller.OWNER_SHA,owner.sha(HERE/'owner.py'))

    def test_missing_or_pending_baseline_refuses_before_source_preparation(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(owner,'BASELINE',Path(tmp)/'baseline.json'):
            with self.assertRaisesRegex(RuntimeError,'missing'):owner.baseline()
            owner.BASELINE.write_text(json.dumps({'schema':'GDN-graph-combined-http-baseline-v1','status':'PENDING','predecessor':{},'serving':{}}))
            with self.assertRaisesRegex(RuntimeError,'CLOSED'):owner.baseline()

    def test_closed_manifest_requires_exact_bytes_for_all_five_refs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();refs={}
            for name in ('journal','audit','plan','numerical','complete'):
                path=root/(name+'.json');path.write_text('{}');refs[name]={'path':str(path),'sha256':owner.sha(path)}
            value={'schema':'GDN-graph-combined-http-baseline-v1','status':'CLOSED',
                'predecessor':{**refs['journal'],'audit':refs['audit']},
                'serving':dict(plan=refs['plan'],audit=refs['numerical'],complete=refs['complete'])}
            path=root/'baseline.json';path.write_text(json.dumps(value))
            with patch.object(owner,'BASELINE',path):
                self.assertEqual(owner.baseline(),value)
                for item in refs.values():
                    changed=Path(item['path']);changed.write_text('{"changed":true}')
                    with self.assertRaisesRegex(RuntimeError,'changed'):owner.baseline()
                    changed.write_text('{}')

    def test_observed_load_intent_is_not_accepted_as_restored_predecessor(self):
        binding={'path':'/CPU/journal.json','sha256':'0'*64,'audit':{'path':'/CPU/audit.json','sha256':'0'*64}}
        fake=SimpleNamespace(baseline=lambda:{'predecessor':{**binding,'sha256':'1'*64}})
        with patch.object(controller,'load',return_value=fake):
            with self.assertRaisesRegex(controller.Refuse,'CLOSED manifest'):controller.predecessor(binding)
        source=(HERE/'controller.py').read_text()
        self.assertIn("prior.get('state')=='RESTORED'",source)
        self.assertIn("restored.get('http_worker_returncode')==0",source)
        self.assertIn("not prior.get('post_monitor_failed')",source)

    def test_no_terminal_restore_never_grants_qualification(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp);(path/'journal.json').write_text('{"state":"LOAD_INTENT"}')
            value=restore.audit(path)
            self.assertEqual(value['status'],'INCOMPLETE');self.assertFalse(value['qualified'])
            self.assertFalse(value['performance_qualified']);self.assertFalse(value['numerical_qualified'])

if __name__=='__main__':unittest.main()
