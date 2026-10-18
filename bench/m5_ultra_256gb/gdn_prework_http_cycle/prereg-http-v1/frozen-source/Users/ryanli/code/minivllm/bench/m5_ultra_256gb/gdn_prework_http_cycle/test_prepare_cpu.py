"""Registration wiring only: actual builder, synthetic closed receipts, no backend."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('gdn_http_registration_cpu',HERE/'prepare.py')
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v)+'\n')

class Tests(unittest.TestCase):
    def fixture(self,root):
        art=root/'art';art.mkdir();reg=root/'reg';archive=root/'archive';child=root/'previous-child'
        audit=archive/'serving-audit.json'
        for name,value in [('restored-journal.json',{'state':'RESTORED'}),('restoration-audit.json',{'status':'PASS'}),('archive-verification.json',{'status':'PASS'})]:write(archive/name,value)
        write(audit,{'child_directory':str(child)})
        write(child/'plan.json',{});write(child/'complete.json',{})
        write(art/'route-http-formal-v2-execution/final.json',{'journal_state':'RESTORED','recovery_returncode':0})
        old=root/'old.json';write(old,dict(qualification={'path':'CPU','sha256':'a'*64},http={}))
        here=root/'code';here.mkdir();(here/'controller.py').write_text('# actual fixture controller\n')
        (here/'execute.py').write_text('# actual fixture driver\n')
        owner=NS(SERVING_AUDIT=audit)
        owner.plan_template=lambda q,c,h,serving:dict(qualification=q,cold_restart=c,http=h,serving=serving,execution={})
        def cpu(args):
            self.actual_args=args
            p=Path(args[args.index('--plan')+1]);digest=args[args.index('--plan-sha256')+1];out=Path(args[args.index('--out')+1])
            self.assertEqual(digest,m.sha(p))
            write(out/'prepared.json',dict(status='PREPARED',owner_plan_sha256=digest,capture_specs=[{'concurrency':1},{'concurrency':8}]))
            return 0
        owner.main=cpu
        controller=NS(predecessor=lambda _:({'state':'RESTORED'},[]),CONTRACT='contract',PLAN_SHA='b'*64)
        return dict(HERE=here,ROOT=root,ART=art,REG=reg,OLD=old,ARCHIVE=archive,JOURNAL=art/'journal',OWNER=art/'owner',CPU=art/'cpu',EXECUTION=art/'execution',PY=Path(sys.executable)),owner,controller

    def test_default_is_inert(self):
        with patch.object(m,'prepare',side_effect=AssertionError('no preparation')),patch.object(m,'seal_prepared',side_effect=AssertionError('no seal')),contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(m.main([]),0)

    def test_actual_builder_saved_command_has_plan_digest_not_source_digest(self):
        with tempfile.TemporaryDirectory() as t,contextlib.ExitStack() as stack:
            values,o,c=self.fixture(Path(t).resolve())
            for k,v in values.items():stack.enter_context(patch.object(m,k,v))
            stack.enter_context(patch.object(m,'module',side_effect=lambda n:{'owner':o,'controller':c}[n]))
            stack.enter_context(patch.object(m,'collect_sources',return_value={str(values['HERE']/'execute.py'):m.sha(values['HERE']/'execute.py')}))
            self.assertEqual(m.main(['--run']),0)
            commands=m.read(values['REG']/'commands.json');digest=m.sha(values['REG']/'owner-plan.json')
            self.assertEqual(commands['cpu_prepare'][4:],self.actual_args)
            self.assertEqual(commands['readonly_service_prepare'][commands['readonly_service_prepare'].index('--owner-plan-sha256')+1],digest)
            self.assertEqual(m.read(values['REG']/'registration-result.json')['owner_plan_sha256'],digest)
            self.assertFalse((values['REG']/'receipt.json').exists())
            self.assertFalse(values['JOURNAL'].exists())

    def test_failed_cpu_preparation_keeps_attempt_and_never_seals(self):
        with tempfile.TemporaryDirectory() as t,contextlib.ExitStack() as stack:
            values,o,c=self.fixture(Path(t).resolve());o.main=lambda _:1
            for k,v in values.items():stack.enter_context(patch.object(m,k,v))
            stack.enter_context(patch.object(m,'module',side_effect=lambda n:{'owner':o,'controller':c}[n]))
            stack.enter_context(patch.object(m,'collect_sources',return_value={}))
            self.assertEqual(m.main(['--run']),1)
            self.assertEqual(m.read(values['REG']/'registration-result.json')['status'],'FAIL')
            self.assertTrue((values['REG']/'owner-plan.json').exists())
            self.assertFalse((values['REG']/'receipt.json').exists())

if __name__=='__main__':unittest.main()
