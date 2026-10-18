"""Actual registrar writes with only expensive owner/controller validation faked.

Temporary source/spec/receipt files exercise the saved commands and final seal;
no process, service, backend, or model access is used. This specifically rejects
the v1 bug where a source-pin iteration overwrote the owner's plan SHA.
"""
from contextlib import ExitStack
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SOURCE=Path(__file__).with_name('preregister.py').resolve()
PHASES=('r1-c1-off','r1-c1-on','r1-c8-on','r1-c8-off',
        'r2-c8-off','r2-c8-on','r2-c1-on','r2-c1-off',
        'r3-c1-off','r3-c1-on','r3-c8-on','r3-c8-off')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_bytes())
def write(p,value):Path(p).write_text(json.dumps(value,indent=2)+'\n')


class RegistrarPlanDigestTests(unittest.TestCase):
    def setUp(self):
        self.stack=ExitStack();self.addCleanup(self.stack.close)
        self.root=Path(self.stack.enter_context(tempfile.TemporaryDirectory())).resolve()
        spec=importlib.util.spec_from_file_location('_route_formal_regression',SOURCE)
        self.reg=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.reg)
        self.here=self.root/'source';self.art=self.root/'artifacts';self.pre=self.root/'prepared-specs'
        for p in (self.here,self.art,self.pre):p.mkdir()
        self.p=self.here/'prereg-formal-v2';(self.pre/'specs').mkdir()
        for key,value in dict(HERE=self.here,ART=self.art,PREPARED=self.pre,P=self.p,NAME='trial-v2').items():
            self.stack.enter_context(patch.object(self.reg,key,value))
        names=('owner.py','controller.py','execute.py','test_owner_cpu.py','test_controller_cpu.py',
               'test_audit_restore_cpu.py','test_audit_formal_cpu.py','test_execute_cpu.py','test_preregister_cpu.py')
        for i,name in enumerate(names):(self.here/name).write_text('# distinct synthetic source '+str(i)+'\n')
        q=self.root/'numeric.json';write(q,dict(execution=dict(python='/not/executed/python')))
        self.q=dict(path=str(q),sha256=sha(q))
        write(self.pre/'result.json',dict(status='PASS',prepared_specs=12,source_unchanged=True,
            inputs_unchanged=True,device_run=False,execution_registered=False))
        for phase in PHASES:
            C=int(phase.split('-')[1][1:])
            write(self.pre/'specs'/(phase+'.json'),dict(phase=phase,concurrency=C,base_plan=self.q,
                transport={'port':18023},deadlines={'overall':3300},environment={},
                baseline_qualification={'fixture':'baseline'+str(C)},route_qualification={'fixture':'route'+str(C)}))
        journal=self.root/'prior.json';audit=self.root/'prior-audit.json'
        write(journal,{'state':'RESTORED'});write(audit,{'status':'PASS'})
        self.predecessor=dict(path=str(journal),sha256=sha(journal),audit=dict(path=str(audit),sha256=sha(audit)))
        self.prior=dict(state='RESTORED',replacement={'pid':123,'fixture':True})
        self.owner=SimpleNamespace(PHASES=PHASES,plan_template=self.plan_template,validate_plan=self.validate)
        self.controller=SimpleNamespace(CONTRACT='synthetic-contract',PLAN_SHA='a'*64,JOURNAL_SCHEMA='synthetic-journal',
            predecessor=lambda x:(self.prior,()),owner_plan=self.controller_plan,
            source_pins=lambda p:{str(self.here/'owner.py'):sha(self.here/'owner.py'),
                                  str(self.here/'controller.py'):sha(self.here/'controller.py')})
        self.stack.enter_context(patch.object(self.reg,'modules',return_value=(self.owner,self.controller)))

    def plan_template(self,q,cold,http,baseline,route,specs):
        return dict(schema='synthetic-expensive-owner-validation-is-faked',qualification=q,cold_restart=cold,
            execution=dict(owner=dict(path=str(self.here/'owner.py'),sha256=sha(self.here/'owner.py'))),
            prepared_specs=specs,trials=list(PHASES),deadline_seconds=40524)

    def validate(self,args):
        self.assertEqual(args.plan_sha256,sha(args.plan))

    def controller_plan(self,path,digest,journal):
        self.assertEqual(digest,sha(path))
        self.assertEqual(str(journal),read(path)['cold_restart']['journal_path'])

    def build(self):
        result=self.reg.prepare(self.predecessor)
        digest=sha(self.p/'owner-plan.json');prepared=read(self.p/'preparation.json')
        self.assertEqual(result['owner_plan_sha256'],digest)
        self.assertEqual(prepared['owner_plan_sha256'],digest)
        self.assertNotEqual(digest,list(prepared['source_pins'].values())[-1],
                            'regression fixture must distinguish final source hash from plan hash')
        commands=read(self.p/'commands.json')
        for name,flag in (('cpu_prepare','--plan-sha256'),('owner','--plan-sha256'),
                          ('readonly_service_prepare','--owner-plan-sha256'),
                          ('independent_formal_audit','--plan-sha256')):
            command=commands[name]
            self.assertEqual(command[command.index(flag)+1],digest,name)
        return digest,prepared

    def write_prepared(self,digest):
        plan=read(self.p/'owner-plan.json');j=self.art/'trial-v2-cold-journal';j.mkdir()
        write(j/'journal.json',dict(schema=self.controller.JOURNAL_SCHEMA,state='PREPARED',original=self.prior['replacement'],
            owner_plan=dict(path=str(self.p/'owner-plan.json'),sha256=digest),owner_source=plan['execution']['owner'],
            controller_sha256=sha(self.here/'controller.py'),predecessor=self.predecessor,trials=list(PHASES),
            owner_directory=str(self.art/'trial-v2-owner'),plan_sha256=self.controller.PLAN_SHA))
        out=self.art/'trial-v2-cpu-prepare';out.mkdir()
        write(out/'prepared.json',dict(status='PREPARED',owner_plan_sha256=digest))

    def test_real_written_commands_preparation_and_seal_keep_plan_sha(self):
        digest,prepared=self.build();self.write_prepared(digest)
        result=self.reg.seal_receipt()
        self.assertEqual(result['owner_plan_sha256'],digest)
        self.assertEqual(read(self.p/'receipt.json')['owner_plan_sha256'],digest)
        self.assertEqual(result['source_pins'][str(self.p/'owner-plan.json')],digest)
        self.assertEqual(result['prepared_journal_sha256'],sha(self.art/'trial-v2-cold-journal/journal.json'))

    def test_wrong_claimed_plan_digest_is_rejected_before_receipt(self):
        digest,prepared=self.build();self.write_prepared(digest)
        prepared['owner_plan_sha256']=list(prepared['source_pins'].values())[-1]
        write(self.p/'preparation.json',prepared)
        with self.assertRaises(AssertionError):self.reg.seal_receipt()
        self.assertFalse((self.p/'receipt.json').exists())


if __name__=='__main__':unittest.main()
