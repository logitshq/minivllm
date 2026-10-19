"""Focused real argparse + fake lifecycle tests; no subprocess/service/backend."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent

def load(name):
    spec=importlib.util.spec_from_file_location('glm_v3_registration_cpu_'+name,HERE/(name+'.py'))
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
r=load('preregister');d=load('execute');o=load('owner');c=load('controller')
a=load('audit_restore');closed=load('audit_closed')
def write(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value)+'\n')


class Tests(unittest.TestCase):
    def setUp(self):
        t=tempfile.TemporaryDirectory();self.addCleanup(t.cleanup);self.root=Path(t.name).resolve()
        self.owner=self.root/'owner';self.journal=self.root/'journal';self.journal.mkdir()
        self.plan=dict(execution={'owner':dict(path=str(HERE/'owner.py'),sha256=r.OWNER_SHA)},
            outputs={'owner':str(self.owner)},cold_restart=dict(journal_path=str(self.journal),
                controller=dict(path=str(HERE/'controller.py'),sha256=r.sha(HERE/'controller.py')),
                proposal_plan_sha256=c.PLAN_SHA,approved_contract=c.CONTRACT))
        self.commands=r.build_commands(self.root/'plan.json','a'*64,self.plan,self.root/'diagnostic.json','CPU-python')

    def test_default_descriptions_never_run_and_incomplete_arguments_reject(self):
        for m,call in ((r,'prepare'),(d,'preflight'),(o,'run')):
            with self.subTest(module=m.__name__),patch.object(m,call,side_effect=AssertionError('operational entry')),contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(m.main([]),0)
                with self.assertRaises(ValueError):m.main(['--run'])

    def test_real_owner_argparse_accepts_actual_generated_run_command(self):
        seen=[]
        with patch.object(o,'run',side_effect=lambda args:seen.append(args) or 0):
            self.assertEqual(o.main(self.commands['owner'][4:]),0)
        self.assertEqual(len(seen),1);self.assertTrue(seen[0].run)
        self.assertEqual(seen[0].out,self.owner);self.assertEqual(seen[0].plan_sha256,'a'*64)
        missing=[x for x in self.commands['owner'][4:] if x!='--run']
        with patch.object(o,'run',side_effect=AssertionError('must not start')):
            with self.assertRaises(ValueError):o.main(missing)

    def test_real_controller_argparse_accepts_prepare_stop_recover_and_rejects_run(self):
        calls=[]
        fake_journal=SimpleNamespace(directory=self.journal,data={},close=lambda:None)
        with patch.object(c,'Journal',return_value=fake_journal),patch.object(c,'System',return_value=object()),\
             patch.object(c.signal,'getitimer',return_value=(0.0,0.0)),\
             patch.object(c.signal,'signal',return_value=None),patch.object(c.signal,'setitimer'),\
             patch.object(c,'prepare',side_effect=lambda *args:calls.append(('prepare',args))),\
             patch.object(c,'stop',side_effect=lambda *args:calls.append(('stop',args))),\
             patch.object(c,'recover',side_effect=lambda *args:calls.append(('recover',args))):
            for key in ('controller_prepare','controller_stop','controller_recover'):
                self.assertEqual(c.main(self.commands[key][4:]),0)
            with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit) as raised:
                c.main(self.commands['controller_prepare'][4:]+['--run'])
            self.assertEqual(raised.exception.code,2)
        self.assertEqual([x[0] for x in calls],['prepare','stop','recover'])
        self.assertEqual(calls[0][1][-1],'a'*64)
        self.assertIs(calls[-1][1][-1],False)

    def test_actual_audit_argparse_accepts_saved_commands_with_fake_audit_only(self):
        for key,m in (('restoration_audit',a),('closed_audit',closed)):
            with self.subTest(key=key),patch.object(m,'audit',return_value={'status':'PASS','scope':'CPU fake only'}) as audit:
                self.assertEqual(m.main(self.commands[key][4:]),0)
                self.assertEqual(audit.call_args.args[0],self.journal)
        self.assertNotEqual(self.commands['restoration_audit'][-1],self.commands['closed_audit'][-1])

    def test_actual_registration_rejects_unsealed_E033_and_old_E032_before_output(self):
        args=['--run']
        for name,value in dict(out=self.root/'reg',owner=self.owner,journal=self.root/'new-journal',
            analysis=self.root/'result.json',python=self.root/'python',probe_plan=self.root/'probe.json',
            verification=self.root/'inventory.json',predecessor_journal=self.root/'restored-journal.json',
            predecessor_audit=self.root/'restoration-audit.json',probe_plan_sha256='1'*64,
            verification_sha256='2'*64,predecessor_journal_sha256='3'*64,predecessor_audit_sha256='4'*64,
            child_limit_bytes=220*(1<<30)).items():
            args+=['--'+name.replace('_','-'),str(value)]
        # Explicit fake unsealed pin keeps this rejection test deterministic after sealing.
        with patch.object(c,'BOUND_MANIFEST_SHA','UNSEALED'),patch.object(r,'module',return_value=c):
            with self.assertRaisesRegex(c.Refuse,'not sealed'):r.main(args)
        self.assertFalse((self.root/'reg').exists())
        old=SimpleNamespace(PRIOR_ARCHIVE=r.REQUIRED_PREDECESSOR_ARCHIVE.with_name('E032-gdn-graph-combined-formal-v1'))
        with patch.object(r,'module',return_value=old):
            with self.assertRaisesRegex(ValueError,'deferred until the actual E033'):r.main(args)
        self.assertFalse((self.root/'reg').exists())

    def prepared(self):
        folder=self.root/'reg';folder.mkdir();python=self.root/'python';python.write_text('CPU interpreter identity')
        path=folder/'owner-plan.json';write(path,self.plan);ph=r.sha(path);analysis=self.root/'diagnostic.json'
        commands=r.build_commands(path,ph,self.plan,analysis,str(python));write(folder/'commands.json',commands)
        write(folder/'plan-build.json',{'outputs':{'analysis':str(analysis)}})
        data=dict(schema='GLM-cold-load-journal-v3',state='PREPARED',owner_directory=str(self.owner),
            owner_plan=dict(path=str(path),sha256=ph),owner_source=self.plan['execution']['owner'],
            trials=['glm-official-load'],controller_sha256=r.sha(HERE/'controller.py'))
        write(self.journal/'journal.json',data)
        pins={str(p):r.sha(p) for p in (HERE/'execute.py',HERE/'preregister.py',HERE/'controller.py',HERE/'owner.py',path,folder/'commands.json',python)}
        receipt=dict(schema='GLM-cold-load-prepared-receipt-v3',experiment_started=False,source_pins=pins,
            owner_plan_sha256=ph,prepared_journal_sha256=r.sha(self.journal/'journal.json'),outer_python=str(python))
        write(folder/'receipt.json',receipt);return folder,receipt

    def test_real_preflight_rejects_rehashed_cli_corruption(self):
        folder,receipt=self.prepared()
        with patch.object(d,'REGISTRAR_SHA',r.sha(HERE/'preregister.py')):
            actual=d.preflight(folder,self.root/'execute');self.assertIn('--run',actual['commands']['owner'])
            commands=r.read(folder/'commands.json');commands['owner'].remove('--run');write(folder/'commands.json',commands)
            receipt['source_pins'][str(folder/'commands.json')]=r.sha(folder/'commands.json');write(folder/'receipt.json',receipt)
            with self.assertRaisesRegex(ValueError,'saved exact GLM commands'):d.preflight(folder,self.root/'execute')

    def test_actual_driver_unknown_launch_recovers_without_no_experiment(self):
        folder,receipt=self.prepared()
        with patch.object(d,'REGISTRAR_SHA',r.sha(HERE/'preregister.py')):
            ctx=d.preflight(folder,self.root/'execute')
        seen=[]
        def command(out,name,argv):
            seen.append((name,list(argv)))
            if name=='stop':write(self.journal/'journal.json',{'state':'STOPPED'});return 0
            if name=='glm-owner':raise RuntimeError('CPU unknown launch')
            write(self.journal/'journal.json',{'state':'RESTORED'});return 0
        with self.assertRaisesRegex(RuntimeError,'unknown launch'):d.execute(ctx,command)
        self.assertEqual([x[0] for x in seen],['stop','glm-owner','recover'])
        self.assertNotIn('--no-experiment',seen[-1][1])
        result=d.read(ctx['execution']/'final.json');self.assertFalse(result['owner_invocation_returned'])
        self.assertEqual(result['recovery_returncode'],0)


if __name__=='__main__':unittest.main()
