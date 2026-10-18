"""Prepared CPU fake driver failures; no subprocess/service/device is executed."""
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
sp=importlib.util.spec_from_file_location('glm_driver_cpu',HERE/'execute.py');d=importlib.util.module_from_spec(sp);sp.loader.exec_module(d)
def write(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value)+'\n')


class Tests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name).resolve()
        self.out=self.root/'execution';self.owner=self.root/'owner';self.journal=self.root/'journal';self.journal.mkdir()
        write(self.journal/'journal.json',{'state':'PREPARED'})
        self.commands=dict(controller_stop=['CPU-python','controller','stop'],
            owner=['CPU-python','owner','--run','--plan','CPU-plan','--plan-sha256','a'*64,'--out',str(self.owner)],
            controller_recover=['CPU-python','controller','recover','--watchdog-seconds','3600'])
        self.context=dict(execution=self.out,owner=self.owner,journal=self.journal,commands=self.commands,
            prepared_raw=(self.journal/'journal.json').read_bytes(),receipt_raw=b'{"CPU-fixture":true}',owner_plan_sha256='a'*64)
        self.calls=[];self.owner_rc=0;self.owner_error=None;self.stop_rc=0;self.stop_error=None;self.stop_state='STOPPED'
        self.create_owner=True;self.recovery_error=None

    def fake(self,folder,phase,argv):
        self.assertEqual(folder,self.out);self.calls.append((phase,list(argv)))
        if phase=='stop':
            self.assertEqual(argv,self.commands['controller_stop']);write(self.journal/'journal.json',{'state':self.stop_state})
            if self.stop_error:raise self.stop_error
            return self.stop_rc
        if phase=='glm-owner':
            self.assertEqual(argv,self.commands['owner']);self.assertIn('--run',argv)
            self.assertEqual(d.read(self.out/'owner-invocation.json')['returned'],False)
            if self.create_owner:self.owner.mkdir()
            if self.owner_error:raise self.owner_error
            return self.owner_rc
        self.assertEqual(phase,'recover');self.assertIn(argv,[self.commands['controller_recover'],self.commands['controller_recover']+['--no-experiment']])
        if self.recovery_error:raise self.recovery_error
        write(self.journal/'journal.json',{'state':'RESTORED'});return 0

    def test_default_and_partial_arguments_inert(self):
        with patch.object(d,'preflight',side_effect=AssertionError('no preparation')),contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(d.main([]),0)
            with self.assertRaises(ValueError):d.main(['--registration',str(self.root)])
            with self.assertRaises(ValueError):d.main(['--run'])

    def test_direct_owner_uses_run_and_normal_finally_restore(self):
        self.assertEqual(d.execute(self.context,self.fake),0)
        self.assertEqual([x[0] for x in self.calls],['stop','glm-owner','recover'])
        self.assertNotIn('--no-experiment',self.calls[-1][1])
        final=d.read(self.out/'final.json');self.assertEqual(final['journal_state'],'RESTORED');self.assertFalse(final['observed_diagnostic_complete'])

    def test_failed_waited_owner_still_restores(self):
        self.owner_rc=1;self.assertEqual(d.execute(self.context,self.fake),1)
        self.assertEqual(self.calls[-1][0],'recover');self.assertEqual(d.read(self.out/'final.json')['owner_returncode'],1)

    def test_waited_owner_no_directory_can_prove_no_experiment(self):
        self.owner_rc=1;self.create_owner=False;self.assertEqual(d.execute(self.context,self.fake),1)
        self.assertIn('--no-experiment',self.calls[-1][1])

    def test_unknown_owner_outcome_cannot_claim_no_experiment(self):
        self.create_owner=False;self.owner_error=RuntimeError('unknown Popen')
        with self.assertRaisesRegex(RuntimeError,'unknown Popen'):d.execute(self.context,self.fake)
        self.assertNotIn('--no-experiment',self.calls[-1][1]);self.assertFalse(d.read(self.out/'final.json')['owner_invocation_returned'])

    def test_stop_failure_after_mutation_still_recovers(self):
        self.stop_state='STOP_INTENT';self.stop_error=RuntimeError('unknown bootout')
        with self.assertRaisesRegex(RuntimeError,'bootout'):d.execute(self.context,self.fake)
        self.assertEqual([x[0] for x in self.calls],['stop','recover']);self.assertIn('--no-experiment',self.calls[-1][1])

    def test_unmutated_prepared_failure_does_not_load(self):
        self.stop_state='PREPARED';self.stop_rc=1;self.assertEqual(d.execute(self.context,self.fake),1)
        self.assertEqual([x[0] for x in self.calls],['stop'])

    def test_recovery_failure_is_retained_not_success(self):
        self.recovery_error=RuntimeError('recovery incomplete')
        with self.assertRaisesRegex(RuntimeError,'recovery incomplete'):d.execute(self.context,self.fake)
        self.assertTrue((self.out/'recovery-error.json').is_file());self.assertEqual(d.read(self.out/'final.json')['journal_state'],'STOPPED')

    def prepared(self):
        reg=d.registrar();folder=self.root/'registration';folder.mkdir();python=self.root/'venv/bin/python';python.parent.mkdir(parents=True);python.write_text('CPU entry')
        plan=dict(execution={'owner':dict(path=str(HERE/'owner.py'),sha256=d.sha(HERE/'owner.py'))},
            outputs=dict(owner=str(self.owner)),cold_restart=dict(journal_path=str(self.journal),
            controller=dict(path=str(HERE/'controller.py'),sha256=d.sha(HERE/'controller.py')),
            proposal_plan_sha256='a'*64,approved_contract='cold-restart-persisted-visible-v1'))
        path=folder/'owner-plan.json';write(path,plan);ph=d.sha(path);analysis=self.root/'analysis.json'
        commands=reg.build_commands(path,ph,plan,analysis,str(python));write(folder/'commands.json',commands)
        write(folder/'plan-build.json',{'outputs':{'analysis':str(analysis)}})
        journal=dict(schema='GLM-cold-load-journal-v1',state='PREPARED',owner_directory=str(self.owner),owner_plan=dict(path=str(path),sha256=ph),
            owner_source=plan['execution']['owner'],trials=['glm-official-load'],controller_sha256=d.sha(HERE/'controller.py'))
        write(self.journal/'journal.json',journal)
        pins={str(p):d.sha(p) for p in (HERE/'execute.py',HERE/'preregister.py',HERE/'controller.py',HERE/'owner.py',path,folder/'commands.json',python)}
        receipt=dict(schema='GLM-cold-load-prepared-receipt-v1',experiment_started=False,source_pins=pins,
            owner_plan_sha256=ph,prepared_journal_sha256=d.sha(self.journal/'journal.json'),outer_python=str(python))
        write(folder/'receipt.json',receipt);return folder,receipt

    def test_preflight_binds_saved_run_argv_and_actual_prepared_bytes(self):
        folder,receipt=self.prepared();value=d.preflight(folder,self.out);self.assertIn('--run',value['commands']['owner'])
        journal=d.read(self.journal/'journal.json');journal['state']='STOPPED';write(self.journal/'journal.json',journal)
        with self.assertRaisesRegex(ValueError,'actual unchanged PREPARED'):d.preflight(folder,self.out)

    def test_corrupt_saved_owner_command_cannot_hide_behind_rehashed_file(self):
        folder,receipt=self.prepared();commands=d.read(folder/'commands.json');commands['owner'].remove('--run');write(folder/'commands.json',commands)
        receipt['source_pins'][str(folder/'commands.json')]=d.sha(folder/'commands.json');write(folder/'receipt.json',receipt)
        with self.assertRaisesRegex(ValueError,'saved exact GLM commands'):d.preflight(folder,self.out)


if __name__=='__main__':unittest.main()
