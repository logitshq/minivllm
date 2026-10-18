"""Three fake-only control-flow regressions; never execute a child or service."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SOURCE=Path(__file__).with_name('execute_combined_v2.py')

class DriverTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve();self.journal=self.root/'journal';self.journal.mkdir()
        self.owner=self.root/'owner';self.output=self.root/'execution'
        sp=importlib.util.spec_from_file_location('_combined_driver_v2_fake',SOURCE)
        self.m=importlib.util.module_from_spec(sp);sp.loader.exec_module(self.m)
        self.m.OUT=self.output;self.m.PYTHON=self.root/'venv/bin/python';self.m.HERE=self.root
        self.m.PLAN=self.root/'owner-plan.json'
        manifest=self.root/'lifecycle.json'
        manifest.write_text(json.dumps({'sources':{'controller':{'path':str(SOURCE),'sha256':self.hash(SOURCE)}}}))
        self.m.PLAN.write_text(json.dumps({'lifecycle':{'path':str(manifest),'sha256':self.hash(manifest)},
                                          'outputs':{'journal':str(self.journal),'out':str(self.owner)}}))
        self.m.PLAN_SHA=self.hash(self.m.PLAN)
        self.write_state('PREPARED');self.calls=[]

    def hash(self,p):return hashlib.sha256(p.read_bytes()).hexdigest()
    def write_state(self,state):
        (self.journal/'journal.json').write_text(json.dumps({'state':state,'owner_plan':{'path':str(self.m.PLAN),'sha256':self.m.PLAN_SHA}}))
    def run_driver(self,backend):
        def call(command,**kwargs):
            self.calls.append(command)
            return backend(command)
        with patch.object(self.m.sys,'argv',[str(SOURCE),'--run']),patch.object(self.m.subprocess,'run',side_effect=call):
            return self.m.main()
    def final(self):return json.loads((self.output/'final.json').read_bytes())

    def test_waited_preentry_owner_failure_allows_no_experiment(self):
        def backend(command):
            if 'stop' in command:self.write_state('STOPPED');return SimpleNamespace(returncode=0)
            if str(self.root/'owner.py') in command:
                self.assertFalse(self.owner.exists());return SimpleNamespace(returncode=2)
            self.assertIn('--no-experiment',command);self.write_state('RESTORED');return SimpleNamespace(returncode=0)
        self.assertEqual(self.run_driver(backend),1)
        self.assertEqual(self.final()['codes'],{'stop':0,'owner':2,'recover':0})
        self.assertIs(self.final()['owner_invocation_returned'],True)
        self.assertFalse(self.owner.exists())

    def test_unknown_owner_launch_does_not_invent_no_experiment(self):
        def backend(command):
            if 'stop' in command:self.write_state('STOPPED');return SimpleNamespace(returncode=0)
            if str(self.root/'owner.py') in command:raise OSError('unknown launch outcome')
            self.assertNotIn('--no-experiment',command);return SimpleNamespace(returncode=1)
        self.assertEqual(self.run_driver(backend),1)
        self.assertIs(self.final()['owner_invocation_attempted'],True)
        self.assertIs(self.final()['owner_invocation_returned'],False)
        self.assertEqual(self.final()['journal_state'],'STOPPED')
        self.assertEqual(self.final()['errors'][0]['type'],'OSError')

    def test_stop_exception_recovers_and_zero_codes_need_restored(self):
        def backend(command):
            if 'stop' in command:self.write_state('STOPPED');raise OSError('stop wait error')
            self.assertIn('recover',command);self.assertIn('--no-experiment',command)
            self.write_state('RESTORED');return SimpleNamespace(returncode=0)
        self.assertEqual(self.run_driver(backend),1)
        self.assertIs(self.final()['owner_invocation_attempted'],False)
        self.assertEqual(self.final()['journal_state'],'RESTORED')
        # A dishonest/incomplete zero-return controller cannot create driver PASS.
        self.m.OUT=self.root/'second-execution';self.output=self.m.OUT;self.write_state('PREPARED')
        def incomplete(command):
            self.write_state('LOAD_INTENT' if 'recover' in command else 'STOPPED')
            return SimpleNamespace(returncode=0)
        self.assertEqual(self.run_driver(incomplete),1)
        self.assertEqual(self.final()['codes'],{'stop':0,'owner':0,'recover':0})
        self.assertEqual(self.final()['journal_state'],'LOAD_INTENT')

if __name__=='__main__':unittest.main()
