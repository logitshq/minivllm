"""Narrow E028 adapter tests; existing owner fakes, never real process/device IO."""
import builtins
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


m = load(HERE / 'model_owner.py', 'e028_model_owner_tested')
fakes = load(HERE.parent / 'pareto_realistic/test_qualify_absent_owner_cpu.py', 'e028_existing_owner_fakes')


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2) + '\n')


class OwnerAdapterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name).resolve()
        self.out = self.directory / 'owner'
        self.journal = self.directory / 'journal'
        self.journal.mkdir()
        (self.journal / '.lock').touch()
        self.original = HERE.parents[2] / '.artifacts/m5-ultra-256gb/realistic-qualification-v1-inputs/plan.json'
        self.child_path = self.directory / 'e028.json'
        self.child = {'schema': 'E028-model-qualification-plan-v1',
            'qualification': {'path': str(self.original), 'sha256': m.ORIGINAL_PLAN_SHA},
            'sources': {**{str(p): h for p, h in m.FIXED.items()},
                        str(HERE / 'model_qualification.py'): m.sha(HERE / 'model_qualification.py')}}
        write(self.child_path, self.child)
        self.plan_path = self.directory / 'wrapper.json'
        base = m.load_owner()
        self.wrapper = {'schema': 'E023-service-absent-owner-plan-v1',
            'execution': {'owner': {'path': str(HERE / 'model_owner.py'), 'sha256': m.sha(HERE / 'model_owner.py')}},
            'qualification': {'path': str(self.child_path), 'sha256': m.sha(self.child_path)},
            'cold_restart': {'journal_path': str(self.journal),
                'controller': {'path': str(m.CONTROLLER), 'sha256': m.CONTROLLER_SHA},
                'approved_contract': base.CONTRACT, 'proposal_plan_sha256': base.PROPOSAL_SHA},
            'deadline_seconds': 1830}
        self.commit()

    def commit(self):
        write(self.plan_path, self.wrapper)
        self.args = NS(plan=self.plan_path, plan_sha256=m.sha(self.plan_path), out=self.out)

    def child_args(self):
        return NS(plan=self.child_path, plan_sha256=m.sha(self.child_path), out=self.out, deadline_seconds=1830)

    def test_exact_plan_and_private_base_locations(self):
        base = m.configured(self.args)
        wrapper, raw, numeric = base.validate_plan(self.args)
        self.assertEqual(wrapper, self.wrapper)
        self.assertEqual(base.sha_bytes(raw), self.args.plan_sha256)
        self.assertEqual(numeric, json.loads(self.original.read_text()))
        self.assertEqual(base.NUMERICAL_SHA, m.sha(self.child_path))
        self.assertEqual(Path(base.__file__), m.BASE)
        self.assertEqual(base.HERE, m.BASE.parent)
        for fn in (base.run, base.stopped, base.absence, base.load_resources):
            self.assertEqual(Path(fn.__code__.co_filename), m.BASE)

    def test_prepare_keeps_real_inherited_plan_then_new_command_and_sources(self):
        self.out.mkdir()
        base = m.load_owner()
        calls = []
        env = {'MLX_FIXED': 'same sanitized object'}
        def inherited(args, check):
            calls.append(args)
            self.assertEqual(args.plan, self.original)
            self.assertEqual(args.plan_sha256, m.ORIGINAL_PLAN_SHA)
            self.assertEqual(args.out, self.out / 'e023-input-preflight')
            self.assertTrue(args.out.is_dir())
            check()
            write(args.out / 'verified-pins.json', {'fake': 'CPU only; no runtime hash'})
            return ['OLD_QUALIFIER_MUST_NOT_EXECUTE'], env
        checks = []
        command, actual_env = m.prepare(self.child_args(), lambda: checks.append(True), base, inherited)
        self.assertEqual(len(calls), 1)
        self.assertIs(actual_env, env)
        numeric = json.loads(self.original.read_text())
        self.assertEqual(command, [numeric['execution']['python'], '-I', '-B', str(HERE / 'model_qualification.py'),
            '--plan', str(self.child_path), '--plan-sha256', m.sha(self.child_path), '--out', str(self.out / 'qualification')])
        self.assertGreaterEqual(len(checks), 7)
        pins = json.loads((self.out / 'adapter-pins.json').read_text())
        self.assertEqual(set(pins), set(self.child['sources']) | {str(m.BASE)})
        for path, receipt in pins.items():
            self.assertEqual(m.sha(self.out / 'frozen-source' / path.lstrip('/')), receipt['sha256'])
        self.assertEqual((self.out / 'qualification-plan.json').read_bytes(), self.child_path.read_bytes())

    def test_bad_source_old_plan_deadline_and_owner_fail_before_launch(self):
        base = m.load_owner()
        for key, value in [('schema', 'wrong'), ('deadline_seconds', 1800), ('deadline_seconds', True)]:
            original = self.wrapper[key]
            self.wrapper[key] = value
            self.commit()
            with self.assertRaises(ValueError): m.validate_plan(self.args, base)
            self.wrapper[key] = original
        self.commit()
        self.child['qualification']['sha256'] = '0' * 64
        write(self.child_path, self.child)
        self.wrapper['qualification']['sha256'] = m.sha(self.child_path)
        self.commit()
        with self.assertRaisesRegex(ValueError, 'original E023'): m.validate_plan(self.args, base)
        self.child['qualification']['sha256'] = m.ORIGINAL_PLAN_SHA
        self.child['sources'][str(HERE / 'candidate.py')] = '0' * 64
        write(self.child_path, self.child)
        self.wrapper['qualification']['sha256'] = m.sha(self.child_path)
        self.commit()
        with self.assertRaisesRegex(ValueError, 'four E028'): m.validate_plan(self.args, base)
        self.assertFalse(self.out.exists())

    def setup_run(self):
        base = m.configured(self.args)
        self.j = {'schema': 'omlx-cold-restart-journal-v2', 'state': 'STOPPED', 'id': 'CPU-E028-journal',
            'removed_at': time.time() - 10, 'plan_sha256': base.PROPOSAL_SHA,
            'controller_sha256': m.CONTROLLER_SHA, 'approved_contract': base.CONTRACT,
            'original': base.ORIGINAL, 'owner_directory': str(self.out),
            'owner_plan': {'path': str(self.plan_path), 'sha256': self.args.plan_sha256},
            'owner_source': self.wrapper['execution']['owner'], 'events': []}
        write(self.journal / 'journal.json', self.j)
        system, resources = fakes.FakeSystem(self), fakes.Resources()
        base.load_resources = lambda numeric: resources
        base.load_prepare = lambda: lambda args, check: (['CPU_ONLY_E028_CHILD'], {})
        return base, system, resources

    def fake_run(self, base, system):
        # Accidental real backend calls fail immediately, even on error paths.
        with patch.object(base.subprocess, 'Popen', side_effect=AssertionError('real spawn')), \
             patch.object(base.subprocess, 'run', side_effect=AssertionError('real process probe')), \
             patch.object(base.signal, 'signal', side_effect=AssertionError('real signal')), \
             patch.object(base.signal, 'setitimer', side_effect=AssertionError('real alarm')):
            return base.run(self.args, system=system)

    def test_frozen_run_direct_child_reap_and_e028_plan_receipt(self):
        base, system, _ = self.setup_run()
        self.assertEqual(self.fake_run(base, system), 0)
        self.assertEqual(system.spawned, [['CPU_ONLY_E028_CHILD']])
        release = json.loads((self.out / 'release.json').read_text())
        self.assertTrue(release['waited_reaped'] and release['strict_absence'])
        result = json.loads((self.out / 'complete.json').read_text())
        self.assertEqual(result['numerical_plan_sha256'], m.sha(self.child_path))
        self.assertEqual(result['status'], 'PASS')

    def test_pressure_two_remains_no_child_and_controller_accepts_receipt(self):
        base, system, resources = self.setup_run()
        resources.pressure = 2
        self.assertEqual(self.fake_run(base, system), 1)
        terminal = json.loads((self.out / 'terminal.json').read_text())
        self.assertEqual(terminal['status'], 'NO_CHILD_STARTED')
        self.assertEqual(system.spawned, [])
        ctl = load(m.CONTROLLER, '_e028_controller_cpu_only')
        proof = ctl.reaped(NS(directory=self.journal, data=self.j), NS(process=lambda pid: None))
        self.assertTrue(proof['no_child_started'])

    def test_preparation_failure_preserves_terminal_without_fake_release(self):
        base, system, _ = self.setup_run()
        def fail(*args): raise ValueError('inherited source mismatch')
        base.load_prepare = lambda: fail
        self.assertEqual(self.fake_run(base, system), 1)
        self.assertEqual(json.loads((self.out / 'first-error.json').read_text())['phase'], 'prepare')
        self.assertEqual(json.loads((self.out / 'terminal.json').read_text())['status'], 'NO_CHILD_STARTED')
        self.assertFalse((self.out / 'release.json').exists())

    def test_help_import_does_not_load_backend(self):
        original = builtins.__import__
        def guarded(name, *args, **kwargs):
            if name.split('.')[0] in ('mlx', 'numpy', 'torch', 'tutorial_code'):
                raise AssertionError('device import')
            return original(name, *args, **kwargs)
        with patch('builtins.__import__', guarded), patch('sys.stdout', new=io.StringIO()):
            subject = load(HERE / 'model_owner.py', '_e028_help_only')
            with patch.object(subject, 'load_owner', side_effect=AssertionError('loaded owner')):
                with self.assertRaises(SystemExit) as caught: subject.main(['--help'])
                self.assertEqual(caught.exception.code, 0)
        self.assertFalse(any(x == 'mlx' or x.startswith('mlx.') for x in sys.modules))


if __name__ == '__main__':
    unittest.main()
