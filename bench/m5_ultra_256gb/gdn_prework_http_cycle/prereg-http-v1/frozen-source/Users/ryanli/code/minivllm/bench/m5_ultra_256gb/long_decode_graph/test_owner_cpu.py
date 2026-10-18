"""CPU-only E024 preparation routing; inherited owner lifecycle tested separately."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('e024_owner_test', HERE / 'owner.py')
owner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(owner)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class AdapterTests(unittest.TestCase):
    def fixture(self, root):
        runtime = root / 'runtime.bin'
        runtime.write_bytes(b'CPU FAKE RUNTIME; NOT IMPORTED')
        env = {f'MLX_TEST_{i}': str(i) for i in range(4)}
        inherited = {'schema': 'E023-native-qualification-plan-v1', 'execution': {
            'python': sys.executable, 'qualifier': {'path': str(owner.LEGACY.with_name('qualify_native.py')), 'sha256': owner.SUPPORT_SHA},
            'environment': env, 'pins': {str(runtime): sha(runtime)}}}
        inherited_path = root / 'e023.json'
        owner.write(root, inherited_path.name, inherited)
        inherited_pin = {'path': str(inherited_path), 'sha256': sha(inherited_path)}
        qualification = {'schema': 'E024-diagnostic-qualification-v1', 'qualifier_sha256': owner.QUALIFIER_SHA,
            'e023_plan': inherited_pin, 'e023_qualifier': inherited['execution']['qualifier'],
            'candidate': {'path': str(HERE / 'candidate.py'), 'sha256': owner.CANDIDATE_SHA}, 'deadline_s': 1800}
        qualification_path = root / 'qualification.json'
        owner.write(root, qualification_path.name, qualification)
        paths = [owner.LEGACY, owner.LEGACY.with_name('qualify_native.py')] + [HERE / n for n in
            ('owner.py', 'qualify.py', 'candidate.py', 'model-source-pins.json', 'test_owner_cpu.py', 'test_cpu.py', 'test_qualify_cpu.py')]
        plan = {'schema': 'E024-diagnostic-owner-v1', 'qualification_plan': {'path': str(qualification_path), 'sha256': sha(qualification_path)},
            'inherited_plan': inherited_pin, 'execution': {**inherited['execution'],
            'qualifier': {'path': str(HERE / 'qualify.py'), 'sha256': owner.QUALIFIER_SHA},
            'pins': {**inherited['execution']['pins'], **{str(p): sha(p) for p in paths}}}}
        return plan, inherited, qualification

    def args(self, root, plan):
        owner.write(root, 'owner-plan.json', plan)
        out = root / 'new-output'
        out.mkdir()
        return NS(plan=root / 'owner-plan.json', plan_sha256=sha(root / 'owner-plan.json'), out=out,
                  deadline_seconds=1830, suspended_pid=123)

    def legacy_prepare(self, inherited, seen):
        def prepare(args, check):
            check()
            actual = json.loads(args.plan.read_text())
            self.assertEqual(actual, inherited)
            self.assertEqual(sha(args.plan), args.plan_sha256)
            seen.append(args)
            owner.write(args.out, 'plan.json', actual)
            owner.write(args.out, 'verified-pins.json', {p: {'sha256': s, 'bytes': Path(p).stat().st_size,
                 'source_copied': False} for p, s in actual['execution']['pins'].items()})
            return ['THIS E023 COMMAND MUST NOT EXECUTE'], {**actual['execution']['environment'], 'CLEAN_MARKER': 'yes'}
        return prepare

    def test_fixed_qualifier_cli_and_true_inherited_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, seen, checks = Path(tmp), [], []
            plan, inherited, qualification = self.fixture(root)
            args = self.args(root, plan)
            command, env = owner.prepare(args, lambda: checks.append(1), self.legacy_prepare(inherited, seen))
            self.assertEqual(command, [sys.executable, '-I', '-B', str(HERE / 'qualify.py'), '--plan',
                plan['qualification_plan']['path'], '--plan-sha256', plan['qualification_plan']['sha256'],
                '--out', str(args.out / 'qualification')])
            self.assertEqual(env['CLEAN_MARKER'], 'yes')
            self.assertEqual(len(seen), 1)
            self.assertGreater(len(checks), 9)
            self.assertEqual(json.loads((args.out / 'e023-input-preflight/plan.json').read_text()), inherited)
            self.assertEqual(json.loads((args.out / 'plan.json').read_text()), plan)
            self.assertEqual(json.loads((args.out / 'qualification-plan.json').read_text()), qualification)
            self.assertEqual(set(qualification), {'schema', 'qualifier_sha256', 'e023_plan', 'e023_qualifier', 'candidate', 'deadline_s'})
            verified = json.loads((args.out / 'verified-pins.json').read_text())
            self.assertIn('inherited_evidence', verified[str(root / 'runtime.bin')])
            self.assertFalse(any(p.name == 'runtime.bin' for p in (args.out / 'frozen-source').rglob('*')))

    def test_environment_python_or_runtime_swap_rejected_before_legacy(self):
        for change in ('environment', 'python', 'runtime'):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as tmp:
                root, seen = Path(tmp), []
                plan, inherited, _ = self.fixture(root)
                if change == 'environment':
                    plan['execution']['environment'] = {'MLX_NEW': '1'}
                elif change == 'python':
                    plan['execution']['python'] = '/other/python'
                else:
                    plan['execution']['pins'][str(root / 'runtime.bin')] = '0' * 64
                args = self.args(root, plan)
                with self.assertRaises(ValueError):
                    owner.prepare(args, lambda: None, self.legacy_prepare(inherited, seen))
                self.assertEqual(seen, [])

    def test_wrong_qualification_binding_and_extra_fields_rejected(self):
        for field, value in (('e023_plan', {'path': '/other', 'sha256': '0' * 64}), ('execution', {})):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                root, seen = Path(tmp), []
                plan, inherited, qualification = self.fixture(root)
                qualification[field] = value
                owner.write(root, 'qualification.json', qualification)
                plan['qualification_plan']['sha256'] = sha(root / 'qualification.json')
                with self.assertRaises(ValueError):
                    owner.prepare(self.args(root, plan), lambda: None, self.legacy_prepare(inherited, seen))
                self.assertEqual(seen, [])

    def test_missing_new_test_pin_and_changed_source_hash_rejected(self):
        for mode in ('missing', 'wrong-sha'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                root, seen = Path(tmp), []
                plan, inherited, _ = self.fixture(root)
                key = str(HERE / 'test_owner_cpu.py')
                if mode == 'missing':
                    del plan['execution']['pins'][key]
                else:
                    plan['execution']['pins'][key] = '0' * 64
                with self.assertRaises(ValueError):
                    owner.prepare(self.args(root, plan), lambda: None, self.legacy_prepare(inherited, seen))

    def test_deadline_reaches_each_pin_chunk_before_delegating(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, seen = Path(tmp), []
            plan, inherited, _ = self.fixture(root)
            def expired():
                raise TimeoutError('injected absolute deadline')
            with self.assertRaisesRegex(TimeoutError, 'absolute deadline'):
                owner.prepare(self.args(root, plan), expired, self.legacy_prepare(inherited, seen))
            self.assertEqual(seen, [])

    def test_inherited_prepare_exception_propagates_without_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan, _, _ = self.fixture(root)
            def failed(*args):
                raise RuntimeError('inherited runtime pin rejected')
            with self.assertRaisesRegex(RuntimeError, 'runtime pin rejected'):
                owner.prepare(self.args(root, plan), lambda: None, failed)

    def test_legacy_sha_validated_before_import(self):
        self.assertTrue(callable(owner.load_legacy().run))
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / 'bad.py'
            bad.write_text('raise AssertionError("must not import")')
            with patch.object(owner, 'LEGACY', bad), self.assertRaisesRegex(ValueError, 'owner SHA'):
                owner.load_legacy()

    def test_main_reuses_legacy_run_with_prepare_adapter(self):
        sentinel, legacy = object(), NS(prepare=lambda *args: sentinel)
        legacy.run = lambda args: 7 if callable(legacy.prepare) and args.deadline_seconds == 1830 else 99
        with patch.object(owner, 'load_legacy', return_value=legacy):
            self.assertEqual(owner.main(['--plan', '/unused', '--plan-sha256', '0' * 64,
                '--out', '/unused-new', '--suspended-pid', '123', '--deadline-seconds', '1830']), 7)

    def test_help_has_no_import_or_filesystem_side_effects(self):
        code = "import runpy,sys;sys.argv=[sys.argv[1],'--help'];\ntry: runpy.run_path(sys.argv[0],run_name='__main__')\nexcept SystemExit as e: assert e.code==0\nassert not any(n=='mlx' or n.startswith('mlx.') or n in ('resources_v2','e024_fixed_e023_owner') for n in sys.modules)"
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run([sys.executable, '-I', '-B', '-c', code, str(HERE / 'owner.py')], cwd=tmp, capture_output=True, text=True, timeout=5)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(list(Path(tmp).iterdir()), [])


if __name__ == '__main__':
    unittest.main()
