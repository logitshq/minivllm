"""CPU fake-process checks: no sockets, resource commands, model or MLX imports."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

PATH = Path(__file__).with_name('qualify_owner.py')
spec = importlib.util.spec_from_file_location('owner_under_test', PATH)
owner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(owner)


class Child:
    pid = 987654321
    def __init__(self, clock, code=None, finish=True, ignore_term=False, terminate_error=False):
        self.clock, self.returncode, self.finish = clock, code, finish
        self.ignore_term, self.terminate_error = ignore_term, terminate_error
        self.terminated = self.killed = False
        self.waits = []
    def poll(self):
        return self.returncode
    def wait(self, timeout=None):
        self.waits.append(timeout)
        if self.returncode is None:
            if self.finish:
                self.returncode = 0
            else:
                self.clock[0] += timeout + 0.001
                raise subprocess.TimeoutExpired('fake-child', timeout)
        return self.returncode
    def terminate(self):
        self.terminated = True
        if self.terminate_error:
            raise OSError('injected terminate failure')
        if not self.ignore_term:
            self.returncode = -15
    def kill(self):
        self.killed, self.returncode = True, -9


class Resources:
    def __init__(self, fail_labels=()):
        self.fail_labels = fail_labels
    def snapshot(self, label, pid=None, suspended_pids=()):
        if label in self.fail_labels:
            raise RuntimeError('injected probe error: ' + label)
        return {'label': label, 'pressure_level_value': 1, 'power_metadata': {'source': 'AC Power'},
                'processinfo_thermal_metadata': {'code': 0}, 'native_inference_processes': [
                    {'pid': suspended_pids[0], 'authorized_suspended': True}]}
    def guard(self, samples):
        pass
    def deltas(self, samples):
        return {'samples': len(samples)}


class OwnerTests(unittest.TestCase):
    def exercise(self, *, code=None, finish=True, ignore_term=False, terminate_error=False,
                 fail_labels=(), absence=True, qualifier_pass=True):
        with tempfile.TemporaryDirectory() as tmp:
            out, clock = Path(tmp) / 'new-owner', [0.0]
            args = SimpleNamespace(out=out, deadline_seconds=1, suspended_pid=123)
            child = Child(clock, code, finish, ignore_term, terminate_error)
            def prepare(args, check):
                (args.out / 'qualification').mkdir()
                owner.write(args.out / 'qualification', 'complete.json', {'status': 'PASS' if qualifier_pass else 'FAIL'})
                return ['fixed-fake-qualifier'], {}
            with patch.object(owner, 'prepare', side_effect=prepare), patch.object(owner.time, 'monotonic', side_effect=lambda: clock[0]), \
                 patch.object(owner.subprocess, 'Popen', return_value=child) as launch, \
                 patch.object(owner.subprocess, 'run', return_value=SimpleNamespace(returncode=1 if absence else 0, stdout='' if absence else str(child.pid))):
                result = owner.run(args, Resources(fail_labels))
            data = {p.name: json.loads(p.read_text()) for p in out.glob('*.json')}
            return result, child, data, launch.call_count

    def test_success_requires_child_pass_reap_absence(self):
        result, child, data, launches = self.exercise()
        self.assertEqual((result, launches, data['complete.json']['status']), (0, 1, 'PASS'))
        self.assertTrue(data['release.json']['waited_reaped'])
        self.assertTrue(data['release.json']['strict_absence'])
        self.assertFalse(child.terminated)

    def test_timeout_terminates_kills_waits_and_preserves_first(self):
        result, child, data, _ = self.exercise(finish=False, ignore_term=True)
        self.assertEqual(result, 1)
        self.assertTrue(child.terminated and child.killed)
        self.assertIn('absolute deadline', data['first-error.json']['traceback'])
        self.assertEqual(data['first-error.json']['phase'], 'execution')
        self.assertEqual(data['release.json']['returncode'], -9)
        self.assertIn(15, child.waits)

    def test_probe_exception_still_reaps(self):
        result, child, data, _ = self.exercise(fail_labels=('child-running',))
        self.assertEqual(result, 1)
        self.assertTrue(child.terminated)
        self.assertTrue(data['release.json']['strict_absence'])
        self.assertIn('probe error', data['first-error.json']['traceback'])
        self.assertEqual(data['complete.json']['samples'], 2)  # preflight and postflight

    def test_preflight_failure_launches_nothing(self):
        result, _, data, launches = self.exercise(fail_labels=('preflight',))
        self.assertEqual((result, launches), (1, 0))
        self.assertNotIn('release.json', data)

    def test_child_failure_survives_cleanup_and_postflight_errors(self):
        result, _, data, _ = self.exercise(code=7, absence=False, fail_labels=('postflight',))
        self.assertEqual(result, 1)
        self.assertIn('qualifier exit 7', data['first-error.json']['traceback'])
        self.assertEqual([x['phase'] for x in data['errors.json']], ['execution', 'cleanup', 'postflight'])
        self.assertNotIn('release.json', data)

    def test_terminate_exception_falls_back_to_kill(self):
        result, child, data, _ = self.exercise(fail_labels=('child-running',), terminate_error=True)
        self.assertEqual(result, 1)
        self.assertTrue(child.killed)
        self.assertTrue(data['release.json']['strict_absence'])
        self.assertEqual([x['phase'] for x in data['errors.json']], ['execution', 'terminate/wait'])

    def test_exit_zero_without_numerical_pass_is_failure(self):
        result, _, data, _ = self.exercise(qualifier_pass=False)
        self.assertEqual(result, 1)
        self.assertIn('did not record PASS', data['first-error.json']['traceback'])

    def test_help_imports_no_mlx_or_resource_and_creates_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            program = "import runpy,sys; sys.argv=[sys.argv[1],'--help'];\ntry: runpy.run_path(sys.argv[0],run_name='__main__')\nexcept SystemExit as e: assert e.code==0\nassert not any(n=='mlx' or n.startswith('mlx.') or n in ('resources_v2','resource_base') for n in sys.modules)"
            result = subprocess.run([sys.executable, '-I', '-B', '-c', program, str(PATH)], cwd=tmp, capture_output=True, text=True, timeout=5)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('--deadline-seconds', result.stdout)
            self.assertEqual(list(Path(tmp).iterdir()), [])


if __name__ == '__main__':
    unittest.main()
