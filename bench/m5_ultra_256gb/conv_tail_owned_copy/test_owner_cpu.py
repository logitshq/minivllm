"""No GPU/services/listeners: E026 owner guards, pin failure and child cleanup."""
import copy
from contextlib import ExitStack
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('tiny_owner_cpu_review', HERE / 'owner.py')
o = importlib.util.module_from_spec(spec); spec.loader.exec_module(o)


def sample(pressure=2, pid=None):
    native = [{'pid': 74384, 'ppid': 1, 'state': 'T', 'owned': False}]
    if pid: native.append({'pid': pid, 'ppid': os.getpid(), 'state': 'R', 'owned': True})
    return {'pressure_level_value': pressure, 'pressure_level': {'returncode': 0},
        'vm_stat': {'returncode': 0, 'text': 'Mach Virtual Memory Statistics: (page size of 16384 bytes)\nPages free: 4194304.'},
        'swap': {'returncode': 0}, 'swap_used_mb': 324.0, 'vm_counters': {'Swapouts': 5},
        'power_metadata': {'source': 'AC Power'}, 'processinfo_thermal_metadata': {'code': 0},
        'native_process_inventory_ok': True, 'process_inventory_ok': True,
        'native_inference_processes': native, 'known_inference_processes': []}


class FakeChild:
    pid = 12345
    def __init__(self, mode='done'):
        self.mode, self.returncode, self.actions = mode, None, []
    def poll(self): return self.returncode
    def wait(self, timeout):
        self.actions.append(('wait', timeout))
        if self.returncode is not None: return self.returncode
        if self.mode == 'done': self.returncode = 0; return 0
        if self.mode == 'childfail': self.returncode = 7; return 7
        if timeout == 2: raise TimeoutError('external arm deadline')
        raise subprocess.TimeoutExpired('fake', timeout)
    def terminate(self): self.actions.append(('terminate',))
    def kill(self): self.actions.append(('kill',)); self.returncode = -9


class OwnerTests(unittest.TestCase):
    def test_pressure_recovery_cannot_worsen_again(self):
        rows = [sample(2), sample(2), sample(1)]
        o.guard(rows)
        with self.assertRaisesRegex(RuntimeError, 'pressure'):
            o.guard(rows + [sample(2)])

    def test_actual_pages_swap_and_foreign_process_guards(self):
        baseline = sample()
        for edit in ('free', 'swapouts', 'used', 'foreign', 'lease', 'parent', 'power', 'thermal'):
            current = sample(pid=12345)
            if edit == 'free': current['vm_stat']['text'] = current['vm_stat']['text'].replace('4194304', '1'); current['free_percent'] = 100
            if edit == 'swapouts': current['vm_counters']['Swapouts'] += 1
            if edit == 'used': current['swap_used_mb'] += 16.01
            if edit == 'foreign': current['native_inference_processes'].append({'pid': 11111, 'ppid': os.getpid(), 'state': 'T', 'owned': True})
            if edit == 'lease': current['native_inference_processes'][0]['state'] = 'S'
            if edit == 'parent': current['native_inference_processes'][1]['ppid'] += 1
            if edit == 'power': current['power_metadata']['source'] = 'Battery Power'
            if edit == 'thermal': current['processinfo_thermal_metadata']['code'] = 2
            with self.subTest(edit=edit), self.assertRaises(RuntimeError): o.guard([baseline, current], 12345)
        good = sample(pid=12345); good['swap_used_mb'] += 16
        o.guard([baseline, good], 12345)
        self.assertEqual(good['real_free_bytes'], 64 << 30)

    def prepare_fixture(self, root):
        root = root.resolve()
        here, base = root / 'bench/tiny', root / 'bench'
        here.mkdir(parents=True)
        files = [here / n for n in ('candidate.py', 'screen.py', 'test_cpu.py', 'test_screen_cpu.py', 'test_owner_cpu.py')]
        files += [base / n for n in ('resources_v2.py', 'resources.py', 'resource_base.py', 'suspended_service.py')]
        runtime = root / 'runtime'
        files += [runtime / n for n in ('mlx/core.fake.so', 'mlx/lib/libmlx.dylib', 'mlx/lib/mlx.metallib', 'numpy/__init__.py', 'numpy/_core/_multiarray_umath.fake.so')]
        python = root / 'python'
        files += [python]
        for p in files: p.parent.mkdir(parents=True, exist_ok=True); p.write_text('# CPU fixture\n')
        python.chmod(0o700)
        manifest = here / 'source-pins.json'; manifest.write_text('{"sources":{}}'); files.append(manifest)
        files.append(Path(o.__file__).resolve())
        pins = {str(p): o.sha(p) for p in files}
        plan = {'schema': 'E026-tiny-owner-v1', 'arms': o.ARMS, 'arm_seconds': 30, 'total_seconds': 180,
                'execution': {'python': str(python), 'screen': {'path': str(here / 'screen.py'), 'sha256': pins[str(here / 'screen.py')]},
                              'pins': pins, 'environment': o.ENV}}
        path = root / 'plan.json'; path.write_text(json.dumps(plan))
        out = root / 'out'; out.mkdir()
        return here, base, NS(plan=path, plan_sha256=o.sha(path), out=out, suspended_pid=74384)

    def test_pins_are_checked_and_environment_cleaned(self):
        with tempfile.TemporaryDirectory() as tmp:
            here, base, args = self.prepare_fixture(Path(tmp))
            with patch.object(o, 'HERE', here), patch.object(o, 'BASE', base), patch.dict(os.environ, {'MLX_UNREGISTERED': 'evil'}):
                _, _, env = o.prepare(args, lambda: None)
                self.assertNotIn('MLX_UNREGISTERED', env)
                self.assertEqual({k: v for k, v in env.items() if k.startswith('MLX_')}, o.ENV)
                (here / 'candidate.py').write_text('# changed\n')
                with self.assertRaisesRegex(RuntimeError, 'pin mismatch'): o.prepare(args, lambda: None)

    def test_plan_sha_and_block_hash_deadline_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, _, args = self.prepare_fixture(Path(tmp)); args.plan_sha256 = '0' * 64
            with self.assertRaisesRegex(RuntimeError, 'plan SHA'): o.prepare(args, lambda: None)
            def deadline(): raise TimeoutError('hash deadline')
            with self.assertRaises(TimeoutError): o.sha(args.plan, deadline)

    def test_partial_child_result_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            o.write(out, 'complete.json', {'status': 'PASS', 'arm': 'off', 'cases': []})
            with self.assertRaisesRegex(RuntimeError, 'incomplete'):
                o.validate_child(NS(contract=lambda arm: {}), out, 'off', 12345, {})

    def run_mock(self, root, mode='done', fail_probe=False, fail_reap=False, partial=False):
        args = NS(out=root / 'out', plan=root / 'plan.json', plan_sha256='0' * 64, suspended_pid=74384)
        children = []
        def popen(*a, **kw):
            child = FakeChild(mode); children.append(child); return child
        def probe(label, pid=None, **kw):
            if fail_probe and label.endswith('/running'): raise RuntimeError('probe failed')
            return sample(pid=pid)
        plan = {'execution': {'python': '/not-executed', 'pins': {}}}
        with ExitStack() as stack:
            for name, value in (('getitimer', lambda *a: (0.0, 0.0)), ('signal', lambda *a: None), ('setitimer', lambda *a: None)):
                stack.enter_context(patch.object(o.signal, name, value))
            stack.enter_context(patch.object(o, 'prepare', return_value=(plan, object(), {})))
            stack.enter_context(patch.object(o.subprocess, 'Popen', side_effect=popen))
            stack.enter_context(patch.object(o.subprocess, 'run', return_value=NS(returncode=0 if fail_reap else 1, stdout='12345' if fail_reap else '', stderr='')))
            validate = stack.enter_context(patch.object(o, 'validate_child'))
            if partial: validate.side_effect = RuntimeError('partial child evidence')
            else: validate.side_effect = lambda screen, out, arm, pid, pins: {'arm': arm, 'pid': pid, 'status': 'PASS'}
            status = o.run(args, NS(snapshot=probe))
        return status, children, json.loads((args.out / 'complete.json').read_text()), args.out

    def test_deadline_terminates_kills_reaps_and_stops_matrix(self):
        with tempfile.TemporaryDirectory() as tmp:
            status, children, result, out = self.run_mock(Path(tmp), mode='timeout')
            self.assertEqual(status, 1); self.assertEqual(len(children), 1)
            self.assertIn(('terminate',), children[0].actions); self.assertIn(('kill',), children[0].actions)
            self.assertTrue(json.loads((out / 'off/release.json').read_text())['strict_absence'])
            self.assertIn('external arm deadline', result['errors'][0]['traceback'])

    def test_probe_exception_reaps_direct_child(self):
        with tempfile.TemporaryDirectory() as tmp:
            status, children, result, out = self.run_mock(Path(tmp), mode='timeout', fail_probe=True)
            self.assertEqual(status, 1); self.assertEqual(len(children), 1)
            self.assertIn(('kill',), children[0].actions)
            self.assertIn('probe failed', result['errors'][0]['traceback'])
            self.assertTrue((out / 'off/release.json').exists())

    def test_child_failure_keeps_first_error_when_absence_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            status, children, result, out = self.run_mock(Path(tmp), mode='childfail', fail_reap=True)
            self.assertEqual(status, 1); self.assertEqual(len(children), 1)
            self.assertIn('child exited 7', result['errors'][0]['traceback'])
            self.assertTrue(any(e['phase'] == 'cleanup' for e in result['errors']))
            self.assertEqual(json.loads((out / 'first-error.json').read_text()), result['errors'][0])

    def test_exit_zero_partial_is_not_pass_and_halts_next_arm(self):
        with tempfile.TemporaryDirectory() as tmp:
            status, children, result, _ = self.run_mock(Path(tmp), partial=True)
            self.assertEqual(status, 1); self.assertEqual(len(children), 1)
            self.assertEqual(result['status'], 'FAIL')

    def test_all_three_children_reaped_in_fixed_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            status, children, result, out = self.run_mock(Path(tmp))
            self.assertEqual(status, 0); self.assertEqual(len(children), 3)
            self.assertEqual([x['arm'] for x in result['arms']], o.ARMS)
            for arm in o.ARMS:
                row = json.loads((out / arm / 'release.json').read_text())
                self.assertTrue(row['waited_reaped'] and row['strict_absence'])

    def test_help_import_does_not_touch_mlx_resources_or_socket(self):
        code = '''
import importlib.abc, runpy, socket, sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,name,*args):
  if name.split('.')[0] in {'mlx','numpy','torch','resources','resources_v2','resource_base'}: raise AssertionError(name)
sys.meta_path.insert(0,Block());socket.socket=lambda *a,**k: (_ for _ in ()).throw(AssertionError('socket'))
sys.argv=[sys.argv[1],'--help'];runpy.run_path(sys.argv[0],run_name='__main__')
'''
        result = subprocess.run([sys.executable, '-I', '-B', '-c', code, str(HERE / 'owner.py')], capture_output=True, text=True, timeout=5)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
