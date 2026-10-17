"""CPU-only contract/evidence tests. Never import MLX or start a listener."""
import importlib.util
import json
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

PATH = Path(__file__).with_name('screen_v2.py')
spec = importlib.util.spec_from_file_location('e025_screen_v2_tested', PATH)
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)


class Scalar:
    def __init__(self, value): self.value = value
    def item(self): return self.value


class Mask:
    def __init__(self, values): self.values = values
    def reshape(self, *_): return self
    def astype(self, *_): return self


class Array:
    def __init__(self, raw=b'\0\0', shape=(1,), dtype='bfloat16', finite=True):
        self.raw, self.shape, self.dtype, self.finite = raw, shape, dtype, finite
    def view(self, _): return self
    def __ne__(self, other): return Mask([a != b for a, b in zip(self.raw, other.raw)])
    def __getitem__(self, item):
        return Array(self.raw, (item.stop, *self.shape[1:]), self.dtype, self.finite)


class FakeMX:
    uint8, int32, bfloat16, uint32 = 'uint8', 'int32', 'bfloat16', 'uint32'
    def __init__(self): self.saved = []
    def eval(self, *_): pass
    def save_safetensors(self, path, arrays):
        self.saved.append((path, {k: v.dtype for k, v in arrays.items()}))
        Path(path).write_bytes(b''.join(v.raw for v in arrays.values()))
    def isfinite(self, value): return value.finite
    def all(self, value): return Scalar(value)
    def array_equal(self, left, right): return Scalar(left.raw == right.raw)
    def argmax(self, mask): return Scalar(next(i for i, value in enumerate(mask.values) if value))


class ScreenTests(unittest.TestCase):
    def test_fixed_coverage_seed_geometry_and_budgets(self):
        p = s.fixed_plan()
        self.assertEqual((p['expected_checks'], p['expected_phases'], p['expected_kernels']), (33, 68, 6))
        self.assertEqual(p['normal_M'], [19])
        self.assertEqual(p['fallback_M'], [1, 11, 12, 32, 33, 128])
        self.assertEqual(p['seed'], 20261017)
        self.assertEqual(p['cache_limit_bytes'], 0)
        self.assertEqual(p['mlx_observation_budget_bytes'], 1 << 30)
        self.assertEqual(p['physical_observation_budget_bytes'], 2 << 30)
        self.assertEqual([g['split_k'] for g in p['geometries']], [2, 2, 3])
        self.assertFalse(p['performance_qualified'])
        self.assertFalse(p['full_model_qualified'])

    def test_cpu_only_cli_blocks_runtime_imports(self):
        code = '''import importlib.abc,runpy,sys
class Deny(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,*args):
  if fullname.split('.')[0] in ('mlx','numpy','torch','tutorial_code'): raise AssertionError('runtime imported: '+fullname)
sys.meta_path.insert(0,Deny())
sys.argv=[sys.argv[1],*sys.argv[2:]]
runpy.run_path(sys.argv[0],run_name='__main__')
'''
        for argv, rc in (([], 0), (['--describe'], 0), (['--help'], 0), (['--run'], 2)):
            result = subprocess.run([sys.executable, '-I', '-B', '-c', code, str(PATH), *argv], capture_output=True, text=True, timeout=5)
            self.assertEqual(result.returncode, rc, result.stderr)

    def test_source_pins_include_fixed_support_and_reject_change(self):
        pins = s.source_pins()
        self.assertIn(str(PATH.resolve()), pins)
        self.assertIn(str(s.LEDGER_PATH), pins)
        self.assertIn(str(s.PHYSICAL_PATH), pins)
        with mock.patch.object(s, 'sha', return_value='0' * 64):
            with self.assertRaisesRegex(ValueError, 'fixed source SHA'):
                s.source_pins()

    def test_each_success_and_failure_saves_original_dtype_pair(self):
        mx = FakeMX()
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            a = Array(b'\0\0\0\x80', (2,))
            good = s.record_pair(mx, out, 0, 'success', a, a)
            bad = s.record_pair(mx, out, 1, 'signed-zero-mismatch', a, Array(b'\0\0\0\0', (2,)))
            self.assertEqual(good['status'], 'PASS')
            self.assertEqual(bad['status'], 'FAIL')
            self.assertEqual(bad['first_differing_byte_index'], 3)
            for row in (good, bad):
                raw = out / row['rawpair']['path']
                self.assertEqual(s.sha(raw), row['rawpair']['sha256'])
                self.assertEqual(raw.stat().st_size, 8)
            self.assertEqual(mx.saved[0][1], {'reference': 'bfloat16', 'candidate': 'bfloat16'})
            self.assertEqual(json.loads((out / 'first-difference.json').read_text())['label'], 'signed-zero-mismatch')

    def test_nonfinite_equal_bits_still_fail(self):
        with tempfile.TemporaryDirectory() as td:
            a = Array(b'\xc0\x7f', finite=False)
            row = s.record_pair(FakeMX(), Path(td), 0, 'nan', a, a)
            self.assertTrue(row['bytes_equal'])
            self.assertFalse(row['finite'])
            self.assertEqual(row['status'], 'FAIL')
            self.assertTrue((Path(td) / row['rawpair']['path']).exists())

    def test_metadata_mismatch_fails_even_equal_bytes(self):
        with tempfile.TemporaryDirectory() as td:
            row = s.record_pair(FakeMX(), Path(td), 0, 'shape', Array(shape=(1,)), Array(shape=(1, 1)))
            self.assertFalse(row['metadata_equal'])
            self.assertEqual(row['status'], 'FAIL')

    def test_first_difference_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            for i in range(2):
                s.record_pair(FakeMX(), out, i, f'bad{i}', Array(b'\0\0'), Array(b'\1\0'))
            self.assertEqual(json.loads((out / 'first-difference.json').read_text())['label'], 'bad0')
            self.assertTrue((out / 'pairs/001.safetensors').exists())

    def test_fixture_manifest_exact_names_and_hash(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td); raw = out / 'input.safetensors'; raw.write_bytes(b'CPU fixture')
            data = dict(schema='E025-projection-inputs-v1', cases={n: dict(path=str(raw), sha256=s.sha(raw)) for n, _, _ in s.c.SHAPES})
            path = out / 'manifest.json'; s.write(path, data)
            self.assertEqual(s.fixture_manifest(path, s.sha(path)), data)
            raw.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'tensor SHA'):
                s.fixture_manifest(path, s.sha(path))
            data['cases']['extra'] = {}
            s.write(path, data)
            with self.assertRaisesRegex(ValueError, 'exact three-case'):
                s.fixture_manifest(path, s.sha(path))

    def test_sigterm_is_not_reported_as_alarm(self):
        with self.assertRaisesRegex(TimeoutError, 'SIGALRM'):
            s.stop_signal(signal.SIGALRM)
        for sig in (signal.SIGTERM, signal.SIGINT):
            with self.assertRaisesRegex(InterruptedError, signal.Signals(sig).name):
                s.stop_signal(sig)

    def test_failure_retains_successful_pair_input_attempt_and_kernel_receipt(self):
        mx = FakeMX()
        mx.random = types.SimpleNamespace(seed=lambda _: None)
        mx.device_info = lambda: {'device_name': 'mock'}
        mx.metal = types.SimpleNamespace(device_info=lambda: {'architecture': 'mock'})
        mx.load = lambda _: dict(activation=Array(shape=(128, 5120)),
            weight=Array(shape=(6144, 640), dtype='uint32'), scales=Array(shape=(6144, 320), dtype='uint8'))
        class Op:
            receipts = []
            def __init__(self): self.kernels = {'example': 1}
            def _hardware(self): return True
            def original(self, *_): return Array(shape=(19, 6144))
            def __call__(self, *_): return self.original()
            def partials(self, *_):
                self.receipts = [{'kind': 'reference', 'compile_attempt': True}]
                raise RuntimeError('original kernel compile failure')
        enabled, disabled = Op(), Op()
        result = dict(contract=s.fixed_plan(), checks=[], inputs=[], normal_cases_passed=0, fallback_cases_passed=0)
        ledger = types.SimpleNamespace(phase=mock.Mock())
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            def snapshot(_mx, _out, name, arrays, suffix):
                if suffix == 'after': raise OSError('input snapshot failure')
                return dict(path='before.safetensors', sha256='a' * 64)
            with mock.patch.object(s.c, 'SameSplitPartial', side_effect=[enabled, disabled]), mock.patch.object(s, 'input_snapshot', side_effect=snapshot):
                with self.assertRaisesRegex(RuntimeError, 'original kernel compile failure'):
                    s.execute(mx, out, result, {'cases': {'gdn_z': {'path': 'mock'}}}, ledger, lambda _: None)
            self.assertEqual(len(result['checks']), 1)
            self.assertTrue((out / result['checks'][0]['rawpair']['path']).is_file())
            self.assertTrue(json.loads((out / 'input-exit-error.json').read_text())['primary_failure_preserved'])
            self.assertEqual(json.loads((out / 'kernel-source-receipts.json').read_text())[0]['kind'], 'reference')
            self.assertEqual(enabled.kernels, {})
            self.assertNotIn(mock.call('final_release'), ledger.phase.call_args_list)


if __name__ == '__main__':
    unittest.main()
