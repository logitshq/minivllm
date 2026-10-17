"""Small raw-parser tests; no OS commands, active evidence or device imports."""
import copy
import importlib.util
from pathlib import Path
import unittest

PATH = Path(__file__).with_name('resource_reparse.py')
SPEC = importlib.util.spec_from_file_location('resource_reparse_under_test', PATH)
m = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(m)


def sample():
    command = lambda text: dict(returncode=0, text=text, error='')
    meta = lambda cmd, text: dict(command=cmd, returncode=0, stdout=text, stderr='')
    ps = ['/bin/ps', '-p', '12', '-o', 'pid=,ppid=,rss=,comm=']
    thermal = ['/usr/bin/osascript', '-l', 'JavaScript', '-e',
               'ObjC.import("Foundation"); String($.NSProcessInfo.processInfo.thermalState)']
    return dict(label='matrix:running', monotonic_s=1.5, owned_model_pid=12,
        pressure_level=command('1'), pressure_level_value=1,
        swap=command('vm.swapusage: total = 1024.00M  used = 429.06M  free = 594.94M'), swap_used_mb=429.06,
        memory_pressure=command('System-wide memory free percentage: 96%'), free_percent=96,
        vm_stat=command('Mach Virtual Memory Statistics: (page size of 16384 bytes)\nPages free: 4000.\n'
            + '\n'.join(k + ': 10.' for k in m.COUNTERS)), vm_counters={k: 10 for k in m.COUNTERS},
        thermal=command('Note: No thermal warning level has been recorded'),
        power_metadata=dict(status='known', source='AC Power', raw=meta(['/usr/bin/pmset', '-g', 'batt'], "Now drawing from 'AC Power'")),
        processinfo_thermal_metadata=dict(status='known', code=0, name='nominal', raw=meta(thermal, '0')),
        exact_owned_pid_metadata=dict(status='known', pid=12, ppid=11, rss_kib=1000, executable='/python', raw=meta(ps, '12 11 1000 /python')),
        process_inventory_ok=True, native_process_inventory_ok=True, authorized_suspended_pids=[],
        known_inference_processes=[], known_inference_process_count=0, only_owned_known_inference_processes=True,
        exactly_one_owned_known_model=False, no_active_foreign_native_processes=True,
        native_inference_processes=[dict(pid=12, ppid=11, owned=True, authorized_suspended=False)])


class RawTests(unittest.TestCase):
    def test_exact_raw_values_and_free_units(self):
        parsed = m.parse_sample(sample(), 12, 11)
        self.assertEqual(parsed['free_bytes'], 4000 * 16384)
        self.assertEqual(parsed['exact_pid_rss_kib'], 1000)

    def test_metadata_corruption_and_duplicate_raw_rejected(self):
        for change in (lambda x: x.update(pressure_level_value=True),
                       lambda x: x.update(swap_used_mb=0.0),
                       lambda x: x['vm_stat'].update(text=x['vm_stat']['text']+'\nSwapouts: 10.'),
                       lambda x: x['power_metadata'].update(source='Battery Power'),
                       lambda x: x['processinfo_thermal_metadata'].update(code=1),
                       lambda x: x['exact_owned_pid_metadata'].update(ppid=99)):
            row = sample(); change(row)
            with self.assertRaises(ValueError): m.parse_sample(row, 12, 11)

    def test_probe_error_rejected(self):
        row = sample(); row['vm_stat']['returncode'] = 1
        with self.assertRaises(ValueError): m.parse_sample(row, 12, 11)

    def test_exact_pid_exit_race_stays_unknown(self):
        row = sample(); exact = row['exact_owned_pid_metadata']
        exact.update(status='unknown', ppid=None, rss_kib=None, executable=None)
        exact['raw'].update(returncode=1, stdout='')
        parsed = m.parse_sample(row, 12, 11)
        self.assertEqual(parsed['exact_pid_probe_status'], 'unknown')
        self.assertIsNone(parsed['exact_pid_rss_kib'])

    def test_foreign_inventory_not_sole_gpu_claim(self):
        row = sample(); process = row['native_inference_processes'][0]
        process.update(pid=99, ppid=1, owned=False)
        row['no_active_foreign_native_processes'] = False
        self.assertFalse(m.parse_sample(row, 12, 11)['known_inventory_clean'])

    def test_duplicate_json_rejected(self):
        with self.assertRaises(ValueError): m.unique([('x', 1), ('x', 1)])


if __name__ == '__main__':
    unittest.main()
