"""Small thread/pipe fakes only. No real process, signal, network or device.

These tests verify independent scheduling and failure propagation, not Apple
physical correctness or real-time guarantees. All Popen handles are explicit
fakes; the existing physical parser is reused with synthetic command outputs.
"""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import threading
import time
import unittest
from unittest.mock import patch


def load(name):
    spec = importlib.util.spec_from_file_location('glm_test_'+name, Path(__file__).with_name(name+'.py'))
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    return value


m = load('live_monitor')
guard = load('physical_guard')


class Model:
    pid = 543210
    def __init__(self): self.returncode = None
    def poll(self): return self.returncode


class ReadProcess:
    """communicate has real thread waits but no OS process behind this handle."""
    pid = 543211
    def __init__(self, output='ok', errors='', code=0, ready=True, stubborn=False):
        self.output = output; self.errors = errors; self.code = code; self.returncode = None
        self.ready = threading.Event(); self.actions = []; self.stubborn = stubborn
        if ready: self.ready.set()
    def poll(self): return self.returncode
    def communicate(self, timeout):
        if not self.ready.wait(timeout): raise subprocess.TimeoutExpired('fake-read', timeout)
        self.returncode = self.code; return self.output, self.errors
    def wait(self, timeout):
        self.actions.append('wait')
        if self.returncode is None: raise subprocess.TimeoutExpired('fake-wait', timeout)
        return self.returncode
    def terminate(self):
        self.actions.append('terminate')
        if not self.stubborn: self.code = -15; self.ready.set()
    def kill(self): self.actions.append('kill'); self.code = -9; self.ready.set()


class Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name); self.child = Model()
        # A forgotten injection must fail instead of executing a real command.
        self.popen_guard = patch.object(subprocess, 'Popen', side_effect=AssertionError('real Popen forbidden'))
        self.popen_guard.start(); self.addCleanup(self.popen_guard.stop)

    def run_monitor(self, physical, metadata, **kwargs):
        values = dict(out=self.root/'monitor', physical=physical, metadata=metadata,
                      work_deadline=time.monotonic()+1, physical_interval=.08, metadata_interval=.2,
                      probe_seconds=.5, join_seconds=.6,
                      popen=lambda *a, **k: (_ for _ in ()).throw(AssertionError('unexpected command')))
        values.update(kwargs)
        return m.monitor(self.child, **values)

    def commands(self, child=None, popen=None):
        self.cancel = threading.Event()
        return m.Commands(self.root, self.cancel, popen=popen or (lambda *a, **k: child))

    def test_slow_metadata_does_not_stop_physical_lane_and_records_real_gaps(self):
        entered = threading.Event(); release = threading.Event(); count = []
        def physical(index, commands, deadline):
            self.assertTrue(entered.wait(.2)); count.append(index)
            if index == 3: release.set(); self.child.returncode = 0
            return dict(physical_index=index)
        def metadata(index, commands, deadline):
            entered.set(); self.assertTrue(release.wait(.4)); return dict(slow_metadata_finished=True)
        result = self.run_monitor(physical, metadata)
        self.assertEqual(result['status'], 'COMPLETE'); self.assertGreaterEqual(len(count), 4)
        self.assertEqual(len(result['lanes']['metadata']['samples']), 1)
        self.assertGreater(result['lanes']['metadata']['samples'][0]['elapsed_seconds'], .08)
        self.assertTrue(all(0 <= row['gap_seconds'] <= .08 for row in result['lanes']['physical']['samples']))
        self.assertFalse(result['model_child_release_proven'])
        self.assertFalse(result['physical_resource_safety_qualified'])

    def test_metadata_error_propagates_even_after_model_successful_exit(self):
        entered = threading.Event(); both = threading.Event()
        def physical(*args): entered.set(); self.assertTrue(both.wait(.2)); return {}
        def metadata(*args):
            self.assertTrue(entered.wait(.1)); both.set(); self.child.returncode = 0
            raise RuntimeError('raw process inventory missing')
        with self.assertRaises(m.MonitorError) as caught:
            self.run_monitor(physical, metadata)
        row = caught.exception.report
        self.assertEqual(row['status'], 'INCOMPLETE')
        self.assertTrue(any('raw process inventory missing' in error['message'] for error in row['errors']))
        self.assertEqual(json.loads((self.root/'monitor/monitor.json').read_bytes()), row)

    def test_failed_footprint_after_normal_exit_is_not_exempted(self):
        entered = threading.Event(); both = threading.Event()
        def physical(*args):
            entered.set(); self.assertTrue(both.wait(.1)); raise ValueError('footprint PID disappeared')
        def metadata(*args):
            self.assertTrue(entered.wait(.1)); self.child.returncode = 0; both.set(); return {}
        with self.assertRaises(m.MonitorError) as caught:
            self.run_monitor(physical, metadata)
        self.assertEqual(caught.exception.report['status'], 'INCOMPLETE')
        self.assertTrue(any('PID disappeared' in e['message'] for e in caught.exception.report['errors']))

    def test_uncooperative_callback_has_bounded_join_and_is_explicitly_unresolved(self):
        release = threading.Event(); exited = threading.Event()
        def physical(*args):
            try: release.wait(.6); return {}
            finally: exited.set()
        try:
            began = time.monotonic()
            with self.assertRaises(m.MonitorError) as caught:
                self.run_monitor(physical, lambda *args: {}, probe_seconds=.04, join_seconds=.04)
            self.assertLess(time.monotonic()-began, .4)
            report = caught.exception.report
            self.assertIn('glm-monitor-physical', report['unjoined_workers'])
            self.assertFalse(report['read_commands_settled'])
        finally:
            release.set(); self.assertTrue(exited.wait(.2))
            # Wait for the test-owned thread's final evidence write before temp cleanup.
            for thread in threading.enumerate():
                if thread.name == 'glm-monitor-physical': thread.join(.2)

    def test_every_command_has_elapsed_wait_and_secret_argv_output_is_redacted(self):
        child = ReadProcess(output='123 1 --api-key secret-value\n')
        commands = self.commands(child)
        result = commands.read(['/bin/ps', '-axo', 'pid=,ppid=,args='])
        self.assertIn('secret-value', result['stdout'])  # only the frozen parser receives raw text
        row = json.loads((self.root/'command-0000.json').read_bytes())
        self.assertNotIn('stdout', row); self.assertNotIn('secret-value', json.dumps(row))
        self.assertEqual(row['stdout_bytes'], len(result['stdout'].encode()))
        self.assertGreaterEqual(row['elapsed_seconds'], 0); self.assertTrue(row['waited_reaped'])
        self.assertEqual(child.actions, ['wait'])
        self.assertEqual(json.loads((self.root/'command-0000.child.json').read_bytes())['pid'], child.pid)

    def test_command_timeout_uses_returned_handle_term_kill_and_wait(self):
        child = ReadProcess(ready=False, stubborn=True); commands = self.commands(child)
        with self.assertRaises(TimeoutError): commands.read(['/bin/ps', '-p', '543210', '-o', 'pid='], timeout=.01)
        row = commands.snapshot()[0]
        self.assertEqual(child.actions, ['terminate', 'kill', 'wait'])
        self.assertEqual(row['returncode'], -9); self.assertTrue(row['waited_reaped'])
        self.assertFalse(row['unknown_launch']); self.assertEqual(row['status'], 'INCOMPLETE')
        self.assertEqual(len(row['cleanup_errors']), 1)

    def test_unknown_constructor_outcome_is_not_no_child(self):
        def unknown(*args, **kwargs): raise OSError('Popen outcome unknown')
        commands = self.commands(popen=unknown)
        with self.assertRaises(OSError): commands.read(['/bin/ps', '-p', '543210', '-o', 'pid='])
        row = commands.snapshot()[0]
        self.assertTrue(row['unknown_launch']); self.assertFalse(row['waited_reaped'])
        self.assertTrue((self.root/'command-0000.intent.json').is_file())
        self.assertFalse((self.root/'command-0000.child.json').exists())
        with self.assertRaises(m.MonitorError) as caught:
            self.run_monitor(lambda i, c, d: c.read(['/usr/bin/vm_stat']), lambda *a: {}, popen=unknown)
        self.assertTrue(caught.exception.report['unresolved_read_commands'][0]['unknown_launch'])
        self.assertFalse(caught.exception.report['read_commands_settled'])

    def test_known_unreaped_probe_cannot_be_a_settled_failure(self):
        class Broken(ReadProcess):
            def communicate(self, timeout): raise OSError('pipe failed')
            def terminate(self): raise PermissionError('TERM failed')
            def kill(self): raise PermissionError('KILL failed')
        process = Broken()
        with self.assertRaises(m.MonitorError) as caught:
            self.run_monitor(lambda i, c, d: c.read(['/usr/bin/vm_stat']), lambda *a: {},
                             popen=lambda *a, **k: process)
        report = caught.exception.report
        self.assertFalse(report['read_commands_settled'])
        row = report['unresolved_read_commands'][0]
        self.assertEqual(row['pid'], process.pid); self.assertFalse(row['unknown_launch'])
        self.assertFalse(row['waited_reaped']); self.assertEqual(len(row['cleanup_errors']), 2)

    def test_cancel_of_inflight_command_is_settled_before_monitor_returns(self):
        command_started = threading.Event(); process = ReadProcess(ready=False)
        def popen(*args, **kwargs): command_started.set(); return process
        def physical(index, commands, deadline): return commands.read(['/usr/bin/vm_stat'])
        def metadata(*args):
            self.assertTrue(command_started.wait(.2)); raise ValueError('power gate failed')
        with self.assertRaises(m.MonitorError) as caught:
            self.run_monitor(physical, metadata, popen=popen)
        report = caught.exception.report
        self.assertTrue(report['read_commands_settled']); self.assertFalse(report['unjoined_workers'])
        self.assertEqual(process.actions, ['terminate', 'wait'])
        self.assertTrue(report['commands']['physical'][0]['waited_reaped'])
        self.assertEqual(report['status'], 'INCOMPLETE')

    def test_masked_command_timeout_still_cannot_complete(self):
        process = ReadProcess(ready=False)
        def physical(index, commands, deadline):
            try: commands.read(['/usr/bin/vm_stat'], timeout=.01)
            except TimeoutError: pass
            self.child.returncode = 0; return {}
        with self.assertRaises(m.MonitorError) as caught:
            self.run_monitor(physical, lambda *args: {}, popen=lambda *a, **k: process)
        self.assertTrue(any('masked' in row['message'] for row in caught.exception.report['errors']))

    def test_frozen_physical_measure_accepts_commands_api_and_keeps_raw_parse(self):
        pid = self.child.pid
        def popen(argv, **kwargs):
            if argv[0] == '/usr/bin/footprint':
                Path(argv[-1]).write_text(json.dumps({'unit':'byte', 'bytes per unit':1, 'errors':[], 'warnings':[],
                    'processes':[{'pid':pid, 'footprint':100, 'auxiliary':{'phys_footprint':90,'phys_footprint_peak':95}}],
                    'auxiliary':{'sys_footprint':30*guard.GIB}})); output = ''
            else:
                output = {'kern.memorystatus_vm_pressure_level':'1', 'vm.swapusage':'used = 0.00M',
                          'hw.memsize':str(guard.RAM), '/usr/bin/vm_stat':'Swapouts: 0.\n'}[argv[-1]]
            return ReadProcess(output=output)
        commands = self.commands(popen=popen)
        sample = guard.measure(self.root/'physical', pid, run=commands.run)
        self.assertEqual(sample['status'], 'MEASURED'); self.assertEqual(sample['parsed']['peak_bytes'], 100)
        self.assertEqual(len(commands.snapshot()), 5)
        self.assertTrue(all(row['waited_reaped'] for row in commands.snapshot()))

    def test_frozen_checkpoint_accepts_read_api_and_retains_all_seven_command_timings(self):
        path = Path(__file__).resolve().parents[1]/'glm_flash_cycle/owner.py'
        spec = importlib.util.spec_from_file_location('glm_original_checkpoint_cpu', path)
        owner = importlib.util.module_from_spec(spec); spec.loader.exec_module(owner)
        def popen(argv, **kwargs):
            output = errors = ''; code = 0
            if argv[:2] == ['/bin/ps', '-p'] or argv[0] == '/usr/sbin/lsof': code = 1
            elif argv[0] == '/bin/launchctl': code = 113; errors = 'Could not find service'
            elif argv[0] == '/usr/bin/pmset': output = "Now drawing from 'AC Power'"
            elif argv[0] == '/usr/bin/osascript': output = '0'
            elif argv[:2] == ['/bin/ps', '-axo']:
                output = '%d %d fake-owned-child\n' % (self.child.pid, os.getpid())
            else: raise AssertionError(argv)
            return ReadProcess(output=output, errors=errors, code=code)
        commands = self.commands(popen=popen); kept = []
        result = owner.checkpoint(commands, old_pid=543200, child_pid=self.child.pid, owner_pid=os.getpid(),
                                  deadline=time.monotonic()+.5, record=kept.append)
        self.assertEqual(result['status'], 'PASS'); self.assertEqual(kept, [result])
        rows = commands.snapshot(); self.assertEqual(len(rows), 7)
        self.assertTrue(all(row['waited_reaped'] and row['elapsed_seconds'] >= 0 for row in rows))
        self.assertNotIn('stdout', result['commands'][-1]); self.assertNotIn('stdout', rows[-1])

    def test_nonzero_model_exit_and_no_live_child_never_complete(self):
        for code in (-15, 0):
            with self.subTest(code=code):
                self.child.returncode = code
                with self.assertRaises(m.MonitorError) as caught:
                    self.run_monitor(lambda *args: {}, lambda *args: {}, out=self.root/('exit'+str(code)))
                self.assertEqual(caught.exception.report['status'], 'INCOMPLETE')

    def test_live_checks_use_incremental_ranges_not_full_command_history_copies(self):
        snapshots = []; original = m.Commands.snapshot; entered = threading.Event()
        def snapshot(commands):
            self.assertEqual(threading.current_thread(), threading.main_thread())
            snapshots.append(commands); return original(commands)
        def physical(index, commands, deadline):
            self.assertTrue(entered.wait(.2))
            commands.read(['/usr/bin/vm_stat'])
            if index == 2: self.child.returncode = 0
            return {}
        def metadata(*args): entered.set(); return {}
        with patch.object(m.Commands, 'snapshot', snapshot):
            report = self.run_monitor(physical, metadata, popen=lambda *a, **k: ReadProcess())
        self.assertEqual(len(snapshots), 2)  # once per lane, only after all live work
        self.assertEqual([r['command_range'] for r in report['lanes']['physical']['samples']],
                         [dict(start=i, stop=i+1) for i in range(3)])
        self.assertTrue(report['read_commands_settled'])
        commands = self.commands(popen=lambda *a, **k: (_ for _ in ()).throw(OSError('unknown launch')))
        with self.assertRaises(OSError): commands.read(['/usr/bin/vm_stat'])
        self.assertEqual(commands.summary(), dict(count=1, failed=1, unresolved=1))


if __name__ == '__main__': unittest.main()
