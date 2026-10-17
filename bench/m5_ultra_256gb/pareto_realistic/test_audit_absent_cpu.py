"""Synthetic stdlib corruption tests, not device or recovery qualification.

Whole-auditor routing tests replace only numerical pins/stream work with explicit
CPU fakes; separate tests exercise the unchanged numerical auditor's self-test.
No model/runtime hashing, measurement imports, subprocesses, signals, or HTTP.
"""
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('absent_auditor_cpu', HERE / 'audit_absent_qualification.py')
a = importlib.util.module_from_spec(spec); spec.loader.exec_module(a)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve(); self.owner = self.root / 'owner'; self.journal = self.root / 'journal'
        self.owner.mkdir(); self.journal.mkdir(); (self.owner / 'qualification').mkdir()
        self.t = 1700000000.
        self.owner_pid, self.child_pid, self.replacement_pid, self.worker_pid = 20001, 20002, 30001, 30002
        numeric = HERE.parents[2] / '.artifacts/m5-ultra-256gb/realistic-qualification-v1-inputs/plan.json'
        self.numeric = json.loads(numeric.read_text())
        for dest in (self.owner / 'plan.json', self.owner / 'qualification/plan.json'):
            dest.write_bytes(numeric.read_bytes())
        self.owner_source = {'path': str(HERE / 'qualify_absent_owner.py'), 'sha256': a.OWNER_SHA}
        self.ctl_source = {'path': str(HERE.parent / 'omlx_cold_restart/controller_v2.py'), 'sha256': a.CONTROLLER_SHA}
        wrapper = {'schema': 'E023-service-absent-owner-plan-v1', 'execution': {'owner': self.owner_source},
            'qualification': {'path': str(numeric), 'sha256': a.PLAN_SHA},
            'cold_restart': {'journal_path': str(self.journal), 'controller': self.ctl_source,
                             'approved_contract': a.CONTRACT, 'proposal_plan_sha256': a.PROPOSAL_SHA}, 'deadline_seconds': 1830}
        write(self.owner / 'owner-plan.json', wrapper)
        self.wrapper_sha = a.sha(self.owner / 'owner-plan.json')
        (self.owner / 'owner-source').mkdir()
        for row in (self.owner_source, self.ctl_source):
            (self.owner / 'owner-source' / Path(row['path']).name).write_bytes(Path(row['path']).read_bytes())
        status = {'version': '0.7.0', 'default_model': a.TARGET, 'loaded': [a.TARGET], 'loading': 0,
                  'target_loading': False, 'path': a.MODEL, 'pinned': False, 'engine_type': 'vlm', 'config_model_type': 'qwen4_exp'}
        events = [self.event('prepared_v2_read_only', 0), self.event('bootout_intent', 1),
                  self.event('bootout_returned', 2), self.event('service_absent', 3)]
        self.data = {'schema': 'omlx-cold-restart-journal-v2', 'state': 'STOPPED', 'id': 'CPU-SYNTHETIC',
            'plan_sha256': a.PROPOSAL_SHA, 'controller_sha256': a.CONTROLLER_SHA, 'support_sha256': a.SUPPORT_SHA,
            'approved_contract': a.CONTRACT, 'original': copy.deepcopy(a.ORIGINAL), 'owner_directory': str(self.owner),
            'owner_source': self.owner_source, 'owner_plan': {'path': str(self.root / 'wrapper.json'), 'sha256': self.wrapper_sha},
            'removed_at': self.t + 3, 'global_settings_stat': {'dev': 1, 'ino': 1, 'size': 123, 'mtime_ns': 100},
            'model_metadata': {'CPU fixture only': 'not a model hash'}, 'original_status': status,
            'log_start': {'dev': 1, 'ino': 2, 'size': 50, 'mtime_ns': 100}, 'resource_samples': [], 'events': events}
        write(self.owner / 'stopped-journal.json', self.data)
        attempt = {'schema': 'cold-service-owner-attempt-v1', 'owner_pid': self.owner_pid,
            'owner_identity': self.identity(self.owner_pid), 'owner_source': self.owner_source, 'owner_plan_sha256': self.wrapper_sha,
            'journal_id': self.data['id'], 'journal_path': str(self.journal), 'controller_sha256': a.CONTROLLER_SHA,
            'removed_at': self.t + 3, 'created_at': self.t + 4, 'phase': 'PRESPAWN',
            'stopped_journal_sha256': a.sha(self.owner / 'stopped-journal.json')}
        write(self.owner / 'attempt.json', attempt)
        write(self.owner / 'spawn-intent.json', {'schema': 'cold-service-owner-spawn-intent-v1', 'owner_pid': self.owner_pid,
            'journal_id': self.data['id'], 'owner_plan_sha256': self.wrapper_sha, 'attempt_sha256': a.sha(self.owner / 'attempt.json'), 'time': self.t + 9.5})
        write(self.owner / 'child.json', {'pid': self.child_pid, 'owner_pid': self.owner_pid, 'spawned_at': self.t + 10})
        write(self.owner / 'release.json', {'pid': self.child_pid, 'returncode': 0, 'waited_reaped': True, 'strict_absence': True, 'time': self.t + 20})
        write(self.owner / 'child-absence.json', {'command': ['/bin/ps', '-p', str(self.child_pid), '-o', 'pid='], 'returncode': 1, 'stdout': ''})
        launch = {'command': [self.numeric['execution']['python'], '-I', '-B', self.numeric['execution']['qualifier']['path'],
            '--plan', str(numeric), '--plan-sha256', a.PLAN_SHA, '--out', str(self.owner / 'qualification')],
            'environment': self.numeric['execution']['environment'], 'numerical_plan_sha256': a.PLAN_SHA}
        write(self.owner / 'launch.json', launch)
        write(self.owner / 'complete.json', {'schema': 'E023-service-absent-owner-result-v1', 'status': 'PASS', 'errors': [],
            'owner_pid': self.owner_pid, 'owner_plan_sha256': self.wrapper_sha, 'numerical_plan_sha256': a.PLAN_SHA,
            'spawn_intent_written': True, 'child_started': True, 'samples': 4, 'absence_probes': 4, 'elapsed_wall_s': 18})
        self.qualified = {'status': 'PASS', 'plan_sha256': a.PLAN_SHA, 'numeric_only_not_http': True,
            'source_model_pins_rechecked': True, 'total_requests': 37, 'qualifier_sha256': self.numeric['execution']['qualifier']['sha256'],
            'cases': ['CPU_FAKE_math_has_separate_tests']}
        write(self.owner / 'qualification/complete.json', self.qualified)
        (self.owner / 'qualification/steps.jsonl').write_text('CPU FAKE NEVER SENT TO REAL STREAM PARSER\n')
        self.resources()
        self.recovery(status)
        write(self.journal / 'journal.json', self.data)
        self.fakebase = NS(audit_pins=lambda *unused: ([{'eos_token_ids': [99], 'normal_eos': True, 'max_output_tokens': 512}] * 8,
                               {}, {'CPU_MOCK_ONLY': True}), stream_ledger=self.stream)
        self.patch = patch.object(a, 'source_auditor', return_value=self.fakebase)
        self.patch.start(); self.addCleanup(self.patch.stop)

    def event(self, name, delta):
        return {'event': name, 'time': self.t + delta, 'monotonic': 1000 + delta}

    def identity(self, pid, service=False):
        return {'pid': pid, 'ppid': 1, 'start': 'Sat Oct 17 00:00:00 2026', 'comm': 'omlx-server' if service else 'python'}

    def absence(self):
        commands = [(['/bin/ps', '-p', '74384', '-o', 'pid='], 1), (['/bin/launchctl', 'print', a.JOB], 113),
            (['/usr/sbin/lsof', '-nP', '-a', '-iTCP:8001', '-sTCP:LISTEN', '-Fpn'], 1)]
        return {'status': 'PASS', 'checks': [{'command': cmd, 'returncode': rc, 'stdout_bytes': 0,
            'stdout_sha256': hashlib.sha256(b'').hexdigest(), 'absent': True} for cmd, rc in commands]}

    def resources(self):
        rows = []
        for i, (label, delta) in enumerate(zip(('before-prepare', 'after-prepare-before-spawn', 'child-running', 'postflight'), (5, 9, 15, 21))):
            pid = self.child_pid if i == 2 else None
            x = {'label': label, 'recorded_at': datetime.fromtimestamp(self.t + delta, timezone.utc).isoformat(),
                'monotonic_s': 1000 + delta, 'owned_model_pid': pid, 'pressure_level': {'returncode': 0, 'text': '1'},
                'pressure_level_value': 1, 'memory_pressure': {'returncode': 0, 'text': 'System-wide memory free percentage: 75%'},
                'free_percent': 75, 'swap': {'returncode': 0, 'text': 'vm.swapusage: used = 0.00M'}, 'swap_used_mb': 0.,
                'vm_stat': {'returncode': 0, 'text': 'Swapouts: 0\nPageouts: 1'}, 'vm_counters': {'Swapouts': 0, 'Pageouts': 1},
                'power_metadata': {'source': 'AC Power', 'raw': {'returncode': 0, 'stdout': "Now drawing from 'AC Power'"}},
                'processinfo_thermal_metadata': {'code': 0, 'raw': {'returncode': 0, 'stdout': '0'}},
                'process_inventory_ok': True, 'native_process_inventory_ok': True, 'no_active_foreign_native_processes': True,
                'only_owned_known_inference_processes': True, 'authorized_suspended_pids': [], 'known_inference_processes': [],
                'native_inference_processes': [] if pid is None else [{'pid': pid, 'owned': True, 'authorized_suspended': False}],
                'exact_owned_pid_metadata': {'status': 'known', 'pid': pid, 'ppid': self.owner_pid,
                    'raw': {'returncode': 0, 'stdout': f'{pid} {self.owner_pid} 123 /usr/bin/python'}}}
            write(self.owner / f'resource-{i:03d}.json', x)
            write(self.owner / f'absence-{i:03d}.json', {'label': label, 'time': self.t + delta - .01, 'resource_index_if_sampled': i, **self.absence()})
            if i:
                x.update(swap_used_mb_delta=0., vm_counter_delta={'Swapouts': 0, 'Pageouts': 0})
            rows.append(x)
        write(self.owner / 'resources.json', rows)
        write(self.owner / 'pre-spawn-absence.json', self.absence())

    def physical(self, index, label, delta, gib):
        folder = self.journal / 'evidence' / f'physical-cpu-{index}'
        path, cp = folder / 'footprint.raw.json', folder / 'commands.json'
        write(path, {'unit': 'byte', 'bytes per unit': 1, 'errors': [], 'warnings': [], 'auxiliary': {'sys_footprint': gib * a.GIB}})
        commands = [['/usr/bin/footprint', '-p', '30000', '--noCategories', '--sysFootprint', '-f', 'bytes', '-j', str(path)],
                    ['/usr/sbin/sysctl', '-n', 'kern.memorystatus_vm_pressure_level'], ['/usr/sbin/sysctl', 'vm.swapusage'],
                    ['/usr/sbin/sysctl', '-n', 'hw.memsize']]
        write(cp, [{'command': cmd, 'returncode': 0, 'stdout': text, 'stderr': '', 'elapsed_s': .1}
                   for cmd, text in zip(commands, ('', '1\n', 'used = 0.00M', str(256 * a.GIB)))])
        row = {'time': self.t + delta, 'system_footprint': gib * a.GIB, 'pressure': 1, 'swap_bytes': 0, 'physical_ram': 256 * a.GIB,
               'raw': {'path': str(path), 'sha256': a.sha(path), 'commands_path': str(cp), 'commands_sha256': a.sha(cp)}}
        write(folder / 'parsed.json', row)
        row['label'] = label
        return row

    def recovery(self, status):
        self.data['state'] = 'RESTORED'; self.data['final_status'] = status
        self.data['replacement'] = self.identity(self.replacement_pid, True)
        paths = [self.owner / n for n in ('attempt.json', 'stopped-journal.json', 'spawn-intent.json', 'child.json', 'release.json')]
        self.data['reap'] = {'no_experiment': False, 'no_child_started': False, 'owner_pid': self.owner_pid,
                             'receipts': [{'path': str(p), 'sha256': a.sha(p)} for p in paths]}
        self.data.update(exact_runtime_signature_restored=False, warm_kv_restored=False, load_baseline_swap_bytes=0,
                         post_worker_pid=self.worker_pid, load_worker={'nonce': 'CPU-nonce', 'directory': str(self.journal / 'load-worker'),
                                                                     'parent_identity': self.identity(30000)})
        events = self.data['events']; events.append(self.event('experiment_terminal_checked', 30))
        labels = [('cold-stability-1', 40, 30), ('cold-stability-2', 45, 30), ('cold-stability-1', 55, 30),
                  ('cold-stability-2', 60, 30), ('during-load-POST', 62, 120), ('load-reconciliation', 64, 180),
                  ('loaded-stability-1', 66, 180), ('loaded-stability-2', 71, 180)]
        for i, (label, delta, gib) in enumerate(labels):
            self.data['resource_samples'].append(self.physical(i, label, delta, gib))
            events.append(self.event('resource_sample_retained', delta + .01))
            if i == 1:
                events.extend([self.event('bootstrap_intent', 46), self.event('bootstrap_returned', 47), self.event('job_reconciled', 50)])
            if i == 3:
                events.extend([self.event('load_intent_monitored', 61), self.event('post_worker_started', 61.1)])
            if i == 4:
                events.append(self.event('post_worker_returned', 63))
            if i == 5:
                events.append(self.event('raw_log_window_retained', 65))
        events.append(self.event('restored_persisted_visible_variant', 72))
        write(self.journal / 'load-worker/claim.json', {'pid': self.worker_pid, 'ppid': 30000, 'nonce': 'CPU-nonce', 'time': self.t + 61.2})
        write(self.journal / 'load-worker/result.json', {'status': 'RETURNED', 'pid': self.worker_pid, 'nonce': 'CPU-nonce', 'response': {'CPU_FAKE': True}})
        write(self.journal / 'load-worker/release.json', {'pid': self.worker_pid, 'returncode': 0, 'waited_reaped': True, 'strict_absence': True,
                                                       'scope': 'HTTP client worker only'})
        self.data['post_result'] = {'path': str(self.journal / 'load-worker/result.json'), 'sha256': a.sha(self.journal / 'load-worker/result.json')}
        before = {'dev': 1, 'ino': 2, 'size': 100, 'mtime_ns': 101}; self.data['recovery_log_start'] = before
        markers = ['Qwen4-Exp PLE mode for ' + a.MODEL + ': resident', 'Qwen4-Exp Lightning MTP enabled for ' + a.MODEL,
                   'Fused 1 resident Qwen4-Exp PLE table into one packed device-side embedding',
                   'VLMBatchedEngine loaded: ' + a.MODEL, 'Loaded model: ' + a.TARGET + ' (']
        self.data['log_markers'] = markers
        raw = ('\n'.join('2026 - INFO - ' + s for s in ['Loading model: ' + a.TARGET] + markers) + '\n').encode()
        path = self.journal / 'evidence/log-cpu/window.raw.log'; path.parent.mkdir(); path.write_bytes(raw)
        receipt = {'path': str(path), 'sha256': a.sha(path), 'bytes': len(raw), 'source': a.LOG, 'before': before,
                   'captured': {**before, 'size': 100 + len(raw)}, 'start_offset': 100, 'end_offset': 100 + len(raw)}
        write(path.parent / 'window.json', receipt); self.data['log_window'] = receipt

    def stream(self, path, fixtures, config, eos, claimed, *, partial):
        self.assertFalse(partial)
        return {'cases': [{'completed': [{}] * n} for n in (8, 16, 13)], 'partial_cases': [], 'truncated_final_line': None,
                'CPU_MOCK_ONLY': True}

    def run_audit(self):
        return a.audit(self.owner, self.journal)

    def mutate(self, path, fn):
        data = a.read(path); fn(data); write(path, data)

    def test_cpu_mechanism_success_not_a_real_qualification(self):
        result = self.run_audit()
        self.assertEqual(result['status'], 'PASS', result)
        self.assertTrue(result['qualified'])
        self.assertTrue(result['pins']['CPU_MOCK_ONLY'] and result['ledger']['CPU_MOCK_ONLY'])
        self.assertEqual(result['service_restore']['physical_samples'], 8)

    def test_missing_terminal_evidence_is_incomplete_without_ledger(self):
        for path in (self.owner / 'complete.json', self.owner / 'release.json', self.journal / 'load-worker/release.json'):
            with self.subTest(path=path):
                raw = path.read_bytes(); path.unlink()
                r = self.run_audit(); self.assertEqual(r['status'], 'INCOMPLETE'); self.assertFalse(r['qualified'] or r['ledger_opened'])
                path.write_bytes(raw)

    def test_recovery_not_finished_never_opens_ledger(self):
        self.mutate(self.journal / 'journal.json', lambda d: d.update(state='LOAD_INTENT'))
        r = self.run_audit(); self.assertEqual(r['status'], 'INCOMPLETE'); self.assertFalse(r['ledger_opened'])

    def test_missing_raw_restoration_evidence_is_incomplete(self):
        Path(self.data['log_window']['path']).unlink()
        r = self.run_audit(); self.assertEqual(r['status'], 'INCOMPLETE'); self.assertFalse(r['qualified'])

    def test_owner_binding_corruptions_fail(self):
        for name, change in [('spawn-intent.json', lambda d: d.update(attempt_sha256='0'*64)),
                ('child.json', lambda d: d.update(owner_pid=999)), ('release.json', lambda d: d.update(returncode=-15)),
                ('attempt.json', lambda d: d.update(journal_path=str(self.root / 'other'))),
                ('launch.json', lambda d: d['command'].__setitem__(-1, '/wrong/out'))]:
            path = self.owner / name; raw = path.read_bytes()
            with self.subTest(name=name):
                self.mutate(path, change); r = self.run_audit(); self.assertEqual(r['status'], 'FAIL'); self.assertFalse(r['qualified'])
            path.write_bytes(raw)

    def test_numerical_receipt_cannot_shrink_37_or_drop_final_pins(self):
        path = self.owner / 'qualification/complete.json'
        for key, value in [('total_requests', 36), ('status', 'INCOMPLETE'), ('source_model_pins_rechecked', False),
                           ('numeric_only_not_http', False)]:
            write(path, {**self.qualified, key: value})
            with self.subTest(key=key):
                r = self.run_audit(); self.assertEqual(r['status'], 'FAIL'); self.assertFalse(r['qualified'])

    def test_full_ledger_counts_and_truncation_are_hard_gates(self):
        original = self.fakebase.stream_ledger
        for key, value in [('cases', [{'completed': [{}]*8}, {'completed': [{}]*15}, {'completed': [{}]*13}]),
                           ('partial_cases', [{}]), ('truncated_final_line', {'bytes': 4})]:
            def stream(*args, **kw):
                row = original(*args, **kw); row[key] = value; return row
            self.fakebase.stream_ledger = stream
            with self.subTest(key=key):
                r = self.run_audit(); self.assertEqual(r['status'], 'FAIL'); self.assertFalse(r['qualified'])

    def test_wrong_absence_command_rc_or_hash_fails(self):
        row = self.absence()
        for key, value in [('returncode', 0), ('stdout_bytes', 1), ('stdout_sha256', '0'*64), ('absent', False)]:
            bad = copy.deepcopy(row); bad['checks'][1][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): a.absence(bad)
        for key, value in [('returncode', True), ('stdout_bytes', False), ('absent', 1)]:
            bad = copy.deepcopy(row); bad['checks'][0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): a.absence(bad)

    def test_suspended_exemption_or_foreign_process_fails(self):
        path = self.owner / 'resource-002.json'; raw = path.read_bytes()
        for change in (lambda d: d.update(authorized_suspended_pids=[74384]),
                       lambda d: d['native_inference_processes'].append({'pid': 74384, 'owned': False, 'authorized_suspended': True}),
                       lambda d: d['known_inference_processes'].append({'pid': 111, 'owned': False})):
            self.mutate(path, change)
            self.assertEqual(self.run_audit()['status'], 'FAIL')
            path.write_bytes(raw)

    def test_resource_phase_or_raw_pressure_change_fails(self):
        path = self.owner / 'resource-001.json'; raw = path.read_bytes()
        for change in (lambda d: d.update(label='child-running'), lambda d: d.update(pressure_level_value=2),
                       lambda d: d['pressure_level'].update(text='2')):
            self.mutate(path, change)
            self.assertEqual(self.run_audit()['status'], 'FAIL')
            path.write_bytes(raw)

    def test_raw_physical_units_global_field_and_command_pin(self):
        row = copy.deepcopy(self.data['resource_samples'][0]); path = Path(row['raw']['path']); original = path.read_bytes()
        for change in (lambda f: f.update(unit='KiB'), lambda f: f.update(warnings=['bad']),
                       lambda f: f['auxiliary'].update(sys_footprint=True)):
            self.mutate(path, change); row['raw']['sha256'] = a.sha(path)
            with self.assertRaises(ValueError): a.physical(row, self.journal)
            path.write_bytes(original)

    def test_all_post_load_samples_obey_bound_and_swap(self):
        for key, value in [('pressure', 2), ('system_footprint', 241*a.GIB), ('swap_bytes', 1)]:
            data = copy.deepcopy(self.data); data['resource_samples'][4][key] = value
            # Bypass raw/parser equality only for this narrow gate corruption test.
            with patch.object(a, 'physical', return_value={}), self.subTest(key=key), self.assertRaises(ValueError):
                a.restore_evidence(data, self.journal, self.t + 20)

    def test_physical_readings_cannot_precede_reap_or_their_phase(self):
        for change in ('all-stale', 'post-before-load', 'future', 'reordered'):
            data = copy.deepcopy(self.data)
            if change == 'all-stale':
                for row in data['resource_samples']: row['time'] -= 1000
            elif change == 'post-before-load': data['resource_samples'][4]['time'] = self.t + 60.5
            elif change == 'future': data['resource_samples'][4]['time'] = self.t + 63
            else: data['resource_samples'][0], data['resource_samples'][1] = data['resource_samples'][1], data['resource_samples'][0]
            # Matching parsed.json timestamps cannot make stale raw evidence current.
            changed = []
            for row in data['resource_samples']:
                path = Path(row['raw']['path']).parent / 'parsed.json'
                changed.append((path, path.read_bytes()))
                write(path, {k: v for k, v in row.items() if k != 'label'})
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, 'chronological|window'):
                a.restore_evidence(data, self.journal, self.t + 21)
            for path, raw in changed: path.write_bytes(raw)

    def test_sustained_swap_guard_is_independently_recomputed(self):
        aggregate = a.read(self.owner / 'resources.json')
        for i, swap in ((2, 70.), (3, 140.)):
            path = self.owner / f'resource-{i:03d}.json'
            row = a.read(path); row['swap_used_mb'] = swap; row['swap']['text'] = f'vm.swapusage: used = {swap:.2f}M'
            write(path, row)
            aggregate[i] = {**row, 'swap_used_mb_delta': 70., 'vm_counter_delta': {'Swapouts': 0, 'Pageouts': 0}}
        write(self.owner / 'resources.json', aggregate)
        result = self.run_audit()
        self.assertEqual(result['status'], 'FAIL')
        self.assertIn('sustained swap', result['first_failure']['message'])

    def test_gate_interval_and_stability(self):
        rows = copy.deepcopy(self.data['resource_samples'][:2])
        a.physical_gate(rows)
        for change in (lambda r: r[1].update(time=r[0]['time']+3),
                       lambda r: r[1].update(system_footprint=r[0]['system_footprint']+2*a.GIB),
                       lambda r: r[1].update(swap_bytes=1)):
            bad = copy.deepcopy(rows); change(bad)
            with self.assertRaises(ValueError): a.physical_gate(bad)

    def test_no_child_terminal_and_partial_never_pass(self):
        write(self.owner / 'terminal.json', {'status': 'NO_CHILD_STARTED'})
        self.assertEqual(self.run_audit()['status'], 'FAIL')

    def test_wrong_controller_reap_binding_fails(self):
        self.mutate(self.journal / 'journal.json', lambda d: d['reap'].update(owner_pid=999))
        self.assertEqual(self.run_audit()['status'], 'FAIL')

    def test_controller_reap_before_owner_postflight_fails(self):
        def corrupt(d):
            for event in d['events']:
                if event['event'] == 'experiment_terminal_checked':
                    event['time'] = self.t + 20.5
        self.mutate(self.journal / 'journal.json', corrupt)
        self.assertEqual(self.run_audit()['status'], 'FAIL')

    def test_physical_event_count_or_raw_path_escape_fails(self):
        data = copy.deepcopy(self.data)
        data['events'] = [e for e in data['events'] if not (e['event'] == 'resource_sample_retained' and e['time'] == self.t + 40.01)]
        with self.assertRaises(ValueError): a.restore_evidence(data, self.journal, self.t + 21)
        row = copy.deepcopy(self.data['resource_samples'][0]); row['raw']['path'] = str(self.root / 'outside.json')
        write(self.root / 'outside.json', {})
        with self.assertRaises(ValueError): a.physical(row, self.journal)

    def test_new_log_requires_resident_mtp_and_same_window(self):
        for mutation in ('missing-mtp', 'ssd', 'offset', 'rotation'):
            data = copy.deepcopy(self.data); row = data['log_window']; path = Path(row['path']); original = path.read_bytes()
            receipt = path.parent / 'window.json'; receipt_original = receipt.read_bytes()
            if mutation == 'missing-mtp': path.write_bytes(original.replace(b'Lightning MTP enabled', b'Lightning MTP missing'))
            elif mutation == 'ssd': path.write_bytes(original + b'2026 - INFO - PLE: ssd\n')
            elif mutation == 'offset': row['start_offset'] += 1
            else: row['captured']['ino'] += 1
            if mutation in ('missing-mtp', 'ssd'):
                row.update(sha256=a.sha(path), bytes=path.stat().st_size, end_offset=100+path.stat().st_size)
                row['captured']['size'] = row['end_offset']
            write(receipt, row)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                a.restore_evidence(data, self.journal, self.t + 20)
            path.write_bytes(original); receipt.write_bytes(receipt_original)

    def test_worker_nonce_and_reap_corruptions_fail(self):
        for name, change in [('claim.json', lambda d: d.update(nonce='wrong')), ('release.json', lambda d: d.update(strict_absence=False))]:
            path = self.journal / 'load-worker' / name; original = path.read_bytes()
            self.mutate(path, change)
            with self.subTest(name=name): self.assertEqual(self.run_audit()['status'], 'FAIL')
            path.write_bytes(original)

    def test_help_and_unknown_never_execute_audit(self):
        with patch.object(a, 'audit', side_effect=AssertionError('must not audit')), patch('sys.stdout', new=io.StringIO()), patch('sys.stderr', new=io.StringIO()):
            with self.assertRaises(SystemExit) as help_result: a.main(['--help'])
            self.assertEqual(help_result.exception.code, 0)
            with self.assertRaises(SystemExit) as unknown: a.main(['--unknown'])
            self.assertEqual(unknown.exception.code, 2)

    def test_new_receipt_only_and_incomplete_exit(self):
        out = self.root / 'audit.json'; (self.owner / 'complete.json').unlink()
        argv = ['--owner', str(self.owner), '--journal', str(self.journal), '--out', str(out)]
        self.assertEqual(a.main(argv), 2)
        self.assertFalse(a.read(out)['qualified'])
        before = out.read_bytes()
        with self.assertRaises(ValueError): a.main(argv)
        self.assertEqual(out.read_bytes(), before)

    def test_frozen_numerical_auditor_selftest_without_measurement_import(self):
        self.patch.stop()
        base = a.source_auditor()
        base.self_test()
        import sys
        self.assertFalse(any(n == 'mlx' or n.startswith('mlx.') for n in sys.modules))

    def test_actual_frozen_owner_produces_archived_source_layout(self):
        # Exercise actual run() with the existing fake OS/resource/prepare backend.
        # Unlike the audit fixture above, this test never creates owner-source.
        path = HERE / 'test_qualify_absent_owner_cpu.py'
        need_sha = 'e643bfb26e7aad617d52ee9c2572deed7296085e7ee910c716d4d17dd38ab5a2'
        self.assertEqual(a.sha(path), need_sha)
        spec = importlib.util.spec_from_file_location('frozen_owner_cpu_fixture', path)
        helpers = importlib.util.module_from_spec(spec); spec.loader.exec_module(helpers)
        fixture = helpers.OwnerTests('test_success_same_child_plan_and_reap')
        fixture.setUp()
        try:
            self.assertFalse(fixture.out.exists())
            self.assertEqual(fixture.run_owner(), 0)
            folder = fixture.out / 'owner-source'
            self.assertEqual({p.name for p in folder.iterdir()}, {'qualify_absent_owner.py', 'controller_v2.py'})
            self.assertEqual(a.sha(folder / 'qualify_absent_owner.py'), a.OWNER_SHA)
            self.assertEqual(a.sha(folder / 'controller_v2.py'), a.CONTROLLER_SHA)
            self.assertEqual(a.read(fixture.out / 'complete.json')['status'], 'PASS')
        finally:
            fixture.doCleanups()


if __name__ == '__main__':
    unittest.main()
