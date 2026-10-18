"""Independent stdlib audit of a completed E023 service-absent run.

--owner DIR --journal DIR --out NEW_JSON. Missing terminal evidence is INCOMPLETE;
only the unchanged 37-request numerical workload can be qualified. Reuses the
source-pinned original numerical auditor, never measurement/controller code.
Resources and new load-log bytes are independently recomputed. Final HTTP and
PID/job/listener observations were retained only as pinned-controller receipts;
this audit does not invent their raw responses or make a current live claim.
No GPU imports, HTTP, process probes, service actions, or large JSONL read_text.
"""
import argparse
from datetime import datetime
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
AUDITOR_SHA = 'b6d90413911f2b17e0951a2fcfdb5fe00666aa06d4382b603e29e5dc4659d4f1'
OWNER_SHA = '5beb6c9bcfa4e22041a386cdbea3f5ecee210cdcb9b345512c50b6c6f873c62d'
CONTROLLER_SHA = 'f06d0ba60e0966fe83741f9c5e60b74404a84154242760d005170a47327496be'
SUPPORT_SHA = 'aad297fb08515d6304fd3ae435301318650e8443ad22628c1ed2e4791887f0f3'
PLAN_SHA = '21ba52e43ce54c2b463ab46bfd212c53d06d6303530c77939e3e6b801df547fb'
PROPOSAL_SHA = 'e96a4a4d5be6f6a69d597387bbc566eadedae81065bfa879445352d986515cb8'
CONTRACT = 'cold-restart-persisted-visible-v1'
TARGET = 'Qwen3.8-Flash-Next-oQ6e-mtp'
MODEL = '/Users/ryanli/code/models/mlx/' + TARGET
LOG = '/Users/ryanli/.omlx/logs/service.stderr.log'
JOB = 'gui/501/local.omlx.server'
GIB = 1 << 30
ORIGINAL = {'pid': 74384, 'ppid': 1, 'start': 'Fri Oct 16 08:01:58 2026', 'comm': 'omlx-server'}
SCOPE = 'Fixed E023 numerical traces only; no HTTP, performance, or E024 qualification. Recovery receipts plus independently checked raw physical/log evidence; no current live probe.'


class Incomplete(Exception):
    pass


def need(ok, message):
    if not ok:
        raise ValueError(message)


def finite(x):
    return type(x) in (int, float) and math.isfinite(x)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while part := f.read(1 << 20):
            h.update(part)
    return h.hexdigest()


def require(path):
    path = Path(path)
    if not path.is_file():
        raise Incomplete('missing evidence: ' + str(path))
    return path


def read(path):
    path = require(path)
    need(path.suffix != '.jsonl', 'ledger must be streamed')
    with path.open() as f:
        return json.load(f)


def within(path, root):
    path, root = Path(path), Path(root).resolve()
    need(path.is_absolute() and '..' not in path.parts and path.resolve().is_relative_to(root), 'artifact escaped expected directory')
    return require(path)


def source_auditor():
    path = HERE / 'audit_qualification.py'
    need(sha(path) == AUDITOR_SHA, 'frozen numerical auditor changed')
    spec = importlib.util.spec_from_file_location('_e023_numerical_audit_only', path)
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)
    return base


def status_receipt(row):
    expected = {'version': '0.7.0', 'default_model': TARGET, 'loaded': [TARGET], 'loading': 0,
                'target_loading': False, 'path': MODEL, 'pinned': False,
                'engine_type': 'vlm', 'config_model_type': 'qwen4_exp'}
    need(row == expected and type(row['loading']) is int and row['target_loading'] is False
         and row['pinned'] is False, 'restored persisted-visible status receipt mismatch')


def identity(row, *, pid=None, service=False):
    need(set(row) == {'pid', 'ppid', 'start', 'comm'} and type(row['pid']) is int and row['pid'] > 1
         and type(row['ppid']) is int and row['ppid'] >= 1 and isinstance(row['start'], str) and row['start']
         and isinstance(row['comm'], str) and row['comm'], 'process identity receipt malformed')
    need(pid is None or row['pid'] == pid, 'process identity PID mismatch')
    if service:
        need(row['ppid'] == 1 and row['comm'] == 'omlx-server' and row['pid'] != 74384, 'replacement service identity')


def owner_binding(owner, journal, data):
    complete = read(owner / 'complete.json')
    need(complete['schema'] == 'E023-service-absent-owner-result-v1' and complete['status'] == 'PASS'
         and complete['errors'] == [] and complete['spawn_intent_written'] is True and complete['child_started'] is True,
         'owner is not a completed successful child run')
    need(not (owner / 'terminal.json').exists(), 'no-child terminal cannot accompany numerical PASS')
    wrapper = read(owner / 'owner-plan.json')
    wrapper_sha = sha(owner / 'owner-plan.json')
    need(set(wrapper) == {'schema', 'execution', 'qualification', 'cold_restart', 'deadline_seconds'}
         and wrapper['schema'] == 'E023-service-absent-owner-plan-v1', 'wrapper schema/fields')
    need(type(wrapper['deadline_seconds']) in (int, float) and finite(wrapper['deadline_seconds'])
         and 1800 <= wrapper['deadline_seconds'] <= 1830, 'owner work deadline')
    need(finite(complete['elapsed_wall_s']) and 0 < complete['elapsed_wall_s'] <= wrapper['deadline_seconds'] + 65,
         'owner exceeded work/cleanup bounds')
    expected_owner = {'path': str(HERE / 'qualify_absent_owner.py'), 'sha256': OWNER_SHA}
    expected_ctl = {'path': str(HERE.parent / 'omlx_cold_restart/controller_v2.py'), 'sha256': CONTROLLER_SHA}
    need(wrapper['execution'] == {'owner': expected_owner}, 'fixed owner source binding')
    need(wrapper['cold_restart'] == {'journal_path': str(journal), 'controller': expected_ctl,
                                    'approved_contract': CONTRACT, 'proposal_plan_sha256': PROPOSAL_SHA}, 'cold restart binding')
    for row in (expected_owner, expected_ctl):
        need(sha(owner / 'owner-source' / Path(row['path']).name) == row['sha256'], 'archived owner/controller source hash')
    q = wrapper['qualification']
    need(set(q) == {'path', 'sha256'} and Path(q['path']).is_absolute() and q['sha256'] == PLAN_SHA, 'unchanged numerical plan binding')
    need(complete['owner_plan_sha256'] == wrapper_sha and complete['numerical_plan_sha256'] == PLAN_SHA, 'owner final plan bindings')
    frozen = read(owner / 'stopped-journal.json')
    need(frozen['state'] == 'STOPPED' and data['state'] == 'RESTORED', 'STOPPED-to-RESTORED states required')
    expected = {'schema': 'omlx-cold-restart-journal-v2', 'plan_sha256': PROPOSAL_SHA,
                'controller_sha256': CONTROLLER_SHA, 'support_sha256': SUPPORT_SHA, 'approved_contract': CONTRACT,
                'original': ORIGINAL, 'owner_directory': str(owner), 'owner_source': expected_owner}
    need(all(frozen.get(k) == data.get(k) == v for k, v in expected.items()), 'journal stable controller/service binding')
    need(frozen['owner_plan'] == data['owner_plan'] and data['owner_plan']['sha256'] == wrapper_sha
         and Path(data['owner_plan']['path']).is_absolute(), 'journal wrapper plan binding')
    for key in ('id', 'removed_at', 'global_settings_stat', 'model_metadata', 'original_status', 'log_start'):
        need(frozen[key] == data[key], 'journal stable field changed: ' + key)
    need(isinstance(data['id'], str) and data['id'] and finite(data['removed_at']) and data['removed_at'] > 0, 'journal runtime identity')
    status_receipt(data['original_status'])
    need(data['events'][:len(frozen['events'])] == frozen['events'], 'STOPPED event prefix was rewritten')
    a, intent, child, release = [read(owner / name) for name in ('attempt.json', 'spawn-intent.json', 'child.json', 'release.json')]
    owner_pid = complete['owner_pid']; identity(a['owner_identity'], pid=owner_pid)
    need(a['schema'] == 'cold-service-owner-attempt-v1' and a['owner_pid'] == owner_pid
         and a['owner_source'] == expected_owner and a['owner_plan_sha256'] == wrapper_sha
         and a['journal_id'] == data['id'] and a['journal_path'] == str(journal)
         and a['controller_sha256'] == CONTROLLER_SHA and a['removed_at'] == data['removed_at']
         and a['phase'] == 'PRESPAWN' and a['stopped_journal_sha256'] == sha(owner / 'stopped-journal.json'), 'durable attempt binding')
    need(intent['schema'] == 'cold-service-owner-spawn-intent-v1' and intent['owner_pid'] == owner_pid
         and intent['journal_id'] == data['id'] and intent['owner_plan_sha256'] == wrapper_sha
         and intent['attempt_sha256'] == sha(owner / 'attempt.json'), 'durable spawn intent binding')
    pid = child['pid']
    need(type(pid) is int and pid > 1 and pid not in (owner_pid, 74384) and child['owner_pid'] == owner_pid, 'direct child binding')
    need(release['pid'] == pid and type(release['returncode']) is int and release['returncode'] == 0
         and release['waited_reaped'] is True and release['strict_absence'] is True, 'numerical child exit/reap')
    times = [data['removed_at'], a['created_at'], intent['time'], child['spawned_at'], release['time']]
    need(all(finite(t) for t in times) and times == sorted(times), 'owner chronological boundaries')
    raw = read(owner / 'child-absence.json')
    need(type(raw['returncode']) is int and raw == {'command': ['/bin/ps', '-p', str(pid), '-o', 'pid='], 'returncode': 1, 'stdout': ''}, 'strict raw child absence')
    launch = read(owner / 'launch.json')
    plan = read(owner / 'plan.json')
    need(sha(owner / 'plan.json') == sha(owner / 'qualification/plan.json') == PLAN_SHA, 'numerical saved plan SHA')
    need(launch['command'] == [plan['execution']['python'], '-I', '-B', plan['execution']['qualifier']['path'],
                              '--plan', q['path'], '--plan-sha256', PLAN_SHA, '--out', str(owner / 'qualification')]
         and launch['environment'] == plan['execution']['environment'] and launch['numerical_plan_sha256'] == PLAN_SHA,
         'exact numerical child command/environment')
    reap = data['reap']
    paths = [owner / n for n in ('attempt.json', 'stopped-journal.json', 'spawn-intent.json', 'child.json', 'release.json')]
    need(reap['no_experiment'] is False and reap['no_child_started'] is False and reap == {'no_experiment': False, 'no_child_started': False, 'owner_pid': owner_pid,
                  'receipts': [{'path': str(p), 'sha256': sha(p)} for p in paths]}, 'controller did not bind this owner reap')
    return complete, plan, launch, owner_pid, pid, child['spawned_at'], release['time'], a['created_at']


def absence(row):
    commands = [(['/bin/ps', '-p', '74384', '-o', 'pid='], 1),
                (['/bin/launchctl', 'print', JOB], 113),
                (['/usr/sbin/lsof', '-nP', '-a', '-iTCP:8001', '-sTCP:LISTEN', '-Fpn'], 1)]
    need(row['status'] == 'PASS' and len(row['checks']) == 3, 'service absence failed/incomplete')
    for r, (command, rc) in zip(row['checks'], commands):
        need(type(r['returncode']) is int and type(r['stdout_bytes']) is int and r['absent'] is True
             and r == {'command': command, 'returncode': rc, 'stdout_sha256': hashlib.sha256(b'').hexdigest(),
                   'stdout_bytes': 0, 'absent': True}, 'raw service absence code/output proof')


def owner_resources(owner, complete, owner_pid, child, spawned, released, created):
    count = complete['samples']
    need(type(count) is int and count >= 4 and complete['absence_probes'] == count, 'successful probe coverage')
    need(sorted(p.name for p in owner.glob('resource-*.json')) == [f'resource-{i:03d}.json' for i in range(count)], 'resource sequence')
    need(sorted(p.name for p in owner.glob('absence-*.json')) == [f'absence-{i:03d}.json' for i in range(count)], 'absence sequence')
    rows, known, deltas = [], 0, []
    for i in range(count):
        x, a = read(owner / f'resource-{i:03d}.json'), read(owner / f'absence-{i:03d}.json')
        label = 'before-prepare' if i == 0 else 'after-prepare-before-spawn' if i == 1 else 'postflight' if i == count - 1 else 'child-running'
        need(x['label'] == a['label'] == label and a['resource_index_if_sampled'] == i, 'resource/absence phase pairing')
        absence(a)
        wall = datetime.fromisoformat(x['recorded_at']).timestamp()
        need(finite(a['time']) and created <= a['time'] <= wall and finite(x['monotonic_s']), 'resource timestamp')
        if rows:
            need(x['monotonic_s'] > rows[-1]['monotonic_s'], 'resource monotonic ordering')
        need(x['pressure_level']['returncode'] == 0 and x['pressure_level']['text'] == '1'
             and type(x['pressure_level_value']) is int and x['pressure_level_value'] == 1, 'raw pressure must be 1')
        free = re.search(r'free percentage:\s*(\d+)', x['memory_pressure']['text'])
        swap = re.search(r'used = ([0-9.]+)M', x['swap']['text'])
        need(free and swap and x['memory_pressure']['returncode'] == x['swap']['returncode'] == x['vm_stat']['returncode'] == 0,
             'raw memory/swap/counter probe')
        need(int(free[1]) == x['free_percent'] and 0 <= x['free_percent'] <= 100
             and finite(x['swap_used_mb']) and x['swap_used_mb'] >= 0 and float(swap[1]) == x['swap_used_mb'], 'memory/swap parse')
        need({'Swapouts', 'Pageouts'} <= set(x['vm_counters']), 'required counters absent')
        for key, value in x['vm_counters'].items():
            match = re.search(r'^' + re.escape(key) + r':\s*(\d+)', x['vm_stat']['text'], re.M)
            need(type(value) is int and value >= 0 and match and int(match[1]) == value, 'raw VM counter mismatch')
        power, thermal = x['power_metadata'], x['processinfo_thermal_metadata']
        need(power['source'] == 'AC Power' and power['raw']['returncode'] == 0 and 'AC Power' in power['raw']['stdout']
             and type(thermal['code']) is int and thermal['code'] in (0, 1) and thermal['raw']['returncode'] == 0
             and thermal['raw']['stdout'] == str(thermal['code']), 'power/thermal raw mismatch')
        need(x['process_inventory_ok'] is True and x['native_process_inventory_ok'] is True
             and x['no_active_foreign_native_processes'] is True and x['only_owned_known_inference_processes'] is True
             and x['authorized_suspended_pids'] == [], 'inventory/empty suspended guard')
        for entry in x['native_inference_processes']:
            need(label == 'child-running' and entry['pid'] == child and entry['owned'] is True
                 and entry['authorized_suspended'] is False, 'foreign native inference or suspended exemption')
        for entry in x['known_inference_processes']:
            need(label == 'child-running' and entry['pid'] == child and entry['owned'] is True, 'foreign recognized inference')
        need(x['owned_model_pid'] == (child if label == 'child-running' else None), 'resource owned PID target')
        if label == 'child-running':
            need(spawned <= wall <= released, 'child probe outside child lifetime')
            exact = x['exact_owned_pid_metadata']
            if exact['status'] == 'known':
                fields = exact['raw']['stdout'].split(None, 3)
                need(exact['pid'] == child and exact['ppid'] == owner_pid and exact['raw']['returncode'] == 0
                     and len(fields) == 4 and [int(fields[0]), int(fields[1])] == [child, owner_pid], 'raw direct child ownership')
                known += 1
            else:
                need(i == count - 2 and exact['raw']['returncode'] == 1 and not exact['raw']['stdout'].strip(), 'unexplained child disappearance')
        elif label == 'postflight':
            need(wall >= released, 'postflight before child reap')
        else:
            need(wall <= spawned, 'preflight after child spawn')
        if rows:
            previous = rows[-1]
            d = {'swap': x['swap_used_mb'] - previous['swap_used_mb'],
                 'Swapouts': x['vm_counters']['Swapouts'] - previous['vm_counters']['Swapouts']}
            deltas.append(d)
            need(not (len(deltas) >= 2 and (all(d['swap'] > 64 for d in deltas[-2:])
                       or all(d['Swapouts'] > 0 for d in deltas[-2:]))), 'sustained swap guard violation')
        rows.append(x)
    need(known > 0, 'no direct child observation')
    aggregate = read(owner / 'resources.json')
    need(len(aggregate) == count, 'aggregate resource count')
    for i, (raw, full) in enumerate(zip(rows, aggregate)):
        need({k: v for k, v in full.items() if k not in ('swap_used_mb_delta', 'vm_counter_delta')} == raw, 'raw/aggregate mismatch')
        if i:
            need(full['swap_used_mb_delta'] == raw['swap_used_mb'] - rows[i - 1]['swap_used_mb']
                 and full['vm_counter_delta'] == {k: v - rows[i - 1]['vm_counters'][k] for k, v in raw['vm_counters'].items()}, 'aggregate counter delta')
    absence(read(owner / 'pre-spawn-absence.json'))
    return {'samples': count, 'absence_probes': count + 1, 'known_owned_samples': known,
            'pressure': 1, 'authorized_suspended_pids': [],
            'last_recorded_at': datetime.fromisoformat(rows[-1]['recorded_at']).timestamp(),
            'swap_delta_mb': rows[-1]['swap_used_mb'] - rows[0]['swap_used_mb'],
            'swapouts_delta': rows[-1]['vm_counters']['Swapouts'] - rows[0]['vm_counters']['Swapouts']}


def physical(row, journal):
    raw = row['raw']; path = within(raw['path'], journal / 'evidence')
    commands_path = within(raw['commands_path'], journal / 'evidence')
    need(path.parent == commands_path.parent and sha(path) == raw['sha256'] and sha(commands_path) == raw['commands_sha256'], 'physical raw binding')
    f, commands = read(path), read(commands_path)
    need(f['unit'] == 'byte' and type(f['bytes per unit']) is int and f['bytes per unit'] == 1
         and f['errors'] == f['warnings'] == [], 'physical units/errors/warnings')
    need(len(commands) == 4 and all(type(c['returncode']) is int and c['returncode'] == 0
         and finite(c['elapsed_s']) and 0 <= c['elapsed_s'] <= 11 for c in commands), 'physical probe command coverage')
    first = commands[0]['command']
    need(len(first) == 9 and first[:2] == ['/usr/bin/footprint', '-p'] and first[2].isdigit()
         and int(first[2]) > 1 and first[3:] == ['--noCategories', '--sysFootprint', '-f', 'bytes', '-j', str(path)], 'physical command target')
    expected = [['/usr/sbin/sysctl', '-n', 'kern.memorystatus_vm_pressure_level'],
                ['/usr/sbin/sysctl', 'vm.swapusage'], ['/usr/sbin/sysctl', '-n', 'hw.memsize']]
    need([c['command'] for c in commands[1:]] == expected, 'physical sysctl command identities')
    value = f['auxiliary']['sys_footprint']; pressure, swap, ram = [c['stdout'].strip() for c in commands[1:]]
    match = re.search(r'used = ([\d.]+)M', swap)
    need(type(value) is int and value > 0 and pressure.isdigit() and ram.isdigit()
         and match and math.isfinite(float(match[1])) and finite(row['time']), 'physical raw values')
    parsed = {'system_footprint': value, 'pressure': int(pressure),
              'swap_bytes': round(float(match[1]) * (1 << 20)), 'physical_ram': int(ram)}
    need(all(type(row[k]) is int and row[k] == v for k, v in parsed.items()), 'physical parsed/raw mismatch')
    need(read(path.parent / 'parsed.json') == {k: v for k, v in row.items() if k != 'label'}, 'physical standalone receipt mismatch')
    return parsed


def physical_gate(rows, *, loaded=False):
    need(len(rows) == 2 and [r['label'] for r in rows] ==
         (['loaded-stability-1', 'loaded-stability-2'] if loaded else ['cold-stability-1', 'cold-stability-2']), 'two-sample physical gate labels')
    limit = 240 * GIB if loaded else 40 * GIB
    for r in rows:
        need(r['physical_ram'] == 256 * GIB and r['pressure'] == 1 and 0 < r['system_footprint'] <= limit
             and r['swap_bytes'] >= 0, 'physical admission bound')
    need(4 <= rows[1]['time'] - rows[0]['time'] <= 60 and rows[1]['swap_bytes'] <= rows[0]['swap_bytes']
         and abs(rows[1]['system_footprint'] - rows[0]['system_footprint']) <= GIB, 'physical stability interval/swap')


def restore_evidence(data, journal, release_time):
    need(data['state'] == 'RESTORED' and not data.get('post_monitor_failed')
         and data.get('recovery_complete', True) is True
         and data['exact_runtime_signature_restored'] is False and data['warm_kv_restored'] is False, 'recovery completion/scope')
    status_receipt(data['final_status']); identity(data['replacement'], service=True)
    events, rows = data['events'], data['resource_samples']
    need(events and all(finite(e['time']) and finite(e['monotonic']) for e in events), 'event timestamps')
    need(all(b['monotonic'] >= a['monotonic'] for a, b in zip(events, events[1:])), 'journal event order')
    positions = {}
    for name in ('bootstrap_intent', 'job_reconciled', 'load_intent_monitored', 'post_worker_started', 'raw_log_window_retained', 'restored_persisted_visible_variant'):
        indices = [i for i, e in enumerate(events) if e['event'] == name]
        need(len(indices) == 1, 'missing/duplicate recovery event: ' + name)
        positions[name] = indices[0]
    need(list(positions.values()) == sorted(positions.values()), 'recovery event ordering')
    need(events[positions['bootstrap_intent']]['time'] >= release_time, 'bootstrap preceded numerical child reap')
    reap_events = [e for e in events[:positions['bootstrap_intent']] if e['event'] == 'experiment_terminal_checked']
    need(reap_events and reap_events[-1]['time'] >= release_time, 'controller reap check preceded owner postflight')
    sample_events = [i for i, e in enumerate(events) if e['event'] == 'resource_sample_retained']
    need(len(sample_events) == len(rows) and len(rows) >= 8, 'retained physical sample coverage')
    need(len({r['raw']['path'] for r in rows}) == len(rows)
         and len({r['raw']['commands_path'] for r in rows}) == len(rows), 'physical probe artifacts reused across samples')
    parsed = [physical(r, journal) for r in rows]
    need(all(b['time'] >= a['time'] for a, b in zip(rows, rows[1:])), 'physical sample chronological order')
    need(all(index > 0 and release_time <= row['time']
             and events[index - 1]['time'] <= row['time'] <= events[index]['time']
             for row, index in zip(rows, sample_events)), 'physical sample outside owner-reap/prior-event/retained-event window')
    for name in ('bootstrap_intent', 'load_intent_monitored', 'restored_persisted_visible_variant'):
        count = sum(i < positions[name] for i in sample_events)
        gate = rows[count - 2:count]
        physical_gate(gate, loaded=name == 'restored_persisted_visible_variant')
        lower = (reap_events[-1]['time'] if name == 'bootstrap_intent' else
                 events[positions['job_reconciled' if name == 'load_intent_monitored' else 'raw_log_window_retained']]['time'])
        need(all(lower <= row['time'] <= events[positions[name]]['time'] for row in gate), 'physical gate outside its recovery phase')
    before_load = sum(i < positions['load_intent_monitored'] for i in sample_events)
    need(type(data['load_baseline_swap_bytes']) is int and data['load_baseline_swap_bytes'] == rows[before_load - 1]['swap_bytes'], 'POST baseline swap binding')
    need(all(r['label'] in ('cold-stability-1', 'cold-stability-2') for r in rows[:before_load]), 'unexpected pre-load physical phase')
    after = rows[before_load:]
    need(any(r['label'] == 'during-load-POST' for r in after) and any(r['label'] == 'load-reconciliation' for r in after), 'POST and reconciliation monitoring missing')
    need(all(r['label'] in ('during-load-POST', 'load-reconciliation', 'loaded-stability-1', 'loaded-stability-2') for r in after), 'unexpected post-load phase')
    for r in after:
        need(r['physical_ram'] == 256 * GIB and r['pressure'] == 1 and 0 < r['system_footprint'] <= 240 * GIB
             and 0 <= r['swap_bytes'] <= data['load_baseline_swap_bytes'], 'post-load physical/swap violation')
    worker = data['load_worker']; identity(worker['parent_identity'])
    need(worker['directory'] == str(journal / 'load-worker'), 'fixed HTTP worker folder')
    claim, result, release = [read(journal / 'load-worker' / n) for n in ('claim.json', 'result.json', 'release.json')]
    pid = data['post_worker_pid']
    need(type(pid) is int and pid > 1 and pid != data['replacement']['pid']
         and claim['pid'] == result['pid'] == release['pid'] == pid and claim['ppid'] == worker['parent_identity']['pid']
         and claim['nonce'] == result['nonce'] == worker['nonce'], 'HTTP worker direct ownership/nonce')
    need(release['waited_reaped'] is True and release['strict_absence'] is True and type(release['returncode']) is int
         and release['scope'] == 'HTTP client worker only', 'HTTP worker reap receipt')
    need(result['status'] in ('RETURNED', 'UNKNOWN')
         and release['returncode'] == (0 if result['status'] == 'RETURNED' else 1), 'HTTP worker result/exit')
    need(finite(claim['time']) and events[positions['load_intent_monitored']]['time'] <= claim['time']
         <= events[positions['restored_persisted_visible_variant']]['time'], 'HTTP worker claim outside recovery')
    need(data['post_result'] == {'path': str(journal / 'load-worker/result.json'), 'sha256': sha(journal / 'load-worker/result.json')}, 'HTTP result SHA binding')
    log = data['log_window']; path = within(log['path'], journal / 'evidence')
    need(path.stat().st_size <= 4 << 20, 'log window bound')
    raw = path.read_bytes()
    need(log['source'] == LOG and log['sha256'] == hashlib.sha256(raw).hexdigest() and log['bytes'] == len(raw)
         and log['before'] == data['recovery_log_start'], 'new log window provenance')
    need(log['start_offset'] == log['before']['size'] and log['end_offset'] == log['captured']['size']
         and log['end_offset'] - log['start_offset'] == len(raw)
         and (log['before']['dev'], log['before']['ino']) == (log['captured']['dev'], log['captured']['ino']), 'new log offsets/rotation')
    need(read(path.parent / 'window.json') == log, 'standalone log receipt')
    lines = raw.decode('utf-8', errors='strict').splitlines()
    starts = [i for i, line in enumerate(lines) if ' - INFO - Loading model: ' + TARGET in line]
    need(len(starts) == 1, 'new target load transaction missing/ambiguous')
    lines = lines[starts[0]:]
    markers = ['Qwen4-Exp PLE mode for ' + MODEL + ': resident', 'Qwen4-Exp Lightning MTP enabled for ' + MODEL,
               'Fused 1 resident Qwen4-Exp PLE table into one packed device-side embedding',
               'VLMBatchedEngine loaded: ' + MODEL, 'Loaded model: ' + TARGET + ' (']
    need(data['log_markers'] == markers and all(any(' - INFO - ' in line and marker in line for line in lines) for marker in markers), 'resident/MTP/new-load markers')
    need(not any('PLE' in line and ('forced to SSD' in line or ': mmap' in line or ': ssd' in line) for line in lines), 'SSD/mmap PLE recovery not accepted')
    return {'status': 'PERSISTED_VISIBLE_RECEIPTS_CONSISTENT', 'physical_samples': len(parsed),
            'max_recorded_system_footprint_bytes': max(r['system_footprint'] for r in rows),
            'raw_log_bytes': len(raw), 'raw_log_sha256': log['sha256'], 'replacement': data['replacement'],
            'http_worker_returncode': release['returncode'], 'exact_runtime_signature_restored': False, 'warm_kv_restored': False,
            'scope': 'Raw physical/log checks independent; final HTTP status and PID/job/listener observations exist only as pinned-controller receipts.'}


def audit(owner, journal):
    owner, journal = Path(owner).resolve(), Path(journal).resolve()
    result = {'qualified': False, 'ledger_opened': False, 'scope': SCOPE}
    try:
        data = read(journal / 'journal.json')
        if data.get('state') != 'RESTORED':
            raise Incomplete('controller not RESTORED')
        for name in ('final_status', 'log_window', 'reap', 'load_worker', 'post_result', 'replacement', 'resource_samples'):
            if name not in data:
                raise Incomplete('missing final controller field: ' + name)
        for path in [owner / 'complete.json', owner / 'qualification/complete.json', owner / 'release.json',
                     journal / 'load-worker/release.json', journal / 'load-worker/result.json']:
            require(path)
        complete, plan, launch, opid, cpid, spawned, released, created = owner_binding(owner, journal, data)
        result['owner_resources'] = owner_resources(owner, complete, opid, cpid, spawned, released, created)
        result['service_restore'] = restore_evidence(data, journal, max(released, result['owner_resources']['last_recorded_at']))
        qualified = read(owner / 'qualification/complete.json')
        need(qualified['status'] == 'PASS' and qualified['plan_sha256'] == PLAN_SHA
             and qualified['numeric_only_not_http'] is True and qualified['source_model_pins_rechecked'] is True
             and type(qualified['total_requests']) is int and qualified['total_requests'] == 37
             and qualified['qualifier_sha256'] == plan['execution']['qualifier']['sha256'], 'complete 37-request numerical receipt')
        base = source_auditor()
        fixtures, config, pins = base.audit_pins(owner, plan, launch)
        result['pins'] = pins
        eos = fixtures[0]['eos_token_ids']
        need(all(f['eos_token_ids'] == eos and f['normal_eos'] is True and f['max_output_tokens'] == 512 for f in fixtures), 'normal EOS/fixed cap fixture contract')
        require(owner / 'qualification/steps.jsonl')
        result['ledger_opened'] = True
        ledger = base.stream_ledger(owner / 'qualification/steps.jsonl', fixtures, config, eos, qualified['cases'], partial=False)
        result['ledger'] = ledger
        need([len(c['completed']) for c in ledger['cases']] == [8, 16, 13]
             and ledger['partial_cases'] == [] and ledger['truncated_final_line'] is None, 'all 37 requests and untruncated complete ledger required')
        result.update(status='PASS', numerical_status='PASS', qualified=True, plan_sha256=PLAN_SHA,
                      numerical_auditor_sha256=AUDITOR_SHA, owner_source_sha256=OWNER_SHA, controller_source_sha256=CONTROLLER_SHA)
    except (Incomplete, FileNotFoundError) as error:
        result.update(status='INCOMPLETE', numerical_status='NOT_QUALIFIED', reason=str(error))
    except Exception as error:
        result.update(status='FAIL', numerical_status='NOT_QUALIFIED', first_failure={'type': type(error).__name__, 'message': str(error)})
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--owner', required=True, type=Path)
    p.add_argument('--journal', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    args = p.parse_args(argv)
    need(not args.out.exists() and args.out.parent.is_dir(), 'new output in existing directory required')
    result = audit(args.owner, args.journal)
    result['audit_source_sha256'] = sha(__file__)
    with args.out.open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False); f.write('\n'); f.flush(); os.fsync(f.fileno())
    return {'PASS': 0, 'FAIL': 1, 'INCOMPLETE': 2}[result['status']]


if __name__ == '__main__':
    raise SystemExit(main())
