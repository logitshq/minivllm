"""Independent CPU-only E022 v2 audit. Never imports its measurement/launch code.

Reconstructs all 36 request contracts from saved SSE and clocks, all six process
medians and the preregistered paired gate. Reuses only the independent E020 audit
parser, with stricter required-evidence checks. No socket, signal, model import,
large weight hash, or mutation of source evidence. --self-test uses CPU fixtures.

Forked from the unchanged v1 auditor. Only v2 pins/schema, accepted-socket raw
integer/normalized boolean verification, and its corruption tests differ. The
full request, source, resources, lifecycle and paired-performance gates remain.
"""
from __future__ import annotations

import argparse
import copy
import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys
import traceback

import external_spec_http_audit as base

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PREREG = HERE / 'native_transport_v2/prereg-v2'
PLAN_SHA = 'e850dcdcc1d160ecddf512529758b32fc9a8241c6af515fb72901c4a6b28c81c'
JOBS_SHA = 'af64454278849b2d30c9577a3de3e36eb9cd3dd507b9f27a829f192f060cb8b8'
V1_AUDITOR_SOURCE_SHA256 = '47c25daa3d74cc6bc09408f2356578fe1942229a018cab181dc1633373b542de'
MATRIX = [(1, 'off'), (1, 'on'), (2, 'on'), (2, 'off'), (3, 'off'), (3, 'on')]
GATE = {'all_correctness_resource_reap_checks': True, 'paired_wall_geomean_min': 1.05,
        'every_paired_wall_speedup_strictly_above': 1.0, 'every_paired_TTFT_speedup_min': .97}
METRICS = {'median_wall_s': 'E2E_s', 'median_TTFT_s': 'TTFT_s',
           'median_output_tok_s': 'aggregate_output_tok_s',
           'median_client_estimated_interactivity_tok_s': 'client_estimated_interactivity_tok_s'}


class Audit(base.Audit):
    def __init__(self):
        super().__init__()
        self.read_files = set()

    def load(self, path, required=True):
        path = Path(path)
        if path.is_file():
            self.read_files.add(path.resolve())
        return super().load(path, required=required)

    def fields(self, record, required, where):
        return self.check(isinstance(record, dict) and set(required) <= set(record),
                          where, 'missing required fields: ' + ', '.join(sorted(set(required) - set(record or {}))))

    def pin(self, path, expected, size=None):
        path = Path(path)
        self.check(path.is_file(), path, 'pinned evidence file missing')
        if path.is_file():
            self.read_files.add(path.resolve())
            self.check(base.sha(path) == expected, path, 'SHA differs from frozen pin')
            if size is not None:
                self.check(path.stat().st_size == size, path, 'size differs from frozen pin')


def flag(command, name):
    if command.count(name) != 1:
        return None
    at = command.index(name)
    return command[at + 1] if at + 1 < len(command) else None


def request_audit(audit, path, row, plan, tokenizer, pair, phase, index, model_id):
    """E020's saved data fields are raw SSE records, not raw TCP/chunk bytes."""
    required = {'request_id', 'method', 'path', 'body', 'http_status', 'done', 'usage',
                'finish_reason', 'sse_events', 'text_events', 'error', 'absolute_request_timeout_s',
                'start_perf_counter_s', 'cleanup_complete_perf_counter_s', 'contract_failures',
                'qualified_raw_ids_source', 'output_ids_when_available'}
    audit.fields(row, required, path)
    expected_body = {'model': model_id, 'prompt': plan['prompt_text'], 'temperature': 0,
        'repetition_penalty': 1.0, 'max_tokens': 32, 'stream': True,
        'stream_options': {'include_usage': True}}
    audit.check(row.get('body') == expected_body, path, 'full request body differs from frozen native contract')
    audit.check(row.get('request_id') == f'pair{pair}-native-none-{phase}-{index}', path, 'wrong request ID')
    audit.check(row.get('absolute_request_timeout_s') == 300 and row.get('qualified_raw_ids_source') == 'native',
                path, 'deadline or raw-ID provenance differs')
    usage_events, raw_events, finished = [], [], []
    events = row.get('sse_events') or []
    for i, event in enumerate(events):
        if event.get('data') == '[DONE]':
            continue
        payload = json.loads(event['data'])
        if payload.get('usage') is not None:
            usage_events.append((i, payload['usage']))
        if 'benchmark_output_ids' in payload:
            raw_events.append((i, payload['benchmark_output_ids']))
        for choice in payload.get('choices', []):
            audit.check(choice.get('index', 0) == 0, path, 'nonzero choice index')
            if choice.get('finish_reason') is not None:
                finished.append((i, choice['finish_reason']))
    last = len(events) - 2
    audit.check(len(usage_events) == len(raw_events) == len(finished) == 1 and
                all(x[0] == last for x in usage_events + raw_events + finished),
                path, 'usage/raw IDs/finish must appear once in the final event before DONE')
    if usage_events:
        usage = usage_events[0][1]
        audit.check(usage == {'prompt_tokens': 19, 'completion_tokens': 32, 'total_tokens': 51,
                              'prompt_tokens_details': {'cached_tokens': 0}},
                    path, 'native final usage shape/value differs')
    if raw_events:
        raw = raw_events[0][1]
        audit.check(isinstance(raw, list) and len(raw) == 32 and all(type(x) is int and x >= 0 for x in raw),
                    path, 'invalid actual native raw IDs')
    result = base.request_audit(audit, path, row, plan, tokenizer, pair, 'native-none', phase, index)
    audit.check(result['valid'], path, 'raw SSE contract failed: ' + '; '.join(result['failures']))
    audit.check(row.get('contract_failures') == [], path, 'runner recorded request failure')
    return result


def socket_audit(audit, receipt, release, plan, arm, where):
    audit.fields(receipt, {'schema', 'pid', 'server_source', 'server_source_sha256',
                          'tcp_nodelay_requested', 'tcp_nodelay_actual', 'tcp_nodelay_actual_enabled', 'wbufsize', 'rbufsize', 'scope'}, where)
    expected = arm == 'on'
    audit.check(receipt.get('schema') == 'E022-socket-v2' and type(receipt.get('pid')) is int
                and receipt.get('pid') == release.get('pid'), where, 'accepted socket PID/schema differs')
    audit.check(receipt.get('tcp_nodelay_requested') is expected and
                type(receipt.get('tcp_nodelay_actual')) is int and
                bool(receipt.get('tcp_nodelay_actual')) is expected and
                receipt.get('tcp_nodelay_actual_enabled') is expected,
                where, 'raw integer/normalized/requested TCP_NODELAY does not match arm')
    audit.check('error' not in receipt and 'error_type' not in receipt, where, 'accepted socket receipt contains failure')
    audit.check(receipt.get('wbufsize') == 0 and receipt.get('rbufsize') == -1,
                where, 'buffering differs from frozen socket contract')
    audit.check(receipt.get('server_source') == plan['reused_sources']['native_server'] and
                receipt.get('server_source_sha256') == plan['native_server_sha256'],
                where, 'socket loaded a different native shim')


def restore_audit(audit, lease, resume, where):
    audit.fields(lease, {'service_pid', 'service_identity', 'owner_pid', 'owner_identity',
                        'child_pid', 'paused_at', 'deadline', 'closed', 'authorization', 'jobs', 'current_job'}, where)
    audit.fields(resume, {'identity_match', 'state_before', 'state_after', 'time'}, where)
    audit.check(lease.get('service_pid') == 74384 and isinstance(lease.get('service_identity'), str)
                and lease['service_identity'].endswith('omlx-server'), where, 'wrong authorized service identity')
    audit.check(lease.get('closed') is True and lease.get('current_job') == 0 and
                bool(lease.get('owner_identity')) and bool(lease.get('authorization')), where, 'lease not completed')
    audit.check(resume.get('identity_match') is True and resume.get('state_before') == 'T'
                and isinstance(resume.get('state_after'), str) and resume['state_after'].startswith(('S', 'R'))
                and 'T' not in resume['state_after'], where, 'restore identity/stopped-to-running transition failed')
    values = [lease.get('paused_at'), resume.get('time'), lease.get('deadline')]
    audit.check(all(base.number(x) for x in values) and values[0] <= values[1] <= values[2],
                where, 'restore time outside authorized lease')


def resources_raw(audit, folder, requests, pid, controller, lease, resume):
    paths = sorted(folder.glob('resource-[0-9][0-9][0-9].json'))
    audit.check(sorted(p.name for p in folder.glob('resource-*.json')) == [p.name for p in paths],
                folder, 'unexpected resource evidence filenames')
    audit.check([p.name for p in paths] == [f'resource-{i:03}.json' for i in range(len(paths))],
                folder, 'resource file numbering incomplete')
    raw = [audit.load(p) for p in paths]
    labels = [r['label'] for r in raw]
    expected = ['before-launch', 'ready'] + [f'{side}-{phase}-{i}'
        for phase in ('warm', 'formal') for i in range(3) for side in ('before', 'after')] + ['after-unload']
    audit.check([x for x in labels if x != 'readiness-wait'] == expected,
                folder, 'resource probes missing/repeated/out of order')
    if 'readiness-wait' in labels:
        audit.check(all(0 < i < labels.index('ready') for i, x in enumerate(labels) if x == 'readiness-wait'),
                    folder, 'readiness probe outside load interval')
    for path, row in zip(paths, raw):
        audit.fields(row, {'label', 'recorded_at', 'monotonic_s', 'owned_model_pid', 'memory_pressure',
            'pressure_level', 'swap', 'vm_stat', 'thermal', 'free_percent', 'swap_used_mb',
            'pressure_level_value', 'vm_counters', 'process_inventory_ok', 'known_inference_processes',
            'native_process_inventory_ok', 'native_inference_processes', 'authorized_suspended_pids',
            'power_metadata', 'processinfo_thermal_metadata', 'exact_owned_pid_metadata'}, path)
        for key in ('memory_pressure', 'pressure_level', 'swap', 'vm_stat', 'thermal'):
            probe = row[key]
            audit.check(probe.get('returncode') == 0 and not probe.get('error'), path, key + ' command failed/unknown')
        free = re.search(r'free percentage:\s*(\d+)', row['memory_pressure']['text'])
        swap = re.search(r'used = ([0-9.]+)M', row['swap']['text'])
        pressure = row['pressure_level']['text']
        audit.check(free is not None and int(free[1]) == row['free_percent'], path, 'free percent differs from command')
        audit.check(swap is not None and float(swap[1]) == row['swap_used_mb'], path, 'swap differs from command')
        audit.check(pressure.isdigit() and int(pressure) == row['pressure_level_value'], path, 'pressure differs from command')
        counters = {}
        for key in base.COUNTERS:
            match = re.search(r'^' + key + r':\s*(\d+)', row['vm_stat']['text'], re.M)
            audit.check(match is not None, path, 'missing raw VM counter: ' + key)
            if match:
                counters[key] = int(match[1])
        audit.check(counters == row['vm_counters'], path, 'VM counters differ from command')
        power, thermal = row['power_metadata'], row['processinfo_thermal_metadata']
        audit.check(power.get('status') == 'known' and power.get('source') == 'AC Power' and
            power.get('raw', {}).get('returncode') == 0 and
            "Now drawing from 'AC Power'" in power['raw'].get('stdout', '') and not power['raw'].get('stderr'),
            path, 'AC power not supported by raw command')
        audit.check(thermal.get('status') == 'known' and thermal.get('code') in (0, 1) and
            thermal.get('raw', {}).get('returncode') == 0 and
            thermal['raw'].get('stdout') == str(thermal.get('code')) and not thermal['raw'].get('stderr'),
            path, 'thermal state not supported by raw command')
        audit.check(row['authorized_suspended_pids'] == [74384], path, 'different authorized suspended service set')
        expected_native_pids = {74384} if row['label'] in ('before-launch', 'after-unload') else {74384, pid}
        native_pids = [p.get('pid') for p in row['native_inference_processes']]
        audit.check(set(native_pids) == expected_native_pids and len(native_pids) == len(expected_native_pids),
                    path, 'native inventory missing/duplicate/foreign process')
        if row['label'] not in ('before-launch', 'after-unload'):
            owned = row['exact_owned_pid_metadata']
            command = owned.get('raw') or {}
            values = command.get('stdout', '').split(None, 3)
            audit.check(command.get('returncode') == 0 and not command.get('stderr') and len(values) == 4
                and values[:2] == [str(pid), str(controller)] and values[2].isdigit()
                and int(values[2]) == owned.get('rss_kib') and values[3] == owned.get('executable'),
                path, 'exact owned PID/parent/RSS differs from raw ps')
            audit.check(type(row.get('owned_pid_rss_kib')) is int and row['owned_pid_rss_kib'] > 0,
                        path, 'owned server RSS unavailable')
        else:
            audit.check(row['owned_model_pid'] is None and not any(p.get('pid') == pid
                for p in row['native_inference_processes']), path, 'model remains at no-owner lifecycle boundary')
    aggregate = audit.load(folder / 'resources.json')
    audit.check(isinstance(aggregate, list) and len(aggregate) == len(raw), folder, 'resource aggregate missing/incomplete')
    result = base.resource_audit(audit, folder, pid, controller, requests, lease, resume)
    audit.check(result['valid'], folder, 'resource guard failed: ' + '; '.join(result['issues']))
    for before, after in zip(requests, requests[1:]):
        left = next(x for x in raw if x['label'] == f'after-{before["phase"]}-{before["index"]}')
        right = next(x for x in raw if x['label'] == f'before-{after["phase"]}-{after["index"]}')
        audit.check(right['monotonic_s'] - left['monotonic_s'] >= 2 - 1e-4, folder, 'fixed two-second cooldown missing')
    return result


def pair_gate(rows):
    """Independent gate arithmetic; no import of native_transport.common."""
    if [(r['pair'], r['arm']) for r in rows] != MATRIX:
        raise ValueError('six fresh arms must appear in preregistered order')
    pairs = []
    valid = all(r['valid'] for r in rows)
    for pair in (1, 2, 3):
        off = next(r for r in rows if r['pair'] == pair and r['arm'] == 'off')
        on = next(r for r in rows if r['pair'] == pair and r['arm'] == 'on')
        okay = off['valid'] and on['valid']
        if okay and not all(base.number(r[k]) and r[k] > 0 for r in (off, on)
                            for k in ('median_wall_s', 'median_TTFT_s', 'median_output_tok_s')):
            raise ValueError('nonpositive/nonfinite valid process median')
        pairs.append({'pair': pair, 'valid': okay,
            'wall_speedup': off['median_wall_s'] / on['median_wall_s'] if okay else None,
            'TTFT_speedup': off['median_TTFT_s'] / on['median_TTFT_s'] if okay else None,
            'throughput_speedup': on['median_output_tok_s'] / off['median_output_tok_s'] if okay else None})
    gms = {name: math.exp(math.fsum(math.log(p[name]) for p in pairs) / 3) if valid else None
           for name in ('wall_speedup', 'TTFT_speedup', 'throughput_speedup')}
    advance = bool(valid and gms['wall_speedup'] >= 1.05 and
        all(p['wall_speedup'] > 1 for p in pairs) and all(p['TTFT_speedup'] >= .97 for p in pairs))
    return {'pairs': pairs, 'all_correctness_resource_reap_checks': valid,
            'paired_wall_geomean': gms['wall_speedup'], 'all_paired_geomeans': gms,
            'advance': advance, 'gate': GATE, 'matrix_complete': True}


def process_audit(audit, folder, pair, arm, plan, parent, tokenizer, native_manifest, controller, lease, resume):
    beginning = len(audit.integrity_errors)
    for bad in ('socket-failure.json', 'first-failure.json', 'failure.json'):
        audit.check(not (folder / bad).exists(), folder / bad, 'permanent failure evidence present')
    launch, ready, release, cleanup, declared = [audit.load(folder / x) or {} for x in
        ('launch.json', 'ready.json', 'release.json', 'cleanup.json', 'result.json')]
    socket = audit.load(folder / 'socket.json') or {}
    sock_result = audit.load(folder / 'transport-result.json') or {}
    route = audit.load(folder / 'route.json') or {}
    audit.fields(declared, {'pair', 'arm', 'valid', 'contract_failures', 'fatal_failure',
        'cleanup_errors', 'request_count', 'formal_count', 'diagnostic_medians_including_invalid_samples',
        *METRICS, 'request_files', 'resource_samples', 'release', 'route_valid', 'scope'}, folder / 'result.json')
    audit.fields(sock_result, {'pair', 'arm', 'valid', 'native_result', 'socket_receipt',
        'transport_failures', 'native_contract_failures', 'diagnostic_medians_including_invalid_samples',
        'median_wall_s', 'median_TTFT_s', 'median_output_tok_s'}, folder / 'transport-result.json')
    owner = base.lifecycle_audit(audit, folder, controller, lease, resume)
    audit.check(owner['valid'], folder, 'ownership/reap failed: ' + '; '.join(owner['issues']))
    pid, port = owner['pid'], release.get('port')
    audit.check(type(pid) is int and pid > 0 and pid not in (74384, controller, lease['owner_pid']), folder, 'invalid/fresh server PID')
    audit.check((release.get('returncode') == -15 and release.get('kill_escalated') is False) or
                (release.get('returncode') == -9 and release.get('kill_escalated') is True),
                folder, 'unexpected server exit; expected controller SIGTERM or recorded SIGKILL escalation')
    for filename, record in [('owned-pid.json', audit.load(folder / 'owned-pid.json')), ('ready.json', ready)]:
        probe = record.get('ps_at_launch', record.get('ps', {}))
        audit.check(probe.get('returncode') == 0 and not probe.get('stderr'), folder / filename, 'owner ps command failed')
    command = launch.get('command', [])
    expected_command = [parent['runtime_versions']['controller-native']['python'], '-B',
        str(HERE / 'native_transport_v2/serve.py'), '--server-source', plan['reused_sources']['native_server'],
        '--server-source-sha256', plan['native_server_sha256'], '--tcp-nodelay', arm,
        '--socket-receipt', str(folder / 'socket.json'), '--socket-failure', str(folder / 'socket-failure.json'),
        '--model', parent['workload']['model_dir'], '--port', str(port),
        '--package-root', parent['native_source_root'], '--source-manifest', parent['native_source_manifest']]
    audit.check(command == expected_command and launch.get('controller_pid') == controller and
        launch.get('pair') == pair and launch.get('arm') == 'native-none' and launch.get('cwd') == str(ROOT),
        folder / 'launch.json', 'direct launch/arm/path/controller differs')
    audit.check(launch.get('environment') == dict(parent['environment'], HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
                and launch.get('files') == {} and launch.get('symlinks') == {}, folder, 'launch environment/extra files differ')
    identity = ready.get('models', {}).get('benchmark_identity', {})
    expected_identity = {'source_files': native_manifest['files'], 'manifest_all_sources_match': True,
        'cache_limit_gib': 2, 'compile_decode': True, 'fuse_decode': True, 'drafter': None,
        'gates_compile': True, 'norm_reuse': False, 'grouped_prefill': 'unmodified core default',
        'step_observer': False, 'max_seqs': 4, 'normal_eos': True}
    audit.check(identity == expected_identity and ready.get('models', {}).get('data') == [{'id': 'native-current-target-only'}],
                folder / 'ready.json', 'actual native model/source/settings differ')
    audit.check(route.get('valid') is True and route.get('failures') == [] and route.get('identity') == identity,
                folder / 'route.json', 'target-only route identity differs')
    audit.fields(cleanup, {'first_failure', 'cleanup_errors', 'release_valid'}, folder / 'cleanup.json')
    audit.check(cleanup.get('first_failure') is None and cleanup.get('cleanup_errors') == [] and cleanup.get('release_valid') is True,
                folder, 'cleanup error/missing proof')
    socket_audit(audit, socket, release, plan, arm, folder / 'socket.json')
    log_path = folder / 'server.log'
    audit.check(log_path.is_file(), log_path, 'required server log missing')
    if log_path.is_file():
        audit.read_files.add(log_path.resolve())
        log = log_path.read_text(errors='replace')
        audit.check(not re.search(r'Traceback \(most recent call last\)|\[METAL\].*(?:failed|Error)|Fatal Python error', log),
                    log_path, 'server traceback/Metal/fatal error')
    request_paths = [folder / f'{phase}-{i}.json' for phase in ('warm', 'formal') for i in range(3)]
    audit.check(sorted(p.name for p in folder.glob('warm-*.json')) == [f'warm-{i}.json' for i in range(3)] and
                sorted(p.name for p in folder.glob('formal-*.json')) == [f'formal-{i}.json' for i in range(3)],
                folder, 'request set missing or contains replacement/extra requests')
    requests = [request_audit(audit, p, audit.load(p), parent, tokenizer, pair, phase, i, 'native-current-target-only')
                for phase in ('warm', 'formal') for i in range(3) for p in [folder / f'{phase}-{i}.json']]
    resource = resources_raw(audit, folder, requests, pid, controller, lease, resume)
    medians = {name: statistics.median(r[field] for r in requests if r['phase'] == 'formal') for name, field in METRICS.items()}
    audit.check(declared.get('pair') == pair and declared.get('arm') == 'native-none' and declared.get('valid') is True
                and declared.get('request_count') == 6 and declared.get('formal_count') == 3 and
                declared.get('contract_failures') == [] and declared.get('fatal_failure') is None and
                declared.get('cleanup_errors') == [] and declared.get('route_valid') is True and
                declared.get('release') == release and declared.get('resource_samples') == resource['samples'] and
                declared.get('request_files') == [str(p) for p in request_paths], folder / 'result.json', 'process summary disagrees with evidence')
    for name, value in medians.items():
        audit.check(base.equal(declared.get(name), value) and
                    base.equal(declared.get('diagnostic_medians_including_invalid_samples', {}).get(name), value),
                    folder / 'result.json', 'formal raw median differs: ' + name)
    audit.check(sock_result.get('pair') == pair and sock_result.get('arm') == arm and sock_result.get('valid') is True
                and sock_result.get('socket_receipt') == socket and sock_result.get('transport_failures') == []
                and sock_result.get('native_contract_failures') == [] and
                sock_result.get('native_result') == str(folder / 'result.json'), folder / 'transport-result.json', 'transport summary differs')
    for name in ('median_wall_s', 'median_TTFT_s', 'median_output_tok_s'):
        audit.check(base.equal(sock_result.get(name), medians[name]), folder / 'transport-result.json', 'transport median differs: ' + name)
    audit.check(sock_result.get('diagnostic_medians_including_invalid_samples') == declared.get('diagnostic_medians_including_invalid_samples'),
                folder / 'transport-result.json', 'transport diagnostic medians differ')
    return {'pair': pair, 'arm': arm, 'valid': len(audit.integrity_errors) == beginning,
        **medians, 'pid': pid, 'port': port, 'requests': requests, 'resources': resource,
        'lifecycle': owner, 'socket_receipt': socket}


def independent_audit(audit, run, lease_dir):
    prereg_path = PREREG / 'plan.json'
    audit.pin(prereg_path, PLAN_SHA)
    plan = audit.load(prereg_path)
    audit.check(plan['schema'] == 'E022-v2' and plan['configs'] == [{'pair': p, 'arm': a} for p, a in MATRIX]
                and plan['gate'] == GATE and plan['normal_eos'] is True and plan['concurrency'] == 1
                and plan['warm_requests'] == plan['formal_requests'] == 3
                and plan['cooldown_seconds_after_each_request'] == 2
                and plan['expected_counts'] == {'servers': 6, 'warm_requests': 18, 'formal_requests': 18, 'formal_tokens_if_valid': 576},
                prereg_path, 'preregistration contract changed')
    audit.check(audit.load(run / 'plan.json') == plan, run, 'runtime/precommitted plan differs')
    for filename, sha in plan['pins'].items():
        audit.pin(filename, sha, plan['pin_stats'][filename]['bytes'])
    audit.pin(plan['e020_plan'], plan['e020_plan_sha256'])
    parent = audit.load(plan['e020_plan'])
    audit.check(len(parent['pins']) == plan['inherited_pin_count'] and parent['environment'] == plan['environment'],
                prereg_path, 'inherited E020 pins/environment differ')
    for label in ('before', 'after'):
        audit.check(audit.load(run / f'pins-{label}.json') ==
            {'inherited_all_full_sha_match': True, 'new_pins_match': True}, run, 'missing/invalid runner before/after full hash receipt')
    attempt = audit.load(run / 'attempt.json')
    audit.fields(attempt, {'pid', 'parent_pid', 'plan', 'plan_sha256', 'lease', 'configs'}, run / 'attempt.json')
    audit.check(attempt['plan_sha256'] == PLAN_SHA and attempt['configs'] == plan['configs']
                and attempt['lease'] == str(lease_dir / 'lease.json'), run, 'attempt plan/config/lease differs')
    audit.pin(attempt['plan'], PLAN_SHA)
    audit.pin(PREREG / 'jobs.json', JOBS_SHA)
    jobs = audit.load(PREREG / 'jobs.json')
    lease, resume, job = [audit.load(lease_dir / p) for p in ('lease.json', 'resume.json', 'job-0.json')]
    restore_audit(audit, lease, resume, lease_dir)
    audit.check(not (lease_dir / 'watchdog-resume.json').exists(), lease_dir, 'watchdog restoration indicates interrupted lease')
    audit.check(lease['jobs'] == jobs and len(jobs) == 1 and job.get('command') == jobs[0], lease_dir, 'executed job differs from preregistration')
    command = jobs[0]
    audit.check(flag(command, '--out') == str(run) and flag(command, '--lease') == str(lease_dir / 'lease.json') and
                flag(command, '--plan') == attempt['plan'] and flag(command, '--plan-sha256') == PLAN_SHA and
                not any('guarded_v2' in part for part in command), lease_dir, 'job does not directly own requested controller')
    controller = attempt['pid']
    audit.check(type(controller) is int and controller > 0 and attempt['parent_pid'] == lease['owner_pid'] and
                controller == lease['child_pid'] == job.get('pid') and job.get('waited') is True and job.get('returncode') == 0,
                lease_dir, 'controller parent/exit/wait receipt differs')
    manifest = audit.load(parent['native_source_manifest'])
    audit.pin(parent['native_source_manifest'], parent['pins'][parent['native_source_manifest']])
    core = Path(parent['native_source_root']) / 'tutorial_code/miniengine'
    audit.check(sorted(str(p.relative_to(core)) for p in core.rglob('*.py')) == sorted(manifest['files']), core, 'frozen core file set differs')
    for relative, sha in manifest['files'].items():
        audit.pin(core / relative, sha)
    tokenizer_path = Path(parent['workload']['model_dir']) / 'tokenizer.json'
    audit.pin(tokenizer_path, parent['pins'][str(tokenizer_path)])
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    audit.check(parent['oracle_ids']['native'] == parent['workload']['continuation_ids']
                and len(parent['workload']['prompt_ids']) == 19 and len(parent['oracle_ids']['native']) == 32,
                'oracle', 'native oracle/count differs from frozen workload')
    expected_dirs = {str(Path(arm) / f'pair{pair}-native-none') for pair, arm in MATRIX}
    actual_dirs = {str(p.relative_to(run)) for arm in ('off', 'on') for p in (run / arm).glob('pair*') if p.is_dir()}
    audit.check(actual_dirs == expected_dirs, run, 'fresh server directory matrix differs')
    audit.check(not (run / 'failure.json').exists(), run, 'controller failure receipt present')
    processes = []
    for pair, arm in MATRIX:
        folder = run / arm / f'pair{pair}-native-none'
        try:
            processes.append(process_audit(audit, folder, pair, arm, plan, parent, tokenizer, manifest, controller, lease, resume))
        except Exception as error:
            audit.check(False, folder, 'malformed/missing evidence: ' + type(error).__name__ + ': ' + str(error))
            processes.append({'pair': pair, 'arm': arm, 'valid': False, 'audit_exception': traceback.format_exc()})
    pids = [p.get('pid') for p in processes]
    audit.check(None not in pids and len(set(pids)) == 6, run, 'six unique fresh server PIDs not proven')
    for left, right in zip(processes, processes[1:]):
        if left.get('resources') and right.get('resources'):
            audit.check(left['resources']['last_monotonic_s'] < right['resources']['first_monotonic_s'], run, 'process intervals overlap/reorder')
    requests = [r for p in processes for r in p.get('requests', [])]
    audit.check(len(requests) == 36 and sum(r['phase'] == 'formal' for r in requests) == 18, run, 'not 36 actual requests')
    for left, right in zip(requests, requests[1:]):
        audit.check(left['cleanup_s'] <= right['start_s'], run, 'C1 requests overlap/reorder')
    gate = pair_gate(processes)
    complete = audit.load(run / 'complete.json')
    audit.fields(complete, {'rows', 'pairs', 'matrix_complete', 'all_correctness_resource_reap_checks',
        'paired_wall_geomean', 'advance', 'gate', 'scope', 'status', 'inherited_pins_full_sha_match',
        'new_pins_sha_match', 'service_restore'}, run / 'complete.json')
    progress = audit.load(run / 'progress.json')
    audit.check(progress == {key: complete[key] for key in progress}, run / 'progress.json', 'final progress differs from completed matrix')
    audit.check(set(progress) == {'rows', 'pairs', 'matrix_complete', 'all_correctness_resource_reap_checks',
        'paired_wall_geomean', 'advance', 'gate', 'scope'}, run / 'progress.json', 'final progress is incomplete')
    audit.check(len(complete.get('rows', [])) == 6 and complete.get('gate') == GATE, run, 'complete matrix/gate differs')
    for stored, actual in zip(complete.get('rows', []), processes):
        audit.check(stored == audit.load(run / actual['arm'] / f'pair{actual["pair"]}-native-none/transport-result.json'), run, 'complete row differs from per-process evidence')
    for key in ('matrix_complete', 'all_correctness_resource_reap_checks', 'paired_wall_geomean', 'advance'):
        audit.check(base.equal(complete.get(key), gate[key]), run, 'complete disagrees with independently reconstructed gate: ' + key)
    audit.check(len(complete.get('pairs', [])) == 3, run, 'missing paired ratios')
    for stored, actual in zip(complete.get('pairs', []), gate['pairs']):
        audit.check(set(stored) == set(actual) and all(base.equal(stored[k], v) for k, v in actual.items()), run, 'paired ratio differs')
    audit.check(complete.get('status') == ('PASS_ADVANCE' if gate['advance'] else 'FAIL_GATE_RESULTS_RETAINED') and
                complete.get('inherited_pins_full_sha_match') is True and complete.get('new_pins_sha_match') is True,
                run, 'complete status/pin receipts differ')
    return {'processes': processes, 'gate': gate, 'request_count': len(requests),
        'formal_actual_output_tokens': sum(r['actual_usage_count'] for r in requests if r['phase'] == 'formal'),
        'restore': {'service_pid': lease['service_pid'], 'service_identity': lease['service_identity'], **resume},
        'inherited_pin_count': len(parent['pins']), 'plan_sha256': PLAN_SHA}


def self_test():
    """Small immutable fixture copies; no original experiment data are modified."""
    from types import SimpleNamespace
    class Tokenizer:
        def encode(self, text, add_special_tokens=False):
            return SimpleNamespace(ids=list(range(19 if text == 'prompt' else len(text))))
    plan = {'oracle_ids': {'native': list(range(32))}, 'prompt_text': 'prompt',
            'workload': {'prompt_ids': list(range(19))},
            'reused_sources': {'native_server': '/frozen/server.py'}, 'native_server_sha256': 'a' * 64}
    row = {'pair': 1, 'arm': 'native-none', 'phase': 'warm', 'index': 0, 'request_id': 'pair1-native-none-warm-0',
        'method': 'POST', 'http_status': 200, 'path': '/v1/completions',
        'body': {'model': 'm', 'prompt': 'prompt', 'temperature': 0, 'repetition_penalty': 1.0,
                 'max_tokens': 32, 'stream': True, 'stream_options': {'include_usage': True}},
        'start_perf_counter_s': 10., 'cleanup_complete_perf_counter_s': 11.1,
        'absolute_request_timeout_s': 300, 'error': None, 'contract_failures': [], 'qualified_raw_ids_source': 'native'}
    usage = {'prompt_tokens': 19, 'completion_tokens': 32, 'total_tokens': 51, 'prompt_tokens_details': {'cached_tokens': 0}}
    row['sse_events'] = [{'client_perf_counter_s': 10.1, 'data': json.dumps({'choices': [{'text': 'x', 'finish_reason': None}]})},
        {'client_perf_counter_s': 10.8, 'data': json.dumps({'choices': [{'text': 'x' * 31, 'finish_reason': None}]})},
        {'client_perf_counter_s': 10.9, 'data': json.dumps({'choices': [{'text': '', 'finish_reason': 'length'}],
                                                       'usage': usage, 'benchmark_output_ids': list(range(32))})},
        {'client_perf_counter_s': 11., 'data': '[DONE]'}]
    # Populate the stored scalar fields independently of measurement code.
    row.update(done=True, usage=usage, finish_reason='length', output_ids_when_available=list(range(32)),
        first_text_perf_counter_s=10.1, done_perf_counter_s=11., text='x' * 32,
        text_events=[{'client_perf_counter_s': 10.1, 'text': 'x'}, {'client_perf_counter_s': 10.8, 'text': 'x' * 31}],
        visible_roundtrip_ids=list(range(32)), actual_visible_canonical_count=32, actual_usage_output_count=32,
        cached_prompt_tokens_when_available=0, cache_usage_field_present=True, first_chunk_text='x',
        first_chunk_reconstructed_tokens=1, first_chunk_prefix_verified=True, full_visible_roundtrip_count_matches_usage=True,
        client_total_s=1., client_ttft_text_s=10.1 - 10., client_post_first_chunk_s=11. - 10.1,
        output_tok_s=32., aggregate_output_tok_s=32., actual_request_count=1, group_window_s=1.,
        total_completion_tokens=32, client_estimated_interactivity_tok_s=31 / (11. - 10.1),
        stream_cleanup_after_done_s=11.1 - 11.)
    tests = []
    def check_request(value):
        audit = Audit()
        request_audit(audit, 'synthetic', value, plan, Tokenizer(), 1, 'warm', 0, 'm')
        return audit.integrity_errors
    assert not check_request(row)
    for name, mutate in [('time', lambda r: r['sse_events'][0].update(client_perf_counter_s=9.)),
                         ('raw_ids', lambda r: r['sse_events'][2].update(data=r['sse_events'][2]['data'].replace('[0, 1,', '[999, 1,'))),
                         ('missing_usage', lambda r: r.pop('usage'))]:
        changed = copy.deepcopy(row); mutate(changed)
        assert check_request(changed), name
        tests.append(name)
    socket = {'schema': 'E022-socket-v2', 'pid': 123, 'server_source': '/frozen/server.py',
        'server_source_sha256': 'a' * 64, 'tcp_nodelay_requested': True, 'tcp_nodelay_actual': 4, 'tcp_nodelay_actual_enabled': True,
        'wbufsize': 0, 'rbufsize': -1, 'scope': 'first accepted'}
    a = Audit(); socket_audit(a, socket, {'pid': 123}, plan, 'on', 'synthetic'); assert not a.integrity_errors
    a = Audit(); socket_audit(a, socket, {'pid': 123}, plan, 'off', 'synthetic'); assert a.integrity_errors
    tests.append('socket_arm')
    # Boolean socket options may return a platform bitmask, not integer 1.
    positive_socket_readbacks = []
    for raw, expected_arm in ((0, 'off'), (1, 'on'), (4, 'on')):
        value = socket | {'tcp_nodelay_requested': expected_arm == 'on',
                          'tcp_nodelay_actual': raw, 'tcp_nodelay_actual_enabled': bool(raw)}
        a = Audit(); socket_audit(a, value, {'pid': 123}, plan, expected_arm, 'synthetic')
        assert not a.integrity_errors, (raw, a.integrity_errors)
        positive_socket_readbacks.append({'raw': raw, 'normalized': bool(raw), 'arm': expected_arm})
    for name, mutate in [
        ('socket_raw_bool', lambda r: r.update(tcp_nodelay_actual=True)),
        ('socket_raw_float', lambda r: r.update(tcp_nodelay_actual=4.0)),
        ('socket_raw_string', lambda r: r.update(tcp_nodelay_actual='4')),
        ('socket_raw_null', lambda r: r.update(tcp_nodelay_actual=None)),
        ('socket_raw_zero_on', lambda r: r.update(tcp_nodelay_actual=0)),
        ('socket_missing_raw', lambda r: r.pop('tcp_nodelay_actual')),
        ('socket_normalized_inconsistent', lambda r: r.update(tcp_nodelay_actual_enabled=False)),
        ('socket_normalized_integer', lambda r: r.update(tcp_nodelay_actual_enabled=1)),
        ('socket_missing_normalized', lambda r: r.pop('tcp_nodelay_actual_enabled')),
        ('socket_requested_integer', lambda r: r.update(tcp_nodelay_requested=1)),
        ('socket_old_schema', lambda r: r.update(schema='E022-socket-v1')),
        ('socket_wrong_pid', lambda r: r.update(pid=456)),
        ('socket_error_receipt', lambda r: r.update(error='setup failed')),
    ]:
        changed = copy.deepcopy(socket); mutate(changed)
        a = Audit(); socket_audit(a, changed, {'pid': 123}, plan, 'on', 'synthetic')
        assert a.integrity_errors, name
        tests.append(name)
    lease = {'service_pid': 74384, 'service_identity': 'start omlx-server', 'owner_pid': 1, 'owner_identity': 'owner',
        'child_pid': 2, 'paused_at': 1., 'deadline': 3., 'closed': True, 'authorization': 'authorized', 'jobs': [], 'current_job': 0}
    restore = {'identity_match': True, 'state_before': 'T', 'state_after': 'S', 'time': 2.}
    a = Audit(); restore_audit(a, lease, restore, 'synthetic'); assert not a.integrity_errors
    a = Audit(); restore_audit(a, lease, restore | {'identity_match': False}, 'synthetic'); assert a.integrity_errors
    tests.append('restore_identity')
    rows = [{'pair': p, 'arm': arm, 'valid': True, 'median_wall_s': 1.1 if arm == 'off' else 1.,
             'median_TTFT_s': .1, 'median_output_tok_s': 32 / (1.1 if arm == 'off' else 1.)} for p, arm in MATRIX]
    assert pair_gate(rows)['advance']
    rows[0]['median_wall_s'] = 1.; assert not pair_gate(rows)['advance']; tests.append('strict_pair_gate')
    return {'status': 'PASS_CPU_SYNTHETIC_NEGATIVE_ASSERTIONS', 'rejected_mutations': tests,
            'positive_socket_readbacks': positive_socket_readbacks,
            'runtime_experiment_audited': False, 'GPU_work': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path)
    parser.add_argument('--lease', type=Path, help='lease directory, or its lease.json')
    parser.add_argument('--out', type=Path, help='new output directory')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), indent=2)); return
    if not all((args.run, args.lease, args.out)):
        parser.error('--run, --lease and --out required unless --self-test')
    run, lease_dir, out = args.run.resolve(), args.lease.resolve(), args.out.resolve()
    if lease_dir.name == 'lease.json':
        lease_dir = lease_dir.parent
    if out == run or run in out.parents or out == lease_dir or lease_dir in out.parents:
        parser.error('audit output must be outside original run/lease evidence')
    out.mkdir(parents=True, exist_ok=False)
    audit, result = Audit(), {}
    try:
        result = independent_audit(audit, run, lease_dir)
    except Exception:
        audit.check(False, run, 'required evidence unavailable/malformed; see audit_exception')
        result['audit_exception'] = traceback.format_exc()
    passed = not audit.integrity_errors
    result.update(schema='E022-independent-audit-v2', status='PASS_INDEPENDENT_AUDIT' if passed else 'FAIL_INDEPENDENT_AUDIT',
        performance_decision=('ADVANCE' if result.get('gate', {}).get('advance') else 'REJECT_PERFORMANCE_GATE') if passed else 'NOT_ELIGIBLE',
        integrity_errors=audit.integrity_errors, CPU_only=True, run=str(run), lease=str(lease_dir),
        source_sha256=base.sha(__file__), independent_parser_sha256=base.sha(Path(base.__file__)),
        v1_auditor_source_sha256=V1_AUDITOR_SOURCE_SHA256, prereg_jobs_sha256=JOBS_SHA,
        evidence_scope={'requests': 'Persisted SSE data records and client receipt clocks; no TCP packet/chunk capture.',
            'socket': 'Raw integer and normalized boolean checked against requested arm; first accepted socket receipt. Later readbacks enforced by pinned fatal handler, not individually persisted.',
            'restore': 'Persisted identity_match plus prior identity and T-to-running state; post-restore raw identity string was not persisted.',
            'resources': 'Sampled raw command values; process inventory native rows are persisted parser results, not full ps output. No GPU peak inference.',
            'pins': 'Small new/frozen core/tokenizer files independently hashed. Large weights/runtime inherited full hashes are runner before/after receipts; not reread.'})
    audit.read_files.update((Path(__file__).resolve(), Path(base.__file__).resolve()))
    manifest = {str(p): {'bytes': p.stat().st_size, 'sha256': base.sha(p)} for p in sorted(audit.read_files) if p.is_file()}
    base.write(out / 'input-manifest.json', manifest)
    result['input_manifest_sha256'] = base.sha(out / 'input-manifest.json')
    base.write(out / 'audit.json', result)
    print(result['status'], result['performance_decision'], 'errors', len(audit.integrity_errors))
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
