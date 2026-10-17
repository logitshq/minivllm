"""Independent stdlib audit of E026 tiny bits/backing evidence; no GPU imports.

This file never imports the screen, owner, MLX, NumPy or model code. It checks
recorded observations; it cannot independently rerun device arithmetic. Memory
chart data are diagnostic only, never throughput/latency or Pareto evidence.
"""
import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import re
import struct
import tempfile
import traceback


SCREEN_SHA = '6112207b4525cad51b68d1d69002cd6cb33d58a4b097667c9372a8a051939698'
CANDIDATE_SHA = '4d9658d34734e3c252375c4b76fcb2eccf83555121d7eb51d7f3e81eb2c48ccc'
MANIFEST_SHA = '3f749bb8608c6693903cc4a4579bbcd76cefcd1cf8069e097e61f7b1700dee32'
LENGTHS = (1, 19, 935, 978)
KINDS = ('bits-0', 'bits-30720', 'bits-61440', 'finite')
ARMS = ('off', 'on-lazy', 'on-materialize')
PHASES = ('before_parent', 'after_parent_and_tail_eval', 'after_helper_graph_build',
          'after_output_eval', 'after_validation_before_drop', 'after_drop_sources',
          'after_gc', 'after_sync', 'after_clear_cache', 'after_drop_output',
          'after_final_gc_sync_clear_cache')
WIDTH, WORDS, TAIL_BYTES, LIMIT = 10240, 30720, 61440, 64 << 20


def need(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(1 << 20):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    need(Path(path).suffix != '.jsonl', 'JSONL must be read line by line')
    with Path(path).open() as stream:
        return json.load(stream)


def expected_word(kind, index):
    if kind == 'finite':
        return (index // 2) % 32640 + (32768 if index % 2 else 0)
    need(kind in KINDS, 'unknown kind')
    return (int(kind[5:]) + index) % 65536


def input_parent_sha(length, tail_bytes):
    h = hashlib.sha256()
    remaining = length * WIDTH * 2
    block = b'\x80\x3f' * 4096  # fixed +1.0 BF16 prefix, no full parent allocation
    while remaining:
        size = min(remaining, len(block))
        h.update(block[:size]); remaining -= size
    h.update(tail_bytes)
    return h.hexdigest()


def audit_bits(directory, case):
    length, kind = case['length'], case['kind']
    case_id = f'T{length}-{kind}'
    need(case['case'] == case_id and case['status'] == 'PASS', 'case identity/status')
    need(case['dtype'] == 'bfloat16' and case['shape'] == [3, WIDTH] and case['bit_equal'] is True, 'original dtype/shape/bit receipt')
    expected_name, actual_name = case_id + '.expected-bf16-u16le.bin', case_id + '.actual-bf16-u16le.bin'
    need(case['expected_file'] == expected_name and case['actual_file'] == actual_name, 'fixed raw filenames')
    expected, actual = (directory / expected_name).read_bytes(), (directory / actual_name).read_bytes()
    need(len(expected) == len(actual) == TAIL_BYTES and expected == actual, 'raw bytes mismatch or truncated')
    need(hashlib.sha256(expected).hexdigest() == case['expected_sha256'] == case['actual_sha256'], 'raw SHA receipt')
    words = struct.unpack('<30720H', actual)
    need(all(word == expected_word(kind, i) for i, word in enumerate(words)), 'raw fixture does not match independent pattern')
    need(case['parent_logical_bytes'] == (length + 3) * WIDTH * 2, 'parent shape/bytes')
    need(input_parent_sha(length, expected) == case['parent_sha256_before'] == case['parent_sha256_after'], 'full original parent SHA not reproduced')
    finite = all((word & 0x7f80) != 0x7f80 for word in words)
    if kind == 'finite':
        need(finite and case['finite'] is True and case['nan_payloads_are_data_not_finite_qualification'] is False, 'finite fixture contract')
    else:
        need(case['finite'] is None and case['nan_payloads_are_data_not_finite_qualification'] is True, 'NaN payloads incorrectly claimed finite qualification')
    counts = {'words': len(words), 'positive_zero': words.count(0), 'negative_zero': words.count(32768),
              'nan': sum((w & 0x7f80) == 0x7f80 and (w & 0x7f) != 0 for w in words),
              'infinity': sum((w & 0x7fff) == 0x7f80 for w in words)}
    return set(words) if kind != 'finite' else set(), counts


def audit_phases(rows, case, arm, child_pid, owner_pid):
    need([r['phase'] for r in rows] == list(PHASES), 'exact 11 memory phases required')
    need(all(r['case'] == case['case'] and r['pid'] == child_pid and r['ppid'] == owner_pid
             and r['groups_are_evaluated'] is True for r in rows), 'phase case/PID/evaluated roots')
    for i, row in enumerate(rows):
        for field in ('active_bytes', 'peak_bytes', 'cache_bytes', 'unique_buffer_bytes'):
            need(type(row[field]) is int and row[field] >= 0, 'invalid memory integer')
        need(row['unique_buffer_bytes'] <= row['active_bytes'] <= row['peak_bytes'] < LIMIT, 'observed memory guard exceeded')
        need(0 <= row['elapsed_s'] < 30, 'child phase beyond 30s diagnostic')
        if i:
            need(row['monotonic_s'] > rows[i-1]['monotonic_s'] and row['elapsed_s'] > rows[i-1]['elapsed_s'], 'nonmonotonic memory observations')
            need(row['peak_bytes'] >= rows[i-1]['peak_bytes'], 'peak reset inside case')
    parent = rows[1]['group_buffer_bytes']['parent']
    output = rows[3]['group_buffer_bytes']['output']
    need(parent >= case['parent_logical_bytes'], 'parent allocator smaller than logical payload')
    union = parent if arm == 'off' else parent + output
    all_groups = {'parent': parent, 'tail': parent, 'output': output}
    expected_groups = [{}, {'parent': parent, 'tail': parent},
        {'parent': parent, 'tail': parent} if arm == 'on-lazy' else all_groups,
        all_groups, all_groups, *[{'output': output} for _ in range(4)], {}, {}]
    expected_unions = [0, parent, parent if arm == 'on-lazy' else union,
                       union, union, output, output, output, output, 0, 0]
    for row, groups, backing in zip(rows, expected_groups, expected_unions):
        need(row['group_buffer_bytes'] == groups and row['unique_buffer_bytes'] == backing, 'group/full buffer union contract')
    release = rows[4]['active_bytes'] - rows[7]['active_bytes']
    if arm == 'off':
        need(output == parent, 'OFF lost original backing alias')
    else:
        need(TAIL_BYTES <= output <= TAIL_BYTES + 16384 and output < parent and release >= parent, 'owned copy/release not established')
    need(rows[-1]['active_bytes'] <= rows[0]['active_bytes'], 'final active arrays not returned to baseline')
    memory = {'original_backing_bytes': parent, 'retained_output_backing_bytes': output,
        'active_bytes_released_after_source_drop': release,
        'retained_backing_reduction_bytes': parent - output, 'source_release_proven_for_on': arm != 'off'}
    need(case['memory'] == memory, 'memory summary differs from independent phase calculation')
    return {'arm': arm, 'case': case['case'], 'length': case['length'], 'kind': case['kind'],
        **memory, 'observed_peak_bytes': max(r['peak_bytes'] for r in rows),
        'active_before_drop_bytes': rows[4]['active_bytes'], 'active_after_sync_bytes': rows[7]['active_bytes'],
        'cache_after_sync_bytes': rows[7]['cache_bytes'], 'active_final_bytes': rows[-1]['active_bytes']}


def audit_child(directory, arm, child_pid, owner_pid):
    complete = read_json(directory / 'complete.json')
    need(complete['status'] == 'PASS' and complete['arm'] == arm and complete['pid'] == child_pid
         and complete['ppid'] == owner_pid and complete['device_run'] is True, 'child terminal/owner identity')
    need(complete['source_pins_rechecked'] is True and complete['elapsed_wall_s'] < 30, 'final pin/deadline receipt')
    c = complete['contract']
    need(c['schema'] == 'E026-tiny-backing-v1' and c['arm'] == arm and c['case_count'] == 16
         and c['lengths'] == list(LENGTHS) and c['fixture_kinds'] == list(KINDS)
         and c['cache_limit_bytes'] == 0 and c['memory_boundary_guard_bytes'] == LIMIT
         and c['child_best_effort_alarm_seconds'] == 30
         and c['candidate_sha256'] == CANDIDATE_SHA and c['source_manifest_sha256'] == MANIFEST_SHA, 'fixed child contract')
    need(complete['runtime']['device']['device_name'] == 'Apple M5 Ultra'
         and complete['runtime']['metal']['architecture'] == 'applegpu_g17d', 'actual backend identity')
    pins = complete['source_pins']
    by_name = {Path(path).name: path for path in pins}
    need(pins[by_name['screen.py']] == SCREEN_SHA and pins[by_name['candidate.py']] == CANDIDATE_SHA
         and pins[by_name['source-pins.json']] == MANIFEST_SHA, 'executed fixed source SHA')
    manifest = read_json(by_name['source-pins.json'])
    source_root = Path(by_name['screen.py']).parent
    expected_pins = {str(source_root / 'screen.py'): SCREEN_SHA,
                     str(source_root / 'candidate.py'): CANDIDATE_SHA,
                     str(source_root / 'source-pins.json'): MANIFEST_SHA}
    expected_pins.update({str(source_root / v['copy']): v['sha256'] for v in manifest['sources'].values()})
    need(pins == expected_pins and all(sha(path) == expected for path, expected in pins.items()), 'independent source copy hash mismatch')
    rows, ledger_sha = [], hashlib.sha256()
    with (directory / 'phases.jsonl').open('rb') as stream:
        for raw in stream:
            ledger_sha.update(raw)
            need(raw.endswith(b'\n'), 'truncated phase line')
            rows.append(json.loads(raw))
    need(len(rows) == 176, 'exact 16*11 memory observations required')
    expected_cases = [(t, k) for t in LENGTHS for k in KINDS]
    need([(x['length'], x['kind']) for x in complete['cases']] == expected_cases, 'fixed ordered 16-case matrix')
    covered, totals, chart = {t: set() for t in LENGTHS}, Counter(), []
    for i, case in enumerate(complete['cases']):
        words, counts = audit_bits(directory, case)
        covered[case['length']].update(words); totals.update(counts)
        chart.append(audit_phases(rows[i*11:(i+1)*11], case, arm, child_pid, owner_pid))
    need(all(words == set(range(65536)) for words in covered.values()), 'all 65536 payloads not covered at every length')
    need(all(rows[i]['monotonic_s'] > rows[i-1]['monotonic_s'] for i in range(1, len(rows))), 'phase monotonicity across case boundaries')
    return {'arm': arm, 'status': 'PASS_CHILD_ONLY', 'pid': child_pid, 'case_count': 16,
        'phase_count': 176, 'source_pins': len(pins), 'raw_word_counts': dict(totals),
        'distinct_payload_words_per_length': {str(t): len(v) for t, v in covered.items()},
        'phase_ledger_sha256': ledger_sha.hexdigest(), 'memory_chart_rows': chart,
        'scope': 'Raw bits/backing observations only; requires separate owner/resource/lease audit before any qualification.'}


def audit_resources(owner, owner_pid, children, count):
    aggregate = read_json(owner / 'resources.json')
    paths = sorted(owner.glob('resource-[0-9][0-9][0-9].json'))
    need(len(paths) == len(aggregate) == count, 'raw/aggregate resource count')
    need([p.name for p in paths] == [f'resource-{i:03d}.json' for i in range(count)], 'resource sequence')
    samples = [read_json(p) for p in paths]
    need(samples == aggregate and samples[0]['label'] == 'preflight'
         and samples[-1]['label'] == 'postflight', 'raw/aggregate/pre-post resources')
    expected_labels = ['preflight']
    for arm in ARMS:
        expected_labels += [arm + '/before', arm + '/after']
    expected_labels += ['postflight']
    need([s['label'] for s in samples if not s['label'].endswith('/running')] == expected_labels, 'all arm boundary probes')
    first, low_pressure, running_counts, free_bytes = samples[0], 2, Counter(), []
    active_arm = None
    for i, s in enumerate(samples):
        for key in ('pressure_level', 'swap', 'vm_stat'):
            need(s[key]['returncode'] == 0, 'resource raw command failure')
        pressure = int(s['pressure_level']['text'].strip())
        need(pressure == s['pressure_level_value'] and pressure in (1, 2) and pressure <= low_pressure,
             'pressure worsened/critical/unknown')
        low_pressure = pressure
        vm = s['vm_stat']['text']
        size = int(re.search(r'page size of (\d+) bytes', vm)[1])
        free = int(re.search(r'^Pages free:\s*(\d+)\.', vm, re.M)[1]) * size
        need(free == s['real_free_bytes'] and free >= 50 << 30, 'physical free bytes below floor')
        free_bytes.append(free)
        for key, value in s['vm_counters'].items():
            need(int(re.search(r'^' + key + r':\s*(\d+)', vm, re.M)[1]) == value, 'raw VM counter mismatch')
        need(s['vm_counters']['Swapouts'] == first['vm_counters']['Swapouts'], 'new/reset Swapouts')
        used = float(re.search(r'used = ([0-9.]+)M', s['swap']['text'])[1])
        need(used == s['swap_used_mb'] and 0 <= used <= first['swap_used_mb'] + 16, 'swap used changed beyond contract')
        need(s['power_metadata']['source'] == 'AC Power' and 'AC Power' in s['power_metadata']['raw']['stdout']
             and s['processinfo_thermal_metadata']['code'] in (0, 1)
             and int(s['processinfo_thermal_metadata']['raw']['stdout']) == s['processinfo_thermal_metadata']['code'], 'power/thermal contract')
        need(s['native_process_inventory_ok'] is True and s['process_inventory_ok'] is True, 'unknown process inventory')
        service = [r for r in s['native_inference_processes'] if r['pid'] == 74384]
        need(len(service) == 1 and 'T' in service[0]['state'], 'authorized service not stopped')
        arm = s['label'].split('/')[0] if '/' in s['label'] else None
        if s['label'].endswith('/before'):
            need(active_arm is None, 'overlapping child arm resource windows')
            active_arm = arm
        elif s['label'].endswith('/running'):
            need(active_arm == arm, 'running probe outside corresponding child window')
        elif s['label'].endswith('/after'):
            need(active_arm == arm, 'after probe outside child window')
            active_arm = None
        expected_pid = children[arm] if s['label'].endswith('/running') else None
        if expected_pid is not None:
            running_counts[arm] += 1
            need(s['owned_model_pid'] == expected_pid, 'resource target child mismatch')
            exact = s['exact_owned_pid_metadata']
            # A child can finish while a resource probe is collecting its ps.
            if exact['status'] == 'known':
                need(exact['pid'] == expected_pid and exact['ppid'] == owner_pid, 'resource direct child identity')
                raw = exact['raw']['stdout'].split(None, 3)
                need([int(raw[0]), int(raw[1])] == [expected_pid, owner_pid], 'raw PID/PPID mismatch')
            else:
                need(exact['raw']['returncode'] == 1 and not exact['raw']['stdout'].strip(), 'unexplained unknown owned process')
        for process in s['native_inference_processes'] + s['known_inference_processes']:
            need(process['pid'] in (74384, expected_pid), 'foreign inference process')
            if process['pid'] == expected_pid:
                need(process['ppid'] == owner_pid, 'foreign parent for child')
        if i:
            need(s['monotonic_s'] > samples[i-1]['monotonic_s'], 'resource times not monotonic')
    need(set(running_counts) == set(ARMS), 'running probes missing for arm')
    return {'samples': count, 'running_samples': dict(running_counts),
        'initial_pressure': first['pressure_level_value'], 'final_pressure': samples[-1]['pressure_level_value'],
        'minimum_physical_free_bytes': min(free_bytes), 'swapouts_delta': 0,
        'max_swap_used_growth_mib': max(s['swap_used_mb'] for s in samples) - first['swap_used_mb'],
        'scope': 'Fixed tiny diagnostic policy only; sampled boundaries cannot rule out unobserved transients, and do not satisfy a pressure-1 performance contract.'}


def audit(owner, lease, plan_sha256):
    need(re.fullmatch('[0-9a-f]{64}', plan_sha256), 'explicit frozen plan SHA required')
    required = [owner / 'complete.json', owner / 'resources.json', lease / 'lease.json',
                lease / 'resume.json', lease / 'job-0.json']
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        return {'status': 'INCOMPLETE', 'qualified': False, 'missing_terminal_evidence': missing}
    complete, lease_data = read_json(required[0]), read_json(required[2])
    if not lease_data['closed']:
        return {'status': 'INCOMPLETE', 'qualified': False, 'reason': 'service lease not closed'}
    need(complete['status'] == 'PASS' and complete['errors'] == [] and complete['elapsed_wall_s'] <= 180,
         'owner did not pass all tiny resource/runtime gates')
    owner_pid = complete['owner_pid']
    plan = read_json(owner / 'plan.json')
    need(sha(owner / 'plan.json') == plan_sha256 and plan['schema'] == 'E026-tiny-owner-v1'
         and plan['arms'] == list(ARMS) and plan['arm_seconds'] == 30 and plan['total_seconds'] == 180, 'frozen owner plan')
    attempt = read_json(owner / 'attempt.json')
    need(attempt['owner_pid'] == owner_pid and attempt['plan_sha256'] == plan_sha256
         and attempt['environment'] == plan['execution']['environment'], 'owner launch identity/environment')
    need(attempt['environment'] == {'MLX_MAX_OPS_PER_BUFFER': '50', 'MLX_MAX_MB_PER_BUFFER': '50',
         'MLX_BFS_MAX_WIDTH': '20', 'MLX_METAL_FAST_SYNCH': '0'}, 'four fixed MLX environment values')
    pins = plan['execution']['pins']; verified = read_json(owner / 'verified-pins.json')
    need(set(pins) == set(verified), 'owner verified pin coverage')
    for path, expected in pins.items():
        row = verified[path]
        copied = Path(path).suffix in ('.py', '.json', '.md', '.h', '.cpp', '.txt')
        source = owner / 'frozen-source' / path.lstrip('/') if copied else Path(path)
        need(row['sha256'] == expected and row['source_copied'] is copied and source.stat().st_size == row['bytes']
             and sha(source) == expected, 'independent owner source/runtime pin mismatch: ' + path)
    screen = plan['execution']['screen']
    need(screen['sha256'] == SCREEN_SHA and pins[screen['path']] == SCREEN_SHA, 'owner screen source identity')
    children, results, releases = {}, [], []
    for arm in ARMS:
        directory = owner / arm
        child, release = read_json(directory / 'child.json'), read_json(directory / 'release.json')
        pid = child['pid']; children[arm] = pid
        need(child['arm'] == arm and child['ppid'] == owner_pid, 'direct child identity')
        need(child['command'] == [plan['execution']['python'], '-I', '-B', screen['path'], '--run', '--arm', arm,
             '--out', str(directory / 'screen')], 'actual command arm/path')
        need(release['pid'] == pid and release['returncode'] == 0 and release['waited_reaped'] is True
             and release['strict_absence'] is True and release['cleanup_notes'] == []
             and release['absence_probe']['returncode'] == 1 and not release['absence_probe']['stdout'].strip(), 'child exit/reap/absence')
        loaded = read_json(directory / 'screen/complete.json')['runtime']
        resolved_pins = {str(Path(path).resolve()) for path in pins}
        need(all(str(Path(loaded[key]).resolve()) in resolved_pins for key in ('mlx_file', 'numpy_file')), 'child runtime path outside owner pins')
        results.append(audit_child(directory / 'screen', arm, pid, owner_pid)); releases.append(release)
    need(len(set(children.values())) == 3, 'three fresh child processes required')
    need(complete['arms'] == [{'arm': arm, 'pid': children[arm], 'status': 'PASS', 'cases': 16, 'phases': 176}
                            for arm in ARMS], 'owner arm summary')
    resources = audit_resources(owner, owner_pid, children, complete['samples'])
    job, resume = read_json(lease / 'job-0.json'), read_json(lease / 'resume.json')
    need(lease_data['child_pid'] == job['pid'] == owner_pid and len(lease_data['jobs']) == 1
         and job['command'] == lease_data['jobs'][0] and job['returncode'] == 0 and job['waited'] is True, 'outer owner exit/wait')
    need(lease_data['service_pid'] == 74384 and lease_data['paused_at'] < resume['time'] <= lease_data['deadline']
         and resume['identity_match'] is True and 'T' in resume['state_before']
         and resume['state_after'] and 'T' not in resume['state_after'], 'same service restoration')
    return {'status': 'PASS', 'qualified': True, 'qualification_scope': 'E026 tiny bits/backing diagnostic only',
        'performance_qualified': False, 'full_model_qualified': False, 'plan_sha256': plan_sha256,
        'owner_pid': owner_pid, 'owner_execution_pins_verified': len(pins), 'arms': results,
        'resources': resources, 'releases': releases, 'service_restore': resume,
        'chartdata': {'kind': 'memory_only_not_pareto', 'units': 'bytes',
                      'rows': [row for result in results for row in result['memory_chart_rows']]},
        'limits': 'Recorded observed active/peak bounds are not a hard allocation cap. No full-model numeric qualification, E023 cause, throughput/latency or production adoption.'}


def self_test():
    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp)
        for kind in KINDS:
            raw = struct.pack('<30720H', *(expected_word(kind, i) for i in range(WORDS)))
            name = f'T1-{kind}'
            expected, actual = name + '.expected-bf16-u16le.bin', name + '.actual-bf16-u16le.bin'
            (path / expected).write_bytes(raw); (path / actual).write_bytes(raw)
            parent_sha = input_parent_sha(1, raw)
            need(parent_sha == hashlib.sha256(b'\x80\x3f' * WIDTH + raw).hexdigest(), 'streamed parent SHA logic')
            case = {'case': name, 'status': 'PASS', 'length': 1, 'kind': kind, 'dtype': 'bfloat16',
                'shape': [3, WIDTH], 'bit_equal': True, 'expected_file': expected, 'actual_file': actual,
                'expected_sha256': hashlib.sha256(raw).hexdigest(), 'actual_sha256': hashlib.sha256(raw).hexdigest(),
                'parent_logical_bytes': 81920, 'parent_sha256_before': parent_sha, 'parent_sha256_after': parent_sha,
                'finite': True if kind == 'finite' else None,
                'nan_payloads_are_data_not_finite_qualification': kind != 'finite'}
            audit_bits(path, case)
            corrupted = bytearray(raw); corrupted[0] ^= 1; (path / actual).write_bytes(corrupted)
            try:
                audit_bits(path, case)
            except ValueError:
                pass
            else:
                raise AssertionError('raw bit corruption accepted')
        for arm in ARMS:
            parent, output = 81920, 81920 if arm == 'off' else 61440
            union = parent if arm == 'off' else parent + output
            groups = [{}, {'parent': parent, 'tail': parent},
                {'parent': parent, 'tail': parent} if arm == 'on-lazy' else {'parent': parent, 'tail': parent, 'output': output},
                *[{'parent': parent, 'tail': parent, 'output': output} for _ in range(2)],
                *[{'output': output} for _ in range(4)], {}, {}]
            allocations = [0, parent, parent if arm == 'on-lazy' else union, union, union,
                           output, output, output, output, 0, 0]
            rows = []
            for i, (phase, g, active) in enumerate(zip(PHASES, groups, allocations)):
                rows.append({'case': 'T1-finite', 'phase': phase, 'pid': 2, 'ppid': 1,
                    'groups_are_evaluated': True, 'monotonic_s': 100 + i * .01, 'elapsed_s': 1 + i * .01,
                    'active_bytes': active, 'peak_bytes': max(allocations[:i+1]), 'cache_bytes': 0,
                    'unique_buffer_bytes': active, 'group_buffer_bytes': g})
            case = {'case': 'T1-finite', 'length': 1, 'kind': 'finite', 'parent_logical_bytes': parent,
                'memory': {'original_backing_bytes': parent, 'retained_output_backing_bytes': output,
                    'active_bytes_released_after_source_drop': union - output,
                    'retained_backing_reduction_bytes': parent - output, 'source_release_proven_for_on': arm != 'off'}}
            audit_phases(rows, case, arm, 2, 1)
            for field, value in (('unique_buffer_bytes', 1), ('pid', 3), ('peak_bytes', LIMIT)):
                bad = copy.deepcopy(rows); bad[3][field] = value
                try:
                    audit_phases(bad, case, arm, 2, 1)
                except ValueError:
                    pass
                else:
                    raise AssertionError('memory/PID corruption accepted: ' + field)
    need(len(PHASES) == 11 and len(LENGTHS) * len(KINDS) * len(PHASES) == 176, 'fixed phase count')
    print('PASS: independent raw patterns/finite/parent SHA and byte-corruption checks; no device execution')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--owner', type=Path)
    parser.add_argument('--lease', type=Path)
    parser.add_argument('--plan-sha256')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args(argv)
    if args.self_test:
        self_test(); return 0
    if not all((args.owner, args.lease, args.plan_sha256, args.out)):
        parser.error('--owner --lease --plan-sha256 --out required')
    with args.out.open('x') as output:
        try:
            result = audit(args.owner.resolve(), args.lease.resolve(), args.plan_sha256)
        except Exception as error:
            result = {'status': 'FAIL', 'qualified': False, 'error': str(error), 'traceback': traceback.format_exc()}
        result['audit_source_sha256'] = sha(__file__)
        json.dump(result, output, indent=2); output.write('\n')
    print(json.dumps({'status': result['status'], 'out': str(args.out)}))
    return {'PASS': 0, 'INCOMPLETE': 2, 'FAIL': 1}[result['status']]


if __name__ == '__main__':
    raise SystemExit(main())
