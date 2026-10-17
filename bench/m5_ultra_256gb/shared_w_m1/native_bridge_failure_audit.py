"""Independent stdlib audit of the fixed E027 native bridge rejection.

This audits a numerical FAIL and its successful cleanup, never grants candidate
qualification. Reuses pinned independent raw audit primitives, not producers.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import struct
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
BASE_SHA = '57bb96046ca651ef34bc7f36f0e2cc6b535c571a93063311528ac81d3b04f25e'
SCREEN_SHA = '74f1471247abc8a6157791940fd932dff7fc1d2bb266792c6d3c921360092543'
OWNER_SHA = '59273b5b7ab9f51854646b5b98227cb75d99b25f579a93ae13c4c637a5030118'
PLAN_SHA = 'f52346375b8a2ccecb861240c283d1215d8fee2880f41ae38116ac5b9c3ada59'


def need(value, text):
    if not value: raise ValueError(text)


def read(path): return json.loads(Path(path).read_text())
def same(a, b): return json.dumps(a, sort_keys=True, allow_nan=False) == json.dumps(b, sort_keys=True, allow_nan=False)
def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while data := f.read(1 << 20): h.update(data)
    return h.hexdigest()


def load_audit():
    need(sha(HERE / 'audit.py') == BASE_SHA, 'independent base auditor changed')
    spec = importlib.util.spec_from_file_location('_native_bridge_raw_audit', HERE / 'audit.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module, module.helpers()


def audit_raw(out, child, base, primitive):
    cases, checks, cache = child['cases'], child['checks'], {}
    expected = base.expected_cases()[:19]
    need([r['case'] for r in cases] == [r['case'] for r in expected], 'exact fixed case prefix')
    need(len(checks) == 56 and same(checks, read(out / 'checks.json')), '56 comparison coverage')
    need([r['status'] for r in cases] == ['PASS'] * 18 + ['FAIL'], 'case status prefix')
    banks = child['bank_receipts']
    need([b['bank'] for b in banks] == [b['id'] for b in base.expected_banks()[:6]], 'six complete banks')
    for bank, meta in zip(banks, base.expected_banks()):
        need([bank['launch_count'], bank['kernel_build_count']] == [3, 1], 'bank callable reuse')
        for name, dtype, shape in (('weight', 'uint32', [meta['N'], 640]), ('scales', 'uint8', [meta['N'], 320])):
            a, b = bank['inputs_before'][name], bank['inputs_after'][name]
            p = base.descriptor(out, a, dtype, shape, primitive, cache)
            q = base.descriptor(out, b, dtype, shape, primitive, cache)
            need(same(base.bare(a), base.bare(b)) and p.read_bytes() == q.read_bytes(), 'bank bytes changed')
            for side, value in (('before', a), ('after', b)):
                base.layout(out, bank['layouts_' + side][name], value, meta['id'] + '/' + name + '-' + side,
                            f'banks/{meta["id"]}/{name}-layout-{side}.bin')
    label_pairs = [('control', 'original-expression-control-vs-native-M1'),
                   ('native_multirow', 'native-multirow-vs-native-M1'), ('shared', 'shared-W-vs-native-M1')]
    checked, compared, failure = 0, 0, None
    for index, (row, meta) in enumerate(zip(cases, expected)):
        case, t, n = meta['case'], meta['T'], meta['N']
        need(same(row, read(out / 'cases' / case / 'case.json')), 'case JSON binding')
        need([row[k] for k in ('T', 'N', 'K', 'bank')] == [t, n, 5120, meta['id']], 'case dimensions')
        need(row['inputs_unchanged'] is True and set(row['inputs_before']) == set(row['inputs_after']) == {'x', 'weight', 'scales'}, 'input coverage')
        for name, dtype, shape in (('x', 'bfloat16', [t, 5120]), ('weight', 'uint32', [n, 640]), ('scales', 'uint8', [n, 320])):
            a, b = row['inputs_before'][name], row['inputs_after'][name]
            path = base.descriptor(out, a, dtype, shape, primitive, cache)
            q = base.descriptor(out, b, dtype, shape, primitive, cache, with_file=name == 'x')
            need(same(base.bare(a), base.bare(b)), 'input hashes/identity changed')
            if q: need(path.read_bytes() == q.read_bytes(), 'X before/after actual bytes')
            if index < 18 and name != 'x':
                bank = banks[index // 3]
                need(same(a, bank['inputs_before'][name]) and same(b, base.bare(bank['inputs_after'][name])), 'case/bank binding')
            if name == 'scales':
                need(a['sha256'] == base.repeated_sha(base.SCALE_CODES, n * 320), 'scale bank pattern')
            if meta['bank'] != 'random':
                if name == 'x': need(a['sha256'] == base.structured_input_sha(meta['bank'], t, 5120), 'structured X pattern')
                if name == 'weight':
                    pair = (0x76543210, 0xfedcba98) if meta['bank'] == 'codes_signed_boundaries' else (0x77777777, 0xffffffff)
                    need(a['sha256'] == base.repeated_sha(struct.pack('<2I', *pair), n * 640 * 4), 'structured W pattern')
            elif name == 'x':
                need(all(v[0] & 0x7fff <= 0x4080 for v in struct.iter_unpack('<H', path.read_bytes())), 'random X magnitude')
        for side in ('before', 'after'):
            base.layout(out, row['layout_' + side], row['inputs_' + side]['x'], case + '/x-' + side,
                        f'cases/{case}/x-layout-{side}.bin')
        pairs = label_pairs if index < 18 else label_pairs[:2]
        need(set(row['outputs']) == {'native'} | {k for k, _ in pairs}, 'no missing or unexpected implementation output')
        products = {name: base.descriptor(out, desc, 'bfloat16', [t, n], primitive, cache)
                    for name, desc in row['outputs'].items()}
        for name, label in pairs:
            saved = checks[checked]; checked += 1
            need(saved['case'] == case and saved['label'] == label and same(saved['reference'], row['outputs']['native'])
                 and same(saved['actual'], row['outputs'][name]), 'comparison output binding')
            diff = primitive.compare_raw(products['native'], products[name], 'BF16')
            compared += products['native'].stat().st_size
            need(saved['exact'] is diff['equal'] and saved['status'] == ('PASS' if diff['equal'] else 'FAIL'), 'comparison raw equality')
            if not diff['equal']:
                need(checked == 56 and case == 'synthetic-N10240-random-T2' and name == 'native_multirow', 'unexpected first failure')
                a, b = products['native'].read_bytes(), products[name].read_bytes()
                offset = next(i for i, (left, right) in enumerate(zip(a, b)) if left != right)
                need(saved['first_differing_byte'] == offset == 13142, 'first differing byte')
                element = offset // 2
                word_a, word_b = struct.unpack_from('<H', a, element * 2)[0], struct.unpack_from('<H', b, element * 2)[0]
                differing = sum(a[i:i + 2] != b[i:i + 2] for i in range(0, len(a), 2))
                failure = dict(case=case, label=label, byte_offset=offset, element_index=element,
                    coordinate=[element // n, element % n], reference_bf16_bits=word_a, native_multirow_bf16_bits=word_b,
                    reference_float32=struct.unpack('<f', struct.pack('<I', word_a << 16))[0],
                    native_multirow_float32=struct.unpack('<f', struct.pack('<I', word_b << 16))[0],
                    differing_bf16_elements=differing, total_bf16_elements=t*n,
                    reference_sha256=sha(products['native']), native_multirow_sha256=sha(products[name]))
        if index < 18:
            order = base.ROWS.index(t)
            need(same(row['route'], dict(launch_count_before=order, launch_count_after=order+1,
                      kernel_build_count=1, required_route='shared-W', fallback=False)), 'shared actual host route')
        else:
            need('route' not in row and not (out / 'cases' / case / 'shared.bf16.bin').exists(), 'shared executed after rejection')
            for name in ('weight', 'scales'):
                record = read(out / 'banks' / meta['id'] / (name + '-layout-before.json'))
                base.layout(out, record, row['inputs_before'][name], meta['id'] + '/' + name + '-before',
                            f'banks/{meta["id"]}/{name}-layout-before.bin')
    need(failure is not None and checked == 56, 'required observed numerical failure')
    layouts = len(list(out.glob('banks/*/*-layout-*.bin'))) + len(list(out.glob('cases/*/x-layout-*.bin')))
    need(layouts == 64, '64 saved descriptor records')
    need(len(list((out / 'cases').iterdir())) == 19, 'unrecorded extra cases')
    return dict(cases=19, completed_cases=18, comparisons=56, passed_comparisons=55, failed_comparisons=1,
                completed_banks=6, partially_observed_banks=1, actual_gpu_descriptors=layouts,
                tensor_files_checked=len(cache), compared_output_bytes=compared, first_difference=failure)


def audit_phases(out, child, primitive):
    rows = primitive.jsonl(out / 'phases.jsonl')
    expected = [('global', 'initial')] + [(c['case'], phase) for c in child['cases'] for phase in ('before_case', 'after_case')]
    need(len(rows) == len(child['physical_receipts']) == len(expected) == 39, 'partial phases completeness')
    elapsed, peak, auxpeak = -1, -1, -1
    for i, (row, pair) in enumerate(zip(rows, expected)):
        need([row[k] for k in ('index', 'case', 'phase', 'pid', 'ppid', 'status')]
             == [i, *pair, child['pid'], child['ppid'], 'PASS'], 'phase sequence/identity')
        need(type(row['elapsed_s']) in (int, float) and math.isfinite(row['elapsed_s'])
             and elapsed <= row['elapsed_s'] <= child['elapsed_wall_s'], 'phase time')
        elapsed = row['elapsed_s']
        need(all(type(row[k]) is int and row[k] >= 0 for k in ('active_bytes', 'cache_bytes', 'peak_bytes'))
             and row['cache_bytes'] == 0 and row['active_bytes'] <= row['peak_bytes'] <= 1 << 30
             and row['peak_bytes'] >= peak, 'MLX observation budget/cache')
        peak = row['peak_bytes']
        raw = out / f'physical-{i:03d}.raw.json'; cmd = read(out / f'physical-{i:03d}.command.json')
        need(cmd['command'] == ['/usr/bin/footprint', '-p', str(child['pid']), '--noCategories', '-f', 'bytes', '-j', str(raw)]
             and cmd['returncode'] == 0, 'self physical command')
        receipt = dict(path=raw.name, sha256=sha(raw), pid=child['pid'], phase=pair[1], case=pair[0])
        need(same(row['physical_receipt'], receipt) and same(child['physical_receipts'][i], receipt), 'physical receipt')
        values = primitive.physical_values(read(raw), child['pid'])
        for field in ('current_bytes', 'peak_bytes', 'top_bytes', 'aux_current_bytes', 'aux_peak_bytes'):
            need(row['physical_' + field] == values[field], 'raw physical values')
        need(values['aux_peak_bytes'] >= auxpeak, 'auxiliary peak monotonicity'); auxpeak = values['aux_peak_bytes']
    return dict(observed_phases=39, final_release_phase_observed=False,
                maximum_mlx_peak_bytes=peak, maximum_physical_peak_bytes=max(r['physical_peak_bytes'] for r in rows),
                maximum_physical_current_bytes=max(r['physical_current_bytes'] for r in rows))


def audit_resources(owner, result, launch, primitive):
    rows = read(owner / 'resources.json')
    need([r['label'] for r in rows] == ['preflight', 'running', 'postflight']
         and result['samples'] == 3 and result['physical_samples'] == 1, 'failure resource lifecycle')
    first, pressure, times = rows[0], 2, []
    for row in rows:
        need(row['pressure_level']['returncode'] == 0 and type(row['pressure_level_value']) is int
             and int(row['pressure_level']['text'].strip()) == row['pressure_level_value']
             and row['pressure_level_value'] in (1, 2) and row['pressure_level_value'] <= pressure, 'pressure gate')
        pressure = row['pressure_level_value']; times.append(row['monotonic_s'])
        vm = row['vm_stat']; page = int(re.search(r'page size of (\d+) bytes', vm['text'])[1])
        free = int(re.search(r'^Pages free:\s*(\d+)\.', vm['text'], re.M)[1]) * page
        need(vm['returncode'] == 0 and free == row['real_free_bytes'] and free >= 50 << 30, 'actual free pages')
        for key, value in row['vm_counters'].items():
            need(type(value) is int and int(re.search('^' + key + r':\s*(\d+)', vm['text'], re.M)[1]) == value, 'raw VM counter')
        need(row['vm_counters']['Swapouts'] == first['vm_counters']['Swapouts'], 'Swapouts increase')
        used = float(re.search(r'used = ([\d.]+)M', row['swap']['text'])[1])
        need(row['swap']['returncode'] == 0 and used == row['swap_used_mb'] and 0 <= used <= first['swap_used_mb'] + 16, 'swap budget')
        power, thermal = row['power_metadata'], row['processinfo_thermal_metadata']
        need(power['status'] == thermal['status'] == 'known' and power['source'] == 'AC Power'
             and power['raw']['returncode'] == thermal['raw']['returncode'] == 0 and 'AC Power' in power['raw']['stdout']
             and type(thermal['code']) is int and thermal['code'] in (0, 1) and thermal['raw']['stdout'].strip() == str(thermal['code']), 'power/thermal')
        need(row['native_process_inventory_ok'] is True and row['process_inventory_ok'] is True
             and row['authorized_suspended_pids'] == [74384], 'known sole ownership')
        native = row['native_inference_processes']; service = [v for v in native if v['pid'] == 74384]
        need(len(service) == 1 and service[0]['authorized_suspended'] is True and 'T' in service[0]['state'], 'service paused')
        allowed = (74384, launch['pid']) if row['label'] == 'running' else (74384,)
        for process in native + row['known_inference_processes']:
            need(process['pid'] in allowed, 'foreign process')
            if process['pid'] == launch['pid']:
                need(process['owned'] is True and process['ppid'] == launch['ppid'], 'direct child ownership')
    need(times == sorted(set(times)), 'resource time order')
    running = rows[1]
    need(running['owned_model_pid'] == launch['pid'] and running['physical_receipt'] == 'footprint-000.json'
         and running['terminal_exit0_observed'] is False, 'live footprint binding')
    exact = running['exact_owned_pid_metadata']
    need(exact['status'] == 'known' and exact['pid'] == launch['pid'] and exact['ppid'] == launch['ppid'], 'live exact PID')
    value = primitive.owner_footprint(owner, read(owner / 'footprint-000.json'), launch['pid'], launch['ppid'])
    need(value is not None, 'actual measured owner physical sample')
    return dict(samples=3, owner_physical_samples=1, first_pressure=rows[0]['pressure_level_value'],
                last_pressure=rows[-1]['pressure_level_value'], minimum_free_bytes=min(r['real_free_bytes'] for r in rows),
                swapouts_growth=0, maximum_swap_growth_mib=max(r['swap_used_mb'] for r in rows)-first['swap_used_mb'])


def audit(owner, lease):
    base, primitive = load_audit()
    plan, result = read(owner / 'plan.json'), read(owner / 'complete.json')
    out = owner / 'child/screen'; child = read(out / 'complete.json')
    need(sha(owner / 'plan.json') == PLAN_SHA and plan['schema'] == 'E027-native-bridge-v1'
         and [plan['expected_cases'], plan['expected_comparisons']] == [33, 99]
         and same(plan['resource_policy'], primitive.POLICY) and same(plan['execution']['environment'], primitive.ENV), 'fixed preregistration')
    pins = base.source_pins(owner, plan)
    need(pins[str(HERE / 'native_bridge_screen.py')] == SCREEN_SHA and pins[str(HERE / 'native_bridge_owner.py')] == OWNER_SHA, 'executed bridge source')
    need(result['schema'] == 'E027-native-bridge-owner-v1' and result['status'] == child['status'] == 'FAIL'
         and same(result['failures'], read(owner / 'failures.json')) and len(result['failures']) == 1
         and 0 < result['elapsed_wall_s'] <= 480 and 0 < child['elapsed_wall_s'] <= 120
         and child['device_run'] is True and child['performance_qualified'] is False and child['full_model_qualified'] is False, 'recorded failure identity')
    need(child['error'] == 'native multirow differs from M1; direct replacement NOT qualified'
         and 'source_pins_rechecked' not in child and 'bundle_rechecked' not in child, 'failure reason/unfinished terminal gates')
    contract = child['contract']
    need(contract['schema'] == 'M5-shared-w-native-bridge-v1' and same(contract['cases'], base.expected_cases())
         and [contract['expected_cases'], contract['expected_comparisons'], contract['expected_phases'], contract['expected_layouts']] == [33, 99, 68, 110], 'full registered workload')
    for path, digest in child['sources'].items():
        need(pins.get(path) == digest and sha(out / 'frozen-source' / Path(path).relative_to(base.ROOT)) == digest, 'loaded source receipt')
    need(child['sources'][str(HERE / 'native_bridge_screen.py')] == SCREEN_SHA, 'actual new screen loaded')
    runtime = child['runtime']
    need(runtime['mlx_version'] == '0.32.3' and runtime['device']['device_name'] == 'Apple M5 Ultra'
         and runtime['device']['architecture'] == 'applegpu_g17d', 'actual runtime/device')
    launch, release = read(owner / 'child/launch.json'), read(owner / 'child/release.json')
    expected = [plan['execution']['python'], '-I', '-B', str(HERE / 'native_bridge_screen.py'), '--run', '--out', str(out),
                '--bundle-manifest', plan['bundle']['path'], '--bundle-sha256', plan['bundle']['sha256']]
    need(launch['command'] == expected and same(launch['environment'], primitive.ENV) and launch['pid'] == child['pid']
         and launch['ppid'] == result['owner_pid'] == child['ppid'] and launch['arm_seconds'] == 120
         and 0 < launch['arm_deadline_monotonic'] - launch['spawned_monotonic'] <= 120, 'direct bounded child')
    need(release['pid'] == child['pid'] and release['returncode'] == 1 and release['waited_reaped'] is True
         and release['strict_absence'] is True and release['cleanup_notes'] == []
         and release['absence_probe']['returncode'] == 1 and not release['absence_probe']['stdout'].strip(), 'failure reaped/absent')
    raw = audit_raw(out, child, base, primitive)
    memory = audit_phases(out, child, primitive); resources = audit_resources(owner, result, launch, primitive)
    data, resume, job = read(lease / 'lease.json'), read(lease / 'resume.json'), read(lease / 'job-0.json')
    command = job['command']; need(len(command) == 10 and command[1:4] == ['-I', '-B', str(HERE / 'native_bridge_owner.py')], 'lease owner command')
    flags = dict(zip(command[4::2], command[5::2]))
    need(set(flags) == {'--plan', '--plan-sha256', '--out'} and flags['--plan-sha256'] == PLAN_SHA
         and flags['--out'] == str(owner) and sha(flags['--plan']) == PLAN_SHA
         and str(Path(command[0]).resolve()) in {str(Path(p).resolve()) for p in pins}, 'lease exact plan binding')
    need(data['closed'] is True and data['jobs'] == [command] and data['child_pid'] == job['pid'] == result['owner_pid']
         and job['returncode'] == 1 and job['waited'] is True, 'closed failed-job lease')
    need(data['service_pid'] == 74384 and data['service_identity'] == 'Fri Oct 16 08:01:58 2026     omlx-server'
         and data['paused_at'] < resume['time'] <= data['deadline'] and resume['identity_match'] is True
         and resume['state_before'] == 'T' and resume['state_after'] == 'S', 'same service restored')
    manifest = {str(path): dict(bytes=path.stat().st_size, sha256=sha(path))
                for directory in (owner, lease) for path in sorted(directory.rglob('*')) if path.is_file()}
    return dict(audit_status='PASS', outcome='NUMERICAL_REJECT', qualified=False, performance_qualified=False,
        full_model_qualified=False, raw=raw, memory=memory, resources=resources, release=release, service_restore=resume,
        plan_sha256=PLAN_SHA, source_pins_checked=len(pins), owner=str(owner), lease=str(lease), raw_root=str(out),
        evidence_file_manifest=manifest,
        limits=['Only the first 18 cases and two checks in case 19 executed; remaining 14 cases untested.',
                'No shared candidate call in the failed case; whole three-geometry native substitution is rejected.',
                'Different native multirow and M1 rounding does not establish that the native operation is incorrect.',
                'Final child release phase and child terminal source/bundle rechecks were not reached; parent pins were rechecked.',
                'Last partial bank has full W/scales before bytes and unchanged after hashes, but no full after raw bank files.',
                'No full model, HTTP, speed, or production qualification; smaller shapes are not promoted from the passing prefix.'])


def corruption_tests(owner, lease):
    """Only mutate in-memory metadata; never write original run artifacts."""
    original_read = read
    mutations = [
        ('false child PASS', owner / 'child/screen/complete.json', lambda x: x.update(status='PASS')),
        ('dropped case', owner / 'child/screen/complete.json', lambda x: x['cases'].pop()),
        ('forged first byte', owner / 'child/screen/checks.json', lambda x: x[-1].update(first_differing_byte=13144)),
        ('not reaped', owner / 'child/release.json', lambda x: x.update(waited_reaped=False)),
        ('unrestored service', lease / 'resume.json', lambda x: x.update(identity_match=False)),
        ('swap growth', owner / 'resources.json', lambda x: x[-1]['vm_counters'].update(Swapouts=x[-1]['vm_counters']['Swapouts'] + 1)),
    ]
    rejected = []
    for name, target, mutate in mutations:
        def modified(path):
            result = original_read(path)
            if Path(path) == target: mutate(result)
            return result
        with patch.dict(globals(), read=modified):
            try: audit(owner, lease)
            except ValueError: rejected.append(name)
            else: raise AssertionError('corruption accepted: ' + name)
    return rejected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--owner', type=Path, required=True); parser.add_argument('--lease', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--self-test', action='store_true'); args = parser.parse_args()
    need(not args.out.exists(), 'refuse overwrite')
    result = audit(args.owner.resolve(), args.lease.resolve())
    if args.self_test:
        result['corruption_tests_rejected'] = corruption_tests(args.owner.resolve(), args.lease.resolve())
    result.update(audit_source_sha256=sha(__file__), base_audit_sha256=BASE_SHA)
    with args.out.open('x') as f: json.dump(result, f, indent=2, allow_nan=False); f.write('\n')
    print(json.dumps({key: result[key] for key in ('audit_status', 'outcome', 'qualified')}))


if __name__ == '__main__': main()
