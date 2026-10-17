"""Independent CPU audit of the E025 v2 early numerical rejection.

Reads safetensors directly with stdlib: no MLX, NumPy, device, process or service
calls. Source helper reuse is limited to rebuilding pinned Metal text. An audit
PASS means the recorded early FAIL is supported; it never qualifies the candidate.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import struct
import sys
import traceback

PLAN_SHA = '3beae0672fd90d6a83a3869f66164273229751151d99868259dcb3f1c4b16a12'
LABELS = ['gdn_z-M19/OFF-vs-original', 'gdn_z-M19/reference-vs-original-QMM', 'gdn_z-M19/every-partial']


def need(ok, message):
    if not ok: raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''): h.update(block)
    return h.hexdigest()


def read(path): return json.loads(Path(path).read_text())
def write(path, data): Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
def rows(path): return [json.loads(s) for s in Path(path).read_text().splitlines()]


def safe_path(root, relative):
    path = (root / relative).resolve()
    need(path.is_relative_to(root.resolve()), 'artifact path escapes root')
    return path


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        need(key not in result, 'duplicate JSON header key')
        result[key] = value
    return result


def tensors(path):
    raw = Path(path).read_bytes()
    need(len(raw) >= 8, 'safetensors prefix missing')
    size = struct.unpack('<Q', raw[:8])[0]
    need(0 < size <= 1 << 20 and 8 + size <= len(raw), 'safetensors header bounds')
    header = json.loads(raw[8:8 + size], object_pairs_hook=unique_object)
    data = raw[8 + size:]; spans = []; result = {}
    for name, info in header.items():
        if name == '__metadata__':
            need(info is None or isinstance(info, dict), 'bad safetensors metadata'); continue
        need(set(info) == {'dtype', 'shape', 'data_offsets'}, 'tensor metadata fields')
        shape, offsets, dtype = info['shape'], info['data_offsets'], info['dtype']
        need(isinstance(shape, list) and all(type(v) is int and v >= 0 for v in shape), 'invalid shape')
        need(dtype in {'BF16', 'U32', 'U8'}, 'unexpected dtype')
        need(isinstance(offsets, list) and len(offsets) == 2 and all(type(v) is int for v in offsets), 'invalid offsets')
        lo, hi = offsets; width = {'BF16': 2, 'U32': 4, 'U8': 1}[dtype]
        need(0 <= lo <= hi <= len(data) and hi - lo == math.prod(shape) * width, 'tensor byte/shape mismatch')
        value = data[lo:hi]
        finite = all((x[0] & 0x7f80) != 0x7f80 for x in struct.iter_unpack('<H', value)) if dtype == 'BF16' else None
        result[name] = dict(shape=shape, dtype=dtype, bytes=hi-lo, sha256=hashlib.sha256(value).hexdigest(), finite=finite, raw=value)
        spans.append((lo, hi))
    cursor = 0
    for lo, hi in sorted(spans):
        need(lo == cursor, 'safetensors data gap/overlap'); cursor = hi
    need(cursor == len(data), 'safetensors trailing payload')
    return result


def meta(tensor): return {k: v for k, v in tensor.items() if k != 'raw'}
def bf16(word): return struct.unpack('<f', struct.pack('<I', word << 16))[0]


def compare_pair(path, expected_shape):
    data = tensors(path)
    need(set(data) == {'reference', 'candidate'}, 'pair tensor names')
    a, b = (data[k] for k in ('reference', 'candidate'))
    need(a['shape'] == b['shape'] == expected_shape and a['dtype'] == b['dtype'] == 'BF16', 'pair shape/dtype')
    need(a['finite'] is True and b['finite'] is True, 'nonfinite pair')
    differences = [i for i, (x, y) in enumerate(zip(a['raw'], b['raw'])) if x != y]
    words_a, words_b = ([v[0] for v in struct.iter_unpack('<H', t['raw'])] for t in (a, b))
    differing = [i for i, (x, y) in enumerate(zip(words_a, words_b)) if x != y]
    details = []
    for i in differing:
        coordinates, residual = [], i
        for dim in reversed(expected_shape): coordinates.insert(0, residual % dim); residual //= dim
        details.append(dict(element_index=i, coordinates=coordinates, reference_hex=f'{words_a[i]:04x}',
            candidate_hex=f'{words_b[i]:04x}', reference=bf16(words_a[i]), candidate=bf16(words_b[i]),
            absolute_difference=abs(bf16(words_a[i])-bf16(words_b[i]))))
    return dict(shape=expected_shape, dtype='BF16', finite=True, bytes_equal=not differences,
        reference=meta(a), candidate=meta(b), elements=len(words_a), differing_bytes=len(differences),
        differing_elements=len(differing), first_differing_byte_index=differences[0] if differences else None,
        max_absolute_difference=max((d['absolute_difference'] for d in details), default=0), differences=details)


def physical(raw, pid):
    need(raw.get('unit') == 'byte' and type(raw.get('bytes per unit')) is int and raw['bytes per unit'] == 1, 'physical units')
    need(not raw.get('errors') and not raw.get('warnings'), 'physical errors/warnings')
    ps = raw['processes']; need(len(ps) == 1 and type(ps[0]['pid']) is int and ps[0]['pid'] == pid, 'physical PID')
    p = ps[0]; values = [p['footprint'], p['auxiliary']['phys_footprint'], p['auxiliary']['phys_footprint_peak']]
    need(all(type(v) is int and 0 <= v <= 2 << 30 for v in values), 'physical 2GiB raw bound')
    top, current, peak = values
    need(peak >= current, 'auxiliary peak less than current')
    return dict(top_bytes=top, aux_current_bytes=current, aux_peak_bytes=peak,
                current_bytes=max(top, current), peak_bytes=max(top, peak))


def audit(owner, lease, plan_path):
    need(sha(plan_path) == PLAN_SHA and sha(owner / 'plan.json') == PLAN_SHA, 'preregistered plan SHA')
    plan = read(plan_path); pins = plan['execution']['pins']
    before, after = (read(owner / n) for n in ('pins-before.json', 'pins-after.json'))
    need(set(before) == set(after) == set(pins) and len(pins) == 570, 'source/runtime pin coverage')
    copied = 0; live = 0
    for path, expected in pins.items():
        a, b = before[path], after[path]
        need(a['sha256'] == b['sha256'] == expected and a['stat'] == b['stat'], 'before/after pin mismatch: ' + path)
        frozen = owner / 'frozen-source' / path.lstrip('/')
        proof = frozen if a['source_copied'] else Path(path)
        need(sha(proof) == expected, 'independent source/runtime SHA mismatch: ' + str(proof))
        need(proof.stat().st_size == a['stat']['size'], 'pin byte length mismatch')
        copied += bool(a['source_copied']); live += not a['source_copied']
    screen = owner / 'child/screen'; complete = read(screen / 'complete.json')
    need(complete['schema'] == 'E025-same-split-result-v2' and complete['status'] == 'FAIL' and complete['device_run'] is True, 'not expected early failure')
    need(complete['contract'] == plan['contract'] == read(screen / 'plan.json'), 'contract drift')
    need(complete['synthetic_only'] is True and complete['fixtures'] is None, 'unexpected model/fixture input')
    need(complete['source_pins_rechecked'] is False and complete['performance_qualified'] is False
         and complete['full_model_qualified'] is False, 'unreached qualification claimed')
    need(complete['normal_cases_passed'] == complete['fallback_cases_passed'] == 0, 'completed cases claimed')
    need(complete['error']['type'] == 'ValueError' and complete['error']['message'] == 'strict numerical gate rejected: ' + LABELS[2], 'primary failure changed')
    need(all(pins.get(p) == h for p, h in complete['sources'].items()), 'screen source identity')
    checks = complete['checks']; need(checks == read(screen / 'checks.json') and len(checks) == 3, 'check receipt mismatch')
    raw_checks = []
    for i, row in enumerate(checks):
        need(row['index'] == i and row['label'] == LABELS[i], 'execution check order')
        path = safe_path(screen, row['rawpair']['path']); need(sha(path) == row['rawpair']['sha256'], 'pair file SHA')
        result = compare_pair(path, [19, 6144] if i < 2 else [2, 19, 6144])
        need(row['left_shape'] == row['right_shape'] == result['shape'], 'reported pair shape')
        need(row['left_dtype'] == row['right_dtype'] == 'mlx.core.bfloat16' and row['metadata_equal'] is True, 'reported pair dtype')
        need(row['finite'] is row['left_finite'] is row['right_finite'] is True and row['bytes_equal'] == result['bytes_equal'], 'reported gate differs from raw')
        need(row['status'] == ('PASS' if result['bytes_equal'] else 'FAIL'), 'reported gate status')
        if not result['bytes_equal']: need(row['first_differing_byte_index'] == result['first_differing_byte_index'], 'first byte mismatch')
        raw_checks.append(dict(index=i, label=row['label'], **result))
    need([r['bytes_equal'] for r in raw_checks] == [True, True, False], 'early failure pattern changed')
    need(read(screen / 'first-difference.json') == checks[2], 'first-difference receipt mismatch')
    native0 = tensors(screen / checks[0]['rawpair']['path'])['reference']['raw']
    native1 = tensors(screen / checks[1]['rawpair']['path'])['reference']['raw']
    need(native0 == native1, 'native oracle changed between first two checks')
    inputs = complete['inputs']; need(inputs == read(screen / 'inputs.json') and len(inputs) == 1, 'input coverage')
    item = inputs[0]; input_tensors = []
    for stage in ('before', 'after'):
        p = safe_path(screen, item[stage]['path']); need(sha(p) == item[stage]['sha256'], 'input SHA')
        input_tensors.append(tensors(p))
    a, b = input_tensors; need(set(a) == set(b) == {'activation', 'weight', 'scales'}, 'input tensors')
    for k, shape, dtype in [('activation', [128,5120], 'BF16'), ('weight', [6144,640], 'U32'), ('scales', [6144,320], 'U8')]:
        need(a[k]['shape'] == b[k]['shape'] == shape and a[k]['dtype'] == b[k]['dtype'] == dtype, 'input layout')
        need(a[k]['raw'] == b[k]['raw'], 'input bytes changed')
    need(a['activation']['finite'] is True and item['x_finite'] is True and item['bytes_equal'] is True, 'input finite/read-only claim')
    phases = rows(screen / 'phases.jsonl')
    expected = [('initial','global'), ('before_check',LABELS[0]), ('after_check',LABELS[0]),
                ('before_check',LABELS[1]), ('after_check',LABELS[1]), ('before_check',LABELS[2])]
    need(len(phases) == len(complete['physical_receipts']) == 6, 'partial physical coverage')
    previous_time = previous_peak = 0
    for i, (row, pair) in enumerate(zip(phases, expected)):
        need((row['phase'],row['case']) == pair and row['index'] == i and row['pid'] == complete['pid'] and row['ppid'] == complete['ppid'], 'phase identity/order')
        need(row['status'] == 'PASS' and row['elapsed_s'] >= previous_time, 'phase status/time')
        previous_time = row['elapsed_s']
        receipt = row['physical_receipt']; need(receipt == complete['physical_receipts'][i], 'physical receipt order')
        p = safe_path(screen, receipt['path']); need(sha(p) == receipt['sha256'], 'self physical SHA')
        values = physical(read(p), complete['pid'])
        for key, value in values.items(): need(row['physical_'+key] == value, 'derived physical mismatch')
        need(values['aux_peak_bytes'] >= previous_peak, 'physical auxiliary historical peak regressed')
        previous_peak = values['aux_peak_bytes']
        command = read(screen / f'physical-{i:03d}.command.json')
        need(command['returncode'] == 0 and command['command'][2] == str(complete['pid']), 'self physical command')
        need(all(type(row[k]) is int and 0 <= row[k] for k in ('active_bytes','peak_bytes','cache_bytes')), 'MLX readings')
        need(row['cache_bytes'] == 0 and row['peak_bytes'] <= 1 << 30 and row['active_bytes']+row['cache_bytes'] <= 1 << 30, 'MLX observed budget')
    kernels = read(screen / 'kernel-source-receipts.json')
    need(len(kernels) == complete['kernel_count'] == 2 and [k['kind'] for k in kernels] == ['reference','candidate'], 'kernel coverage')
    source = next(p for p in pins if p.endswith('/prefill_nax_same_split/candidate.py'))
    frozen_candidate = owner / 'frozen-source' / source.lstrip('/')
    spec = importlib.util.spec_from_file_location('e025_audit_source_builder', frozen_candidate)
    helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
    include = next(p.split('/mlx/include/')[0]+'/mlx/include' for p in pins if '/mlx/include/mlx/' in p)
    frozen_include = owner / 'frozen-source' / include.lstrip('/')
    for k in kernels:
        geometry = helper.geometry(19,5120,6144)
        need(json.loads(json.dumps(geometry)) == k['geometry'], 'kernel geometry')
        built = helper.build_source(k['kind'], geometry, frozen_include)
        need(built['pins'] == k['pins'] and hashlib.sha256(built['header'].encode()).hexdigest() == k['header_sha256']
             and hashlib.sha256(built['source'].encode()).hexdigest() == k['body_sha256'], 'kernel source reconstruction')
    launch, release, outer = (read(owner / p) for p in ('child/launch.json','child/release.json','complete.json'))
    pid = complete['pid']; ppid = complete['ppid']
    need(launch['pid'] == release['pid'] == pid and launch['ppid'] == outer['owner_pid'] == ppid, 'owned process chain')
    need(release['returncode'] == 1 and release['waited_reaped'] is True and release['strict_absence'] is True,
         'child failure/reap receipt')
    need(release['absence_probe'] == dict(returncode=1, stdout='', stderr='') and not release['cleanup_notes'], 'strict absence probe')
    need(outer['status'] == 'FAIL' and len(outer['failures']) == 1 and 'numerical child failed' in outer['failures'][0]['traceback'], 'owner failure cause')
    need(0 < complete['elapsed_wall_s'] <= 120 and 0 < outer['elapsed_wall_s'] <= 480, 'deadlines')
    resources = read(owner / 'resources.json'); need([r['label'] for r in resources] == ['preflight','running','postflight'], 'resource partial order')
    derived_resources = []; best_pressure = 2; initial_swap = initial_swapouts = None
    for row in resources:
        for key in ('vm_stat','swap','pressure_level','thermal'): need(row[key]['returncode'] == 0, 'resource probe error')
        text = row['vm_stat']['text']; page_size = int(re.search(r'page size of (\d+) bytes', text)[1])
        free = int(re.search(r'^Pages free:\s+(\d+)\.', text, re.M)[1]) * page_size
        swaps = int(re.search(r'^Swapouts:\s+(\d+)\.', text, re.M)[1])
        swapused = float(re.search(r'used = ([0-9.]+)M', row['swap']['text'])[1])
        pressure = int(row['pressure_level']['text'])
        if initial_swap is None: initial_swap, initial_swapouts = swapused, swaps
        need(pressure in (1,2) and pressure <= best_pressure, 'pressure worsened'); best_pressure = min(best_pressure,pressure)
        need(free == row['real_free_bytes'] and free >= 50 << 30 and swaps == initial_swapouts
             and swapused <= initial_swap+16, 'free/swap resource gate')
        need(row['power_metadata']['raw']['returncode'] == 0 and "Now drawing from 'AC Power'" in row['power_metadata']['raw']['stdout'], 'power raw')
        need(row['processinfo_thermal_metadata']['raw']['returncode'] == 0 and int(row['processinfo_thermal_metadata']['raw']['stdout']) in (0,1), 'thermal raw')
        need(row['native_process_inventory_ok'] is True and row['process_inventory_ok'] is True, 'process inventory missing')
        for process in row['native_inference_processes']:
            need((process['pid'] == pid and process['owned'] is True and process['ppid'] == ppid)
                 or (process['pid'] == 74384 and process['authorized_suspended'] is True and 'T' in process['state']), 'active foreign process')
        derived_resources.append(dict(label=row['label'], free_bytes=free, pressure=pressure, swap_used_mib=swapused, swapouts=swaps))
    physical_owner = read(owner / 'footprint-000.json'); raw_owner = owner / physical_owner['raw_file']
    need(sha(raw_owner) == physical_owner['raw_sha256'] and physical_owner['returncode'] == 0 and physical_owner['pid'] == pid, 'owner footprint raw')
    parsed = physical(read(raw_owner), pid)
    need(physical_owner['parsed'] == dict(pid=pid, **parsed), 'owner physical derived values')
    l, resume, job = (read(lease / p) for p in ('lease.json','resume.json','job-0.json'))
    need(l['service_pid'] == 74384 and l['closed'] is True and l['child_pid'] == ppid, 'lease identity/closure')
    need(job['pid'] == ppid and job['returncode'] == 1 and job['waited'] is True and job['command'] == l['jobs'][0], 'lease owned job failure/wait')
    need(resume['identity_match'] is True and resume['state_before'] == 'T' and resume['state_after'] == 'S'
         and l['paused_at'] <= resume['time'] <= l['deadline'], 'service restoration identity/order')
    need('--plan-sha256' in job['command'] and job['command'][job['command'].index('--plan-sha256')+1] == PLAN_SHA, 'lease plan binding')
    return dict(schema='E025-failure-audit-v2', status='PASS', experiment_status='REJECTED_NUMERICAL_PARTIAL',
        plan_sha256=PLAN_SHA, checks=raw_checks, native_QMM_reference_gate_passed_first=True,
        inputs={k:meta(v) for k,v in a.items()}, input_full_bytes_unchanged=True,
        sources=dict(pins=570, before_after_receipts_match=True, frozen_copies_rehashed=copied,
                     current_uncopied_files_rehashed=live, kernel_sources_reconstructed=2,
                     child_final_source_recheck_reached=False),
        observed_resources=dict(phases=6, owner_samples=len(resources), owner_physical_samples=1,
            mlx_peak_bytes=max(r['peak_bytes'] for r in phases), mlx_active_bytes=max(r['active_bytes'] for r in phases),
            physical_peak_bytes=max(r['physical_peak_bytes'] for r in phases), raw_system_samples=derived_resources,
            scope='Only retained observations; no after-failing-check/final-release or full-resource qualification'),
        lifecycle=dict(owner_pid=ppid, child_pid=pid, child_exit=1, strict_reaped_absent=True,
                       service_pid=74384, same_identity_restored_T_to_S=True),
        not_run=['NAX final reduction comparison','ON callable','all fallback checks','mlp_down','out','final_release','child final source recheck'],
        conclusion='NAX partial materialized and differs in exact BF16 bits; native SIMD reduced reference first matched native QMM. Unrun final/model outputs are not judged.',
        performance_qualified=False, full_model_qualified=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--owner', required=True, type=Path)
    ap.add_argument('--lease', required=True, type=Path)
    ap.add_argument('--plan', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    args = ap.parse_args(); args.out.mkdir(parents=True, exist_ok=False)
    try:
        result = audit(args.owner.resolve(), args.lease.resolve(), args.plan.resolve())
    except BaseException as error:
        result = dict(schema='E025-failure-audit-v2', status='FAIL', error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
    result['auditor'] = dict(path=str(Path(__file__).resolve()), sha256=sha(__file__))
    artifacts = []
    for prefix, root in [('owner',args.owner), ('lease',args.lease), ('inputs',args.plan.parent)]:
        for p in sorted(root.rglob('*')):
            if p.is_file(): artifacts.append(dict(path=prefix+'/'+str(p.relative_to(root)), bytes=p.stat().st_size, sha256=sha(p)))
    write(args.out / 'artifact-manifest.json', dict(schema='E025-failure-artifacts-v1', files=artifacts,
        roots=dict(owner=str(args.owner.resolve()), lease=str(args.lease.resolve()), inputs=str(args.plan.parent.resolve())),
        total_bytes=sum(r['bytes'] for r in artifacts)))
    result['artifact_manifest_sha256'] = sha(args.out / 'artifact-manifest.json')
    write(args.out / 'audit.json', result)
    print(json.dumps(dict(status=result['status'], out=str(args.out))))
    return int(result['status'] != 'PASS')


if __name__ == '__main__':
    raise SystemExit(main())
