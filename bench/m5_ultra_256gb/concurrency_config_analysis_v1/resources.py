"""Offline E045 resource/restoration review; saved evidence only, default inert.

This does not perform live process, HTTP, GPU, or memory probes. Physical/raw
command bytes are reparsed. Full process inventories and repeated start-identity
assertions were not retained as raw ps text; that narrower scope stays explicit.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re

GIB = 1 << 30
R = Path('/Users/ryanli/code/minivllm')
B = R / 'bench/m5_ultra_256gb'
LABELS = ('c1-start50', 'c1-mb200', 'c1-mb16', 'c1-mb800', 'c1-end50',
          'c8-start50', 'c8-mb800', 'c8-mb16', 'c8-mb200', 'c8-end50')


def need(ok, message):
    if not ok:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def norm(text):
    return ' '.join(text.split())


def match(pattern, text):
    found = re.search(pattern, text, re.M)
    need(found is not None, 'Unparseable retained text: ' + pattern)
    return found[1]


def command(row, expected):
    need(row['command'] == expected and row['returncode'] == 0
         and isinstance(row['stdout'], str) and isinstance(row['stderr'], str)
         and not row['stderr'].strip(), 'Command unknown/failed/different: ' + str(expected))
    return row['stdout'].strip()


def physical(folder, read, binary):
    row = read(folder / 'sample.json'); pid = row['pid']
    need(row['schema'] == 'GLM-physical-sample-v1' and row['status'] == 'MEASURED'
         and row.get('error') is None and type(pid) is int and pid > 1, 'Physical sample failed/unknown')
    begin, end = row['started_monotonic'], row['ended_monotonic']
    need(all(type(x) in (int, float) and math.isfinite(x) for x in (begin, end))
         and 0 <= end - begin <= 5 and row['started_at'] <= row['ended_at'], 'Physical duration')
    raw_path = folder / 'footprint.raw.json'; raw = binary(raw_path)
    need(row['raw'] == dict(path=str(raw_path), bytes=len(raw), sha256=digest(raw)), 'Footprint raw SHA/size/path')
    expected = [['/usr/bin/footprint', '-p', str(pid), '--noCategories', '--sysFootprint', '-f', 'bytes', '-j', str(raw_path)],
        ['/usr/sbin/sysctl', '-n', 'kern.memorystatus_vm_pressure_level'],
        ['/usr/sbin/sysctl', 'vm.swapusage'], ['/usr/sbin/sysctl', '-n', 'hw.memsize'], ['/usr/bin/vm_stat']]
    need(len(row['commands']) == 5, 'Five complete physical commands')
    texts = [command(c, argv) for c, argv in zip(row['commands'], expected)]
    report = json.loads(raw)
    need(report['unit'] == 'byte' and report['bytes per unit'] == 1
         and report['errors'] == report['warnings'] == [] and len(report['processes']) == 1,
         'Footprint report units/errors/process count')
    proc = report['processes'][0]; aux = proc['auxiliary']
    top, cur, peak, system = proc['footprint'], aux['phys_footprint'], aux['phys_footprint_peak'], report['auxiliary']['sys_footprint']
    need(proc['pid'] == pid and all(type(x) is int and x >= 0 for x in (top, cur, peak, system))
         and peak >= cur and system > 0, 'Raw physical fields')
    need(texts[1].isdigit() and texts[3].isdigit(), 'Pressure/RAM text')
    page_bytes = int(match(r'page size of (\d+) bytes', texts[4])); need(page_bytes == 16384, 'Page size')
    value = dict(pid=pid, top_bytes=top, aux_current_bytes=cur, aux_peak_bytes=peak,
        current_bytes=max(top, cur), peak_bytes=max(top, peak), system_bytes=system,
        pressure=int(texts[1]), physical_ram=int(texts[3]),
        swap_bytes=round(float(match(r'used = (\d+(?:\.\d+)?)M', texts[2])) * (1 << 20)),
        swapouts=int(match(r'^Swapouts:\s*(\d+)\.', texts[4])))
    need(value == row['parsed'], 'Raw physical reparse disagrees with saved parsed data')
    need(value['physical_ram'] == 256 * GIB and value['pressure'] == 1 and system <= 240 * GIB,
         'Physical observed system budget/pressure/RAM')
    return row, value


def metadata(sample, owner, service, baseline, identity):
    for name in ('memory_pressure', 'pressure_level', 'swap', 'vm_stat', 'thermal'):
        value = sample[name]
        need(value['returncode'] == 0 and not value['error'].strip(), 'Resource command failed: ' + name)
    pressure = int(sample['pressure_level']['text'])
    free = int(match(r'free percentage:\s*(\d+)', sample['memory_pressure']['text']))
    swap = float(match(r'used = (\d+(?:\.\d+)?)M', sample['swap']['text']))
    counters = {name: int(match(r'^' + name + r':\s*(\d+)', sample['vm_stat']['text']))
        for name in ('Swapouts', 'Swapins', 'Pageouts', 'Compressions', 'Decompressions')}
    need(pressure == sample['pressure_level_value'] == 1 and free == sample['free_percent']
         and 0 <= free <= 100 and swap == sample['swap_used_mb'] and counters == sample['vm_counters'],
         'Base resource reparse/pressure/free/counters')
    need(round(swap * (1 << 20)) <= baseline['swap_bytes'] and counters['Swapouts'] == baseline['swapouts'],
         'Metadata swap/Swapouts grew')
    power = sample['power_metadata']; thermal = sample['processinfo_thermal_metadata']
    text = command(power['raw'], ['/usr/bin/pmset', '-g', 'batt'])
    need(match(r"Now drawing from '([^']+)'", text) == power['source'] == 'AC Power'
         and power['status'] == 'known', 'AC source')
    code = int(command(thermal['raw'], ['/usr/bin/osascript', '-l', 'JavaScript', '-e',
        'ObjC.import("Foundation"); String($.NSProcessInfo.processInfo.thermalState)']))
    need(code == thermal['code'] and code in (0, 1) and thermal['status'] == 'known'
         and thermal['name'] == {0: 'nominal', 1: 'fair'}[code], 'Thermal state')
    pid = sample['owned_model_pid']; owned = sample['exact_owned_pid_metadata']
    if pid is None:
        need(owned['status'] == 'unknown' and owned['pid'] is None and owned['raw'] is None,
             'No-server metadata must not invent an owned process')
    else:
        fields = command(owned['raw'], ['/bin/ps', '-p', str(pid), '-o', 'pid=,ppid=,rss=,comm=']).split(None, 3)
        need(len(fields) == 4 and all(x.isdigit() for x in fields[:3]), 'Exact PID raw format')
        need(owned['status'] == 'known' and int(fields[0]) == owned['pid'] == pid
             and int(fields[1]) == owned['ppid'] == owner and int(fields[2]) == owned['rss_kib']
             and fields[3] == owned['executable'] == identity['executable'], 'Exact PID/PPID/executable binding')
    need(sample['process_inventory_ok'] is True and sample['native_process_inventory_ok'] is True
         and sample['authorized_suspended_pids'] == [service['pid']], 'Inventory availability/authorized service')
    native = sample['native_inference_processes']; known = sample['known_inference_processes']
    need(len({p['pid'] for p in native}) == len(native)
         and sample['known_inference_process_count'] == len(known), 'Inventory identity/count')
    for proc in native:
        need(proc['owned'] is (proc['pid'] == pid) and proc['authorized_suspended'] is
             (proc['pid'] == service['pid'] and 'T' in proc['state']), 'Inventory flags')
    suspended = [p for p in native if p['pid'] == service['pid']]
    need(len(suspended) == 1 and suspended[0]['authorized_suspended'] is True
         and suspended[0]['ppid'] == service['process']['ppid']
         and suspended[0]['executable'] == service['process']['executable'], 'Original service paused observation')
    need(all(p['owned'] or p['authorized_suspended'] for p in native)
         and sample['no_active_foreign_native_processes'] is True
         and all(p['owned'] is (p['pid'] == pid) for p in known)
         and all(p['owned'] or p['pid'] == service['pid'] for p in known), 'Recomputed process guard')
    need(sample['only_owned_known_inference_processes'] is all(p['owned'] for p in known), 'Unmodified original inventory flag')
    if pid is not None:
        ours = [p for p in native if p['pid'] == pid]
        need(len(ours) == 1 and ours[0]['ppid'] == owner and ours[0]['executable'] == identity['executable'], 'Native owned process binding')
    return dict(thermal=code, free_percent=free, swap_bytes=round(swap * (1 << 20)), swapouts=counters['Swapouts'])


def health(row):
    target = 'Qwen3.8-Flash-Next-oQ6e-mtp'; status = row['/api/status']
    need(row['/health']['status'] == 'healthy' and status['version'] == '0.7.0'
         and status['active_requests'] == status['waiting_requests'] == status['models_loading'] == 0
         and status['default_model'] == target and status['loaded_models'] == [target], 'Saved health/status contract')
    models = row['/v1/models/status']['models']; loaded = [m for m in models if m['loaded']]
    need(len(loaded) == 1 and not any(m['is_loading'] for m in models), 'Saved model-status contract')
    model = loaded[0]
    need(model['id'] == target and model['model_path'] == str(R.parent / 'models/mlx' / target)
         and model['engine_type'] == 'vlm' and model['config_model_type'] == 'qwen4_exp'
         and model['pinned'] is False, 'Saved restored loaded model')


def audit(execution):
    refs = {}; result = dict(schema='E045-saved-resource-review-v1', status='INCOMPLETE', errors=[],
        live_checks_performed=False, performance_qualified=False, hard_allocation_cap=False,
        limitations=['Observed physical ceilings do not bound inter-sample allocation peaks or establish physical fit.',
            'Native/known inventories are saved parsed observations; full original ps inventory text was not retained.',
            'Initial child start identity and raw PID/PPID/executable are checked; repeated start/service identity assertions are source-bound, not independently replayable raw ps.',
            'Final/lease/health are original recorded facts, not fresh live restoration checks; GUI/unrecognized GPU use remains unknown.'])
    def binary(path):
        path = Path(path); need(not path.is_symlink(), 'Unexpected symlink: ' + str(path))
        raw = path.read_bytes(); refs[str(path)] = dict(bytes=len(raw), sha256=digest(raw)); return raw
    def read(path):
        return json.loads(binary(path))
    try:
        final = read(execution / 'final.json'); result['original_final_status'] = final['status']
        plan = read(execution / 'execution-plan.json'); screen = Path(plan['outputs']['screen']); lease_dir = Path(plan['outputs']['lease'])
        need(str(execution) == plan['outputs']['execution'] and read(screen / 'plan.json') == plan, 'Output/plan binding')
        registered = read(execution / 'plan.json'); invocation = read(execution / 'invocation.json')
        need({k: v for k, v in plan.items() if k != 'physical_baseline'} == registered
             and refs[str(execution / 'plan.json')]['sha256'] == invocation['plan_sha256'], 'Original preregistration binding')
        need(plan['physical'] == dict(child_limit_gib=64, system_limit_gib=240, swap_growth_bytes=0,
             swapouts_growth=0, probe_seconds=5, interval_seconds=3, hard_allocation_cap=False)
             and plan['max_observation_gap_seconds'] == 12, 'Fixed observed resource policy')
        for path in (B/'resources_v2.py', B/'resources.py', B/'resource_base.py', B/'glm_flash/physical_guard.py',
                     B/'suspended_service.py', B/'concurrency_config_v2/runner.py', B/'concurrency_config_v2/execute.py',
                     B/'nvfp4_m8_register/real-prereg-v2/execute.py'):
            need(digest(binary(path)) == plan['pins'][str(path)], 'Source binding: ' + str(path))
        whole = read(screen / 'result.json'); result['original_screen_status'] = whole['status']
        lease = read(lease_dir / 'lease.json'); job = read(lease_dir / 'job-0.json'); resume = read(lease_dir / 'resume.json')
        runner = read(execution / 'runner-identity.json'); service = plan['service']; owner = runner['pid']
        result['original_lease_closed'] = lease['closed']; result['original_lease_returncode'] = final['lease_returncode']
        need(service['pid'] == 52818 and norm(lease['service_identity']) == service['identity'] == final['service_identity_after']
             and lease['service_pid'] == service['pid'], 'Original service identity')
        jobs = read(execution / 'jobs.json')
        need(len(jobs) == 1 and lease['jobs'] == jobs and job['command'] == jobs[0]
             and job['pid'] == lease['child_pid'] == owner and runner['ppid'] == lease['owner_pid']
             and job['waited'] is True and job['returncode'] == 0, 'Runner command/parent/wait')
        need(jobs[0][jobs[0].index('--plan-sha256') + 1] == refs[str(execution / 'execution-plan.json')]['sha256'], 'Runner execution-plan SHA')
        need(lease['closed'] is True and lease['current_job'] == 0 and resume['identity_match'] is True
             and resume['state_after'] and 'T' not in resume['state_after']
             and lease['paused_at'] < resume['time'], 'Original lease/resume receipt')
        health(read(execution / 'before-health.json')); health(read(execution / 'after-health.json'))
        need(final['status'] == 'PASS_CAPTURE_AND_RESTORE' and final['errors'] == [] and final['lease_returncode'] == 0
             and whole['status'] == 'PASS_CAPTURE_ONLY' and whole['errors'] == []
             and [r['label'] for r in whole['trials']] == list(LABELS), 'Original complete cycle status')
        base_row, baseline = physical(execution / 'physical-before', read, binary)
        need(base_row['pid'] == base_row['owner_pid'] and baseline == plan['physical_baseline'], 'Original physical baseline')
        result['baseline'] = baseline; children = {}; checkpoints = []
        for label in LABELS:
            folder = screen / 'trials' / label; child = read(folder / 'child.json'); release = read(folder / 'release.json')
            complete = read(folder / 'complete.json'); pid = child['pid']; identity = read(screen / 'resources' / f'identity-{pid}.json')
            fields = command(read(folder / 'child-identity.json'), ['/bin/ps', '-p', str(pid), '-o', 'pid=,ppid=,lstart=,comm=']).split(None, 7)
            need(len(fields) == 8 and int(fields[0]) == pid and int(fields[1]) == owner
                 and identity == dict(pid=pid, ppid=owner, start=' '.join(fields[2:7]), executable=fields[7]), 'Server start identity/PPID')
            need(pid not in children and release['pid'] == pid and release['returncode'] in (0, -15)
                 and all(release[k] is True for k in ('waited_reaped', 'strict_absence', 'port_absent'))
                 and complete['status'] == 'PASS' and complete['errors'] == []
                 and complete['started_at'] <= child['time'] < release['time'] <= complete['ended_at'], 'Server release binding')
            events = [json.loads(x) for x in binary(folder / 'events.jsonl').splitlines()]
            if release['returncode'] == -15:
                need(any(e['label'] == 'server-term' and e['pid'] == pid for e in events), 'Owned TERM event')
            absent = read(folder / 'child-absence.json')
            need(absent['command'] == ['/bin/ps', '-p', str(pid), '-o', 'pid='] and absent['returncode'] == 1
                 and absent['stdout'].strip() == absent['stderr'].strip() == '', 'Saved exact server absence')
            children[pid] = dict(identity=identity, child=child, release=release, label=label)
            checkpoints.extend(e for e in events if e['label'] == 'checkpoint')
        folder = screen / 'resources'; paths = sorted(folder.glob('physical-*/sample.json')); metas = sorted(folder.glob('resource-*.json'))
        need(len(paths) == len(metas) == whole['physical_samples'] == whole['resource_samples'] == len(checkpoints) > 0, 'Complete physical/metadata/checkpoint matrix')
        values = []; metadata_rows = []; gaps = []; prior = base_row
        for i, (path, metapath, event) in enumerate(zip(paths, metas, checkpoints)):
            need(path == folder / f'physical-{i:04d}' / 'sample.json'
                 and metapath == folder / f'resource-{i:04d}.json', 'Contiguous resource sequence')
            row, value = physical(path.parent, read, binary); sample = read(metapath); pid = sample['owned_model_pid']
            gap = row['started_monotonic'] - prior['ended_monotonic']; gaps.append(gap)
            need(0 <= gap <= 12 and row['owner_pid'] == owner and row['pid'] == (owner if pid is None else pid), 'Cadence/physical owner binding')
            need(value['swap_bytes'] <= baseline['swap_bytes'] and value['swapouts'] == baseline['swapouts'], 'Physical swap/Swapouts grew')
            need(row['ended_monotonic'] <= sample['monotonic_s'] <= event['monotonic']
                 and event['pid'] == pid and event['phase'] == sample['label']
                 and lease['paused_at'] <= row['started_at'] <= event['time'] <= resume['time'], 'Probe/checkpoint/pause interval')
            identity = None
            if pid is not None:
                binding = children[pid]; identity = binding['identity']
                need(binding['child']['time'] <= row['started_at'] <= row['ended_at'] <= binding['release']['time']
                     and max(value[k] for k in ('top_bytes', 'aux_current_bytes', 'aux_peak_bytes')) <= 64 * GIB,
                     'Live server observation/release/64-GiB ceiling')
            metadata_rows.append(metadata(sample, owner, service, baseline, identity)); values.append(value); prior = row
        result.update(status='PASS_SAVED_OBSERVATIONS', physical_samples=len(values), metadata_samples=len(metadata_rows),
            servers=len(children), max_observation_gap_seconds=max(gaps), baseline_to_first_gap_seconds=gaps[0],
            max_system_bytes=max(v['system_bytes'] for v in values),
            max_child_current_bytes=max(v['current_bytes'] for v in values if v['pid'] in children),
            max_child_historical_peak_bytes=max(v['aux_peak_bytes'] for v in values if v['pid'] in children),
            observed_pressure_values=sorted({v['pressure'] for v in values}),
            observed_thermal_codes=sorted({v['thermal'] for v in metadata_rows}),
            max_swap_growth_bytes=max(v['swap_bytes'] - baseline['swap_bytes'] for v in values),
            swapouts_values=sorted({v['swapouts'] for v in values}),
            all_saved_health_contracts_pass=True, recorded_cycle_restored=True,
            sampled_native_inventory_guard_pass=True, raw_full_process_inventory_reparsed=False)
    except Exception as exc:
        result['status'] = 'FAIL_OR_UNKNOWN'; result['errors'].append(dict(type=type(exc).__name__, message=str(exc)))
    for path, ref in refs.items():
        try:
            need(digest(Path(path).read_bytes()) == ref['sha256'], 'Input changed: ' + path)
        except Exception as exc:
            result['status'] = 'FAIL_OR_UNKNOWN'
            result['errors'].append(dict(type=type(exc).__name__, message=str(exc)))
    result['inputs'] = refs
    result['review_source_sha256'] = digest(Path(__file__).read_bytes())
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--execution', type=Path); parser.add_argument('--out', type=Path)
    args = parser.parse_args(argv)
    if args.execution is None and args.out is None:
        print('Inert: saved evidence only. Supply --execution closed-execution-directory --out fresh-directory.'); return 0
    need(args.execution is not None and args.out is not None, 'Supply both --execution and --out')
    args.out.mkdir(parents=True, exist_ok=False); result = audit(args.execution.resolve())
    (args.out / 'resource-audit.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: result[k] for k in ('status', 'errors')}))
    return 0 if result['status'] == 'PASS_SAVED_OBSERVATIONS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
