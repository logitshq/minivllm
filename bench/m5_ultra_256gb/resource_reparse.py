"""Offline cold single-child resource reparse; stdlib, no runtime/OS probes.

Read only a completed owner's existing receipts. PASS is limited to recorded
samples and the original resource guards, never sole-GPU, restoration, numeric
or performance qualification. Native/legacy inventory lists have no saved raw
ps table; those lists remain producer receipts. Exact owned-PID ps is reparsed.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re

COUNTERS = ('Swapouts', 'Swapins', 'Pageouts', 'Compressions', 'Decompressions')
THERMAL_NAMES = {0: 'nominal', 1: 'fair', 2: 'serious', 3: 'critical'}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def unique(pairs):
    result = {}
    for key, value in pairs:
        need(key not in result, 'duplicate JSON key: ' + key)
        result[key] = value
    return result


def same(a, b):
    return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def match(pattern, text):
    rows = re.findall(pattern, text, re.M)
    need(len(rows) == 1, 'missing/duplicate raw field: ' + pattern)
    return rows[0]


def raw(row, name):
    item = row[name]
    need(type(item['returncode']) is int and item['returncode'] == 0
         and item['error'] == '' and type(item['text']) is str, 'failed raw command: ' + name)
    return item['text']


def metadata_raw(value, command):
    item = value['raw']
    need(item['command'] == command and type(item['returncode']) is int
         and item['returncode'] == 0 and item['stderr'] == '', 'failed/wrong metadata command')
    return item['stdout']


def parse_sample(row, child_pid, owner_pid):
    text = raw(row, 'pressure_level').strip()
    need(re.fullmatch(r'\d+', text) is not None, 'pressure value')
    pressure = int(text)
    swap = float(match(r'\bused = ([0-9]+(?:\.[0-9]+)?)M\b', raw(row, 'swap')))
    vm_text = raw(row, 'vm_stat')
    vm = {key: int(match(r'^' + key + r':\s*(\d+)\.\s*$', vm_text)) for key in COUNTERS}
    page_size = int(match(r'page size of (\d+) bytes', vm_text))
    free_bytes = page_size * int(match(r'^Pages free:\s*(\d+)\.\s*$', vm_text))
    free_percent = int(match(r'free percentage:\s*(\d+)%', raw(row, 'memory_pressure')))
    raw(row, 'thermal')  # Preserve/check raw pmset command; no invented thermal interpretation.
    need(same(pressure, row['pressure_level_value']) and same(swap, row['swap_used_mb'])
         and same(vm, row['vm_counters']) and same(free_percent, row['free_percent']), 'resource parsed metadata differs')
    need(page_size > 0 and 0 <= free_percent <= 100, 'memory units/percentage')
    power = row['power_metadata']
    power_text = metadata_raw(power, ['/usr/bin/pmset', '-g', 'batt'])
    source = match(r"Now drawing from '([^']+)'", power_text)
    need(power['status'] == 'known' and source == power['source'], 'power parsed metadata')
    thermal = row['processinfo_thermal_metadata']
    thermal_text = metadata_raw(thermal, ['/usr/bin/osascript', '-l', 'JavaScript', '-e',
        'ObjC.import("Foundation"); String($.NSProcessInfo.processInfo.thermalState)']).strip()
    need(re.fullmatch(r'\d+', thermal_text) is not None, 'thermal raw integer')
    thermal_code = int(thermal_text)
    need(thermal['status'] == 'known' and same(thermal_code, thermal['code'])
         and thermal['name'] == THERMAL_NAMES.get(thermal_code), 'thermal parsed metadata')
    pid = row['owned_model_pid']
    need(pid is None or type(pid) is int and pid == child_pid, 'sample child PID')
    exact = row['exact_owned_pid_metadata']; exact_rss = None
    if pid is None:
        need(exact['status'] == 'unknown' and exact['pid'] is None and exact['raw'] is None, 'no owned PID probe')
    else:
        item = exact['raw']
        need(exact['pid'] == pid and item['command'] == ['/bin/ps', '-p', str(pid), '-o', 'pid=,ppid=,rss=,comm=']
             and type(item['returncode']) is int and item['stderr'] == '', 'exact PID command')
        fields = item['stdout'].split(None, 3)
        known = item['returncode'] == 0 and len(fields) == 4 and all(x.isdigit() for x in fields[:3]) and int(fields[0]) == pid
        if known:
            observed = dict(pid=int(fields[0]), ppid=int(fields[1]), rss_kib=int(fields[2]), executable=fields[3])
            need(exact['status'] == 'known' and all(same(exact[k], v) for k, v in observed.items())
                 and observed['ppid'] == owner_pid, 'exact PID raw/parent mismatch')
            exact_rss = observed['rss_kib']
        else:
            need(exact['status'] == 'unknown' and all(exact.get(k) is None for k in ('ppid', 'rss_kib', 'executable')),
                 'unavailable exact PID must stay unknown')
    need(row['native_process_inventory_ok'] is True and row['process_inventory_ok'] is True
         and row['authorized_suspended_pids'] == [], 'inventory unavailable/suspended exemption')
    native, legacy = row['native_inference_processes'], row['known_inference_processes']
    for process in native + legacy:
        need(type(process['pid']) is int and type(process['ppid']) is int
             and process['owned'] is (process['pid'] == pid), 'inventory ownership flag')
    need(row['known_inference_process_count'] == len(legacy)
         and row['only_owned_known_inference_processes'] is all(p['owned'] for p in legacy)
         and row['exactly_one_owned_known_model'] is (len(legacy) == 1 and legacy[0]['owned'])
         and row['no_active_foreign_native_processes'] is all(p['owned'] or p['authorized_suspended'] for p in native),
         'inventory receipt flags')
    inventory_clean = all(p['owned'] and p['pid'] == child_pid and p['ppid'] == owner_pid for p in native + legacy)
    inventory_clean = inventory_clean and all(p['authorized_suspended'] is False for p in native)
    return dict(label=row['label'], monotonic_s=row['monotonic_s'], pressure=pressure, swap_mib=swap, vm=vm,
        page_size_bytes=page_size, free_bytes=free_bytes, free_percent=free_percent, power_source=source,
        thermal_code=thermal_code, exact_pid_probe_status=exact['status'], exact_pid_rss_kib=exact_rss,
        known_inventory_clean=inventory_clean, native_inventory_count=len(native), legacy_inventory_count=len(legacy))


def audit(owner):
    owner = Path(owner).resolve(); pins = {}
    def read(name):
        path = owner / name; data = path.read_bytes()
        pins[name] = dict(sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))
        return json.loads(data, object_pairs_hook=unique)
    complete = read('complete.json'); release = read('release.json'); index = read('children.json')
    need(complete['status'] in ('PASS', 'FAIL') and complete['all_children_reaped'] is True
         and release['all_children_reaped'] is True, 'closed owner with reaped child required')
    need(index['schema'] == 'cold-service-owner-children-v1' and len(index['children']) == 1
         and same(index['children'], release['children']) and pins['children.json']['sha256'] == release['children_sha256'],
         'one closed child/index binding')
    child = index['children'][0]; pid, owner_pid = child['pid'], complete['owner_pid']
    need(type(pid) is int and type(owner_pid) is int and pid > 1 and owner_pid > 1
         and release['owner_pid'] == owner_pid and complete['planned_trials'] == [child['phase']], 'actual child/owner phase')
    item = child['release']; name = item['path']
    need(name == 'phases/' + child['phase'] + '/release.json', 'phase release path')
    phase_release = read(name)
    need(pins[name]['sha256'] == item['sha256'] and phase_release['pid'] == pid
         and phase_release['waited_reaped'] is True and phase_release['strict_absence'] is True, 'closed child release receipt')
    paths = sorted(owner.glob('resource-*.json'))
    need(type(complete['samples']) is int and complete['samples'] >= 3
         and [p.name for p in paths] == [f'resource-{i:03d}.json' for i in range(complete['samples'])], 'complete sample sequence')
    original = [read(path.name) for path in paths]
    rows = [parse_sample(row, pid, owner_pid) for row in original]
    need([r['label'] for r in rows] == ['before-prepare', 'after-prepare-before-spawn']
         + [child['phase'] + ':running'] * (len(rows)-3) + ['postflight'], 'fixed single-child sample phases')
    need(all(type(r['monotonic_s']) in (int, float) and math.isfinite(r['monotonic_s']) for r in rows)
         and all(a['monotonic_s'] < b['monotonic_s'] for a, b in zip(rows, rows[1:])), 'strict sample clock order')
    saved = read('resources.json'); need(len(saved) == len(original), 'summary sample count')
    violations = []
    for i, (row, parsed, combined) in enumerate(zip(original, rows, saved)):
        expected = dict(row)
        if i:
            expected.update(swap_used_mb_delta=parsed['swap_mib']-rows[i-1]['swap_mib'],
                vm_counter_delta={k: parsed['vm'][k]-rows[i-1]['vm'][k] for k in COUNTERS})
        need(same(combined, expected), 'summary copy/delta differs at sample ' + str(i))
        bad = []
        if parsed['pressure'] != 1: bad.append('pressure_not_1')
        if parsed['power_source'] != 'AC Power': bad.append('not_ac_power')
        if parsed['thermal_code'] not in (0, 1): bad.append('thermal_above_fair')
        if not parsed['known_inventory_clean']: bad.append('foreign_known_process_receipt')
        if i >= 2:
            if all(rows[n]['swap_mib']-rows[n-1]['swap_mib'] > 64 for n in (i-1, i)):
                bad.append('original_guard_sustained_swap_growth')
            if all(rows[n]['vm']['Swapouts']-rows[n-1]['vm']['Swapouts'] > 0 for n in (i-1, i)):
                bad.append('original_guard_sustained_swapouts')
        violations += [dict(sample=i, reason=reason) for reason in bad]
    return dict(schema='cold-single-child-resource-reparse-v1', status='REJECT' if violations else 'PASS',
        owner_directory=str(owner), owner_status=complete['status'], phase=child['phase'], child_pid=pid, owner_pid=owner_pid,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), raw_file_pins=pins,
        samples=len(rows), observed=rows, guard_violations=violations,
        pressure_values=sorted({r['pressure'] for r in rows}), thermal_codes=sorted({r['thermal_code'] for r in rows}),
        swap_delta_mib=rows[-1]['swap_mib']-rows[0]['swap_mib'],
        max_swap_growth_mib=max(r['swap_mib'] for r in rows)-rows[0]['swap_mib'],
        vm_delta={k: rows[-1]['vm'][k]-rows[0]['vm'][k] for k in COUNTERS},
        minimum_free_bytes=min(r['free_bytes'] for r in rows),
        scope='Raw pressure/swap/VM/power/NSProcessInfo thermal and exact owned-PID ps reparsed. Native/legacy process inventories checked only as producer receipts; their complete raw ps tables were not saved. Free bytes are Pages free times page size, not available-memory or physical footprint. No exclusion of unrecognized runtimes or GUI GPU activity.',
        resource_scope='Recorded samples and original pressure1/AC/thermal/known-inventory/sustained-swap guards only; not service restoration or unsampled intervals.',
        sole_gpu_qualified=False, restoration_qualified=False, numerical_qualified=False, performance_qualified=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--owner', type=Path, required=True); parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    need(not args.out.exists(), 'new output required')
    try:
        result = audit(args.owner)
    except Exception as error:
        result = dict(schema='cold-single-child-resource-reparse-v1', status='FAIL',
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), owner_directory=str(args.owner.resolve()),
            error=dict(type=type(error).__name__, message=str(error)), performance_qualified=False)
    with args.out.open('x') as stream:
        json.dump(result, stream, indent=2); stream.write('\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('raw_file_pins', 'observed')}))
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
