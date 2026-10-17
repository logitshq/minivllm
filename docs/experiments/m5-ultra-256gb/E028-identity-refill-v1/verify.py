"""Offline E028 v1 failure verification. Stdlib only; never imports run code.

Optional --runtime-pins also hashes the original installed runtime paths.
The default verifies archived source copies and original pin receipts.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
PLAN_SHA = '652486a29e021bd8009bc27a1bdede607c63aa56d9fb84bf70940860a8e437ee'


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def verify(runtime_pins=False):
    raw = ROOT / 'raw'
    owner = raw / 'identity-refill-v1-owner'
    cpu = raw / 'identity-refill-v1-cpu-prepare'
    inp = raw / 'identity-refill-v1-inputs'
    lease_dir = raw / 'identity-refill-v1-service-lease'
    screen = owner / 'child/screen'
    plan = read(inp / 'plan.json')
    need(sha(inp / 'plan.json') == PLAN_SHA, 'original plan SHA')
    need(read(owner / 'plan.json') == plan == read(cpu / 'plan.json'), 'plan copies')
    registration = read(inp / 'registration.json')
    need(registration['plan_sha256'] == PLAN_SHA and registration['jobs_sha256'] == sha(inp / 'jobs.json'), 'registration')
    pins = plan['execution']['pins']
    need(len(pins) == registration['pins'] == 570, 'pin count')
    frozen_count = 0
    for folder, record_names in ((owner, ['pins-before.json', 'pins-after.json']), (cpu, ['pins-before.json'])):
        records = [read(folder / name) for name in record_names]
        for record in records:
            need(set(record) == set(pins), 'pin receipt inventory')
            for path, expected in pins.items():
                need(record[path]['sha256'] == expected, 'pin receipt SHA')
        before = records[0]
        for path, expected in pins.items():
            if before[path]['source_copied']:
                saved = folder / 'frozen-source' / path.lstrip('/')
                need(sha(saved) == expected, 'archived source mismatch: ' + path)
                frozen_count += 1
            if len(records) == 2:
                need(records[1][path]['stat'] == before[path]['stat'], 'runtime stat changed during run')
    runtime_bytes = 0
    if runtime_pins:
        for path, expected in pins.items():
            need(sha(path) == expected, 'current runtime pin mismatch: ' + path)
            runtime_bytes += Path(path).stat().st_size
    complete, child = read(owner / 'complete.json'), read(screen / 'complete.json')
    launch, release = read(owner / 'child/launch.json'), read(owner / 'child/release.json')
    need(complete['status'] == child['status'] == 'FAIL', 'failure statuses')
    need(complete['owner_pid'] == launch['ppid'] == child['ppid'] == 58415, 'owner binding')
    need(launch['pid'] == release['pid'] == child['pid'] == 58427, 'child binding')
    need(child['contract'] == plan['contract'] == read(screen / 'plan.json'), 'child contract')
    need(child['checks'] == [] and child['source_pins_rechecked'] is False, 'no finished checks')
    need(len(child['cases']) == 1, 'one attempted case')
    case = child['cases'][0]
    need(case == read(screen / 'cases/identity-2-to-4/case.json'), 'case copy')
    need(case['id'] == 'identity-2-to-4' and case['status'] == 'FAIL'
         and case['snapshots'] == case['witness'] == {}, 'no snapshots or rebuild witness')
    need(not list(screen.rglob('*.safetensors')) and not (screen / 'events.jsonl').read_bytes(), 'no saved output/comparison events')
    error = child['error']
    need(error['type'] == 'TypeError' and error['message'] == "unsupported operand type(s) for |: 'int' and 'mlx.core.array'", 'input generator failure')
    need('in pattern' in error['traceback'] and '0x3e00 | low' in error['traceback']
         and 'in rebuild' not in error['traceback'], 'failure before rebuild')
    need(child['device_run'] is True and child['device'] == {'name':'Apple M5 Ultra','architecture':'applegpu_g17d','mlx_version':'0.32.3'}, 'device receipt')
    for path, expected in child['sources'].items():
        need(pins[path] == expected, 'screen/source pin binding')
    need(launch['command'] == [plan['execution']['python'], '-I', '-B', plan['execution']['screen'], '--run', '--out',
         registration['registered_paths']['owner'] + '/child/screen'], 'child command')
    need(launch['environment'] == plan['execution']['environment'], 'child environment')
    need(release['returncode'] == 1 and release['waited_reaped'] is True and release['strict_absence'] is True
         and release['absence_probe'] == {'returncode':1,'stdout':'','stderr':''}, 'child cleanup')
    phases = [json.loads(line) for line in (screen / 'phases.jsonl').read_text().splitlines()]
    need([(r['index'],r['case'],r['phase']) for r in phases] == [(0,'global','initial'),(1,'identity-2-to-4','before_case')], 'two partial phases')
    for i, row in enumerate(phases):
        receipt = row['physical_receipt']
        need(receipt == child['physical_receipts'][i] and row['status'] == 'PASS'
             and row['pid'] == 58427 and row['ppid'] == 58415, 'phase identity')
        need(sha(screen / receipt['path']) == receipt['sha256'], 'physical SHA')
        data = read(screen / receipt['path'])
        need(data['unit'] == 'byte' and data['bytes per unit'] == 1 and data['errors'] == data['warnings'] == [], 'physical units/errors')
        need(len(data['processes']) == 1 and data['processes'][0]['pid'] == 58427, 'physical PID')
        p = data['processes'][0]
        top, cur, peak = p['footprint'], p['auxiliary']['phys_footprint'], p['auxiliary']['phys_footprint_peak']
        need(all(type(v) is int and 0 <= v <= 2 << 30 for v in (top,cur,peak)) and peak >= cur, 'physical budget')
        need([row[k] for k in ('physical_top_bytes','physical_aux_current_bytes','physical_aux_peak_bytes','physical_current_bytes','physical_peak_bytes')]
             == [top,cur,peak,max(top,cur),max(top,peak)], 'physical parsed values')
        need(row['active_bytes'] == row['peak_bytes'] == row['cache_bytes'] == 0, 'no materialized MLX state in phases')
        command = read(screen / f'physical-{i:03d}.command.json')
        need(command['returncode'] == 0 and command['command'] == ['/usr/bin/footprint','-p','58427','--noCategories','-f','bytes','-j',
             registration['registered_paths']['owner'] + f'/child/screen/physical-{i:03d}.raw.json'], 'self footprint command')
    observed = read(owner / 'footprint-000.json')
    need(observed['returncode'] == 66 and observed['child_poll_after_probe'] == 1
         and observed['raw_exists'] is False and not (owner / observed['raw_file']).exists(), 'secondary failed-child footprint race')
    need(complete['physical_samples'] == 0, 'no successful owner footprint measurements')
    resources = read(owner / 'resources.json')
    need([r['label'] for r in resources] == ['preflight','running','postflight'] and complete['samples'] == 3, 'resource count')
    for row in resources:
        need(row['pressure_level_value'] == 2 and row['pressure_level']['returncode'] == 0 and row['pressure_level']['text'].strip() == '2', 'pressure raw')
        vm = row['vm_stat']; swap = row['swap']
        need(vm['returncode'] == swap['returncode'] == 0, 'resource commands')
        page = int(re.search(r'page size of (\d+) bytes', vm['text'])[1])
        free = int(re.search(r'^Pages free:\s*(\d+)\.', vm['text'], re.M)[1]) * page
        swapout = int(re.search(r'^Swapouts:\s*(\d+)\.', vm['text'], re.M)[1])
        used = float(re.search(r'used = ([\d.]+)M', swap['text'])[1])
        need(free == row['real_free_bytes'] and free >= 50 << 30, 'real free')
        need(swapout == row['vm_counters']['Swapouts'] == 316235 and used == row['swap_used_mb'] == 445.06, 'unchanged raw swap')
        need(row['native_process_inventory_ok'] is True and row['process_inventory_ok'] is True
             and row['authorized_suspended_pids'] == [74384], 'inventory receipt')
        inventory = row['native_inference_processes'] + row['known_inference_processes']
        need(all(p['pid'] in (74384,58427) for p in inventory), 'no foreign recognized inference')
        service = [p for p in inventory if p['pid'] == 74384]
        need(len(service) == 1 and service[0]['state'] == 'T' and service[0]['authorized_suspended'] is True, 'suspended service')
        power, thermal = row['power_metadata'], row['processinfo_thermal_metadata']
        need(power['source'] == 'AC Power' and power['raw']['returncode'] == 0 and 'AC Power' in power['raw']['stdout'], 'power')
        need(thermal['code'] == 0 and thermal['raw']['returncode'] == 0 and thermal['raw']['stdout'] == '0', 'thermal')
    lease, job, resume = (read(lease_dir / f) for f in ('lease.json','job-0.json','resume.json'))
    jobs = read(inp / 'jobs.json')
    expected_command = [plan['execution']['python'],'-I','-B',plan['execution']['screen'].replace('/screen.py','/owner.py'),
                        '--plan',registration['registered_paths']['inputs']+'/plan.json','--plan-sha256',PLAN_SHA,
                        '--out',registration['registered_paths']['owner']]
    need(jobs == lease['jobs'] == [job['command']] == [expected_command], 'lease/owner/plan command binding')
    need(lease['child_pid'] == job['pid'] == 58415 and job['returncode'] == 1 and job['waited'] is True, 'lease reaped owner')
    need(lease['closed'] is True and lease['service_pid'] == 74384
         and lease['service_identity'] == 'Fri Oct 16 08:01:58 2026     omlx-server', 'lease identity')
    need(resume['identity_match'] is True and resume['state_before'] == 'T' and resume['state_after'] == 'S'
         and lease['paused_at'] < resume['time'] <= lease['deadline'], 'restoration receipt')
    need(not complete['performance_qualified'] and not child['performance_qualified'] and not child['full_model_qualified'], 'scope')
    return dict(audit_status='PASS_EVIDENCE_VERIFICATION',experiment_status='INPUT_GENERATOR_FAILURE_NOT_QUALIFIED',
                verifier_sha256=sha(__file__),
                numerical_qualified=False,performance_qualified=False,full_model_qualified=False,
                plan_sha256=PLAN_SHA,pins=570,archived_source_copies_verified=frozen_count,
                runtime_pins_rehashed=570 if runtime_pins else 0,runtime_pin_bytes=runtime_bytes,
                completed_cases=0,attempted_cases=1,comparisons=0,snapshots=0,physical_phases=2,
                resources=3,successful_owner_physical_samples=0,child_pid=58427,child_returncode=1,owner_pid=58415,
                child_reaped=True,owner_waited=True,service_pid=74384,service_identity_restored=True,
                candidate_rebuild_reached=False,candidate_factory_and_OFF_initialization_reached=True,
                restoration_scope='Pinned suspended_service source + saved same-identity T-to-S receipt; no new live service probe',
                source_pins=child['sources'],lease_source_sha256=pins[next(p for p in pins if p.endswith('/suspended_service.py'))])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-pins', action='store_true')
    args = parser.parse_args()
    print(json.dumps(verify(args.runtime_pins), indent=2, allow_nan=False))
