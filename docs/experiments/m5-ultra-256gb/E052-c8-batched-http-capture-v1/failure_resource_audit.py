"""E052 v1 exact readiness failure: saved resource/restore scope only.

Raw resource reader algorithms are frozen E045. The copied audit control flow
requires the actual known FAIL/exit1/readiness-error values; it never rewrites a
saved status or feeds a fake PASS view. No device or live probe.
"""
import argparse,base64,hashlib,importlib.util,json
from pathlib import Path
R=Path('/Users/ryanli/code/minivllm');B=R/'bench/m5_ultra_256gb'
HELPER=B/'concurrency_config_analysis_v1/resources.py'
HELPER_SHA='0f1b4c0092f5af909a9bb5e3afb87a2dead071b2c8ae4bd7ffbb43fea7d961f9'
PLAN_SHA='3e14ce911125eb46147943d5024510414271198984588dd91fea8319df8947e6'
SPEC_SHA='8f7d0095c98d70bb4d72609a1a91e48581da32fc0695e6363f0fadbafa0c281c'
LABELS=('c8-batched-mb50-two-waves',);GIB=1<<30
def digest(raw):return hashlib.sha256(raw).hexdigest()
assert digest(HELPER.read_bytes())==HELPER_SHA
spec=importlib.util.spec_from_file_location('_e052_original_raw_resource_readers',HELPER)
h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
need,norm,command,health,physical,metadata=h.need,h.norm,h.command,h.health,h.physical,h.metadata

def audit(execution):
    refs = {}; result = dict(schema='E052-saved-failure-resource-review-v1', status='INCOMPLETE', errors=[],
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
             and job['waited'] is True and job['returncode'] == 1, 'Runner command/parent/wait')
        need(jobs[0][jobs[0].index('--plan-sha256') + 1] == refs[str(execution / 'execution-plan.json')]['sha256'], 'Runner execution-plan SHA')
        need(lease['closed'] is True and lease['current_job'] == 0 and resume['identity_match'] is True
             and resume['state_after'] and 'T' not in resume['state_after']
             and lease['paused_at'] < resume['time'], 'Original lease/resume receipt')
        health(read(execution / 'before-health.json')); health(read(execution / 'after-health.json'))
        need(final['status'] == 'FAIL' and final['errors'] == [dict(type='AssertionError',message='screen/lease failed'),dict(type='AssertionError',message='closure: ')] and final['lease_returncode'] == 1
             and whole['status'] == 'FAIL' and whole['errors'] == [dict(type='AssertionError',message='trial failed: '+LABELS[0])]
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
                 and complete['status'] == 'FAIL' and complete['errors'] == [dict(phase='readiness',type='ValueError',message='actual startup identity differs from capture contract')]
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
        result.update(status='PASS_SAVED_FAILURE_RESTORE', physical_samples=len(values), metadata_samples=len(metadata_rows),
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

def review(execution):
    result=audit(execution)
    def binary(path):
        raw=path.read_bytes();result['inputs'][str(path)]=dict(bytes=len(raw),sha256=digest(raw));return raw
    def read(path):return json.loads(binary(path))
    try:
        plan=read(execution/'plan.json');need(digest(binary(execution/'plan.json'))==PLAN_SHA,'exact E052 parent plan')
        need(plan['schema']=='E052-c8-batched-http-plan-v1' and len(plan['specs'])==1,'one registered capture')
        trial=Path(plan['outputs']['screen'])/'trials'/LABELS[0]
        spec_value=read(trial/'spec.json');need(digest(binary(trial/'spec.json'))==SPEC_SHA==plan['specs'][0]['sha256'],'original spec')
        need(spec_value['schema']=='c8-batched-http-spec-v1' and spec_value['label']==LABELS[0],'actual spec schema')
        complete=read(trial/'complete.json')
        need(complete['schema']=='c8-batched-http-result-v1' and complete['cleanup_complete'] is complete['server_reaped'] is True,'failed child cleanup')
        for name,reference in complete['artifacts'].items():
            raw=binary(trial/name);need(len(raw)==reference['bytes'] and digest(raw)==reference['sha256'],'closed artifact integrity')
        events=[json.loads(row) for row in binary(trial/'events.jsonl').splitlines()]
        readiness=[e['raw'] for e in events if e['label']=='readiness'];success=[row for row in readiness if row['status']==200]
        need(len(success)==1 and all(row['status'] in (200,503) for row in readiness),'one successful readiness body')
        actual=json.loads(base64.b64decode(success[0]['body_base64'],validate=True))['benchmark_identity'];expected=spec_value['identity']
        need(expected['diagnostic_only'] is True and 'diagnostic_only' not in actual
             and actual=={k:v for k,v in expected.items() if k!='diagnostic_only'},'only observed diagnostic_only difference')
        trace=[json.loads(row) for row in binary(trial/'trace.jsonl').splitlines()]
        route=[json.loads(row) for row in binary(trial/'route-events.jsonl').splitlines()]
        need(trace==[dict(event='startup',identity=actual)] and len(route)==1
             and route[0]['event']=='route-startup' and route[0]['identity']==actual,'startup-only traces')
        need(not (trial/'group.json').exists() and not (trial/'client').exists() and not (trial/'joined-input').exists(),'no inference group or raw join')
        for path,expected_sha in plan['pins'].items():need(digest(binary(Path(path)))==expected_sha,'original small source/input pin')
        result.update(original_experiment_status='FAIL',generation_requests=0,readiness_polls=len(readiness),
            successful_readiness_bodies=len(success),identity_difference=dict(key='diagnostic_only',expected=True,actual='ABSENT'),
            original_failure_preserved=True,full_state_qualified=False,numerical_qualified=False,http_qualified=False,engine_pareto_point=False,
            scope='Only exact saved readiness failure, resource observations and restoration. No generation request, raw join, numerical or performance qualification.')
    except Exception as exc:
        result['status']='FAIL_OR_UNKNOWN';result['errors'].append(dict(type=type(exc).__name__,message=str(exc)))
    for path,reference in result['inputs'].items():
        if digest(Path(path).read_bytes())!=reference['sha256']:
            result['status']='FAIL_OR_UNKNOWN';result['errors'].append(dict(type='ChangedInput',message=path))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execution',type=Path);p.add_argument('--out',type=Path);a=p.parse_args()
    if a.execution is None and a.out is None:print('Inert: explicit closed --execution and fresh --out required.')
    else:
        need(a.execution is not None and a.out is not None,'both arguments required');result=review(a.execution.resolve())
        with a.out.open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
        print(json.dumps({k:result[k] for k in ('status','errors')}));raise SystemExit(0 if result['status']=='PASS_SAVED_FAILURE_RESTORE' else 1)
