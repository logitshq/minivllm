"""E054 saved resource/terminal-handshake review; no device or live probes.

Uses frozen E045 pure raw-data readers only. Run after restoration. Physical
footprints are never added together, and sampling is not an allocation cap.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

R=Path('/Users/ryanli/code/minivllm');B=R/'bench/m5_ultra_256gb'
HELPER=B/'concurrency_config_analysis_v1/resources.py'
HELPER_SHA='0f1b4c0092f5af909a9bb5e3afb87a2dead071b2c8ae4bd7ffbb43fea7d961f9'
PLAN_SHA='f613b5df1494e9064ba84e2dd025b437177891b00bf5d98bcf18d03e5eb960ce'

def sha(raw):return hashlib.sha256(raw).hexdigest()
def need(ok,message):
    if not ok:raise ValueError(message)

def review(execution):
    refs={};result=dict(schema='E054-saved-resource-review-v1',status='INCOMPLETE',errors=[],
        live_checks_performed=False,numerical_qualified_here=False,performance_qualified=False,hard_allocation_cap=False,
        limitations=['Sampled observations do not bound inter-sample allocation peaks or establish physical fit.',
            'Known/native process inventories and repeated start-time assertions have source-bound saved scope; their full raw ps history is unavailable.',
            'Final, lease and health are recorded facts, not new live checks. Saved tensor-byte and timing audits are separate; this reader grants neither numerical nor performance qualification.'])
    def binary(path):
        path=Path(path);need(not path.is_symlink(),'unexpected symlink');raw=path.read_bytes()
        refs[str(path)]=dict(bytes=len(raw),sha256=sha(raw));return raw
    def read(path):return json.loads(binary(path))
    try:
        need(sha(binary(HELPER))==HELPER_SHA,'frozen pure E045 readers changed')
        spec=importlib.util.spec_from_file_location('_e054_saved_resource_readers',HELPER)
        h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
        plan=read(execution/'execution-plan.json');registered=read(execution/'plan.json')
        need(sha(binary(execution/'plan.json'))==PLAN_SHA,'exact prepared plan')
        need({k:v for k,v in plan.items() if k!='physical_baseline'}==registered,'only observed baseline added')
        need(plan['schema']=='E054-c1-verifier-cost-owner-plan-v1' and plan['outputs']['execution']==str(execution),'E054 output binding')
        owner=Path(plan['outputs']['screen']);lease_dir=Path(plan['outputs']['lease']);child_dir=Path(plan['outputs']['numerical'])
        need(read(owner/'plan.json')==plan,'executed owner plan')
        for name,digest in plan['pins'].items():need(sha(binary(Path(name)))==digest,'source/input changed: '+name)
        copied=read(execution/'source-copies.json')
        need({r['original']:r['sha256'] for r in copied}==plan['pins'],'all frozen source copies')
        for row in copied:need(sha(binary(Path(row['copy'])))==row['sha256'] and refs[row['copy']]['bytes']==row['bytes'],'source copy integrity')
        need(read(execution/'runtime-pins-verified.json')==plan['runtime_pins'],'recorded runtime pin verification')
        invocation=read(execution/'invocation.json');need(invocation['plan_sha256']==PLAN_SHA,'original invocation plan')
        final=read(execution/'final.json');whole=read(owner/'result.json')
        result.update(original_parent_status=final['status'],original_owner_status=whole['status'])
        lease=read(lease_dir/'lease.json');job=read(lease_dir/'job-0.json');resume=read(lease_dir/'resume.json')
        runner=read(execution/'runner-identity.json');service=plan['service'];jobs=read(execution/'jobs.json')
        need(service['pid']==lease['service_pid']==52818 and h.norm(lease['service_identity'])==service['identity']==final['service_identity_after'],'original service identity')
        need(len(jobs)==1 and lease['jobs']==jobs and job['command']==jobs[0]
             and job['pid']==lease['child_pid']==runner['pid'] and runner['ppid']==lease['owner_pid']
             and job['waited'] is True and job['returncode']==0,'bound owner waited by lease')
        need(jobs[0][jobs[0].index('--plan-sha256')+1]==sha(binary(execution/'execution-plan.json')),'executed plan SHA')
        need(lease['closed'] is True and lease['current_job']==0 and resume['identity_match'] is True
             and resume['state_after'] and 'T' not in resume['state_after'],'recorded lease restoration')
        h.health(read(execution/'before-health.json'));h.health(read(execution/'after-health.json'))
        result.update(saved_health_contracts_pass=True,recorded_service_restored=True,original_lease_returncode=final['lease_returncode'])
        complete=read(child_dir/'complete.json');done=read(owner/'numerical-done.json');ack=read(owner/'numerical-exit-ack.json')
        need(complete['schema']=='C1-verifier-cost-result-v1' and complete['status']=='PASS_LOCAL_COST_ONLY'
             and complete['numerical_screen_qualified'] is complete['device_cleanup_completed'] is complete['source_model_pins_rechecked'] is True
             and complete['http_qualified'] is complete['performance_qualified'] is complete['engine_pareto_point'] is False
             and complete['plan_sha256']==plan['numerical_plan_sha256'],'registered local cost-only producer')
        need(len(complete['qualification'])==6 and {(c['T'],c['k']) for c in complete['qualification']}=={(4,1),(4,2),(4,4),(8,1),(8,4),(8,8)}
             and all(c['status']=='PASS' for c in complete['qualification']), 'six local selected-state/continuation receipts')
        need(len(complete['raw_comparisons'])==16 and len(complete['timings'])==4
             and all(len(t['samples'])==100 for t in complete['timings']), 'registered16 comparisons/400 timing rows')
        result.update(original_child_status=complete['status'],original_child_local_numerical_screen_qualified=True,
            full_history_qualified_here=False,http_qualified=False,engine_pareto_point=False,
            numerical_and_timing_audit_performed_here=False)
        pid=complete['pid'];identity=read(owner/'resources'/f'identity-{pid}.json')
        need(identity['pid']==pid==done['pid']==ack['pid'] and identity['ppid']==complete['ppid']==done['ppid']==ack['owner_pid']==runner['pid'],'direct numerical PID/PPID')
        need(read(owner/'original-qualifier-command.json')==plan['numerical_command']==done['command']==ack['command'],'unchanged qualifier command')
        command=read(owner/'child-command.json')
        need(command==[str(R/'.venv/bin/python'),'-I','-B',str(B/'rowwise_native_verify_cost_v1/terminal_hold.py'),'--run',
             '--command-json',str(owner/'original-qualifier-command.json'),'--command-sha256',sha(binary(owner/'original-qualifier-command.json')),'--control',str(owner)],'terminal wrapper invocation')
        need(done['schema']=='E054-terminal-hold-v1' and done['wrapper_errors']==[] and done['qualifier_complete']==dict(
             path=str(child_dir/'complete.json'),sha256=sha(binary(child_dir/'complete.json')),status=complete['status']), 'original terminal complete SHA/status')
        need(ack['done_sha256']==sha(binary(owner/'numerical-done.json')) and ack['final_physical_and_metadata_complete'] is True,'exact final-observation ack')
        reap=read(owner/'child-reap.json');raw_reap=read(owner/'child-reap-raw.json');absence=raw_reap['absence']
        need(raw_reap['ownership']=='direct-Popen-returned' and raw_reap['owner_pid']==runner['pid'] and raw_reap['pid']==pid
             and raw_reap['status']=='REAPED' and raw_reap['waited_reaped'] is raw_reap['strict_absence'] is True and raw_reap['errors']==[],'raw direct child reap')
        need(absence['command']==['/bin/ps','-p',str(pid),'-o','pid='] and absence['returncode']==1
             and absence['stdout'].strip()==absence['stderr'].strip()=='','saved strict numerical absence')
        need(reap['pid']==pid and reap['identity']==identity and reap['waited'] is reap['absent'] is True
             and reap['errors']==[] and reap['raw_status']=='REAPED','derived child reap')
        need(reap['returncode']==raw_reap['returncode']==done['qualifier_returncode']==whole['child_returncode']==0,'actual wait exit status')
        base_row,baseline=h.physical(execution/'physical-before',read,binary)
        need(base_row['pid']==base_row['owner_pid'] and baseline==plan['physical_baseline'],'admission baseline')
        need(plan['physical']==dict(child_limit_gib=64,system_limit_gib=240,swap_growth_bytes=0,swapouts_growth=0,
             probe_seconds=5,interval_seconds=3,hard_allocation_cap=False) and plan['max_observation_gap_seconds']==12,'fixed observation policy')
        folder=owner/'resources';paths=sorted(folder.glob('physical-*/sample.json'));metas=sorted(folder.glob('resource-*.json'));probes=sorted(folder.glob('probe-*.json'))
        need(len(paths)==len(metas)==len(probes)==whole['physical_samples']==whole['resource_samples']>=2,'complete observation matrix')
        values=[];metadata=[];gaps=[];prior=base_row
        for i,(path,meta,probe_path) in enumerate(zip(paths,metas,probes)):
            need(path==folder/f'physical-{i:04d}'/'sample.json' and meta==folder/f'resource-{i:04d}.json' and probe_path==folder/f'probe-{i:04d}.json','contiguous observations')
            row,value=h.physical(path.parent,read,binary);sample=read(meta);probe=read(probe_path)
            expected_pid=None if i==0 else pid
            label='paused-admission' if i==0 else 'numerical-done-before-exit' if i==len(paths)-1 else 'c1-verifier-cost-running'
            code="import sys,json;sys.path.insert(0,sys.argv[1]);import resources_v2 as r;print(json.dumps(r.snapshot(sys.argv[2],None if sys.argv[3]=='None' else int(sys.argv[3]),suspended_pids=(int(sys.argv[4]),))))"
            text=h.command(probe,[str(R/'.venv/bin/python'),'-I','-B','-c',code,str(B),label,str(expected_pid),str(service['pid'])])
            need(json.loads(text)==sample and sample['label']==label and sample['owned_model_pid']==expected_pid,'raw metadata subprocess/stdout/phase')
            gap=row['started_monotonic']-prior['ended_monotonic'];gaps.append(gap)
            need(0<=gap<=12 and row['owner_pid']==runner['pid'] and row['pid']==(runner['pid'] if i==0 else pid),'observation gap/owner')
            need(row['ended_monotonic']<=sample['monotonic_s'] and lease['paused_at']<=row['started_at']<=row['ended_at']<=resume['time'],'sampling inside lease')
            need(value['swap_bytes']<=baseline['swap_bytes'] and value['swapouts']==baseline['swapouts']
                 and max(value[k] for k in ('top_bytes','aux_current_bytes','aux_peak_bytes'))<=64*(1<<30),'zero swap and 64-GiB sampled threshold')
            metadata.append(h.metadata(sample,runner['pid'],service,baseline,identity if i else None));values.append(value);prior=row
        need(ack['physical_sha256']==sha(binary(paths[-1])) and ack['metadata_sha256']==sha(binary(metas[-1])),'ack bound final samples')
        need(done['ended_unix']<=prior['started_at']<=prior['ended_at']<=raw_reap['time']
             and prior['ended_monotonic']<=sample['monotonic_s']<done['ack_deadline_monotonic'],'terminal final sample before ack/reap')
        result.update(physical_samples=len(values),metadata_samples=len(metadata),child_pid=pid,owner_pid=runner['pid'],
            max_gap_seconds=max(gaps),baseline_to_first_gap_seconds=gaps[0],
            max_system_bytes=max(v['system_bytes'] for v in values),
            max_child_current_bytes=max(v['current_bytes'] for v in values if v['pid']==pid),
            max_child_historical_peak_bytes=max(v['aux_peak_bytes'] for v in values if v['pid']==pid),
            pressure_values=sorted({v['pressure'] for v in values}),thermal_codes=sorted({v['thermal'] for v in metadata}),
            swap_growth_bytes=max(v['swap_bytes']-baseline['swap_bytes'] for v in values),swapouts_values=sorted({v['swapouts'] for v in values}),
            terminal_final_sample_ack_wait_reap_verified=True,raw_full_process_inventory_reparsed=False)
        need(final['status']=='PASS_CAPTURE_AND_RESTORE' and final['errors']==[] and final['lease_returncode']==0
             and whole['status']=='PASS_CAPTURE_ONLY' and whole['errors']==[],'original owner/driver outcome')
        result['status']='PASS_SAVED_OBSERVATIONS'
    except Exception as exc:result.update(status='FAIL_OR_UNKNOWN');result['errors'].append(dict(type=type(exc).__name__,message=str(exc)))
    for path,ref in refs.items():
        try:need(sha(Path(path).read_bytes())==ref['sha256'],'input changed: '+path)
        except Exception as exc:result['status']='FAIL_OR_UNKNOWN';result['errors'].append(dict(type=type(exc).__name__,message=str(exc)))
    result['inputs']=refs;result['review_source_sha256']=sha(Path(__file__).read_bytes());return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execution',type=Path);p.add_argument('--out',type=Path);a=p.parse_args()
    if a.execution is None and a.out is None:print('Inert: explicit --execution closed-directory --out fresh.json required after restoration.')
    else:
        need(a.execution is not None and a.out is not None,'both arguments required');result=review(a.execution.resolve())
        with a.out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
        print(json.dumps({k:result[k] for k in ('status','errors')}));raise SystemExit(0 if result['status']=='PASS_SAVED_OBSERVATIONS' else 1)
