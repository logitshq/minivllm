"""Default-OFF E057 shared-MLP+head cost audit; stdlib saved evidence only.

Frozen E054v2 byte/shape/finite/continuation/statistics helpers remain unchanged.
The actual E057 schema and shared-output qualification are checked separately.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import time
import traceback

HERE=Path(__file__).resolve().parent;B=HERE.parent;R=B.parents[1]
COST_AUDIT=B/'rowwise_native_verify_cost_analysis_v2/audit.py'
COST_AUDIT_SHA='483fac0a92befa1d946a9b7dd75a578f0564a167e175f29661da6cf81847afcb'
CANARY_AUDIT=B/'rowwise_mlp_shared_analysis_v1/audit.py'
CANARY_AUDIT_SHA='3ce4f312ce3e7a4c76ea429dcd9617d23430bfa0bb4bfa7105379562502240a5'
BASE=B/'rowwise_native_verify_cost_v1/screen.py'
BASE_SHA='679ad62a8b2973c9ea6d767da3135782345ca09c870ff071ff1494005b914d4d'
SCREEN=B/'rowwise_mlp_shared_cost_v1/screen.py'
SCREEN_SHA='2922675fca185ca0fd04f3da6d22c96268fe4b41d66bdd305a2ec914f778be9c'
OWNER=B/'rowwise_mlp_shared_cost_device_v1'
CANARY=B/'rowwise_mlp_shared/canary.py'
E056_PARENT=B/'rowwise_mlp_shared_device_v1/prereg-v1/plan.json'
PARENT_SCHEMA='E057-c1-mlp-shared-cost-owner-plan-v1';OWNER_SCHEMA='E057-c1-mlp-shared-cost-owner-v1'
TERMINAL_SCHEMA='E057-terminal-hold-v1';SUCCESS='PASS_LOCAL_COST_ONLY'


def load(path,digest,name):
    if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:raise ValueError('frozen auditor helper changed '+str(path))
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


cost=load(COST_AUDIT,COST_AUDIT_SHA,'_e057_frozen_e054_v2');h=cost.base
c56=load(CANARY_AUDIT,CANARY_AUDIT_SHA,'_e057_frozen_e056_audit')
base=load(BASE,BASE_SHA,'_e057_frozen_e054_contract')
need,same,read,sha,parse=h.need,h.same,h.read,h.sha,h.parse
SOURCE_PINS={str(SCREEN):SCREEN_SHA,str(BASE):BASE_SHA,
 str(B/'rowwise_mlp_shared/candidate.py'):'02200afa33fd6daa3f0f4dac3436b717616cf620f4b6cd7e59f1d8f263658950',
 str(CANARY):'c1aeae682a5822a0483926fb588f3e184d307a1aeb7fc3a2380a0d16bf85ffd8',
 str(CANARY_AUDIT):CANARY_AUDIT_SHA,str(B/'nvfp4_multirow_shared/candidate.py'):'d1826b3c4d07d44adbae071ef2da43197348c09b2d8be1035eb1a0c2788d132e'}
DEPENDENCY_PINS={p:s for p,s in h.SOURCE_PINS.items() if not p.endswith(('/execute.py','/runner.py','/terminal_hold.py'))}


def contract():
    return dict(schema='E057-independent-shared-MLP-cost-contract-v1',enabled_by_default=False,raw_files=32,
        full_output_state_pairs=16,tensor_pairs=2064,selected_prefixes=6,timing_count=400,group_count=20,
        qualified_mlp_calls=768,qualified_head_calls=6,timed_mlp_observer=False,device_execution=False,
        full_model_qualified=False,http_qualified=False,performance_qualified=False,engine_pareto_point=False,
        scope='Actual shared MLP+head output propagation, selected prefix and native continuation bytes; full verifier/commit local cost. No draft/acceptance/scheduler/HTTP costs, no400 full-tensor timing claim.')


def producer_contract():
    return dict(schema='C1-MLP-shared-cost-contract-v1',enabled_by_default=False,widths=[4,8],used=1023,
        inherited_device_loop=dict(path=str(BASE),sha256=BASE_SHA),shared_head=True,shared_mlp_calls_per_forward=128,
        qualification='Six k=1,T/2,T: fullT logits + selected128 states + native continuation;32 raw files/2064 tensor pairs.',
        timing='Unchanged E054400 calls: native device AR(T) primary, host AR(T) secondary, versus MLP+head verifier/commit.',
        timed_mlp_observer=False,head_only_baseline='Historical E054 only; not a paired third arm.',
        compiled_qualified=False,full_model_qualified=False,http_qualified=False,performance_qualified=False,engine_pareto_point=False)


def mlp_receipts(rows):
    need(len(rows)==6,'six actual shared MLP qualification records')
    for row,(width,k) in zip(rows,((4,1),(4,2),(4,4),(8,1),(8,4),(8,8))):
        expected=dict(T=width,k=k,status='PASS',call_count=128,
            scope='Python qualification calls; not GPU dispatch counts or timed observer',calls=[
                dict(layer=i,projection=name,T=width,K=5120,N=17408,weight_shape=[17408,640],scales_shape=[17408,320],
                     input_dtype='mlx.core.bfloat16',output_dtype='mlx.core.bfloat16') for i in range(64) for name in ('gate','up')])
        same(row,expected,'complete ordered128 actual shared MLP callbacks at '+str((width,k)))


def closed_owner(parent_path,parent_sha,child,remember):
    """Saved closure only; do not import/run producer owner or probe live state."""
    def get(path):
        path=Path(path);need(path.is_file() and path.stat().st_size<16<<20,'bounded small evidence '+str(path))
        remember[str(path)]=sha(path);return read(path)
    parent=get(parent_path);need(remember[str(parent_path)]==parent_sha,'explicit actual parent SHA')
    need(set(parent)=={'schema','pins','runtime_pins','numerical_command','child_environment','service','numerical_plan_sha256',
        'screen_deadline_seconds','driver_deadline_seconds','max_observation_gap_seconds','physical','outputs','scope'}
        and parent['schema']==PARENT_SCHEMA and parent['outputs']['numerical']==str(child),'fixed executed parent/child fields')
    need(set(parent['outputs'])=={'execution','screen','lease','numerical'} and set(parent['service'])=={'pid','identity','process'},'exact parent paths/service fields')
    same([parent[k] for k in ('screen_deadline_seconds','driver_deadline_seconds','max_observation_gap_seconds')],[1650,1800,12],'registered bounded owner intervals')
    pins=parent['pins'];need(all(pins.get(p)==s for p,s in {**DEPENDENCY_PINS,**SOURCE_PINS}.items()),'frozen numerical source closure')
    need(all(str(OWNER/name) in pins for name in ('execute.py','runner.py','terminal_hold.py')),'preregistered owner source closure')
    execution=Path(parent['outputs']['execution']);owner=Path(parent['outputs']['screen']);lease=Path(parent['outputs']['lease'])
    final=get(execution/'final.json');state=get(lease/'lease.json');resume=get(lease/'resume.json');job=get(lease/'job-0.json')
    need(final['status']=='PASS_CAPTURE_AND_RESTORE' and final['errors']==[] and final['lease_returncode']==0
         and final['service_identity_after']==parent['service']['identity'],'driver successful restore receipt')
    need(state['closed'] is True and state['service_pid']==parent['service']['pid']
         and ' '.join(state['service_identity'].split())==parent['service']['identity'],'closed same-service lease')
    need(resume['identity_match'] is True and resume['state_after'] and 'T' not in resume['state_after'],'resumed service receipt')
    same(get(execution/'plan.json'),parent,'retained preregistered parent plan')
    actual=get(execution/'execution-plan.json');same({k:v for k,v in actual.items() if k!='physical_baseline'},parent,'only registered physical baseline added')
    need('physical_baseline' in actual,'saved admission baseline');same(get(owner/'plan.json'),actual,'same actual owner plan')
    command=[str(R/'.venv/bin/python'),'-I','-B',str(OWNER/'runner.py'),'--plan',str(execution/'execution-plan.json'),
             '--plan-sha256',remember[str(execution/'execution-plan.json')],'--out',str(owner)]
    same(state['jobs'],[command],'one owner job');same(job['command'],command,'exact waited owner command')
    identity=get(execution/'runner-identity.json')
    need(job['waited'] is True and job['returncode']==0 and job['pid']==state['child_pid']==identity['pid']
         and identity['ppid']==state['owner_pid'],'owner PID/wait chain')
    status=get(owner/'result.json');reap=get(owner/'child-reap.json')
    need(status['schema']==OWNER_SCHEMA and status['status']=='PASS_CAPTURE_ONLY' and status['errors']==[]
         and status['child_reaped'] is True and status['child_returncode']==0,'numerical owner PASS/reap')
    need(reap['waited'] is True and reap['absent'] is True and reap['returncode']==0 and reap['errors']==[],'saved numerical reap')
    done=get(owner/'numerical-done.json');ack=get(owner/'numerical-exit-ack.json')
    need(done['schema']==TERMINAL_SCHEMA and done['pid']==reap['pid'] and done['ppid']==job['pid']
         and done['qualifier_returncode']==0 and done['wrapper_errors']==[],'terminal original successful exit')
    same(done['command'],parent['numerical_command'],'terminal exact shared cost command');same(ack['command'],done['command'],'ack command')
    need(ack['pid']==done['pid'] and ack['owner_pid']==done['ppid'] and ack['done_sha256']==remember[str(owner/'numerical-done.json')]
         and ack['final_physical_and_metadata_complete'] is True,'bound terminal acknowledgement')
    n=status['physical_samples'];m=status['resource_samples'];need(type(n) is int and type(m) is int and n==m and n>=2,'complete final observations')
    for name,path in (('physical_sha256',owner/'resources'/f'physical-{n-1:04d}'/'sample.json'),('metadata_sha256',owner/'resources'/f'resource-{m-1:04d}.json')):
        get(path);need(ack[name]==remember[str(path)],'ack bound final observation '+name)
    need(not (owner/'terminal-hold-error.json').exists(),'no terminal failure')
    copies=get(execution/'source-copies.json');need(len(copies)==len(pins) and {c['original'] for c in copies}==set(pins),'complete retained source closure')
    for row in copies:
        path=Path(row['copy']);original=Path(row['original'])
        need(path.resolve().is_relative_to(execution.resolve()/'sources') and path.stat().st_size==original.stat().st_size==row['bytes']<16<<20,'bounded source copy')
        need(row['sha256']==pins[row['original']]==sha(path)==sha(original),'retained/current source pin mismatch')
        remember[str(path)]=row['sha256'];remember[str(original)]=row['sha256']
    same(get(execution/'runtime-pins-verified.json'),parent['runtime_pins'],'saved runtime payload verification receipt')
    complete=get(child/'complete.json');need(done['qualifier_complete']==dict(path=str(child/'complete.json'),sha256=remember[str(child/'complete.json')],status=SUCCESS),'complete terminal SHA/status')
    need(complete['pid']==done['pid'] and complete['ppid']==done['ppid'],'numerical complete identity')
    plan=get(child/'plan.json');need(remember[str(child/'plan.json')]==complete['plan_sha256']==parent['numerical_plan_sha256'],'actual child plan SHA')
    expected=[str(R/'.venv/bin/python'),'-I','-B',str(SCREEN),'--run','--plan',str(OWNER/'prereg-v1/child-plan.json'),
              '--plan-sha256',parent['numerical_plan_sha256'],'--out',str(child)]
    same(parent['numerical_command'],expected,'exact registered shared cost invocation')
    need(pins[expected[6]]==parent['numerical_plan_sha256'],'bound original child plan')
    need(set(plan)=={'schema','adapter_sha256','e054_plan','e056_plan','e056_result','e056_audit','deadline_s'}
         and plan['schema']=='C1-MLP-shared-cost-plan-v1' and plan['adapter_sha256']==SOURCE_PINS[str(SCREEN)],'exact shared cost child plan')
    need(type(plan['deadline_s']) in (int,float) and math.isfinite(plan['deadline_s']) and 0<plan['deadline_s']<=1800,'bounded child deadline')
    prior={}
    for key in ('e054_plan','e056_plan','e056_result','e056_audit'):
        ref=plan[key];need(set(ref)=={'path','sha256'} and Path(ref['path']).is_absolute() and pins[ref['path']]==ref['sha256'],'actual parent-bound prior ref '+key)
        prior[key]=get(ref['path']);need(remember[ref['path']]==ref['sha256'],'unchanged prior evidence')
    e056plan=prior['e056_plan'];previous=prior['e056_result'];audit=prior['e056_audit']
    need(plan['e054_plan']['sha256']==c56.E054['e054_plan'],'same fixed original E054 plan')
    same(e056plan['e054_plan'],plan['e054_plan'],'same E054 identity through E056')
    need(e056plan['schema']=='C1-MLP-shared-activation-plan-v1' and e056plan['canary_sha256']==c56.SOURCE_PINS[str(CANARY)],'frozen E056 canary plan')
    need(previous['schema']=='C1-MLP-shared-activation-canary-result-v1' and previous['status']=='PASS_PROJECTION_CANARY_ONLY'
         and previous['plan_sha256']==plan['e056_plan']['sha256'],'actual completed E056 canary')
    c56.producer_receipts(previous)
    need(audit['schema']=='E056-MLP-canary-independent-v1' and audit['status']=='PASS' and audit['projection_bytes_audited'] is True
         and audit['audit_source_sha256']==CANARY_AUDIT_SHA and audit['complete_sha256']==plan['e056_result']['sha256']
         and audit['raw_files']==30 and audit['input_files']==6 and audit['tensor_pairs']==12
         and audit['provenance_receipt_pairs']==4 and audit['provenance_receipt_arrays']==516
         and audit['raw_full_state_checked'] is False,'actual independent E056 byte/provenance audit')
    # E056's audit binds its exact completed parent and original parameter hashes.
    need(pins.get(str(E056_PARENT))==audit['parent_plan_sha256'],'parent-bound audited E056 owner plan')
    p56=get(E056_PARENT);need(remember[str(E056_PARENT)]==audit['parent_plan_sha256'],'same actual E056 parent')
    need(p56['schema']==c56.PARENT_SCHEMA and p56['service']['pid']==parent['service']['pid']
         and p56['service']['identity']==parent['service']['identity'],'same-service audited E056 owner')
    need(p56['outputs']['numerical']==str(Path(plan['e056_result']['path']).parent)
         and p56['numerical_plan_sha256']==plan['e056_plan']['sha256'],'audited E056 output/child identity')
    same(complete['seed'],previous['seed'],'actual E042 seed through E056');need(complete['seed']['used']==1023 and complete['seed']['position']==1022,'fixed seed position')
    parameter_path=Path(plan['e056_result']['path']).parent/'parameters-before.json'
    prior_parameters=get(parameter_path);need(remember[str(parameter_path)]==audit['source_and_small_evidence'][str(parameter_path)],'audited E056 parameter receipt')
    return parent,plan,complete,prior_parameters

def numerical_files(child,complete,result,remember):
    reader=h.raw_reader(); indices={}; raw_stats={}; seen=set()
    same([r['label'] for r in complete['raw_comparisons']],h.expected_labels(),'all16 ordered raw pairs')
    for row in complete['raw_comparisons']:
        label=row['label']; width=1 if label.endswith('/continuation') else int(label[1]); need(len(row['files'])==2,'exact reference+actual pair')
        pair=[]
        for arm,desc in zip(('reference','actual'),row['files']):
            relative=label+'/'+arm+'.safetensors'; need(desc['path']==relative,'fixed saved array path')
            path=child/relative; need(not path.is_symlink(),'no snapshot alias'); st=path.stat(); raw_stats[str(path)]=(st.st_size,st.st_mtime_ns,st.st_ino)
            sidecar=path.with_suffix('.json'); remember[str(sidecar)]=sha(sidecar); same(read(sidecar),desc,'exact saved sidecar')
            same(desc['metadata'],h.snapshot_metadata(),'complete snapshot metadata'); index=reader.snapshot_index(child,desc); h.check_layout(index,width)
            pair.append(index); seen.add(relative); result['raw_files']+=1
        result['pairs'].append(h.compare_pair(reader,*pair,label)); result['tensor_pairs']+=len(pair[0]['tensors']); indices[label]=pair[0]
    need({str(p.relative_to(child)) for p in child.rglob('*.safetensors')}==seen and len(seen)==32 and result['tensor_pairs']==2064,'all32 files/2064 tensor pairs; no hidden failure arrays')
    full=reader.tensor_bytes(indices['T8/device'],'root/logits'); ids=h.greedy(full,8)
    need(ids[:7]==complete['seed']['tokens'][1:] and not set(ids)&set(complete['seed']['original_fixture']['eos_token_ids']),'saved causal outputIDs and normal EOS')
    state_anchors={}
    for label,index in indices.items():
        parts=label.split('/'); width=int(parts[0][1:]); count=int(parts[1][1:]) if parts[1].startswith('k') else width
        prefix=count+int(label.endswith('/continuation'))
        raw=reader.tensor_bytes(index,'root/logits')
        if label.endswith('/continuation'):
            if count<8: need(raw==full[count*248320*2:(count+1)*248320*2],'continuation same oracle row '+label)
            need(not set(h.greedy(raw,1))&set(complete['seed']['original_fixture']['eos_token_ids']),'continuation normal EOS')
        else: need(raw==full[:width*248320*2],'same original oracle logits '+label)
        states={k:{f:v[f] for f in ('shape','encoding','sha256')} for k,v in index['tensors'].items() if k.startswith('root/state/')}
        if prefix in state_anchors: same(states,state_anchors[prefix],'same full saved oracle state at prefix '+str(prefix))
        else: state_anchors[prefix]=states
    return ids,raw_stats

def audit(child,parent_path,parent_sha):
    started=time.monotonic();child=Path(child).resolve();parent_path=Path(parent_path).resolve();remember={}
    result=dict(schema='E057-shared-MLP-cost-independent-v1',status='INCOMPLETE',contract=contract(),child=str(child),parent_plan_sha256=parent_sha,
        audit_source_sha256=sha(__file__),cost_audit_sha256=COST_AUDIT_SHA,canary_audit_sha256=CANARY_AUDIT_SHA,raw_reader_sha256=h.RAW_SHA,
        raw_files=0,tensor_pairs=0,pairs=[],timings=[],local_cost_audited=False,shared_mlp_cost_audited=False,
        full_model_qualified=False,http_qualified=False,performance_qualified=False,engine_pareto_point=False,
        limits=['All16 saved full-output/state pairs are independently checked;400 timed samples retain IDs, not full tensor snapshots.',
                'Six ordered128-MLP-call records and one head per qualified forward are source-bound Python receipts, not GPU dispatch counters.',
                'Pre-timing1349-parameter equality is enforced at the frozen producer write boundary; offline audit compares saved original-byte hash receipts.',
                'Break-even assumes allT proposed tokens are useful. Draft, acceptance, rollback, scheduler and HTTP cost are excluded; no engine winner claim.',
                'Saved owner/restoration/source closure is checked; no live process/resource probe or model payload rehash.'])
    try:
        parent,plan,complete,original_parameters=closed_owner(parent_path,parent_sha,child,remember)
        need(complete['schema']=='C1-MLP-shared-cost-result-v1' and complete['status']==SUCCESS,'actual E057 producer successful terminal')
        for key in ('numerical_screen_qualified','pre_timing_parameters_unchanged','device_cleanup_completed','source_model_pins_rechecked'):
            need(complete[key] is True,'complete numerical/pre-timing gate '+key)
        for key in ('full_model_qualified','http_qualified','performance_qualified','engine_pareto_point'):need(complete[key] is False,'excluded claim '+key)
        same(complete['contract'],producer_contract(),'actual E057 shared-output contract')
        same(complete['inherited_contract'],base.contract(),'unchanged inherited E054 measuring contract')
        same(complete['inherited_plan'],plan['e054_plan'],'explicit inherited plan, not original E054 execution')
        need(complete['adaptation']=='Original E054 device code with private candidate_run/write/helper dependencies; actual shared MLP+head candidate, not original E054 run.','explicit actual adaptation identity')
        same(complete['sources'],SOURCE_PINS,'actual E057 execution source map')
        same(complete['device'],dict(name='Apple M5 Ultra',architecture='applegpu_g17d',mlx_version='0.32.3'),'actual device receipt')
        need(not complete.get('error') and not (child/'failure.json').exists(),'no producer failure')
        elapsed=complete['elapsed_wall_s'];need(type(elapsed) in (int,float) and math.isfinite(elapsed) and 0<elapsed<=plan['deadline_s'],'actual bounded elapsed')
        mlp_receipts(complete['mlp_qualifications'])
        ids,raw_stats=numerical_files(child,complete,result,remember)
        # These exact source-frozen helpers retain the original output/commit/head,
        # canonical root/root receipt and full400 duration statistical contracts.
        h.qualification(complete['qualification'],ids);cost.producer_checks(complete)
        reader=h.raw_reader();reader.parameter_receipt(original_parameters);original_values=reader.parameter_values(original_parameters)
        for name in ('parameters-before.json','parameters-before-timing.json','parameters-after.json'):
            path=child/name;remember[str(path)]=sha(path);receipt=read(path);reader.parameter_receipt(receipt)
            same(reader.parameter_values(receipt),original_values,'all1349 original parameter hashes before/before-timing/after')
        timing_path=child/'timings.jsonl';remember[str(timing_path)]=sha(timing_path)
        result['timings'],total_ns=h.timing_ledger(timing_path,complete['timings'],ids,plan['deadline_s'])
        need(total_ns<=elapsed*1e9,'nonoverlapping timings bounded by whole run')
        memory=complete['memory_after_cleanup'];need(set(memory)=={'active_bytes','peak_bytes','cache_bytes'} and all(type(v) is int and v>=0 for v in memory.values()),'complete memory receipt')
        need(all(sha(p)==s for p,s in remember.items()),'small source/evidence changed during audit')
        need(all((Path(p).stat().st_size,Path(p).stat().st_mtime_ns,Path(p).stat().st_ino)==v for p,v in raw_stats.items()),'raw file changed during audit')
        result.update(status='PASS',local_cost_audited=True,shared_mlp_cost_audited=True,timing_count=400,group_count=20,selected_prefix_count=6,
            qualified_mlp_calls=768,qualified_head_calls=6,timed_mlp_observer=False,parameter_count=1349,parameter_receipt_sets=3,
            total_retained_ns=total_ns,saved_oracle_ids=ids,complete_sha256=remember[str(child/'complete.json')],source_and_small_evidence=remember)
    except BaseException as error:
        result.update(status='FAIL',local_cost_audited=False,shared_mlp_cost_audited=False,error=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc()))
    result['elapsed_wall_s']=time.monotonic()-started;return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',action='store_true');p.add_argument('--child',type=Path)
    p.add_argument('--parent-plan',type=Path);p.add_argument('--parent-plan-sha256');p.add_argument('--out',type=Path);a=p.parse_args(argv)
    if not a.run:
        if any(v is not None for v in (a.child,a.parent_plan,a.parent_plan_sha256,a.out)):p.error('explicit --run required')
        print(json.dumps(contract(),indent=2));return 0
    need(all(v is not None for v in (a.child,a.parent_plan,a.parent_plan_sha256,a.out)) and not a.out.exists(),'explicit complete inputs and fresh output')
    result=audit(a.child,a.parent_plan,a.parent_plan_sha256)
    with a.out.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({k:result[k] for k in ('status','raw_files','tensor_pairs','elapsed_wall_s')}))
    return 0 if result['status']=='PASS' else 1


if __name__=='__main__':raise SystemExit(main())
