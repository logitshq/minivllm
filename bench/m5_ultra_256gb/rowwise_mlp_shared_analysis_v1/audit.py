"""Default-OFF stdlib audit of closed E056 saved MLP projection bytes.

Only the 30 small activation/projection files are independently reread. Full
logits/state, canonical and parameter invariance remain source-bound receipts.
No backend, model payload, timing, live process or service operation.
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
HELPER=B/'rowwise_native_verify_cost_analysis_v1/audit.py'
HELPER_SHA='65b83525aa5e51c2de1a562894706a98df230463a4acd152d5c1ea3b90fd36bf'
if hashlib.sha256(HELPER.read_bytes()).hexdigest()!=HELPER_SHA:raise ValueError('frozen stdlib audit helpers changed')
_spec=importlib.util.spec_from_file_location('_e056_saved_byte_helpers',HELPER)
h=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(h)
need,same,read,sha,parse=h.need,h.same,h.read,h.sha,h.parse
CANARY=B/'rowwise_mlp_shared/canary.py';OWNER=B/'rowwise_mlp_shared_device_v1'
SOURCE_PINS={
 str(CANARY):'c1aeae682a5822a0483926fb588f3e184d307a1aeb7fc3a2380a0d16bf85ffd8',
 str(CANARY.with_name('candidate.py')):'02200afa33fd6daa3f0f4dac3436b717616cf620f4b6cd7e59f1d8f263658950',
 str(B/'rowwise_native_verify_cost_v1/screen.py'):'679ad62a8b2973c9ea6d767da3135782345ca09c870ff071ff1494005b914d4d',
 str(B/'nvfp4_multirow_shared/candidate.py'):'d1826b3c4d07d44adbae071ef2da43197348c09b2d8be1035eb1a0c2788d132e'}
DEPENDENCY_PINS={p:s for p,s in h.SOURCE_PINS.items() if not p.endswith(('/execute.py','/runner.py','/terminal_hold.py'))}
E054=dict(e054_plan='3d10d60dc72eff7e2c70444239c61be09c865b6e15109c7f43b8cdf8a6a66c2c',
 e054_result='b719a1c25257e0b0fc93e80a8d49215e223d02f708ab5f879558ffe94968e9de',
 e054_audit='7fe77b62600f36b27823b68b80e54ee32815e9b830178870679540b5e3835972')
E055_PLAN=B/'c8_batched_formal_v1/prereg-v1/plan.json'
E055_PLAN_SHA='fc24144f4c0676ca6de3aa26f93bb6eabb2976d8addbc403ec35e228b484ed4b'
E054_PARENT_SHA='f613b5df1494e9064ba84e2dd025b437177891b00bf5d98bcf18d03e5eb960ce'
WIDTHS=(4,8);LAYERS=(0,32,63);BF16='mlx.core.bfloat16'
PARENT_SCHEMA='E056-c1-mlp-shared-canary-owner-plan-v1';OWNER_SCHEMA='E056-c1-mlp-canary-owner-v1'
TERMINAL_SCHEMA='E056-terminal-hold-v1';SUCCESS='PASS_PROJECTION_CANARY_ONLY'


def contract():
    return dict(schema='E056-independent-MLP-canary-contract-v1',enabled_by_default=False,raw_files=30,input_files=6,
        projection_pairs=12,projection_tensor_pairs=12,provenance_pairs=4,provenance_arrays_per_pair=129,
        raw_state_files=0,canonical_receipt_arrays=655,parameter_count=1349,device_execution=False,
        raw_full_state_checked=False,full_model_qualified=False,http_qualified=False,performance_qualified=False,engine_pareto_point=False)


def producer_contract():
    return dict(schema='C1-MLP-shared-activation-contract-v1',enabled_by_default=False,used=1023,widths=[4,8],layers=[0,32,63],
        projection_pairs=12,input_files=6,output_files=24,raw_files=30,provenance_pairs=4,provenance_tensors_per_pair=129,
        callback_count=256,native_m1_projection_calls=1536,shadow_projection_calls=12,
        returned_projection_values='original native M1 only; shared shadow output never returned',
        inputs='actual per-row post_attention_layernorm from native-returning teacher-forced model traversal',
        provenance='T4/T8 full logits and complete selected k=T state versus original verifier and native causal oracle',
        compiled_qualified=False,full_model_qualified=False,http_qualified=False,performance_qualified=False,engine_pareto_point=False)


def expected_checks():
    def rows(label,spec):
        return [dict(check=label,name=name,shape=shape,dtype=h.DTYPES[dtype],reference_finite=True,candidate_finite=True,bit_equal=True)
                for name,(shape,dtype) in spec.items()]
    wanted=[]
    for width in WIDTHS:
        for layer in LAYERS:
            for name in ('gate','up'):wanted+=rows(f'T{width}/layer{layer}/{name}',{'root/output':([width,17408],'BF16')})
        for arm in ('native-causal','frozen-verifier'):wanted+=rows(f'T{width}/provenance/{arm}',h.layout(width))
    # snapshot(flat mx.load dictionary) adds a root/ to keys already root/.
    canonical={'root/'+key:value for key,value in h.layout(1,True).items()}
    for arm in ('canonical','oracle','T4','T8','frozen'):wanted+=rows('canonical-preserved/'+arm,canonical)
    return wanted


def producer_receipts(complete):
    same(complete['contract'],producer_contract(),'exact frozen canary contract')
    for key in ('projection_canary_complete','native_returning_provenance_complete','device_cleanup_completed','source_model_pins_rechecked'):
        need(complete[key] is True,'producer completed gate '+key)
    for key in ('full_model_qualified','http_qualified','performance_qualified','engine_pareto_point'):
        need(complete[key] is False,'excluded producer claim '+key)
    counters=dict(projection_pairs=12,raw_files=30,callback_count=256,native_m1_projection_calls=1536,shadow_projection_calls=12,shared_outputs_returned=0)
    same({k:complete[k] for k in counters},counters,'exact completed counts')
    same(complete['traversals'],[dict(T=w,callback_count=128,native_m1_projection_calls=128*w,
        shadow_projection_calls=6,shared_outputs_returned=0,status='PASS') for w in WIDTHS],'two complete actual native-returning traversals')
    same(complete['provenance'],[dict(T=w,k=w,status='PASS',labels=[f'T{w}/provenance/{arm}' for arm in ('native-causal','frozen-verifier')],
        full_logits=True,selected_state_arrays=128,raw_state_saved=False,shared_outputs_returned=0) for w in WIDTHS],'source-bound full-array provenance receipts')
    order=lambda row:(row['check'],row['name'])
    same(sorted(complete['checks'],key=order),sorted(expected_checks(),key=order),'all1183 source-bound projection/provenance/canonical checks')
    teacher=complete['teacher'];need(set(teacher)=={'inputs','output_ids','native_steps'} and teacher['native_steps']==9,'native teacher receipt')
    inputs,ids=teacher['inputs'],teacher['output_ids']
    need(len(inputs)==len(ids)==9 and all(type(t) is int and 0<=t<248320 for t in inputs+ids),'complete native teacher IDs')
    need(inputs[:8]==complete['seed']['tokens'] and inputs[1:]==ids[:8],'causal teacher recurrence and original E042 inputIDs')
    need(not (set(inputs)|set(ids))&set(complete['seed']['original_fixture']['eos_token_ids']),'normal teacher EOS')


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
    same(done['command'],parent['numerical_command'],'terminal exact canary command');same(ack['command'],done['command'],'ack command')
    need(ack['pid']==done['pid'] and ack['owner_pid']==done['ppid'] and ack['done_sha256']==remember[str(owner/'numerical-done.json')]
         and ack['final_physical_and_metadata_complete'] is True,'bound terminal acknowledgement')
    n=status['physical_samples'];m=status['resource_samples'];need(type(n) is int and type(m) is int and n==m and n>=2,'complete final observations')
    for name,path in (('physical_sha256',owner/'resources'/f'physical-{n-1:04d}'/'sample.json'),('metadata_sha256',owner/'resources'/f'resource-{m-1:04d}.json')):
        get(path);need(ack[name]==remember[str(path)],'ack bound final observation '+name)
    need(not (owner/'terminal-hold-error.json').exists(),'no terminal failure')
    need(pins.get(str(E055_PLAN))==E055_PLAN_SHA,'actual E055 predecessor plan pin')
    predecessor=get(E055_PLAN);need(remember[str(E055_PLAN)]==E055_PLAN_SHA,'unchanged E055 plan')
    lp=Path(predecessor['outputs']['lease']);ep=Path(predecessor['outputs']['execution'])
    prior=[]
    for path in (lp/'lease.json',lp/'job-0.json',lp/'resume.json',ep/'final.json'):
        prior.append(get(path));need(pins.get(str(path))==remember[str(path)],'parent-bound closed predecessor receipt')
    old_lease,old_job,old_resume,old_final=prior
    need(old_lease['closed'] is True and old_job['waited'] is True and old_job['returncode']==0
         and old_job['pid']==old_lease['child_pid'] and old_job['command']==old_lease['jobs'][0],'closed E055 owner job')
    need(old_lease['service_pid']==parent['service']['pid'] and ' '.join(old_lease['service_identity'].split())==parent['service']['identity']
         and old_resume['identity_match'] is True and old_resume['state_after'] and 'T' not in old_resume['state_after']
         and old_final['status']=='PASS_CAPTURE_AND_RESTORE' and old_final['service_identity_after']==parent['service']['identity'],'same-service restored E055 predecessor')
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
    expected=[str(R/'.venv/bin/python'),'-I','-B',str(CANARY),'--run','--plan',str(OWNER/'prereg-v1/child-plan.json'),
              '--plan-sha256',parent['numerical_plan_sha256'],'--out',str(child)]
    same(parent['numerical_command'],expected,'exact registered canary invocation')
    need(pins[expected[6]]==parent['numerical_plan_sha256'],'bound original child plan')
    need(set(plan)=={'schema','canary_sha256','e054_plan','e054_result','e054_audit','deadline_s'}
         and plan['schema']=='C1-MLP-shared-activation-plan-v1' and plan['canary_sha256']==SOURCE_PINS[str(CANARY)],'exact frozen child plan')
    need(type(plan['deadline_s']) in (int,float) and math.isfinite(plan['deadline_s']) and 0<plan['deadline_s']<=1800,'bounded child deadline')
    prior={}
    for key,digest in E054.items():
        ref=plan[key];need(set(ref)=={'path','sha256'} and Path(ref['path']).is_absolute() and pins[ref['path']]==ref['sha256']==digest,'actual bound E054 ref '+key)
        prior[key]=get(ref['path']);need(remember[ref['path']]==digest,'unchanged E054 evidence')
    previous=prior['e054_result'];audit=prior['e054_audit']
    need(previous['schema']=='C1-verifier-cost-result-v1' and previous['status']=='PASS_LOCAL_COST_ONLY' and previous['plan_sha256']==E054['e054_plan'],'actual completed E054')
    need(audit['schema']=='E054-local-cost-independent-v2' and audit['status']=='PASS' and audit['local_cost_audited'] is True
         and audit['complete_sha256']==E054['e054_result'] and audit['parent_plan_sha256']==E054_PARENT_SHA,'independently audited E054')
    same(complete['seed'],previous['seed'],'actual E042 seed through audited E054');need(complete['seed']['used']==1023 and complete['seed']['position']==1022,'fixed seed position')
    # E054's independent receipt binds the original parameter hash receipt. No
    # actual parameter/canonical/model tensor payload is opened here.
    parameter_path=Path(plan['e054_result']['path']).parent/'parameters-before.json'
    prior_parameters=get(parameter_path);need(remember[str(parameter_path)]==audit['source_and_small_evidence'][str(parameter_path)],'audited original parameter receipt')
    return parent,plan,complete,prior_parameters


def raw_captures(child,complete,result,remember,raw_stats):
    reader=h.raw_reader();seen=set();captures=complete['captures']
    same([(r['T'],r['layer']) for r in captures],[(w,i) for w in WIDTHS for i in LAYERS],'six ordered actual activation captures')
    def saved(desc,relative,width,size,key):
        need(desc['path']==relative,'exact saved tensor path');path=child/relative
        need(path.resolve()==path and path.is_file() and path.stat().st_size<1<<20,'small regular raw tensor without alias')
        st=path.stat();raw_stats[str(path)]=(st.st_size,st.st_mtime_ns,st.st_ino)
        sidecar=path.with_suffix('.json');remember[str(sidecar)]=sha(sidecar);same(read(sidecar),desc,'exact saved sidecar')
        same(desc['metadata'],{key:{'array':'root/'+key}},'actual snapshot metadata')
        index=reader.snapshot_index(child,desc);need(set(index['tensors'])=={'root/'+key},'one complete saved original tensor')
        tensor=index['tensors']['root/'+key];same(tensor['shape'],[width,size],'full saved original shape');need(tensor['encoding']=='BF16','original BF16 encoding')
        seen.add(relative);result['raw_files']+=1;return index
    for row in captures:
        width,layer=row['T'],row['layer'];tag=f'T{width}/layer{layer}'
        saved(row['input'],tag+'/input.safetensors',width,5120,'x');result['input_files']+=1
        same([p['name'] for p in row['projections']],['gate','up'],'actual gate then up')
        for pair in row['projections']:
            name=pair['name'];need(pair['status']=='PASS' and len(pair['files'])==2,'two completed projection outputs')
            indices=[saved(desc,tag+'/'+name+'-'+arm+'.safetensors',width,17408,'output') for arm,desc in zip(('native','shadow'),pair['files'])]
            result['pairs'].append(h.compare_pair(reader,*indices,tag+'/'+name));result['tensor_pairs']+=1
    need({str(p.relative_to(child)) for p in child.rglob('*.safetensors')}==seen and len(seen)==30
         and result['input_files']==6 and result['tensor_pairs']==12,'exact30 raw files/6inputs/12 pairs; no hidden failure output')


def audit(child,parent_path,parent_sha):
    started=time.monotonic();child=Path(child).resolve();parent_path=Path(parent_path).resolve();remember={};raw_stats={}
    result=dict(schema='E056-MLP-canary-independent-v1',status='INCOMPLETE',contract=contract(),child=str(child),parent_plan_sha256=parent_sha,
        audit_source_sha256=sha(__file__),helper_sha256=HELPER_SHA,raw_reader_sha256=h.RAW_SHA,raw_files=0,input_files=0,tensor_pairs=0,pairs=[],
        projection_bytes_audited=False,raw_full_state_checked=False,full_model_qualified=False,http_qualified=False,performance_qualified=False,engine_pareto_point=False,
        limits=['Independently reads all30 original BF16 input/output files and compares12 native/shadow projection pairs including finite bits.',
                'Full-logit/selected128-state provenance, canonical preservation and original parameter hashes are source-bound producer receipts, not offline state/model byte checks.',
                'Saved closure/source identities are checked; no live service/process/resource probe or timing/performance measurement. Shared values never feed later layers.'])
    try:
        parent,plan,complete,original_parameters=closed_owner(parent_path,parent_sha,child,remember)
        need(complete['schema']=='C1-MLP-shared-activation-canary-result-v1' and complete['status']==SUCCESS,'complete producer success')
        need(not complete.get('error') and not (child/'failure.json').exists(),'no producer failure')
        same(complete['sources'],SOURCE_PINS,'exact numerical source receipt')
        same(complete['device'],dict(name='Apple M5 Ultra',architecture='applegpu_g17d',mlx_version='0.32.3'),'fixed device receipt')
        elapsed=complete['elapsed_wall_s'];need(type(elapsed) in (int,float) and math.isfinite(elapsed) and 0<elapsed<=plan['deadline_s'],'bounded elapsed')
        producer_receipts(complete);raw_captures(child,complete,result,remember,raw_stats)
        reader=h.raw_reader();reader.parameter_receipt(original_parameters);original_values=reader.parameter_values(original_parameters)
        for name in ('parameters-before.json','parameters-after.json'):
            path=child/name;remember[str(path)]=sha(path);receipt=read(path);reader.parameter_receipt(receipt)
            same(reader.parameter_values(receipt),original_values,'all1349 original parameter value receipts unchanged')
        memory=complete['memory_after_cleanup'];need(set(memory)=={'active_bytes','peak_bytes','cache_bytes'} and all(type(v) is int and v>=0 for v in memory.values()),'complete cleanup memory receipt')
        need(all(sha(p)==s for p,s in remember.items()),'small source/evidence changed during audit')
        need(all((Path(p).stat().st_size,Path(p).stat().st_mtime_ns,Path(p).stat().st_ino)==v for p,v in raw_stats.items()),'raw file changed during audit')
        result.update(status='PASS',projection_bytes_audited=True,provenance_receipt_pairs=4,provenance_receipt_arrays=516,
            canonical_receipt_arrays=655,parameter_receipt_count=1349,complete_sha256=remember[str(child/'complete.json')],source_and_small_evidence=remember)
    except BaseException as error:
        result.update(status='FAIL',projection_bytes_audited=False,error=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc()))
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
