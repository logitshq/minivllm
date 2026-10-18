"""B2/B8 real-activation local GDN canary; no backend import until device().

Exact native batched reference, independent row IDs/lengths/greedy tokens and
normal EOS. Four fixed chains (early/late at B2/B8), GDN ordinals 0/24/47,
three consecutive candidate states each. First mismatch/error stops. This does
not qualify the candidate full model, scheduler, HTTP, B3..7 or performance.
The owner must separately enforce resources, cold service and hard deadline.
"""
import argparse
from contextlib import ExitStack
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
import traceback

HERE=Path(__file__).resolve().parent
BASE=HERE.parent/'gdn_prework_unbound/qualify.py'
BASE_SHA='f3088a98d2aca6b4f267e7d4c258efbbf9f83e1beccb6745c88ae42343f88e89'
CANDIDATE_SHA='acd2a23802296b9e420108203cc215c3170cd1eccd9a859e1a787432f98a5353'

def need(ok,message):
    if not ok:raise ValueError(message)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path,name,digest):
    need(sha(path)==digest,'pinned executable source changed: '+str(path))
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

_base=load(BASE,'_gdn_batched_original_canary_helpers',BASE_SHA)
HELPER,HELPER_SHA,OLD,OLD_SHA,OLD_PLAN_SHA,CORE_PINS=(getattr(_base,k) for k in
    ('HELPER','HELPER_SHA','OLD','OLD_SHA','OLD_PLAN_SHA','CORE_PINS'))
DecodeCapture,replay_chain,checked_invoke,IncompleteCoverage=(_base.DecodeCapture,_base.replay_chain,_base.checked_invoke,_base.IncompleteCoverage)


def validate(plan):
    need(sha(BASE)==BASE_SHA,'frozen single-row canary helper changed')
    need(set(plan)=={'schema','qualifier_sha256','candidate','e024_plan','deadline_s'}
         and plan['schema']=='GDN-batched-prework-real-layer-plan-v1','exact batched canary plan')
    need(plan['qualifier_sha256']==sha(__file__),'actual qualifier source')
    need(type(plan['deadline_s']) in (int,float) and math.isfinite(plan['deadline_s'])
         and 0<plan['deadline_s']<=1800,'bounded child deadline')
    need(plan['candidate']==dict(path=str(HERE/'candidate.py'),sha256=CANDIDATE_SHA),'fixed adjacent default-OFF candidate')
    candidate=load(HERE/'candidate.py','_gdn_batched_candidate',CANDIDATE_SHA)
    ref=plan['e024_plan']
    need(set(ref)=={'path','sha256'} and Path(ref['path']).is_absolute()
         and ref['sha256']==OLD_PLAN_SHA==sha(ref['path']),'unchanged inherited E024 plan')
    h=load(HELPER,'_gdn_batched_exact_helpers',HELPER_SHA)
    old=load(OLD,'_gdn_batched_old_contract',OLD_SHA)
    original=json.loads(Path(ref['path']).read_bytes());old.validate_plan(original)
    support=old.load_pinned(original['e023_qualifier'],'_gdn_batched_comparator',old.SUPPORT_SHA)
    inherited=support.pinned_json(original['e023_plan'])
    need(h.sha(CORE_PINS)==old.MODEL_SOURCE_SHA,'frozen complete core manifest')
    return h,old,support,candidate,original,inherited


def fixed_contract():
    return dict(schema='GDN-batched-prework-real-layer-contract-v1',batches=[2,8],
        cases=['b2-early','b2-late','b8-early','b8-late'],starts=[1,129],steps_per_chain=3,
        fixture_selection='sort original8 by (prompt length,fixture_id); B2 shortest2 early/longest2 late; B8 all8 both',
        gdn_ordinals=[0,24,47],expected_layer_calls=36,reference_chains=4,candidate_chains=12,
        configuration=dict(max_len=2048,max_seqs=8,max_batched_tokens=16384,cache_limit_gib=2,
            compile_decode=False,fuse_decode=False,gates_compile=True,norm_reuse=False),
        inputs='native batched full-model greedy rollout; each row consumes its own actual prior ID',
        state='independent initial candidate state, then own prior state for all three calls',
        normal_eos=True,early_eos='any row EOS before final window output is INCOMPLETE; no forced tokens or retirement',
        exact_arrays=['output:BF16[B,5120]','conv:BF16[B,3,10240]','S:FP32[B,48,128,128]'],
        saved_reference_logits='20 prefill outputs plus12 captured full-vocab batched outputs; intervening tokens have producer receipts only',
        full_model_qualified=False,http_qualified=False,performance_qualified=False,
        excluded=['candidate full-model rollout','ahead/scheduling','B3..7','parameter replacement',
            'row permutation/removal/refill','prefill optimization','E024/E028 composition','synthetic/strided coverage'])


def choose_cases(fixtures):
    _base.choose_cases(fixtures)  # unchanged original8 D512 contract
    ordered=sorted(fixtures,key=lambda f:(len(f['prompt_ids']),f['fixture_id']))
    return [dict(case_id=f'b{B}-{age}',batch_size=B,start=start,
        fixtures=(ordered[:2] if age=='early' else ordered[-2:]) if B==2 else list(ordered))
        for B in (2,8) for age,start in (('early',1),('late',129))]


def rollout_window(first_tokens,eos_ids,cap,start,advance,emit):
    need(type(first_tokens) is list and len(first_tokens) in (2,8)
         and all(type(t) is int and t>=0 for t in first_tokens),'actual B2/B8 first greedy tokens')
    tokens=list(first_tokens);raw=[[t] for t in tokens];records=[]
    for step in range(1,start+3):
        ended=[i for i,t in enumerate(tokens) if t in eos_ids or len(raw[i])>=cap]
        if ended:
            emit(dict(event='coverage-shortfall',decode_step=step,ended_rows=ended,raw_ids=[list(v) for v in raw]))
            raise IncompleteCoverage('normal EOS/cap in a real batch row before the fixed chain completed')
        captured=step>=start;next_tokens,batch=advance(list(tokens),step,captured)
        need(type(next_tokens) is list and len(next_tokens)==len(tokens)
             and all(type(t) is int and t>=0 for t in next_tokens),'actual native batched greedy IDs')
        for row,t in zip(raw,next_tokens):row.append(t)
        emit(dict(event='reference-token',decode_step=step,input_tokens=tokens,output_tokens=list(next_tokens),captured=captured))
        if captured:records.append(batch)
        tokens=list(next_tokens)
    need(len(records)==3,'three actual batched decode steps')
    return raw,records


def device(plan,h,old,support,candidate,original,inherited,out,deadline,emit,result):
    source=support.pinned_json(inherited['source_manifest'])['files']
    manifest=support.pinned_json(inherited['model_manifest']); fixtures=support.pinned_json(inherited['fixtures'])
    root=Path(inherited['package_root']).resolve()/'tutorial_code/miniengine'; model_path=Path(inherited['model']).resolve()
    need(manifest['schema']=='E023-model-v1' and Path(manifest['model']).resolve()==model_path
         and manifest['revision']==model_path.name and Path(fixtures['model']).resolve()==model_path,'fixed model/fixture identity')
    need(source==json.loads(CORE_PINS.read_bytes())['files']
         and set(source)=={str(p.relative_to(root)) for p in root.rglob('*.py')}
         and all(source.get(k)==v for k,v in candidate.PINS.items()), 'actual frozen complete core')
    required={'config.json','generation_config.json'}|{str(p.relative_to(model_path)) for p in model_path.rglob('*.safetensors')}
    need(required<=set(manifest['files']) and any(x.endswith('.safetensors') for x in required),'all model shard pins')
    support.verify_files(root,source,deadline); support.verify_files(model_path,manifest['files'],deadline)
    need({k:v for k,v in os.environ.items() if k.startswith('MLX_')}==inherited['execution']['environment'],'original MLX environment')
    sys.path.insert(0,str(root.parent.parent))
    import mlx.core as mx
    import tutorial_code.miniengine as package
    from tutorial_code.miniengine import cache, context
    from tutorial_code.miniengine.model import gdn
    from tutorial_code.miniengine.engine.llm_engine import LLMEngine
    from mlx.utils import tree_flatten
    need(Path(package.__file__).resolve().parent==root and Path(gdn.__file__).resolve()==root/'model/gdn.py'
         and Path(cache.__file__).resolve()==root/'cache.py' and Path(context.__file__).resolve()==root/'context.py', 'actual loaded root')
    result['device_run']=True
    result['device']=dict(name=mx.device_info().get('device_name'),architecture=mx.metal.device_info().get('architecture'),
                          mlx_version=importlib.metadata.version('mlx'))
    need(result['device']==dict(name='Apple M5 Ultra',architecture='applegpu_g17d',mlx_version='0.32.3')
         and mx.default_device()==mx.gpu,'actual fixed device/runtime')
    def memory(name):
        deadline(); mx.synchronize(); emit(dict(event='memory',phase=name,active_bytes=mx.get_active_memory(),
            peak_bytes=mx.get_peak_memory(),cache_bytes=mx.get_cache_memory(),not_performance=True))
    engine=model=layer=logits=None; layers={}; records=[]; batch=None; caches=[]; handles={}
    cmp=support.Comparator(mx,out)
    try:
        memory('before-load')
        engine=LLMEngine(str(model_path),max_len=2048,max_seqs=8,max_batched_tokens=16384,
                         cache_limit_gb=2,dtype=mx.bfloat16,compile_decode=False,fuse_decode=False)
        model=engine.model
        need(engine.runner.spec is None and len(model.layers)==64,'original full target-only model')
        indices=[i for i,layer in enumerate(model.layers) if not layer.is_attn]
        need(len(indices)==48,'actual48 GDN layers')
        for i in indices:
            model.layers[i].linear_attn.set_gates_compile(True); model.layers[i].linear_attn.set_norm_reuse(False)
        layers={indices[n]:model.layers[indices[n]].linear_attn for n in (0,24,47)}
        result['layer_indices']=list(layers)
        def layout(snapshot,B,output=False):
            expected={'root/'+('output' if output else 'x'):((B,5120),mx.bfloat16),
                      'root/conv':((B,3,10240),mx.bfloat16),'root/S':((B,48,128,128),mx.float32)}
            need(set(snapshot['arrays'])==set(expected) and all((snapshot['arrays'][k].shape,snapshot['arrays'][k].dtype)==v
                 for k,v in expected.items()),'complete original-dtype real layer layout')
        for f in fixtures['fixtures']: cmp.meta('fixture EOS',sorted(f['eos_token_ids']),sorted(engine.eos_ids))
        def parameters(name):
            values={str(i):h.parameter_hashes(mx,layer,tree_flatten,deadline) for i,layer in layers.items()}
            path=name+'.json'; h.write(out/path,values)
            result[name]=dict(path=path,sha256=h.sha(out/path,deadline),scope='all parameters of the three selected layers')
            return {i:h.parameter_values(v) for i,v in values.items()}
        parameters_before=parameters('parameters-before')
        for case in choose_cases(fixtures['fixtures']):
            case_id,B,start=case['case_id'],case['batch_size'],case['start'];selected=case['fixtures']
            prefix='captures/'+case_id;lengths=[len(f['prompt_ids']) for f in selected]
            row_ids=[f['fixture_id'] for f in selected];first=[];pending=[];prefill=[]
            h.write(out/(case_id+'-fixtures.json'),selected)
            for row,fixture in enumerate(selected):
                deadline();caches=cache.make_cache(model)
                logits=model(mx.array(fixture['prompt_ids'],dtype=mx.int32),context.ForwardContext('prefill',caches),last_only=True)
                mx.eval(logits,*support.snapshot(caches)['arrays'].values())
                need(logits.shape in ((1,248320),(248320,)) and logits.dtype==mx.bfloat16,'full-vocab original prefill output')
                first.append(int(mx.argmax(logits,axis=-1).item()))
                prefill.append(h.dump(mx,out,f'{prefix}/prefill-row-{row}/full-logits.safetensors',support.snapshot(logits),deadline))
                pending.append((fixture['fixture_id'],caches,len(fixture['prompt_ids'])+512))
                caches=[]
            batch=cache.DecodeBatch(model);batch.rebuild([],pending);mx.eval(*batch.arrays());pending.clear()
            cmp.meta('actual batch row IDs',batch.req_ids,row_ids);cmp.meta('actual initial batch lengths',batch.lens,lengths)
            emit(dict(event='reference-prefill',case_id=case_id,row_ids=row_ids,first_tokens=first,reference_full_logits=prefill))
            with DecodeCapture(layers,gdn,support.snapshot) as capture:
                def advance(input_tokens,step,active):
                    deadline();capture.begin(step,active);positions=[n+step-1 for n in lengths];used=[p+1 for p in positions]
                    cmp.meta('actual batch positions',batch.lens,positions);cmp.meta('stable batch rows',batch.req_ids,row_ids)
                    ctx=context.ForwardContext('decode',batch.layers,positions=mx.array(positions,dtype=mx.int32),
                        lens=mx.array(used,dtype=mx.int32),max_len=max(used))
                    full_logits=model(mx.array(input_tokens,dtype=mx.int32),ctx)
                    mx.eval(full_logits,*batch.arrays());batch.lens=used
                    need(full_logits.shape==(B,248320) and full_logits.dtype==mx.bfloat16,'actual full-vocab native batched logits')
                    sampled=mx.argmax(full_logits,axis=-1).tolist();observed=capture.finish();saved={}
                    for i,row in observed.items():
                        base=f'{prefix}/step-{step}/layer-{i}'
                        layout(row['input'],B);layout(row['output'],B,True)
                        saved[i]=dict(layer=i,batch_size=B,row_ids=row_ids,decode_step=step,positions=positions,used=used,input_tokens=input_tokens,
                            reference_input=h.dump(mx,out,base+'/reference-input.safetensors',row['input'],deadline),
                            reference_output=h.dump(mx,out,base+'/reference-output.safetensors',row['output'],deadline))
                    if active:
                        receipt=h.dump(mx,out,f'{prefix}/step-{step}/reference-full-logits.safetensors',support.snapshot(full_logits),deadline)
                        emit(dict(event='reference-capture',case_id=case_id,row_ids=row_ids,batch_size=B,decode_step=step,used=used,
                            reference_full_logits=receipt,layers=saved))
                    observed.clear();return sampled,saved
                raw,case_records=rollout_window(first,engine.eos_ids,512,start,advance,
                    lambda row:emit(dict(case_id=case_id,row_ids=row_ids,batch_size=B,**row)))
            records.append(dict(case_id=case_id,batch_size=B,row_ids=row_ids,start=start,prompt_tokens=lengths,
                raw_ids=raw,prefill_logits=prefill,steps=case_records))
            h.write(out/'captures.json',records);result['reference_chains']+=1
            batch.clear();batch=None;del logits,capture,advance,pending,prefill
            gc.collect();mx.synchronize();mx.clear_cache();memory('reference-chain-released')
        cmp.meta('reference selected parameters unchanged',parameters_before,parameters('parameters-after-reference'))
        # The original-model run is now idle. Three instance hooks exist only
        # during local replay; all state is loaded from saved original dtype.
        with ExitStack() as scopes:
            handles={i:scopes.enter_context(candidate.install(layer,gdn,enabled=True)) for i,layer in layers.items()}
            for case in records:
                for index,layer in layers.items():
                    chain=[step[index] for step in case['steps']]
                    replay_calls=0
                    def initialize(first):
                        arrays=mx.load(str(out/first['reference_input']['path'])); mx.eval(*arrays.values())
                        return cache.BatchState(arrays['root/conv'],arrays['root/S'])
                    def input_at(record):
                        ref=record['reference_input']; path=out/ref['path']
                        need(h.sha(path,deadline)==ref['sha256'],'saved actual input unchanged')
                        arrays=mx.load(str(path)); x=mx.load(str(path))['root/x']; mx.eval(x,*arrays.values())
                        return x,dict(metadata=ref['metadata'],arrays=arrays)
                    def invoke(current_layer,x,state):
                        nonlocal replay_calls
                        before=dict(handles[index].calls)
                        value=checked_invoke(handles[index],current_layer,x,state,mx.eval)
                        emit(dict(event='candidate-dispatch',case_id=case['case_id'],batch_size=case['batch_size'],row_ids=case['row_ids'],layer=index,
                            decode_step=chain[replay_calls]['decode_step'],used=chain[replay_calls]['used'],
                            before=before,after=dict(handles[index].calls)))
                        replay_calls+=1
                        result['candidate_calls']+=1
                        return value
                    def compare(kind,ordinal,value,expected):
                        record=chain[ordinal]; tag=f"{case['case_id']}/layer-{index}/step-{record['decode_step']}"
                        if kind=='output':
                            output,state=value; actual=support.snapshot(dict(output=output,conv=state.conv,S=state.S))
                            layout(actual,case['batch_size'],True)
                            ref=record['reference_output']; path=out/ref['path']
                            need(h.sha(path,deadline)==ref['sha256'],'saved actual reference output unchanged')
                            arrays=mx.load(str(path)); mx.eval(*arrays.values())
                            saved=h.dump(mx,out,f'candidate/{tag}/output.safetensors',actual,deadline)
                            record['candidate_output']=saved
                            # Persist usable outputs even if the following strict comparator rejects.
                            h.write(out/'captures.json',records)
                            cmp.compare(tag+'/complete-output',dict(metadata=ref['metadata'],arrays=arrays),actual)
                            result['candidate_compared_calls']+=1
                        else:
                            # Retain actual original handles, not pre-call views:
                            # __setitem__ descriptor replacement is visible here.
                            actual=support.snapshot(value); layout(actual,case['batch_size'])
                            if kind=='input-before':
                                left=list(value.values()); right=list(expected['arrays'].values())
                                need(mx.get_array_buffer_size(left,right)==mx.get_array_buffer_size(left)+mx.get_array_buffer_size(right),
                                     'candidate input/state storage independent of raw reference')
                            cmp.compare(tag+'/'+kind,expected,actual)
                        emit(dict(event='comparison',kind=kind,case_id=case['case_id'],batch_size=case['batch_size'],row_ids=case['row_ids'],layer=index,
                                  decode_step=record['decode_step'],used=record['used'],checks=cmp.rows))
                        cmp.rows=[]
                    replay_chain(layer,chain,initialize=initialize,input_at=input_at,invoke=invoke,
                        observe=lambda x,st:dict(x=x,conv=st.conv,S=st.S),compare=compare)
                    result['candidate_chains']+=1
            result['dispatch_counters']={str(i):dict(hd.calls) for i,hd in handles.items()}
        result['instance_hooks_restored']=all('_decode' not in vars(layer) for layer in layers.values())
        need(result['instance_hooks_restored'] and result['reference_chains']==4 and result['candidate_calls']==result['candidate_compared_calls']==36 and result['candidate_chains']==12,
             'all36 batched real-layer calls/twelve independent-state chains, cleanup')
        cmp.meta('candidate selected parameters unchanged',parameters_before,parameters('parameters-after-candidate'))
        del parameters,initialize,input_at,invoke,compare,scopes
    finally:
        result['dispatch_counters']={str(i):dict(hd.calls) for i,hd in handles.items()}
        result['instance_hooks_restored']=bool(layers) and all('_decode' not in vars(item) for item in layers.values())
        if batch is not None: batch.clear()
        caches.clear(); layers.clear(); records.clear(); model=engine=layer=logits=None
        gc.collect(); mx.synchronize(); mx.clear_cache(); memory('after-release')
    support.verify_files(root,source,deadline); support.verify_files(model_path,manifest['files'],deadline)
    for key in ('source_manifest','model_manifest','fixtures'): support.pinned_json(inherited[key])
    validate(plan); result['source_model_pins_rechecked']=True


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True); parser.add_argument('--plan-sha256',required=True)
    parser.add_argument('--out',type=Path,required=True); args=parser.parse_args(argv)
    args.out.mkdir(parents=True,exist_ok=False); started=time.monotonic(); previous={}
    result=dict(schema='GDN-batched-prework-real-layer-result-v1',status='INCOMPLETE',device_run=False,
        contract=fixed_contract(),pid=os.getpid(),ppid=os.getppid(),plan_sha256=args.plan_sha256,
        reference_chains=0,candidate_chains=0,candidate_calls=0,candidate_compared_calls=0,canary_qualified=False,
        full_model_qualified=False,http_qualified=False,performance_qualified=False,
        instance_hooks_restored=False,source_model_pins_rechecked=False)
    def write(name,value):
        (args.out/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    try:
        need(sha(args.plan)==args.plan_sha256,'exact child plan SHA')
        plan=json.loads(args.plan.read_bytes()); write('plan.json',plan)
        h,old,support,candidate,original,inherited=validate(plan)
        write('inherited-e024-plan.json',original); write('inherited-e023-plan.json',inherited)
        result['sources']={str(Path(__file__).resolve()):plan['qualifier_sha256'],str(HERE/'candidate.py'):CANDIDATE_SHA,str(BASE):BASE_SHA,
            str(HELPER):HELPER_SHA,str(OLD):OLD_SHA,str(CORE_PINS):old.MODEL_SOURCE_SHA,
            original['e023_qualifier']['path']:old.SUPPORT_SHA}
        need(signal.getitimer(signal.ITIMER_REAL)==(0.0,0.0),'existing alarm')
        def deadline():
            if time.monotonic()-started>=plan['deadline_s']: raise TimeoutError('batched real-layer canary deadline')
        def stop(signum,frame):
            if signum==signal.SIGALRM: raise TimeoutError('batched real-layer canary deadline')
            raise InterruptedError('batched real-layer canary cancellation '+signal.Signals(signum).name)
        previous={s:signal.getsignal(s) for s in (signal.SIGALRM,signal.SIGINT,signal.SIGTERM)}
        for sig in previous: signal.signal(sig,stop)
        signal.setitimer(signal.ITIMER_REAL,plan['deadline_s'])
        with (args.out/'events.jsonl').open('x') as stream:
            def emit(row):
                stream.write(json.dumps(dict(**row,elapsed_wall_s=time.monotonic()-started),allow_nan=False)+'\n'); stream.flush()
            device(plan,h,old,support,candidate,original,inherited,args.out,deadline,emit,result)
        need(sha(args.plan)==args.plan_sha256,'plan changed')
        result.update(status='PASS',canary_qualified=True)
    except IncompleteCoverage as exc:
        result.update(status='INCOMPLETE',error=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()))
    except BaseException as exc:
        result.update(status='FAIL',error=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()))
    finally:
        if previous:
            signal.setitimer(signal.ITIMER_REAL,0)
            for sig,value in previous.items(): signal.signal(sig,value)
        result['elapsed_wall_s']=time.monotonic()-started
        write('complete.json',result)
    print(json.dumps(dict(status=result['status'],canary_qualified=result['canary_qualified'],candidate_calls=result['candidate_calls'])),flush=True)
    return 0 if result['status']=='PASS' else 1


if __name__=='__main__':raise SystemExit(main())
