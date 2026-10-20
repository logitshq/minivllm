"""Default-OFF actual-normalized-activation MLP projection canary; no timing.

Only an externally owned child may --run. The shared outputs are shadows;
every decoder MLP consumes original M1 projection outputs, including at probes.
"""
import argparse
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
from types import SimpleNamespace

HERE=Path(__file__).resolve().parent; B=HERE.parent
SCREEN=B/'rowwise_native_verify_cost_v1/screen.py'
SCREEN_SHA='679ad62a8b2973c9ea6d767da3135782345ca09c870ff071ff1494005b914d4d'
CANDIDATE=HERE/'candidate.py'
CANDIDATE_SHA='02200afa33fd6daa3f0f4dac3436b717616cf620f4b6cd7e59f1d8f263658950'
SHARED=B/'nvfp4_multirow_shared/candidate.py'
SHARED_SHA='d1826b3c4d07d44adbae071ef2da43197348c09b2d8be1035eb1a0c2788d132e'
E054_PLAN_SHA='3d10d60dc72eff7e2c70444239c61be09c865b6e15109c7f43b8cdf8a6a66c2c'
E054_COMPLETE_SHA='b719a1c25257e0b0fc93e80a8d49215e223d02f708ab5f879558ffe94968e9de'
E054_AUDIT_SHA='7fe77b62600f36b27823b68b80e54ee32815e9b830178870679540b5e3835972'
E054_PARENT_SHA='f613b5df1494e9064ba84e2dd025b437177891b00bf5d98bcf18d03e5eb960ce'
LAYERS=(0,32,63); WIDTHS=(4,8)


def need(ok,message):
    if not ok: raise ValueError(message)


def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4<<20),b''):value.update(block)
    return value.hexdigest()


def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def load(path,name,digest):
    need(sha(path)==digest,'sealed source '+str(path))
    spec=importlib.util.spec_from_file_location(name,path); module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module


def pinned(ref,expected):
    need(set(ref)=={'path','sha256'} and Path(ref['path']).is_absolute() and ref['sha256']==expected==sha(ref['path']),'fixed evidence ref')
    return json.loads(Path(ref['path']).read_bytes())


def contract():
    return dict(schema='C1-MLP-shared-activation-contract-v1',enabled_by_default=False,used=1023,widths=list(WIDTHS),layers=list(LAYERS),
        projection_pairs=12,input_files=6,output_files=24,raw_files=30,provenance_pairs=4,provenance_tensors_per_pair=129,callback_count=256,native_m1_projection_calls=1536,shadow_projection_calls=12,
        returned_projection_values='original native M1 only; shared shadow output never returned',
        inputs='actual per-row post_attention_layernorm from native-returning teacher-forced model traversal',
        provenance='T4/T8 full logits and complete selected k=T state versus original verifier and native causal oracle',
        compiled_qualified=False,full_model_qualified=False,http_qualified=False,performance_qualified=False,engine_pareto_point=False)


def validate(plan):
    need(set(plan)=={'schema','canary_sha256','e054_plan','e054_result','e054_audit','deadline_s'}
         and plan['schema']=='C1-MLP-shared-activation-plan-v1','exact canary plan')
    need(plan['canary_sha256']==sha(__file__),'canary source SHA')
    need(type(plan['deadline_s']) in (int,float) and math.isfinite(plan['deadline_s']) and 0<plan['deadline_s']<=1800,'bounded canary deadline')
    old_plan=pinned(plan['e054_plan'],E054_PLAN_SHA); previous=pinned(plan['e054_result'],E054_COMPLETE_SHA)
    audit=pinned(plan['e054_audit'],E054_AUDIT_SHA)
    need(previous['schema']=='C1-verifier-cost-result-v1' and previous['status']=='PASS_LOCAL_COST_ONLY'
         and previous['plan_sha256']==E054_PLAN_SHA,'actual completed cost prerequisite')
    need(audit['schema']=='E054-local-cost-independent-v2' and audit['status']=='PASS' and audit['local_cost_audited'] is True
         and audit['complete_sha256']==E054_COMPLETE_SHA and audit['parent_plan_sha256']==E054_PARENT_SHA,'independent E054 prerequisite')
    screen=load(SCREEN,'_mlp_canary_cost_helpers',SCREEN_SHA)
    q,deps,seed=screen.validate(old_plan)
    need(seed==previous['seed'] and seed['used']==1023 and seed['position']==1022,'actual E042 canonical seed')
    candidate=load(CANDIDATE,'_mlp_canary_candidate',CANDIDATE_SHA)
    shared=load(SHARED,'_mlp_canary_shared_kernel',SHARED_SHA)
    return screen,q,deps,seed,candidate,shared,old_plan


class NativeReturningProbe:
    """One ordered gate/up traversal. All outputs returned are native M1 values."""
    def __init__(self,model,mx,width,shared,emit):
        need(width in WIDTHS and len(model.layers)==64 and callable(shared) and callable(emit),'fixed probe domain')
        self.mx,self.width,self.shared,self.emit=mx,width,shared,emit
        self.ops=[(i,name,getattr(layer.mlp,name+'_proj')) for i,layer in enumerate(model.layers) for name in ('gate','up')]
        need(len({(id(op.weight),id(op.scales)) for _,_,op in self.ops})==128,'distinct actual live projection sources')
        self.cursor=0;self.shadow_calls=0;self.native_calls=0;self.input=None

    def __call__(self,x,weight,scales):
        need(self.cursor<len(self.ops),'extra projection callback')
        layer,name,op=self.ops[self.cursor]
        need(weight is op.weight and scales is op.scales,'ordered actual projection source identity')
        need(x.shape==(self.width,5120) and x.dtype==self.mx.bfloat16,'actual normalized row shape/dtype')
        if name=='gate':self.input=x
        else:need(x is self.input,'same actual gate/up normalized array')
        # A native Linear call for EACH row: never native M=T QMM or a custom B1 oracle.
        native=self.mx.concatenate([op(x[t:t+1]) for t in range(self.width)],axis=0)
        need(native.shape==(self.width,17408) and native.dtype==self.mx.bfloat16,'complete original M1 projection outputs')
        self.native_calls+=self.width;self.cursor+=1
        if layer in LAYERS:
            shadow=self.shared(x,weight,scales)
            need(shadow is not None and shadow.shape==native.shape and shadow.dtype==native.dtype,'explicit shadow shared projection rejected')
            self.shadow_calls+=1;self.emit(self.width,layer,name,x,native,shadow)
        if name=='up':self.input=None
        return native

    def finish(self):
        need(self.cursor==128 and self.native_calls==128*self.width and self.shadow_calls==6 and self.input is None,'complete native-returning traversal')
        return dict(T=self.width,callback_count=self.cursor,native_m1_projection_calls=self.native_calls,
                    shadow_projection_calls=self.shadow_calls,shared_outputs_returned=0,status='PASS')



def compare_provenance(support,bridge,cmp,width,layout,oracle,baseline,output):
    """Full logits + selected complete cache only; receipts are not raw state dumps."""
    need(width in WIDTHS and len(baseline.prefixes)==len(output.prefixes)==width
         and all(len(v)==64 for v in (oracle['selected'][width],baseline.prefixes[-1],output.prefixes[-1])),
         'complete selected provenance states')
    def snapshot(logits,layers):
        return support.snapshot(dict(logits=logits,state=bridge.canonical_state(layers,layout)))
    actual=snapshot(output.logits,output.prefixes[-1]);labels=[]
    for arm,reference in (
        ('native-causal',snapshot(oracle['logits'][:width],oracle['selected'][width])),
        ('frozen-verifier',snapshot(baseline.logits,baseline.prefixes[-1]))):
        label=f'T{width}/provenance/{arm}';cmp.compare(label,reference,actual);labels.append(label)
    return dict(T=width,k=width,status='PASS',labels=labels,full_logits=True,selected_state_arrays=128,
                raw_state_saved=False,shared_outputs_returned=0)


def device(plan,validated,out,deadline,result):
    screen,q,deps,seed,candidate,shared,old_plan=validated
    bridge,frozen,h,old,support,original,inherited=deps
    source=support.pinned_json(inherited['source_manifest'])['files'];manifest=support.pinned_json(inherited['model_manifest'])
    fixtures=support.pinned_json(inherited['fixtures']);root=Path(inherited['package_root']).resolve()/'tutorial_code/miniengine'
    model_path=Path(inherited['model']).resolve()
    need(source==json.loads(bridge.CORE_PINS.read_bytes())['files'] and set(source)=={str(p.relative_to(root)) for p in root.rglob('*.py')},'frozen complete original model source')
    need(Path(manifest['model']).resolve()==model_path and manifest['revision']==model_path.name
         and Path(fixtures['model']).resolve()==model_path,'original model and fixtures')
    required={'config.json','generation_config.json','tokenizer.json','tokenizer_config.json'}|{str(p.relative_to(model_path)) for p in model_path.rglob('*.safetensors')}
    need(required<=set(manifest['files']) and any(p.endswith('.safetensors') for p in required),'all original payload pins')
    support.verify_files(root,source,deadline);support.verify_files(model_path,manifest['files'],deadline)
    need({k:v for k,v in os.environ.items() if k.startswith('MLX_')}==inherited['execution']['environment'],'original MLX environment')
    canonical=Path(old_plan['e042_result']['path']).parent/seed['canonical']['path']
    need(h.sha(canonical,deadline)==seed['canonical']['sha256'],'actual canonical bytes')
    sys.path.insert(0,str(root.parent.parent))
    import mlx.core as mx
    import mlx.nn as nn
    import tutorial_code.miniengine as package
    from tutorial_code.miniengine import cache,context
    from tutorial_code.miniengine.engine.llm_engine import LLMEngine
    from mlx.utils import tree_flatten
    need(Path(package.__file__).resolve().parent==root,'actual frozen package')
    result['device']=dict(name=mx.device_info().get('device_name'),architecture=mx.device_info().get('architecture'),mlx_version=importlib.metadata.version('mlx'))
    need(result['device']==dict(name='Apple M5 Ultra',architecture='applegpu_g17d',mlx_version='0.32.3') and mx.default_device()==mx.gpu,'fixed device/runtime')
    runtime=SimpleNamespace(mx=mx,silu=nn.silu,BatchKV=cache.BatchKV,BatchState=cache.BatchState,ForwardContext=context.ForwardContext,DecodeBatch=cache.DecodeBatch)
    engine=model=oracle=project=layer=probe=output=baseline=baseline_ctx=ctx=data=native=shadow=None;storage={};cmp=support.Comparator(mx,out)
    try:
        engine=LLMEngine(str(model_path),max_len=2048,max_seqs=1,max_batched_tokens=16384,cache_limit_gb=2,
                         dtype=mx.bfloat16,compile_decode=True,fuse_decode=True)
        model=engine.model;layout=tuple(layer.is_attn for layer in model.layers)
        need(len(layout)==64 and sum(layout)==16 and engine.runner.spec is None and engine.runner._ahead is None,'fresh original target-only model')
        need(runtime.silu is type(model.layers[0].mlp).__call__.__globals__['nn'].silu,'original model SiLU function')
        for layer in model.layers:
            if not layer.is_attn:layer.linear_attn.set_gates_compile(True);layer.linear_attn.set_norm_reuse(False)
        need(seed['fixture_id']==bridge.selected(fixtures['fixtures'])[0][0]['fixture_id'],'original shortest fixture')
        need(all(sorted(f['eos_token_ids'])==sorted(engine.eos_ids) for f in fixtures['fixtures']),'normal original EOS')
        def parameters(name):
            value=h.parameter_hashes(mx,model,tree_flatten,deadline);write(out/(name+'.json'),value);return h.parameter_values(value)
        before=parameters('parameters-before')
        project=shared.build(mx,enabled=True);need(project is not None,'explicit frozen shared kernel admitted')
        for arm in ('canonical','oracle','T4','T8','frozen'):
            storage[arm]=mx.load(str(canonical));mx.eval(*storage[arm].values())
        need(storage['canonical']['root/positions'].tolist()==[1022] and storage['canonical']['root/lens'].tolist()==[1023],'original seed position/lens')
        oracle=screen.native(runtime,engine,h,layout,storage['oracle'],1022,9,True,True)
        need(oracle['inputs'][:8]==seed['tokens'] and not (set(oracle['inputs'])|set(oracle['ids']))&set(engine.eos_ids),'native causal teacher and normal EOS')
        result['teacher']=dict(inputs=oracle['inputs'],output_ids=oracle['ids'],native_steps=9)
        # Retain only k=T4/T8 states for provenance, not all nine native versions.
        oracle['selected']={width:oracle['prefixes'][width-1] for width in WIDTHS}
        oracle['prefixes'].clear();oracle['holder'].layers.clear()
        captured={}
        def emit(width,index,name,x,native,shadow):
            deadline();tag=f'T{width}/layer{index}';key=(width,index)
            if name=='gate':
                need(key not in captured,'fresh actual activation capture')
                row=dict(T=width,layer=index,input=h.dump(mx,out,tag+'/input.safetensors',support.snapshot(dict(x=x)),deadline),projections=[])
                captured[key]=row;result['captures'].append(row)
            else:need(key in captured,'gate capture precedes up')
            row=captured[key];snapshots=[support.snapshot(dict(output=v)) for v in (native,shadow)]
            files=[h.dump(mx,out,tag+'/'+name+'-'+arm+'.safetensors',snap,deadline) for arm,snap in zip(('native','shadow'),snapshots)]
            result['raw_files']=sum(1+2*len(row['projections']) for row in result['captures'])+2
            comparison=dict(name=name,status='PENDING',files=files);row['projections'].append(comparison);write(out/'complete.json',result)
            cmp.compare(tag+'/'+name,*snapshots);comparison['status']='PASS'
            result['projection_pairs']+=1;write(out/'complete.json',result)
        for width in WIDTHS:
            deadline();probe=NativeReturningProbe(model,mx,width,project,emit)
            ctx=h.context_from_input(storage['T'+str(width)],layout,runtime,1023)
            output=candidate.verify(model,mx.array(oracle['inputs'][:width],dtype=mx.int32),ctx,runtime,
                                    enabled=True,mlp_input_project=probe,head_project=None)
            need(output is not None,'new layer-loop admission')
            q.prefix_ownership(output,ctx,layout,width)
            baseline_ctx=h.context_from_input(storage['frozen'],layout,runtime,1023)
            baseline=frozen.verify(model,mx.array(oracle['inputs'][:width],dtype=mx.int32),baseline_ctx,runtime,enabled=True,head_project=None)
            need(baseline is not None,'frozen original verifier admission')
            q.prefix_ownership(baseline,baseline_ctx,layout,width)
            mx.eval(output.logits,baseline.logits,*screen.arrays(output.prefixes[-1]),*screen.arrays(baseline.prefixes[-1]));mx.synchronize()
            result['provenance'].append(compare_provenance(support,bridge,cmp,width,layout,oracle,baseline,output))
            result['traversals'].append(probe.finish());result['checks']=cmp.rows;write(out/'complete.json',result)
            output=probe=baseline=baseline_ctx=ctx=None
        for arm,data in storage.items():cmp.compare('canonical-preserved/'+arm,support.snapshot(storage['canonical']),support.snapshot(data))
        cmp.meta('all original parameter values unchanged',before,parameters('parameters-after'))
        need(engine.runner.decode_model._functions=={} and engine.runner.decode_model._packed is None and engine.runner._ahead is None,'original uncompiled used>256 path')
        need([(r['T'],r['layer']) for r in result['captures']]==[(w,i) for w in WIDTHS for i in LAYERS]
             and result['projection_pairs']==12 and [p['T'] for p in result['provenance']]==list(WIDTHS) and all([p['name'] for p in r['projections']]==['gate','up'] for r in result['captures']),'all selected activation/projection captures')
        result.update(checks=cmp.rows,raw_files=30,callback_count=sum(t['callback_count'] for t in result['traversals']),
            native_m1_projection_calls=sum(t['native_m1_projection_calls'] for t in result['traversals']),
            shadow_projection_calls=sum(t['shadow_projection_calls'] for t in result['traversals']),shared_outputs_returned=0,
            projection_canary_complete=True,native_returning_provenance_complete=True)
        need(len(list(out.rglob('*.safetensors')))==30,'exact30 small raw capture files')
    except BaseException as error:
        traceback.clear_frames(error.__traceback__);raise
    finally:
        storage.clear();model=oracle=project=layer=probe=output=baseline=baseline_ctx=ctx=data=native=shadow=None
        if engine is not None:engine.runner.batch.clear();engine.runner.staging.clear();engine.runner.feats.clear();engine.runner._ahead=None
        engine=None;gc.collect();mx.synchronize();mx.clear_cache();result['device_cleanup_completed']=True
        result['memory_after_cleanup']=dict(active_bytes=mx.get_active_memory(),peak_bytes=mx.get_peak_memory(),cache_bytes=mx.get_cache_memory())
    support.verify_files(root,source,deadline);support.verify_files(model_path,manifest['files'],deadline)
    validate(plan);result['source_model_pins_rechecked']=True


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',action='store_true');p.add_argument('--plan',type=Path)
    p.add_argument('--plan-sha256');p.add_argument('--out',type=Path);a=p.parse_args(argv)
    if not a.run:
        if any(v is not None for v in (a.plan,a.plan_sha256,a.out)):p.error('explicit --run required')
        print(json.dumps(contract(),indent=2));return 0
    need(all(v is not None for v in (a.plan,a.plan_sha256,a.out)),'explicit complete run arguments')
    a.out.mkdir(parents=True,exist_ok=False);started=time.monotonic();previous={}
    result=dict(schema='C1-MLP-shared-activation-canary-result-v1',status='INCOMPLETE',pid=os.getpid(),ppid=os.getppid(),plan_sha256=a.plan_sha256,
                contract=contract(),captures=[],traversals=[],provenance=[],native_returning_provenance_complete=False,projection_pairs=0,raw_files=0,projection_canary_complete=False,
                device_cleanup_completed=False,source_model_pins_rechecked=False,full_model_qualified=False,http_qualified=False,performance_qualified=False,engine_pareto_point=False)
    try:
        need(sha(a.plan)==a.plan_sha256,'plan SHA');plan=json.loads(a.plan.read_bytes());write(a.out/'plan.json',plan)
        validated=validate(plan);result['seed']=validated[3]
        result['sources']={str(Path(__file__).resolve()):plan['canary_sha256'],str(SCREEN):SCREEN_SHA,str(CANDIDATE):CANDIDATE_SHA,str(SHARED):SHARED_SHA}
        write(a.out/'complete.json',result)
        def deadline():
            if time.monotonic()-started>=plan['deadline_s']:raise TimeoutError('canary child deadline')
        def stop(signum,frame):raise InterruptedError('bounded canary signal '+str(signum))
        need(signal.getitimer(signal.ITIMER_REAL)==(0.0,0.0),'existing alarm')
        previous={s:signal.getsignal(s) for s in (signal.SIGALRM,signal.SIGINT,signal.SIGTERM)}
        for s in previous:signal.signal(s,stop)
        signal.setitimer(signal.ITIMER_REAL,plan['deadline_s'])
        device(plan,validated,a.out,deadline,result);deadline()
        need(result['projection_canary_complete'] and result['native_returning_provenance_complete'] and result['device_cleanup_completed'] and result['source_model_pins_rechecked'],'complete canary+cleanup')
        need(sha(a.plan)==a.plan_sha256,'unchanged plan');result['status']='PASS_PROJECTION_CANARY_ONLY'
    except BaseException as error:
        result.update(status='FAIL',error=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc()))
    finally:
        if previous:
            signal.setitimer(signal.ITIMER_REAL,0)
            for s,value in previous.items():signal.signal(s,value)
        result['elapsed_wall_s']=time.monotonic()-started;write(a.out/'complete.json',result)
    print(json.dumps({k:result[k] for k in ('status','projection_pairs','raw_files')}))
    return 0 if result['status']=='PASS_PROJECTION_CANARY_ONLY' else 1


if __name__=='__main__':raise SystemExit(main())
