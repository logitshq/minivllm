"""E024 forward-only go/no-go, never HTTP latency or a Pareto point.

CPU-safe CLI --plan PATH --plan-sha256 SHA --out NEW. Exact five-field plan:
E024-route-timing-plan-v1/schema, qualifier_sha256, candidate {path,sha256},
e024_plan {path,sha256}, deadline_s<=1800. Requires separately qualified matrix.
Four B1/B8, used1023/1024 points; at each point OFF then ON. All ON outputs must
equal the OFF eager oracle. Two warmups per arm, then three ABBA repetitions.
Inputs are copied/materialized before timing. Forward + all128 states/logits
eval + synchronize are timed; comparison, hashes, observers and teardown are not.
Every sample owns its output cleanup, and cleanup is reported separately.
"""
import argparse
import gc
import hashlib
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
MATRIX_SHA='6130d948450c825302d9857450204f8503c0dab449050a1bb59eaa51b77920e3'
POINTS=tuple((B,used) for used in (1023,1024) for B in (1,8))


def matrix():
    path=HERE/'qualifier_matrix.py'
    if hashlib.sha256(path.read_bytes()).hexdigest()!=MATRIX_SHA: raise ValueError('fixed matrix support changed')
    spec=importlib.util.spec_from_file_location('_e024_forward_matrix_support',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def fixed_plan():
    return dict(schema='E024-route-timing-contract-v1',points=[dict(B=B,used=u) for B,u in POINTS],
        gate_order=['off','on'],order='each point OFF then ON; all ON outputs exact against OFF eager',
        warm_arms=['A','C','C','A'],formal_arms=list('ABBA'.replace('B','C'))*3,
        expected_configs=8,expected_warm_samples=32,expected_formal_samples=96,
        original_dtype=True,output_arrays=129,input_arrays=131,KV_backing=1536,
        prefill_checkpoints=[256,1022,1023],
        warm_excluded=True,compile_excluded=True,observer_off_timing=True,cache_limit_gib=2,
        sample_prepare='independent identity-gather cache copy, evaluate and synchronize outside timer',
        sample_timer='forward + evaluate logits and all128 state arrays + synchronize',
        sample_verify='all129 finite original unsigned bits versus the qualified oracle, outside timer',
        sample_cleanup='drop sample-owned arrays then gc/synchronize; recorded separately, no cache clearing',
        hashes='full original bytes for canonical and each untimed qualification arm input-before/input-after/output; parameters before/after',
        raw='first failure available tensors only; observed failure input handles are not historical before payloads',
        metrics=['forward_ms','teardown_ms','owned_lifecycle_ms','diagnostic_tokens_per_second'],
        lifecycle_semantics='forward_ms + teardown_ms; excludes intervening verification, not contiguous request latency',
        rate_semantics='B / forward_seconds, a row-equivalent rate without sampling or HTTP',
        no_speed_acceptance_threshold=True,full_model_qualified=False,performance_qualified=False,
        http_qualified=False,pareto_eligible=False)


def validate(plan):
    m=matrix();q=m.helper();h=q.helpers();need=h.need
    need(set(plan)=={'schema','qualifier_sha256','candidate','e024_plan','deadline_s'}
         and plan['schema']=='E024-route-timing-plan-v1','exact forward timing plan')
    need(plan['qualifier_sha256']==h.sha(__file__),'timing source SHA')
    need(type(plan['deadline_s']) in (int,float) and math.isfinite(plan['deadline_s']) and 0<plan['deadline_s']<=1800,'bounded deadline')
    need(plan['candidate']==dict(path=str(HERE/'candidate.py'),sha256=m.CANDIDATE_SHA),'fixed candidate')
    ref=plan['e024_plan'];need(set(ref)=={'path','sha256'} and Path(ref['path']).is_absolute()
        and ref['sha256']==q.OLD_PLAN_SHA==h.sha(ref['path']),'original E024 plan')
    old=h.load(q.OLD/'qualify.py','_e024_timing_old',q.OLD_SHA)
    original=json.loads(Path(ref['path']).read_text());old.validate_plan(original)
    support=old.load_pinned(original['e023_qualifier'],'_e024_timing_native',old.SUPPORT_SHA)
    inherited=support.pinned_json(original['e023_plan'])
    candidate=h.load(HERE/'candidate.py','_e024_timing_candidate',m.CANDIDATE_SHA)
    need(h.sha(candidate.ORIGINAL)==candidate.ORIGINAL_SHA==old.CANDIDATE_SHA,'original candidate support')
    return m,q,h,old,support,candidate,original,inherited


def set_gates(model,enabled):
    for layer in model.layers:
        if not layer.is_attn:
            layer.linear_attn.set_gates_compile(enabled)
            layer.linear_attn.set_norm_reuse(False)


def witnessed_call(graph,ids,ctx):
    """Actual callable invocation only during untimed numerical qualification.

    Restore the raw compiler callables before timing, including on exceptions.
    No monkeypatch of MLX, production modules or the frozen candidate.
    """
    raw=dict(graph._functions);calls=[]
    class Witness(dict):
        def __setitem__(self,key,fn):
            raw[key]=fn
            def invoke(*args): calls.append(key);return fn(*args)
            super().__setitem__(key,invoke)
    wrapped=Witness()
    for key,fn in raw.items(): wrapped[key]=fn
    graph._functions=wrapped
    try:
        if not graph._eligible(ids,ctx): raise ValueError('expected graph route, fallback rejected')
        value=graph(ids,ctx)
        if len(calls)!=1: raise ValueError('exact one actual compiled invocation required')
        return value,dict(invocations=1,key=repr(calls[0]),bound=calls[0][0])
    finally: graph._functions=raw


def timed_forward(mx,forward,ids,ctx,clock=time.perf_counter_ns):
    """Only the declared numerical product; caller owns copying/checks/release."""
    begin=clock()
    logits=forward(ids,ctx)
    states=[a for cache in ctx.caches for a in
            ((cache.K,cache.V) if hasattr(cache,'K') else (cache.conv,cache.S))]
    mx.eval(logits,*states);mx.synchronize()
    end=clock()
    if end<=begin: raise ValueError('nonpositive forward interval')
    return logits,(end-begin)/1e6


def device(plan,m,q,h,old,support,candidate,original,inherited,out,deadline,emit,result):
    need,write=h.need,h.write
    source=support.pinned_json(inherited['source_manifest'])['files'];model=support.pinned_json(inherited['model_manifest'])
    fixtures=support.pinned_json(inherited['fixtures']);root=Path(inherited['package_root']).resolve()/'tutorial_code/miniengine'
    model_path=Path(inherited['model']).resolve()
    need(model['schema']=='E023-model-v1' and Path(model['model']).resolve()==model_path
         and model['revision']==model_path.name and Path(fixtures['model']).resolve()==model_path,'model/fixture identity')
    need(set(source)=={str(p.relative_to(root)) for p in root.rglob('*.py')}
         and source==json.loads(candidate.PINS.read_text())['files'] and h.sha(candidate.PINS)==old.MODEL_SOURCE_SHA,'fixed core')
    required={'config.json','generation_config.json'}|{str(p.relative_to(model_path)) for p in model_path.rglob('*.safetensors')}
    need(required<=set(model['files']) and any(p.endswith('.safetensors') for p in required),'all model shards')
    support.verify_files(root,source,deadline);support.verify_files(model_path,model['files'],deadline)
    result['mlx_environment']={k:v for k,v in os.environ.items() if k.startswith('MLX_')}
    need(result['mlx_environment']==inherited['execution']['environment'],'exact clean MLX environment')
    sys.path.insert(0,str(root.parent.parent))
    import mlx.core as mx
    import importlib.metadata
    import tutorial_code.miniengine as package
    from tutorial_code.miniengine.cache import make_cache,DecodeBatch
    from tutorial_code.miniengine.context import ForwardContext
    from tutorial_code.miniengine.engine.llm_engine import LLMEngine
    from mlx.utils import tree_flatten
    need(Path(package.__file__).resolve().parent==root,'actual pinned package')
    result['device_run']=True
    result['device']=dict(name=mx.device_info().get('device_name'),architecture=mx.metal.device_info().get('architecture'),version=importlib.metadata.version('mlx'))
    need(result['device']==dict(name='Apple M5 Ultra',architecture='applegpu_g17d',version='0.32.3') and mx.default_device()==mx.gpu,'fixed runtime')
    engines=[];seeds=[];graph=None;cmp=support.Comparator(mx,out)
    def phase(name):
        deadline();mx.synchronize();emit(dict(event='memory',phase=name,active=mx.get_active_memory(),cache=mx.get_cache_memory(),peak=mx.get_peak_memory()))
    def params(index,when):
        rows=h.parameter_hashes(mx,engines[index].model,tree_flatten,deadline);path=f'parameters-{index}-{when}.json';write(out/path,rows)
        result['parameter_receipts'].append(dict(model=index,phase=when,path=path,sha256=h.sha(out/path,deadline)));return rows
    def fresh(canonical):
        arrays=m.clone_arrays(mx,canonical['arrays'])
        ctx=h.context_from_input(arrays,layout,graph._runtime,max(arrays['root/lens'].tolist()))
        return arrays,arrays['root/tokens'],ctx
    def failure_raw(label,canonical,input_snapshot,output,expected=None):
        receipt={};errors=[]
        for name,snap in [('canonical',canonical),('input-handles-at-failure',input_snapshot),('output',output),('expected-output',expected)]:
            if snap is None: continue
            try: receipt[name]=h.dump(mx,out,f'failure/{label}/{name}.safetensors',snap,deadline)
            except BaseException as exc: errors.append(dict(name=name,type=type(exc).__name__,message=str(exc)))
        emit(dict(event='failure-raw',label=label,raw=receipt,errors=errors,
                  input_scope='Original handles as observed at failure; historical before bytes are not claimed. Genuine before hashes are separate.',
                  canonical_scope='Independent prefill/rebuild storage, copied into each execution.'))
    def qualify(name,arm,canonical,oracle=None):
        arrays=ids=ctx=inputs=output=None;row=dict(event='qualification',name=name,arm=arm,status='RUNNING')
        try:
            deadline();arrays,ids,ctx=fresh(canonical)
            inputs=support.snapshot(dict(state=ctx.caches,tokens=ids,positions=ctx.positions,lens=ctx.lens))
            cmp.compare(name+'/'+arm+'/canonical',canonical,inputs)
            row['input_before']=m.hashes(mx,inputs,deadline)
            if arm=='A': logits=engines[0].model(ids,ctx);row['route']=dict(kind='eager',bound=ctx.max_len)
            else: logits,row['route']=witnessed_call(graph,ids,ctx)
            output=support.snapshot(dict(logits=logits,state=ctx.caches));need(len(output['arrays'])==129,'all129 outputs')
            row['input_after']=m.hashes(mx,inputs,deadline);cmp.meta(name+'/'+arm+'/input-unchanged',row['input_before'],row['input_after'])
            row['output']=m.hashes(mx,output,deadline)
            if oracle is not None: cmp.compare(name+'/'+arm+'/OFF-eager-oracle',oracle,output)
            row.update(status='PASS',checks=cmp.rows);return output,row['output']
        except BaseException as exc:
            row.update(status='FAIL',error=dict(type=type(exc).__name__,message=str(exc)),checks=cmp.rows)
            if not isinstance(exc,(InterruptedError,TimeoutError)): failure_raw(name+'-'+arm,canonical,inputs,output,oracle)
            raise
        finally: emit(row);cmp.rows=[]
    def sample(name,arm,index,warm,canonical,expected):
        arrays=ids=ctx=inputs=logits=output=None
        row=dict(event='sample',name=name,arm=arm,index=index,warm=warm,status='RUNNING')
        try:
            deadline();arrays,ids,ctx=fresh(canonical)
            inputs=support.snapshot(dict(state=ctx.caches,tokens=ids,positions=ctx.positions,lens=ctx.lens))
            cmp.compare(name+'/prepared-input',canonical,inputs);cmp.rows=[]
            need(type(graph._functions) is dict,'timed candidate must have no invocation observer')
            need(graph._eligible(ids,ctx),'registered sample must remain graph eligible')
            mx.synchronize()
            logits,row['forward_ms']=timed_forward(mx,engines[0].model if arm=='A' else graph,ids,ctx)
            row['diagnostic_tokens_per_second']=len(canonical['arrays']['root/tokens'])*1000/row['forward_ms']
            emit(dict(row,event='sample-measured'))
            output=support.snapshot(dict(logits=logits,state=ctx.caches))
            cmp.compare(name+'/sample-output',expected,output)
            cmp.compare(name+'/sample-input-unchanged',canonical,inputs)
            row.update(status='PASS',checks=cmp.rows);deadline()
        except BaseException as exc:
            row.update(status='FAIL',error=dict(type=type(exc).__name__,message=str(exc)),checks=cmp.rows)
            if not isinstance(exc,(InterruptedError,TimeoutError)): failure_raw(name+'-sample-'+str(index),canonical,inputs,output,expected)
            raise
        finally:
            primary_failed='error' in row;cleanup_error=None
            release=time.perf_counter_ns()
            try:
                arrays=ids=ctx=inputs=logits=output=None
                gc.collect();mx.synchronize()
            except BaseException as exc:
                cleanup_error=dict(type=type(exc).__name__,message=str(exc))
                row.update(status='FAIL',cleanup_error=cleanup_error)
            row['teardown_ms']=(time.perf_counter_ns()-release)/1e6
            if 'forward_ms' in row: row['owned_lifecycle_ms']=row['forward_ms']+row['teardown_ms']
            emit(row);result['samples'].append({k:v for k,v in row.items() if k!='checks'});cmp.rows=[]
            write(out/'progress.json',result)
            if cleanup_error is not None and not primary_failed:raise RuntimeError('sample cleanup failed: '+cleanup_error['message'])
    try:
        for label in ('reference','candidate'):
            phase('before-load-'+label)
            e=LLMEngine(str(model_path),max_len=2048,max_seqs=8,max_batched_tokens=16384,cache_limit_gb=2,
                        dtype=mx.bfloat16,compile_decode=False,fuse_decode=False);engines.append(e);set_gates(e.model,False)
            need(e.runner.spec is None and len(e.model.layers)==64,'target-only full64 engine')
        weights=[dict(tree_flatten(e.model.parameters())) for e in engines]
        need(not any(weights[0][k] is weights[1][k] for k in weights[0]),'independent model handles')
        cmp.compare('independent-loaded-weights',support.snapshot(weights[0]),support.snapshot(weights[1]));del weights
        emit(dict(event='loaded-weights',checks=cmp.rows));cmp.rows=[]
        before=[params(i,'before') for i in range(2)]
        for f in fixtures['fixtures']:
            for e in engines: cmp.meta('fixture EOS',sorted(f['eos_token_ids']),sorted(e.eos_ids))
        graph=candidate.RoutePreservingDecode(engines[1].model,enabled=True);result['model_source_receipt']=graph.source_receipt
        layout=tuple(l.is_attn for l in engines[0].model.layers)
        diagnostic=old.diagnostic_fixtures(fixtures['fixtures'],1023);write(out/'diagnostic-inputs.json',diagnostic)
        seeds=[[make_cache(e.model) for _ in range(8)] for e in engines];previous=0
        for used in (257,1023,1024):
            for e in engines:set_gates(e.model,False)
            tokens=[];end=used-1
            for i,f in enumerate(diagnostic):
                deadline();logits=[e.model(mx.array(f['prompt_ids'][previous:end],dtype=mx.int32),ForwardContext('prefill',s[i]),last_only=True) for e,s in zip(engines,seeds)]
                cmp.compare(f'prefill-{end}-{i}/logits',support.snapshot(logits[0]),support.snapshot(logits[1]))
                cmp.compare(f'prefill-{end}-{i}/state',support.snapshot(seeds[0][i]),support.snapshot(seeds[1][i]))
                tokens.append(int(mx.argmax(logits[0],axis=-1).item()));emit(dict(event='prefill',row=i,begin=previous,end=end,checks=cmp.rows));cmp.rows=[];del logits
            if used==257:
                previous=end
                continue
            for B in (1,8):
                batch=DecodeBatch(engines[0].model);batch.rebuild([],[(f'row-{i}',c,1536) for i,c in enumerate(seeds[0][:B])])
                canonical=support.snapshot(dict(state=batch.layers,tokens=mx.array(tokens[:B],dtype=mx.int32),
                    positions=mx.array([used-1]*B,dtype=mx.int32),lens=mx.array([used]*B,dtype=mx.int32)))
                mx.eval(*canonical['arrays'].values());del batch
                emit(dict(event='canonical',B=B,used=used,hashes=m.hashes(mx,canonical,deadline)))
                oracle=None
                for enabled in (False,True):
                    for e in engines:set_gates(e.model,enabled)
                    name=f'B{B}-used{used}-gates'+('ON' if enabled else 'OFF');phase('before-'+name)
                    reference,reference_hash=qualify(name,'A',canonical,oracle)
                    if oracle is None: oracle=reference
                    actual,actual_hash=qualify(name,'C',canonical,oracle)
                    cmp.meta(name+'/same-output-hashes',reference_hash,actual_hash)
                    del actual
                    for warm,order in ((True,fixed_plan()['warm_arms']),(False,fixed_plan()['formal_arms'])):
                        for index,arm in enumerate(order):sample(name,arm,index,warm,canonical,reference)
                    result['configs'].append(dict(name=name,status='PASS',B=B,used=used,gates=enabled,
                        qualification_output_hashes=dict(A=reference_hash,C=actual_hash)))
                    del reference;phase('after-'+name)
                del oracle,canonical
                gc.collect();mx.synchronize()
            previous=end
        for i in range(2):cmp.meta(f'model-{i}/unchanged-parameters',h.parameter_values(before[i]),h.parameter_values(params(i,'after')))
        need(len(result['configs'])==8 and len(result['samples'])==128 and all(r['status']=='PASS' for r in result['samples']),'all fixed timing samples')
        need(sum(not r['warm'] for r in result['samples'])==96,'all formal samples')
        result['diagnostic_complete']=True
    finally:
        if graph is not None:graph._functions.clear()
        seeds.clear();engines.clear()
    support.verify_files(root,source,deadline);support.verify_files(model_path,model['files'],deadline)
    for key in ('source_manifest','model_manifest','fixtures'):support.pinned_json(inherited[key])
    validate(plan);result['source_model_pins_rechecked']=True


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--plan-sha256',required=True);ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args(argv);m=matrix();q=m.helper();h=q.helpers();h.need(h.sha(args.plan)==args.plan_sha256,'plan SHA')
    plan=json.loads(args.plan.read_text());m,q,h,old,support,candidate,original,inherited=validate(plan)
    h.need(signal.getitimer(signal.ITIMER_REAL)==(0.,0.),'existing alarm');args.out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    result=dict(schema='E024-route-timing-result-v1',status='INCOMPLETE',contract=fixed_plan(),pid=os.getpid(),ppid=os.getppid(),
        device_run=False,diagnostic_complete=False,source_model_pins_rechecked=False,configs=[],samples=[],parameter_receipts=[],
        full_model_qualified=False,performance_qualified=False,http_qualified=False,pareto_eligible=False,
        plan_sha256=args.plan_sha256,sources={str(Path(__file__).resolve()):plan['qualifier_sha256'],str(HERE/'qualifier_matrix.py'):MATRIX_SHA,
            str(HERE/'qualify.py'):m.CANARY_SHA,str(HERE/'candidate.py'):m.CANDIDATE_SHA,str(q.ISOLATION):q.ISOLATION_SHA,
            str(q.OLD/'qualify.py'):q.OLD_SHA,str(candidate.ORIGINAL):candidate.ORIGINAL_SHA,
            original['e023_qualifier']['path']:old.SUPPORT_SHA,str(candidate.PINS):old.MODEL_SOURCE_SHA})
    h.write(args.out/'plan.json',plan);h.write(args.out/'inherited-e024-plan.json',original);h.write(args.out/'inherited-e023-plan.json',inherited)
    def deadline(*args):
        if args or time.monotonic()-started>=plan['deadline_s']:raise TimeoutError('forward diagnostic absolute deadline')
    def stop(signum,frame):
        if signum==signal.SIGALRM:deadline(signum)
        raise InterruptedError('forward diagnostic cancellation '+signal.Signals(signum).name)
    previous={s:signal.getsignal(s) for s in (signal.SIGALRM,signal.SIGINT,signal.SIGTERM)}
    with (args.out/'events.jsonl').open('x') as stream:
        def emit(row):stream.write(json.dumps(dict(**row,elapsed_wall_s=time.monotonic()-started),allow_nan=False)+'\n');stream.flush()
        try:
            for s in previous:signal.signal(s,stop)
            signal.setitimer(signal.ITIMER_REAL,plan['deadline_s'])
            device(plan,m,q,h,old,support,candidate,original,inherited,args.out,deadline,emit,result)
            h.need(h.sha(args.plan)==args.plan_sha256,'plan changed');result['status']='PASS'
        except BaseException as exc:result.update(status='FAIL',error=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()))
        finally:
            try:
                gc.collect();mx=sys.modules.get('mlx.core')
                if mx is not None:mx.synchronize();mx.clear_cache();result.update(final_active_bytes=mx.get_active_memory(),peak_bytes=mx.get_peak_memory())
            except BaseException as exc:result.update(status='FAIL',cleanup_error=dict(type=type(exc).__name__,message=str(exc)))
            finally:
                signal.setitimer(signal.ITIMER_REAL,0)
                for s,handler in previous.items():signal.signal(s,handler)
                result['elapsed_wall_s']=time.monotonic()-started;h.write(args.out/'complete.json',result)
    print(json.dumps(dict(status=result['status'],out=str(args.out))),flush=True);return int(result['status']!='PASS')


if __name__=='__main__':raise SystemExit(main())
