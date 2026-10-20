"""Default-inert C1 forward/selected-prefix commit cost screen. No owner or HTTP."""
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
import statistics
import sys
import time
import traceback
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
QUALIFIER = HERE.parent/'rowwise_native_verify/qualify.py'
QUALIFIER_SHA = '0a7d81f835be9a1ac2328feec8a455bd85464077cd1661be0350e1772cde701a'
WIDTHS = (4, 8)


def need(ok, message):
    if not ok: raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(8 << 20): digest.update(block)
    return digest.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def load(path, name, expected):
    need(sha(path) == expected, 'source SHA: '+str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module); return module


def pinned(ref):
    need(set(ref) == {'path', 'sha256'} and Path(ref['path']).is_absolute(), 'absolute pinned reference')
    need(sha(ref['path']) == ref['sha256'], 'reference SHA')
    return json.loads(Path(ref['path']).read_bytes())


def contract():
    return dict(schema='C1-verifier-cost-contract-v1', enabled_by_default=False, used=1023, widths=list(WIDTHS),
        qualification_prefixes='k=1,T/2,T; full T logits, selected 128 states, native continuation, canonical unchanged',
        primary_baseline='native device-dependent AR(T); per-step async_eval(token,state), final full logits/state sync',
        secondary_baseline='native host-sequential AR(T), independent numerical oracle',
        timing='fresh descriptors, full logits, selected state, returned IDs, isolated holder commit, release, final sync',
        warmups=3, groups=5, pairs_per_group=10, timed_prefix='k=T only',
        short_prefix_cost_measured=False, compiled_qualified=False, rollback_qualified=False,
        full_model_qualified=False, http_qualified=False, performance_qualified=False, engine_pareto_point=False)


def validate(plan):
    need(set(plan) == {'schema','screen_sha256','e042_plan','e042_result','deadline_s'}
         and plan['schema'] == 'C1-verifier-cost-plan-v1', 'exact cost plan')
    need(plan['screen_sha256'] == sha(__file__), 'actual screen source')
    need(type(plan['deadline_s']) in (int,float) and math.isfinite(plan['deadline_s'])
         and 0 < plan['deadline_s'] <= 1800, 'finite child deadline')
    q = load(QUALIFIER, '_c1_cost_e042_qualifier', QUALIFIER_SHA)
    prior_plan = pinned(plan['e042_plan']); dependencies = q.validate(prior_plan)
    previous = pinned(plan['e042_result'])
    need(previous['status'] == 'PASS' and previous['forward_prefix_qualified'] is True
         and previous['plan_sha256'] == plan['e042_plan']['sha256'], 'qualified E042 provenance')
    seeds = [s for s in previous['reference_blocks'] if s['used'] == 1023]
    need(len(seeds) == 1 and seeds[0]['status'] == 'PASS' and len(seeds[0]['tokens']) == 8, 'one E042 used1023 seed')
    return q, dependencies, seeds[0]


def schedule():
    return [(group, pair, ('native','candidate') if (group*10+pair)%2 == 0 else ('candidate','native'))
            for group in range(5) for pair in range(10)]


def selected_prefixes(width):
    need(width in WIDTHS, 'T4/T8 only'); return (1, width//2, width)


def commit(holder, layers, seed_position, count, pending):
    need(type(count) is int and count > 0 and len(layers) == 64, 'complete selected state')
    holder.req_ids = ['c1-cost-local']; holder.layers = list(layers)
    holder.lens = [seed_position+count]
    return pending


def fresh_context(r, layers, position):
    copied = [r.BatchKV(c.K[:],c.V[:]) if hasattr(c,'K') else r.BatchState(c.conv[:],c.S[:]) for c in layers]
    return r.ForwardContext('decode', copied, positions=r.mx.array([position], dtype=r.mx.int32),
        lens=r.mx.array([position+1],dtype=r.mx.int32), max_len=position+1)


def arrays(layers):
    return [a for c in layers for a in ((c.K,c.V) if hasattr(c,'K') else (c.conv,c.S))]


def native(r, engine, h, layout, storage, position, width, host_sync, retain_prefixes=False):
    """Causal AR: next input ALWAYS comes from this arm's own previous logits."""
    mx=r.mx; ctx=h.context_from_input(storage,layout,r,position+1)
    token=storage['root/tokens'][:]; values=[]; prefixes=[]; inputs=[]
    for step in range(width):
        if step: ctx=fresh_context(r,ctx.caches,position+step)
        if retain_prefixes: inputs.append(int(token.item()))  # oracle qualification only
        logits=engine.runner.decode_model(token.astype(mx.int32),ctx)
        token=mx.argmax(logits,axis=-1).astype(mx.int32)
        if host_sync:
            mx.eval(logits,*arrays(ctx.caches)); token=mx.array([int(token.item())],dtype=mx.int32)
        else: mx.async_eval(token,*arrays(ctx.caches))
        values.append(logits)
        if retain_prefixes: prefixes.append(list(ctx.caches))
    logits=mx.concatenate(values,axis=0); mx.eval(logits,*arrays(ctx.caches),token)
    ids=mx.argmax(logits,axis=-1).tolist()
    holder=r.DecodeBatch(engine.model); pending=commit(holder,ctx.caches,position,width,ids[-1])
    del ctx,values,token
    return dict(logits=logits,holder=holder,pending=pending,ids=ids,prefixes=prefixes,inputs=inputs)


def candidate_run(r,engine,q,candidate,h,layout,storage,position,tokens,count,project,*,audit=False):
    mx=r.mx; width=len(tokens); row={}; ctx=h.context_from_input(storage,layout,r,position+1)
    output=candidate.verify(engine.model,mx.array(tokens,dtype=mx.int32),ctx,r,enabled=True,
        head_project=q.counted_head(project,mx,engine.model,width,row) if audit else project)
    need(output is not None, 'candidate runtime eligibility')
    if audit: q.prefix_ownership(output,ctx,layout,width)
    selected=output.prefixes[count-1]; logits=output.logits
    mx.eval(logits,*arrays(selected))  # rejected full state is not forced only for auditing
    ids=mx.argmax(logits,axis=-1).tolist()
    holder=r.DecodeBatch(engine.model); pending=commit(holder,selected,position,count,ids[count-1])
    del output,selected,ctx
    return dict(logits=logits,holder=holder,pending=pending,ids=ids,head=row.get('head_projection'))


def continuation(r,engine,result,position):
    ctx=fresh_context(r,result['holder'].layers,position)
    logits=engine.runner.decode_model(r.mx.array([result['pending']],dtype=r.mx.int32),ctx)
    r.mx.eval(logits,*arrays(ctx.caches))
    return logits,ctx.caches


def summarize(rows):
    groups=[]
    for group in range(5):
        subset=[x for x in rows if x['group']==group]
        need(len(subset)==20 and {x['pair'] for x in subset}==set(range(10)), 'complete timing group')
        paired=[]
        for pair in range(10):
            samples=[x for x in subset if x['pair']==pair]
            need(len(samples)==2 and {x['arm'] for x in samples}=={'native','candidate'}, 'unique arm pair')
            times={x['arm']:x['elapsed_ns'] for x in samples}
            need(all(type(n) is int and n>0 for n in times.values()), 'positive measured ns')
            paired.append(times['native']/times['candidate'])
        groups.append(dict(group=group,median_paired_ratio=statistics.median(paired),ratios=paired))
    totals={arm:sorted(x['elapsed_ns'] for x in rows if x['arm']==arm) for arm in ('native','candidate')}
    return dict(groups=groups,median_group_ratio=statistics.median(x['median_paired_ratio'] for x in groups),
        totals_ns={arm:dict(p50=statistics.median(v),p90=v[math.ceil(.9*len(v))-1]) for arm,v in totals.items()})


def device(plan,q,deps,seed,out,deadline,result):
    bridge,candidate,h,old,support,original,inherited=deps
    source=support.pinned_json(inherited['source_manifest'])['files']; manifest=support.pinned_json(inherited['model_manifest'])
    fixtures=support.pinned_json(inherited['fixtures']); root=Path(inherited['package_root']).resolve()/'tutorial_code/miniengine'
    model_path=Path(inherited['model']).resolve()
    need(source==json.loads(bridge.CORE_PINS.read_bytes())['files']
         and set(source)=={str(p.relative_to(root)) for p in root.rglob('*.py')}, 'frozen core tree')
    need(Path(manifest['model']).resolve()==model_path and manifest['revision']==model_path.name
         and Path(fixtures['model']).resolve()==model_path, 'original model/fixture identity')
    required={'config.json','generation_config.json','tokenizer.json','tokenizer_config.json'}|{
        str(p.relative_to(model_path)) for p in model_path.rglob('*.safetensors')}
    need(required<=set(manifest['files']) and any(p.endswith('.safetensors') for p in required), 'all original payload pins')
    support.verify_files(root,source,deadline); support.verify_files(model_path,manifest['files'],deadline)
    need({k:v for k,v in os.environ.items() if k.startswith('MLX_')}==inherited['execution']['environment'],'original MLX environment')
    canonical=Path(plan['e042_result']['path']).parent/seed['canonical']['path']
    need(h.sha(canonical,deadline)==seed['canonical']['sha256'], 'original canonical raw bytes')
    sys.path.insert(0,str(root.parent.parent))
    import mlx.core as mx
    import tutorial_code.miniengine as package
    from tutorial_code.miniengine import cache,context
    from tutorial_code.miniengine.engine.llm_engine import LLMEngine
    from mlx.utils import tree_flatten
    need(Path(package.__file__).resolve().parent==root,'actual package root')
    result['device']=dict(name=mx.device_info().get('device_name'),architecture=mx.device_info().get('architecture'),
                          mlx_version=importlib.metadata.version('mlx'))
    need(result['device']==dict(name='Apple M5 Ultra',architecture='applegpu_g17d',mlx_version='0.32.3') and mx.default_device()==mx.gpu,'fixed runtime')
    r=SimpleNamespace(mx=mx,BatchKV=cache.BatchKV,BatchState=cache.BatchState,ForwardContext=context.ForwardContext,DecodeBatch=cache.DecodeBatch)
    engine=model=oracle=project=layer=actual=value=logits=layers=data=None; storage={}; cmp=support.Comparator(mx,out)
    try:
        engine=LLMEngine(str(model_path),max_len=2048,max_seqs=1,max_batched_tokens=16384,cache_limit_gb=2,
                         dtype=mx.bfloat16,compile_decode=True,fuse_decode=True)
        model=engine.model; layout=tuple(layer.is_attn for layer in model.layers)
        need(len(layout)==64 and sum(layout)==16 and engine.runner.spec is None and engine.runner._ahead is None,'fresh target only model')
        for layer in model.layers:
            if not layer.is_attn: layer.linear_attn.set_gates_compile(True); layer.linear_attn.set_norm_reuse(False)
        need(seed['fixture_id']==bridge.selected(fixtures['fixtures'])[0][0]['fixture_id'],'shortest original fixture')
        need(all(sorted(f['eos_token_ids'])==sorted(engine.eos_ids) for f in fixtures['fixtures']),'normal original EOS')
        def parameters(name):
            data=h.parameter_hashes(mx,model,tree_flatten,deadline); write(out/(name+'.json'),data); return h.parameter_values(data)
        before=parameters('parameters-before')
        project=q.load(q.HEAD,'_c1_cost_head',q.HEAD_SHA).build(mx,enabled=True); need(project is not None,'explicit head enabled')
        for arm in ('canonical','oracle','host','device','candidate'):
            storage[arm]=mx.load(str(canonical)); mx.eval(*storage[arm].values())
        position=seed['position']; need(position==1022 and seed['used']==1023,'fixed boundary')
        need(storage['canonical']['root/positions'].tolist()==[position]
             and storage['canonical']['root/lens'].tolist()==[position+1], 'saved absolute position')
        oracle=native(r,engine,h,layout,storage['oracle'],position,9,True,True)
        need(oracle['inputs'][:8]==seed['tokens'] and not (set(oracle['inputs'])|set(oracle['ids'])) & set(engine.eos_ids),'original causal block and normal EOS')
        def snapshot(logits,layers): return support.snapshot(dict(logits=logits,state=bridge.canonical_state(layers,layout)))
        def compare_saved(label,reference,actual):
            desc=[h.dump(mx,out,label+'/'+arm+'.safetensors',snap,deadline) for arm,snap in (('reference',reference),('actual',actual))]
            result['raw_comparisons'].append(dict(label=label,files=desc)); write(out/'complete.json',result)
            cmp.compare(label,reference,actual)
        for width in WIDTHS:
            for count in selected_prefixes(width):
                deadline(); actual=candidate_run(r,engine,q,candidate,h,layout,storage['candidate'],position,oracle['inputs'][:width],count,project,audit=True)
                compare_saved(f'T{width}/k{count}/selected',snapshot(oracle['logits'][:width],oracle['prefixes'][count-1]),snapshot(actual['logits'],actual['holder'].layers))
                need(actual['ids']==oracle['ids'][:width] and actual['pending']==oracle['ids'][count-1]
                     and actual['holder'].lens==[position+count], 'committed lens/pending ID')
                logits,layers=continuation(r,engine,actual,position+count)
                compare_saved(f'T{width}/k{count}/continuation',snapshot(oracle['logits'][count:count+1],oracle['prefixes'][count]),snapshot(logits,layers))
                result['qualification'].append(dict(T=width,k=count,status='PASS',head=actual['head'],committed_lens=[position+count],pending=actual['pending']))
                del actual,logits,layers
            for baseline in ('host','device'):
                actual=native(r,engine,h,layout,storage[baseline],position,width,baseline=='host')
                compare_saved(f'T{width}/{baseline}',snapshot(oracle['logits'][:width],oracle['prefixes'][width-1]),snapshot(actual['logits'],actual['holder'].layers))
                need(actual['ids']==oracle['ids'][:width],'native submission-order IDs'); del actual
        for arm,data in storage.items(): cmp.compare('canonical-preserved/'+arm,support.snapshot(storage['canonical']),support.snapshot(data))
        result['numerical_screen_qualified']=True; result['checks']=cmp.rows; cmp.rows=[]; write(out/'complete.json',result)
        for width in WIDTHS:
            for baseline in ('device','host'):
                def run(arm):
                    return native(r,engine,h,layout,storage[baseline],position,width,baseline=='host') if arm=='native' else candidate_run(r,engine,q,candidate,h,layout,storage['candidate'],position,oracle['inputs'][:width],width,project)
                for _ in range(3):
                    for arm in ('native','candidate'): value=run(arm); mx.synchronize(); del value
                rows=[]; result['timings'].append(dict(T=width,k=width,baseline=baseline,samples=rows))
                for group,pair,order in schedule():
                    for ordinal,arm in enumerate(order):
                        deadline(); mx.synchronize(); start=time.perf_counter_ns(); value=run(arm); mx.synchronize(); elapsed=time.perf_counter_ns()-start
                        need(value['ids']==oracle['ids'][:width] and value['holder'].lens==[position+width],'timed IDs/lens')
                        record=dict(group=group,pair=pair,order=list(order),ordinal=ordinal,arm=arm,elapsed_ns=elapsed,ids=value['ids'],head=value.get('head'))
                        rows.append(record)
                        with (out/'timings.jsonl').open('a') as stream: stream.write(json.dumps(dict(T=width,baseline=baseline,**record))+'\n')
                        del value
                result['timings'][-1]['summary']=summarize(rows); write(out/'complete.json',result)
        for arm,data in storage.items(): cmp.compare('post-timing-canonical/'+arm,support.snapshot(storage['canonical']),support.snapshot(data))
        cmp.meta('parameter values preserved',before,parameters('parameters-after'))
        need(engine.runner.decode_model._functions=={} and engine.runner.decode_model._packed is None and engine.runner._ahead is None,'uncompiled used>256 path')
        result['post_checks']=cmp.rows
    except BaseException as exc:
        traceback.clear_frames(exc.__traceback__); raise
    finally:
        storage.clear(); oracle=project=model=layer=actual=value=logits=layers=data=None
        if engine is not None: engine.runner.batch.clear(); engine.runner.staging.clear(); engine.runner.feats.clear(); engine.runner._ahead=None
        engine=None; gc.collect(); mx.synchronize(); mx.clear_cache(); result['device_cleanup_completed']=True
        result['memory_after_cleanup']=dict(active_bytes=mx.get_active_memory(),peak_bytes=mx.get_peak_memory(),cache_bytes=mx.get_cache_memory())
    support.verify_files(root,source,deadline); support.verify_files(model_path,manifest['files'],deadline)
    validate(plan); result['source_model_pins_rechecked']=True


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--run',action='store_true')
    p.add_argument('--plan',type=Path); p.add_argument('--plan-sha256'); p.add_argument('--out',type=Path); a=p.parse_args(argv)
    if not a.run:
        if any(x is not None for x in (a.plan,a.plan_sha256,a.out)): p.error('explicit --run required')
        print(json.dumps(contract(),indent=2)); return 0
    if any(x is None for x in (a.plan,a.plan_sha256,a.out)): p.error('--run needs all plan/out arguments')
    a.out.mkdir(parents=True,exist_ok=False); start=time.monotonic(); previous={}
    result=dict(schema='C1-verifier-cost-result-v1',status='INCOMPLETE',pid=os.getpid(),ppid=os.getppid(),plan_sha256=a.plan_sha256,
        contract=contract(),qualification=[],raw_comparisons=[],timings=[],numerical_screen_qualified=False,
        device_cleanup_completed=False,source_model_pins_rechecked=False,http_qualified=False,performance_qualified=False,engine_pareto_point=False)
    try:
        need(sha(a.plan)==a.plan_sha256,'plan SHA'); plan=json.loads(a.plan.read_bytes()); write(a.out/'plan.json',plan)
        q,deps,seed=validate(plan)
        result['sources']={str(Path(__file__).resolve()):plan['screen_sha256'],str(QUALIFIER):QUALIFIER_SHA,
            str(q.HERE/'candidate.py'):q.CANDIDATE_SHA,str(q.HEAD):q.HEAD_SHA}
        result['seed']=seed; write(a.out/'complete.json',result)
        def deadline():
            if time.monotonic()-start>=plan['deadline_s']: raise TimeoutError('C1 cost deadline')
        def stop(signum,frame): raise InterruptedError('bounded C1 cost signal '+str(signum))
        need(signal.getitimer(signal.ITIMER_REAL)==(0.0,0.0),'existing alarm')
        previous={s:signal.getsignal(s) for s in (signal.SIGALRM,signal.SIGINT,signal.SIGTERM)}
        for s in previous: signal.signal(s,stop)
        signal.setitimer(signal.ITIMER_REAL,plan['deadline_s'])
        device(plan,q,deps,seed,a.out,deadline,result); deadline()
        need(result['numerical_screen_qualified'] and result['device_cleanup_completed'] and result['source_model_pins_rechecked'],'qualification and cleanup')
        need(sha(a.plan)==a.plan_sha256,'unchanged plan'); result['status']='PASS_LOCAL_COST_ONLY'
    except BaseException as exc:
        result.update(status='FAIL',error=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()))
    finally:
        if previous:
            signal.setitimer(signal.ITIMER_REAL,0)
            for s,value in previous.items(): signal.signal(s,value)
        result['elapsed_wall_s']=time.monotonic()-start; write(a.out/'complete.json',result)
    print(json.dumps(dict(status=result['status'],samples=sum(len(x['samples']) for x in result['timings']))))
    return 0 if result['status']=='PASS_LOCAL_COST_ONLY' else 1


if __name__=='__main__': raise SystemExit(main())
