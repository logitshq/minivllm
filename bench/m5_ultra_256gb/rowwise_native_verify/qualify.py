"""Unsealed, default-inert rowwise C1 numerical qualifier; no device run yet.

No arguments describes the contract. An external cold/resource owner may later
invoke --plan FILE --plan-sha256 SHA --out NEW after both source pins are sealed.
This tests forward prefixes only; it installs no sampler, acceptance or rollback.
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

HERE = Path(__file__).resolve().parent
BRIDGE = HERE.parent/'exact_verifier_bridge/qualify.py'
BRIDGE_SHA = 'UNSEALED'
CANDIDATE_SHA = 'UNSEALED'
WIDTHS = (2,4,8)
USED = (257,1023,1024)
BACKING = 1536


def need(ok, message):
    if not ok: raise ValueError(message)


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sealed(value):
    return isinstance(value,str) and len(value)==64 and all(x in '0123456789abcdef' for x in value)


def load(path, name, digest):
    need(sealed(digest) and sha(path)==digest,'sealed source: '+str(path))
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    # dataclass introspection uses the module's real identity, not a donor alias.
    sys.modules[name]=module
    try: spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name,None); raise
    return module


def contract():
    return dict(schema='Rowwise-native-verifier-contract-v1',enabled_by_default=False,
        sources_sealed=sealed(BRIDGE_SHA) and sealed(CANDIDATE_SHA),widths=list(WIDTHS),used=list(USED),
        cases=9,compared_prefixes=42,states_per_prefix=128,KV_backing=BACKING,
        seed='shortest original D8 fixture; used257 uses its first256 IDs as a NEW diagnostic prompt',
        position='used is the first decode attention bound; pending token writes at used-1',
        teacher='independent original C1 greedy continuation, normal EOS/cap512, no forced continuation',
        oracle='source-pinned runner.decode_model, one actual token and absolute position per call',
        candidate='uncompiled layer-major B1 calls; each prefix retained; fixed teacher IDs only',
        configuration=dict(max_seqs=1,max_len=2048,max_batched_tokens=16384,cache_limit_gib=2,
            compile_decode=True,fuse_decode=True,gates_compile=True,norm_reuse=False),
        arrays='full-vocabulary logits plus every original-dtype state including full KV padding at EACH prefix',
        parameter_check='all model original-dtype values before/after; producer hash receipts, no handle equality',
        exclusions=['E027 M>1 equivalence','E024 compiled qualification','parameter replacement',
                    'accept/rollback','EOS retirement','scheduler/ahead','HTTP','performance'],
        compiled_qualified=False,rollback_qualified=False,full_model_qualified=False,
        http_qualified=False,performance_qualified=False)


def validate(plan):
    need(set(plan)=={'schema','qualifier_sha256','candidate','bridge','e024_plan','deadline_s'}
         and plan['schema']=='Rowwise-native-verifier-plan-v1','exact rowwise plan')
    need(sealed(BRIDGE_SHA) and sealed(CANDIDATE_SHA),'sources remain UNSEALED')
    need(plan['qualifier_sha256']==sha(__file__),'actual qualifier source')
    need(plan['candidate']==dict(path=str(HERE/'candidate.py'),sha256=CANDIDATE_SHA)
         and plan['bridge']==dict(path=str(BRIDGE),sha256=BRIDGE_SHA),'real adjacent source bindings')
    budget=plan['deadline_s']
    need(type(budget) in (int,float) and math.isfinite(budget) and 0<budget<=1800,'bounded child deadline')
    bridge=load(BRIDGE,'_rowwise_reference_bridge',BRIDGE_SHA)
    candidate=load(HERE/'candidate.py','_rowwise_candidate',CANDIDATE_SHA)
    # Validate the bridge's own unchanged source/contract through its public
    # validator. This read-only view is not saved as this experiment's plan.
    h,old,support,original,inherited=bridge.validate(dict(schema='Exact-verifier-bridge-plan-v1',
        qualifier_sha256=BRIDGE_SHA,e024_plan=plan['e024_plan'],deadline_s=budget))
    return bridge,candidate,h,old,support,original,inherited


def seed_cases(bridge, fixtures):
    fixture=bridge.selected(fixtures)[0][0]  # also validates all eight original fixtures
    result=[]
    for used in USED:
        prompt=list(fixture['prompt_ids'][:256] if used==257 else fixture['prompt_ids'])
        start=used-len(prompt)
        need(0<start and start+8<=512,'normal-EOS reference window within cap')
        result.append(dict(fixture_id=fixture['fixture_id'],original_fixture=fixture,
            prompt_ids=prompt,domain='diagnostic-prefix256' if used==257 else 'original-full-prompt',
            used=used,position=used-1,start=start,backing=BACKING))
    return result


def seed_admission(case, batch_lens, position, layers, layout):
    need(position==case['position']==case['used']-1 and position>=256
         and batch_lens==[position] and case['used']+7<=BACKING,'actual committed seed position')
    need(len(layers)==len(layout)==64 and sum(layout)==16,'complete seed layout')
    need(all(c.K.shape==c.V.shape==(1,4,BACKING,256) for c,attn in zip(layers,layout) if attn),
         'fixed complete1536 KV backing; no mask/trim or allocator-inferred capacity')


def output_snapshot(support,bridge,logits,caches,layout,position,input_token,prefix):
    snap=support.snapshot(dict(logits=logits,state=bridge.canonical_state(caches,layout),
        position=position,lens=position+1,max_len=position+1,input_token=input_token,prefix=prefix))
    need(len(snap['arrays'])==129,'full logits plus all128 prefix arrays')
    return snap


def input_snapshot(support,bridge,tokens,ctx,layout):
    snap=support.snapshot(dict(tokens=tokens,state=bridge.canonical_state(ctx.caches,layout),
        positions=ctx.positions,lens=ctx.lens,max_len=ctx.max_len,mode=ctx.mode,prefill_shape=ctx.prefill_shape))
    need(len(snap['arrays'])==131,'all128 inputs plus tokens/position/lens')
    return snap


def acquire_seed(mx,cache,context,bridge,support,h,cmp,engine,layout,case,out,deadline,emit):
    """New seed-domain adapter; the old acquire_seed's p>256/capacity is NOT reused."""
    prompt=case['prompt_ids']; model=engine.model
    caches=cache.make_cache(model); batch=cache.DecodeBatch(model)
    row=dict(**case,reference_outputs=[],status='INCOMPLETE'); anchor_snapshot=None
    prefix=f"seeds/used{case['used']}"
    h.write(out/'current-seed.json',row)
    try:
        logits=model(mx.array(prompt,dtype=mx.int32),context.ForwardContext('prefill',caches),last_only=True)
        mx.eval(logits,*support.snapshot(caches)['arrays'].values())
        token=int(mx.argmax(logits,axis=-1).item())
        row['prefill_token']=token
        # The 256-token diagnostic seed would otherwise allocate only768 slots.
        batch.rebuild([],[('teacher-reference',caches,BACKING)])
        mx.eval(*batch.arrays()); caches.clear()
        def anchor(input_token,step):
            nonlocal anchor_snapshot
            p=len(prompt)+step-1
            seed_admission(case,batch.lens,p,batch.layers,layout)
            anchor_snapshot=support.snapshot(dict(state=bridge.canonical_state(batch.layers,layout),
                tokens=mx.array([input_token],dtype=mx.int32),positions=mx.array([p],dtype=mx.int32),
                lens=mx.array([p+1],dtype=mx.int32)))
            need(len(anchor_snapshot['arrays'])==131,'complete canonical seed')
            row['canonical']=h.dump(mx,out,prefix+'/canonical.safetensors',anchor_snapshot,deadline)
        def advance(input_token,step):
            deadline(); p=len(prompt)+step-1
            ctx=context.ForwardContext('decode',batch.layers,positions=mx.array([p],dtype=mx.int32),
                lens=mx.array([p+1],dtype=mx.int32),max_len=p+1)
            values=engine.runner.decode_model(mx.array([input_token],dtype=mx.int32),ctx)
            mx.eval(values,*batch.arrays()); batch.lens=[p+1]
            if step>=case['start']:
                ordinal=step-case['start']+1
                snap=output_snapshot(support,bridge,values,batch.layers,layout,p,input_token,ordinal)
                row['reference_outputs'].append(h.dump(mx,out,f'{prefix}/reference-p{ordinal}.safetensors',snap,deadline))
            return int(mx.argmax(values,axis=-1).item())
        raw,tokens=bridge.reference_tokens(token,engine.eos_ids,case['start'],advance,anchor,
            lambda value:emit(dict(used=case['used'],fixture_id=case['fixture_id'],**value)))
        row.update(raw_ids=raw,tokens=tokens)
        row['reference_input_after']=h.dump(mx,out,prefix+'/canonical-after-reference.safetensors',anchor_snapshot,deadline)
        arrays=mx.load(str(out/row['canonical']['path'])); mx.eval(*arrays.values())
        cmp.compare(prefix+'/reference-snapshot-preserved',
            dict(metadata=row['canonical']['metadata'],arrays=arrays),anchor_snapshot)
        need(len(row['reference_outputs'])==8,'eight independent reference prefix outputs')
        row['status']='PASS'; return row
    finally:
        h.write(out/'current-seed.json',row)
        caches.clear(); batch.clear()


def prefix_ownership(output,ctx,layout,width):
    need(output is not None and len(output.prefixes)==width,'candidate admitted complete block')
    need(all(len(p)==64 for p in output.prefixes),'complete candidate prefix state')
    for index,attn in enumerate(layout):
        wrappers=[ctx.caches[index],*(p[index] for p in output.prefixes)]
        need(len({id(c) for c in wrappers})==width+1,'prefix-owned cache wrappers')
        if attn:
            for name in ('K','V'):
                need(len({id(getattr(c,name)) for c in wrappers})==width+1,'prefix-owned KV descriptors')


def checked_call(candidate,model,tokens,ctx,runtime,*,before,evaluate,after,compare):
    """Retain/compare actual supplied input even if candidate raises; never retry."""
    before()
    try:
        output=candidate.verify(model,tokens,ctx,runtime,enabled=True)
        evaluate(output)
        return output
    finally:
        after(); compare()


def compare_case(mx,bridge,candidate,h,support,cmp,engine,layout,runtime,seed,width,row,out,deadline):
    descriptor=seed['canonical']; path=out/descriptor['path']
    need(h.sha(path,deadline)==descriptor['sha256'],'canonical raw source')
    arrays=mx.load(str(path)); mx.eval(*arrays.values())
    ctx=h.context_from_input(arrays,layout,runtime,seed['used'])
    tokens=mx.array(seed['tokens'][:width],dtype=mx.int32)
    need(tokens.tolist()==seed['tokens'][:width] and ctx.positions.tolist()==[seed['position']]
         and ctx.lens.tolist()==[seed['used']],'actual teacher IDs and absolute seed metadata')
    tag=f"cases/used{seed['used']}/T{width}"
    def before():
        row['input_before']=h.dump(mx,out,tag+'/input-before.safetensors',input_snapshot(support,bridge,tokens,ctx,layout),deadline)
    def after():
        row['input_after']=h.dump(mx,out,tag+'/input-after.safetensors',input_snapshot(support,bridge,tokens,ctx,layout),deadline)
    def unchanged():
        saved=mx.load(str(out/row['input_before']['path'])); mx.eval(*saved.values())
        cmp.compare(tag+'/input-preserved',dict(metadata=row['input_before']['metadata'],arrays=saved),
                    input_snapshot(support,bridge,tokens,ctx,layout))
        # The loader's original arrays, including complete unused KV tails, also remain unchanged.
        pristine=mx.load(str(path)); mx.eval(*pristine.values())
        cmp.compare(tag+'/canonical-preserved',dict(metadata=descriptor['metadata'],arrays=pristine),
                    dict(metadata=descriptor['metadata'],arrays=arrays))
    def evaluate(output):
        prefix_ownership(output,ctx,layout,width)
        need(output.logits.shape==(width,engine.cfg.vocab_size),'complete T x full vocabulary')
        mx.eval(output.logits,*[a for prefix in output.prefixes for a in
            support.snapshot(bridge.canonical_state(prefix,layout))['arrays'].values()])
    output=checked_call(candidate,engine.model,tokens,ctx,runtime,before=before,evaluate=evaluate,
                        after=after,compare=unchanged)
    row['prefixes']=[]
    for index in range(width):
        deadline(); desc=seed['reference_outputs'][index]
        need(h.sha(out/desc['path'],deadline)==desc['sha256'],'independent reference prefix bytes')
        reference=mx.load(str(out/desc['path'])); mx.eval(*reference.values())
        actual=output_snapshot(support,bridge,output.logits[index:index+1],output.prefixes[index],layout,
                               seed['position']+index,seed['tokens'][index],index+1)
        item=dict(prefix=index+1,reference=desc,status='INCOMPLETE')
        row['prefixes'].append(item)
        item['candidate']=h.dump(mx,out,tag+f'/candidate-p{index+1}.safetensors',actual,deadline)
        # Both original-dtype sides already exist before first-failure comparison.
        cmp.compare(tag+f'/prefix{index+1}',dict(metadata=desc['metadata'],arrays=reference),actual)
        item['status']='PASS'
    row['candidate_greedy_ids']=mx.argmax(output.logits,axis=-1).tolist()
    cmp.meta(tag+'/teacher-reference-output-IDs',seed['raw_ids'][seed['start']:seed['start']+width],
             row['candidate_greedy_ids'])


def rejected_cases(mx,bridge,candidate,h,support,cmp,engine,layout,runtime,seed,out,deadline):
    desc=seed['canonical']; arrays=mx.load(str(out/desc['path'])); mx.eval(*arrays.values())
    ctx=h.context_from_input(arrays,layout,runtime,seed['used'])
    token=mx.array(seed['tokens'][:2],dtype=mx.int32)
    record=dict(cases=[])
    need(candidate.verify(None,None,None,None) is None,'default OFF before inspection')
    for label,tokens,positions,lens,bound in (
            ('T1',token[:1],ctx.positions,ctx.lens,ctx.max_len),
            ('B2',token,mx.array([256,256],dtype=mx.int32),mx.array([257,257],dtype=mx.int32),257),
            ('past1536',token,mx.array([1535],dtype=mx.int32),mx.array([1536],dtype=mx.int32),1536)):
        other=runtime.ForwardContext('decode',ctx.caches,positions=positions,lens=lens,max_len=bound)
        item=dict(label=label,status='INCOMPLETE'); record['cases'].append(item)
        item['input_before']=h.dump(mx,out,'rejections/'+label+'-before.safetensors',
                                   input_snapshot(support,bridge,tokens,other,layout),deadline)
        try:
            need(candidate.verify(engine.model,tokens,other,runtime,enabled=True) is None,'rejected domain '+label)
        finally:
            after=input_snapshot(support,bridge,tokens,other,layout)
            item['input_after']=h.dump(mx,out,'rejections/'+label+'-after.safetensors',after,deadline)
            saved=mx.load(str(out/item['input_before']['path'])); mx.eval(*saved.values())
            cmp.compare('rejected/'+label+'/input-preserved',
                dict(metadata=item['input_before']['metadata'],arrays=saved),after)
            h.write(out/'rejections.json',record)
        item['status']='PASS'; h.write(out/'rejections.json',record)
    return record


def device(plan,bridge,candidate,h,old,support,original,inherited,out,deadline,emit,result):
    source=support.pinned_json(inherited['source_manifest'])['files']
    manifest=support.pinned_json(inherited['model_manifest']); fixtures=support.pinned_json(inherited['fixtures'])
    (out/'fixtures.json').write_bytes(Path(inherited['fixtures']['path']).read_bytes())
    need(sha(out/'fixtures.json')==inherited['fixtures']['sha256'],'original fixture bytes retained')
    root=Path(inherited['package_root']).resolve()/'tutorial_code/miniengine'; model_path=Path(inherited['model']).resolve()
    need(manifest['schema']=='E023-model-v1' and Path(manifest['model']).resolve()==model_path
         and manifest['revision']==model_path.name and Path(fixtures['model']).resolve()==model_path,'fixed model identity')
    need(source==json.loads(bridge.CORE_PINS.read_bytes())['files']
         and set(source)=={str(p.relative_to(root)) for p in root.rglob('*.py')},'complete frozen source tree')
    required={'config.json','generation_config.json','tokenizer.json','tokenizer_config.json'}|{
        str(p.relative_to(model_path)) for p in model_path.rglob('*.safetensors')}
    need(required<=set(manifest['files']) and any(p.endswith('.safetensors') for p in required),'model/tokenizer payload pins')
    support.verify_files(root,source,deadline); support.verify_files(model_path,manifest['files'],deadline)
    need({k:v for k,v in os.environ.items() if k.startswith('MLX_')}==inherited['execution']['environment'],'original MLX environment')
    sys.path.insert(0,str(root.parent.parent))
    import mlx.core as mx
    import tutorial_code.miniengine as package
    from tutorial_code.miniengine import cache, context
    from tutorial_code.miniengine.engine.llm_engine import LLMEngine
    from mlx.utils import tree_flatten
    need(Path(package.__file__).resolve().parent==root and Path(cache.__file__).resolve()==root/'cache.py'
         and Path(context.__file__).resolve()==root/'context.py','actual loaded source root')
    result['device_run']=True
    result['device']=dict(name=mx.device_info().get('device_name'),architecture=mx.metal.device_info().get('architecture'),
                         mlx_version=importlib.metadata.version('mlx'))
    need(result['device']==dict(name='Apple M5 Ultra',architecture='applegpu_g17d',mlx_version='0.32.3')
         and mx.default_device()==mx.gpu,'actual fixed runtime/device')
    runtime=SimpleNamespace(mx=mx,BatchKV=cache.BatchKV,BatchState=cache.BatchState,ForwardContext=context.ForwardContext)
    cmp=support.Comparator(mx,out); engine=model=layer=None
    def memory(phase):
        mx.synchronize(); emit(dict(event='memory',phase=phase,active_bytes=mx.get_active_memory(),
            peak_bytes=mx.get_peak_memory(),cache_bytes=mx.get_cache_memory(),not_performance=True))
    try:
        memory('before-load')
        engine=LLMEngine(str(model_path),max_len=2048,max_seqs=1,max_batched_tokens=16384,
                         cache_limit_gb=2,dtype=mx.bfloat16,compile_decode=True,fuse_decode=True)
        model=engine.model; layout=tuple(layer.is_attn for layer in model.layers)
        need(engine.runner.spec is None and engine.runner._ahead is None and len(layout)==64 and sum(layout)==16,
             'fresh ordinary target-only64-layer model')
        for layer in model.layers:
            if not layer.is_attn:
                layer.linear_attn.set_gates_compile(True); layer.linear_attn.set_norm_reuse(False)
        for fixture in fixtures['fixtures']: cmp.meta('fixture EOS',sorted(fixture['eos_token_ids']),sorted(engine.eos_ids))
        def parameters(name):
            values=h.parameter_hashes(mx,model,tree_flatten,deadline); h.write(out/(name+'.json'),values)
            result[name]=dict(path=name+'.json',sha256=h.sha(out/(name+'.json'),deadline),scope='producer original-byte parameter receipt')
            return h.parameter_values(values)
        before=parameters('parameters-before')
        for case in seed_cases(bridge,fixtures['fixtures']):
            seed=acquire_seed(mx,cache,context,bridge,support,h,cmp,engine,layout,case,out,deadline,emit)
            result['reference_blocks'].append(seed); h.write(out/'complete.json',result)
            if case['used']==257:
                result['rejections']=rejected_cases(mx,bridge,candidate,h,support,cmp,engine,layout,runtime,seed,out,deadline)
            for width in WIDTHS:
                row=dict(used=case['used'],width=width,status='INCOMPLETE',checks=[])
                result['cases'].append(row); h.write(out/'complete.json',result)
                try:
                    compare_case(mx,bridge,candidate,h,support,cmp,engine,layout,runtime,seed,width,row,out,deadline)
                    row['status']='PASS'
                except BaseException as exc:
                    row.update(status='FAIL',error=dict(type=type(exc).__name__,message=str(exc))); raise
                finally:
                    row['checks']=cmp.rows; cmp.rows=[]; h.write(out/'complete.json',result)
                emit(dict(event='case',used=case['used'],width=width,status=row['status']))
                gc.collect(); mx.synchronize(); mx.clear_cache()
        need(len(result['cases'])==9 and sum(len(c['prefixes']) for c in result['cases'])==42
             and all(c['status']=='PASS' for c in result['cases']),'nine complete cases/all42 prefixes')
        need(engine.runner.decode_model._functions=={} and engine.runner.decode_model._packed is None
             and engine.runner._ahead is None,'all used lengths above original compiled256 route; no ahead')
        cmp.meta('all parameter values unchanged',before,parameters('parameters-after'))
        del parameters,before
    except BaseException as exc:
        result['first_device_error']=dict(type=type(exc).__name__,message=str(exc))
        traceback.clear_frames(exc.__traceback__); raise
    finally:
        if engine is not None:
            engine.runner.batch.clear(); engine.runner.staging.clear(); engine.runner.feats.clear(); engine.runner._ahead=None
        engine=model=layer=None
        gc.collect(); mx.synchronize(); mx.clear_cache(); memory('after-release')
        result['device_cleanup_completed']=True
    support.verify_files(root,source,deadline); support.verify_files(model_path,manifest['files'],deadline)
    for key in ('source_manifest','model_manifest','fixtures'): support.pinned_json(inherited[key])
    validate(plan); result['source_model_pins_rechecked']=True


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path); parser.add_argument('--plan-sha256'); parser.add_argument('--out',type=Path)
    args=parser.parse_args(argv)
    if args.plan is None and args.plan_sha256 is None and args.out is None:
        print(json.dumps(contract(),indent=2)); return 0
    if args.plan is None or args.plan_sha256 is None or args.out is None: parser.error('all three owned-child arguments required')
    args.out.mkdir(parents=True,exist_ok=False); started=time.monotonic(); previous={}; bridge=None
    result=dict(schema='Rowwise-native-verifier-result-v1',status='INCOMPLETE',device_run=False,contract=contract(),
        pid=os.getpid(),ppid=os.getppid(),plan_sha256=args.plan_sha256,reference_blocks=[],cases=[],forward_prefix_qualified=False,
        compiled_qualified=False,rollback_qualified=False,full_model_qualified=False,http_qualified=False,performance_qualified=False,
        source_model_pins_rechecked=False,device_cleanup_completed=False)
    def write(name,value): (args.out/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    write('complete.json',result)
    try:
        need(sha(args.plan)==args.plan_sha256,'exact child plan SHA')
        plan=json.loads(args.plan.read_bytes()); write('plan.json',plan)
        bridge,candidate,h,old,support,original,inherited=validate(plan)
        write('inherited-e024-plan.json',original); write('inherited-e023-plan.json',inherited)
        result['sources']={str(Path(__file__).resolve()):plan['qualifier_sha256'],str(HERE/'candidate.py'):CANDIDATE_SHA,
            str(BRIDGE):BRIDGE_SHA,str(bridge.HELPER):bridge.HELPER_SHA,str(bridge.OLD):bridge.OLD_SHA,
            str(bridge.CORE_PINS):old.MODEL_SOURCE_SHA,original['e023_qualifier']['path']:old.SUPPORT_SHA}
        need(signal.getitimer(signal.ITIMER_REAL)==(0.0,0.0),'existing alarm')
        def deadline():
            if time.monotonic()-started>=plan['deadline_s']: raise TimeoutError('rowwise deadline')
        def stop(signum,frame):
            if signum==signal.SIGALRM: raise TimeoutError('rowwise deadline')
            raise InterruptedError('rowwise cancellation '+signal.Signals(signum).name)
        previous={s:signal.getsignal(s) for s in (signal.SIGALRM,signal.SIGINT,signal.SIGTERM)}
        for sig in previous: signal.signal(sig,stop)
        signal.setitimer(signal.ITIMER_REAL,plan['deadline_s'])
        with (args.out/'events.jsonl').open('x') as stream:
            def emit(row):
                stream.write(json.dumps(dict(**row,elapsed_wall_s=time.monotonic()-started),allow_nan=False)+'\n'); stream.flush()
            device(plan,bridge,candidate,h,old,support,original,inherited,args.out,deadline,emit,result)
        deadline(); need(sha(args.plan)==args.plan_sha256,'plan unchanged')
        need(result['device_cleanup_completed'] and result['source_model_pins_rechecked'],'cleanup/postpins completed')
        result.update(status='PASS',forward_prefix_qualified=True)
    except BaseException as exc:
        incomplete=bridge is not None and isinstance(exc,bridge.IncompleteCoverage)
        result.update(status='INCOMPLETE' if incomplete else 'FAIL',
                      error=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()))
    finally:
        if previous:
            signal.setitimer(signal.ITIMER_REAL,0)
            for sig,value in previous.items(): signal.signal(sig,value)
        if result['status']!='PASS': result['forward_prefix_qualified']=False
        result['elapsed_wall_s']=time.monotonic()-started; write('complete.json',result)
    print(json.dumps(dict(status=result['status'],forward_prefix_qualified=result['forward_prefix_qualified'],cases=len(result['cases']))),flush=True)
    return 0 if result['status']=='PASS' else 1


if __name__=='__main__': raise SystemExit(main())
