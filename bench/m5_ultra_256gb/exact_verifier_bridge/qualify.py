"""Default-OFF native C1 -> existing prefill verifier numerical bridge.

No arguments describes the contract. An externally owned child executes only
with --plan FILE --plan-sha256 SHA --out NEW. No drafter, new kernel, scheduler,
rollback, service control or timing qualification is installed here.
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
HELPER = HERE.parent/'long_decode_isolation/qualify.py'
HELPER_SHA = '9d8c17a4bfc7f6ea9d61f79f466cad45059b71c9606d7b8d29c05c40213cad9e'
OLD = HERE.parent/'long_decode_graph/qualify.py'
OLD_SHA = 'c24219f431b08502a26a4f7a83c002214bddd7ea3962d6b1869b313e4c13faf7'
OLD_PLAN_SHA = '5025d37768a5fa74d6e3e256a96ff8b8e4e92b31837183d5ffe031d9473e7123'
CORE_PINS = HERE.parent/'long_decode_graph/model-source-pins.json'
WIDTHS = (2, 4, 8)
STARTS = (1, 129)


def need(ok, message):
    if not ok: raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path, name, digest):
    need(sha(path) == digest, 'pinned executable source changed: '+str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def contract():
    return dict(schema='Exact-verifier-bridge-contract-v1',enabled_by_default=False,
        widths=list(WIDTHS),decode_starts=list(STARTS),expected_cases=12,
        fixtures='shortest and longest original D512, ties by fixture_id; both starts for each',
        oracle='original runner.decode_model, one actual token/position per call',
        candidate='SpecRunner target expression: snapshot, model(block, prefill, capture=())',
        inputs='fixed IDs generated only by an independent original C1 continuation',
        normal_eos=True,cap=512,early_eos='INCOMPLETE; no fabricated continuation',
        configuration=dict(max_seqs=1,max_len=2048,max_batched_tokens=16384,cache_limit_gib=2,
            compile_decode=True,fuse_decode=True,gates_compile=True,norm_reuse=False),
        arrays='all T full-vocabulary logits and all128 final state arrays, including padded KV backing',
        parameter_check='all original-dtype parameter values; handle IDs recorded, not compared',
        excluded=['drafter/features','accept/rollback','intermediate prefix states','scheduler/ahead',
                  'E024 graph','E028 refill','GDN prework','HTTP','performance'],
        rollback_qualified=False,full_model_qualified=False,http_qualified=False,performance_qualified=False)


def validate(plan):
    need(set(plan) == {'schema','qualifier_sha256','e024_plan','deadline_s'}
         and plan['schema'] == 'Exact-verifier-bridge-plan-v1', 'exact bridge plan')
    need(plan['qualifier_sha256'] == sha(__file__), 'actual qualifier source')
    budget = plan['deadline_s']
    need(type(budget) in (int,float) and math.isfinite(budget) and 0 < budget <= 1800, 'bounded child deadline')
    ref = plan['e024_plan']
    need(set(ref) == {'path','sha256'} and Path(ref['path']).is_absolute()
         and ref['sha256'] == OLD_PLAN_SHA == sha(ref['path']), 'unchanged inherited E024 plan')
    h = load(HELPER,'_exact_bridge_raw_helpers',HELPER_SHA)
    old = load(OLD,'_exact_bridge_old_contract',OLD_SHA)
    original = json.loads(Path(ref['path']).read_bytes()); old.validate_plan(original)
    support = old.load_pinned(original['e023_qualifier'],'_exact_bridge_comparator',old.SUPPORT_SHA)
    inherited = support.pinned_json(original['e023_plan'])
    need(h.sha(CORE_PINS) == old.MODEL_SOURCE_SHA, 'frozen complete core manifest')
    return h,old,support,original,inherited


def selected(fixtures):
    need(len(fixtures) == 8 and len({f['fixture_id'] for f in fixtures}) == 8, 'eight distinct original fixtures')
    for f in fixtures:
        need(f['normal_eos'] is True and type(f['max_output_tokens']) is int and f['max_output_tokens'] == 512
             and 886 <= len(f['prompt_ids']) <= 978
             and all(type(t) is int and t >= 0 for t in f['prompt_ids']), 'original D512 input contract')
    ordered = sorted(fixtures,key=lambda f:(len(f['prompt_ids']),f['fixture_id']))
    return [(f,start) for f in (ordered[0],ordered[-1]) for start in STARTS]


class IncompleteCoverage(RuntimeError):
    pass


def reference_tokens(first, eos_ids, start, advance, anchor, emit):
    """Only reference output feeds future input; candidate has no callback here."""
    token=first; raw=[first]; block=[]
    for step in range(1,start+max(WIDTHS)):
        if token in eos_ids or len(raw) >= 512:
            raise IncompleteCoverage('normal EOS/cap before complete eight-token teacher block')
        if step == start: anchor(token,step)
        if step >= start: block.append(token)
        next_token=advance(token,step)
        need(type(next_token) is int and next_token >= 0, 'actual reference greedy ID')
        emit(dict(event='reference-token',decode_step=step,input_token=token,output_token=next_token))
        raw.append(next_token); token=next_token
    need(len(block) == 8, 'complete independent teacher block')
    return raw,block


def canonical_state(caches, layout, *, staging=False):
    """Only field names and singleton B dimension change; no casting or KV slice."""
    need(len(caches) == len(layout) == 64 and sum(layout) == 16, 'full64-layer state')
    if staging:
        return [dict(K=c.keys,V=c.values) if attn else dict(conv=c.conv_state[None],S=c.S[None])
                for c,attn in zip(caches,layout)]
    return [dict(K=c.K,V=c.V) if attn else dict(conv=c.conv,S=c.S) for c,attn in zip(caches,layout)]


def verify_forward(model, tokens, caches, context_factory):
    """The existing target transaction, excluding drafter features and rollback."""
    for c in caches: c.snapshot()
    return model(tokens,context_factory('prefill',caches),capture=())


def device(plan,h,old,support,original,inherited,out,deadline,emit,result):
    source=support.pinned_json(inherited['source_manifest'])['files']
    manifest=support.pinned_json(inherited['model_manifest']); fixtures=support.pinned_json(inherited['fixtures'])
    (out/'fixtures.json').write_bytes(Path(inherited['fixtures']['path']).read_bytes())
    need(sha(out/'fixtures.json')==inherited['fixtures']['sha256'],'exact original fixture bytes retained')
    root=Path(inherited['package_root']).resolve()/'tutorial_code/miniengine'; model_path=Path(inherited['model']).resolve()
    need(manifest['schema']=='E023-model-v1' and Path(manifest['model']).resolve()==model_path
         and manifest['revision']==model_path.name and Path(fixtures['model']).resolve()==model_path, 'fixed model identity')
    need(source==json.loads(CORE_PINS.read_bytes())['files']
         and set(source)=={str(p.relative_to(root)) for p in root.rglob('*.py')}, 'complete frozen core')
    required={'config.json','generation_config.json'}|{str(p.relative_to(model_path)) for p in model_path.rglob('*.safetensors')}
    need(required<=set(manifest['files']) and any(x.endswith('.safetensors') for x in required),'all model shard pins')
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
         and mx.default_device()==mx.gpu,'fixed actual runtime/device')
    runtime=SimpleNamespace(BatchKV=cache.BatchKV,BatchState=cache.BatchState,ForwardContext=context.ForwardContext)
    cmp=support.Comparator(mx,out); engine=model=None
    def memory(phase):
        mx.synchronize(); emit(dict(event='memory',phase=phase,active_bytes=mx.get_active_memory(),
            peak_bytes=mx.get_peak_memory(),cache_bytes=mx.get_cache_memory(),not_performance=True))
    try:
        memory('before-load')
        engine=LLMEngine(str(model_path),max_len=2048,max_seqs=1,max_batched_tokens=16384,
                         cache_limit_gb=2,dtype=mx.bfloat16,compile_decode=True,fuse_decode=True)
        model=engine.model; layout=tuple(layer.is_attn for layer in model.layers)
        need(engine.runner.spec is None and engine.runner._ahead is None and len(layout)==64 and sum(layout)==16,'ordinary target-only64-layer model')
        for layer in model.layers:
            if not layer.is_attn:
                layer.linear_attn.set_gates_compile(True); layer.linear_attn.set_norm_reuse(False)
        for f in fixtures['fixtures']: cmp.meta('fixture EOS',sorted(f['eos_token_ids']),sorted(engine.eos_ids))
        def parameters(name):
            values=h.parameter_hashes(mx,model,tree_flatten,deadline); path=name+'.json'; h.write(out/path,values)
            result[name]=dict(path=path,sha256=h.sha(out/path,deadline),scope='all model parameter original bytes; producer receipt')
            return h.parameter_values(values)
        before=parameters('parameters-before')
        for fixture,start in selected(fixtures['fixtures']):
            seed=acquire_seed(mx,cache,context,support,h,engine,layout,fixture,start,out,deadline,emit)
            result['reference_blocks'].append(seed); h.write(out/'complete.json',result)
            for width in WIDTHS:
                row=dict(fixture_id=fixture['fixture_id'],start=start,width=width,position=seed['position'],
                         tokens=seed['tokens'][:width],canonical=seed['canonical'],status='INCOMPLETE',checks=[])
                result['cases'].append(row); h.write(out/'complete.json',result)
                try:
                    compare_case(mx,cache,context,runtime,support,h,cmp,engine,layout,seed,width,row,out,deadline)
                    row['status']='PASS'
                except BaseException as exc:
                    row.update(status='FAIL',error=dict(type=type(exc).__name__,message=str(exc)))
                    raise
                finally:
                    row['checks']=cmp.rows; cmp.rows=[]; h.write(out/'complete.json',result)
                emit(dict(event='case',**row)); gc.collect(); mx.synchronize(); mx.clear_cache()
        need(len(result['cases'])==12 and all(x['status']=='PASS' for x in result['cases']), 'all12 complete forward cases')
        cmp.meta('all parameters unchanged',before,parameters('parameters-after'))
        del parameters,before
    except BaseException as exc:
        result['first_device_error']=dict(type=type(exc).__name__,message=str(exc))
        # Unwound comparison/helper frames can otherwise keep complete state
        # arrays alive while the release receipt is collected. Preserve raw
        # artifacts and traceback locations, not their tensor-bearing locals.
        traceback.clear_frames(exc.__traceback__)
        raise
    finally:
        # No hook/patch exists. Clearing the actual runner also releases any
        # unexpected pending state; a failure never receives forward qualification.
        if engine is not None:
            engine.runner.batch.clear(); engine.runner.staging.clear(); engine.runner.feats.clear(); engine.runner._ahead=None
        engine=model=None
        # The loop variable is an nn.Module and otherwise retains a last layer.
        layer=None
        gc.collect(); mx.synchronize(); mx.clear_cache(); memory('after-release')
        result['device_cleanup_completed']=True
    support.verify_files(root,source,deadline); support.verify_files(model_path,manifest['files'],deadline)
    for key in ('source_manifest','model_manifest','fixtures'): support.pinned_json(inherited[key])
    validate(plan); result['source_model_pins_rechecked']=True


def acquire_seed(mx,cache,context,support,h,engine,layout,fixture,start,out,deadline,emit):
    """Reference-only continuation and canonical on-disk state, before either arm."""
    model=engine.model; prompt=list(fixture['prompt_ids']); prefix=f"inputs/{fixture['fixture_id']}-s{start}"
    caches=cache.make_cache(model); batch=cache.DecodeBatch(model); saved={}
    try:
        logits=model(mx.array(prompt,dtype=mx.int32),context.ForwardContext('prefill',caches),last_only=True)
        mx.eval(logits,*support.snapshot(caches)['arrays'].values()); token=int(mx.argmax(logits,axis=-1).item())
        batch.rebuild([],[('teacher-reference',caches,len(prompt)+512)]); mx.eval(*batch.arrays()); caches.clear()
        def advance(input_token,step):
            deadline(); p=len(prompt)+step-1
            ctx=context.ForwardContext('decode',batch.layers,positions=mx.array([p],dtype=mx.int32),
                lens=mx.array([p+1],dtype=mx.int32),max_len=p+1)
            values=engine.runner.decode_model(mx.array([input_token],dtype=mx.int32),ctx)
            mx.eval(values,*batch.arrays()); batch.lens=[p+1]
            return int(mx.argmax(values,axis=-1).item())
        def anchor(input_token,step):
            p=len(prompt)+step-1
            need(batch.lens==[p] and p>256 and p+8<=2048,'actual committed anchor and context')
            snapshot=support.snapshot(dict(state=canonical_state(batch.layers,layout),tokens=mx.array([input_token],dtype=mx.int32),
                positions=mx.array([p],dtype=mx.int32),lens=mx.array([p+1],dtype=mx.int32)))
            need(len(snapshot['arrays'])==131,'all128 state plus3 inputs')
            saved.update(position=p,canonical=h.dump(mx,out,prefix+'/canonical.safetensors',snapshot,deadline))
        raw,tokens=reference_tokens(token,engine.eos_ids,start,advance,anchor,
            lambda row:emit(dict(fixture_id=fixture['fixture_id'],start=start,**row)))
        return dict(fixture_id=fixture['fixture_id'],start=start,prompt_ids=prompt,raw_ids=raw,tokens=tokens,**saved)
    finally:
        caches.clear(); batch.clear()


def compare_case(mx,cache,context,runtime,support,h,cmp,engine,layout,seed,width,row,out,deadline):
    path=out/seed['canonical']['path']; need(h.sha(path,deadline)==seed['canonical']['sha256'],'saved canonical bytes')
    left=mx.load(str(path)); right=mx.load(str(path)); mx.eval(*left.values(),*right.values())
    need(mx.get_array_buffer_size(list(left.values()),list(right.values())) ==
         mx.get_array_buffer_size(list(left.values()))+mx.get_array_buffer_size(list(right.values())), 'independent evaluated arm storage')
    position=seed['position']; tokens=list(seed['tokens'][:width]); tag=f"{seed['fixture_id']}/s{seed['start']}/T{width}"
    cmp.compare(tag+'/same-seed',dict(metadata=seed['canonical']['metadata'],arrays=left),dict(metadata=seed['canonical']['metadata'],arrays=right))
    ref=h.context_from_input(left,layout,runtime,position+1)
    cand_decode=h.context_from_input(right,layout,runtime,position+1)
    staging=[]
    for c,attn in zip(cand_decode.caches,layout):
        target=cache.AttnCache() if attn else cache.GDNCache(); target.offset=position
        if attn: target.keys,target.values=c.K,c.V
        else: target.conv_state,target.S=c.conv[0],c.S[0]
        staging.append(target)
    cmp.compare(tag+'/staging-representation',support.snapshot(canonical_state(ref.caches,layout)),
                support.snapshot(canonical_state(staging,layout,staging=True)))
    outputs=[]
    for index,token in enumerate(tokens):
        deadline(); p=position+index
        ref=context.ForwardContext('decode',ref.caches,positions=mx.array([p],dtype=mx.int32),
            lens=mx.array([p+1],dtype=mx.int32),max_len=p+1)
        logits=engine.runner.decode_model(mx.array([token],dtype=mx.int32),ref)
        mx.eval(logits,*support.snapshot(ref.caches)['arrays'].values()); outputs.append(logits)
    reference_logits=mx.concatenate(outputs,axis=0)
    deadline()
    candidate_logits=verify_forward(engine.model,mx.array(tokens,dtype=mx.int32),staging,context.ForwardContext)
    mx.eval(candidate_logits,*support.snapshot(staging)['arrays'].values())
    need(reference_logits.shape==candidate_logits.shape==(width,engine.cfg.vocab_size),'complete T x full vocabulary')
    need(all(c.record is True and c.tape is not None and len(c.tape)==6 for c,attn in zip(staging,layout) if not attn),
         'actual snapshot/record verify transaction')
    cmp.meta(tag+'/actual-offsets',[position+width]*64,[c.offset for c in staging])
    row['verify_bookkeeping']=[dict(kind=type(c).__name__,offset=c.offset,
        record=None if attn else c.record,tape_arrays=[] if attn else [dict(shape=list(x.shape),dtype=str(x.dtype)) for x in c.tape])
        for c,attn in zip(staging,layout)]
    a=support.snapshot(dict(logits=reference_logits,state=canonical_state(ref.caches,layout),offsets=[position+width]*64))
    b=support.snapshot(dict(logits=candidate_logits,state=canonical_state(staging,layout,staging=True),offsets=[c.offset for c in staging]))
    need(len(a['arrays'])==len(b['arrays'])==129,'full logits and128 final state arrays')
    # Persist BOTH complete outputs before invoking the fail-first bit comparator.
    row['reference_output']=h.dump(mx,out,'cases/'+tag+'/reference.safetensors',a,deadline)
    row['candidate_output']=h.dump(mx,out,'cases/'+tag+'/candidate.safetensors',b,deadline)
    row['reference_greedy_ids']=mx.argmax(reference_logits,axis=-1).tolist()
    row['candidate_greedy_ids']=mx.argmax(candidate_logits,axis=-1).tolist()
    cmp.compare(tag+'/full-output',a,b)
    cmp.meta(tag+'/independent-reference-reproduction',seed['raw_ids'][seed['start']:seed['start']+width],row['reference_greedy_ids'])
    canonical=mx.load(str(path)); mx.eval(*canonical.values())
    for label,arrays in (('reference',left),('candidate',right)):
        cmp.compare(tag+'/'+label+'-seed-unchanged',dict(metadata=seed['canonical']['metadata'],arrays=canonical),
                    dict(metadata=seed['canonical']['metadata'],arrays=arrays))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path); parser.add_argument('--plan-sha256'); parser.add_argument('--out',type=Path)
    args=parser.parse_args(argv)
    if args.plan is None and args.plan_sha256 is None and args.out is None:
        print(json.dumps(contract(),indent=2)); return 0
    if args.plan is None or args.plan_sha256 is None or args.out is None: parser.error('all three owned-child arguments are required')
    args.out.mkdir(parents=True,exist_ok=False); started=time.monotonic(); previous={}
    result=dict(schema='Exact-verifier-bridge-result-v1',status='INCOMPLETE',device_run=False,contract=contract(),
        pid=os.getpid(),ppid=os.getppid(),plan_sha256=args.plan_sha256,reference_blocks=[],cases=[],forward_qualified=False,
        rollback_qualified=False,full_model_qualified=False,http_qualified=False,performance_qualified=False,
        source_model_pins_rechecked=False,device_cleanup_completed=False)
    def write(name,value): (args.out/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    write('complete.json',result)
    try:
        need(sha(args.plan)==args.plan_sha256,'exact child plan SHA')
        plan=json.loads(args.plan.read_bytes()); write('plan.json',plan)
        h,old,support,original,inherited=validate(plan)
        write('inherited-e024-plan.json',original); write('inherited-e023-plan.json',inherited)
        result['sources']={str(Path(__file__).resolve()):plan['qualifier_sha256'],str(HELPER):HELPER_SHA,str(OLD):OLD_SHA,
            str(CORE_PINS):old.MODEL_SOURCE_SHA,original['e023_qualifier']['path']:old.SUPPORT_SHA}
        need(signal.getitimer(signal.ITIMER_REAL)==(0.0,0.0),'existing alarm')
        def deadline():
            if time.monotonic()-started>=plan['deadline_s']: raise TimeoutError('exact bridge deadline')
        def stop(signum,frame):
            if signum==signal.SIGALRM: raise TimeoutError('exact bridge deadline')
            raise InterruptedError('exact bridge cancellation '+signal.Signals(signum).name)
        previous={s:signal.getsignal(s) for s in (signal.SIGALRM,signal.SIGINT,signal.SIGTERM)}
        for sig in previous: signal.signal(sig,stop)
        signal.setitimer(signal.ITIMER_REAL,plan['deadline_s'])
        with (args.out/'events.jsonl').open('x') as stream:
            def emit(row):
                stream.write(json.dumps(dict(**row,elapsed_wall_s=time.monotonic()-started),allow_nan=False)+'\n'); stream.flush()
            device(plan,h,old,support,original,inherited,args.out,deadline,emit,result)
        deadline(); need(sha(args.plan)==args.plan_sha256,'plan unchanged')
        need(result['device_cleanup_completed'] and result['source_model_pins_rechecked'],'cleanup and postpins completed')
        result.update(status='PASS',forward_qualified=True)
    except IncompleteCoverage as exc:
        result.update(status='INCOMPLETE',error=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()))
    except BaseException as exc:
        result.update(status='FAIL',error=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()))
    finally:
        if previous:
            signal.setitimer(signal.ITIMER_REAL,0)
            for sig,value in previous.items(): signal.signal(sig,value)
        if result['status']!='PASS': result['forward_qualified']=False
        result['elapsed_wall_s']=time.monotonic()-started; write('complete.json',result)
    print(json.dumps(dict(status=result['status'],forward_qualified=result['forward_qualified'],cases=len(result['cases']))),flush=True)
    return 0 if result['status']=='PASS' else 1


if __name__=='__main__': raise SystemExit(main())
