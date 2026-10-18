"""One B1/used257 canary for the default-OFF route-preserving candidate.

CPU-safe help. CLI --plan PATH --plan-sha256 SHA --out NEW.
Exact plan E024-route-canary-plan-v1 fields: schema, qualifier_sha256,
candidate {path,sha256}, e024_plan {path,sha256}, deadline_s<=1800.
The frozen E024/E023 plans fix the real model, original dtype and fixtures.
PASS requires all129 output arrays exact, finite and all invariants; no HTTP,
40-point, lifecycle or performance qualification. Numeric rejection preserves
both complete outputs and all comparisons before returning FAIL/exit1.
"""
import argparse
import gc
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

HERE = Path(__file__).resolve().parent
ISOLATION = HERE.parent/'long_decode_isolation/qualify.py'
ISOLATION_SHA = '9d8c17a4bfc7f6ea9d61f79f466cad45059b71c9606d7b8d29c05c40213cad9e'
OLD = HERE.parent/'long_decode_graph'
OLD_SHA = 'c24219f431b08502a26a4f7a83c002214bddd7ea3962d6b1869b313e4c13faf7'
OLD_PLAN_SHA = '5025d37768a5fa74d6e3e256a96ff8b8e4e92b31837183d5ffe031d9473e7123'


def helpers():
    import hashlib
    if hashlib.sha256(ISOLATION.read_bytes()).hexdigest() != ISOLATION_SHA:
        raise ValueError('frozen isolation helper changed')
    spec = importlib.util.spec_from_file_location('e024_route_isolation_support',ISOLATION)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def fixed_plan():
    return dict(schema='E024-route-canary-contract-v1',batch=1,used=257,prefill_tokens=256,
        KV_backing=1536,layers=64,states=128,arms=[dict(name='A',mode='eager',bound=257),
        dict(name='C',mode='compiled',bound=1023)],comparisons=['A-C'],
        configuration=dict(max_len=2048,max_seqs=8,max_batched_tokens=16384,cache_limit_gib=2,
            compile_decode=False,fuse_decode=False,gates_compile=False,norm_reuse=False),
        gate='all129 original-dtype finite/bit equality; source/input/parameter invariants',
        shared_parameter_model=True,independent_evaluated_cache_storage=True,
        no_fuse_used=False,full_matrix_qualified=False,full_model_qualified=False,performance_qualified=False)


def validate(plan):
    h = helpers(); need = h.need
    need(set(plan) == {'schema','qualifier_sha256','candidate','e024_plan','deadline_s'}
         and plan['schema'] == 'E024-route-canary-plan-v1', 'exact route canary plan')
    need(plan['qualifier_sha256'] == h.sha(__file__), 'canary qualifier source SHA')
    need(type(plan['deadline_s']) in (int,float) and math.isfinite(plan['deadline_s'])
         and 0 < plan['deadline_s'] <= 1800, 'bounded deadline')
    item = plan['e024_plan']
    need(set(item) == {'path','sha256'} and item['sha256'] == OLD_PLAN_SHA
         and Path(item['path']).is_absolute() and h.sha(item['path']) == OLD_PLAN_SHA,'original E024 plan')
    entry = plan['candidate']
    need(set(entry) == {'path','sha256'} and entry['path'] == str(HERE/'candidate.py'), 'adjacent route candidate required')
    candidate = h.load(entry['path'],'e024_route_candidate',entry['sha256'])
    old = h.load(OLD/'qualify.py','e024_route_old_qualifier',OLD_SHA)
    original = json.loads(Path(item['path']).read_text()); old.validate_plan(original)
    support = old.load_pinned(original['e023_qualifier'],'e024_route_native',old.SUPPORT_SHA)
    inherited = support.pinned_json(original['e023_plan'])
    need(h.sha(original['candidate']['path']) == old.CANDIDATE_SHA == candidate.ORIGINAL_SHA,
         'unchanged original candidate dependency')
    return h,old,support,candidate,original,inherited


def device(plan,h,old,support,candidate,original,inherited,out,deadline,emit,result):
    need,write = h.need,h.write
    source = support.pinned_json(inherited['source_manifest'])['files']
    manifest = support.pinned_json(inherited['model_manifest'])
    fixtures = support.pinned_json(inherited['fixtures'])
    root = Path(inherited['package_root']).resolve()/'tutorial_code/miniengine'
    model_path = Path(inherited['model']).resolve()
    need(manifest['schema'] == 'E023-model-v1' and Path(manifest['model']).resolve() == model_path
        and manifest['revision'] == model_path.name and Path(fixtures['model']).resolve() == model_path,'model/fixture identity')
    need(set(source) == {str(p.relative_to(root)) for p in root.rglob('*.py')}
        and source == json.loads(candidate.PINS.read_text())['files']
        and h.sha(candidate.PINS) == old.MODEL_SOURCE_SHA,'fixed full core sources')
    required = {'config.json','generation_config.json'} | {str(p.relative_to(model_path)) for p in model_path.rglob('*.safetensors')}
    need(required <= set(manifest['files']) and any(x.endswith('.safetensors') for x in required),'all fixed weight shards')
    support.verify_files(root,source,deadline); support.verify_files(model_path,manifest['files'],deadline)
    result['mlx_environment'] = {k:v for k,v in os.environ.items() if k.startswith('MLX_')}
    need(result['mlx_environment'] == inherited['execution']['environment'],'clean inherited MLX environment')
    sys.path.insert(0,str(root.parent.parent))
    import mlx.core as mx
    import tutorial_code.miniengine as package
    from tutorial_code.miniengine.cache import make_cache, DecodeBatch
    from tutorial_code.miniengine.context import ForwardContext
    from tutorial_code.miniengine.engine.llm_engine import LLMEngine
    from mlx.utils import tree_flatten
    need(Path(package.__file__).resolve().parent == root,'actual package root')
    result['device_run'] = True
    result['device'] = dict(name=mx.device_info().get('device_name'),architecture=mx.metal.device_info().get('architecture'),
        mlx_version=importlib.metadata.version('mlx'))
    need(result['device'] == dict(name='Apple M5 Ultra',architecture='applegpu_g17d',mlx_version='0.32.3')
        and mx.default_device() == mx.gpu,'fixed M5/runtime')
    def phase(name):
        deadline(); mx.synchronize()
        emit(dict(event='memory',phase=name,active_bytes=mx.get_active_memory(),cache_bytes=mx.get_cache_memory(),
            peak_bytes=mx.get_peak_memory(),observation_not_perf=True))
    def params(model,relative):
        rows = h.parameter_hashes(mx,model,tree_flatten,deadline); write(out/relative,rows)
        return rows,dict(path=relative,sha256=h.sha(out/relative,deadline))
    engines=[]; seeds=[]; prefills=[]; inputs={}; outputs={}; contexts={}; graph=witness=None
    cmp = support.Comparator(mx,out)
    try:
        for arm in ('reference','independent-prefill'):
            phase('before-load-'+arm)
            engine = LLMEngine(str(model_path),max_len=2048,max_seqs=8,max_batched_tokens=16384,
                cache_limit_gb=2,dtype=mx.bfloat16,compile_decode=False,fuse_decode=False)
            engines.append(engine)
            for layer in engine.model.layers:
                if not layer.is_attn: layer.linear_attn.set_gates_compile(False); layer.linear_attn.set_norm_reuse(False)
            need(engine.runner.spec is None and len(engine.model.layers)==64,'normal full64 target-only model')
        weights = [dict(tree_flatten(e.model.parameters())) for e in engines]
        need(not any(weights[0][k] is weights[1][k] for k in weights[0]),'independent verification models')
        cmp.compare('independent-loaded-weights',support.snapshot(weights[0]),support.snapshot(weights[1]))
        emit(dict(event='loaded-weights',checks=cmp.rows)); cmp.rows=[]; del weights
        before,result['parameters_before'] = params(engines[0].model,'parameters-before.json')
        cmp.meta('model EOS',sorted(engines[0].eos_ids),sorted(engines[1].eos_ids))
        for f in fixtures['fixtures']: cmp.meta('fixture EOS',sorted(f['eos_token_ids']),sorted(engines[0].eos_ids))
        fixture = old.diagnostic_fixtures(fixtures['fixtures'],256)[0]; write(out/'diagnostic-input.json',fixture)
        for e in engines:
            caches = make_cache(e.model); seeds.append(caches)
            prefills.append(e.model(mx.array(fixture['prompt_ids'],dtype=mx.int32),ForwardContext('prefill',caches),last_only=True))
        cmp.compare('verified-prefill-logits',support.snapshot(prefills[0]),support.snapshot(prefills[1]))
        cmp.compare('verified-prefill-state',support.snapshot(seeds[0]),support.snapshot(seeds[1]))
        result['prefill_checks']=cmp.rows; cmp.rows=[]
        token = int(mx.argmax(prefills[0],axis=-1).item()); model=engines[0].model; batch=DecodeBatch(model)
        batch.rebuild([],[('diagnostic-row-0',seeds[0],1536)])
        canonical = support.snapshot(dict(state=batch.layers,tokens=mx.array([token],dtype=mx.int32),
            positions=mx.array([256],dtype=mx.int32),lens=mx.array([257],dtype=mx.int32)))
        need(len(canonical['arrays'])==131,'all canonical input arrays')
        result['canonical']=h.dump(mx,out,'canonical.safetensors',canonical,deadline)
        seeds.clear(); prefills.clear(); engines.pop(); del e,engine,layer,caches,batch,canonical
        gc.collect(); mx.synchronize(); mx.clear_cache(); phase('after-verified-prefill-release')
        after,result['parameters_after_prefill']=params(model,'parameters-after-prefill.json')
        cmp.meta('prefill/unchanged-model',h.parameter_values(before),h.parameter_values(after)); del after
        graph = candidate.RoutePreservingDecode(model,enabled=True); r=graph._runtime
        result['model_source_receipt']=graph.source_receipt
        layout=tuple(layer.is_attn for layer in model.layers)
        for name,mode,bound in (('A','eager',257),('C','compiled',1023)):
            arrays=mx.load(str(out/'canonical.safetensors')); mx.eval(*arrays.values())
            inputs[name]=dict(metadata=result['canonical']['metadata'],arrays=arrays)
            union=mx.get_array_buffer_size(*[x['arrays'] for x in inputs.values()])
            need(union==sum(mx.get_array_buffer_size(x['arrays']) for x in inputs.values()),'independent arm input storage')
            contexts[name]=h.context_from_input(arrays,layout,r,257)
            result['arms'][name]=dict(mode=mode,bound=bound,status='PREPARED',input_buffer_bytes=mx.get_array_buffer_size(arrays),
                input_union_bytes=union,input_before=h.dump(mx,out,f'arms/{name}/input-before.safetensors',inputs[name],deadline))
            cmp.compare(name+'/canonical',dict(metadata=inputs[name]['metadata'],arrays=mx.load(str(out/'canonical.safetensors'))),inputs[name]); cmp.rows=[]
        for name in ('A','C'):
            phase('before-'+name); row=result['arms'][name]; ctx=contexts[name]; ids=inputs[name]['arrays']['root/tokens'][:]
            need(graph._eligible(ids,ctx),'fixed canary graph eligibility')
            if name=='A':
                logits=model(ids,ctx); route=dict(kind='eager',max_len=257,compiled_invocations=0)
            else:
                witness=old.GraphWitness(graph,emit); witness.phase='C-compiled1023'; logits=witness(ids,ctx)
                route=dict(kind='route-preserving-E024-candidate',max_len=1023,compiled_invocations=witness.functions.invocations,
                    actual_compile_objects=witness.functions.compiled_objects)
            outputs[name]=support.snapshot(dict(logits=logits,state=ctx.caches))
            need(len(outputs[name]['arrays'])==129,'full output logits and128 states')
            row['output']=h.dump(mx,out,f'arms/{name}/output.safetensors',outputs[name],deadline)
            row['input_after']=h.dump(mx,out,f'arms/{name}/input-after.safetensors',inputs[name],deadline)
            cmp.meta(name+'/input-unchanged',row['input_before']['tensors'],row['input_after']['tensors'])
            after,row['parameters_after']=params(model,f'parameters-after-{name}.json')
            cmp.meta(name+'/model-unchanged',h.parameter_values(before),h.parameter_values(after)); del after
            row.update(status='COMPLETE',route=route,greedy_ids=mx.argmax(logits,axis=-1).tolist())
            phase('after-'+name); write(out/'progress.json',result); del logits,ids,ctx
        result['comparisons']=[h.compare_finite_outputs(support,mx,out,'A-C',outputs['A'],outputs['C'])]
        write(out/'comparisons.json',result['comparisons'])
        result['numerical_equal']=result['comparisons'][0]['numerical_equal']; result['diagnostic_complete']=True
    finally:
        if graph is not None: graph._functions.clear()
        contexts.clear(); inputs.clear(); outputs.clear(); seeds.clear(); prefills.clear(); engines.clear()
    support.verify_files(root,source,deadline); support.verify_files(model_path,manifest['files'],deadline)
    for field in ('source_manifest','model_manifest','fixtures'): support.pinned_json(inherited[field])
    validate(plan); result['source_model_pins_rechecked']=True


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--plan',type=Path,required=True); ap.add_argument('--plan-sha256',required=True); ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args(argv); h=helpers(); h.need(h.sha(args.plan)==args.plan_sha256,'plan SHA')
    plan=json.loads(args.plan.read_text()); h,old,support,candidate,original,inherited=validate(plan)
    h.need(signal.getitimer(signal.ITIMER_REAL)==(0.0,0.0),'existing alarm')
    args.out.mkdir(parents=True,exist_ok=False); started=time.monotonic()
    result=dict(schema='E024-route-canary-result-v1',status='INCOMPLETE',contract=fixed_plan(),device_run=False,
        pid=os.getpid(),ppid=os.getppid(),arms={},comparisons=[],diagnostic_complete=False,canary_qualified=False,
        numerical_equal=None,source_model_pins_rechecked=False,full_model_qualified=False,performance_qualified=False,
        plan_sha256=args.plan_sha256,qualifier_sha256=plan['qualifier_sha256'],sources={
            str(HERE/'qualify.py'):plan['qualifier_sha256'],str(HERE/'candidate.py'):plan['candidate']['sha256'],
            str(ISOLATION):ISOLATION_SHA,str(OLD/'qualify.py'):OLD_SHA,str(candidate.ORIGINAL):candidate.ORIGINAL_SHA,
            original['e023_qualifier']['path']:old.SUPPORT_SHA,str(candidate.PINS):old.MODEL_SOURCE_SHA})
    h.write(args.out/'plan.json',plan); h.write(args.out/'inherited-e024-plan.json',original); h.write(args.out/'inherited-e023-plan.json',inherited)
    def deadline(*args):
        if args or time.monotonic()-started>=plan['deadline_s']: raise TimeoutError('route canary absolute deadline')
    def stop(signum,frame):
        if signum==signal.SIGALRM: deadline(signum)
        raise InterruptedError('route canary cancellation '+signal.Signals(signum).name)
    previous={s:signal.getsignal(s) for s in (signal.SIGALRM,signal.SIGINT,signal.SIGTERM)}
    with (args.out/'events.jsonl').open('x') as stream:
        def emit(row): stream.write(json.dumps(dict(**row,elapsed_wall_s=time.monotonic()-started))+'\n'); stream.flush()
        try:
            for s in previous: signal.signal(s,stop)
            signal.setitimer(signal.ITIMER_REAL,plan['deadline_s'])
            device(plan,h,old,support,candidate,original,inherited,args.out,deadline,emit,result)
            h.need(h.sha(args.plan)==args.plan_sha256,'plan changed')
            h.need(result['numerical_equal'] is True,'route canary original-dtype bit gate rejected')
            result.update(status='PASS',canary_qualified=True)
        except BaseException as error:
            result.update(status='FAIL',error=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc()))
        finally:
            try:
                gc.collect(); mx=sys.modules.get('mlx.core')
                if mx is not None:
                    mx.synchronize(); mx.clear_cache(); result['final_active_bytes']=mx.get_active_memory(); result['peak_bytes']=mx.get_peak_memory()
            except BaseException as error:
                result.update(status='FAIL',canary_qualified=False,cleanup_error=dict(type=type(error).__name__,message=str(error)))
            finally:
                signal.setitimer(signal.ITIMER_REAL,0)
                for s,handler in previous.items(): signal.signal(s,handler)
                result['elapsed_wall_s']=time.monotonic()-started; h.write(args.out/'complete.json',result)
    print(json.dumps(dict(status=result['status'],out=str(args.out))),flush=True)
    return 0 if result['status']=='PASS' else 1


if __name__=='__main__': raise SystemExit(main())
