"""Fixed E024 route matrix and direct decode reuse; no HTTP/performance claim.

CPU-safe help, --plan PATH --plan-sha256 SHA --out NEW. Exact five-field plan:
schema E024-route-matrix-plan-v1, qualifier_sha256, candidate {path,sha256},
e024_plan {path,sha256}, deadline_s<=1800. Frozen canary/candidate stay unchanged.
40 matrix points +3 uniform crossing +3 mixed crossing +6 batch reuse =52.
First failure stops. PASS points have full device bit checks and scalar hashes;
only the explicitly selected ten full snapshots support offline raw-byte audit.
Failure input dumps are the original handles observed at failure, not historical
before payloads; separately recorded pre-execution hashes retain that boundary.
The external owner enforces full-model resource/deadline/reap contracts.
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

HERE=Path(__file__).resolve().parent
CANARY_SHA='fe8e1c044e97885ec7c97fddce04a6e34f5b9067be4d982c5f6ad1071c59c801'
CANDIDATE_SHA='6531be25acb53713d6ddcd930d2c778f6640a3b1f53304fd5375d20b044525c3'
USED=(257,1023,1024,1280,1536)
MIXED=(257,511,767,900,978,1000,1022,1023)
RAW_POINTS={'matrix-B1-used257':'canonical','matrix-B1-used1023':'both',
            'matrix-B1-used1024':'both','mixed-used1024':'both'}


def helper():
    path=HERE/'qualify.py'
    if hashlib.sha256(path.read_bytes()).hexdigest()!=CANARY_SHA: raise ValueError('fixed canary source')
    spec=importlib.util.spec_from_file_location('e024_matrix_canary_support',path)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def point_plan():
    points=[]
    for used in USED:
        points += [dict(name=f'matrix-B{B}-used{used}',kind='matrix',B=B,lens=[used]*B) for B in range(1,9)]
        if used==1023:
            points += [dict(name=f'uniform-used{n}',kind='uniform-chain',B=1,lens=[n]) for n in (1023,1024,1025)]
        if used in (1023,1024):
            points += [dict(name=f'reuse-used{used}-{i}-B{B}',kind='batch-reuse',B=B,lens=[used]*B)
                       for i,B in enumerate((1,8,1))]
    points += [dict(name=f'mixed-used{1023+i}',kind='mixed-chain',B=8,lens=[n+i for n in MIXED]) for i in range(3)]
    return points


def fixed_plan():
    return dict(schema='E024-route-matrix-contract-v1',points=point_plan(),expected_points=52,
        matrix_points=40,uniform_crossing_points=3,mixed_crossing_points=3,batch_reuse_points=6,
        layers=64,state_arrays=128,output_arrays=129,input_arrays=131,KV_backing=1536,
        prefill_checkpoints=[u-1 for u in USED],max_prefill_tokens_per_model=18730,
        max_decode_output_tokens_per_model=227,history='repeat each frozen D prompt IDs and truncate; diagnostic extension, not generation',
        configuration=dict(max_len=2048,max_seqs=8,max_batched_tokens=16384,cache_limit_gib=2,
            compile_decode=False,fuse_decode=False,gates_compile=False,norm_reuse=False),
        raw_points=RAW_POINTS,expected_pass_full_snapshots=10,
        pass_evidence='All arrays device finite/unsigned-bit comparison plus full-byte SHA receipts; sparse full raw snapshots only',
        failure_evidence='First failure retains original input handles as observed at failure, available outputs and genuine pre-execution hash receipts; no historical before-payload claim, no later points',
        two_callable_families_not_two_machine_graphs=True,parameter_replacement_qualified=False,
        ahead_qualified=False,scheduler_lifecycle_qualified=False,full_model_qualified=False,performance_qualified=False)


def validate(plan):
    q=helper(); h=q.helpers(); need=h.need
    need(set(plan)=={'schema','qualifier_sha256','candidate','e024_plan','deadline_s'}
         and plan['schema']=='E024-route-matrix-plan-v1','exact matrix plan')
    need(plan['qualifier_sha256']==h.sha(__file__),'matrix qualifier source SHA')
    need(type(plan['deadline_s']) in (int,float) and math.isfinite(plan['deadline_s']) and 0<plan['deadline_s']<=1800,'bounded deadline')
    entry=plan['candidate']; oldplan=plan['e024_plan']
    need(entry==dict(path=str(HERE/'candidate.py'),sha256=CANDIDATE_SHA),'frozen route candidate')
    need(set(oldplan)=={'path','sha256'} and oldplan['sha256']==q.OLD_PLAN_SHA
         and Path(oldplan['path']).is_absolute() and h.sha(oldplan['path'])==q.OLD_PLAN_SHA,'original E024 plan')
    candidate=h.load(entry['path'],'e024_matrix_candidate',CANDIDATE_SHA)
    old=h.load(q.OLD/'qualify.py','e024_matrix_old_qualifier',q.OLD_SHA)
    original=json.loads(Path(oldplan['path']).read_text()); old.validate_plan(original)
    support=old.load_pinned(original['e023_qualifier'],'e024_matrix_native',old.SUPPORT_SHA)
    inherited=support.pinned_json(original['e023_plan'])
    need(h.sha(candidate.ORIGINAL)==candidate.ORIGINAL_SHA==old.CANDIDATE_SHA,'original candidate support pin')
    return q,h,old,support,candidate,original,inherited


def clone_arrays(mx,arrays):
    """Materialized identity gather, not array(x)/contiguous identity elision."""
    copied={}
    for name,value in arrays.items():
        copy=mx.take(value,mx.arange(value.shape[0],dtype=mx.int32),axis=0)
        mx.eval(copy)
        if copy is value or mx.get_array_buffer_size(value,copy)!=mx.get_array_buffer_size(value)+mx.get_array_buffer_size(copy):
            raise ValueError('identity gather failed to own independent storage: '+name)
        copied[name]=copy
    return copied


def hashes(mx,snapshot,deadline):
    result={}
    for name,value in snapshot['arrays'].items():
        deadline(); mx.eval(value)
        unsigned={2:mx.uint16,4:mx.uint32}[value.itemsize]
        bits=mx.contiguous(value).view(unsigned); mx.eval(bits)
        result[name]=dict(shape=list(value.shape),dtype=str(value.dtype),bytes=value.nbytes,
            sha256=hashlib.sha256(memoryview(bits).cast('B')).hexdigest(),finite=bool(mx.all(mx.isfinite(value)).item()))
        if not result[name]['finite']: raise ValueError('nonfinite point tensor '+name)
        del bits
    return dict(metadata=snapshot['metadata'],arrays=result)


def own_contexts(mx,h,r,layout,canonical):
    snapshots=[dict(metadata=canonical['metadata'],arrays=clone_arrays(mx,canonical['arrays'])) for _ in range(2)]
    contexts=[h.context_from_input(snap['arrays'],layout,r,max(snap['arrays']['root/lens'].tolist())) for snap in snapshots]
    return contexts,[snap['arrays']['root/tokens'] for snap in snapshots]


def advance_chain(mx,contexts,own_tokens,lens):
    if len(contexts)!=2 or len(own_tokens)!=2: raise ValueError('two independent chain arms')
    for ctx in contexts:
        ctx.positions=mx.array([n-1 for n in lens],dtype=mx.int32)
        ctx.lens=mx.array(lens,dtype=mx.int32); ctx.max_len=max(lens)
    return [mx.array(values,dtype=mx.int32) for values in own_tokens]


def device(plan,q,h,old,support,candidate,original,inherited,out,deadline,emit,result):
    need,write=h.need,h.write
    source=support.pinned_json(inherited['source_manifest'])['files']; manifest=support.pinned_json(inherited['model_manifest'])
    fixtures=support.pinned_json(inherited['fixtures']); root=Path(inherited['package_root']).resolve()/'tutorial_code/miniengine'
    model_path=Path(inherited['model']).resolve()
    need(manifest['schema']=='E023-model-v1' and Path(manifest['model']).resolve()==model_path
         and manifest['revision']==model_path.name and Path(fixtures['model']).resolve()==model_path,'model/fixture identity')
    need(set(source)=={str(p.relative_to(root)) for p in root.rglob('*.py')}
         and source==json.loads(candidate.PINS.read_text())['files'] and h.sha(candidate.PINS)==old.MODEL_SOURCE_SHA,'fixed core tree')
    required={'config.json','generation_config.json'}|{str(p.relative_to(model_path)) for p in model_path.rglob('*.safetensors')}
    need(required<=set(manifest['files']) and any(x.endswith('.safetensors') for x in required),'all model shards')
    support.verify_files(root,source,deadline); support.verify_files(model_path,manifest['files'],deadline)
    result['mlx_environment']={k:v for k,v in os.environ.items() if k.startswith('MLX_')}
    need(result['mlx_environment']==inherited['execution']['environment'],'exact clean MLX environment')
    sys.path.insert(0,str(root.parent.parent))
    import mlx.core as mx
    import tutorial_code.miniengine as package
    from tutorial_code.miniengine.cache import make_cache,DecodeBatch
    from tutorial_code.miniengine.context import ForwardContext
    from tutorial_code.miniengine.engine.llm_engine import LLMEngine
    from mlx.utils import tree_flatten
    need(Path(package.__file__).resolve().parent==root,'actual pinned package')
    result['device_run']=True
    result['device']=dict(name=mx.device_info().get('device_name'),architecture=mx.metal.device_info().get('architecture'),mlx_version=importlib.metadata.version('mlx'))
    need(result['device']==dict(name='Apple M5 Ultra',architecture='applegpu_g17d',mlx_version='0.32.3')
         and mx.default_device()==mx.gpu,'fixed GPU/runtime')
    engines=[]; seeds=[]; graph=witness=None; params_before=[]; cmp=support.Comparator(mx,out)
    def phase(name):
        deadline(); mx.synchronize(); emit(dict(event='memory',phase=name,active_bytes=mx.get_active_memory(),
            cache_bytes=mx.get_cache_memory(),peak_bytes=mx.get_peak_memory(),observation_not_perf=True))
    def parameters(index,when):
        rows=h.parameter_hashes(mx,engines[index].model,tree_flatten,deadline)
        path=f'parameters-{index}-{when}.json'; write(out/path,rows)
        result['parameter_receipts'].append(dict(model=index,phase=when,path=path,sha256=h.sha(out/path,deadline)))
        return rows
    def make_initial(source_caches,tokens,lens):
        batch=DecodeBatch(engines[0].model)
        batch.rebuild([],[(f'diagnostic-row-{i}',c,1536) for i,c in enumerate(source_caches)])
        canonical=support.snapshot(dict(state=batch.layers,tokens=mx.array(tokens,dtype=mx.int32),
            positions=mx.array([n-1 for n in lens],dtype=mx.int32),lens=mx.array(lens,dtype=mx.int32)))
        mx.eval(*canonical['arrays'].values())
        contexts,ids=own_contexts(mx,h,graph._runtime,layout,canonical)
        return contexts,ids,canonical
    def dump_failure(name,row,inputs,outputs,canonical):
        records={}; errors=[]
        row['failure_raw_scope']=dict(input_payload='Original input handles observed at failure; not historical before bytes',
            genuine_before_evidence='input_before hashes recorded before either arm executes',
            canonical_provenance=row['canonical_provenance'],
            canonical_payload='Canonical handles observed at failure; any earlier raw.canonical file was saved before execution')
        for label,snap in [('canonical-handles-at-failure',canonical)]+[(f'{arm}-input-handles-at-failure',snap) for arm,snap in zip(('A','C'),inputs)]:
            if snap is None: continue
            try: records[label]=h.dump(mx,out,f'failure/{name}/{label}.safetensors',snap,deadline)
            except BaseException as error: errors.append(dict(label=label,type=type(error).__name__,message=str(error)))
        for arm,snap in outputs.items():
            try: records[arm+'-output']=h.dump(mx,out,f'failure/{name}/{arm}-output.safetensors',snap,deadline)
            except BaseException as error: errors.append(dict(label=arm+'-output',type=type(error).__name__,message=str(error)))
        row['failure_raw']=records; row['failure_evidence_errors']=errors
    def point(spec,contexts,ids,canonical,*,canonical_provenance='independent-prefill-rebuild-storage'):
        name=spec['name']; row=dict(**spec,status='RUNNING',raw={},checks=[],input_before={},input_after={},outputs={},
            canonical_provenance=canonical_provenance)
        inputs=[]; outputs={}; point_cmp=support.Comparator(mx,out)
        try:
            deadline(); phase('before-'+name)
            for ctx,token in zip(contexts,ids):
                need(ctx.positions.tolist()==[n-1 for n in spec['lens']] and ctx.lens.tolist()==spec['lens']
                     and ctx.max_len==max(spec['lens']),'point exact positions/lens')
                inputs.append(support.snapshot(dict(state=ctx.caches,tokens=token,positions=ctx.positions,lens=ctx.lens)))
            mx.eval(*[value for snap in inputs for value in snap['arrays'].values()])
            sizes=[mx.get_array_buffer_size(snap['arrays']) for snap in inputs]
            need(mx.get_array_buffer_size(*[snap['arrays'] for snap in inputs])==sum(sizes),'two arm input storage distinct')
            row['input_buffer_bytes']=sizes
            for arm,snap in zip(('A','C'),inputs):
                point_cmp.compare(name+'/'+arm+'/canonical',canonical,snap)
                row['input_before'][arm]=hashes(mx,snap,deadline)
            if name in RAW_POINTS: row['raw']['canonical']=h.dump(mx,out,f'raw/{name}/canonical.safetensors',canonical,deadline)
            witness.phase=name
            left=engines[0].model(ids[0],contexts[0])
            outputs['A']=support.snapshot(dict(logits=left,state=contexts[0].caches))
            right=witness(ids[1],contexts[1])
            outputs['C']=support.snapshot(dict(logits=right,state=contexts[1].caches))
            need(all(len(snap['arrays'])==129 for snap in outputs.values()),'complete all-layer output')
            for arm,snap in zip(('A','C'),inputs):
                row['input_after'][arm]=hashes(mx,snap,deadline)
                point_cmp.meta(name+'/'+arm+'/input-unchanged',row['input_before'][arm],row['input_after'][arm])
            for arm,snap in outputs.items(): row['outputs'][arm]=hashes(mx,snap,deadline)
            if RAW_POINTS.get(name)=='both':
                for arm,snap in outputs.items(): row['raw'][arm+'-output']=h.dump(mx,out,f'raw/{name}/{arm}-output.safetensors',snap,deadline)
            point_cmp.rows=[]
            point_cmp.compare(name+'/outputs',outputs['A'],outputs['C'])
            row['checks']=point_cmp.rows
            tokens=[mx.argmax(logits,axis=-1).tolist() for logits in (left,right)]
            point_cmp.meta(name+'/own-greedy',tokens[0],tokens[1]); row['greedy_ids']=tokens
            row.update(status='PASS',candidate_bound=candidate.bound_for_used(max(spec['lens'])))
            phase('after-'+name)
            return contexts,tokens
        except BaseException as error:
            row.update(status='FAIL',checks=point_cmp.rows,error=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc()))
            if isinstance(error,(InterruptedError,TimeoutError)):
                row['failure_evidence_errors']=[dict(label='raw-save-not-started',message='Cancellation/deadline: retain existing evidence and promptly return to owner.')]
            else:
                dump_failure(name,row,inputs,outputs,canonical)
            raise
        finally:
            emit(dict(event='point',**row)); result['points'].append(dict(name=name,status=row['status'],raw=row['raw']))
            write(out/'progress.json',result)
    def direct(spec,source_caches,tokens):
        contexts,ids,canonical=make_initial(source_caches,tokens,spec['lens'])
        point(spec,contexts,ids,canonical)
    def chain(specs,source_caches,tokens):
        contexts,ids,canonical=make_initial(source_caches,tokens,specs[0]['lens'])
        for i,spec in enumerate(specs):
            if i:
                ids=advance_chain(mx,contexts,tokens,spec['lens'])
                canonical=support.snapshot(dict(state=contexts[0].caches,tokens=ids[0],positions=contexts[0].positions,lens=contexts[0].lens))
            contexts,tokens=point(spec,contexts,ids,canonical,canonical_provenance=
                'reference-input-view-after-prior-exact-step' if i else 'independent-prefill-rebuild-storage')
    try:
        for label in ('reference','candidate'):
            phase('before-load-'+label)
            e=LLMEngine(str(model_path),max_len=2048,max_seqs=8,max_batched_tokens=16384,cache_limit_gb=2,
                dtype=mx.bfloat16,compile_decode=False,fuse_decode=False); engines.append(e)
            for layer in e.model.layers:
                if not layer.is_attn: layer.linear_attn.set_gates_compile(False); layer.linear_attn.set_norm_reuse(False)
            need(e.runner.spec is None and len(e.model.layers)==64,'normal target-only full64 model')
        weights=[dict(tree_flatten(e.model.parameters())) for e in engines]
        need(not any(weights[0][k] is weights[1][k] for k in weights[0]),'independent model handles')
        cmp.compare('independent-loaded-weights',support.snapshot(weights[0]),support.snapshot(weights[1])); del weights
        emit(dict(event='loaded-weights',checks=cmp.rows)); cmp.rows=[]
        params_before=[parameters(i,'before') for i in range(2)]
        for f in fixtures['fixtures']:
            for e in engines: cmp.meta('fixture EOS',sorted(f['eos_token_ids']),sorted(e.eos_ids))
        graph=candidate.RoutePreservingDecode(engines[1].model,enabled=True); witness=old.GraphWitness(graph,emit)
        result['model_source_receipt']=graph.source_receipt; layout=tuple(layer.is_attn for layer in engines[0].model.layers)
        diagnostic=old.diagnostic_fixtures(fixtures['fixtures'],1535); write(out/'diagnostic-inputs.json',diagnostic)
        seeds=[[make_cache(e.model) for _ in range(8)] for e in engines]
        points=point_plan(); previous=0
        for used in USED:
            tokens=[]; end=used-1
            for row,f in enumerate(diagnostic):
                deadline()
                logits=[e.model(mx.array(f['prompt_ids'][previous:end],dtype=mx.int32),ForwardContext('prefill',arm[row]),last_only=True)
                        for e,arm in zip(engines,seeds)]
                cmp.compare(f'prefill-{end}-{row}/logits',support.snapshot(logits[0]),support.snapshot(logits[1]))
                cmp.compare(f'prefill-{end}-{row}/state',support.snapshot(seeds[0][row]),support.snapshot(seeds[1][row]))
                tokens.append(int(mx.argmax(logits[0],axis=-1).item()))
                emit(dict(event='prefill',stage='matrix',row=row,begin=previous,end=end,checks=cmp.rows)); cmp.rows=[]; del logits
            for spec in [x for x in points if x['kind']=='matrix' and x['lens'][0]==used]:
                direct(spec,seeds[0][:spec['B']],tokens[:spec['B']])
            if used==1023: chain([x for x in points if x['kind']=='uniform-chain'],seeds[0][:1],tokens[:1])
            if used in (1023,1024):
                for spec in [x for x in points if x['kind']=='batch-reuse' and x['lens'][0]==used]:
                    direct(spec,seeds[0][:spec['B']],tokens[:spec['B']])
            previous=end
        seeds.clear(); gc.collect(); mx.synchronize(); mx.clear_cache(); phase('before-mixed-prefill')
        seeds=[[make_cache(e.model) for _ in range(8)] for e in engines]; tokens=[]
        for row,(f,lens) in enumerate(zip(diagnostic,MIXED)):
            deadline(); end=lens-1
            logits=[e.model(mx.array(f['prompt_ids'][:end],dtype=mx.int32),ForwardContext('prefill',arm[row]),last_only=True)
                    for e,arm in zip(engines,seeds)]
            cmp.compare(f'mixed-prefill-{row}/logits',support.snapshot(logits[0]),support.snapshot(logits[1]))
            cmp.compare(f'mixed-prefill-{row}/state',support.snapshot(seeds[0][row]),support.snapshot(seeds[1][row]))
            tokens.append(int(mx.argmax(logits[0],axis=-1).item()))
            emit(dict(event='prefill',stage='mixed',row=row,begin=0,end=end,checks=cmp.rows)); cmp.rows=[]; del logits
        chain([x for x in points if x['kind']=='mixed-chain'],seeds[0],tokens)
        for i in range(2): cmp.meta(f'model-{i}/unchanged-parameters',h.parameter_values(params_before[i]),h.parameter_values(parameters(i,'after')))
        need([p['name'] for p in result['points']]==[p['name'] for p in points]
             and all(p['status']=='PASS' for p in result['points']),'exact52 point completion')
        need(witness.functions.invocations==52 and {key[0] for key in graph._functions}=={1023,1536},'actual52 graph calls and two families')
        need(sum(len(p['raw']) for p in result['points'])==10,'fixed sparse raw completeness')
        result.update(actual_graph_calls=52,compiled_objects=witness.functions.compiled_objects,matrix_complete=True)
    finally:
        if graph is not None: graph._functions.clear()
        seeds.clear(); engines.clear()
    support.verify_files(root,source,deadline); support.verify_files(model_path,manifest['files'],deadline)
    for field in ('source_manifest','model_manifest','fixtures'): support.pinned_json(inherited[field])
    validate(plan); result['source_model_pins_rechecked']=True


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--plan',type=Path,required=True); ap.add_argument('--plan-sha256',required=True); ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args(argv); q=helper(); h=q.helpers(); h.need(h.sha(args.plan)==args.plan_sha256,'plan SHA')
    plan=json.loads(args.plan.read_text()); q,h,old,support,candidate,original,inherited=validate(plan)
    h.need(signal.getitimer(signal.ITIMER_REAL)==(0.0,0.0),'existing alarm')
    args.out.mkdir(parents=True,exist_ok=False); started=time.monotonic()
    result=dict(schema='E024-route-matrix-result-v1',status='INCOMPLETE',contract=fixed_plan(),device_run=False,
        pid=os.getpid(),ppid=os.getppid(),points=[],parameter_receipts=[],matrix_complete=False,matrix_qualified=False,
        source_model_pins_rechecked=False,full_model_qualified=False,performance_qualified=False,
        plan_sha256=args.plan_sha256,qualifier_sha256=plan['qualifier_sha256'],sources={
            str(Path(__file__).resolve()):plan['qualifier_sha256'],str(HERE/'qualify.py'):CANARY_SHA,str(HERE/'candidate.py'):CANDIDATE_SHA,
            str(q.ISOLATION):q.ISOLATION_SHA,str(q.OLD/'qualify.py'):q.OLD_SHA,str(candidate.ORIGINAL):candidate.ORIGINAL_SHA,
            original['e023_qualifier']['path']:old.SUPPORT_SHA,str(candidate.PINS):old.MODEL_SOURCE_SHA})
    h.write(args.out/'plan.json',plan); h.write(args.out/'inherited-e024-plan.json',original); h.write(args.out/'inherited-e023-plan.json',inherited)
    def deadline(*args):
        if args or time.monotonic()-started>=plan['deadline_s']: raise TimeoutError('route matrix absolute deadline')
    def stop(signum,frame):
        if signum==signal.SIGALRM: deadline(signum)
        raise InterruptedError('route matrix cancellation '+signal.Signals(signum).name)
    previous={s:signal.getsignal(s) for s in (signal.SIGALRM,signal.SIGINT,signal.SIGTERM)}
    with (args.out/'events.jsonl').open('x') as stream:
        def emit(row): stream.write(json.dumps(dict(**row,elapsed_wall_s=time.monotonic()-started))+'\n'); stream.flush()
        try:
            for s in previous: signal.signal(s,stop)
            signal.setitimer(signal.ITIMER_REAL,plan['deadline_s'])
            device(plan,q,h,old,support,candidate,original,inherited,args.out,deadline,emit,result)
            h.need(h.sha(args.plan)==args.plan_sha256,'plan changed')
            result.update(status='PASS',matrix_qualified=True)
        except BaseException as error:
            result.update(status='FAIL',error=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc()))
        finally:
            try:
                gc.collect(); mx=sys.modules.get('mlx.core')
                if mx is not None:
                    mx.synchronize(); mx.clear_cache(); result['final_active_bytes']=mx.get_active_memory(); result['peak_bytes']=mx.get_peak_memory()
            except BaseException as error: result.update(status='FAIL',matrix_qualified=False,cleanup_error=dict(type=type(error).__name__,message=str(error)))
            finally:
                signal.setitimer(signal.ITIMER_REAL,0)
                for s,handler in previous.items(): signal.signal(s,handler)
                result['elapsed_wall_s']=time.monotonic()-started; h.write(args.out/'complete.json',result)
    print(json.dumps(dict(status=result['status'],out=str(args.out))),flush=True)
    return 0 if result['status']=='PASS' else 1


if __name__=='__main__': raise SystemExit(main())
