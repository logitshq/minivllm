"""Default OFF. E046 real-activation B2/B8 local GDN timing; no engine speed claim."""
import argparse
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import signal
import statistics
import sys
import time
import traceback

R = Path('/Users/ryanli/code/minivllm'); B = R/'bench/m5_ultra_256gb'; A = R/'.artifacts/m5-ultra-256gb'
Q = B/'gdn_prework_batched/qualify.py'; EVIDENCE = A/'gdn-batched-v1-numerical'
AUDIT = A/'gdn-batched-v1-independent-numerical.json'
QUALIFIER_SHA = '8b0f0fc0060d918f1eb3a169e702985642f10d6b03760546941308846256425c'
PLAN_SHA = '7df6cff99425698a1fb134df4f25dc8de4e717482f73039e56ec5a0a00930003'
COMPLETE_SHA = '94e6b0a2dbb9c1beebf3026258709532837219b3aaa69379fa9d7ce9c22dde3b'
CAPTURES_SHA = 'ea7b87878037ceec085d97958c9b41972bf80a6cff1a3563771fdf14b9d1dc99'
AUDIT_SHA = 'ce6c9c840360f336febe20d84081da2c84b3e93252b1b7733d3a9761358a1bce'
LAYERS = (0,32,62); GROUPS = 5; PAIRS = 10; WARMUPS = 5; DEADLINE = 900


def need(ok, message):
    if not ok: raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''): h.update(block)
    return h.hexdigest()


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,indent=2,allow_nan=False); stream.write('\n')


def source(path, name, digest):
    need(sha(path)==digest,'fixed source changed: '+str(path))
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def select_captures(captures):
    expected=[(b,age,start) for b in (2,8) for age,start in (('early',1),('late',129))]
    need([c['case_id'] for c in captures]==[f'b{b}-{age}' for b,age,_ in expected],'four exact capture chains')
    selected=[]
    for case,(batch,age,start) in zip(captures,expected):
        need(case['batch_size']==batch and case['start']==start and len(case['steps'])==3,'original three-step chain')
        first=case['steps'][0]; need(set(first)=={str(i) for i in LAYERS},'three selected actual layers')
        for layer in LAYERS:
            row=first[str(layer)]
            need(row['layer']==layer and row['batch_size']==batch and row['decode_step']==start
                 and row['row_ids']==case['row_ids'] and len(row['row_ids'])==batch,'actual first capture identity')
            for kind in ('input','output'):
                ref=row['reference_'+kind]
                need(ref['path']==f'captures/b{batch}-{age}/step-{start}/layer-{layer}/reference-{kind}.safetensors'
                     and ref['finite'] is True,'exact original capture path/finite receipt')
            selected.append(dict(case_id=f'b{batch}-{age}-layer{layer}',**row))
    return selected


def evidence():
    need(sha(AUDIT)==AUDIT_SHA and sha(EVIDENCE/'complete.json')==COMPLETE_SHA
         and sha(EVIDENCE/'captures.json')==CAPTURES_SHA,'actual E046 evidence changed')
    audit=json.loads(AUDIT.read_text()); complete=json.loads((EVIDENCE/'complete.json').read_text())
    need(audit['schema']=='GDN-batched-prework-real-layer-independent-v1' and audit['status']=='PASS'
         and audit['canary_qualified'] is True and audit['raw_files']==140
         and audit['complete_sha256']==COMPLETE_SHA and audit['captures_sha256']==CAPTURES_SHA
         and audit['plan_sha256']==PLAN_SHA and audit['child_directory']==str(EVIDENCE),'actual independent E046 audit')
    need(complete['status']=='PASS' and complete['canary_qualified'] is True and complete['plan_sha256']==PLAN_SHA
         and complete['candidate_calls']==complete['candidate_compared_calls']==36
         and complete['layer_indices']==list(LAYERS),'actual complete E046 canary')
    return select_captures(json.loads((EVIDENCE/'captures.json').read_text())),complete


def paired_order():
    for group in range(GROUPS):
        for pair in range(PAIRS):
            order=('native','candidate') if pair%2==0 else ('candidate','native')
            for slot,arm in enumerate(order): yield dict(group=group,pair=pair,slot=slot,arm=arm)


def fresh_state(cache, arrays):
    # Original native/candidate assign fresh conv/S outputs. Each invocation gets
    # new descriptors over this arm's own immutable evaluated seed storage.
    return arrays['root/x'][:], cache.BatchState(arrays['root/conv'][:],arrays['root/S'][:])


def measured(mx, cache, fn, seed, *, clock=time.perf_counter_ns):
    x,state=fresh_state(cache,seed); mx.eval(x,state.conv,state.S); mx.synchronize()
    start=clock()
    value=fn(x,state,_compile_gates=True,_norm_reuse_normal=True)
    mx.eval(value,state.conv,state.S); mx.synchronize()
    elapsed=clock()-start
    need(type(elapsed) is int and elapsed>0,'positive elapsed nanoseconds')
    return elapsed,dict(output=value,conv=state.conv,S=state.S)


def dispatch(handle, before, arm):
    expected=dict(before)
    if arm=='candidate': expected['candidate']+=1
    need(handle.calls==expected and handle.calls['fallback']==0,'actual candidate dispatch, no fallback')


def run_case(mx, cache, candidate, gdn, layer, row, h, support, out, deadline, emit):
    case=out/row['case_id']; case.mkdir(); batch=row['batch_size']
    for key in ('reference_input','reference_output'):
        ref=row[key]; need(h.sha(EVIDENCE/ref['path'],deadline)==ref['sha256'],'original capture payload SHA')
    input_ref=row['reference_input']; output_ref=row['reference_output']
    seeds={arm:mx.load(str(EVIDENCE/input_ref['path'])) for arm in ('native','candidate')}
    original=mx.load(str(EVIDENCE/input_ref['path'])); reference=mx.load(str(EVIDENCE/output_ref['path']))
    mx.eval(*[v for d in (*seeds.values(),original,reference) for v in d.values()])
    expected={'root/x':((batch,5120),mx.bfloat16),'root/conv':((batch,3,10240),mx.bfloat16),
              'root/S':((batch,48,128,128),mx.float32)}
    for arrays in (*seeds.values(),original):
        need(set(arrays)==set(expected) and all((arrays[k].shape,arrays[k].dtype)==v for k,v in expected.items()),'exact capture input layout')
    left,right=[list(seeds[k].values()) for k in ('native','candidate')]
    need(mx.get_array_buffer_size(left,right)==mx.get_array_buffer_size(left)+mx.get_array_buffer_size(right),'independent arm seed storage')
    cmp=support.Comparator(mx,case); native=layer._decode; samples=[]; saved={}
    with candidate.install(layer,gdn,enabled=True) as handle:
        routes=dict(native=native,candidate=layer._decode)
        def once(arm):
            deadline(); before=dict(handle.calls)
            elapsed,value=measured(mx,cache,routes[arm],seeds[arm]); dispatch(handle,before,arm)
            return elapsed,value
        # Both arms must match the actual E046 original bytes before any timing.
        for arm in routes:
            _,value=once(arm); snap=support.snapshot(value)
            saved[arm]=h.dump(mx,case,arm+'-qualified.safetensors',snap,deadline)
            cmp.compare(arm+'/original-output',dict(metadata=output_ref['metadata'],arrays=reference),snap)
            cmp.compare(arm+'/seed-preserved',dict(metadata=input_ref['metadata'],arrays=original),
                        dict(metadata=input_ref['metadata'],arrays=seeds[arm]))
            del value,snap
        write(case/'numerical.json',dict(status='PASS',input=input_ref,reference=output_ref,saved=saved,checks=cmp.rows))
        cmp.rows=[]
        with (case/'timings.jsonl').open('x') as stream:
            def retain(record): stream.write(json.dumps(record,allow_nan=False)+'\n'); stream.flush()
            for warm in range(WARMUPS):
                for arm in ('native','candidate'):
                    elapsed,value=once(arm); del value
                    retain(dict(phase='warmup',warmup=warm,arm=arm,elapsed_ns=elapsed))
            for item in paired_order():
                elapsed,value=once(item['arm']); del value
                record=dict(phase='measured',**item,elapsed_ns=elapsed); samples.append(record); retain(record)
        for arm,arrays in seeds.items():
            cmp.compare(arm+'/final-seed-preserved',dict(metadata=input_ref['metadata'],arrays=original),
                        dict(metadata=input_ref['metadata'],arrays=arrays))
        counters=dict(handle.calls)
        need(counters==dict(candidate=1+WARMUPS+GROUPS*PAIRS,fallback=0),'all expected local candidate calls')
    need('_decode' not in vars(layer) and layer._decode.__func__ is gdn.GatedDeltaNet._decode,'instance hook restored')
    groups=[]
    for group in range(GROUPS):
        med={arm:statistics.median(s['elapsed_ns'] for s in samples if s['group']==group and s['arm']==arm) for arm in ('native','candidate')}
        groups.append(dict(group=group,median_ns=med,native_over_candidate=med['native']/med['candidate']))
    result=dict(case_id=row['case_id'],status='PASS',batch_size=batch,layer=row['layer'],decode_step=row['decode_step'],
        row_ids=row['row_ids'],timing_count=len(samples),groups=groups,
        median_group_ratio=statistics.median(g['native_over_candidate'] for g in groups),dispatch=counters,final_checks=cmp.rows)
    write(case/'result.json',result); emit(dict(event='case',**result)); return result


def device(args, q, plan, out, deadline, emit, result):
    rows,proof=evidence(); h,old,support,candidate,_,inherited=q.validate(plan)
    root=Path(inherited['package_root'])/'tutorial_code/miniengine'; model_path=Path(inherited['model'])
    pins=support.pinned_json(inherited['source_manifest'])['files']; manifest=support.pinned_json(inherited['model_manifest'])
    need(pins==json.loads(q.CORE_PINS.read_text())['files'] and all(pins.get(k)==v for k,v in candidate.PINS.items()),'frozen native source closure')
    support.verify_files(root,pins,deadline); support.verify_files(model_path,manifest['files'],deadline)
    for name,digest in inherited['execution']['pins'].items(): need(h.sha(name,deadline)==digest,'runtime changed: '+name)
    need({k:v for k,v in os.environ.items() if k.startswith('MLX_')}==inherited['execution']['environment'],'exact inherited environment')
    sys.path.insert(0,str(root.parent.parent))
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from tutorial_code.miniengine import cache
    from tutorial_code.miniengine.model import gdn
    from tutorial_code.miniengine.engine.llm_engine import LLMEngine
    need(Path(cache.__file__).resolve()==root/'cache.py' and Path(gdn.__file__).resolve()==root/'model/gdn.py','actual frozen modules')
    runtime=dict(mlx_version=importlib.metadata.version('mlx'),name=mx.device_info().get('device_name'),architecture=mx.metal.device_info().get('architecture'))
    need(runtime==dict(mlx_version='0.32.3',name='Apple M5 Ultra',architecture='applegpu_g17d') and mx.default_device()==mx.gpu,'actual fixed device')
    result.update(device_run=True,runtime=runtime); engine=None; layers={}
    def memory(phase):
        deadline(); mx.synchronize(); emit(dict(event='memory',phase=phase,active_bytes=mx.get_active_memory(),peak_bytes=mx.get_peak_memory(),cache_bytes=mx.get_cache_memory()))
    try:
        memory('before-load')
        engine=LLMEngine(str(model_path),max_len=2048,max_seqs=8,max_batched_tokens=16384,cache_limit_gb=2,
                         dtype=mx.bfloat16,compile_decode=False,fuse_decode=False)
        need(engine.runner.spec is None and engine.runner._ahead is None and len(engine.model.layers)==64,'ordinary target-only model')
        layers={i:engine.model.layers[i].linear_attn for i in LAYERS}
        for layer in layers.values(): layer.set_gates_compile(True); layer.set_norm_reuse(False)
        def parameters(name):
            records={str(i):h.parameter_hashes(mx,layer,tree_flatten,deadline) for i,layer in layers.items()}
            write(out/(name+'.json'),records); return {i:h.parameter_values(v) for i,v in records.items()}
        ref=proof['parameters-before']; need(sha(EVIDENCE/ref['path'])==ref['sha256'],'original selected-layer parameter receipt')
        expected={i:h.parameter_values(v) for i,v in json.loads((EVIDENCE/ref['path']).read_text()).items()}
        before=parameters('parameters-before'); need(before==expected,'same actual E046 selected-layer parameter bytes')
        memory('loaded')
        for row in rows:
            result['current_case']=row['case_id']
            case=run_case(mx,cache,candidate,gdn,layers[row['layer']],row,h,support,out,deadline,emit)
            result['cases'].append(case); result['case_count']+=1; result['numerical_case_count']+=1
            result['timing_count']+=case['timing_count']; result['candidate_calls']+=case['dispatch']['candidate']
            result['candidate_fallback_calls']+=case['dispatch']['fallback']
            gc.collect(); memory('case-released')
        need(parameters('parameters-after')==before,'selected parameter values unchanged')
        result['instance_hooks_restored']=all('_decode' not in vars(layer) for layer in layers.values())
    except BaseException as error:
        traceback.clear_frames(error.__traceback__); raise
    finally:
        if engine is not None:
            engine.runner.batch.clear(); engine.runner.staging.clear(); engine.runner.feats.clear(); engine.runner._ahead=None
        layers.clear(); engine=layer=None
        gc.collect(); mx.synchronize(); mx.clear_cache(); memory('after-release'); result['device_cleanup_completed']=True
    support.verify_files(root,pins,deadline); support.verify_files(model_path,manifest['files'],deadline)
    evidence(); q.validate(plan); need(sha(args.plan)==args.plan_sha256,'original plan unchanged')
    result['source_model_pins_rechecked']=True


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--run',action='store_true')
    p.add_argument('--plan',type=Path); p.add_argument('--plan-sha256'); p.add_argument('--out',type=Path); args=p.parse_args(argv)
    if not args.run:
        need(args.plan is args.plan_sha256 is args.out is None,'explicit --run required')
        print('DESCRIPTION ONLY: E046 12 actual B2/B8 first captures; native batched vs frozen candidate; 5 warmups +5x10 paired samples.'); return 0
    need(args.plan and args.out and args.out.is_absolute() and not args.out.exists(),'explicit fresh absolute output')
    args.out.mkdir(); start=time.monotonic(); previous={}
    result=dict(schema='GDN-batched-local-timing-result-v1',status='INCOMPLETE',pid=os.getpid(),ppid=os.getppid(),
        source_sha256=sha(__file__),plan_sha256=args.plan_sha256,device_run=False,cases=[],case_count=0,numerical_case_count=0,
        timing_count=0,candidate_calls=0,candidate_fallback_calls=0,instance_hooks_restored=False,
        source_model_pins_rechecked=False,device_cleanup_completed=False,local_timing_screen_complete=False,
        full_model_qualified=False,http_qualified=False,performance_qualified=False,engine_pareto_point=False)
    def deadline():
        if time.monotonic()-start>=DEADLINE: raise TimeoutError('900-second local screen deadline')
    def stop(signum,frame): raise InterruptedError('local timing cancelled: '+signal.Signals(signum).name)
    try:
        need(args.plan_sha256==PLAN_SHA==sha(args.plan),'original E046 child plan binding')
        plan=json.loads(args.plan.read_text()); write(args.out/'plan.json',plan)
        write(args.out/'contract.json',dict(groups=GROUPS,pairs=PAIRS,warmups=WARMUPS,cases=12,timing_count=1200,
            timing='fresh native B2/B8 or original candidate forward+complete output/conv/S eval+sync; independent seed reset outside clock',
            excluded=['load','hash','reset/eval','serialization','parameter checks','HTTP','whole-model performance'],
            original_audit=dict(path=str(AUDIT),sha256=AUDIT_SHA),original_complete_sha256=COMPLETE_SHA,captures_sha256=CAPTURES_SHA))
        need(signal.getitimer(signal.ITIMER_REAL)==(0.0,0.0),'existing alarm')
        previous={s:signal.getsignal(s) for s in (signal.SIGALRM,signal.SIGTERM,signal.SIGINT)}
        for sig in previous: signal.signal(sig,stop)
        signal.setitimer(signal.ITIMER_REAL,DEADLINE)
        q=source(Q,'_e048_original_canary',QUALIFIER_SHA)
        with (args.out/'events.jsonl').open('x') as stream:
            def emit(row): stream.write(json.dumps(dict(**row,elapsed_wall_s=time.monotonic()-start),allow_nan=False)+'\n'); stream.flush()
            device(args,q,plan,args.out,deadline,emit,result)
        need(result['case_count']==result['numerical_case_count']==12 and result['timing_count']==1200
             and result['candidate_calls']==672 and result['candidate_fallback_calls']==0
             and result['instance_hooks_restored'],'complete fixed local timing contract')
        result.update(status='PASS',local_timing_screen_complete=True)
    except BaseException as error:
        result.update(status='FAIL',error=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc()))
    finally:
        if previous:
            signal.setitimer(signal.ITIMER_REAL,0)
            for sig,value in previous.items(): signal.signal(sig,value)
        result['elapsed_wall_s']=time.monotonic()-start; write(args.out/'complete.json',result)
    print(json.dumps({k:result[k] for k in ('status','case_count','timing_count','local_timing_screen_complete')}),flush=True)
    return 0 if result['status']=='PASS' else 1


if __name__=='__main__': raise SystemExit(main())
