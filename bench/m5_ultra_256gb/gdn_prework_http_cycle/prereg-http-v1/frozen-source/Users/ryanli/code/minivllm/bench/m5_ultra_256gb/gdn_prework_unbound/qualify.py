"""Real-activation, local-layer canary for unbound GDN preparation.

Source-only preparation: no device or numerical qualification has run. CLI
--plan FILE --plan-sha256 SHA --out NEW. Exact five-field plan: schema
GDN-prework-real-layer-plan-v1, qualifier_sha256, candidate {path,sha256},
e024_plan {path,sha256}, deadline_s (0 < seconds <=1800). The original E024
plan binds the unchanged E023 model/source/eight original D512 fixtures.

One original eager model, gates ON/norm OFF, actual greedy tokens/normal EOS.
Shortest D fixture: decode1..3; longest: decode129..131. Capture GDN ordinals
0/24/47, then replay each three-step chain using the reference's real x and
the candidate's OWN continued state. Compare/save complete BF16 output/conv
and FP32 S. This is not a candidate full-model rollout, scheduler/HTTP gate or
performance test. Early normal EOS is INCOMPLETE, never forced continuation.
Root must separately own resources, service removal and hard child deadline.
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
from types import MethodType

HERE = Path(__file__).resolve().parent
HELPER = HERE.parent/'long_decode_isolation/qualify.py'
HELPER_SHA = '9d8c17a4bfc7f6ea9d61f79f466cad45059b71c9606d7b8d29c05c40213cad9e'
OLD = HERE.parent/'long_decode_graph/qualify.py'
OLD_SHA = 'c24219f431b08502a26a4f7a83c002214bddd7ea3962d6b1869b313e4c13faf7'
OLD_PLAN_SHA = '5025d37768a5fa74d6e3e256a96ff8b8e4e92b31837183d5ffe031d9473e7123'
CORE_PINS = HERE.parent/'long_decode_graph/model-source-pins.json'
CANDIDATE_SHA = '1bf5a837c6515ca505bc5d1beaa1718b88e1ccd52609f6814bb995afbe6c1260'


def need(ok, message):
    if not ok: raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path, name, digest):
    need(sha(path) == digest, 'pinned executable source changed: '+str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def validate(plan):
    need(set(plan) == {'schema','qualifier_sha256','candidate','e024_plan','deadline_s'}
         and plan['schema'] == 'GDN-prework-real-layer-plan-v1', 'exact canary plan')
    need(plan['qualifier_sha256'] == sha(__file__), 'actual qualifier source')
    budget = plan['deadline_s']
    need(type(budget) in (int,float) and math.isfinite(budget) and 0 < budget <= 1800, 'bounded child deadline')
    ref = plan['candidate']
    need(ref == dict(path=str(HERE/'candidate.py'),sha256=CANDIDATE_SHA), 'fixed adjacent default-OFF candidate')
    candidate = load(ref['path'],'_gdn_prework_candidate',CANDIDATE_SHA)
    ref = plan['e024_plan']
    need(set(ref) == {'path','sha256'} and Path(ref['path']).is_absolute()
         and ref['sha256'] == OLD_PLAN_SHA == sha(ref['path']), 'unchanged inherited E024 plan')
    h = load(HELPER,'_gdn_prework_exact_helpers',HELPER_SHA)
    old = load(OLD,'_gdn_prework_old_contract',OLD_SHA)
    original = json.loads(Path(ref['path']).read_bytes()); old.validate_plan(original)
    support = old.load_pinned(original['e023_qualifier'],'_gdn_prework_original_comparator',old.SUPPORT_SHA)
    inherited = support.pinned_json(original['e023_plan'])
    need(h.sha(CORE_PINS) == old.MODEL_SOURCE_SHA, 'frozen full core manifest')
    return h,old,support,candidate,original,inherited


def fixed_contract():
    return dict(schema='GDN-prework-real-layer-contract-v1',fixture_selection='shortest/longest original D512; ties by fixture_id',
        starts=[1,129],steps_per_chain=3,gdn_ordinals=[0,24,47],expected_layer_calls=18,
        configuration=dict(max_len=2048,max_seqs=1,max_batched_tokens=16384,cache_limit_gib=2,
            compile_decode=False,fuse_decode=False,gates_compile=True,norm_reuse=False),
        inputs='unmodified actual original-model activation; ordinary eager greedy rollout',
        state='candidate initialized once per layer/fixture, then its own prior output state',
        normal_eos=True,early_eos='INCOMPLETE; no ignore_eos or synthetic continuation',
        exact_arrays=['output:BF16[1,5120]','conv:BF16[1,3,10240]','S:FP32[1,48,128,128]'],
        full_model_qualified=False,http_qualified=False,performance_qualified=False,
        excluded=['candidate full-model rollout','ahead/scheduling','B>1','parameter replacement',
                  'prefill optimization','E024/E028 composition','random/special/strided synthetic coverage'])


def choose_cases(fixtures):
    need(len(fixtures) == 8 and len({f['fixture_id'] for f in fixtures}) == 8, 'eight original distinct D fixtures')
    for f in fixtures:
        need(f['normal_eos'] is True and type(f['max_output_tokens']) is int and f['max_output_tokens'] == 512
             and 886 <= len(f['prompt_ids']) <= 978
             and all(type(t) is int and t >= 0 for t in f['prompt_ids']), 'actual original D512 fixture')
    ordered = sorted(fixtures,key=lambda f:(len(f['prompt_ids']),f['fixture_id']))
    return [(ordered[0],1),(ordered[-1],129)]


class IncompleteCoverage(RuntimeError):
    pass


def rollout_window(first_token, eos_ids, cap, start, advance, emit):
    """Original-model token causality; advance receives the previous real ID.

    No fabricated tokens or retries. The prefill output is raw token ordinal1;
    decode d consumes raw ordinal d and produces ordinal d+1. EOS on the final
    captured output is permitted; earlier EOS cannot complete the fixed chain.
    """
    token = first_token; raw = [token]; records = []
    for step in range(1,start+3):
        if token in eos_ids or len(raw) >= cap:
            emit(dict(event='coverage-shortfall',decode_step=step,raw_ids=list(raw),normal_eos=token in eos_ids))
            raise IncompleteCoverage('normal EOS/cap before the fixed three-step activation chain')
        captured = step >= start
        next_token, batch = advance(token,step,captured)
        need(type(next_token) is int and next_token >= 0, 'original greedy ID')
        raw.append(next_token)
        emit(dict(event='reference-token',decode_step=step,input_token=token,output_token=next_token,captured=captured))
        if captured: records.append(batch)
        token = next_token
    need(len(records) == 3, 'three actual sequential decode steps')
    return raw,records


class DecodeCapture:
    """Only three idle instances; no module/class patch or extra evaluation.

    Snapshot handles are kept only through the caller's normal full-model eval,
    then written/released. The pinned ordinary method replaces conv/S handles;
    it does not mutate the prior array payloads captured here.
    """
    def __init__(self, layers, module, snapshot):
        self.bindings=[]; self.rows={}; self.active=False; self.step=None
        self.expected=tuple(layers); self.snapshot=snapshot; self.closed=False
        try:
            for index,layer in layers.items():
                need(type(layer) is module.GatedDeltaNet and '_decode' not in vars(layer)
                     and layer._decode.__func__ is module.GatedDeltaNet._decode, 'untouched ordinary capture layer')
                original=layer._decode
                def call(this,x,state,*,_compile_gates=False,_norm_reuse_normal=False,_index=index,_original=original):
                    if not self.active:
                        return _original(x,state,_compile_gates=_compile_gates,_norm_reuse_normal=_norm_reuse_normal)
                    need(_index not in self.rows and _compile_gates is True and _norm_reuse_normal is True,
                         'one actual ordinary decode per selected layer/step')
                    before=self.snapshot(dict(x=x,conv=state.conv,S=state.S))
                    output=_original(x,state,_compile_gates=_compile_gates,_norm_reuse_normal=_norm_reuse_normal)
                    self.rows[_index]=dict(step=self.step,input=before,
                        output=self.snapshot(dict(output=output,conv=state.conv,S=state.S)))
                    return output
                wrapper=MethodType(call,layer); object.__setattr__(layer,'_decode',wrapper)
                self.bindings.append((layer,wrapper))
        except BaseException:
            self.restore(); raise

    def begin(self, step, active):
        need(not self.rows and not self.closed, 'capture rows must be consumed between steps')
        self.step,self.active=step,active

    def finish(self):
        need(set(self.rows) == (set(self.expected) if self.active else set()), 'complete actual capture layers')
        result,self.rows=self.rows,{}; return result

    def restore(self):
        if self.closed: return
        for layer,wrapper in reversed(self.bindings):
            need(vars(layer).get('_decode') is wrapper, 'foreign capture wrapper must restore first')
            object.__delattr__(layer,'_decode')
        self.bindings.clear(); self.rows.clear(); self.snapshot=None; self.closed=True

    def __enter__(self): return self
    def __exit__(self,*exc): self.restore(); return False


def replay_chain(layer, records, *, initialize, input_at, invoke, observe, compare):
    """Initialize state ONCE; only x comes from each later reference capture.

    Callbacks are concrete I/O/comparison in device(), and fake in CPU tests.
    They never provide a replacement state after initialize. A failed call or
    comparison exits immediately; no ordinary fallback or later step is hidden.
    """
    need(len(records)==3,'fixed three-step replay chain')
    state=initialize(records[0])
    for ordinal,record in enumerate(records):
        x,expected=input_at(record)
        before=observe(x,state)
        compare('input-before',ordinal,before,expected)
        output=invoke(layer,x,state)
        compare('output',ordinal,(output,state),record)
        compare('input-unchanged',ordinal,before,expected)


def checked_invoke(handle, layer, x, state, evaluate):
    """Count only this evaluated candidate call, not a stale last-path value."""
    before=dict(handle.calls)
    output=layer._decode(x,state,_compile_gates=True,_norm_reuse_normal=True)
    evaluate(output,state.conv,state.S)
    need(dict(handle.calls)==dict(candidate=before['candidate']+1,fallback=before['fallback']),
         'actual candidate call; no fallback/stale counter')
    return output


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
        engine=LLMEngine(str(model_path),max_len=2048,max_seqs=1,max_batched_tokens=16384,
                         cache_limit_gb=2,dtype=mx.bfloat16,compile_decode=False,fuse_decode=False)
        model=engine.model
        need(engine.runner.spec is None and len(model.layers)==64,'original full target-only model')
        indices=[i for i,layer in enumerate(model.layers) if not layer.is_attn]
        need(len(indices)==48,'actual48 GDN layers')
        for i in indices:
            model.layers[i].linear_attn.set_gates_compile(True); model.layers[i].linear_attn.set_norm_reuse(False)
        layers={indices[n]:model.layers[indices[n]].linear_attn for n in (0,24,47)}
        result['layer_indices']=list(layers)
        def layout(snapshot,output=False):
            expected={'root/'+('output' if output else 'x'):((1,5120),mx.bfloat16),
                      'root/conv':((1,3,10240),mx.bfloat16),'root/S':((1,48,128,128),mx.float32)}
            need(set(snapshot['arrays'])==set(expected) and all((snapshot['arrays'][k].shape,snapshot['arrays'][k].dtype)==v
                 for k,v in expected.items()),'complete original-dtype real layer layout')
        for f in fixtures['fixtures']: cmp.meta('fixture EOS',sorted(f['eos_token_ids']),sorted(engine.eos_ids))
        def parameters(name):
            values={str(i):h.parameter_hashes(mx,layer,tree_flatten,deadline) for i,layer in layers.items()}
            path=name+'.json'; h.write(out/path,values)
            result[name]=dict(path=path,sha256=h.sha(out/path,deadline),scope='all parameters of the three selected layers')
            return {i:h.parameter_values(v) for i,v in values.items()}
        parameters_before=parameters('parameters-before')
        for fixture,start in choose_cases(fixtures['fixtures']):
            prefix='captures/'+fixture['fixture_id']; original_ids=list(fixture['prompt_ids'])
            h.write(out/(fixture['fixture_id']+'-fixture.json'),fixture)
            caches=cache.make_cache(model)
            logits=model(mx.array(original_ids,dtype=mx.int32),context.ForwardContext('prefill',caches),last_only=True)
            mx.eval(logits,*support.snapshot(caches)['arrays'].values())
            token=int(mx.argmax(logits,axis=-1).item())
            batch=cache.DecodeBatch(model); batch.rebuild([],[('local-canary',caches,len(original_ids)+512)])
            mx.eval(*batch.arrays()); caches.clear()
            with DecodeCapture(layers,gdn,support.snapshot) as capture:
                def advance(input_token,step,active):
                    deadline(); capture.begin(step,active); position=len(original_ids)+step-1
                    ctx=context.ForwardContext('decode',batch.layers,positions=mx.array([position],dtype=mx.int32),
                        lens=mx.array([position+1],dtype=mx.int32),max_len=position+1)
                    full_logits=model(mx.array([input_token],dtype=mx.int32),ctx)
                    mx.eval(full_logits,*batch.arrays()); batch.lens=[position+1]
                    sampled=int(mx.argmax(full_logits,axis=-1).item()); observed=capture.finish(); saved={}
                    for i,row in observed.items():
                        base=f'{prefix}/step-{step}/layer-{i}'
                        layout(row['input']); layout(row['output'],True)
                        saved[i]=dict(layer=i,decode_step=step,position=position,used=position+1,input_token=input_token,
                            reference_input=h.dump(mx,out,base+'/reference-input.safetensors',row['input'],deadline),
                            reference_output=h.dump(mx,out,base+'/reference-output.safetensors',row['output'],deadline))
                    if active:
                        receipt=h.dump(mx,out,f'{prefix}/step-{step}/reference-full-logits.safetensors',support.snapshot(full_logits),deadline)
                        emit(dict(event='reference-capture',fixture_id=fixture['fixture_id'],decode_step=step,used=position+1,
                            reference_full_logits=receipt,layers=saved))
                    observed.clear(); return sampled,saved
                raw,case_records=rollout_window(token,engine.eos_ids,512,start,advance,
                    lambda row:emit(dict(fixture_id=fixture['fixture_id'],**row)))
            records.append(dict(fixture_id=fixture['fixture_id'],start=start,prompt_tokens=len(original_ids),raw_ids=raw,steps=case_records))
            h.write(out/'captures.json',records)
            result['reference_chains']+=1
            batch.clear(); batch=None; del logits,capture,advance
            gc.collect(); mx.synchronize(); mx.clear_cache(); memory('reference-chain-released')
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
                        emit(dict(event='candidate-dispatch',fixture_id=case['fixture_id'],layer=index,
                            decode_step=chain[replay_calls]['decode_step'],used=chain[replay_calls]['used'],
                            before=before,after=dict(handles[index].calls)))
                        replay_calls+=1
                        result['candidate_calls']+=1
                        return value
                    def compare(kind,ordinal,value,expected):
                        record=chain[ordinal]; tag=f"{case['fixture_id']}/layer-{index}/step-{record['decode_step']}"
                        if kind=='output':
                            output,state=value; actual=support.snapshot(dict(output=output,conv=state.conv,S=state.S))
                            layout(actual,True)
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
                            actual=support.snapshot(value); layout(actual)
                            if kind=='input-before':
                                left=list(value.values()); right=list(expected['arrays'].values())
                                need(mx.get_array_buffer_size(left,right)==mx.get_array_buffer_size(left)+mx.get_array_buffer_size(right),
                                     'candidate input/state storage independent of raw reference')
                            cmp.compare(tag+'/'+kind,expected,actual)
                        emit(dict(event='comparison',kind=kind,fixture_id=case['fixture_id'],layer=index,
                                  decode_step=record['decode_step'],used=record['used'],checks=cmp.rows))
                        cmp.rows=[]
                    replay_chain(layer,chain,initialize=initialize,input_at=input_at,invoke=invoke,
                        observe=lambda x,st:dict(x=x,conv=st.conv,S=st.S),compare=compare)
                    result['candidate_chains']+=1
            result['dispatch_counters']={str(i):dict(hd.calls) for i,hd in handles.items()}
        result['instance_hooks_restored']=all('_decode' not in vars(layer) for layer in layers.values())
        need(result['instance_hooks_restored'] and result['candidate_calls']==result['candidate_compared_calls']==18 and result['candidate_chains']==6,
             'all18 real-layer calls/six independent-state chains, cleanup')
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
    result=dict(schema='GDN-prework-real-layer-result-v1',status='INCOMPLETE',device_run=False,
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
        result['sources']={str(Path(__file__).resolve()):plan['qualifier_sha256'],str(HERE/'candidate.py'):CANDIDATE_SHA,
            str(HELPER):HELPER_SHA,str(OLD):OLD_SHA,str(CORE_PINS):old.MODEL_SOURCE_SHA,
            original['e023_qualifier']['path']:old.SUPPORT_SHA}
        need(signal.getitimer(signal.ITIMER_REAL)==(0.0,0.0),'existing alarm')
        def deadline():
            if time.monotonic()-started>=plan['deadline_s']: raise TimeoutError('real-layer canary deadline')
        def stop(signum,frame):
            if signum==signal.SIGALRM: raise TimeoutError('real-layer canary deadline')
            raise InterruptedError('real-layer canary cancellation '+signal.Signals(signum).name)
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


if __name__=='__main__': raise SystemExit(main())
