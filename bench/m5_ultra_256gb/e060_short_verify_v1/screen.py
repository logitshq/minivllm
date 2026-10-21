"""E060: default-OFF real DFlash8 generation with T2/T4/T8 private verification."""
import argparse, gc, hashlib, importlib.util, json, os, signal, sys, time, traceback
from pathlib import Path
from types import SimpleNamespace
HERE=Path(__file__).resolve().parent
R=Path('/Users/ryanli/code/minivllm');B=R/'bench/m5_ultra_256gb'
CAPTURE=(5,19,33,47,61)
ARMS=('native','fixed2','fixed4','fixed8')
def need(ok,message):
    if not ok: raise ValueError(message)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
    return h.hexdigest()
def write(p,value):Path(p).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def load(p,name):
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def pinned(ref):
    need(sha(ref['path'])==ref['sha256'],'pin '+ref['path']);return json.loads(Path(ref['path']).read_text())
def validate(plan):
    need(plan['schema']=='E060-generation-plan-v1' and plan['deadline_s']==1500,'plan identity/deadline')
    for p,digest in plan['sources'].items():need(sha(p)==digest,'source '+p)
    need(pinned(plan['e059_result'])['status']=='PASS_GENERATION_LOCAL_COST','closed E059 producer')
    old=pinned(plan['e058_plan']);complete=pinned(plan['e058_result']);audit=pinned(plan['e058_audit'])
    need(complete['status']=='PASS_FEATURES_LOCAL_COST' and audit['status']=='PASS','E058 qualification')
    module=load(B/'e058_agent_run_v1/screen.py','_e060_e058')
    deps=module.validate(old)
    for p,digest in plan['drafter_pins'].items():need(sha(p)==digest,'drafter '+p)
    return module,deps

def device(plan,out,deadline,result):
    previous,(cost,_,_,_,deps,_)=validate(plan)
    bridge,frozen,h,old,support,original,inherited=deps
    root=Path(inherited['package_root'])/'tutorial_code/miniengine';model_path=Path(inherited['model'])
    sources=support.pinned_json(inherited['source_manifest'])['files'];weights=support.pinned_json(inherited['model_manifest'])['files']
    support.verify_files(root,sources,deadline);support.verify_files(model_path,weights,deadline)
    need({k:v for k,v in os.environ.items() if k.startswith('MLX_')}==inherited['execution']['environment'],'MLX environment')
    sys.path.insert(0,str(R));sys.path.insert(0,str(root.parent.parent))
    import mlx.core as mx
    import mlx.nn as nn
    import importlib.metadata
    import tutorial_code.miniengine as package
    from tutorial_code.miniengine import cache,context
    from tutorial_code.miniengine.engine.llm_engine import LLMEngine
    from tutorial_code.miniengine.spec.dflash2 import load_drafter
    from mlx.utils import tree_flatten
    from transformers import AutoTokenizer
    need(Path(package.__file__).resolve().parent==root.resolve(),'frozen target and drafter code')
    result['device']={'name':mx.device_info().get('device_name'),'architecture':mx.device_info().get('architecture'),'mlx':importlib.metadata.version('mlx')}
    need(result['device']=={'name':'Apple M5 Ultra','architecture':'applegpu_g17d','mlx':'0.32.3'},'M5 runtime')
    rt=SimpleNamespace(mx=mx,silu=nn.silu,BatchKV=cache.BatchKV,BatchState=cache.BatchState,ForwardContext=context.ForwardContext,DecodeBatch=cache.DecodeBatch)
    candidate=load(B/'rowwise_mlp_shared_features/candidate.py','_e060_candidate')
    shared=load(B/'nvfp4_multirow_shared/candidate.py','_e060_projector')
    selection=load(HERE/'selection.py','_e060_selection')
    engine=model=draft=project=tokenizer=seed=gold=actual=ctx=feats=logits=holder=layers=None
    records=[];raw={'schema':'E060-generation-pairs-v1','pairs':[]}
    try:
        engine=LLMEngine(str(model_path),max_len=2048,max_seqs=1,max_batched_tokens=16384,cache_limit_gb=2,dtype=mx.bfloat16,compile_decode=True,fuse_decode=True)
        model=engine.model
        result['eos_ids']=sorted(engine.eos_ids);result['native_runner_qualified']=[]
        for layer in model.layers:
            if not layer.is_attn:layer.linear_attn.set_gates_compile(True);layer.linear_attn.set_norm_reuse(False)
        draft=load_drafter(plan['drafter'],device_selection=True,bounded_kv=False)
        need(tuple(draft.cfg.target_layer_ids)==CAPTURE and draft.cfg.block_size==8,'trained full8 drafter')
        project=shared.build(mx,enabled=True);need(project is not None,'M5 shared kernel')
        tokenizer=AutoTokenizer.from_pretrained(str(model_path),local_files_only=True)
        before=h.parameter_hashes(mx,model,tree_flatten,deadline);write(out/'parameters-before.json',before)
        draft_before=h.parameter_hashes(mx,draft,tree_flatten,deadline);write(out/'drafter-before.json',draft_before)
        arrays=previous.all_arrays
        def snapshot(value):
            data={'logits':value['logits'],'features':value['features']}
            for i,c in enumerate(value['layers']):
                for key in (('K','V') if hasattr(c,'K') else ('conv','S')):data[f'state/{i}/{key}']=getattr(c,key)
            need(len(data)==130,'all arrays');return data
        def prefill(item):
            # Explicit synthetic context; two previously observed texts, no hidden model-generated history.
            unit=tokenizer.encode(item['context'],add_special_tokens=False)
            tail=tokenizer.encode(item['question'],add_special_tokens=False)
            need(unit and len(tail)<1024,'prompt construction')
            ids=(unit*((1024//len(unit))+1))[:1024-len(tail)]+tail
            caches=cache.make_cache(model)
            lo,fe=model(mx.array(ids,dtype=mx.int32),context.ForwardContext('prefill',caches),last_only=True,capture=CAPTURE)
            batch=cache.DecodeBatch(model);batch.rebuild([], [('e060',caches,1536)])
            mx.eval(lo,fe,*arrays(batch.layers));mx.synchronize()
            anchor=int(mx.argmax(lo[0]).item());need(anchor not in engine.eos_ids,'nonterminal seed')
            return dict(layers=batch.layers,features=fe,logits=lo,anchor=anchor,position=1024,prompt_ids=ids)
        result['seed_checks']=[]
        def seed_save(seed,label):
            path=out/('seed-'+label+'.safetensors');mx.save_safetensors(str(path),snapshot(seed))
            return dict(path=str(path.relative_to(out)),sha256=sha(path))
        def seed_check(seed,original,label):
            previous.compare(mx,mx.load(str(out/original['path'])),snapshot(seed))
            after=seed_save(seed,label);need(original['sha256']==after['sha256'],'canonical seed bytes changed')
            result['seed_checks'].append(dict(before=original,after=after))
        def generate(seed,arm,*,capture=False,stress=False):
            """One pending anchor. Timing includes reset, full draft, verify, select and install."""
            draft.reset();layers=seed['layers'];pending=seed['anchor'];position=seed['position'];new_features=seed['features']
            outputs=[pending];feature_chunks=[];rounds=[];last=None
            while len(outputs)<32 and outputs[-1] not in engine.eos_ids:
                deadline();remaining=32-len(outputs)
                if arm=='native':
                    c=cost.fresh_context(rt,layers,position)
                    value=(model(mx.array([pending],dtype=mx.int32),c,capture=CAPTURE) if capture else engine.runner.decode_model(mx.array([pending],dtype=mx.int32),c))
                    if capture:last,fe=value;feature_chunks.append(fe)
                    else:last=value
                    next_token=int(mx.argmax(last[0]).item());mx.eval(last,*arrays(c.caches))
                    layers=c.caches;position+=1;outputs.append(next_token);pending=next_token
                    continue
                width=int(arm[-1])
                if stress:width=(2,4,8)[len(rounds)%3]
                start=time.perf_counter_ns()
                proposals=draft.draft(new_features,pending,position,model.embed_tokens,model.lm_head_fn,block_size=8)
                need(len(proposals)==7,'real draft8');mx.eval([v for l in draft.layers for v in (l.self_attn.ctx_k,l.self_attn.ctx_v)])
                draft_ns=time.perf_counter_ns()-start
                # Exercise first rejection using a deliberately wrong draft token in an untimed qualification only.
                if stress:
                    c0=cost.fresh_context(rt,layers,position);lo0,_=model(mx.array([pending],dtype=mx.int32),c0,capture=CAPTURE)
                    proposals[0]=(int(mx.argmax(lo0[0]).item())+1)%model.cfg.vocab_size
                t=time.perf_counter_ns()
                v=candidate.verify(model,mx.array([pending]+proposals[:width-1],dtype=mx.int32),cost.fresh_context(rt,layers,position),rt,enabled=True,head_project=project,mlp_input_project=project,capture=CAPTURE)
                need(v is not None,'private verification admitted')
                posterior=mx.argmax(v.logits,axis=-1).tolist()
                sel=selection.select(proposals[:width-1],posterior,remaining,engine.eos_ids)
                count=sel.computed_count;layers=v.prefixes[count-1];new_features=v.features[:count]
                last=v.logits[count-1:count]
                mx.eval(last,new_features,*arrays(layers));mx.synchronize()
                verify_commit_ns=time.perf_counter_ns()-t
                if capture:feature_chunks.append(new_features)
                outputs.extend(sel.produced);position+=count;pending=outputs[-1]
                raw_accepted=0
                while raw_accepted<width-1 and proposals[raw_accepted]==posterior[raw_accepted]:raw_accepted+=1
                row={'width':width,'accepted':raw_accepted,'delivered':len(sel.produced),'draft_ns':draft_ns,'verify_commit_ns':verify_commit_ns,'draft_ids':list(proposals),'posterior':posterior,'produced':list(sel.produced),'position_after':position}
                rounds.append(row);v=None
                if sel.finished:break
            mx.eval(last,*arrays(layers));mx.synchronize()
            return dict(ids=outputs,layers=layers,features=mx.concatenate(feature_chunks,axis=0) if capture else None,logits=last,rounds=rounds,position=position)
        def save_pair(label,left,right):
            need(left['ids']==right['ids'] and left['position']==right['position'],'actual greedy outputs/position '+label)
            a,b=snapshot(left),snapshot(right);previous.compare(mx,a,b)
            row={'label':label,'output_ids':left['ids'],'computed_rows':left['position']-1024}
            for arm,values in (('reference',a),('candidate',b)):
                p=out/'raw'/f'{label}-{arm}.safetensors';p.parent.mkdir(exist_ok=True);mx.save_safetensors(str(p),values)
                row[arm]={'path':str(p.relative_to(out)),'sha256':sha(p)}
            raw['pairs'].append(row);write(out/'pairs.json',raw)
        seeds=[];oracles=[];seed_pins=[]
        for item in plan['prompts']:
            seed=prefill(item);seeds.append(seed);seed_pins.append(seed_save(seed,item['label']+'-before'));gold=generate(seed,'native',capture=True);oracles.append(gold)
            for arm in ('fixed2','fixed4','fixed8','stress'):
                actual=generate(seed,'fixed4' if arm=='stress' else arm,capture=True,stress=arm=='stress')
                save_pair(item['label']+'-'+arm,gold,actual);actual=None
            seed_check(seed,seed_pins[-1],item['label']+'-after-qualification')
            # Strong native no-capture runner must also match all state/logits, before it is timed.
            actual=generate(seed,'native');previous.compare(mx,{k:v for k,v in snapshot(gold).items() if k!='features'},{k:v for k,v in snapshot(actual).items() if k!='features'});need(actual['ids']==gold['ids'],'strong native oracle');result['native_runner_qualified'].append(item['label']);actual=None
            write(out/'complete.json',result)
        result['completed_pairs']=len(raw['pairs']);result['numerical_screen_qualified']=True
        mid=h.parameter_hashes(mx,model,tree_flatten,deadline);need(h.parameter_values(before)==h.parameter_values(mid),'target parameters unchanged before timing')
        result['pre_timing_parameters_unchanged']=True;write(out/'parameters-before-timing.json',mid)
        # One full warmup per prompt/arm, then 3 groups with rotation across all four arms.
        for seed,gold in zip(seeds,oracles):
            for arm in ARMS:
                actual=generate(seed,arm);need(actual['ids']==gold['ids'],'warm output IDs');actual=None
        for group in range(3):
            for case,(seed,gold) in enumerate(zip(seeds,oracles)):
                shift=(group+case)%4;order=ARMS[shift:]+ARMS[:shift]
                for arm in order:
                    mx.synchronize();t=time.perf_counter_ns();actual=generate(seed,arm);elapsed=time.perf_counter_ns()-t
                    need(actual['ids']==gold['ids'],'timed output IDs')
                    record={'group':group,'case':case,'arm':arm,'elapsed_ns':elapsed,'output_ids':actual['ids'],'decoded_tokens':len(actual['ids'])-1,'rounds':actual['rounds']}
                    records.append(record);write(out/'timings-partial.json',records);actual=None
        result['timings']=records;result['test_prompt_ids']=[s['prompt_ids'] for s in seeds]
        result['raw_manifest']={'path':'pairs.json','sha256':sha(out/'pairs.json')}
        draft.reset();after=h.parameter_hashes(mx,model,tree_flatten,deadline);draft_after=h.parameter_hashes(mx,draft,tree_flatten,deadline)
        need(h.parameter_values(before)==h.parameter_values(after),'target parameters unchanged')
        need(h.parameter_values(draft_before)==h.parameter_values(draft_after),'draft parameters unchanged')
        write(out/'parameters-after.json',after);write(out/'drafter-after.json',draft_after)
        for i,saved_seed in enumerate(seeds):seed_check(saved_seed,seed_pins[i],f'test{i}-after-timing')
        result['canonical_unchanged']=True
        # Release the closure cells before memory cleanup; saved raw evidence owns no device arrays.
        seeds=oracles=gold=seed=actual=saved_seed=None;generate=prefill=snapshot=save_pair=seed_check=seed_save=None
    except BaseException as error:
        traceback.clear_frames(error.__traceback__);raise
    finally:
        draft=None;model=None;project=None;tokenizer=None;seed=gold=actual=ctx=feats=logits=holder=layers=None
        if engine is not None:engine.runner.batch.clear();engine.runner.staging.clear();engine.runner.feats.clear();engine.runner._ahead=None
        engine=None;gc.collect();mx.synchronize();mx.clear_cache();result['device_cleanup_completed']=True
        result['memory_after_cleanup']={'active':mx.get_active_memory(),'peak':mx.get_peak_memory(),'cache':mx.get_cache_memory()}
    support.verify_files(root,sources,deadline);support.verify_files(model_path,weights,deadline);validate(plan);result['source_model_pins_rechecked']=True

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',action='store_true');p.add_argument('--plan',type=Path);p.add_argument('--plan-sha256');p.add_argument('--out',type=Path);a=p.parse_args()
    if not a.run:print('Default OFF: E060 actual speculative generation.');return 0
    need(a.plan and a.out and sha(a.plan)==a.plan_sha256,'explicit bound plan');plan=json.loads(a.plan.read_text());validate(plan);a.out.mkdir(exist_ok=False);write(a.out/'plan.json',plan);started=time.monotonic()
    result=dict(schema='E060-generation-result-v1',status='INCOMPLETE',pid=os.getpid(),ppid=os.getppid(),plan_sha256=a.plan_sha256,completed_pairs=0,timings=[],numerical_screen_qualified=False,pre_timing_parameters_unchanged=False,canonical_unchanged=False,device_cleanup_completed=False,source_model_pins_rechecked=False,http_qualified=False,performance_qualified=False,engine_pareto_point=False,independent_raw_audit_pending=True)
    previous={}
    try:
        def deadline():need(time.monotonic()-started<plan['deadline_s'],'bounded deadline')
        def stop(*unused):raise InterruptedError('E060 signal')
        need(signal.getitimer(signal.ITIMER_REAL)==(0.,0.),'existing alarm')
        previous={s:signal.getsignal(s) for s in (signal.SIGALRM,signal.SIGINT,signal.SIGTERM)}
        for s in previous:signal.signal(s,stop)
        signal.setitimer(signal.ITIMER_REAL,plan['deadline_s']);device(plan,a.out,deadline,result)
        need(result['completed_pairs']==8 and len(result['timings'])==24,'complete record count');result['status']='PASS_GENERATION_LOCAL_COST'
    except BaseException as e:result.update(status='FAIL',error={'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()})
    finally:
        if previous:
            signal.setitimer(signal.ITIMER_REAL,0)
            for s,handler in previous.items():signal.signal(s,handler)
        result['elapsed_s']=time.monotonic()-started;write(a.out/'complete.json',result)
    print(json.dumps({k:result[k] for k in ('status','completed_pairs','elapsed_s')}));return 0 if result['status']=='PASS_GENERATION_LOCAL_COST' else 1
if __name__=='__main__':raise SystemExit(main())
