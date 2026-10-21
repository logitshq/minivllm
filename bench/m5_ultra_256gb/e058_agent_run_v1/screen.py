"""Bounded E058 real feature/state qualification and local cost; default inert."""
import argparse, gc, hashlib, importlib.util, json, os, signal, sys, time, traceback
from pathlib import Path
from types import SimpleNamespace
HERE=Path(__file__).resolve().parent
R=Path('/Users/ryanli/code/minivllm');B=R/'bench/m5_ultra_256gb'
OLD=B/'rowwise_mlp_shared_cost_v1/screen.py'
OLD_SHA='2922675fca185ca0fd04f3da6d22c96268fe4b41d66bdd305a2ec914f778be9c'
CAPTURE=(5,19,33,47,61)
def need(ok,msg):
 if not ok:raise ValueError(msg)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def load(p,name):
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def pinned(ref):
 need(set(ref)=={'path','sha256'} and Path(ref['path']).is_absolute() and sha(ref['path'])==ref['sha256'],'exact prerequisite')
 return json.loads(Path(ref['path']).read_text())
def validate(plan):
 need(set(plan)=={'schema','deadline_s','sources','e057_plan','e057_result','e057_audit'},'exact plan fields')
 need(plan['schema']=='E058-feature-plan-v1' and type(plan['deadline_s']) is int and 0<plan['deadline_s']<=1500,'bounded plan')
 need({str(HERE/name) for name in ('screen.py','implementation.py','timing.py','audit.py')}<=set(plan['sources']),'own modules pinned')
 for p,digest in plan['sources'].items():need(sha(p)==digest,'source changed '+p)
 need(sha(OLD)==OLD_SHA,'frozen E057 source')
 previous=pinned(plan['e057_plan']);complete=pinned(plan['e057_result']);audit=pinned(plan['e057_audit'])
 need(complete['status']=='PASS_LOCAL_COST_ONLY' and audit['status']=='PASS' and audit['complete_sha256']==plan['e057_result']['sha256'],'closed E057')
 need(complete['plan_sha256']==plan['e057_plan']['sha256'],'same E057 plan/result')
 return load(OLD,'_e058_old_e057').validate(previous)
def all_arrays(layers):return [a for c in layers for a in ((c.K,c.V) if hasattr(c,'K') else (c.conv,c.S))]
def snap(output,k):
 values={'logits':output.logits[:k],'features':output.features[:k]}
 for i,c in enumerate(output.prefixes[k-1]):
  for key in (('K','V') if i%4==3 else ('conv','S')):values[f'state/{i}/{key}']=getattr(c,key)
 need(len(values)==130,'complete feature/state snapshot');return values
def compare(mx,left,right):
 need(left.keys()==right.keys(),'snapshot names')
 flags=[]
 for name in left:
  a,b=left[name],right[name];need(a.shape==b.shape and a.dtype==b.dtype,'snapshot metadata '+name)
  dt=mx.uint16 if a.dtype==mx.bfloat16 else mx.uint32 if a.dtype==mx.float32 else a.dtype
  flags.extend([mx.array_equal(a.view(dt),b.view(dt)),mx.all(mx.isfinite(a)),mx.all(mx.isfinite(b))])
 mx.eval(*flags)
 failed=[name for i,name in enumerate(left) if not all(bool(v.item()) for v in flags[3*i:3*i+3])]
 need(not failed,'original-byte/finite comparison: '+','.join(failed))
def device(plan,out,deadline,result):
 cost,old_candidate,old_plan,q,deps,seed=validate(plan)
 bridge,frozen,h,old,support,original,inherited=deps
 source=support.pinned_json(inherited['source_manifest'])['files'];manifest=support.pinned_json(inherited['model_manifest'])
 root=Path(inherited['package_root']).resolve()/'tutorial_code/miniengine';model_path=Path(inherited['model']).resolve()
 need(source==json.loads(bridge.CORE_PINS.read_text())['files'] and set(source)=={str(p.relative_to(root)) for p in root.rglob('*.py')},'complete frozen model source')
 support.verify_files(root,source,deadline);support.verify_files(model_path,manifest['files'],deadline)
 need({k:v for k,v in os.environ.items() if k.startswith('MLX_')}==inherited['execution']['environment'],'original environment')
 canonical=Path(old_plan['e042_result']['path']).parent/seed['canonical']['path']
 need(h.sha(canonical,deadline)==seed['canonical']['sha256'],'real original seed')
 sys.path.insert(0,str(R));sys.path.insert(0,str(root.parent.parent))
 import importlib.metadata
 import mlx.core as mx
 import mlx.nn as nn
 import tutorial_code.miniengine as package
 from tutorial_code.miniengine import cache,context
 from tutorial_code.miniengine.engine.llm_engine import LLMEngine
 from mlx.utils import tree_flatten
 need(Path(package.__file__).resolve().parent==root,'actual source root')
 result['device']={'name':mx.device_info().get('device_name'),'architecture':mx.device_info().get('architecture'),'mlx':importlib.metadata.version('mlx')}
 need(result['device']=={'name':'Apple M5 Ultra','architecture':'applegpu_g17d','mlx':'0.32.3'} and mx.default_device()==mx.gpu,'actual M5 runtime')
 rt=SimpleNamespace(mx=mx,silu=nn.silu,BatchKV=cache.BatchKV,BatchState=cache.BatchState,ForwardContext=context.ForwardContext,DecodeBatch=cache.DecodeBatch)
 impl=load(HERE/'implementation.py','_e058_agent_impl');timing=load(HERE/'timing.py','_e058_agent_timing')
 shared=load(B/'nvfp4_multirow_shared/candidate.py','_e058_project')
 engine=model=project=storage=reference=candidate=golden=cont_ref=cont_candidate=oracle=ctx=other=None
 left=right=callbacks=values=pristine=layer=candidate_ctx=None
 try:
  engine=LLMEngine(str(model_path),max_len=2048,max_seqs=1,max_batched_tokens=16384,cache_limit_gb=2,dtype=mx.bfloat16,compile_decode=True,fuse_decode=True)
  model=engine.model;layout=tuple(l.is_attn for l in model.layers)
  need(layout==tuple(i%4==3 for i in range(64)) and engine.runner.spec is None and engine.runner._ahead is None,'fixed original model')
  for layer in model.layers:
   if not layer.is_attn:layer.linear_attn.set_gates_compile(True);layer.linear_attn.set_norm_reuse(False)
  before=h.parameter_hashes(mx,model,tree_flatten,deadline);write(out/'parameters-before.json',before)
  storage=mx.load(str(canonical));mx.eval(*storage.values());project=shared.build(mx,enabled=True);need(project is not None,'shared projector')
  oracle=cost.native(rt,engine,h,layout,storage,1022,9,True,True)
  need(oracle['inputs'][:8]==seed['tokens'],'actual native teacher tokens')
  need(not (set(oracle['inputs'])|set(oracle['ids']))&set(engine.eos_ids),'normal EOS scope')
  teacher=list(oracle['inputs']);result['teacher']=teacher;oracle=None
  def fresh(position=1022,layers=None):
   return h.context_from_input(storage,layout,rt,1023) if layers is None else cost.fresh_context(rt,layers,position)
  def golden_forward(ids,initial):
   state=initial.caches;prefixes=[];logits=[];features=[]
   for i in range(ids.shape[0]):
    c=cost.fresh_context(rt,state,int(initial.max_len)-1+i)
    lo,fe=model(ids[i:i+1],c,capture=CAPTURE)
    state=c.caches;prefixes.append(list(state));logits.append(lo);features.append(fe)
   return SimpleNamespace(logits=mx.concatenate(logits,axis=0),features=mx.concatenate(features,axis=0),prefixes=prefixes)
  raw={'schema':'E058-feature-pairs-v1','pairs':[]}
  def save_pair(label,a,b):
   deadline();compare(mx,a,b)
   row={'label':label}
   for name,data in (('reference',a),('candidate',b)):
    path=out/'raw'/f"{label.replace('/','-')}-{name}.safetensors";path.parent.mkdir(exist_ok=True)
    mx.save_safetensors(str(path),data);row[name]={'path':str(path.relative_to(out)),'sha256':sha(path)}
   raw['pairs'].append(row);write(out/'pairs.json',raw)
  for width in (4,8):
   deadline();ids=mx.array(teacher[:width],dtype=mx.int32)
   reference=impl.reference_with_features(model,ids,fresh(),rt)
   golden=golden_forward(ids,fresh())
   candidate_ctx=fresh();candidate=impl.candidate_with_features(model,ids,candidate_ctx,rt,project)
   need(candidate is not None,'candidate admitted');q.prefix_ownership(candidate,candidate_ctx,layout,width)
   for k in range(1,width+1):
    # The agent's independent reference must first agree with actual model capture.
    compare(mx,snap(reference,k),snap(golden,k))
    save_pair(f'T{width}/k{k}/selected',snap(reference,k),snap(candidate,k))
    pending=int(mx.argmax(reference.logits[k-1:k],axis=-1).item());token=mx.array([pending],dtype=mx.int32)
    cont_ref=impl.reference_with_features(model,token,fresh(1022+k,reference.prefixes[k-1]),rt)
    cont_candidate=impl.reference_with_features(model,token,fresh(1022+k,candidate.prefixes[k-1]),rt)
    save_pair(f'T{width}/k{k}/continuation',snap(cont_ref,1),snap(cont_candidate,1))
    result['completed_pairs']=len(raw['pairs']);write(out/'complete.json',result)
    cont_ref=cont_candidate=None
   reference=candidate=golden=candidate_ctx=None
  pristine=mx.load(str(canonical));compare(mx,pristine,storage);pristine=None
  mid=h.parameter_hashes(mx,model,tree_flatten,deadline);write(out/'parameters-before-timing.json',mid)
  need(h.parameter_values(before)==h.parameter_values(mid),'parameter values unchanged before timing')
  result['numerical_screen_qualified']=True;result['pre_timing_parameters_unchanged']=True;write(out/'complete.json',result)
  def callback(width,arm):
   def invoke():
    deadline();ids=mx.array(teacher[:width],dtype=mx.int32);c=fresh()
    if arm=='native':v=golden_forward(ids,c)
    elif arm=='shared_with_features':v=impl.candidate_with_features(model,ids,c,rt,project)
    else:v=old_candidate.verify(model,ids,c,rt,enabled=True,head_project=project,mlp_input_project=project)
    need(v is not None,'timed candidate admitted')
    values=[v.logits,*all_arrays(v.prefixes[-1])]
    if getattr(v,'features',None) is not None:values.append(v.features)
    mx.eval(*values);mx.synchronize();del values,v,c,ids
   return invoke
  callbacks={w:{arm:callback(w,arm) for arm in timing.ARMS} for w in (4,8)}
  partial=[]
  def sample(row):partial.append(row);write(out/'timings-partial.json',partial)
  rows=timing.collect(callbacks,on_sample=sample);result['timings']=rows;result['timing_summary']=timing.reduce(rows)
  result['timing_scope']='Local teacher-forced verifier+full-logit/selected-state materialization and feature capture; native arm uses the original frozen model capture, token-major with supplied tokens, NOT device-dependent AR. No draft, actual acceptance, scheduler or HTTP. Shared-without-features is overhead diagnostic only.'
  after=h.parameter_hashes(mx,model,tree_flatten,deadline);write(out/'parameters-after.json',after)
  need(h.parameter_values(before)==h.parameter_values(after),'parameter values unchanged after timing')
  pristine=mx.load(str(canonical));compare(mx,pristine,storage)
  result['canonical_unchanged']=True;result['raw_manifest']={'path':'pairs.json','sha256':sha(out/'pairs.json')}
 except BaseException as error:
  traceback.clear_frames(error.__traceback__)
  raise
 finally:
  callbacks=None;storage=reference=candidate=golden=cont_ref=cont_candidate=oracle=ctx=other=left=right=values=pristine=project=layer=candidate_ctx=None
  model=None
  if engine is not None:engine.runner.batch.clear();engine.runner.staging.clear();engine.runner.feats.clear();engine.runner._ahead=None
  engine=None;gc.collect();mx.synchronize();mx.clear_cache();result['device_cleanup_completed']=True
  result['memory_after_cleanup']={'active':mx.get_active_memory(),'peak':mx.get_peak_memory(),'cache':mx.get_cache_memory()}
 support.verify_files(root,source,deadline);support.verify_files(model_path,manifest['files'],deadline);validate(plan)
 result['source_model_pins_rechecked']=True

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',action='store_true');p.add_argument('--plan',type=Path);p.add_argument('--plan-sha256');p.add_argument('--out',type=Path);a=p.parse_args()
 if not a.run:print('Default OFF: E058 feature qualification and local cost.');return 0
 need(all((a.plan,a.plan_sha256,a.out)),'complete explicit arguments');need(sha(a.plan)==a.plan_sha256,'plan SHA')
 plan=json.loads(a.plan.read_text());validate(plan);a.out.mkdir(exist_ok=False);write(a.out/'plan.json',plan);start=time.monotonic()
 result={'schema':'E058-feature-result-v1','status':'INCOMPLETE','pid':os.getpid(),'ppid':os.getppid(),'plan_sha256':a.plan_sha256,'completed_pairs':0,'timings':[],
 'numerical_screen_qualified':False,'pre_timing_parameters_unchanged':False,'canonical_unchanged':False,'device_cleanup_completed':False,'source_model_pins_rechecked':False,
 'http_qualified':False,'performance_qualified':False,'engine_pareto_point':False,'independent_raw_audit_pending':True}
 previous={}
 try:
  def deadline():need(time.monotonic()-start<plan['deadline_s'],'child deadline')
  def stop(*unused):raise InterruptedError('E058 bounded signal')
  need(signal.getitimer(signal.ITIMER_REAL)==(0.0,0.0),'existing alarm')
  previous={s:signal.getsignal(s) for s in (signal.SIGALRM,signal.SIGINT,signal.SIGTERM)}
  for s in previous:signal.signal(s,stop)
  signal.setitimer(signal.ITIMER_REAL,plan['deadline_s']);device(plan,a.out,deadline,result)
  need(result['completed_pairs']==24 and len(result['timings'])==60,'complete planned records');result['status']='PASS_FEATURES_LOCAL_COST'
 except BaseException as e:result.update(status='FAIL',error={'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()})
 finally:
  if previous:
   signal.setitimer(signal.ITIMER_REAL,0)
   for s,handler in previous.items():signal.signal(s,handler)
  result['elapsed_s']=time.monotonic()-start;write(a.out/'complete.json',result)
 print(json.dumps({k:result[k] for k in ('status','completed_pairs','elapsed_s')}));return 0 if result['status']=='PASS_FEATURES_LOCAL_COST' else 1
if __name__=='__main__':raise SystemExit(main())
