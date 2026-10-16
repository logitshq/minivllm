"""Bounded V19 model worker; V17 state_arrays/normal runner protocol are reused."""
import ast, gc, importlib.abc, importlib.machinery, importlib.util, json, sys
from pathlib import Path

class SourceFinder(importlib.abc.MetaPathFinder):
 def __init__(self,identity,root,files):self.identity,self.root,self.files=identity,root,files
 def find_spec(self,name,path=None,target=None):
  if name=='tutorial_code':
   spec=importlib.machinery.ModuleSpec(name,None,is_package=True);spec.submodule_search_locations=[str(self.root.parent)];return spec
  if not name.startswith('tutorial_code.miniengine'):return None
  suffix=name.removeprefix('tutorial_code.miniengine').replace('.','/').lstrip('/')
  rel=(suffix+'/__init__.py').lstrip('/');package=rel in self.files
  if not package:rel=suffix+'.py'
  if rel not in self.files:raise ImportError('unpinned source '+name)
  entry={'path':str(self.root/rel),'sha256':self.files[rel],'is_package':package}
  return importlib.util.spec_from_file_location(name,entry['path'],loader=self.identity.VerifiedSourceLoader(entry),submodule_search_locations=[str(Path(entry['path']).parent)] if package else None)

def install_hooks(gdn,gdn_metal,norm,mx,gates,restores):
 """Each hook delegates the actual original callable; finally restores layer context."""
 current=[None];counts={}
 def reset_counts():
  counts.clear();counts.update({k:{} for k in ('norm_public','legacy','decode_norm','prepare','state','prework_dispatch','state_dispatch','compiled_gates','kernel_creations')});counts['compile_creations']=0
 def hit(kind):
  key=str(current[0]);counts[kind][key]=counts[kind].get(key,0)+1
 def replace(obj,name,fn):
  old=getattr(obj,name);restores.append((obj,name,old));setattr(obj,name,fn)
 original_decode=gdn.GatedDeltaNet._decode
 def decode(self,*a,**kw):
  before=current[0];current[0]=next(i for i,l in gates.items() if l is self)
  try:return original_decode(self,*a,**kw)
  finally:current[0]=before
 replace(gdn.GatedDeltaNet,'_decode',decode)
 legacy=gdn_metal.gated_delta_step_metal
 def legacy_call(*a,**kw):hit('legacy');return legacy(*a,**kw)
 replace(gdn_metal,'gated_delta_step_metal',legacy_call)
 if norm:
  for key in ('_prework_kernel','_state_kernel','_norm_probe_kernel'):restores.append((norm,key,getattr(norm,key)))
  production=norm.gated_delta_step_norm_reuse_metal;original_norm=gdn.GatedDeltaNet._decode_norm_reuse
  prepare=norm._prepare;state=norm._state;factory=norm._make_kernel
  def public_call(*a,**kw):
   if current[0] is None:raise ValueError('N2 outside actual GDN decode')
   hit('norm_public');return production(*a,**kw)
  def decode_norm(self,*a,**kw):hit('decode_norm');return original_norm(self,*a,**kw)
  def prepare_call(*a,**kw):hit('prepare');return prepare(*a,**kw)
  def state_call(*a,**kw):hit('state');return state(*a,**kw)
  def factory_call(mx,name,*a,**kw):
   kind={'gdn_norm_reuse_prework_build1':'prework','gdn_norm_reuse_state_build1':'state'}.get(name)
   if kind is None:raise ValueError('unexpected qualification kernel factory '+name)
   original=factory(mx,name,*a,**kw);counts['kernel_creations'][kind]=counts['kernel_creations'].get(kind,0)+1
   def dispatch(*a,**kw):hit(kind+'_dispatch');return original(*a,**kw)
   return dispatch
  replace(norm,'gated_delta_step_norm_reuse_metal',public_call);replace(gdn.GatedDeltaNet,'_decode_norm_reuse',decode_norm)
  replace(norm,'_prepare',prepare_call);replace(norm,'_state',state_call);replace(norm,'_make_kernel',factory_call)
 compile0=mx.compile
 def compile_call(fn,*a,**kw):
  compiled=compile0(fn,*a,**kw)
  if fn is not gdn._gates_explicit:return compiled
  counts['compile_creations']+=1
  def invoked(*a,**kw):hit('compiled_gates');return compiled(*a,**kw)
  return invoked
 replace(mx,'compile',compile_call)
 return counts,reset_counts

def run(args,f,go,result,out,d):
 P=d.P;role=args.worker_role;root=Path(f[role+'_root']);resource=None;loaded=False;finder=source_finder=None;restores=[]
 try:
  resource=d.Resources(out,f,go);resource.sample('worker-before-MLX-import');d.environment_guard(f,result)
  identity=d.pinned('m5_runtime',d.D/'runtime.py',f)
  identity.reject_preloaded(sys.modules)
  result['runtime_preflight']=identity.preflight(f)
  source_finder=SourceFinder(identity,root,f[role+'_sources']);sys.meta_path.insert(0,source_finder)
  import mlx.core as mx
  import mlx.nn as nn
  import mlx.nn.layers.activations as activations
  import numpy as np
  from mlx.utils import tree_flatten
  result['runtime_identity_before']=identity.runtime_identity(f,mx,nn,activations)
  result['device']=mx.metal.device_info()
  if mx.default_device()!=mx.gpu or result['device']['device_name']!='Apple M5 Ultra' or result['device']['architecture']!='applegpu_g17d' or result['device']['memory_size']!=274877906944:raise ValueError('actual device differs')
  from tutorial_code.miniengine.engine.llm_engine import LLMEngine
  from tutorial_code.miniengine.engine.model_runner import ModelRunner
  from tutorial_code.miniengine.engine.scheduler import Scheduler
  from tutorial_code.miniengine.sampling_params import SamplingParams
  from tutorial_code.miniengine.cache import BatchState,DecodeBatch,make_cache
  from tutorial_code.miniengine.context import ForwardContext
  from tutorial_code.miniengine.model import gdn
  from tutorial_code.miniengine.model.layers import prefill_batch_backend_supported,_prefill_backend_identity
  from tutorial_code.miniengine.kernels import gdn_metal
  norm=None
  if role=='candidate':
   from tutorial_code.miniengine.kernels import gdn_norm_reuse_metal as norm
  if prefill_batch_backend_supported():raise ValueError('M5 must retain the existing NAX prefill fallback')
  result['backend']=_prefill_backend_identity()
  # Reuse reviewed schema/function, executing only its frozen AST, not old main.
  old=d.D/'state_reference.py';node=next(x for x in ast.parse(old.read_bytes()).body if isinstance(x,ast.FunctionDef) and x.name=='state_arrays')
  ns={};exec(compile(ast.Module(body=[node],type_ignores=[]),str(old),'exec',dont_inherit=True),ns);state_arrays=ns['state_arrays']
  w=json.loads((P/'workload.json').read_text());expected=json.loads((P/'chapter15-smoke-expected-v1.json').read_text())
  if w['prompt_ids']!=expected['prompt_ids'] or len(w['prompt_ids'])!=19 or w['model_dir']!=expected['checkpoint']['actual_path']:raise ValueError('workload differs')
  if {n:d.digest(Path(w['model_dir'])/n) for n in expected['checkpoint']['metadata_sha256']}!=expected['checkpoint']['metadata_sha256']:raise ValueError('checkpoint metadata differs')
  result['model_load_attempts']=1;d.write(out/'receipt.json',result)
  e=LLMEngine(w['model_dir'],max_seqs=4,max_len=2048,max_batched_tokens=512,dtype=mx.bfloat16,cache_limit_gb=0,drafter_dir=None)
  model=e.model;loaded=True;mx.synchronize();resource.sample('model-ready-cache0-noDraft',True)
  gates={i:l.linear_attn for i,l in enumerate(model.layers) if not l.is_attn};layers=[i for i in range(64) if i%4!=3]
  if list(gates)!=layers:raise ValueError('GDN layers differ')
  def params():return [{'key':k,'object_id':id(v),'shape':list(v.shape),'dtype':str(v.dtype)} for k,v in tree_flatten(model.parameters())]
  initial=params();
  if len(initial)!=1349:raise ValueError('parameter count differs')
  result['parameters']={'initial':initial,'unchanged_at':[]}
  def tree(label):
   if params()!=initial:raise ValueError('NN parameter identity changed: '+label)
   for l in gates.values():
    if any(k.startswith(('_gates_','_compiled_','_norm_reuse_')) for k in l):raise ValueError('private metadata became NN parameter')
   result['parameters']['unchanged_at'].append(label)
  if any(object.__getattribute__(l,'_gates_compile_enabled') is not False or (norm and object.__getattribute__(l,'_norm_reuse_enabled') is not False) for l in gates.values()):raise ValueError('default-off violated')
  if norm and any(getattr(norm,k) is not None for k in ('_prework_kernel','_state_kernel','_norm_probe_kernel')):raise ValueError('N2 import created pipeline')
  pipeline_ids=None;result['pipeline_lifecycle']=[]
  tree('loaded-default-off');result['default_off_checked']=True
  counts,reset_counts=install_hooks(gdn,gdn_metal,norm,mx,gates,restores)
  def toggle(on):
   for l in gates.values():
    l.set_gates_compile(True)
    if norm:l.set_norm_reuse(on)
  catalog=[];result['raw_logits']=catalog
  def canonical(x,raw=None):
   value=np.ascontiguousarray(np.asarray(x.astype(mx.float32),dtype='<f4'));flat=value.reshape(-1)
   for start in range(0,flat.size,1<<18):
    if not np.isfinite(flat[start:start+(1<<18)]).all():raise ValueError('nonfinite state/output')
   sig={'shape':list(x.shape),'dtype':str(x.dtype),'sha256':__import__('hashlib').sha256(memoryview(value).cast('B')).hexdigest(),'canonical_bytes':value.nbytes}
   if raw:
    with raw.open('xb') as stream:stream.write(memoryview(value).cast('B'))
    catalog.append({'file':str(raw.relative_to(out)),'signature':sig})
   return sig
  def idle():
   if e.has_work() or e.requests or e.runner.staging or e.runner.batch.size or e.runner._ahead is not None or e.runner.feats:raise ValueError('cache lifecycle not empty')
  result['rounds']=[];expected_obs=None
  if role=='candidate':expected_obs=json.loads((args.output_dir/'accepted'/'accepted-observations.json').read_text())
  rounds=[('accepted',False)] if role=='accepted' else [('candidate-off',False),('candidate-on',True),('candidate-reset-off',False)]
  for name,on in rounds:
   idle();toggle(on);reset_counts();tree(name+'-before');observations=[];(out/name).mkdir()
   class RecordingModel:
    def __getattr__(self,key):return getattr(model,key)
    def __call__(self,ids,ctx,**kw):
     output=model(ids,ctx,**kw);logits=output[0] if isinstance(output,tuple) else output
     positions=ctx.positions.tolist() if ctx.mode=='decode' else [18];length=positions[0]+1
     if ctx.mode=='decode' and (ctx.lens.tolist()!=[length] or ctx.max_len!=length):raise ValueError('position/lens mismatch')
     index=len(observations);rec={'mode':ctx.mode,'positions':positions,'context_lens':ctx.lens.tolist() if ctx.mode=='decode' else None,'context_max_len':ctx.max_len,'input_ids':ids.tolist(),'logits':canonical(logits,out/name/('%02d-logits.float32le.bin'%index)),'greedy':mx.argmax(logits,axis=-1).tolist(),'states':{}}
     for label,arr in state_arrays(ctx.mode,ctx.caches,length):rec['states'][label]=canonical(arr)
     if len(rec['states'])!=128:raise ValueError('missing state signature')
     observations.append(rec);d.write(out/(name+'-observations.json'),observations)
     if expected_obs is not None and rec!=expected_obs[index]:raise ValueError('strict FP32 signature mismatch '+name+' step '+str(index))
     return output
   e.scheduler=Scheduler(4,2048,512,e.eos_ids);e.runner=ModelRunner(RecordingModel(),2048);e._counter=0
   resource.sample(name+'-before-request',True)
   e.add_request(w['prompt_ids'],SamplingParams(max_tokens=32,temperature=0,ignore_eos=False),req_id='R0');delivered=[];committed=[]
   while e.has_work():
    rows=e.step();step=len(delivered)+1
    if len(rows)!=1 or rows[0][0]!='R0' or rows[0][2]!=(step==32) or step>32:raise ValueError('normal EOS/row differs')
    delivered.append(rows[0][1]);record={'step':step,'batch_req_ids':list(e.runner.batch.req_ids),'batch_lens':list(e.runner.batch.lens),'staging_offsets':{rid:[c.offset for c in caches] for rid,caches in e.runner.staging.items()}}
    if step==1 and (record['batch_req_ids'] or record['staging_offsets']!={'R0':[19]*64}):raise ValueError('prefill cache boundary differs')
    if 1<step<32 and (record['batch_req_ids']!=['R0'] or record['batch_lens']!=[18+step] or record['staging_offsets']):raise ValueError('decode cache boundary differs')
    committed.append(record)
   idle()
   if delivered!=expected['expected_output_ids'] or any(t in e.eos_ids for t in delivered):raise ValueError('expected 32 IDs differ')
   if len(observations)!=32 or [r['mode'] for r in observations]!=['prefill']+['decode']*31 or [r['positions'][0] for r in observations]!=list(range(18,50)) or [r['greedy'][0] for r in observations]!=delivered:raise ValueError('observations differ')
   want={str(i):31 for i in gates}
   for kind in ('norm_public','decode_norm','prepare','state','prework_dispatch','state_dispatch'):
    if counts[kind]!=(want if on else {}):raise ValueError('actual N2 delegated counts differ: '+str(counts))
   if counts['legacy']!=({} if on else want) or counts['compiled_gates']!=want or counts['compile_creations']!=48:raise ValueError('actual legacy/gates counts differ: '+str(counts))
   if counts['kernel_creations']!=({'prework':1,'state':1} if on else {}):raise ValueError('N2 factory creations differ')
   if norm:
    now=tuple(getattr(norm,k) for k in ('_prework_kernel','_state_kernel'))
    if name=='candidate-off' and any(k is not None for k in now):raise ValueError('OFF created N2 factories')
    if on:
     if any(k is None for k in now):raise ValueError('ON missed N2 factories')
     pipeline_ids=tuple(id(k) for k in now)
    if name=='candidate-reset-off' and tuple(id(k) for k in now)!=pipeline_ids:raise ValueError('reset changed N2 factory cache')
    if norm._norm_probe_kernel is not None:raise ValueError('qualification probe entered model path')
    result['pipeline_lifecycle'].append({'round':name,'prework_present':now[0] is not None,'state_present':now[1] is not None,'probe_absent':True,'reset_preserves_factories':name=='candidate-reset-off'})
   row={'name':name,'on':on,'output_ids':delivered,'committed_positions':committed,'observations_file':name+'-observations.json','canonical_array_hashes':4128,'routes':json.loads(json.dumps(counts)),'fresh_cache_idle':True}
   result['rounds'].append(row);tree(name+'-after');e.runner=ModelRunner(model,2048);gc.collect();mx.clear_cache();resource.sample(name+'-after-idle',True);d.write(out/'receipt.json',result)
  if norm:
   resource.sample('fallbacks-before',True)
   helper=d.pinned('fullqual_fallback',d.D/'fallback.py',f)
   helper.run(model,gates,e,w,mx,BatchState,ForwardContext,make_cache,DecodeBatch,toggle,reset_counts,counts,canonical,state_arrays,result)
   tree('fallbacks-after');resource.sample('fallbacks-after',True)
   for l in gates.values():
    for invalid in (None,0,1,'on'):
     try:l.set_norm_reuse(invalid)
     except TypeError:pass
     else:raise ValueError('setter accepted nonbool')
    l.set_norm_reuse(False);l.set_gates_compile(False)
    if object.__getattribute__(l,'_norm_reuse_enabled') or object.__getattribute__(l,'_compiled_decode_gates') is not None:raise ValueError('reset failed')
   tree('final-reset-off')
  for name,module in list(sys.modules.items()):
   if name.startswith('tutorial_code.miniengine'):
    rel=str(Path(module.__file__).relative_to(root));identity.module_identity(name,module,{'path':str(root/rel),'sha256':f[role+'_sources'][rel],'is_package':rel.endswith('__init__.py')})
  result['runtime_identity_after']=identity.runtime_identity(f,mx,nn,activations);result['raw_logits']=catalog
  d.verify_frozen(args.frozen_sha256);result['status']='PASS_WORKER_BOUNDED_SCOPE'
 finally:
  for obj,name,old in reversed(restores):setattr(obj,name,old)
  result['hooks_restored']=all(getattr(obj,name) is old for obj,name,old in restores)
  for v in (source_finder,finder):
   if v in sys.meta_path:sys.meta_path.remove(v)
  result['process_exit_is_unload_boundary']=True
  if resource:
   try:resource.sample('worker-finally-before-process-exit',loaded)
   except BaseException as error:result.setdefault('cleanup_errors',[]).append(str(error));result['status']='STOP_FIRST_FAILURE'
   result['raw_resources']=resource.raws
