"""V27 bounded full32 numeric exploration. --inert is stdlib-only."""
from pathlib import Path
import argparse,ast,copy,hashlib,importlib.metadata,importlib.util,json,os,sys,time,traceback
D=Path(__file__).resolve().parent;P=D.parent;ROOT=P/'candidate-v27-gdn-simd-build1-root';PKG=ROOT/'tutorial_code/miniengine'
LOGIT_BOUND=1/16;CACHE_L2_BOUND=2**-10
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def sources():
 pins=json.loads((D/'fullqual-pins.json').read_text())
 for path,expected in pins.items():
  if sha(path)!=expected:raise ValueError('source changed: '+path)
 manifest=json.loads((D/'source-manifest.json').read_text());actual={str(p.relative_to(ROOT)):sha(p) for p in ROOT.rglob('*') if p.is_file() and p.suffix in ('.py','.metal')}
 if actual!=manifest['files']:raise ValueError('candidate source differs')
 return pins,manifest
def main(argv=None):
 ap=argparse.ArgumentParser();ap.add_argument('--inert',action='store_true');ap.add_argument('--out',type=Path);a=ap.parse_args(argv);pins,manifest=sources();w=json.loads((P/'workload.json').read_text())
 if len(w['prompt_ids'])!=19 or len(w['continuation_ids'])!=32:raise ValueError('fixed19prompt/32output required')
 if a.inert:print(json.dumps({'status':'CPU_INERT_ONLY','MLX_imported':any(n=='mlx' or n.startswith('mlx.') for n in sys.modules),'GPU_run':False,'pins':len(pins),'candidate_py_count':28,'logit_bound':LOGIT_BOUND,'cache_normalized_l2_bound':CACHE_L2_BOUND}));return 0
 if a.out is None or not a.out.is_absolute():raise ValueError('new absolute --out required; root owns timeout/PID cleanup')
 a.out.mkdir(exist_ok=False);r={'status':'STARTED','pid':os.getpid(),'source_manifest_sha256':sha(D/'source-manifest.json'),'source_pins':pins,'protocol_sha256':sha(D/'PROTOCOL.md'),'micro_status':'CONTROL_DRIFT_STOP_RETAINED','micro_qualified':False,'independent_fullmodel_exploration_after_micro_stop':True,'bitwise_equivalence_required':False,'bounds':{'logits_maxabs':LOGIT_BOUND,'cache_normalized_l2':CACHE_L2_BOUND},'steps':[],'warm_layout_replays':[],'model_loads':0,'performance_qualified':False,'service_lifecycle_qualified':False};save=lambda:write(a.out/'receipt.json',r);save();start=time.perf_counter();mx=None;loaded=False;resources=None
 try:
  utility=module('v27_reused_numeric_resources',P/'v24-fullmodel-qual-build1/fullqual.py');resources=utility.Resources(a.out);resources.sample('beforeload',False)
  versions={k:importlib.metadata.version(k) for k in ('mlx','mlx-metal')}
  if set(versions.values())!={'0.32.3'}:raise ValueError('runtime requires0.32.3')
  import mlx.core as mx
  import numpy as np
  from mlx.utils import tree_flatten
  device=mx.device_info()
  if mx.default_device()!=mx.gpu or device.get('device_name')!='Apple M4 Max' or device.get('architecture')!='applegpu_g16s':raise ValueError('requires actual M4 Max GPU')
  r['runtime']={'versions':versions,'device':device,'allocator_cache_limit_bytes':2<<30};mx.set_cache_limit(2<<30)
  sys.path.insert(0,str(ROOT))
  from tutorial_code.miniengine.model.loader import load_model
  from tutorial_code.miniengine.model.qwen3_5 import Qwen3_5ForCausalLM
  from tutorial_code.miniengine.model.gdn import GatedDeltaNet
  from tutorial_code.miniengine.cache import BatchKV,BatchState,DecodeBatch,make_cache
  from tutorial_code.miniengine.context import ForwardContext
  from tutorial_code.miniengine.eos import read_eos_ids
  from tutorial_code.miniengine.kernels.functional_decode import FunctionalDecode,DecodeTypes
  from tutorial_code.miniengine.kernels import gdn_simd_row
  local=module('v27_frozen_local_utils',P/'v27-gdn-simd-row-build1/common.py');descriptor=local.layout_probe(mx)
  class Numeric(utility.Numeric):
   def raw(self,value,label):
    if value.dtype==mx.int32:
     mx.eval(value);h=np.ascontiguousarray(np.asarray(value),dtype='<i4');data=h.tobytes();return data,{'shape':list(value.shape),'dtype':str(value.dtype),'finite':True,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
    return super().raw(value,label)
  numeric=Numeric(mx,np,a.out)
  def float_host(value,label):
   data,sig=numeric.raw(value,label)
   if value.dtype==mx.bfloat16:h=(np.frombuffer(data,dtype='<u2').astype(np.uint32)<<16).view(np.float32).reshape(value.shape)
   elif value.dtype==mx.float32:h=np.frombuffer(data,dtype='<f4').reshape(value.shape)
   elif value.dtype==mx.int32:h=np.frombuffer(data,dtype='<i4').reshape(value.shape)
   else:raise ValueError('unsupported numeric dtype')
   return h,sig,data
  def compare(left,right,label,is_logits=False):
   if left.shape!=right.shape or left.dtype!=right.dtype:raise ValueError('shape/dtype: '+label)
   ha,sa,ba=float_host(left,label+'.A');hb,sb,bb=float_host(right,label+'.B');diff=ha.astype(np.float64)-hb.astype(np.float64);maxabs=float(np.max(np.abs(diff)));l2=float(np.linalg.norm(diff.ravel())/max(1.,np.linalg.norm(ha.astype(np.float64).ravel())))
   ok=(ba==bb) if left.dtype==mx.int32 else maxabs<=LOGIT_BOUND if is_logits else l2<=CACHE_L2_BOUND
   row={'label':label,'A':sa,'B':sb,'max_abs_error':maxabs,'normalized_l2':l2,'pass':ok}
   if not ok:
    (a.out/'first-failure-A.bin').write_bytes(ba);(a.out/'first-failure-B.bin').write_bytes(bb);write(a.out/'first-failure-comparison.json',row);raise ValueError('fixed fullmodel numeric bound failed: '+label)
   return row
  def parameter_identity():
   rows=[(k,id(v),list(v.shape),str(v.dtype)) for k,v in tree_flatten(model.parameters())];return {'count':len(rows),'sha256':hashlib.sha256(json.dumps(rows).encode()).hexdigest()}
  def from_flat(flat):return [BatchKV(flat[2*i][:],flat[2*i+1][:]) if layer.is_attn else BatchState(flat[2*i],flat[2*i+1]) for i,layer in enumerate(model.layers)]
  def flatten(caches):return tuple(x for layer,c in zip(model.layers,caches) for x in ((c.K,c.V) if layer.is_attn else (c.conv,c.S)))
  def layouts(flat):
   rows=[]
   for i,layer in enumerate(model.layers):
    if not layer.is_attn:
     row=descriptor(flat[2*i+1]);row['layer']=i
     if row['shape']!=[1,48,128,128] or row['strides_elements']!=[786432,16384,1,128]:raise ValueError('SIMD state layout mismatch layer'+str(i))
     rows.append(row)
   return rows
  r['stage']='loading';save();model,cfg=load_model(w['model_dir'],dtype=mx.bfloat16);r['model_loads']=1;loaded=True;mx.synchronize();resources.sample('afterload',True,mx)
  gdns=[l.linear_attn for l in model.layers if not l.is_attn]
  if len(model.layers)!=64 or len(gdns)!=48 or not all(type(g) is GatedDeltaNet and g._simd_row_enabled is False for g in gdns):raise ValueError('defaultOFF/model identity')
  params=parameter_identity()
  for g in gdns:g.set_gates_compile(True);g.set_conv_silu(True);g.set_simd_row(True)
  if parameter_identity()!=params or params['count']!=1349:raise ValueError('parameter identity changed')
  reference_path=P/'candidate-v17-gdn-gates-build1-root/tutorial_code/miniengine/model/qwen3_5.py'
  reference_tree=ast.parse(reference_path.read_text());reference_class=next(n for n in reference_tree.body if isinstance(n,ast.ClassDef) and n.name=='Qwen3_5ForCausalLM');reference_fn=copy.deepcopy(next(n for n in reference_class.body if isinstance(n,ast.FunctionDef) and n.name=='__call__'))
  if any(isinstance(n,ast.Name) and n.id=='_v23_normal' for n in ast.walk(reference_fn)):raise ValueError('accepted25 reference unexpectedly carries V23 admission')
  reference_ns=dict(Qwen3_5ForCausalLM.__call__.__globals__);exec(compile(ast.fix_missing_locations(ast.Module(body=[reference_fn],type_ignores=[])),str(reference_path),'exec'),reference_ns);reference_call=reference_ns['__call__']
  r['reference']={'path':str(reference_path),'sha256':sha(reference_path),'method_AST_sha256':hashlib.sha256(ast.dump(reference_fn,include_attributes=False).encode()).hexdigest(),'scope':'actual accepted25 topmodel function with shared unchanged weights; default layer normal-admission=False selects unchanged accepted25 GDN math'}
  helper=FunctionalDecode(model,backend=mx,types=DecodeTypes(Qwen3_5ForCausalLM,ForwardContext,BatchKV,BatchState),runtime_version=versions['mlx'],device_kind='gpu')
  r['flags']={'gates':48,'conv':48,'simd':48,'fixed_before_helper':True};r['parameters_initial']=params;r['stage']='prefill';save()
  caches=make_cache(model);prefill=model(mx.array(w['prompt_ids'],dtype=mx.int32),ForwardContext('prefill',caches),last_only=True);mx.eval(prefill);numeric.raw(prefill,'prefill.logits');first=int(mx.argmax(prefill,axis=-1).item());eos=read_eos_ids(w['model_dir'])
  if first!=w['continuation_ids'][0] or first in eos:raise ValueError('prefill greedy seed/EOS mismatch')
  batch=DecodeBatch(model);batch.rebuild([],[('R0',caches,len(w['prompt_ids'])+32)]);initial=tuple(batch.arrays());mx.eval(*initial)
  if len(initial)!=128:raise ValueError('must compare all128cache arrays')
  state_a=tuple(x[:] for x in initial);state_b=tuple(x[:] for x in initial);seed_before=numeric.signatures(initial,'initial');ids_a=[first];ids_b=[first];warm_ref=None
  r['A_ids']=ids_a;r['B_ids']=ids_b;r['initial_cache_shapes']=[list(x.shape) for x in initial]
  for step in range(1,32):
   position=19+step-1;length=position+1;pos=mx.array([position],dtype=mx.int32);lens=mx.array([length],dtype=mx.int32);ta=mx.array([ids_a[-1]],dtype=mx.int32);tb=mx.array([ids_b[-1]],dtype=mx.int32)
   retained=(*state_a,*state_b,ta,tb,pos,lens);before=numeric.signatures(retained,'retained.before');ctx=ForwardContext('decode',from_flat(state_a),positions=pos,lens=lens,max_len=length)
   # The actual accepted25 topmodel function omits the newer normal admission.
   la=reference_call(model,ta,ctx);next_a=flatten(ctx.caches);mx.eval(la,*next_a)
   result=helper.functional(tb,pos,lens,state_b,max_len=length,greedy=True);lb,next_b=result[0],tuple(result[1:]);mx.eval(lb,*next_b)
   if len(next_a)!=128 or len(next_b)!=128:raise ValueError('full cache outputs missing')
   logit_check=compare(la,lb,'step%d.logits'%step,True);checks=[compare(x,y,'step%d.cache%d'%(step,i)) for i,(x,y) in enumerate(zip(next_a,next_b))];numeric.readonly(retained,before,'step%d.retained'%step)
   tok_a=int(mx.argmax(la,axis=-1).item());tok_b=int(mx.argmax(lb,axis=-1).item())
   if tok_a!=tok_b or tok_a!=w['continuation_ids'][step] or tok_a in eos:raise ValueError('greedy32/EOS mismatch step'+str(step))
   ids_a.append(tok_a);ids_b.append(tok_b);r['steps'].append({'step':step,'position':position,'exact_L':length,'logits':logit_check,'caches':checks,'input_readonly':True,'input_signature_sha256':hashlib.sha256(json.dumps(before,sort_keys=True).encode()).hexdigest(),'output_S_layouts':layouts(next_b),'token_A':tok_a,'token_B':tok_b})
   if step==1:warm_ref=(la,tuple(next_a))
   state_a,state_b=next_a,next_b;r['stage']='decode'+str(step);save()
   if step in (1,16,31):resources.sample('after-decode'+str(step),True,mx)
  numeric.readonly(initial,seed_before,'seed-after-cold-chain')
  # Existing L20 entry was traced with canonical state; hit it with both layouts.
  for index,kind in enumerate(['canonical','persistent','canonical']):
   flat=tuple(local.transposed_storage(x,mx) if kind=='persistent' and not model.layers[i//2].is_attn and i%2 else x[:] for i,x in enumerate(initial));pos=mx.array([19],mx.int32);lens=mx.array([20],mx.int32);token=mx.array([first],mx.int32);inputs=(*flat,pos,lens,token);before=numeric.signatures(inputs,'warm.before');snap=helper.snapshot();result=helper.functional(token,pos,lens,flat,max_len=20,greedy=True);lb,next_b=result[0],tuple(result[1:]);mx.eval(lb,*next_b)
   checks=[compare(x,y,'warm%d.cache%d'%(index,i)) for i,(x,y) in enumerate(zip(warm_ref[1],next_b))];logits=compare(warm_ref[0],lb,'warm%d.logits'%index,True);numeric.readonly(inputs,before,'warm.retained');after=helper.snapshot()
   if after['hits']!=snap['hits']+1 or after['misses']!=31 or after['entries']!=31:raise ValueError('sameL warm cache engagement differs')
   if int(mx.argmax(lb,axis=-1).item())!=w['continuation_ids'][1]:raise ValueError('warm greedy differs')
   r['warm_layout_replays'].append({'input_layout':kind,'logits':logits,'caches':checks,'input_readonly':True,'output_S_layouts':layouts(next_b),'helper_before':snap,'helper_after':after});save()
  if parameter_identity()!=params:raise ValueError('parameter identity after qualification')
  actual={}
  for name,mod in tuple(sys.modules.items()):
   if name.startswith('tutorial_code.miniengine') and getattr(mod,'__file__',None):
    path=str(Path(mod.__file__).resolve())
    if path not in pins or sha(path)!=pins[path]:raise ValueError('imported source outside candidate: '+path)
    actual[name]={'path':path,'sha256':pins[path]}
  r.update(status='PASS_V27_FULL32_GREEDY_BOUNDED_NUMERIC',A_ids=ids_a,B_ids=ids_b,helper_final=helper.snapshot(),parameters_unchanged=True,imported_sources=actual,all128cache_arrays_compared_each_step=True,canonical_and_persistent_functional_hit_covered=True,no_performance_claim=True)
  resources.sample('fullqual-complete',True,mx);return 0
 except BaseException as e:r.update(status='FAIL_STOP',error={'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()});raise
 finally:
  if resources is not None:r['raw_resources']=resources.records
  r['wall_seconds']=time.perf_counter()-start;save()
if __name__=='__main__':raise SystemExit(main())
