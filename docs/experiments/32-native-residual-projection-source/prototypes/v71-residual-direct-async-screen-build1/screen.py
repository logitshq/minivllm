"""Root-only production B-only engine timing; no HTTP, paired comparison or win claim."""
import argparse
import ctypes
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
import traceback

sys.dont_write_bytecode=True
D=Path(__file__).resolve().parent
P=D.parent
ROOT=P/'candidate-v65-residual-carry-build1-root'
LIB=P/'v62-native-direct-pipeline-build2/libv62replay.dylib'
LIB_SHA='35aa6e0d0297c12d3978caf2e70a7082ecbbf60b85e1525133d4b8b0effdc83f'
ADAPTER=P/'v64-native-direct-async-runner-build1/native_runner.py'
ADAPTER_SHA='acd59a4d817cb73fba07f6757b3a170a057e3c78ea707710624dda0a65e4bc07'
GOLD=P/'v23-root-control-build1/fullmodel-v44native-attempt1/child-output/receipt.json'
GOLD_SHA='1dcfc41e7779c575602bd6c73bae4e4800664db42e1ee87e75414335bdc45b6d'
QUALS=json.loads((D/'qualifications.json').read_text())
for binding in QUALS.values():
 if binding['receipt'] is not None:binding['receipt']=Path(binding['receipt'])
CONTROLS={'MLX_MAX_OPS_PER_BUFFER':'50','MLX_MAX_MB_PER_BUFFER':'50','MLX_BFS_MAX_WIDTH':'20','MLX_METAL_FAST_SYNCH':'0'}
sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec);sys.modules[name]=value;spec.loader.exec_module(value);return value

def source_check(*,allow_pending=False):
 pin_path=D/'source-pins.json'
 if not pin_path.exists():
  assert allow_pending,'final source pins/dual-mode V70 qualification not ready'
  pin_path=D/'source-pins.pending.json'
 pins=json.loads(pin_path.read_text())
 for path,digest in pins.items():assert sha(path)==digest,path
 assert sha(LIB)==LIB_SHA and sha(ADAPTER)==ADAPTER_SHA and sha(GOLD)==GOLD_SHA
 for mode,b in QUALS.items():
  if b['binding_status']!='PASS_BOUND':
   assert allow_pending,'V70 dual-mode qualification pending: performance forbidden'
   continue
  assert sha(b['receipt'])==b['sha256'];r=json.loads(b['receipt'].read_text());release=b['receipt'].parent.parent/'external-release.json'
  assert sha(release)==b['release_sha256'];x=json.loads(release.read_text())
  assert x['status']=='PASS_EXTERNAL_RELEASE' and x['GPU_released'] and x['tool_exit']==0 and all(v['returncode']==1 and v['stdout']==v['stderr']=='' for v in x['strict_absence'])
  assert r['status']==b['status'] and r['mode']==mode and r['strict_output_checks']==b['checks']
  assert (r['native_steps'],r['native_completed_steps'],r['total_discarded_steps'],r['resets'])==(161,162,1,6)
  assert r['build_input131_readonly'] and r['captured_originals_readonly']
  assert r['native']['feedback']['icb']['bank_executions']==[84,78]
  assert r['source_pins'][str(LIB)]==pins[str(LIB)] and r['adapter_sha256']==ADAPTER_SHA
  assert r['hidden_norm_metadata_qualified']==129 and r['same_parameter_identity_count']==1349 and r['same_parameter_identity_preserved']
  assert r['B_capture_engagement']['residual_calls']==128 and r['B_capture_engagement']['counter_removed_after_build']
  assert r['V65_capture_hooks_removed_final'] and r['carry_on'] is True
 package=ROOT/'tutorial_code/miniengine';manifest=json.loads((P/'v65-residual-carry-fullqual-build1/candidate-manifest.json').read_text())['files'];expected={k.removeprefix('tutorial_code/miniengine/'):v for k,v in manifest.items()}
 assert {str(f.relative_to(package)):sha(f) for f in package.rglob('*.py')}==expected and len(expected)==29
 gold=json.loads(GOLD.read_text());last=gold['requests'][0]['steps'][-1]['outputs'];assert len(last)==132 and all(v['strict_bytes_equal'] and v['finite'] for v in last)
 return pins,last

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path);ap.add_argument('--tokens',type=int,default=32);ap.add_argument('--inert',action='store_true');args=ap.parse_args()
 pins,gold=source_check(allow_pending=args.inert);w=json.loads((P/'workload.json').read_text());assert args.tokens==32 and len(w['prompt_ids'])==19 and len(w['continuation_ids'])==32
 if args.inert:
  assert not any(n=='mlx' or n.startswith(('mlx.','numpy')) for n in sys.modules)
  print(json.dumps({'status':'PASS_CPU_INERT','pins':len(pins),'warm':3,'measure':3,'decode_steps_per_request':31,'expected_completed':186,'expected_resets':6,'expected_banks':[96,90],'GPU_used':False,'qualifications_pending':[m for m,b in QUALS.items() if b['binding_status']!='PASS_BOUND'],'candidate_source_count':29}));return 0
 assert args.out is not None and args.out.is_absolute();args.out.mkdir()
 r={'status':'STARTED','stage':'beforeload','pid':os.getpid(),'model_loads':0,'source_pins':pins,'requests':[],
    'scope':'production B-only engine exploratory timing; no HTTP/Pareto/paired comparison/adoption claim',
    'schedule':['warm']*3+['measure']*3,'candidate':str(ROOT),'native_library':str(LIB),'adapter':str(ADAPTER),
    'qualification_receipts':{m:{'sha256':b['sha256'],'release_sha256':b['release_sha256']} for m,b in QUALS.items()},
    'timing_definition':{'t0':'before fresh engine.add_request and prefill','t1':'after engine.step first prefill token observed',
     't2':'after remaining31 engine.step calls, final delivered token and drop/empty drain',
     'decode':'post_first_token_decode31_including_seed;31/(t2-t1)',
     'included_decode':'native begin/reset/all131 seed uploads and engine scheduling; first warm additionally includes capture/readonly baselines',
     'whole':'32/(t2-t0)','excluded':'per-request resource probes, native reports,132 final hashes, readonly checks and file writes',
     'comparison_limit':'different scope from raw57 pure loop whose t1 was after seed uploads; no direct paired comparison'},
    'performance_point':'exploratory B-only absolute scopes only','first_warm_capture_descriptive_only':True}
 save=lambda:write(args.out/'receipt.json',r);save();loaded=False;mx=None;resources=None;adapter=None
 try:
  assert os.environ.get('DYLD_INSERT_LIBRARIES')==str(LIB) and all(os.environ.get(k)==v for k,v in CONTROLS.items())
  support=module('v67_resources',P/'v24-fullmodel-qual-build1/fullqual.py');qualified=module('v67_qualified64',P/'v64-native-direct-async-runner-build1/fullqual.py');runtime=module('v67_adapter64',ADAPTER)
  resources=support.Resources(args.out)
  def sample(label):
   value=resources.sample(label,loaded=loaded,mx=mx);cur=resources.history[-1]
   assert cur['processinfo_thermal_metadata']['status']=='known' and cur['processinfo_thermal_metadata']['code']==0
   assert cur['pressure_level_value']==1 and cur['power_metadata']['status']=='known' and cur['power_metadata']['source']=='AC Power'
   if len(resources.history)>1:
    prev=resources.history[-2];assert cur['swap_used_mb']<=prev['swap_used_mb'],'strict swap growth STOP'
    assert cur['vm_counters']['Swapouts']==prev['vm_counters']['Swapouts'],'strict Swapouts change STOP'
   return value
  sample('beforeload')
  sys.path.insert(0,str(ROOT));import mlx.core as mx
  from tutorial_code.miniengine.engine.llm_engine import LLMEngine
  from tutorial_code.miniengine.sampling_params import SamplingParams
  e=LLMEngine(w['model_dir'],max_seqs=1,max_len=2048,max_batched_tokens=512,dtype=mx.bfloat16,cache_limit_gb=2)
  loaded=True;r['model_loads']=1;mx.synchronize();r['afterload']=sample('afterload-before-any-token')
  from mlx.utils import tree_flatten
  from tutorial_code.miniengine.model.layers import RMSNorm
  live_parameters=tree_flatten(e.model.parameters());initial_parameter_identity=[(name,id(value)) for name,value in live_parameters]
  assert len(live_parameters)==1349 and len(set(name for name,_ in live_parameters))==1349 and e.model._residual_carry_enabled is False
  parameter_by_name=dict(live_parameters)
  hidden_norms=[(f'layers.{i}.{key}',getattr(layer,key)) for i,layer in enumerate(e.model.layers) for key in ('input_layernorm','post_attention_layernorm')]+[('norm',e.model.norm)]
  hidden_metadata=[]
  for label,norm in hidden_norms:
   weight=getattr(norm,'weight',None);eps=getattr(norm,'eps',None)
   row={'name':label,'shape':list(getattr(weight,'shape',())),'dtype':str(getattr(weight,'dtype',None)),
        'eps':eps if type(eps) is float and math.isfinite(eps) else None,'eps_repr':repr(eps),'zero_centered':getattr(norm,'zero_centered',None),
        'parameter_object_identity_match':parameter_by_name.get(label+'.weight') is weight}
   row['fusion_eligible']=(type(norm) is RMSNorm and weight is not None and weight.shape==(5120,) and weight.dtype in (mx.bfloat16,mx.float32)
       and norm.zero_centered is True and type(eps) is float and math.isfinite(eps) and 0.0<eps<1.0 and row['parameter_object_identity_match'])
   hidden_metadata.append(row)
  r['hidden_norm_metadata_before_any_token']={'count':len(hidden_metadata),'rows':hidden_metadata,'token_executed':False,'array_parameters':1349}
  write(args.out/'hidden-norm-metadata-before-any-token.json',r['hidden_norm_metadata_before_any_token']);save()
  assert len(hidden_metadata)==129 and all(row['fusion_eligible'] for row in hidden_metadata)
  e.model.set_residual_carry(True)
  gdns=[l.linear_attn for l in e.model.layers if not l.is_attn];assert len(gdns)==48
  for g in gdns:g.set_gates_compile(True);g.set_conv_norm_reuse_cache(True)
  assert getattr(e.runner,'_functional_decode_helper',None) is None
  assert initial_parameter_identity==[(name,id(value)) for name,value in tree_flatten(e.model.parameters())]
  r['carry_on']=True;r['hidden_norm_metadata_qualified']=129
  r['stage']='ready';write(args.out/'model-ready.json',{'pid':os.getpid(),'model_loads':1,'token_executed':False});save();print(json.dumps({'event':'MODEL_READY','pid':os.getpid()}),flush=True)
  sampling=SamplingParams(max_tokens=32,temperature=0,ignore_eos=False)
  ordinary=e.generate([w['prompt_ids']],sampling);assert ordinary==[w['continuation_ids']] and not e.has_work() and e.runner.batch.size==0 and e.runner._ahead is None
  sample('after-ordinary-warm32');adapter=runtime.install(e,LIB,LIB_SHA,diagnostic=False)
  original_build=adapter.build;capture={};baseline=[]
  from tutorial_code.miniengine.kernels import residual_norm_metal as residual_module
  from tutorial_code.miniengine.model.gdn import GatedDeltaNet
  original_residual=residual_module.residual_norm;original_conv=GatedDeltaNet._decode_conv_norm_reuse_cache
  norm_labels={id(norm.weight):label+'.weight' for label,norm in hidden_norms};fusion_calls=[];conv_calls=[]
  assert len(norm_labels)==129
  def observe_first_build(values):
   # First warm only: snapshot BEFORE build, then original outputs BEFORE replay.
   # This is not a backend API recorder and is removed before any measured run.
   before=[bytes(adapter.shared(v,str(i)).bytes_view) for i,v in enumerate(values)]
   assert not fusion_calls and not conv_calls and e.model._residual_carry_enabled is True
   assert residual_module.residual_norm is original_residual and GatedDeltaNet._decode_conv_norm_reuse_cache is original_conv
   def counted_residual(left,delta,weight,eps):
    fusion_calls.append(norm_labels[id(weight)]);return original_residual(left,delta,weight,eps)
   def counted_conv(self,*a,**kw):
    conv_calls.append(id(self));return original_conv(self,*a,**kw)
   residual_module.residual_norm=counted_residual;GatedDeltaNet._decode_conv_norm_reuse_cache=counted_conv
   try:original_build(values)
   finally:
    residual_module.residual_norm=original_residual;GatedDeltaNet._decode_conv_norm_reuse_cache=original_conv
    r['B_first_warm_capture_engagement']={'residual_calls':len(fusion_calls),'norm_weight_order':list(fusion_calls),
        'conv_calls':len(conv_calls),'distinct_GDN_objects':len(set(conv_calls)),
        'counter_removed_after_build':residual_module.residual_norm is original_residual and GatedDeltaNet._decode_conv_norm_reuse_cache is original_conv}
   expected_fusions=[name for i in range(64) for name in (f'layers.{i}.post_attention_layernorm.weight',f'layers.{i+1}.input_layernorm.weight' if i<63 else 'norm.weight')]
   assert fusion_calls==expected_fusions and len(fusion_calls)==128 and len(conv_calls)==48 and set(conv_calls)==set(map(id,gdns))
   assert r['B_first_warm_capture_engagement']['counter_removed_after_build'] is True
   assert len(before)==len(adapter.inputs)==131
   assert all(bytes(h.bytes_view)==b for h,b in zip(adapter.inputs,before))
   baseline.extend(before+[bytes(h.bytes_view) for h in adapter.outputs]);assert len(baseline)==263
   capture.update(build_input131_readonly=True,baseline_before_first_replay=True)
   adapter.build=original_build
  adapter.build=observe_first_build
  for index,phase in enumerate(r['schedule'],1):
   assert not e.has_work() and not e.requests and not e.runner.staging and e.runner.batch.size==0 and adapter.active is None and not adapter.pending
   before=sample(f'request-{index}-before')
   ids=[]
   t0=time.perf_counter_ns()
   rid=e.add_request(w['prompt_ids'],sampling)
   result=e.step()
   assert len(result)==1 and result[0][0]==rid and not result[0][2]
   ids.append(result[0][1])
   t1=time.perf_counter_ns()
   for step in range(31):
    result=e.step()
    assert len(result)==1 and result[0][0]==rid and result[0][2]==(step==30)
    ids.append(result[0][1])
   t2=time.perf_counter_ns()
   after=sample(f'request-{index}-after')
   assert ids==w['continuation_ids'] and len(baseline)==263 and adapter.build==original_build
   assert not e.has_work() and not e.requests and not e.runner.staging and e.runner.batch.size==0 and e.runner._ahead is None and adapter.active is None and not adapter.pending
   finals=qualified.snapshot_rows(adapter,gold,api='v62_copy_quiescent_output',context=('screen-final',index))
   assert all(bytes(h.bytes_view)==b for h,b in zip(adapter.inputs+adapter.outputs,baseline)),'original MLX owner changed'
   native=adapter.report();qualified.assert_pipeline(native,diagnostic=False,delivered=31*index,completed=31*index,requests=index,discarded=0,banks=[16*index,15*index])
   assert adapter.native_steps==adapter.total_submitted==31*index and adapter.total_discarded==0 and adapter.request_count==index
   assert len(adapter.drops)==index and all(x['submitted']==31 and x['discarded']==0 for x in adapter.drops)
   assert native['direct_pipeline']['quiescent_output_copies']==132*index and native['feedback']['CPU_output_compare_bytes_total']==0
   row={'index':index,'phase':phase,'request_id':rid,'ids':ids,'t0_ns':t0,'t1_ns':t1,'t2_ns':t2,
        'post_first_token_decode31_including_seed_ns':t2-t1,'decode_ns_per_step':(t2-t1)/31,'decode31_tokens_per_s':31e9/(t2-t1),
        'whole32_ns':t2-t0,'whole32_tokens_per_s':32e9/(t2-t0),'first_prefill_token_ns':t1-t0,
        'resources':{'before':before,'after':after},'final132_hashes':finals,'capture_owners263_readonly':True,
        'final132_api':'v62_copy_quiescent_output after completed empty drop','native':native}
   r['requests'].append(row);r['stage']={'request':index,'phase':phase,'complete':True};save()
   print(json.dumps({'event':'B_ONLY_ENGINE_EXPLORATORY','request':index,'phase':phase,'post_first_token_decode31_including_seed_ns':t2-t1,'whole32_ns':t2-t0}),flush=True)
  measured=[x for x in r['requests'] if x['phase']=='measure'];assert len(measured)==3
  r['measured_summary']={'n':3,'median_post_first_token_decode_ns_per_step_including_seed':statistics.median(x['decode_ns_per_step'] for x in measured),
    'median_decode31_tokens_per_s':statistics.median(x['decode31_tokens_per_s'] for x in measured),'median_whole32_tokens_per_s':statistics.median(x['whole32_tokens_per_s'] for x in measured),
    'scope':'B-only engine absolute; includes native seed setup; no HTTP/paired performance win'}
  assert capture=={'build_input131_readonly':True,'baseline_before_first_replay':True}
  assert residual_module.residual_norm is original_residual and GatedDeltaNet._decode_conv_norm_reuse_cache is original_conv and len(fusion_calls)==128
  assert initial_parameter_identity==[(name,id(value)) for name,value in tree_flatten(e.model.parameters())]
  r['same_parameter_identity_count']=1349;r['same_parameter_identity_preserved']=True
  r.update(status='PASS_V71_RESIDUAL_DIRECT_ASYNC_B_ONLY_ENGINE_EXPLORATORY_SCREEN',stage='complete',ordinary_warm_tokens=32,
    native_delivered=186,native_completed=186,discarded=0,resets=6,ICB_banks=[96,90],final_output_hash_checks=792,
    capture=capture,backend_API_recording_wrappers=False,readonly_capture_snapshot_first_warm_only=True,source_pins_final_verified=True)
  source_check();save();return 0
 except BaseException as error:
  r.update(status='FAIL_STOP',error=repr(error),traceback=traceback.format_exc());save();raise
 finally:
  if resources is not None:r['finally_resource']=sample('finally-before-exit');save()

if __name__=='__main__':raise SystemExit(main())
