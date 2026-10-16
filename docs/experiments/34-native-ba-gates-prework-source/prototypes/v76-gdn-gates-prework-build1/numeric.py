"""V76 root-only narrow numeric probe; 18 selected ranges, no model or timing."""
import argparse
import ast
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import struct
import sys
import time
import traceback
sys.dont_write_bytecode=True
D=Path(__file__).resolve().parent;P=D.parent
LAYERS=(0,32,62)
BANKROOT=P/'v18-recurrent-numeric-build1/gpu-recurrent-numeric-AC-run1'
BANKS=[f'real-L{l}-D{d}' for l in LAYERS for d in (1,16,31)]+['synthetic-zero','synthetic-random-PCG64-18001','synthetic-amplitude','synthetic-cancellation','synthetic-signedzero-subnormal','synthetic-repeated-head-boundary']
CONTROLS={'MLX_MAX_OPS_PER_BUFFER':'50','MLX_MAX_MB_PER_BUFFER':'50','MLX_BFS_MAX_WIDTH':'20','MLX_METAL_FAST_SYNCH':'0'}
Q={'group_size':16,'bits':4,'mode':'nvfp4'}
REF_HELPER=P/'v44-gdn-activated-normreuse-build1/gdn_activated_norm_reuse_cached_metal.py'
REF_GATES=P/'candidate-v44-conv-normreuse-build1-root/tutorial_code/miniengine/model/gdn.py'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def identity():
 pins=json.loads((D/'source-pins.json').read_text())
 for p,h in pins.items():assert sha(p)==h,('source changed',p)
 fixture=json.loads((D/'fixture.json').read_text());checkpoint=Path(fixture['checkpoint'])
 assert sha(checkpoint/'model.safetensors.index.json')==fixture['index_sha256'] and sha(checkpoint/'config.json')==fixture['config_sha256']
 index=json.loads((checkpoint/'model.safetensors.index.json').read_text())['weight_map'];headers={}
 for layer in LAYERS:
  records=fixture['tensors'][str(layer)];assert len(records)==6
  for rec in records:
   path=Path(rec['path']);assert checkpoint/index[rec['key']]==path
   if path not in headers:
    with path.open('rb') as f:
     size=struct.unpack('<Q',f.read(8))[0];assert 0<size<64<<20;raw=f.read(size)
    headers[path]=(json.loads(raw),8+size,hashlib.sha256(raw).hexdigest())
   table,start,hsh=headers[path];item=table[rec['key']]
   assert hsh==rec['header_sha256'] and item['shape']==rec['shape'] and item['dtype']==rec['dtype']
   lo,hi=item['data_offsets'];assert (start+lo,hi-lo)==(rec['offset'],rec['nbytes']) and start+hi<=path.stat().st_size
 for name in BANKS:
  directory=BANKROOT/name;meta=json.loads((directory/'bank.json').read_text())
  for key,shape,dtype,width in [('u',[1,10240],'bfloat16',2),('g',[1,48],'float32',4),('beta',[1,48],'float32',4),('S',[1,48,128,128],'float32',4)]:
   a=meta['arrays'][key];path=directory/a['file'];assert Path(a['file']).name==a['file']
   assert a['shape']==shape and a['dtype']==dtype and path.stat().st_size==a['bytes']==math.prod(shape)*width and sha(path)==a['sha256']
 assert fixture['selected_tensor_count']==18 and fixture['selected_checkpoint_payload_bytes']==830016
 assert (fixture['cases'],fixture['strict_checks_required'])==(45,720)
 return fixture,pins


def gpu(args,fixture,pins):
 out=args.out;checks=[];resources=[];cases=[];started=time.perf_counter()
 r={'status':'STARTED','pid':os.getpid(),'source_pins':pins,'fixture':fixture,'checks':checks,'cases':cases,
    'scope':'independent Q35 extension local numeric only; 15 historical u/S banks + genericX, not matched full-model inputs',
    'donor_scope':'actual oMLX Q35 prework keeps gates/recurrence separate; this extension is not their Qwen4 BF16 gates',
    'model_loaded':False,'engine_integrated':False,'timing_run':False,'performance_claim':False,'math_mode':'V44 safe unchanged','QMV_JIT_contract_risk':'combined safe context is not assumed byte-equivalent to native QMV/compiled-gates; explicit BA/gates strict checks required',
    'production_prework_buffers':14,'diagnostic_prework_buffers':16,'state_source_name_policy':'exact qualified44/state40, S input6',
    'strict_output_checks':0,'power_domain':args.power_domain,'readonly':[]}
 save=lambda:write(out/'receipt.json',r);save()
 try:
  assert {n:importlib.metadata.version(n) for n in ('mlx','mlx-metal')}=={'mlx':'0.32.3','mlx-metal':'0.32.3'}
  assert all(os.environ.get(k)==v for k,v in CONTROLS.items())
  assert not os.environ.get('DYLD_INSERT_LIBRARIES'),'standalone numeric only; no native injection'
  import mlx.core as mx
  import mlx.nn as nn
  import numpy as np
  info=mx.device_info();assert mx.default_device()==mx.gpu and info['device_name']=='Apple M4 Max' and info['architecture']=='applegpu_g16s'
  mx.set_memory_limit(512<<20);mx.set_cache_limit(64<<20)
  r['runtime']={'mlx':'0.32.3','mlx-metal':'0.32.3','device':info,'memory_limit_bytes':512<<20,'cache_limit_bytes':64<<20,'controls':CONTROLS}
  sys.path.insert(0,str(P));from resource_snapshot_power_v1 import snapshot,deltas,guard
  def sample(label):
   mx.synchronize();raw=snapshot(label,os.getpid());raw['mlx_memory']={'active':mx.get_active_memory(),'cache':mx.get_cache_memory(),'peak':mx.get_peak_memory()}
   write(out/f'resource-{len(resources):02d}-raw.json',raw);resources.append(raw);deltas(resources);write(out/'resources.json',resources);guard(resources)
   assert raw['power_metadata']['status']=='known' and raw['power_metadata']['source']==args.power_domain
   assert raw['processinfo_thermal_metadata']['code']==0 and raw['pressure_level_value']==1
   assert all(x['owned'] for x in raw['known_inference_processes'])
   assert raw['swap_used_mb']<=resources[0]['swap_used_mb'] and raw['vm_counters']['Swapouts']==resources[0]['vm_counters']['Swapouts']
  def raw(value):
   mx.eval(value);view=value.view(mx.uint16) if value.dtype==mx.bfloat16 else value;host=np.asarray(view);data=host.tobytes()
   if value.dtype==mx.bfloat16:finite=not bool(np.any((host&0x7f80)==0x7f80))
   elif value.dtype==mx.float32:finite=bool(np.isfinite(host).all())
   else:finite=True
   meta={'shape':list(value.shape),'dtype':str(value.dtype),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'finite':finite,'address':int(host.ctypes.data)}
   return meta,data
  def signature(values):return {name:raw(v)[0] for name,v in values.items()}
  def compare(a,b,label):
   am,ab=raw(a);bm,bb=raw(b)
   equal=am['shape']==bm['shape'] and am['dtype']==bm['dtype'] and len(ab)==len(bb) and am['finite'] and bm['finite'] and ab==bb
   row={'label':label,'A':am,'B':bm,'strict_bytes_equal':bool(equal)};checks.append(row);save()
   if not equal:
    (out/'first-failure-A.bin').write_bytes(ab);(out/'first-failure-B.bin').write_bytes(bb)
    r['first_failure']={'label':label,'first_differing_byte':next((i for i,(x,y) in enumerate(zip(ab,bb)) if x!=y),None),'A':am,'B':bm};save();raise AssertionError(('STRICT_STOP',label))
  def load_range(rec):
   dtype=np.dtype({'U32':'<u4','U8':'u1','BF16':'<u2','F16':'<f2','F32':'<f4'}[rec['dtype']])
   mapping=np.memmap(rec['path'],mode='r',dtype=dtype,offset=rec['offset'],shape=tuple(rec['shape']))
   try:host=np.array(mapping,copy=True)
   finally:mapping._mmap.close()
   value=mx.array(host)
   if rec['dtype']=='BF16':value=value.view(mx.bfloat16)
   if rec['loader_cast']=='astype(mx.float32)':value=value.astype(mx.float32)
   mx.eval(value);return value
  layout_kernel=mx.fast.metal_kernel(name='v76_actual_operand_shape_strides',input_names=['value'],output_names=['descriptor'],ensure_row_contiguous=False,
   source='for(uint i=0;i<5;i++) descriptor[i]=0; descriptor[0]=value_ndim; for(uint i=0;i<value_ndim;i++){descriptor[1+i]=value_shape[i];descriptor[3+i]=value_strides[i];}')
  r['layouts']=[]
  def layout(value,label):
   assert value.ndim in (1,2)
   d,=layout_kernel(inputs=[value],grid=(1,1,1),threadgroup=(1,1,1),output_shapes=[(5,)],output_dtypes=[mx.uint64]);mx.eval(d);items=d.tolist();ndim=int(items[0])
   assert ndim==value.ndim and items[1:1+ndim]==list(value.shape)
   strides=tuple(int(v) for v in items[3:3+ndim]);r['layouts'].append({'label':label,'shape':list(value.shape),'actual_strides_elements':list(strides),'ensure_row_contiguous':False});return strides
  sample('fresh-no-model')
  reference=module('v76_qualified44',REF_HELPER);candidate=module('v76_candidate',D/'gdn_gates_prework_metal.py')
  gt=ast.parse(REF_GATES.read_text());gatefun=next(n for n in gt.body if isinstance(n,ast.FunctionDef) and n.name=='_gates_explicit')
  namespace={'mx':mx,'nn':nn};exec(compile(ast.Module(body=[gatefun],type_ignores=[]),str(REF_GATES),'exec'),namespace)
  compiled_gates=mx.compile(namespace['_gates_explicit'],shapeless=False)
  rng=np.random.default_rng(20261016);host_x=[np.zeros((1,5120),np.float32),np.tile(np.array([1.,-1.],np.float32),2560)[None]]
  host_x.extend((rng.standard_normal((1,5120))*amp).astype(np.float32) for amp in (.0625,1.,16.,.25,.25,.25))
  xs=[mx.array(h,dtype=mx.bfloat16) for h in host_x];assert len(xs)==8 and len({raw(x)[0]['sha256'] for x in xs})==8
  operands={layer:tuple(load_range(rec) for rec in fixture['tensors'][str(layer)]) for layer in LAYERS}
  banks=[];originals={f'genericX.{i}':x for i,x in enumerate(xs)}
  for layer,values in operands.items():originals.update({f'L{layer}.operand{i}':v for i,v in enumerate(values)})
  for name in BANKS:
   directory=BANKROOT/name;meta=json.loads((directory/'bank.json').read_text())
   hu=np.fromfile(directory/meta['arrays']['u']['file'],dtype='<u2').reshape(1,10240);hs=np.fromfile(directory/meta['arrays']['S']['file'],dtype='<f4').reshape(1,48,128,128)
   u=mx.array(hu).view(mx.bfloat16);S=mx.array(hs);act=nn.silu(u);mx.eval(u,S,act)
   banks.append((name,u,act,S));originals.update({name+'.u':u,name+'.activated':act,name+'.S':S})
  assert len(originals)==71;before=signature(originals);assert all(v['finite'] for v in before.values());r['original_inputs_before']=before
  x_strides=[layout(x,f'genericX.{i}') for i,x in enumerate(xs)]
  op_strides={layer:[layout(v,f'L{layer}.operand{i}') for i,v in enumerate(values)] for layer,values in operands.items()}
  act_strides={name:layout(act,name+'.activated') for name,u,act,S in banks}
  sample('18target-ranges-15savedbanks-and8genericX-ready');save()
  for layer in LAYERS:
   wb,sb,wa,sa,Alog,dtbias=operands[layer]
   for index,(name,u,act,S) in enumerate(banks):
    x=xs[index%8];label=f'L{layer}/{name}/genericX{index%8}'
    b=mx.quantized_matmul(x,wb,sb,transpose=True,**Q);a=mx.quantized_matmul(x,wa,sa,transpose=True,**Q)
    beta,g=compiled_gates(b,a,Alog,dtbias)
    oldprep=reference._prepare(act,mx);expected=(*oldprep,g,beta)
    strides=(act_strides[name],x_strides[index%8],*op_strides[layer])
    production=candidate.prepare(act,x,wb,sb,wa,sa,Alog,dtbias,observed_strides=strides)
    diagnostic=candidate.prepare(act,x,wb,sb,wa,sa,Alog,dtbias,observed_strides=strides,probe=True)
    A=reference._state(*oldprep,g,beta,S,mx);B=candidate.state(production,S)
    mx.eval(*expected,*production,*diagnostic,*A,*B)
    row={'layer':layer,'bank':name,'historical_saved_bank_not_new_capture':True,'generic_X_index':index%8,'not_a_matched_model_activation_tuple':True,'bank_kind':'historical-real' if name.startswith('real-') else 'saved-synthetic'};cases.append(row);save()
    compare(b,diagnostic[6],label+'/b/QMM-vs-probe');compare(a,diagnostic[7],label+'/a/QMM-vs-probe')
    for key,e,p,d in zip(('q','k','v','scales','g','beta'),expected,production,diagnostic[:6]):
     compare(e,p,label+'/'+key+'/original-vs-production');compare(p,d,label+'/'+key+'/production-vs-diagnostic')
    for key,aout,bout in zip(('o','S_new'),A,B):compare(aout,bout,label+'/'+key+'/44-vs-candidate-state')
    # All original arrays are retained; full signatures are checked after each layer.
    del b,a,beta,g,oldprep,expected,production,diagnostic,A,B
   after=signature(originals);readonly={'after_layer':layer,'all71_originals_unchanged':before==after};r['readonly'].append(readonly);save();assert readonly['all71_originals_unchanged']
   sample(f'after-layer-{layer}-15banks-strict')
  assert len(cases)==45 and len(checks)==720 and all(c['strict_bytes_equal'] for c in checks)
  identity();sample('strict-numeric-complete')
  r.update(status='PASS_V76_BA_FP32_GATES_PREWORK_LOCAL_STRICT_45CASES_720CHECKS',fullmodel_qualified=False,readonly_all71=True,actual_compiled_gate_reference=True,source_verified_final=True)
  save();return 0
 except BaseException as error:
  r.update(status='FAIL_STOP',error=repr(error),traceback=traceback.format_exc());save();raise
 finally:
  r['strict_output_checks']=sum(bool(c['strict_bytes_equal']) for c in checks)
  r['wall_seconds']=time.perf_counter()-started;save()


def main():
 ap=argparse.ArgumentParser(description=__doc__);g=ap.add_mutually_exclusive_group(required=True);g.add_argument('--inert',action='store_true');g.add_argument('--run-gpu',action='store_true');ap.add_argument('--out',type=Path);ap.add_argument('--power-domain',choices=['AC Power','Battery Power'],default='AC Power');args=ap.parse_args()
 fixture,pins=identity()
 if args.inert:
  assert not any(n=='numpy' or n=='mlx' or n.startswith('mlx.') for n in sys.modules)
  print(json.dumps({'status':'PASS_CPU_INERT','MLX_imported':False,'GPU_used':False,'checkpoint_payload_loaded':False,'selected_tensor_ranges':18,'selected_payload_bytes':830016,'historical_banks':15,'genericX':8,'cases':45,'strict_checks_required':720,'production_prework_buffers':14,'probe_prework_buffers':16,'pins':len(pins)}));return 0
 assert args.out is not None and args.out.is_absolute();args.out.mkdir(exist_ok=False);return gpu(args,fixture,pins)
if __name__=='__main__':raise SystemExit(main())
