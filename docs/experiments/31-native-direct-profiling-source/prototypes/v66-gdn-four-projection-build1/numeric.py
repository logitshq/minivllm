"""Root-only strict V66 local numeric qualification. Inert reads source/headers only."""
import argparse,hashlib,importlib.metadata,importlib.util,json,os,sys,traceback
from pathlib import Path
sys.dont_write_bytecode=True
D=Path(__file__).resolve().parent;P=D.parent;LAYERS=(0,32,62)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def module(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
old=module('v66_subset_loader',P/'v22-nvfp4-qmv-r2-build1/micro.py')
h=module('v66_four_projection',D/'four_projection.py')

def identity():
 pins=json.loads((D/'source-pins.json').read_text())
 for p,digest in pins.items():assert sha(p)==digest,p
 checkpoint=old.CHECKPOINT;index_path=checkpoint/'model.safetensors.index.json';index=json.loads(index_path.read_text())['weight_map']
 config=json.loads((checkpoint/'config.json').read_text());assert config['quantization']==old.Q and config['text_config']['hidden_size']==h.K
 parsed={};tensors={}
 for layer in LAYERS:
  records=[]
  for name,n in zip(h.NAMES,h.ROWS):
   for suffix,dtype,shape in [('weight','U32',(n,h.K//8)),('scales','U8',(n,h.K//16))]:
    tail=f'.layers.{layer}.linear_attn.in_proj_{name}.{suffix}';keys=[k for k in index if k.endswith(tail)];assert len(keys)==1,tail
    key=keys[0];path=checkpoint/index[key]
    if path not in parsed:parsed[path]=old.header(path)
    records.append(old.tensor_meta(path,key,dtype,shape,parsed[path]))
  tensors[str(layer)]=records
 bank=old.REAL_BANK;receipt_path=bank/'receipt.json';receipt=json.loads(receipt_path.read_text())
 assert sha(receipt_path)=='94a9f5ab7ae3780f306e66d71e4f3b0f2fb559eef4d5a89ae179ef63ef3355c8' and receipt['A_status']=='ACTUAL64_NUMERIC_PASS'
 real=[]
 for layer in LAYERS:
  row=next(x for x in receipt['rows'] if x['layer']==layer);path=bank/f'L{layer:02d}-x.raw.le.bin'
  assert row['x']['shape']==[1,h.K] and row['x']['dtype']=='mlx.core.bfloat16' and path.stat().st_size==2*h.K and sha(path)==row['x']['raw_sha256']
  real.append({'path':str(path),'sha256':sha(path),'layer':layer,'shape':[1,h.K],'dtype':'BF16','origin':'historical first normal decode MLP post_attention_layernorm input; generic BF16 test vector, NOT GDN input or fresh capture'})
 assert sum(x['nbytes'] for rs in tensors.values() for x in rs)==142387200
 return {'source_pins':pins,'source_pins_sha256':sha(D/'source-pins.json'),'checkpoint':str(checkpoint),'index_sha256':sha(index_path),'config_sha256':sha(checkpoint/'config.json'),'tensors':tensors,'historical_vectors':real,'historical_receipt_sha256':sha(receipt_path),'selected_weight_scale_payload_bytes':142387200,'layers':list(LAYERS),'synthetic_inputs':8,'historical_inputs':3,'cases':33,'strict_output_comparisons_required':396,'MAIN_MTL_buffer_argument_count':13,'native_graph_adoption_requires_actual_argument_reflection':True}

def gpu(out,ident):
 assert importlib.metadata.version('mlx')==importlib.metadata.version('mlx-metal')=='0.32.3'
 assert all(os.environ.get(k)==v for k,v in old.CONTROLS.items())
 assert not any(k.startswith(('DYLD_','METAL_','OMLX_','LITHOS_','MONOLITH_')) or k in ('PYTHONPATH','PYTHONHOME') for k in os.environ)
 import mlx.core as mx
 import numpy as np
 info=mx.device_info();h.runtime(mx,'0.32.3',info['device_name'],info['architecture'])
 mx.set_cache_limit(2<<30)
 sys.path.insert(0,str(P));from resource_snapshot_power_v1 import snapshot,deltas,guard
 resources=[];checks=[];readonly=[]
 receipt={'status':'STARTED','pid':os.getpid(),'identity':ident,'model_loaded':False,'full_model_qualified':False,'engine_integrated':False,'timing_run':False,'performance_claim':False,'old_V16_status_preserved':'STOP-MICRO-GATE','checks':checks,'readonly':readonly,'layouts':[],'prepare':[],'scope':'3 selected layers; local strict numeric only; historical vectors are MLP-origin, not GDN; no packed weights'}
 save=lambda:write(out/'receipt.json',receipt)
 def probe(label):
  mx.synchronize();raw=snapshot(label,os.getpid());write(out/f'resource-{len(resources):02d}-raw.json',raw);resources.append(raw);deltas(resources);write(out/'resources.json',resources);guard(resources)
  s=resources[-1];assert s['power_metadata']['source']=='AC Power' and s['processinfo_thermal_metadata']['code']==0 and s['pressure_level_value']==1 and all(x['owned'] for x in s['known_inference_processes'])
  assert s['swap_used_mb']<=resources[0]['swap_used_mb'] and s['vm_counters']['Swapouts']==resources[0]['vm_counters']['Swapouts']
 def fingerprint(value):
  mx.eval(value);view=value.view(mx.uint16) if value.dtype==mx.bfloat16 else value;a=np.asarray(view);b=a.tobytes()
  return {'shape':list(value.shape),'dtype':str(value.dtype),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'shared_address':int(a.ctypes.data)},b
 layout_probe=mx.fast.metal_kernel(name='v66_observed_operand_strides',input_names=['value'],output_names=['descriptor'],source=(D/'layout_probe.metal').read_text(),ensure_row_contiguous=False)
 def layout(value,label):
  desc,=layout_probe(inputs=[value],grid=(1,1,1),threadgroup=(1,1,1),output_shapes=[(4,)],output_dtypes=[mx.uint64]);mx.eval(desc);v=desc.tolist()
  assert v[:2]==list(value.shape);row={'label':label,'shape':v[:2],'observed_strides_elements':v[2:],'dtype':str(value.dtype),'ensure_row_contiguous':False,'row_contiguous_derived':v[2:]==[value.shape[1],1]};receipt['layouts'].append(row);save();assert row['row_contiguous_derived'];return tuple(v[2:])
 def compare(a,b,n,label):
  mx.eval(a,b);assert a.shape==b.shape==(1,n) and a.dtype==b.dtype==mx.bfloat16
  am,ab=fingerprint(a);bm,bb=fingerprint(b);finite=bool(mx.all(mx.isfinite(a)).item() and mx.all(mx.isfinite(b)).item())
  row={'label':label,'shape':[1,n],'dtype':'mlx.core.bfloat16','bytes':len(ab),'finite':finite,'A_sha256':am['sha256'],'B_sha256':bm['sha256'],'differing_bytes':sum(x!=y for x,y in zip(ab,bb)),'strict_bytes_equal':ab==bb}
  checks.append(row);save();assert finite and ab==bb,('STRICT_STOP',row)
 try:
  save();probe('fresh-no-model')
  rng=np.random.default_rng(20261016);host=[np.zeros((1,h.K),np.float32),np.tile(np.array([1.,-1.],np.float32),h.K//2)[None]]
  host.extend((rng.standard_normal((1,h.K))*a).astype(np.float32) for a in (.0625,1,16,.25,.25,.25))
  vectors=[(f'synthetic-{j}',mx.array(v,dtype=mx.bfloat16)) for j,v in enumerate(host)]
  assert len({fingerprint(v)[0]['sha256'] for _,v in vectors})==8
  for rec in ident['historical_vectors']:
   words=np.fromfile(rec['path'],dtype='<u2',count=h.K);assert words.shape==(h.K,)
   value=mx.array(words,dtype=mx.uint16).view(mx.bfloat16).reshape(1,h.K);assert fingerprint(value)[0]['sha256']==rec['sha256'];vectors.append((f"historical-MLP-L{rec['layer']}",value))
  xs={name:layout(v,name) for name,v in vectors};assert len(vectors)==11
  inputs=[{'label':name,**fingerprint(v)[0],'finite':bool(mx.all(mx.isfinite(v)).item()),'origin':'synthetic' if name.startswith('synthetic') else 'historical MLP input, NOT GDN'} for name,v in vectors];write(out/'inputs.json',inputs);assert all(rec['finite'] for rec in inputs)
  operands={layer:tuple(old.load_range(np,mx,rec) for rec in ident['tensors'][str(layer)]) for layer in LAYERS}
  originals=[(f'L{layer}.{j}',v) for layer in LAYERS for j,v in enumerate(operands[layer])]+vectors
  before={name:fingerprint(v)[0] for name,v in originals};receipt['original_input_metadata']=before;save();probe('all24-target-ranges-and11-vectors-ready')
  control=h.R4Controls(mx)
  for layer in LAYERS:
   ops=operands[layer];strides=[layout(v,f'L{layer}.{j}') for j,v in enumerate(ops)];active=mx.get_active_memory()
   fused=h.prepare(mx,ops,observed_strides=strides,runtime_version='0.32.3',device_name=info['device_name'],architecture=info['architecture'])
   assert all(a is b for a,b in zip(fused.operands,ops)) and mx.get_active_memory()==active
   receipt['prepare'].append({'layer':layer,'same_original_handles':True,'expected_extra_weight_bytes':0,'mlx_active_before':active,'mlx_active_after':mx.get_active_memory()});save()
   for name,x in vectors:
    reference=tuple(mx.quantized_matmul(x,ops[2*j],ops[2*j+1],transpose=True,**old.Q) for j in range(4))
    same_r4=control.project(ops,x,x_strides=xs[name]);candidate=fused.project(x,x_strides=xs[name]);mx.eval(*reference,*same_r4,*candidate)
    ranges=[]
    for j,(projection,n) in enumerate(zip(h.NAMES,h.ROWS)):
     compare(reference[j],same_r4[j],n,f'L{layer}/{name}/{projection}/QMM-vs-same-R4')
     compare(reference[j],candidate[j],n,f'L{layer}/{name}/{projection}/QMM-vs-four-pointer')
     compare(same_r4[j],candidate[j],n,f'L{layer}/{name}/{projection}/same-R4-vs-four-pointer')
     meta,_=fingerprint(candidate[j]);ranges.append((meta['shared_address'],meta['shared_address']+meta['bytes']))
    assert len({id(v) for v in candidate})==4 and all(a[1]<=b[0] or b[1]<=a[0] for j,a in enumerate(ranges) for b in ranges[j+1:]),'candidate outputs alias'
    original_ranges=[(m['shared_address'],m['shared_address']+m['bytes']) for m in before.values()]
    assert all(a[1]<=b[0] or b[1]<=a[0] for a in ranges for b in original_ranges),'candidate output overlaps immutable original'
   after={name:fingerprint(v)[0] for name,v in originals};row={'after_layer':layer,'all35_originals_unchanged':before==after};readonly.append(row);save();assert row['all35_originals_unchanged']
  assert len(checks)==396 and len(readonly)==3 and len(receipt['layouts'])==35
  probe('strict-numeric-complete');receipt.update(status='PASS_V66_FOUR_ORIGINAL_POINTER_GDN_PROJECTION_LOCAL_STRICT',strict_output_checks=len(checks),readonly_all35=True,all4_outputs_consumed=True,selected_layers=list(LAYERS),runtime={'mlx':'0.32.3','device':info,'cache_limit_bytes':2<<30},extra_packed_weight_bytes=0,source_verified_final=identity()['source_pins_sha256']==ident['source_pins_sha256']);save();return 0
 except BaseException as error:receipt.update(status='FAIL_STOP',error=repr(error),traceback=traceback.format_exc());save();raise
 finally:mx.synchronize()

def main():
 ap=argparse.ArgumentParser(description=__doc__);g=ap.add_mutually_exclusive_group(required=True);g.add_argument('--inert',action='store_true');g.add_argument('--run-gpu',action='store_true');ap.add_argument('--out',type=Path);args=ap.parse_args();ident=identity()
 if args.inert:
  assert not any(n=='numpy' or n=='mlx' or n.startswith('mlx.') for n in sys.modules);print(json.dumps({'status':'PASS_CPU_INERT','pins':len(ident['source_pins']),'selected_tensor_ranges':24,'selected_payload_bytes':142387200,'cases':33,'strict_checks_required':396,'historical_inputs':'3 MLP-origin vectors, NOT GDN','MLX_imported':False,'GPU_used':False,'checkpoint_payload_loaded':False}));return 0
 assert args.out and args.out.is_absolute();args.out.mkdir(exist_ok=False);return gpu(args.out,ident)
if __name__=='__main__':raise SystemExit(main())
