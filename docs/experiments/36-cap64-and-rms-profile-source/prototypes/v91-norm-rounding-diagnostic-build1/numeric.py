"""V91 four-path RMS rounding diagnostic, root-only GPU opt-in; no timing/adoption gate."""
import argparse, hashlib, importlib.util, importlib.metadata, itertools, json, math, os, random, struct, sys, traceback
from pathlib import Path
sys.dont_write_bytecode=True
D=Path(__file__).resolve().parent;P=D.parent
K=5120;EPS=1e-6
CHECKPOINT=Path('/Users/ryanli/.cache/huggingface/hub/models--mlx-community--Qwen3.8-27B-nvfp4/snapshots/5ff8ef173ad0d7c3aae92f0be43031a6ab8067c6')
NORM=P/'candidate-v44-conv-normreuse-build1-root/tutorial_code/miniengine/kernels/norm_metal.py'
CONTROLS={'MLX_MAX_OPS_PER_BUFFER':'50','MLX_MAX_MB_PER_BUFFER':'50','MLX_BFS_MAX_WIDTH':'20','MLX_METAL_FAST_SYNCH':'0'}
LABELS=('zero','alternating_unit','powers_of_two_cycle','small_signed','ones_with_outliers','ramp_decimal','fixed_random_unit_91001','fixed_random_16_91002','historical_postnorm_output_generic_NOT_matched_prenorm')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def module(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m

def identity():
 pins=json.loads((D/'source-pins.json').read_text())
 for p,h in pins.items():assert sha(p)==h,('source changed',p)
 plan=json.loads((D/'inputs.json').read_text());assert len(plan['gammas'])==3 and len(LABELS)==9
 # Exact selected small detached ranges, never a full shard.
 for g in plan['gammas']:
  with Path(g['path']).open('rb') as f:
   size=struct.unpack('<Q',f.read(8))[0];header=f.read(size);assert hashlib.sha256(header).hexdigest()==g['header_sha256']
   entry=json.loads(header)[g['key']];assert entry['dtype']=='BF16' and entry['shape']==[K]
   assert [x+8+size for x in entry['data_offsets']]==[g['offset'],g['offset']+g['nbytes']]
   f.seek(g['offset']);raw=f.read(g['nbytes']);assert len(raw)==2*K and hashlib.sha256(raw).hexdigest()==g['raw_sha256']
 assert sha(plan['historical']['path'])==plan['historical']['sha256']
 return pins,plan

def synthetic():
 r1=random.Random(91001);r2=random.Random(91002)
 return [[0.0]*K,[(-1.0 if i&1 else 1.0) for i in range(K)],
 [(-1.0 if i&1 else 1.0)*2.0**((i%17)-8) for i in range(K)],
 [(-1.0 if i&1 else 1.0)*2.0**-16 for i in range(K)],
 [128.0 if i==0 else -128.0 if i in (127,255,5119) else 1.0 for i in range(K)],
 [(i%257-128)/127.0 for i in range(K)],
 [r1.uniform(-1,1) for i in range(K)],[r2.uniform(-16,16) for i in range(K)]]

def bf16_values(raw):return [struct.unpack('<f',struct.pack('<I',u<<16))[0] for u in struct.unpack('<'+'H'*(len(raw)//2),raw)]
def compare(left,right):
 assert len(left)==len(right)==2*K
 aa=struct.unpack('<'+'H'*K,left);bb=struct.unpack('<'+'H'*K,right)
 # Monotone bit order; -0/+0 are adjacent distinct encodings (ULP distance1).
 ordered=lambda u:(~u)&0xffff if u&0x8000 else u|0x8000
 valsA=bf16_values(left);valsB=bf16_values(right);assert all(map(math.isfinite,valsA+valsB))
 return {'strict_bytes_equal':left==right,'different_elements':sum(a!=b for a,b in zip(aa,bb)),
 'max_BF16_ULP_distance':max(abs(ordered(a)-ordered(b)) for a,b in zip(aa,bb)),
 'max_absolute_difference':max(abs(a-b) for a,b in zip(valsA,valsB)),
 'signed_zero_bit_differences':sum(a!=b and (a&0x7fff)==(b&0x7fff)==0 for a,b in zip(aa,bb)),
 'comparison_is_adoption_or_oldgold_gate':False}

def fallback_expression(mx,x,gamma,eps):
 # GPU-op emulation of official fast.cpp CPU fallback formula, not actual CPU fallback execution.
 xf=x.astype(mx.float32)
 normalized=xf*mx.rsqrt(mx.mean(xf*xf,axis=-1,keepdims=True)+eps)
 normalized_bf16=normalized.astype(x.dtype)
 mx.eval(normalized_bf16)  # Explicit diagnostic materialization preserves this cast boundary.
 return normalized_bf16*gamma

def gpu(args,pins,plan):
 versions={n:importlib.metadata.version(n) for n in ('mlx','mlx-metal')};assert versions=={'mlx':'0.32.3','mlx-metal':'0.32.3'}
 assert sys.executable=='/Users/ryanli/code/minivllm/.venv/bin/python'
 assert all(os.environ.get(k)==v for k,v in CONTROLS.items())
 assert not any(k.startswith(('DYLD_','METAL_')) for k in os.environ),'no injected native library/Metal overrides'
 assert not any(k.startswith('MLX_') and k not in CONTROLS for k in os.environ),'unknown MLX override'
 import mlx.core as mx
 import numpy as np
 sys.path.insert(0,str(P));from resource_snapshot_power_v1 import snapshot,guard,deltas
 info=mx.device_info();assert mx.default_device()==mx.gpu and info['device_name']=='Apple M4 Max' and info['architecture']=='applegpu_g16s'
 mx.set_memory_limit(512<<20);mx.set_cache_limit(64<<20)
 out=args.out;out.mkdir(exist_ok=False);(out/'raw').mkdir();resources=[];cases=[]
 result={'status':'STARTED','scope':'four RMS rounding paths, local27cases108outputs; no timing/model/adoption','source_pins':pins,'runtime':{'versions':versions,'device':info,'controls':CONTROLS,'memory_limit_bytes':512<<20,'cache_limit_bytes':64<<20},'model_loaded':False,'performance_run':False,'oldgold_gate':False,'adopted':False,'formula_protocol':json.loads((D/'protocol.json').read_text())}
 def save():write(out/'receipt.json',result)
 def probe(label):
  mx.synchronize();raw=snapshot(label,os.getpid());write(out/f'resource-{len(resources):02d}-raw.json',raw);resources.append(raw);deltas(resources);guard(resources)
  assert raw['power_metadata']['source']=='AC Power' and raw['processinfo_thermal_metadata']['code']==0 and raw['pressure_level_value']==1
  assert raw['swap_used_mb']==resources[0]['swap_used_mb'] and raw['vm_counters']['Swapouts']==resources[0]['vm_counters']['Swapouts']
  assert all(x['owned'] for x in raw['known_inference_processes']);write(out/'resources.json',resources)
 def raw(a):
  mx.eval(a);assert a.dtype==mx.bfloat16;return np.asarray(a.view(mx.uint16)).tobytes()
 def finite(a):mx.eval(a);assert bool(mx.all(mx.isfinite(a)).item())
 def load_gamma(g):
  mapping=np.memmap(g['path'],mode='r',dtype='<u2',offset=g['offset'],shape=(K,));detached=np.array(mapping,copy=True);assert not np.shares_memory(mapping,detached)
  mm=mapping._mmap;del mapping;mm.close();assert mm.closed
  a=mx.array(detached,dtype=mx.uint16).view(mx.bfloat16);mx.eval(a);assert hashlib.sha256(raw(a)).hexdigest()==g['raw_sha256'];return a
 try:
  save();probe('before-small-load')
  norm=module('v91_frozen44_norm',NORM);explicit=module('v91_original_explicit',D/'explicit_reference.py')
  gammas=[load_gamma(g) for g in plan['gammas']]
  offsets=[(g.astype(mx.float32)-1.0).astype(mx.bfloat16) for g in gammas];mx.eval(offsets)
  xs=[mx.array(values,dtype=mx.bfloat16).reshape(1,K) for values in synthetic()]
  historic=np.fromfile(plan['historical']['path'],dtype='<u2');assert historic.nbytes==2*K
  xs.append(mx.array(historic,dtype=mx.uint16).view(mx.bfloat16).reshape(1,K));mx.eval(xs)
  originals=gammas+offsets+xs
  for a in originals:finite(a)
  before=[hashlib.sha256(raw(a)).hexdigest() for a in originals]
  for i,x in enumerate(xs):(out/'raw'/f'input-x{i:02d}.bf16.raw').write_bytes(raw(x))
  weights=[]
  for i,(g,w) in enumerate(zip(gammas,offsets)):
   b=raw(g);ww=raw(w);rt=(1.0+w).astype(mx.bfloat16);finite(rt)
   for label,blob in [('gamma',b),('loader-offset',ww),('effective-roundtrip',raw(rt))]:(out/'raw'/f'weight-g{i}-{label}.bf16.raw').write_bytes(blob)
   weights.append({'gamma':plan['gammas'][i],'GPU_gamma_offset_roundtrip':compare(b,raw(rt)),'raw_gamma_sha256':hashlib.sha256(b).hexdigest(),'offset_sha256':hashlib.sha256(ww).hexdigest()})
  result['weight_diagnostics']=weights
  probe('prepared-before-case0')
  for gi,(gamma,offset) in enumerate(zip(gammas,offsets)):
   for xi,x in enumerate(xs):
    assert x.shape==(1,K) and gamma.shape==offset.shape==(K,)
    outputs={'A':mx.fast.rms_norm(x,gamma,EPS),
     'B':norm.norm(x,offset,EPS,zero_centered=True),
     'C':fallback_expression(mx,x,gamma,EPS),
     'D':explicit._rms_norm_explicit(x,gamma,EPS)}
    mx.eval(list(outputs.values()));blobs={};metadata={}
    for arm,a in outputs.items():
     assert a.shape==(1,K) and a.dtype==mx.bfloat16;finite(a);blob=raw(a);assert len(blob)==2*K
     name=f'case-g{gi}-x{xi:02d}-{arm}.bf16.raw';(out/'raw'/name).write_bytes(blob);blobs[arm]=blob
     metadata[arm]={'raw':f'raw/{name}','shape':[1,K],'dtype':'bfloat16','bytes':len(blob),'finite':True,'sha256':hashlib.sha256(blob).hexdigest()}
    comparisons={a+'-'+b:compare(blobs[a],blobs[b]) for a,b in itertools.combinations('ABCD',2)}
    after=[hashlib.sha256(raw(a)).hexdigest() for a in originals];assert before==after,'original input/weight changed'
    cases.append({'gamma':gi,'input':xi,'input_label':LABELS[xi],'outputs':metadata,'comparisons':comparisons,'all15_originals_readonly':True});write(out/'cases.json',cases)
   probe(f'after-gamma{gi}')
  assert len(cases)==27 and sum(len(x['outputs']) for x in cases)==108
  result.update(status='DIAGNOSTIC_COMPLETED_NO_ADOPTION',cases=27,outputs=108,pairwise_diagnostic_comparisons=162,readonly_original_arrays=15,readonly_case_checkpoints=27,all_finite=True,all15_originals_readonly=True,
   unequal_pair_counts={p:sum(not x['comparisons'][p]['strict_bytes_equal'] for x in cases) for p in ('A-B','A-C','A-D','B-C','B-D','C-D')},
   D_compile_context='original44 @partial(mx.compile,shapeless=True), default backend compiler options; cold first call is diagnostic, never timed',
   A_B_C_backend_FMA_or_selected_PSO='NOT_OBSERVED; API/source contexts are in protocol, no guessed safe/fast flags')
  save();return 0
 except BaseException as e:
  result.update(status='DIAGNOSTIC_FAILED_STOP',error=repr(e),traceback=traceback.format_exc(),completed_cases=len(cases));save();raise
 finally:
  try:probe('finally-before-owned-exit')
  except BaseException as e:result.update(status='DIAGNOSTIC_FAILED_STOP',resource_final_error=repr(e));save();raise
  identity();save()

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--inert',action='store_true');ap.add_argument('--execute-gpu',action='store_true');ap.add_argument('--out',type=Path);args=ap.parse_args()
 pins,plan=identity()
 if args.inert or not args.execute_gpu:
  assert not any(n=='mlx' or n.startswith(('mlx.','numpy')) for n in sys.modules)
  print(json.dumps({'status':'PASS_CPU_INERT_ONLY','pins':len(pins),'gammas':3,'inputs':9,'cases':27,'outputs':108,'GPU_used':False}));return 0
 assert args.out and args.out.is_absolute();return gpu(args,pins,plan)
if __name__=='__main__':raise SystemExit(main())
