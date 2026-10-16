"""V72 root-only 8-real-weight-bank native range ICB; CPU inert mode imports no MLX/NumPy/native library."""
import argparse
import ctypes
import hashlib
import importlib.util
import importlib.metadata
import json
import os
from pathlib import Path
import statistics
import sys
import time
import traceback
sys.dont_write_bytecode=True
D=Path(__file__).resolve().parent;P=D.parent;LIB=D/'libv72replay.dylib'
LAYERS=(0,8,16,24,32,40,48,56);K=5120;N=17408
CONTROLS={'MLX_MAX_OPS_PER_BUFFER':'50','MLX_MAX_MB_PER_BUFFER':'50','MLX_BFS_MAX_WIDTH':'20','MLX_METAL_FAST_SYNCH':'0'}
PRIME=128;REPEATS=1024
SCHEDULE=(('A','A'),('B','B'))+(('A','B'),('B','A'))*3+(('A','A'),('B','B'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def module(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m

def source_check():
 pins=json.loads((D/'source-pins.json').read_text())
 for p,h in pins.items():assert sha(p)==h,('changed source',p)
 assert str(LIB) in pins and str(D/'micro.py') in pins
 fail=P/'v23-root-control-build1/fullmodel-v72num-attempt1/child-output/receipt.json'
 assert sha(fail)=='b4159e4d4758fa1ffd20f11165c6428e7301975b14178f7a88fba7722be27c20' and json.loads(fail.read_text())['status']=='FAIL_STOP'
 proof=json.loads((D/'source-write-proof.json').read_text());assert proof['status']=='PASS_CPU_PINNED_SOURCE_STORE_ONLY_OUTPUT_PROOF' and proof['reflection_policy']=='RW retained; source-only write inference for exact two observed output args'
 return pins

def identity():
 old=module('v72_frozen_v22_cpu_support',P/'v22-nvfp4-qmv-r2-build1/micro.py')
 index=json.loads((old.CHECKPOINT/'model.safetensors.index.json').read_text())['weight_map']
 bank=old.REAL_BANK;receipt=json.loads((bank/'receipt.json').read_text());saved={r['layer']:r for r in receipt['rows']};rows=[]
 for layer in LAYERS:
  tensors=[]
  for suffix,dtype,shape in [('weight','U32',(N,K//8)),('scales','U8',(N,K//16))]:
   keys=[k for k in index if k.endswith(f'.layers.{layer}.mlp.gate_proj.'+suffix)];assert len(keys)==1
   tensors.append(old.tensor_meta(old.CHECKPOINT/index[keys[0]],keys[0],dtype,shape))
  x=bank/f'L{layer:02d}-x.raw.le.bin';assert x.stat().st_size==10240 and sha(x)==saved[layer]['x']['raw_sha256']
  rows.append({'layer':layer,'tensors':tensors,'X_path':str(x),'X_sha256':sha(x),'X_shape':[1,K],'X_dtype':'bfloat16'})
 assert sum(t['nbytes'] for r in rows for t in r['tensors'])==401080320
 return old,rows

def gpu(args,pins,old,rows):
 import mlx.core as mx
 import numpy as np
 sys.path.insert(0,str(P));from resource_snapshot_power_v1 import snapshot,guard,deltas
 assert sys.executable=='/Users/ryanli/code/minivllm/.venv/bin/python'
 assert os.environ.get('DYLD_INSERT_LIBRARIES')==str(LIB)
 assert all(os.environ.get(k)==v for k,v in CONTROLS.items())
 assert importlib.metadata.version('mlx')==importlib.metadata.version('mlx-metal')=='0.32.3'
 info=mx.device_info();assert mx.default_device()==mx.gpu and info['device_name']=='Apple M4 Max' and info['architecture']=='applegpu_g16s'
 mx.set_cache_limit(2<<30)
 prior=module('v72_shared',P/'v33-fixed-buffer-replay-worker-build3/fullqual.py')
 helper=module('v72_frozen69',P/'v69-nvfp4-r8-reuse-build1/geometry.py')
 out=args.out;out.mkdir(exist_ok=False);(out/'numeric').mkdir();resources=[]
 result={'status':'STARTED','scope':'8-bank stationary native ICB range micro; no fullmodel/E2E/HTTP/adoption','model_loads':0,'source_pins':pins,'banks':rows,'candidate_R8_source':str(P/'v69-nvfp4-r8-reuse-build1'),'old69_STOP_unchanged':True,'timing_run':not args.qualify_only,'adopted':False}
 write(out/'receipt.json',result)
 def probe(label):
  mx.synchronize();raw=snapshot(label,os.getpid());p=out/f'resource-{len(resources):02d}-raw.json';write(p,raw);resources.append(raw);deltas(resources);guard(resources)
  assert raw['power_metadata']['source']=='AC Power' and raw['processinfo_thermal_metadata']['code']==0 and raw['pressure_level_value']==1
  assert raw['swap_used_mb']==resources[0]['swap_used_mb'] and raw['vm_counters']['Swapouts']==resources[0]['vm_counters']['Swapouts']
  assert all(x['owned'] for x in raw['known_inference_processes'])
  write(out/'resources.json',resources);return {'raw':p.name,'sha256':sha(p),'mlx_active_bytes':mx.get_active_memory(),'mlx_cache_bytes':mx.get_cache_memory(),'mlx_peak_bytes':mx.get_peak_memory()}
 U=ctypes.c_uint64;I=ctypes.c_uint32;V=ctypes.c_void_p;S=ctypes.c_char_p
 lib=ctypes.CDLL(None)
 for name,types in {'v33_pin':[U,U,S],'v33_begin':[],'v33_end_build':[],'v33_register_output':[U,U,S],'v72_register_slot':[I,I]+[U]*8,'v72_build_ranges':[],'v72_loop_range':[I,I],'v72_copy_output':[I,I,U,V],'v72_verify_readonly':[]}.items():
  f=getattr(lib,name);f.argtypes=types;f.restype=ctypes.c_int
 for name in ['v33_report_json','v72_report_json']:
  f=getattr(lib,name);f.argtypes=[];f.restype=S
 def report():return json.loads(lib.v72_report_json().decode())
 def call(name,*values):
  rc=getattr(lib,name)(*values)
  if rc:write(out/'native-failure.json',report());raise RuntimeError((name,rc))
 def raw(a):return np.asarray(a.view(mx.uint16)).tobytes()
 def checked_raw(a):
  mx.eval(a);assert a.shape==(1,N) and a.dtype==mx.bfloat16 and bool(mx.all(mx.isfinite(a)).item());return raw(a)
 checks=[];native_checks=[];owners=[];owner_hashes=[];captured=[];output_holders=[]
 def readonly(label):
  now=[hashlib.sha256(h.bytes_view).hexdigest() for h in owners+output_holders]
  assert now==owner_hashes,('originalinput/output changed',label)
  call('v72_verify_readonly');return {'label':label,'arrays':len(now),'all_sha256_unchanged':True}
 def compare(a,b,label):
  aa,bb=checked_raw(a),checked_raw(b);record={'label':label,'A_sha256':hashlib.sha256(aa).hexdigest(),'B_sha256':hashlib.sha256(bb).hexdigest(),'bytes':len(aa),'strict':aa==bb};checks.append(record);write(out/'numeric-checks.json',checks);assert aa==bb,record
 try:
  probe('before-any-weight-load');banks=[];initial_input_hashes=[];compiled={'A':[],'B':[]};startup=[];traces={'A':[0]*8,'B':[0]*8}
  for slot,row in enumerate(rows):
   w,s=[old.load_range(np,mx,t) for t in row['tensors']]
   x=mx.array(np.fromfile(row['X_path'],dtype='<u2'),dtype=mx.uint16).view(mx.bfloat16).reshape(1,K)
   mx.eval(w,s,x);assert hashlib.sha256(raw(x)).hexdigest()==row['X_sha256']
   z=mx.zeros((1,K),dtype=mx.bfloat16);rng=np.random.default_rng(20261016+row['layer']);rnd=mx.array(rng.standard_normal((1,K)).astype(np.float32),dtype=mx.bfloat16)
   mx.eval(z,rnd);r8=helper.prepare(mx,w,s,runtime_version='0.32.3',device_name=info['device_name'],architecture=info['architecture']);assert r8.inputs[0] is w and r8.inputs[1] is s
   banks.append((w,s,x,z,rnd,r8))
   for label,a in zip(('weight','scales','realX','zeroX','randomX'),(w,s,x,z,rnd)):
    owners.append(prior.Shared(mx,np,a,f'layer{row["layer"]}.{label}'));initial_input_hashes.append(hashlib.sha256(owners[-1].bytes_view).hexdigest())
   def make_a(w=w,s=s,j=slot):
    def fn(a):traces['A'][j]+=1;return mx.quantized_matmul(a,w,s,transpose=True,**old.Q)
    return fn
   def make_b(r8=r8,j=slot):
    def fn(a):traces['B'][j]+=1;return r8.project(a)
    return fn
   for arm,factory in [('A',make_a),('B',make_b)]:
    t=time.perf_counter_ns();fn=mx.compile(factory(),shapeless=False);v=fn(x);mx.eval(v);startup.append({'arm':arm,'slot':slot,'cold_compile_eval_ns':time.perf_counter_ns()-t});compiled[arm].append(fn)
   for vi,a in enumerate((x,z,rnd)):
    e=mx.quantized_matmul(a,w,s,transpose=True,**old.Q);aa=compiled['A'][slot](a);bb=compiled['B'][slot](a)
    compare(e,aa,f'bank{slot}.vector{vi}.eager-A');compare(e,bb,f'bank{slot}.vector{vi}.eager-B');compare(aa,bb,f'bank{slot}.vector{vi}.A-B')
  assert len(checks)==72 and all(v==[1]*8 for v in traces.values())
  assert initial_input_hashes==[hashlib.sha256(h.bytes_view).hexdigest() for h in owners], 'numeric72 changed immutable input'
  result['MLX72_input_arrays_readonly']=40;write(out/'startup.json',startup)
  write(out/'inputs.json',[{**h.meta,'sha256':digest} for h,digest in zip(owners,initial_input_hashes)])
  result['weights_payload_bytes']=401080320;result['native_weight_repack_bytes']=0;result['compiled_trace_counts']=traces
  mx.synchronize();probe('weight-numeric72-ready-before-capture')
  # Pin exact W/scales/realX once; zero/random qualification arrays are never captured.
  for i in range(8):
   for h in owners[5*i:5*i+3]:call('v33_pin',h.address,h.nbytes,h.label.encode())
  expected=[]
  for arm in ('A','B'):
   for slot,b in enumerate(banks):expected.append(checked_raw(compiled[arm][slot](b[2])))
  before_inputs=[hashlib.sha256(h.bytes_view).hexdigest() for h in owners]
  call('v33_begin')
  for arm in ('A','B'):
   for slot,b in enumerate(banks):
    y=compiled[arm][slot](b[2]);mx.eval(y);mx.synchronize();captured.append(y)
  call('v33_end_build')
  assert before_inputs==[hashlib.sha256(h.bytes_view).hexdigest() for h in owners]
  for i,y in enumerate(captured):
   h=prior.Shared(mx,np,y,f'{"A" if i<8 else "B"}.{i%8}.original-output');output_holders.append(h)
   assert bytes(h.bytes_view)==expected[i];call('v33_register_output',h.address,h.nbytes,h.label.encode());(out/'numeric'/f'expected-{i:02d}.raw').write_bytes(expected[i])
  for i,h in enumerate(output_holders):
   j=i%8;w,s,x=owners[5*j:5*j+3];call('v72_register_slot',i//8,j,w.address,w.nbytes,s.address,s.nbytes,x.address,x.nbytes,h.address,h.nbytes)
  call('v72_build_ranges');n=report();assert n['enabled'] and n['capture_dispatches']==n['commands_in_single_ICB']==16
  write(out/'native-built.json',n);write(out/'capture-native40.json',json.loads(lib.v33_report_json().decode()))
  owner_hashes=[hashlib.sha256(h.bytes_view).hexdigest() for h in owners+output_holders]
  write(out/'capture-output-baselines.json',[{**h.meta,'sha256':owner_hashes[len(owners)+i]} for i,h in enumerate(output_holders)])
  def verify_native(arm,label):
   rows=[]
   for slot in range(8):
    buf=ctypes.create_string_buffer(34816);call('v72_copy_output',arm,slot,34816,buf);b=buf.raw;reference=expected[arm*8+slot]
    p=out/'numeric'/f'{label}-slot{slot}.raw';p.write_bytes(b);row={'arm':arm,'slot':slot,'raw_file':str(p.relative_to(out)),'sha256':hashlib.sha256(b).hexdigest(),'expected_sha256':hashlib.sha256(reference).hexdigest(),'strict':b==reference};assert b==reference,row;rows.append(row)
   return rows
  for groups in (1,2,3,4,5,7,8,16):
   for arm in (0,1):
    call('v72_loop_range',arm,groups);n=report();assert n['observed_encoded_dispatches']==8*groups and n['last_CB_completed'] and n['buffer_barriers_per_last_loop']==groups
    native_checks.extend(verify_native(arm,f'qual-g{groups}-a{arm}'))
  assert len(native_checks)==128;write(out/'native128-checks.json',native_checks);result['readonly_after_native128']=readonly('native128')
  result.update(status='PASS_V72_72_MLX_128_NATIVE_STRICT_NO_TIMING',MLX_strict_checks=72,native_strict_checks=128,qualification=True,model_loaded=False)
  write(out/'receipt.json',result);probe('after-native128-before-any-timing')
  if args.qualify_only:return 0
  assert args.qualified_receipt and args.qualified_sha and sha(args.qualified_receipt)==args.qualified_sha,'fresh perf requires actual prior V72 numerical receipt+SHA'
  prior_qual=json.loads(args.qualified_receipt.read_text());assert prior_qual['status']=='PASS_V72_72_MLX_128_NATIVE_STRICT_NO_TIMING' and prior_qual['MLX_strict_checks']==72 and prior_qual['native_strict_checks']==128 and prior_qual['source_pins']==pins
  pairs=[]
  for pi,order in enumerate(SCHEDULE):
   before=probe(f'pair{pi}-before');blocks=[]
   for arm_name in order:
    arm=0 if arm_name=='A' else 1
    call('v72_loop_range',arm,PRIME)
    t0=time.perf_counter_ns();call('v72_loop_range',arm,REPEATS);t1=time.perf_counter_ns()
    n=report();assert n['observed_encoded_dispatches']==8192 and n['last_CB_completed'] and n['buffer_barriers_per_last_loop']==1024
    blocks.append({'arm':arm_name,'prime_groups':PRIME,'timed_groups':REPEATS,'encoded_dispatches':8192,'wall_ns':t1-t0,'ns_per_dispatch':(t1-t0)/8192,'native':n,'outputs':verify_native(arm,f'timed-p{pi}-{len(blocks)}')})
   after=probe(f'pair{pi}-after');ro=readonly(f'pair{pi}')
   a,b=blocks;pair={'pair':pi,'order':order,'blocks':blocks,'resources':[before,after],'readonly':ro,'kind':'control' if order[0]==order[1] else 'AB'}
   if pair['kind']=='control':pair['slot2_div_slot1']=b['wall_ns']/a['wall_ns']
   else:
    vals={x['arm']:x['wall_ns'] for x in blocks};pair['reduction']=1-vals['B']/vals['A']
   pairs.append(pair);write(out/'pairs.json',pairs)
  ab=[x for x in pairs if x['kind']=='AB'];controls={a:[x['slot2_div_slot1'] for x in pairs if x['kind']=='control' and x['order'][0]==a] for a in ('A','B')}
  med={a:statistics.median(next(b['ns_per_dispatch'] for b in p['blocks'] if b['arm']==a) for p in ab) for a in ('A','B')};gain=1-med['B']/med['A'];wins=sum(x['reduction']>0 for x in ab)
  control_pass=all(all(.95<=x<=1.05 for x in xs) and .98<=statistics.median(xs)<=1.02 for xs in controls.values())
  status='CONTROL_DRIFT_STOP' if not control_pass else 'EXPLORATORY_PASS_NOT_ADOPTED' if gain>=.05 and wins>=5 else 'STOP_MICRO_GATE'
  result.update(status=status,MLX_strict_checks=72,native_strict_checks=128,medians_ns_per_dispatch=med,median_reduction=gain,wins=wins,pairs=6,controls=controls,control_passed=control_pass,scope='oneffi oneCB per8192dispatches; range8×1024 immutable independent realbankQMV; wall includes encode/commitwait, no HTTP/fullmodel',native_final=report())
  assert traces=={'A':[1]*8,'B':[1]*8};result['no_Python_retrace']=True;write(out/'receipt.json',result)
  return 0 if status=='EXPLORATORY_PASS_NOT_ADOPTED' else 2
 except BaseException as e:
  result.update(status='FAIL_STOP',error=repr(e),traceback=traceback.format_exc());write(out/'receipt.json',result);raise
 finally:
  try:probe('finally-before-root-owned-exit')
  except BaseException as e:result.update(status='FAIL_STOP',finally_error=repr(e));write(out/'receipt.json',result);raise
  source_check();write(out/'receipt.json',result)

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path);ap.add_argument('--inert',action='store_true');ap.add_argument('--qualify-only',action='store_true');ap.add_argument('--qualified-receipt',type=Path);ap.add_argument('--qualified-sha')
 args=ap.parse_args();pins=source_check();old,rows=identity()
 if not args.inert and not args.qualify_only:
  assert args.qualified_receipt and args.qualified_sha and sha(args.qualified_receipt)==args.qualified_sha, 'perf requires prior actual qualification receipt/SHA before runtime import'
  q=json.loads(args.qualified_receipt.read_text());assert q['status']=='PASS_V72_72_MLX_128_NATIVE_STRICT_NO_TIMING' and q['source_pins']==pins and q['timing_run'] is False
 if args.inert:
  assert not any(k=='mlx' or k.startswith('mlx.') or k=='numpy' or k.startswith('numpy.') for k in sys.modules)
  print(json.dumps({'status':'PASS_CPU_INERT','pins':len(pins),'layers':LAYERS,'weights_payload_bytes':401080320,'capture_dispatches':16,'ranges':[[0,8],[8,16]],'MLX_checks':72,'native_checks':128,'prime_groups':PRIME,'timed_groups':REPEATS,'GPU_used':False}));return 0
 assert args.out and args.out.is_absolute();return gpu(args,pins,old,rows)
if __name__=='__main__':raise SystemExit(main())
