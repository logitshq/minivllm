"""Root-only stationary A44 versus one qualified GDN variant; CPU mode is inert."""
import argparse,copy,gc,hashlib,importlib.metadata,importlib.util,json,os,signal,statistics,subprocess,sys,time,traceback
from pathlib import Path
sys.dont_write_bytecode=True
D=Path(__file__).resolve().parent;P=D.parent
PYTHON="/Users/ryanli/code/minivllm/.venv/bin/python"
PRIME,REPEATS=2048,8192
SCHEDULE=[("A","A"),("B","B"),("A","B"),("B","A"),("A","B"),("B","A"),("A","B"),("B","A"),("A","A"),("B","B")]
CONTROLS={"MLX_MAX_OPS_PER_BUFFER":"50","MLX_MAX_MB_PER_BUFFER":"50","MLX_BFS_MAX_WIDTH":"20","MLX_METAL_FAST_SYNCH":"0"}
REAL=[f"real-L{l}-D{d}" for l in (0,32,62) for d in (1,16,31)]
SYNTHETIC=["synthetic-zero","synthetic-random-PCG64-18001","synthetic-amplitude","synthetic-cancellation","synthetic-signedzero-subnormal","synthetic-repeated-head-boundary"]
TIMING_BANKS=REAL[:8]
CONTRACT={"u":((1,10240),"bfloat16"),"g":((1,48),"float32"),"beta":((1,48),"float32"),"S":((1,48,128,128),"float32")}
BANKROOT=P/"v18-recurrent-numeric-build1/gpu-recurrent-numeric-AC-run1"
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,value):Path(p).write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def identity(power):
 cfg=json.loads((D/"config.json").read_text());pins=json.loads((D/"source-pins.json").read_text())
 for path,h in pins.items():assert sha(path)==h,("source changed",path)
 assert sha(cfg["qualification_receipt"])==cfg["qualification_sha256"]
 q=json.loads(Path(cfg["qualification_receipt"]).read_text());assert q["status"]==cfg["qualification_status"] and q["strict_output_checks"]==30
 if cfg["version"]==60:assert q["strict_exp_bank_checks"]==15
 candidate_numeric=module("candidate_numeric_cpu",cfg["candidate_numeric"]);candidate_numeric.identity()
 parent=json.loads(Path(cfg["parent_qualification_receipt"]).read_text());assert parent["status"]=="PASS_V44_ACTIVATED_NORMREUSE_CACHE_STRICT_BYTES_9REAL_6SYNTHETIC" and parent["strict_output_checks"]==30
 plan=json.loads((P/"v18-activation-runtime-identity-build3.json").read_text())
 for key in ("required_modules","source_modules","native_images"):
  rows=plan[key]["required"] if key=="native_images" else plan[key]
  if isinstance(rows,dict):rows=rows.values()
  for rec in rows:assert sha(rec["path"])==rec["sha256"],rec["path"]
 assert sha(plan["metal_library"]["path"])==plan["metal_library"]["sha256"]
 return {"config":cfg,"source_pins":pins,"power_domain":power,"PRIME":PRIME,"REPEATS":REPEATS,"schedule":SCHEDULE,"timing_banks":TIMING_BANKS,"both_arms_mx_compile":False,"activation_outside_timing":True,"feedback":False,"limit_bytes":512<<20,"cache_bytes":64<<20,"model_loaded":False}

def block(mx,invoke,variant):
 prime_start=time.perf_counter_ns()
 for j in range(PRIME):result=invoke(variant,j)
 mx.synchronize();prime_end=time.perf_counter_ns()
 start=time.perf_counter_ns()
 for j in range(REPEATS):result=invoke(variant,j)
 mx.synchronize();end=time.perf_counter_ns()
 return {"variant":variant,"prime_start_ns":prime_start,"prime_end_ns":prime_end,"prime_repeats":PRIME,"start_ns":start,"end_ns":end,"total_ns":end-start,"ns_per_call":(end-start)/REPEATS,"repeats":REPEATS,"outputs_consumed_per_call":2},result

def controls_pass(values):return all(len(a)==2 and .98<=statistics.median(a)<=1.02 and all(.95<=v<=1.05 for v in a)for a in values.values())

def gpu(out,ident):
 assert sys.executable==PYTHON
 assert {n:importlib.metadata.version(n)for n in("mlx","mlx-metal")}=={"mlx":"0.32.3","mlx-metal":"0.32.3"}
 assert all(os.environ.get(k)==v for k,v in CONTROLS.items())
 assert not any(k.startswith(("DYLD_","METAL_","OMLX_","LITHOS_","MONOLITH_"))or k in("PYTHONPATH","PYTHONHOME")or(k.startswith("MLX_")and k not in CONTROLS)for k in os.environ)
 import mlx.core as mx
 import mlx.nn as nn
 import numpy as np
 sys.path.insert(0,str(P));from resource_snapshot_power_v1 import snapshot,deltas,guard
 info=mx.device_info();assert mx.default_device()==mx.gpu and info["device_name"]=="Apple M4 Max" and info["architecture"]=="applegpu_g16s"
 mx.set_memory_limit(512<<20);mx.set_cache_limit(64<<20)
 cfg=ident["config"];ahelper=module("actual44_A",cfg["A_helper"]);bhelper=module("qualified_variant_B",cfg["B_helper"])
 helpers={"A":ahelper.gated_delta_step_activated_norm_reuse_metal,"B":bhelper.gated_delta_step_activated_norm_reuse_metal}
 resources=[];checks=[];pairs=[];readonly=[];banks={};host_owners=[];first_error=None;final_error=None;result=None
 def probe(label):
  mx.synchronize();raw_resource=snapshot(label,os.getpid());write(out/("resource-%02d-raw.json"%len(resources)),raw_resource)
  row=copy.deepcopy(raw_resource);row["mlx_memory"]={"active":mx.get_active_memory(),"cache":mx.get_cache_memory(),"peak":mx.get_peak_memory()};resources.append(row);deltas(resources);write(out/"resources.json",resources);guard(resources)
  assert row["power_metadata"]["status"]=="known" and row["power_metadata"]["source"]==ident["power_domain"]
  assert row["processinfo_thermal_metadata"]["status"]=="known" and row["processinfo_thermal_metadata"]["code"] in(0,1)
  assert row["pressure_level_value"]==1 and row["exact_owned_pid_metadata"]["status"]=="known"
  assert all(x["owned"]for x in row["known_inference_processes"]),"foreign inference"
  assert row["swap_used_mb"]<=resources[0]["swap_used_mb"] and row["vm_counters"]["Swapouts"]==resources[0]["vm_counters"]["Swapouts"]
 def raw(value):
  mx.eval(value)
  return np.ascontiguousarray(np.asarray(value.view(mx.uint16)if value.dtype==mx.bfloat16 else value),dtype="<u2"if value.dtype==mx.bfloat16 else"<f4").tobytes()
 def fingerprint(value):
  data=raw(value);return {"shape":list(value.shape),"dtype":str(value.dtype),"bytes":len(data),"sha256":hashlib.sha256(data).hexdigest()}
 def signatures():return {n:{k:fingerprint(v)for k,v in b.items()}for n,b in banks.items()}
 def check_readonly(label):
  after=signatures();row={"label":label,"unchanged":after==initial,"before":initial,"after":after};readonly.append(row);write(out/"readonly.json",readonly);assert row["unchanged"],"immutable saved input mutated"
 def call(variant,bank):return helpers[variant](bank["activated"],bank["g"],bank["beta"],bank["S"])
 try:
  write(out/"runtime.json",{"device":info,"both_arms_mx_compile":False,"math_policy":{"A":{"math_mode":"safe"},"B":{"math_mode":"safe"}},"memory_limit_bytes":512<<20,"cache_limit_bytes":64<<20,"model_loaded":False,"power_domain":ident["power_domain"]})
  probe("fresh-no-model")
  for name in REAL+SYNTHETIC:
   meta=json.loads((BANKROOT/name/"bank.json").read_text());bank={}
   for key,(shape,dtype)in CONTRACT.items():
    rec=meta["arrays"][key];h=np.fromfile(BANKROOT/name/rec["file"],dtype="<u2"if dtype=="bfloat16"else"<f4").reshape(shape);host_owners.append(h)
    value=mx.array(h);bank[key]=value.view(mx.bfloat16)if dtype=="bfloat16"else value
   bank["activated"]=nn.silu(bank["u"]);mx.eval(*bank.values());banks[name]=bank
  mx.synchronize();initial=signatures();write(out/"inputs.json",{"banks":initial,"timing_banks":TIMING_BANKS,"activated_materialized_before_any_pair":True,"historical_saved_not_new_capture":True});probe("all-inputs-activation-ready")
  for name in REAL+SYNTHETIC:
   A=call("A",banks[name]);B=call("B",banks[name]);mx.eval(*A,*B)
   for key,a,b,shape in zip(("o","S_new"),A,B,((1,48,128),(1,48,128,128))):
    assert a.shape==b.shape==shape and a.dtype==b.dtype==mx.float32
    finite=bool(mx.all(mx.isfinite(a)).item()and mx.all(mx.isfinite(b)).item());da,db=raw(a),raw(b)
    row={"bank":name,"output":key,"A":fingerprint(a),"B":fingerprint(b),"finite":finite,"strict_bytes_equal":finite and da==db};checks.append(row);write(out/"bytes.json",checks)
    if not row["strict_bytes_equal"]:
     (out/"first-failure-A.bin").write_bytes(da);(out/"first-failure-B.bin").write_bytes(db);raise AssertionError(("strict A44 vs candidate bytes",row))
   del A,B,a,b,da,db
  check_readonly("all15-strict-qualified");assert len(checks)==30
  write(out/"qualification.json",{"status":"STRICT_A44_VS_VARIANT_15BANKS_30OUTPUTS_PASS","checks":30,"readonly":True,"A":cfg["A_helper"],"B":cfg["B_helper"],"activation_outside_pair":True,"both_arms_mx_compile":False,"timing_allowed":True})
  probe("strict30-before-timing")
  def invoke(variant,j):
   values=call(variant,banks[TIMING_BANKS[j%8]])
   mx.eval(*values)
   return values
  for index,order in enumerate(SCHEDULE):
   probe(f"pair{index}-before")
   a,ao=block(mx,invoke,order[0]);b,bo=block(mx,invoke,order[1])
   for rec,values in((a,ao),(b,bo)):rec["hashes"]=[hashlib.sha256(raw(v)).hexdigest()for v in values]
   del values
   assert a["hashes"]==b["hashes"],"post-pair output hashes differ"
   row={"pair":index,"order":order,"blocks":[a,b],"kind":"position-control"if order[0]==order[1]else"AB","slot2_div_slot1":b["ns_per_call"]/a["ns_per_call"]}
   if row["kind"]=="AB":
    by={a["variant"]:a,b["variant"]:b};row["reduction"]=1-by["B"]["ns_per_call"]/by["A"]["ns_per_call"]
   pairs.append(row);write(out/"pairs.json",pairs);del ao,bo;probe(f"pair{index}-after")
  check_readonly("all15-post-timing")
  ab=[r for r in pairs if r["kind"]=="AB"];ctrl={v:[r["slot2_div_slot1"]for r in pairs if r["kind"]=="position-control"and r["order"][0]==v]for v in("A","B")}
  med={v:statistics.median(next(b["ns_per_call"]for b in r["blocks"]if b["variant"]==v)for r in ab)for v in("A","B")};red=1-med["B"]/med["A"];wins=sum(r["reduction"]>0 for r in ab);ok=controls_pass(ctrl)
  status="CONTROL_DRIFT_STOP"if not ok else"EXPLORATORY-PASS"if red>=.05 and wins>=5 else"STOP-MICRO-GATE"
  result={"status":status,"checks":30,"all_bytes_exact":True,"readonly":True,"medians_ns":med,"median_reduction":red,"wins":wins,"pairs":6,"control_passed":ok,"control_slot2_div_slot1":ctrl,"minimum_reduction":.05,"minimum_wins":5,"power_domain":ident["power_domain"],"scope":"stationary immutable saved-input A44 versus one candidate only; neither mx.compile; local wall time not E2E","memory_lifetime":"only final tuple retained per block; invoke never feeds output back or appends arrays; actual peak inresources","model_loaded":False,"adopted":False,"full_model_qualified":False,"E2E_run":False,"AC_frontier_entry":False};write(out/"result.json",result)
 except BaseException as error:
  first_error=error;write(out/"first-failure.json",{"error":repr(error),"traceback":traceback.format_exc(),"completed_strict_checks":sum(c["strict_bytes_equal"]for c in checks),"timing_allowed":(out/"qualification.json").exists()})
 finally:
  try:probe("finally-before-child-exit")
  except BaseException as error:final_error=error;write(out/"finally-resource-failure.json",{"error":repr(error),"traceback":traceback.format_exc()})
 if first_error is not None:raise first_error
 if final_error is not None:raise final_error
 return 0 if result["status"]=="EXPLORATORY-PASS"else 2

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("mode",choices=("cpu","run","child"));ap.add_argument("--out",type=Path,required=True);ap.add_argument("--owner-pid",type=int);ap.add_argument("--power-domain",choices=("AC Power","Battery Power"),default="AC Power");args=ap.parse_args();ident=identity(args.power_domain)
 if args.mode=="cpu":
  args.out.mkdir(exist_ok=False);assert not any(n=="mlx"or n.startswith(("mlx.","numpy"))for n in sys.modules);write(args.out/"cpu.json",{"status":"PASS_CPU_INERT","identity":ident,"MLX_imported":False,"GPU_used":False,"model_loaded":False});return 0
 assert sys.executable==PYTHON
 if args.mode=="child":
  assert args.owner_pid==os.getppid()and args.owner_pid>1
  try:return gpu(args.out,ident)
  except BaseException as error:write(args.out/"result.json",{"status":"FAIL-STOP","error":repr(error),"adopted":False,"power_domain":args.power_domain});traceback.print_exc();return 1
  finally:write(args.out/"child-finally.json",{"pid":os.getpid(),"exiting":True})
 args.out.mkdir(exist_ok=False);write(args.out/"identity.json",ident);sys.path.insert(0,str(P/"v25-c1-screen-build1"));from support import identity as process_identity,process_parent,cleanup_owned
 proc=None;birth=None;rc=1
 def stop(sig,frame):raise KeyboardInterrupt("owned stationarymicro supervisor interrupted")
 previous=signal.signal(signal.SIGTERM,stop)
 with(args.out/"child.log").open("w")as log:
  try:
   argv=[sys.executable,"-B",str(Path(__file__).resolve()),"child","--out",str(args.out),"--owner-pid",str(os.getpid()),"--power-domain",args.power_domain]
   proc=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT);birth=process_identity(proc.pid);assert birth is not None and process_parent(proc.pid)["ppid"]==os.getpid();write(args.out/"child.json",{"pid":proc.pid,"owner_pid":os.getpid(),"argv":argv,"birth_identity":birth,"timeout_s":170});rc=proc.wait(timeout=170)
  except BaseException as error:write(args.out/"supervisor-failure.json",{"error":repr(error),"traceback":traceback.format_exc()});rc=1
  finally:
   if proc is not None:write(args.out/"cleanup.json",cleanup_owned(proc,birth,args.out/"cleanup-identities.json"))
   signal.signal(signal.SIGTERM,previous)
 print(f"child exit={rc};exactownedchild reaped;evidence={args.out}",flush=True);return rc
if __name__=="__main__":raise SystemExit(main())
