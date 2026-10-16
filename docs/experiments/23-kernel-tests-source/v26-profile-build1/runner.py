"""Diagnostic B1 decode profile. CPU --inert default; root owns actual GPU child."""
import argparse
import cProfile
import ctypes
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import pstats
import sys
import time
import traceback

sys.dont_write_bytecode=True
D=Path(__file__).resolve().parent
P=D.parent
LIB=D/"libv26trace.dylib"

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+"\n")

def identity(root,manifest):
    m=json.loads(manifest.read_text());files=m["files"]
    base=root if all(n.startswith("tutorial_code/miniengine/") for n in files) else root/"tutorial_code/miniengine"
    for name,h in files.items():assert sha(base/name)==h,("source changed",name)
    return {"package_root":str(root),"manifest":str(manifest),"manifest_sha256":sha(manifest),"source_count":len(files),"driver_sha256":sha(__file__),"native_source_sha256":sha(D/"trace.mm"),"native_library_sha256":sha(LIB),"workload_sha256":sha(P/"workload.json")}

def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--package-root",type=Path,default=P/"candidate-v17-gdn-gates-build1-root")
    ap.add_argument("--source-manifest",type=Path,default=P/"source-manifest-v17-gdn-gates-build1.json")
    ap.add_argument("--out",type=Path)
    ap.add_argument("--measured",type=int,default=4)
    ap.add_argument("--execute-gpu",action="store_true")
    args=ap.parse_args(argv);assert 1<=args.measured<=4
    ident=identity(args.package_root.resolve(),args.source_manifest.resolve())
    if not args.execute_gpu:
        assert not any(k=="mlx" or k.startswith("mlx.") for k in sys.modules)
        print(json.dumps({"status":"CPU_READY_ONLY","GPU_executed":False,"identity":ident,"warm_requests":3,"measured_B1_forwards":args.measured}))
        return 0
    assert args.out is not None and args.out.is_absolute()
    assert os.environ.get("DYLD_INSERT_LIBRARIES")==str(LIB)
    expected_env={"MLX_MAX_OPS_PER_BUFFER":"50","MLX_MAX_MB_PER_BUFFER":"50","MLX_BFS_MAX_WIDTH":"20","MLX_METAL_FAST_SYNCH":"0"}
    assert all(os.environ.get(k)==v for k,v in expected_env.items())
    versions={n:importlib.metadata.version(n) for n in ("mlx","mlx-metal")}
    assert versions=={"mlx":"0.32.3","mlx-metal":"0.32.3"}
    args.out.mkdir(exist_ok=False)
    receipt={"status":"STARTED","identity":ident,"versions":versions,"environment":expected_env,"DYLD_INSERT_LIBRARIES":str(LIB),"measured":[],"warm":[],"model_loads":0,"diagnostic_only":True,"performance_point":False,"Pareto_eligible":False,"adopted":False,"ahead_enabled":False}
    history=[];mx=None;native=None;armed=False
    save=lambda:write(args.out/"receipt.json",receipt)
    try:
        # The library must already be injected before model/PSO construction.
        native=ctypes.CDLL(None)
        native.v26_begin.restype=ctypes.c_int
        native.v26_step.argtypes=[ctypes.c_int]
        native.v26_end.argtypes=[ctypes.c_char_p];native.v26_end.restype=ctypes.c_int
        import mlx.core as mx
        sys.path.insert(0,str(args.package_root.resolve()));sys.path.insert(0,str(P))
        import resource_snapshot_power_v1 as resources
        from tutorial_code.miniengine.model.loader import load_model
        from tutorial_code.miniengine.cache import make_cache,DecodeBatch
        from tutorial_code.miniengine.context import ForwardContext
        device=mx.device_info();assert device["device_name"]=="Apple M4 Max" and device["architecture"]=="applegpu_g16s"
        receipt["device_info"]=device
        def probe(label):
            mx.synchronize();raw=resources.snapshot(label,os.getpid())
            write(args.out/("resource-%02d-%s.json"%(len(history),label)),raw)
            history.append(raw);resources.deltas(history);write(args.out/"resources.json",history);resources.guard(history)
            assert raw["pressure_level_value"]==1 and raw["power_metadata"]["source"]=="AC Power" and raw["processinfo_thermal_metadata"]["code"] in (0,1)
            assert not [x for x in raw["known_inference_processes"] if x["pid"]!=os.getpid()]
        probe("beforeload")
        mx.set_cache_limit(2<<30);w=json.loads((P/"workload.json").read_text())
        model,cfg=load_model(w["model_dir"],dtype=mx.bfloat16);receipt["model_loads"]=1
        mx.synchronize();probe("afterload-beforetoken")
        for layer in model.layers:
            if not layer.is_attn:
                layer.linear_attn.set_gates_compile(True)
                if hasattr(layer.linear_attn,"set_conv_silu"):layer.linear_attn.set_conv_silu(False)
        receipt["gates_on"]=True;receipt["conv_off"]=True;receipt["functional_off"]=True
        shapes=[]
        for i,layer in enumerate(model.layers):
            for owner,component in [(layer.mlp,"MLP"),(layer.self_attn if layer.is_attn else layer.linear_attn,"attention" if layer.is_attn else "GDN")]:
                for name in ("gate_proj","up_proj","down_proj","q_proj","k_proj","v_proj","o_proj","in_proj_qkv","in_proj_z","in_proj_b","in_proj_a","out_proj"):
                    op=getattr(owner,name,None)
                    if op is not None and hasattr(op,"in_features"):shapes.append({"layer":i,"name":name,"component":component,"K":op.in_features,"N":op.out_features})
        if hasattr(model,"lm_head"):shapes.append({"layer":None,"name":"lm_head","component":"logits","K":model.lm_head.in_features,"N":model.lm_head.out_features})
        receipt["linear_shapes"]=shapes
        write(args.out/"model-ready.json",{"pid":os.getpid(),"ready":True,"model_loads":1});save()
        def setup():
            caches=make_cache(model)
            logits=model(mx.array(w["prompt_ids"],mx.int32),ForwardContext("prefill",caches),last_only=True)
            batch=DecodeBatch(model);batch.rebuild([],[("profile",caches,128)])
            mx.eval(logits,*batch.arrays());seed=int(mx.argmax(logits,axis=-1).item());assert seed==w["continuation_ids"][0]
            return batch,seed
        def decode(batch,token,i):
            position=len(w["prompt_ids"])+i
            ctx=ForwardContext("decode",batch.layers,positions=mx.array([position],mx.int32),lens=mx.array([position+1],mx.int32),max_len=position+1)
            then=time.perf_counter_ns();logits=model(mx.array([token],mx.int32),ctx);built=time.perf_counter_ns()
            pred=mx.argmax(logits,axis=-1);mx.eval(pred,*batch.arrays());evaluated=time.perf_counter_ns()
            value=int(pred.item());assert value==w["continuation_ids"][i+1]
            return value,{"step":i,"position":position,"id":value,"host_model_graph_call_ns":built-then,"host_eval_and_wait_ns":evaluated-built,"scope":"host wall; eval includes scheduling/Metal encode/GPU wait, not GPU duration"}
        for warm in range(3):
            batch,token=setup()
            for i in range(31):token,_=decode(batch,token,i)
            batch.clear();mx.synchronize();receipt["warm"].append({"request":warm,"outputs":32,"IDs_exact":True});save();probe("warm%d"%warm)
        batch,token=setup();mx.synchronize();probe("before-measured")
        assert native.v26_begin()==0;armed=True
        profiler=cProfile.Profile();profiler.enable()
        for i in range(args.measured):
            native.v26_step(i);token,row=decode(batch,token,i);receipt["measured"].append(row)
        mx.synchronize();profiler.disable();profiler.dump_stats(str(args.out/"host.prof"))
        text=io.StringIO();pstats.Stats(profiler,stream=text).sort_stats("tottime").print_stats(60)
        (args.out/"host-profile.txt").write_text(text.getvalue())
        assert native.v26_end(os.fsencode(args.out/"native.json"))==0;armed=False
        batch.clear();mx.synchronize();probe("after-measured")
        receipt["status"]="CAPTURED_INSTRUMENTED_DIAGNOSTIC_ONLY";save()
        return 0
    except BaseException as error:
        receipt.update(status="FAIL_STOP",error=repr(error),traceback=traceback.format_exc());raise
    finally:
        try:
            if mx is not None:mx.synchronize()
            if armed and native is not None:
                code=native.v26_end(os.fsencode(args.out/"native.json"));receipt["finally_native_end_code"]=code
                if code:receipt["status"]="FAIL_STOP"
            if "probe" in locals():probe("finally")
        except BaseException as error:
            receipt.update(status="FAIL_STOP",finally_error=repr(error));save();raise
        save()

if __name__=="__main__":raise SystemExit(main())
