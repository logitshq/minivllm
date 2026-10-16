"""V44 saved9real/6synthetic strict bytes qualification; CPU inert default."""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
sys.dont_write_bytecode=True
D=Path(__file__).resolve().parent;P=D.parent
BASE=P/"candidate-v17-gdn-gates-build1-root/tutorial_code/miniengine/kernels/gdn_metal.py"
BANKROOT=P/"v18-recurrent-numeric-build1/gpu-recurrent-numeric-AC-run1"
REAL=[f"real-L{l}-D{d}" for l in (0,32,62) for d in (1,16,31)]
SYNTHETIC=["synthetic-zero","synthetic-random-PCG64-18001","synthetic-amplitude","synthetic-cancellation","synthetic-signedzero-subnormal","synthetic-repeated-head-boundary"]
CONTRACT={"u":((1,10240),"bfloat16"),"g":((1,48),"float32"),"beta":((1,48),"float32"),"S":((1,48,128,128),"float32")}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+"\n")
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def identity():
    pins=json.loads((D/"source-pins.json").read_text())
    for path,h in pins.items():assert sha(path)==h,("pinned source changed",path)
    for name in REAL+SYNTHETIC:
        directory=BANKROOT/name;meta=json.loads((directory/"bank.json").read_text())
        for key,(shape,dtype) in CONTRACT.items():
            a=meta["arrays"][key];f=directory/a["file"]
            assert Path(a["file"]).name==a["file"] and tuple(a["shape"])==shape and a["dtype"]==dtype
            assert f.stat().st_size==a["bytes"]==math.prod(shape)*(2 if dtype=="bfloat16" else 4)
            assert sha(f)==a["sha256"],("saved bank binary changed",name,key)
    return pins

def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("--out",type=Path);ap.add_argument("--execute-gpu",action="store_true");args=ap.parse_args(argv)
    pins=identity()
    if not args.execute_gpu:
        assert not any(x=="mlx" or x.startswith(("mlx.","numpy")) for x in sys.modules)
        print(json.dumps({"status":"CPU_READY_ONLY","GPU_executed":False,"saved_real_banks":9,"saved_synthetic_banks":6,"input_binary_SHA_verified":True,"policy":"V19 unchanged math_mode=safe; baseline accepted25 omitted options","fullmodel_qualified":False}));return 0
    assert args.out is not None and args.out.is_absolute();args.out.mkdir(exist_ok=False)
    r={"status":"STARTED","pins":pins,"cases":[],"strict_FP32_raw_bytes":True,"activated_input":"nn.silu(savedRAWu) computed outside candidate; helper performs no SiLU","conv_fusion_qualified":False,"engine_connected":False,"model_loaded":False,"timing_run":False,"compile_policy":"V19 unchanged compile_options={math_mode:safe}; baseline accepted25 default omitted options","output_layout":"canonical S_new[base+d*Dv+dv]; original grid/TG","register_reuse":"intent only; actual allocation/spills/reloads unmeasured"}
    start=time.perf_counter();save=lambda:write(args.out/"receipt.json",r);save()
    try:
        versions={n:importlib.metadata.version(n) for n in ("mlx","mlx-metal")};assert versions=={"mlx":"0.32.3","mlx-metal":"0.32.3"}
        controls={"MLX_MAX_OPS_PER_BUFFER":"50","MLX_MAX_MB_PER_BUFFER":"50","MLX_BFS_MAX_WIDTH":"20","MLX_METAL_FAST_SYNCH":"0"};assert all(os.environ.get(k)==v for k,v in controls.items())
        import mlx.core as mx
        import mlx.nn as nn
        import numpy as np
        info=mx.device_info();assert mx.default_device()==mx.gpu and info["device_name"]=="Apple M4 Max" and info["architecture"]=="applegpu_g16s"
        mx.set_memory_limit(512<<20);mx.set_cache_limit(64<<20)
        r["runtime"]={"versions":versions,"device":info,"controls":controls,"memory_limit":512<<20,"cache_limit":64<<20}
        baseline=module("v44_original",BASE);candidate=module("v44_candidate",D/"gdn_activated_norm_reuse_cached_metal.py")
        def raw(a):
            mx.eval(a)
            if a.dtype==mx.bfloat16:
                h=np.ascontiguousarray(np.asarray(a.view(mx.uint16)),dtype="<u2");assert not np.any((h&0x7f80)==0x7f80)
            else:
                assert a.dtype==mx.float32;h=np.ascontiguousarray(np.asarray(a),dtype="<f4");assert np.isfinite(h).all()
            data=h.tobytes();return data,{"shape":list(a.shape),"dtype":str(a.dtype),"bytes":len(data),"sha256":hashlib.sha256(data).hexdigest()}
        def signatures(values):return {key:raw(a)[1] for key,a in values.items()}
        for name in REAL+SYNTHETIC:
            directory=BANKROOT/name;meta=json.loads((directory/"bank.json").read_text());bank={};hosts=[]
            for key,(shape,dtype) in CONTRACT.items():
                a=meta["arrays"][key];h=np.fromfile(directory/a["file"],dtype="<u2" if dtype=="bfloat16" else "<f4").reshape(shape);hosts.append(h)
                value=mx.array(h);bank[key]=value.view(mx.bfloat16) if dtype=="bfloat16" else value
            act=nn.silu(bank["u"]);q,k,v=mx.split(act,[2048,4096],axis=-1)
            q=mx.repeat(q.reshape(1,16,128),3,axis=1);k=mx.repeat(k.reshape(1,16,128),3,axis=1);v=v.reshape(1,48,128)
            inputs=dict(bank,activated=act,q=q,k=k,v=v);before=signatures(inputs)
            A=baseline.gated_delta_step_metal(q,k,v,bank["g"],bank["beta"],bank["S"])
            B=candidate.gated_delta_step_activated_norm_reuse_metal(act,bank["g"],bank["beta"],bank["S"])
            mx.eval(*A,*B)
            after=signatures(inputs);row={"bank":name,"kind":meta["kind"],"bank_sha256":sha(directory/"bank.json"),"historical_saved_input_not_new_capture":True,"input_before":before,"input_after":after,"input_readonly":before==after,"outputs":[]}
            r["cases"].append(row);save()
            assert before==after,("input mutation",name)
            for key,a,b in zip(("o","S_new"),A,B):
                assert a.shape==b.shape and a.dtype==b.dtype==mx.float32
                da,sa=raw(a);db,sb=raw(b);equal=da==db
                row["outputs"].append({"name":key,"strict_equal":equal,"A":sa,"B":sb});save()
                if not equal:
                    (args.out/"first-failure-A.bin").write_bytes(da);(args.out/"first-failure-B.bin").write_bytes(db)
                    r["first_failure"]={"bank":name,"output":key,"A":sa,"B":sb,"first_byte_offset":next(i for i,(x,y) in enumerate(zip(da,db)) if x!=y)};save();raise AssertionError("strict raw bytes differ")
            del A,B,bank,inputs,hosts
        assert len(r["cases"])==15 and all(x["input_readonly"] and len(x["outputs"])==2 and all(y["strict_equal"] for y in x["outputs"]) for x in r["cases"])
        r.update(status="PASS_V44_ACTIVATED_NORMREUSE_CACHE_STRICT_BYTES_9REAL_6SYNTHETIC",GPU_numeric_run=True,strict_output_checks=30,fullmodel_qualified=False);return 0
    except BaseException as error:
        r.update(status="FAIL_STOP",error=repr(error),traceback=traceback.format_exc());raise
    finally:
        r["wall_seconds"]=time.perf_counter()-start;save()

if __name__=="__main__":raise SystemExit(main())
