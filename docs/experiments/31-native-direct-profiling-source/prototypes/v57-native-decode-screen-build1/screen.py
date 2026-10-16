"""Root-only B-only native exploratory timing; inert never imports MLX."""
import argparse
from contextlib import contextmanager
import ctypes
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys
import time
import traceback

sys.dont_write_bytecode = True
D = Path(__file__).resolve().parent
P = D.parent
ROOT = P / "candidate-v44-conv-normreuse-build1-root"
LIB = P / "v57-native-direct-feedback-build1/libv57replay.dylib"
QUAL = P / "v23-root-control-build1/fullmodel-v57native-attempt1/child-output/receipt.json"
QUAL_SHA = "2a62473708e5e2d2113d603b0348fc8760d230534bd53f5d9856ea99851c2c73"
CONTROLS = {"MLX_MAX_OPS_PER_BUFFER": "50", "MLX_MAX_MB_PER_BUFFER": "50",
            "MLX_BFS_MAX_WIDTH": "20", "MLX_METAL_FAST_SYNCH": "0"}


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value): Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def module(name, path):
    spec=importlib.util.spec_from_file_location(name,path); value=importlib.util.module_from_spec(spec)
    sys.modules[name]=value; spec.loader.exec_module(value); return value


def source_check():
    pins=json.loads((D/"source-pins.json").read_text())
    for path,digest in pins.items(): assert sha(path)==digest,("source changed",path)
    assert sha(QUAL)==QUAL_SHA
    release=QUAL.parent.parent/"external-release.json"
    assert sha(release)=="d520151aca17257bfdb7a3f2437c5e3a9429fe0fef12538bafba1cab46378771"
    r=json.loads(release.read_text());assert r["status"]=="PASS_EXTERNAL_RELEASE" and r["GPU_released"] and r["tool_exit"]==0
    assert all(x["returncode"]==1 and x["stdout"]==x["stderr"]=="" for x in r["strict_absence"])
    q=json.loads(QUAL.read_text()); assert q["status"]=="PASS_NATIVE_SYNC_DIRECT_FEEDBACK_CONV_ACTIVATED_NORMREUSE_CACHE_ICB_BASE_RESET_CHANGED_VS_ACCEPTED25_STRICT"
    assert q["native_steps"]==93 and len(q["requests"])==3
    assert q["A_saved_baseline_receipt_sha256"]=="c57e7625baa5a4ce450fc7306233e766f1efd71636713080d748780624542c8a"
    engagement=q["B_conv_normreuse_cache_capture_engagement"]
    assert engagement["calls"]==engagement["distinct_GDN_objects"]==48 and engagement["counter_removed_after_capture"] is True
    base,reset=q["requests"][:2]
    assert base["name"]=="base" and reset["name"]=="reset-base" and len(base["steps"])==len(reset["steps"])==31
    expected=base["steps"][-1]["outputs"]; assert len(expected)==132
    assert all(x["strict_bytes_equal"] and x["finite"] for x in expected)
    assert [(x["label"],x["A_sha256"]) for x in expected]==[(x["label"],x["A_sha256"]) for x in reset["steps"][-1]["outputs"]]
    assert q["native_library"]==str(LIB) and q["source_pins"][str(LIB)]==pins[str(LIB)]
    manifest=json.loads((P/"v44-native-fullqual-build1/candidate-manifest.json").read_text())["files"]
    expected_source={k.removeprefix("tutorial_code/miniengine/"):v for k,v in manifest.items()}
    package=ROOT/"tutorial_code/miniengine"
    assert {str(f.relative_to(package)):sha(f) for f in package.rglob("*.py")}==expected_source and len(expected_source)==27
    return pins,q,expected


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out",type=Path);ap.add_argument("--tokens",type=int,default=32);ap.add_argument("--inert",action="store_true")
    args=ap.parse_args(argv);pins,qualification,expected_final=source_check()
    w=json.loads((P/"workload.json").read_text());assert args.tokens==32 and len(w["prompt_ids"])==19
    if args.inert:
        assert not any(n=="mlx" or n.startswith("mlx.") for n in sys.modules)
        print(json.dumps({"status":"PASS_CPU_INERT","pins":len(pins),"MLX_imported":False,"GPU_used":False,
                          "requests":6,"warm":3,"measure":3,"qualification_sha256":QUAL_SHA}));return 0
    assert args.out is not None and args.out.is_absolute();args.out.mkdir(exist_ok=False)
    receipt={"status":"STARTED","stage":"beforeload","pid":os.getpid(),"model_loads":0,"source_pins":pins,
             "qualification_receipt":str(QUAL),"qualification_sha256":QUAL_SHA,"requests":[],
             "scope":"B-only exploratory native timing; no HTTP/Pareto/fair A comparison/adoption",
             "schedule":["warm"]*3+["measure"]*3,"fixed_cap":256,"expected_decode_steps":31,
             "timing_definition":{"t0":"immediately before fresh prefill",
                                  "t1":"after prefill first-token/flat-seed eval, native reset and all131 seed uploads",
                                  "t2":"after31 native decode steps and4B prediction observation per step",
                                  "decode":"31/(t2-t1)","whole_request":"32/(t2-t0)",
                                  "excluded_from_timed_decode":"A/model calls, state copy-to-CPU/hash/compare, resources/logging/file writes"}}
    save=lambda:write(args.out/"receipt.json",receipt)
    save();mx=None;resources=None;loaded=False;native=None;armed=False
    parameter_holders=[];seed_holders=[];output_holders=[]
    try:
        assert os.environ.get("DYLD_INSERT_LIBRARIES")==str(LIB)
        assert all(os.environ.get(k)==v for k,v in CONTROLS.items())
        versions={n:importlib.metadata.version(n) for n in ("mlx","mlx-metal")}
        assert versions=={"mlx":"0.32.3","mlx-metal":"0.32.3"}
        support=module("v35_resources",P/"v24-fullmodel-qual-build1/fullqual.py")
        base=module("v35_native_support",P/"v44-native-fullqual-build1/fullqual.py")
        prior=module("v35_shared",P/"v33-fixed-buffer-replay-worker-build3/fullqual.py")
        resources=support.Resources(args.out)
        def sample(label):
            result=resources.sample(label,loaded=loaded,mx=mx)
            current=resources.history[-1]
            assert current["processinfo_thermal_metadata"]["code"]==0,"thermal must be nominal0"
            assert current["pressure_level_value"]==1
            if len(resources.history)>1:
                previous=resources.history[-2]
                assert current["swap_used_mb"]<=previous["swap_used_mb"],"strict swap growth STOP"
                assert current["vm_counters"]["Swapouts"]==previous["vm_counters"]["Swapouts"],"strict Swapouts growth/reset STOP"
            return result
        sample("beforeload");native=base.Native(args.out)
        assert not any(n=="mlx" or n.startswith(("mlx.","tutorial_code.")) for n in sys.modules)
        import mlx.core as mx
        import numpy as np
        from mlx.utils import tree_flatten
        info=mx.device_info();assert mx.default_device()==mx.gpu and info["device_name"]=="Apple M4 Max" and info["architecture"]=="applegpu_g16s"
        mx.set_cache_limit(2<<30)
        receipt["runtime"]={"versions":versions,"device":info,"cache_limit_bytes":2<<30,"environment":{**CONTROLS,"DYLD_INSERT_LIBRARIES":str(LIB)}}
        sys.path.insert(0,str(ROOT))
        from tutorial_code.miniengine.model.loader import load_model
        from tutorial_code.miniengine.cache import BatchKV,BatchState,DecodeBatch,make_cache
        from tutorial_code.miniengine.context import ForwardContext
        model,cfg=load_model(w["model_dir"],dtype=mx.bfloat16);loaded=True;receipt["model_loads"]=1;mx.synchronize()
        receipt["afterload"]=sample("afterload-before-any-token")
        gdns=[l.linear_attn for l in model.layers if not l.is_attn];assert len(model.layers)==64 and len(gdns)==48
        for g in gdns:
            g.set_gates_compile(True)
            g.set_conv_norm_reuse_cache(True)
        from tutorial_code.miniengine.model.gdn import GatedDeltaNet as GDN
        candidate_norm=GDN._decode_conv_norm_reuse_cache
        norm_capture_calls=[]
        @contextmanager
        def capture_norm_engagement():
            assert GDN._decode_conv_norm_reuse_cache is candidate_norm
            def counted(self,*args,**kwargs):
                norm_capture_calls.append(id(self))
                return candidate_norm(self,*args,**kwargs)
            GDN._decode_conv_norm_reuse_cache=counted
            try: yield
            finally: GDN._decode_conv_norm_reuse_cache=candidate_norm
        receipt.update(stage="ready",gates_on=48,conv_on=48,conv_off=False,normreuse_cache_on=48,whole_functional_off=True)
        write(args.out/"model-ready.json",{"pid":os.getpid(),"token_executed":False,"model_loads":1});save()
        print(json.dumps({"event":"MODEL_READY","pid":os.getpid()}),flush=True)
        def shared(v,label):return prior.Shared(mx,np,v,label)
        def layers(flat):return [BatchKV(flat[2*i][:],flat[2*i+1][:]) if l.is_attn else BatchState(flat[2*i],flat[2*i+1]) for i,l in enumerate(model.layers)]
        def states(ctx):return tuple(v for l,c in zip(model.layers,ctx.caches) for v in ((c.K,c.V) if l.is_attn else (c.conv,c.S)))
        def setup():
            caches=make_cache(model)
            logits=model(mx.array(w["prompt_ids"],mx.int32),ForwardContext("prefill",caches),last_only=True)
            batch=DecodeBatch(model);batch.rebuild([],[("v35",caches,51)])
            flat=tuple(v[:] for v in batch.arrays());assert len(flat)==128
            mx.eval(logits,*flat);first=int(mx.argmax(logits,axis=-1).item())
            seed=(mx.array([first],mx.int32),mx.array([19],mx.int32),mx.array([20],mx.int32),*flat)
            mx.eval(*seed);mx.synchronize();return seed,first
        # Same untimed initialization as qualified V34: one ordinary32 warm.
        warm,token=setup();flat=warm[3:];warm_ids=[token]
        for i in range(31):
            position=19+i
            ctx=ForwardContext("decode",layers(flat),positions=mx.array([position],mx.int32),lens=mx.array([position+1],mx.int32),max_len=position+1)
            logits=model(mx.array([token],mx.int32),ctx);flat=states(ctx)
            pred=mx.argmax(logits,axis=-1);mx.eval(pred,*flat);token=int(pred.item());warm_ids.append(token)
        assert warm_ids==w["continuation_ids"];del warm,flat,ctx,logits,pred
        mx.synchronize();sample("after-initial-warm32")
        capture_seed,first=setup();assert first==w["continuation_ids"][0]
        for name,v in tree_flatten(model.parameters()):
            h=shared(v,"parameter."+name);parameter_holders.append(h);native.call("v33_pin",h.address,h.nbytes,h.label.encode())
        assert len(parameter_holders)==1349
        input_labels=("tokens","positions","lens")+tuple(f"state.{i}" for i in range(128))
        output_labels=("logits",)+tuple(f"state.{i}" for i in range(128))+("prediction","next_positions","next_lens")
        seed_holders=[shared(v,n) for v,n in zip(capture_seed,input_labels)]
        ctx=ForwardContext("decode",layers(capture_seed[3:]),positions=capture_seed[1],lens=capture_seed[2],max_len=256)
        mx.eval(*(v for c,l in zip(ctx.caches,model.layers) for v in ((c.K,c.V) if l.is_attn else (c.conv,c.S))))
        mx.synchronize();sample("before-capture")
        native.call("v33_begin");armed=True
        with capture_norm_engagement():
            logits=model(capture_seed[0],ctx)
            captured=(logits,*states(ctx),mx.argmax(logits,axis=-1).astype(mx.int32),capture_seed[1]+1,capture_seed[2]+1)
            mx.eval(*captured);mx.synchronize()
        native.call("v33_end_build");armed=False
        assert len(norm_capture_calls)==48 and set(norm_capture_calls)==set(map(id,gdns))
        assert GDN._decode_conv_norm_reuse_cache is candidate_norm
        receipt["B_conv_normreuse_cache_capture_engagement"]={"calls":48,"distinct_GDN_objects":48,"counter_removed_after_capture":True,"activated_helper":True,"conv_fusion_ON":True,"no_double_SiLU":True}
        assert len(captured)==132
        for v,label in zip(captured,output_labels):
            h=shared(v,label);output_holders.append(h);native.call("v33_register_output",h.address,h.nbytes,label.encode())
        for i,h in enumerate(seed_holders):native.call("v34_register_input",h.address,h.nbytes,(1,2,3)[i] if i<3 else 0,h.label.encode())
        for i,h in enumerate(seed_holders):
            other=output_holders[(129,130,131)[i] if i<3 else i-2];assert h.nbytes==other.nbytes
            native.call("v34_link",h.address,other.address,h.nbytes)
        pingpong_requested=[]
        for layer_index,layer in enumerate(model.layers):
            if layer.is_attn: continue
            input_state=seed_holders[4+2*layer_index]
            output_state=output_holders[2+2*layer_index]
            assert input_state.meta["shape"]==output_state.meta["shape"]==[1,48,128,128]
            assert input_state.meta["dtype"]==output_state.meta["dtype"]=="mlx.core.float32"
            assert input_state.nbytes==output_state.nbytes==3145728
            native.call("v36_register_pingpong",input_state.address,output_state.address,input_state.nbytes)
            pingpong_requested.append(layer_index)
        assert len(pingpong_requested)==48
        receipt["pingpong_layers_requested"]=pingpong_requested
        native.call("v34_seal_feedback",128,256)
        native.call("v38_enable_icb")
        native.lib.v57_enable_direct_feedback.argtypes=[];native.lib.v57_enable_direct_feedback.restype=ctypes.c_int
        native.call("v57_enable_direct_feedback")
        receipt["capture_native"]=native.report("native-sealed")
        base.assert_icb(receipt["capture_native"],[])
        f=receipt["capture_native"]["feedback"];assert f["sealed"] and f["input_count"]==131 and f["output_count"]==132 and f["link_count"]==131
        pp=f["pingpong"];assert (pp["requested_count"],pp["eligible_count"],pp["staged_link_count"])==(48,48,83)
        for h,expected in zip(output_holders,expected_final):
            assert h.label==expected["label"] and h.meta["shape"]==expected["shape"] and h.meta["dtype"]==expected["dtype"] and h.nbytes==expected["bytes"]
        sample("after-capture");save()
        step_fn=native.lib.v34_replay_step;copy_fn=native.lib.v34_copy_output
        timing_fn=native.lib.v36_timing_json;timing_fn.argtypes=[];timing_fn.restype=ctypes.c_char_p
        prediction=ctypes.c_int32();prediction_ptr=ctypes.byref(prediction);prediction_address=output_holders[129].address
        for index,phase in enumerate(receipt["schedule"]):
            before=sample(f"request-{index+1}-before")
            t0=time.perf_counter_ns()
            request_seed,first=setup()
            request_holders=[shared(v,n) for v,n in zip(request_seed,input_labels)]
            native.call("v34_reset")
            for original,current in zip(seed_holders,request_holders):
                assert original.nbytes==current.nbytes and original.meta["shape"]==current.meta["shape"] and original.meta["dtype"]==current.meta["dtype"]
                native.call("v34_write_input",original.address,original.nbytes,ctypes.c_void_p(current.address))
            ids=[first]
            t1=time.perf_counter_ns()
            # Timed loop: native graph+GPUfeedback,4B observation,append only.
            for _ in range(31):
                rc=step_fn()
                if rc:raise RuntimeError("native step failure")
                rc=copy_fn(prediction_address,4,prediction_ptr)
                if rc:raise RuntimeError("native prediction observation failure")
                ids.append(prediction.value)
            t2=time.perf_counter_ns()
            after=sample(f"request-{index+1}-after")
            assert ids==w["continuation_ids"],"canonical IDs differ"
            checks=[]
            for h,expected in zip(output_holders,expected_final):
                target=ctypes.create_string_buffer(h.nbytes)
                native.call("v34_copy_output",h.address,h.nbytes,ctypes.cast(target,ctypes.c_void_p))
                digest=hashlib.sha256(memoryview(target).cast("B")).hexdigest()
                assert digest==expected["A_sha256"],("final staged output differs from strict qualification",h.label)
                checks.append({"label":h.label,"bytes":h.nbytes,"sha256":digest,"qualified_A_sha256":expected["A_sha256"]})
            report=native.report(f"native-request-{index+1}");f=report["feedback"]
            base.assert_icb(report,[31]*(index+1))
            direct=f["direct_feedback"]
            assert direct["enabled"] and direct["last_result_valid"] and not direct["capture_stage_encoder_used"]
            assert direct["direct_links"]==83 and direct["direct_bytes_per_step"]==19726348
            assert direct["banks_checked"]==2 and direct["input_output_range_pairs_checked"]==34584
            assert f["GPU_blit_bytes_per_step"]==19726348
            assert f["request_steps"]==31 and f["total_steps"]==(index+1)*31 and f["resets"]==index+1
            assert f["request_seed_uploaded_ranges"]==131 and f["CPU_seed_upload_bytes_total"]==sum(h.nbytes for h in seed_holders)*(index+1)
            assert f["CPU_output_compare_bytes_total"]==0 and f["CPU_full_state_restore_per_step"] is False
            pp=f["pingpong"];assert (pp["requested_count"],pp["eligible_count"],pp["staged_link_count"])==(48,48,83)
            # Export exactly once per request, after timing and final output hashes.
            timing_raw=timing_fn();assert timing_raw is not None
            timing_path=args.out/f"native-timing-request-{index+1:02d}.json"
            timing_path.write_bytes(timing_raw+b"\n")
            timing_data=json.loads(timing_raw)
            assert timing_data["record_count"]==(index+1)*31 and f["timing_records_stored"]==(index+1)*31
            current_rows=[r for r in timing_data["records"] if r["request"]==index+1]
            assert len(current_rows)==31 and [r["step"] for r in current_rows]==list(range(1,32))
            row={"index":index+1,"phase":phase,"ids":ids,"t0_ns":t0,"t1_ns":t1,"t2_ns":t2,
                 "prefill_seed_ready_ns":t1-t0,"decode31_ns":t2-t1,"whole32_ns":t2-t0,
                 "decode_ns_per_step":(t2-t1)/31,"decode31_tokens_per_s":31e9/(t2-t1),"whole32_tokens_per_s":32e9/(t2-t0),
                 "resources":{"before":before,"after":after},"final132_hash_checks":checks,
                 "final_hash_cpu_copy_bytes_outside_timing":sum(h.nbytes for h in output_holders),"native":report,
                 "native_timing":{"raw_file":timing_path.name,"raw_sha256":sha(timing_path),
                                  "current_request_records":31,"total_records":timing_data["record_count"],
                                  "scope":"request-after native spans and whole CB GPU envelope; not exclusive kernels"}}
            receipt["requests"].append(row);receipt["stage"]={"request":index+1,"phase":phase,"complete":True};save()
            print(json.dumps({"event":"B_ONLY_EXPLORATORY","request":index+1,"phase":phase,"decode31_ns":row["decode31_ns"],
                              "decode_ns_per_step":row["decode_ns_per_step"],"decode31_tokens_per_s":row["decode31_tokens_per_s"],
                              "whole32_ns":row["whole32_ns"],"whole32_tokens_per_s":row["whole32_tokens_per_s"]}),flush=True)
        measured=[r for r in receipt["requests"] if r["phase"]=="measure"]
        receipt["measured_summary"]={"n":3,"median_decode_ns_per_step":statistics.median(r["decode_ns_per_step"] for r in measured),
                                     "median_decode31_tokens_per_s":statistics.median(r["decode31_tokens_per_s"] for r in measured),
                                     "median_whole32_tokens_per_s":statistics.median(r["whole32_tokens_per_s"] for r in measured),
                                     "scope":"B-only exploratory absolute timing; not HTTP/Pareto or an A/B performance win"}
        assert all(g._gates_compile_enabled and callable(g._compiled_decode_gates) for g in gdns)
        assert all(g._conv_norm_reuse_cache_enabled is True for g in gdns)
        assert GDN._decode_conv_norm_reuse_cache is candidate_norm
        for name,mod in tuple(sys.modules.items()):
            if name.startswith("tutorial_code.miniengine") and getattr(mod,"__file__",None):
                path=str(Path(mod.__file__).resolve());assert path in pins and sha(path)==pins[path]
        source_check();receipt.update(status="PASS_SYNC_DIRECT_FEEDBACK_CONV_ACTIVATED_NORMREUSE_CACHE_ICB_S_PINGPONG_B_ONLY_EXPLORATORY_SCREEN",stage="complete");save();return 0
    except BaseException as error:
        receipt.update(status="FAIL_STOP",error=repr(error),traceback=traceback.format_exc(),capture_armed=armed)
        if native is not None:
            try:native.report("native-on-failure")
            except BaseException as report_error:receipt["native_report_error"]=repr(report_error)
        save();raise
    finally:
        try:
            if mx is not None:mx.synchronize()
            if resources is not None:receipt["finally_resource"]=sample("finally-before-exit")
        except BaseException as error:
            receipt.update(status="FAIL_STOP",finally_error=repr(error));save();raise
        save()


if __name__=="__main__":raise SystemExit(main())
