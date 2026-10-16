"""Root-only V100 vendor plainRMS leaf physicalCAP64 B-only raw-native exploratory timing; inert never imports MLX."""
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
ROOT = P / "candidate-v92-vendor-rms-leaf-build1-root"
LIB = P / "v89-native-capacity64-build1/libv89replay.dylib"
QUAL = P / "v23-root-control-build1/fullmodel-v97native-attempt1/child-output/receipt.json"
QUAL_SHA = "6c9621109e8f23e20b88244324b7b104e7672ca67a89342af52c4ebfe0b65947"
CAP = 64
RELEASE = P / "v23-root-control-build1/fullmodel-v97native-attempt1/external-release.json"
RELEASE_SHA = "803c1fe49aac490e36afe985d83a1ffbe481745ae86a0b6918e3d264b84fede3"
INDEPENDENT_REVIEW97 = P / "v97-independent-results-review-build1/verification.json"
INDEPENDENT_REVIEW97_SHA = "3c568203af0a045cb611e088765f54c41d73ead09fb85a2a4e9b7a9dfa0239aa"
CONTROLS = {"MLX_MAX_OPS_PER_BUFFER": "50", "MLX_MAX_MB_PER_BUFFER": "50",
            "MLX_BFS_MAX_WIDTH": "20", "MLX_METAL_FAST_SYNCH": "0"}


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value): Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def module(name, path):
    spec=importlib.util.spec_from_file_location(name,path); value=importlib.util.module_from_spec(spec)
    sys.modules[name]=value; spec.loader.exec_module(value); return value


def source_check():
    assert INDEPENDENT_REVIEW97 is not None and INDEPENDENT_REVIEW97_SHA is not None,"HOLD_INDEPENDENT97_SAVED_RESULT_REVIEW_PENDING"
    pins=json.loads((D/"source-pins.json").read_text())
    for path,digest in pins.items():assert sha(path)==digest,("source changed",path)
    assert sha(INDEPENDENT_REVIEW97)==INDEPENDENT_REVIEW97_SHA
    independent=json.loads(INDEPENDENT_REVIEW97.read_text())
    assert independent["status"]=="PASS_INDEPENDENT_SAVED_RESULT_AUDIT_NO_ADOPTION"
    assert independent["receipt_SHA256"]==QUAL_SHA and independent["old_profile_PASS"] is False
    assert independent["prefill_seed_checks"]==393 and independent["prefill_logits_checks"]==3
    assert independent["gamma_readonly_arrays"]==322 and independent["native_cached_gamma_fullrange_readonly"]==161
    assert sha(QUAL)==QUAL_SHA and sha(RELEASE)==RELEASE_SHA
    q=json.loads(QUAL.read_text());release=json.loads(RELEASE.read_text())
    assert q["status"]=="DIAGNOSTIC_COMPLETED_NEW_REFERENCE_STRICT_NO_ADOPTION"
    assert q["old_profile_pass"] is False and q["adopted"] is False and q["performance_point"] is False
    assert q["native_steps"]==93 and q["strict_output_checks"]==12276 and q["canonical_KV_tail_zero_checks"]==2976
    assert q["fixed_cap"]==64 and q["canonical_capacity"]==256 and q["allocation_bucket_restored_finally"]
    assert q["same_parameter_identity_count"]==1349 and q["same_parameter_identity_preserved"]
    assert q["independent_prefill_summary"]["seed131_strict_checks"]==393 and q["independent_prefill_summary"]["last_logits_strict_checks"]==3
    assert q["gamma322_readonly_final"]["arrays"]==322 and q["gamma322_readonly_final"]["strict_bytes_equal"]
    assert len(q["derived_cachedgamma161_native_readonly_proof"])==161
    assert q["native_parameter_pins"]["total_pins"]==1510
    assert release["status"]=="PASS_EXTERNAL_RELEASE" and release["GPU_released"] and release["tool_exit"]==0
    assert all(x["returncode"]==1 and x["stdout"]==x["stderr"]=="" for x in release["strict_absence"])
    prior_pins=json.loads((D/"qualification-source-pins.json").read_text())
    assert q["source_pins"]==prior_pins and len(prior_pins)==142
    assert all(pins[path]==digest for path,digest in prior_pins.items())
    engagement=q["B_conv_normreuse_cache_capture_engagement"]
    assert engagement["calls"]==engagement["distinct_GDN_objects"]==48 and engagement["counter_removed_after_capture"] is True
    plain=q["B_vendor_plain_rms_capture_engagement"]
    assert plain["calls"]==plain["distinct_plain_norm_objects"]==161 and plain["all161_prepared_enabled"] and plain["counter_removed_before_replay"]
    base,reset=q["requests"][:2]
    assert base["name"]=="base" and reset["name"]=="reset-base" and len(base["steps"])==len(reset["steps"])==31
    expected_ids=base["A_ids"]
    assert expected_ids==base["B_ids"]==reset["A_ids"]==reset["B_ids"]==q["ordinary_warm32_diagnostic"]["new_B_ids"]
    assert len(expected_ids)==32 and all(t not in q["eos_ids"] for t in expected_ids)
    expected=base["steps"][-1]["outputs"];assert len(expected)==132
    assert all(x["strict_bytes_equal"] and x["finite"] and x["canonical_B_sha256"]==x["A_sha256"] for x in expected)
    assert sum(x["physical_tail_zero_checked"] for x in expected)==32
    assert [(x["label"],x["A_sha256"]) for x in expected]==[(x["label"],x["A_sha256"]) for x in reset["steps"][-1]["outputs"]]
    assert q["native_library"]==str(LIB) and q["source_pins"][str(LIB)]==pins[str(LIB)]
    manifest=json.loads((P/"v92-vendor-rms-leaf-build1/candidate-manifest.json").read_text())["files"]
    expected_source={k.removeprefix("tutorial_code/miniengine/"):v for k,v in manifest.items()}
    package=ROOT/"tutorial_code/miniengine"
    assert {str(f.relative_to(package)):sha(f) for f in package.rglob("*.py")}==expected_source and len(expected_source)==27
    return pins,q,expected


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out",type=Path);ap.add_argument("--tokens",type=int,default=32);ap.add_argument("--inert",action="store_true")
    args=ap.parse_args(argv);pins,qualification,expected_final=source_check()
    w=json.loads((P/"workload.json").read_text());assert args.tokens==32 and len(w["prompt_ids"])==19
    expected_ids=qualification["requests"][0]["A_ids"]
    if args.inert:
        assert not any(n=="mlx" or n.startswith("mlx.") for n in sys.modules)
        print(json.dumps({"status":"PASS_CPU_INERT","pins":len(pins),"MLX_imported":False,"GPU_used":False,
                          "requests":6,"warm":3,"measure":3,"qualification_sha256":QUAL_SHA}));return 0
    assert args.out is not None and args.out.is_absolute();args.out.mkdir(exist_ok=False)
    receipt={"status":"STARTED","stage":"beforeload","pid":os.getpid(),"model_loads":0,"source_pins":pins,
             "qualification_receipt":str(QUAL),"qualification_sha256":QUAL_SHA,"qualification_release_sha256":RELEASE_SHA,"independent97_review_sha256":INDEPENDENT_REVIEW97_SHA,"profile":"vendor RMSNorm leaf rounding","old_profile_pass":False,"whole_omlx_semantics":False,"adopted":False,"HTTP_or_Pareto_point":False,"fair_AB_win_claim":False,"expected_new_profile_raw_ids":expected_ids,"requests":[],
             "scope":"another vendor plainnorm leaf profile B-only exploration; no oldprofile PASS/wholeOMLX/HTTP/Pareto/fair A comparison/adoption",
             "schedule":["warm"]*3+["measure"]*3,"fixed_cap":CAP,"CPU_canonical_cap":256,"expected_decode_steps":31,
             "timing_definition":{"t0":"immediately before fresh prefill",
                                  "t1":"after prefill first-token/flat-seed eval, native reset and all131 seed uploads",
                                  "t2":"after31 native decode steps and4B prediction observation per step",
                                  "decode":"31/(t2-t1)","whole_request":"32/(t2-t0)",
                                  "excluded_from_timed_decode":"A/model calls, state copy-to-CPU/hash/compare, resources/logging/file writes"}}
    save=lambda:write(args.out/"receipt.json",receipt)
    save();mx=None;resources=None;loaded=False;native=None;armed=False
    parameter_holders=[];seed_holders=[];output_holders=[]
    allocation_classes=None;original_allocation_steps=None
    try:
        assert os.environ.get("DYLD_INSERT_LIBRARIES")==str(LIB)
        assert all(os.environ.get(k)==v for k,v in CONTROLS.items())
        versions={n:importlib.metadata.version(n) for n in ("mlx","mlx-metal")}
        assert versions=={"mlx":"0.32.3","mlx-metal":"0.32.3"}
        support=module("v35_resources",P/"v24-fullmodel-qual-build1/fullqual.py")
        base=module("v35_native_support",P/"v89-cap64-native-fullqual-build1/fullqual.py")
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
        import mlx.nn as nn
        from mlx.utils import tree_flatten
        info=mx.device_info();assert mx.default_device()==mx.gpu and info["device_name"]=="Apple M4 Max" and info["architecture"]=="applegpu_g16s"
        mx.set_cache_limit(2<<30)
        receipt["runtime"]={"versions":versions,"device":info,"cache_limit_bytes":2<<30,"environment":{**CONTROLS,"DYLD_INSERT_LIBRARIES":str(LIB)}}
        sys.path.insert(0,str(ROOT))
        from tutorial_code.miniengine.model.loader import load_model
        from tutorial_code.miniengine.cache import AttnCache,BatchKV,BatchState,DecodeBatch,make_cache
        from tutorial_code.miniengine.context import ForwardContext
        model,cfg=load_model(w["model_dir"],dtype=mx.bfloat16);loaded=True;receipt["model_loads"]=1;mx.synchronize()
        receipt["afterload"]=sample("afterload-before-any-token")
        initial_parameter_identity=[(name,id(v)) for name,v in tree_flatten(model.parameters())]
        allocation_classes=(AttnCache,DecodeBatch);original_allocation_steps=(AttnCache.step,DecodeBatch.step)
        assert original_allocation_steps==(256,256)
        AttnCache.step=CAP;DecodeBatch.step=CAP
        kv_state_indices={2*i+j for i,layer in enumerate(model.layers) if layer.is_attn for j in (0,1)}
        kv_output_labels={f"state.{i}" for i in kv_state_indices};assert len(kv_output_labels)==32
        receipt["allocation_bucket"]={"original":list(original_allocation_steps),"effective":[64,64],"before_any_token":True,"source_files_unchanged":True}
        gdns=[l.linear_attn for l in model.layers if not l.is_attn];assert len(model.layers)==64 and len(gdns)==48
        for g in gdns:
            g.set_gates_compile(True)
            g.set_conv_norm_reuse_cache(True)
        from tutorial_code.miniengine.model.gdn import GatedDeltaNet as GDN
        candidate_norm=GDN._decode_conv_norm_reuse_cache
        # Only plain RMSNorm changes profile; GatedRMSNorm/GDN math stays current44 in BOTH arms.
        vendor=module("v100_vendor_reference",P/"v92-vendor-rms-leaf-build1/vendor_reference.py")
        profile=vendor.prepare_model(model,mx=mx,np=np,nn=nn,device_info=info,expected_parameter_count=1349)
        assert len(profile.entries)==161
        profile.validate_current_bindings()
        from tutorial_code.miniengine.model.layers import RMSNorm as PlainRMSNorm
        plain_norms={id(e.norm) for e in profile.entries};assert len(plain_norms)==161
        plain_names={id(e.norm):e.name for e in profile.entries}
        assert all(type(e.norm) is PlainRMSNorm for e in profile.entries)
        candidate_plain=PlainRMSNorm.__call__
        gamma_holders=[];cached_gamma_holders=[];gamma_checks=[]
        for e in profile.entries:
            assert e.norm.weight is e.source_weight
            assert e.raw_gamma is not e.cached_gamma and e.reference.weight is e.raw_gamma
            Araw=prior.Shared(mx,np,e.raw_gamma,"reference.rawgamma."+e.name)
            Bcached=prior.Shared(mx,np,e.cached_gamma,"derived.cachedgamma."+e.name)
            assert Araw.nbytes==Bcached.nbytes==e.nbytes
            assert Araw.address+Araw.nbytes<=Bcached.address or Bcached.address+Bcached.nbytes<=Araw.address,"independent gamma ranges overlap"
            assert Araw.meta["shape"]==Bcached.meta["shape"] and Araw.meta["dtype"]==Bcached.meta["dtype"]
            assert native.lib.memcmp(Araw.address,Bcached.address,Araw.nbytes)==0,("raw/checkpoint gamma vs B cache differs",e.name)
            am,bm=base.metadata_finite(np,Araw),base.metadata_finite(np,Bcached)
            assert am["sha256"]==e.raw_sha256 and bm["sha256"]==e.cached_sha256
            gamma_holders.extend((Araw,Bcached));cached_gamma_holders.append(Bcached)
            gamma_checks.append({"name":e.name,"checkpoint_key":e.checkpoint_key,"A_raw":am,"B_cached":bm,"strict_bytes_equal":True,"separate_objects":True,"separate_addresses":True,"A_B_ranges_disjoint":True})
        assert len(gamma_holders)==322 and len(cached_gamma_holders)==161
        gamma_baselines=[h.raw() for h in gamma_holders]
        assert initial_parameter_identity==[(name,id(v)) for name,v in tree_flatten(model.parameters())] and len(initial_parameter_identity)==1349
        profile.set_enabled(True);profile.validate_current_bindings()
        receipt["vendor_rms_profile_preparation"]={"metadata":profile.metadata(),"snapshot":profile.snapshot(),"plain_norms":161,"parameter_arrays":1349,"same_parameter_identity":True,"derived_cachedgamma_arrays":161,"independent_raw_gamma_arrays":161,"all161_gamma_pairs_strict":gamma_checks,"prepared_before_any_token":True}
        receipt["gamma_readonly_baseline"]={"arrays":322,"bytes":sum(h.nbytes for h in gamma_holders),"once_only_raw_snapshots":True}
        sample("after-vendor-profile-before-any-token");save()
        norm_capture_calls=[];plain_capture_calls=[];plain_capture_rows=[]
        @contextmanager
        def capture_norm_engagement():
            assert GDN._decode_conv_norm_reuse_cache is candidate_norm and PlainRMSNorm.__call__ is candidate_plain
            def counted(self,*a,**kw):
                norm_capture_calls.append(id(self));return candidate_norm(self,*a,**kw)
            def plain_counted(self,x):
                assert id(self) in plain_norms,"unexpected plainRMSNorm instance during capture"
                assert object.__getattribute__(self,"_vendor_rms_leaf_enabled") is True and self._vendor_rms_leaf_current()
                assert x.dtype==mx.bfloat16 and x.ndim>=1 and self.weight.shape==(x.shape[-1],),"vendor RMS branch ineligible"
                plain_capture_calls.append(id(self))
                y=candidate_plain(self,x)
                plain_capture_rows.append({"name":plain_names[id(self)],"norm_id":id(self),"input_shape":list(x.shape),"input_dtype":str(x.dtype),"output_shape":list(y.shape),"output_dtype":str(y.dtype),"cached_gamma_id":id(object.__getattribute__(self,"_vendor_rms_leaf_gamma")),"actual_branch_predicates_passed":True})
                return y
            GDN._decode_conv_norm_reuse_cache=counted;PlainRMSNorm.__call__=plain_counted
            try:yield
            finally:
                GDN._decode_conv_norm_reuse_cache=candidate_norm;PlainRMSNorm.__call__=candidate_plain
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
        assert warm_ids==expected_ids;receipt["initial_ordinary_warm_new_profile_raw_ids"]=list(warm_ids);del warm,flat,ctx,logits,pred
        mx.synchronize();sample("after-initial-warm32")
        capture_seed,first=setup();assert first==expected_ids[0]
        for name,v in tree_flatten(model.parameters()):
            h=shared(v,"parameter."+name);parameter_holders.append(h);native.call("v33_pin",h.address,h.nbytes,h.label.encode())
        assert len(parameter_holders)==1349
        for h in cached_gamma_holders:native.call("v33_pin",h.address,h.nbytes,h.label.encode())
        receipt["native_parameter_pins"]={"model_parameter_arrays":1349,"derived_cachedgamma_arrays":161,"total_pins":1510,"reference_rawgamma_pinned_into_B":False}
        input_labels=("tokens","positions","lens")+tuple(f"state.{i}" for i in range(128))
        output_labels=("logits",)+tuple(f"state.{i}" for i in range(128))+("prediction","next_positions","next_lens")
        seed_holders=[shared(v,n) for v,n in zip(capture_seed,input_labels)]
        ctx=ForwardContext("decode",layers(capture_seed[3:]),positions=capture_seed[1],lens=capture_seed[2],max_len=CAP)
        mx.eval(*(v for c,l in zip(ctx.caches,model.layers) for v in ((c.K,c.V) if l.is_attn else (c.conv,c.S))))
        mx.synchronize();sample("before-capture")
        native.call("v33_begin");armed=True
        with capture_norm_engagement():
            logits=model(capture_seed[0],ctx)
            captured=(logits,*states(ctx),mx.argmax(logits,axis=-1).astype(mx.int32),capture_seed[1]+1,capture_seed[2]+1)
            mx.eval(*captured);mx.synchronize()
        native.call("v33_end_build");armed=False
        assert len(norm_capture_calls)==48 and set(norm_capture_calls)==set(map(id,gdns))
        assert len(plain_capture_calls)==161 and set(plain_capture_calls)==plain_norms
        assert PlainRMSNorm.__call__ is candidate_plain
        profile.validate_current_bindings()
        receipt["B_vendor_plain_rms_capture_engagement"]={"calls":161,"distinct_plain_norm_objects":161,"all161_prepared_enabled":True,"counter_removed_before_replay":True,"actual_branch_rows":plain_capture_rows,"GatedRMSNorm_hooked":False,"math_profile":"actual MXfast.rms_norm with independently cached originalgamma"}
        receipt["B_conv_normreuse_cache_capture_engagement"]={"calls":48,"distinct_GDN_objects":48,"normreuse_cache_ON_metadata":True,
                                               "counter_removed_after_capture":GDN._decode_conv_norm_reuse_cache is candidate_norm}
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
        native.call("v34_seal_feedback",128,CAP)
        native.call("v38_enable_icb")
        receipt["capture_native"]=native.report("native-sealed")
        native_gamma_proof=[]
        for h in cached_gamma_holders:
            rows=[v for v in receipt["capture_native"]["buffers"] if v["used"] and v["zero_copy"] and v["original_contents"]<=h.address and h.address+h.nbytes<=v["original_contents"]+v["length"] and v["role"].startswith(h.label+f" full-range={h.address}+{h.nbytes} binding-delta=0 readonly:")]
            assert len(rows)==1,("cachedgamma native readonly fullrange unproven",h.label,rows)
            native_gamma_proof.append({"label":h.label,"address":h.address,"bytes":h.nbytes,"original_buffer":rows[0]["original_buffer"],"role":rows[0]["role"],"full_range_readonly":True})
        assert len(native_gamma_proof)==161
        receipt["derived_cachedgamma161_native_readonly_proof"]=native_gamma_proof
        base.assert_icb(receipt["capture_native"],[])
        f=receipt["capture_native"]["feedback"];assert f["sealed"] and f["input_count"]==131 and f["output_count"]==132 and f["link_count"]==131 and f["capacity"]==CAP
        pp=f["pingpong"];assert (pp["requested_count"],pp["eligible_count"],pp["staged_link_count"])==(48,48,83)
        for h,expected in zip(output_holders,expected_final):
            assert h.label==expected["label"] and h.meta["shape"]==expected["physical_shape"] and h.meta["dtype"]==expected["dtype"] and h.nbytes==expected["physical_bytes"]
        assert all(h.meta["shape"]==[1,4,CAP,256] for i,h in enumerate(seed_holders[3:]) if i in kv_state_indices)
        assert all(h.meta["shape"]==[1,4,CAP,256] for h in output_holders if h.label in kv_output_labels)
        sample("after-capture");save()
        seed_baselines=[h.raw() for h in seed_holders]
        output_baselines=[h.raw() for h in output_holders]
        receipt["captured263_original_owners_readonly_baselines"]={"seed_arrays":131,"captured_outputs":132,"physical_KV_capacity":CAP,"bytes":sum(map(len,seed_baselines))+sum(map(len,output_baselines)),"scope":"once-only backups of true64 owners outside all clocks"}
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
            assert ids==expected_ids,"new97 reference raw IDs differ"
            checks=[]
            for h,expected in zip(output_holders,expected_final):
                target=ctypes.create_string_buffer(h.nbytes)
                native.call("v34_copy_output",h.address,h.nbytes,ctypes.cast(target,ctypes.c_void_p))
                physical=memoryview(target).cast("B")
                canonical,canonical_shape=base.canonical_output(physical,h.meta["shape"],h.meta["dtype"],h.label in kv_output_labels,50)
                digest=hashlib.sha256(canonical).hexdigest()
                assert canonical_shape==expected["shape"] and len(canonical)==expected["bytes"]
                assert digest==expected["A_sha256"],("canonical256 final output differs from actual97 independent newA",h.label)
                checks.append({"label":h.label,"shape":canonical_shape,"dtype":h.meta["dtype"],"bytes":len(canonical),"sha256":digest,"qualified_A_sha256":expected["A_sha256"],"physical_shape":h.meta["shape"],"physical_bytes":h.nbytes,"physical_sha256":hashlib.sha256(physical).hexdigest(),"physical_tail_zero_checked":h.label in kv_output_labels,"CPU_canonical_zero_fill":h.label in kv_output_labels})
            assert len(checks)==132 and sum(v["physical_tail_zero_checked"] for v in checks)==32
            owners_readonly={"capture_seed":base.originals_readonly(native,seed_holders,seed_baselines,f"request{index+1}.capture_seed"),"captured_reference":base.originals_readonly(native,output_holders,output_baselines,f"request{index+1}.captured_reference")}
            gamma_readonly=base.originals_readonly(native,gamma_holders,gamma_baselines,f"request{index+1}.raw-and-cached-gamma")
            profile.validate_current_bindings();assert PlainRMSNorm.__call__ is candidate_plain
            report=native.report(f"native-request-{index+1}");f=report["feedback"]
            base.assert_icb(report,[31]*(index+1))
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
                 "final_hash_cpu_copy_bytes_outside_timing":sum(h.nbytes for h in output_holders),"native":report,"captured263_original_owners_readonly":owners_readonly,"gamma322_readonly":gamma_readonly,
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
        assert initial_parameter_identity==[(name,id(v)) for name,v in tree_flatten(model.parameters())] and len(initial_parameter_identity)==1349
        receipt["same1349_parameter_owners_final"]=True
        profile.validate_current_bindings();assert profile.snapshot()["enabled_count"]==161 and PlainRMSNorm.__call__ is candidate_plain
        receipt["gamma322_readonly_final"]=base.originals_readonly(native,gamma_holders,gamma_baselines,"final.raw-and-cached-gamma")
        receipt["vendor_rms_profile_final_snapshot"]=profile.snapshot()
        receipt["terminal_canonical_hash_checks"]=sum(len(r["final132_hash_checks"]) for r in receipt["requests"])
        assert receipt["terminal_canonical_hash_checks"]==792
        source_check();receipt.update(status="COMPLETED_V100_VENDOR_PLAIN_RMS_B_ONLY_EXPLORATORY_SCREEN_NO_ADOPTION",stage="complete");save();return 0
    except BaseException as error:
        receipt.update(status="FAIL_STOP",error=repr(error),traceback=traceback.format_exc(),capture_armed=armed)
        if native is not None:
            try:native.report("native-on-failure")
            except BaseException as report_error:receipt["native_report_error"]=repr(report_error)
        save();raise
    finally:
        if allocation_classes is not None:
            for klass,old_step in zip(allocation_classes,original_allocation_steps):klass.step=old_step
            receipt["allocation_bucket_restored_finally"]=True
        try:
            if mx is not None:mx.synchronize()
            if resources is not None:receipt["finally_resource"]=sample("finally-before-exit")
        except BaseException as error:
            receipt.update(status="FAIL_STOP",finally_error=repr(error));save();raise
        save()


if __name__=="__main__":raise SystemExit(main())
