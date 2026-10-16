"""Root-only V86 BA/gates/activated-prework GDN B-only native exploratory timing; inert never imports MLX."""
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
ROOT = P / "candidate-v82-ba-gates-prework-build1-root"
LIB = P / "v40-native-icb-build2/libv40replay.dylib"
QUAL = P / "v23-root-control-build1/fullmodel-v82native-attempt1/child-output/receipt.json"
QUAL_SHA = "6553eae93b7f79bed92753749cb04b156bd4343db422ec88043a75c8072265af"
RELEASE = P / "v23-root-control-build1/fullmodel-v82native-attempt1/external-release.json"
RELEASE_SHA = "d452c839515007bbc1c809a3ffe900972eef71ad3531e6af1e3e265f427c6847"
OLD44 = P / "v23-root-control-build1/fullmodel-v44native-attempt1/child-output/receipt.json"
OLD44_SHA = "1dcfc41e7779c575602bd6c73bae4e4800664db42e1ee87e75414335bdc45b6d"
CONTROLS = {"MLX_MAX_OPS_PER_BUFFER": "50", "MLX_MAX_MB_PER_BUFFER": "50",
            "MLX_BFS_MAX_WIDTH": "20", "MLX_METAL_FAST_SYNCH": "0"}


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value): Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def module(name, path):
    spec=importlib.util.spec_from_file_location(name,path); value=importlib.util.module_from_spec(spec)
    sys.modules[name]=value; spec.loader.exec_module(value); return value


def source_check():
    pins=json.loads((D/"source-pins.json").read_text())
    for path,digest in pins.items():assert sha(path)==digest,("source changed",path)
    assert sha(QUAL)==QUAL_SHA and sha(RELEASE)==RELEASE_SHA
    q=json.loads(QUAL.read_text());release=json.loads(RELEASE.read_text())
    assert q["status"]=="PASS_V82_BA_FP32_GATES_PREWORK_GDN_FULL_ICB_BASE_RESET_CHANGED_STRICT"
    assert q["strict_output_checks"]==12276 and q["native_steps"]==93 and len(q["requests"])==3
    assert q["ba_gates_target288_readonly_final"] and q["BA_gates_prework_ON"]==48
    assert q["same_parameter_identity_count"]==1349 and q["same_parameter_identity_preserved"]
    assert release["status"]=="PASS_EXTERNAL_RELEASE" and release["GPU_released"] and release["tool_exit"]==0
    assert all(v["returncode"]==1 and v["stdout"]==v["stderr"]=="" for v in release["strict_absence"])
    original_pins=json.loads((D/"qualification-source-pins.json").read_text())
    assert q["source_pins"]==original_pins and len(original_pins)==107
    assert all(pins[path]==digest for path,digest in original_pins.items())
    assert sha(OLD44)==OLD44_SHA
    assert q["A_saved_baseline_receipt_sha256"]=="c57e7625baa5a4ce450fc7306233e766f1efd71636713080d748780624542c8a"
    engagement=q["B_ba_gates_prework_capture_engagement"]
    assert engagement["calls"]==engagement["distinct_GDN_objects"]==48 and engagement["observers_removed_before_replay"] is True
    assert len(engagement["prework_calls"])==48 and all(v["outputs"]==6 for v in engagement["prework_calls"])
    prep=q["ba_gates_prework_preparation"]
    assert prep["status"]=="READY_ACTUAL_LAYOUT_PREPARED" and len(prep["layout_rows"])==384 and len(prep["prepared"])==48
    assert prep["readonly_after_prepare"] and prep["original_parameter_ranges_disjoint"]
    proof=q["ba_gates_prework_native_binding_proof"]
    assert proof["Python_capture_calls"]==48 and len(proof["original_target_ranges"])==288
    assert all(v["full_original_parameter_range_readonly"] for v in proof["original_target_ranges"])
    base,reset=q["requests"][:2]
    assert base["name"]=="base" and reset["name"]=="reset-base" and len(base["steps"])==len(reset["steps"])==31
    expected=base["steps"][-1]["outputs"];assert len(expected)==132
    assert all(v["strict_bytes_equal"] and v["finite"] for v in expected)
    assert expected==json.loads(OLD44.read_text())["requests"][0]["steps"][-1]["outputs"]
    assert [(v["label"],v["A_sha256"]) for v in expected]==[(v["label"],v["A_sha256"]) for v in reset["steps"][-1]["outputs"]]
    assert q["native_library"]==str(LIB) and q["source_pins"][str(LIB)]==pins[str(LIB)]
    manifest=json.loads((P/"v82-native-fullqual-build1/candidate-manifest.json").read_text())["files"]
    expected_source={k.removeprefix("tutorial_code/miniengine/"):v for k,v in manifest.items()}
    package=ROOT/"tutorial_code/miniengine"
    assert {str(f.relative_to(package)):sha(f) for f in package.rglob("*") if f.is_file()}==expected_source and sum(k.endswith(".py") for k in expected_source)==29
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
             "qualification_receipt":str(QUAL),"qualification_sha256":QUAL_SHA,"qualification_release_sha256":RELEASE_SHA,"old44_A_gold_sha256":OLD44_SHA,"requests":[],
             "individual_argument_reflection":"NOT_OBSERVED; sourceABI14 pointers, native40 report lacks per-argument table",
             "clockscope_comparison":"Same raw-native V44 t1 after seed uploads; do not compare directly to V67/V71 engine decode clocks including seed setup",
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
        base=module("v35_native_support",P/"v82-native-fullqual-build1/fullqual.py")
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
        initial_parameter_identity=[(name,id(v)) for name,v in tree_flatten(model.parameters())]
        gdns=[l.linear_attn for l in model.layers if not l.is_attn];assert len(model.layers)==64 and len(gdns)==48
        for g in gdns:
            g.set_gates_compile(True)
            g.set_conv_norm_reuse_cache(True)
        from tutorial_code.miniengine.model.gdn import GatedDeltaNet as GDN
        candidate_norm=GDN._decode_conv_norm_reuse_cache
        candidate_decode=GDN._decode_dispatch
        candidate_ba=GDN._decode_ba_gates_prework
        assert all(g._ba_gates_prework_enabled is False and g._ba_gates_prework_helper is None for g in gdns)
        ba_capture_calls=[];prework_capture_calls=[]
        @contextmanager
        def capture_norm_engagement():
            from tutorial_code.miniengine.model.gdn_ba_prework import PreparedBA
            original_prework=PreparedBA.prework
            assert GDN._decode_ba_gates_prework is candidate_ba
            def counted(self,*a,**kw):
                ba_capture_calls.append(id(self));return candidate_ba(self,*a,**kw)
            def prework_counted(self,activated,x):
                out=original_prework(self,activated,x)
                assert len(out)==6 and len({id(v) for v in out})==6
                assert [v.shape for v in out]==[(1,16,128),(1,16,128),(1,48,128),(1,16,2),(1,48),(1,48)] and all(v.dtype==mx.float32 for v in out)
                prework_capture_calls.append({"helper_id":id(self),"x_id":id(x),"activated_id":id(activated),"outputs":6,"output_shapes":[list(v.shape) for v in out],"output_dtypes":[str(v.dtype) for v in out]})
                return out
            GDN._decode_ba_gates_prework=counted;PreparedBA.prework=prework_counted
            try:yield
            finally:
                GDN._decode_ba_gates_prework=candidate_ba;PreparedBA.prework=original_prework
        target_holders=[];target_before=[]
        def target_fingerprints():
            return [{**h.meta,"sha256":hashlib.sha256(h.bytes_view).hexdigest()} for h in target_holders]
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
        # One extra qualified44 B decode observes true norm X and conv23 activation.
        # BA stays OFF; these preparation probes are outside capture and all timed requests.
        layout_seed,layout_first=setup();layout_x={};layout_activated={};active_gdn=[]
        from tutorial_code.miniengine.kernels import gdn_conv_silu_metal as conv_module
        original_conv=conv_module.conv_silu_raw_tail
        assert all(g._ba_gates_prework_enabled is False for g in gdns)
        def retain_layout_x(self,x,st,*a,**kw):
            assert id(self) not in layout_x;layout_x[id(self)]=x;active_gdn.append(id(self))
            try:return candidate_norm(self,x,st,*a,**kw)
            finally:active_gdn.pop()
        def retain_activated(*a,**kw):
            out=original_conv(*a,**kw);assert len(active_gdn)==1 and active_gdn[0] not in layout_activated
            layout_activated[active_gdn[0]]=out[0];return out
        GDN._decode_conv_norm_reuse_cache=retain_layout_x;conv_module.conv_silu_raw_tail=retain_activated
        try:
            layout_ctx=ForwardContext("decode",layers(layout_seed[3:]),positions=layout_seed[1],lens=layout_seed[2],max_len=20)
            layout_logits=model(layout_seed[0],layout_ctx)
            mx.eval(layout_logits,*states(layout_ctx),*layout_x.values(),*layout_activated.values());mx.synchronize()
            assert int(mx.argmax(layout_logits,axis=-1).item())==w["continuation_ids"][1]
        finally:
            GDN._decode_conv_norm_reuse_cache=candidate_norm;conv_module.conv_silu_raw_tail=original_conv
        assert set(layout_x)==set(layout_activated)==set(map(id,gdns)) and len(layout_x)==48 and not active_gdn
        from tutorial_code.miniengine.model.layers import RMSNorm
        assert all(type(layer.input_layernorm) is RMSNorm for layer in model.layers)
        layout_kernel=mx.fast.metal_kernel(name="v82_prepare_observed_live_strides",input_names=["value"],output_names=["descriptor"],source=(D/"layout_probe.metal").read_text(),ensure_row_contiguous=False)
        layout_rows=[]
        def observed_layout(value,label):
            assert value.ndim in (1,2)
            descriptor,=layout_kernel(inputs=[value],grid=(1,1,1),threadgroup=(1,1,1),output_shapes=[(5,)],output_dtypes=[mx.uint64])
            mx.eval(descriptor);v=descriptor.tolist();ndim=int(v[0]);assert ndim==value.ndim and v[1:1+ndim]==list(value.shape)
            strides=tuple(int(x) for x in v[3:3+ndim]);expected=1;canonical=True
            for dim,stride in zip(reversed(value.shape),reversed(strides)):
                canonical=canonical and (dim==1 or stride==expected);expected*=dim
            row={"label":label,"shape":list(value.shape),"dtype":str(value.dtype),"actual_strides_elements":list(strides),"row_contiguous_derived_singleton_aware":canonical,"internal_MLX_flags_observed":False,"ensure_row_contiguous":False}
            layout_rows.append(row);save();assert canonical,("layout STOP",row);return strides
        prepared=[]
        receipt["ba_gates_prework_preparation"]={"status":"PREPARING","extra_ordinary_qualified44_decode_steps":1,"included_in_timed_requests":False,"layout_rows":layout_rows,"prepared":prepared}
        save()
        for layer_index,layer in enumerate(model.layers):
            if layer.is_attn:continue
            g=layer.linear_attn;ops=g._ba_gates_prework_operands();operand_strides=[];parameter_ranges=[]
            for suffix,value in zip(("in_proj_b.weight","in_proj_b.scales","in_proj_a.weight","in_proj_a.scales","A_log","dt_bias"),ops):
                label=f"parameter.layers.{layer_index}.linear_attn.{suffix}"
                holder=shared(value,label);target_holders.append(holder)
                target_before.append({**holder.meta,"sha256":hashlib.sha256(holder.bytes_view).hexdigest()})
                operand_strides.append(observed_layout(value,label));parameter_ranges.append((holder.address,holder.nbytes))
            x=layout_x[id(g)];activated=layout_activated[id(g)]
            assert x.shape==(1,5120) and x.dtype==mx.bfloat16 and activated.shape==(1,10240) and activated.dtype==mx.bfloat16
            x_strides=observed_layout(x,f"layer.{layer_index}.actual_input_layernorm_X")
            activated_strides=observed_layout(activated,f"layer.{layer_index}.actual_conv23_activation")
            xholder=shared(x,f"layer.{layer_index}.layout_X");aholder=shared(activated,f"layer.{layer_index}.layout_activated")
            input_before=[hashlib.sha256(h.bytes_view).hexdigest() for h in (xholder,aholder)]
            active=mx.get_active_memory()
            g.prepare_ba_gates_prework(observed_parameter_strides=operand_strides,observed_x=x,observed_x_strides=x_strides,
                observed_activated=activated,observed_activated_strides=activated_strides,observed_parameter_ranges=parameter_ranges,
                runtime_version=versions["mlx"],device_name=info["device_name"],architecture=info["architecture"],
                input_layout_contract="qualified-original-input-layernorm-B1-and-conv23")
            assert mx.get_active_memory()==active and g._ba_gates_prework_enabled is False
            helper=g._ba_gates_prework_helper
            assert all(a is b for a,b in zip(helper.operands,ops)) and input_before==[hashlib.sha256(h.bytes_view).hexdigest() for h in (xholder,aholder)]
            prepared.append({"layer":layer_index,"helper_id":id(helper),"original_operand_ids":[id(v) for v in ops],"x_strides":list(x_strides),"activated_strides":list(activated_strides),"observed_parameter_strides":[list(v) for v in operand_strides],"parameter_ranges":[list(v) for v in parameter_ranges],"mlx_active_before_prepare":active,"mlx_active_after_prepare":mx.get_active_memory(),"same_original_handles":True,"X_activated_readonly":True})
        assert len(prepared)==48 and len(target_holders)==288 and len(layout_rows)==384
        assert sum(h.nbytes for h in target_holders)==13289472 and target_fingerprints()==target_before
        ranges=[(h.address,h.address+h.nbytes) for h in target_holders]
        assert all(b<=c or d<=a for i,(a,b) in enumerate(ranges) for c,d in ranges[i+1:]),"original target parameter ranges alias"
        assert initial_parameter_identity==[(name,id(value)) for name,value in tree_flatten(model.parameters())] and len(initial_parameter_identity)==1349
        for g in gdns:g.set_ba_gates_prework(True)
        receipt["ba_gates_prework_preparation"]={"status":"READY_ACTUAL_LAYOUT_PREPARED","extra_ordinary_qualified44_decode_steps":1,"included_in_timed_requests":False,"purpose":"actual48 original norm-output and conv23 activation strides; no performance timing","layout_rows":layout_rows,"prepared":prepared,"original288_target_arrays":target_before,"readonly_after_prepare":True,"target_payload_bytes":13289472,"extra_packed_weight_bytes":0,"parameter_owners":1349,"same_parameter_identity_after_prepare":True,"source_contract":"unchanged original RMSNorm and conv23 output stores; restricted engine path, arbitrary externalX not qualified","target_parameter_holders_retained":True,"original_parameter_ranges_disjoint":True}
        save();del layout_seed,layout_ctx,layout_logits,layout_x,layout_activated,x,activated,xholder,aholder
        mx.synchronize();sample("after-BA-gates-layout-prepare")
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
        assert len(ba_capture_calls)==48 and set(ba_capture_calls)==set(map(id,gdns))
        assert len(prework_capture_calls)==48 and {r["helper_id"] for r in prework_capture_calls}=={id(g._ba_gates_prework_helper) for g in gdns}
        from tutorial_code.miniengine.model.gdn_ba_prework import PreparedBA
        receipt["B_ba_gates_prework_capture_engagement"]={"calls":48,"distinct_GDN_objects":48,"all48_enabled":all(g._ba_gates_prework_enabled for g in gdns),"prework_calls":prework_capture_calls,"six_output_consumption":"state40 consumes Q/K/V/scales/g/beta; original norm/out/tail unchanged; whole132 comparison","observers_removed_before_replay":GDN._decode_ba_gates_prework is candidate_ba and PreparedBA.prework.__name__=="prework"}
        assert receipt["B_ba_gates_prework_capture_engagement"]["observers_removed_before_replay"]
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
        receipt["capture_native"]=native.report("native-sealed")
        fused_pso=[v for v in receipt["capture_native"]["feedback"]["icb"]["PSO_records"] if "gdn_activated_BA_gates_prework_v76_build1" in v["function"]]
        assert 1<=len(fused_pso)<=48 and all(v["actual_supports_ICB"] for v in fused_pso)
        buffer_rows=receipt["capture_native"]["buffers"];binding_proof=[]
        parameter_by_name={v.label:v for v in parameter_holders}
        for holder in target_holders:
            owner=parameter_by_name[holder.label];assert owner.original is holder.original and owner.address==holder.address and owner.nbytes==holder.nbytes
            matches=[v for v in buffer_rows if v["used"] and v["zero_copy"] and v["original_contents"]<=holder.address and holder.address+holder.nbytes<=v["original_contents"]+v["length"] and v["role"].startswith(holder.label+f" full-range={holder.address}+{holder.nbytes} binding-delta=0 readonly:")]
            assert len(matches)==1 and "gdn_activated_BA_gates_prework_v76_build1" in matches[0]["role"],("unproven fused original weight range",holder.label,matches)
            binding_proof.append({"parameter":holder.label,"original_object_id":id(holder.original),"address":holder.address,"bytes":holder.nbytes,"actual_original_buffer":matches[0]["original_buffer"],"role":matches[0]["role"],"full_original_parameter_range_readonly":True})
        assert len(binding_proof)==288
        receipt["ba_gates_prework_native_binding_proof"]={"fused_PSOs":fused_pso,"Python_capture_calls":48,"original_target_ranges":binding_proof,"buffer_argument_count_source_proof":14,"individual_argument_reflection":"NOT_OBSERVED; native40 report omits per-argument table; source8inputs+6outputs; observer still required"}
        base.assert_icb(receipt["capture_native"],[])
        f=receipt["capture_native"]["feedback"];assert f["sealed"] and f["input_count"]==131 and f["output_count"]==132 and f["link_count"]==131
        pp=f["pingpong"];assert (pp["requested_count"],pp["eligible_count"],pp["staged_link_count"])==(48,48,83)
        for h,expected in zip(output_holders,expected_final):
            assert h.label==expected["label"] and h.meta["shape"]==expected["shape"] and h.meta["dtype"]==expected["dtype"] and h.nbytes==expected["bytes"]
        sample("after-capture");save()
        seed_baselines=[h.raw() for h in seed_holders]
        output_baselines=[h.raw() for h in output_holders]
        receipt["captured263_original_owners_readonly_baselines"]={"seed_arrays":131,"captured_outputs":132,"bytes":sum(map(len,seed_baselines))+sum(map(len,output_baselines)),"scope":"once-only backups outside all clocks; actual model owners retained"}
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
            owners_readonly={"capture_seed":base.originals_readonly(native,seed_holders,seed_baselines,f"request{index+1}.capture_seed"),"captured_reference":base.originals_readonly(native,output_holders,output_baselines,f"request{index+1}.captured_reference")}
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
                 "final_hash_cpu_copy_bytes_outside_timing":sum(h.nbytes for h in output_holders),"native":report,"captured263_original_owners_readonly":owners_readonly,
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
        assert all(g._gates_compile_enabled and g._conv_norm_reuse_cache_enabled for g in gdns)
        assert GDN._decode_conv_norm_reuse_cache is candidate_norm and GDN._decode_ba_gates_prework is candidate_ba
        assert all(g._ba_gates_prework_enabled and g._ba_gates_prework_helper is not None for g in gdns)
        assert initial_parameter_identity==[(name,id(v)) for name,v in tree_flatten(model.parameters())] and len(initial_parameter_identity)==1349
        assert target_fingerprints()==target_before
        receipt["target288_original_parameters_readonly_final"]=True
        receipt["same1349_parameter_owners_final"]=True
        for name,mod in tuple(sys.modules.items()):
            if name.startswith("tutorial_code.miniengine") and getattr(mod,"__file__",None):
                path=str(Path(mod.__file__).resolve());assert path in pins and sha(path)==pins[path]
        source_check();receipt.update(status="PASS_V86_BA_GATES_PREWORK_GDN_NATIVE_B_ONLY_EXPLORATORY_SCREEN",stage="complete");save();return 0
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
