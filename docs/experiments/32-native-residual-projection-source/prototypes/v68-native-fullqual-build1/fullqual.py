"""Root-only V68 original-pointer four-GDN-projection+convnorm ICB versus accepted25 full numeric qualification; inert never imports MLX."""
import argparse
import ast
from contextlib import contextmanager
import ctypes
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import sys
import traceback

sys.dont_write_bytecode = True
D = Path(__file__).resolve().parent
P = D.parent
ROOT = P / "candidate-v68-four-projection-build1-root"
BASELINE_GDN = P / "candidate-v17-gdn-gates-build1-root/tutorial_code/miniengine/model/gdn.py"
BASELINE_QUAL = P / "v23-root-control-build1/fullmodel-v38native-attempt1/child-output/receipt.json"
BASELINE_QUAL_SHA = "c57e7625baa5a4ce450fc7306233e766f1efd71636713080d748780624542c8a"
LIB = P / "v40-native-icb-build2/libv40replay.dylib"
ACTUAL44 = P / "v23-root-control-build1/fullmodel-v44native-attempt1/child-output/receipt.json"
ACTUAL44_SHA = "1dcfc41e7779c575602bd6c73bae4e4800664db42e1ee87e75414335bdc45b6d"
NUM66 = P / "v23-root-control-build1/numeric-v66b2-attempt1/child-output/receipt.json"
NUM66_SHA = "af5d31fa9aeff992c36df169ebdd1d14a7ea2b04102c91ae95fade50582bf305"
CAP = 256
CONTROLS = {"MLX_MAX_OPS_PER_BUFFER": "50", "MLX_MAX_MB_PER_BUFFER": "50",
            "MLX_BFS_MAX_WIDTH": "20", "MLX_METAL_FAST_SYNCH": "0"}


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def source_check(library):
    pins = json.loads((D / "source-pins.json").read_text())
    for path, digest in pins.items(): assert sha(path) == digest, ("source changed", path)
    assert str(library) in pins, "unbound native library"
    assert sha(BASELINE_QUAL)==BASELINE_QUAL_SHA
    expected = json.loads((D / "candidate-manifest.json").read_text())["files"]
    expected = {k.removeprefix("tutorial_code/miniengine/"): v for k, v in expected.items()}
    package = ROOT / "tutorial_code/miniengine"
    actual = {str(f.relative_to(package)): sha(f) for f in package.rglob("*") if f.is_file()}
    assert actual == expected and sum(k.endswith(".py") for k in actual) == 28
    assert sha(ACTUAL44)==ACTUAL44_SHA and sha(NUM66)==NUM66_SHA
    n66=json.loads(NUM66.read_text())
    assert n66["status"]=="PASS_V66_FOUR_ORIGINAL_POINTER_GDN_PROJECTION_LOCAL_STRICT" and n66["strict_output_checks"]==396 and n66["readonly_all35"]
    assert json.loads(ACTUAL44.read_text())["status"]=="PASS_NATIVE_CONV_ACTIVATED_NORMREUSE_CACHE_ICB_BASE_RESET_CHANGED_VS_ACCEPTED25_STRICT"
    tiny=json.loads((P/"v23-root-control-build1/fullmodel-v40smoke-attempt2/child-output/receipt.json").read_text())
    assert tiny["status"]=="PASS_TINY_V40_NORMREUSE_CACHE_ICB_PINGPONG_3REQUESTS_15STEPS" and tiny["strict_output_checks"]==75
    local44_path=P/"v23-root-control-build1/numeric-v44-attempt1/child-output/receipt.json"
    assert sha(local44_path)=="1c22f2e29c78b885600c1be4a06fac357154a7eaa2ee82aa31b284921ff12392"
    local44=json.loads(local44_path.read_text())
    assert local44["status"]=="PASS_V44_ACTIVATED_NORMREUSE_CACHE_STRICT_BYTES_9REAL_6SYNTHETIC" and local44["strict_output_checks"]==30
    return pins


class Native:
    def __init__(self, out):
        self.out = out
        self.lib = ctypes.CDLL(None)
        U, V, S = ctypes.c_uint64, ctypes.c_void_p, ctypes.c_char_p
        specs = {"v33_pin": [U, U, S], "v33_begin": [], "v33_end_build": [],
                 "v33_register_output": [U, U, S],
                 "v34_register_input": [U, U, ctypes.c_uint32, S],
                 "v34_link": [U, U, U], "v34_seal_feedback": [ctypes.c_uint32, ctypes.c_uint32],
                 "v34_reset": [], "v34_write_input": [U, U, V], "v34_replay_step": [],
                 "v34_compare_output": [U, U, V], "v34_copy_output": [U, U, V],
                 "v36_register_pingpong": [U, U, U], "v38_enable_icb": []}
        for name, args in specs.items():
            fn = getattr(self.lib, name); fn.argtypes = args; fn.restype = ctypes.c_int
        self.lib.v34_report_json.argtypes = []
        self.lib.v34_report_json.restype = S
        self.lib.memcmp.argtypes = [V, V, ctypes.c_size_t]
        self.lib.memcmp.restype = ctypes.c_int

    def report(self, label):
        raw = self.lib.v34_report_json(); assert raw is not None
        value = json.loads(raw.decode()); write(self.out / (label + ".json"), value)
        return value

    def call(self, name, *args):
        rc = getattr(self.lib, name)(*args)
        if rc:
            self.report("native-first-failure")
            raise RuntimeError(f"{name} failed rc={rc}")

    def same(self, shared, baseline):
        assert shared.nbytes == len(baseline)
        # c_char_p keeps the immutable Python bytes alive; memcmp reads exactly n.
        return self.lib.memcmp(shared.address, ctypes.c_char_p(baseline), shared.nbytes) == 0


def metadata_finite(np, holder):
    dtype = holder.meta["dtype"]
    if dtype.endswith("bfloat16"):
        v = np.frombuffer(holder.bytes_view, dtype="<u2")
        finite = not bool(np.any((v & 0x7f80) == 0x7f80))
    elif dtype.endswith("float32"):
        finite = bool(np.isfinite(np.frombuffer(holder.bytes_view, dtype="<f4")).all())
    elif dtype.endswith("int32"):
        finite = True
    else:
        raise RuntimeError("unexpected dtype: " + dtype)
    assert finite, ("nonfinite", holder.label)
    return {**holder.meta, "finite": finite, "sha256": hashlib.sha256(holder.bytes_view).hexdigest()}


def originals_readonly(native, holders, baselines, label):
    assert len(holders) == len(baselines)
    for h, baseline in zip(holders, baselines):
        assert native.same(h, baseline), ("original changed", label, h.label)
    return {"label": label, "arrays": len(holders), "strict_bytes_equal": True,
            "bytes": sum(h.nbytes for h in holders), "method": "C memcmp of pinned original and once-only raw snapshot"}


def assert_icb(report, request_lengths):
    fb=report["feedback"];icb=fb["icb"]
    total=sum(request_lengths)
    expected_banks=[sum((n+1)//2 for n in request_lengths),sum(n//2 for n in request_lengths)]
    assert icb["enabled"] is True and icb["banks"]==2
    assert icb["execution_count"]==fb["total_steps"]==total
    assert icb["bank_executions"]==expected_banks
    assert icb["commands_per_bank"]>=report["dispatches"]>0
    assert icb["constant_bank_bytes"]>0
    assert icb["unique_original_PSOs"]>0 and len(icb["PSO_records"])==icb["unique_original_PSOs"]
    resources=icb["resources"]
    assert resources["read_only_count"]>0 and resources["mutable_arena_count"]==1
    assert resources["icb_read_declarations"]>0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path); ap.add_argument("--tokens", type=int, default=32)
    ap.add_argument("--native-library", type=Path, default=LIB); ap.add_argument("--inert", action="store_true")
    args = ap.parse_args(argv); library = args.native_library.resolve(); pins = source_check(library)
    w = json.loads((P / "workload.json").read_text())
    assert args.tokens == 32 and len(w["prompt_ids"]) == 19 and len(w["continuation_ids"]) == 32
    base_prompt = w["prompt_ids"]
    changed_prompt = base_prompt[:10] + [base_prompt[3]] + base_prompt[10:]
    assert len(changed_prompt) == 20
    cases = [("base", base_prompt, True), ("reset-base", base_prompt, True),
             ("changed-prefill", changed_prompt, False)]
    baseline_qualification=json.loads(BASELINE_QUAL.read_text())
    assert baseline_qualification["status"]=="PASS_NATIVE_ICB_S_PINGPONG_BASE_RESET_CHANGED_STRICT"
    assert baseline_qualification["declared_prompts"]=={n:p for n,p,_ in cases}
    archived_cases={r["name"]:r for r in baseline_qualification["requests"]}
    assert all(len(r["steps"])==31 for r in archived_cases.values())
    actual44_cases={r["name"]:r for r in json.loads(ACTUAL44.read_text())["requests"]}
    assert all(len(r["steps"])==31 for r in actual44_cases.values())
    if args.inert:
        assert not any(n == "mlx" or n.startswith("mlx.") for n in sys.modules)
        print(json.dumps({"status": "PASS_CPU_INERT", "pins": len(pins), "MLX_imported": False,
                          "GPU_used": False, "requests": [[n, len(p)] for n, p, _ in cases],
                          "inputs": 131, "outputs": 132, "feedback_links": 131, "GDN_S_pingpong_requested": 48, "staged_links_required": 83, "required_actual_ICB_banks": 2}))
        return 0
    assert args.out is not None and args.out.is_absolute()
    args.out.mkdir(exist_ok=False)
    receipt = {"status": "STARTED", "stage": "beforeload", "pid": os.getpid(), "model_loads": 0,
               "source_pins": pins, "native_library": str(library), "fixed_cap": CAP, "requests": [],
               "declared_prompts": {n: p for n, p, _ in cases}, "performance_point": False,
               "ICB_scope": "actual native ICB qualification; unrelated to mx.compile", "adopted": False, "input_upload_scope": "all131 once per request only",
               "B_token_feedback": "captured MLX argmax -> staged GPU blit; no CPU/saved-continuation feeding",
               "A_saved_baseline_receipt":str(BASELINE_QUAL),"A_saved_baseline_receipt_sha256":BASELINE_QUAL_SHA,
               "actual44_gold_sha256":ACTUAL44_SHA,"local66_numeric_sha256":NUM66_SHA,
               "four_projection_scope":"qualified original input-layernorm normalB1 only; defaultOFF; no per-token layout probe; possible loss of qkv/conv overlap",
               "declared_fused_kernel_buffer_arguments":13,"actual_individual_argument_reflection":"NOT_OBSERVED: native40 report has no per-argument table; frozen source ABI9inputs+4outputs, no implicit descriptor inputs"}
    save = lambda: write(args.out / "receipt.json", receipt)
    save(); mx = None; resources = None; native = None; loaded = False; armed = False
    parameter_holders, seed_holders, output_holders = [], [], []
    try:
        assert os.environ.get("DYLD_INSERT_LIBRARIES") == str(library)
        assert all(os.environ.get(k) == v for k, v in CONTROLS.items())
        versions = {n: importlib.metadata.version(n) for n in ("mlx", "mlx-metal")}
        assert versions == {"mlx": "0.32.3", "mlx-metal": "0.32.3"}
        support = module("v34_resources", P / "v24-fullmodel-qual-build1/fullqual.py")
        prior = module("v34_shared", P / "v33-fixed-buffer-replay-worker-build3/fullqual.py")
        resources = support.Resources(args.out); resources.sample("beforeload", loaded=False)
        native = Native(args.out)
        assert not any(n == "mlx" or n.startswith(("mlx.", "tutorial_code.")) for n in sys.modules)
        import mlx.core as mx
        import numpy as np
        from mlx.utils import tree_flatten
        info = mx.device_info()
        assert mx.default_device() == mx.gpu and info["device_name"] == "Apple M4 Max" and info["architecture"] == "applegpu_g16s"
        previous_limit = mx.set_cache_limit(2 << 30)
        receipt["runtime"] = {"versions": versions, "device": info, "cache_limit_bytes": 2 << 30,
                              "previous_cache_limit_bytes": previous_limit, "environment": {**CONTROLS, "DYLD_INSERT_LIBRARIES": str(library)}}
        sys.path.insert(0, str(ROOT))
        from tutorial_code.miniengine.model.loader import load_model
        from tutorial_code.miniengine.cache import BatchKV, BatchState, DecodeBatch, make_cache
        from tutorial_code.miniengine.context import ForwardContext
        from tutorial_code.miniengine.eos import read_eos_ids
        model, cfg = load_model(w["model_dir"], dtype=mx.bfloat16)
        loaded = True; receipt["model_loads"] = 1; mx.synchronize()
        receipt["afterload"] = resources.sample("afterload-before-any-token", loaded=True, mx=mx)
        initial_parameter_identity=[(name,id(value)) for name,value in tree_flatten(model.parameters())]
        gdns = [l.linear_attn for l in model.layers if not l.is_attn]
        assert len(model.layers) == 64 and len(gdns) == 48
        for g in gdns:
            g.set_gates_compile(True)
            g.set_conv_norm_reuse_cache(True)
        from tutorial_code.miniengine.model import gdn as gdn_module
        GDN=gdn_module.GatedDeltaNet
        assert all(type(g) is GDN and g._conv_norm_reuse_cache_enabled is True for g in gdns)
        old_tree=ast.parse(BASELINE_GDN.read_text())
        old_class=next(n for n in old_tree.body if isinstance(n,ast.ClassDef) and n.name=="GatedDeltaNet")
        old_decode=next(n for n in old_class.body if isinstance(n,ast.FunctionDef) and n.name=="_decode")
        legacy_namespace={}
        exec(compile(ast.Module(body=[old_decode],type_ignores=[]),str(BASELINE_GDN),"exec"),vars(gdn_module),legacy_namespace)
        legacy_decode=legacy_namespace["_decode"]
        candidate_decode,candidate_norm=GDN._decode_dispatch,GDN._decode_conv_norm_reuse_cache
        candidate_four=GDN._decode_conv_norm_reuse_four_projection
        assert all(g._four_projection_enabled is False and g._four_projection_helper is None for g in gdns)
        baseline_calls=[]
        @contextmanager
        def baseline_decode():
            assert GDN._decode_dispatch is candidate_decode and GDN._decode_conv_norm_reuse_cache is candidate_norm
            flags=[g._four_projection_enabled for g in gdns]
            for g in gdns:g.set_four_projection(False)
            assert all(g._four_projection_enabled is False for g in gdns)
            GDN._decode_dispatch=legacy_decode;GDN._decode_conv_norm_reuse_cache=legacy_decode
            baseline_calls.append({"method":"oldaccepted25_decode_AST","four_projection_OFF":48,"legacy_dispatch_observed":GDN._decode_dispatch is legacy_decode})
            try: yield
            finally:
                GDN._decode_dispatch=candidate_decode;GDN._decode_conv_norm_reuse_cache=candidate_norm
                for g,flag in zip(gdns,flags):g.set_four_projection(flag)
                assert GDN._decode_dispatch is candidate_decode and GDN._decode_conv_norm_reuse_cache is candidate_norm
        four_capture_calls=[];projection_capture_calls=[]
        @contextmanager
        def capture_norm_engagement():
            from tutorial_code.miniengine.kernels.v66_four_projection.four_projection import FourProjection
            original_project=FourProjection.project
            assert GDN._decode_conv_norm_reuse_four_projection is candidate_four
            def counted(self,*a,**kw):
                four_capture_calls.append(id(self))
                return candidate_four(self,*a,**kw)
            def project_counted(self,x,**kw):
                out=original_project(self,x,**kw)
                assert len(out)==4 and len({id(v) for v in out})==4
                assert [v.shape for v in out]==[(1,n) for n in (10240,6144,48,48)] and all(v.dtype==mx.bfloat16 for v in out)
                projection_capture_calls.append({"helper_id":id(self),"x_id":id(x),"outputs":4,"output_shapes":[list(v.shape) for v in out],"output_dtypes":[str(v.dtype) for v in out]})
                return out
            GDN._decode_conv_norm_reuse_four_projection=counted;FourProjection.project=project_counted
            try:yield
            finally:
                GDN._decode_conv_norm_reuse_four_projection=candidate_four;FourProjection.project=original_project
        target_holders=[];target_before=[]
        def target_fingerprints():
            return [{**h.meta,"sha256":hashlib.sha256(h.bytes_view).hexdigest()} for h in target_holders]
        receipt["A_baseline_source"]={"path":str(BASELINE_GDN),"sha256":sha(BASELINE_GDN),
                                      "method":"_decode AST compiled into actualcandidate namespace; temporaryclassbinddispatchandnormreuseentry;finallyrestore",
                                      "source_proof":json.loads((D/"baseline-source-proof.json").read_text()),"same_weights_load":True}

        eos = read_eos_ids(w["model_dir"]); receipt["eos_ids"] = sorted(eos)
        receipt.update(stage="ready", gates_on=48, conv_on=48, conv_off=False, normreuse_cache_on=48, whole_functional_off=True)
        write(args.out / "model-ready.json", {"pid": os.getpid(), "token_executed": False, "model_loads": 1})
        save(); print(json.dumps({"event": "MODEL_READY", "pid": os.getpid()}), flush=True)

        def shared(value, label): return prior.Shared(mx, np, value, label)
        def ctx_layers(flat):
            return [BatchKV(flat[2*i][:], flat[2*i+1][:]) if layer.is_attn else BatchState(flat[2*i], flat[2*i+1])
                    for i, layer in enumerate(model.layers)]
        def state_of(ctx):
            return tuple(v for layer, cache in zip(model.layers, ctx.caches)
                         for v in ((cache.K,cache.V) if layer.is_attn else (cache.conv,cache.S)))
        def outputs(logits, state, positions, lens):
            return (logits, *state, mx.argmax(logits, axis=-1).astype(mx.int32), positions + 1, lens + 1)
        def setup(prompt):
            caches = make_cache(model)
            with baseline_decode():
                logits = model(mx.array(prompt, mx.int32), ForwardContext("prefill", caches), last_only=True)
            batch = DecodeBatch(model); batch.rebuild([], [("v34", caches, len(prompt) + 32)])
            flat = tuple(v[:] for v in batch.arrays()); assert len(flat) == 128
            mx.eval(logits, *flat)
            first = int(mx.argmax(logits, axis=-1).item())
            return (mx.array([first], mx.int32), mx.array([len(prompt)], mx.int32),
                    mx.array([len(prompt)+1], mx.int32), *flat), first
        # One ordinary32 warm request: no native capture or feedback.
        warm, warm_token = setup(base_prompt); warm_flat = warm[3:]; warm_ids = [warm_token]
        for i in range(31):
            pos = len(base_prompt) + i
            ctx = ForwardContext("decode", ctx_layers(warm_flat), positions=mx.array([pos],mx.int32),
                                 lens=mx.array([pos+1],mx.int32), max_len=pos+1)
            with baseline_decode():
                logits = model(mx.array([warm_token], mx.int32), ctx); warm_flat = state_of(ctx)
                pred = mx.argmax(logits,axis=-1); mx.eval(pred,*warm_flat)
            warm_token = int(pred.item()); warm_ids.append(warm_token)
        assert warm_ids == w["continuation_ids"]
        del warm, warm_flat, ctx, logits, pred
        mx.synchronize(); resources.sample("after-warm32", loaded=True, mx=mx)
        # Additional single ordinary decode solely qualifies actual norm-output layout.
        # It is outside capture, the 93 numerical steps, and any future timing.
        layout_seed,layout_first=setup(base_prompt);layout_x={}
        with baseline_decode():
            def retain_layout_x(self,x,*a,**kw):
                assert id(self) not in layout_x
                layout_x[id(self)]=x
                return legacy_decode(self,x,*a,**kw)
            GDN._decode_dispatch=retain_layout_x
            try:
                layout_ctx=ForwardContext("decode",ctx_layers(layout_seed[3:]),positions=layout_seed[1],lens=layout_seed[2],max_len=len(base_prompt)+1)
                layout_logits=model(layout_seed[0],layout_ctx)
                mx.eval(layout_logits,*state_of(layout_ctx),*layout_x.values());mx.synchronize()
                assert int(mx.argmax(layout_logits,axis=-1).item())==w["continuation_ids"][1]
            finally:GDN._decode_dispatch=legacy_decode
        assert set(layout_x)==set(map(id,gdns)) and len(layout_x)==48
        from tutorial_code.miniengine.model.layers import RMSNorm
        assert all(type(layer.input_layernorm) is RMSNorm for layer in model.layers)
        layout_kernel=mx.fast.metal_kernel(name="v68_prepare_observed_live_strides",input_names=["value"],output_names=["descriptor"],source=(D/"layout_probe.metal").read_text(),ensure_row_contiguous=False)
        layout_rows=[]
        def observed_layout(value,label):
            descriptor,=layout_kernel(inputs=[value],grid=(1,1,1),threadgroup=(1,1,1),output_shapes=[(4,)],output_dtypes=[mx.uint64])
            mx.eval(descriptor);v=descriptor.tolist();assert v[:2]==list(value.shape)
            expected=[value.shape[1],1];contiguous=v[3]==1 and (value.shape[0]==1 or v[2]==expected[0])
            row={"label":label,"shape":v[:2],"dtype":str(value.dtype),"actual_strides_elements":v[2:],"row_contiguous_derived_singleton_aware":contiguous,"internal_MLX_flags_observed":False,"ensure_row_contiguous":False}
            layout_rows.append(row);save();assert contiguous,("layout STOP",row);return tuple(v[2:])
        prepared=[]
        receipt["four_projection_preparation"]={"status":"PREPARING","extra_ordinary_decode_steps":1,"included_in_93_steps":False,"layout_rows":layout_rows,"prepared":prepared,"original384_target_arrays":target_before}
        save()
        for layer_index,layer in enumerate(model.layers):
            if layer.is_attn:continue
            g=layer.linear_attn;operand_strides=[]
            for projection,op in zip(("qkv","z","b","a"),g._four_projection_ops()):
                for suffix,value in (("weight",op.weight),("scales",op.scales)):
                    label=f"parameter.layers.{layer_index}.linear_attn.in_proj_{projection}.{suffix}"
                    holder=shared(value,label);target_holders.append(holder)
                    target_before.append({**holder.meta,"sha256":hashlib.sha256(holder.bytes_view).hexdigest()})
                    operand_strides.append(observed_layout(value,label))
            x=layout_x[id(g)];assert x.shape==(1,5120) and x.dtype==mx.bfloat16
            x_strides=observed_layout(x,f"layer.{layer_index}.actual_input_layernorm_X")
            xholder=shared(x,f"layer.{layer_index}.layout_X");x_before=hashlib.sha256(xholder.bytes_view).hexdigest()
            active=mx.get_active_memory()
            g.prepare_four_projection(observed_weight_scale_strides=operand_strides,observed_x_strides=x_strides,runtime_version=versions["mlx"],device_name=info["device_name"],architecture=info["architecture"],input_layout_contract="qualified-original-input-layernorm-B1")
            assert mx.get_active_memory()==active and g._four_projection_enabled is False
            helper=g._four_projection_helper;ops=tuple(v for op in g._four_projection_ops() for v in (op.weight,op.scales))
            assert all(a is b for a,b in zip(helper.operands,ops)) and hashlib.sha256(xholder.bytes_view).hexdigest()==x_before
            prepared.append({"layer":layer_index,"helper_id":id(helper),"original_operand_ids":[id(v) for v in ops],"x_strides":list(x_strides),"observed_weight_scale_strides":[list(v) for v in operand_strides],"mlx_active_before_prepare":active,"mlx_active_after_prepare":mx.get_active_memory(),"same_original_handles":True,"X_readonly":True})
        assert len(prepared)==48 and len(target_holders)==384 and len(layout_rows)==432
        assert sum(h.nbytes for h in target_holders)==2278195200 and target_fingerprints()==target_before
        assert initial_parameter_identity==[(name,id(value)) for name,value in tree_flatten(model.parameters())] and len(initial_parameter_identity)==1349
        for g in gdns:g.set_four_projection(True)
        receipt["four_projection_preparation"]={"status":"READY_ACTUAL_LAYOUT_PREPARED","extra_ordinary_decode_steps":1,"included_in_93_steps":False,"purpose":"independent actual48 original norm-output strides; no performance timing","layout_rows":layout_rows,"prepared":prepared,"original384_target_arrays":target_before,"readonly_after_prepare":True,"target_payload_bytes":2278195200,"extra_packed_weight_bytes":0,"parameter_owners":1349,"same_parameter_identity_after_prepare":True,"source_contract":"unchanged original RMSNorm→norm_metal output_shapes=[x.shape],linear row*D stores; restricted engine path, arbitrary externalX not qualified","target_parameter_holders_retained":True}
        save();del layout_seed,layout_ctx,layout_logits,layout_x,x,xholder
        mx.synchronize();resources.sample("after-four-projection-layout-prepare",loaded=True,mx=mx)
        capture_seed, first = setup(base_prompt); assert first == w["continuation_ids"][0]
        mx.eval(*capture_seed); mx.synchronize()
        for name, value in tree_flatten(model.parameters()):
            h = shared(value, "parameter." + name); parameter_holders.append(h)
            native.call("v33_pin", h.address, h.nbytes, h.label.encode())
        assert len(parameter_holders) == 1349
        labels_in = ("tokens", "positions", "lens") + tuple(f"state.{i}" for i in range(128))
        labels_out = ("logits",) + tuple(f"state.{i}" for i in range(128)) + ("prediction", "next_positions", "next_lens")
        seed_holders = [shared(v,n) for v,n in zip(capture_seed,labels_in)]
        seed_baselines = [h.raw() for h in seed_holders]  # once at capture, never per-step
        capture_ctx = ForwardContext("decode", ctx_layers(capture_seed[3:]), positions=capture_seed[1], lens=capture_seed[2], max_len=CAP)
        mx.eval(*(v for cache,layer in zip(capture_ctx.caches,model.layers) for v in ((cache.K,cache.V) if layer.is_attn else (cache.conv,cache.S))))
        mx.synchronize(); resources.sample("seed-ready-before-capture", loaded=True, mx=mx)
        native.call("v33_begin"); armed = True
        with capture_norm_engagement():
            logits = model(capture_seed[0], capture_ctx)
            captured = outputs(logits, state_of(capture_ctx), capture_seed[1], capture_seed[2]); assert len(captured) == 132
            mx.eval(*captured); mx.synchronize()
        native.call("v33_end_build"); armed = False
        assert len(four_capture_calls)==48 and set(four_capture_calls)==set(map(id,gdns))
        assert len(projection_capture_calls)==48 and {r["helper_id"] for r in projection_capture_calls}=={id(g._four_projection_helper) for g in gdns}
        from tutorial_code.miniengine.kernels.v66_four_projection.four_projection import FourProjection
        receipt["B_four_projection_capture_engagement"]={"calls":48,"distinct_GDN_objects":48,"all48_enabled":all(g._four_projection_enabled for g in gdns),"projection_calls":projection_capture_calls,"four_output_consumption":"unchanged conv uses qkv; original output norm uses z; gates use b/a; whole132 numerical comparison","observers_removed_before_replay":GDN._decode_conv_norm_reuse_four_projection is candidate_four and FourProjection.project.__name__=="project"}
        assert receipt["B_four_projection_capture_engagement"]["observers_removed_before_replay"]
        for value,label in zip(captured,labels_out):
            h = shared(value,label); output_holders.append(h)
            native.call("v33_register_output",h.address,h.nbytes,label.encode())
        for i,h in enumerate(seed_holders):
            native.call("v34_register_input",h.address,h.nbytes,(1,2,3)[i] if i<3 else 0,h.label.encode())
        for i,h in enumerate(seed_holders):
            output_index = (129,130,131)[i] if i<3 else i-2
            other=output_holders[output_index]; assert h.nbytes==other.nbytes
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
            pingpong_requested.append({"layer":layer_index,"input_label":input_state.label,
                                       "output_label":output_state.label,"input_address":input_state.address,
                                       "output_address":output_state.address,"bytes":input_state.nbytes})
        assert len(pingpong_requested)==48
        receipt["pingpong_requested"]=pingpong_requested
        native.call("v34_seal_feedback",128,CAP)
        native.call("v38_enable_icb")
        receipt["ICB_enable_called_after_feedback_seal"]=True
        output_baselines=[h.raw() for h in output_holders]  # once at capture only
        receipt["capture"]={"inputs":[metadata_finite(np,h) for h in seed_holders],
                            "outputs":[metadata_finite(np,h) for h in output_holders],
                            "native":native.report("native-sealed"),"parameter_arrays":len(parameter_holders)}
        assert receipt["capture"]["native"]["status"] == "OK"
        fused_pso=[v for v in receipt["capture"]["native"]["feedback"]["icb"]["PSO_records"] if "v66_gdn_four_original_pointers_r4" in v["function"]]
        assert 1<=len(fused_pso)<=48 and all(v["actual_supports_ICB"] for v in fused_pso)
        buffer_rows=receipt["capture"]["native"]["buffers"];binding_proof=[]
        parameter_by_name={v.label:v for v in parameter_holders}
        for holder in target_holders:
            owner=parameter_by_name[holder.label];assert owner.original is holder.original and owner.address==holder.address and owner.nbytes==holder.nbytes
            matches=[v for v in buffer_rows if v["used"] and v["zero_copy"] and v["original_contents"]<=holder.address and holder.address+holder.nbytes<=v["original_contents"]+v["length"] and v["role"].startswith(holder.label+f" full-range={holder.address}+{holder.nbytes} binding-delta=0 readonly:")]
            assert len(matches)==1 and "v66_gdn_four_original_pointers_r4" in matches[0]["role"],("unproven fused original weight range",holder.label,matches)
            binding_proof.append({"parameter":holder.label,"original_object_id":id(holder.original),"address":holder.address,"bytes":holder.nbytes,"actual_original_buffer":matches[0]["original_buffer"],"role":matches[0]["role"],"full_original_parameter_range_readonly":True})
        assert len(binding_proof)==384
        receipt["four_projection_native_binding_proof"]={"fused_PSOs":fused_pso,"Python_capture_calls":48,"original_target_ranges":binding_proof,"buffer_argument_count_source_proof":13,"individual_argument_reflection":"NOT_OBSERVED; native40 report omits per-argument table; source9inputs+4outputs,noimplicitdescriptorbuffers"}
        assert_icb(receipt["capture"]["native"],[])
        fb = receipt["capture"]["native"]["feedback"]
        assert fb["sealed"] and (fb["input_count"],fb["output_count"],fb["link_count"],fb["state_count"],fb["capacity"]) == (131,132,131,128,CAP)
        assert fb["CPU_full_state_restore_per_step"] is False
        pp=fb["pingpong"]
        assert (pp["requested_count"],pp["eligible_count"],pp["staged_link_count"])==(48,48,83)
        resources.sample("after-capture-seal",loaded=True,mx=mx); save()
        total_steps = 0
        for name,prompt,canonical in cases:
            seed_values,first = (capture_seed,first) if name=="base" else setup(prompt)
            mx.eval(*seed_values); mx.synchronize()
            case_holders=[shared(v,n) for v,n in zip(seed_values,labels_in)]
            case_baselines=[h.raw() for h in case_holders]  # once per request only
            changed_inputs = [i for i,(old,new) in enumerate(zip(seed_holders,case_holders))
                              if native.lib.memcmp(old.address,new.address,old.nbytes) != 0]
            if canonical:
                assert changed_inputs == [], ("base/reset seed drift",changed_inputs)
            else:
                assert any(i >= 3 for i in changed_inputs), "changed prefill did not alter cache inputs"
            resources.sample("before-request-"+name,loaded=True,mx=mx)
            native.call("v34_reset")
            for original,current in zip(seed_holders,case_holders):
                assert original.nbytes==current.nbytes and original.meta["shape"]==current.meta["shape"] and original.meta["dtype"]==current.meta["dtype"]
                native.call("v34_write_input",original.address,original.nbytes,ctypes.c_void_p(current.address))
            ids_a,ids_b=[first],[first]
            case={"name":name,"prompt_ids":prompt,"prompt_length":len(prompt),"A_ids":ids_a,"B_ids":ids_b,
                  "seed_upload_calls":131,"seed_arrays":131,"steps":[],"first_token":first,
                  "changed_seed_inputs_vs_capture":changed_inputs}
            receipt["requests"].append(case); receipt["stage"]={"request":name,"phase":"seed-loaded"}; save()
            if canonical: assert first==w["continuation_ids"][0] and first not in eos
            a_flat=tuple(v[:] for v in seed_values[3:]); a_token=first
            for step in range(1,32):
                if a_token in eos: break  # Changed prompt honors actual greedy EOS.
                position=len(prompt)+step-1; length=position+1
                retained=[shared(v,f"A.input.{i}") for i,v in enumerate(a_flat)]
                before=[hashlib.sha256(h.bytes_view).hexdigest() for h in retained]
                positions,lens=mx.array([position],mx.int32),mx.array([length],mx.int32)
                ctx=ForwardContext("decode",ctx_layers(a_flat),positions=positions,lens=lens,max_len=length)
                with baseline_decode():
                    logits_a=model(mx.array([a_token],mx.int32),ctx)
                    a_state=state_of(ctx); expected_values=outputs(logits_a,a_state,positions,lens)
                    mx.eval(*expected_values); mx.synchronize()
                assert GDN._decode_conv_norm_reuse_cache is candidate_norm
                assert before==[hashlib.sha256(h.bytes_view).hexdigest() for h in retained], "reference retained state changed"
                expected=[shared(v,n) for v,n in zip(expected_values,labels_out)]
                native.call("v34_replay_step"); total_steps+=1
                equal=[]
                archived_step=archived_cases[name]["steps"][step-1]
                assert archived_step["step"]==step and archived_step["A_exact_L"]==length
                archived_outputs={r["label"]:r for r in archived_step["outputs"]}
                assert len(archived_outputs)==132
                for original,reference in zip(output_holders,expected):
                    assert original.meta["shape"]==reference.meta["shape"] and original.meta["dtype"]==reference.meta["dtype"] and original.nbytes==reference.nbytes
                    sig=metadata_finite(np,reference)
                    archived=archived_outputs[reference.label]
                    assert sig["shape"]==archived["shape"] and sig["dtype"]==archived["dtype"] and sig["nbytes"]==archived["bytes"]
                    assert sig["sha256"]==archived["A_sha256"],("A differs from actualsavedV38 oldbaseline",name,step,reference.label)
                    gold44=actual44_cases[name]["steps"][step-1]["outputs"][len(equal)]
                    assert gold44["label"]==reference.label and gold44["shape"]==sig["shape"] and gold44["dtype"]==sig["dtype"] and gold44["bytes"]==sig["nbytes"] and gold44["A_sha256"]==sig["sha256"],("A differs from actual44",name,step,reference.label)
                    native.call("v34_compare_output",original.address,original.nbytes,ctypes.c_void_p(reference.address))
                    equal.append({"label":reference.label,"shape":sig["shape"],"dtype":sig["dtype"],
                                  "bytes":sig["nbytes"],"finite":True,"strict_bytes_equal":True,"A_sha256":sig["sha256"]})
                # Only 4B prediction is copied; it is observed, never uploaded to B.
                target=ctypes.c_int32()
                native.call("v34_copy_output",output_holders[129].address,4,ctypes.byref(target))
                a_token=int(expected_values[129].item()); b_token=int(target.value)
                assert a_token==b_token
                if canonical: assert a_token==w["continuation_ids"][step] and a_token not in eos
                ids_a.append(a_token);ids_b.append(b_token);a_flat=a_state
                ro={"seed":originals_readonly(native,seed_holders,seed_baselines,name+".capture-seed"),
                    "capture_outputs":originals_readonly(native,output_holders,output_baselines,name+".capture-output"),
                    "request_seed":originals_readonly(native,case_holders,case_baselines,name+".request-seed")}
                event={"step":step,"A_exact_L":length,"B_fixed_L":CAP,"A_token":a_token,"B_token":b_token,
                       "outputs":equal,"originals_readonly":ro,"reference_input_digest_readonly":True,
                       "native_prediction_copied_bytes":4,"seed_uploads_this_step":0,
                       "A_saved_V38_hash_checks":132}
                if step%4==0:
                    event["resources"]=resources.sample(name+f"-decode-{step:02d}",loaded=True,mx=mx)
                case["steps"].append(event);receipt["stage"]={"request":name,"step":step};save()
                print(json.dumps({"event":"NATIVE_DECODE_EQUAL","request":name,"step":step}),flush=True)
            assert ids_a==ids_b
            if canonical: assert ids_a==w["continuation_ids"] and len(case["steps"])==31
            case.update(status="PASS_REQUEST_STRICT",terminated_by_EOS=ids_a[-1] in eos,
                        native=native.report("native-after-"+name))
            assert_icb(case["native"],[len(r["steps"]) for r in receipt["requests"]])
            fb=case["native"]["feedback"]
            assert fb["sealed"] and fb["request_steps"]==len(case["steps"]) and fb["total_steps"]==total_steps
            assert fb["resets"]==len(receipt["requests"]) and fb["request_seed_uploaded_ranges"]==131
            assert fb["CPU_seed_upload_bytes_total"]==sum(h.nbytes for h in seed_holders)*len(receipt["requests"])
            assert fb["CPU_output_compare_bytes_total"]==sum(h.nbytes for h in output_holders)*total_steps
            assert fb["CPU_full_state_restore_per_step"] is False
            pp=fb["pingpong"]
            assert (pp["requested_count"],pp["eligible_count"],pp["staged_link_count"])==(48,48,83)
            resources.sample("after-"+name,loaded=True,mx=mx);save()
        assert all(g._gates_compile_enabled and callable(g._compiled_decode_gates) for g in gdns)
        assert all(g._conv_norm_reuse_cache_enabled is True for g in gdns)
        assert GDN._decode_dispatch is candidate_decode and GDN._decode_conv_norm_reuse_cache is candidate_norm
        receipt["A_baseline_temporary_bind_calls"]=len(baseline_calls)
        receipt["A_baseline_restore_final"]=True
        receipt["actual_imported_sources"]={}
        for name,mod in tuple(sys.modules.items()):
            if name.startswith("tutorial_code.miniengine") and getattr(mod,"__file__",None):
                path=str(Path(mod.__file__).resolve());assert path in pins and sha(path)==pins[path]
                receipt["actual_imported_sources"][name]=path
        assert initial_parameter_identity==[(name,id(value)) for name,value in tree_flatten(model.parameters())]
        receipt["same_parameter_identity_count"]=len(initial_parameter_identity)
        receipt["same_parameter_identity_preserved"]=True
        assert total_steps==93 and all(len(r["steps"])==31 and r["A_ids"]==actual44_cases[r["name"]]["A_ids"] for r in receipt["requests"])
        assert all(g._four_projection_enabled and g._four_projection_helper is not None for g in gdns)
        assert target_fingerprints()==target_before
        receipt["four_projection_target384_readonly_final"]=True
        receipt["A_four_OFF_legacy_dispatch_observations"]=baseline_calls
        source_check(library)
        receipt.update(status="PASS_V68_FOUR_ORIGINAL_POINTER_GDN_FULL_ICB_BASE_RESET_CHANGED_STRICT",stage="complete",native_steps=total_steps,
                       source_pins_verified_final=True,gates_callable_count=48,four_projection_ON=48,strict_output_checks=93*132)
        save();return 0
    except BaseException as error:
        receipt.update(status="FAIL_STOP",error=repr(error),traceback=traceback.format_exc(),capture_armed=armed)
        if native is not None:
            try:native.report("native-on-failure")
            except BaseException as report_error:receipt["native_report_error"]=repr(report_error)
        save();raise
    finally:
        try:
            if mx is not None:mx.synchronize()
            if resources is not None:resources.sample("finally-before-exit",loaded=loaded,mx=mx)
        except BaseException as error:
            receipt.update(status="FAIL_STOP",finally_error=repr(error));save();raise
        save()


if __name__=="__main__":raise SystemExit(main())
