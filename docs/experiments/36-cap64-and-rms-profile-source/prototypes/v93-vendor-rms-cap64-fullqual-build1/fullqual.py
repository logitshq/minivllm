"""Root-only V93 vendor plainRMS profile physicalCAP64; independent raw-gamma A vs native B; old gold diagnostic only; inert never imports MLX."""
import argparse
from contextlib import contextmanager, nullcontext
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
ROOT = P / "candidate-v92-vendor-rms-leaf-build1-root"
OLD_PROFILE_QUAL = P / "v23-root-control-build1/fullmodel-v38native-attempt1/child-output/receipt.json"
OLD_PROFILE_QUAL_SHA = "c57e7625baa5a4ce450fc7306233e766f1efd71636713080d748780624542c8a"
LIB = P / "v89-native-capacity64-build1/libv89replay.dylib"
VENDOR_MANIFEST = P / "v92-vendor-rms-leaf-build1/candidate-manifest.json"
VENDOR_FACTORY = P / "v92-vendor-rms-leaf-build1/vendor_reference.py"
VENDOR_PINS = P / "v92-vendor-rms-leaf-build1/source-pins.json"
CAP = 64
CANONICAL_CAP = 256
ACTUAL44 = P / "v23-root-control-build1/fullmodel-v44native-attempt1/child-output/receipt.json"
ACTUAL44_SHA = "1dcfc41e7779c575602bd6c73bae4e4800664db42e1ee87e75414335bdc45b6d"
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
    assert library==LIB and str(library) in pins,"unbound native library"
    assert sha(VENDOR_MANIFEST)=="9a5c20c8ad321a9218f11f2816e5e0a4c519da7fa17a727cb3c1336e2ad0a36d"
    assert sha(VENDOR_FACTORY)=="78f11edf4ecded35e3cec7e660f10a672ae64f2301c39f02052093729f41a55d"
    assert sha(VENDOR_PINS)=="afcfd3486a82216c64d83f67294c6a7bc994a1097939aa3b822a8a1b6c12b03b"
    assert all(pins[path]==digest for path,digest in json.loads(VENDOR_PINS.read_text()).items())
    assert sha(OLD_PROFILE_QUAL)==OLD_PROFILE_QUAL_SHA and sha(ACTUAL44)==ACTUAL44_SHA
    assert json.loads(ACTUAL44.read_text())["status"]=="PASS_NATIVE_CONV_ACTIVATED_NORMREUSE_CACHE_ICB_BASE_RESET_CHANGED_VS_ACCEPTED25_STRICT"
    proof=json.loads((P/"v89-native-capacity64-build1/source-proof.json").read_text())
    assert proof["inverse_patch_exact_frozen40"] is True and proof["allowed_capacities"]==[64,256]
    expected = json.loads((D / "candidate-manifest.json").read_text())["files"]
    expected = {k.removeprefix("tutorial_code/miniengine/"): v for k, v in expected.items()}
    package = ROOT / "tutorial_code/miniengine"
    actual = {str(f.relative_to(package)): sha(f) for f in package.rglob("*.py")}
    assert actual == expected and len(actual) == 27
    tiny=json.loads((P/"v23-root-control-build1/fullmodel-v40smoke-attempt2/child-output/receipt.json").read_text())
    assert tiny["status"]=="PASS_TINY_V40_NORMREUSE_CACHE_ICB_PINGPONG_3REQUESTS_15STEPS" and tiny["strict_output_checks"]==75
    local44_path=P/"v23-root-control-build1/numeric-v44-attempt1/child-output/receipt.json"
    assert sha(local44_path)=="1c22f2e29c78b885600c1be4a06fac357154a7eaa2ee82aa31b284921ff12392"
    local44=json.loads(local44_path.read_text())
    assert local44["status"]=="PASS_V44_ACTIVATED_NORMREUSE_CACHE_STRICT_BYTES_9REAL_6SYNTHETIC" and local44["strict_output_checks"]==30
    actual89=P/"v23-root-control-build1/fullmodel-v89native-attempt1/child-output/receipt.json"
    assert sha(actual89)=="40428c19e12c69c80209885f58246b585d0c0659c3890ae2396f5f7310143e86"
    assert json.loads(actual89.read_text())["status"]=="PASS_V89_PHYSICAL_CAP64_CANONICAL256_NATIVE_ICB_BASE_RESET_CHANGED_OLD38_44_STRICT"
    release=P/"v23-root-control-build1/fullmodel-v89native-attempt1/external-release.json"
    assert sha(release)=="ea7cd3efd15404a89829be54368becb54cd65e17327e16232adf401ff2a50fdc"
    r=json.loads(release.read_text());assert r["GPU_released"] and r["tool_exit"]==0
    assert all(row["returncode"]==1 and row["stdout"]==row["stderr"]=="" for row in r["strict_absence"])
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
        if name=="v34_seal_feedback":
            assert len(args)==2 and all(type(v) is int and 0<=v<=0xffffffff for v in args),"uint32 ABI overflow/type rejected before ctypes narrowing"
            assert args[1] in (64,256),"unknown capacity rejected before ctypes narrowing"
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


def canonical_output(raw, shape, dtype, is_kv, valid_length):
    """CPU-only byte canonicalization. Never creates/pads an MLX tensor."""
    raw=bytes(raw);shape=list(shape)
    if not is_kv:return raw,shape
    assert shape==[1,4,CAP,256] and dtype=="mlx.core.bfloat16"
    assert isinstance(valid_length,int) and 0<valid_length<=CAP
    element_bytes=2;head_bytes=CAP*256*element_bytes;valid_bytes=valid_length*256*element_bytes
    assert len(raw)==4*head_bytes
    padding=bytes((CANONICAL_CAP-CAP)*256*element_bytes)
    heads=[]
    for head in range(4):
        current=raw[head*head_bytes:(head+1)*head_bytes]
        assert current[valid_bytes:]==bytes(head_bytes-valid_bytes),("physical KV tail nonzero",head,valid_length)
        heads.append(current+padding)
    result=b"".join(heads)
    assert len(result)==1*4*CANONICAL_CAP*256*element_bytes
    return result,[1,4,CANONICAL_CAP,256]


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
    baseline_qualification=json.loads(OLD_PROFILE_QUAL.read_text())
    assert baseline_qualification["status"]=="PASS_NATIVE_ICB_S_PINGPONG_BASE_RESET_CHANGED_STRICT"
    assert baseline_qualification["declared_prompts"]=={n:p for n,p,_ in cases}
    archived_cases={r["name"]:r for r in baseline_qualification["requests"]}
    gold44=json.loads(ACTUAL44.read_text());archive44={r["name"]:r for r in gold44["requests"]}
    assert gold44["declared_prompts"]==baseline_qualification["declared_prompts"]
    assert all(len(r["A_ids"])==32 and r["A_ids"]==r["B_ids"]==archive44[name]["A_ids"]==archive44[name]["B_ids"] for name,r in archived_cases.items())
    assert max(len(prompt)+31 for _,prompt,_ in cases)<=CAP
    assert all(len(r["steps"])==31 for r in archived_cases.values())
    if args.inert:
        assert not any(n == "mlx" or n.startswith("mlx.") for n in sys.modules)
        print(json.dumps({"status": "PASS_CPU_INERT", "pins": len(pins), "MLX_imported": False,
                          "GPU_used": False, "numerical_profile":"vendor RMSNorm leaf rounding", "old_profile_pass":False, "requests": [[n, len(p)] for n, p, _ in cases],
                          "inputs": 131, "outputs": 132, "feedback_links": 131, "GDN_S_pingpong_requested": 48, "staged_links_required": 83, "required_actual_ICB_banks": 2, "physical_KV_capacity": CAP, "CPU_canonical_KV_capacity":CANONICAL_CAP}))
        return 0
    assert args.out is not None and args.out.is_absolute()
    args.out.mkdir(exist_ok=False)
    receipt = {"status": "STARTED", "stage": "beforeload", "pid": os.getpid(), "model_loads": 0,
               "source_pins": pins, "native_library": str(library), "fixed_cap": CAP, "requests": [], "numerical_profile":"vendor RMSNorm leaf rounding", "old_profile_pass":False,
               "declared_prompts": {n: p for n, p, _ in cases}, "performance_point": False,
               "ICB_scope": "actual native ICB qualification; unrelated to mx.compile", "adopted": False, "input_upload_scope": "all131 once per request only",
               "B_token_feedback": "captured MLX argmax -> staged GPU blit; no CPU/saved-continuation feeding",
               "old_profile_ID_and_hash_diagnostic_receipt":str(OLD_PROFILE_QUAL),"old_profile_ID_and_hash_diagnostic_receipt_sha256":OLD_PROFILE_QUAL_SHA,
               "actual44_gold_sha256":ACTUAL44_SHA,"canonical_capacity":CANONICAL_CAP,
               "canonicalization":"CPU raw BF16 per-head append-zero only after physical64 readback; no MLX padded256 input/state",
               "scope":"sync fixedCAP64 numerical only, no async/service/performance/adoption"}
    save = lambda: write(args.out / "receipt.json", receipt)
    save(); mx = None; resources = None; native = None; loaded = False; armed = False
    parameter_holders, seed_holders, output_holders = [], [], []
    allocation_classes = None; original_allocation_steps = None
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
        import mlx.nn as nn
        from mlx.utils import tree_flatten
        info = mx.device_info()
        assert mx.default_device() == mx.gpu and info["device_name"] == "Apple M4 Max" and info["architecture"] == "applegpu_g16s"
        previous_limit = mx.set_cache_limit(2 << 30)
        receipt["runtime"] = {"versions": versions, "device": info, "cache_limit_bytes": 2 << 30,
                              "previous_cache_limit_bytes": previous_limit, "environment": {**CONTROLS, "DYLD_INSERT_LIBRARIES": str(library)}}
        sys.path.insert(0, str(ROOT))
        from tutorial_code.miniengine.model.loader import load_model
        from tutorial_code.miniengine.cache import AttnCache, BatchKV, BatchState, DecodeBatch, make_cache
        from tutorial_code.miniengine.context import ForwardContext
        from tutorial_code.miniengine.eos import read_eos_ids
        model, cfg = load_model(w["model_dir"], dtype=mx.bfloat16)
        loaded = True; receipt["model_loads"] = 1; mx.synchronize()
        receipt["afterload"] = resources.sample("afterload-before-any-token", loaded=True, mx=mx)
        initial_parameter_identity=[(name,id(value)) for name,value in tree_flatten(model.parameters())]
        # Allocation metadata only; model/cache source stays frozen. Set BEFORE any token.
        allocation_classes=(AttnCache,DecodeBatch);original_allocation_steps=(AttnCache.step,DecodeBatch.step)
        assert original_allocation_steps==(256,256)
        AttnCache.step=CAP;DecodeBatch.step=CAP
        kv_state_indices={2*i+j for i,layer in enumerate(model.layers) if layer.is_attn for j in (0,1)}
        kv_output_labels={f"state.{i}" for i in kv_state_indices};assert len(kv_output_labels)==32
        receipt["allocation_bucket"]={"original":list(original_allocation_steps),"effective":[AttnCache.step,DecodeBatch.step],"before_any_token":True,"source_files_unchanged":True}
        gdns = [l.linear_attn for l in model.layers if not l.is_attn]
        assert len(model.layers) == 64 and len(gdns) == 48
        for g in gdns:
            g.set_gates_compile(True)
            g.set_conv_norm_reuse_cache(True)
        from tutorial_code.miniengine.model import gdn as gdn_module
        GDN=gdn_module.GatedDeltaNet
        assert all(type(g) is GDN and g._conv_norm_reuse_cache_enabled is True for g in gdns)
        candidate_decode,candidate_norm=GDN._decode_dispatch,GDN._decode_conv_norm_reuse_cache
        # Only plain RMSNorm changes profile; GatedRMSNorm/GDN math stays current44 in BOTH arms.
        vendor=module("v93_vendor_reference",P/"v92-vendor-rms-leaf-build1/vendor_reference.py")
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
            am,bm=metadata_finite(np,Araw),metadata_finite(np,Bcached)
            assert am["sha256"]==e.raw_sha256 and bm["sha256"]==e.cached_sha256
            gamma_holders.extend((Araw,Bcached));cached_gamma_holders.append(Bcached)
            gamma_checks.append({"name":e.name,"checkpoint_key":e.checkpoint_key,"A_raw":am,"B_cached":bm,"strict_bytes_equal":True,"separate_objects":True,"separate_addresses":True,"A_B_ranges_disjoint":True})
        assert len(gamma_holders)==322 and len(cached_gamma_holders)==161
        gamma_baselines=[h.raw() for h in gamma_holders]
        assert initial_parameter_identity==[(name,id(v)) for name,v in tree_flatten(model.parameters())] and len(initial_parameter_identity)==1349
        profile.set_enabled(True);profile.validate_current_bindings()
        receipt["vendor_rms_profile_preparation"]={"metadata":profile.metadata(),"snapshot":profile.snapshot(),"plain_norms":161,"parameter_arrays":1349,"same_parameter_identity":True,"derived_cachedgamma_arrays":161,"independent_raw_gamma_arrays":161,"all161_gamma_pairs_strict":gamma_checks,"prepared_before_any_token":True}
        receipt["gamma_readonly_baseline"]={"arrays":322,"bytes":sum(h.nbytes for h in gamma_holders),"once_only_raw_snapshots":True}
        resources.sample("after-vendor-profile-before-any-token",loaded=True,mx=mx);save()
        baseline_calls=[]
        @contextmanager
        def raw_gamma_reference():
            assert GDN._decode_dispatch is candidate_decode and GDN._decode_conv_norm_reuse_cache is candidate_norm
            baseline_calls.append("independent_checkpoint_raw_gamma_NNRMSNorm_current44GDN")
            with profile.reference_context():yield
            assert GDN._decode_dispatch is candidate_decode and GDN._decode_conv_norm_reuse_cache is candidate_norm
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
        receipt["A_reference_profile"]={"method":"independent checkpoint rawgamma nn.RMSNorm factory/context",
                                       "GDN_math":"same current44 conv+gates+normreuse in A/B; no legacy dispatch binding",
                                       "old_gold_scope":"all96 IDs hard gate; canonical132 hashes diagnostic equality/difference only"}

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
        labels_in = ("tokens", "positions", "lens") + tuple(f"state.{i}" for i in range(128))
        labels_out = ("logits",) + tuple(f"state.{i}" for i in range(128)) + ("prediction", "next_positions", "next_lens")
        receipt["independent_prefill_qualifications"]=[]
        def setup(prompt, reference, label):
            # ALL reference graph construction, eval and Shared views remain inside A context.
            with raw_gamma_reference() if reference else nullcontext():
                caches=make_cache(model)
                logits=model(mx.array(prompt,mx.int32),ForwardContext("prefill",caches),last_only=True)
                batch=DecodeBatch(model);batch.rebuild([],[("v93",caches,len(prompt)+32)])
                flat=tuple(v[:] for v in batch.arrays());assert len(flat)==128
                mx.eval(logits,*flat);first=int(mx.argmax(logits,axis=-1).item())
                assert all(v.shape==(1,4,CAP,256) and v.dtype==mx.bfloat16 for i,v in enumerate(flat) if i in kv_state_indices)
                assert all(c.keys.shape[2]==CAP and c.values.shape[2]==CAP for layer,c in zip(model.layers,caches) if layer.is_attn)
                seed=(mx.array([first],mx.int32),mx.array([len(prompt)],mx.int32),mx.array([len(prompt)+1],mx.int32),*flat)
                mx.eval(*seed);mx.synchronize()
                holders=[shared(v,label+"."+n) for v,n in zip(seed,labels_in)]
                last_logits=shared(logits,label+".prefill_last_logits")
                return {"seed":seed,"first":first,"logits":logits,"holders":holders,"baselines":[h.raw() for h in holders],
                        "last_logits":last_logits,"logits_baseline":last_logits.raw()}
        def independent_prefill_pair(prompt,name):
            A=setup(prompt,True,name+".A");B=setup(prompt,False,name+".B")
            assert len(A["holders"])==len(B["holders"])==131
            checks=[]
            for ah,bh,label in zip(A["holders"],B["holders"],labels_in):
                assert ah.original is not bh.original,("prefill seed handles were shared",name,label)
                assert ah.meta["shape"]==bh.meta["shape"] and ah.meta["dtype"]==bh.meta["dtype"] and ah.nbytes==bh.nbytes
                if native.lib.memcmp(ah.address,bh.address,ah.nbytes)!=0:
                    (args.out/"first-failure-prefill-A.bin").write_bytes(ah.raw())
                    (args.out/"first-failure-prefill-B.bin").write_bytes(bh.raw())
                    write(args.out/"first-failure-prefill.json",{"request":name,"label":label,"A":ah.meta,"B":bh.meta})
                    raise AssertionError(("independent prefill seed differs",name,label))
                am,bm=metadata_finite(np,ah),metadata_finite(np,bh)
                if label in kv_output_labels:
                    canonical_output(ah.bytes_view,ah.meta["shape"],ah.meta["dtype"],True,len(prompt))
                    canonical_output(bh.bytes_view,bh.meta["shape"],bh.meta["dtype"],True,len(prompt))
                checks.append({"label":label,"shape":am["shape"],"dtype":am["dtype"],"bytes":ah.nbytes,"A_sha256":am["sha256"],"B_sha256":bm["sha256"],"strict_bytes_equal":True,"separate_array_handles":True})
            ah,bh=A["last_logits"],B["last_logits"]
            assert ah.meta["shape"]==bh.meta["shape"] and ah.meta["dtype"]==bh.meta["dtype"] and ah.nbytes==bh.nbytes
            if native.lib.memcmp(ah.address,bh.address,ah.nbytes)!=0:
                (args.out/"first-failure-prefill-A.bin").write_bytes(ah.raw())
                (args.out/"first-failure-prefill-B.bin").write_bytes(bh.raw())
                write(args.out/"first-failure-prefill.json",{"request":name,"label":"last_logits","A":ah.meta,"B":bh.meta})
                raise AssertionError(("independent prefill logits differ",name))
            assert A["first"]==B["first"]==archived_cases[name]["A_ids"][0]
            event={"name":name,"prompt_ids":prompt,"A":"independent rawgamma nn.RMSNorm reference_context through eval","B":"preparedgamma normal candidate path",
                   "seed131":checks,"seed131_strict_bytes_equal":True,"A_B_states_retained_separately":True,"B_seed_origin":"own B prefill only",
                   "last_logits":{"A":metadata_finite(np,ah),"B":metadata_finite(np,bh),"strict_bytes_equal":True},"first_token":A["first"],"first_token_matches_old_ID":True}
            receipt["independent_prefill_qualifications"].append(event);save()
            return {"A":A,"B":B,"proof":event}
        # One ordinary32 warm request: no native capture or feedback.
        warm_record=setup(base_prompt,False,"untimed-warm-B");warm,warm_token=warm_record["seed"],warm_record["first"];warm_flat=warm[3:];warm_ids=[warm_token]
        for i in range(31):
            pos = len(base_prompt) + i
            ctx = ForwardContext("decode", ctx_layers(warm_flat), positions=mx.array([pos],mx.int32),
                                 lens=mx.array([pos+1],mx.int32), max_len=pos+1)
            logits = model(mx.array([warm_token], mx.int32), ctx); warm_flat = state_of(ctx)
            pred = mx.argmax(logits,axis=-1); mx.eval(pred,*warm_flat)
            warm_token = int(pred.item()); warm_ids.append(warm_token)
        assert warm_ids == w["continuation_ids"]
        del warm, warm_flat, warm_record, ctx, logits, pred
        mx.synchronize(); resources.sample("after-warm32", loaded=True, mx=mx)
        capture_pair=independent_prefill_pair(base_prompt,"base")
        capture_seed,first=capture_pair["B"]["seed"],capture_pair["B"]["first"];assert first==w["continuation_ids"][0]
        mx.eval(*capture_seed); mx.synchronize()
        for name, value in tree_flatten(model.parameters()):
            h = shared(value, "parameter." + name); parameter_holders.append(h)
            native.call("v33_pin", h.address, h.nbytes, h.label.encode())
        assert len(parameter_holders) == 1349
        for h in cached_gamma_holders:native.call("v33_pin",h.address,h.nbytes,h.label.encode())
        receipt["native_parameter_pins"]={"model_parameter_arrays":1349,"derived_cachedgamma_arrays":161,"total_pins":1510,"reference_rawgamma_pinned_into_B":False}
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
        assert len(norm_capture_calls)==48 and set(norm_capture_calls)==set(map(id,gdns))
        assert len(plain_capture_calls)==161 and set(plain_capture_calls)==plain_norms
        assert PlainRMSNorm.__call__ is candidate_plain
        profile.validate_current_bindings()
        receipt["B_vendor_plain_rms_capture_engagement"]={"calls":161,"distinct_plain_norm_objects":161,"all161_prepared_enabled":True,"counter_removed_before_replay":True,"actual_branch_rows":plain_capture_rows,"GatedRMSNorm_hooked":False,"math_profile":"actual MXfast.rms_norm with independently cached originalgamma"}
        receipt["B_conv_normreuse_cache_capture_engagement"]={"calls":48,"distinct_GDN_objects":48,"normreuse_cache_ON_metadata":True,
                                               "counter_removed_after_capture":GDN._decode_conv_norm_reuse_cache is candidate_norm}
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
        assert all(h.meta["shape"]==[1,4,CAP,256] for i,h in enumerate(seed_holders[3:]) if i in kv_state_indices)
        assert all(h.meta["shape"]==[1,4,CAP,256] for h in output_holders if h.label in kv_output_labels)
        receipt["capture"]={"inputs":[metadata_finite(np,h) for h in seed_holders],
                            "outputs":[metadata_finite(np,h) for h in output_holders],
                            "native":native.report("native-sealed"),"parameter_arrays":len(parameter_holders),"derived_cachedgamma_arrays":161}
        assert receipt["capture"]["native"]["status"] == "OK"
        native_gamma_proof=[]
        for h in cached_gamma_holders:
            rows=[v for v in receipt["capture"]["native"]["buffers"] if v["used"] and v["zero_copy"] and v["original_contents"]<=h.address and h.address+h.nbytes<=v["original_contents"]+v["length"] and v["role"].startswith(h.label+f" full-range={h.address}+{h.nbytes} binding-delta=0 readonly:")]
            assert len(rows)==1,("cachedgamma native readonly fullrange unproven",h.label,rows)
            native_gamma_proof.append({"label":h.label,"address":h.address,"bytes":h.nbytes,"original_buffer":rows[0]["original_buffer"],"role":rows[0]["role"],"full_range_readonly":True})
        assert len(native_gamma_proof)==161
        receipt["derived_cachedgamma161_native_readonly_proof"]=native_gamma_proof
        assert_icb(receipt["capture"]["native"],[])
        fb = receipt["capture"]["native"]["feedback"]
        assert fb["sealed"] and (fb["input_count"],fb["output_count"],fb["link_count"],fb["state_count"],fb["capacity"]) == (131,132,131,128,CAP)
        assert fb["CPU_full_state_restore_per_step"] is False
        pp=fb["pingpong"]
        assert (pp["requested_count"],pp["eligible_count"],pp["staged_link_count"])==(48,48,83)
        resources.sample("after-capture-seal",loaded=True,mx=mx); save()
        total_steps = 0
        old_gold_counts={"equal":0,"different":0,"first_differences":[]}
        receipt["old_profile_canonical132_hash_diagnostic"]=old_gold_counts
        for name,prompt,canonical in cases:
            seed_pair=capture_pair if name=="base" else independent_prefill_pair(prompt,name)
            seed_values,first=seed_pair["B"]["seed"],seed_pair["B"]["first"]
            a_seed_values=seed_pair["A"]["seed"]
            case_holders=seed_pair["B"]["holders"];case_baselines=seed_pair["B"]["baselines"]
            a_seed_holders=seed_pair["A"]["holders"];a_seed_baselines=seed_pair["A"]["baselines"]
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
                  "changed_seed_inputs_vs_capture":changed_inputs,"independent_prefill_proof":name,"B_seed_source":"own independent B prefill"}
            receipt["requests"].append(case); receipt["stage"]={"request":name,"phase":"seed-loaded"}; save()
            assert first==archived_cases[name]["A_ids"][0] and first not in eos
            a_flat=tuple(v[:] for v in a_seed_values[3:]);a_token=seed_pair["A"]["first"]
            for step in range(1,32):
                if a_token in eos: break  # Changed prompt honors actual greedy EOS.
                position=len(prompt)+step-1; length=position+1
                with raw_gamma_reference():
                    retained=[shared(v,f"A.input.{i}") for i,v in enumerate(a_flat)]
                    before=[hashlib.sha256(h.bytes_view).hexdigest() for h in retained]
                    positions,lens=mx.array([position],mx.int32),mx.array([length],mx.int32)
                    ctx=ForwardContext("decode",ctx_layers(a_flat),positions=positions,lens=lens,max_len=length)
                    logits_a=model(mx.array([a_token],mx.int32),ctx)
                    a_state=state_of(ctx);expected_values=outputs(logits_a,a_state,positions,lens)
                    mx.eval(*expected_values);mx.synchronize()
                    expected=[shared(v,n) for v,n in zip(expected_values,labels_out)]
                assert GDN._decode_conv_norm_reuse_cache is candidate_norm and PlainRMSNorm.__call__ is candidate_plain
                assert before==[hashlib.sha256(h.bytes_view).hexdigest() for h in retained],"reference retained state changed"
                native.call("v34_replay_step"); total_steps+=1
                equal=[]
                archived_step=archived_cases[name]["steps"][step-1]
                assert archived_step["step"]==step and archived_step["A_exact_L"]==length
                archived_outputs={r["label"]:r for r in archived_step["outputs"]}
                assert len(archived_outputs)==132
                for original,reference in zip(output_holders,expected):
                    assert original.meta["shape"]==reference.meta["shape"] and original.meta["dtype"]==reference.meta["dtype"] and original.nbytes==reference.nbytes
                    sig=metadata_finite(np,reference)
                    is_kv=reference.label in kv_output_labels
                    a_canonical,canonical_shape=canonical_output(reference.bytes_view,sig["shape"],sig["dtype"],is_kv,length)
                    canonical_digest=hashlib.sha256(a_canonical).hexdigest()
                    archived=archived_outputs[reference.label]
                    archived44=next(v for v in archive44[name]["steps"][step-1]["outputs"] if v["label"]==reference.label)
                    assert archived44==archived,("old gold disagreement",name,step,reference.label)
                    assert canonical_shape==archived["shape"] and sig["dtype"]==archived["dtype"] and len(a_canonical)==archived["bytes"]
                    old_profile_equal=canonical_digest==archived["A_sha256"]
                    old_gold_counts["equal" if old_profile_equal else "different"]+=1
                    if not old_profile_equal and len(old_gold_counts["first_differences"])<8:
                        old_gold_counts["first_differences"].append({"request":name,"step":step,"label":reference.label,"new_A_sha256":canonical_digest,"old_A_sha256":archived["A_sha256"]})
                    # First compare EVERY physical output byte against live independent A.
                    native.call("v34_compare_output",original.address,original.nbytes,ctypes.c_void_p(reference.address))
                    if is_kv:
                        # CPU-only canonicalization; never upload old256 state into native arena.
                        target=ctypes.create_string_buffer(original.nbytes)
                        native.call("v34_copy_output",original.address,original.nbytes,ctypes.cast(target,ctypes.c_void_p))
                        b_canonical,b_shape=canonical_output(target.raw,original.meta["shape"],original.meta["dtype"],True,length)
                        if b_shape!=canonical_shape or b_canonical!=a_canonical:
                            (args.out/"first-failure-A-canonical.bin").write_bytes(a_canonical)
                            (args.out/"first-failure-B-canonical.bin").write_bytes(b_canonical)
                            raise AssertionError(("B canonical CAP256 bytes differ",name,step,reference.label))
                    equal.append({"label":reference.label,"shape":canonical_shape,"dtype":sig["dtype"],
                                  "bytes":len(a_canonical),"finite":True,"strict_bytes_equal":True,"A_sha256":canonical_digest,
                                  "physical_shape":sig["shape"],"physical_bytes":sig["nbytes"],
                                  "physical_A_sha256":sig["sha256"],"canonical_B_sha256":canonical_digest,
                                  "physical_tail_zero_checked":is_kv,"canonical_CPU_zero_fill":is_kv,"old_profile_sha256":archived["A_sha256"],"old_profile_strict_equal":old_profile_equal})
                # Only 4B prediction is copied; it is observed, never uploaded to B.
                target=ctypes.c_int32()
                native.call("v34_copy_output",output_holders[129].address,4,ctypes.byref(target))
                a_token=int(expected_values[129].item()); b_token=int(target.value)
                assert a_token==b_token
                assert a_token==archived_cases[name]["A_ids"][step] and a_token not in eos
                ids_a.append(a_token);ids_b.append(b_token);a_flat=a_state
                ro={"seed":originals_readonly(native,seed_holders,seed_baselines,name+".capture-seed"),
                    "capture_outputs":originals_readonly(native,output_holders,output_baselines,name+".capture-output"),
                    "request_seed":originals_readonly(native,case_holders,case_baselines,name+".B-request-seed"),
                    "A_request_seed":originals_readonly(native,a_seed_holders,a_seed_baselines,name+".A-request-seed")}
                event={"step":step,"A_exact_L":length,"B_fixed_L":CAP,"A_token":a_token,"B_token":b_token,
                       "outputs":equal,"originals_readonly":ro,"reference_input_digest_readonly":True,
                       "native_prediction_copied_bytes":4,"numerical_KV_canonical_readback_bytes":4194304,"seed_uploads_this_step":0,
                       "oldV38_canonical_hash_diagnostic_comparisons":132,"oldV44_canonical_hash_diagnostic_comparisons":132,"canonical132_new_A_B_strict_checks":132,"KV_physical64_tail_zero_checks":32}
                if step%4==0:
                    event["resources"]=resources.sample(name+f"-decode-{step:02d}",loaded=True,mx=mx)
                case["steps"].append(event);receipt["stage"]={"request":name,"step":step};save()
                print(json.dumps({"event":"NATIVE_DECODE_EQUAL","request":name,"step":step}),flush=True)
            assert ids_a==ids_b
            assert ids_a==archived_cases[name]["A_ids"] and len(case["steps"])==31
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
            case["gamma322_readonly"]=originals_readonly(native,gamma_holders,gamma_baselines,name+".raw-and-cached-gamma")
            for arm in ("A","B"):
                prefill=seed_pair[arm]
                assert native.same(prefill["last_logits"],prefill["logits_baseline"]),("prefill logits changed",name,arm)
            profile.validate_current_bindings()
            assert PlainRMSNorm.__call__ is candidate_plain
            resources.sample("after-"+name,loaded=True,mx=mx);save()
        assert total_steps==93 and all(len(r["steps"])==31 for r in receipt["requests"])
        assert all(g._gates_compile_enabled and callable(g._compiled_decode_gates) for g in gdns)
        assert all(g._conv_norm_reuse_cache_enabled is True for g in gdns)
        assert GDN._decode_dispatch is candidate_decode and GDN._decode_conv_norm_reuse_cache is candidate_norm
        receipt["A_reference_context_calls"]=len(baseline_calls)
        assert PlainRMSNorm.__call__ is candidate_plain and profile.snapshot()["enabled_count"]==161
        assert len(baseline_calls)==96
        receipt["A_reference_restore_final"]=True
        receipt["actual_imported_sources"]={}
        for name,mod in tuple(sys.modules.items()):
            if name.startswith("tutorial_code.miniengine") and getattr(mod,"__file__",None):
                path=str(Path(mod.__file__).resolve());assert path in pins and sha(path)==pins[path]
                receipt["actual_imported_sources"][name]=path
        assert initial_parameter_identity==[(name,id(value)) for name,value in tree_flatten(model.parameters())]
        receipt["same_parameter_identity_count"]=len(initial_parameter_identity)
        receipt["same_parameter_identity_preserved"]=True
        assert len(receipt["independent_prefill_qualifications"])==3
        receipt["independent_prefill_summary"]={"requests":3,"seed131_strict_checks":393,"last_logits_strict_checks":3,"firsttoken_oldID_checks":3,"separate_A_B_prefill":True,"B_seeded_from_own_prefill_only":True}
        assert old_gold_counts["equal"]+old_gold_counts["different"]==12276
        assert sum(len(r["A_ids"]) for r in receipt["requests"])==sum(len(r["B_ids"]) for r in receipt["requests"])==96
        receipt["new_profile_reference"]={"A":"independent raw checkpoint gamma NN.RMSNorm","B":"prepared cachedgamma MXfast.rms_norm","physical132_strict_checks":12276,"old96IDs_hard_gate":True,"old_canonical_gold":"diagnostic only; no oldprofilePASS assertion"}
        receipt["gamma322_readonly_final"]=originals_readonly(native,gamma_holders,gamma_baselines,"final.raw-and-cached-gamma")
        profile.validate_current_bindings();receipt["vendor_rms_profile_final_snapshot"]=profile.snapshot()
        source_check(library)
        receipt.update(status="PASS_V93_VENDOR_PLAIN_RMS_CAP64_NATIVE_ICB_VS_INDEPENDENT_RAW_GAMMA_STRICT_96_OLD_IDS",stage="complete",native_steps=total_steps,
                       source_pins_verified_final=True,gates_callable_count=48,strict_output_checks=12276,canonical_KV_tail_zero_checks=2976)
        save();return 0
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
            if resources is not None:resources.sample("finally-before-exit",loaded=loaded,mx=mx)
        except BaseException as error:
            receipt.update(status="FAIL_STOP",finally_error=repr(error));save();raise
        save()


if __name__=="__main__":raise SystemExit(main())
