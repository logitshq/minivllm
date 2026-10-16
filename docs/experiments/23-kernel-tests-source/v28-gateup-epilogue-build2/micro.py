#!/usr/bin/env python3
"""Root-only V28 gate/up QMV + typed epilogue numeric and balanced micro. CPU mode never imports MLX."""
import argparse
import copy
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time
import traceback

sys.dont_write_bytecode = True
BASE = Path(__file__).resolve().parent
P = BASE.parent
OLD = P / "v22-nvfp4-qmv-r2-build1"
PYTHON = "/Users/ryanli/code/minivllm/.venv/bin/python"
spec = importlib.util.spec_from_file_location("v26_readonly_v22_support", OLD / "micro.py")
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)  # stdlib-only; __main__ guard prevents execution
from gateup import prepare, K, N

Q, REPEATS, SCHEDULE = old.Q, old.REPEATS, old.SCHEDULE
PRIME = 2048
old.PRIME = PRIME  # The reused block() resolves PRIME in this private imported module.
CONTROLS = old.CONTROLS
sha, write = old.sha, old.write


def identity():
    frozen = json.loads((BASE / "source-manifest.json").read_text())
    for name, digest in frozen["files"].items():
        assert sha(BASE / name) == digest, ("prototype changed", name)
    for rec in frozen["upstream_sources"].values():
        assert sha(rec["path"]) == rec["sha256"], rec["path"]
    index_path = old.CHECKPOINT / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())["weight_map"]
    tensors = []
    for projection in ("gate_proj", "up_proj"):
        for suffix, dtype, shape in [("weight", "U32", (N, K // 8)), ("scales", "U8", (N, K // 16))]:
            keys = [key for key in index if key.endswith(".layers.0.mlp." + projection + "." + suffix)]
            assert len(keys) == 1
            key = keys[0]
            tensors.append(old.tensor_meta(old.CHECKPOINT / index[key], key, dtype, shape))
    config = json.loads((old.CHECKPOINT / "config.json").read_text())
    assert config["quantization"] == Q and config["text_config"]["hidden_size"] == K
    runtime_plan = P / "v18-activation-runtime-identity-build3.json"
    plan = json.loads(runtime_plan.read_text())
    for key in ("required_modules", "source_modules", "native_images"):
        rows = plan[key]["required"] if key == "native_images" else plan[key]
        if isinstance(rows, dict): rows = rows.values()
        for rec in rows:
            assert sha(rec["path"]) == rec["sha256"], rec["path"]
    assert sha(plan["metal_library"]["path"]) == plan["metal_library"]["sha256"]
    return {"driver_sha256": sha(__file__), "prototype_manifest_sha256": sha(BASE / "source-manifest.json"),
            "protocol_sha256": sha(BASE / "protocol.json"), "runtime_plan_sha256": sha(runtime_plan),
            "checkpoint": str(old.CHECKPOINT), "index_sha256": sha(index_path),
            "config_sha256": sha(old.CHECKPOINT / "config.json"), "tensors": tensors,
            "target_tensor_payload_bytes": sum(t["nbytes"] for t in tensors),
            "real_inputs": old.real_metadata(), "mechanism": "two native R4 reductions + typed BF16 epilogue; no weight copy",
            "plan": {"prime": PRIME, "repeats": REPEATS, "schedule": SCHEDULE,
                     "controls": CONTROLS, "power": "AC Power", "timeout_s": 170,
                     "minimum_reduction": .05, "minimum_wins": 5, "full_model": False}}


def gpu(out, ident):
    import numpy as np
    import mlx.core as mx
    import mlx.nn as nn
    sys.path.insert(0, str(P))
    from resource_snapshot_power_v1 import snapshot, deltas, guard
    assert sys.executable == PYTHON and importlib.metadata.version("mlx") == importlib.metadata.version("mlx-metal") == "0.32.3"
    device = mx.metal.device_info()
    assert mx.default_device() == mx.gpu and device["device_name"] == "Apple M4 Max" and device["architecture"] == "applegpu_g16s"
    assert all(os.environ.get(k) == v for k, v in CONTROLS.items())
    assert not any(k.startswith(("DYLD_", "METAL_", "OMLX_", "LITHOS_", "MONOLITH_")) or k in ("PYTHONPATH", "PYTHONHOME") or (k.startswith("MLX_") and k not in CONTROLS) for k in os.environ)
    mx.set_cache_limit(2 << 30)
    write(out / "runtime.json", {"mlx": mx.__file__, "version": "0.32.3", "device": device,
          "controls": {k: os.environ[k] for k in CONTROLS}, "allocator_cache_limit_bytes": 2 << 30,
          "math_policy_override": False, "model_loaded": False, "rows_per_simd": 4, "projection_count": 2, "weights_repacked": False})
    resources, checks, startups, pairs, readonly = [], [], [], [], []

    def probe(label):
        mx.synchronize()
        raw = snapshot(label, os.getpid())
        write(out / ("resource-%02d-raw.json" % len(resources)), raw)
        s = copy.deepcopy(raw)
        s["mlx_memory"] = {"active": mx.get_active_memory(), "cache": mx.get_cache_memory(), "peak": mx.get_peak_memory()}
        resources.append(s); deltas(resources); write(out / "resources.json", resources); guard(resources)
        assert s["power_metadata"]["status"] == "known" and s["power_metadata"]["source"] == "AC Power"
        assert s["processinfo_thermal_metadata"]["status"] == "known" and s["processinfo_thermal_metadata"]["code"] in (0, 1)
        assert s["pressure_level_value"] == 1 and s["exact_owned_pid_metadata"]["status"] == "known"
        assert all(x["owned"] for x in s["known_inference_processes"]), "known foreign inference process"
        assert s["swap_used_mb"] <= resources[0]["swap_used_mb"] and s["vm_counters"]["Swapouts"] == resources[0]["vm_counters"]["Swapouts"]
        return s

    def raw(value):
        return np.asarray(value.view(mx.uint16) if value.dtype == mx.bfloat16 else value).tobytes()

    def fingerprint(value):
        data = raw(value)
        return {"shape": list(value.shape), "dtype": str(value.dtype), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}

    def compare(left, right, label):
        mx.eval(left, right)
        assert left.shape == right.shape == (1, N) and left.dtype == right.dtype == mx.bfloat16
        lb, rb = raw(left), raw(right)
        finite = bool(mx.all(mx.isfinite(left)).item() and mx.all(mx.isfinite(right)).item())
        item = {"label": label, "shape": [1, N], "dtype": str(left.dtype), "nbytes": len(lb),
                "finite": finite, "left_sha256": hashlib.sha256(lb).hexdigest(),
                "right_sha256": hashlib.sha256(rb).hexdigest(), "differing_bytes": sum(a != b for a, b in zip(lb, rb)),
                "exact": finite and len(lb) == len(rb) and lb == rb}
        checks.append(item); write(out / "bytes.json", checks)
        assert item["exact"], ("strict BF16 bytes; STOP before timing", item)

    first_error = None
    final_error = None
    result = None
    try:
        probe("fresh-no-model")
        rng = np.random.default_rng(old.SEED)
        host = [np.zeros((1, K), np.float32), np.tile(np.array([1., -1.], np.float32), K // 2)[None]]
        host += [(rng.standard_normal((1, K)) * amp).astype(np.float32) for amp in (.0625, 1, 16, .25, .25, .25)]
        pool = [mx.array(x, dtype=mx.bfloat16) for x in host]; mx.eval(*pool)
        real = ident["real_inputs"]["inputs"][0]
        words = np.fromfile(real["path"], dtype="<u2", count=K)
        assert words.shape == (K,)
        historical = mx.array(words, dtype=mx.uint16).view(mx.bfloat16).reshape(1, K); mx.eval(historical)
        assert fingerprint(historical)["sha256"] == real["raw_sha256"]
        inputs = [{"id": j, **fingerprint(x), "synthetic": True} for j, x in enumerate(pool)]
        assert len({r["sha256"] for r in inputs}) == 8
        inputs.append({"id": "historical-real-L0", **fingerprint(historical), "synthetic": False}); write(out / "inputs.json", inputs)
        values = [old.load_range(np, mx, rec) for rec in ident["tensors"]]
        gate_weight, gate_scales, up_weight, up_scales = values
        assert sum(v.nbytes for v in values) == ident["target_tensor_payload_bytes"] == 100270080
        before_prepare = probe("weights-materialized-before-prepare")
        begin = time.perf_counter_ns()
        fused = prepare(mx, *values, runtime_version="0.32.3", device_name=device["device_name"], architecture=device["architecture"])
        host_end = time.perf_counter_ns(); mx.synchronize(); eval_end = time.perf_counter_ns()
        assert all(a is b for a, b in zip(fused.inputs, values))
        after_prepare = probe("one-prepare-no-weight-copy")
        write(out / "prepare.json", {"host_call_ns": host_end - begin, "synchronize_host_ns": eval_end - host_end,
              "total_ns": eval_end - begin, "expected_extra_weight_bytes": 0, "same_input_handles": True,
              "before_mlx_memory": before_prepare["mlx_memory"], "after_mlx_memory": after_prepare["mlx_memory"],
              "before_rss_kib": before_prepare.get("owned_pid_rss_kib"), "after_rss_kib": after_prepare.get("owned_pid_rss_kib")})
        bank_path = P / "swiglu-boundary-build2/check_swiglu_boundary_numeric_v2.py"
        bank_spec = importlib.util.spec_from_file_location("v28_frozen_swiglu_banks", bank_path)
        bank_module = importlib.util.module_from_spec(bank_spec); bank_spec.loader.exec_module(bank_module)
        pointwise_banks = []
        for label, gw, uw in bank_module.banks("bfloat16"):
            g = mx.array(np.asarray(gw, dtype="<u2"), dtype=mx.uint16).view(mx.bfloat16).reshape(1, N)
            u = mx.array(np.asarray(uw, dtype="<u2"), dtype=mx.uint16).view(mx.bfloat16).reshape(1, N)
            mx.eval(g, u)
            assert raw(g) == bank_module.raw_bytes(gw, "bfloat16") and raw(u) == bank_module.raw_bytes(uw, "bfloat16")
            pointwise_banks.append((label, g, u))
        native_kernel = old.make_kernel(mx)
        def original_pair(x):
            return [mx.quantized_matmul(x, w, sc, transpose=True, **Q) for w, sc in ((gate_weight, gate_scales), (up_weight, up_scales))]
        def parent_swiglu(g, u): return nn.silu(g) * u
        compiled_swiglu = mx.compile(parent_swiglu, shapeless=False)
        def original(x):
            g, u = original_pair(x)
            return parent_swiglu(g, u)
        def native_r4(x, projection): return old.project(mx, native_kernel, values[projection * 2], values[projection * 2 + 1], x, 4)
        originals = [*values, *pool, historical, *(a for _, g, u in pointwise_banks for a in (g, u))]
        initial = [fingerprint(x) for x in originals]
        def check_readonly(label):
            after = [fingerprint(x) for x in originals]
            row = {"label": label, "before": initial, "after": after, "unchanged": initial == after}
            readonly.append(row); write(out / "readonly.json", readonly)
            assert row["unchanged"], "input mutation"
        # Establish the actual installed MLP pointwise math, including its JIT
        # boundary. A mismatch is STOP; no adaptive formula/rounding selection.
        for label, g, u in pointwise_banks:
            compare(parent_swiglu(g, u), compiled_swiglu(g, u), "plain-compiled-swiglu." + label)
            compare(parent_swiglu(g, u), fused.pointwise(g, u), "typed-pointwise." + label)
        assert len(checks) == 16
        write(out / "pointwise-qualification.json", {"status": "STRICT-MLP-TYPED-POINTWISE-PASS", "checks": 16,
              "reference": "installed nn.silu(gate)*up, also compared to mx.compile same formula", "formula_selected_adaptively": False})
        check_readonly("pointwise-before-QMV-qualification")
        qualify = [(f"synthetic{j}", x) for j, x in enumerate(pool)] + [("historical-real-L0", historical)]
        start = time.perf_counter_ns(); z = original(pool[0]); mx.eval(z)
        startups.append({"variant": "original-gate-up-swiglu", "first_call_ns_including_eval": time.perf_counter_ns() - start})
        write(out / "startup.json", startups)
        for label, x in qualify:
            refs = original_pair(x)
            for i, name in enumerate(("gate", "up")): compare(refs[i], native_r4(x, i), "native-R4-QMM." + name + "." + label)
        check_readonly("all-native-R4-control-pass")
        probe("all-native-R4-control-pass-before-fused")
        start = time.perf_counter_ns(); z = fused.project(pool[0], debug=True); mx.eval(*z)
        startups.append({"variant": "fused-debug", "first_call_ns_including_eval": time.perf_counter_ns() - start}); write(out / "startup.json", startups)
        for label, x in qualify:
            g, u = original_pair(x)
            candidate, cg, cu = fused.project(x, debug=True)
            compare(g, cg, "fused-debug-gate." + label)
            compare(u, cu, "fused-debug-up." + label)
            compare(parent_swiglu(g, u), candidate, "fused-debug-swiglu." + label)
        # Production emits only the final output; qualify that exact kernel too.
        start = time.perf_counter_ns(); z = fused.project(pool[0])[0]; mx.eval(z)
        startups.append({"variant": "fused-production", "first_call_ns_including_eval": time.perf_counter_ns() - start}); write(out / "startup.json", startups)
        for label, x in qualify: compare(original(x), fused.project(x)[0], "fused-production." + label)
        check_readonly("all-qualification-inputs")
        assert len(checks) == 70 and all(c["exact"] for c in checks)
        write(out / "qualification.json", {"status": "STRICT-GATE-UP-EPILOGUE-BYTES-PASS", "checks": 70,
              "pointwise_checks": 16, "native_r4_checks": 18, "debug_checks": 27, "production_checks": 9,
              "readonly": True, "timing_allowed": True, "production_kernel_qualified": True})
        probe("all-byte-pass-before-timing")
        def invoke(variant, j):
            value = original(pool[j % 8]) if variant == "A" else fused.project(pool[j % 8])[0]
            mx.eval(value)
            return [value]
        for index, order in enumerate(SCHEDULE):
            probe(f"pair{index}-before")
            a, ao = old.block(mx, invoke, order[0]); b, bo = old.block(mx, invoke, order[1])
            for rec, outputs in ((a, ao), (b, bo)):
                rec["segments_per_call"] = 2
                rec["hashes"] = [hashlib.sha256(raw(v)).hexdigest() for v in outputs]
            assert a["hashes"] == b["hashes"], "paired output mismatch"
            row = {"pair": index, "order": order, "blocks": [a, b],
                   "kind": "position-control" if order[0] == order[1] else "AB",
                   "slot2_div_slot1": b["ns_per_call"] / a["ns_per_call"]}
            if row["kind"] == "AB":
                by = {a["variant"]: a, b["variant"]: b}
                row["reduction"] = 1 - by["B"]["ns_per_call"] / by["A"]["ns_per_call"]
            pairs.append(row); write(out / "pairs.json", pairs); del ao, bo
            probe(f"pair{index}-after")
        check_readonly("post-timing-all-inputs")
        ab = [r for r in pairs if r["kind"] == "AB"]
        controls = {v: [r["slot2_div_slot1"] for r in pairs if r["kind"] == "position-control" and r["order"][0] == v] for v in ("A", "B")}
        medians = {v: statistics.median(next(b["ns_per_call"] for b in r["blocks"] if b["variant"] == v) for r in ab) for v in ("A", "B")}
        reduction = 1 - medians["B"] / medians["A"]; wins = sum(r["reduction"] > 0 for r in ab)
        control_ok = old.control_passes(controls)
        status = "CONTROL_DRIFT_STOP" if not control_ok else "EXPLORATORY-PASS" if reduction >= .05 and wins >= 5 else "STOP-MICRO-GATE"
        result = {"status": status, "all_bytes_exact": True, "checks": 70, "readonly": True,
                  "medians_ns": medians, "median_reduction": reduction, "wins": wins, "pairs": 6,
                  "control_passed": control_ok, "control_slot2_div_slot1": controls,
                  "minimum_reduction": .05, "minimum_wins": 5, "power_domain": "AC single-matrix exploration",
                  "scope": "L0 gate+up only; A 2 native QMM + nn.silu(gate)*up / B one fused R4; no model inference",
                  "target_tensor_payload_bytes": 100270080, "extra_packed_bytes": 0,
                  "adopted": False, "full_model_qualified": False, "E2E_run": False, "AC_frontier_entry": False}
        write(out / "result.json", result)
    except BaseException as exc:
        first_error = exc
        write(out / "first-failure.json", {"error": repr(exc), "traceback": traceback.format_exc(), "timing_allowed": (out / "qualification.json").exists()})
    finally:
        try: probe("finally-before-owned-child-exit")
        except BaseException as exc:
            final_error = exc; write(out / "finally-resource-failure.json", {"error": repr(exc), "traceback": traceback.format_exc()})
    if first_error is not None: raise first_error
    if final_error is not None: raise final_error
    return 0 if result["status"] == "EXPLORATORY-PASS" else 2


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=("cpu", "run", "child")); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--owner-pid", type=int); args = ap.parse_args(); ident = identity()
    if args.mode == "cpu":
        args.out.mkdir(exist_ok=False)
        assert not any(k == "mlx" or k.startswith(("mlx.", "tutorial_code")) for k in sys.modules)
        write(args.out / "cpu.json", {"status": "PASS_CPU_SOURCE_HEADER", "identity": ident,
              "MLX_imported": False, "GPU_used": False, "checkpoint_payload_loaded": False})
        return 0
    assert sys.executable == PYTHON, "use exact pinned native interpreter"
    if args.mode == "child":
        assert args.owner_pid == os.getppid() and args.owner_pid > 1
        try: return gpu(args.out, ident)
        except BaseException as exc:
            write(args.out / "result.json", {"status": "FAIL-STOP", "error": repr(exc), "pid": os.getpid(), "adopted": False,
                  "timing_allowed": (args.out / "qualification.json").exists()}); traceback.print_exc(); return 1
        finally: write(args.out / "child-finally.json", {"pid": os.getpid(), "exiting": True})
    args.out.mkdir(exist_ok=False); write(args.out / "identity.json", ident)
    sys.path.insert(0, str(P / "v25-c1-screen-build1"))
    from support import identity as process_identity, process_parent, cleanup_owned
    proc = None; birth = None; rc = 1
    def stop(sig, frame): raise KeyboardInterrupt("owned V28 supervisor interrupted")
    previous = signal.signal(signal.SIGTERM, stop)
    with (args.out / "child.log").open("w") as log:
        try:
            argv = [sys.executable, "-B", str(Path(__file__).resolve()), "child", "--out", str(args.out), "--owner-pid", str(os.getpid())]
            proc = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT)
            birth = process_identity(proc.pid); assert birth is not None
            assert process_parent(proc.pid)["ppid"] == os.getpid()
            write(args.out / "child.json", {"pid": proc.pid, "owner_pid": os.getpid(), "argv": argv, "birth_identity": birth, "timeout_s": 170})
            rc = proc.wait(timeout=170)
        except BaseException as exc:
            write(args.out / "supervisor-failure.json", {"error": repr(exc), "traceback": traceback.format_exc()}); rc = 1
        finally:
            if proc is not None:
                cleanup = cleanup_owned(proc, birth, args.out / "cleanup-identities.json"); write(args.out / "cleanup.json", cleanup)
            signal.signal(signal.SIGTERM, previous)
    print(f"V28 child exit={rc}; exact owned child reaped; evidence={args.out}", flush=True)
    return rc


if __name__ == "__main__": raise SystemExit(main())
