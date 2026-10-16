#!/usr/bin/env python3
"""Root-only V26 one-matrix numeric + balanced micro. CPU mode never imports MLX."""
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
from rowblock import prepare, metadata, K, N

Q, PRIME, REPEATS, SCHEDULE = old.Q, old.PRIME, old.REPEATS, old.SCHEDULE
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
    for suffix, dtype, shape in [("weight", "U32", (N, K // 8)), ("scales", "U8", (N, K // 16))]:
        keys = [key for key in index if key.endswith(".layers.0.mlp.gate_proj." + suffix)]
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
            "real_inputs": old.real_metadata(), "layout": metadata(),
            "plan": {"prime": PRIME, "repeats": REPEATS, "schedule": SCHEDULE,
                     "controls": CONTROLS, "power": "AC Power", "timeout_s": 170,
                     "minimum_reduction": .05, "minimum_wins": 5, "full_model": False}}


def gpu(out, ident):
    import numpy as np
    import mlx.core as mx
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
          "math_policy_override": False, "model_loaded": False, "rows_per_simd": 4})
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
        weight, scales = [old.load_range(np, mx, rec) for rec in ident["tensors"]]
        assert weight.nbytes + scales.nbytes == ident["target_tensor_payload_bytes"] == 50135040
        native_kernel = old.make_kernel(mx)
        def original(x): return mx.quantized_matmul(x, weight, scales, transpose=True, **Q)
        def native_r4(x): return old.project(mx, native_kernel, weight, scales, x, 4)
        originals = [weight, scales, *pool, historical]
        initial = [fingerprint(x) for x in originals]
        def check_readonly(label, arrays, before):
            after = [fingerprint(x) for x in arrays]
            row = {"label": label, "before": before, "after": after, "unchanged": before == after}
            readonly.append(row); write(out / "readonly.json", readonly)
            assert row["unchanged"], "input mutation"
        qualify = [(f"synthetic{j}", x) for j, x in enumerate(pool)] + [("historical-real-L0", historical)]
        for name, fn in (("original", original), ("native-R4", native_r4)):
            start = time.perf_counter_ns(); z = fn(pool[0]); mx.eval(z)
            startups.append({"variant": name, "first_call_ns_including_eval": time.perf_counter_ns() - start})
            write(out / "startup.json", startups)
        for label, x in qualify: compare(original(x), native_r4(x), "R4-QMM." + label)
        check_readonly("native-R4-gate", originals, initial)
        before_pack = probe("native-R4-pass-before-packing")
        begin = time.perf_counter_ns()
        packed = prepare(mx, weight, scales, runtime_version="0.32.3", device_name=device["device_name"], architecture=device["architecture"])
        host_end = time.perf_counter_ns(); mx.eval(packed.weight, packed.scales); eval_end = time.perf_counter_ns()
        after_pack = probe("packing-materialized")
        write(out / "packing.json", {"host_call_ns": host_end - begin, "eval_host_ns": eval_end - host_end,
              "total_ns_including_eval": eval_end - begin, "expected_extra_packed_bytes": 50135040,
              "packed_payload_bytes": packed.weight.nbytes + packed.scales.nbytes,
              "before_mlx_memory": before_pack["mlx_memory"], "after_mlx_memory": after_pack["mlx_memory"],
              "before_rss_kib": before_pack.get("owned_pid_rss_kib"), "after_rss_kib": after_pack.get("owned_pid_rss_kib"),
              "scope": "allocator/RSS are separate observations; not native program-cache bytes"})
        for value, src, unit in ((packed.weight, weight, 64), (packed.scales, scales, 32)):
            inv = np.asarray(value).reshape(N // 8, K // 512, 8, unit).transpose(0, 2, 1, 3).reshape(src.shape)
            assert inv.tobytes() == raw(src), "actual packed payload/scale inverse bytes mismatch"
        del inv; gc.collect()
        packed_arrays = [packed.weight, packed.scales]; packed_initial = [fingerprint(x) for x in packed_arrays]
        start = time.perf_counter_ns(); z = packed.project(pool[0]); mx.eval(z)
        startups.append({"variant": "packed-R4", "first_call_ns_including_eval": time.perf_counter_ns() - start}); write(out / "startup.json", startups)
        for label, x in qualify:
            candidate = packed.project(x)
            compare(original(x), candidate, "packed-QMM." + label)
            compare(native_r4(x), candidate, "packed-R4." + label)
        check_readonly("qualification-original-inputs", originals, initial)
        check_readonly("qualification-packed-inputs", packed_arrays, packed_initial)
        assert len(checks) == 27 and all(c["exact"] for c in checks)
        write(out / "qualification.json", {"status": "STRICT-NATIVE-R4-PACKED-BYTES-PASS", "checks": 27,
              "readonly": True, "actual_inverse_pack_bytes_exact": True, "timing_allowed": True})
        probe("all-byte-pass-before-timing")
        def invoke(variant, j):
            value = original(pool[j % 8]) if variant == "A" else packed.project(pool[j % 8])
            mx.eval(value)
            return [value]
        for index, order in enumerate(SCHEDULE):
            probe(f"pair{index}-before")
            a, ao = old.block(mx, invoke, order[0]); b, bo = old.block(mx, invoke, order[1])
            for rec, values in ((a, ao), (b, bo)):
                rec["hashes"] = [hashlib.sha256(raw(v)).hexdigest() for v in values]
            assert a["hashes"] == b["hashes"], "paired output mismatch"
            row = {"pair": index, "order": order, "blocks": [a, b],
                   "kind": "position-control" if order[0] == order[1] else "AB",
                   "slot2_div_slot1": b["ns_per_call"] / a["ns_per_call"]}
            if row["kind"] == "AB":
                by = {a["variant"]: a, b["variant"]: b}
                row["reduction"] = 1 - by["B"]["ns_per_call"] / by["A"]["ns_per_call"]
            pairs.append(row); write(out / "pairs.json", pairs); del ao, bo
            probe(f"pair{index}-after")
        check_readonly("post-timing-original-inputs", originals, initial)
        check_readonly("post-timing-packed-inputs", packed_arrays, packed_initial)
        ab = [r for r in pairs if r["kind"] == "AB"]
        controls = {v: [r["slot2_div_slot1"] for r in pairs if r["kind"] == "position-control" and r["order"][0] == v] for v in ("A", "B")}
        medians = {v: statistics.median(next(b["ns_per_call"] for b in r["blocks"] if b["variant"] == v) for r in ab) for v in ("A", "B")}
        reduction = 1 - medians["B"] / medians["A"]; wins = sum(r["reduction"] > 0 for r in ab)
        control_ok = old.control_passes(controls)
        status = "CONTROL_DRIFT_STOP" if not control_ok else "EXPLORATORY-PASS" if reduction >= .05 and wins >= 5 else "STOP-MICRO-GATE"
        result = {"status": status, "all_bytes_exact": True, "checks": 27, "readonly": True,
                  "medians_ns": medians, "median_reduction": reduction, "wins": wins, "pairs": 6,
                  "control_passed": control_ok, "control_slot2_div_slot1": controls,
                  "minimum_reduction": .05, "minimum_wins": 5, "power_domain": "AC single-matrix exploration",
                  "scope": "L0 gate only; A native QMM / B packed R4; no full-model or adoption inference",
                  "target_tensor_payload_bytes": 50135040, "extra_packed_bytes": 50135040,
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
    def stop(sig, frame): raise KeyboardInterrupt("owned V26 supervisor interrupted")
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
    print(f"V26 child exit={rc}; exact owned child reaped; evidence={args.out}", flush=True)
    return rc


if __name__ == "__main__": raise SystemExit(main())
