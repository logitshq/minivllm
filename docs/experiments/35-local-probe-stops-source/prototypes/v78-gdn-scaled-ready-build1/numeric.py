"""V78 local strict qualification against V44; default is CPU-only identity checks."""
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
sys.dont_write_bytecode = True
D = Path(__file__).resolve().parent
P = D.parent
BASE = P / "v44-gdn-activated-normreuse-build1/gdn_activated_norm_reuse_cached_metal.py"
BANKROOT = P / "v18-recurrent-numeric-build1/gpu-recurrent-numeric-AC-run1"
REAL = [f"real-L{layer}-D{step}" for layer in (0, 32, 62) for step in (1, 16, 31)]
SYNTHETIC = ["synthetic-zero", "synthetic-random-PCG64-18001", "synthetic-amplitude", "synthetic-cancellation", "synthetic-signedzero-subnormal", "synthetic-repeated-head-boundary"]
CONTRACT = {"u": ((1, 10240), "bfloat16"), "g": ((1, 48), "float32"), "beta": ((1, 48), "float32"), "S": ((1, 48, 128, 128), "float32")}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def identity():
    pins = json.loads((D / "source-pins.json").read_text())
    for path, expected in pins.items():
        assert sha(path) == expected, ("pinned source changed", path)
    for name in REAL + SYNTHETIC:
        directory = BANKROOT / name
        meta = json.loads((directory / "bank.json").read_text())
        for key, (shape, dtype) in CONTRACT.items():
            array = meta["arrays"][key]
            path = directory / array["file"]
            assert Path(array["file"]).name == array["file"]
            assert tuple(array["shape"]) == shape and array["dtype"] == dtype
            assert path.stat().st_size == array["bytes"] == math.prod(shape) * (2 if dtype == "bfloat16" else 4)
            assert sha(path) == array["sha256"], ("saved bank changed", name, key)
    return pins


def completed_counts(cases):
    return {"strict_output_checks": sum(x.get("strict_equal") is True for row in cases for x in row.get("outputs", [])),
            "strict_ready_checks": sum(x.get("strict_equal") is True for row in cases for x in row.get("ready", [])),
            "strict_diagnostic_checks": sum(x.get("strict_equal") is True for row in cases for x in row.get("diagnostics", [])),
            "input_readonly_cases": sum(row.get("input_readonly") is True for row in cases)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--execute-gpu", action="store_true")
    args = ap.parse_args(argv)
    pins = identity()
    if not args.execute_gpu:
        assert not any(name == "mlx" or name.startswith(("mlx.", "numpy")) for name in sys.modules)
        print(json.dumps({"status": "CPU_READY_ONLY", "GPU_executed": False, "saved_real_banks": 9, "saved_synthetic_banks": 6, "input_binary_SHA_verified": True, "A": str(BASE), "B": "V78 scaled-ready", "fullmodel_qualified": False, "timing_prepared": False}))
        return 0
    assert args.out is not None and args.out.is_absolute()
    args.out.mkdir(exist_ok=False)
    r = {"status": "STARTED", "pins": pins, "cases": [], "GPU_numeric_run": False,
         "A": str(BASE), "B": str(D / "gdn_activated_scaled_ready_metal.py"),
         "strict_FP32_raw_bytes": True, "expected_output_checks": 30, "expected_ready_checks": 30,
         "expected_diagnostic_checks": 30, "expected_input_readonly_cases": 15,
         "activated_input": "one externally materialized nn.silu(saved BF16 u) shared by A/B; neither helper applies SiLU",
         "saved_S_bank_fixed": True, "g_beta_fixed": True, "output_feedback": False,
         "compile_policy": {"A": {"math_mode": "safe"}, "B": {"math_mode": "safe"}, "reference_probe": {"math_mode": "safe"}},
         "ready_reference": "separate qualification-only safe Metal kernel: FP32 qs * rawq and ks * rawk, original operand order",
         "qualification_repeats_prework": True, "timing_run": False, "engine_connected": False,
         "model_loaded": False, "native_whitelist_changed": False, "fullmodel_qualified": False,
         "claim": "local bytes gate only; FP32 materialization and FMA/contraction equivalence require actual validation"}
    start = time.perf_counter()
    save = lambda: write(args.out / "receipt.json", r)
    save()
    try:
        versions = {name: importlib.metadata.version(name) for name in ("mlx", "mlx-metal")}
        assert versions == {"mlx": "0.32.3", "mlx-metal": "0.32.3"}
        controls = {"MLX_MAX_OPS_PER_BUFFER": "50", "MLX_MAX_MB_PER_BUFFER": "50", "MLX_BFS_MAX_WIDTH": "20", "MLX_METAL_FAST_SYNCH": "0"}
        assert all(os.environ.get(key) == value for key, value in controls.items())
        import mlx.core as mx
        import mlx.nn as nn
        import numpy as np
        info = mx.device_info()
        assert mx.default_device() == mx.gpu and info["device_name"] == "Apple M4 Max" and info["architecture"] == "applegpu_g16s"
        mx.set_memory_limit(512 << 20)
        mx.set_cache_limit(64 << 20)
        r["runtime"] = {"versions": versions, "device": info, "controls": controls, "memory_limit": 512 << 20, "cache_limit": 64 << 20}
        baseline = module("v78_A44", BASE)
        candidate = module("v78_B", D / "gdn_activated_scaled_ready_metal.py")
        probe = mx.fast.metal_kernel(name="gdn_scaled_ready_reference_v78_build1", input_names=["q_raw", "k_raw", "scales"], output_names=["q_expected", "k_expected"], source=(D / "reference-scaled-ready-probe.metal").read_text(), header=baseline.HEADER, compile_options={"math_mode": "safe"})

        def raw(array):
            mx.eval(array)
            if array.dtype == mx.bfloat16:
                host = np.ascontiguousarray(np.asarray(array.view(mx.uint16)), dtype="<u2")
                assert not np.any((host & 0x7f80) == 0x7f80)
            else:
                assert array.dtype == mx.float32
                host = np.ascontiguousarray(np.asarray(array), dtype="<f4")
                assert np.isfinite(host).all()
            data = host.tobytes()
            return data, {"shape": list(array.shape), "dtype": str(array.dtype), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}

        def signatures(values):
            return {key: raw(array)[1] for key, array in values.items()}

        def compare(row, category, key, a, b):
            assert a.shape == b.shape and a.dtype == b.dtype == mx.float32
            da, sa = raw(a)
            db, sb = raw(b)
            row[category].append({"name": key, "strict_equal": da == db, "A": sa, "B": sb})
            r.update(completed_counts(r["cases"]))
            save()
            if da != db:
                (args.out / "first-failure-A.bin").write_bytes(da)
                (args.out / "first-failure-B.bin").write_bytes(db)
                r["first_failure"] = {"bank": row["bank"], "category": category, "output": key, "A": sa, "B": sb, "first_byte_offset": next(i for i, (x, y) in enumerate(zip(da, db)) if x != y)}
                save()
                raise AssertionError("strict FP32 raw bytes differ")

        r["GPU_numeric_run"] = True
        for name in REAL + SYNTHETIC:
            directory = BANKROOT / name
            meta = json.loads((directory / "bank.json").read_text())
            row = {"bank": name, "kind": meta["kind"], "bank_sha256": sha(directory / "bank.json"), "historical_saved_input_not_new_capture": True, "ready": [], "diagnostics": [], "outputs": []}
            r["cases"].append(row)
            save()
            bank = {}
            hosts = []
            for key, (shape, dtype) in CONTRACT.items():
                host = np.fromfile(directory / meta["arrays"][key]["file"], dtype="<u2" if dtype == "bfloat16" else "<f4").reshape(shape)
                hosts.append(host)
                value = mx.array(host)
                bank[key] = value.view(mx.bfloat16) if dtype == "bfloat16" else value
            activated = nn.silu(bank["u"])
            mx.eval(activated, bank["g"], bank["beta"], bank["S"])
            mx.synchronize()
            a_ready = baseline._prepare(activated, mx)
            b_ready = candidate._prepare(activated, mx)
            expected = tuple(probe(inputs=[a_ready[0], a_ready[1], a_ready[3]], grid=(2048, 1, 1), threadgroup=(128, 1, 1), output_shapes=[(1, 16, 128)] * 2, output_dtypes=[mx.float32] * 2))
            mx.eval(*a_ready, *b_ready, *expected)
            inputs = dict(bank, activated=activated)
            for prefix, values in (("A44", a_ready), ("B78", b_ready)):
                inputs.update({prefix + "_" + key: value for key, value in zip(("q_ready", "k_ready", "v_ready", "scales"), values)})
            inputs.update({"expected_q": expected[0], "expected_k": expected[1]})
            row["input_before"] = signatures(inputs)
            # Real A/B use their public two-kernel APIs, independently of qualification probes.
            A = baseline.gated_delta_step_activated_norm_reuse_metal(activated, bank["g"], bank["beta"], bank["S"])
            B = candidate.gated_delta_step_activated_norm_reuse_metal(activated, bank["g"], bank["beta"], bank["S"])
            mx.eval(*A, *B)
            row["input_after"] = signatures(inputs)
            row["input_readonly"] = row["input_before"] == row["input_after"]
            r.update(completed_counts(r["cases"]))
            save()
            assert row["input_readonly"], ("input mutation", name)
            for key, a, b in zip(("q_ready", "k_ready"), expected, b_ready[:2]):
                compare(row, "ready", key, a, b)
            for key, index in (("v_ready", 2), ("scales", 3)):
                compare(row, "diagnostics", key, a_ready[index], b_ready[index])
            for key, a, b in zip(("o", "S_new"), A, B):
                compare(row, "outputs", key, a, b)
            r.update(completed_counts(r["cases"]))
            save()
            del A, B, bank, inputs, hosts, a_ready, b_ready, expected, activated
        counts = completed_counts(r["cases"])
        assert len(r["cases"]) == 15 and counts == {"strict_output_checks": 30, "strict_ready_checks": 30, "strict_diagnostic_checks": 30, "input_readonly_cases": 15}
        r.update(counts)
        r.update(status="PASS_V78_SCALED_READY_LOCAL_STRICT_9REAL_6SYNTHETIC", fullmodel_qualified=False)
        return 0
    except BaseException as error:
        r.update(completed_counts(r["cases"]))
        r.update(status="FAIL_STOP", error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        r["diagnostic_wall_seconds_not_performance"] = time.perf_counter() - start
        save()


if __name__ == "__main__":
    raise SystemExit(main())
