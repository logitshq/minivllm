"""Root-only numerical worker. --inert/--help never import MLX or sample resources."""
from pathlib import Path
import argparse
import copy
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import resource
import sys
import time
import traceback

D = Path(__file__).resolve().parent
P = D.parent
ACCEPTED = P / "candidate-v17-gdn-gates-build1-root/tutorial_code/miniengine"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def source_check():
    pins = json.loads((D / "source-pins.json").read_text())
    for path, expected in pins.items():
        if sha(path) != expected:
            raise ValueError("Pinned source changed: " + path)
    return pins


class CheckFailure(RuntimeError):
    def __init__(self, kind, **details):
        super().__init__(kind)
        self.details = dict(kind=kind, **details)


class Resources:
    def __init__(self, out):
        self.out, self.history, self.records = out, [], []
        self.probe = module("v24_original_resource", P / "resource_snapshot_power_v1.py")

    def sample(self, label, loaded=False, mx=None):
        raw = self.probe.snapshot(label, os.getpid())
        path = self.out / ("resource-%02d-raw.json" % len(self.records))
        write(path, raw)  # Original raw exists before any annotation or guard.
        self.records.append(dict(file=path.name, sha256=sha(path)))
        derived = copy.deepcopy(raw)
        if loaded and not any(x["pid"] == os.getpid()
                              for x in derived["known_inference_processes"]):
            derived["known_inference_processes"].append(
                dict(pid=os.getpid(), owned=True, engine="explicit-loaded-v24-fullmodel"))
        known = derived["known_inference_processes"]
        derived["known_inference_process_count"] = len(known)
        derived["only_owned_known_inference_processes"] = all(x["owned"] for x in known)
        derived["exactly_one_owned_known_model"] = len(known) == 1 and known[0]["owned"]
        derived["model_loaded_by_worker"] = loaded
        derived["process_rss_high_water_bytes_macos"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        derived["mlx_allocator"] = None if mx is None else {
            "active_bytes": mx.get_active_memory(),
            "cache_bytes": mx.get_cache_memory(),
            "peak_active_bytes": mx.get_peak_memory(),
            "compiled_program_bytes": None,
            "scope": "allocator/tensor counters; not compiled-program cache"}
        self.history.append(derived)
        self.probe.deltas(self.history)
        write(self.out / "resources-derived.json", self.history)
        self.probe.guard(self.history)  # Pinned original pressure/paging/foreign guard.
        if derived["pressure_level_value"] != 1:
            raise CheckFailure("resource_pressure_requires_1", label=label)
        power = derived["power_metadata"]
        thermal = derived["processinfo_thermal_metadata"]
        if power["status"] != "known" or power["source"] != "AC Power":
            raise CheckFailure("resource_requires_known_AC", label=label)
        if thermal["status"] != "known" or thermal["code"] not in (0, 1):
            raise CheckFailure("resource_requires_thermal_0_or_1", label=label)
        if (derived["exact_owned_pid_metadata"]["status"] != "known" or
                len(known) != int(loaded) or any(not x["owned"] for x in known)):
            raise CheckFailure("resource_process_inventory_differs", label=label)
        return dict(raw_file=path.name, raw_sha256=sha(path),
                    exact_owned_pid_metadata=derived["exact_owned_pid_metadata"],
                    mlx_allocator=derived["mlx_allocator"])


class Numeric:
    def __init__(self, mx, np, out):
        self.mx, self.np, self.out = mx, np, out

    def raw(self, value, label):
        mx, np = self.mx, self.np
        mx.eval(value)
        if value.dtype == mx.bfloat16:
            host = np.ascontiguousarray(np.asarray(value.view(mx.uint16)), dtype="<u2")
            finite = not bool(np.any((host & 0x7f80) == 0x7f80))
        elif value.dtype == mx.float32:
            host = np.ascontiguousarray(np.asarray(value), dtype="<f4")
            finite = bool(np.isfinite(host).all())
        else:
            raise CheckFailure("unexpected_numerical_dtype", label=label, dtype=str(value.dtype))
        data = host.tobytes()
        sig = dict(shape=list(value.shape), dtype=str(value.dtype), finite=finite,
                   bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
        if not finite:
            (self.out / "first-failure-nonfinite.bin").write_bytes(data)
            raise CheckFailure("nonfinite", label=label, signature=sig,
                               raw_file="first-failure-nonfinite.bin")
        return data, sig

    def signatures(self, flat, label):
        return [self.raw(a, f"{label}.{i}")[1] for i, a in enumerate(flat)]

    def readonly(self, flat, before, label):
        after = self.signatures(flat, label)
        if after != before:
            i = next(i for i, (a, b) in enumerate(zip(before, after)) if a != b)
            raise CheckFailure("retained_input_changed", label=label, array_index=i,
                               before=before[i], after=after[i])
        return dict(equal=True, count=len(flat), before=before, after=after)

    def equal(self, left, right, label, persist=None):
        if left.shape != right.shape or left.dtype != right.dtype:
            raise CheckFailure("shape_or_dtype_differs", label=label,
                               left_shape=list(left.shape), right_shape=list(right.shape),
                               left_dtype=str(left.dtype), right_dtype=str(right.dtype))
        a, sa = self.raw(left, label + ".A")
        b, sb = self.raw(right, label + ".B")
        if a != b:
            (self.out / "first-failure-A.bin").write_bytes(a)
            (self.out / "first-failure-B.bin").write_bytes(b)
            offset = next(i for i, (x, y) in enumerate(zip(a, b)) if x != y)
            raise CheckFailure("strict_bytes_differ", label=label, byte_offset=offset,
                               A=sa, B=sb, raw_files=["first-failure-A.bin", "first-failure-B.bin"])
        result = dict(label=label, strict_equal=True, A=sa, B=sb)
        if persist:
            for side, raw in (("A", a), ("B", b)):
                name = f"{persist}-{side}.bin"
                (self.out / name).write_bytes(raw)
                result["raw_" + side] = name
        return result


def arguments(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--tokens", type=int, default=32)
    ap.add_argument("--inert", action="store_true")
    return ap.parse_args(argv)


def main(argv=None):
    args = arguments(argv)
    pins = source_check()
    workload = json.loads((D / "workload.json").read_text())
    if args.tokens != 32 or len(workload["prompt_ids"]) != 19 or len(workload["continuation_ids"]) != 32:
        raise ValueError("Only the fixed19-prompt/32-output numerical workload is admitted")
    if args.inert:
        print(json.dumps(dict(status="CPU_INERT_ONLY", pinned_files=len(pins),
                              MLX_imported=any(k == "mlx" or k.startswith("mlx.") for k in sys.modules),
                              GPU_executed=False)))
        return 0
    if args.out is None or not args.out.is_absolute():
        raise ValueError("Root execution requires --out NEW_ABSOLUTE_DIRECTORY")
    args.out.mkdir(exist_ok=False)
    receipt = dict(status="STARTED", stage="beforeload", pid=os.getpid(),
                   no_performance_claim=True, engine_integrated=False,
                   expected_outputs=32, expected_decode_calls=31, steps=[],
                   model_load_attempts=0, model_loads=0, raw_resources=[],
                   numerical_coverage=dict(exact_L_cold_chain=True,
                       same_L_warm_hit=False, LRU_eviction=False, service_lifecycle=False),
                   source_pins=pins, workload=workload)
    save = lambda: write(args.out / "receipt.json", receipt)
    save()
    mx, loaded, resources, helper = None, False, None, None
    started = time.perf_counter()
    try:
        resources = Resources(args.out)
        resources.sample("beforeload", loaded=False)
        versions = {name: importlib.metadata.version(name) for name in ("mlx", "mlx-metal")}
        if versions != {"mlx": "0.32.3", "mlx-metal": "0.32.3"}:
            raise CheckFailure("runtime_version_differs", versions=versions)
        import mlx.core as mx
        import numpy as np
        from mlx.utils import tree_flatten
        if mx.default_device() != mx.gpu:
            raise CheckFailure("actual_default_device_not_gpu")
        receipt["runtime"] = dict(versions=versions, device_info=mx.device_info(),
                                  python=sys.executable, numpy_version=np.__version__,
                                  selected_environment={k: v for k, v in os.environ.items()
                                      if k.startswith(("MLX_", "METAL_", "DYLD_"))})
        receipt["cache_limit_bytes"] = 2 << 30
        receipt["previous_cache_limit_bytes"] = mx.set_cache_limit(2 << 30)
        sys.path.insert(0, str(ACCEPTED.parent.parent))
        from tutorial_code.miniengine.model.loader import load_model
        from tutorial_code.miniengine.model.qwen3_5 import Qwen3_5ForCausalLM
        from tutorial_code.miniengine.cache import BatchKV, BatchState, DecodeBatch, make_cache
        from tutorial_code.miniengine.context import ForwardContext
        from tutorial_code.miniengine.eos import read_eos_ids

        def imported_identity():
            actual = {}
            for name, mod in tuple(sys.modules.items()):
                if name.startswith("tutorial_code.miniengine") and getattr(mod, "__file__", None):
                    path = str(Path(mod.__file__).resolve())
                    if path not in pins or sha(path) != pins[path]:
                        raise CheckFailure("imported_source_not_accepted", module=name, path=path)
                    actual[name] = dict(path=path, sha256=pins[path])
            return actual

        receipt["imported_sources_before_load"] = imported_identity()
        receipt["stage"] = "loading"; receipt["model_load_attempts"] += 1; save()
        model, cfg = load_model(workload["model_dir"], dtype=mx.bfloat16)
        receipt["model_loads"] += 1; loaded = True
        mx.synchronize()
        receipt["afterload"] = resources.sample("afterload-before-any-token", loaded=True, mx=mx)
        if len(model.layers) != 64:
            raise CheckFailure("model_layer_count_differs", count=len(model.layers))
        for layer in model.layers:
            if not layer.is_attn:
                layer.linear_attn.set_gates_compile(True)
        def parameter_identity():
            rows = [(k, id(v), list(v.shape), str(v.dtype)) for k, v in tree_flatten(model.parameters())]
            return dict(count=len(rows), identity_sha256=hashlib.sha256(json.dumps(rows).encode()).hexdigest())
        parameter_start = parameter_identity()
        receipt["parameters_initial"] = parameter_start
        receipt["stage"] = "ready"; save()
        write(args.out / "model-ready.json", dict(pid=os.getpid(), stage="ready", model_loads=1,
                                                  resource=receipt["afterload"], token_executed=False))
        print(json.dumps(dict(event="MODEL_READY", pid=os.getpid())), flush=True)
        candidate = module("v24_frozen_functional", D / "functional_decode.py")
        helper = candidate.FunctionalDecode(model, backend=mx,
            types=candidate.DecodeTypes(Qwen3_5ForCausalLM, ForwardContext, BatchKV, BatchState),
            runtime_version=versions["mlx"], device_kind="gpu")
        numeric = Numeric(mx, np, args.out)
        caches = make_cache(model)
        receipt["stage"] = "prefill"; save()
        prefill_logits = model(mx.array(workload["prompt_ids"], dtype=mx.int32),
                               ForwardContext("prefill", caches), last_only=True)
        mx.eval(prefill_logits)
        raw, seed_sig = numeric.raw(prefill_logits, "prefill.logits")
        (args.out / "prefill-logits.bin").write_bytes(raw)
        first = int(mx.argmax(prefill_logits, axis=-1).item())
        eos = read_eos_ids(workload["model_dir"])
        receipt["eos_ids"] = sorted(eos)
        if first != workload["continuation_ids"][0] or first in eos:
            raise CheckFailure("prefill_seed_differs_or_EOS", token=first)
        batch = DecodeBatch(model)
        batch.rebuild([], [("R0", caches, len(workload["prompt_ids"]) + args.tokens)])
        initial_flat = tuple(batch.arrays())
        mx.eval(*initial_flat)
        if len(initial_flat) != 128:
            raise CheckFailure("cache_array_count_differs", count=len(initial_flat))
        a_flat = tuple(a[:] for a in initial_flat)
        b_flat = tuple(a[:] for a in initial_flat)
        seed_state = numeric.signatures(initial_flat, "seed.state")
        receipt["prefill"] = dict(logits=seed_sig, seed_token=first, states=seed_state,
                                   raw_logits="prefill-logits.bin", array_count=128)
        a_ids, b_ids = [first], [first]
        receipt["A_ids"], receipt["B_ids"] = a_ids, b_ids

        def layers_from_flat(flat):
            # Never pass the retained external K/V array objects to write().
            return [BatchKV(flat[2*i][:], flat[2*i+1][:]) if layer.is_attn
                    else BatchState(flat[2*i], flat[2*i+1])
                    for i, layer in enumerate(model.layers)]

        for step in range(1, args.tokens):
            position = len(workload["prompt_ids"]) + step - 1
            length = position + 1
            receipt["stage"] = dict(phase="decode", step=step, position=position, exact_L=length)
            save()
            before_a = numeric.signatures(a_flat, "A.retained.before")
            before_b = numeric.signatures(b_flat, "B.retained.before")
            positions = mx.array([position], dtype=mx.int32)
            lens = mx.array([length], dtype=mx.int32)
            ctx_a = ForwardContext("decode", layers_from_flat(a_flat),
                                    positions=positions, lens=lens, max_len=length)
            then = time.perf_counter_ns()
            logits_a = model(mx.array([a_ids[-1]], dtype=mx.int32), ctx_a)
            a_call_ns = time.perf_counter_ns() - then
            next_a = tuple(a for layer, cache in zip(model.layers, ctx_a.caches)
                           for a in ((cache.K, cache.V) if layer.is_attn else (cache.conv, cache.S)))
            then = time.perf_counter_ns(); mx.eval(logits_a, *next_a)
            a_eval_ns = time.perf_counter_ns() - then
            numeric.readonly(a_flat, before_a, "A.retained.after_A")
            numeric.readonly(b_flat, before_b, "B.retained.after_A")
            then = time.perf_counter_ns()
            result_b = helper.functional(mx.array([b_ids[-1]], dtype=mx.int32), positions,
                                         lens, b_flat, max_len=length, greedy=True)
            b_call_ns = time.perf_counter_ns() - then
            logits_b, next_b = result_b[0], result_b[1:]
            then = time.perf_counter_ns(); mx.eval(logits_b, *next_b)
            b_eval_ns = time.perf_counter_ns() - then
            logits_sig = numeric.equal(logits_a, logits_b, f"step{step}.logits",
                                       persist=f"step{step:02d}-logits")
            if len(next_a) != 128 or len(next_b) != 128:
                raise CheckFailure("decode_cache_array_count_differs", step=step)
            state_sigs = [numeric.equal(a, b, f"step{step}.state.{i}")
                          for i, (a, b) in enumerate(zip(next_a, next_b))]
            ro_a = numeric.readonly(a_flat, before_a, "A.retained.after_AB")
            ro_b = numeric.readonly(b_flat, before_b, "B.retained.after_AB")
            ta, tb = int(mx.argmax(logits_a, axis=-1).item()), int(mx.argmax(logits_b, axis=-1).item())
            if ta != tb or ta != workload["continuation_ids"][step] or ta in eos:
                raise CheckFailure("greedy_ID_differs_or_early_EOS", step=step, A=ta, B=tb,
                                   archived=workload["continuation_ids"][step])
            a_ids.append(ta); b_ids.append(tb)
            a_flat, b_flat = next_a, tuple(next_b)
            event = dict(step=step, position=position, exact_L=length, A_token=ta, B_token=tb,
                         logits=logits_sig, states=state_sigs, retained_input_readonly=dict(A=ro_a, B=ro_b),
                         host_ns=dict(A_fncall=a_call_ns, B_fncall=b_call_ns,
                                      A_eval=a_eval_ns, B_eval=b_eval_ns),
                         timing_scope="numerical/instrumented host calls; not serving or performance qualification",
                         helper=helper.snapshot())
            if step % 4 == 0:
                mx.synchronize()
                event["resources"] = resources.sample(f"decode-{step:02d}", loaded=True, mx=mx)
            receipt["steps"].append(event)
            save()
            print(json.dumps(dict(event="DECODE_EQUAL", step=step, exact_L=length)), flush=True)
        numeric.readonly(initial_flat, seed_state, "shared_prefill_state.final")
        if parameter_identity() != parameter_start:
            raise CheckFailure("model_parameter_identity_changed")
        receipt["imported_sources_final"] = imported_identity()
        receipt["helper_final"] = helper.snapshot()
        if receipt["helper_final"]["misses"] != 31 or receipt["helper_final"]["entries"] != 31:
            raise CheckFailure("expected_31_exact_L_cold_graphs_not_observed")
        receipt["final_cache"] = dict(A=numeric.signatures(a_flat, "final.A"),
                                      B=numeric.signatures(b_flat, "final.B"))
        source_check()
        receipt["status"] = "PASS_32_GREEDY_FULLMODEL_NUMERICAL_ONLY"
        receipt["stage"] = "complete"
    except Exception as error:
        receipt["status"] = "FAIL_STOP"
        receipt["first_failure"] = getattr(error, "details", dict(
            kind=type(error).__name__, message=str(error)))
        receipt["traceback"] = traceback.format_exc()
    finally:
        try:
            if mx is not None:
                mx.synchronize()
        except Exception as error:
            receipt["finally_synchronize_failure"] = dict(type=type(error).__name__, message=str(error))
            receipt["status"] = "FAIL_STOP"
        try:
            if resources is not None:
                receipt["finally_resource"] = resources.sample("finally", loaded=loaded, mx=mx)
        except Exception as error:
            receipt["finally_failure"] = dict(type=type(error).__name__, message=str(error))
            receipt["status"] = "FAIL_STOP"
        receipt["raw_resources"] = resources.records if resources else []
        receipt["helper_final"] = helper.snapshot() if helper else None
        receipt["instrumented_wall_seconds"] = time.perf_counter() - started
        save()
    return 0 if receipt["status"] == "PASS_32_GREEDY_FULLMODEL_NUMERICAL_ONLY" else 1


if __name__ == "__main__":
    raise SystemExit(main())
