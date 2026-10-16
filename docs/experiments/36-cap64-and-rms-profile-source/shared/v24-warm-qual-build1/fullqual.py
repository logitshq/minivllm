"""Root-only same-L warm numerical worker; --inert is CPU-safe."""
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import sys
import time
import traceback

D = Path(__file__).resolve().parent
P = D.parent
COLD = P / "v24-fullmodel-qual-build1"
ACCEPTED = P / "candidate-v17-gdn-gates-build1-root/tutorial_code/miniengine"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def source_check():
    pins = json.loads((D / "source-pins.json").read_text())
    for path, expected in pins.items():
        if sha(path) != expected:
            raise ValueError("source changed: " + path)
    return pins


def arguments(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--tokens", type=int, default=32)
    ap.add_argument("--inert", action="store_true")
    return ap.parse_args(argv)


def main(argv=None):
    args = arguments(argv)
    pins = source_check()
    config = json.loads((D / "workload.json").read_text())
    workload = config["workload"]
    if args.tokens != 32 or len(workload["prompt_ids"]) != 19:
        raise ValueError("fixed --tokens32 compatibility interface and19 prompt required")
    if args.inert:
        print(json.dumps(dict(status="CPU_INERT_ONLY", pins=len(pins),
            MLX_imported=any(k == "mlx" or k.startswith("mlx.") for k in sys.modules),
            GPU_executed=False)))
        return 0
    if args.out is None or not args.out.is_absolute():
        raise ValueError("Root execution requires --out NEW_ABSOLUTE_DIRECTORY")
    args.out.mkdir(exist_ok=False)
    result = dict(status="STARTED", stage="beforeload", pid=os.getpid(),
        scope="same-L20 four-case warm numerical qualification; not32-output generation",
        interface_tokens_argument=32, no_performance_claim=True,
        model_load_attempts=0, model_loads=0, groups=[], cases=[], raw_resources=[],
        source_pins=pins, config=config, service_integrated=False,
        pending=["ModelRunner integration and ahead/service lifecycle", "native program memory attribution"])
    save = lambda: write(args.out / "receipt.json", result)
    save()
    mx, resources, helper, loaded = None, None, None, False
    started = time.perf_counter()
    try:
        previous = json.loads(Path(config["prior_cold_receipt"]).read_text())
        if previous["status"] != "PASS_32_GREEDY_FULLMODEL_NUMERICAL_ONLY":
            raise ValueError("prior exact-L cold chain is not PASS")
        result["prior_cold"] = dict(path=config["prior_cold_receipt"],
                                   sha256=sha(config["prior_cold_receipt"]), status=previous["status"])
        base = module("v24_frozen_cold_numeric_resources", COLD / "fullqual.py")
        resources = base.Resources(args.out)
        resources.sample("beforeload", loaded=False)
        versions = {n: importlib.metadata.version(n) for n in ("mlx", "mlx-metal")}
        if versions != {"mlx": "0.32.3", "mlx-metal": "0.32.3"}:
            raise base.CheckFailure("runtime_version_differs", versions=versions)
        import mlx.core as mx
        import numpy as np
        from mlx.utils import tree_flatten
        if mx.default_device() != mx.gpu:
            raise base.CheckFailure("default_device_not_gpu")
        result["runtime"] = dict(versions=versions, device_info=mx.device_info(),
            python=sys.executable, numpy_version=np.__version__,
            selected_environment={k: v for k, v in os.environ.items()
                if k.startswith(("MLX_", "METAL_", "DYLD_"))})
        result["cache_limit_bytes"] = 2 << 30
        result["previous_cache_limit_bytes"] = mx.set_cache_limit(2 << 30)
        sys.path.insert(0, str(ACCEPTED.parent.parent))
        from tutorial_code.miniengine.model.loader import load_model
        from tutorial_code.miniengine.model.qwen3_5 import Qwen3_5ForCausalLM
        from tutorial_code.miniengine.cache import BatchKV, BatchState, DecodeBatch, make_cache
        from tutorial_code.miniengine.context import ForwardContext
        from tutorial_code.miniengine.eos import read_eos_ids

        def imported_identity():
            found = {}
            for name, mod in tuple(sys.modules.items()):
                if name.startswith("tutorial_code.miniengine") and getattr(mod, "__file__", None):
                    path = str(Path(mod.__file__).resolve())
                    if path not in pins or sha(path) != pins[path]:
                        raise base.CheckFailure("imported_source_not_accepted", module=name, path=path)
                    found[name] = dict(path=path, sha256=pins[path])
            return found

        result["imported_sources_initial"] = imported_identity()
        result["stage"] = "loading"; result["model_load_attempts"] += 1; save()
        model, cfg = load_model(workload["model_dir"], dtype=mx.bfloat16)
        result["model_loads"] += 1; loaded = True
        mx.synchronize()
        result["afterload"] = resources.sample("afterload-before-any-token", loaded=True, mx=mx)
        gdn = [l.linear_attn for l in model.layers if not l.is_attn]
        if len(model.layers) != 64 or len(gdn) != 48:
            raise base.CheckFailure("layer_counts_differ")
        for layer in gdn:
            layer.set_gates_compile(True)

        def parameter_identity():
            rows = [(k, id(v), list(v.shape), str(v.dtype)) for k, v in tree_flatten(model.parameters())]
            return dict(count=len(rows), identity_sha256=hashlib.sha256(json.dumps(rows).encode()).hexdigest())

        parameter_start = parameter_identity()
        result["parameters_initial"] = parameter_start
        result["stage"] = "ready"; save()
        write(args.out / "model-ready.json", dict(pid=os.getpid(), stage="ready", model_loads=1,
                                                  token_executed=False, resource=result["afterload"]))
        print(json.dumps(dict(event="MODEL_READY", pid=os.getpid())), flush=True)
        prototype = module("v24_frozen_functional_warm", COLD / "functional_decode.py")
        helper = prototype.FunctionalDecode(model, backend=mx,
            types=prototype.DecodeTypes(Qwen3_5ForCausalLM, ForwardContext, BatchKV, BatchState),
            runtime_version=versions["mlx"], device_kind="gpu")
        numeric = base.Numeric(mx, np, args.out)
        prompts = [list(workload["prompt_ids"]), list(workload["prompt_ids"])]
        change = config["prompt_change"]
        if prompts[1][change["index"]] != change["old_token"]:
            raise base.CheckFailure("prompt_change_source_differs")
        prompts[1][change["index"]] = change["new_token"]
        if sum(a != b for a, b in zip(*prompts)) != 1:
            raise base.CheckFailure("requires_one_prompt_token_change")
        if any(type(t) is not int or not 0 <= t < cfg.vocab_size
               for ids in prompts for t in ids):
            raise base.CheckFailure("illegal_prompt_token")
        eos = read_eos_ids(workload["model_dir"])
        result["eos_ids"] = sorted(eos)
        groups = []

        def prefill(group_index):
            ids = prompts[group_index]
            result["stage"] = dict(phase="prefill", group=group_index); save()
            caches = make_cache(model)
            logits = model(mx.array(ids, dtype=mx.int32), ForwardContext("prefill", caches), last_only=True)
            mx.eval(logits)
            raw, logit_sig = numeric.raw(logits, f"group{group_index}.prefill")
            raw_name = f"group{group_index}-prefill-logits.bin"
            (args.out / raw_name).write_bytes(raw)
            token = int(mx.argmax(logits, axis=-1).item())
            if token in eos:
                raise base.CheckFailure("prefill_seed_is_EOS", group=group_index, token=token)
            batch = DecodeBatch(model)
            batch.rebuild([], [("R0", caches, len(ids) + 32)])
            flat = tuple(batch.arrays()); mx.eval(*flat)
            if len(flat) != 128 or any(cache.K.shape[2] != 256
                    for layer, cache in zip(model.layers, batch.layers) if layer.is_attn):
                raise base.CheckFailure("requires128_arrays_and256_KV_capacity")
            sigs = numeric.signatures(flat, f"group{group_index}.seed")
            meta = dict(group=group_index, prompt_ids=ids, prompt_length=19,
                        first_greedy_token=token, logits=logit_sig, raw_logits=raw_name,
                        states=sigs, arrays=128, KV_capacity=256)
            result["groups"].append(meta); save()
            return dict(flat=flat, token=token, meta=meta)

        decode_calls = 0
        def eval_decode(logits, flat):
            nonlocal decode_calls
            then = time.perf_counter_ns(); mx.eval(logits, *flat)
            elapsed = time.perf_counter_ns() - then
            decode_calls += 1
            if decode_calls % 4 == 0:
                mx.synchronize()
                resources.sample(f"decode-call-{decode_calls:02d}", loaded=True, mx=mx)
            return elapsed

        def run_case(label, group, token, role):
            if not 0 <= token < cfg.vocab_size:
                raise base.CheckFailure("illegal_decode_token", label=label, token=token)
            result["stage"] = dict(phase="decode", case=label, exact_L=20); save()
            a_flat = tuple(a[:] for a in group["flat"])
            b_flat = tuple(a[:] for a in group["flat"])
            before_a = numeric.signatures(a_flat, label + ".A.input")
            before_b = numeric.signatures(b_flat, label + ".B.input")
            positions, lens = mx.array([19], dtype=mx.int32), mx.array([20], dtype=mx.int32)
            ctx = ForwardContext("decode", [BatchKV(a_flat[2*i][:], a_flat[2*i+1][:]) if l.is_attn
                else BatchState(a_flat[2*i], a_flat[2*i+1]) for i, l in enumerate(model.layers)],
                positions=positions, lens=lens, max_len=20)
            then = time.perf_counter_ns()
            a_logits = model(mx.array([token], dtype=mx.int32), ctx)
            a_call = time.perf_counter_ns() - then
            a_next = tuple(a for l, c in zip(model.layers, ctx.caches)
                           for a in ((c.K, c.V) if l.is_attn else (c.conv, c.S)))
            a_eval = eval_decode(a_logits, a_next)
            numeric.readonly(a_flat, before_a, label + ".A.after_A")
            numeric.readonly(b_flat, before_b, label + ".B.after_A")
            helper_before = helper.snapshot()
            then = time.perf_counter_ns()
            b_result = helper.functional(mx.array([token], dtype=mx.int32), positions, lens,
                                         b_flat, max_len=20, greedy=True)
            b_call = time.perf_counter_ns() - then
            b_logits, b_next = b_result[0], b_result[1:]
            b_eval = eval_decode(b_logits, b_next)
            logits_sig = numeric.equal(a_logits, b_logits, label + ".logits", persist=label + "-logits")
            if len(a_next) != 128 or len(b_next) != 128:
                raise base.CheckFailure("output_array_count_differs", label=label)
            state_sigs = [numeric.equal(a, b, f"{label}.state.{i}")
                          for i, (a, b) in enumerate(zip(a_next, b_next))]
            ro_a = numeric.readonly(a_flat, before_a, label + ".A.after_AB")
            ro_b = numeric.readonly(b_flat, before_b, label + ".B.after_AB")
            ta, tb = int(mx.argmax(a_logits, axis=-1).item()), int(mx.argmax(b_logits, axis=-1).item())
            if ta != tb:
                raise base.CheckFailure("greedy_output_differs", label=label, A=ta, B=tb)
            after = helper.snapshot()
            event = dict(label=label, input_role=role, input_token=token,
                prefill_first_greedy_token=group["token"], exact_L=20,
                input_positions=[19], input_lens=[20], input_states=before_b,
                logits=logits_sig, states=state_sigs, A_token=ta, B_token=tb,
                output_is_EOS=ta in eos, retained_input_readonly=dict(A=ro_a, B=ro_b),
                helper_before=helper_before, helper_after=after,
                cache_hit_delta=after["hits"]-helper_before["hits"],
                cache_miss_delta=after["misses"]-helper_before["misses"],
                host_ns=dict(A_fncall=a_call, B_fncall=b_call, A_eval=a_eval, B_eval=b_eval),
                timing_scope="instrumented numerical bookkeeping; not performance")
            result["cases"].append(event); save()
            print(json.dumps(dict(event="WARM_CASE_EQUAL", case=label,
                                  hit_delta=event["cache_hit_delta"])), flush=True)
            return event

        groups.append(prefill(0))
        if groups[0]["token"] != workload["continuation_ids"][0]:
            raise base.CheckFailure("base_prefill_seed_differs_from_cold_workload")
        t0 = groups[0]["token"]
        cold = run_case("base-cold", groups[0], t0, "actual_first_greedy_seed")
        repeat = run_case("base-warm-repeat", groups[0], t0, "actual_first_greedy_seed_replay")
        if (repeat["logits"]["A"] != cold["logits"]["A"] or
                [s["A"] for s in repeat["states"]] != [s["A"] for s in cold["states"]]):
            raise base.CheckFailure("same_input_reference_replay_changed")
        if cold["A_token"] != workload["continuation_ids"][1]:
            raise base.CheckFailure("base_decode_differs_from_cold_workload")
        groups.append(prefill(1))
        cache_case = run_case("changed-prefill-warm-shared-token", groups[1], t0,
            "actual_first_greedy_seed" if groups[1]["token"] == t0 else "controlled_cache_contrast")
        distinct = groups[1]["token"]
        if distinct == t0:
            distinct = next(t for t in config["token_contrast_priority"] if t != t0)
        token_case = run_case("changed-prefill-warm-distinct-token", groups[1], distinct,
            "actual_first_greedy_seed" if groups[1]["token"] != t0 else "controlled_token_contrast")

        def state_hashes(event, field):
            return [s["sha256"] for s in event["input_states"]] if field == "input" else [s["A"]["sha256"] for s in event["states"]]
        changed_inputs = [i for i, (a, b) in enumerate(zip(state_hashes(cold, "input"),
                                                        state_hashes(cache_case, "input"))) if a != b]
        def changed_outputs(a, b):
            return (a["logits"]["A"]["sha256"] != b["logits"]["A"]["sha256"] or
                    state_hashes(a, "output") != state_hashes(b, "output"))
        if not changed_inputs or not changed_outputs(cold, cache_case):
            raise base.CheckFailure("cache_value_contrast_not_observable")
        if (token_case["input_token"] == cache_case["input_token"] or
                state_hashes(token_case, "input") != state_hashes(cache_case, "input") or
                not changed_outputs(cache_case, token_case)):
            raise base.CheckFailure("token_value_contrast_not_observable")
        final = helper.snapshot()
        if (final["misses"], final["hits"], final["entries"], final["evictions"]) != (1, 3, 1, 0):
            raise base.CheckFailure("expected_one_cold_three_warm_calls_not_observed")
        if cold["cache_miss_delta"] != 1 or any(e["cache_hit_delta"] != 1
                for e in (repeat, cache_case, token_case)):
            raise base.CheckFailure("individual_compile_hit_engagement_differs")
        for i, group in enumerate(groups):
            numeric.readonly(group["flat"], group["meta"]["states"], f"group{i}.seed.final")
        if parameter_identity() != parameter_start:
            raise base.CheckFailure("parameter_identity_changed")
        result["contrasts"] = dict(cache_changed_input_indices=changed_inputs,
            cache_contrast_same_token=True, token_contrast_same_state=True,
            token_values=[cache_case["input_token"], token_case["input_token"]],
            original_outputs_changed_and_candidate_strictly_matched=True)
        result["imported_sources_final"] = imported_identity()
        result["decode_calls"] = decode_calls
        result["helper_final"] = final
        source_check()
        result["status"] = "PASS_SAME_L_WARM_FULLMODEL_NUMERICAL_ONLY"
        result["stage"] = "complete"
    except Exception as error:
        result["status"] = "FAIL_STOP"
        result["first_failure"] = getattr(error, "details", dict(kind=type(error).__name__, message=str(error)))
        result["traceback"] = traceback.format_exc()
    finally:
        try:
            if mx is not None:
                mx.synchronize()
        except Exception as error:
            result["finally_sync_failure"] = str(error); result["status"] = "FAIL_STOP"
        try:
            if resources is not None:
                result["finally_resource"] = resources.sample("finally", loaded=loaded, mx=mx)
        except Exception as error:
            result["finally_resource_failure"] = str(error); result["status"] = "FAIL_STOP"
        result["raw_resources"] = resources.records if resources else []
        result["helper_final"] = helper.snapshot() if helper else None
        result["instrumented_wall_seconds"] = time.perf_counter() - started
        save()
    return 0 if result["status"] == "PASS_SAME_L_WARM_FULLMODEL_NUMERICAL_ONLY" else 1


if __name__ == "__main__":
    raise SystemExit(main())
