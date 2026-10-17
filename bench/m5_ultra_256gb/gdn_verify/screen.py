"""E018 sole-owner GPU microqualification; model used only to obtain real inputs."""
import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time
import traceback
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tutorial_code"))


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    try:
        run(args)
    except BaseException as exc:
        write(args.out / "failure.json", {"type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc()})
        raise


def run(args):
    import importlib.metadata
    import mlx.core as mx
    import numpy as np
    from candidate import verify_history
    from miniengine.cache import BatchState, make_cache
    from miniengine.context import ForwardContext
    from miniengine.kernels import gdn_metal
    from miniengine.model.loader import load_model

    identity = {"mlx": importlib.metadata.version("mlx"), "device": mx.device_info()}
    assert identity["mlx"] == "0.32.3" and identity["device"]["device_name"] == "Apple M5 Ultra", identity
    assert gdn_metal.SPLIT == 4 and gdn_metal.VALUE_TILE == 32
    mx.set_cache_limit(2 * 1024 ** 3)
    workload = json.loads(args.workload.read_text())
    assert len(workload["prompt_ids"]) == 19 and len(workload["continuation_ids"]) >= 8
    target = Path(workload["model_dir"])
    sources = list((ROOT / "tutorial_code/miniengine").rglob("*.py")) + list(Path(__file__).parent.glob("*.py"))
    plan = {"identity": identity, "workload_sha256": digest(args.workload),
            "source_sha256": {str(p): digest(p) for p in sources},
            "model": {"path": str(target), "revision": target.name,
                      "files": {p.name: digest(p) for p in sorted(target.glob("*"))
                                if p.is_file() and p.suffix in (".json", ".safetensors")}},
            "lengths": [1, 3, 5, 8], "states": ["real", "zero", "independent_random"],
            "warmups_per_arm": 2, "abba_blocks": 3,
            "gate": {"all_output_and_history_bytes_exact": True, "all_keep_prefixes_exact": True,
                     "initial_state_unchanged": True, "T3_T5_T8_geomean_speedup_min": 1.25,
                     "T1_speedup_min": 0.97, "no_T_regression_below": 0.97},
            "scope": "isolated literal helper vs same helper serial chain; not old chunk WY equivalence or whole-model speculation",
            "pool_limit_bytes": 2 * 1024 ** 3}
    write(args.out / "plan.json", plan)

    # First GDN layer has no upstream decoder layers; its real per-token inputs
    # can be captured without running the rest of the target at every position.
    loaded = load_model(str(target))
    model = loaded[0] if isinstance(loaded, tuple) else loaded
    assert not model.layers[0].is_attn
    assert sum(not layer.is_attn for layer in model.layers) == 48
    caches = make_cache(model)
    logits = model(mx.array(workload["prompt_ids"]), ForwardContext("prefill", caches), last_only=True)
    mx.eval(logits, caches[0].S, caches[0].conv_state)
    mixer = model.layers[0].linear_attn
    state = BatchState(caches[0].conv_state[None], caches[0].S[None])
    original = gdn_metal.gated_delta_step_metal
    captured = []

    def capture(q, k, v, g, beta, initial):
        captured.append((q, k, v, g, beta, initial))
        return original(q, k, v, g, beta, initial)

    input_ids = workload["continuation_ids"][:8]
    with patch.object(gdn_metal, "gated_delta_step_metal", capture):
        for token in input_ids:
            x = model.layers[0].input_layernorm(model.embed_tokens(mx.array([token])))
            y = mixer._decode(x, state)
            mx.eval(y, state.S, state.conv)
    assert len(captured) == 8
    arrays = {name: mx.stack([row[i] for row in captured]) for i, name in enumerate(("q", "k", "v", "g", "beta"))}
    arrays["initial_state"] = captured[0][5]
    mx.eval(*arrays.values())
    mx.save_safetensors(str(args.out / "real-inputs.safetensors"), arrays)
    write(args.out / "real-inputs.json", {"layer_index": 0, "prefix_ids": workload["prompt_ids"],
          "input_ids": input_ids, "scope": "real layer-0 continuation inputs; no claim these eight tokens are a fresh whole-model rollout",
          "sha256": digest(args.out / "real-inputs.safetensors"),
          "arrays": {name: {"shape": list(value.shape), "dtype": str(value.dtype)} for name, value in arrays.items()}})
    # Reload materialized standalone arrays so benchmarking does not retain a
    # lazily captured graph or target weights through an observer reference.
    del arrays, captured, model, loaded, mixer, caches, state, logits, y, x
    gc.collect()
    mx.clear_cache()
    preparation_peak_memory = mx.get_peak_memory()
    after_target_release_memory = mx.get_active_memory()
    mx.reset_peak_memory()
    arrays = mx.load(str(args.out / "real-inputs.safetensors"))
    mx.eval(*arrays.values())
    q, k, v, g, beta = (arrays[name] for name in ("q", "k", "v", "g", "beta"))
    initial = arrays["initial_state"]
    random_state = mx.array(np.random.default_rng(11).normal(0, 0.125, size=initial.shape).astype(np.float32))
    states = {"real": initial, "zero": mx.zeros_like(initial), "independent_random": random_state}
    mx.eval(*states.values())

    def serial(inputs, start):
        state = start
        outputs, history = [], []
        for t in range(inputs[0].shape[0]):
            output, state = original(*(x[t] for x in inputs), state)
            outputs.append(output)
            history.append(state)
        return outputs, history

    def raw(value):
        return np.array(value).tobytes()

    def equality(label, reference, candidate):
        a, b = raw(reference), raw(candidate)
        return {"label": label, "shape": list(reference.shape), "dtype": str(reference.dtype),
                "exact": reference.shape == candidate.shape and reference.dtype == candidate.dtype and a == b,
                "reference_sha256": hashlib.sha256(a).hexdigest(), "candidate_sha256": hashlib.sha256(b).hexdigest()}

    checks = []
    for state_name, start in states.items():
        for length in plan["lengths"]:
            before = raw(start)
            inputs = [x[:length] for x in (q, k, v, g, beta)]
            outputs, history = serial(inputs, start)
            candidate_out, candidate_hist = verify_history(*inputs, start)
            mx.eval(outputs, history, candidate_out, candidate_hist)
            comparisons = []
            for t in range(length):
                comparisons += [equality(f"output-{t}", outputs[t], candidate_out[t]),
                                equality(f"history-{t}", history[t], candidate_hist[t])]
            for keep in range(length + 1):
                if keep:
                    _, independent_history = serial([x[:keep] for x in inputs], start)
                    mx.eval(independent_history)
                    expected = independent_history[-1]
                    chosen = candidate_hist[keep - 1]
                else:
                    expected = chosen = start
                comparisons.append(equality(f"independent-prefix-keep-{keep}", expected, chosen))
            initial_unchanged = raw(start) == before
            row = {"state": state_name, "T": length, "comparisons": comparisons,
                   "initial_state_unchanged": initial_unchanged,
                   "passed": initial_unchanged and all(c["exact"] for c in comparisons)}
            checks.append(row)
            write(args.out / f"check-{state_name}-T{length}.json", row)
    write(args.out / "qualification.json", checks)
    assert all(row["passed"] for row in checks), "strict literal output/history/prefix qualification failed"
    del outputs, history, candidate_out, candidate_hist, independent_history, expected, chosen, _
    del states, random_state, start, inputs
    gc.collect()
    mx.clear_cache()
    before_timing_active_memory = mx.get_active_memory()

    timing = []
    for length in plan["lengths"]:
        inputs = [x[:length] for x in (q, k, v, g, beta)]
        def measure(arm):
            started = time.perf_counter_ns()
            result = serial(inputs, initial) if arm == "reference" else verify_history(*inputs, initial)
            mx.eval(result)
            return (time.perf_counter_ns() - started) / 1e6
        first = {arm: measure(arm) for arm in ("reference", "candidate")}
        warmups = {arm: [measure(arm) for _ in range(2)] for arm in ("reference", "candidate")}
        rounds = []
        for block in range(3):
            for arm in ("reference", "candidate", "candidate", "reference"):
                rounds.append({"block": block, "arm": arm, "ms": measure(arm)})
        medians = {arm: statistics.median(row["ms"] for row in rounds if row["arm"] == arm)
                   for arm in ("reference", "candidate")}
        row = {"T": length, "first_post_qualification_ms": first, "warmups_ms": warmups,
               "rounds": rounds, "medians_ms": medians, "speedup": medians["reference"] / medians["candidate"]}
        timing.append(row)
        write(args.out / f"timing-T{length}.json", row)
    speedups = {row["T"]: row["speedup"] for row in timing}
    geomean = math.exp(statistics.mean(math.log(speedups[t]) for t in (3, 5, 8)))
    advance = geomean >= 1.25 and all(value >= 0.97 for value in speedups.values())
    for path, expected in plan["source_sha256"].items():
        assert digest(path) == expected, "source changed during screen: " + path
    assert {p.name: digest(p) for p in sorted(target.glob("*"))
            if p.is_file() and p.suffix in (".json", ".safetensors")} == plan["model"]["files"], "target files changed during screen"
    assert digest(args.workload) == plan["workload_sha256"], "workload changed during screen"
    status = "PASS_MODULE_SCREEN" if advance else "REJECT_PERFORMANCE_GATE"
    write(args.out / "summary.json", {"status": status, "qualification_passed": True, "advance": advance,
          "qualification_cases": len(checks), "exact_comparisons": sum(len(r["comparisons"]) for r in checks),
          "timing": timing, "T3_T5_T8_geomean_speedup": geomean,
          "scope": plan["scope"], "micro_peak_memory_bytes": mx.get_peak_memory(),
          "preparation_peak_memory_bytes": preparation_peak_memory,
          "after_target_release_active_memory_bytes": after_target_release_memory,
          "before_timing_active_memory_bytes": before_timing_active_memory,
          "cold_compilation_claim": False, "no_whole_model_correctness_or_speed_claim": True})
    print(json.dumps({"status": status, "qualification_passed": True, "advance": advance,
                      "speedups": speedups, "T3_T5_T8_geomean": geomean}), flush=True)


if __name__ == "__main__":
    main()
