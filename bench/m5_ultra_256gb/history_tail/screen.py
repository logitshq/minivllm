"""E019 sole-owner extraction micro screen; no model, drafter, or core patch."""
import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import statistics
import sys
import time
import traceback


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
WIDTH = 25600
LENGTHS = (19, 128, 512, 2048)
SEED = 19019
POOL_BYTES = 2 * 1024 ** 3


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def build_cases():
    """All cases are declared before importing a device runtime or seeing data."""
    cases = []
    for n in LENGTHS:
        layouts = ["single", "short_tail", "cross_chunks", "offset_single",
                   "row_stride_cross", "column_stride_cross"]
        if n - 7 >= 65:
            layouts += ["old64_then_append", "old65_compress_then_append"]
        for layout in layouts:
            for role, fed in [(f"tail{tail}", n - tail) for tail in (1, 3, 7)] + [
                    ("reset", 0), ("caught_up", n)]:
                cases.append({"id": f"N{n}-{layout}-{role}", "N": n, "layout": layout,
                              "role": role, "fed": fed, "start": n, "rows": n - fed,
                              "timed": fed < n,
                              "gate_group": ("none" if fed == n else "long_tail"
                                             if n in (512, 2048) and role.startswith("tail") else "other")})
    return cases


def build_history(mx, flat, n, layout):
    """Preparation only: restore exact view layout from a frozen backing tensor."""
    dense = flat[:n * WIDTH].reshape(n, WIDTH)
    if layout == "single":
        return [dense]
    if layout == "short_tail":
        return [dense[:-7], dense[-7:]]
    if layout == "offset_single":
        return [flat[WIDTH + 1:WIDTH + 1 + n * WIDTH].reshape(n, WIDTH)]
    if layout == "row_stride_cross":
        dense = flat[:2 * n * WIDTH].reshape(2 * n, WIDTH)[::2]
    elif layout == "column_stride_cross":
        dense = flat[:2 * n * WIDTH].reshape(n, 2 * WIDTH)[:, ::2]
    elif layout in ("old64_then_append", "old65_compress_then_append"):
        count = 64 if layout == "old64_then_append" else 65
        previous = n - 7
        assert previous >= count
        parts = [dense[(i * previous) // count:((i + 1) * previous) // count] for i in range(count)]
        if len(parts) > 64:
            parts[:] = [mx.concatenate(parts)]
        parts.append(dense[-7:])
        return parts
    elif layout != "cross_chunks":
        raise ValueError("unknown frozen layout: " + layout)
    return [dense[:-3], dense[-3:-1], dense[-1:]]


def expected_concat_rows(history_rows, needed):
    if needed == 0:
        return []
    selected = []
    for rows in reversed(history_rows):
        take = min(needed, rows)
        if take:
            selected.append(take)
            needed -= take
        if not needed:
            break
    assert needed == 0
    selected.reverse()
    return [selected] if len(selected) > 1 else []


def evaluate_gate(cases, timing):
    by_id = {row["id"]: row for row in timing}
    assert len(by_id) == len(timing) and set(by_id) == {c["id"] for c in cases if c["timed"]}
    long_tail = [by_id[c["id"]]["speedup"] for c in cases if c["gate_group"] == "long_tail"]
    other = [by_id[c["id"]] for c in cases if c["gate_group"] == "other"]
    assert long_tail and other and all(math.isfinite(x) and x > 0 for x in long_tail)
    assert all(math.isfinite(row["speedup"]) and row["speedup"] > 0 for row in other)
    geomean = math.exp(statistics.mean(math.log(value) for value in long_tail))
    misses = [{"id": row["id"], "speedup": row["speedup"]} for row in other if row["speedup"] < 0.97]
    return {"long_tail_points": len(long_tail), "other_points": len(other),
            "long_tail_geomean": geomean, "other_below_097": misses,
            "advance": geomean >= 1.25 and not misses}


def run(args):
    import mlx.core as mx
    import numpy as np
    from candidate import full_history_reference, pending_features

    identity = {"python": sys.version, "mlx": importlib.metadata.version("mlx"),
                "mlx_metal": importlib.metadata.version("mlx-metal"),
                "numpy": importlib.metadata.version("numpy"), "device": mx.device_info()}
    identity["candidate_loaded_source"] = str(Path(pending_features.__code__.co_filename).resolve())
    assert Path(identity["candidate_loaded_source"]) == HERE / "candidate.py"
    assert Path(full_history_reference.__code__.co_filename).resolve() == HERE / "candidate.py"
    assert identity["mlx"] == identity["mlx_metal"] == "0.32.3", identity
    assert identity["device"]["device_name"] == "Apple M5 Ultra" and mx.default_device() == mx.gpu, identity
    mx.set_cache_limit(POOL_BYTES)
    sources = sorted(HERE.glob("*.py")) + [
        ROOT / "tutorial_code/miniengine/engine/spec_runner.py",
        ROOT / "tutorial_code/miniengine/engine/model_runner.py",
        ROOT / "docs/optimization-guide/tutorials/019-drafter-history-tail.md"]
    cases = build_cases()
    plan = {"identity": identity, "cases": cases, "width": WIDTH, "dtype": "bfloat16", "seed": SEED,
            "source_sha256": {str(p): digest(p) for p in sources}, "pool_limit_bytes": POOL_BYTES,
            "warmups_per_arm": 2, "abba_blocks": 3, "timing_metric": "build_eval_and_release_ms",
            "reference": "literal lazy concat(history)[fed:start]; only final result eval, never force full concat",
            "consumer": "output eval and host BF16 bit read only; no real fc/norm/drafter",
            "cold_compilation_claim": False, "no_model_or_full_draft_claim": True,
            "compression_domain": "64/65 nonempty old chunks require N-7>=65; absent at N19 by construction",
            "gate": {"long_N512_N2048_all_tails_geomean_min": 1.25, "all_other_nonempty_points_min": 0.97},
            "inputs": {}}
    write(args.out / "plan.json", plan)

    # Fixed finite BF16 data: preparation and existing >64 compression are
    # common to both arms and never charged to only one timed expression.
    for n in LENGTHS:
        flat = ((mx.arange((2 * n + 4) * WIDTH, dtype=mx.int32) + SEED) % 257 - 128)
        flat = (flat.astype(mx.float32) / 128).astype(mx.bfloat16)
        mx.eval(flat)
        path = args.out / f"inputs-N{n}.safetensors"
        mx.save_safetensors(str(path), {"flat": flat})
        plan["inputs"][str(n)] = {"file": path.name, "sha256": digest(path), "bytes": path.stat().st_size,
                                    "flat_shape": list(flat.shape), "dtype": str(flat.dtype),
                                    "generation": "bf16(((arange + 19019) % 257 - 128) / 128)"}
        del flat
    mx.clear_cache()
    preparation_peak = mx.get_peak_memory()
    write(args.out / "plan.json", plan)
    backing = {n: mx.load(str(args.out / plan["inputs"][str(n)]["file"]))["flat"] for n in LENGTHS}
    mx.eval(*backing.values())
    histories = {}
    for case in cases:
        key = (case["N"], case["layout"])
        if key not in histories:
            histories[key] = build_history(mx, backing[case["N"]], *key)
            mx.eval(histories[key])
    history_objects = {key: tuple(map(id, history)) for key, history in histories.items()}
    mx.reset_peak_memory()

    def bits(value):
        # Same-size reinterpretation preserves BF16 sign/NaN bits and works on
        # logical row/column views; do not convert BF16 values to FP32 here.
        assert value.dtype == mx.bfloat16
        return np.array(value.view(mx.uint16)).tobytes(order="C")

    def snapshot(history):
        return [{"shape": list(piece.shape), "dtype": str(piece.dtype),
                 "sha256": hashlib.sha256(bits(piece)).hexdigest()} for piece in history]

    def observe_concat(log):
        def concatenate(pieces, axis=0):
            log.append({"axis": axis, "rows": [piece.shape[0] for piece in pieces]})
            return mx.concatenate(pieces, axis=axis)
        return concatenate

    checks = []
    for case in cases:
        history = histories[(case["N"], case["layout"])]
        ids = tuple(map(id, history))
        before = snapshot(history)
        reference_calls, candidate_calls = [], []
        reference = full_history_reference(history, case["fed"], case["start"],
                                           concatenate=observe_concat(reference_calls))
        candidate = pending_features(history, case["fed"], case["start"], width=WIDTH,
                                     dtype=mx.bfloat16, concatenate=observe_concat(candidate_calls))
        if case["timed"]:
            assert reference is not None and candidate is not None
            mx.eval(reference, candidate)
            reference_bits, candidate_bits = bits(reference), bits(candidate)
            exact = (reference.shape == candidate.shape == (case["rows"], WIDTH)
                     and reference.dtype == candidate.dtype == mx.bfloat16
                     and reference_bits == candidate_bits)
            output = {"reference_sha256": hashlib.sha256(reference_bits).hexdigest(),
                      "candidate_sha256": hashlib.sha256(candidate_bits).hexdigest(),
                      "shape": list(candidate.shape), "dtype": str(candidate.dtype),
                      "candidate_same_piece_indices": [i for i, piece in enumerate(history) if candidate is piece]}
            del reference_bits, candidate_bits
        else:
            exact = reference is None and candidate is None
            output = {"reference_none": reference is None, "candidate_none": candidate is None}
        after = snapshot(history)
        expected_rows = expected_concat_rows([p.shape[0] for p in history], case["rows"])
        expected_reference_rows = [[p.shape[0] for p in history]] if case["timed"] and len(history) > 1 else []
        calls_match = ([call["rows"] for call in candidate_calls] == expected_rows
                       and all(call["axis"] == 0 for call in candidate_calls)
                       and [call["rows"] for call in reference_calls] == expected_reference_rows
                       and all(call["axis"] == 0 for call in reference_calls))
        unchanged = before == after and ids == tuple(map(id, history))
        row = {**case, "output": output, "exact": exact, "inputs_before": before, "inputs_after": after,
               "input_objects_unchanged": ids == tuple(map(id, history)), "inputs_unchanged": unchanged,
               "reference_python_concat_calls": reference_calls, "candidate_python_concat_calls": candidate_calls,
               "expected_candidate_concat_rows": expected_rows, "expected_reference_concat_rows": expected_reference_rows,
               "concat_call_contract": calls_match,
               "python_calls_do_not_prove_physical_metal_traffic": True,
               "passed": exact and unchanged and calls_match}
        checks.append(row)
        write(args.out / f"qualification-{case['id']}.json", row)
        if not row["passed"] and reference is not None and candidate is not None:
            mx.save_safetensors(str(args.out / f"failure-output-{case['id']}.safetensors"),
                                {"reference": reference, "candidate": candidate})
        del reference, candidate
    write(args.out / "qualification.json", checks)
    assert len(checks) == len(cases) and all(row["passed"] for row in checks), "E019 byte/input/concat qualification failed; no timing"
    gc.collect()
    mx.clear_cache()
    before_timing_active = mx.get_active_memory()

    def measure(call):
        # All input objects stay alive outside the arm; only this call's result
        # is released here. Candidate returning an input adds/drops a reference.
        # Reference's lazy full concat is never separately evaluated or retained.
        started = time.perf_counter_ns()
        result = call()
        mx.eval(result)
        evaluated = time.perf_counter_ns()
        del result
        released = time.perf_counter_ns()
        return {"build_and_eval_ms": (evaluated - started) / 1e6,
                "release_ms": (released - evaluated) / 1e6,
                "build_eval_and_release_ms": (released - started) / 1e6}

    timing = []
    for case in cases:
        if not case["timed"]:
            continue
        history = histories[(case["N"], case["layout"])]
        calls = {"reference": lambda: full_history_reference(history, case["fed"], case["start"], concatenate=mx.concatenate),
                 "candidate": lambda: pending_features(history, case["fed"], case["start"], width=WIDTH,
                                                         dtype=mx.bfloat16, concatenate=mx.concatenate)}
        first = {arm: measure(calls[arm]) for arm in ("reference", "candidate")}
        warmups = {arm: [measure(calls[arm]) for _ in range(2)] for arm in ("reference", "candidate")}
        rounds = []
        for block in range(3):
            for arm in ("reference", "candidate", "candidate", "reference"):
                rounds.append({"block": block, "arm": arm, **measure(calls[arm])})
        medians = {arm: statistics.median(row["build_eval_and_release_ms"] for row in rounds if row["arm"] == arm)
                   for arm in ("reference", "candidate")}
        row = {**case, "first_post_qualification_ms": first, "warmups_ms": warmups, "rounds": rounds,
               "median_build_eval_and_release_ms": medians, "speedup": medians["reference"] / medians["candidate"]}
        timing.append(row)
        write(args.out / f"timing-{case['id']}.json", row)
    write(args.out / "timing.json", timing)
    gate = evaluate_gate(cases, timing)
    post_inputs = {f"N{n}-{layout}": snapshot(history) for (n, layout), history in histories.items()}
    before_inputs = {f"N{row['N']}-{row['layout']}": row["inputs_before"] for row in checks}
    write(args.out / "inputs-after-timing.json", post_inputs)
    assert post_inputs == before_inputs, "history changed during timed extraction"
    assert history_objects == {key: tuple(map(id, history)) for key, history in histories.items()}, "history objects replaced during timing"
    for path, expected in plan["source_sha256"].items():
        assert digest(path) == expected, "source changed during screen: " + path
    for metadata in plan["inputs"].values():
        assert digest(args.out / metadata["file"]) == metadata["sha256"], "frozen input changed"
    status = "PASS_MODULE_SCREEN" if gate["advance"] else "REJECT_PERFORMANCE_GATE"
    summary = {"status": status, "qualification_passed": True, "qualification_cases": len(checks),
               "timed_cases": len(timing), "untimed_none_cases": sum(not c["timed"] for c in cases),
               "gate": gate, "advance": gate["advance"], "input_preparation_peak_bytes": preparation_peak,
               "micro_peak_bytes": mx.get_peak_memory(), "before_timing_active_bytes": before_timing_active,
               "pool_limit_bytes": POOL_BYTES, "all_timed_histories_unchanged": True,
               "source_and_frozen_inputs_unchanged": True, "cold_compilation_claim": False,
               "scope": "pure feature extraction; no full draft, model, token, accept, or Metal traffic claim"}
    write(args.out / "summary.json", summary)
    print(json.dumps({"status": status, "qualification_cases": len(checks), "timed_cases": len(timing),
                      "long_tail_geomean": gate["long_tail_geomean"],
                      "other_below_097_count": len(gate["other_below_097"])}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    try:
        run(args)
    except BaseException as exc:
        write(args.out / "failure.json", {"type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
