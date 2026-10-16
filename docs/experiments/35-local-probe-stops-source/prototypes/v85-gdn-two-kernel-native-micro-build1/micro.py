"""V85 local native GDN two-kernel micro; frozen V81 numerical preparation, CPU inert default."""
import argparse
import ctypes
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import sys
import statistics
import time
import traceback
sys.dont_write_bytecode = True
D = Path(__file__).resolve().parent
P = D.parent
BASE81 = P / "v81-gdn-two-kernel-native-range-build1"
LIB = BASE81 / "libv81replay.dylib"
V81_RECEIPT = P / "v23-root-control-build1/fullmodel-v81num-attempt1/child-output/receipt.json"
V81_SHA = "bd8e1b359462f210a5f539f0922a571b4f101eba730748b3a838b771f0f42f26"
PRIME = 128
REPEATS = 1024
SCHEDULE = (("A", "A"), ("B", "B")) + (("A", "B"), ("B", "A")) * 3 + (("A", "A"), ("B", "B"))
A_PATH = P / "v44-gdn-activated-normreuse-build1/gdn_activated_norm_reuse_cached_metal.py"
B_PATH = P / "v78-gdn-scaled-ready-build1/gdn_activated_scaled_ready_metal.py"
V78_RECEIPT = P / "v23-root-control-build1/numeric-v78-attempt1/child-output/receipt.json"
V78_SHA = "f5d8fc89eb16ad29f0efed700b1a6a8200a975fb145b2db99a12b3632ec7807a"
CONTROLS = {"MLX_MAX_OPS_PER_BUFFER": "50", "MLX_MAX_MB_PER_BUFFER": "50", "MLX_BFS_MAX_WIDTH": "20", "MLX_METAL_FAST_SYNCH": "0"}
GROUPS = (1, 2, 3, 4, 5, 7, 8, 16)
ARRAY_NAMES = ("activated", "g", "beta", "S", "q_ready", "k_ready", "v_ready", "scales", "o", "S_new")
ARRAY_BYTES = (20480, 192, 192, 3145728, 8192, 8192, 24576, 128, 24576, 3145728)


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
        assert sha(path) == expected, ("pinned file changed", path)
    assert str(D / "micro.py") in pins and str(LIB) in pins
    baseline = module("v85_frozen81_cpu_support", BASE81 / "qualify.py")
    old_pins, support = baseline.source_check()
    # The extended driver dictionary is deliberately not required to equal old56.
    assert len(old_pins) == 56
    assert all(pins.get(path) == expected for path, expected in old_pins.items()), "all original56 pins must be an unchanged subset"
    assert sha(V81_RECEIPT) == V81_SHA
    actual = json.loads(V81_RECEIPT.read_text())
    assert actual["status"] == "PASS_V81_30_PUBLIC_60_CAPTURED_480_NATIVE_STRICT_NO_TIMING"
    assert actual["pins"] == old_pins
    assert [actual[key] for key in ("public_strict_checks", "captured_strict_checks", "native_strict_checks")] == [30, 60, 480]
    assert actual["numeric_qualified"] and actual["actual_reflection_verified"] and actual["actual_S_Snew_slots"] == [6, 9]
    assert actual["timing_run"] is False and actual["original_arrays_readonly"] == 255
    return pins, support


def score(pairs):
    assert len(pairs) == 10
    ab = [pair for pair in pairs if pair["kind"] == "AB"]
    assert len(ab) == 6
    controls = {arm: [pair["slot2_div_slot1"] for pair in pairs if pair["kind"] == "control" and pair["order"][0] == arm] for arm in ("A", "B")}
    assert all(len(values) == 2 for values in controls.values())
    medians = {arm: statistics.median(next(block["ns_per_dispatch"] for block in pair["blocks"] if block["arm"] == arm) for pair in ab) for arm in ("A", "B")}
    gain = 1 - medians["B"] / medians["A"]
    wins = sum(pair["reduction"] > 0 for pair in ab)
    control_passed = all(all(.95 <= value <= 1.05 for value in values) and .98 <= statistics.median(values) <= 1.02 for values in controls.values())
    status = "CONTROL_DRIFT_STOP" if not control_passed else "EXPLORATORY_PASS_NOT_ADOPTED" if gain >= .05 and wins >= 5 else "STOP_MICRO_GATE"
    return {"status": status, "medians_ns_per_dispatch": medians, "medians_ns_per_bank_two_dispatches": {arm: value * 2 for arm, value in medians.items()},
            "median_reduction": gain, "wins": wins, "AB_BA_pairs": 6, "AA_BB_control_pairs": 4,
            "controls": controls, "control_passed": control_passed, "adopted": False}


def performance(args, result, pins, save, probe, call, report, expected, readonly):
    assert result["status"] == "PASS_V81_30_PUBLIC_60_CAPTURED_480_NATIVE_STRICT_NO_TIMING"
    assert result["numeric_qualified"] and result["actual_reflection_verified"] and result["actual_S_Snew_slots"] == [6, 9]
    assert result["pins"] == pins and sha(args.qualified_receipt) == args.qualified_sha == V81_SHA
    result.update(status="STARTED_V85_PERFORMANCE", timing_run=True, performance_enabled=True,
                  scope="V85 local15-fixed-bank native two-kernel GDN range micro; oneFFI/oneCB encode+commit+wait per block; no model/HTTP/adoption",
                  qualified_v81_receipt=str(args.qualified_receipt), qualified_v81_receipt_sha256=args.qualified_sha,
                  original56_pins_preserved=True, native_library=str(LIB), native_math_and_banks_unchanged=True,
                  prime_groups=PRIME, timed_groups=REPEATS, dispatches_per_timed_block=REPEATS * 30,
                  timed_terminal_checks=[], timed_terminal_checks_completed=0, pairs=[], adopted=False,
                  terminal_check_scope="final15banks times o/S per timed block only; not every dispatch")
    save()
    for pair_index, order in enumerate(SCHEDULE):
        probe(f"v85-pair{pair_index}-before")
        pair = {"pair": pair_index, "order": list(order), "kind": "control" if order[0] == order[1] else "AB", "blocks": []}
        result["pairs"].append(pair)
        for block_index, arm_name in enumerate(order):
            arm = 0 if arm_name == "A" else 1
            previous = report()
            call("v81_loop_range", arm, PRIME)
            start = time.perf_counter_ns()
            call("v81_loop_range", arm, REPEATS)
            end = time.perf_counter_ns()
            native = report()
            assert end > start and native["last_arm"] == arm and native["last_groups"] == REPEATS
            assert native["observed_encoded_dispatches"] == 30720 and native["last_CB_completed"]
            assert native["trailing_Buffers_barriers_last_loop"] == REPEATS and native["internal_state_barriers_per_range"] == 15
            assert native["total_encoded_dispatches"] - previous["total_encoded_dispatches"] == (PRIME + REPEATS) * 30
            assert native["total_groups"] - previous["total_groups"] == PRIME + REPEATS and native["loops"] - previous["loops"] == 2
            block = {"block": block_index, "arm": arm_name, "prime_groups": PRIME, "timed_groups": REPEATS,
                     "encoded_dispatches": 30720, "wall_ns": end - start, "ns_per_dispatch": (end - start) / 30720,
                     "ns_per_bank_two_dispatches": (end - start) / (REPEATS * 15), "native": native}
            pair["blocks"].append(block)
            first_check = len(result["timed_terminal_checks"])
            # These copies/hashes run after the clock. Only each block's final slots are checked.
            for bank in range(15):
                for output, size in enumerate(ARRAY_BYTES[8:]):
                    buffer = ctypes.create_string_buffer(size)
                    call("v81_copy_output", arm, bank, output, size, buffer)
                    data, reference = buffer.raw, expected[bank][output]
                    row = {"pair": pair_index, "block": block_index, "arm": arm_name, "bank": bank,
                           "output": ("o", "S_new")[output], "bytes": size,
                           "actual_sha256": hashlib.sha256(data).hexdigest(), "expected_sha256": hashlib.sha256(reference).hexdigest(),
                           "strict_equal": data == reference}
                    result["timed_terminal_checks"].append(row)
                    result["timed_terminal_checks_completed"] = sum(item["strict_equal"] for item in result["timed_terminal_checks"])
                    if data != reference:
                        (args.out / "timed-first-failure-A.raw").write_bytes(reference)
                        (args.out / "timed-first-failure-B.raw").write_bytes(data)
                        result["timed_first_failure"] = dict(row, first_byte_offset=next((i for i, (a, b) in enumerate(zip(reference, data)) if a != b), min(len(reference), len(data))))
                        save()
                        raise AssertionError(("timed terminal raw bytes differ", row))
            assert len(result["timed_terminal_checks"]) - first_check == 30
            block["terminal_check_indices"] = [first_check, len(result["timed_terminal_checks"])]
            block["terminal_strict_checks"] = 30
            save()
        first, second = pair["blocks"]
        if pair["kind"] == "control":
            pair["slot2_div_slot1"] = second["wall_ns"] / first["wall_ns"]
        else:
            values = {block["arm"]: block["wall_ns"] for block in pair["blocks"]}
            pair["reduction"] = 1 - values["B"] / values["A"]
        pair["readonly"] = readonly(f"v85-pair{pair_index}")
        probe(f"v85-pair{pair_index}-after")
        save()
    assert len(result["timed_terminal_checks"]) == result["timed_terminal_checks_completed"] == 600
    assert sum(len(pair["blocks"]) for pair in result["pairs"]) == 20
    result["performance_gate"] = score(result["pairs"])
    result.update(result["performance_gate"])
    # Keep pair records; score's pair counts have explicit AB_BA/AA_BB names.
    result.update(native_final=report(), timed_blocks=20, timed_terminal_strict_checks=600,
                  terminal_check_scope="600 final o/S array checks over20timed blocks; not all614400timed dispatch outputs")
    save()
    return 0 if result["status"] == "EXPLORATORY_PASS_NOT_ADOPTED" else 2


def gpu(args, pins, support):
    # These assertions precede runtime imports; root owns preload/process/GPU supervision.
    assert sys.executable == "/Users/ryanli/code/minivllm/.venv/bin/python"
    assert os.environ.get("DYLD_INSERT_LIBRARIES") == str(LIB)
    assert all(os.environ.get(key) == value for key, value in CONTROLS.items())
    assert importlib.metadata.version("mlx") == importlib.metadata.version("mlx-metal") == "0.32.3"
    import mlx.core as mx
    import mlx.nn as nn
    import numpy as np
    sys.path.insert(0, str(P))
    from resource_snapshot_power_v1 import snapshot, guard, deltas
    info = mx.device_info()
    assert mx.default_device() == mx.gpu and info["device_name"] == "Apple M4 Max" and info["architecture"] == "applegpu_g16s"
    mx.set_memory_limit(512 << 20)
    mx.set_cache_limit(64 << 20)
    out = args.out
    out.mkdir(exist_ok=False)
    (out / "numeric").mkdir()
    result = {"status": "STARTED", "scope": "V81 fixed15-bank local two-kernel native ICB numeric only", "pins": pins,
              "prior_v78_receipt": str(V78_RECEIPT), "prior_v78_receipt_sha256": V78_SHA,
              "cases": [], "public_AB_checks": [], "captured_checks": [], "native_checks": [],
              "timing_run": False, "performance_enabled": False, "model_loaded": False, "HTTP_run": False,
              "feedback": False, "adopted": False, "runtime": {"device": info, "controls": CONTROLS, "mlx": "0.32.3"}}
    save = lambda: write(out / "receipt.json", result)
    save()
    resources = []
    def probe(label):
        mx.synchronize()
        raw = snapshot(label, os.getpid())
        path = out / f"resource-{len(resources):02d}-raw.json"
        write(path, raw)
        resources.append(raw)
        deltas(resources)
        guard(resources)
        assert raw["power_metadata"]["source"] == "AC Power" and raw["processinfo_thermal_metadata"]["code"] == 0 and raw["pressure_level_value"] == 1
        assert raw["swap_used_mb"] == resources[0]["swap_used_mb"] and raw["vm_counters"]["Swapouts"] == resources[0]["vm_counters"]["Swapouts"]
        assert all(row["owned"] for row in raw["known_inference_processes"])
        write(out / "resources.json", resources)
    U, I, V, C = ctypes.c_uint64, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_char_p
    lib = ctypes.CDLL(None)
    for name, types in {"v33_pin": [U, U, C], "v33_begin": [], "v33_end_build": [], "v33_register_output": [U, U, C],
                        "v81_register_bank": [I, I, ctypes.POINTER(U), ctypes.POINTER(U)], "v81_build_ranges": [],
                        "v81_loop_range": [I, I], "v81_copy_output": [I, I, I, U, V], "v81_verify_readonly": []}.items():
        fn = getattr(lib, name)
        fn.argtypes, fn.restype = types, ctypes.c_int
    for name in ("v33_report_json", "v81_report_json"):
        fn = getattr(lib, name)
        fn.argtypes, fn.restype = [], C
    def report():
        return json.loads(lib.v81_report_json().decode())
    def call(name, *values):
        rc = getattr(lib, name)(*values)
        if rc:
            write(out / "native-failure.json", report())
            write(out / "capture-native40-failure.json", json.loads(lib.v33_report_json().decode()))
            raise RuntimeError((name, rc))
    def raw(array):
        mx.eval(array)
        if array.dtype == mx.bfloat16:
            host = np.ascontiguousarray(np.asarray(array.view(mx.uint16)), dtype="<u2")
            assert not np.any((host & 0x7f80) == 0x7f80)
        else:
            assert array.dtype == mx.float32
            host = np.ascontiguousarray(np.asarray(array), dtype="<f4")
            assert np.isfinite(host).all()
        return host.tobytes()
    def digest(data):
        return hashlib.sha256(data).hexdigest()
    def compare(category, a, b, metadata):
        row = dict(metadata, strict_equal=a == b, bytes=len(a), A_sha256=digest(a), B_sha256=digest(b))
        result[category].append(row)
        save()
        if a != b:
            (out / "first-failure-A.raw").write_bytes(a)
            (out / "first-failure-B.raw").write_bytes(b)
            result["first_failure"] = dict(row, first_byte_offset=next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b))))
            save()
            raise AssertionError(("strict raw bytes differ", row))
    owners, captured_holders, banks, expected, captured = [], [], [], [], []
    owner_hashes = []
    def readonly(label):
        now = [digest(bytes(holder.bytes_view)) for holder in owners + captured_holders]
        assert now == owner_hashes, ("original input or captured allocation mutated", label)
        call("v81_verify_readonly")
        return {"label": label, "original_arrays": len(now), "all_raw_sha256_unchanged": True, "native_shape_constants_unchanged": True}
    try:
        probe("before-bank-load")
        a_helper, b_helper = module("v81_A44", A_PATH), module("v81_B78", B_PATH)
        prior = module("v81_shared_memory_support", P / "v33-fixed-buffer-replay-worker-build3/fullqual.py")
        # Exactly 15 original u/g/beta/S banks; activation is external and fixed.
        for slot, name in enumerate(support.REAL + support.SYNTHETIC):
            directory = support.BANKROOT / name
            meta = json.loads((directory / "bank.json").read_text())
            bank = {}
            for key, (shape, dtype) in support.CONTRACT.items():
                host = np.fromfile(directory / meta["arrays"][key]["file"], dtype="<u2" if dtype == "bfloat16" else "<f4").reshape(shape)
                value = mx.array(host)
                bank[key] = value.view(mx.bfloat16) if dtype == "bfloat16" else value
            activated = nn.silu(bank["u"])
            mx.eval(activated, bank["g"], bank["beta"], bank["S"])
            mx.synchronize()
            bank["activated"] = activated
            for key in ("u", "activated", "g", "beta", "S"):
                owners.append(prior.Shared(mx, np, bank[key], f"bank{slot}.{name}.{key}"))
            before = [digest(bytes(holder.bytes_view)) for holder in owners[-5:]]
            A = a_helper.gated_delta_step_activated_norm_reuse_metal(activated, bank["g"], bank["beta"], bank["S"])
            B = b_helper.gated_delta_step_activated_norm_reuse_metal(activated, bank["g"], bank["beta"], bank["S"])
            mx.eval(*A, *B)
            reference = [raw(array) for array in A]
            for output, (aa, bb) in enumerate(zip(reference, B)):
                compare("public_AB_checks", aa, raw(bb), {"bank": slot, "name": name, "output": ("o", "S_new")[output]})
                (out / "numeric" / f"expected-b{slot:02d}-o{output}.raw").write_bytes(aa)
            assert before == [digest(bytes(holder.bytes_view)) for holder in owners[-5:]]
            banks.append(bank)
            expected.append(reference)
            result["cases"].append({"slot": slot, "bank": name, "kind": meta["kind"], "bank_json_sha256": sha(directory / "bank.json"), "input_readonly_before_capture": True})
            del A, B
        assert len(banks) == 15 and len(result["public_AB_checks"]) == 30
        result["fixed_input_payload_bytes"] = sum(holder.nbytes for i, holder in enumerate(owners) if i % 5 != 0)
        result["saved_S_payload_bytes"] = 15 * 3145728
        result["fixed_input_working_set_note"] = "S alone is47,185,920 decimal bytes (45MiB); 15 distinct S banks stay resident and rotate, no feedback"
        write(out / "inputs.json", [{**holder.meta, "sha256": digest(bytes(holder.bytes_view))} for holder in owners])
        mx.synchronize()
        probe("fifteen-fixed-banks-numeric-ready-before-capture")
        # Original u is not used in capture; only activated/g/beta/S are pinned.
        for i in range(15):
            for holder in owners[5 * i + 1:5 * i + 5]:
                call("v33_pin", holder.address, holder.nbytes, holder.label.encode())
        input_hashes = [digest(bytes(holder.bytes_view)) for holder in owners]
        call("v33_begin")
        for arm, helper in enumerate((a_helper, b_helper)):
            for slot, bank in enumerate(banks):
                ready = helper._prepare(bank["activated"], mx)
                # Keep every intermediate owner; capture actual prework completion before state.
                mx.eval(*ready)
                mx.synchronize()
                outputs = helper._state(*ready, bank["g"], bank["beta"], bank["S"], mx)
                mx.eval(*outputs)
                mx.synchronize()
                captured.append((ready, outputs))
        call("v33_end_build")
        assert input_hashes == [digest(bytes(holder.bytes_view)) for holder in owners]
        for index, (ready, outputs) in enumerate(captured):
            arm, slot = divmod(index, 15)
            holders = []
            for name, array in zip(ARRAY_NAMES[4:], (*ready, *outputs)):
                holder = prior.Shared(mx, np, array, f"arm{arm}.bank{slot}.{name}")
                captured_holders.append(holder)
                holders.append(holder)
                call("v33_register_output", holder.address, holder.nbytes, holder.label.encode())
            # Match captured manual two-kernel paths to the same public A44 terminal reference.
            for output, array in enumerate(outputs):
                compare("captured_checks", expected[slot][output], raw(array), {"arm": arm, "bank": slot, "output": ("o", "S_new")[output]})
            registered = owners[5 * slot + 1:5 * slot + 5] + holders
            assert tuple(holder.nbytes for holder in registered) == ARRAY_BYTES
            call("v81_register_bank", arm, slot, (U * 10)(*(holder.address for holder in registered)), (U * 10)(*ARRAY_BYTES))
        assert len(captured_holders) == 180 and len(result["captured_checks"]) == 60
        call("v81_build_ranges")
        built = report()
        assert built["enabled"] and built["capture_dispatches"] == built["ICB_commands"] == 60
        assert built["immutable_registered_ranges"] == built["immutable_allocations"] == 60 and built["mutable_registered_outputs"] == 180
        assert built["producer_consumer_edges_verified"] and built["original_and_arena_ranges_disjoint"]
        assert built["ranges"] == [[0, 30], [30, 60]] and built["internal_state_barriers_per_range"] == 15
        assert len(built["dispatch_records"]) == 60 and len(built["PSOs"]) == 4
        # Every state must actually expose S ReadOnly at6 and store-only S_new at9.
        for record in built["dispatch_records"]:
            if record["stage"] != 1:
                continue
            arguments = {arg["name"]: arg for arg in record["actual_active_arguments"]}
            assert arguments["S"]["index"] == 6 and arguments["S"]["reflection_access"] == 0
            assert arguments["S_new"]["index"] == 9 and arguments["S_new"]["reflection_access"] in (1, 2)
            assert record["ICB_barrier_before"] and record["captured_boundaries_before"] > 0
        write(out / "native-built.json", built)
        write(out / "capture-native40.json", json.loads(lib.v33_report_json().decode()))
        owner_hashes[:] = [digest(bytes(holder.bytes_view)) for holder in owners + captured_holders]
        write(out / "captured-originals.json", [{**holder.meta, "sha256": owner_hashes[len(owners) + i]} for i, holder in enumerate(captured_holders)])
        readonly_checks = [readonly("after-build")]
        for groups in GROUPS:
            for arm in (0, 1):
                call("v81_loop_range", arm, groups)
                actual = report()
                assert actual["observed_encoded_dispatches"] == 30 * groups and actual["last_CB_completed"]
                assert actual["trailing_Buffers_barriers_last_loop"] == groups
                for slot in range(15):
                    for output, size in enumerate(ARRAY_BYTES[8:]):
                        buffer = ctypes.create_string_buffer(size)
                        call("v81_copy_output", arm, slot, output, size, buffer)
                        data = buffer.raw
                        compare("native_checks", expected[slot][output], data, {"groups": groups, "arm": arm, "bank": slot, "output": ("o", "S_new")[output]})
                        # All480 comparisons save hashes; save the final60 terminal raw arrays as well.
                        if groups == GROUPS[-1]:
                            path = out / "numeric" / f"native-final-a{arm}-b{slot:02d}-o{output}.raw"
                            path.write_bytes(data)
                            result["native_checks"][-1]["raw_file"] = str(path.relative_to(out))
                readonly_checks.append(readonly(f"groups{groups}.arm{arm}"))
                write(out / f"native-g{groups}-a{arm}.json", actual)
                save()
        assert len(result["native_checks"]) == 480 and all(row["strict_equal"] for row in result["native_checks"])
        assert len(readonly_checks) == 17
        result.update(status="PASS_V81_30_PUBLIC_60_CAPTURED_480_NATIVE_STRICT_NO_TIMING", public_strict_checks=30,
                      captured_strict_checks=60, native_strict_checks=480, readonly_checks=readonly_checks,
                      original_arrays_readonly=255, actual_reflection_verified=True, actual_S_Snew_slots=[6, 9],
                      numeric_qualified=True, native_final=report(), timing_run=False, performance_enabled=False)
        probe("after-native-numeric-before-exit")
        save()
        if args.performance:
            return performance(args, result, pins, save, probe, call, report, expected, readonly)
        return 0
    except BaseException as error:
        result.update(status="FAIL_STOP", error=repr(error), traceback=traceback.format_exc(), numeric_qualified=False)
        save()
        raise
    finally:
        result["completed_strict_counts"] = {key: sum(row.get("strict_equal") is True for row in result[key]) for key in ("public_AB_checks", "captured_checks", "native_checks")}
        try:
            probe("finally-before-root-owned-exit")
            source_check()
        except BaseException as error:
            result.update(status="FAIL_STOP", final_validation_error=repr(error), numeric_qualified=False)
            save()
            raise
        save()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--inert", action="store_true")
    group.add_argument("--qualify-only", action="store_true")
    group.add_argument("--performance", action="store_true")
    ap.add_argument("--qualified-receipt", type=Path)
    ap.add_argument("--qualified-sha")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    pins, support = source_check()
    if args.performance:
        assert args.qualified_receipt and args.qualified_receipt.is_absolute()
        assert args.qualified_receipt.resolve() == V81_RECEIPT.resolve()
        assert args.qualified_sha == V81_SHA and sha(args.qualified_receipt) == args.qualified_sha
    if not (args.qualify_only or args.performance):
        assert not any(name == "mlx" or name.startswith(("mlx.", "numpy")) for name in sys.modules)
        print(json.dumps({"status": "PASS_CPU_INERT", "pins": len(pins), "old_pins_preserved": 56,
                          "GPU_used": False, "native_loaded": False, "native_library": str(LIB),
                          "saved_banks": 15, "dispatches_per_timed_block": 30720,
                          "AB_BA_pairs": 6, "AA_BB_controls": 4, "terminal_checks_per_timed_block": 30,
                          "expected_timed_terminal_checks": 600, "performance_requires_actual81_receipt_SHA": True}))
        return 0
    assert args.out and args.out.is_absolute()
    return gpu(args, pins, support)


if __name__ == "__main__":
    raise SystemExit(main())
