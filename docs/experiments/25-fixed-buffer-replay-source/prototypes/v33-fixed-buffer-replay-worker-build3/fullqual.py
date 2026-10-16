"""Root-only fixed-L20 native replay qualification; --inert never imports MLX."""
import argparse
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
ROOT = P / "candidate-v17-gdn-gates-build1-root"
LIB = P / "v33-native-replay-build4/libv33replay.dylib"
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
    manifest = json.loads((P / "source-manifest-v17-gdn-gates-build1.json").read_text())
    actual = {str(f.relative_to(ROOT / "tutorial_code/miniengine")): sha(f)
              for f in (ROOT / "tutorial_code/miniengine").rglob("*.py")}
    expected = {key.removeprefix("tutorial_code/miniengine/"): value for key, value in manifest["files"].items()}
    assert actual == expected and len(actual) == 25
    return pins


class Native:
    def __init__(self, out):
        self.out = out
        self.lib = ctypes.CDLL(None)  # already injected before model construction
        specs = {
            "v33_pin": [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_char_p],
            "v33_begin": [], "v33_end_build": [],
            "v33_register_output": [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_char_p],
            "v33_restore_replay": [],
            "v33_copy_output": [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_void_p],
        }
        for name, args in specs.items():
            fn = getattr(self.lib, name); fn.argtypes = args; fn.restype = ctypes.c_int
        self.lib.v33_report_json.argtypes = []
        self.lib.v33_report_json.restype = ctypes.c_char_p

    def report(self, label):
        raw = self.lib.v33_report_json()
        assert raw is not None
        value = json.loads(raw.decode())
        write(self.out / (label + ".json"), value)
        return value

    def call(self, name, *args):
        code = getattr(self.lib, name)(*args)
        if code:
            self.report("native-first-failure")
            raise RuntimeError(f"{name} failed rc={code}")


class Shared:
    """Pin the MLX owner, bit-view, memoryviews and NumPy zero-copy buffer view."""
    def __init__(self, mx, np, value, label):
        self.original, self.label = value, label
        # NumPy/PEP3118 cannot name BF16. Equal-width integer View is metadata;
        # native actual contents-range resolution must still prove this address.
        self.alias = value.view(mx.uint16) if value.dtype == mx.bfloat16 else value
        mx.eval(self.alias)
        self.buffer = memoryview(self.alias)
        assert self.buffer.c_contiguous, ("noncontiguous buffer rejected", label)
        self.bytes_view = self.buffer.cast("B")
        self.numpy_view = np.frombuffer(self.bytes_view, dtype=np.uint8)
        assert self.numpy_view.base is not None and self.numpy_view.nbytes == self.buffer.nbytes == value.nbytes
        self.address = int(self.numpy_view.__array_interface__["data"][0])
        self.nbytes = int(value.nbytes)
        assert self.address > 0 and self.nbytes > 0
        self.meta = {"label": label, "address": self.address, "nbytes": self.nbytes,
                     "shape": list(value.shape), "dtype": str(value.dtype),
                     "original_object_id": id(value), "alias_object_id": id(self.alias),
                     "memoryview_c_contiguous": True, "numpy_api": "frombuffer; no copy fallback",
                     "bf16_equal_width_bit_view": self.alias is not value}

    def raw(self): return bytes(self.bytes_view)


def finite(np, raw, dtype):
    if dtype.endswith("bfloat16"):
        words = np.frombuffer(raw, dtype="<u2")
        return not bool(np.any((words & 0x7f80) == 0x7f80))
    if dtype.endswith("float32"):
        return bool(np.isfinite(np.frombuffer(raw, dtype="<f4")).all())
    if dtype.endswith("int32"): return True
    raise RuntimeError("unexpected snapshot dtype: " + dtype)


def snapshot(np, out, shared, directory, ordinal):
    data = shared.raw()
    assert finite(np, data, shared.meta["dtype"]), ("nonfinite", shared.label)
    filename = f"{directory}/{ordinal:03d}.bin"
    (out / filename).write_bytes(data)
    return {**shared.meta, "raw_file": filename, "sha256": hashlib.sha256(data).hexdigest(), "finite": True}


def readonly(out, holders, records, label):
    total = 0
    for holder, record in zip(holders, records):
        expected, current = (out / record["raw_file"]).read_bytes(), holder.raw()
        assert len(current) == record["nbytes"] and current == expected, ("original input/reference changed", label, holder.label)
        total += len(current)
    return {"label": label, "strict_bytes_equal": True, "arrays": len(records), "bytes": total}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path); ap.add_argument("--tokens", type=int, default=32)
    ap.add_argument("--native-library", type=Path, default=LIB); ap.add_argument("--inert", action="store_true")
    args = ap.parse_args(argv); library = args.native_library.resolve(); pins = source_check(library)
    w = json.loads((P / "workload.json").read_text())
    assert args.tokens == 32 and len(w["prompt_ids"]) == 19 and len(w["continuation_ids"]) == 32
    if args.inert or args.out is None:
        assert not any(n == "mlx" or n.startswith("mlx.") for n in sys.modules)
        print(json.dumps({"status": "PASS_CPU_INERT", "MLX_imported": False, "GPU_used": False,
                          "pins": len(pins), "scope": "one fixed L20 capture and two seed restores"}))
        return 0
    assert args.out.is_absolute()
    args.out.mkdir(exist_ok=False); (args.out / "seed").mkdir(); (args.out / "expected").mkdir()
    receipt = {"status": "STARTED", "stage": "beforeload", "pid": os.getpid(), "model_loads": 0,
               "source_pins": pins, "native_library": str(library), "warm_requests": 1,
               "capture_L": 20, "replays": [], "performance_point": False, "adopted": False,
               "autoregressive_replay": False, "ICB_claim": False, "teacherforced_fixed_seed": True}
    save = lambda: write(args.out / "receipt.json", receipt)
    save(); mx = None; resources = None; loaded = False; armed = False; native = None
    # All Shared objects remain live through both replays, including parameters.
    parameter_holders, seed_holders, output_holders = [], [], []
    try:
        assert os.environ.get("DYLD_INSERT_LIBRARIES") == str(library), "library must be injected at child startup"
        assert all(os.environ.get(k) == v for k, v in CONTROLS.items())
        assert {n: importlib.metadata.version(n) for n in ("mlx", "mlx-metal")} == {"mlx": "0.32.3", "mlx-metal": "0.32.3"}
        support = module("v33_resource_support", P / "v24-fullmodel-qual-build1/fullqual.py")
        resources = support.Resources(args.out); resources.sample("beforeload", loaded=False)
        native = Native(args.out)
        assert not any(n == "mlx" or n.startswith(("mlx.", "tutorial_code.")) for n in sys.modules), "preloaded runtime/model"
        import mlx.core as mx
        import numpy as np
        from mlx.utils import tree_flatten
        info = mx.device_info(); assert mx.default_device() == mx.gpu and info["device_name"] == "Apple M4 Max" and info["architecture"] == "applegpu_g16s"
        mx.set_cache_limit(2 << 30)
        receipt["runtime"] = {"device": info, "mlx_version": "0.32.3", "cache_limit_bytes": 2 << 30,
                              "environment": {**CONTROLS, "DYLD_INSERT_LIBRARIES": str(library)}}
        sys.path.insert(0, str(ROOT))
        from tutorial_code.miniengine.model.loader import load_model
        from tutorial_code.miniengine.cache import BatchKV, BatchState, DecodeBatch, make_cache
        from tutorial_code.miniengine.context import ForwardContext
        imported_sources = {}
        for name, mod in tuple(sys.modules.items()):
            if name.startswith("tutorial_code.miniengine") and getattr(mod, "__file__", None):
                path = str(Path(mod.__file__).resolve())
                assert path in pins and sha(path) == pins[path], ("imported source not pinned", name, path)
                imported_sources[name] = path
        receipt["actual_imported_sources"] = imported_sources
        model, cfg = load_model(w["model_dir"], dtype=mx.bfloat16)
        receipt["model_loads"] = 1; loaded = True; mx.synchronize()
        receipt["afterload"] = resources.sample("afterload-before-any-token", loaded=True, mx=mx)
        gdns = [layer.linear_attn for layer in model.layers if not layer.is_attn]
        assert len(model.layers) == 64 and len(gdns) == 48
        for gdn in gdns:
            gdn.set_gates_compile(True)
            if hasattr(gdn, "set_conv_silu"): gdn.set_conv_silu(False)
        receipt.update(stage="ready", gates_on=True, conv_off=True, functional_off=True)
        write(args.out / "model-ready.json", {"pid": os.getpid(), "model_loads": 1, "token_executed": False})
        save(); print(json.dumps({"event": "MODEL_READY", "pid": os.getpid()}), flush=True)

        def setup():
            caches = make_cache(model)
            logits = model(mx.array(w["prompt_ids"], dtype=mx.int32), ForwardContext("prefill", caches), last_only=True)
            batch = DecodeBatch(model); batch.rebuild([], [("v33", caches, 19 + 32)])
            mx.eval(logits, *batch.arrays())
            first = int(mx.argmax(logits, axis=-1).item()); assert first == w["continuation_ids"][0]
            return batch, first

        batch, token = setup()
        warm_ids = [token]
        for i in range(31):
            position = 19 + i
            ctx = ForwardContext("decode", batch.layers, positions=mx.array([position], mx.int32), lens=mx.array([position+1], mx.int32), max_len=position+1)
            logits = model(mx.array([token], mx.int32), ctx)
            pred = mx.argmax(logits, axis=-1); mx.eval(pred, *batch.arrays())
            token = int(pred.item()); assert token == w["continuation_ids"][i+1]; warm_ids.append(token)
        batch.clear(); mx.synchronize(); receipt["warm_ids"] = warm_ids
        assert all(callable(g._compiled_decode_gates) for g in gdns)
        resources.sample("after-one-32-output-warm", loaded=True, mx=mx)
        batch, seed_token = setup()
        seed_flat = tuple(value[:] for value in batch.arrays()); assert len(seed_flat) == 128
        tokens, positions, lens = mx.array([seed_token], mx.int32), mx.array([19], mx.int32), mx.array([20], mx.int32)
        mx.eval(tokens, positions, lens, *seed_flat); mx.synchronize()
        seed_values = (tokens, positions, lens, *seed_flat)
        for key, value in tree_flatten(model.parameters()):
            holder = Shared(mx, np, value, "parameter." + key); parameter_holders.append(holder)
            native.call("v33_pin", holder.address, holder.nbytes, holder.label.encode())
        assert len(parameter_holders) == 1349
        receipt["parameter_ranges"] = [h.meta for h in parameter_holders]
        receipt["parameter_pin_scope"] = "provisional actual pointers only; native validates every USED zero-copy binding against real contents/range/read-only reflection"
        native.report("native-parameter-pins")
        for i, value in enumerate(seed_values): seed_holders.append(Shared(mx, np, value, ("tokens", "positions", "lens")[i] if i < 3 else f"state.{i-3}"))
        seed_records = [snapshot(np, args.out, h, "seed", i) for i, h in enumerate(seed_holders)]
        receipt["seed_inputs"] = seed_records; save()
        ctx_layers = [BatchKV(seed_flat[2*i][:], seed_flat[2*i+1][:]) if layer.is_attn else BatchState(seed_flat[2*i], seed_flat[2*i+1])
                      for i, layer in enumerate(model.layers)]
        ctx = ForwardContext("decode", ctx_layers, positions=positions, lens=lens, max_len=20)
        mx.eval(*(v for c, layer in zip(ctx_layers, model.layers) for v in ((c.K,c.V) if layer.is_attn else (c.conv,c.S))))
        mx.synchronize(); resources.sample("seed-ready-before-capture", loaded=True, mx=mx)
        native.call("v33_begin"); armed = True
        captured_logits = model(tokens, ctx)
        captured_state = tuple(v for c, layer in zip(ctx.caches, model.layers) for v in ((c.K,c.V) if layer.is_attn else (c.conv,c.S)))
        assert len(captured_state) == 128
        mx.eval(captured_logits, *captured_state); mx.synchronize()
        native.call("v33_end_build"); armed = False
        built = native.report("native-built-arena")
        assert built["status"] == "OK" and built["arena_built"] and built["dispatches"] > 0 and built["native_replays"] == 0
        assert built["parameter_ranges_registered"] == len(parameter_holders)
        resources.sample("after-capture-arena-build", loaded=True, mx=mx)
        output_values = (captured_logits, *captured_state)
        for i, value in enumerate(output_values):
            holder = Shared(mx, np, value, "logits" if i == 0 else f"state.{i-1}"); output_holders.append(holder)
            native.call("v33_register_output", holder.address, holder.nbytes, holder.label.encode())
        expected = [snapshot(np, args.out, h, "expected", i) for i, h in enumerate(output_holders)]
        assert len(expected) == 129
        mapped = native.report("native-output-mappings")
        assert mapped["status"] == "OK" and len(mapped["outputs"]) == 129
        by_label = {row["label"]: row for row in mapped["outputs"]}
        assert len(by_label) == 129
        for holder in output_holders:
            row = by_label[holder.label]
            assert row["address"] == holder.address and row["bytes"] == holder.nbytes
            assert row["original_buffer"] > 0 and row["original_offset"] >= 0
        receipt["expected_outputs"] = expected
        receipt["seed_after_original_capture"] = readonly(args.out, seed_holders, seed_records, "after-original")
        receipt["stage"] = "native-replay-ready"; save()
        resources.sample("all-output-addresses-mapped", loaded=True, mx=mx)
        for replay in range(2):
            resources.sample(f"before-replay-{replay+1}", loaded=True, mx=mx)
            native.call("v33_restore_replay")
            rows = []
            for holder, record in zip(output_holders, expected):
                target = ctypes.create_string_buffer(holder.nbytes)
                native.call("v33_copy_output", holder.address, holder.nbytes, ctypes.cast(target, ctypes.c_void_p))
                actual, reference = target.raw, (args.out / record["raw_file"]).read_bytes()
                if actual != reference or not finite(np, actual, record["dtype"]):
                    (args.out / "first-replay-output.bin").write_bytes(actual)
                    raise RuntimeError(f"strict replay bytes/nonfinite failure replay={replay+1} output={holder.label}")
                rows.append({"label": holder.label, "bytes": len(actual), "finite": True, "strict_equal": True, "sha256": hashlib.sha256(actual).hexdigest()})
            seed_ro = readonly(args.out, seed_holders, seed_records, f"after-replay-{replay+1}")
            reference_ro = readonly(args.out, output_holders, expected, f"reference-after-replay-{replay+1}")
            replay_report = native.report(f"native-after-replay-{replay+1}")
            assert replay_report["status"] == "OK" and replay_report["native_replays"] == replay+1
            receipt["replays"].append({"replay": replay+1, "outputs": rows, "seed_readonly": seed_ro, "captured_reference_readonly": reference_ro})
            resources.sample(f"after-replay-{replay+1}", loaded=True, mx=mx); save()
        receipt.update(status="PASS_FIXED_L20_NATIVE_REPLAY_2_STRICT", stage="complete", output_arrays_per_replay=129, seed_arrays=131)
        save(); return 0
    except BaseException as error:
        receipt.update(status="FAIL_STOP", error=repr(error), traceback=traceback.format_exc(), capture_armed=armed)
        if native is not None:
            try: native.report("native-on-failure")
            except BaseException as report_error: receipt["native_report_error"] = repr(report_error)
        save(); raise
    finally:
        try:
            if mx is not None: mx.synchronize()
            if resources is not None: resources.sample("finally-before-exit", loaded=loaded, mx=mx)
        except BaseException as error:
            receipt.update(status="FAIL_STOP", finally_error=repr(error)); save(); raise
        save()


if __name__ == "__main__": raise SystemExit(main())
