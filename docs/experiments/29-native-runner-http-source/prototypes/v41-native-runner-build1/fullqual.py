"""Root-only C1 runner lifecycle qualification against frozen actual A hashes."""
import argparse
import ctypes
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import traceback

sys.dont_write_bytecode = True
D = Path(__file__).resolve().parent
P = D.parent
ROOT = P / "candidate-v39-conv-build1-root"
LIB = P / "v38-native-icb-build1/libv38replay.dylib"
LIB_SHA = "9274227bdfdc968e6849500ea3f84060b08448250367de5068815113c1c9387f"
QUAL = P / "v23-root-control-build1/fullmodel-v39native-attempt1/child-output/receipt.json"
QUAL_SHA = "33870a3a417135c5824f3252f60dbbb82e9f96f726c19038fe34f2c5734caf53"


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, v): Path(p).write_text(json.dumps(v, indent=2, allow_nan=False) + "\n")


def module(name, p):
    spec = importlib.util.spec_from_file_location(name, p)
    value = importlib.util.module_from_spec(spec); sys.modules[name] = value
    spec.loader.exec_module(value); return value


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path); ap.add_argument("--inert", action="store_true")
    args = ap.parse_args()
    pins = json.loads((D / "source-pins.json").read_text())
    for path, digest in pins.items(): assert sha(path) == digest, path
    assert sha(QUAL) == QUAL_SHA and sha(LIB) == LIB_SHA
    qualified = json.loads(QUAL.read_text())
    assert qualified["status"] == "PASS_NATIVE_CONV_ICB_BASE_RESET_CHANGED_VS_ACCEPTED25_STRICT"
    expected = qualified["requests"]
    assert [c["name"] for c in expected] == ["base", "reset-base", "changed-prefill"]
    w = json.loads((P / "workload.json").read_text())
    if args.inert:
        assert not any(n == "mlx" or n.startswith("mlx.") for n in sys.modules)
        print(json.dumps({"status": "PASS_CPU_INERT", "pins": len(pins), "GPU_used": False,
                          "qualification_scope": "C1 queued three fresh requests; all132 outputs each step compared to actual prior A hashes"}))
        return 0
    assert args.out is not None and args.out.is_absolute()
    args.out.mkdir()
    r = {"status": "STARTED", "stage": "beforeload", "pid": os.getpid(), "model_loads": 0,
         "source_pins": pins, "prior_qualification_sha256": QUAL_SHA, "requests": [],
         "performance_point": False, "adopted": False, "scheduler_max_seqs": 1}
    save = lambda: write(args.out / "receipt.json", r)
    save(); loaded = False; mx = None; resources = None
    try:
        assert os.environ.get("DYLD_INSERT_LIBRARIES") == str(LIB)
        support = module("v41_resources", P / "v24-fullmodel-qual-build1/fullqual.py")
        resources = support.Resources(args.out); resources.sample("beforeload", loaded=False)
        sys.path.insert(0, str(ROOT))
        import mlx.core as mx
        from tutorial_code.miniengine.engine.llm_engine import LLMEngine
        from tutorial_code.miniengine.sampling_params import SamplingParams
        e = LLMEngine(w["model_dir"], max_seqs=1, max_len=2048, max_batched_tokens=512,
                      dtype=mx.bfloat16, cache_limit_gb=2)
        loaded = True; r["model_loads"] = 1; mx.synchronize()
        r["afterload"] = resources.sample("afterload-before-token", loaded=True, mx=mx)
        r["stage"] = "ready"
        write(args.out / "model-ready.json", {"pid": os.getpid(), "model_loads": 1, "token_executed": False})
        save(); print(json.dumps({"event": "MODEL_READY", "pid": os.getpid()}), flush=True)
        gdns = [l.linear_attn for l in e.model.layers if not l.is_attn]
        assert len(gdns) == 48
        for g in gdns: g.set_gates_compile(True); g.set_conv_silu(True)
        assert e.runner._functional_decode_helper is None
        sampling = SamplingParams(max_tokens=32, temperature=0, ignore_eos=False)
        ordinary = e.generate([w["prompt_ids"]], sampling)
        assert ordinary == [w["continuation_ids"]]
        assert e.runner._ahead is None and e.runner.batch.size == 0 and not e.requests
        resources.sample("after-ordinary-warm", loaded=True, mx=mx)
        runtime = module("v41_adapter", D / "native_runner.py")
        adapter = runtime.install(e, LIB, LIB_SHA)
        captured_originals = {}
        original_build = adapter.build
        def observe_build(values):
            before = [bytes(adapter.shared(v, str(i)).bytes_view) for i, v in enumerate(values)]
            original_build(values)
            assert len(before) == len(adapter.inputs) == 131
            for h, original in zip(adapter.inputs, before):
                assert bytes(h.bytes_view) == original, ("build changed input", h.label)
            captured_originals["before_any_native_replay"] = before + [bytes(h.bytes_view) for h in adapter.outputs]
            r["build_input131_readonly"] = True
        adapter.build = observe_build
        requests = {}
        # Queue all three before the next schedule; this tests FCFS handoff and reset.
        for case in expected:
            rid = e.add_request(case["prompt_ids"], sampling)
            item = {"name": case["name"], "request_id": rid, "B_ids": [], "steps": []}
            r["requests"].append(item); requests[rid] = (case, item)
        baseline_originals = None; checks = 0; first_tokens = 0; schedules = 0
        while e.has_work():
            schedules += 1; assert schedules <= 200
            before = adapter.native_steps
            results = e.step()
            assert len(results) == 1, "C1 whole prompts must deliver exactly one token per step"
            rid, token, done = results[0]; case, item = requests[rid]
            item["B_ids"].append(token); ordinal = len(item["B_ids"])
            assert token == case["A_ids"][ordinal-1]
            if ordinal == 1:
                assert adapter.native_steps == before
                first_tokens += 1
            else:
                assert adapter.native_steps == before + 1
                reference = case["steps"][ordinal-2]
                assert reference["step"] == ordinal-1
                output_rows = []
                for h, prior in zip(adapter.outputs, reference["outputs"]):
                    assert h.label == prior["label"] and h.nbytes == prior["bytes"]
                    assert list(h.shape) == prior["shape"] and str(h.dtype) == prior["dtype"]
                    dest = ctypes.create_string_buffer(h.nbytes)
                    adapter.call("v34_copy_output", h.address, h.nbytes, dest)
                    actual_sha = hashlib.sha256(dest.raw).hexdigest()
                    assert actual_sha == prior["A_sha256"], (item["name"], ordinal, h.label)
                    output_rows.append({"label": h.label, "bytes": h.nbytes,
                                        "sha256": actual_sha, "qualified_A_sha256": prior["A_sha256"]})
                assert len(output_rows) == 132
                checks += 132; item["steps"].append({"step": ordinal-1, "checks": output_rows})
                if baseline_originals is None:
                    baseline_originals = captured_originals["before_any_native_replay"]
            if done:
                assert ordinal == 32 and item["B_ids"] == case["A_ids"]
                assert adapter.active is None and e.runner.batch.size == 0 and e.runner._ahead is None
                assert rid not in e.requests and rid not in e.runner.staging
                assert baseline_originals is not None
                for h, original in zip(adapter.inputs + adapter.outputs, baseline_originals):
                    assert bytes(h.bytes_view) == original, ("captured MLX owner mutated", h.label)
                item["captured_originals_readonly"] = True
                item["status"] = "PASS_32_GREEDY_AND_DROP"
                resources.sample("after-request-" + item["name"], loaded=True, mx=mx)
                print(json.dumps({"event": "RUNNER_REQUEST_STRICT_PASS", "name": item["name"], "steps": 31}), flush=True)
                save()
        assert checks == 12276 and first_tokens == 3 and adapter.native_steps == 93
        assert adapter.request_count == 3 and not e.requests and not e.runner.staging
        native = adapter.report(); f = native["feedback"]
        assert native["status"] == "OK" and f["resets"] == 3 and f["total_steps"] == 93
        assert f["icb"]["execution_count"] == 93 and f["icb"]["bank_executions"] == [48, 45]
        assert f["pingpong"]["eligible_count"] == 48
        r["core_native_after_three_queued_requests"] = native
        # Exercise whole-request fallback and a prefill-only completion, then
        # prove another native request resets correctly after both paths.
        fallback = e.generate([w["prompt_ids"]], SamplingParams(
            max_tokens=2, temperature=0.7, top_k=1, ignore_eos=False))
        assert fallback == [w["continuation_ids"][:2]] and adapter.native_steps == 93
        prefill_only = e.generate([w["prompt_ids"]], SamplingParams(max_tokens=1, temperature=0))
        assert prefill_only == [w["continuation_ids"][:1]] and adapter.native_steps == 93
        final_request = e.generate([w["prompt_ids"]], sampling)
        assert final_request == [w["continuation_ids"]] and adapter.native_steps == 124
        final_checks = []
        for h, prior in zip(adapter.outputs, expected[0]["steps"][-1]["outputs"]):
            assert h.label == prior["label"] and h.nbytes == prior["bytes"]
            dest = ctypes.create_string_buffer(h.nbytes)
            adapter.call("v34_copy_output", h.address, h.nbytes, dest)
            actual_sha = hashlib.sha256(dest.raw).hexdigest()
            assert actual_sha == prior["A_sha256"], ("post-fallback", h.label)
            final_checks.append({"label": h.label, "sha256": actual_sha,
                                 "qualified_A_sha256": prior["A_sha256"]})
        assert len(final_checks) == 132 and adapter.request_count == 4
        assert adapter.active is None and not e.requests and not e.runner.staging
        assert e.runner.batch.size == 0 and e.runner._ahead is None
        for h, original in zip(adapter.inputs + adapter.outputs, baseline_originals):
            assert bytes(h.bytes_view) == original, ("post-fallback owner mutated", h.label)
        native = adapter.report(); f = native["feedback"]
        assert native["status"] == "OK" and f["resets"] == 4 and f["total_steps"] == 124
        assert f["icb"]["execution_count"] == 124 and f["icb"]["bank_executions"] == [64, 60]
        r.update(status="PASS_NATIVE_RUNNER_C1_QUEUED_RESET_FALLBACK_STRICT", stage="complete",
                 strict_output_checks=checks+132, core_all132_steps=93, first_prefill_tokens=first_tokens,
                 native_steps=124, request_count=4, ordinary_warm_tokens=32,
                 whole_request_fallback_ids=fallback, prefill_only_ids=prefill_only,
                 post_fallback_final132=final_checks, post_fallback_ids=final_request,
                 terminal_cleanup=True, captured_originals_readonly=True, native=native)
        save(); return 0
    except BaseException as error:
        r.update(status="FAIL", stage="exception", error={"type": type(error).__name__, "message": str(error),
                                                         "traceback": traceback.format_exc()})
        save(); traceback.print_exc(); return 1
    finally:
        if resources is not None:
            r["finally_resource"] = resources.sample("finally", loaded=loaded, mx=mx)
            save()


if __name__ == "__main__": raise SystemExit(main())
