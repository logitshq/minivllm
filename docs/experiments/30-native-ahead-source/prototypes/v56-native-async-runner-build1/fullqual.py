"""Root-only async depth2 C1 lifecycle qualification; strict normal output gold and separate early-stop drain proof."""
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
ROOT = P / "candidate-v44-conv-normreuse-build1-root"
LIB = P / "v55-native-async-build2/libv55replay.dylib"
LIB_SHA = "bb3c6ebb035eb204a09e4e99828aa7438a11ff991a3a883cdb310a4128fa1e3e"
QUAL = P / "v23-root-control-build1/fullmodel-v44native-attempt1/child-output/receipt.json"
QUAL_SHA = "1dcfc41e7779c575602bd6c73bae4e4800664db42e1ee87e75414335bdc45b6d"
ADAPTER_SHA = "af252476a377e9624173a8dbb028650c283b56d5beb0248639b0f00c271dbd59"
TINY = P / "v23-root-control-build1/fullmodel-v55smoke-attempt1/child-output/receipt.json"
TINY_SHA = "9b5d6a58b387e0c11ef6d7091dd4e98efd80242041967bb3d94027cd8ca2d658"
TINY_RELEASE = P / "v23-root-control-build1/fullmodel-v55smoke-attempt1/external-release.json"
TINY_RELEASE_SHA = "62da405079d2ff8648a48256c1767e6acff879b899e3beeeb626781f6f6ce0f6"


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, v): Path(p).write_text(json.dumps(v, indent=2, allow_nan=False) + "\n")


def module(name, p):
    spec = importlib.util.spec_from_file_location(name, p)
    value = importlib.util.module_from_spec(spec); sys.modules[name] = value
    spec.loader.exec_module(value); return value


def assert_pipeline(native, *, delivered, completed, requests, discarded, banks,
                    request_delivered=31, request_completed=31, request_discarded=0,
                    last_readable=True):
    f, p = native["feedback"], native["pipeline"]
    assert native["status"] == "OK" and not native["error"]
    assert p["enabled"] and p["depth"] == 2 and p["maximum_outstanding"] == 2
    assert p["total_submitted"] == p["total_completed"] == completed
    assert p["total_delivered"] == delivered and p["total_discarded"] == discarded
    assert p["total_failed"] == p["request_failed"] == 0
    assert p["request_submitted"] == p["request_completed"] == request_completed
    assert p["request_delivered"] == request_delivered and p["request_discarded"] == request_discarded
    assert p["inflight"] == 0 and p["pending"] == [] and all(not x["busy"] for x in p["slots"])
    assert p["last_result_readable"] is last_readable
    assert f["total_steps"] == f["icb"]["execution_count"] == f["icb"]["encoded_count"] == completed
    assert f["resets"] == requests and f["icb"]["bank_executions"] == banks
    assert f["pingpong"]["eligible_count"] == 48


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path); ap.add_argument("--inert", action="store_true")
    args = ap.parse_args()
    pins = json.loads((D / "source-pins.json").read_text())
    for path, digest in pins.items(): assert sha(path) == digest, path
    manifest = json.loads((P / "v44-native-fullqual-build1/candidate-manifest.json").read_text())
    source_package = ROOT / "tutorial_code/miniengine"
    expected_source = {k.removeprefix("tutorial_code/miniengine/"): v for k, v in manifest["files"].items()}
    actual_source = {str(f.relative_to(source_package)): sha(f) for f in source_package.rglob("*.py")}
    assert actual_source == expected_source and len(actual_source) == 27
    assert sha(QUAL) == QUAL_SHA and sha(LIB) == LIB_SHA
    assert sha(D / "native_runner.py") == ADAPTER_SHA
    assert sha(TINY) == TINY_SHA and sha(TINY_RELEASE) == TINY_RELEASE_SHA
    tiny, released = json.loads(TINY.read_text()), json.loads(TINY_RELEASE.read_text())
    assert tiny["status"] == "PASS_TINY_V55_ASYNC_DEPTH2_FIFO_3REQUESTS_15STEPS_75STRICT"
    assert tiny["strict_output_checks"] == 75 and tiny["native"]["pipeline"]["depth"] == 2
    assert tiny["native"]["pipeline"]["total_completed"] == tiny["native"]["pipeline"]["total_delivered"] == 15
    assert released["status"] == "PASS_EXTERNAL_RELEASE" and released["GPU_released"] and released["tool_exit"] == 0
    assert all(x["returncode"] == 1 and x["stdout"] == x["stderr"] == "" for x in released["strict_absence"])
    qualified = json.loads(QUAL.read_text())
    assert qualified["status"] == "PASS_NATIVE_CONV_ACTIVATED_NORMREUSE_CACHE_ICB_BASE_RESET_CHANGED_VS_ACCEPTED25_STRICT"
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
         "source_pins": pins, "prior_qualification_sha256": QUAL_SHA,
         "tiny_qualification_sha256": TINY_SHA, "tiny_release_sha256": TINY_RELEASE_SHA, "requests": [],
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
        for g in gdns: g.set_gates_compile(True); g.set_conv_norm_reuse_cache(True)
        assert getattr(e.runner, "_functional_decode_helper", None) is None
        assert all(g._gates_compile_enabled is True and g._conv_norm_reuse_cache_enabled is True for g in gdns)
        r.update(source_root=str(ROOT), gates_on=48, conv_on=48, normreuse_cache_on=48,
                 conv_normreuse_cache_on=48, conv_off=False,
                 whole_functional_off=True, adapter_source=str(P / "v56-native-async-runner-build1/native_runner.py"),
                 adapter_sha256=sha(P / "v56-native-async-runner-build1/native_runner.py"))
        sampling = SamplingParams(max_tokens=32, temperature=0, ignore_eos=False)
        ordinary = e.generate([w["prompt_ids"]], sampling)
        assert ordinary == [w["continuation_ids"]]
        assert e.runner._ahead is None and e.runner.batch.size == 0 and not e.requests
        resources.sample("after-ordinary-warm", loaded=True, mx=mx)
        runtime = module("v56_adapter", P / "v56-native-async-runner-build1/native_runner.py")
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
        assert_pipeline(native, delivered=93, completed=93, requests=3, discarded=0, banks=[48, 45])
        assert adapter.total_submitted == 93 and adapter.total_discarded == 0
        assert all(x["submitted"] == 31 and x["discarded"] == 0 for x in adapter.drops)
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
        assert_pipeline(native, delivered=124, completed=124, requests=4, discarded=0, banks=[64, 60])
        assert not adapter.pending and adapter.total_submitted == 124 and adapter.total_discarded == 0
        assert len(adapter.drops) == 4 and all(x["submitted"] == 31 and x["discarded"] == 0 for x in adapter.drops)
        r.update(stage="normal-four-complete", normal_strict_output_checks=checks+132,
                 normal_native_delivered=124, normal_native_completed=124,
                 core_all132_steps=93, first_prefill_tokens=first_tokens, ordinary_warm_tokens=32,
                 whole_request_fallback_ids=fallback, prefill_only_ids=prefill_only,
                 post_fallback_final132=final_checks, post_fallback_ids=final_request,
                 normal_four_native=native)
        save()
        # Static stop token index6=103725 is unique in the preceding canonical prefix.
        # Index5=220 would stop earlier at index2 and is deliberately not used.
        stop_index = 6; stop_token = w["continuation_ids"][stop_index]
        assert stop_token == 103725 and stop_token not in w["continuation_ids"][:stop_index]
        early = e.generate([w["prompt_ids"]], SamplingParams(
            max_tokens=32, temperature=0, ignore_eos=False, stop_token_ids=[stop_token]))
        assert early == [w["continuation_ids"][:7]]
        assert adapter.native_steps == 130 and adapter.total_submitted == 131 and adapter.total_discarded == 1
        assert adapter.request_count == 5 and not adapter.pending and adapter.active is None
        assert not e.requests and not e.runner.staging and e.runner.batch.size == 0 and e.runner._ahead is None
        early_native = adapter.report()
        assert_pipeline(early_native, delivered=130, completed=131, requests=5, discarded=1,
                        banks=[68, 63], request_delivered=6, request_completed=7,
                        request_discarded=1, last_readable=False)
        assert len(adapter.drops) == 5 and adapter.drops[-1]["submitted"] == 7 and adapter.drops[-1]["discarded"] == 1
        for h, original in zip(adapter.inputs + adapter.outputs, baseline_originals):
            assert bytes(h.bytes_view) == original, ("early-stop owner mutated", h.label)
        # Nonempty drain invalidates the previous result. No old132 copy/claim here.
        r["early_stop"] = {"stop_token_index": stop_index, "stop_token_id": stop_token,
            "max_tokens": 32, "ids": early, "native_delivered_delta": 6,
            "native_completed_delta": 7, "discarded_delta": 1,
            "all132_output_checks": 0, "scope": "IDs plus FIFO/drain/drop/readonly proof only; no full-cache comparison for these six delivered steps",
            "native": early_native, "drop": adapter.drops[-1], "captured_originals_readonly": True}
        resources.sample("after-early-stop-drain", loaded=True, mx=mx); save()
        post_early = e.generate([w["prompt_ids"]], sampling)
        assert post_early == [w["continuation_ids"]] and adapter.native_steps == 161
        assert adapter.total_submitted == 162 and adapter.total_discarded == 1 and adapter.request_count == 6
        assert not adapter.pending and adapter.active is None and not e.requests and not e.runner.staging
        assert e.runner.batch.size == 0 and e.runner._ahead is None
        post_early_checks = []
        for h, prior in zip(adapter.outputs, expected[0]["steps"][-1]["outputs"]):
            assert h.label == prior["label"] and h.nbytes == prior["bytes"]
            assert list(h.shape) == prior["shape"] and str(h.dtype) == prior["dtype"]
            dest = ctypes.create_string_buffer(h.nbytes)
            adapter.call("v34_copy_output", h.address, h.nbytes, dest)
            actual_sha = hashlib.sha256(dest.raw).hexdigest()
            assert actual_sha == prior["A_sha256"], ("post-early-stop", h.label)
            post_early_checks.append({"label": h.label, "bytes": h.nbytes, "sha256": actual_sha,
                                      "qualified_A_sha256": prior["A_sha256"]})
        assert len(post_early_checks) == 132
        for h, original in zip(adapter.inputs + adapter.outputs, baseline_originals):
            assert bytes(h.bytes_view) == original, ("post-early-stop owner mutated", h.label)
        native = adapter.report()
        assert_pipeline(native, delivered=161, completed=162, requests=6, discarded=1, banks=[84, 78])
        assert len(adapter.drops) == 6 and adapter.drops[-1]["submitted"] == 31 and adapter.drops[-1]["discarded"] == 0
        r.update(status="PASS_ASYNC_NATIVE_RUNNER_C1_QUEUED_RESET_EARLYSTOP_FALLBACK_STRICT", stage="complete",
                 strict_output_checks=checks+264, normal_strict_output_checks=12408,
                 native_steps=161, native_steps_definition="adapter delivered steps; native feedback/ICB completed includes one discarded future",
                 native_completed_steps=162, total_submitted_steps=162, total_discarded_steps=1,
                 request_count=6, resets=6, post_early_stop_ids=post_early,
                 post_early_stop_final132=post_early_checks, terminal_cleanup=True,
                 captured_originals_readonly=True, all_drop_records=list(adapter.drops), native=native)
        assert r["strict_output_checks"] == 12540
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
