"""E015: instrumented official oMLX EnginePool qualification, never HTTP timing.

Run with the installed official oMLX environment, in one sole GPU-owner process.
Imports which can initialize MLX are deferred until after argument parsing.
"""
from __future__ import annotations

import argparse
import asyncio
import dataclasses
import functools
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import platform
import sys
import traceback
from unittest.mock import patch


EXPECTED = {"omlx": "0.7.1.dev1", "mlx": "0.32.3", "mlx-metal": "0.32.3",
            "mlx-vlm": "0.7.7", "mlx-lm": "0.31.4.dev132+g94cdcae13"}
PREFIXES = ("omlx", "mlx", "mlx_lm", "mlx_vlm")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def freeze_distributions():
    versions = {name: importlib.metadata.version(name) for name in EXPECTED}
    assert versions == EXPECTED, versions
    files = {}
    for name in EXPECTED:
        dist = importlib.metadata.distribution(name)
        for entry in dist.files or []:
            if str(entry).endswith((".py", ".so", ".dylib", ".metallib", "METADATA")):
                path = Path(dist.locate_file(entry)).resolve()
                if path.is_file():
                    files[str(path)] = digest(path)
    return {"versions": versions, "files": files}


def model_identity(directory):
    directory = Path(directory).resolve()
    files = [p for p in directory.iterdir() if p.is_file() and
             (p.suffix in (".json", ".safetensors") or p.name == "chat_template.jinja")]
    return {"path": str(directory), "revision": directory.name,
            "files": {p.name: {"size": p.stat().st_size, "sha256": digest(p)}
                      for p in sorted(files)}}


def loaded_sources():
    result = {}
    for name, module in tuple(sys.modules.items()):
        if any(name == p or name.startswith(p + ".") for p in PREFIXES):
            path = getattr(module, "__file__", None)
            if path and Path(path).is_file():
                path = Path(path).resolve()
                result[name] = {"file": str(path), "sha256": digest(path)}
    return result


class Observer:
    """Call-through observers; no tensor evaluation, sampler or return changes."""

    def __init__(self, out):
        self.out = Path(out)
        self.events = []
        self.current = None
        self.patches = []
        self.seams = {}

    def event(self, kind, **data):
        record = {"kind": kind, **data}
        self.events.append(record)
        # Retain partial evidence even when the model raises or the child is killed.
        with (self.out / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")

    def hook(self, owner, name, wrapper):
        original = getattr(owner, name)
        path = inspect.getsourcefile(original)
        self.seams[name] = {"module": original.__module__, "qualname": original.__qualname__,
                            "file": path, "sha256": digest(path)}
        replacement = functools.wraps(original)(wrapper(original))
        item = patch.object(owner, name, replacement)
        item.start()
        self.patches.append(item)

    def install(self):
        from omlx.engine.vlm import VLMBatchedEngine
        from omlx.scheduler import Scheduler
        from omlx.speculative import vlm_mtp
        from mlx_vlm.speculative import mtp

        def load(original):
            def call(path):
                result = original(path)
                self.event("load", path=str(Path(path).resolve()), loaded=result is not None,
                           kind_resolved=getattr(result, "draft_kind", None),
                           model_class=type(result.model).__module__ + "." + type(result.model).__qualname__ if result else None)
                return result
            return call

        def attach(original):
            def call(engine, drafter):
                result = original(engine, drafter)
                scheduler = engine._engine.engine.scheduler
                self.event("attach", engine_class=type(engine).__module__ + "." + type(engine).__qualname__,
                           scheduler_same_drafter=scheduler._vlm_mtp_drafter is drafter,
                           configured_block_size=scheduler._vlm_mtp_draft_block_size,
                           model_block_size=getattr(drafter.model.config, "block_size", None))
                return result
            return call

        def route(original):
            def call(scheduler, request, *args, **kwargs):
                self.current = request.request_id
                uid = original(scheduler, request, *args, **kwargs)
                self.event("route", request_id=request.request_id, uid=uid,
                           prompt_ids=list(request.prompt_token_ids),
                           registered=uid in scheduler._vlm_mtp_active if uid is not None else False)
                return uid
            return call

        def verify(original):
            def call(lm, verify_input, *args, **kwargs):
                result = original(lm, verify_input, *args, **kwargs)
                self.event("verify", request_id=self.current, input_shape=list(verify_input.shape),
                           sample_target_tokens=kwargs.get("sample_target_tokens", True))
                return result
            return call

        def round_record(original):
            def call(model, accepted, draft_count):
                before = counter_values(model)
                result = original(model, accepted, draft_count)
                # oMLX passes _MTPResetBindingProxy. Its list append delegates
                # through __getattr__, but counter assignment lives on the
                # proxy itself. Observe the actual argument, not its drafter.
                underlying = getattr(model, "_drafter", model)
                self.event("round", request_id=self.current, accepted=int(accepted), proposed=int(draft_count),
                           counter_owner_id=id(model), underlying_model_id=id(underlying),
                           counter_owner_class=type(model).__module__ + "." + type(model).__qualname__,
                           counters_before=before, counters_after=counter_values(model))
                return result
            return call

        def outputs(original):
            def call(scheduler, responses):
                prompts = {rid: list(req.prompt_token_ids or []) for rid, req in scheduler.running.items()}
                result = original(scheduler, responses)
                for output in result[0]:
                    if output.finished:
                        self.event("output", request_id=output.request_id,
                                   raw_ids=list(output.output_token_ids), prompt_ids=prompts.get(output.request_id),
                                   finish_reason=output.finish_reason, prompt_tokens=output.prompt_tokens,
                                   completion_tokens=output.completion_tokens, cached_tokens=output.cached_tokens,
                                   error=output.error)
                return result
            return call

        self.hook(vlm_mtp, "load_vlm_mtp_drafter", load)
        self.hook(VLMBatchedEngine, "set_vlm_mtp_drafter", attach)
        self.hook(Scheduler, "_route_to_vlm_mtp", route)
        self.hook(mtp, "_mtp_verify_target", verify)
        self.hook(mtp, "_record_speculative_round", round_record)
        self.hook(Scheduler, "_process_batch_responses", outputs)
        write(self.out / "observer-seams.json", self.seams)

    def close(self):
        for item in reversed(self.patches):
            item.stop()


async def qualify(args, workload, observer):
    from omlx.engine_pool import EnginePool
    from omlx.model_settings import ModelSettings
    from omlx.scheduler import SchedulerConfig
    from omlx.settings import burst_decode_env

    os.environ.update(burst_decode_env("balanced"))
    root = args.out / "models"
    root.mkdir()
    model_id = "Qwen3.8-27B-nvfp4"
    (root / model_id).symlink_to(Path(workload["model_dir"]).resolve())
    settings = ModelSettings(vlm_mtp_enabled=args.mode == "mtp",
                             vlm_mtp_draft_model=str(args.drafter.resolve()) if args.mode == "mtp" else None)
    scheduler_config = SchedulerConfig(max_num_seqs=1, completion_batch_size=1,
                                       chunked_prefill=False, prefill_speed_priority=True,
                                       decode_fairness=True, paged_ssd_cache_dir=None, hot_cache_max_size=0)
    scheduler_record = dataclasses.asdict(scheduler_config)
    scheduler_record["policy"] = scheduler_config.policy.value
    write(args.out / "settings.json", {"model": settings.to_dict(),
          "scheduler": scheduler_record, "burst": "balanced",
          "request": {"prompt_ids": workload["prompt_ids"], "max_tokens": 32,
                      "temperature": 0, "repetition_penalty": 1.0, "normal_eos": True},
          "scope": "official EnginePool API; no HTTP server/admission enforcer, no throughput claim"})
    pool = EnginePool(scheduler_config=scheduler_config)
    engine = None
    drafter = None
    try:
        pool.discover_models(str(root))
        discovered = {key: {"path": str(value.model_path), "engine_type": value.engine_type}
                      for key, value in pool._entries.items()}
        write(args.out / "discovery.json", discovered)
        assert set(discovered) == {model_id}, "unexpected discovered model or missing target"
        assert Path(discovered[model_id]["path"]).resolve() == Path(workload["model_dir"]).resolve()
        engine = await pool.get_engine(model_id, runtime_settings=settings)
        assert type(engine).__name__ == "VLMBatchedEngine", type(engine)
        write(args.out / "effective-engine.json", {
            "class": type(engine).__module__ + "." + type(engine).__qualname__,
            "settings": engine._model_settings.to_dict(),
            "stop_token_ids": sorted(engine._engine.engine.scheduler._get_stop_tokens()),
            "normal_eos": True})
        write(args.out / "loaded-sources.json", loaded_sources())
        drafter = engine.vlm_mtp_drafter
        assert (drafter is not None) == (args.mode == "mtp"), "drafter soft fallback"
        loads = [e for e in observer.events if e["kind"] == "load"]
        attaches = [e for e in observer.events if e["kind"] == "attach"]
        if args.mode == "mtp":
            assert len(loads) == len(attaches) == 1, (loads, attaches)
            assert loads[0]["loaded"] and loads[0]["kind_resolved"] == "mtp", loads
            assert loads[0]["path"] == str(args.drafter.resolve()), loads
            assert attaches[0]["scheduler_same_drafter"], attaches
            assert attaches[0]["configured_block_size"] is None and attaches[0]["model_block_size"] == 3, attaches
        else:
            assert not loads and not attaches
        requests = []
        for index in range(3):
            event_start = len(observer.events)
            underlying_counters_before = counter_values(drafter.model) if drafter else {}
            result = await asyncio.wait_for(engine.generate(
                prompt=list(workload["prompt_ids"]), max_tokens=32,
                temperature=0, repetition_penalty=1.0), timeout=180)
            events = observer.events[event_start:]
            outputs = [e for e in events if e["kind"] == "output"]
            assert len(outputs) == 1, outputs
            raw = outputs[0]
            assert raw["prompt_ids"] == workload["prompt_ids"], raw
            assert len(raw["raw_ids"]) == raw["completion_tokens"] == result.completion_tokens == 32, raw
            assert raw["prompt_tokens"] == result.prompt_tokens == 19, raw
            assert raw["finish_reason"] == result.finish_reason == "length", raw
            assert raw["cached_tokens"] == result.cached_tokens == 0 and raw["error"] is None, raw
            routes = [e for e in events if e["kind"] == "route"]
            verifies = [e for e in events if e["kind"] == "verify"]
            rounds = [e for e in events if e["kind"] == "round"]
            row = {"index": index, "raw": raw, "text": result.text, "routes": routes,
                   "verifies": verifies, "rounds": rounds,
                   "underlying_counters_before": underlying_counters_before,
                   "underlying_counters_after": counter_values(drafter.model) if drafter else {}}
            if args.mode == "mtp":
                assert len(routes) == 1 and routes[0]["uid"] < 0 and routes[0]["registered"], routes
                assert routes[0]["request_id"] == raw["request_id"]
                assert verifies and len(verifies) == len(rounds), (verifies, rounds)
                assert all(e["request_id"] == raw["request_id"] for e in verifies + rounds)
                draft_lens = list(map(int, drafter.model.draft_lens))
                accept_lens = list(map(int, drafter.model.accept_lens))
                assert draft_lens == [r["proposed"] for r in rounds]
                assert accept_lens == [r["accepted"] for r in rounds]
                assert [v["input_shape"] for v in verifies] == [[1, n + 1] for n in draft_lens]
                assert all(0 <= a <= d for a, d in zip(accept_lens, draft_lens))
                assert len({r["counter_owner_id"] for r in rounds}) == 1, "counter proxy changed within request"
                assert all(r["underlying_model_id"] == id(drafter.model) for r in rounds)
                assert all(r["counter_owner_class"] == "omlx.speculative.vlm_mtp._MTPResetBindingProxy" for r in rounds)
                for round_index, r in enumerate(rounds):
                    before, after = r["counters_before"], r["counters_after"]
                    assert {key: after[key] - before[key] for key in before} == {
                        "rounds": 1, "accepted": r["accepted"], "drafted": r["proposed"]}
                    if round_index:
                        assert before == rounds[round_index - 1]["counters_after"], "counter chain discontinuity"
                row["counters_before"] = rounds[0]["counters_before"]
                row["counters_after"] = rounds[-1]["counters_after"]
                row["counter_owner_id"] = rounds[0]["counter_owner_id"]
                row["underlying_model_id"] = id(drafter.model)
                delta = {key: row["counters_after"][key] - row["counters_before"][key]
                         for key in row["counters_before"]}
                assert delta == {"rounds": len(rounds), "accepted": sum(accept_lens), "drafted": sum(draft_lens)}, delta
                assert sum(draft_lens) > 0 and sum(accept_lens) > 0, "no useful speculation observed"
                assert 1 + sum(1 + n for n in accept_lens) >= 32, "inadequate round delivery accounting"
                row.update(draft_lens=draft_lens, accept_lens=accept_lens, counter_delta=delta)
            else:
                assert not routes and not verifies and not rounds
            requests.append(row)
            write(args.out / f"request-{index}.json", row)
        assert all(r["raw"]["raw_ids"] == requests[0]["raw"]["raw_ids"] for r in requests)
        assert len({r["raw"]["request_id"] for r in requests}) == 3, "request identity reused"
        return {"status": "PASS_ARM_ONLY_REQUIRES_FRESH_CONTROL_COMPARISON", "mode": args.mode,
                "requests": requests, "loads": loads, "attaches": attaches, "no_performance_claim": True}
    except BaseException as exc:
        write(args.out / "request-failure.json", {"type": type(exc).__name__, "error": str(exc),
                                                "traceback": traceback.format_exc()})
        raise
    finally:
        # The observer must not retain drafter/target weights during the official
        # unload-and-settle barrier. EnginePool owns the remaining references.
        drafter = None
        try:
            await pool.shutdown()
            retained = [key for key, entry in pool._entries.items() if entry.engine is not None or entry.is_loading]
            engine_cleared = engine is None or (not engine._loaded and engine._engine is None and
                                                engine._vlm_model is None and engine._vlm_mtp_drafter is None)
            write(args.out / "pool-shutdown.json", {"completed": not retained and engine_cleared,
                  "returned": True, "retained_entries": retained, "engine_references_cleared": engine_cleared,
                  "process_release_proved_separately_by_owner_reap": True})
            assert not retained and engine_cleared, "official shutdown retained engine state"
        except BaseException as exc:
            write(args.out / "cleanup-failure.json", {"type": type(exc).__name__, "error": str(exc),
                                                     "traceback": traceback.format_exc()})
            raise


def counter_values(model):
    return {name: int(getattr(model, "speculative_total_" + name, 0))
            for name in ("rounds", "accepted", "drafted")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("none", "mtp"), required=True)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--drafter", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    write(args.out / "attempt.json", {"pid": os.getpid(), "mode": args.mode,
          "workload": str(args.workload), "stage": "freeze_before_model_load", "no_performance_claim": True})
    observer = Observer(args.out)
    try:
        workload = json.loads(args.workload.read_text())
        assert len(workload["prompt_ids"]) == 19
        sources = freeze_distributions()
        plan = {"mode": args.mode, "pid": os.getpid(), "python": sys.executable,
                "platform": platform.platform(), "workload_sha256": digest(args.workload),
                "harness": {str(p): digest(p) for p in Path(__file__).parent.glob("*.py")},
                "distribution_identity": sources,
                "target": model_identity(workload["model_dir"]), "drafter": model_identity(args.drafter),
                "environment": {k: os.environ.get(k) for k in (
                    "MLX_MAX_OPS_PER_BUFFER", "MLX_MAX_MB_PER_BUFFER", "MLX_BFS_MAX_WIDTH", "MLX_METAL_FAST_SYNCH")},
                "scope": "3 serial requests, 19 -> 32, greedy normal EOS; qualified engine API only",
                "observer_effect": "synchronous CPU event writes perturb timings; no performance results"}
        write(args.out / "plan.json", plan)
        observer.install()
        result = asyncio.run(qualify(args, workload, observer))
        for path, expected in sources["files"].items():
            assert digest(path) == expected, "distribution changed during qualification: " + path
        write(args.out / "result.json", result)
        print(json.dumps({"mode": args.mode, "status": result["status"], "requests": 3}), flush=True)
    except BaseException as exc:
        write(args.out / "failure.json", {"type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc()})
        raise
    finally:
        observer.close()


if __name__ == "__main__":
    main()
