"""Measure Lithos target-only through its real Chat Completions API.

Owns one isolated loopback server. Stops only that child process. No model download.
"""
import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
import secrets
import socket
import statistics
import subprocess
import sys
import time

import httpx
from transformers import AutoTokenizer


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--model-dir", type=Path, default=Path(
        "/Users/ryanli/.cache/huggingface/hub/models--mlx-community--Qwen3.8-27B-nvfp4/snapshots/5ff8ef173ad0d7c3aae92f0be43031a6ab8067c6"))
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-context", type=int, default=2048)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--startup-timeout", type=float, default=900)
    parser.add_argument("--prepare-only", action="store_true", help="Validate metadata/prompt IDs and prepare symlink view on CPU; no server or GPU device")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if min(args.repeat, args.max_new_tokens, args.max_context) < 1:
        parser.error("repeat, output cap and context must be positive")
    root, model = args.workspace.resolve(), args.model_dir.resolve()
    # eos.py is a pure JSON module; importing it does not import MLX or execute a model.
    sys.path.insert(0, str(root))
    from tutorial_code.miniengine.eos import read_eos_ids
    runtime = root / ".artifacts/lithos-baseline"
    configuration = json.loads((model / "config.json").read_text())
    quant = configuration.get("quantization", configuration.get("quantization_config", {}))
    if quant != {"group_size": 16, "bits": 4, "mode": "nvfp4"}:
        raise RuntimeError("This baseline adapter is qualified for MLX NVFP4 group16/bits4")
    served_checkpoint = model
    generation_policy = "existing generation_config.json"
    eos_ids = sorted(read_eos_ids(str(model)))
    if not eos_ids:
        raise RuntimeError("Could not resolve stopping IDs from model/tokenizer metadata")
    if (model / "generation_config.json").exists():
        stored = json.loads((model / "generation_config.json").read_text())["eos_token_id"]
        stored = stored if isinstance(stored, list) else [stored]
        if set(stored) != set(eos_ids):
            raise RuntimeError("Existing Lithos generation EOS metadata differs from tutorial baseline")
    else:
        # Some locally converted models omit this file. Upstream text_config.eos can
        # otherwise hide the tokenizer's EOS. Preserve the baseline's EOS set
        # in a metadata-only view; weights/config/tokenizer bytes remain unchanged.
        served_checkpoint = runtime / "checkpoints" / f"{model.name}-{sha(model / 'config.json')[:12]}"
        served_checkpoint.mkdir(parents=True, exist_ok=True)
        for original in model.iterdir():
            link = served_checkpoint / original.name
            if not link.exists():
                link.symlink_to(original, target_is_directory=original.is_dir())
        (served_checkpoint / "generation_config.json").write_text(json.dumps({"eos_token_id": eos_ids}) + "\n")
        generation_policy = "metadata-only symlink view: add EOS set read from unchanged nested model config and tokenizer metadata; no tensor copies or edits"
    cli = runtime / ".venv/bin/lithos-metal"
    installation = json.loads((runtime / "installation.json").read_text())
    if version("lithos-metal") != installation["distribution_version"]:
        raise RuntimeError("Run this module under the pinned Lithos Python environment")
    from monolith.models.qwen3_5 import weights
    if sha(Path(weights.__file__)) != installation["checkpoint_adapter_sha256"]:
        raise RuntimeError("Installed checkpoint adapter differs from pinned upstream fix")
    trace = json.loads((root / "tutorial_code/examples/chapter01-trace.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True, trust_remote_code=False)
    messages = [{"role": "user", "content": trace["prompt"]}]
    ids = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False,
                                        add_generation_prompt=True, enable_thinking=False)
    if ids != trace["prompt_ids"]:
        raise RuntimeError("Lithos chat-template input IDs differ from the tutorial baseline")
    if len(ids) + args.max_new_tokens > args.max_context:
        parser.error("Context must hold this prompt plus the output budget")
    if args.prepare_only:
        from monolith.models import resolve_model
        from monolith.models.qwen3_5.weights import bind_checkpoint_formats
        from collections import Counter
        cls = resolve_model(configuration["architectures"][0])
        if cls is None:
            raise RuntimeError("No registered Lithos model package")
        cpu_model = cls.from_checkpoint(str(served_checkpoint), max_context=args.max_context)
        bound = bind_checkpoint_formats(cpu_model, str(served_checkpoint))
        prepared = {"baseline": "Lithos Metal", "status": "CPU prepared, GPU unmeasured",
                    "model_dir": str(model), "served_metadata_view": str(served_checkpoint),
                    "generation_config_policy": generation_policy, "eos_ids": eos_ids,
                    "prompt_ids": ids, "prompt_ids_match_baseline": True,
                    "bindings": len(bound), "bound_formats": dict(Counter(bound.values())),
                    "native_device_created": False, "server_started": False}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(prepared, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(prepared, ensure_ascii=False, indent=2))
        return
    run_dir = runtime / "runs" / secrets.token_hex(6)
    run_dir.mkdir(parents=True)
    log_path = run_dir / "server.log"
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    key = secrets.token_urlsafe(24)
    env = {**os.environ, "LITHOS_METAL_API_KEY": key, "HF_HUB_OFFLINE": "1",
           "TOKENIZERS_PARALLELISM": "false", "PYTHONDONTWRITEBYTECODE": "1"}
    command = [str(cli), "serve", "--model", str(served_checkpoint), "--no-draft", "--pack", str(runtime / "packs"),
               "--local-files-only", "--host", "127.0.0.1", "--port", str(port),
               "--max-context", str(args.max_context), "--keep-warm", "0", "--no-warmup"]
    measurements = []
    resident_buffers = None
    with log_path.open("w") as log:
        startup_started = time.perf_counter()
        proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
        try:
            with httpx.Client(headers={"Authorization": f"Bearer {key}"}, timeout=900, trust_env=False) as client:
                base = f"http://127.0.0.1:{port}"
                deadline = time.monotonic() + args.startup_timeout
                while True:
                    if proc.poll() is not None:
                        raise RuntimeError(f"Lithos exited at startup; inspect {log_path}")
                    try:
                        response = client.get(base + "/v1/models", timeout=1)
                        if response.status_code == 200:
                            model_name = response.json()["data"][0]["id"]
                            startup_elapsed_s = time.perf_counter() - startup_started
                            break
                    except httpx.HTTPError:
                        pass
                    if time.monotonic() > deadline:
                        raise TimeoutError(f"Lithos readiness timeout; inspect {log_path}")
                    time.sleep(.2)
                body = {"model": model_name, "messages": messages, "temperature": 0,
                        "top_p": 1, "max_completion_tokens": args.max_new_tokens}
                # --no-warmup omits upstream's extra sampled/long-context startup workloads.
                # This explicit full-workload warmup includes greedy compilation and weight loading.
                warm_started = time.perf_counter()
                warm = client.post(base + "/v1/chat/completions", json=body)
                warm.raise_for_status()
                warmup = {"client_elapsed_s": time.perf_counter() - warm_started, "usage": warm.json()["usage"],
                          "finish_reason": warm.json()["choices"][0]["finish_reason"]}
                assert warmup["usage"]["prompt_tokens"] == len(ids)
                assert warmup["finish_reason"] in {"stop", "length"}
                assert 0 < warmup["usage"]["completion_tokens"] <= args.max_new_tokens
                for _ in range(args.repeat):
                    started, first_text, usage, finish, done = time.perf_counter(), None, None, None, False
                    pieces = []
                    with client.stream("POST", base + "/v1/chat/completions", json={**body, "stream": True,
                                       "stream_options": {"include_usage": True}}) as response:
                        response.raise_for_status()
                        for line in response.iter_lines():
                            if not line.startswith("data:"):
                                continue
                            payload = line[5:].strip()
                            if payload == "[DONE]":
                                done = True
                                break
                            event = json.loads(payload)
                            if "error" in event:
                                raise RuntimeError(f"Lithos stream error: {event['error']}")
                            if event.get("usage") is not None:
                                usage = event["usage"]
                            for choice in event.get("choices", []):
                                if choice.get("finish_reason") is not None:
                                    finish = choice["finish_reason"]
                                text = choice.get("delta", {}).get("content", "")
                                if text:
                                    first_text = first_text or time.perf_counter()
                                    pieces.append(text)
                    elapsed = time.perf_counter() - started
                    assert done and usage and finish in {"stop", "length"}
                    assert usage["prompt_tokens"] == len(ids)
                    count = usage["completion_tokens"]
                    assert 0 < count <= args.max_new_tokens
                    measurements.append({"client_elapsed_s": elapsed,
                        "client_ttft_text_s": None if first_text is None else first_text - started,
                        "client_completion_tokens_per_s": count / elapsed,
                        "usage": usage, "logical_completion_tokens": count - int(finish == "stop"),
                        "finish_reason": finish, "text": "".join(pieces), "done": done})
                memory = client.get(base + "/api/ps")
                memory.raise_for_status()
                resident_buffers = memory.json()["models"][0]["size_vram"]
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=15)
    # Backend's exact cache and GPU timing counters are logged for warmup and every request.
    import ast
    metrics = []
    for line in log_path.read_text().splitlines():
        if "Generation: " in line:
            metrics.append(ast.literal_eval(line.split("Generation: ", 1)[1]))
    if len(metrics) != args.repeat + 1:
        raise RuntimeError(f"Expected warmup plus {args.repeat} backend metric records, got {len(metrics)}")
    for entry, counters in zip([warmup, *measurements], metrics):
        assert counters["cached_prompt_tokens"] == 0
        assert counters["verify_tokens"] == 1  # target-only, no speculative proposals
        entry["backend_metrics"] = counters
    from monolith.runtime import _native
    info = _native.Device().info()
    from monolith.backends.metal import config_for_device
    profile = config_for_device(info.gpu_cores, info.apple_family, info.name)
    profile_text = re.search(r"Serving backend=(\S+)", log_path.read_text())
    report = {"baseline": "Lithos Metal", "status": "measured local target-only",
        "recorded_at": datetime.now(timezone.utc).isoformat(), "installation": installation,
        "versions": {name: version(name) for name in ("lithos-metal", "numpy", "transformers", "tokenizers")},
        "host": {"chip": info.name, "gpu_cores": info.gpu_cores, "apple_family": info.apple_family,
                 "unified_memory_bytes": int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)),
                 "macos": subprocess.check_output(["sw_vers", "-productVersion"], text=True).strip()},
        "backend": None if profile_text is None else profile_text.group(1).rstrip(","),
        "backend_profile": profile.raw,
        "checkpoint": {"path": str(model), "revision": model.name if model.parent.name == "snapshots" else None,
                       "format": "MLX NVFP4 group16 bits4",
                       "served_metadata_view": str(served_checkpoint), "generation_config_policy": generation_policy,
                       "config_sha256": sha(model / "config.json"), "index_sha256": sha(model / "model.safetensors.index.json"),
                       "source_weight_bytes": sum(p.stat().st_size for p in model.glob("*.safetensors"))},
        "prompt": trace["prompt"], "prompt_ids": ids, "prompt_ids_match_baseline": True,
        "sampling": {"temperature": 0, "top_p": 1, "eos": "normal stopping",
                     "eos_ids": eos_ids}, "draft": None,
        "concurrency": 1, "max_new_tokens_cap": args.max_new_tokens, "max_context": args.max_context,
        "cache_policy": "serving prefix cache enabled; exact short 19-token prompt had zero cached prompt tokens on every run",
        "timing_scope": "client POST through SSE DONE, including HTTP/service/prefill/decode; excludes installation, packing, model loading and explicit full-workload warmup",
        "backend_timing_scope": "prefill_gpu_ms/decode_gpu_ms are sums of Metal command-buffer GPU end minus start; include dispatch/queue contents, not isolated arithmetic kernels; steps can include queued done/no-op submissions and are not output-token counts",
        "counting": "Lithos API completion_tokens counts committed IDs including terminal EOS (decoded text hides special tokens)",
        "kernel_route": "registered M4 fallback; no automatic mixer megakernels or tensor accelerator; profile availability is not per-kernel dispatch evidence",
        "normalization_policy": "official Session default commute_norm=True; BF16 normalization/projection fusion can change rounding versus unfused paths",
        "memory": {"server_resident_buffer_bytes": resident_buffers,
                   "scope": "Lithos /api/ps size_vram: unique session buffer allocations; excludes CPU packs and process RSS"},
        "warmup": warmup, "runs": measurements,
        "startup": {"process_to_api_ready_s": startup_elapsed_s,
                    "scope": "server process start through models API readiness; includes checkpoint packing when absent and imports/tokenizer setup; --no-warmup defers GPU session loading/compilation to explicit warmup",
                    "pack_cache": str(runtime / "packs")},
        "median_client_completion_tokens_per_s": statistics.median(x["client_completion_tokens_per_s"] for x in measurements),
        "median_client_elapsed_s": statistics.median(x["client_elapsed_s"] for x in measurements),
        "median_client_ttft_text_s": statistics.median(x["client_ttft_text_s"] for x in measurements if x["client_ttft_text_s"] is not None),
        "server_log": str(log_path), "owned_server_exit_code": proc.returncode}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "median_client_completion_tokens_per_s": report["median_client_completion_tokens_per_s"],
                      "actual_output_counts": [x["usage"]["completion_tokens"] for x in measurements]}, indent=2))


if __name__ == "__main__":
    main()
