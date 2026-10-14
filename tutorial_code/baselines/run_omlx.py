"""Run one warmed oMLX baseline request and record its exact configuration.

This is a baseline smoke/measurement adapter, not an inference-engine component.
It owns an isolated foreground server and terminates only that child process.
"""
import argparse
import hashlib
import json
import secrets
import socket
import subprocess
import time
from pathlib import Path

import httpx
from tokenizers import Tokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path,
                        default=Path("/Users/ryanli/code/minivllm/models/Qwen3.5-0.8B-nvfp4"))
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be positive")
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")
    model_dir = args.model_dir.resolve()
    if not (model_dir / "config.json").is_file():
        parser.error("--model-dir must point to a downloaded MLX model")
    root = Path(__file__).resolve().parents[2]
    runtime = root / ".artifacts/omlx-baseline"
    python = runtime / ".venv/bin/python"
    cli = runtime / ".venv/bin/omlx"
    if not cli.exists():
        parser.error("Run tutorial_code/baselines/setup_omlx.py first")

    probe = subprocess.check_output([str(python), "-c",
        "import json; from importlib.metadata import version; "
        "from omlx.custom_kernels import native_kernel_status; "
        "print(json.dumps({'versions':{n:version(n) for n in ['omlx','mlx','mlx-lm','transformers']},"
        "'native_kernels':native_kernel_status()}))"], text=True)
    metadata = json.loads(probe)
    for name in ["qwen35_prefill", "decode_fast"]:
        if not metadata["native_kernels"][name]["available"]:
            raise RuntimeError(f"Missing native kernel: {name}")

    trace = json.loads((root / "tutorial_code/examples/chapter01-trace.json").read_text())
    tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    # Send the literal ChatML text through /completions to avoid adding a second template.
    prompt = tok.decode(trace["prompt_ids"], skip_special_tokens=False)
    assert tok.encode(prompt, add_special_tokens=False).ids == trace["prompt_ids"]
    run_dir = runtime / "runs" / secrets.token_hex(6)
    models = run_dir / "models"
    models.mkdir(parents=True)
    model_name = model_dir.name
    (models / model_name).symlink_to(model_dir, target_is_directory=True)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    key = secrets.token_urlsafe(24)
    command = [str(cli), "serve", "--model-dir", str(models),
               "--base-path", str(run_dir / "state"), "--host", "127.0.0.1",
               "--port", str(port), "--api-key", key, "--headless", "--no-hf-cache",
               "--no-cache", "--max-concurrent-requests", "1", "--sse-keepalive-mode", "off"]
    log_path = run_dir / "server.log"
    with log_path.open("w") as log:
        proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            base = f"http://127.0.0.1:{port}"
            with httpx.Client(headers={"Authorization": f"Bearer {key}"},
                              timeout=180, trust_env=False) as client:
                deadline = time.monotonic() + 60
                while True:
                    if proc.poll() is not None:
                        raise RuntimeError("oMLX exited during startup")
                    try:
                        r = client.get(base + "/v1/models", timeout=1)
                        if r.status_code == 200:
                            break
                    except httpx.HTTPError:
                        pass
                    if time.monotonic() > deadline:
                        raise TimeoutError("oMLX readiness timeout")
                    time.sleep(.2)
                assert model_name in [m["id"] for m in r.json()["data"]]
                body = {"model": model_name, "prompt": prompt, "temperature": 0,
                        "max_tokens": args.max_new_tokens, "repetition_penalty": 1.0}
                # Load weights and compile paths before the recorded request.
                warm = client.post(base + "/v1/completions", json=body)
                warm.raise_for_status()
                def measured_request():
                    started = time.perf_counter()
                    first_text_time = None
                    text = []
                    usage = None
                    finish_reason = None
                    done = False
                    with client.stream("POST", base + "/v1/completions",
                                       json={**body, "stream": True,
                                             "stream_options": {"include_usage": True}}) as response:
                        response.raise_for_status()
                        for line in response.iter_lines():
                            if not line.startswith("data:"):
                                continue
                            payload = line[5:].strip()
                            if payload == "[DONE]":
                                done = True
                                break
                            chunk = json.loads(payload)
                            if chunk.get("usage") is not None:
                                usage = chunk["usage"]
                            for choice in chunk.get("choices", []):
                                if choice.get("finish_reason") is not None:
                                    finish_reason = choice["finish_reason"]
                                piece = choice.get("text", "")
                                if piece:
                                    first_text_time = first_text_time or time.perf_counter()
                                    text.append(piece)
                    elapsed = time.perf_counter() - started
                    assert done and usage is not None, "Stream missing DONE or token usage"
                    assert usage["prompt_tokens"] == len(trace["prompt_ids"]), "Prompt token counts differ"
                    assert usage["completion_tokens"] > 0
                    assert usage["prompt_tokens_details"]["cached_tokens"] == 0, "Unexpected cross-request prefix-cache hit"
                    assert finish_reason in {"stop", "length"}, "Stream missing normal completion finish reason"
                    cfg_raw = (model_dir / "config.json").read_bytes()
                    report = {
                        "baseline": "omlx", "purpose": "optimized-engine reference smoke" if args.max_new_tokens <= 8 else "optimized-engine warmed client measurement",
                        **metadata, "model_dir": str(model_dir),
                        "config_sha256": hashlib.sha256(cfg_raw).hexdigest(),
                        "quantization": json.loads(cfg_raw).get("quantization"),
                        "request": {"prompt_ids": trace["prompt_ids"], "temperature": 0,
                                    "max_tokens": args.max_new_tokens, "repetition_penalty": 1.0},
                        "settings": {"concurrency": 1, "paged_prefix_cache": False,
                                     "speculative_decoding": False, "warmup_output_tokens": args.max_new_tokens,
                                     "warmup_actual_output_tokens": warm.json().get("usage", {}).get("completion_tokens"),
                                     "transport": "HTTP SSE; client timing includes server overhead"},
                        "usage": usage, "text": "".join(text),
                        "finish_reason": finish_reason,
                        "output_count_definition": "oMLX completion_tokens excludes a terminal EOS; text may omit other special IDs. Rates use server-reported completion_tokens.",
                        "client_latency_s": elapsed,
                        "client_ttft_text_s": first_text_time - started if first_text_time else None,
                        "output_tokens_per_client_second": usage["completion_tokens"] / elapsed,
                        "scope": "functional smoke" if args.max_new_tokens <= 8 else "warmed HTTP client measurement; actual completion length recorded; EOS stopping enabled",
                    }
                    return report
                runs = [measured_request() for _ in range(args.repeat)]
                if args.repeat == 1:
                    report = runs[0]
                else:
                    import statistics
                    report = {"baseline": "omlx", "purpose": "repeated warmed HTTP client measurements",
                              "repeat": args.repeat, "runs": runs,
                              "median_client_latency_s": statistics.median(r["client_latency_s"] for r in runs),
                              "median_client_ttft_text_s": statistics.median(v for r in runs if (v := r["client_ttft_text_s"]) is not None) if any(r["client_ttft_text_s"] is not None for r in runs) else None,
                              "scope": "HTTP SSE client timing includes server overhead; independent of pure engine timing"}
                print(json.dumps(report, ensure_ascii=False, indent=2))
                if args.out:
                    args.out.parent.mkdir(parents=True, exist_ok=True)
                    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        except Exception:
            print(log_path.read_text()[-5000:].replace(key, "[redacted]"))
            raise
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()


if __name__ == "__main__":
    main()
