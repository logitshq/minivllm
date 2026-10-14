"""Serving benchmark client: a vllm-bench-serve lookalike for OpenAI-compatible servers.

Mirrors `vllm bench serve --dataset-name random --random-input-len N --random-output-len M
--num-prompts P --max-concurrency C`: P prompts of random tokens, all queued at t=0, at most C in
flight, streaming /v1/completions, temperature 0. Reports request / output-token throughput,
TTFT, TPOT, ITL (mean / median / p99) and writes a JSON next to the run.

EOS suppression: vLLM honours `ignore_eos`; other servers get `logit_bias` of -100 on every EOS id.
"""
import argparse, asyncio, json, random, statistics, time
from pathlib import Path

import httpx
from transformers import AutoTokenizer


def pct(xs, p):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))]


def build_prompts(tok, n, input_len, seed):
    rng = random.Random(seed)
    special = set(tok.all_special_ids)
    vocab = tok.vocab_size
    prompts = []
    for _ in range(n):
        ids = [i for i in (rng.randrange(vocab) for _ in range(input_len * 2)) if i not in special][:input_len]
        text = tok.decode(ids, skip_special_tokens=True)
        # random ids rarely re-tokenise to the same length; trim until the re-encoded length fits
        for _ in range(8):
            re_ids = tok.encode(text, add_special_tokens=False)
            if len(re_ids) <= input_len:
                break
            ids = ids[: max(1, len(ids) - (len(re_ids) - input_len))]
            text = tok.decode(ids, skip_special_tokens=True)
        prompts.append((text, len(re_ids)))
    return prompts


async def one_request(client, sem, url, body, result):
    async with sem:
        t0 = time.perf_counter()
        ttft = None
        chunk_times = []
        usage = None
        text_parts = 0
        try:
            async with client.stream("POST", url, json=body, timeout=None) as r:
                if r.status_code != 200:
                    result["error"] = f"HTTP {r.status_code}: {(await r.aread())[:200]!r}"
                    return
                async for line in r.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    obj = json.loads(payload)
                    if obj.get("usage"):
                        usage = obj["usage"]
                    choices = obj.get("choices") or []
                    if choices and choices[0].get("text"):
                        now = time.perf_counter()
                        if ttft is None:
                            ttft = now - t0
                        chunk_times.append(now)
                        text_parts += 1
        except Exception as e:  # noqa: BLE001
            result["error"] = repr(e)
            return
        t1 = time.perf_counter()
        out_tokens = (usage or {}).get("completion_tokens") or text_parts
        result.update(
            latency=t1 - t0,
            ttft=ttft,
            output_tokens=out_tokens,
            prompt_tokens=(usage or {}).get("prompt_tokens"),
            itl=[b - a for a, b in zip(chunk_times, chunk_times[1:])],
            tpot=((t1 - t0 - ttft) / (out_tokens - 1)) if ttft is not None and out_tokens > 1 else None,
        )


async def main(a):
    tok = AutoTokenizer.from_pretrained(a.tokenizer or a.model)
    prompts = build_prompts(tok, a.num_prompts, a.input_len, a.seed)
    eos_ids = sorted({i for i in [tok.eos_token_id, *(tok.convert_tokens_to_ids(t) for t in ["<|im_end|>", "<|endoftext|>"] if t in tok.get_vocab())] if i is not None})
    print(f"{a.num_prompts} prompts, re-tokenised lengths mean {statistics.mean(p[1] for p in prompts):.0f}; eos ids {eos_ids}; eos mode {a.eos_mode}")

    url = a.base_url.rstrip("/") + "/v1/completions"
    sem = asyncio.Semaphore(a.max_concurrency)
    results = [dict(idx=i) for i in range(a.num_prompts)]
    bodies = []
    for text, _ in prompts:
        body = dict(model=a.model, prompt=text, max_tokens=a.output_len, temperature=0.0, stream=True,
                    stream_options={"include_usage": True})
        if a.eos_mode == "vllm":
            body["ignore_eos"] = True
        elif a.eos_mode == "logit_bias":
            body["logit_bias"] = {str(i): -100.0 for i in eos_ids}
        bodies.append(body)

    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=a.max_concurrency + 4)) as client:
        # warm the server once so model load / compile is not in the measurement
        await one_request(client, sem, url, {**bodies[0], "max_tokens": 8}, {})
        t0 = time.perf_counter()
        await asyncio.gather(*(one_request(client, sem, url, b, r) for b, r in zip(bodies, results)))
        dur = time.perf_counter() - t0

    ok = [r for r in results if "error" not in r]
    errs = [r for r in results if "error" in r]
    out_tok = sum(r["output_tokens"] for r in ok)
    in_tok = sum(r["prompt_tokens"] or 0 for r in ok)
    ttfts = [r["ttft"] for r in ok if r["ttft"] is not None]
    tpots = [r["tpot"] for r in ok if r["tpot"] is not None]
    itls = [x for r in ok for x in r["itl"]]
    summary = dict(
        name=a.name, base_url=a.base_url, model=a.model, num_prompts=a.num_prompts, input_len=a.input_len,
        output_len=a.output_len, max_concurrency=a.max_concurrency, eos_mode=a.eos_mode,
        duration_s=dur, completed=len(ok), failed=len(errs),
        request_throughput=len(ok) / dur, output_token_throughput=out_tok / dur,
        total_token_throughput=(out_tok + in_tok) / dur,
        mean_output_tokens=out_tok / max(1, len(ok)), mean_prompt_tokens=in_tok / max(1, len(ok)),
        ttft_ms=dict(mean=1e3 * statistics.mean(ttfts), median=1e3 * statistics.median(ttfts), p99=1e3 * pct(ttfts, 99)) if ttfts else None,
        tpot_ms=dict(mean=1e3 * statistics.mean(tpots), median=1e3 * statistics.median(tpots), p99=1e3 * pct(tpots, 99)) if tpots else None,
        itl_ms=dict(mean=1e3 * statistics.mean(itls), median=1e3 * statistics.median(itls), p99=1e3 * pct(itls, 99)) if itls else None,
        errors=[r["error"] for r in errs][:5],
    )
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    w = 36
    print("=" * 60)
    print(f"{'Successful requests:':<{w}}{len(ok)}  (failed {len(errs)})")
    print(f"{'Benchmark duration (s):':<{w}}{dur:.2f}")
    print(f"{'Total input tokens:':<{w}}{in_tok}")
    print(f"{'Total generated tokens:':<{w}}{out_tok}")
    print(f"{'Request throughput (req/s):':<{w}}{len(ok)/dur:.2f}")
    print(f"{'Output token throughput (tok/s):':<{w}}{out_tok/dur:.2f}")
    print(f"{'Total token throughput (tok/s):':<{w}}{(out_tok+in_tok)/dur:.2f}")
    for label, d in [("TTFT", summary["ttft_ms"]), ("TPOT", summary["tpot_ms"]), ("ITL", summary["itl_ms"])]:
        if d:
            print(f"{'Mean ' + label + ' (ms):':<{w}}{d['mean']:.2f}")
            print(f"{'Median ' + label + ' (ms):':<{w}}{d['median']:.2f}")
            print(f"{'P99 ' + label + ' (ms):':<{w}}{d['p99']:.2f}")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(dict(summary=summary, requests=results), ensure_ascii=False))
        print("saved", a.out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://localhost:8080")
    p.add_argument("--model", required=True, help="model name as the server knows it")
    p.add_argument("--tokenizer", default=None, help="HF tokenizer path (default: --model)")
    p.add_argument("--num-prompts", type=int, default=100)
    p.add_argument("--input-len", type=int, default=1024)
    p.add_argument("--output-len", type=int, default=1024)
    p.add_argument("--max-concurrency", type=int, default=32)
    p.add_argument("--eos-mode", choices=["vllm", "logit_bias", "none"], default="logit_bias")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--name", default="run")
    p.add_argument("--out", default=None)
    asyncio.run(main(p.parse_args()))
