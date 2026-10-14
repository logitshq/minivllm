"""直接驱动 LLMEngine 的吞吐测试（不经过 server）：B 个请求同时提交，各 1024 进 / N 出，ignore_eos。
报：总输出 tok/s、decode 阶段 tok/s（所有 prompt 算完之后）、平均 TTFT、峰值内存。
用法：uv run bench/throughput.py <model_dir> --batch 1 8 16 32 --output-len 128
"""
import argparse, glob, os, random, time
import mlx.core as mx

from minivllm.engine.llm_engine import LLMEngine
from minivllm.sampling_params import SamplingParams


def run(eng, B, input_len, output_len, seed=0):
    rng = random.Random(seed)
    prompts = [[rng.randrange(1000, 200000) for _ in range(input_len)] for _ in range(B)]
    sp = SamplingParams(max_tokens=output_len, ignore_eos=True)
    t0 = time.perf_counter()
    ids = [eng.add_request(p, sp) for p in prompts]
    first = {}; n_out = 0; decode_t0 = None; decode_tokens = 0
    while eng.has_work():
        res = eng.step()
        now = time.perf_counter()
        for rid, tok, fin in res:
            n_out += 1
            first.setdefault(rid, now - t0)
        if decode_t0 is None and len(first) == B:
            decode_t0 = now                              # 所有 prompt 都算完了，从这里开始算纯 decode
        elif decode_t0 is not None:
            decode_tokens += len(res)
    total = time.perf_counter() - t0
    return dict(B=B, total_s=total, out_tps=n_out / total, decode_tps=decode_tokens / (time.perf_counter() - decode_t0) if decode_t0 else float("nan"),
                ttft_mean=sum(first.values()) / B, peak_gb=mx.get_peak_memory() / 2**30)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("--batch", type=int, nargs="+", default=[1, 8, 16, 32])
    ap.add_argument("--input-len", type=int, default=1024); ap.add_argument("--output-len", type=int, default=128)
    ap.add_argument("--max-batched-tokens", type=int, default=512)
    a = ap.parse_args()
    eng = LLMEngine(a.model, max_seqs=max(a.batch), max_len=a.input_len + a.output_len, max_batched_tokens=a.max_batched_tokens)
    print(f"{'B':>3} {'total s':>8} {'out tok/s':>10} {'decode tok/s':>13} {'TTFT mean s':>12} {'peak GB':>8}")
    for B in a.batch:
        mx.reset_peak_memory()
        r = run(eng, B, a.input_len, a.output_len)
        print(f"{r['B']:>3} {r['total_s']:>8.1f} {r['out_tps']:>10.1f} {r['decode_tps']:>13.1f} {r['ttft_mean']:>12.2f} {r['peak_gb']:>8.1f}", flush=True)
