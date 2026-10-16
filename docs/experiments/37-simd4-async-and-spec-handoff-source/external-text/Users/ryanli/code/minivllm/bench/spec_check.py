"""DFlash2 投机解码的端到端检查：和不投机的 greedy 逐 token 比，报接受率和 tok/s。
用法：uv run python -u bench/spec_check.py <target_dir> <drafter_dir> [N]"""
import sys, time, mlx.core as mx
from minivllm.engine.llm_engine import LLMEngine
from minivllm.sampling_params import SamplingParams
from minivllm.tokenizer import ChatTokenizer

d, dd = sys.argv[1], sys.argv[2]; N = int(sys.argv[3]) if len(sys.argv) > 3 else 128
tok = ChatTokenizer(d)
prompts = {"kv": "用一句话解释什么是 KV cache。",
           "math": "The function $f$ satisfies $f(x)+f(y)=f(x+y)-xy-1$ for all real $x,y$. If $f(1)=1$, find all integers $n$ with $f(n)=n$. Reason step by step.",
           "code": "Write a Python function that parses an ISO-8601 date string and returns a datetime, with tests."}
def run(eng, ids, sp):
    """返回 (tokens, decode-only 秒数)：从第一个 token 之后开始计时。"""
    rid = eng.add_request(ids, sp); out = []; t0 = None
    while eng.has_work():
        res = eng.step()
        if res and t0 is None:
            t0 = time.perf_counter(); out += [tk for _, tk, _ in res]; continue
        out += [tk for _, tk, _ in res]
    return out, time.perf_counter() - t0, rid

SP = {"kv": SamplingParams(max_tokens=N), "math": SamplingParams(max_tokens=N, ignore_eos=True), "code": SamplingParams(max_tokens=N, ignore_eos=True)}
plain = LLMEngine(d, max_seqs=2, max_len=4096, max_batched_tokens=512)
base, t_plain = {}, {}
for k, p in prompts.items():
    ids = tok.apply_chat_template([{"role": "user", "content": p}])
    out, dt, _ = run(plain, ids, SP[k])
    base[k] = out; t_plain[k] = dt; print(f"plain  {k:5s}: {(len(out)-1)/dt:5.1f} tok/s decode-only, {len(out)} tokens", flush=True)
del plain
for blk in [None, 8, 5]:
  spec = LLMEngine(d, max_seqs=2, max_len=4096, max_batched_tokens=512, drafter_dir=dd, spec_block=blk)
  print(f"--- block {blk or 'adaptive'}")
  for k, p in prompts.items():
    ids = tok.apply_chat_template([{"role": "user", "content": p}])
    out, dt, rid = run(spec, ids, SP[k])
    acc, rounds = spec.runner.spec.state[rid].stats if rid in spec.runner.spec.state else spec.runner.spec.finished_stats.get(rid, (0, 0))
    match = next((i for i, (a, b) in enumerate(zip(out, base[k])) if a != b), min(len(out), len(base[k])))
    print(f"dflash {k:5s}: {(len(out)-1)/dt:5.1f} tok/s decode-only ({(len(out)-1)/dt/((len(base[k])-1)/t_plain[k]):.2f}x); {len(out)} tokens in {rounds} rounds = {(len(out)-1)/max(rounds,1):.2f} tok/round, {acc/max(rounds,1):.2f} drafts accepted/round; greedy matches plain for {match}/{len(out)}", flush=True)
    if blk is None: print("   ", repr(tok.decode(out))[:110], flush=True)
  del spec
