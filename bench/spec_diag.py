"""诊断 DFlash2：每轮 draft 的逐位置命中率、选路器 vs 原始 argmax、各阶段耗时。bf16 和 4bit drafter 各跑一次。
用法：uv run python -u bench/spec_diag.py <target_dir> <drafter_dir>"""
import sys, time, mlx.core as mx
from minivllm.cache import make_cache
from minivllm.forward_context import ForwardContext
from minivllm.model.loader import load_model
from minivllm.spec.dflash2 import load_drafter
from minivllm.tokenizer import ChatTokenizer

d, dd = sys.argv[1], sys.argv[2]
tok = ChatTokenizer(d)
model, cfg = load_model(d)
ids = tok.apply_chat_template([{"role": "user", "content": "The function $f$ satisfies $f(x)+f(y)=f(x+y)-xy-1$ for all real $x,y$. If $f(1)=1$, find all integers $n$ with $f(n)=n$. Reason step by step."}])
for name, quant in [("4bit", {"group_size": 64, "bits": 4, "mode": "affine"}), ("bf16", None)]:
    drafter = load_drafter(dd, quant)
    cap = tuple(drafter.cfg.target_layer_ids)
    caches = make_cache(model); ctx = ForwardContext("prefill", caches)
    logits, feats = model(mx.array(ids), ctx, last_only=True, capture=cap)
    anchor = int(mx.argmax(logits[-1]).item()); start = len(ids); hist = [feats]
    pos_hits = [0] * 7; raw_hits = [0] * 7; rounds = 0; acc_total = 0; t_draft = t_verify = t_roll = 0.0; produced_all = [anchor]
    for r in range(24):
        fed = drafter.layers[0].self_attn.ctx_len
        full = mx.concatenate(hist) if len(hist) > 1 else hist[0]
        feed = full[fed:start] if fed < start else None
        mx.synchronize(); t = time.perf_counter()
        drafts = drafter.draft(feed, anchor, start, model.embed_tokens, model.lm_head_fn)
        mx.synchronize(); t_draft += time.perf_counter() - t; t = time.perf_counter()
        # 原始 argmax（不走选路器）
        h_ctx = drafter.hidden_norm(drafter.fc(feed)) if feed is not None else None
        block = [anchor] + drafts
        for c in caches: c.snapshot()
        lg, nf = model(mx.array(block), ctx, capture=cap); post = mx.argmax(lg, axis=-1); mx.eval(post, nf); post = post.tolist()
        mx.synchronize(); t_verify += time.perf_counter() - t; t = time.perf_counter()
        n = 0
        while n < 7 and drafts[n] == post[n]: n += 1
        for i in range(7): pos_hits[i] += int(drafts[i] == post[i])
        for c in caches: c.rollback(n + 1)
        mx.synchronize(); t_roll += time.perf_counter() - t
        hist.append(nf[: n + 1]); produced_all += drafts[:n] + [post[n]]
        anchor = post[n]; start += n + 1; rounds += 1; acc_total += n
    print(f"[{name}] {rounds} rounds: {acc_total/rounds:.2f} drafts accepted/round ({(acc_total+rounds)/rounds:.2f} tok/round); per-position hit rate {[round(h/rounds, 2) for h in pos_hits]}")
    print(f"[{name}] time/round: draft {t_draft/rounds*1e3:.0f} ms, verify {t_verify/rounds*1e3:.0f} ms, rollback {t_roll/rounds*1e3:.0f} ms → {(acc_total+rounds)/(t_draft+t_verify+t_roll):.1f} tok/s")
    print("   ", repr(tok.decode(produced_all))[:120])
