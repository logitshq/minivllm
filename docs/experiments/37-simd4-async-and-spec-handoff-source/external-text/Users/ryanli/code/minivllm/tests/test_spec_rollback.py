"""投机解码的回滚（0.8B fp32）：verify 一块 8 个 token、只保留前 m 个、继续 decode == 直接按这 m 个 token decode。"""
import glob, os
import numpy as np
import mlx.core as mx

from minivllm.cache import make_cache
from minivllm.forward_context import ForwardContext
from minivllm.model.loader import load_model

HF_DIR = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/*"))[0]
PROMPT = np.load("tests/data/hf_0.8b.npz")["prompt"].tolist()


def logits_np(x):
    return np.array(x.astype(mx.float32))


def test_verify_rollback_continue():
    model, _ = load_model(HF_DIR, dtype=mx.float32)
    block = [79852, 18887, 220, 98435, 5, 6, 7, 8]          # 锚点 + 7 个"草稿"（后面几个故意是错的）
    cont = [98618, 103806, 95779]
    for keep in [1, 4, 8]:                                   # 保留 锚点+0 / 锚点+3 / 全部
        # 参考：prefill prompt，然后逐 token 喂 block[:keep] + cont
        ref_caches = ForwardContext("prefill", make_cache(model))
        model(mx.array(PROMPT), ref_caches)
        ref = [logits_np(model(mx.array([t]), ref_caches))[0] for t in block[:keep] + cont]
        # 投机：prefill prompt，snapshot，verify 整块，rollback(keep)，再喂 cont
        caches = make_cache(model); ctx = ForwardContext("prefill", caches)
        model(mx.array(PROMPT), ctx)
        for c in caches:
            c.snapshot()
        ver = logits_np(model(mx.array(block), ctx))
        for c in caches:
            c.rollback(keep)
        assert caches[0].offset == len(PROMPT) + keep and caches[3].offset == len(PROMPT) + keep
        got = [ver[i] for i in range(keep)] + [logits_np(model(mx.array([t]), ctx))[0] for t in cont]
        for a, b in zip(got, ref):
            assert np.abs(a - b).max() < 1e-2, keep
