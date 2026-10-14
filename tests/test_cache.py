"""第 2 步：cache 的正确性。判据只有一个：带 cache 的逐 token 计算 == 整句一次前向。"""
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


def test_prefill_then_decode_equals_full_forward():
    """fp32：prefill 19 个 token，再逐个 decode 30 个；每一步的 logits 都要和整句前向对应位置一致，greedy 相同。"""
    model, _ = load_model(HF_DIR, dtype=mx.float32)
    ref = np.load("tests/data/hf_0.8b.npz")
    full_ids = PROMPT + ref["fp32_greedy"].tolist()[:30]
    full = logits_np(model(mx.array(full_ids)))                        # [T, V] 整句
    cache = ForwardContext("prefill", make_cache(model))
    out = logits_np(model(mx.array(PROMPT), cache))                    # prefill
    assert np.abs(out - full[: len(PROMPT)]).max() < 1e-2
    for t in range(len(PROMPT), len(full_ids)):
        step = logits_np(model(mx.array([full_ids[t]]), cache))        # decode：只算 1 个 token
        assert np.abs(step[0] - full[t]).max() < 1e-2, f"position {t}"
        assert step[0].argmax() == full[t].argmax()
    assert cache.caches[0].offset == len(full_ids) and cache.caches[3].offset == len(full_ids)


def test_chunked_prefill_equals_one_shot():
    """分块 prefill（7 个一块，最后一块 5 个）== 一次性 prefill：验证注意力的右下角 causal 对齐和 GDN 的状态接续。"""
    model, _ = load_model(HF_DIR, dtype=mx.float32)
    one = logits_np(model(mx.array(PROMPT)))
    cache = ForwardContext("prefill", make_cache(model))
    outs = [logits_np(model(mx.array(PROMPT[i:i + 7]), cache)) for i in range(0, len(PROMPT), 7)]
    chunked = np.concatenate(outs)
    assert chunked.shape == one.shape
    assert np.abs(chunked - one).max() < 1e-2


def test_bf16_greedy_with_cache_matches_no_cache():
    """bf16：带 cache 的 greedy（递推路径）和不带 cache 的 greedy（分块路径）至少前 8 个 token 相同。
    两条路的 fp32 中间结果差 1e-4，bf16 下 top-2 打平的位置会翻（换三角求解 kernel 后从 32/32 变成 12/32）；
    真正的判据是上面的 fp32 测试。"""
    model, _ = load_model(HF_DIR, dtype=mx.bfloat16)
    ids = list(PROMPT); no_cache = []
    for _ in range(32):
        no_cache.append(int(mx.argmax(model(mx.array(ids))[-1]).item())); ids.append(no_cache[-1])
    cache = ForwardContext("prefill", make_cache(model))
    nxt = int(mx.argmax(model(mx.array(PROMPT), cache)[-1]).item()); with_cache = [nxt]
    for _ in range(31):
        nxt = int(mx.argmax(model(mx.array([nxt]), cache)[-1]).item()); with_cache.append(nxt)
    n = next((i for i, (a, b) in enumerate(zip(no_cache, with_cache)) if a != b), 32)
    print(f"bf16 cache vs no-cache greedy: {n}/32 identical")
    assert n >= 8
