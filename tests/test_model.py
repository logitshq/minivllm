"""第 1 步的验证：整句前向（无 cache）的模型实现是否正确。
参考数据由 tests/make_refs.py（HF）和 tests/data/mlxlm_0.8b_nvfp4.npz（mlx-lm）生成。
"""
import glob, os
import numpy as np
import pytest
import mlx.core as mx

from minivllm.model.loader import load_model

HF_DIR = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/*"))[0]
NVFP4_DIR = "models/Qwen3.5-0.8B-nvfp4"


def greedy_no_cache(model, prompt, n):
    ids, out = list(prompt), []
    for _ in range(n):
        out.append(int(mx.argmax(model(mx.array(ids))[-1]).item()))
        ids.append(out[-1])
    return out


def first_mismatch(a, b):
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))


def test_fp32_matches_hf_exactly():
    """fp32 对 fp32：这是真正的正确性测试。logits 差要在 1e-3 以内，48 个 greedy token 逐个相同。"""
    ref = np.load("tests/data/hf_0.8b.npz")
    model, _ = load_model(HF_DIR, dtype=mx.float32)
    logits = np.array(model(mx.array(ref["prompt"])).astype(mx.float32))
    assert np.abs(logits - ref["fp32_logits"]).max() < 1e-3
    assert first_mismatch(greedy_no_cache(model, ref["prompt"].tolist(), 48), ref["fp32_greedy"].tolist()) == 48


def test_bf16_close_to_hf():
    """bf16 对 bf16：跨框架累加顺序不同，只要求 prompt 上每个位置的 argmax 相同，greedy 至少对 8 个。"""
    ref = np.load("tests/data/hf_0.8b.npz")
    model, _ = load_model(HF_DIR, dtype=mx.bfloat16)
    logits = np.array(model(mx.array(ref["prompt"])).astype(mx.float32))
    assert (logits.argmax(-1) == ref["bf16_logits"].argmax(-1)).all()
    assert np.abs(logits - ref["bf16_logits"]).max() < 2.0
    assert first_mismatch(greedy_no_cache(model, ref["prompt"].tolist(), 48), ref["bf16_greedy"].tolist()) >= 8


def test_nvfp4_matches_mlxlm():
    """nvfp4 对 mlx-lm 同一个 checkpoint：量化 GEMM 相同，GDN/attention 的 kernel 不同（我们是 fp32 递推，它是 Metal kernel），
    所以和 bf16 一档的标准一样：logits 差在 2 以内、prompt 上多数位置 argmax 相同、greedy 至少对 8 个。
    真正的对齐证据是 27B：24 个 greedy token 和 mlx-lm 逐个相同（tests/data/ours_27b_greedy.txt，手动跑）。"""
    ref = np.load("tests/data/mlxlm_0.8b_nvfp4.npz")
    model, cfg = load_model(NVFP4_DIR)
    assert cfg.quantization["mode"] == "nvfp4"
    logits = np.array(model(mx.array(ref["prompt"])).astype(mx.float32))
    assert (logits.argmax(-1) == ref["logits"].argmax(-1)).mean() >= 0.8
    assert np.abs(logits - ref["logits"]).max() < 2.0
    n = first_mismatch(greedy_no_cache(model, ref["prompt"].tolist(), 48), ref["greedy"].tolist())
    print(f"nvfp4 greedy matches mlx-lm for {n} tokens")
    assert n >= 8
