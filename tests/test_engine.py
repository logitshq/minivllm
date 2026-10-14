"""第 4 步：引擎端到端。fp32 的 0.8B，判据是和单序列 / HF 的 greedy 完全一致，以及批量不变性。"""
import glob, os
import numpy as np
import mlx.core as mx

from minivllm.engine.llm_engine import LLMEngine
from minivllm.sampling_params import SamplingParams

HF_DIR = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/*"))[0]
REF = np.load("tests/data/hf_0.8b.npz")
PROMPT = REF["prompt"].tolist()
P2 = PROMPT[:12] + [248046, 198, 248045, 74455, 198]        # 截断的变体
P3 = [248045, 846, 198, 108386, 248046, 198, 248045, 74455, 198]


def test_single_request_matches_hf_greedy():
    eng = LLMEngine(HF_DIR, max_seqs=2, max_len=256, max_batched_tokens=64, dtype=mx.float32)
    out = eng.generate([PROMPT], SamplingParams(max_tokens=32, ignore_eos=True))[0]
    assert out == REF["fp32_greedy"].tolist()[:32]


def test_batch_invariance_and_chunked_prefill():
    """三个请求一起跑（批量 decode、分块 prefill 预算 8）== 各自单独跑（预算 512）。"""
    sp = SamplingParams(max_tokens=24, ignore_eos=True)
    solo = LLMEngine(HF_DIR, max_seqs=1, max_len=256, max_batched_tokens=512, dtype=mx.float32)
    alone = [solo.generate([p], sp)[0] for p in (PROMPT, P2, P3)]
    together = LLMEngine(HF_DIR, max_seqs=3, max_len=256, max_batched_tokens=8, dtype=mx.float32)
    batched = together.generate([PROMPT, P2, P3], sp)
    for a, b in zip(alone, batched):
        assert a == b
    assert alone[0] == REF["fp32_greedy"].tolist()[:24]


def test_eos_and_slot_reuse():
    """不 ignore_eos 时在 EOS 停；5 个请求挤 2 个槽，全部完成。"""
    eng = LLMEngine(HF_DIR, max_seqs=2, max_len=256, max_batched_tokens=32, dtype=mx.bfloat16)
    outs = eng.generate([PROMPT, P2, P3, PROMPT, P3], SamplingParams(max_tokens=64))
    assert len(outs) == 5
    for o in outs:
        assert 1 <= len(o) <= 64
        assert o[-1] in eng.eos_ids or len(o) == 64
    assert not eng.has_work() and eng.runner.staging == {} and sorted(eng.scheduler.free_slots) == [0, 1]
