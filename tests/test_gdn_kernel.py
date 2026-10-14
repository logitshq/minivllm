"""分块 GDN == 逐 token 递推（递推已经通过 fp32 整模型对齐 HF 验证过）。"""
import mlx.core as mx
import numpy as np

from minivllm.kernels.gdn import gated_delta_rule_chunk, gated_delta_rule_recurrent


def rand_inputs(T, Hv=4, Dk=32, Dv=32, seed=0):
    mx.random.seed(seed)
    q, k, v = (mx.random.normal((T, Hv, d)) for d in (Dk, Dk, Dv))
    g = -mx.random.uniform(0.0, 0.5, (T, Hv))            # log 衰减 ≤ 0
    beta = mx.sigmoid(mx.random.normal((T, Hv)))
    return q, k, v, g, beta


def test_chunk_equals_recurrent():
    for T in [1, 7, 64, 200]:
        q, k, v, g, beta = rand_inputs(T)
        o1, S1 = gated_delta_rule_recurrent(q, k, v, g, beta)
        o2, S2 = gated_delta_rule_chunk(q, k, v, g, beta)
        assert np.abs(np.array(o1 - o2)).max() < 1e-4, T
        assert np.abs(np.array(S1 - S2)).max() < 1e-4, T


def test_chunk_state_continuation():
    """分两段算（带初始状态）== 一次算完。"""
    q, k, v, g, beta = rand_inputs(150)
    o, S = gated_delta_rule_chunk(q, k, v, g, beta)
    oa, Sa = gated_delta_rule_chunk(*(t[:70] for t in (q, k, v, g, beta)))
    ob, Sb = gated_delta_rule_chunk(*(t[70:] for t in (q, k, v, g, beta)), state=Sa)
    assert np.abs(np.array(mx.concatenate([oa, ob]) - o)).max() < 1e-4
    assert np.abs(np.array(Sb - S)).max() < 1e-4


def test_chunk_with_correlated_keys():
    """真实数据的形状：相邻 token 的键高度相关、beta 接近 1。显式求逆在这里会炸（曾经的 bug），前向代入不会。"""
    mx.random.seed(3)
    T, Hv, D = 128, 4, 64
    base = mx.random.normal((1, Hv, D))
    k = base + 0.05 * mx.random.normal((T, Hv, D))
    q, v = mx.random.normal((T, Hv, D)), mx.random.normal((T, Hv, D))
    g = -mx.random.uniform(0.0, 0.1, (T, Hv))
    beta = mx.full((T, Hv), 0.98)
    o1, S1 = gated_delta_rule_recurrent(q, k, v, g, beta)
    o2, S2 = gated_delta_rule_chunk(q, k, v, g, beta)
    assert np.abs(np.array(o1 - o2)).max() < 1e-3 * np.abs(np.array(o1)).max()
    assert np.abs(np.array(S1 - S2)).max() < 1e-3 * np.abs(np.array(S1)).max()
