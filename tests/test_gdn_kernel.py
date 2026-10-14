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


def test_metal_step_equals_batch_step():
    """Metal 版 decode 一步 == 纯 MLX 版 == 逐 token 递推。"""
    from minivllm.kernels.gdn import gated_delta_step_batch
    from minivllm.kernels.gdn_metal import gated_delta_step_metal
    mx.random.seed(5)
    B, Hv, Dk, Dv = 3, 4, 128, 128
    q, k, v = (mx.random.normal((B, Hv, d)) for d in (Dk, Dk, Dv))
    g = -mx.random.uniform(0.0, 3.0, (B, Hv)); beta = mx.sigmoid(mx.random.normal((B, Hv)))
    S = mx.random.normal((B, Hv, Dk, Dv))
    o1, S1 = gated_delta_step_batch(q, k, v, g, beta, S)
    o2, S2 = gated_delta_step_metal(q, k, v, g, beta, S)
    assert np.abs(np.array(o1 - o2)).max() < 1e-4 * np.abs(np.array(o1)).max()
    assert np.abs(np.array(S1 - S2)).max() < 1e-5 * np.abs(np.array(S1)).max()
    for b in range(B):   # 和逐 token 递推（带初始状态）逐序列对比
        o3, S3 = gated_delta_rule_recurrent(q[b:b+1], k[b:b+1], v[b:b+1], g[b:b+1], beta[b:b+1], S[b])
        assert np.abs(np.array(o3[0] - o2[b])).max() < 1e-4 * np.abs(np.array(o3)).max()


def test_metal_solve_equals_python_solve():
    from minivllm.kernels.gdn import _solve_unit_lower
    from minivllm.kernels.gdn_metal import solve_unit_lower_metal
    mx.random.seed(7)
    A = mx.tril(mx.random.normal((3, 5, 64, 64)), -1)
    V = mx.random.normal((3, 5, 64, 256))
    X1, X2 = _solve_unit_lower(A, V), solve_unit_lower_metal(A, V)
    assert np.abs(np.array(X1 - X2)).max() < 1e-3 * np.abs(np.array(X1)).max()
