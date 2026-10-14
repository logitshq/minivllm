"""Gated DeltaNet 的两个"kernel"（纯 MLX，没有 Metal）：
- gated_delta_rule_recurrent：逐 token 递推，decode（T=1）用它。
- gated_delta_rule_chunk：按 64 个 token 分块，块内用矩阵乘一次算完，块间递推。prefill 用它。
  照 HF 的 torch_chunk_gated_delta_rule 移植；三角方程组用前向代入解（MLX 的 solve_triangular 在 CPU 上）。

记号：q, k [T, Hv, Dk]，v [T, Hv, Dv]，g（log 衰减，≤ 0）和 beta（写入强度，(0,1)）[T, Hv] fp32。
递推：S_t = S_{t-1}·exp(g_t) + k_t ⊗ (v_t − k_tᵀS_{t-1})·β_t ；o_t = q_tᵀS_t。q、k 先 L2 归一化，q 再乘 Dk^-0.5。
"""
import mlx.core as mx


def l2norm(x: mx.array, eps: float = 1e-6) -> mx.array:
    return x * mx.rsqrt((x * x).sum(-1, keepdims=True) + eps)


def gated_delta_rule_recurrent(q, k, v, g, beta, state=None):
    T, Hv, Dk = k.shape
    Dv = v.shape[-1]
    q, k, v = (t.astype(mx.float32) for t in (q, k, v))
    q, k = l2norm(q), l2norm(k)
    q = q * Dk ** -0.5
    S = mx.zeros((Hv, Dk, Dv), dtype=mx.float32) if state is None else state
    outs = []
    for t in range(T):
        S = S * mx.exp(g[t])[:, None, None]
        kv_mem = (S * k[t][:, :, None]).sum(axis=1)             # [Hv, Dv]
        delta = (v[t] - kv_mem) * beta[t][:, None]
        S = S + k[t][:, :, None] * delta[:, None, :]
        outs.append((S * q[t][:, :, None]).sum(axis=1))
    return mx.stack(outs), S


def _solve_unit_lower(A: mx.array, V: mx.array) -> mx.array:
    """解 (I + A) X = V，A 严格下三角 [..., C, C]，V [..., C, N]。前向代入：x_i = v_i − Σ_{j<i} A_ij x_j。
    不要显式求逆：真实数据里相邻 token 的键高度相关、β≈1 时，(I+A)^{-1} 的元素按 2^(i−j) 增长再相消，fp32 会炸。
    解本身（就是 delta rule 的修正量 u）是有界的，前向代入稳定。"""
    C = A.shape[-2]
    rows = [V[..., 0:1, :]]
    for i in range(1, C):
        X = mx.concatenate(rows, axis=-2)                                     # [..., i, N]
        rows.append(V[..., i:i + 1, :] - A[..., i:i + 1, :i] @ X)
    return mx.concatenate(rows, axis=-2)


def _solve(A, V):
    try:                                   # Metal 版（一次发射）；没有 Metal 时退回纯 MLX 的前向代入
        from .gdn_metal import solve_unit_lower_metal
        return solve_unit_lower_metal(A, V)
    except ImportError:
        return _solve_unit_lower(A, V)


def gated_delta_rule_chunk(q, k, v, g, beta, state=None, chunk: int = 64):
    T, Hv, Dk = k.shape
    Dv = v.shape[-1]
    q, k, v = (t.astype(mx.float32).transpose(1, 0, 2) for t in (q, k, v))     # [Hv, T, D]
    g, beta = g.T, beta.T                                                     # [Hv, T]
    q, k = l2norm(q), l2norm(k)
    q = q * Dk ** -0.5
    pad = (chunk - T % chunk) % chunk
    if pad:
        q, k, v = (mx.pad(t, [(0, 0), (0, pad), (0, 0)]) for t in (q, k, v))
        g, beta = (mx.pad(t, [(0, 0), (0, pad)]) for t in (g, beta))
    nc = (T + pad) // chunk
    q, k, v = (t.reshape(Hv, nc, chunk, -1) for t in (q, k, v))               # [Hv, nc, C, D]
    g, beta = g.reshape(Hv, nc, chunk), beta.reshape(Hv, nc, chunk)
    v_beta, k_beta = v * beta[..., None], k * beta[..., None]

    cum = mx.cumsum(g, axis=-1)                                               # [Hv, nc, C] 块内累计 log 衰减
    pair = cum[..., :, None] - cum[..., None, :]                              # (i, j): 从 j 到 i 的衰减
    lower = mx.tril(mx.ones((chunk, chunk), dtype=mx.bool_))
    pair = mx.exp(mx.where(lower, pair, -mx.inf))                             # 上三角 → 0
    A = mx.tril((k_beta @ k.transpose(0, 1, 3, 2)) * pair, -1)                # 严格下三角
    intra = (q @ k.transpose(0, 1, 3, 2)) * pair                              # 块内注意力
    sol = _solve(A, mx.concatenate([v_beta, k_beta * mx.exp(cum)[..., None]], axis=-1))
    new_values, k_cumdecay = sol[..., :Dv], sol[..., Dv:]                     # "u"：块内 delta rule 合并后的新值；读旧状态的衰减键
    q = q * mx.exp(cum)[..., None]
    k = k * mx.exp(cum[..., -1:] - cum)[..., None]
    chunk_decay = mx.exp(cum[..., -1])                                        # [Hv, nc]

    S = mx.zeros((Hv, Dk, Dv), dtype=mx.float32) if state is None else state
    outs = []
    for i in range(nc):
        v_new = new_values[:, i] - k_cumdecay[:, i] @ S                       # 减掉旧状态已经"知道"的部分
        outs.append(q[:, i] @ S + intra[:, i] @ v_new)                        # 读旧状态 + 块内
        S = S * chunk_decay[:, i][:, None, None] + k[:, i].transpose(0, 2, 1) @ v_new
    o = mx.concatenate(outs, axis=1)[:, :T]                                   # [Hv, T, Dv]
    return o.transpose(1, 0, 2), S


def gated_delta_step_batch(q, k, v, g, beta, S):
    """decode 一步、B 个序列一起：q, k [B, Hv, Dk]，v [B, Hv, Dv]，g, beta [B, Hv]，S [B, Hv, Dk, Dv] fp32。
    返回 o [B, Hv, Dv] 和新状态。数学和逐 token 递推一样，只是把 o 的读取合并进去：
        o = qᵀS_new = e·(qᵀS) + (q·k)·δ，其中 δ = (v − kᵀS)·β，e = exp(g)
    这样 S 只读两次（qᵀS 和 kᵀS 合成一次矩阵乘、更新时一次）、写一次。S 是 decode 时最大的内存流量（27B 每层 3 MB/序列）。"""
    q, k, v = (t.astype(mx.float32) for t in (q, k, v))
    q, k = l2norm(q), l2norm(k)
    q = q * k.shape[-1] ** -0.5
    e = mx.exp(g)[..., None]                                                  # [B, Hv, 1]
    qk_S = mx.concatenate([q[..., None, :], k[..., None, :]], axis=-2) @ S    # [B, Hv, 2, Dv]
    qS, kS = qk_S[..., 0, :], qk_S[..., 1, :]
    delta = (v - e * kS) * beta[..., None]                                    # kᵀ(e·S) = e·kᵀS
    o = e * qS + (q * k).sum(-1, keepdims=True) * delta
    S = S * e[..., None] + k[..., :, None] * delta[..., None, :]
    return o, S
