"""Gated DeltaNet（24 层里的 18 层）。递推/分块两个 kernel 在 kernels/gdn.py；这里是投影、卷积、门和归一化。

每个 token t（每个 v 头独立，状态 S 是 Dk×Dv 的矩阵，fp32）：
    S = S * exp(g_t)                       # 衰减，g_t = -exp(A_log) * softplus(a_t + dt_bias) < 0
    kv_mem = k_t^T S                       # 当前状态对 k_t 的"记忆"
    S = S + k_t ⊗ (v_t - kv_mem) * beta_t  # delta rule：只写入新信息，beta_t = sigmoid(b_t)
    o_t = q_t^T S
q、k 先做 L2 归一化，q 再除以 sqrt(Dk)。进 q/k/v 之前先过一个深度可分离的因果卷积（kernel 4）+ silu。
"""
import mlx.core as mx
import mlx.nn as nn

from ..config import ModelConfig
from ..kernels.gdn import gated_delta_rule_chunk, gated_delta_rule_recurrent
from ..kernels.gdn_metal import gated_delta_step_metal
from .layers import GatedRMSNorm, Linear


def causal_conv1d(x: mx.array, w: mx.array, state: mx.array | None = None) -> tuple[mx.array, mx.array]:
    """x: [T, C]，w: [C, K]，state: 之前的 K-1 个输入 [K-1, C]（None = 补 0，即序列开头）。
    y[t] = sum_j w[:, j] * x[t - (K-1) + j]。等价于 HF 的 conv1d(padding=K-1) 取前 T 个。
    返回 y 和新的 state（最后 K-1 个输入），decode 时 T=1 也走这条路。"""
    K = w.shape[1]
    T = x.shape[0]
    xp = mx.concatenate([mx.zeros((K - 1, x.shape[1]), x.dtype) if state is None else state.astype(x.dtype), x])
    y = sum(xp[j:j + T] * w[:, j] for j in range(K))
    return y, xp[T:]


class GatedDeltaNet(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.Hk, self.Hv = cfg.linear_num_key_heads, cfg.linear_num_value_heads
        self.Dk, self.Dv = cfg.linear_key_head_dim, cfg.linear_value_head_dim
        key_dim, value_dim = self.Hk * self.Dk, self.Hv * self.Dv
        self.conv_dim = 2 * key_dim + value_dim
        q = cfg.quantization
        self.in_proj_qkv = Linear(cfg.hidden_size, self.conv_dim, q)
        self.in_proj_z = Linear(cfg.hidden_size, value_dim, q)
        self.in_proj_b = Linear(cfg.hidden_size, self.Hv, q)
        self.in_proj_a = Linear(cfg.hidden_size, self.Hv, q)
        self.conv1d_weight = mx.zeros((self.conv_dim, cfg.linear_conv_kernel_dim), dtype=mx.bfloat16)
        self.A_log = mx.zeros((self.Hv,), dtype=mx.float32)
        self.dt_bias = mx.zeros((self.Hv,), dtype=mx.float32)
        self.norm = GatedRMSNorm(self.Dv, cfg.rms_norm_eps)
        self.out_proj = Linear(value_dim, cfg.hidden_size, q)

    def __call__(self, x: mx.array, ctx=None, layer_idx: int = 0) -> mx.array:
        if ctx is not None and ctx.mode == "decode":
            return self._decode(x, ctx.caches[layer_idx])
        return self._prefill(x, ctx.caches[layer_idx] if ctx is not None else None)

    def _gates(self, b, a):
        beta = mx.sigmoid(b.astype(mx.float32))
        g = -mx.exp(self.A_log) * nn.softplus(a.astype(mx.float32) + self.dt_bias)
        return beta, g

    def _split_heads(self, qkv, n):
        q, k, v = mx.split(qkv, [self.Hk * self.Dk, 2 * self.Hk * self.Dk], axis=-1)
        q, k, v = q.reshape(n, self.Hk, self.Dk), k.reshape(n, self.Hk, self.Dk), v.reshape(n, self.Hv, self.Dv)
        if self.Hv > self.Hk:                                        # 27B：16 个 k 头服务 48 个 v 头
            rep = self.Hv // self.Hk
            q, k = mx.repeat(q, rep, axis=1), mx.repeat(k, rep, axis=1)
        return q, k, v

    def _prefill(self, x: mx.array, cache) -> mx.array:
        """单序列：x [T, hidden]。T=1 走递推，否则走分块；状态接 cache。"""
        T = x.shape[0]
        qkv = self.in_proj_qkv(x)
        z = self.in_proj_z(x).reshape(T, self.Hv, self.Dv)
        conv_state = cache.conv_state if cache is not None and cache.offset > 0 else None
        raw = qkv
        qkv, new_conv_state = causal_conv1d(qkv, self.conv1d_weight.astype(qkv.dtype), conv_state)
        q, k, v = self._split_heads(nn.silu(qkv), T)
        beta, g = self._gates(self.in_proj_b(x), self.in_proj_a(x))
        if cache is not None and cache.record:
            cache.tape = (raw, q, k, v, g, beta)                      # 投机解码回滚用
        S0 = cache.S if cache is not None and cache.offset > 0 else None
        rule = gated_delta_rule_recurrent if T == 1 else gated_delta_rule_chunk   # decode 递推，prefill 分块
        o, S = rule(q, k, v, g, beta, S0)
        if cache is not None:
            cache.conv_state, cache.S, cache.offset = new_conv_state, S, cache.offset + T
        o = self.norm(o.astype(x.dtype), z)
        return self.out_proj(o.reshape(T, -1))

    def _decode(self, x: mx.array, st) -> mx.array:
        """批量：x [B, hidden]；st.conv [B, K-1, C]，st.S [B, Hv, Dk, Dv]，整体替换。"""
        B = x.shape[0]
        qkv = self.in_proj_qkv(x)                                   # [B, C]
        z = self.in_proj_z(x).reshape(B, self.Hv, self.Dv)
        w = self.conv1d_weight.astype(qkv.dtype)                    # [C, K]
        xp = mx.concatenate([st.conv.astype(qkv.dtype), qkv[:, None]], axis=1)   # [B, K, C]
        qkv = (xp * w.T[None]).sum(axis=1)                          # 因果卷积在最后一个位置的值
        q, k, v = self._split_heads(nn.silu(qkv), B)
        beta, g = self._gates(self.in_proj_b(x), self.in_proj_a(x))
        o, S = gated_delta_step_metal(q, k, v, g, beta, st.S)
        st.conv, st.S = xp[:, 1:], S
        o = self.norm(o.astype(x.dtype), z)
        return self.out_proj(o.reshape(B, -1))
