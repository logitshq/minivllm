"""Chapter 06: reuse KV with position offset and right-aligned causal mask."""
import mlx.core as mx
import mlx.nn as nn

from ..config import ModelConfig
from .layers import Linear, RMSNorm, apply_rope, prefill_linear


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads, self.n_kv_heads, self.head_dim = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        self.rotary_dim, self.theta = cfg.rotary_dim, cfg.rope_theta
        self.scale = cfg.head_dim ** -0.5
        q = cfg.quantization
        self.q_proj = Linear(cfg.hidden_size, self.n_heads * self.head_dim * 2, q)
        self.k_proj = Linear(cfg.hidden_size, self.n_kv_heads * self.head_dim, q)
        self.v_proj = Linear(cfg.hidden_size, self.n_kv_heads * self.head_dim, q)
        self.o_proj = Linear(self.n_heads * self.head_dim, cfg.hidden_size, q)
        self.q_norm = RMSNorm(self.head_dim, cfg.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, cfg.rms_norm_eps)

    def __call__(self, x: mx.array, ctx=None, layer_idx: int = 0) -> mx.array:
        if ctx is not None and ctx.mode == "prefill_batch":
            return self._prefill_batch(x, ctx, layer_idx)
        if ctx is not None and ctx.mode == "decode":
            return self._decode(x, ctx, layer_idx)
        return self._prefill(x, ctx.caches[layer_idx] if ctx is not None else None)

    def _project(self, x):
        T = x.shape[0]
        qg = self.q_proj(x).reshape(T, self.n_heads, 2 * self.head_dim)
        q, gate = mx.split(qg, 2, axis=-1)                       # 每个头：前 D 是 q，后 D 是门
        k = self.k_proj(x).reshape(T, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).reshape(T, self.n_kv_heads, self.head_dim)
        return self.q_norm(q), self.k_norm(k), v, gate.reshape(T, -1)   # 按头做 RMSNorm（零中心）

    def _prefill(self, x: mx.array, cache) -> mx.array:
        """单序列：x [T, hidden]；cache=None 时整句前向，否则接着 cache.offset 往后算。"""
        T = x.shape[0]
        offset = cache.offset if cache is not None else 0      # 已有多少 token：RoPE 的位置从这里起算
        q, k, v, gate = self._project(x)
        q = apply_rope(q.transpose(1, 0, 2)[None], offset, self.rotary_dim, self.theta)   # [1, H, T, D]
        k = apply_rope(k.transpose(1, 0, 2)[None], offset, self.rotary_dim, self.theta)
        v = v.transpose(1, 0, 2)[None]
        if cache is not None:
            k, v = cache.update(k, v)                            # [1, Hkv, offset+T, D]
        # T 个 query 对 offset+T 个 key：causal 掩码按右下角对齐（第 i 个新 token 能看到 offset+i 之前的全部）
        mask = "causal" if T > 1 else None
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)  # GQA 自动广播
        o = o[0].transpose(1, 0, 2).reshape(T, -1)
        return self.o_proj(o * mx.sigmoid(gate))


    def _prefill_batch(self, x: mx.array, ctx, layer_idx: int) -> mx.array:
        """Fresh independent equal-T rows; share projections, never attention history."""
        B, T = ctx.prefill_shape
        caches = ctx.caches[layer_idx]
        qg = prefill_linear(self.q_proj, x, (B, T)).reshape(B * T, self.n_heads, 2 * self.head_dim)
        q, gate = mx.split(qg, 2, axis=-1)
        k = prefill_linear(self.k_proj, x, (B, T)).reshape(B * T, self.n_kv_heads, self.head_dim)
        v = prefill_linear(self.v_proj, x, (B, T)).reshape(B * T, self.n_kv_heads, self.head_dim)
        q, k = self.q_norm(q), self.k_norm(k)
        gate = gate.reshape(B * T, -1)
        q = q.reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        q = apply_rope(q, 0, self.rotary_dim, self.theta)
        k = apply_rope(k, 0, self.rotary_dim, self.theta)
        for row, cache in enumerate(caches):
            cache.update(k[row:row + 1], v[row:row + 1])
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale,
                                                mask="causal" if T > 1 else None)
        o = o.transpose(0, 2, 1, 3).reshape(B * T, -1)
        return prefill_linear(self.o_proj, o * mx.sigmoid(gate), (B, T))


    def _decode(self, x: mx.array, ctx, layer_idx: int) -> mx.array:
        """批量：x [B, hidden]，每行一个新 token，位置各不相同。"""
        B = x.shape[0]
        kv = ctx.caches[layer_idx]
        q, k, v, gate = self._project(x)                         # [B, H, D], [B, Hkv, D]
        q = apply_rope(q[:, :, None], ctx.positions, self.rotary_dim, self.theta)   # [B, H, 1, D]，每行自己的位置
        k = apply_rope(k[:, :, None], ctx.positions, self.rotary_dim, self.theta)[:, :, 0]
        kv.write(k, v, ctx.positions)
        L = ctx.max_len
        K, V = kv.K[:, :, :L], kv.V[:, :, :L]
        mask = (mx.arange(L)[None, :] < ctx.lens[:, None])[:, None, None, :]        # 每行只看自己的历史
        o = mx.fast.scaled_dot_product_attention(q, K, V, scale=self.scale, mask=mask)
        o = o[:, :, 0].reshape(B, -1)
        return self.o_proj(o * mx.sigmoid(gate))
