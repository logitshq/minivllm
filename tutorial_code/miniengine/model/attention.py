"""Chapter 03: GQA, per-head q/gate split, partial RoPE and output gate. No cache yet."""
import mlx.core as mx
import mlx.nn as nn

from ..config import ModelConfig
from .layers import Linear, RMSNorm, apply_rope


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
        if ctx is not None:
            raise ValueError("Cache arrives in chapter 06; batch decode in chapter 08")
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

