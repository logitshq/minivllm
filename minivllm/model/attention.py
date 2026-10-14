"""全注意力层（24 层里每 4 层有 1 层）。第 1 步没有 cache：整句一起算，causal mask。

和普通 Qwen3 的区别：q_proj 的输出是两倍宽，一半是 q、一半是输出门，按头交错排列
（view 成 [T, H, 2*D] 后沿最后一维对半切），最后 attn_out * sigmoid(gate)。
"""
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

    def __call__(self, x: mx.array) -> mx.array:
        T, _ = x.shape
        qg = self.q_proj(x).reshape(T, self.n_heads, 2 * self.head_dim)
        q, gate = mx.split(qg, 2, axis=-1)                       # 每个头：前 D 是 q，后 D 是门
        k = self.k_proj(x).reshape(T, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).reshape(T, self.n_kv_heads, self.head_dim)
        q, k = self.q_norm(q), self.k_norm(k)                    # 按头做 RMSNorm（零中心）
        q = apply_rope(q.transpose(1, 0, 2)[None], 0, self.rotary_dim, self.theta)   # [1, H, T, D]
        k = apply_rope(k.transpose(1, 0, 2)[None], 0, self.rotary_dim, self.theta)
        v = v.transpose(1, 0, 2)[None]
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask="causal")  # GQA 自动广播
        o = o[0].transpose(1, 0, 2).reshape(T, -1)
        o = o * mx.sigmoid(gate.reshape(T, -1))
        return self.o_proj(o)
