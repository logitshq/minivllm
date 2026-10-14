"""Chapter 03: Chapter 02 mathematics registered as MLX modules. Raw weights only."""
import mlx.core as mx
import mlx.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float, zero_centered: bool = True):
        super().__init__()
        self.weight = mx.zeros((dim,)) if zero_centered else mx.ones((dim,))
        self.eps = eps
        self.zero_centered = zero_centered

    def __call__(self, x: mx.array) -> mx.array:
        w = 1.0 + self.weight if self.zero_centered else self.weight
        xf = x.astype(mx.float32)
        y = xf * mx.rsqrt(mx.mean(xf * xf, axis=-1, keepdims=True) + self.eps)
        return (y * w).astype(x.dtype)


class GatedRMSNorm(nn.Module):
    """GDN 输出的归一化：norm 之后乘权重，再乘 silu(gate)。HF 的 Qwen3_5RMSNormGated。"""

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = mx.ones((dim,))
        self.eps = eps

    def __call__(self, x: mx.array, gate: mx.array) -> mx.array:
        xf = x.astype(mx.float32)
        y = (xf * mx.rsqrt(mx.mean(xf * xf, axis=-1, keepdims=True) + self.eps) * self.weight).astype(x.dtype)
        return y * nn.silu(gate.astype(mx.float32)).astype(y.dtype)


def apply_rope(x: mx.array, offset: int, rotary_dim: int, theta: float) -> mx.array:
    """x: [B, H, T, D]。只旋转前 rotary_dim 维；traditional=False 就是 HF 的 rotate_half 写法。"""
    return mx.fast.rope(x, rotary_dim, traditional=False, base=theta, scale=1.0, offset=offset)


class Linear(nn.Module):
    def __init__(self, in_features, out_features, quantization=None):
        super().__init__()
        if quantization is not None:
            raise ValueError("Quantized weights arrive in chapter 07")
        self.weight = mx.zeros((out_features, in_features), dtype=mx.float32)

    def __call__(self, x):
        return x @ self.weight.T


class Embedding(nn.Module):
    def __init__(self, vocab_size, dim, quantization=None):
        super().__init__()
        if quantization is not None:
            raise ValueError("Quantized weights arrive in chapter 07")
        self.weight = mx.zeros((vocab_size, dim), dtype=mx.float32)

    def __call__(self, ids):
        return self.weight[ids]

    def as_linear(self, x):
        return x @ self.weight.T
