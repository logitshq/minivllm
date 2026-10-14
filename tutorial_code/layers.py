"""Chapter 02: unquantized building blocks implemented with MLX."""
from dataclasses import dataclass

import mlx.core as mx


@dataclass
class Embedding:
    weight: mx.array                 # [vocab_size, hidden_size]

    def __call__(self, token_ids: mx.array) -> mx.array:
        return self.weight[token_ids]


@dataclass
class Linear:
    weight: mx.array                 # [out_features, in_features]

    def __call__(self, x: mx.array) -> mx.array:
        return x @ self.weight.T     # [..., in] → [..., out]


@dataclass
class RMSNorm:
    weight: mx.array                 # [hidden_size], stored as w for Qwen
    eps: float = 1e-6
    zero_centered: bool = True

    def __call__(self, x: mx.array) -> mx.array:
        # The explicit math is kept here before introducing fused kernels.
        xf = x.astype(mx.float32)
        mean_square = mx.mean(xf * xf, axis=-1, keepdims=True)
        normalized = xf * mx.rsqrt(mean_square + self.eps)
        scale = 1.0 + self.weight if self.zero_centered else self.weight
        return (normalized * scale).astype(x.dtype)


def silu(x: mx.array) -> mx.array:
    return x * mx.sigmoid(x)


@dataclass
class MLP:
    gate_proj: Linear                # H → I
    up_proj: Linear                  # H → I
    down_proj: Linear                # I → H

    def __call__(self, x: mx.array) -> mx.array:
        gate = silu(self.gate_proj(x))
        value = self.up_proj(x)
        return self.down_proj(gate * value)


@dataclass
class FeedForwardBlock:
    """The second sub-block of a decoder layer: x + MLP(norm(x))."""
    norm: RMSNorm
    mlp: MLP

    def __call__(self, x: mx.array) -> mx.array:
        return x + self.mlp(self.norm(x))
