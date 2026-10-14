"""Chapter 02: two building blocks, using unquantized MLX weights."""
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
