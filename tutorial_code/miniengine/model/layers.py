"""Chapter 07: add packed quantized Linear/Embedding while retaining explicit RMSNorm math."""
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
    """无 bias 的线性层。bf16 时只有 weight [out, in]；量化时 weight 是打包的 uint32，外加 scales（affine 还有 biases）。"""

    def __init__(self, in_features: int, out_features: int, quantization: dict | None = None):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.q = quantization
        if quantization is None:
            self.weight = mx.zeros((out_features, in_features), dtype=mx.bfloat16)
        else:
            gs, bits, mode = quantization["group_size"], quantization["bits"], quantization["mode"]
            self.weight = mx.zeros((out_features, in_features * bits // 32), dtype=mx.uint32)
            scale_dtype = mx.uint8 if mode in ("nvfp4", "mxfp4", "mxfp8") else mx.bfloat16
            self.scales = mx.zeros((out_features, in_features // gs), dtype=scale_dtype)
            if mode == "affine":
                self.biases = mx.zeros((out_features, in_features // gs), dtype=mx.bfloat16)

    def __call__(self, x: mx.array) -> mx.array:
        if self.q is None:
            return x @ self.weight.T
        return mx.quantized_matmul(
            x, self.weight, self.scales, getattr(self, "biases", None), transpose=True,
            group_size=self.q["group_size"], bits=self.q["bits"], mode=self.q["mode"],
        )


class Embedding(nn.Module):
    """词表嵌入。量化版只解量化用到的行；tie_word_embeddings 时同一份权重当 lm_head 用（as_linear）。"""

    def __init__(self, vocab_size: int, dim: int, quantization: dict | None = None):
        super().__init__()
        self.q = quantization
        if quantization is None:
            self.weight = mx.zeros((vocab_size, dim), dtype=mx.bfloat16)
        else:
            gs, bits, mode = quantization["group_size"], quantization["bits"], quantization["mode"]
            self.weight = mx.zeros((vocab_size, dim * bits // 32), dtype=mx.uint32)
            scale_dtype = mx.uint8 if mode in ("nvfp4", "mxfp4", "mxfp8") else mx.bfloat16
            self.scales = mx.zeros((vocab_size, dim // gs), dtype=scale_dtype)
            if mode == "affine":
                self.biases = mx.zeros((vocab_size, dim // gs), dtype=mx.bfloat16)

    def __call__(self, ids: mx.array) -> mx.array:
        if self.q is None:
            return self.weight[ids]
        b = getattr(self, "biases", None)
        return mx.dequantize(self.weight[ids], self.scales[ids], None if b is None else b[ids],
                             group_size=self.q["group_size"], bits=self.q["bits"], mode=self.q["mode"])

    def as_linear(self, x: mx.array) -> mx.array:
        if self.q is None:
            return x @ self.weight.T
        return mx.quantized_matmul(x, self.weight, self.scales, getattr(self, "biases", None), transpose=True,
                                   group_size=self.q["group_size"], bits=self.q["bits"], mode=self.q["mode"])
