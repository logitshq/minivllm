"""基础层。全部是几行的小函数/小模块，和 HF 的 modeling_qwen3_5.py 一一对应。

关键细节（都是踩过坑才知道的）：
- Qwen3.5 的 RMSNorm 是"零中心"的：y = norm(x) * (1 + w)，权重存的是 w，初始为 0。mx.fast.rms_norm 算的是 norm(x) * w，所以传 1 + w。
- GDN 后面的门控 RMSNorm 不是零中心的：y = norm(x) * w * silu(z)。
- RoPE 只转 head_dim 的前 1/4（256 维里转 64 维），用 rotate_half 式（非交错），theta 1e7。
  原版是 mrope（时间/高/宽三路位置），纯文本时三路位置相同，数学上退化成普通 RoPE。测试里会数值验证这一点。
- Linear 没有 bias。量化权重用 mx.quantized_matmul，nvfp4 只有 scales 没有 biases。
"""
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
        return mx.fast.rms_norm(x, w.astype(x.dtype), self.eps)


class GatedRMSNorm(nn.Module):
    """GDN 输出的归一化：norm 之后乘权重，再乘 silu(gate)。HF 的 Qwen3_5RMSNormGated。"""

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = mx.ones((dim,))
        self.eps = eps

    def __call__(self, x: mx.array, gate: mx.array) -> mx.array:
        y = mx.fast.rms_norm(x, self.weight.astype(x.dtype), self.eps)
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
