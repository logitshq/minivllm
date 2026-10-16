"""Chapter 07: add packed quantized Linear/Embedding while retaining explicit RMSNorm math."""
import mlx.core as mx
import mlx.nn as nn
from functools import partial, lru_cache
from importlib.metadata import version
import re
from ..kernels.norm_metal import norm as _metal_norm


@partial(mx.compile, shapeless=True)
def _rms_norm_explicit(x, w, eps):
    xf = x.astype(mx.float32)
    y = xf * mx.rsqrt(mx.mean(xf * xf, axis=-1, keepdims=True) + eps)
    return (y * w).astype(x.dtype)


@partial(mx.compile, shapeless=True)
def _gated_rms_norm_explicit(x, w, gate, eps):
    xf = x.astype(mx.float32)
    y = (xf * mx.rsqrt(mx.mean(xf * xf, axis=-1, keepdims=True) + eps) * w).astype(x.dtype)
    return y * nn.silu(gate.astype(mx.float32)).astype(y.dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float, zero_centered: bool = True):
        super().__init__()
        self.weight = mx.zeros((dim,)) if zero_centered else mx.ones((dim,))
        self.eps = eps
        self.zero_centered = zero_centered

    def __call__(self, x: mx.array) -> mx.array:
        if (x.dtype == mx.bfloat16 and x.ndim >= 2
                and self.weight.shape == (x.shape[-1],)
                and self.weight.dtype in (mx.bfloat16, mx.float16, mx.float32)):
            return _metal_norm(x, self.weight, self.eps, zero_centered=self.zero_centered)
        w = 1.0 + self.weight if self.zero_centered else self.weight
        return _rms_norm_explicit(x, w, self.eps)


class GatedRMSNorm(nn.Module):
    """GDN 输出的归一化：norm 之后乘权重，再乘 silu(gate)。HF 的 Qwen3_5RMSNormGated。"""

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = mx.ones((dim,))
        self.eps = eps

    def __call__(self, x: mx.array, gate: mx.array) -> mx.array:
        if (x.dtype == mx.bfloat16 and x.ndim >= 2 and gate.shape == x.shape
                and self.weight.shape == (x.shape[-1],)
                and self.weight.dtype in (mx.bfloat16, mx.float16, mx.float32)
                and gate.dtype in (mx.bfloat16, mx.float16, mx.float32)):
            return _metal_norm(x, self.weight, self.eps, gate=gate)
        return _gated_rms_norm_explicit(x, self.weight, gate, self.eps)


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
        self._output_dtype = mx.bfloat16  # Activations; packed uint32 weights remain packed.
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
                             group_size=self.q["group_size"], bits=self.q["bits"], mode=self.q["mode"]).astype(self._output_dtype)

    def as_linear(self, x: mx.array) -> mx.array:
        if self.q is None:
            return x @ self.weight.T
        return mx.quantized_matmul(x, self.weight, self.scales, getattr(self, "biases", None), transpose=True,
                                   group_size=self.q["group_size"], bits=self.q["bits"], mode=self.q["mode"])


@lru_cache(maxsize=1)
def _prefill_backend_identity():
    try:
        return version("mlx"), str(mx.metal.device_info().get("architecture", ""))
    except Exception:
        return None, None


@lru_cache(maxsize=1)
def _prefill_nax_identity():
    return (_prefill_backend_identity() == ("0.32.3", "applegpu_g17d")
            and mx.device_info().get("device_name") == "Apple M5 Ultra")


def prefill_batch_backend_supported(shape=None):
    # Legacy QMM may flatten rows. Qualified M5 NAX keeps M=T instead.
    if mx.default_device() != mx.gpu:
        return False
    ver, arch = _prefill_backend_identity()
    match = re.fullmatch(r"applegpu_g(\d+)[a-z]*", arch or "")
    if ver == "0.32.3" and match is not None and int(match[1]) < 17:
        return True
    return (_prefill_nax_identity() and (shape is None or
            (2 <= shape[0] <= 32 and 1 <= shape[1] <= 512 and shape[0] * shape[1] <= 4096)))


def prefill_linear(op, x, shape):
    """Share only the pinned splitK1 reduction; keep other projections per row.

    MLX0.32.3 quantized.cpp:1081-1191 stores splitK partials in BF16.
    Changing M can change both the partition and its rounding. The product
    below>256 proves floor(512/(ceil(N/32)*ceil(M/32)))<=1 before K adjustment.
    K>4096 and T>=14 also stay above the qualified devices' qmv limits.
    M5 NAX changes BM at M=32: broadcast weights to retain M=T and the
    original reduction tile. Unknown metadata/shape keeps per-row calls.
    """
    B, T = shape
    q = getattr(op, "q", None)
    K, N = getattr(op, "in_features", 0), getattr(op, "out_features", 0)
    if (prefill_batch_backend_supported(shape) and x.dtype == mx.bfloat16 and x.ndim == 2
            and x.shape == (B * T, K) and B >= 2 and T >= 14 and K > 4096
            and K % 32 == 0 and N > 0 and N % 32 == 0
            and q == {"group_size": 16, "bits": 4, "mode": "nvfp4"}
            and op.weight.ndim == 2 and op.weight.dtype == mx.uint32
            and op.scales.dtype == mx.uint8
            and ((N + 31) // 32) * ((T + 31) // 32) > 256
            and ((N + 31) // 32) * ((B * T + 31) // 32) > 256):
        if _prefill_nax_identity():
            if K % 64 == 0:
                weight = mx.broadcast_to(op.weight, (B,) + op.weight.shape)
                scales = mx.broadcast_to(op.scales, (B,) + op.scales.shape)
                return mx.quantized_matmul(x.reshape(B, T, K), weight, scales,
                                           transpose=True, **q).reshape(B * T, N)
        else:
            return op(mx.contiguous(x))
    return mx.concatenate([op(row) for row in x.reshape(B, T, -1)], axis=0)
