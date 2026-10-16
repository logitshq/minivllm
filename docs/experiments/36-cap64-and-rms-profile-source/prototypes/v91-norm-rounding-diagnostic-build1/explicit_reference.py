from functools import partial
import mlx.core as mx

@partial(mx.compile, shapeless=True)
def _rms_norm_explicit(x, w, eps):
    xf = x.astype(mx.float32)
    y = xf * mx.rsqrt(mx.mean(xf * xf, axis=-1, keepdims=True) + eps)
    return (y * w).astype(x.dtype)
