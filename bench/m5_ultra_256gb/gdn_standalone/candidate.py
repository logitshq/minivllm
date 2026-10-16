"""Experimental GDN fusion with unchanged, separate projection GEMMs.

No runtime import or default flag enables this prototype. The context manager
is for a single-threaded, owned qualification process only.
"""
from contextlib import contextmanager
import mlx.core as mx

from tutorial_code.miniengine.cache import BatchState
from tutorial_code.miniengine.kernels import gdn_decode_fused as kernel
from tutorial_code.miniengine.model.gdn import GatedDeltaNet, _m5_ultra_decode_identity


def fused(raw, conv, w, g, beta, state):
    B = raw.shape[0]
    if not (9 <= B <= 64 and raw.shape == (B, 10240)
            and conv.shape == (B, 3, 10240) and w.shape == (10240, 4)
            and raw.dtype == conv.dtype == w.dtype == mx.bfloat16
            and g.shape == beta.shape == (B, 48)
            and state.shape == (B, 48, 128, 128)
            and g.dtype == beta.dtype == state.dtype == mx.float32):
        raise ValueError('standalone GDN prototype requires qualified B9..64 shapes')
    q, k, v, scales, tail = kernel.batch_kernel(
        inputs=[raw, conv, w], grid=(64 * 128, B, 1), threadgroup=(128, 1, 1),
        output_shapes=[(B, 16, 128), (B, 16, 128), (B, 48, 128),
                       (B, 16, 2), (B, 3, 10240)],
        output_dtypes=[mx.float32] * 4 + [mx.bfloat16])
    o, updated = kernel.state_kernel(
        inputs=[q, k, v, scales, g, beta, state],
        grid=(128, 48 * kernel.nr.SPLIT, B),
        threadgroup=(kernel.nr.VALUE_TILE, kernel.nr.SPLIT, 1),
        output_shapes=[(B, 48, 128), (B, 48, 128, 128)],
        output_dtypes=[mx.float32, mx.float32])
    return o, updated, tail


REFERENCE = GatedDeltaNet._decode


def decode(self, x, st, *, _compile_gates=False, _norm_reuse_normal=False):
    eligible = (_norm_reuse_normal is True and type(st) is BatchState
                and set(vars(st)) == {'conv', 'S'}
                and (self.Hk, self.Hv, self.Dk, self.Dv) == (16, 48, 128, 128)
                and x.ndim == 2 and 9 <= x.shape[0] <= 64
                and x.shape[1] == 5120 and x.dtype == mx.bfloat16
                and mx.default_device() == mx.gpu and _m5_ultra_decode_identity())
    if not eligible:
        return REFERENCE(self, x, st, _compile_gates=_compile_gates,
                         _norm_reuse_normal=_norm_reuse_normal)
    # Keep all four original projections: no concatenation, padding or alternate
    # matmul reduction route. Only their elementwise consumers are fused.
    raw = self.in_proj_qkv(x)
    z = self.in_proj_z(x).reshape(x.shape[0], self.Hv, self.Dv)
    beta, g = self._gates(self.in_proj_b(x), self.in_proj_a(x))
    o, state, tail = fused(raw, st.conv, self.conv1d_weight, g, beta, st.S)
    st.conv, st.S = tail, state
    return self.out_proj(self.norm(o.astype(x.dtype), z).reshape(x.shape[0], -1))


@contextmanager
def enabled(active=True):
    previous = GatedDeltaNet._decode
    if active:
        GatedDeltaNet._decode = decode
    try:
        yield
    finally:
        GatedDeltaNet._decode = previous
