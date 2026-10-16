"""Lazy experimental recurrence. Import uses stdlib only; engine is not connected.
Logical state remains [B,Hv,Dk,Dv]; output backing storage is [B,Hv,Dv,Dk].
"""
from pathlib import Path
_KERNEL = None
SOURCE = Path(__file__).with_suffix('.metal').read_text()
COMPILE_OPTIONS = {'math_mode': 'safe'}
def gated_delta_step_simd_serial(q, k, v, g, beta, S, *, backend=None):
    if backend is None:
        import mlx.core as backend
    mx = backend
    B = S.shape[0]
    if (B not in (1, 2, 4) or S.shape != (B, 48, 128, 128) or S.dtype != mx.float32
            or q.shape != (B, 16, 128) or k.shape != q.shape
            or q.dtype not in (mx.bfloat16, mx.float32) or k.dtype != q.dtype
            or v.shape != (B, 48, 128) or v.dtype != q.dtype
            or g.shape != (B, 48) or beta.shape != g.shape
            or g.dtype != mx.float32 or beta.dtype != mx.float32):
        raise ValueError('V29 requires compact Hk16/Hv48, Dk=Dv128, B1/2/4 and FP32 state/g/beta')
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = mx.fast.metal_kernel(name='v29_gdn_simd_serial',
            input_names=['q','k','v','g','beta','S'], output_names=['o','S_physical'],
            source=SOURCE, ensure_row_contiguous=False, compile_options=COMPILE_OPTIONS)
    o, physical = _KERNEL(inputs=[q,k,v,g,beta,S], grid=(32,128,B*48),
        threadgroup=(32,4,1), output_shapes=[(B,48,128),(B,48,128,128)],
        output_dtypes=[mx.float32,mx.float32])
    return o, mx.swapaxes(physical, -1, -2)
