"""Lazy V30 fixed128 canonical GDN. Intent: reuse32 state values per thread.
No engine connection; actual register allocation/spill/reload elimination is unmeasured.
Default Metal compile policy is exactly the accepted25 constructor's omitted options.
"""
from pathlib import Path
SOURCE=Path(__file__).with_suffix(".metal").read_text()
_KERNEL=None

def gated_delta_step_cached(q,k,v,g,beta,S,*,backend=None):
    if backend is None:
        import mlx.core as backend
    mx=backend
    if not (S.shape==(1,48,128,128) and S.dtype==mx.float32
            and q.shape==k.shape==v.shape==(1,48,128)
            and q.dtype==k.dtype==v.dtype and q.dtype in (mx.bfloat16,mx.float32)
            and g.shape==beta.shape==(1,48) and g.dtype==beta.dtype==mx.float32):
        raise ValueError("V30 fixed B1/Hv48/Dk128/Dv128 and canonical FP32 state")
    global _KERNEL
    if _KERNEL is None:
        _KERNEL=mx.fast.metal_kernel(name="v30_gdn_register_cache",
            input_names=["q","k","v","g","beta","S"],output_names=["o","S_new"],
            source=SOURCE.replace("SPLIT","4").replace("VALUE_TILE","32"))
    return tuple(_KERNEL(inputs=[q.astype(mx.float32),k.astype(mx.float32),v.astype(mx.float32),g.astype(mx.float32),beta.astype(mx.float32),S],
        grid=(128,48*4,1),threadgroup=(32,4,1),
        output_shapes=[(1,48,128),(1,48,128,128)],output_dtypes=[mx.float32,mx.float32]))
