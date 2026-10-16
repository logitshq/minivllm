"""Batched convolution, BF16 activation and serial GDN norms in one preparation pass.

Keep BF16 rounding at every convolution addition, as in MLX's column sum.
The existing norm-reuse state kernel retains the original reduction order.
"""
import mlx.core as mx
from . import gdn_norm_reuse_metal as nr
HEADER=nr.HEADER+r'''
inline float prepared_column(uint c, const device bfloat16_t* raw,
        const device bfloat16_t* conv, const device bfloat16_t* w,
        device bfloat16_t* new_conv) {
    bfloat16_t p0 = conv[c] * w[4*c];
    bfloat16_t p1 = conv[10240+c] * w[4*c+1];
    bfloat16_t p2 = conv[20480+c] * w[4*c+2];
    bfloat16_t p3 = raw[c] * w[4*c+3];
    bfloat16_t s = p0 + p1;
    s += p2; s += p3;
    bfloat16_t u = bfloat16_t(s);
    new_conv[c] = conv[10240+c];
    new_conv[10240+c] = conv[20480+c];
    new_conv[20480+c] = raw[c];
    return float(v18_bf16_activation(u).activated);
}
'''
source=nr._PREWORK_SOURCE
for index in ('h * 128 + lane','2048 + h * 128 + lane','4096 + h * 128 + lane'):
    source=source.replace('float(v18_bf16_activation(u['+index+']).activated)',
                          'prepared_column('+index+', raw, conv, w, new_conv)')
assert 'u[' not in source
# Retain the already-qualified single-row pipelines exactly.
single_kernel=mx.fast.metal_kernel(name='gdn_decode_conv_prework_bf16', input_names=['raw','conv','w'],
    output_names=['q_ready','k_ready','v_ready','scales','new_conv'],source=source,header=HEADER,
    compile_options=nr.COMPILE_OPTIONS)
# Batch indices only: arithmetic and per-head reduction order are unchanged.
source = source.replace("    uint group = threadgroup_position_in_grid.x;", """    uint group = threadgroup_position_in_grid.x;
    uint batch = threadgroup_position_in_grid.y;
    raw += batch * 10240; conv += batch * 30720; new_conv += batch * 30720;
    q_ready += batch * 2048; k_ready += batch * 2048;
    v_ready += batch * 6144; scales += batch * 32;""")
batch_kernel=mx.fast.metal_kernel(name='gdn_decode_conv_prework_batch_bf16', input_names=['raw','conv','w'],
    output_names=['q_ready','k_ready','v_ready','scales','new_conv'], source=source, header=HEADER,
    compile_options=nr.COMPILE_OPTIONS)
state_source = nr._STATE_SOURCE.replace('(size_t)(h / 3) * Dk', '(size_t)(b * 16 + h / 3) * Dk').replace('scales[(h / 3)', 'scales[(b * 16 + h / 3)').replace('SPLIT',str(nr.SPLIT)).replace('VALUE_TILE',str(nr.VALUE_TILE))
state_kernel=mx.fast.metal_kernel(name='gdn_norm_reuse_batch_state',input_names=['q_ready','k_ready','v','scales','g','beta','S'],output_names=['o','S_new'],source=state_source,header=nr.HEADER,compile_options=nr.COMPILE_OPTIONS)
def fused(raw,conv,w,g,beta,S):
    B=raw.shape[0]
    if not (1 <= B <= 8 and raw.shape == (B,10240) and conv.shape == (B,3,10240) and w.shape == (10240,4)
            and raw.dtype == conv.dtype == w.dtype == mx.bfloat16
            and g.shape == beta.shape == (B,48) and S.shape == (B,48,128,128)
            and g.dtype == beta.dtype == S.dtype == mx.float32):
        raise ValueError("fused GDN requires qualified BF16 batch shapes")
    if B == 1:
        q,k,v,scales,new_conv=single_kernel(inputs=[raw,conv,w],grid=(64*128,1,1),threadgroup=(128,1,1),
            output_shapes=[(1,16,128),(1,16,128),(1,48,128),(1,16,2),(1,3,10240)],
            output_dtypes=[mx.float32]*4+[mx.bfloat16])
        o,state=nr._state(q,k,v,scales,g,beta,S,mx)
        return o,state,new_conv
    q,k,v,scales,new_conv=batch_kernel(inputs=[raw,conv,w],grid=(64*128,B,1),threadgroup=(128,1,1),
       output_shapes=[(B,16,128),(B,16,128),(B,48,128),(B,16,2),(B,3,10240)],
       output_dtypes=[mx.float32]*4+[mx.bfloat16])
    o,state=state_kernel(inputs=[q,k,v,scales,g,beta,S],grid=(128,48*nr.SPLIT,B),threadgroup=(nr.VALUE_TILE,nr.SPLIT,1),output_shapes=[(B,48,128),(B,48,128,128)],output_dtypes=[mx.float32,mx.float32])
    return o,state,new_conv
