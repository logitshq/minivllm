"""Internal rank>=2 norm used after layer shape guards.

Independently implemented to match MLX v0.32.3 (MIT) arithmetic and row
reduction: https://github.com/ml-explore/mlx/blob/v0.32.3/mlx/backend/metal/reduce.cpp
and https://github.com/ml-explore/mlx/blob/v0.32.3/mlx/backend/metal/kernels/reduction/reduce_row.h .
Sigmoid follows the pinned unary_ops.h formula. Layer wrappers preserve rank1
and broadcast-gate behavior through the original compiled explicit fallback.
"""
import mlx.core as mx

SOURCE=r'''
    uint tid = thread_position_in_threadgroup.x;
    uint row = threadgroup_position_in_grid.y;
    uint D = x_shape[x_ndim - 1];
    uint lane = tid % 32;
    uint sg = tid / 32;
    uint nsg = TG / 32;
    threadgroup float sums[32];
    float part = 0.0f;
    for (uint first = tid * 4; first < D; first += TG * 4) {
        for (uint j = 0; j < 4 && first + j < D; j++) {
            float xf = float(x[(size_t)row * D + first + j]);
            float sq = xf * xf;
            part = part + sq;
        }
    }
    float total = metal::simd_sum(part);
    if (nsg > 1) {
        if (lane == 0) sums[sg] = total;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0) {
            float group_part = lane < nsg ? sums[lane] : 0.0f;
            float final_sum = metal::simd_sum(group_part);
            if (lane == 0) sums[0] = final_sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        total = sums[0];
    }
    float mean = total * (1.0f / float(D));
    float inv = metal::precise::rsqrt(mean + eps[0]);
    for (uint i = tid; i < D; i += TG) {
        float xf = float(x[(size_t)row * D + i]);
        float wf = float(weight[i]);
        if (ZERO_CENTERED) wf = float(W(1.0f + wf));
        float normalized = xf * inv;
        T weighted = T(normalized * wf);
        if (GATED) {
            float value = float(gate[(size_t)row * D + i]);
            float exp_value = metal::precise::exp(metal::abs(value));
            float sig = 1.0f / (1.0f + exp_value);
            sig = value < 0.0f ? sig : 1.0f - sig;
            T activated = T(value * sig);
            weighted = T(float(weighted) * float(activated));
        }
        output[(size_t)row * D + i] = weighted;
    }
'''
kernel=mx.fast.metal_kernel(name='explicit_norm_row',input_names=['x','weight','gate','eps'],output_names=['output'],source=SOURCE,header='#pragma clang fp contract(off)\n')
EPS_VALUES={}
def norm(x,weight,eps=1e-6,zero_centered=False,gate=None):
 D=x.shape[-1];rows=x.size//D
 tg=32 if D<=512 else (128 if D<=1024 else min(1024,((D+3)//4+31)//32*32))
 gated=gate is not None
 if eps not in EPS_VALUES:EPS_VALUES[eps]=mx.array([eps],mx.float32)
 return kernel(inputs=[x,weight,gate if gated else x,EPS_VALUES[eps]],template=[('T',x.dtype),('W',weight.dtype),('TG',tg),('ZERO_CENTERED',zero_centered),('GATED',gated)],grid=(tg,rows,1),threadgroup=(tg,1,1),output_shapes=[x.shape],output_dtypes=[x.dtype])[0]
