"""CPU-prepared, unqualified residual + chapter13 norm; no engine integration.

Import is inert. Calling the operator lazily imports pinned MLX and may use GPU.
Same row-contiguous preparation as the old norm is included in the dispatch.
No value scan: NaN/inf use the original typed Add and unchanged norm expressions;
raw nonfinite propagation, including payload bits, is still GPU-UNVERIFIED.
"""
import math

SOURCE = r'''
    uint tid = thread_position_in_threadgroup.x;
    uint row = threadgroup_position_in_grid.y;
    uint D = left_shape[left_ndim - 1];
    uint lane = tid % 32;
    uint sg = tid / 32;
    uint nsg = TG / 32;
    threadgroup float sums[32];
    float part = 0.0f;
    for (uint first = tid * 4; first < D; first += TG * 4) {
        for (uint j = 0; j < 4 && first + j < D; j++) {
            T residual_value = left[(size_t)row * D + first + j] + delta[(size_t)row * D + first + j];
            float xf = float(residual_value);
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
        T residual_value = left[(size_t)row * D + i] + delta[(size_t)row * D + i];
        residual[(size_t)row * D + i] = residual_value;
        float xf = float(residual_value);
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
HEADER = '#pragma clang fp contract(off)\n'
_KERNEL = None


def _runtime():
    from importlib.metadata import version
    if version('mlx') != '0.32.3':
        raise RuntimeError('residual_norm requires pinned MLX 0.32.3')
    import mlx.core as mx
    return mx


def _host_geometry(value, shape, name):
    actual = getattr(value, 'shape', None)
    if (type(actual) is not tuple or actual != shape
            or any(type(d) is not int for d in actual)
            or type(getattr(value, 'ndim', None)) is not int
            or value.ndim != len(shape)
            or type(getattr(value, 'size', None)) is not int
            or value.size != 5120):
        raise ValueError(f'{name} requires exact shape {shape} and known host ndim/size')


def residual_norm(left, delta, weight, eps):
    """Return independently owned (BF16 residual, zero-centered BF16 norm).

    Only BF16 [1,5120] operands and BF16/FP32 weight[5120], Python float eps.
    Invalid metadata raises; no silent fallback or nonfinite-value eligibility.
    Physical strides are not publicly exposed by pinned MLX: the same Metal
    wrapper row-contiguous preparation as old norm handles layout, at its cost.
    """
    _host_geometry(left, (1, 5120), 'left')
    _host_geometry(delta, (1, 5120), 'delta')
    _host_geometry(weight, (5120,), 'weight')
    if type(eps) is not float or not math.isfinite(eps) or not 0.0 < eps < 1.0:
        raise ValueError('eps requires a finite Python float with 0 < eps < 1')
    mx = _runtime()
    if (getattr(left, 'dtype', None) != mx.bfloat16
            or getattr(delta, 'dtype', None) != mx.bfloat16
            or getattr(weight, 'dtype', None) not in (mx.bfloat16, mx.float32)):
        raise ValueError('requires BF16 operands and BF16/FP32 weight')
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = mx.fast.metal_kernel(
            name='residual_explicit_norm_row_build1',
            input_names=['left', 'delta', 'weight', 'gate', 'eps'],
            output_names=['residual', 'output'], source=SOURCE, header=HEADER,
            ensure_row_contiguous=True,
        )
    D = left.shape[-1]
    tg = 32 if D<=512 else (128 if D<=1024 else min(1024,((D+3)//4+31)//32*32))
    # Fresh scalar input, never a weight/activation capture or value-based cache.
    eps_value = mx.array([eps], mx.float32)
    return tuple(_KERNEL(
        inputs=[left, delta, weight, left, eps_value],
        template=[('T', left.dtype), ('W', weight.dtype), ('TG', tg),
                  ('ZERO_CENTERED', True), ('GATED', False)],
        grid=(tg, 1, 1), threadgroup=(tg, 1, 1),
        output_shapes=[left.shape, left.shape], output_dtypes=[left.dtype, left.dtype],
    ))
