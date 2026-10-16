"""Pinned B1 NVFP4 QMV: one output row per SIMD group, original reduction order.

Arithmetic adapted from MLX 0.32.3 fp_qmv_fast_impl (MIT), Apple Inc.
https://github.com/ml-explore/mlx/blob/v0.32.3/mlx/backend/metal/kernels/fp_quantized.h
See THIRD_PARTY_NOTICES/MLX.txt. FP contraction MUST remain disabled: contraction
changes a vocabulary-head result even when both results round to BF16.
"""
# Copyright © 2025-2026 Apple Inc. (adapted arithmetic)
# SPDX-License-Identifier: MIT
import mlx.core as mx
HEADER='''
#pragma clang fp contract(off)
float f4(uint bits) {
    half v = as_type<half>(ushort((bits & 7) << 9));
    v *= 16384.0;
    return float(bits & 8 ? -v : v);
}
float f8(uchar bits) {
    ushort v = bits & 127;
    ushort sign = ushort((bits >> 7) & 1) << 15;
    ushort u = (v << 7) | (((v + 1) >> 7) << 14) | sign;
    half converted = as_type<half>(u);
    return float(half(converted * 256.0));
}
'''
SOURCE='''
    const uint K = x_shape[1], N = scales_shape[0];
    const uint lane = thread_position_in_threadgroup.x;
    const uint sg = thread_position_in_threadgroup.y;
    const uint row0 = threadgroup_position_in_grid.y * 2 * R + sg * R;
    const device ushort* packed = reinterpret_cast<const device ushort*>(w);
    float result[R] = {0};
    for (uint k = 0; k < K; k += 512) {
        float xx[16];
        for (uint i = 0; i < 16; ++i) xx[i] = float(x[k + lane * 16 + i]);
        for (uint r = 0; r < R; ++r) {
            uint row = row0 + r;
            float scale = f8(scales[row * (K / 16) + k / 16 + lane]);
            float accum = 0;
            for (uint j = 0; j < 4; ++j) {
                uint p = packed[row * (K / 4) + k / 4 + lane * 4 + j];
                accum += (xx[4*j] * f4(p) + xx[4*j+1] * f4(p >> 4)
                         + xx[4*j+2] * f4(p >> 8) + xx[4*j+3] * f4(p >> 12));
            }
            result[r] += scale * accum;
        }
    }
    for (uint r = 0; r < R; ++r) {
        result[r] = simd_sum(result[r]);
        if (lane == 0) out[row0 + r] = T(result[r]);
    }
'''
kernel=mx.fast.metal_kernel(name='m5_nvfp4_qmv_r1_exact',input_names=['x','w','scales'],output_names=['out'],source=SOURCE,header=HEADER,compile_options={'math_mode':'safe'})
def supported(x, w, scales):
    return (x.ndim == 2 and x.shape[0] == 1 and x.dtype == mx.bfloat16
            and x.shape[1] % 512 == 0 and scales.ndim == 2 and scales.shape[0] % 4 == 0
            and w.shape == (scales.shape[0], x.shape[1] // 8) and w.dtype == mx.uint32
            and scales.shape[1] == x.shape[1] // 16 and scales.dtype == mx.uint8)


def qmv(x, w, scales):
    if not supported(x, w, scales):
        return mx.quantized_matmul(x, w, scales, transpose=True, group_size=16, bits=4, mode='nvfp4')
    N = scales.shape[0]
    return kernel(inputs=[x, w, scales], template=[('T', x.dtype), ('R', 1)],
                  grid=(32, N, 1), threadgroup=(32, 2, 1),
                  output_shapes=[(1, N)], output_dtypes=[x.dtype])[0]
