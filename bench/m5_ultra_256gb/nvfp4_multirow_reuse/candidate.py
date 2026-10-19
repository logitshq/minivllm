"""Default-OFF T2/4/8 NVFP4 projection with register reuse, not qualified.

No backend/model import or work at module import or build(enabled=False).
Explicit build creates a callable; Metal compilation/evaluation can be lazy.
"""
# Copyright © 2023-2026 Apple Inc. (adapted arithmetic)
# SPDX-License-Identifier: MIT
# Adapted from MLX 0.32.3 fp_qmv_fast_impl/qdot and the local B1 kernel.
# https://github.com/ml-explore/mlx/blob/v0.32.3/mlx/backend/metal/kernels/fp_quantized.h
# License: ../../../THIRD_PARTY_NOTICES/MLX.txt
HEADER = '''
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
SOURCE = '''
    const uint K = x_shape[1], N = scales_shape[0];
    const uint lane = thread_position_in_threadgroup.x;
    const uint sg = thread_position_in_threadgroup.y;
    const uint row = threadgroup_position_in_grid.y * 2 + sg;
    const device ushort* packed = reinterpret_cast<const device ushort*>(w);
    float result[ROWS] = {0};
    for (uint k = 0; k < K; k += 512) {
        float scale = f8(scales[row * (K / 16) + k / 16 + lane]);
        float accum[ROWS] = {0};
        for (uint j = 0; j < 4; ++j) {
            uint p = packed[row * (K / 4) + k / 4 + lane * 4 + j];
            float w0 = f4(p), w1 = f4(p >> 4);
            float w2 = f4(p >> 8), w3 = f4(p >> 12);
            for (uint t = 0; t < ROWS; ++t) {
                uint base = t * K + k + lane * 16 + 4 * j;
                accum[t] += (float(x[base]) * w0 + float(x[base+1]) * w1
                             + float(x[base+2]) * w2 + float(x[base+3]) * w3);
            }
        }
        for (uint t = 0; t < ROWS; ++t) result[t] += scale * accum[t];
    }
    for (uint t = 0; t < ROWS; ++t) {
        result[t] = simd_sum(result[t]);
        if (lane == 0) out[t * N + row] = DT(result[t]);
    }
'''


def supported(mx, x, w, scales):
    """Metadata only; stride handling belongs to ensure_row_contiguous=True."""
    shapes = [getattr(a, 'shape', ()) for a in (x, w, scales)]
    if not all(type(s) is tuple and len(s) == 2 for s in shapes):
        return False
    (rows, K), ws, ss = shapes
    if not all(type(v) is int and v > 0 for s in shapes for v in s):
        return False
    N = ss[0]
    return (rows in (2, 4, 8) and K % 512 == 0 and N % 4 == 0
            and N * K < (1 << 32) and rows * K < (1 << 32)
            and ws == (N, K // 8) and ss == (N, K // 16)
            and getattr(x, 'dtype', None) == mx.bfloat16
            and getattr(w, 'dtype', None) == mx.uint32
            and getattr(scales, 'dtype', None) == mx.uint8)


def build(mx, *, enabled=False):
    """Explicit opt-in only. Unsupported domains return None, never M=T QMM.

    The caller must pin actual source/runtime/model and qualify original bytes.
    Row-contiguous staging of x, w AND scales is delegated explicitly to MLX;
    any staging, allocation and lazy execution must be included in measurements.
    No eval, parameter packing, mutation, integration or qualification here.
    """
    if type(enabled) is not bool:
        raise TypeError('enabled must be bool')
    if not enabled:
        return None
    from importlib.metadata import PackageNotFoundError, version
    try:
        if version('mlx') != '0.32.3':
            return None
    except PackageNotFoundError:
        return None
    if (mx.default_device() != mx.gpu
            or mx.metal.device_info().get('architecture') != 'applegpu_g17d'
            or mx.device_info().get('device_name') != 'Apple M5 Ultra'):
        return None
    kernel = mx.fast.metal_kernel(
        name='m5_nvfp4_multirow_reuse_r1_v1',
        input_names=['x', 'w', 'scales'], output_names=['out'],
        source=SOURCE, header=HEADER, ensure_row_contiguous=True,
        compile_options={'math_mode': 'safe'})

    def project(x, w, scales):
        if mx.default_device() != mx.gpu or not supported(mx, x, w, scales):
            return None
        return kernel(inputs=[x, w, scales], template=[('DT', x.dtype), ('ROWS', x.shape[0])],
                      grid=(32, scales.shape[0], 1), threadgroup=(32, 2, 1),
                      output_shapes=[(x.shape[0], scales.shape[0])],
                      output_dtypes=[x.dtype])[0]
    return project
