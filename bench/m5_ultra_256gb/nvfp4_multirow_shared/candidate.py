"""Default-OFF K5120 NVFP4 prototype: parallel token SIMD groups share weight loads.

No backend/model import or work until explicit build(enabled=True). This source
has no device/numerical/performance qualification. Unsupported input returns None.
"""
# Copyright © 2023-2026 Apple Inc. (adapted arithmetic)
# SPDX-License-Identifier: MIT
# Arithmetic adapted from MLX 0.32.3 fp_qmv_fast_impl/qdot and local B1 kernel.
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
    const uint K = 5120, N = scales_shape[0];
    const uint lane = thread_position_in_threadgroup.x;
    const uint sg = thread_position_in_threadgroup.y;
    const uint local_row = sg % 2, token = sg / 2;
    const uint first_row = threadgroup_position_in_grid.y * 2;
    const uint tid = sg * 32 + lane, threads = ROWS * 64;
    threadgroup uint shared_w[1280];
    threadgroup uchar shared_s[640];
    // Adjacent linear threads copy adjacent U32 words/U8 scales. Each item has
    // exactly one writer, and every thread reaches the same single barrier.
    for (uint i = tid; i < 1280; i += threads)
        shared_w[i] = w[(first_row + i / 640) * 640 + i % 640];
    for (uint i = tid; i < 640; i += threads)
        shared_s[i] = scales[(first_row + i / 320) * 320 + i % 320];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const threadgroup ushort* packed = reinterpret_cast<const threadgroup ushort*>(shared_w);
    float result = 0;
    for (uint k = 0; k < K; k += 512) {
        float scale = f8(shared_s[local_row * 320 + k / 16 + lane]);
        float accum = 0;
        for (uint j = 0; j < 4; ++j) {
            uint p = packed[local_row * 1280 + k / 4 + lane * 4 + j];
            float w0 = f4(p), w1 = f4(p >> 4);
            float w2 = f4(p >> 8), w3 = f4(p >> 12);
            uint base = token * K + k + lane * 16 + 4 * j;
            accum += (float(x[base]) * w0 + float(x[base+1]) * w1
                      + float(x[base+2]) * w2 + float(x[base+3]) * w3);
        }
        result += scale * accum;
    }
    result = simd_sum(result);
    if (lane == 0) out[token * N + first_row + local_row] = DT(result);
'''


def supported(mx, x, w, scales):
    shapes = [getattr(a, 'shape', ()) for a in (x, w, scales)]
    if not all(type(s) is tuple and len(s) == 2 for s in shapes):
        return False
    (rows, K), ws, ss = shapes
    if not all(type(v) is int and v > 0 for s in shapes for v in s):
        return False
    N = ss[0]
    return (rows in (2, 4, 8) and K == 5120 and N % 4 == 0
            and N * K < (1 << 32) and rows * K < (1 << 32)
            and ws == (N, 640) and ss == (N, 320)
            and getattr(x, 'dtype', None) == mx.bfloat16
            and getattr(w, 'dtype', None) == mx.uint32
            and getattr(scales, 'dtype', None) == mx.uint8)


def build(mx, *, enabled=False):
    """Explicit fixed-device opt-in. No eval/packing/mutation or M=T QMM fallback.

    Each threadgroup owns two output columns and 2*ROWS SIMD groups, one per
    (token,column). Its 5,760-byte shared tile is read-only after one barrier.
    All layout staging through ensure_row_contiguous is part of later timing.
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
        name='m5_nvfp4_multirow_shared_v1', input_names=['x', 'w', 'scales'],
        output_names=['out'], source=SOURCE, header=HEADER,
        ensure_row_contiguous=True, compile_options={'math_mode': 'safe'})

    def project(x, w, scales):
        if mx.default_device() != mx.gpu or not supported(mx, x, w, scales):
            return None
        T, N = x.shape[0], scales.shape[0]
        return kernel(inputs=[x, w, scales], template=[('DT', x.dtype), ('ROWS', T)],
                      grid=(32, N * T, 1), threadgroup=(32, 2 * T, 1),
                      output_shapes=[(T, N)], output_dtypes=[x.dtype])[0]
    return project
