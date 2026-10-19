"""Default-OFF native-M8 arithmetic hypothesis with parallel shared-weight loads.

No backend/model work at import or while disabled. CPU mechanisms only;
exact native M8 output bits and speed must be established on the real device.
The oracle is one native quantized_matmul(X[8,K],W,S), NEVER independent M1.
"""
# Copyright © 2025-2026 Apple Inc. (adapted arithmetic)
# SPDX-License-Identifier: MIT
# MLX v0.32.3 fp_qmv_wide_impl, fp4_e2m1 and fp8_e4m3 conversions.
# https://github.com/ml-explore/mlx/blob/v0.32.3/mlx/backend/metal/kernels/fp_quantized.h
# License: ../../../THIRD_PARTY_NOTICES/MLX.txt; local source pins: references.json.
# Native wide source has no contract(off) pragma; do not borrow the B1 policy.
HEADER = '''
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
    const uint N = scales_shape[0];
    const uint lane = thread_position_in_threadgroup.x;
    const uint sg = thread_position_in_threadgroup.y;
    const uint k_lane = lane % 16;
    const uint local_row = (sg % 2) * 2 + lane / 16;
    const uint token = sg / 2;
    const uint first_row = threadgroup_position_in_grid.y * 4;
    const uint tid = sg * 32 + lane;
    threadgroup uint shared_w[2560];
    threadgroup uchar shared_s[1280];
    // Unique coalesced copies: 4 output rows, shared across all 8 input rows.
    for (uint i = tid; i < 2560; i += 512)
        shared_w[i] = w[(first_row + i / 640) * 640 + i % 640];
    for (uint i = tid; i < 1280; i += 512)
        shared_s[i] = scales[(first_row + i / 320) * 320 + i % 320];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const threadgroup ushort* packed = reinterpret_cast<const threadgroup ushort*>(shared_w);
    float result = 0;
    for (uint g = k_lane; g < 320; g += 16) {
        const uint k0 = g * 16;
        float s = f8(shared_s[local_row * 320 + g]);
        float4 w4[4];
#pragma unroll
        for (uint i = 0; i < 4; ++i) {
            uint p = packed[local_row * 1280 + g * 4 + i];
            w4[i] = float4(f4(p), f4(p >> 4), f4(p >> 8), f4(p >> 12));
        }
        const device vec<DT, 4>* xv4 = (const device vec<DT, 4>*)(x + token * 5120 + k0);
        float acc = 0;
#pragma unroll
        for (uint j = 0; j < 4; ++j) {
            acc += dot(w4[j], float4(xv4[j]));
        }
        result += s * acc;
    }
    // Same kl16 shuffle ladder as native qmv_wide. Only half-SIMD leaders store.
    result += simd_shuffle_down(result, 8);
    result += simd_shuffle_down(result, 4);
    result += simd_shuffle_down(result, 2);
    result += simd_shuffle_down(result, 1);
    if (k_lane == 0) out[token * N + first_row + local_row] = DT(result);
'''


def supported(mx, x, w, scales):
    shapes = [getattr(a, 'shape', ()) for a in (x, w, scales)]
    if not all(type(s) is tuple and len(s) == 2 for s in shapes):
        return False
    if not all(type(v) is int and v > 0 for s in shapes for v in s):
        return False
    N = shapes[2][0]
    return (shapes == [(8,5120),(N,640),(N,320)] and N % 4 == 0
            and N * 5120 < (1 << 32)
            and getattr(x, 'dtype', None) == mx.bfloat16
            and getattr(w, 'dtype', None) == mx.uint32
            and getattr(scales, 'dtype', None) == mx.uint8)


def build(mx, *, enabled=False):
    """Fixed M8/K5120 NVFP4 only; unsupported explicit opt-in returns None.

    No fallback/retry, eval, packing or parameter mutation. Contiguous staging
    remains explicit in MLX and belongs inside any eventual timing boundary.
    Safe math plus native vector-dot source is a hypothesis, not a binary proof.
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
        name='m5_nvfp4_m8_shared_wide_v1', input_names=['x','w','scales'],
        output_names=['out'], source=SOURCE, header=HEADER,
        ensure_row_contiguous=True, compile_options={'math_mode':'safe'})

    def project(x, w, scales):
        if mx.default_device() != mx.gpu or not supported(mx,x,w,scales):
            return None
        N = scales.shape[0]
        return kernel(inputs=[x,w,scales], template=[('DT',x.dtype)],
                      grid=(32,N*4,1), threadgroup=(32,16,1),
                      output_shapes=[(8,N)], output_dtypes=[x.dtype])[0]
    return project
