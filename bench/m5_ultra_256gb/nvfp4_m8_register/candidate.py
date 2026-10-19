"""Default-OFF native wide helper instantiated at nv8, not two nv4 tiles.

One 64-thread group decodes each weight group once for eight accumulators.
No shared memory, barrier, repack, eval or fallback. Native M8 remains the oracle;
CPU/source checks do not establish actual GPU equivalence or performance.
"""
# Copyright © 2025-2026 Apple Inc. (retained Metal helpers)
# SPDX-License-Identifier: MIT; see ../../../THIRD_PARTY_NOTICES/MLX.txt.
# https://github.com/ml-explore/mlx/tree/v0.32.3/mlx/backend/metal/kernels
HEADER_ROOT = '/Users/ryanli/code/minivllm/.venv/lib/python3.13/site-packages/mlx/include/mlx/backend/metal/kernels'
HEADER_PINS = {
    'fp_quantized.h':'359a1ea9008f74cc15c61e9d39c6300dcffb9cfb9cc0ebd59ec7c00b789d719f',
    'fp4.h':'e3159945d7e5ae24f7a1ca9c4875b2a5d15cfb6952405df24a0df1bb2622a93a',
    'fp8.h':'9a9d77a6295f66b4ada4311c3b855b6ecb7a2b38f28623109e3c61d2e17e81ae',
}
SOURCE = """
fp_qmv_wide_impl<DT, 16, 4, 8, 16>(
    w, scales, x, out, x_shape[1], scales_shape[0], x_shape[0],
    threadgroup_position_in_grid, simdgroup_index_in_threadgroup,
    thread_index_in_simdgroup);
"""


def native_header():
    """Pinned installed conversion types and helper bodies, copied verbatim."""
    import hashlib
    from pathlib import Path
    texts = {}
    for name, expected in HEADER_PINS.items():
        raw = (Path(HEADER_ROOT)/name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError('native header SHA mismatch: '+name)
        texts[name] = raw.decode()
    q = texts['fp_quantized.h']
    start = q.index('template <int wsize = 8, int bits = 4>')
    end = q.index('template <typename T, typename U, int values_per_thread>')
    wide = q.index('template <typename T, int group_size, int bits, int vecs_per_tg, int k_lanes>')
    stop = q.index('template <typename T, int group_size, int bits, bool has_global_scale = false>',wide)
    preamble = ('#include <metal_stdlib>\n#include <metal_simdgroup>\nusing namespace metal;\n'
                '#ifndef METAL_FUNC\n#define METAL_FUNC inline\n#endif\n'
                'static constant constexpr const int SIMD_SIZE = 32;\n')
    return preamble+texts['fp4.h']+texts['fp8.h']+q[start:end]+q[wide:stop]


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
        name='m5_nvfp4_m8_register_wide_v1', input_names=['x','w','scales'],
        output_names=['out'], source=SOURCE, header=native_header(),
        ensure_row_contiguous=True, compile_options={'math_mode':'safe'})

    def project(x, w, scales):
        if mx.default_device() != mx.gpu or not supported(mx,x,w,scales):
            return None
        N = scales.shape[0]
        return kernel(inputs=[x,w,scales], template=[('DT',x.dtype)],
                      grid=(32,N//2,1), threadgroup=(32,2,1),
                      output_shapes=[(8,N)], output_dtypes=[x.dtype])[0]
    return project
