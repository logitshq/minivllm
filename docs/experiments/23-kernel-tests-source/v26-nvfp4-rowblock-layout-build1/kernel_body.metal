// Address-only variant of MLX0.32.3 fp_qmv_fast_impl<T,16,4,false,4>.
// Copyright © 2025-2026 Apple Inc. MIT; see LICENSE-MLX.txt.
// w: [N/8,K/512,8,64] U32; scales: [N/8,K/512,8,32] U8.
const uint local = thread_position_in_threadgroup.x;
const uint simd_gid = local / 32;
const uint simd_lid = local % 32;
const uint block = threadgroup_position_in_grid.y;
constexpr int values_per_thread = 16;
constexpr int block_size = 512;
constexpr int results_per_simdgroup = 4;
typedef float U;
thread U x_thread[values_per_thread];
thread U result[results_per_simdgroup] = {0};
const device uint8_t* ws = (const device uint8_t*)weight;
const device T* xp = x + simd_lid * values_per_thread;
device T* yp = output + block * 8 + simd_gid * 4;
for (int k = 0; k < K; k += block_size) {
  load_vector<T, U, values_per_thread>(xp, x_thread);
  for (int row = 0; row < results_per_simdgroup; row++) {
    const uint slab_row = simd_gid * 4 + row;
    const ulong tile = (ulong(block) * (K / block_size) + k / block_size) * 8 + slab_row;
    auto wl = (const device uint8_t*)(ws + tile * 256 + simd_lid * 8);
    const device auto* sl = scales + tile * 32 + simd_lid;
    U s = dequantize_scale<U, 16>(sl[0]);
    result[row] += qdot<U, values_per_thread, 4>(wl, x_thread, s);
  }
  xp += block_size;
}
for (int row = 0; row < results_per_simdgroup; row++) {
  result[row] = simd_sum(result[row]);
  if (simd_lid == 0) yp[row] = static_cast<T>(result[row]);
}
