// Two native R4 reductions share x loads, then a typed BF16 epilogue.
// Copyright © 2025-2026 Apple Inc. MIT; see LICENSE-MLX.txt.
const uint local = thread_position_in_threadgroup.x;
const uint simd_gid = local / 32;
const uint simd_lid = local % 32;
const uint out_row = threadgroup_position_in_grid.y * 8 + simd_gid * 4;
constexpr int values_per_thread = 16;
constexpr int block_size = 512;
typedef float U;
thread U x_thread[values_per_thread];
thread U gate_result[4] = {0};
thread U up_result[4] = {0};
const device uint8_t* wg = (const device uint8_t*)gate_weight + out_row * (K / 2) + simd_lid * 8;
const device uint8_t* wu = (const device uint8_t*)up_weight + out_row * (K / 2) + simd_lid * 8;
const device uint8_t* sg = gate_scales + out_row * (K / 16) + simd_lid;
const device uint8_t* su = up_scales + out_row * (K / 16) + simd_lid;
const device T* xp = x + simd_lid * values_per_thread;
for (int k = 0; k < K; k += block_size) {
  load_vector<T, U, values_per_thread>(xp, x_thread);
  for (int row = 0; row < 4; row++) {
    auto gw = (const device uint8_t*)(wg + row * (K / 2));
    const device auto* gs = sg + row * (K / 16);
    U gate_scale = dequantize_scale<U, 16>(gs[0]);
    gate_result[row] += qdot<U, values_per_thread, 4>(gw, x_thread, gate_scale);
    auto uw = (const device uint8_t*)(wu + row * (K / 2));
    const device auto* us = su + row * (K / 16);
    U up_scale = dequantize_scale<U, 16>(us[0]);
    up_result[row] += qdot<U, values_per_thread, 4>(uw, x_thread, up_scale);
  }
  wg += block_size / 2; wu += block_size / 2;
  sg += block_size / 16; su += block_size / 16;
  xp += block_size;
}
for (int row = 0; row < 4; row++) {
  gate_result[row] = simd_sum(gate_result[row]);
  up_result[row] = simd_sum(up_result[row]);
  if (simd_lid == 0) {
    T gate_value = static_cast<T>(gate_result[row]);
    T up_value = static_cast<T>(up_result[row]);
    output[out_row + row] = v28_swiglu<T>(gate_value, up_value);
#if V28_DEBUG_OUT
    gate_output[out_row + row] = gate_value;
    up_output[out_row + row] = up_value;
#endif
  }
}
