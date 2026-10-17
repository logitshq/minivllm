// V124: cached packed words; arithmetic expression is the pinned M1 bits4 branch.
inline float v124_qdot_cached(
    const thread uint16_t* ws, const thread float* x_thread, float scale) {
  float accum = 0;
  for (int i = 0; i < (16 / 4); i++) {
    accum +=
        (x_thread[4 * i] * Dequantize<4>{}(ws[i]) +
         x_thread[4 * i + 1] * Dequantize<4>{}(ws[i] >> 4) +
         x_thread[4 * i + 2] * Dequantize<4>{}(ws[i] >> 8) +
         x_thread[4 * i + 3] * Dequantize<4>{}(ws[i] >> 12));
  }
  return scale * accum;
}

template <typename T, int VERIFY_T, int K_SIZE, int N_SIZE>
METAL_FUNC void v124_shared_m1(
    const device uint32_t* weight, const device uint8_t* scales,
    const device T* x, device T* output,
    uint tile, uint simd_gid, uint lane) {
  static_assert(VERIFY_T == 2 || VERIFY_T == 4 || VERIFY_T == 5);
  static_assert(K_SIZE % 512 == 0 && N_SIZE % 8 == 0);
  const int out_row = int(tile) * 8 + int(simd_gid) * 4;
  const device uint8_t* ws = (const device uint8_t*)weight +
      out_row * (K_SIZE / 2) + int(lane) * 8;
  const device uint8_t* sc = scales + out_row * (K_SIZE / 16) + int(lane);
  thread float result[VERIFY_T][4] = {0};
  for (int k = 0; k < K_SIZE; k += 512) {
    for (int row = 0; row < 4; row++) {
      const device uint16_t* wl =
          (const device uint16_t*)(ws + row * (K_SIZE / 2));
      thread uint16_t cached[4];
      for (int i = 0; i < 4; i++) cached[i] = wl[i];
      float scale = dequantize_scale<float, 16>(sc[row * (K_SIZE / 16)]);
      // Only this row's 16 activation values are live. This re-reads small X
      // across the four output rows instead of keeping VERIFY_T*16 float Xs.
      for (int token = 0; token < VERIFY_T; token++) {
        thread float x_thread[16];
        load_vector<T, float, 16>(
            x + token * K_SIZE + k + int(lane) * 16, x_thread);
        result[token][row] += v124_qdot_cached(cached, x_thread, scale);
      }
    }
    ws += 256;
    sc += 32;
  }
  for (int row = 0; row < 4; row++) {
    for (int token = 0; token < VERIFY_T; token++) {
      result[token][row] = simd_sum(result[token][row]);
      if (lane == 0) {
        output[token * N_SIZE + out_row + row] = static_cast<T>(result[token][row]);
      }
    }
  }
}
