// Independent register-state SIMD variant, preserving tutorial FP32 loop order.
// Geometry/register-layout source: mlx-vlm qwen3_5/gated_delta.py:35-102 (MIT).
// Every shuffle is outside lane-conditional control and has a source in 0..31.
uint bh = thread_position_in_grid.z;
uint b = bh / 48;
uint h = bh % 48;
uint hk = h / 3;
uint dv = thread_position_in_grid.y;
uint lane = thread_index_in_simdgroup;
uint Dk = S_shape[2];
uint seg_size = Dk / 4;
float qv[4], kvv[4], state[4];
for (uint i = 0; i < 4; ++i) {
    uint d = lane * 4 + i;
    qv[i] = float(q[b * q_strides[0] + hk * q_strides[1] + d * q_strides[2]]);
    kvv[i] = float(k[b * k_strides[0] + hk * k_strides[1] + d * k_strides[2]]);
    size_t si = size_t(b) * S_strides[0] + size_t(h) * S_strides[1]
              + size_t(d) * S_strides[2] + size_t(dv) * S_strides[3];
    state[i] = S[si];
}
float qs = 0.0f, ks = 0.0f;
if (lane == 0) {
    float qn = 0.0f, kn = 0.0f;
    for (uint d = 0; d < Dk; ++d) {
        float qd = float(q[b * q_strides[0] + hk * q_strides[1] + d * q_strides[2]]);
        float kd = float(k[b * k_strides[0] + hk * k_strides[1] + d * k_strides[2]]);
        qn += qd * qd;
        kn += kd * kd;
    }
    qs = metal::rsqrt(qn + 1e-6f) * metal::rsqrt((float)Dk);
    ks = metal::rsqrt(kn + 1e-6f);
}
qs = simd_shuffle(qs, 0);
ks = simd_shuffle(ks, 0);
float ex = metal::exp(g[b * g_strides[0] + h * g_strides[1]]);
float be = beta[b * beta_strides[0] + h * beta_strides[1]];
float part = 0.0f;
for (uint offset = 0; offset < seg_size; ++offset) {
    uint owner = (lane < 4 ? lane : 0) * 8 + offset / 4;
    // Shuffle RAW operands, not a pre-rounded product, to preserve FMA policy.
    float key = simd_shuffle(kvv[offset % 4], owner);
    float old_state = simd_shuffle(state[offset % 4], owner);
    if (lane < 4) part += ks * key * old_state;
}
float dot = 0.0f;
for (uint seg = 0; seg < 4; ++seg) dot += simd_shuffle(part, seg);
float delta = (float(v[b * v_strides[0] + h * v_strides[1] + dv * v_strides[2]]) - ex * dot) * be;
for (uint i = 0; i < 4; ++i) {
    uint d = lane * 4 + i;
    float next = ex * state[i] + ks * kvv[i] * delta;
    state[i] = next;
    S_physical[(size_t(bh) * 128 + dv) * 128 + d] = next;
}
part = 0.0f;
for (uint offset = 0; offset < seg_size; ++offset) {
    uint owner = (lane < 4 ? lane : 0) * 8 + offset / 4;
    float query = simd_shuffle(qv[offset % 4], owner);
    float next_state = simd_shuffle(state[offset % 4], owner);
    if (lane < 4) part += qs * query * next_state;
}
float acc = 0.0f;
for (uint seg = 0; seg < 4; ++seg) acc += simd_shuffle(part, seg);
if (lane == 0) o[size_t(bh) * 128 + dv] = acc;
