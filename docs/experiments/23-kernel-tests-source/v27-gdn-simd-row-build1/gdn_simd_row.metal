// Independently written compact-head, register-resident T1 GDN recurrence.
// Algorithm source: mlx-vlm/models/qwen3_5/gated_delta.py:35-102 (MIT).
// Keep this tutorial's normalize/decay/update expressions; SIMD sum order differs.
uint bh = thread_position_in_grid.z;
uint b = bh / 48;
uint h = bh % 48;
uint hk = h / 3;
uint dv = thread_position_in_grid.y;
uint lane = thread_index_in_simdgroup;
float qv[4], kvv[4], state[4];
float qn = 0.0f, kn = 0.0f;
for (uint i = 0; i < 4; ++i) {
    uint d = lane * 4 + i;
    qv[i] = float(q[b * q_strides[0] + hk * q_strides[1] + d * q_strides[2]]);
    kvv[i] = float(k[b * k_strides[0] + hk * k_strides[1] + d * k_strides[2]]);
    qn += qv[i] * qv[i];
    kn += kvv[i] * kvv[i];
    size_t si = size_t(b) * S_strides[0] + size_t(h) * S_strides[1]
              + size_t(d) * S_strides[2] + size_t(dv) * S_strides[3];
    state[i] = S[si];
}
qn = simd_sum(qn);
kn = simd_sum(kn);
float qs = metal::rsqrt(qn + 1e-6f) * metal::rsqrt(128.0f);
float ks = metal::rsqrt(kn + 1e-6f);
float ex = metal::exp(g[b * g_strides[0] + h * g_strides[1]]);
float be = beta[b * beta_strides[0] + h * beta_strides[1]];
float dot = 0.0f;
for (uint i = 0; i < 4; ++i) dot += ks * kvv[i] * state[i];
dot = simd_sum(dot);
float delta = (float(v[b * v_strides[0] + h * v_strides[1] + dv * v_strides[2]]) - ex * dot) * be;
float acc = 0.0f;
for (uint i = 0; i < 4; ++i) {
    uint d = lane * 4 + i;
    float next = ex * state[i] + ks * kvv[i] * delta;
    S_physical[(size_t(bh) * 128 + dv) * 128 + d] = next;
    acc += qs * qv[i] * next;
}
acc = simd_sum(acc);
if (lane == 0) o[size_t(bh) * 128 + dv] = acc;
