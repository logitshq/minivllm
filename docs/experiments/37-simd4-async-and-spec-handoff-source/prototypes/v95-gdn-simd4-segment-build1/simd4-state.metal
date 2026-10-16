
    uint lane = thread_index_in_simdgroup;
    uint dv_in_simd = lane % 8;
    uint dv_local = thread_position_in_threadgroup.y * 8 + dv_in_simd;
    uint dv = threadgroup_position_in_grid.x * VALUE_TILE + dv_local;
    uint sp = lane / 8;
    uint h  = threadgroup_position_in_grid.y;
    uint b  = threadgroup_position_in_grid.z;
    uint Hv = S_shape[1];
    uint Dk = S_shape[2];
    uint Dv = S_shape[3];
    uint bh = b * Hv + h;
    uint seg = Dk / SPLIT;
    uint d0 = sp * seg;
    size_t base = (size_t)bh * Dk * Dv;
    const device float* qh = q_ready + (size_t)(h / 3) * Dk;
    const device float* kh = k_ready + (size_t)(h / 3) * Dk;
    float qs = scales[(h / 3) * 2];
    float ks = scales[(h / 3) * 2 + 1];
    float ex = metal::exp(g[bh]);
    float be = beta[bh];
    float kv = 0.0f;
    float held[32];
    #pragma unroll 32
    for (uint j = 0; j < 32; j++) {
        uint d = d0 + j;
        held[j] = S[base + d * Dv + dv];
        kv += ks * kh[d] * held[j];
    }
    float kv_part = kv;
    kv = 0.0f;
    for (uint i = 0; i < SPLIT; i++) kv += simd_shuffle(kv_part, dv_in_simd + i * 8);
    float delta = (v[(size_t)bh * Dv + dv] - ex * kv) * be;
    float acc = 0.0f;
    #pragma unroll 32
    for (uint j = 0; j < 32; j++) {
        uint d = d0 + j;
        float s = ex * held[j] + ks * kh[d] * delta;
        S_new[base + d * Dv + dv] = s;
        acc += qs * qh[d] * s;
    }
    float acc_part = acc;
    acc = 0.0f;
    for (uint i = 0; i < SPLIT; i++) acc += simd_shuffle(acc_part, dv_in_simd + i * 8);
    if (sp == 0) {
        o[(size_t)bh * Dv + dv] = acc;
    }
