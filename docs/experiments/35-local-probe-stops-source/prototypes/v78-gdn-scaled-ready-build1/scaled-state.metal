
    uint dv_local = thread_position_in_threadgroup.x;
    uint dv = threadgroup_position_in_grid.x * VALUE_TILE + dv_local;
    uint sp = thread_position_in_threadgroup.y;
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
    threadgroup float red[SPLIT * VALUE_TILE];
    float ex = metal::exp(g[bh]);
    float be = beta[bh];
    float kv = 0.0f;
    float held[32];
    #pragma unroll 32
    for (uint j = 0; j < 32; j++) {
        uint d = d0 + j;
        held[j] = S[base + d * Dv + dv];
        kv += kh[d] * held[j];
    }
    red[sp * VALUE_TILE + dv_local] = kv;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    kv = 0.0f;
    for (uint i = 0; i < SPLIT; i++) kv += red[i * VALUE_TILE + dv_local];
    float delta = (v[(size_t)bh * Dv + dv] - ex * kv) * be;
    float acc = 0.0f;
    #pragma unroll 32
    for (uint j = 0; j < 32; j++) {
        uint d = d0 + j;
        float s = ex * held[j] + kh[d] * delta;
        S_new[base + d * Dv + dv] = s;
        acc += qh[d] * s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    red[sp * VALUE_TILE + dv_local] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sp == 0) {
        acc = 0.0f;
        for (uint i = 0; i < SPLIT; i++) acc += red[i * VALUE_TILE + dv_local];
        o[(size_t)bh * Dv + dv] = acc;
    }
