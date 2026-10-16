
    uint lane = thread_position_in_threadgroup.x;
    uint group = threadgroup_position_in_grid.x;
    threadgroup float q_storage[128];
    threadgroup float k_storage[128];
    if (group < 16) {
        uint h = group;
        uint Dk = 128;
        const threadgroup float* qh = q_storage;
        const threadgroup float* kh = k_storage;
        q_storage[lane] = float(activated[h * 128 + lane]);
        k_storage[lane] = float(activated[2048 + h * 128 + lane]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane == 0) {
    float qn = 0.0f, kn = 0.0f;
    for (uint d = 0; d < Dk; d++) { qn += qh[d] * qh[d]; kn += kh[d] * kh[d]; }
    float qs = metal::rsqrt(qn + 1e-6f) * metal::rsqrt((float)Dk);   // L2 归一化 + Dk^-0.5
    float ks = metal::rsqrt(kn + 1e-6f);
            scales[h * 2] = qs;
            scales[h * 2 + 1] = ks;
        }
        q_ready[h * 128 + lane] = qh[lane];
        k_ready[h * 128 + lane] = kh[lane];
    } else {
        uint h = group - 16;
        v_ready[h * 128 + lane] = float(activated[4096 + h * 128 + lane]);
    }
