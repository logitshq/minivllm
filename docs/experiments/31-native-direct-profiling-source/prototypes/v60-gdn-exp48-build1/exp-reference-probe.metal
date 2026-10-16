
    uint dv_local = thread_position_in_threadgroup.x;
    uint dv = threadgroup_position_in_grid.x * 32 + dv_local;
    uint sp = thread_position_in_threadgroup.y;
    uint bh = threadgroup_position_in_grid.y;
    float ex = metal::exp(g[bh]);
    original_exp[((size_t)bh * 128 + dv) * 4 + sp] = ex;
