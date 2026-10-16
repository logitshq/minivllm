
    uint x = thread_position_in_threadgroup.x;
    uint y = thread_position_in_threadgroup.y;
    uint lane = thread_index_in_simdgroup;
    uint sp = lane / 8;
    uint dv_local = y * 8 + lane % 8;
    uint row = y * 32 + x;
    mapping[row * 7 + 0] = x;
    mapping[row * 7 + 1] = y;
    mapping[row * 7 + 2] = lane;
    mapping[row * 7 + 3] = sp;
    mapping[row * 7 + 4] = dv_local;
    mapping[row * 7 + 5] = sp * 32;
    mapping[row * 7 + 6] = marker[0];
