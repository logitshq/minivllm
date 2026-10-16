// Qualification only: operands and left multiplication match V44 state source.
    uint i = thread_position_in_grid.x;
    uint h = i / 128;
    float qs = scales[h * 2];
    float ks = scales[h * 2 + 1];
    q_expected[i] = qs * q_raw[i];
    k_expected[i] = ks * k_raw[i];
