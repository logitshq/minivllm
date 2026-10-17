fp_qmv_fast_impl<T, 16, 4, false, 4>(
    weight, scales, nullptr, x, output, v124_k, v124_n,
    uint3(0, threadgroup_position_in_grid.y, 0),
    thread_position_in_threadgroup.x / 32, thread_position_in_threadgroup.x % 32);
