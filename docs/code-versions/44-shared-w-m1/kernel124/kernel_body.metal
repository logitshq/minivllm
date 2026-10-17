v124_shared_m1<T, VERIFY_T, K_SIZE, N_SIZE>(
    weight, scales, x, output, threadgroup_position_in_grid.y,
    thread_position_in_threadgroup.x / 32, thread_position_in_threadgroup.x % 32);
