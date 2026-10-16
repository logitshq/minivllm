uint local = thread_position_in_threadgroup.x;
fp_qmv_fast_impl<T, 16, 4, false, 4>(weight, scales, nullptr, x, output,
    v66_k, v66_control_n, uint3(0, threadgroup_position_in_grid.y, 0), local / 32, local % 32);
