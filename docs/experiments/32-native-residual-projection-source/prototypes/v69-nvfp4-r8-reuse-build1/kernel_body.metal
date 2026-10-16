uint local = thread_position_in_threadgroup.x;
uint3 tile = uint3(0, threadgroup_position_in_grid.y, 0);
fp_qmv_fast_impl<T, 16, 4, false, R>(weight, scales, nullptr, x, output,
    v69_in_vec_size, v69_out_vec_size, tile, local / 32, local % 32);
