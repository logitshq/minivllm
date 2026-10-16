// One uniform branch per TG; the native R4 math and local tile row are unchanged.
uint local = thread_position_in_threadgroup.x;
uint tile = threadgroup_position_in_grid.y;
if (tile < 1280) {
    fp_qmv_fast_impl<T, 16, 4, false, 4>(w_qkv, s_qkv, nullptr, x, y_qkv,
        v66_k, v66_n_qkv, uint3(0, tile, 0), local / 32, local % 32);
} else if (tile < 2048) {
    fp_qmv_fast_impl<T, 16, 4, false, 4>(w_z, s_z, nullptr, x, y_z,
        v66_k, v66_n_z, uint3(0, tile - 1280, 0), local / 32, local % 32);
} else if (tile < 2054) {
    fp_qmv_fast_impl<T, 16, 4, false, 4>(w_b, s_b, nullptr, x, y_b,
        v66_k, v66_n_b, uint3(0, tile - 2048, 0), local / 32, local % 32);
} else {
    fp_qmv_fast_impl<T, 16, 4, false, 4>(w_a, s_a, nullptr, x, y_a,
        v66_k, v66_n_a, uint3(0, tile - 2054, 0), local / 32, local % 32);
}
