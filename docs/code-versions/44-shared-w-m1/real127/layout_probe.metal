// Metadata only. No contiguous conversion or input write.
if (thread_position_in_grid.x == 0) {
    for (uint d = 0; d < 2; ++d) {
        descriptor[d] = ulong(value_shape[d]);
        descriptor[2 + d] = ulong(value_strides[d]);
    }
}
