// Read actual original product descriptors without a contiguous conversion.
// Probe output is metadata only, not tensor numerical data.
if (thread_position_in_grid.x == 0) {
    for (uint d = 0; d < 3; ++d) {
        layout[d] = ulong(product_shape[d]);
        layout[3 + d] = ulong(product_strides[d]);
    }
}
