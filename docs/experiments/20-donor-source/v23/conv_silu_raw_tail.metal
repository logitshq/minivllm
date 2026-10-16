// V23 prototype: MLX metal_kernel body, not a standalone Metal translation unit.
// SPDX-License-Identifier: MIT
// See LICENSE-MLX-MIT for the MLX-derived typed arithmetic helpers.
// SUM_Y is the measured reference reduction's logical y count, not this grid's y.
uint channel = thread_position_in_grid.x;
if (channel >= 10240) {
    return;
}
// One physical owner evaluates the declared logical y-local folds in order.
// No FP32 accumulator, atomics, input writes, or cross-channel sharing.
bfloat16_t partial[SUM_Y];
for (uint y = 0; y < SUM_Y; ++y) {
    bfloat16_t total = bfloat16_t(0);
    for (uint tap = y; tap < 4; tap += SUM_Y) {
        bfloat16_t input_value = (tap < 3)
            ? old_tail[tap * 10240 + channel]
            : raw[channel];
        bfloat16_t product = V18NativeMultiply()(input_value, w[channel * 4 + tap]);
        // Match Sum<bfloat16_t>::operator()(product, total), including +zero.
        total = bfloat16_t(product + total);
    }
    partial[y] = total;
}
bfloat16_t conv = partial[0];
for (uint y = 1; y < SUM_Y; ++y) {
    conv = bfloat16_t(partial[y] + conv);
}
act[channel] = v18_bf16_activation(conv).activated;
// New output storage only; old_tail and raw remain immutable.
new_tail[channel] = old_tail[10240 + channel];
new_tail[10240 + channel] = old_tail[20480 + channel];
new_tail[20480 + channel] = raw[channel];
