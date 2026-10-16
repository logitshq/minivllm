"""E009 prototype: retain each lane's 32 FP32 state elements across reductions.

Original arithmetic and reduction order; independent output buffer. Whether
Metal keeps this in registers or spills it must be measured, not assumed.
"""
from contextlib import contextmanager
import mlx.core as mx
import mlx.nn as nn
from tutorial_code.miniengine.kernels import gdn_metal as original
from tutorial_code.miniengine.model.gdn import _m5_ultra_decode_identity

REFERENCE = original.gated_delta_step_metal
source = original._SOURCE.replace('uint seg = Dk / SPLIT;', 'constexpr uint seg = 32;')
old = 'for (uint d = d0; d < d0 + seg; d++) kv += ks * kh[d] * S[base + d * Dv + dv];'
assert source.count(old) == 1
source = source.replace(old, '''float saved[32];
    for (uint d = d0; d < d0 + seg; d++) {
        float previous = S[base + d * Dv + dv];
        saved[d - d0] = previous;
        kv += ks * kh[d] * previous;
    }''')
old = 'float s = ex * S[base + d * Dv + dv] + ks * kh[d] * delta;'
assert source.count(old) == 1
source = source.replace(old, 'float s = ex * saved[d - d0] + ks * kh[d] * delta;')
kernel = mx.fast.metal_kernel(
    name='gdn_decode_state_reuse_experiment',
    input_names=['q', 'k', 'v', 'g', 'beta', 'S'], output_names=['o', 'S_new'],
    source=source.replace('SPLIT', str(original.SPLIT)).replace('VALUE_TILE', str(original.VALUE_TILE)))


def step(q, k, v, g, beta, state):
    B = state.shape[0]
    if not (9 <= B <= 64 and state.shape == (B, 48, 128, 128)
            and q.shape == k.shape == v.shape == (B, 48, 128)
            and g.shape == beta.shape == (B, 48)
            and state.dtype == mx.float32 and mx.default_device() == mx.gpu
            and _m5_ultra_decode_identity()):
        return REFERENCE(q, k, v, g, beta, state)
    original.DISPATCH_COUNTS['decode_step'] += 1
    return tuple(kernel(inputs=[q.astype(mx.float32), k.astype(mx.float32),
                        v.astype(mx.float32), g.astype(mx.float32), beta.astype(mx.float32), state],
                        grid=(128, 48 * original.SPLIT, B),
                        threadgroup=(original.VALUE_TILE, original.SPLIT, 1),
                        output_shapes=[(B, 48, 128), (B, 48, 128, 128)],
                        output_dtypes=[mx.float32, mx.float32]))


def fused(raw, conv, w, g, beta, state):
    # Same signature as E007's screen; preparation is unchanged in this experiment.
    B = raw.shape[0]
    xp = mx.concatenate([conv, raw[:, None]], axis=1)
    u = (xp * w.T[None]).sum(axis=1)
    q, k, v = mx.split(nn.silu(u), [2048, 4096], axis=-1)
    q = mx.repeat(q.reshape(B, 16, 128), 3, axis=1)
    k = mx.repeat(k.reshape(B, 16, 128), 3, axis=1)
    o, state = step(q, k, v.reshape(B, 48, 128), g, beta, state)
    return o, state, xp[:, 1:]


@contextmanager
def enabled(active=True):
    previous = original.gated_delta_step_metal
    if active:
        original.gated_delta_step_metal = step
    try:
        yield
    finally:
        original.gated_delta_step_metal = previous
