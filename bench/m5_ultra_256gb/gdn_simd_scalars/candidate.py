"""Unqualified B1 GDN scalar sharing experiment; default OFF, import is inert.

Only each SIMD group's lane 0 computes the original serial q/k norms and gate
scalars. Broadcasts move FP32 values without changing the state dot/reduction
source. No model, request cache, global patch, eval, or captured array is owned.
"""

ORIGINAL_SCALARS = """    float qn = 0.0f, kn = 0.0f;
    for (uint d = 0; d < Dk; d++) { qn += qh[d] * qh[d]; kn += kh[d] * kh[d]; }
    float qs = metal::rsqrt(qn + 1e-6f) * metal::rsqrt((float)Dk);   // L2 归一化 + Dk^-0.5
    float ks = metal::rsqrt(kn + 1e-6f);
    float ex = metal::exp(g[bh]);
    float be = beta[bh];
"""

SHARED_SCALARS = """    float qs = 0.0f, ks = 0.0f, ex = 0.0f, be = 0.0f;
    if (thread_index_in_simdgroup == 0) {
        float qn = 0.0f, kn = 0.0f;
        for (uint d = 0; d < Dk; d++) { qn += qh[d] * qh[d]; kn += kh[d] * kh[d]; }
        qs = metal::rsqrt(qn + 1e-6f) * metal::rsqrt((float)Dk);
        ks = metal::rsqrt(kn + 1e-6f);
        ex = metal::exp(g[bh]);
        be = beta[bh];
    }
    qs = simd_broadcast(qs, ushort(0));
    ks = simd_broadcast(ks, ushort(0));
    ex = simd_broadcast(ex, ushort(0));
    be = simd_broadcast(be, ushort(0));
"""

# The recurrence is literal, not generated from a runtime source transformation.
# A CPU source test verifies the only difference from the pinned native body.
SOURCE = """
    uint dv_local = thread_position_in_threadgroup.x;
    uint dv = threadgroup_position_in_grid.x * VALUE_TILE + dv_local;
    uint sp = thread_position_in_threadgroup.y;
    uint h  = threadgroup_position_in_grid.y;
    uint b  = threadgroup_position_in_grid.z;
    uint Hv = S_shape[1];
    uint Dk = S_shape[2];
    uint Dv = S_shape[3];
    uint bh = b * Hv + h;
    uint seg = Dk / SPLIT;
    uint d0 = sp * seg;
    size_t base = (size_t)bh * Dk * Dv;
    const device float* qh = q + (size_t)bh * Dk;
    const device float* kh = k + (size_t)bh * Dk;
    threadgroup float red[SPLIT * VALUE_TILE];
    float qs = 0.0f, ks = 0.0f, ex = 0.0f, be = 0.0f;
    if (thread_index_in_simdgroup == 0) {
        float qn = 0.0f, kn = 0.0f;
        for (uint d = 0; d < Dk; d++) { qn += qh[d] * qh[d]; kn += kh[d] * kh[d]; }
        qs = metal::rsqrt(qn + 1e-6f) * metal::rsqrt((float)Dk);
        ks = metal::rsqrt(kn + 1e-6f);
        ex = metal::exp(g[bh]);
        be = beta[bh];
    }
    qs = simd_broadcast(qs, ushort(0));
    ks = simd_broadcast(ks, ushort(0));
    ex = simd_broadcast(ex, ushort(0));
    be = simd_broadcast(be, ushort(0));
    float kv = 0.0f;
    for (uint d = d0; d < d0 + seg; d++) kv += ks * kh[d] * S[base + d * Dv + dv];
    red[sp * VALUE_TILE + dv_local] = kv;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    kv = 0.0f;
    for (uint i = 0; i < SPLIT; i++) kv += red[i * VALUE_TILE + dv_local];
    float delta = (v[(size_t)bh * Dv + dv] - ex * kv) * be;
    float acc = 0.0f;
    for (uint d = d0; d < d0 + seg; d++) {
        float s = ex * S[base + d * Dv + dv] + ks * kh[d] * delta;
        S_new[base + d * Dv + dv] = s;
        acc += qs * qh[d] * s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    red[sp * VALUE_TILE + dv_local] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sp == 0) {
        acc = 0.0f;
        for (uint i = 0; i < SPLIT; i++) acc += red[i * VALUE_TILE + dv_local];
        o[(size_t)bh * Dv + dv] = acc;
    }
"""


def make_step(mx_module, baseline_module, backend_supported, *, enabled=False):
    """Return a per-owner explicit-input step; this does not install it anywhere.

    The future device harness supplies the real pinned M5 backend predicate.
    All calls outside B1/BF16 QKV + FP32 gates/state use the untouched helper.
    Changed baseline source rejects construction when ON, never silently tests
    another comparator. Enabled kernel creation/dispatch failures propagate.
    """
    if type(enabled) is not bool or not callable(backend_supported):
        raise TypeError('enabled must be bool and backend_supported callable')
    if enabled:
        expected = SOURCE.replace(SHARED_SCALARS, ORIGINAL_SCALARS)
        if (SOURCE.count(SHARED_SCALARS) != 1 or baseline_module._SOURCE != expected
                or baseline_module.SPLIT != 4 or baseline_module.VALUE_TILE != 32):
            raise ValueError('native source/geometry changed; requalify the comparator')
    return _Step(mx_module, baseline_module.gated_delta_step_metal, backend_supported, enabled)


class _Step:
    def __init__(self, mx, fallback, backend_supported, enabled):
        self._mx, self._fallback, self._backend_supported = mx, fallback, backend_supported
        self._enabled, self._kernel = enabled, None
        self.calls = {'candidate': 0, 'fallback': 0}
        self.last_path = None

    def __call__(self, q, k, v, g, beta, S):
        mx = self._mx
        eligible = (self._enabled and self._backend_supported()
                    and all(a.shape == (1, 48, 128) and a.dtype == mx.bfloat16 for a in (q, k, v))
                    and all(a.shape == (1, 48) and a.dtype == mx.float32 for a in (g, beta))
                    and S.shape == (1, 48, 128, 128) and S.dtype == mx.float32)
        if not eligible:
            self.calls['fallback'] += 1; self.last_path = 'fallback'
            return self._fallback(q, k, v, g, beta, S)
        if self._kernel is None:
            # Same default compiler options, casts, geometry and two outputs as
            # the original; no extra materialization or math-mode override.
            self._kernel = mx.fast.metal_kernel(name='gdn_simd_shared_scalars_b1',
                input_names=['q', 'k', 'v', 'g', 'beta', 'S'], output_names=['o', 'S_new'],
                source=SOURCE.replace('SPLIT', '4').replace('VALUE_TILE', '32'))
        self.last_path = 'candidate-attempt'
        result = self._kernel(inputs=[a.astype(mx.float32) for a in (q, k, v, g, beta)] + [S],
            grid=(128, 48 * 4, 1), threadgroup=(32, 4, 1),
            output_shapes=[(1, 48, 128), (1, 48, 128, 128)], output_dtypes=[mx.float32, mx.float32])
        self.calls['candidate'] += 1; self.last_path = 'candidate'
        return result
