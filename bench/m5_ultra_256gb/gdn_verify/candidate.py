"""E018 isolated literal GDN chain; never enabled in the model or cache code.

Keep the pinned original SPLIT=4, VALUE_TILE=32 arithmetic and reductions.
Each thread owns 32 FP32 state values, carried across at most eight tokens.
"""
import mlx.core as mx


SOURCE = r"""
    uint dv_local = thread_position_in_threadgroup.x;
    uint dv = threadgroup_position_in_grid.x * 32 + dv_local;
    uint sp = thread_position_in_threadgroup.y;
    uint h = threadgroup_position_in_grid.y;
    uint b = threadgroup_position_in_grid.z;
    uint Hv = S_shape[1];
    uint Dk = S_shape[2];
    uint Dv = S_shape[3];
    uint bh = b * Hv + h;
    uint seg = Dk / 4;
    uint d0 = sp * seg;
    size_t base = (size_t)bh * Dk * Dv;
    size_t state_stride = (size_t)S_shape[0] * Hv * Dk * Dv;
    size_t head_stride = (size_t)S_shape[0] * Hv;
    threadgroup float red[4 * 32];
    float carried[32];
    for (uint d = d0; d < d0 + seg; d++) {
        carried[d - d0] = S[base + d * Dv + dv];
    }
    for (uint t = 0; t < T; t++) {
        size_t tbh = (size_t)t * head_stride + bh;
        const device float* qh = q + tbh * Dk;
        const device float* kh = k + tbh * Dk;
        float qn = 0.0f, kn = 0.0f;
        for (uint d = 0; d < Dk; d++) { qn += qh[d] * qh[d]; kn += kh[d] * kh[d]; }
        float qs = metal::rsqrt(qn + 1e-6f) * metal::rsqrt((float)Dk);
        float ks = metal::rsqrt(kn + 1e-6f);
        float ex = metal::exp(g[tbh]);
        float be = beta[tbh];
        float kv = 0.0f;
        for (uint d = d0; d < d0 + seg; d++) kv += ks * kh[d] * carried[d - d0];
        red[sp * 32 + dv_local] = kv;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        kv = 0.0f;
        for (uint i = 0; i < 4; i++) kv += red[i * 32 + dv_local];
        float delta = (v[tbh * Dv + dv] - ex * kv) * be;
        float acc = 0.0f;
        for (uint d = d0; d < d0 + seg; d++) {
            float s = ex * carried[d - d0] + ks * kh[d] * delta;
            carried[d - d0] = s;
            history[(size_t)t * state_stride + base + d * Dv + dv] = s;
            acc += qs * qh[d] * s;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        red[sp * 32 + dv_local] = acc;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sp == 0) {
            acc = 0.0f;
            for (uint i = 0; i < 4; i++) acc += red[i * 32 + dv_local];
            output[tbh * Dv + dv] = acc;
        }
        // The next token must not overwrite red while split 0 reads this one.
        if (t + 1 < T) threadgroup_barrier(mem_flags::mem_threadgroup);
    }
"""

_kernel = mx.fast.metal_kernel(
    name="e018_literal_gdn_chain", input_names=["q", "k", "v", "g", "beta", "S"],
    output_names=["output", "history"], source=SOURCE)


def verify_history(q, k, v, g, beta, state):
    """q/k/v [T,1,48,128], gates [T,1,48], state [1,48,128,128]."""
    if (q.ndim != 4 or q.shape != k.shape or q.shape != v.shape
            or q.shape[0] not in (1, 3, 5, 8) or q.shape[1:] != (1, 48, 128)
            or g.shape != beta.shape or g.shape != q.shape[:3]
            or state.shape != (1, 48, 128, 128) or state.dtype != mx.float32):
        raise ValueError("E018 supports only pinned B1/H48/D128 and T1/3/5/8")
    if any(x.dtype not in (mx.float32, mx.float16, mx.bfloat16) for x in (q, k, v, g, beta)):
        raise TypeError("E018 inputs must be floating point")
    t = q.shape[0]
    return _kernel(
        inputs=[x.astype(mx.float32) for x in (q, k, v, g, beta)] + [state],
        template=[("T", t)], grid=(128, 48 * 4, 1), threadgroup=(32, 4, 1),
        output_shapes=[(t, 1, 48, 128), (t, 1, 48, 128, 128)],
        output_dtypes=[mx.float32, mx.float32])
