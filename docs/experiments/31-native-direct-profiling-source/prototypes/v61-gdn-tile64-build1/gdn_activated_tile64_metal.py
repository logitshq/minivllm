"""V44 already-activated BF16 norm-reuse + held32 candidate; GPU unqualified.
V61 tile64 only: state strings, SPLIT4, held32/serial order and safe policy stay exact.
Prework only loads already-activated BF16 values; it never applies SiLU.
Import is inert; no engine connection or performance claim.
"""

SPLIT = 4
VALUE_TILE = 64
COMPILE_OPTIONS = {"math_mode": "safe"}

HEADER = r'''// Typed bodies pinned to MLX 0.32.3 unary_ops.h and binary_ops.h.
struct V18NativeSigmoid {
  template <typename T>
  T operator()(T x) thread {
    auto y = 1 / (1 + metal::precise::exp(metal::abs(x)));
    return (x < 0) ? y : 1 - y;
  }
};
struct V18NativeMultiply {
  template <typename T>
  T operator()(T x, T y) thread {
    return x * y;
  }
};
struct V18BF16Activation {
    bfloat16_t sigmoid_value;
    bfloat16_t activated;
};
inline V18BF16Activation v18_bf16_activation(bfloat16_t u) {
    bfloat16_t sigmoid_value = V18NativeSigmoid()(u);
    bfloat16_t activated = V18NativeMultiply()(u, sigmoid_value);
    return {sigmoid_value, activated};
}
'''

_NORM_BODY = r'''    float qn = 0.0f, kn = 0.0f;
    for (uint d = 0; d < Dk; d++) { qn += qh[d] * qh[d]; kn += kh[d] * kh[d]; }
    float qs = metal::rsqrt(qn + 1e-6f) * metal::rsqrt((float)Dk);   // L2 归一化 + Dk^-0.5
    float ks = metal::rsqrt(kn + 1e-6f);
'''

_OLD_Q = '    const device float* qh = q + (size_t)bh * Dk;'

_OLD_K = '    const device float* kh = k + (size_t)bh * Dk;'

_READY_Q = '    const device float* qh = q_ready + (size_t)(h / 3) * Dk;'

_READY_K = '    const device float* kh = k_ready + (size_t)(h / 3) * Dk;'

_SCALE_LOAD = '    float qs = scales[(h / 3) * 2];\n    float ks = scales[(h / 3) * 2 + 1];\n'

_PREWORK_SOURCE = r'''
    uint lane = thread_position_in_threadgroup.x;
    uint group = threadgroup_position_in_grid.x;
    threadgroup float q_storage[128];
    threadgroup float k_storage[128];
    if (group < 16) {
        uint h = group;
        uint Dk = 128;
        const threadgroup float* qh = q_storage;
        const threadgroup float* kh = k_storage;
        q_storage[lane] = float(activated[h * 128 + lane]);
        k_storage[lane] = float(activated[2048 + h * 128 + lane]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane == 0) {
    float qn = 0.0f, kn = 0.0f;
    for (uint d = 0; d < Dk; d++) { qn += qh[d] * qh[d]; kn += kh[d] * kh[d]; }
    float qs = metal::rsqrt(qn + 1e-6f) * metal::rsqrt((float)Dk);   // L2 归一化 + Dk^-0.5
    float ks = metal::rsqrt(kn + 1e-6f);
            scales[h * 2] = qs;
            scales[h * 2 + 1] = ks;
        }
        q_ready[h * 128 + lane] = qh[lane];
        k_ready[h * 128 + lane] = kh[lane];
    } else {
        uint h = group - 16;
        v_ready[h * 128 + lane] = float(activated[4096 + h * 128 + lane]);
    }
'''

_STATE_SOURCE = r'''
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
    const device float* qh = q_ready + (size_t)(h / 3) * Dk;
    const device float* kh = k_ready + (size_t)(h / 3) * Dk;
    threadgroup float red[SPLIT * VALUE_TILE];
    float qs = scales[(h / 3) * 2];
    float ks = scales[(h / 3) * 2 + 1];
    float ex = metal::exp(g[bh]);
    float be = beta[bh];
    float kv = 0.0f;
    float held[32];
    #pragma unroll 32
    for (uint j = 0; j < 32; j++) {
        uint d = d0 + j;
        held[j] = S[base + d * Dv + dv];
        kv += ks * kh[d] * held[j];
    }
    red[sp * VALUE_TILE + dv_local] = kv;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    kv = 0.0f;
    for (uint i = 0; i < SPLIT; i++) kv += red[i * VALUE_TILE + dv_local];
    float delta = (v[(size_t)bh * Dv + dv] - ex * kv) * be;
    float acc = 0.0f;
    #pragma unroll 32
    for (uint j = 0; j < 32; j++) {
        uint d = d0 + j;
        float s = ex * held[j] + ks * kh[d] * delta;
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
'''

_NORM_PROBE_SOURCE = r'''
    uint h = thread_position_in_grid.x;
    uint Dk = 128;
    const device float* qh = q_ready + (size_t)h * Dk;
    const device float* kh = k_ready + (size_t)h * Dk;
    float qn = 0.0f, kn = 0.0f;
    for (uint d = 0; d < Dk; d++) { qn += qh[d] * qh[d]; kn += kh[d] * kh[d]; }
    float qs = metal::rsqrt(qn + 1e-6f) * metal::rsqrt((float)Dk);   // L2 归一化 + Dk^-0.5
    float ks = metal::rsqrt(kn + 1e-6f);
    norm_stats[h * 4] = qn;
    norm_stats[h * 4 + 1] = kn;
    norm_stats[h * 4 + 2] = qs;
    norm_stats[h * 4 + 3] = ks;
'''

_prework_kernel = None
_state_kernel = None
_norm_probe_kernel = None


def _runtime():
    import mlx.core as mx
    return mx


def _validate_shapes(activated, g, beta, S):
    if not (getattr(activated, "shape", None) == (1, 10240)
            and getattr(g, "shape", None) == getattr(beta, "shape", None) == (1, 48)
            and getattr(S, "shape", None) == (1, 48, 128, 128)):
        raise ValueError("norm reuse requires B1/Hk16/Hv48/D128 fixed shapes")


def _validate_dtypes(activated, g, beta, S, mx):
    if not (activated.dtype == mx.bfloat16 and g.dtype == beta.dtype == S.dtype == mx.float32):
        raise ValueError("activated norm reuse requires already-activated BF16 and FP32 g/beta/S")


def _make_kernel(mx, name, input_names, output_names, source):
    return mx.fast.metal_kernel(
        name=name, input_names=input_names, output_names=output_names,
        source=source, header=HEADER, compile_options=COMPILE_OPTIONS,
    )


def _prepare(activated, mx):
    global _prework_kernel
    if _prework_kernel is None:
        _prework_kernel = _make_kernel(
            mx, "gdn_activated_norm_reuse_prework_v44_build1", ["activated"],
            ["q_ready", "k_ready", "v_ready", "scales"], _PREWORK_SOURCE,
        )
    return tuple(_prework_kernel(
        inputs=[activated], grid=(64 * 128, 1, 1), threadgroup=(128, 1, 1),
        output_shapes=[(1, 16, 128), (1, 16, 128), (1, 48, 128), (1, 16, 2)],
        output_dtypes=[mx.float32] * 4,
    ))


def _state(q_ready, k_ready, v_ready, scales, g, beta, S, mx):
    global _state_kernel
    if _state_kernel is None:
        _state_kernel = _make_kernel(
            mx, "gdn_tile64_norm_reuse_cached_state_v61_build1", ["q_ready", "k_ready", "v", "scales", "g", "beta", "S"],
            ["o", "S_new"], _STATE_SOURCE.replace("SPLIT", str(SPLIT)).replace("VALUE_TILE", str(VALUE_TILE)),
        )
    return tuple(_state_kernel(
        inputs=[q_ready, k_ready, v_ready, scales, g, beta, S],
        grid=(128, 48 * SPLIT, 1), threadgroup=(VALUE_TILE, SPLIT, 1),
        output_shapes=[(1, 48, 128), (1, 48, 128, 128)],
        output_dtypes=[mx.float32, mx.float32],
    ))


def gated_delta_step_activated_norm_reuse_metal(activated, g, beta, S):
    """Only operator API: already-activated BF16[1,10240], FP32 g/beta[1,48], S[1,48,128,128]."""
    _validate_shapes(activated, g, beta, S)
    mx = _runtime()
    _validate_dtypes(activated, g, beta, S, mx)
    q_ready, k_ready, v_ready, scales = _prepare(activated, mx)
    return _state(q_ready, k_ready, v_ready, scales, g, beta, S, mx)


def _norm_probe(q_ready, k_ready, mx):
    """Qualification only: old serial norm body, outputs qn/kn/qs/ks per head."""
    if not (q_ready.shape == k_ready.shape == (1, 16, 128)
            and q_ready.dtype == k_ready.dtype == mx.float32):
        raise ValueError("norm probe requires FP32 Q/K[1,16,128]")
    global _norm_probe_kernel
    if _norm_probe_kernel is None:
        _norm_probe_kernel = _make_kernel(
            mx, "gdn_norm_reuse_serial_norm_probe_build1", ["q_ready", "k_ready"], ["norm_stats"], _NORM_PROBE_SOURCE,
        )
    return tuple(_norm_probe_kernel(
        inputs=[q_ready, k_ready], grid=(16, 1, 1), threadgroup=(16, 1, 1),
        output_shapes=[(1, 16, 4)], output_dtypes=[mx.float32],
    ))[0]
