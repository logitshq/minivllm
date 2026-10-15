"""V18 isolated post-conv input preparation; unqualified, independent opt-in.
Original recurrent equations remain unchanged after cooperative BF16 activation.
Logical threadgroup scratch 1664 B vs 512 B. Activations/layer/token 55296
vs 10240; no speed claim. Pipeline/128-thread launch remains a GPU gate.
"""
import mlx.core as mx

SPLIT = 4
VALUE_TILE = 32
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
_SOURCE = r'''
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
    threadgroup float q_storage[128];
    threadgroup float k_storage[128];
    threadgroup float v_tile[32];
    const threadgroup float* qh = q_storage;
    const threadgroup float* kh = k_storage;
    uint j = sp * VALUE_TILE + dv_local;
    q_storage[j] = float(v18_bf16_activation(u[(h / 3) * 128 + j]).activated);
    k_storage[j] = float(v18_bf16_activation(u[2048 + (h / 3) * 128 + j]).activated);
    if (sp == 0) {
        v_tile[dv_local] = float(v18_bf16_activation(u[4096 + h * 128 + dv]).activated);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float red[SPLIT * VALUE_TILE];
    float qn = 0.0f, kn = 0.0f;
    for (uint d = 0; d < Dk; d++) { qn += qh[d] * qh[d]; kn += kh[d] * kh[d]; }
    float qs = metal::rsqrt(qn + 1e-6f) * metal::rsqrt((float)Dk);   // L2 归一化 + Dk^-0.5
    float ks = metal::rsqrt(kn + 1e-6f);
    float ex = metal::exp(g[bh]);
    float be = beta[bh];
    float kv = 0.0f;
    for (uint d = d0; d < d0 + seg; d++) kv += ks * kh[d] * S[base + d * Dv + dv];
    red[sp * VALUE_TILE + dv_local] = kv;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    kv = 0.0f;
    for (uint i = 0; i < SPLIT; i++) kv += red[i * VALUE_TILE + dv_local];
    float delta = (v_tile[dv_local] - ex * kv) * be;
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
'''
_PROBE_SOURCE = r'''    uint i = thread_position_in_grid.x;
    const uint V18_PROBE_COUNT = 65280;
    if (i < V18_PROBE_COUNT) {
        V18BF16Activation values = v18_bf16_activation(u[i]);
        sigmoid_bits[i] = as_type<ushort>(values.sigmoid_value);
        silu_bits[i] = as_type<ushort>(values.activated);
    }
'''

_kernel = None
_probe_kernel = None

def _make_kernel(name, input_names, output_names, source):
    return mx.fast.metal_kernel(
        name=name, input_names=input_names, output_names=output_names,
        source=source, header=HEADER, compile_options=COMPILE_OPTIONS,
    )

def gated_delta_step_postconv_metal(u, g, beta, S):
    """Strict internal B1 entry. Caller must establish normal-mode eligibility."""
    if not (u.shape == (1, 10240) and u.dtype == mx.bfloat16
            and g.shape == beta.shape == (1, 48)
            and g.dtype == beta.dtype == S.dtype == mx.float32
            and S.shape == (1, 48, 128, 128)):
        raise ValueError("V18 postconv requires fixed B1 BF16 u and FP32 S/g/beta")
    global _kernel
    if _kernel is None:
        _kernel = _make_kernel(
            "v18_gdn_postconv_step", ["u", "g", "beta", "S"], ["o", "S_new"],
            _SOURCE.replace("SPLIT", str(SPLIT)).replace("VALUE_TILE", str(VALUE_TILE)),
        )
    return tuple(_kernel(
        inputs=[u, g, beta, S], grid=(128, 48 * SPLIT, 1),
        threadgroup=(VALUE_TILE, SPLIT, 1),
        output_shapes=[(1, 48, 128), (1, 48, 128, 128)],
        output_dtypes=[mx.float32, mx.float32],
    ))

def activation_probe(u):
    """Qualification only; exact helper/header/options shared with production."""
    if u.dtype != mx.bfloat16 or u.ndim != 1 or u.shape != (65280,):
        raise ValueError("probe requires exactly BF16[65280]")
    global _probe_kernel
    if _probe_kernel is None:
        _probe_kernel = _make_kernel(
            "v18_bf16_activation_probe", ["u"], ["sigmoid_bits", "silu_bits"], _PROBE_SOURCE,
        )
    return tuple(_probe_kernel(
        inputs=[u], grid=(65280, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[u.shape, u.shape], output_dtypes=[mx.uint16, mx.uint16],
    ))
