"""V76 independent extension: activated norm prework + exact BA QMV/FP32 gates.
CPU-prepared local probe only. This is not oMLX's actual Q35 decode path.
No engine integration, performance claim, altered state math or weight repack.
"""
from pathlib import Path

SPLIT = 4
VALUE_TILE = 32
COMPILE_OPTIONS = {"math_mode": "safe"}
D = Path(__file__).resolve().parent

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
        // Only the first SIMD in each value-head group performs both QMV rows.
        if (lane < 32) {
            const uint simd_lid = lane;
            const device bfloat16_t* xp = x + simd_lid * 16;
            const device uint8_t* wb = (const device uint8_t*)w_b + h * 2560 + simd_lid * 8;
            const device uint8_t* wa = (const device uint8_t*)w_a + h * 2560 + simd_lid * 8;
            const device uint8_t* sb = s_b + h * 320 + simd_lid;
            const device uint8_t* sa = s_a + h * 320 + simd_lid;
            thread float x_thread[16];
            float result_b = 0.0f;
            float result_a = 0.0f;
            for (int kk = 0; kk < 5120; kk += 512) {
                load_vector<bfloat16_t, float, 16>(xp, x_thread);
                float scale_b = dequantize_scale<float, 16>(sb[0]);
                result_b += qdot<float, 16, 4>(wb, x_thread, scale_b);
                float scale_a = dequantize_scale<float, 16>(sa[0]);
                result_a += qdot<float, 16, 4>(wa, x_thread, scale_a);
                wb += 256;
                wa += 256;
                sb += 32;
                sa += 32;
                xp += 512;
            }
            result_b = simd_sum(result_b);
            result_a = simd_sum(result_a);
            if (simd_lid == 0) {
                const bfloat16_t bv = static_cast<bfloat16_t>(result_b);
                const bfloat16_t av = static_cast<bfloat16_t>(result_a);
                // V76_DIAGNOSTIC_BA_STORES
                const float bf = static_cast<float>(bv);
                const float af = static_cast<float>(av);
                const float beta_value = V76NativeSigmoid()(bf);
                const float apd = V76NativeAdd()(af, dt_bias[h]);
                const float softplus_value = V76NativeLogAddExp()(apd, 0.0f);
                const float ea = V76NativeExp()(A_log[h]);
                const float neg_ea = V76NativeNegative()(ea);
                const float g_value = V76NativeMultiply()(neg_ea, softplus_value);
                g_ready[h] = g_value;
                beta_ready[h] = beta_value;
            }
        }
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

_prework_kernel = None
_prework_probe_kernel = None
_state_kernel = None

def _runtime():
    import mlx.core as mx
    return mx

def _make_kernel(mx, name, input_names, output_names, source):
    return mx.fast.metal_kernel(
        name=name, input_names=input_names, output_names=output_names,
        source=source, header=HEADER, compile_options=COMPILE_OPTIONS,
    )

def _state(q_ready, k_ready, v_ready, scales, g, beta, S, mx):
    global _state_kernel
    if _state_kernel is None:
        _state_kernel = _make_kernel(
            mx, "gdn_norm_reuse_cached_state_v40_build1", ["q_ready", "k_ready", "v", "scales", "g", "beta", "S"],
            ["o", "S_new"], _STATE_SOURCE.replace("SPLIT", str(SPLIT)).replace("VALUE_TILE", str(VALUE_TILE)),
        )
    return tuple(_state_kernel(
        inputs=[q_ready, k_ready, v_ready, scales, g, beta, S],
        grid=(128, 48 * SPLIT, 1), threadgroup=(VALUE_TILE, SPLIT, 1),
        output_shapes=[(1, 48, 128), (1, 48, 128, 128)],
        output_dtypes=[mx.float32, mx.float32],
    ))

INPUT_NAMES = ["activated", "x", "w_b", "s_b", "w_a", "s_a", "A_log", "dt_bias"]
OUTPUT_NAMES = ["q_ready", "k_ready", "v_ready", "scales", "g_ready", "beta_ready"]
SHAPES = ((1,10240),(1,5120),(48,640),(48,320),(48,640),(48,320),(48,),(48,))


def _validate(inputs, observed_strides, mx):
    if len(inputs) != 8 or len(observed_strides) != 8:
        raise ValueError("exact8 original inputs and observed stride records required")
    types=(mx.bfloat16,mx.bfloat16,mx.uint32,mx.uint8,mx.uint32,mx.uint8,mx.float32,mx.float32)
    for value,shape,dtype,strides in zip(inputs,SHAPES,types,observed_strides):
        if value.shape != shape or value.dtype != dtype or len(strides) != len(shape):
            raise ValueError("unsupported input metadata")
        strides=tuple(strides)
        if any(type(v) is not int or v < 0 for v in strides):
            raise ValueError("actual nonnegative integer element strides required")
        expected=1
        for dim,stride in zip(reversed(shape),reversed(strides)):
            if dim>1 and stride!=expected:raise ValueError("noncanonical input; no packing/copy fallback")
            expected*=dim


def prepare(activated,x,w_b,s_b,w_a,s_a,A_log,dt_bias,*,observed_strides,probe=False):
    """Production8inputs/6outputs; qualification probe adds only BF16 b/a outputs."""
    if type(probe) is not bool:raise TypeError("probe must be bool")
    mx=_runtime();inputs=(activated,x,w_b,s_b,w_a,s_a,A_log,dt_bias)
    _validate(inputs,observed_strides,mx)
    global _prework_kernel,_prework_probe_kernel
    kernel=_prework_probe_kernel if probe else _prework_kernel
    names=OUTPUT_NAMES+(["b_probe","a_probe"] if probe else [])
    source=_PREWORK_SOURCE.replace('// V76_DIAGNOSTIC_BA_STORES',
        'b_probe[h] = bv; a_probe[h] = av;' if probe else '// no diagnostic stores')
    if kernel is None:
        kernel=mx.fast.metal_kernel(name='gdn_activated_BA_gates_prework_v76_build1'+('_probe' if probe else ''),
            input_names=INPUT_NAMES,output_names=names,source=source,
            header=(D/'qmv_header.metal').read_text()+HEADER+(D/'gates_header.metal').read_text(),
            compile_options=COMPILE_OPTIONS,ensure_row_contiguous=False)
        if probe:_prework_probe_kernel=kernel
        else:_prework_kernel=kernel
    shapes=[(1,16,128),(1,16,128),(1,48,128),(1,16,2),(1,48),(1,48)]+([(1,48),(1,48)] if probe else [])
    dtypes=[mx.float32]*6+([mx.bfloat16]*2 if probe else [])
    return tuple(kernel(inputs=list(inputs),grid=(64*128,1,1),threadgroup=(128,1,1),output_shapes=shapes,output_dtypes=dtypes))


def state(prepared,S):
    mx=_runtime()
    if len(prepared)!=6 or S.shape!=(1,48,128,128) or S.dtype!=mx.float32:
        raise ValueError("fixedB1 six prepared outputs and canonical FP32 state required")
    q,k,v,scales,g,beta=prepared
    return _state(q,k,v,scales,g,beta,S,mx)
