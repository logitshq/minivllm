"""Chapter 09: original minivllm Metal GDN step and triangular solve.
Counters record dispatched kernels; checks evaluate outputs before reporting success.
"""
import mlx.core as mx


DISPATCH_COUNTS = {"decode_step": 0, "unit_lower_solve": 0}

def reset_dispatch_counts():
    for name in DISPATCH_COUNTS:
        DISPATCH_COUNTS[name] = 0

# 每个 (序列, 头) 一个 threadgroup：Dv 列 × SPLIT 段。线程 (t, s) 管第 t 列、dk ∈ [s·Dk/SPLIT, (s+1)·Dk/SPLIT)。
# 两次段间归约走 threadgroup 内存。B 小的时候线程数是瓶颈（B=1 只有 48 个头），拆段把线程数乘 SPLIT。
SPLIT = 4
_SOURCE = """
    uint dv = thread_position_in_threadgroup.x;
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
    threadgroup float red[SPLIT * 128];
    float qn = 0.0f, kn = 0.0f;
    for (uint d = 0; d < Dk; d++) { qn += qh[d] * qh[d]; kn += kh[d] * kh[d]; }
    float qs = metal::rsqrt(qn + 1e-6f) * metal::rsqrt((float)Dk);   // L2 归一化 + Dk^-0.5
    float ks = metal::rsqrt(kn + 1e-6f);
    float ex = metal::exp(g[bh]);
    float be = beta[bh];
    float kv = 0.0f;
    for (uint d = d0; d < d0 + seg; d++) kv += ks * kh[d] * S[base + d * Dv + dv];
    red[sp * Dv + dv] = kv;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    kv = 0.0f;
    for (uint i = 0; i < SPLIT; i++) kv += red[i * Dv + dv];
    float delta = (v[(size_t)bh * Dv + dv] - ex * kv) * be;
    float acc = 0.0f;
    for (uint d = d0; d < d0 + seg; d++) {
        float s = ex * S[base + d * Dv + dv] + ks * kh[d] * delta;
        S_new[base + d * Dv + dv] = s;
        acc += qs * qh[d] * s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    red[sp * Dv + dv] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sp == 0) {
        acc = 0.0f;
        for (uint i = 0; i < SPLIT; i++) acc += red[i * Dv + dv];
        o[(size_t)bh * Dv + dv] = acc;
    }
"""

_kernel = mx.fast.metal_kernel(
    name="gdn_decode_step", input_names=["q", "k", "v", "g", "beta", "S"], output_names=["o", "S_new"],
    source=_SOURCE.replace("SPLIT", str(SPLIT)),
)


def gated_delta_step_metal(q, k, v, g, beta, S):
    """和 gated_delta_step_batch 同样的接口：q, k [B, Hv, Dk]，v [B, Hv, Dv]，g, beta [B, Hv]，S [B, Hv, Dk, Dv] fp32。
    归一化、缩放、exp 都在 kernel 里做。"""
    B, Hv, Dk, Dv = S.shape
    if Dk % SPLIT or Dv > 128 or Dv * SPLIT > 1024:
        raise ValueError("Metal GDN requires Dk divisible by 4 and Dv <= 128")
    DISPATCH_COUNTS["decode_step"] += 1
    o, S_new = _kernel(
        inputs=[q.astype(mx.float32), k.astype(mx.float32), v.astype(mx.float32), g.astype(mx.float32), beta.astype(mx.float32), S],
        grid=(Dv, Hv * SPLIT, B), threadgroup=(Dv, SPLIT, 1),
        output_shapes=[(B, Hv, Dv), (B, Hv, Dk, Dv)], output_dtypes=[mx.float32, mx.float32],
    )
    return o, S_new


# ---------------------------------------------------------------------------------------------
# 第二个 kernel：分块 GDN 里的单位下三角求解 (I + A) X = V，前向代入。
# 纯 MLX 版是 63 步 Python 循环（每步 concat + 小矩阵乘），每层约 190 次发射。这里一个 threadgroup 解一个 (头, 块)：
# 线程 m 管第 m 列，按行 i = 0..C-1 顺序算 x_i[m] = v_i[m] − Σ_{j<i} A[i][j]·x_j[m]，行与行之间用 barrier 同步。
# X 放在 device 内存（64×256 个 float 超过 threadgroup 内存的 32 KB），刚写的行在 L1 里，读回很便宜。
_SOLVE_SOURCE = """
    uint m = thread_position_in_threadgroup.x;
    uint n = threadgroup_position_in_grid.x;
    uint C = A_shape[1];
    uint M = V_shape[2];
    const device float* An = A + (size_t)n * C * C;
    const device float* Vn = V + (size_t)n * C * M;
    device float* Xn = X + (size_t)n * C * M;
    for (uint i = 0; i < C; i++) {
        float acc = Vn[i * M + m];
        for (uint j = 0; j < i; j++) acc -= An[i * C + j] * Xn[j * M + m];
        Xn[i * M + m] = acc;
        threadgroup_barrier(mem_flags::mem_device);
    }
"""

_solve_kernel = mx.fast.metal_kernel(name="unit_lower_solve", input_names=["A", "V"], output_names=["X"], source=_SOLVE_SOURCE)


def solve_unit_lower_metal(A: mx.array, V: mx.array) -> mx.array:
    """A [..., C, C] 严格下三角，V [..., C, M]，返回 X [..., C, M]，fp32。M 必须 ≤ 1024（一个 threadgroup）。"""
    *lead, C, _ = A.shape
    M = V.shape[-1]
    if M > 1024:
        raise ValueError("one Metal threadgroup supports M <= 1024")
    DISPATCH_COUNTS["unit_lower_solve"] += 1
    N = 1
    for d in lead:
        N *= d
    A2 = A.astype(mx.float32).reshape(N, C, C)
    V2 = V.astype(mx.float32).reshape(N, C, M)
    (X,) = _solve_kernel(inputs=[A2, V2], grid=(M * N, 1, 1), threadgroup=(M, 1, 1), output_shapes=[(N, C, M)], output_dtypes=[mx.float32])
    return X.reshape(*lead, C, M)
