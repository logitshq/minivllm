"""Chapter 04: explicit fp32 Gated DeltaNet recurrence. State decays before prediction."""
import mlx.core as mx


def l2norm(x: mx.array, eps: float = 1e-6) -> mx.array:
    return x * mx.rsqrt((x * x).sum(-1, keepdims=True) + eps)


def gated_delta_rule_recurrent(q, k, v, g, beta, state=None):
    T, Hv, Dk = k.shape
    Dv = v.shape[-1]
    q, k, v = (t.astype(mx.float32) for t in (q, k, v))
    q, k = l2norm(q), l2norm(k)
    q = q * Dk ** -0.5
    S = mx.zeros((Hv, Dk, Dv), dtype=mx.float32) if state is None else state
    outs = []
    for t in range(T):
        S = S * mx.exp(g[t])[:, None, None]
        kv_mem = (S * k[t][:, :, None]).sum(axis=1)             # [Hv, Dv]
        delta = (v[t] - kv_mem) * beta[t][:, None]
        S = S + k[t][:, :, None] * delta[:, None, :]
        outs.append((S * q[t][:, :, None]).sum(axis=1))
    return mx.stack(outs), S


