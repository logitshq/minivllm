"""Chapter 06: keep convolution tail and GDN state across chunks and single-token steps."""
import mlx.core as mx
import mlx.nn as nn
from functools import lru_cache

from ..config import ModelConfig
from ..kernels.gdn import gated_delta_rule_chunk, gated_delta_rule_recurrent, gated_delta_step_batch
from .layers import GatedRMSNorm, Linear, prefill_linear, prefill_batch_backend_supported, _prefill_backend_identity
from ..cache import BatchState
from ..context import ForwardContext



@lru_cache(maxsize=1)
def _m5_ultra_decode_identity():
    return (_prefill_backend_identity() == ("0.32.3", "applegpu_g17d")
            and mx.device_info().get("device_name") == "Apple M5 Ultra")


def gdn_decode_backend_supported():
    """Opt-in decode qualification domain; does not enable NAX batched prefill."""
    return (mx.default_device() == mx.gpu
            and (prefill_batch_backend_supported() or _m5_ultra_decode_identity()))


def causal_conv1d(x: mx.array, w: mx.array, state: mx.array | None = None) -> tuple[mx.array, mx.array]:
    """x: [T, C]，w: [C, K]，state: 之前的 K-1 个输入 [K-1, C]（None = 补 0，即序列开头）。
    y[t] = sum_j w[:, j] * x[t - (K-1) + j]。等价于 HF 的 conv1d(padding=K-1) 取前 T 个。
    返回 y 和新的 state（最后 K-1 个输入），decode 时 T=1 也走这条路。"""
    K = w.shape[1]
    T = x.shape[0]
    xp = mx.concatenate([mx.zeros((K - 1, x.shape[1]), x.dtype) if state is None else state.astype(x.dtype), x])
    y = sum(xp[j:j + T] * w[:, j] for j in range(K))
    return y, xp[T:]


def _gates_explicit(b, a, A_log, dt_bias):
    beta = mx.sigmoid(b.astype(mx.float32))
    g = -mx.exp(A_log) * nn.softplus(a.astype(mx.float32) + dt_bias)
    return beta, g


class GatedDeltaNet(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.Hk, self.Hv = cfg.linear_num_key_heads, cfg.linear_num_value_heads
        self.Dk, self.Dv = cfg.linear_key_head_dim, cfg.linear_value_head_dim
        key_dim, value_dim = self.Hk * self.Dk, self.Hv * self.Dv
        self.conv_dim = 2 * key_dim + value_dim
        q = cfg.quantization
        self.in_proj_qkv = Linear(cfg.hidden_size, self.conv_dim, q)
        self.in_proj_z = Linear(cfg.hidden_size, value_dim, q)
        self.in_proj_b = Linear(cfg.hidden_size, self.Hv, q)
        self.in_proj_a = Linear(cfg.hidden_size, self.Hv, q)
        self.conv1d_weight = mx.zeros((self.conv_dim, cfg.linear_conv_kernel_dim), dtype=mx.bfloat16)
        self.A_log = mx.zeros((self.Hv,), dtype=mx.float32)
        self.dt_bias = mx.zeros((self.Hv,), dtype=mx.float32)
        self.norm = GatedRMSNorm(self.Dv, cfg.rms_norm_eps)
        self.out_proj = Linear(value_dim, cfg.hidden_size, q)
        self.set_gates_compile(False)
        self.set_norm_reuse(False)

    def __call__(self, x: mx.array, ctx=None, layer_idx: int = 0, *, _norm_reuse_normal=False, _projection=None) -> mx.array:
        if ctx is not None and ctx.mode == "prefill_batch":
            return self._prefill_batch(x, ctx, layer_idx)
        if ctx is not None and ctx.mode == "decode":
            allow_compile = (type(ctx) is ForwardContext and
                             not any(getattr(ctx, key, False) for key in ("record", "capture", "speculative", "verify")))
            if _projection is not None:
                return self._decode_fused(x, ctx.caches[layer_idx], _projection, allow_compile)
            return self._decode(x, ctx.caches[layer_idx], _compile_gates=allow_compile,
                                _norm_reuse_normal=(_norm_reuse_normal is True and allow_compile))
        return self._prefill(x, ctx.caches[layer_idx] if ctx is not None else None)

    def _gates(self, b, a):
        beta = mx.sigmoid(b.astype(mx.float32))
        g = -mx.exp(self.A_log) * nn.softplus(a.astype(mx.float32) + self.dt_bias)
        return beta, g

    def set_gates_compile(self, enabled=True):
        """Experimental opt-in only; private Python metadata adds no array parameters."""
        if type(enabled) is not bool:
            raise TypeError("enabled must be bool")
        object.__setattr__(self, "_gates_compile_enabled", enabled)
        object.__setattr__(self, "_compiled_decode_gates", None)

    def _decode_gates(self, b, a, x, st):
        # All weights are explicit inputs, so replacement cannot leave captured weights.
        eligible = (object.__getattribute__(self, "_gates_compile_enabled")
                    and type(st) is BatchState and not any(getattr(st, key, False) for key in ("record", "capture", "speculative", "verify"))
                    and self.Hv == 48 and x.ndim == 2 and x.shape == (1, 5120)
                    and x.dtype == mx.bfloat16
                    and b.shape == a.shape == (1, 48) and b.dtype == a.dtype == mx.bfloat16
                    and self.A_log.shape == self.dt_bias.shape == (48,)
                    and self.A_log.dtype == self.dt_bias.dtype == mx.float32
                    and gdn_decode_backend_supported())
        if eligible:
            for op in (self.in_proj_b, self.in_proj_a):
                if not (type(op) is Linear and op.q == {"group_size": 16, "bits": 4, "mode": "nvfp4"}
                        and op.in_features == 5120 and op.out_features == 48
                        and op.weight.shape == (48, 640) and op.weight.dtype == mx.uint32
                        and op.scales.shape == (48, 320) and op.scales.dtype == mx.uint8
                        and getattr(op, "biases", None) is None):
                    eligible = False
                    break
        if not eligible:
            return self._gates(b, a)
        fn = object.__getattribute__(self, "_compiled_decode_gates")
        if fn is None:
            fn = mx.compile(_gates_explicit, shapeless=False)
            object.__setattr__(self, "_compiled_decode_gates", fn)
        return fn(b, a, self.A_log, self.dt_bias)

    def set_norm_reuse(self, enabled=True):
        """Independent default-off experiment; metadata adds no MLX parameters."""
        if type(enabled) is not bool:
            raise TypeError("enabled must be bool")
        object.__setattr__(self, "_norm_reuse_enabled", enabled)

    def _norm_reuse_eligible(self, u, g, beta, x, st, normal):
        # Every rejection precedes st.conv/st.S reassignment. Unknown metadata fails closed.
        return (normal is True and type(self) is GatedDeltaNet
                and object.__getattribute__(self, "_norm_reuse_enabled")
                and type(st) is BatchState and set(vars(st)) == {"conv", "S"}
                and (self.Hk, self.Hv, self.Dk, self.Dv, self.conv_dim) == (16, 48, 128, 128, 10240)
                and x.shape == (1, 5120) and x.dtype == mx.bfloat16
                and u.shape == (1, 10240) and u.dtype == mx.bfloat16
                and self.conv1d_weight.shape == (10240, 4) and self.conv1d_weight.dtype == mx.bfloat16
                and st.conv.shape == (1, 3, 10240) and st.conv.dtype == mx.bfloat16
                and st.S.shape == (1, 48, 128, 128) and st.S.dtype == mx.float32
                and g.shape == beta.shape == (1, 48) and g.dtype == beta.dtype == mx.float32
                and gdn_decode_backend_supported())

    def _split_heads(self, qkv, n):
        q, k, v = mx.split(qkv, [self.Hk * self.Dk, 2 * self.Hk * self.Dk], axis=-1)
        q, k, v = q.reshape(n, self.Hk, self.Dk), k.reshape(n, self.Hk, self.Dk), v.reshape(n, self.Hv, self.Dv)
        if self.Hv > self.Hk:                                        # 27B：16 个 k 头服务 48 个 v 头
            rep = self.Hv // self.Hk
            q, k = mx.repeat(q, rep, axis=1), mx.repeat(k, rep, axis=1)
        return q, k, v

    def _prefill(self, x: mx.array, cache) -> mx.array:
        """单序列：x [T, hidden]。T=1 走递推，否则走分块；状态接 cache。"""
        T = x.shape[0]
        qkv = self.in_proj_qkv(x)
        z = self.in_proj_z(x).reshape(T, self.Hv, self.Dv)
        conv_state = cache.conv_state if cache is not None and cache.offset > 0 else None
        raw = qkv
        qkv, new_conv_state = causal_conv1d(qkv, self.conv1d_weight.astype(qkv.dtype), conv_state)
        q, k, v = self._split_heads(nn.silu(qkv), T)
        beta, g = self._gates(self.in_proj_b(x), self.in_proj_a(x))
        if cache is not None and cache.record:
            cache.tape = (raw, q, k, v, g, beta)
        S0 = cache.S if cache is not None and cache.offset > 0 else None
        if T == 1:
            o, S = gated_delta_rule_recurrent(q, k, v, g, beta, S0)                 # decode：递推
        else:
            chunk = 64 if T > 32 else (16 if T > 8 else 8)                          # 投机解码 verify 的 8 个 token 用小块
            o, S = gated_delta_rule_chunk(q, k, v, g, beta, S0, chunk=chunk)
        if cache is not None:
            cache.conv_state, cache.S, cache.offset = new_conv_state, S, cache.offset + T
        o = self.norm(o.astype(x.dtype), z)
        return self.out_proj(o.reshape(T, -1))


    def _prefill_batch(self, x: mx.array, ctx, layer_idx: int) -> mx.array:
        """Fold only independent row/head axes; preserve each sequence's time axis."""
        B, T = ctx.prefill_shape
        caches = ctx.caches[layer_idx]
        raw = prefill_linear(self.in_proj_qkv, x, (B, T)).reshape(B, T, self.conv_dim)
        z = prefill_linear(self.in_proj_z, x, (B, T)).reshape(B * T, self.Hv, self.Dv)
        w = self.conv1d_weight.astype(raw.dtype)
        convolved, tails = [], []
        for row in range(B):
            y, tail = causal_conv1d(raw[row], w)
            convolved.append(y); tails.append(tail)
        q, k, v = self._split_heads(nn.silu(mx.stack(convolved).reshape(B * T, self.conv_dim)), B * T)
        q = q.reshape(B, T, self.Hv, self.Dk).transpose(1, 0, 2, 3).reshape(T, B * self.Hv, self.Dk)
        k = k.reshape(B, T, self.Hv, self.Dk).transpose(1, 0, 2, 3).reshape(T, B * self.Hv, self.Dk)
        v = v.reshape(B, T, self.Hv, self.Dv).transpose(1, 0, 2, 3).reshape(T, B * self.Hv, self.Dv)
        beta, g = self._gates(prefill_linear(self.in_proj_b, x, (B, T)), prefill_linear(self.in_proj_a, x, (B, T)))
        beta = beta.reshape(B, T, self.Hv).transpose(1, 0, 2).reshape(T, B * self.Hv)
        g = g.reshape(B, T, self.Hv).transpose(1, 0, 2).reshape(T, B * self.Hv)
        if T == 1:
            o, state = gated_delta_rule_recurrent(q, k, v, g, beta)
        else:
            chunk = 64 if T > 32 else (16 if T > 8 else 8)
            o, state = gated_delta_rule_chunk(q, k, v, g, beta, chunk=chunk)
        state = state.reshape(B, self.Hv, self.Dk, self.Dv)
        for row, cache in enumerate(caches):
            cache.conv_state, cache.S, cache.offset = tails[row], state[row], T
        o = o.reshape(T, B, self.Hv, self.Dv).transpose(1, 0, 2, 3).reshape(B * T, self.Hv, self.Dv)
        o = self.norm(o.astype(x.dtype), z)
        return prefill_linear(self.out_proj, o.reshape(B * T, -1), (B, T))


    def _norm_reuse_initial_eligible(self, x, st, normal):
        return (normal is True and type(self) is GatedDeltaNet
                and object.__getattribute__(self, "_norm_reuse_enabled")
                and type(st) is BatchState and set(vars(st)) == {"conv", "S"}
                and (self.Hk, self.Hv, self.Dk, self.Dv, self.conv_dim) == (16, 48, 128, 128, 10240)
                and x.shape == (1, 5120) and x.dtype == mx.bfloat16
                and self.conv1d_weight.shape == (10240, 4) and self.conv1d_weight.dtype == mx.bfloat16
                and st.conv.shape == (1, 3, 10240) and st.conv.dtype == mx.bfloat16
                and st.S.shape == (1, 48, 128, 128) and st.S.dtype == mx.float32
                and gdn_decode_backend_supported())

    def _decode(self, x: mx.array, st, *, _compile_gates=False, _norm_reuse_normal=False) -> mx.array:
        """批量：x [B, hidden]；st.conv [B, K-1, C]，st.S [B, Hv, Dk, Dv]，整体替换。"""
        if self._norm_reuse_initial_eligible(x, st, _norm_reuse_normal):
            return self._decode_norm_reuse(x, st, _compile_gates=_compile_gates, _norm_reuse_normal=True)
        B = x.shape[0]
        qkv = self.in_proj_qkv(x)                                   # [B, C]
        z = self.in_proj_z(x).reshape(B, self.Hv, self.Dv)
        w = self.conv1d_weight.astype(qkv.dtype)                    # [C, K]
        xp = mx.concatenate([st.conv.astype(qkv.dtype), qkv[:, None]], axis=1)   # [B, K, C]
        qkv = (xp * w.T[None]).sum(axis=1)                          # 因果卷积在最后一个位置的值
        q, k, v = self._split_heads(nn.silu(qkv), B)
        if _compile_gates and object.__getattribute__(self, "_gates_compile_enabled"):
            beta, g = self._decode_gates(self.in_proj_b(x), self.in_proj_a(x), x, st)
        else:
            beta, g = self._gates(self.in_proj_b(x), self.in_proj_a(x))
        from ..kernels.gdn_metal import gated_delta_step_metal
        o, S = gated_delta_step_metal(q, k, v, g, beta, st.S)
        st.conv, st.S = xp[:, 1:], S
        o = self.norm(o.astype(x.dtype), z)
        return self.out_proj(o.reshape(B, -1))

    def _decode_norm_reuse(self, x: mx.array, st, *, _compile_gates=False, _norm_reuse_normal=False) -> mx.array:
        """批量：x [B, hidden]；st.conv [B, K-1, C]，st.S [B, Hv, Dk, Dv]，整体替换。"""
        B = x.shape[0]
        qkv = self.in_proj_qkv(x)                                   # [B, C]
        z = self.in_proj_z(x).reshape(B, self.Hv, self.Dv)
        w = self.conv1d_weight.astype(qkv.dtype)                    # [C, K]
        xp = mx.concatenate([st.conv.astype(qkv.dtype), qkv[:, None]], axis=1)   # [B, K, C]
        qkv = (xp * w.T[None]).sum(axis=1)                          # 因果卷积在最后一个位置的值
        if _compile_gates and object.__getattribute__(self, "_gates_compile_enabled"):
            beta, g = self._decode_gates(self.in_proj_b(x), self.in_proj_a(x), x, st)
        else:
            beta, g = self._gates(self.in_proj_b(x), self.in_proj_a(x))
        if self._norm_reuse_eligible(qkv, g, beta, x, st, _norm_reuse_normal):
            from ..kernels.gdn_norm_reuse_metal import gated_delta_step_norm_reuse_metal
            o, S = gated_delta_step_norm_reuse_metal(qkv, g, beta, st.S)
        else:
            q, k, v = self._split_heads(nn.silu(qkv), B)
            from ..kernels.gdn_metal import gated_delta_step_metal
            o, S = gated_delta_step_metal(q, k, v, g, beta, st.S)
        st.conv, st.S = xp[:, 1:], S
        o = self.norm(o.astype(x.dtype), z)
        return self.out_proj(o.reshape(B, -1))

    def _decode_fused(self, x, st, projection, compile_gates):
        from ..kernels.gdn_decode_fused import fused
        qkv, z, b, a = projection(x)
        beta, g = self._decode_gates(b, a, x, st) if compile_gates else self._gates(b, a)
        o, state, conv = fused(qkv, st.conv, self.conv1d_weight, g, beta, st.S)
        st.conv, st.S = conv, state
        z = z.reshape(1, self.Hv, self.Dv)
        return projection.linear(self.out_proj, self.norm(o.astype(x.dtype), z).reshape(1, -1))
