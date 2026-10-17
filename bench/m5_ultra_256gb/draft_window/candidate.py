"""E014 scoped benchmark prototype. No production module installs this patch."""
from contextlib import contextmanager

import mlx.core as mx

from tutorial_code.miniengine.model.layers import apply_rope
from tutorial_code.miniengine.spec.dflash2 import DraftAttention

ORIGINAL = DraftAttention.__call__


def bounded_attention(self, x, ctx_new, start):
    # Keep projection input shapes and arithmetic identical to the original.
    if ctx_new is not None and ctx_new.shape[0] > 0:
        k, v = self._kv(ctx_new, start - ctx_new.shape[0])
        self.ctx_k = k if self.ctx_k is None else mx.concatenate([self.ctx_k, k], axis=2)
        self.ctx_v = v if self.ctx_v is None else mx.concatenate([self.ctx_v, v], axis=2)
        self.ctx_len += ctx_new.shape[0]  # Logical position is independent of residency.
    # Amortize compaction: at most 2W resident rows after a call. Append can
    # transiently use 2W + incoming rows; the next block still reads only W-L.
    if self.ctx_k is not None and self.ctx_k.shape[2] > 2 * self.window:
        self.ctx_k = mx.contiguous(self.ctx_k[:, :, -self.window:])
        self.ctx_v = mx.contiguous(self.ctx_v[:, :, -self.window:])
    L = x.shape[0]
    q = apply_rope(self.q_norm(self.q_proj(x).reshape(L, self.H, self.D)).transpose(1, 0, 2)[None], start, self.D, self.theta)
    kb, vb = self._kv(x, start)
    lo = max(0, self.ctx_k.shape[2] - self.window + L)
    k = mx.concatenate([self.ctx_k[:, :, lo:], kb], axis=2)
    v = mx.concatenate([self.ctx_v[:, :, lo:], vb], axis=2)
    o = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.D ** -0.5, mask=None)
    return self.o_proj(o[0].transpose(1, 0, 2).reshape(L, -1))


@contextmanager
def enabled(bounded=False):
    """A mode owns its cache trajectory: switching back cannot recover dropped rows."""
    if type(bounded) is not bool:
        raise TypeError('bounded flag must be bool')
    previous = DraftAttention.__call__
    DraftAttention.__call__ = bounded_attention if bounded else ORIGINAL
    try:
        yield
    finally:
        DraftAttention.__call__ = previous
