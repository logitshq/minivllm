"""Chapter 06: single-request KV history, GDN conv tail and recurrent state. No batching or rollback yet."""
import mlx.core as mx


class AttnCache:
    step = 256

    def __init__(self):
        self.keys = None
        self.values = None
        self.offset = 0

    def update(self, k: mx.array, v: mx.array) -> tuple[mx.array, mx.array]:
        """k, v: [1, Hkv, T, D]。写入后返回 [1, Hkv, offset+T, D] 的完整历史。"""
        T = k.shape[2]
        end = self.offset + T
        if self.keys is None or end > self.keys.shape[2]:
            n = (end + self.step - 1) // self.step * self.step
            B, H, _, D = k.shape
            new_k, new_v = mx.zeros((B, H, n, D), k.dtype), mx.zeros((B, H, n, D), v.dtype)
            if self.keys is not None:
                new_k[:, :, : self.offset] = self.keys[:, :, : self.offset]
                new_v[:, :, : self.offset] = self.values[:, :, : self.offset]
            self.keys, self.values = new_k, new_v
        self.keys[:, :, self.offset:end] = k
        self.values[:, :, self.offset:end] = v
        self.offset = end
        return self.keys[:, :, :end], self.values[:, :, :end]


class GDNCache:
    """A fixed-size conv tail and fp32 state, independent for each request."""

    def __init__(self):
        self.conv_state = None     # [K-1, C]
        self.S = None              # [Hv, Dk, Dv] fp32
        self.offset = 0


def make_cache(model) -> list:
    return [AttnCache() if layer.is_attn else GDNCache() for layer in model.layers]


