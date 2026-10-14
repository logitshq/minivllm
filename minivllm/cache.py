"""单条序列的 cache（第 2 步）。每一层一个对象：注意力层存 K/V，GDN 层存卷积尾巴和递推状态。

AttnCache：K/V 预分配成 [1, Hkv, N, D]，N 按 256 为步长增长，新 token 按位置写入（slice 赋值），
           attention 只看 [:offset + T]。这和 mlx-lm 的 KVCache 一样；vLLM 的 paged cache 是它的推广（第 4 步）。
GDNCache： conv_state [K-1, C] 是最近 K-1 个"卷积输入"（投影之后、卷积之前），S [Hv, Dk, Dv] fp32 是递推状态。
           两者加起来就是 GDN 层关于整个历史的全部信息，大小不随序列长度变。
"""
import mlx.core as mx


class AttnCache:
    step = 256

    def __init__(self):
        self.keys = None
        self.values = None
        self.offset = 0            # 已经在 cache 里的 token 数

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
    def __init__(self):
        self.conv_state = None     # [K-1, C]
        self.S = None              # [Hv, Dk, Dv] fp32
        self.offset = 0


def make_cache(model) -> list:
    return [AttnCache() if layer.is_attn else GDNCache() for layer in model.layers]
