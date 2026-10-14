"""cache 的两种形态。

单序列（prefill 用，第 2 步的设计）：AttnCache 存 K/V [1, Hkv, N, D]，GDNCache 存卷积尾巴 [K-1, C] 和状态 S [Hv, Dk, Dv]。
一个请求在 prefill 阶段用自己的单序列 cache，prompt 算完后并进 decode batch。

批量（decode 用）：DecodeBatch 把 B 个在跑请求的 cache 排成紧凑的批量数组：
  每个注意力层  K, V [B, Hkv, L, D]   每行按绝对位置写（第 i 行的第 pos_i 列），用长度掩码读
  每个 GDN 层   conv [B, K-1, C]，S [B, Hv, Dk, Dv]   每步整体替换，没有写回
为什么不是"大池子 + 按槽散写"：MLX 的数组不可变，往 128 MB 的 buffer 里散写 8 行会把整个 buffer 拷一遍（实测 0.24 ms），
GDN 状态池 4.8 GB 每步散写一次就是 10 ms。紧凑数组每步只拷 K/V 一次（写一列），GDN 状态零拷贝；
代价是请求进出 batch 时要重排一次（gather + concatenate），每个请求一生只发生两次。
"""
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
    def __init__(self):
        self.conv_state = None     # [K-1, C]
        self.S = None              # [Hv, Dk, Dv] fp32
        self.offset = 0


def make_cache(model) -> list:
    return [AttnCache() if layer.is_attn else GDNCache() for layer in model.layers]


class BatchKV:
    """一个注意力层的批量 K/V。rows 顺序 = DecodeBatch.req_ids 的顺序。"""

    def __init__(self, K: mx.array, V: mx.array):
        self.K, self.V = K, V                  # [B, Hkv, L, D]

    def write(self, k: mx.array, v: mx.array, pos: mx.array) -> None:
        """k, v: [B, Hkv, D]；pos: [B] 每行写到自己的列。"""
        rows = mx.arange(k.shape[0])
        self.K[rows, :, pos] = k
        self.V[rows, :, pos] = v


class BatchState:
    """一个 GDN 层的批量状态。每步整体替换。"""

    def __init__(self, conv: mx.array, S: mx.array):
        self.conv, self.S = conv, S            # [B, K-1, C], [B, Hv, Dk, Dv]


class DecodeBatch:
    """在跑请求的批量 cache + 它们的长度。layers[i] 是 BatchKV 或 BatchState。"""
    step = 256

    def __init__(self, model):
        self.model = model
        self.req_ids: list[str] = []
        self.lens: list[int] = []              # 每行已在 cache 里的 token 数
        self.layers: list = [None] * len(model.layers)

    @property
    def size(self) -> int:
        return len(self.req_ids)

    def rebuild(self, keep: list[int], new: list[tuple[str, list, int]]) -> None:
        """一次重排：保留 keep 里的行（按给定顺序），再追加 new 里的请求 (req_id, 单序列 cache 列表, 需要的最大长度)。"""
        new_lens = [self.lens[i] for i in keep] + [c[0].offset if c[0] is not None else 0 for _, c, _ in new]
        # L 取所有行需要的最大长度，向上到 256 的倍数；不够的行补零
        need = max([n for _, _, n in new] + ([self.layers_len()] if keep else [0]))
        L = (need + self.step - 1) // self.step * self.step
        keep_idx = mx.array(keep) if keep else None
        for i, layer in enumerate(self.model.layers):
            old = self.layers[i]
            if layer.is_attn:
                rows_k, rows_v = [], []
                if keep:
                    oldK, oldV = old.K[keep_idx], old.V[keep_idx]
                    if oldK.shape[2] < L:
                        pad = [(0, 0), (0, 0), (0, L - oldK.shape[2]), (0, 0)]
                        oldK, oldV = mx.pad(oldK, pad), mx.pad(oldV, pad)
                    rows_k.append(oldK[:, :, :L]); rows_v.append(oldV[:, :, :L])
                for _, caches, _ in new:
                    c = caches[i]
                    k, v = c.keys[:, :, :c.offset], c.values[:, :, :c.offset]
                    pad = [(0, 0), (0, 0), (0, L - c.offset), (0, 0)]
                    rows_k.append(mx.pad(k, pad)); rows_v.append(mx.pad(v, pad))
                self.layers[i] = BatchKV(mx.concatenate(rows_k), mx.concatenate(rows_v))
            else:
                convs = [old.conv[keep_idx]] if keep else []
                Ss = [old.S[keep_idx]] if keep else []
                for _, caches, _ in new:
                    c = caches[i]
                    convs.append(c.conv_state[None]); Ss.append(c.S[None])
                self.layers[i] = BatchState(mx.concatenate(convs), mx.concatenate(Ss))
        self.req_ids = [self.req_ids[i] for i in keep] + [rid for rid, _, _ in new]
        self.lens = new_lens

    def layers_len(self) -> int:
        for l in self.layers:
            if isinstance(l, BatchKV):
                return l.K.shape[2]
        return 0

    def clear(self) -> None:
        self.req_ids, self.lens, self.layers = [], [], [None] * len(self.model.layers)

    def arrays(self) -> list[mx.array]:
        out = []
        for l in self.layers:
            if l is not None:
                out += [l.K, l.V] if isinstance(l, BatchKV) else [l.conv, l.S]
        return out
