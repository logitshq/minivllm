"""Chapter 10: compact batch caches, offset rollback for attention, tape replay for GDN."""
import mlx.core as mx


class AttnCache:
    step = 256

    def __init__(self):
        self.keys = None
        self.values = None
        self.offset = 0
        self._snap = 0

    def snapshot(self):
        self._snap = self.offset

    def rollback(self, n: int):
        """只保留 verify 块的前 n 个 token：K/V 多写的几列留着，下次直接覆盖。"""
        self.offset = self._snap + n

    def update(self, k: mx.array, v: mx.array) -> tuple[mx.array, mx.array]:
        """k, v: [1, Hkv, T, D]。写入后返回 [1, Hkv, offset+T, D] 的完整历史。"""
        transaction = getattr(self, "_verify108", None)
        if transaction is not None:
            return transaction.attention_update(k, v, mx)
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
    """conv_state [K-1, C]，S [Hv, Dk, Dv] fp32。
    投机解码的回滚：verify 前 snapshot()，verify 时层把每个 token 的递推输入记在 tape 里，接受 n 个之后 rollback(n)
    从快照状态把这 n 个 token 重新递推一遍（"磁带回放"），conv_state 也从快照 + 前 n 个原始输入重建。"""

    def __init__(self):
        self.conv_state = None     # [K-1, C]
        self.S = None              # [Hv, Dk, Dv] fp32
        self.offset = 0
        self.record = False        # True 时 _prefill 会把 tape 存进来
        self.tape = None           # (raw_qkv [T, C] 卷积前, q, k, v, g, beta)
        self._snap = None

    def snapshot(self):
        self._snap = (self.conv_state, self.S, self.offset)
        self.record = True

    def rollback(self, n: int):
        """只保留 verify 块的前 n 个 token。"""
        from .kernels.gdn import gated_delta_rule_chunk
        if self._snap is None or self.tape is None:
            raise RuntimeError("snapshot and recorded verify inputs are required")
        conv0, S0, off0 = self._snap
        raw, q, k, v, g, beta = self.tape
        T = raw.shape[0]
        if not 0 <= n <= T:
            raise ValueError("rollback length must be within the verify block")
        if n < T:
            if n == 0:
                self.S = S0 if S0 is not None else mx.zeros_like(self.S)
            else:
                _, self.S = gated_delta_rule_chunk(q[:n], k[:n], v[:n], g[:n], beta[:n], S0)
            K1 = self.conv_state.shape[0]
            prev = mx.zeros((K1, raw.shape[1]), raw.dtype) if conv0 is None else conv0.astype(raw.dtype)
            self.conv_state = mx.concatenate([prev, raw[:n]])[-K1:]
            self.offset = off0 + n
        self.record, self.tape, self._snap = False, None, None


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

    def extract(self, row: int) -> list:
        """把第 row 行抽成单序列 cache 列表（AttnCache / GDNCache），供投机解码的 solo 模式用。"""
        out = []
        n = self.lens[row]
        for i, layer in enumerate(self.model.layers):
            l = self.layers[i]
            if layer.is_attn:
                c = AttnCache(); c.keys, c.values, c.offset = l.K[row:row + 1], l.V[row:row + 1], n
            else:
                c = GDNCache(); c.conv_state, c.S, c.offset = l.conv[row], l.S[row], n
            out.append(c)
        return out

    def clear(self) -> None:
        self.req_ids, self.lens, self.layers = [], [], [None] * len(self.model.layers)

    def arrays(self) -> list[mx.array]:
        out = []
        for l in self.layers:
            if l is not None:
                out += [l.K, l.V] if isinstance(l, BatchKV) else [l.conv, l.S]
        return out
