"""V108 opt-in, bounded single-invocation prefix cache transaction.

No MLX import at module import. Prefix GDN arrays are literal old44 helper
results; attention prefixes copy the same invocation's already-projected K/V.
"""
from itertools import count

_IDS = count(1)


def select_delivery(drafts, posterior, budget, eos_ids=(), stop_ids=(), ignore_eos=False):
    if type(budget) is not int or budget < 1:
        raise ValueError("positive remaining output budget required")
    if len(posterior) != len(drafts) + 1:
        raise ValueError("one target prediction per anchor/draft input required")
    accepted = 0
    while accepted < len(drafts) and drafts[accepted] == posterior[accepted]:
        accepted += 1
    produced = list(drafts[:accepted]) + [posterior[accepted]]
    delivered = produced[:budget]
    if not ignore_eos:
        stops = set(eos_ids) | set(stop_ids)
        for i, token in enumerate(delivered):
            if token in stops:
                delivered = delivered[:i + 1]
                break
    # Frontier invariant: all delivered except the last are cached. The old
    # anchor is also cached, so the verified input count equals delivery count.
    return delivered, accepted, len(delivered)


class PrefixTrace:
    def __init__(self, cache, length, owner):
        self.cache, self.length, self.owner = cache, length, owner
        self.start_offset = cache.offset
        self.kind = "attention" if hasattr(cache, "keys") else "gdn"
        self.producer_owners = []
        self.updates = 0
        if self.kind == "attention":
            self.first = [cache.keys, cache.values]
        else:
            self.first = [cache.conv_state, cache.S]
        self.history = [tuple(self.first)]

    def record_gdn(self, conv, state, producer_conv, producer_state):
        if self.kind != "gdn" or len(self.history) > self.length:
            raise RuntimeError("wrong/repeated GDN prefix producer")
        self.history.append((conv, state))
        self.producer_owners.append((producer_conv, producer_state))
        self.updates += 1

    def attention_update(self, keys, values, mx):
        if self.kind != "attention" or self.updates or keys.shape[2] != self.length or keys.shape != values.shape:
            raise RuntimeError("one complete batched K/V update required per transaction")
        cache, off = self.cache, self.start_offset
        old_k, old_v = self.first
        capacity = old_k.shape[2] if old_k is not None else cache.step
        if off + self.length > capacity:
            raise ValueError("verify inputs exceed fixed attention capacity")
        shape = (keys.shape[0], keys.shape[1], capacity, keys.shape[3])
        new_k, new_v = mx.zeros(shape, keys.dtype), mx.zeros(shape, values.dtype)
        if off:
            if old_k is None or old_v is None:
                raise RuntimeError("nonempty attention offset without K/V")
            new_k[:, :, :off, :] = old_k[:, :, :off, :]
            new_v[:, :, :off, :] = old_v[:, :, :off, :]
        self.producer_owners.append((keys, values))
        for i in range(self.length):
            # Separate Python array handles prevent later __setitem__ graph
            # replacement from mutating an earlier witness handle.
            next_k, next_v = new_k[:], new_v[:]
            next_k[:, :, off + i:off + i + 1, :] = keys[:, :, i:i + 1, :]
            next_v[:, :, off + i:off + i + 1, :] = values[:, :, i:i + 1, :]
            self.history.append((next_k, next_v))
            new_k, new_v = next_k, next_v
        self.updates = 1
        cache.keys, cache.values, cache.offset = new_k, new_v, off + self.length
        return new_k[:, :, :cache.offset, :], new_v[:, :, :cache.offset, :]

    def complete(self):
        return len(self.history) == self.length + 1 and (self.updates == self.length if self.kind == "gdn" else self.updates == 1)

    def witness(self, keep):
        if type(keep) is not int or not 0 <= keep <= self.length:
            raise ValueError("prefix count outside verify inputs")
        if not self.complete():
            raise RuntimeError("incomplete same-invocation prefix history")
        return self.history[keep], self.start_offset + keep

    def apply(self, keep):
        values = self.history[keep]
        if self.kind == "attention":
            self.cache.keys, self.cache.values = values
        else:
            self.cache.conv_state, self.cache.S = values
        self.cache.offset = self.start_offset + keep


class VerifyTransaction:
    def __init__(self, caches, length):
        if type(length) is not int or not 1 <= length <= 8:
            raise ValueError("V108 verify length must be 1..8")
        if not caches or any(getattr(c, "_verify108", None) is not None or getattr(c, "record", False) for c in caches):
            raise RuntimeError("nested/legacy-recorded transaction forbidden")
        if len({id(c) for c in caches}) != len(caches):
            raise ValueError("aliased cache objects")
        if len({c.offset for c in caches}) != 1:
            raise ValueError("cache frontiers differ")
        self.id, self.length, self.closed, self.keep = next(_IDS), length, False, None
        self.traces = [PrefixTrace(c, length, self.id) for c in caches]
        for trace in self.traces:
            trace.cache._verify108 = trace
            if trace.kind == "gdn":
                trace.cache.record = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if not self.closed:
            self.abort()

    def witness(self, keep):
        return [trace.witness(keep) for trace in self.traces]

    def eval_arrays(self):
        # Request every literal history/producer owner as an eval output in
        # the same boundary. No copy or recomputed substitute defines gold.
        values = [a for trace in self.traces for pair in trace.history + trace.producer_owners for a in pair if a is not None]
        return list({id(a): a for a in values}.values())

    def _detach(self):
        for trace in self.traces:
            trace.cache._verify108 = None
            if trace.kind == "gdn":
                trace.cache.record, trace.cache.tape, trace.cache._snap = False, None, None

    def commit(self, keep):
        if self.closed:
            raise RuntimeError("transaction already closed")
        self.witness(keep)  # Validate every producer before mutating any cache.
        for trace in self.traces:
            trace.apply(keep)
        self.keep, self.closed = keep, True
        self._detach()

    def abort(self):
        if self.closed:
            raise RuntimeError("transaction already closed")
        for trace in self.traces:
            trace.apply(0)
        self.keep, self.closed = 0, True
        self._detach()


def fused_recurrent_history(raw, g, beta, weights, cache, mx):
    """One verify invocation; exact44 conv/state helpers, no chunk replay."""
    from ..kernels.gdn_conv_silu_metal import conv_silu_raw_tail
    from ..kernels.gdn_activated_norm_reuse_cached_metal import gated_delta_step_activated_norm_reuse_metal
    trace = getattr(cache, "_verify108", None)
    length = raw.shape[0]
    if trace is None or trace.kind != "gdn" or trace.length != length or not cache.record:
        raise RuntimeError("active GDN transaction required")
    if not (1 <= length <= 8 and raw.shape == (length, 10240) and raw.dtype == mx.bfloat16
            and weights.shape == (10240, 4) and weights.dtype == mx.bfloat16
            and g.shape == beta.shape == (length, 48) and g.dtype == beta.dtype == mx.float32):
        raise ValueError("V108 requires fixed27B BF16 raw/weights and FP32 g/beta")
    conv = mx.zeros((1, 3, 10240), mx.bfloat16) if cache.conv_state is None else cache.conv_state[None]
    state = mx.zeros((1, 48, 128, 128), mx.float32) if cache.S is None else cache.S[None]
    if conv.shape != (1, 3, 10240) or state.shape != (1, 48, 128, 128) or state.dtype != mx.float32 or conv.dtype != mx.bfloat16:
        raise ValueError("V108 requires canonical44 state layout")
    outputs = []
    for t in range(length):
        activated, next_conv = conv_silu_raw_tail(raw[t:t + 1], conv, weights)
        output, next_state = gated_delta_step_activated_norm_reuse_metal(activated, g[t:t + 1], beta[t:t + 1], state)
        trace.record_gdn(next_conv[0], next_state[0], next_conv, next_state)
        outputs.append(output)
        conv, state = next_conv, next_state
    cache.conv_state, cache.S = trace.history[-1]
    cache.offset = trace.start_offset + length
    return mx.concatenate(outputs, axis=0)
