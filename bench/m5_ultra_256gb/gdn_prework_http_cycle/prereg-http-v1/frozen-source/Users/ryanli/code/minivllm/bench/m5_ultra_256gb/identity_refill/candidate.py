"""E028 bench-only identity refill; no MLX import, installation, or evaluation.

The eligible rebuild is cache.py (SHA-256 below) with only the old-row gathers
and their index allocation removed. Slices, padding, concatenation, metadata,
and the original superclass fallback remain. This is not device qualification:
lazy graph/backing lifetimes still need the independent OFF/ON screen.
"""

REFERENCE_CACHE_SHA256 = "056a3bf3bc42163e93f55eb914fa35d5c081f803da9b5faa740588a67057a791"


def make_batch_class(cache_module, mx_module):
    """Return a per-instance, default-OFF DecodeBatch subclass; patch nothing.

    The caller supplies the already loaded cache and matching MLX modules.
    ``counters`` counts rebuild attempts, including failed superclass calls;
    ``last_path`` and ``last_reason`` are Python branch witnesses, not device
    execution or measured work. Toggle ``enabled`` only between rebuild calls.
    """
    if cache_module.mx is not mx_module:
        raise ValueError("cache and candidate must use the same array module")
    mx = mx_module

    class IdentityRefillBatch(cache_module.DecodeBatch):
        def __init__(self, model, enabled=False):
            if type(enabled) is not bool:
                raise TypeError("enabled must be bool")
            super().__init__(model)
            self.enabled = enabled
            self.counters = {"identity_refill": 0, "fallback": 0}
            self.last_path = None
            self.last_reason = None

        def _identity_layout(self, keep, new):
            """Route/shape guard; retain DecodeBatch's valid producer contract.

            In particular, do not rescan per-layer new dtypes, offsets or IDs.
            Those and value validity remain inputs to the original rebuild.
            """
            if self.enabled is not True:
                return None, "disabled"
            B = self.size
            if not B or not new:
                return None, "empty_old_or_new"
            if (type(keep) is not list or len(keep) != B
                    or any(type(k) is not int or k != i for i, k in enumerate(keep))):
                return None, "nonidentity_keep"
            try:
                if (type(new) is not list or len(self.lens) != B or len(self.layers) != len(self.model.layers)
                        or not self.layers or any(len(c) != len(self.layers) for _, c, _ in new)):
                    return None, "layout"
                need = max([n for _, _, n in new] + [self.layers_len()])
                L = (need + self.step - 1) // self.step * self.step
                for old, layer in zip(self.layers, self.model.layers):
                    if layer.is_attn:
                        if (not isinstance(old, cache_module.BatchKV) or len(old.K.shape) != 4
                                or old.K.shape[0] != B or old.V.shape != old.K.shape):
                            return None, "layout"
                        if old.K.shape[2] != L:
                            return None, "capacity_changed"
                    elif (not isinstance(old, cache_module.BatchState)
                          or len(old.conv.shape) != 3 or len(old.S.shape) != 4
                          or old.conv.shape[0] != B or old.S.shape[0] != B):
                        return None, "layout"
                return L, "identity_keep_unchanged_capacity"
            except (AttributeError, TypeError, ValueError, IndexError):
                return None, "layout"

        def rebuild(self, keep, new):
            L, reason = self._identity_layout(keep, new)
            self.last_reason = reason
            if L is None:
                self.last_path = "fallback"
                self.counters["fallback"] += 1
                return super().rebuild(keep, new)
            self.last_path = "identity_refill"
            self.counters["identity_refill"] += 1
            new_lens = [self.lens[i] for i in keep] + [c[0].offset if c[0] is not None else 0 for _, c, _ in new]
            for i, layer in enumerate(self.model.layers):
                old = self.layers[i]
                if layer.is_attn:
                    rows_k, rows_v = [], []
                    oldK, oldV = old.K, old.V  # Sole attention change: no identity gather.
                    if oldK.shape[2] < L:
                        pad = [(0, 0), (0, 0), (0, L - oldK.shape[2]), (0, 0)]
                        oldK, oldV = mx.pad(oldK, pad), mx.pad(oldV, pad)
                    rows_k.append(oldK[:, :, :L]); rows_v.append(oldV[:, :, :L])
                    for _, caches, _ in new:
                        c = caches[i]
                        k, v = c.keys[:, :, :c.offset], c.values[:, :, :c.offset]
                        pad = [(0, 0), (0, 0), (0, L - c.offset), (0, 0)]
                        rows_k.append(mx.pad(k, pad)); rows_v.append(mx.pad(v, pad))
                    self.layers[i] = cache_module.BatchKV(mx.concatenate(rows_k), mx.concatenate(rows_v))
                else:
                    convs, Ss = [old.conv], [old.S]  # Sole GDN change: no identity gathers.
                    for _, caches, _ in new:
                        c = caches[i]
                        convs.append(c.conv_state[None]); Ss.append(c.S[None])
                    self.layers[i] = cache_module.BatchState(mx.concatenate(convs), mx.concatenate(Ss))
            self.req_ids = [self.req_ids[i] for i in keep] + [rid for rid, _, _ in new]
            self.lens = new_lens

    return IdentityRefillBatch
