"""Standalone exact-L prototype. Imports stdlib only; the GPU owner injects MLX.

No engine installation or monkeypatching occurs. The caller must keep model
weights/configuration fixed until invalidate()/replace_model(), and call from
one worker. This wrapper caches functional compiled graphs, not Metal ICBs.
"""
from collections import OrderedDict
from dataclasses import dataclass
from time import perf_counter_ns


@dataclass(frozen=True)
class DecodeTypes:
    model: type
    context: type
    kv: type
    state: type


class IneligibleDecode(ValueError):
    pass


class FunctionalDecode:
    def __init__(self, model, *, backend, types, runtime_version,
                 device_kind, max_entries=64):
        if runtime_version != "0.32.3" or device_kind != "gpu":
            raise ValueError("Only explicitly identified MLX 0.32.3 GPU is admitted")
        if type(max_entries) is not int or not 1 <= max_entries <= 64:
            raise ValueError("max_entries must be an integer in 1..64")
        self._model, self._mx, self._types = model, backend, types
        self._limit, self._entries, self._epoch = max_entries, OrderedDict(), 0
        self._counts = dict(hits=0, misses=0, evictions=0, invalidations=0,
                            fallbacks=0, failures=0, factory_ns=0,
                            cold_call_ns=0, warm_call_ns=0)
        self._last_fallback = None

    @property
    def model(self):
        return self._model

    def invalidate(self):
        """Call at a quiescent boundary after changing weights/config/opt-ins."""
        self._entries.clear()
        self._epoch += 1
        self._counts["invalidations"] += 1

    def replace_model(self, model):
        self.invalidate()
        self._model = model

    def snapshot(self):
        """Host counters and metadata only: no device query or array retention."""
        return dict(self._counts, entries=len(self._entries), limit=self._limit,
                    epoch=self._epoch, last_fallback=self._last_fallback,
                    keys=list(self._entries), native_compiled_bytes=None)

    def _is_array(self, value):
        return type(value) is self._mx.array

    def _pack(self, caches):
        flat = []
        for layer, cache in zip(self._model.layers, caches):
            flat.extend((cache.K, cache.V) if layer.is_attn
                        else (cache.conv, cache.S))
        return tuple(flat)

    def _unpack(self, flat):
        caches = []
        for i, layer in enumerate(self._model.layers):
            a, b = flat[2 * i:2 * i + 2]
            # __setitem__ changes an mx.array's graph handle. New views isolate
            # the explicitly supplied original handles, as in accepted ahead.
            caches.append(self._types.kv(a[:], b[:]) if layer.is_attn
                          else self._types.state(a, b))
        return caches

    def _guard(self, tokens, ctx, greedy):
        mx, types, model = self._mx, self._types, self._model
        if greedy is not True or type(model) is not types.model:
            return "ordinary greedy model required"
        if type(ctx) is not types.context or ctx.mode != "decode":
            return "ordinary decode context required"
        fields = {"mode", "caches", "positions", "lens", "max_len", "prefill_shape"}
        if set(vars(ctx)) - fields or getattr(ctx, "prefill_shape", None) is not None:
            return "capture/spec/verify/record or unknown context metadata"
        if not self._is_array(tokens) or tokens.shape != (1,) or tokens.dtype != mx.int32:
            return "B1 int32 tokens required"
        if any(not self._is_array(a) or a.shape != (1,) or a.dtype != mx.int32
               for a in (ctx.positions, ctx.lens)):
            return "explicit B1 int32 positions/lens required"
        if type(ctx.max_len) is not int or ctx.max_len < 1:
            return "positive exact Python L required"
        if len(ctx.caches) != len(model.layers) or not model.layers:
            return "cache/layer count mismatch"
        cfg = model.cfg
        if (cfg.hidden_size != 5120 or
                cfg.quantization != {"group_size": 16, "bits": 4, "mode": "nvfp4"}):
            return "accepted 27B NVFP4 model configuration required"
        if ((cfg.linear_num_key_heads, cfg.linear_num_value_heads,
             cfg.linear_key_head_dim, cfg.linear_value_head_dim,
             cfg.linear_conv_kernel_dim) != (16, 48, 128, 128, 4)):
            return "accepted conv/GDN configuration required"
        for layer, cache in zip(model.layers, ctx.caches):
            if layer.is_attn:
                if type(cache) is not types.kv or set(vars(cache)) != {"K", "V"}:
                    return "plain BatchKV required"
                k, v = cache.K, cache.V
                if (not self._is_array(k) or not self._is_array(v) or
                        k.shape != v.shape or len(k.shape) != 4 or
                        k.shape[:2] != (1, cfg.num_key_value_heads) or
                        k.shape[3] != cfg.head_dim or k.shape[2] < ctx.max_len or
                        k.dtype != mx.bfloat16 or v.dtype != mx.bfloat16):
                    return "KV shape/dtype/capacity mismatch"
            else:
                if type(cache) is not types.state or set(vars(cache)) != {"conv", "S"}:
                    return "plain BatchState required"
                if (not self._is_array(cache.conv) or not self._is_array(cache.S) or
                        cache.conv.shape != (1, 3, 10240) or
                        cache.S.shape != (1, 48, 128, 128) or
                        cache.conv.dtype != mx.bfloat16 or cache.S.dtype != mx.float32):
                    return "GDN state shape/dtype mismatch"
        return None

    def _make_pure(self, length):
        # Do not close over the caller's context, caches, tokens or positions.
        model, types = self._model, self._types
        kinds = tuple(layer.is_attn for layer in model.layers)

        def pure(tokens, positions, lens, *flat):
            caches = [types.kv(flat[2*i][:], flat[2*i+1][:]) if attn
                      else types.state(flat[2*i], flat[2*i+1])
                      for i, attn in enumerate(kinds)]
            ctx = types.context("decode", caches, positions=positions,
                                lens=lens, max_len=length)
            logits = model(tokens, ctx)
            outputs = []
            for attn, cache in zip(kinds, caches):
                outputs.extend((cache.K, cache.V) if attn else (cache.conv, cache.S))
            return (logits, *outputs)

        return pure

    def _invoke(self, tokens, ctx, flat):
        args = (tokens, ctx.positions, ctx.lens, *flat)
        signature = tuple((tuple(a.shape), str(a.dtype)) for a in args)
        key = (self._epoch, 1, ctx.max_len, signature)
        cold = key not in self._entries
        if cold:
            self._counts["misses"] += 1
            started = perf_counter_ns()
            try:
                fn = self._mx.compile(self._make_pure(ctx.max_len), shapeless=False)
            except Exception:
                self._counts["failures"] += 1
                raise
            finally:
                self._counts["factory_ns"] += perf_counter_ns() - started
        else:
            self._counts["hits"] += 1
            fn = self._entries[key]
            self._entries.move_to_end(key)
        started = perf_counter_ns()
        try:
            result = tuple(fn(*args))
            if len(result) != len(flat) + 1 or not self._is_array(result[0]):
                raise RuntimeError("compiled output structure mismatch")
            if any(not self._is_array(a) or a.shape != old.shape or a.dtype != old.dtype
                   for a, old in zip(result[1:], flat)):
                raise RuntimeError("compiled state metadata mismatch")
        except Exception:
            self._entries.pop(key, None)
            self._counts["failures"] += 1
            raise  # The original caller context has not been committed.
        finally:
            field = "cold_call_ns" if cold else "warm_call_ns"
            self._counts[field] += perf_counter_ns() - started
        if cold:
            self._entries[key] = fn
            if len(self._entries) > self._limit:
                self._entries.popitem(last=False)
                self._counts["evictions"] += 1
        return result

    def functional(self, tokens, positions, lens, flat_state, *, max_len, greedy=False):
        """Return (logits, *new_states); never commit into caller cache objects."""
        if len(flat_state) != 2 * len(self._model.layers):
            raise IneligibleDecode("flat state/layer count mismatch")
        ctx = self._types.context("decode", self._unpack(flat_state),
                                  positions=positions, lens=lens, max_len=max_len)
        reason = self._guard(tokens, ctx, greedy)
        if reason:
            raise IneligibleDecode(reason)
        return self._invoke(tokens, ctx, tuple(flat_state))

    def forward(self, tokens, ctx=None, *, greedy=False, **model_kwargs):
        """Future normal-decode call-site adapter; unsupported calls stay original."""
        reason = "model keyword arguments" if model_kwargs else self._guard(tokens, ctx, greedy)
        if reason:
            self._counts["fallbacks"] += 1
            self._last_fallback = reason
            return self._model(tokens, ctx, **model_kwargs)
        result = self._invoke(tokens, ctx, self._pack(ctx.caches))
        # Install graph handles only after the whole functional call succeeds.
        # Evaluation, ahead isolation, sampling and scheduler commit stay outside.
        for i, (layer, cache) in enumerate(zip(self._model.layers, ctx.caches)):
            if layer.is_attn:
                cache.K, cache.V = result[2*i+1:2*i+3]
            else:
                cache.conv, cache.S = result[2*i+1:2*i+3]
        return result[0]
