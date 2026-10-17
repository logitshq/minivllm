"""Default-OFF E024 route-preserving bound experiment, not device-qualified.

Only B1..8 and used257..1536. Bound1023 below1024, otherwise1536.
Two callable families may have multiple shape specializations. No installation,
core monkeypatch, SDPA override, tolerance change, or performance claim.
"""
import hashlib
import importlib.util
from pathlib import Path

ORIGINAL = Path(__file__).resolve().parent.parent/'long_decode_graph/candidate.py'
ORIGINAL_SHA = 'eb14b671a66764fe677219762090acb495b89d384cf078f3f63b918a40aa2b84'
PINS = ORIGINAL.with_name('model-source-pins.json')
MIN_USED_LENGTH = 257


def bound_for_used(used):
    if type(used) is not int or not 257 <= used <= 1536:
        return None
    return 1023 if used < 1024 else 1536


def _load_runtime(model):
    # Reuse the unchanged original model/source/backend verifier at its real path.
    if hashlib.sha256(ORIGINAL.read_bytes()).hexdigest() != ORIGINAL_SHA:
        raise ValueError('fixed original E024 candidate source changed')
    spec = importlib.util.spec_from_file_location('e024_route_original_runtime', ORIGINAL)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module._load_runtime(model)


class RoutePreservingDecode:
    """Bench callable: RoutePreservingDecode(engine.model, enabled=True)(tokens, ctx).

    Enablement and model ownership are fixed at construction. Replace array
    parameters on the original model normally; recreate this object after any
    model structure/non-array configuration change (except gates/norm flags).
    """
    def __init__(self, model, *, enabled=False):
        if type(enabled) is not bool:
            raise TypeError('enabled must be bool')
        self._model = model
        self._enabled = enabled
        self._runtime = _load_runtime(model) if enabled else None
        self._functions = {}
        self._bound = 1536
        self._minimum = MIN_USED_LENGTH
        self._layer_ids = tuple(id(layer) for layer in model.layers) if enabled else ()
        self.source_receipt = self._runtime.source_receipt if enabled else None

    @property
    def model(self):
        return self._model

    @property
    def enabled(self):
        return self._enabled

    @property
    def bound(self):
        return self._bound

    def _eligible(self, tokens, ctx):
        r, model = self._runtime, self._model
        mx = r.mx
        bound = bound_for_used(getattr(ctx, 'max_len', None))
        if bound is None:
            return False
        if not (type(ctx) is r.ForwardContext and ctx.mode == 'decode'
                and set(vars(ctx)) == {'mode', 'caches', 'positions', 'lens', 'max_len', 'prefill_shape'}
                and ctx.prefill_shape is None and tokens.ndim == 1
                and 1 <= tokens.shape[0] <= 8 and tokens.dtype in (mx.int32, mx.int64)
                and type(ctx.max_len) is int and self._minimum <= ctx.max_len <= self._bound
                and ctx.positions is not None and ctx.lens is not None
                and ctx.positions.shape == ctx.lens.shape == tokens.shape
                and ctx.positions.dtype in (mx.int32, mx.int64)
                and ctx.lens.dtype in (mx.int32, mx.int64)
                and model.cfg.hidden_size == 5120 and len(model.layers) == 64
                and tuple(id(layer) for layer in model.layers) == self._layer_ids
                and model.cfg.quantization == {'group_size': 16, 'bits': 4, 'mode': 'nvfp4'}
                and model.embed_tokens._output_dtype == mx.bfloat16
                and mx.default_device() == mx.gpu and r.identity()
                and len(ctx.caches) == len(model.layers)):
            return False
        B = tokens.shape[0]
        for layer, cache in zip(model.layers, ctx.caches):
            if layer.is_attn:
                if not (type(cache) is r.BatchKV and set(vars(cache)) == {'K', 'V'}
                        and cache.K.ndim == cache.V.ndim == 4
                        and cache.K.shape == cache.V.shape and cache.K.shape[0] == B
                        and cache.K.shape[2] >= self._bound
                        and cache.K.dtype == cache.V.dtype == mx.bfloat16):
                    return False
            elif not (type(cache) is r.BatchState and set(vars(cache)) == {'conv', 'S'}
                      and cache.conv.shape == (B, 3, 10240) and cache.conv.dtype == mx.bfloat16
                      and cache.S.shape == (B, 48, 128, 128) and cache.S.dtype == mx.float32):
                return False
        return True

    def __call__(self, tokens, ctx):
        model = self._model
        if not self._enabled or not self._eligible(tokens, ctx):
            return model(tokens, ctx)
        r, bound = self._runtime, bound_for_used(ctx.max_len)
        key = (bound, tuple((layer.linear_attn._gates_compile_enabled,
                             layer.linear_attn._norm_reuse_enabled)
                            for layer in model.layers if not layer.is_attn))
        if key not in self._functions:
            layout = tuple(layer.is_attn for layer in model.layers)
            def forward(ids, positions, lens, states):
                # Independent KV array wrappers protect the caller's handles
                # from __setitem__ while sharing evaluated input storage.
                caches = [r.BatchKV(s[0][:], s[1][:]) if attn else r.BatchState(*s)
                          for attn, s in zip(layout, states)]
                logits = model(ids, r.ForwardContext('decode', caches,
                    positions=positions, lens=lens, max_len=bound))
                return logits, r.states(caches)
            self._functions[key] = r.mx.compile(forward, inputs=model)
        logits, states = self._functions[key](tokens, ctx.positions, ctx.lens, r.states(ctx.caches))
        if len(states) != len(ctx.caches) or any(len(pair) != 2 for pair in states):
            raise RuntimeError('E024 graph returned an incomplete state tree')
        for cache, state in zip(ctx.caches, states):
            if type(cache) is r.BatchKV:
                cache.K, cache.V = state
            else:
                cache.conv, cache.S = state
        return logits
