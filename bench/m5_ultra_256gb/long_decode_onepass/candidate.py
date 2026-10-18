"""Default-OFF bench candidate: collect long-route guards/key/states in one pass.

build(model) returns the original model without source/runtime access. Enabled
construction verifies and subclasses the frozen E024 candidate. No installation,
parameter snapshot, epoch shortcut, global patch, eval or qualification claim.
The injected factory exists for CPU contract tests; actual use goes through build.
"""
import hashlib
import importlib.util
from pathlib import Path

ORIGINAL = Path(__file__).resolve().parent.parent / 'route_decode_graph/candidate.py'
ORIGINAL_SHA = '6531be25acb53713d6ddcd930d2c778f6640a3b1f53304fd5375d20b044525c3'


def make_decoder_class(original):
    """Keep the original constructor/runtime verifier and ordinary fallback."""
    class OnePassDecode(original):
        def _prepare(self, tokens, ctx):
            r, model = self._runtime, self._model
            mx = r.mx
            used = getattr(ctx, 'max_len', None)
            if type(used) is not int or not 257 <= used <= 1536:
                return None
            bound = 1023 if used < 1024 else 1536
            layers = model.layers
            if not (type(ctx) is r.ForwardContext and ctx.mode == 'decode'
                    and set(vars(ctx)) == {'mode', 'caches', 'positions', 'lens', 'max_len', 'prefill_shape'}
                    and ctx.prefill_shape is None and tokens.ndim == 1
                    and 1 <= tokens.shape[0] <= 8 and tokens.dtype in (mx.int32, mx.int64)
                    and type(ctx.max_len) is int and self._minimum <= ctx.max_len <= self._bound
                    and ctx.positions is not None and ctx.lens is not None
                    and ctx.positions.shape == ctx.lens.shape == tokens.shape
                    and ctx.positions.dtype in (mx.int32, mx.int64)
                    and ctx.lens.dtype in (mx.int32, mx.int64)
                    and model.cfg.hidden_size == 5120 and len(layers) == 64
                    and len(self._layer_ids) == 64
                    and model.cfg.quantization == {'group_size': 16, 'bits': 4, 'mode': 'nvfp4'}
                    and model.embed_tokens._output_dtype == mx.bfloat16
                    and mx.default_device() == mx.gpu and r.identity()
                    and len(ctx.caches) == len(layers)):
                return None
            B = tokens.shape[0]
            states, flags = [], []
            for layer, cache, identity in zip(layers, ctx.caches, self._layer_ids):
                if id(layer) != identity:
                    return None
                if layer.is_attn:
                    if not (type(cache) is r.BatchKV and set(vars(cache)) == {'K', 'V'}
                            and cache.K.ndim == cache.V.ndim == 4
                            and cache.K.shape == cache.V.shape and cache.K.shape[0] == B
                            and cache.K.shape[2] >= self._bound
                            and cache.K.dtype == cache.V.dtype == mx.bfloat16):
                        return None
                    states.append((cache.K, cache.V))
                else:
                    if not (type(cache) is r.BatchState and set(vars(cache)) == {'conv', 'S'}
                            and cache.conv.shape == (B, 3, 10240) and cache.conv.dtype == mx.bfloat16
                            and cache.S.shape == (B, 48, 128, 128) and cache.S.dtype == mx.float32):
                        return None
                    states.append((cache.conv, cache.S))
                    flags.append((layer.linear_attn._gates_compile_enabled,
                                  layer.linear_attn._norm_reuse_enabled))
            return bound, (bound, tuple(flags)), states

        def __call__(self, tokens, ctx):
            model = self._model
            if not self._enabled:
                return model(tokens, ctx)
            prepared = self._prepare(tokens, ctx)
            if prepared is None:
                return model(tokens, ctx)
            bound, key, inputs = prepared
            r = self._runtime
            if key not in self._functions:
                layout = tuple(layer.is_attn for layer in model.layers)
                def forward(ids, positions, lens, states):
                    # Exact original trace body; copies only the KV handles.
                    caches = [r.BatchKV(s[0][:], s[1][:]) if attn else r.BatchState(*s)
                              for attn, s in zip(layout, states)]
                    logits = model(ids, r.ForwardContext('decode', caches,
                        positions=positions, lens=lens, max_len=bound))
                    return logits, r.states(caches)
                self._functions[key] = r.mx.compile(forward, inputs=model)
            logits, states = self._functions[key](tokens, ctx.positions, ctx.lens, inputs)
            # Validate the complete returned tree before any state assignment.
            if len(states) != len(ctx.caches) or any(len(pair) != 2 for pair in states):
                raise RuntimeError('E024 graph returned an incomplete state tree')
            for cache, state in zip(ctx.caches, states):
                if type(cache) is r.BatchKV:
                    cache.K, cache.V = state
                else:
                    cache.conv, cache.S = state
            return logits
    return OnePassDecode


def _original_class():
    if hashlib.sha256(ORIGINAL.read_bytes()).hexdigest() != ORIGINAL_SHA:
        raise RuntimeError('frozen E024 route source changed')
    spec = importlib.util.spec_from_file_location('_long_onepass_original', ORIGINAL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.RoutePreservingDecode


def build(model, *, enabled=False):
    """Return a callable; caller owns quiescence and any decoder installation.

    Parameter replacement remains ordinary inputs=model capture. As with the
    original, recreate after model structure/non-array configuration changes,
    except the dynamically keyed gates/norm flags. No hot reload with ahead work.
    """
    if type(enabled) is not bool:
        raise TypeError('enabled must be bool')
    if not enabled:
        return model
    return make_decoder_class(_original_class())(model, enabled=True)
