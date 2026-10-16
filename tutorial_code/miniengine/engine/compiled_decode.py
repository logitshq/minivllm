"""Opt-in, bounded M5 Ultra decode graph reuse; prefill and speculation stay eager.

The first 256 positions share one masked attention shape. State is explicit in
and out, so graph reuse never owns a request's cache. Weights are captured as
inputs (not constants), including when an array parameter is replaced.
"""
import mlx.core as mx

from ..cache import BatchKV, BatchState
from ..context import ForwardContext
from ..model.gdn import _m5_ultra_decode_identity


class CompiledDecode:
    def __init__(self, model):
        self.model = model
        self._functions = {}

    def __call__(self, tokens, ctx):
        model = self.model
        eligible = (
            type(ctx) is ForwardContext and ctx.mode == "decode"
            and set(vars(ctx)) == {"mode", "caches", "positions", "lens", "max_len", "prefill_shape"}
            and ctx.prefill_shape is None and tokens.shape in ((1,), (2,), (4,))
            and 0 < ctx.max_len <= 256
            and model.cfg.hidden_size == 5120 and len(model.layers) == 64
            and model.cfg.quantization == {"group_size": 16, "bits": 4, "mode": "nvfp4"}
            and model.embed_tokens._output_dtype == mx.bfloat16
            and mx.default_device() == mx.gpu and _m5_ultra_decode_identity()
            and len(ctx.caches) == len(model.layers)
            and all(type(c) is (BatchKV if layer.is_attn else BatchState)
                    and set(vars(c)) == ({"K", "V"} if layer.is_attn else {"conv", "S"})
                    and (not layer.is_attn or c.K.shape[2] >= 256)
                    for layer, c in zip(model.layers, ctx.caches))
        )
        if not eligible:
            return model(tokens, ctx)
        # These Python options affect tracing, unlike array weights. A change
        # selects another graph instead of silently retaining an old branch.
        key = tuple((layer.linear_attn._gates_compile_enabled,
                     layer.linear_attn._norm_reuse_enabled)
                    for layer in model.layers if not layer.is_attn)
        if key not in self._functions:
            def forward(ids, positions, lens, states):
                layers = [BatchKV(s[0][:], s[1][:]) if layer.is_attn else BatchState(*s)
                          for layer, s in zip(model.layers, states)]
                logits = model(ids, ForwardContext("decode", layers, positions=positions,
                                                  lens=lens, max_len=256))
                return logits, self._states(layers)
            self._functions[key] = mx.compile(forward, inputs=model)
        logits, states = self._functions[key](tokens, ctx.positions, ctx.lens, self._states(ctx.caches))
        for layer, state in zip(ctx.caches, states):
            if isinstance(layer, BatchKV):
                layer.K, layer.V = state
            else:
                layer.conv, layer.S = state
        return logits

    @staticmethod
    def _states(layers):
        return [(c.K, c.V) if isinstance(c, BatchKV) else (c.conv, c.S) for c in layers]
