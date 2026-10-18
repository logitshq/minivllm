"""Opt-in, bounded M5 Ultra decode graph reuse; prefill and speculation stay eager.

The first 256 positions share one masked attention shape. State is explicit in
and out, so graph reuse never owns a request's cache. Weights are captured as
inputs (not constants), including when an array parameter is replaced.
The qualified batch range is 1..8; larger batches retain eager execution.
"""
import mlx.core as mx

from ..cache import BatchKV, BatchState
from ..context import ForwardContext
from ..model.gdn import _m5_ultra_decode_identity


class CompiledDecode:
    def __init__(self, model, *, fuse_decode=False):
        self.model = model
        self._functions = {}
        self._fuse_decode = fuse_decode
        self._packed = None

    def __call__(self, tokens, ctx):
        model = self.model
        eligible = (
            type(ctx) is ForwardContext and ctx.mode == "decode"
            and set(vars(ctx)) == {"mode", "caches", "positions", "lens", "max_len", "prefill_shape"}
            and ctx.prefill_shape is None and tokens.ndim == 1 and 1 <= tokens.shape[0] <= 8
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
        fused = self._fuse_decode and all(
            layer.is_attn or (
                (layer.linear_attn.Hk, layer.linear_attn.Hv,
                 layer.linear_attn.Dk, layer.linear_attn.Dv) == (16, 48, 128, 128)
                and layer.linear_attn.conv1d_weight.shape == (10240, 4)
                and layer.linear_attn.conv1d_weight.dtype == mx.bfloat16
                and c.conv.shape == (tokens.shape[0], 3, 10240) and c.conv.dtype == mx.bfloat16
                and c.S.shape == (tokens.shape[0], 48, 128, 128) and c.S.dtype == mx.float32)
            for layer, c in zip(model.layers, ctx.caches))
        if fused:
            if self._packed is None:
                from .packed_decode import PackedDecode
                self._packed = PackedDecode(model)
            if not self._packed.refresh():
                return model(tokens, ctx)
        # These Python options affect tracing, unlike array weights. A change
        # selects another graph instead of silently retaining an old branch.
        key = (fused, tuple((layer.linear_attn._gates_compile_enabled,
                     layer.linear_attn._norm_reuse_enabled)
                    for layer in model.layers if not layer.is_attn))
        if key not in self._functions:
            projections = self._packed.layers if fused else None
            def forward(ids, positions, lens, states):
                layers = [BatchKV(s[0][:], s[1][:]) if layer.is_attn else BatchState(*s)
                          for layer, s in zip(model.layers, states)]
                logits = model(ids, ForwardContext("decode", layers, positions=positions,
                                                  lens=lens, max_len=256),
                               _decode_projections=projections)
                return logits, self._states(layers)
            inputs = [model, self._packed.arrays] if fused else model
            self._functions[key] = mx.compile(forward, inputs=inputs)
        clean = (self._packed.groups if fused else self._packed.clean_groups()) if self._packed else []
        marker = model.embed_tokens.weight
        logits, states = self._functions[key](tokens, ctx.positions, ctx.lens, self._states(ctx.caches))
        # mx.compile may replace captured array wrappers when tracing another
        # batch shape. This forward never writes parameters. Rebind only packs
        # already current before the call; actual replacements must stay dirty.
        if model.embed_tokens.weight is not marker:
            for projection in clean:
                projection.sources = projection.current_sources()
        for layer, state in zip(ctx.caches, states):
            if isinstance(layer, BatchKV):
                layer.K, layer.V = state
            else:
                layer.conv, layer.S = state
        return logits

    @staticmethod
    def _states(layers):
        return [(c.K, c.V) if isinstance(c, BatchKV) else (c.conv, c.S) for c in layers]
