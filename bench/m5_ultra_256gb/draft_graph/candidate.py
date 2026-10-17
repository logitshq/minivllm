"""E016 isolated graph upper bound; request state is always explicit.

This prototype freezes layer parameters, refuses parameter replacement, and has
no production integration. Exact shape specialization is intentional: growing
KV shapes may retrace, and the screen retains those first-call costs.
"""
from copy import copy

import mlx.core as mx
from mlx.utils import tree_flatten


def states(drafter):
    return [(layer.self_attn.ctx_k, layer.self_attn.ctx_v) for layer in drafter.layers]


def restore(drafter, values):
    for layer, (key, value) in zip(drafter.layers, values, strict=True):
        layer.self_attn.ctx_k, layer.self_attn.ctx_v = key, value
        layer.self_attn.ctx_len = key.shape[2]


class BackboneGraph:
    def __init__(self, drafter):
        self.drafter = drafter
        self.traces = []
        self._frozen = self._parameters()

        def forward(context, embedded, position, cache):
            self.traces.append({
                'context_shape': list(context.shape),
                'embedded_shape': list(embedded.shape),
                'kv_shapes': [[list(k.shape), list(v.shape)] for k, v in cache],
            })
            hidden, new_cache = embedded, []
            for source, (key, value) in zip(drafter.layers, cache, strict=True):
                # Copies contain the same immutable weights. Only the copied
                # attention shell receives the explicit request state.
                layer = copy(source)
                layer.self_attn = copy(source.self_attn)
                layer.self_attn.ctx_k, layer.self_attn.ctx_v = key, value
                layer.self_attn.ctx_len = key.shape[2]
                hidden = layer(hidden, context, position)
                new_cache.append((layer.self_attn.ctx_k, layer.self_attn.ctx_v))
            return hidden, new_cache

        self._compiled = mx.compile(forward)

    def _parameters(self):
        return tuple((name, value) for name, value in tree_flatten(
            [layer.parameters() for layer in self.drafter.layers])
            if name.rsplit('.', 1)[-1] not in ('ctx_k', 'ctx_v'))

    def assert_frozen(self):
        current = self._parameters()
        if len(current) != len(self._frozen) or any(
            name != old_name or value is not old_value
            for (name, value), (old_name, old_value) in zip(current, self._frozen)
        ):
            raise RuntimeError('E016 freezes parameters; rebuild graph after replacing a weight')

    def __call__(self, context, embedded, position, cache):
        self.assert_frozen()
        if len(cache) != len(self.drafter.layers) or any(k is None or v is None for k, v in cache):
            raise ValueError('E016 requires explicit initialized KV for every layer')
        if context is None or context.shape[0] == 0:
            raise ValueError('E016 initial screen requires nonempty context')
        return self._compiled(context, embedded, position, cache)

    def draft(self, features, anchor, start, embed_fn, head_fn, block_size):
        """Benchmark-only adapter with the original surrounding draft operations."""
        draft = self.drafter
        context = draft.hidden_norm(draft.fc(features))
        embedded = embed_fn(mx.array([anchor] + [draft.cfg.mask_token_id] * (block_size - 1)))
        hidden, cache = self(context, embedded, mx.array(start, dtype=mx.int32), states(draft))
        restore(draft, cache)
        backbone_hidden = hidden
        hidden = draft.norm(backbone_hidden)[1:]
        logits = head_fn(hidden).astype(mx.float32)
        return draft._select(hidden, logits, anchor), hidden, logits, cache, backbone_hidden
