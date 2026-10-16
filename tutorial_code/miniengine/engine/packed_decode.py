"""Derived decode weights, owned by the runner rather than model parameters.

Concatenate output rows once: the same NVFP4 dot products need fewer launches.
Original parameter arrays stay intact; replacement refreshes the derived copy.
"""
import itertools

import mlx.core as mx

from ..model.layers import Linear


class PackedProjection:
    def __init__(self, *ops):
        self.ops = ops
        self.splits = list(itertools.accumulate(op.out_features for op in ops))[:-1]
        self.arrays = []
        self.sources = ()

    def current_sources(self):
        return tuple(a for op in self.ops for a in (op.weight, op.scales))

    def supported(self):
        return all(type(op) is Linear
                   and op.q == {"group_size": 16, "bits": 4, "mode": "nvfp4"}
                   and op.in_features == 5120 and op.out_features % 16 == 0
                   and op.weight.shape == (op.out_features, 640) and op.weight.dtype == mx.uint32
                   and op.scales.shape == (op.out_features, 320) and op.scales.dtype == mx.uint8
                   and getattr(op, "biases", None) is None for op in self.ops)

    def __call__(self, x):
        y = mx.quantized_matmul(x, *self.arrays, transpose=True,
                                group_size=16, bits=4, mode="nvfp4")
        return mx.split(y, self.splits, axis=-1)


class PackedDecode:
    def __init__(self, model):
        self.layers = []
        for layer in model.layers:
            mlp = PackedProjection(layer.mlp.gate_proj, layer.mlp.up_proj)
            if layer.is_attn:
                a = layer.self_attn
                mixer = PackedProjection(a.q_proj, a.k_proj, a.v_proj)
            else:
                a = layer.linear_attn
                mixer = PackedProjection(a.in_proj_qkv, a.in_proj_z, a.in_proj_b, a.in_proj_a)
            self.layers.append((mlp, mixer))
        self.groups = [p for layer in self.layers for p in layer]
        # Keep this outer tree stable: mx.compile captures it as mutable inputs.
        self.arrays = [p.arrays for p in self.groups]

    def refresh(self):
        changed = [(p, p.current_sources()) for p in self.groups]
        changed = [(p, sources) for p, sources in changed
                   if len(sources) != len(p.sources)
                   or any(a is not b for a, b in zip(sources, p.sources))]
        if any(not p.supported() for p, _ in changed):
            return False
        for p, sources in changed:
            p.arrays[:] = [mx.concatenate(sources[::2]), mx.concatenate(sources[1::2])]
            p.sources = sources
        if changed:
            mx.eval([p.arrays for p, _ in changed])
        return True
