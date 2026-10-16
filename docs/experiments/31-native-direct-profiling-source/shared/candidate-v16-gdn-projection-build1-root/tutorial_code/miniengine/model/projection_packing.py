"""Opt-in NVFP4 output-row packing. Storage preservation is not GPU qualification.

Enable only after load_model has returned and caller-owned weight collections are
released. No automatic activation, model-specific sizes, padding or recoding.
Use reload_projection_weights for in-place reloads; external tensor replacement
invalidates on the next eligible decode; never race an in-flight forward.
"""
import mlx.core as mx

from .layers import Linear, prefill_batch_backend_supported

_NVFP4 = {"group_size": 16, "bits": 4, "mode": "nvfp4"}


def _layout(ops):
    if len(ops) < 2 or any(op is earlier for i, op in enumerate(ops) for earlier in ops[:i]):
        return None
    rows, width = [], None
    for op in ops:
        q = getattr(op, "q", None)
        if (type(op) is not Linear or type(q) is not dict or q != _NVFP4
                or type(q["group_size"]) is not int or type(q["bits"]) is not int
                or type(q["mode"]) is not str):
            return None
        k, n = getattr(op, "in_features", None), getattr(op, "out_features", None)
        if (type(k) is not int or type(n) is not int or k <= 0 or k % 16
                or n <= 0 or (width is not None and width != k)
                or hasattr(op, "bias") or hasattr(op, "biases")):
            return None
        w, s = getattr(op, "weight", None), getattr(op, "scales", None)
        if (type(w) is not mx.array or type(s) is not mx.array
                or w.dtype != mx.uint32 or s.dtype != mx.uint8
                or w.shape != (n, k // 8) or s.shape != (n, k // 16)):
            return None
        width = k
        rows.append(n)
    return width, tuple(rows)


class PackedProjectionGroup:
    """A plain object, deliberately not a Module/list/dict of new parameters."""

    def __init__(self, ops, layout, weight, scales, views):
        self.ops, self.layout = ops, layout
        self.weight, self.scales, self.views = weight, scales, views
        total, self.boundaries = 0, []
        for n in layout[1][:-1]:
            total += n
            self.boundaries.append(total)

    def invalidate(self):
        self.weight = self.scales = None
        self.ops = self.views = ()

    def project(self, x, ops):
        # Replaced arrays or changed quantization metadata must not use stale slabs.
        if (self.weight is None or len(ops) != len(self.ops)
                or any(op is not old for op, old in zip(ops, self.ops))
                or _layout(ops) != self.layout
                or any(op.weight is not w or op.scales is not s
                       for op, (w, s) in zip(ops, self.views))):
            self.invalidate()
            return None
        if (not prefill_batch_backend_supported() or type(x) is not mx.array
                or x.dtype != mx.bfloat16 or x.shape != (1, self.layout[0])):
            return None
        output = mx.quantized_matmul(
            x, self.weight, self.scales, None, transpose=True,
            group_size=16, bits=4, mode="nvfp4",
        )
        return tuple(mx.split(output, self.boundaries, axis=-1))


def pack_projection_group(ops):
    """Pack arbitrary output sizes transactionally; unsupported groups return None.

    MLX arrays have unified storage rather than a public per-array device field;
    the pinned backend guard checks actual default GPU/version/architecture at
    construction and execution. All operations use that same default device.
    Callers must not change the default device concurrently.
    """
    ops = tuple(ops)
    layout = _layout(ops)
    if layout is None or not prefill_batch_backend_supported():
        return None
    weight = mx.concatenate([op.weight for op in ops], axis=0)
    scales = mx.concatenate([op.scales for op in ops], axis=0)
    mx.eval(weight, scales)  # Materialize concat before releasing its inputs.
    views, start = [], 0
    for n in layout[1]:
        views.append((weight[start:start + n], scales[start:start + n]))
        start += n
    views = tuple(views)
    mx.eval(*(a for pair in views for a in pair))
    group = PackedProjectionGroup(ops, layout, weight, scales, views)
    previous = tuple((op.weight, op.scales) for op in ops)
    try:
        for op, (w, s) in zip(ops, views):
            op.weight, op.scales = w, s
    except Exception:
        for op, (w, s) in zip(ops, previous):
            op.weight, op.scales = w, s
        group.invalidate()
        raise
    return group


def decode_context_supported(ctx, st):
    from ..context import ForwardContext
    from ..cache import BatchState
    if (type(ctx) is not ForwardContext or ctx.mode != "decode"
            or ctx.prefill_shape is not None or type(st) is not BatchState
            or any(getattr(obj, key, False) is not False for obj in (ctx, st)
                   for key in ("record", "capture", "spec"))
            or type(ctx.max_len) is not int or ctx.max_len <= 0):
        return False
    for a in (ctx.positions, ctx.lens):
        if (type(a) is not mx.array or a.shape != (1,)
                or a.dtype not in (mx.int32, mx.int64, mx.uint32, mx.uint64)):
            return False
    return (type(st.conv) is mx.array and type(st.S) is mx.array
            and st.conv.dtype == mx.bfloat16 and st.S.dtype == mx.float32
            and st.conv.ndim == 3 and st.S.ndim == 4
            and st.conv.shape[0] == st.S.shape[0] == 1)


def _gdn_modules(model):
    from .gdn import GatedDeltaNet
    for layer in model.layers:
        op = getattr(layer, "linear_attn", None)
        if type(op) is GatedDeltaNet:
            yield op


def disable_gdn_projection_packing(model):
    """Drop helpers; fallback Linear row views continue to own their weights."""
    for gdn in _gdn_modules(model):
        group = getattr(gdn, "_projection_packing", None)
        if group is not None:
            group.invalidate()
        gdn._projection_packing = None


def enable_gdn_projection_packing(model):
    """Explicit post-load activation, one group at a time; returns scalar receipts.

    No caller-owned original arrays/weights dicts may be retained. This API does
    not prove allocator sharing, numerical equality, or performance.
    """
    disable_gdn_projection_packing(model)
    receipts = []
    for gdn in _gdn_modules(model):
        ops = (gdn.in_proj_qkv, gdn.in_proj_z, gdn.in_proj_b, gdn.in_proj_a)
        group = pack_projection_group(ops)
        gdn._projection_packing = group
        receipts.append({"enabled": group is not None,
                         "rows": None if group is None else group.layout[1]})
    return receipts


def reload_projection_weights(model, weights, *, strict=True):
    """Invalidate before strict reload, including when loading raises."""
    disable_gdn_projection_packing(model)
    return model.load_weights(weights, strict=strict)


def load_model_with_gdn_projection_packing(model_dir, dtype=mx.bfloat16, *, enabled=False):
    """Default-off wrapper. Packing starts after the loader frame has returned."""
    from .loader import load_model
    model, cfg = load_model(model_dir, dtype=dtype)
    if enabled:
        enable_gdn_projection_packing(model)
    return model, cfg
