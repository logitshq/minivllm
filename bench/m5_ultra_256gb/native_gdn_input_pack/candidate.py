"""Default-OFF, one-shot B1 native-QMM GDN input projection experiment.

No MLX import, model installation, graph capture or hot cache. ``prepare`` is an
explicit idle-time materialization; callers must rebuild it from current arrays
for every qualification case. A prepared object is a derived snapshot, never a
freshness check for replacement or __setitem__ on the original parameters.
"""

QUANT = {"group_size": 16, "bits": 4, "mode": "nvfp4"}
WIDTHS = (10240, 6144, 48, 48)  # qkv, z, b, a
SPLITS = (10240, 16384, 16432)
INPUT_WIDTH = 5120
PACKED_WIDTH = 16480


def _array_is(array, shape, dtype):
    return getattr(array, "shape", None) == shape and getattr(array, "dtype", None) == dtype


class PreparedPack:
    """Materialized derived arrays; original Linear objects are not retained.

    This object is deliberately not a live-parameter or compiled-model adapter.
    Do not mutate its arrays or reuse it across source parameter changes.
    """

    def __init__(self, mx, weight, scales):
        self._mx, self.weight, self.scales = mx, weight, scales

    def project(self, x):
        """Return qkv/z/b/a, or None for inputs outside BF16[1,5120].

        None asks the caller to use the four original Linears; this function
        never executes fallback itself. Kernel errors propagate. No eval/sync,
        casts, re-quantization, custom qmv or parameter refresh occurs here.
        """
        mx = self._mx
        if not _array_is(x, (1, INPUT_WIDTH), mx.bfloat16):
            return None
        y = mx.quantized_matmul(x, self.weight, self.scales, None,
                                transpose=True, **QUANT)
        if not _array_is(y, (1, PACKED_WIDTH), mx.bfloat16):
            raise ValueError("native QMM must return BF16[1,16480]; no output cast allowed")
        outputs = tuple(mx.split(y, list(SPLITS), axis=-1))
        if len(outputs) != 4 or any(not _array_is(a, (1, n), mx.bfloat16)
                                    for a, n in zip(outputs, WIDTHS)):
            raise ValueError("native QMM split metadata differs from qkv/z/b/a")
        return outputs


def prepare(mx, ops, *, enabled=False, linear_type=None):
    """Explicit idle-only concat/eval of four current Linear parameter pairs.

    OFF returns None before inspecting mx, ops or linear_type. ON requires the
    actual native Linear class supplied by the qualification caller. All four
    operands are validated before any array operation. Preparation leaves the
    original Linear attributes untouched; errors propagate without a fallback.
    The caller must keep source parameters stable during this preparation/use.
    """
    if type(enabled) is not bool:
        raise TypeError("enabled must be bool")
    if not enabled:
        return None
    if not isinstance(linear_type, type) or type(ops) not in (tuple, list) or len(ops) != 4:
        raise ValueError("expected actual Linear class and four qkv/z/b/a operations")
    pairs = []
    for op, n in zip(ops, WIDTHS):
        if type(op) is not linear_type:
            raise ValueError("only exact native Linear instances are supported")
        q = getattr(op, "q", None)
        if (type(q) is not dict or q != QUANT
                or type(q.get("group_size")) is not int or type(q.get("bits")) is not int
                or type(getattr(op, "in_features", None)) is not int or op.in_features != INPUT_WIDTH
                or type(getattr(op, "out_features", None)) is not int or op.out_features != n
                or getattr(op, "biases", None) is not None):
            raise ValueError("native Linear quantization, geometry or bias differs")
        weight, scales = getattr(op, "weight", None), getattr(op, "scales", None)
        if not (_array_is(weight, (n, 640), mx.uint32)
                and _array_is(scales, (n, 320), mx.uint8)):
            raise ValueError("expected current U32[N,640] weight and U8[N,320] scales")
        pairs.append((weight, scales))
    weight = mx.concatenate([w for w, _ in pairs], axis=0)
    scales = mx.concatenate([s for _, s in pairs], axis=0)
    if not (_array_is(weight, (PACKED_WIDTH, 640), mx.uint32)
            and _array_is(scales, (PACKED_WIDTH, 320), mx.uint8)):
        raise ValueError("derived pack metadata differs")
    mx.eval(weight, scales)  # Explicit one-time preparation, outside decode/graphs/timing.
    return PreparedPack(mx, weight, scales)
