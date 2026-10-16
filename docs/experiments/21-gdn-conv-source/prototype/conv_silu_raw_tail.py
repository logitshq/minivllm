"""Unintegrated V23 prototype. Importing this module never imports MLX.

The GPU owner must establish the reference product layout and launch Y before
calling conv_silu_raw_tail(..., sum_y=Y, backend=mlx.core). No default or measured
branch selection is provided. This module has no engine hooks or numerical claim.
"""
from functools import lru_cache
from pathlib import Path

CHANNELS = 10240
GRID = (CHANNELS, 1, 1)
THREADGROUP = (256, 1, 1)
OUTPUT_SHAPES = ((1, CHANNELS), (1, 3, CHANNELS))
COMPILE_OPTIONS = {"math_mode": "safe"}  # Same option as V18 activation-only PASS.
_ROOT = Path(__file__).resolve().parent
HEADER = (_ROOT / "native_activation.metal").read_text()
SOURCE = (_ROOT / "conv_silu_raw_tail.metal").read_text()


def validate_sum_y(sum_y):
    """Validate an already measured reference Y; never infer or tune it."""
    if type(sum_y) is not int or sum_y not in (1, 2, 3, 4):
        raise ValueError("sum_y must be a verified integer in {1, 2, 3, 4}")
    return sum_y


def validate_metadata(raw, old_tail, w, *, sum_y, bfloat16):
    """Pure CPU metadata check; this does not certify layout or numerical math."""
    validate_sum_y(sum_y)
    expected = ((1, CHANNELS), (1, 3, CHANNELS), (CHANNELS, 4))
    for name, value, shape in zip(("raw", "old_tail", "w"), (raw, old_tail, w), expected):
        if getattr(value, "shape", None) != shape:
            raise ValueError(f"{name} must have shape {shape}")
        if getattr(value, "dtype", None) != bfloat16:
            raise ValueError(f"{name} must be BF16")


@lru_cache(maxsize=1)
def _kernel(backend):
    # backend is supplied explicitly; no import, compile, eval or allocation at import.
    return backend.fast.metal_kernel(
        name="v23_gdn_conv_silu_raw_tail",
        input_names=["raw", "old_tail", "w"], output_names=["act", "new_tail"],
        source=SOURCE, header=HEADER, ensure_row_contiguous=True,
        compile_options=COMPILE_OPTIONS,
    )


def conv_silu_raw_tail(raw, old_tail, w, *, sum_y, backend):
    """Return lazy (act BF16[1,C], new_tail BF16[1,3,C]) in distinct outputs.

    PRECONDITION: caller recorded original product flags/strides and actual
    kernel M or launch Y, then froze this sum_y from that evidence. The caller
    must not choose Y by comparing outputs. The four branches express order,
    not a performance sweep. This launch has not been compiled or GPU-tested.
    """
    validate_metadata(raw, old_tail, w, sum_y=sum_y, bfloat16=backend.bfloat16)
    if any(not isinstance(value, backend.array) for value in (raw, old_tail, w)):
        raise TypeError("inputs must be MLX arrays from the supplied backend")
    if backend.default_device() != backend.gpu:
        raise ValueError("this unqualified prototype requires the GPU backend")
    return tuple(_kernel(backend)(
        inputs=[raw, old_tail, w], template=[("SUM_Y", sum_y)],
        grid=GRID, threadgroup=THREADGROUP,
        output_shapes=list(OUTPUT_SHAPES), output_dtypes=[backend.bfloat16] * 2,
    ))
