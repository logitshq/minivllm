"""Single-shape, standalone address-layout experiment. Import uses stdlib only.

Backend is injected by the GPU owner. prepare() builds lazy packing operations;
the caller must eval packed arrays before numerical comparison or timing.
No engine patch, implicit activation conversion, fallback, or opt-in is installed.
"""
from pathlib import Path

BASE = Path(__file__).resolve().parent
K, N, ROWS, STEP = 5120, 17408, 8, 512


def metadata(k=K, n=N):
    if k != K or n != N:
        raise ValueError("V26 build1 qualifies only L0 gate [17408,5120]")
    payload = n * k // 2
    scales = n * k // 16
    return {
        "k": k, "n": n, "rows_per_group": ROWS,
        "weight_shape": (n // ROWS, k // STEP, ROWS, STEP // 8),
        "scale_shape": (n // ROWS, k // STEP, ROWS, STEP // 16),
        "payload_bytes": payload, "scale_bytes": scales,
        "original_bytes": payload + scales, "packed_bytes": payload + scales,
        "padding_bytes": 0, "grid": (64, n // ROWS, 1), "threadgroup": (64, 1, 1),
        "arithmetic": "native R4/K512/qdot/FP8/simd_sum/BF16; address changes only",
        "math_policy_override": False,
    }


def packed_offset(row, k_step, lane, part, *, k=K, bytes_per_lane=8):
    """Byte address for CPU contract; part is one of 8 payload bytes or one scale."""
    if k % STEP or not 0 <= row or not 0 <= k_step < k // STEP:
        raise ValueError("invalid row or K step")
    if not 0 <= lane < 32 or not 0 <= part < bytes_per_lane:
        raise ValueError("invalid lane byte")
    tile = ((row // ROWS) * (k // STEP) + k_step) * ROWS + row % ROWS
    return (tile * 32 + lane) * bytes_per_lane + part


class PackedQMV:
    def __init__(self, mx, weight, scales):
        self.mx = mx
        self.original_weight = weight
        self.original_scales = scales
        self.weight = mx.contiguous(weight.reshape(N // ROWS, ROWS, K // STEP, STEP // 8)
                                    .transpose(0, 2, 1, 3)).reshape(-1)
        self.scales = mx.contiguous(scales.reshape(N // ROWS, ROWS, K // STEP, STEP // 16)
                                   .transpose(0, 2, 1, 3)).reshape(-1)
        self.kernel = None

    def project(self, x):
        mx = self.mx
        if x.shape != (1, K) or x.dtype != mx.bfloat16:
            raise ValueError("V26 build1 accepts only B1 BF16 input [1,5120]")
        if self.kernel is None:
            self.kernel = mx.fast.metal_kernel(
                name="v26_nvfp4_rowblock_r4", input_names=["weight", "scales", "x"],
                output_names=["output"], source=(BASE / "kernel_body.metal").read_text(),
                header=(BASE / "kernel_header.metal").read_text())
        return self.kernel(inputs=[self.weight, self.scales, x],
                           template=[("T", mx.bfloat16), ("K", K)],
                           grid=(64, N // ROWS, 1), threadgroup=(64, 1, 1),
                           output_shapes=[(1, N)], output_dtypes=[mx.bfloat16])[0]


def prepare(mx, weight, scales, *, runtime_version, device_name, architecture):
    """Explicit owner action, outside measurement. This function can use the GPU."""
    if (runtime_version, device_name, architecture) != ("0.32.3", "Apple M4 Max", "applegpu_g16s"):
        raise ValueError("unqualified runtime/device")
    if mx.default_device() != mx.gpu:
        raise ValueError("GPU backend required")
    if weight.shape != (N, K // 8) or weight.dtype != mx.uint32:
        raise ValueError("expected native U32 payload")
    if scales.shape != (N, K // 16) or scales.dtype != mx.uint8:
        raise ValueError("expected native U8 NVFP4 group16 scales")
    return PackedQMV(mx, weight, scales)
