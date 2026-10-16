"""Standalone NVFP4 factor-fold candidate. Injected backend; stdlib-only import."""
from pathlib import Path

BASE = Path(__file__).resolve().parent
K, N = 5120, 17408


class FoldedQMV:
    def __init__(self, mx, weight, scales):
        self.mx = mx
        self.inputs = (weight, scales)
        self.kernel = None

    def project(self, x):
        mx = self.mx
        if x.shape != (1, K) or x.dtype != mx.bfloat16:
            raise ValueError("only BF16 B1 [1,5120] is eligible")
        if self.kernel is None:
            self.kernel = mx.fast.metal_kernel(
                name="v31_nvfp4_factor_fold_r4", input_names=["weight", "scales", "x"],
                output_names=["output"], source=(BASE / "kernel_body.metal").read_text(),
                header=(BASE / "folded_header.metal").read_text())
        return self.kernel(inputs=[*self.inputs, x], template=[("T", mx.bfloat16), ("R", 4)],
                           grid=(64, N // 8, 1), threadgroup=(64, 1, 1),
                           output_shapes=[(1, N)], output_dtypes=[mx.bfloat16])[0]


def prepare(mx, weight, scales, *, runtime_version, device_name, architecture):
    if (runtime_version, device_name, architecture) != ("0.32.3", "Apple M4 Max", "applegpu_g16s") or mx.default_device() != mx.gpu:
        raise ValueError("unqualified runtime/device")
    if weight.shape != (N, K // 8) or weight.dtype != mx.uint32:
        raise ValueError("native U32 payload required")
    if scales.shape != (N, K // 16) or scales.dtype != mx.uint8:
        raise ValueError("native U8 group16 scales required")
    return FoldedQMV(mx, weight, scales)
