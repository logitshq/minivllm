"""Injected-backend standalone B1 L0 gate/up experiment; import is stdlib-only."""
from pathlib import Path

BASE = Path(__file__).resolve().parent
K, N = 5120, 17408


class GateUp:
    def __init__(self, mx, gate_weight, gate_scales, up_weight, up_scales):
        self.mx = mx
        self.inputs = (gate_weight, gate_scales, up_weight, up_scales)
        self.kernels = {}
        self.pointwise_kernel = None

    def _epilogue_source(self):
        return (BASE / "epilogue.metal").read_text()

    def pointwise(self, gate, up):
        mx = self.mx
        if any(v.shape != (1, N) or v.dtype != mx.bfloat16 for v in (gate, up)):
            raise ValueError("BF16 gate/up [1,17408] required")
        if self.pointwise_kernel is None:
            self.pointwise_kernel = mx.fast.metal_kernel(
                name="v28_typed_pointwise", input_names=["gate", "up"], output_names=["output"],
                source=(BASE / "pointwise_body.metal").read_text(), header=self._epilogue_source())
        return self.pointwise_kernel(inputs=[gate, up], template=[("T", mx.bfloat16)],
                                     grid=(N, 1, 1), threadgroup=(256, 1, 1),
                                     output_shapes=[(1, N)], output_dtypes=[mx.bfloat16])[0]

    def project(self, x, *, debug=False):
        mx = self.mx
        if x.shape != (1, K) or x.dtype != mx.bfloat16 or type(debug) is not bool:
            raise ValueError("BF16 B1 [1,5120] and bool debug required")
        if debug not in self.kernels:
            outputs = ["output", "gate_output", "up_output"] if debug else ["output"]
            self.kernels[debug] = mx.fast.metal_kernel(
                name="v28_gateup_r4_debug" if debug else "v28_gateup_r4",
                input_names=["gate_weight", "gate_scales", "up_weight", "up_scales", "x"],
                output_names=outputs, source=(BASE / "kernel_body.metal").read_text(),
                header=(f"#define V28_DEBUG_OUT {int(debug)}\n" +
                        (BASE / "kernel_header.metal").read_text() + self._epilogue_source()))
        count = 3 if debug else 1
        return self.kernels[debug](inputs=[*self.inputs, x], template=[("T", mx.bfloat16), ("K", K)],
                                   grid=(64, N // 8, 1), threadgroup=(64, 1, 1),
                                   output_shapes=[(1, N)] * count, output_dtypes=[mx.bfloat16] * count)


def prepare(mx, gate_weight, gate_scales, up_weight, up_scales, *, runtime_version, device_name, architecture):
    if (runtime_version, device_name, architecture) != ("0.32.3", "Apple M4 Max", "applegpu_g16s") or mx.default_device() != mx.gpu:
        raise ValueError("unqualified runtime/device")
    for w, s in ((gate_weight, gate_scales), (up_weight, up_scales)):
        if w.shape != (N, K // 8) or w.dtype != mx.uint32 or s.shape != (N, K // 16) or s.dtype != mx.uint8:
            raise ValueError("native NVFP4 group16 U32/U8 [17408,5120] required")
    return GateUp(mx, gate_weight, gate_scales, up_weight, up_scales)
