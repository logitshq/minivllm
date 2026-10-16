"""V83 native R4 SG2 with original FP4 decode and outer K-loop unroll_count(2); backend injected, stdlib-only import."""
from pathlib import Path
BASE = Path(__file__).resolve().parent
K, N = 5120, 17408  # Initial local screen target only.
ROWS, SIMD_GROUPS, THREADS = 4, 2, 64
ELIGIBLE_SHAPES = frozenset([(48, 5120), (1024, 5120), (5120, 6144), (5120, 17408), (6144, 5120), (10240, 5120), (12288, 5120), (17408, 5120), (248320, 5120)])

class Unroll2QMV:
    def __init__(self, mx, weight, scales, n, k):
        self.mx, self.inputs, self.N, self.K = mx, (weight, scales), n, k
        self.kernel = None

    def project(self, x):
        mx = self.mx
        if x.shape != (1, self.K) or x.dtype != mx.bfloat16:
            raise ValueError("only declared BF16 normalB1 activation is eligible")
        if self.kernel is None:
            header = (BASE / "kernel_header.metal").read_text()
            header += f"\nconstant int v83_in_vec_size = {self.K};\nconstant int v83_out_vec_size = {self.N};\n"
            self.kernel = mx.fast.metal_kernel(
                name=f"v83_nvfp4_unroll2_r4_sg2_n{self.N}_k{self.K}",
                input_names=["weight", "scales", "x"], output_names=["output"],
                source=(BASE / "kernel_body.metal").read_text(), header=header,
                ensure_row_contiguous=True)
        return self.kernel(inputs=[*self.inputs, x], template=[("T", mx.bfloat16), ("R", ROWS)],
                           grid=(THREADS, (self.N + ROWS * SIMD_GROUPS - 1) // (ROWS * SIMD_GROUPS), 1),
                           threadgroup=(THREADS, 1, 1), output_shapes=[(1, self.N)],
                           output_dtypes=[mx.bfloat16])[0]

def prepare(mx, weight, scales, *, runtime_version, device_name, architecture):
    if (runtime_version, device_name, architecture) != ("0.32.3", "Apple M4 Max", "applegpu_g16s") or mx.default_device() != mx.gpu:
        raise ValueError("unqualified runtime/device")
    if len(weight.shape) != 2 or weight.dtype != mx.uint32:
        raise ValueError("native U32 payload required")
    n, k = weight.shape[0], weight.shape[1] * 8
    if (n, k) not in ELIGIBLE_SHAPES or k % 512 or n % (ROWS * SIMD_GROUPS):
        raise ValueError("only current-checkpoint complete-R4-SG2 shapes are source-eligible")
    if scales.shape != (n, k // 16) or scales.dtype != mx.uint8:
        raise ValueError("native U8 group16 scales required")
    return Unroll2QMV(mx, weight, scales, n, k)
