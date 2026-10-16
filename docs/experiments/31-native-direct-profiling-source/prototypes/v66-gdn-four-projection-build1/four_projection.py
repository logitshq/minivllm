"""V66 standalone zero-copy four-GDN-projection helper. Backend injected; stdlib only.

Only observed contiguous B1 operands are qualified. Numeric runner probes actual
strides with ensure_row_contiguous=False. No packed weights or engine integration.
"""
from pathlib import Path
D = Path(__file__).resolve().parent
K = 5120
NAMES = ("qkv", "z", "b", "a")
ROWS = (10240, 6144, 48, 48)
THREADS, SIMD_GROUPS, RESULTS_PER_SIMD = 64, 2, 4
MAIN_BUFFER_ARGUMENTS = 13
INPUT_NAMES = tuple(v for name in NAMES for v in ("w_"+name, "s_"+name)) + ("x",)
OUTPUT_NAMES = tuple("y_"+name for name in NAMES)


def header():
    text = (D / "kernel_header.metal").read_text() + "\nconstant int v66_k = 5120;\n"
    return text + "".join(f"constant int v66_n_{name} = {n};\n" for name,n in zip(NAMES, ROWS))


def runtime(mx, runtime_version, device_name, architecture):
    if (runtime_version, device_name, architecture) != ("0.32.3", "Apple M4 Max", "applegpu_g16s") or mx.default_device() != mx.gpu:
        raise ValueError("only pinned MLX0.32.3/M4Max GPU is eligible")


class FourProjection:
    def __init__(self, mx, operands):
        self.mx, self.operands, self.kernel = mx, tuple(operands), None

    def project(self, x, *, x_strides):
        mx = self.mx
        if x.shape != (1, K) or x.dtype != mx.bfloat16 or tuple(x_strides) != (K, 1):
            raise ValueError("requires observed contiguous BF16 B1 K5120 input")
        if self.kernel is None:
            self.kernel = mx.fast.metal_kernel(name="v66_gdn_four_original_pointers_r4",
                input_names=INPUT_NAMES, output_names=OUTPUT_NAMES,
                source=(D / "kernel_body.metal").read_text(), header=header(),
                ensure_row_contiguous=False)
        return tuple(self.kernel(inputs=[*self.operands, x], template=[("T", mx.bfloat16)],
            grid=(64, 2060, 1), threadgroup=(64, 1, 1),
            output_shapes=[(1,n) for n in ROWS], output_dtypes=[mx.bfloat16]*4))


def prepare(mx, operands, *, observed_strides, runtime_version, device_name, architecture):
    runtime(mx, runtime_version, device_name, architecture)
    operands = tuple(operands)
    expected = tuple(v for n in ROWS for v in ((K//8,1),(K//16,1)))
    if len(operands) != 8 or tuple(tuple(s) for s in observed_strides) != expected:
        raise ValueError("four original W/scale pairs and independently observed contiguous strides required")
    if len({id(x) for x in operands}) != 8:
        raise ValueError("operand handles must be distinct")
    for j,n in enumerate(ROWS):
        w,s = operands[2*j:2*j+2]
        if w.shape != (n,K//8) or w.dtype != mx.uint32 or s.shape != (n,K//16) or s.dtype != mx.uint8:
            raise ValueError("native U32[N,640]/U8[N,320] payloads required")
    # Strong references preserve every exact original handle; no data operation here.
    return FourProjection(mx, operands)


class R4Controls:
    """Four separate custom kernels, with the same header/math as the fused candidate."""
    def __init__(self, mx): self.mx, self.kernels = mx, {}
    def project(self, operands, x, *, x_strides):
        mx = self.mx
        if x.shape != (1,K) or x.dtype != mx.bfloat16 or tuple(x_strides)!=(K,1):
            raise ValueError("observed contiguous B1 input required")
        out=[]
        for j,n in enumerate(ROWS):
            if n not in self.kernels:
                self.kernels[n]=mx.fast.metal_kernel(name=f"v66_native_r4_control_n{n}",
                    input_names=["weight","scales","x"],output_names=["output"],
                    source=(D/"control_body.metal").read_text(),
                    header=header()+f"\nconstant int v66_control_n = {n};\n",ensure_row_contiguous=False)
            out.append(self.kernels[n](inputs=[*operands[2*j:2*j+2],x],template=[("T",mx.bfloat16)],
                grid=(64,n//8,1),threadgroup=(64,1,1),output_shapes=[(1,n)],output_dtypes=[mx.bfloat16])[0])
        return tuple(out)
