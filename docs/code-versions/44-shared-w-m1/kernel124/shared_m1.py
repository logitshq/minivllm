"""V124 standalone source prototype. Injected backend; no MLX import or engine patch.

The caller supplies independently observed strides; the helper never normalizes
or copies operands. CPU checks do not qualify Metal arithmetic or performance.
"""
from pathlib import Path
D = Path(__file__).resolve().parent
ROWS = (2, 4, 5)
SHAPES = frozenset(((48, 5120), (1024, 5120), (5120, 6144),
    (5120, 17408), (6144, 5120), (10240, 5120), (12288, 5120),
    (17408, 5120), (248320, 5120)))
Q = {'group_size': 16, 'bits': 4, 'mode': 'nvfp4'}


def validate_geometry(n, k, weight_strides, scale_strides):
    if type(n) is not int or type(k) is not int or (n, k) not in SHAPES:
        raise ValueError('only complete current-target SG2/R4 shapes')
    if tuple(weight_strides) != (k // 8, 1) or tuple(scale_strides) != (k // 16, 1):
        raise ValueError('original packed W and scales must be observed canonical')


def validate_input(shape, strides, k):
    if len(shape) != 2 or type(shape[0]) is not int or shape[0] not in ROWS or shape[1] != k:
        raise ValueError('single sequence T2/4/5 by K only')
    if tuple(strides) != (k, 1):
        raise ValueError('observed contiguous rows required; no hidden copy')


def prepare(mx, op, *, weight_strides, scale_strides,
            runtime_version, device_name, architecture):
    if (runtime_version, device_name, architecture) != ('0.32.3', 'Apple M4 Max', 'applegpu_g16s') or mx.default_device() != mx.gpu:
        raise ValueError('only pinned MLX0.32.3 M4 Max GPU')
    if getattr(op, 'q', None) != Q or getattr(op, 'biases', None) is not None:
        raise ValueError('NVFP4 g16/b4 and no bias/global scale only')
    w, s = op.weight, op.scales
    if w.ndim != 2 or w.dtype != mx.uint32 or s.dtype != mx.uint8:
        raise ValueError('original U32 weights/U8 scales only')
    n, k = w.shape[0], w.shape[1] * 8
    validate_geometry(n, k, weight_strides, scale_strides)
    if s.shape != (n, k // 16) or w is s:
        raise ValueError('independent complete original W/scales handles required')
    return SharedM1(mx, op, w, s, n, k)


class SharedM1:
    def __init__(self, mx, op, weight, scales, n, k):
        self.mx, self.op, self.weight, self.scales = mx, op, weight, scales
        self.n, self.k, self.kernel = n, k, None

    def project(self, x, *, observed_x_strides):
        mx = self.mx
        validate_input(x.shape, observed_x_strides, self.k)
        if x.dtype != mx.bfloat16:
            raise ValueError('BF16 only')
        if (self.op.weight is not self.weight or self.op.scales is not self.scales
                or self.op.q != Q or getattr(self.op, 'biases', None) is not None
                or self.weight.shape != (self.n, self.k // 8) or self.weight.dtype != mx.uint32
                or self.scales.shape != (self.n, self.k // 16) or self.scales.dtype != mx.uint8):
            raise RuntimeError('prepared original operand identity/metadata changed')
        if self.kernel is None:
            self.kernel = mx.fast.metal_kernel(
                name='v124_shared_weight_m1_r4', input_names=['weight', 'scales', 'x'],
                output_names=['output'], source=(D / 'kernel_body.metal').read_text(),
                header=(D / 'kernel_header.metal').read_text() + '\n' + (D / 'shared_impl.metal').read_text(),
                ensure_row_contiguous=False)
        return self.kernel(inputs=[self.weight, self.scales, x],
            template=[('T', mx.bfloat16), ('VERIFY_T', x.shape[0]), ('K_SIZE', self.k), ('N_SIZE', self.n)],
            grid=(64, self.n // 8, 1), threadgroup=(64, 1, 1),
            output_shapes=[(x.shape[0], self.n)], output_dtypes=[mx.bfloat16])[0]
