"""Default-OFF M5 shared-W M1 numerical candidate; injected MLX only.

No runtime imports, model loading, engine patch, repack, or hidden contiguous
copy. ``prepare(..., enabled=False)`` returns None without touching backend or
operands; the caller explicitly retains its baseline path. Enabled preparation
requires independently observed canonical GPU strides, the exact NVFP4 operator
and M5 Ultra/MLX 0.32.3 backend. Unsupported cases fail, never silently fall back.

Keep one prepared object per original operator. Its Metal callable is created
on the first project and reused, with original W/scales passed on every call.
Only the original operator, W/scales, source strings, and callable are retained;
X/output are never cached. No global cache or monkeypatch is installed.

The Metal/header/license are the pinned, unchanged M4 publication bytes. They
preserve each lane's K512 traversal, four-term qdot expression, simd_sum and
BF16 store; compilation/register allocation on M5 still requires fresh strict
device qualification. CPU tests establish no numerical or performance result.
Independent native reference: concatenate per-row mx.quantized_matmul calls
on X[i:i+1], original W/scales, transpose=True and Q; no candidate code is used
to compute that reference. Caller must evaluate/compare complete finite BF16
bits and verify all operands unchanged. Whole-model and HTTP qualification are
separate, as are benchmarks against a qualified broadcast-M1 baseline.
"""
from pathlib import Path
import hashlib

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / 'docs/code-versions/44-shared-w-m1/kernel124'
SOURCE_PINS = {
    'kernel_header.metal': '17759af8d59a7ba185040630de07af69dd3335538e19094bd1f541c2b1140c6b',
    'shared_impl.metal': '6dfe076f5a45af62aa14c644826791f3aa143d2e92bc230a0f07c39886fcecc9',
    'kernel_body.metal': '54746243dbaf9b73734dcf8337691016c53b5b68e277afa01451d7a72dd3e2b0',
    'LICENSE-MLX.txt': 'ccfab7ccb2ea306f71531c8ca77bb55507606cd90768b1e32b8b52ab5b48cf01',
}
ROWS = (2, 4, 5)
SHAPES = frozenset(((48, 5120), (1024, 5120), (5120, 6144),
    (5120, 17408), (6144, 5120), (10240, 5120), (12288, 5120),
    (17408, 5120), (248320, 5120)))
Q = {'group_size': 16, 'bits': 4, 'mode': 'nvfp4'}
RUNTIME = ('0.32.3', 'Apple M5 Ultra', 'applegpu_g17d')


def contract():
    return {'schema': 'm5-shared-w-m1-candidate-v1', 'default_enabled': False,
            'runtime': list(RUNTIME), 'rows': list(ROWS), 'geometries_N_K': [list(v) for v in sorted(SHAPES)],
            'quantization': dict(Q), 'biases': None, 'weight_dtype': 'uint32', 'scale_dtype': 'uint8',
            'input_output_dtype': 'bfloat16', 'source_directory': str(SOURCE), 'source_pins': dict(SOURCE_PINS),
            'grid': [64, 'N//8', 1], 'threadgroup': [64, 1, 1], 'ensure_row_contiguous': False,
            'unsupported': 'raise; no implicit fallback', 'prepared_lifetime': 'caller-owned original operator lifetime',
            'kernel_cache': 'one lazy Metal callable per prepared object; no X/output cache',
            'counter_scope': 'kernel_build_count counts Python Metal callable construction; launch_count counts returned lazy graphs, not completed GPU work',
            'stride_evidence': 'caller must independently observe actual GPU descriptor strides on current operands',
            'numerical_qualified': False, 'performance_qualified': False, 'full_model_qualified': False}


def _ints(values, expected):
    return (type(values) in (tuple, list) and len(values) == len(expected)
            and all(type(v) is int and v == n for v, n in zip(values, expected)))


def validate_geometry(n, k, weight_strides, scale_strides):
    if type(n) is not int or type(k) is not int or (n, k) not in SHAPES:
        raise ValueError('only nine complete SG2/R4 target geometries are registered')
    if not _ints(weight_strides, (k // 8, 1)) or not _ints(scale_strides, (k // 16, 1)):
        raise ValueError('original packed W/scales require observed canonical element strides')


def validate_input(shape, strides, k):
    if (type(shape) not in (tuple, list) or len(shape) != 2 or type(shape[0]) is not int
            or shape[0] not in ROWS or type(shape[1]) is not int or shape[1] != k):
        raise ValueError('only a single sequence with T2/4/5 and full K is registered')
    if not _ints(strides, (k, 1)):
        raise ValueError('actual X requires canonical row/column element strides; no hidden copy')


def _quantization(op):
    q = getattr(op, 'q', None)
    return (type(q) is dict and set(q) == set(Q) and type(q['group_size']) is int
            and type(q['bits']) is int and q == Q and getattr(op, 'biases', None) is None)


def _read_sources():
    result = {}
    for name, expected in SOURCE_PINS.items():
        path = SOURCE / name
        if path.is_symlink() or not path.resolve().is_relative_to(SOURCE.resolve()):
            raise ValueError('pinned source must remain inside immutable publication directory')
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError('shared-W source SHA mismatch: ' + name)
        result[name] = raw.decode('utf-8')
    return result


def prepare(mx, op, *, enabled=False, weight_strides=None, scale_strides=None,
            runtime_version=None, device_name=None, architecture=None):
    if type(enabled) is not bool:
        raise TypeError('enabled must be bool')
    if not enabled:
        return None
    if ((runtime_version, device_name, architecture) != RUNTIME or mx.default_device() != mx.gpu
            or mx.device_info().get('device_name') != RUNTIME[1]
            or mx.metal.device_info().get('architecture') != RUNTIME[2]):
        raise ValueError('only actual M5 Ultra/applegpu_g17d GPU with pinned MLX0.32.3 is registered')
    if not _quantization(op):
        raise ValueError('exact NVFP4 g16/b4, no biases, original packed operands required')
    w, s = op.weight, op.scales
    if (w.ndim != 2 or s.ndim != 2 or w.dtype != mx.uint32 or s.dtype != mx.uint8
            or any(type(v) is not int for v in (*w.shape, *s.shape))):
        raise ValueError('original rank2 U32 weights and rank2 U8 scales required')
    n, k = w.shape[0], w.shape[1] * 8
    validate_geometry(n, k, weight_strides, scale_strides)
    if not _ints(s.shape, (n, k // 16)) or w is s:
        raise ValueError('independent original weight/scale handles with complete shapes required')
    sources = _read_sources()
    return SharedM1(mx, op, w, s, n, k, sources)


class SharedM1:
    """Reusable preparation; not thread-safe and never an engine installation."""

    def __init__(self, mx, op, weight, scales, n, k, sources):
        self.mx, self.op, self.weight, self.scales = mx, op, weight, scales
        self.n, self.k = n, k
        self._body = sources['kernel_body.metal']
        self._header = sources['kernel_header.metal'] + '\n' + sources['shared_impl.metal']
        self._kernel = None
        # MLX execution is lazy: these are host-call witnesses, not GPU timings
        # or proof of completed Metal compilation/execution.
        self.kernel_build_count = 0
        self.launch_count = 0

    def project(self, x, *, observed_x_strides):
        mx = self.mx
        if mx.default_device() != mx.gpu:
            raise RuntimeError('prepared shared-W route requires the original GPU backend')
        if (self.op.weight is not self.weight or self.op.scales is not self.scales or not _quantization(self.op)
                or not _ints(self.weight.shape, (self.n, self.k // 8)) or self.weight.dtype != mx.uint32
                or not _ints(self.scales.shape, (self.n, self.k // 16)) or self.scales.dtype != mx.uint8):
            raise RuntimeError('prepared original operand identity/metadata changed')
        validate_input(x.shape, observed_x_strides, self.k)
        if x.dtype != mx.bfloat16:
            raise ValueError('BF16 input is required')
        if self._kernel is None:
            kernel = mx.fast.metal_kernel(
                name='m5_shared_weight_m1_r4_v1', input_names=['weight', 'scales', 'x'],
                output_names=['output'], source=self._body, header=self._header,
                ensure_row_contiguous=False)
            self._kernel = kernel
            self.kernel_build_count += 1
        result = self._kernel(inputs=[self.weight, self.scales, x],
            template=[('T', mx.bfloat16), ('VERIFY_T', x.shape[0]), ('K_SIZE', self.k), ('N_SIZE', self.n)],
            grid=(64, self.n // 8, 1), threadgroup=(64, 1, 1),
            output_shapes=[(x.shape[0], self.n)], output_dtypes=[mx.bfloat16])[0]
        self.launch_count += 1
        return result
