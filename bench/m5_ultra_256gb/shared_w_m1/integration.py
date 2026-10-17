"""CPU-safe, default-OFF projection bridge; no installed engine replacement.

The current and frozen E020 target use Linear.__call__(X), including the solo
SpecRunner verify -> ForwardContext('prefill') -> GDN/Attention prefill path.
For canonical BF16 X[T,5120], T2/4/5 and original 2-D NVFP4 W, MLX 0.32.3
dispatches native **fp_qmv_wide**, not per-row M1. It already reuses W across T.
Its k_lanes=16, float4 dot and shuffle ladder differ from the M1/candidate
K512 traversal, four-term expression and simd_sum. E027's 33 local M1 tests
therefore do NOT qualify replacing native Linear, verify or batched decode.

This module makes that missing comparison executable without changing its
baseline. Example (caller owns GPU/resource admission and raw-bit auditing):

    with prepare_qualification(mx, op, enabled=True, ...) as bridge:
        native = bridge.native_multirow(x, observed_x_strides=actual_strides)
        reference = bridge.row_m1(x, observed_x_strides=actual_strides)
        # Evaluate both complete outputs and compare original BF16 bits here.
        # On ANY mismatch, save both arrays and stop before calling shared.
        shared = bridge.shared(x, observed_x_strides=actual_strides)
        # Independently compare full finite bits; verify unchanged operands.

The three explicit entries are diagnostics, not an automatic equality checker.
No caller-supplied boolean or receipt enables native substitution. Even a local
bridge PASS would still need propagated full-model/cache/raw-token correctness
and same-round HTTP throughput versus complete-request latency qualification.
Never place these observer calls in mx.compile or count them as performance.

Only E027-tested (N,K)=(48,5120),(1024,5120),(10240,5120), T2/4/5 are admitted.
One caller-owned scope keeps one reusable candidate preparation for the original
operator. It never copies/rebinds W/scales, patches classes/modules/instances,
or retains X/outputs. Source pins describe the inspected route, not an unobserved
GPU-kernel witness. No MLX import, model loading or device execution occurs here.
"""
import hashlib
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
CANDIDATE_SHA = '7af958ddc747142cd2996920ed7d09e6ef3d48844aaceaef3612578070fc49fc'
QUALIFIED_GEOMETRIES = frozenset(((48, 5120), (1024, 5120), (10240, 5120)))
ROWS = (2, 4, 5)
Q = {'group_size': 16, 'bits': 4, 'mode': 'nvfp4'}
INSPECTED_SOURCE_PINS = {
    'model/layers.py': 'c2c9936df2a34877edb5e1702152583e54549f69b37d7c93e9996af1e7c20908',
    'engine/spec_runner.py': 'b6e83506633b92e75998dd51d85488fccff12aa3bb69611ad31729a0c9040a73',
    'model/gdn.py': 'c88096bf3b51a1ad7a24dd4f4e2c71bbb6b220b288b83a9242ca190d6c0a5030',
    'model/attention.py': '112997ec74e55ecdb88641e86f7bb29ad8c6c6a73481f1afe463a4332156d52c',
}
MLX_DISPATCH_SHA = '36b79607ee01bc213511dcdad590b2dc3b96099300551cbc3e7e6c7e13ba5982'


def contract():
    return {'schema': 'm5-shared-w-native-bridge-api-v1', 'default_enabled': False,
            'rows': list(ROWS), 'geometries_N_K': [list(v) for v in sorted(QUALIFIED_GEOMETRIES)],
            'native_expression': 'mx.quantized_matmul(X, W, scales, None, transpose=True, **Q)',
            'native_dispatch_static': 'fp_qmv_wide; k_lanes=16; vecs_per_tg=T',
            'm1_expression': 'concatenate per-row native QMM(X[i:i+1], W, scales)',
            'native_route_qualified': False, 'engine_installed': False,
            'performance_qualified': False, 'full_model_qualified': False,
            'candidate_sha256': CANDIDATE_SHA, 'inspected_core_pins': dict(INSPECTED_SOURCE_PINS),
            'inspected_mlx_dispatch_sha256': MLX_DISPATCH_SHA,
            'scope': 'same-input numerical bridge only; caller must gate before shared',
            'counter_scope': 'host returned lazy graphs, not completed GPU work or compiler count'}


def _enabled(value):
    if type(value) is not bool:
        raise TypeError('enabled must be bool')
    return value


def prepare_native_route(mx, op, *, enabled=False):
    """OFF is inert; ON cannot silently replace the unqualified native baseline."""
    if not _enabled(enabled):
        return None
    raise ValueError('native qmv_wide substitution is unqualified; use the explicit bridge diagnostic')


def _candidate():
    path = HERE / 'candidate.py'
    if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != CANDIDATE_SHA:
        raise ValueError('frozen E027 candidate source mismatch')
    spec = importlib.util.spec_from_file_location('_m5_shared_w_bridge_candidate', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare_qualification(mx, op, *, enabled=False, weight_strides=None,
                          scale_strides=None, runtime_version=None,
                          device_name=None, architecture=None):
    """Return a diagnostic context, or None when OFF before touching operands."""
    if not _enabled(enabled):
        return None
    # Narrow the older nine-shape implementation to the actually tested domain.
    shape = getattr(op.weight, 'shape', ())
    if (len(shape) != 2 or any(type(v) is not int for v in shape)
            or (shape[0], shape[1] * 8) not in QUALIFIED_GEOMETRIES):
        raise ValueError('only the three E027-tested geometries may enter this bridge')
    helper = _candidate()
    prepared = helper.prepare(mx, op, enabled=True, weight_strides=weight_strides,
        scale_strides=scale_strides, runtime_version=runtime_version,
        device_name=device_name, architecture=architecture)
    return ProjectionBridge(mx, op, helper, prepared)


class ProjectionBridge:
    """Explicit projection-only scope; no model hook or automatic adoption."""

    def __init__(self, mx, op, helper, prepared):
        self.mx, self.op, self.helper, self.prepared = mx, op, helper, prepared
        self.active, self.closed = False, False
        self.native_calls = self.row_m1_calls = self.row_m1_launches = 0
        self._final_counts = None

    def __enter__(self):
        if self.active or self.closed:
            raise RuntimeError('bridge scope already active or closed')
        self.active = True
        return self

    def _check(self, x, strides):
        if not self.active or self.closed:
            raise RuntimeError('projection requires an active bridge scope')
        p, mx = self.prepared, self.mx
        if (mx.default_device() != mx.gpu or self.op.weight is not p.weight
                or self.op.scales is not p.scales or not self.helper._quantization(self.op)
                or tuple(p.weight.shape) != (p.n, p.k // 8) or p.weight.dtype != mx.uint32
                or tuple(p.scales.shape) != (p.n, p.k // 16) or p.scales.dtype != mx.uint8):
            raise RuntimeError('original GPU/operator/packed operands changed')
        self.helper.validate_input(x.shape, strides, p.k)
        if x.dtype != mx.bfloat16:
            raise ValueError('BF16 input required')

    def native_multirow(self, x, *, observed_x_strides):
        self._check(x, observed_x_strides)
        result = self.mx.quantized_matmul(x, self.op.weight, self.op.scales, None,
                                         transpose=True, **Q)
        self.native_calls += 1
        return result

    def row_m1(self, x, *, observed_x_strides):
        self._check(x, observed_x_strides)
        # Deliberately explicit alternate oracle, never called a native baseline.
        outputs = [self.mx.quantized_matmul(x[i:i + 1], self.op.weight,
                   self.op.scales, None, transpose=True, **Q) for i in range(x.shape[0])]
        result = self.mx.concatenate(outputs, axis=0)
        self.row_m1_calls += 1
        self.row_m1_launches += x.shape[0]
        return result

    def shared(self, x, *, observed_x_strides):
        self._check(x, observed_x_strides)
        return self.prepared.project(x, observed_x_strides=observed_x_strides)

    def counts(self):
        if self.closed:
            return dict(self._final_counts)
        return {'native_multirow': self.native_calls, 'row_m1': self.row_m1_calls,
                'row_m1_launches': self.row_m1_launches,
                'shared': self.prepared.launch_count,
                'kernel_callable_builds': self.prepared.kernel_build_count}

    def close(self):
        if not self.closed:
            self._final_counts = self.counts()
            self.active, self.closed = False, True
            self.mx = self.op = self.helper = self.prepared = None

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False
