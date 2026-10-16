"""E010 experiment only: keep M=T while broadcasting packed NAX weights."""
from contextlib import contextmanager
import mlx.core as mx


def prefill_linear(op, x, shape):
    B, T = shape
    K, N = getattr(op, 'in_features', 0), getattr(op, 'out_features', 0)
    if (B >= 2 and T >= 14 and K > 4096 and K % 64 == 0 and N % 32 == 0
            and x.dtype == mx.bfloat16 and x.shape == (B * T, K)
            and getattr(op, 'q', None) == {'group_size': 16, 'bits': 4, 'mode': 'nvfp4'}
            and ((N + 31) // 32) * ((T + 31) // 32) > 256):
        # Explicit weight batch prevents MLX0.32.3 from flattening M=B*T.
        # The inequality proves the original single-sequence split_k is one.
        # Both calls therefore use NAX with the SAME M-dependent BM tile.
        w = mx.broadcast_to(op.weight, (B,) + op.weight.shape)
        s = mx.broadcast_to(op.scales, (B,) + op.scales.shape)
        return mx.quantized_matmul(x.reshape(B, T, K), w, s, transpose=True,
                                   **op.q).reshape(B * T, N)
    return mx.concatenate([op(row) for row in x.reshape(B, T, -1)], axis=0)


@contextmanager
def enabled(active=True):
    from tutorial_code.miniengine.model import layers, attention, gdn, qwen3_5
    from tutorial_code.miniengine.engine import model_runner
    modules = (layers, attention, gdn, qwen3_5)
    old = [m.prefill_linear for m in modules]
    gate = model_runner.prefill_batch_backend_supported
    if active:
        for m in modules:
            m.prefill_linear = prefill_linear
        model_runner.prefill_batch_backend_supported = lambda: True
    try:
        yield
    finally:
        for m, fn in zip(modules, old):
            m.prefill_linear = fn
        model_runner.prefill_batch_backend_supported = gate
