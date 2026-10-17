"""Local opt-in NVFP4 shape probe only. No MLX import or model integration."""


def broadcast_m1(op, x, mx):
    q = getattr(op, 'q', None)
    bias = getattr(op, 'biases', None)
    if q != {'group_size':16, 'bits':4, 'mode':'nvfp4'} or bias is not None:
        raise ValueError('V113 requires target NVFP4 g16/b4 without bias')
    if x.ndim != 2 or x.dtype != mx.bfloat16 or not 1 <= x.shape[0] <= 8:
        raise ValueError('V113 requires T1..8 by K BF16 input')
    T, K = x.shape
    w, scales = op.weight, op.scales
    N = w.shape[0]
    if (K not in (5120,6144,17408) or N <= 0 or N % 8 or w.shape != (N,K//8)
            or w.dtype != mx.uint32 or scales.shape != (N,K//16) or scales.dtype != mx.uint8):
        raise ValueError('V113 fixed target packed matrix geometry mismatch')
    # No broadcast_to(T,...), tile, repeat, contiguous, dequantize, or weight copy.
    # Rank3 W prevents the host's rank2-W flattening shortcut.
    y = mx.quantized_matmul(x[:,None,:], w[None,...], scales[None,...], None,
                            transpose=True, group_size=16, bits=4, mode='nvfp4')
    if y.shape != (T,1,N) or y.dtype != x.dtype:
        raise RuntimeError('V113 quantized_matmul returned wrong batch/M/output shape')
    return y.reshape(T,N)
