"""E026 default-OFF BF16 GDN tail compaction; no imports initialize MLX.

Only the existing [3,10240] convolution tail is passed here. Its uint16 view is
copied with pinned MLX 0.32.3 contiguous, then viewed as BF16; no arithmetic or
qkv/state computation changes. This does not patch any model or production core.

enabled=True constructs a LAZY compaction. materialize=True is a separate,
synchronous diagnostic intervention at this call site. Both default to False.
No GPU numeric, ownership, integration or performance qualification is claimed.
"""
from importlib.metadata import version


def supported_backend(mx):
    return (version('mlx') == '0.32.3' and mx.default_device() == mx.gpu
            and mx.device_info().get('device_name') == 'Apple M5 Ultra'
            and mx.metal.device_info().get('architecture') == 'applegpu_g17d')


def own_conv_tail(mx, tail, *, enabled=False, materialize=False):
    """Compact excess backing; an already compact input may share its buffer.

    In v0.32.3, contiguous copies even row-contiguous input if the whole backing
    is > logical bytes +16KiB. A BF16 [T+3,10240] parent exceeds that threshold
    for every T>=1. This is NOT a promise that the old backing is globally freed:
    other arrays/graphs/observers may retain it, and a lazy copy has not run yet.
    """
    if type(enabled) is not bool or type(materialize) is not bool:
        raise TypeError('enabled and materialize must be bool')
    if materialize and not enabled:
        raise ValueError('materialize requires enabled=True')
    if not enabled:
        return tail
    if (tail.ndim != 2 or tail.shape != (3, 10240) or tail.dtype != mx.bfloat16
            or not supported_backend(mx)):
        return tail
    result = mx.contiguous(tail.view(mx.uint16)).view(mx.bfloat16)
    if materialize:
        mx.eval(result)
    return result
