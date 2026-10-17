"""V114 instance-local, active-V108-transaction-only broadcastM1 scope.

No MLX import/array creation at module import; no global monkeypatch. The
verbatim V113 helper constructs W/scales singleton views, never packed copies.
"""
from contextlib import contextmanager
from importlib.metadata import version
from threading import get_ident
from .broadcast_m1_113 import broadcast_m1

_FLAG = '_verify_m1_scope114'
_ABSENT = object()


def prepare_verify_m1(model, mx):
    from ..model.layers import Linear
    if version('mlx') != '0.32.3' or version('mlx-metal') != '0.32.3' or mx.default_device() != mx.gpu:
        raise RuntimeError('V114 requires pinned MLX0.32.3 GPU runtime')
    info = mx.device_info()
    if info.get('device_name') != 'Apple M4 Max' or info.get('architecture') != 'applegpu_g16s':
        raise RuntimeError('V114 requires the source-qualified M4 Max')
    ops = tuple((name, op) for name, op in model.named_modules() if type(op) is Linear)
    if len(ops) != 497 or len({id(op) for _, op in ops}) != 497 or not any(name == 'lm_head' and op is model.lm_head for name, op in ops):
        raise ValueError('exact497 original target Linear instances including untied lm_head required')
    if getattr(model.cfg, 'tie_word_embeddings', True):
        raise ValueError('V114 only supports fixed untied27B target')
    return ops


class VerifyScope:
    def __init__(self, ops, caches):
        traces = tuple(getattr(cache, '_verify108', None) for cache in caches)
        if len(traces) != 64 or any(trace is None for trace in traces):
            raise RuntimeError('V114 requires all64 active target transaction traces')
        if len({trace.owner for trace in traces}) != 1 or len({trace.length for trace in traces}) != 1:
            raise RuntimeError('different verify transaction owners/lengths')
        length = traces[0].length
        if type(length) is not int or not 1 <= length <= 8:
            raise ValueError('V114 strict verify length1..8 required')
        self.ops, self.length, self.owner = ops, length, traces[0].owner
        self.thread = get_ident()
        self.active = False
        self.names = {id(op): name for name, op in ops}
        self.calls = {}

    def metadata(self):
        return {'transaction_owner': self.owner, 'verify_T': self.length,
                'original_Linear_instances': len(self.ops), 'broadcast_M1_calls': sum(self.calls.values()),
                'distinct_Linear_calls': len(self.calls), 'every_Linear_once': len(self.calls) == 497 and all(n == 1 for n in self.calls.values()),
                'verify_lm_head_calls': self.calls.get('lm_head', 0),
                'thread_owner_matches': self.thread == get_ident(), 'flags_restored_before_eval_or_draft': not self.active,
                'pointer_policy': 'current original W/scales singleton views; no explicit packed-weight materialization',
                'actual_PSO_observed': False}


def dispatch_verify_m1(op, x, mx):
    scope = getattr(op, _FLAG, None)
    if scope is None or not scope.active or scope.thread != get_ident() or id(op) not in scope.names:
        raise RuntimeError('no active owning target verify scope')
    if x.ndim != 2 or x.shape[0] != scope.length:
        raise ValueError('target Linear input rows differ from active verify length')
    result = broadcast_m1(op, x, mx)
    name = scope.names[id(op)]
    scope.calls[name] = scope.calls.get(name, 0) + 1
    return result


@contextmanager
def verify_m1_context(runner, caches):
    if not getattr(runner, '_verify_m1_enabled114', False):
        yield None
        return
    ops = getattr(runner, '_verify_m1_ops114', ())
    if len(ops) != 497:
        raise RuntimeError('V114 enabled without preparation')
    scope = VerifyScope(ops, caches)
    previous = [(op, vars(op).get(_FLAG, _ABSENT)) for _, op in ops]
    if any(value is not _ABSENT and value is not None for op, value in previous):
        raise RuntimeError('nested or leaked per-instance verify flag')
    assigned = []
    try:
        for op, value in previous:
            object.__setattr__(op, _FLAG, scope)
            assigned.append((op, value))
        scope.active = True
        yield scope
        if len(scope.calls) != 497 or any(n != 1 for n in scope.calls.values()):
            raise RuntimeError('actual target verify did not call all497 Linears exactly once')
    finally:
        scope.active = False
        for op, value in reversed(assigned):
            if value is _ABSENT:
                object.__delattr__(op, _FLAG)
            else:
                object.__setattr__(op, _FLAG, value)
