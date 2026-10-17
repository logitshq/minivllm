"""Deferred same-input comparisons. Import is standard-library only."""
from collections import Counter


def prefix_lengths(q_length, k_length):
    if not 1 <= q_length <= k_length:
        raise ValueError('invalid causal geometry')
    return list(range(k_length-q_length+1, k_length+1))


class Hooks:
    def __init__(self, mx, nn, model, layers, gdn_module):
        self.mx, self.nn = mx, nn
        self.names = {id(m): name for name, m in model.named_modules()}
        self.layers, self.gdn_module = layers, gdn_module
        self.records, self.restore = [], []
        self.active = False
        self.T = None
        self.compiled_gates = None

    def patch(self, owner, name, replacement):
        original = getattr(owner, name)
        self.restore.append((owner, name, original))
        setattr(owner, name, replacement)
        return original

    def record(self, kind, name, inputs, output, build, metadata=None):
        self.records.append(dict(order=len(self.records), kind=kind, name=name,
                                 inputs=inputs, output=output, build=build,
                                 metadata=metadata or {}))

    def install(self):
        mx, nn, L = self.mx, self.nn, self.layers
        def method(cls, name, kind, gated=False):
            original = getattr(cls, name)
            def wrapped(op, x, *args, **kwargs):
                y = original(op, x, *args, **kwargs)
                if self.active and x.ndim >= 2 and x.shape[0] == self.T:
                    inputs = (x, *args)
                    def build():
                        ys = [original(op, x[i:i+1], *(a[i:i+1] for a in args), **kwargs) for i in range(self.T)]
                        return [('row_M1', mx.concatenate(ys, axis=0))]
                    q = getattr(op, 'q', None)
                    meta = {'quantization': q, 'input_features': getattr(op, 'in_features', x.shape[-1]),
                            'output_features': getattr(op, 'out_features', y.shape[-1]),
                            'weight_dtype': str(op.weight.dtype), 'weight_shape': list(op.weight.shape)}
                    self.record(kind, self.names[id(op)], inputs, y, build, meta)
                return y
            self.patch(cls, name, wrapped)
        method(L.Linear, '__call__', 'Linear')
        method(L.Embedding, 'as_linear', 'Embedding.as_linear')
        method(L.RMSNorm, '__call__', 'RMSNorm')
        method(L.GatedRMSNorm, '__call__', 'GatedRMSNorm', True)

        original_gates = self.gdn_module.GatedDeltaNet._gates
        def gates(op, b, a):
            y = original_gates(op, b, a)
            if self.active and b.shape[0] == self.T:
                def build():
                    rows = [original_gates(op, b[i:i+1], a[i:i+1]) for i in range(self.T)]
                    explicit = tuple(mx.concatenate([r[j] for r in rows], axis=0) for j in range(2))
                    if self.compiled_gates is None:
                        self.compiled_gates = mx.compile(self.gdn_module._gates_explicit, shapeless=False)
                    rows = [self.compiled_gates(b[i:i+1], a[i:i+1], op.A_log, op.dt_bias) for i in range(self.T)]
                    compiled = tuple(mx.concatenate([r[j] for r in rows], axis=0) for j in range(2))
                    return [('row_M1_explicit', explicit), ('row_M1_compiled_decode', compiled)]
                self.record('GDN.gates', self.names[id(op)], (b, a), y, build,
                            {'outputs': ['beta', 'g'], 'compiled_decode_reference': 'original _gates_explicit with all weights explicit'})
            return y
        self.patch(self.gdn_module.GatedDeltaNet, '_gates', gates)

        for owner, name in ((nn, 'silu'), (nn, 'softplus'), (mx, 'sigmoid')):
            original = getattr(owner, name)
            def unary(x, *args, _original=original, _name=name, **kwargs):
                y = _original(x, *args, **kwargs)
                if self.active and x.ndim >= 2 and x.shape[0] == self.T:
                    def build():
                        return [('row_M1', mx.concatenate([_original(x[i:i+1], *args, **kwargs) for i in range(self.T)], axis=0))]
                    self.record('pointwise.'+_name, 'functional', (x,), y, build)
                return y
            self.patch(owner, name, unary)

        original_sdpa = mx.fast.scaled_dot_product_attention
        def sdpa(q, k, v, *args, **kwargs):
            y = original_sdpa(q, k, v, *args, **kwargs)
            if self.active:
                assert not args and q.shape[2] == self.T and kwargs.get('mask') == 'causal'
                ends = prefix_lengths(q.shape[2], k.shape[2])
                def build():
                    results = []
                    for mode in ('row_M1_prefix_None', 'row_M1_prefix_decode_boolmask'):
                        rows = []
                        for i, end in enumerate(ends):
                            call = dict(kwargs)
                            call['mask'] = None if mode.endswith('None') else mx.ones((1, 1, 1, end), dtype=mx.bool_)
                            rows.append(original_sdpa(q[:, :, i:i+1, :], k[:, :, :end, :], v[:, :, :end, :], **call))
                        results.append((mode, mx.concatenate(rows, axis=2)))
                    return results
                self.record('attention.SDPA', 'functional', (q, k, v), y, build,
                            {'q_shape': list(q.shape), 'k_shape': list(k.shape), 'v_shape': list(v.shape),
                             'causal_prefix_lengths': ends, 'scale': kwargs['scale'],
                             'scope': 'same already-projected/normalized/rotated QKV; no cache mutation or new token inputs'})
            return y
        self.patch(mx.fast, 'scaled_dot_product_attention', sdpa)

        original_rope = mx.fast.rope
        def rope(x, *args, **kwargs):
            y = original_rope(x, *args, **kwargs)
            if self.active:
                assert x.ndim == 4 and x.shape[2] == self.T and type(kwargs['offset']) is int
                def build():
                    rows = []
                    for i in range(self.T):
                        call = dict(kwargs);call['offset'] += i
                        rows.append(original_rope(x[:, :, i:i+1, :], *args, **call))
                    return [('row_M1_offset_incremented', mx.concatenate(rows, axis=2))]
                self.record('attention.RoPE', 'functional', (x,), y, build, {'offset': kwargs['offset']})
            return y
        self.patch(mx.fast, 'rope', rope)

    def close(self):
        self.active = False
        for owner, name, original in reversed(self.restore):
            setattr(owner, name, original)
        self.restore.clear()


def summarize(rows):
    counts = {}
    first_by_kind = {}
    linear_geometry = {}
    for row in rows:
        key = row['kind'] + ':' + row['variant']
        c = counts.setdefault(key, Counter(calls=0, strict_equal=0, strict_different=0, different_elements=0))
        c['calls'] += 1;c['strict_equal'] += int(row['strict_equal'])
        c['strict_different'] += int(not row['strict_equal']);c['different_elements'] += row['different_elements']
        if not row['strict_equal'] and key not in first_by_kind:
            first_by_kind[key] = {k: row[k] for k in ('order','name','component','first_difference')}
        if row['kind'] in ('Linear','Embedding.as_linear'):
            meta = row['metadata'];q = meta['quantization']
            geometry = f"{row['kind']}:{q['mode'] if q else 'unquantized'}:K{meta['input_features']}:N{meta['output_features']}"
            g = linear_geometry.setdefault(geometry, {'calls':0,'strict_equal':0,'strict_different':0,'first_difference':None})
            g['calls'] += 1;g['strict_equal'] += int(row['strict_equal']);g['strict_different'] += int(not row['strict_equal'])
            if not row['strict_equal'] and g['first_difference'] is None:
                g['first_difference'] = {k: row[k] for k in ('order','name','component','first_difference')}
    first = next((r for r in rows if not r['strict_equal']), None)
    return {'by_kind': {k: dict(v) for k, v in counts.items()}, 'first_difference_by_kind':first_by_kind,
            'linear_by_quantization_geometry':linear_geometry, 'first_differing_captured_operation': first}
