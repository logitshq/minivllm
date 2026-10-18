"""Default-OFF stdlib FP32 state-log experiment, never a runtime adapter.

Every arithmetic operation below rounds to binary32. No FMA, exp, normalization,
parallel reduction, GPU, MLX, model loading or production installation is used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import struct
import sys


def f32(x):
    value = struct.unpack('<f', struct.pack('<f', x))[0]
    if not math.isfinite(value):
        raise ValueError('nonfinite FP32')
    return value


def raw(values):
    return b''.join(struct.pack('<f', x) for x in values)


def flat(state):
    return tuple(x for row in state for x in row)


class Dense:
    """Independent eager oracle: compute each new matrix, then output."""
    def __init__(self, state):
        self.alive = True
        self.reset(state)

    def check(self):
        if not self.alive:
            raise RuntimeError('retired state')

    def reset(self, state):
        self.check()
        self.s = tuple(tuple(f32(x) for x in row) for row in state)
        self.dk, self.dv = len(self.s), len(self.s[0])

    def state(self):
        self.check()
        return self.s

    def step(self, x):
        self.check()
        ex, ks, key, beta, value, qs, query = x
        delta = []
        for j in range(self.dv):
            kv = f32(0)
            for i in range(self.dk):
                kv = f32(kv + f32(f32(ks * key[i]) * self.s[i][j]))
            delta.append(f32(f32(value[j] - f32(ex * kv)) * beta))
        updated = []
        for i in range(self.dk):
            row = []
            for j in range(self.dv):
                row.append(f32(f32(ex * self.s[i][j]) +
                               f32(f32(ks * key[i]) * delta[j])))
            updated.append(tuple(row))
        self.s = tuple(updated)
        output = []
        for j in range(self.dv):
            acc = f32(0)
            for i in range(self.dk):
                acc = f32(acc + f32(f32(qs * query[i]) * self.s[i][j]))
            output.append(acc)
        return tuple(output)

    def clone(self):
        self.check()
        return Dense(self.s)

    def retire(self):
        self.check()
        self.alive = False
        self.s = ()


class Deferred:
    """Bounded original-factor log; merge means sequential replay, never sum."""
    def __init__(self, state, merge):
        if type(merge) is not int or merge not in (1, 2, 4, 8):
            raise ValueError('merge must be 1/2/4/8')
        self.merge, self.alive = merge, True
        self.base_writes = self.log_writes = self.steps = self.merges = 0
        self.max_log_entries = 0
        self.reset(state)

    def check(self):
        if not self.alive:
            raise RuntimeError('retired state')

    def reset(self, state):
        self.check()
        self.base = tuple(tuple(f32(x) for x in row) for row in state)
        self.dk, self.dv = len(self.base), len(self.base[0])
        self.log = ()

    def cell(self, i, j, tail=None):
        self.check()
        value = self.base[i][j]
        entries = self.log if tail is None else self.log + (tail,)
        for ex, ks, key, delta in entries:
            decayed = f32(ex * value)
            scaled_key = f32(ks * key[i])
            correction = f32(scaled_key * delta[j])
            value = f32(decayed + correction)
        return value

    def state(self):
        self.check()
        return tuple(tuple(self.cell(i, j) for j in range(self.dv))
                     for i in range(self.dk))

    def step(self, x):
        self.check()
        ex, ks, key, beta, value, qs, query = x
        correction = []
        for column in range(self.dv):
            accumulator = f32(0)
            for row in range(self.dk):
                scaled_key = f32(ks * key[row])
                product = f32(scaled_key * self.cell(row, column))
                accumulator = f32(accumulator + product)
            prediction = f32(ex * accumulator)
            residual = f32(value[column] - prediction)
            correction.append(f32(residual * beta))
        entry = (ex, ks, tuple(key), tuple(correction))
        output = []
        for column in range(self.dv):
            accumulator = f32(0)
            for row in range(self.dk):
                scaled_query = f32(qs * query[row])
                product = f32(scaled_query * self.cell(row, column, entry))
                accumulator = f32(accumulator + product)
            output.append(accumulator)
        self.steps += 1
        if len(self.log) + 1 == self.merge:
            self.base = tuple(tuple(self.cell(i, j, entry) for j in range(self.dv))
                              for i in range(self.dk))
            self.log = ()
            self.base_writes += self.dk * self.dv
            self.merges += 1
        else:
            self.log += (entry,)
            self.log_writes += 2 + self.dk + self.dv  # ex, ks, key, delta
        self.max_log_entries = max(self.max_log_entries, len(self.log))
        return tuple(output)

    def clone(self):
        self.check()
        clone = Deferred(self.base, self.merge)
        clone.log = self.log  # all entries immutable; later updates replace tuple
        return clone

    def retire(self):
        self.check()
        self.alive = False
        self.base, self.log = (), ()


def inputs(seed, bank, count, dk=8, dv=6):
    rng = random.Random(seed)
    for t in range(count):
        if bank == 'random':
            key = [rng.uniform(-0.3, 0.3) for _ in range(dk)]
            value = [rng.uniform(-2, 2) for _ in range(dv)]
            query = [rng.uniform(-1, 1) for _ in range(dk)]
            ex, ks, beta, qs = rng.uniform(.8, 1), .75, .3, .5
        elif bank == 'cancellation':
            key = [(-1 if i % 2 else 1) * .125 for i in range(dk)]
            value = [(-1 if (t+j) % 2 else 1) * 2**12 for j in range(dv)]
            query = [(-1 if (t+i) % 2 else 1) for i in range(dk)]
            ex, ks, beta, qs = 1.0, 1.25, .125, .25
        elif bank == 'rounding_edges':
            tiny = struct.unpack('<f', struct.pack('<I', 1))[0]
            key = [(.125 + (i % 3) * 2**-26) * (-1 if i % 2 else 1)
                   for i in range(dk)]
            value = [tiny if (t+j) % 5 == 0 else (-1 if j % 2 else 1) * (1+2**-23)
                     for j in range(dv)]
            query = [1 + (i % 3) * 2**-23 for i in range(dk)]
            ex, ks, beta, qs = (0, 1, 1-2**-24)[t % 3], 1+2**-23, .5, .5
        else:
            raise ValueError(bank)
        yield (f32(ex), f32(ks), tuple(map(f32, key)), f32(beta),
               tuple(map(f32, value)), f32(qs), tuple(map(f32, query)))


class ComparisonFailure(AssertionError):
    def __init__(self, detail):
        super().__init__(json.dumps(detail, sort_keys=True))
        self.detail = detail


def equal_bits(label, left, right):
    a, b = raw(left), raw(right)
    if a != b:
        first = next((i for i in range(min(len(a), len(b))) if a[i] != b[i]),
                     min(len(a), len(b)))
        raise ComparisonFailure({'label': label, 'first_byte': first,
                                 'fp32_index': first // 4,
                                 'expected_word_le': a[first//4*4:first//4*4+4].hex(),
                                 'actual_word_le': b[first//4*4:first//4*4+4].hex(),
                                 'expected_bytes': len(a), 'actual_bytes': len(b)})
    return len(a)


def counterexample():
    # These factors are also produced by the stated scalar GDN-like delta rule.
    initial, key = f32(2**24), f32(.5)
    value1, value2 = f32(2**23 + 2), f32(-25165824)
    delta1 = f32(value1 - f32(key * initial))
    state1 = f32(initial + f32(key * delta1))
    delta2 = f32(value2 - f32(key * state1))
    sequential = f32(state1 + f32(key * delta2))
    combined = f32(initial + f32(key * f32(delta1 + delta2)))
    if raw([sequential]) == raw([combined]):
        raise AssertionError('algebraic counterexample disappeared')
    return {'initial': initial, 'ex': 1.0, 'ks': 1.0, 'key': key, 'beta': 1.0,
            'values': [value1, value2], 'deltas': [delta1, delta2],
            'first_state': state1, 'sequential': sequential,
            'combined_update_sum': combined,
            'sequential_le': raw([sequential]).hex(),
            'combined_le': raw([combined]).hex(),
            'scope': 'two-step scalar toy recurrence, not a model trajectory'}


def run_case(merge, steps, bank):
    initial = tuple(tuple(f32((i-j) / 32) for j in range(6)) for i in range(8))
    dense, log = Dense(initial), Deferred(initial, merge)
    data = list(inputs(913, bank, steps))
    trajectory = hashlib.sha256()
    comparisons = compared_bytes = 0
    lifecycle = []
    for t, x in enumerate(data):
        if t == steps // 2 + 1:
            dense.reset(initial)
            log.reset(initial)
            equal_bits('reset', flat(dense.state()), flat(log.state()))
            lifecycle.append({'step': t, 'event': 'reset-clears-pending-log'})
        y, z = dense.step(x), log.step(x)
        compared_bytes += equal_bits(f'step {t} output', y, z)
        a, b = flat(dense.state()), flat(log.state())
        compared_bytes += equal_bits(f'step {t} state', a, b)
        comparisons += 2
        trajectory.update(raw(a) + raw(y))
        if t == 4:  # nonempty log for M=2/4/8
            dcopy, lcopy = dense.clone(), log.clone()
            before = raw(a)
            branch_input = next(inputs(177, 'random', 1))
            equal_bits('clone output', dcopy.step(branch_input), lcopy.step(branch_input))
            equal_bits('clone state', flat(dcopy.state()), flat(lcopy.state()))
            if raw(flat(dense.state())) != before or raw(flat(log.state())) != before:
                raise AssertionError('clone mutated original')
            dcopy.retire()
            lcopy.retire()
            for retired in (dcopy, lcopy):
                for operation in (retired.state, retired.clone,
                                  lambda r=retired: r.step(branch_input),
                                  lambda r=retired: r.reset(initial)):
                    try:
                        operation()
                    except RuntimeError:
                        pass
                    else:
                        raise AssertionError('retired handle accepted access')
            lifecycle.append({'step': t, 'event': 'clone-independent-branch-then-retire',
                              'retired_operations_rejected': 8})
        if len(log.log) >= merge:
            raise AssertionError('unbounded log')
    pending = len(log.log)
    segments = (steps // 2 + 1, steps - (steps // 2 + 1))
    expected_merges = sum(n // merge for n in segments)
    if log.merges != expected_merges or log.log_writes != (steps - expected_merges) * 16:
        raise AssertionError('persistent write accounting')
    writes = {'dense_state_fp32_writes': steps * 8 * 6,
              'deferred_dense_base_fp32_writes': log.base_writes,
              'deferred_log_fp32_writes': log.log_writes,
              'deferred_total_persistent_fp32_writes': log.base_writes + log.log_writes,
              'merges': log.merges, 'reset_segments': segments, 'pending_entries': pending,
              'max_pending_entries': log.max_log_entries,
              'flush_at_end': False,
              'extra_base_writes_if_final_flush_required': 48 if pending else 0,
              'scope': 'theoretical persistent recurrence writes only; excludes initialization, reset, '
                       'clone, verification, output, temporary storage, replay reads/computation; '
                       'not measured Python or device traffic'}
    return {'merge': merge, 'steps': steps, 'bank': bank, 'shape': [8, 6],
            'status': 'PASS', 'state_comparisons': steps, 'output_comparisons': steps,
            'rollout_comparisons': comparisons, 'compared_bytes': compared_bytes,
            'lifecycle_exact_comparisons': 5,
            'lifecycle_compared_bytes': 24 + 4 * 192,
            'trajectory_sha256': trajectory.hexdigest(), 'lifecycle': lifecycle,
            'write_model': writes}


def experiment():
    result = {'schema': 'state-log-exact-reference-v1', 'status': 'INCOMPLETE',
              'toy_bit_exact': False, 'metal_qualified': False, 'model_qualified': False,
              'performance_qualified': False, 'cases': [], 'error': None,
              'arithmetic': 'binary32 after every scalar multiply/add/subtract, serial ascending-k '
                            'reductions, supplied ex/ks/qs, no FMA/exp/normalization'}
    active_case = None
    try:
        result['algebraic_counterexample'] = counterexample()
        for bank in ('random', 'cancellation', 'rounding_edges'):
            for steps in (64, 128):
                for merge in (1, 2, 4, 8):
                    active_case = {'merge': merge, 'steps': steps, 'bank': bank}
                    result['cases'].append(run_case(merge, steps, bank))
        result.update(status='PASS', toy_bit_exact=True)
    except Exception as error:
        result.update(status='FAIL', error={'type': type(error).__name__,
                      'message': str(error), 'active_case': active_case, 'first_difference': getattr(error, 'detail', None)})
    result['completed_cases'] = len(result['cases'])
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true', help='explicitly run CPU toy only')
    parser.add_argument('--out', type=Path, help='new JSON file; never overwritten')
    args = parser.parse_args(argv)
    if not args.run:
        print(json.dumps({'enabled': False, 'description': __doc__,
                          'required': '--run --out NEW_JSON'}, indent=2))
        return 0
    if args.out is None:
        parser.error('--run requires --out')
    with args.out.open('x') as stream:
        result = experiment()
        result['source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({k: result[k] for k in ('status', 'completed_cases', 'toy_bit_exact',
                                           'metal_qualified', 'performance_qualified')}))
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
