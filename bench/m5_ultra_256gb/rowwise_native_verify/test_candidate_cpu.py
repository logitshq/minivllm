"""Prepared stdlib-only control-flow tests; not run during formal timing.

The fake arrays store small scalar fingerprints, not native tensor payloads.
They test traversal, state ownership and guards. They cannot establish MLX
slice semantics, floating-point equality, kernel dispatch or performance.
"""
from dataclasses import dataclass
from types import SimpleNamespace
import unittest

from bench.m5_ultra_256gb.rowwise_native_verify import candidate as c


class Array:
    def __init__(self, shape, dtype, values):
        self.shape, self.dtype, self.values = tuple(shape), dtype, tuple(values)

    @property
    def ndim(self): return len(self.shape)

    def __getitem__(self, index):
        if not isinstance(index, slice): raise AssertionError('only first-axis slices expected')
        values = self.values[index]
        return Array((len(values), *self.shape[1:]), self.dtype, values)

    def __add__(self, value):
        return Array(self.shape, self.dtype, [x + value for x in self.values])


class StateArray:
    def __init__(self, shape, dtype, cells):
        self.shape, self.dtype, self.cells = tuple(shape), dtype, dict(cells)

    @property
    def ndim(self): return len(self.shape)

    def copy(self): return StateArray(self.shape, self.dtype, self.cells)

    def __getitem__(self, index):
        if index != slice(None): raise AssertionError('state must retain its entire backing')
        # Explicit fake value-copy semantics: real MLX requires a device gate.
        return self.copy()


class MX:
    int32, int64, bfloat16, float32 = 'i32', 'i64', 'bf16', 'f32'

    @staticmethod
    def concatenate(arrays, axis=0):
        arrays = list(arrays)
        if axis != 0 or not arrays: raise AssertionError('unexpected concatenate')
        tail, dtype = arrays[0].shape[1:], arrays[0].dtype
        if any(a.shape[1:] != tail or a.dtype != dtype for a in arrays):
            raise AssertionError('incompatible row shapes')
        values = tuple(v for a in arrays for v in a.values)
        return Array((len(values), *tail), dtype, values)


class BatchKV:
    def __init__(self, K, V): self.K, self.V = K, V


class BatchState:
    def __init__(self, conv, S): self.conv, self.S = conv, S


@dataclass
class ForwardContext:
    mode: str
    caches: list
    positions: object = None
    lens: object = None
    max_len: int = 0
    prefill_shape: object = None


RUNTIME = SimpleNamespace(mx=MX, ForwardContext=ForwardContext,
                          BatchKV=BatchKV, BatchState=BatchState)


def clone_caches(caches):
    return [BatchKV(s.K.copy(), s.V.copy()) if type(s) is BatchKV
            else BatchState(s.conv.copy(), s.S.copy()) for s in caches]


def state_value(caches):
    result = []
    for state in caches:
        arrays = (state.K, state.V) if type(state) is BatchKV else (state.conv, state.S)
        result.extend((a.shape, a.dtype, tuple(sorted(a.cells.items()))) for a in arrays)
    return tuple(result)


def seed(used=257):
    caches = []
    for i in range(64):
        if i % 4 == 3:
            shape = (1, 4, 1536, 256)
            # Preserve both visible history and nonzero unwritten tail padding.
            caches.append(BatchKV(StateArray(shape, 'bf16', {0:i+1, 100:i+7, 1535:9000+i}),
                                  StateArray(shape, 'bf16', {0:i+2, 100:i+8, 1535:8000+i})))
        else:
            caches.append(BatchState(StateArray((1,3,10240), 'bf16', {0:i+1, 1:i+2, 2:7000+i}),
                                     StateArray((1,48,128,128), 'f32', {0:i+3, 1:i+4, 2:6000+i})))
    return ForwardContext('decode', caches, Array((1,), 'i32', [used-1]),
                          Array((1,), 'i32', [used]), used)


class Embed:
    _output_dtype = 'bf16'

    def __init__(self, model): self.model = model

    def __call__(self, tokens):
        if tokens.shape != (1,): raise AssertionError('embedding must remain one row')
        self.model.events.append(('embed', tokens.values[0], tokens.shape))
        return Array((1,5120), 'bf16', [tokens.values[0]*11 + self.model.embedding_weight])

    def as_linear(self, x): return self.model.head(x, 'tied-head')


class Layer:
    def __init__(self, model, index):
        self.model, self.index, self.is_attn, self.weight = model, index, index % 4 == 3, index+19

    def __call__(self, x, ctx, index, *, _norm_reuse_normal=False):
        if index != self.index or ctx.mode != 'decode' or x.shape != (1,5120):
            raise AssertionError('unexpected native B1 layer route')
        if not _norm_reuse_normal: raise AssertionError('original normal-context flag lost')
        pos, length = ctx.positions.values[0], ctx.lens.values[0]
        if length != pos+1 or length != ctx.max_len: raise AssertionError('invalid canonical C1 context')
        self.model.events.append(('layer', index, pos, length, ctx.max_len, x.shape))
        state = ctx.caches[index]
        value = (x.values[0] + self.weight + pos) % 1000003
        if self.is_attn:
            # Mimic native KV assignment into the supplied wrapper, then read
            # only the visible prefix. No other layer's cache is consulted.
            state.K.cells[pos], state.V.cells[pos] = value, value+1
            total = sum(v for p,v in state.K.cells.items() if p < length)
            total += sum(v for p,v in state.V.cells.items() if p < length)
            value = (value + total) % 1000003
        else:
            conv, recurrent = dict(state.conv.cells), dict(state.S.cells)
            conv[0] = (conv[0]*3 + value) % 1000003
            recurrent[0] = (recurrent[0]*5 + conv[0] + value) % 1000003
            # Native GDN replaces these two array attributes as whole values.
            state.conv = StateArray(state.conv.shape, state.conv.dtype, conv)
            state.S = StateArray(state.S.shape, state.S.dtype, recurrent)
            value = (value + conv[0] + recurrent[0]) % 1000003
        if self.model.fail_at == (index,pos): raise self.model.failure
        return Array(x.shape, x.dtype, [value])


class Model:
    def __init__(self, tied=False):
        self.cfg = SimpleNamespace(hidden_size=5120,
            quantization={'group_size':16,'bits':4,'mode':'nvfp4'}, tie_word_embeddings=tied)
        self.events, self.embedding_weight, self.head_weight = [], 23, 29
        self.fail_at, self.failure = None, RuntimeError('fake layer failed after its private state update')
        self.embed_tokens = Embed(self)
        self.layers = [Layer(self,i) for i in range(64)]

    def norm(self, x):
        if x.shape != (1,5120): raise AssertionError('norm row shape changed')
        self.events.append(('norm', x.shape))
        return Array(x.shape, x.dtype, [(x.values[0]*7) % 1000003])

    def head(self, x, name):
        if x.shape != (1,5120): raise AssertionError('head row shape changed')
        self.events.append((name, x.shape))
        return Array((1,248320), 'bf16', [x.values[0]+self.head_weight])

    def lm_head(self, x): return self.head(x, 'untied-head')


def token_major_reference(model, tokens, canonical):
    """Independent sequential caller; never calls candidate.verify."""
    state, outputs, prefixes = clone_caches(canonical.caches), [], []
    for step, token in enumerate(tokens.values):
        position = canonical.positions.values[0] + step
        ctx = ForwardContext('decode', state, Array((1,), 'i32', [position]),
                             Array((1,), 'i32', [position+1]), canonical.max_len+step)
        x = model.embed_tokens(Array((1,), tokens.dtype, [token]))
        for index, layer in enumerate(model.layers):
            x = layer(x, ctx, index, _norm_reuse_normal=True)
        x = model.norm(x)
        outputs.append(model.embed_tokens.as_linear(x) if model.cfg.tie_word_embeddings else model.lm_head(x))
        prefixes.append(clone_caches(state))
    return SimpleNamespace(logits=MX.concatenate(outputs), prefixes=prefixes)


class Tests(unittest.TestCase):
    def test_default_off_does_not_inspect_any_argument(self):
        class Poison:
            def __getattribute__(self, name): raise AssertionError('OFF inspected '+name)
        self.assertIsNone(c.verify(Poison(), Poison(), Poison(), Poison()))
        for flag in (0, 1, None, 'on'):
            with self.subTest(flag=flag), self.assertRaises(TypeError):
                c.verify(Poison(), Poison(), Poison(), Poison(), enabled=flag)

    def test_every_prefix_matches_independent_token_major_control_flow(self):
        for width in (2,4,8):
            for used in (257,1023,1024):
                with self.subTest(width=width,used=used):
                    tokens = Array((width,), 'i32', range(41,41+width))
                    canonical, model, reference_model = seed(used), Model(), Model()
                    before = state_value(canonical.caches)
                    actual = c.verify(model,tokens,canonical,RUNTIME,enabled=True)
                    expected = token_major_reference(reference_model,tokens,canonical)
                    self.assertEqual(actual.logits.shape,(width,248320))
                    self.assertEqual(actual.logits.values,expected.logits.values)
                    self.assertEqual(len(actual.prefixes),width)
                    for actual_prefix,expected_prefix in zip(actual.prefixes,expected.prefixes):
                        self.assertEqual(len(state_value(actual_prefix)),128)
                        self.assertEqual(state_value(actual_prefix),state_value(expected_prefix))
                    self.assertEqual(state_value(canonical.caches),before)
                    self.assertEqual([(e[1],e[2]) for e in model.events if e[0]=='layer'],
                                     [(i,used-1+t) for i in range(64) for t in range(width)])
                    self.assertEqual([(e[1],e[2]) for e in reference_model.events if e[0]=='layer'],
                                     [(i,used-1+t) for t in range(width) for i in range(64)])

    def test_absolute_positions_lengths_and_max_len_cross_1023_1024(self):
        model, canonical = Model(), seed(1023)
        tokens = Array((4,), 'i64', [2,7,3,9])
        c.verify(model,tokens,canonical,RUNTIME,enabled=True)
        expected = [(1022+t,1023+t,1023+t) for t in range(4)]
        for index in range(64):
            self.assertEqual([e[2:5] for e in model.events if e[0]=='layer' and e[1]==index],expected)
        self.assertEqual(canonical.positions.values,(1022,))
        self.assertEqual(canonical.lens.values,(1023,)); self.assertEqual(canonical.max_len,1023)
        self.assertEqual([e[1] for e in model.events if e[0]=='embed'],list(tokens.values))

    def test_prefixes_own_kv_handles_and_keep_full_padding_and_recurrent_states(self):
        canonical, model = seed(), Model()
        before = state_value(canonical.caches)
        actual = c.verify(model,Array((8,),'i32',range(8)),canonical,RUNTIME,enabled=True)
        saved = [state_value(prefix) for prefix in actual.prefixes]
        for index in range(64):
            wrappers = [canonical.caches[index],*(p[index] for p in actual.prefixes)]
            self.assertEqual(len({id(x) for x in wrappers}),9)
            if model.layers[index].is_attn:
                for name in ('K','V'):
                    arrays = [getattr(x,name) for x in wrappers]
                    self.assertEqual(len({id(x) for x in arrays}),9)
                    self.assertTrue(all(x.cells[1535] == arrays[0].cells[1535] for x in arrays))
                    getattr(actual.prefixes[-1][index],name).cells[1535] = -1
            else:
                for name in ('conv','S'):
                    arrays = [getattr(x,name) for x in wrappers]
                    self.assertEqual(len({id(x) for x in arrays}),9)
                    self.assertTrue(all(x.cells[2] == arrays[0].cells[2] for x in arrays))
        self.assertEqual([state_value(p) for p in actual.prefixes[:-1]],saved[:-1])
        self.assertEqual(state_value(canonical.caches),before)

    def test_shape_batch_and_capacity_rejections_happen_before_layer_or_embedding(self):
        changes = [lambda m,t,s:setattr(t,'shape',(1,2)),
                   lambda m,t,s:setattr(t,'shape',(3,)),
                   lambda m,t,s:setattr(t,'dtype','f32'),
                   lambda m,t,s:setattr(s,'positions',Array((2,),'i32',[256,256])),
                   lambda m,t,s:setattr(s,'lens',Array((2,),'i32',[257,257])),
                   lambda m,t,s:setattr(s,'max_len',256),
                   lambda m,t,s:setattr(s,'max_len',1536),
                   lambda m,t,s:setattr(s.caches[3].K,'shape',(2,4,1536,256)),
                   lambda m,t,s:setattr(s.caches[3].V,'shape',(1,4,1024,256)),
                   lambda m,t,s:setattr(s.caches[0].S,'shape',(2,48,128,128)),
                   lambda m,t,s:s.caches.pop(),
                   lambda m,t,s:setattr(m.cfg,'hidden_size',4096)]
        for index,change in enumerate(changes):
            with self.subTest(index=index):
                model,canonical,tokens = Model(),seed(),Array((2,),'i32',[1,2])
                change(model,tokens,canonical)
                before = state_value(canonical.caches)
                self.assertIsNone(c.verify(model,tokens,canonical,RUNTIME,enabled=True))
                self.assertEqual(model.events,[])
                self.assertEqual(state_value(canonical.caches),before)
        accepted = c.verify(Model(),Array((8,),'i32',range(8)),seed(1529),RUNTIME,enabled=True)
        self.assertEqual(len(accepted.prefixes),8)  # last used length is exactly 1536

    def test_unknown_context_or_cache_metadata_is_not_silently_forwarded(self):
        for target in ('context','gdn','attention'):
            with self.subTest(target=target):
                model,canonical = Model(),seed()
                obj = canonical if target=='context' else canonical.caches[0 if target=='gdn' else 3]
                obj.record = True
                self.assertIsNone(c.verify(model,Array((2,),'i32',[1,2]),canonical,RUNTIME,enabled=True))
                self.assertEqual(model.events,[])
        class ExtraContext(ForwardContext): pass
        canonical = seed()
        other = ExtraContext(**vars(canonical))
        self.assertIsNone(c.verify(Model(),Array((2,),'i32',[1,2]),other,RUNTIME,enabled=True))

    def test_admitted_exception_propagates_once_without_mutating_canonical_input(self):
        model,canonical = Model(),seed()
        before = state_value(canonical.caches)
        model.fail_at = (3,257)  # mutate the second private KV wrapper, then raise
        with self.assertRaises(RuntimeError) as caught:
            c.verify(model,Array((4,),'i32',[1,2,3,4]),canonical,RUNTIME,enabled=True)
        self.assertIs(caught.exception,model.failure)
        calls = [(e[1],e[2]) for e in model.events if e[0]=='layer']
        self.assertEqual(calls,[(i,256+t) for i in range(3) for t in range(4)]+[(3,256),(3,257)])
        self.assertEqual(state_value(canonical.caches),before)
        self.assertFalse(any(e[0] in ('norm','untied-head','tied-head') for e in model.events))

    def test_both_head_routes_keep_all_operations_one_row_and_teacher_inputs_fixed(self):
        for tied in (False,True):
            with self.subTest(tied=tied):
                model,canonical,tokens = Model(tied),seed(),Array((4,),'i32',[4,1,8,2])
                actual = c.verify(model,tokens,canonical,RUNTIME,enabled=True)
                expected = token_major_reference(Model(tied),tokens,canonical)
                self.assertEqual(actual.logits.values,expected.logits.values)
                self.assertEqual([e[1] for e in model.events if e[0]=='embed'],[4,1,8,2])
                self.assertEqual(sum(e[0]=='norm' for e in model.events),4)
                self.assertEqual(sum(e[0]==('tied-head' if tied else 'untied-head') for e in model.events),4)
                self.assertEqual(tokens.values,(4,1,8,2))

    def test_parameter_replacement_is_observed_without_parameter_or_input_assignment(self):
        model,canonical,tokens = Model(),seed(),Array((2,),'i32',[5,6])
        before = state_value(canonical.caches)
        first = c.verify(model,tokens,canonical,RUNTIME,enabled=True)
        model.layers[7].weight += 101
        expected_model = Model(); expected_model.layers[7].weight = model.layers[7].weight
        parameters = tuple(layer.weight for layer in model.layers)
        second = c.verify(model,tokens,canonical,RUNTIME,enabled=True)
        expected = token_major_reference(expected_model,tokens,canonical)
        self.assertEqual(second.logits.values,expected.logits.values)
        self.assertNotEqual(first.logits.values,second.logits.values)
        self.assertEqual(tuple(layer.weight for layer in model.layers),parameters)
        self.assertEqual(state_value(canonical.caches),before)


if __name__ == '__main__': unittest.main()
