"""CPU scalar control-flow tests; not numerical/device qualification."""
from types import SimpleNamespace
import unittest

from bench.m5_ultra_256gb.rowwise_native_verify import test_candidate_cpu as nf
from bench.m5_ultra_256gb.rowwise_mlp_shared_features import test_cpu as ef
from bench.m5_ultra_256gb.rowwise_mlp_shared import test_cpu as sf
from bench.m5_ultra_256gb.rowwise_mlp_shared_features.round import select_greedy
import implementation as impl


class PaddedMX(nf.MX):
    """Row concat via the native fake; hidden concat preserves row fingerprints."""
    def __init__(self):
        self.calls = []

    def concatenate(self, arrays, axis=0):
        arrays = list(arrays)
        self.calls.append((axis, arrays))
        if axis != 0:
            raise AssertionError(f'unexpected concatenate axis {axis}')
        hidden = len(arrays) == 5 and all(
            len(a.shape) == 2 and a.shape[0] == len(arrays[0].values)
            and a.shape[1] == 5120 for a in arrays)
        token = len(arrays) > 1 and all(
            len(a.shape) == 2 and a.shape[1] == 25600 for a in arrays)
        if not hidden and not token:
            return nf.MX.concatenate(arrays, axis)
        if token:
            return nf.Array((len(arrays), 25600), 'bf16',
                            [tuple(a.values) for a in arrays])
        T = len(arrays[0].values)
        width = sum(a.shape[1] for a in arrays)
        return nf.Array((T, width), 'bf16', [
            tuple(v for a in arrays for v in a.values) for _ in range(T)
        ])


class Recorder:
    def __init__(self, layer, rows):
        self.layer, self.rows = layer, rows

    def __call__(self, *args, **kwargs):
        result = self.layer(*args, **kwargs)
        self.rows.append(result)
        return result


REFERENCE_RUNTIME = SimpleNamespace(**{**vars(nf.RUNTIME), 'mx': PaddedMX()})
CANDIDATE_RUNTIME = SimpleNamespace(**{**vars(sf.RUNTIME), 'mx': ef.MX})


class HeadDispatch:
    """Valid untied NVFP4 metadata; records and emulates only head reuse."""
    def __init__(self, model, project):
        self.model, self.project = model, project
        self.head_calls = 0
        self.weight = nf.Array((248320, 640), 'u32', [29])
        self.scales = nf.Array((248320, 320), 'u8', [1])
        self.q = model.cfg.quantization
        self.in_features, self.out_features = 5120, 248320
        self.biases = None

    def __call__(self, x, weight, scales):
        if weight is self.weight and scales is self.scales:
            self.head_calls += 1
            return sf.Array((1, 248320), 'bf16', [
                (v * 29 + 1) % sf.P for v in x.values])
        return self.project(x, weight, scales)


def untied_model(project=None):
    model = sf.Model()
    model.cfg.tie_word_embeddings = False
    head = HeadDispatch(model, model.projector if project is None else project)
    model.lm_head = head
    return model


class Tests(unittest.TestCase):
    def check_reference(self, width, used, tied, token_ids=range(41, 49)):
        tokens = nf.Array((width,), 'i32', list(token_ids)[:width])
        canonical = nf.seed(used)
        before = (nf.state_value(canonical.caches), canonical.positions.values,
                  canonical.lens.values, canonical.max_len)
        reference_model = nf.Model(tied=tied)
        captures = {i: [] for i in impl.CAPTURE}
        for i in captures:
            reference_model.layers[i] = Recorder(reference_model.layers[i], captures[i])
        expected = nf.token_major_reference(reference_model, tokens, canonical)
        recorded = tuple(list(captures[i]) for i in impl.CAPTURE)
        runtime = SimpleNamespace(**{**vars(nf.RUNTIME), 'mx': PaddedMX()})
        actual = impl.reference_with_features(
            nf.Model(tied=tied), tokens, canonical, runtime)

        self.assertEqual(set(vars(actual)), {'logits', 'features', 'prefixes'})
        self.assertEqual(actual.logits.shape, (width, 248320))
        self.assertEqual(actual.logits.values, expected.logits.values)
        self.assertEqual(actual.features.shape, (width, 25600))
        self.assertEqual(actual.features.dtype, 'bf16')
        expected_features = tuple(
            tuple(recorded[i][t].values[0] for i in range(5))
            for t in range(width))
        self.assertEqual(actual.features.values, expected_features)

        # Five hidden-axis concatenations (one per capture layer, T rows each)
        # followed by one token-axis concatenation of five (T,5120) arrays.
        hidden = [call for call in runtime.mx.calls if len(call[1]) == width]
        token = [call for call in runtime.mx.calls if len(call[1]) == 5]
        self.assertEqual(len(runtime.mx.calls), 6)
        self.assertTrue(all(axis == 0 for axis, _ in runtime.mx.calls))
        self.assertEqual(len(hidden), 5)
        for index, (_, arrays) in enumerate(hidden):
            self.assertEqual([tuple(a.values) for a in arrays],
                             [tuple(r.values) for r in recorded[index]])
        self.assertEqual(len(token), 1)
        self.assertEqual(token[0][1][0].shape, (width, 5120))
        self.assertEqual([tuple(a.values) for a in token[0][1]],
                         [tuple(tuple(recorded[i][t].values) for i in range(5))
                          for t in range(width)])

        self.assertEqual(len(actual.prefixes), width)
        self.assertEqual([nf.state_value(p) for p in actual.prefixes],
                         [nf.state_value(p) for p in expected.prefixes])
        self.assertEqual((nf.state_value(canonical.caches), canonical.positions.values,
                          canonical.lens.values, canonical.max_len), before)
        self.assertEqual(tokens.values, tuple(list(token_ids)[:width]))
        return actual, canonical, recorded

    def test_reference_all_layers_prefixes_logits_and_positions(self):
        cases = ((1, 257), (1, 1023), (4, 1023), (4, 1024),
                 (4, 1025), (8, 1023), (8, 1024))
        for width, used in cases:
            for tied in (False, True):
                with self.subTest(width=width, used=used, tied=tied):
                    self.check_reference(width, used, tied)

    def test_reference_sample_matches_original_token_major_every_layer(self):
        # Full 64-layer event order and per-layer output for one continuation case.
        width, used = 4, 1024
        tokens = nf.Array((width,), 'i64', [2, 7, 3, 9])
        canonical = nf.seed(used)
        reference_model = nf.Model(tied=False)
        captures = {i: [] for i in impl.CAPTURE}
        for i in captures:
            reference_model.layers[i] = Recorder(reference_model.layers[i], captures[i])
        expected = nf.token_major_reference(reference_model, tokens, canonical)
        model = nf.Model(tied=False)
        actual = impl.reference_with_features(model, tokens, canonical, REFERENCE_RUNTIME)
        self.assertEqual(expected.logits.values, actual.logits.values)
        self.assertEqual([nf.state_value(p) for p in expected.prefixes],
                         [nf.state_value(p) for p in actual.prefixes])
        native = [e for e in reference_model.events if e[0] == 'layer']
        self.assertEqual([e for e in model.events if e[0] == 'layer'], native)
        self.assertEqual([(e[1], e[2], e[3], e[4]) for e in native],
                         [(i, used - 1 + t, used + t, used + t)
                          for t in range(width) for i in range(64)])
        expected_features = tuple(
            tuple(captures[i][t].values[0] for i in impl.CAPTURE)
            for t in range(width))
        self.assertEqual(actual.features.shape,
                         (width, len(impl.CAPTURE) * 5120))
        self.assertEqual(actual.features.values, expected_features)
        self.assertEqual(actual.features.shape, (width, 25600))
        self.assertEqual(model.events[-1][0], 'untied-head')

    def test_reference_owns_every_prefix_and_full_padding(self):
        for width in (1, 4, 8):
            with self.subTest(width=width):
                actual, canonical, _ = self.check_reference(width, 1023, False)
                wrappers = [canonical.caches, *actual.prefixes]
                for index in range(64):
                    handles = [w[index] for w in wrappers]
                    self.assertEqual(len({id(h) for h in handles}), width + 1)
                    names = ('K', 'V') if index % 4 == 3 else ('conv', 'S')
                    for name in names:
                        arrays = [getattr(h, name) for h in handles]
                        self.assertEqual(len({id(a) for a in arrays}), width + 1)
                actual.prefixes[-1][63].K.cells[1535] = -1
                self.assertNotEqual(canonical.caches[63].K.cells[1535], -1)
                self.assertNotEqual(actual.prefixes[-2][63].K.cells[1535], -1)
                actual.prefixes[0][0].conv.cells[2] = -7
                self.assertNotEqual(canonical.caches[0].conv.cells[2], -7)

    def test_reference_rejects_unsupported_token_rows_before_work(self):
        for shape in ((3,), (1, 2), (0,)):
            model, canonical = nf.Model(), nf.seed()
            before = nf.state_value(canonical.caches)
            with self.assertRaises(ValueError):
                impl.reference_with_features(
                    model, nf.Array(shape, 'i32', [1] * (shape[0] if shape else 1)),
                    canonical, nf.RUNTIME)
            self.assertEqual(model.events, [])
            self.assertEqual(nf.state_value(canonical.caches), before)

    def test_candidate_wrapper_forwards_both_projectors_and_three_fields(self):
        for width in (2, 4, 8):
            with self.subTest(width=width):
                roles, calls = [], []

                def project(x, weight, scales):
                    roles.append('project')
                    calls.append(x.shape)
                    return model.projector(x, weight, scales)

                model = untied_model(project)
                project = model.lm_head

                ctx = nf.seed(1024)
                before = nf.state_value(ctx.caches)
                actual = impl.candidate_with_features(
                    model, sf.tokens(width), ctx, CANDIDATE_RUNTIME, project)
                self.assertEqual(set(vars(actual)), {'logits', 'features', 'prefixes'})
                self.assertEqual(len(calls), 128)
                self.assertEqual(roles, ['project'] * 128)
                self.assertEqual(model.lm_head.head_calls, 1)
                self.assertTrue(all(shape in ((width, 5120), (2, 5120), (4, 5120))
                                    for shape in calls))
                self.assertEqual(actual.features.shape, (width, 25600))
                self.assertEqual(len(actual.features.rows), width)
                self.assertEqual(len(actual.prefixes), width)
                self.assertIn(('native', 0, 'head', (1, 5120)), model.events)
                self.assertNotIn('tied-head', [event[0] for event in model.events])
                expected = nf.token_major_reference(
                    untied_model(), sf.tokens(width), ctx)
                self.assertEqual(actual.logits.values, expected.logits.values)
                self.assertEqual([nf.state_value(p) for p in actual.prefixes],
                                 [nf.state_value(p) for p in expected.prefixes])
                self.assertEqual(nf.state_value(ctx.caches), before)

    def test_candidate_features_match_independent_layer_captures(self):
        for width in (2, 4, 8):
            with self.subTest(width=width):
                record = untied_model()
                captures = {i: [] for i in impl.CAPTURE}
                for i in captures:
                    record.layers[i] = Recorder(record.layers[i], captures[i])
                ctx = nf.seed(1023)
                expected = nf.token_major_reference(record, sf.tokens(width), ctx)
                model = untied_model()
                actual = impl.candidate_with_features(
                    model, sf.tokens(width), ctx, CANDIDATE_RUNTIME, model.lm_head)
                self.assertEqual(actual.features.rows,
                                 tuple(zip(*(tuple(captures[i]) for i in impl.CAPTURE))))
                self.assertEqual(actual.logits.values, expected.logits.values)
                self.assertEqual([nf.state_value(p) for p in actual.prefixes],
                                 [nf.state_value(p) for p in expected.prefixes])

    def test_greedy_selection_uses_exact_accepted_prefix_and_features(self):
        for width in (2, 4, 8):
            with self.subTest(width=width):
                model = untied_model()
                ctx = nf.seed(1023)
                output = impl.candidate_with_features(
                    model, sf.tokens(width), ctx, CANDIDATE_RUNTIME, model.lm_head)
                prefixes = [nf.state_value(p) for p in output.prefixes]
                features = output.features.rows
                self.assertEqual(len(features), width)
                drafts = list(range(100, 99 + width))
                for accepted in (0, (width - 2) // 2, width - 1):
                    with self.subTest(accepted=accepted):
                        posterior = drafts + [201]
                        posterior[accepted] = 201
                        chosen = select_greedy(drafts, posterior, output)
                        self.assertEqual(chosen.accepted_drafts, accepted)
                        self.assertIs(chosen.layers, output.prefixes[accepted])
                        self.assertEqual(chosen.features.shape, (accepted + 1, 25600))
                        self.assertEqual(chosen.features.rows, features[:accepted + 1])
                        self.assertEqual(chosen.computed_count, accepted + 1)
                        self.assertEqual(chosen.next_anchor, 201)
                        self.assertEqual(chosen.produced,
                                         tuple(drafts[:accepted]) + (201,))
                self.assertEqual([nf.state_value(p) for p in output.prefixes], prefixes)
                self.assertEqual(nf.state_value(ctx.caches),
                                 nf.state_value(nf.seed(1023).caches))


if __name__ == '__main__':
    unittest.main()
