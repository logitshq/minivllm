"""Scalar control-flow tests only; no MLX, model load or GPU qualification."""
from types import SimpleNamespace
import unittest
from bench.m5_ultra_256gb.rowwise_mlp_shared_features import candidate as c
from bench.m5_ultra_256gb.rowwise_mlp_shared_features.round import select_greedy
from bench.m5_ultra_256gb.rowwise_mlp_shared import candidate as old
from bench.m5_ultra_256gb.rowwise_mlp_shared import test_cpu as f


class FeatureRows:
    def __init__(self, rows):
        self.rows = tuple(rows)
        self.shape = (len(self.rows), 25600)
        self.dtype = 'bf16'

    def __getitem__(self, index):
        return FeatureRows(self.rows[index])


class MX(f.MX):
    feature_concats = 0

    @staticmethod
    def concatenate(arrays, axis=0):
        if axis != -1:
            return f.MX.concatenate(arrays, axis)
        MX.feature_concats += 1
        if len(arrays) != 5 or any(a.shape != arrays[0].shape for a in arrays):
            raise AssertionError('five same-width hidden feature arrays')
        return FeatureRows(zip(*(a.values for a in arrays)))


RUNTIME = SimpleNamespace(**dict(vars(f.RUNTIME), mx=MX))


class Recorder:
    def __init__(self, layer, rows):
        self.layer, self.rows = layer, rows

    def __getattr__(self, name):
        return getattr(self.layer, name)

    def __call__(self, *args, **kwargs):
        result = self.layer(*args, **kwargs)
        # Independent token-major oracle: after the complete DecoderLayer.
        self.rows.append(result.values[0])
        return result


class Tests(unittest.TestCase):
    def run_candidate(self, width, used=1023, capture=c.CAPTURE):
        model = f.Model()
        ctx = f.f.seed(used)
        result = c.verify(model, f.tokens(width), ctx, RUNTIME, enabled=True,
                          mlp_input_project=model.projector, capture=capture)
        return result, ctx, model

    def test_default_off_does_not_inspect_capture_or_inputs(self):
        class Poison:
            def __getattribute__(self, name):
                raise AssertionError(name)
        self.assertIsNone(c.verify(Poison(), Poison(), Poison(), Poison(),
                                  capture=Poison()))

    def test_capture_location_order_prefixes_and_logits_match_native(self):
        for width in (2, 4, 8):
            for used in (257, 1023, 1024):
                with self.subTest(width=width, used=used):
                    actual, ctx, _ = self.run_candidate(width, used)
                    reference = f.Model()
                    captures = {i: [] for i in c.CAPTURE}
                    for i in captures:
                        reference.layers[i] = Recorder(reference.layers[i], captures[i])
                    expected = f.f.token_major_reference(reference, f.tokens(width), ctx)
                    self.assertEqual(actual.features.rows,
                                     tuple(zip(*(captures[i] for i in c.CAPTURE))))
                    self.assertEqual(actual.logits.values, expected.logits.values)
                    self.assertEqual([f.f.state_value(p) for p in actual.prefixes],
                                     [f.f.state_value(p) for p in expected.prefixes])
                    self.assertEqual(f.f.state_value(ctx.caches),
                                     f.f.state_value(f.f.seed(used).caches))
                    for count in range(1, width+1):
                        self.assertEqual(actual.features[:count].rows,
                                         actual.features.rows[:count])

    def test_capture_off_matches_frozen_path_without_feature_concat(self):
        MX.feature_concats = 0
        actual, ctx, model = self.run_candidate(4, capture=())
        prior = f.Model()
        expected = old.verify(prior, f.tokens(4), ctx, RUNTIME, enabled=True,
                              mlp_input_project=prior.projector)
        self.assertIsNone(actual.features)
        self.assertEqual(MX.feature_concats, 0)
        self.assertEqual(actual.logits.values, expected.logits.values)
        self.assertEqual(model.events, prior.events)

    def test_shared_head_keeps_identical_captured_features(self):
        baseline, _, _ = self.run_candidate(4)
        model = f.Model()
        model.lm_head = f.Linear(model, 0, 'head', 5120, 248320)
        calls = []
        def head(x, w, s):
            self.assertIs(w, model.lm_head.weight)
            self.assertIs(s, model.lm_head.scales)
            calls.append(x.shape)
            return f.Array((4, 248320), 'bf16',
                           [(v*w.values[0]+s.values[0]) % f.P for v in x.values])
        actual = c.verify(model, f.tokens(4), f.f.seed(1023), RUNTIME,
                          enabled=True, head_project=head,
                          mlp_input_project=model.projector, capture=c.CAPTURE)
        self.assertEqual(actual.features.rows, baseline.features.rows)
        self.assertEqual(calls, [(4, 5120)])
        reference = f.Model()
        reference.lm_head = f.Linear(reference, 0, 'head', 5120, 248320)
        expected = f.f.token_major_reference(reference, f.tokens(4), f.f.seed(1023))
        self.assertEqual(actual.logits.values, expected.logits.values)

    def test_only_fixed_capture_tuple_admitted_before_work(self):
        for capture in ([5, 19, 33, 47, 61], (61, 47, 33, 19, 5),
                        (5, 5, 19, 33, 61), (5,), None):
            model = f.Model()
            with self.assertRaises((TypeError, ValueError)):
                c.verify(model, f.tokens(4), f.f.seed(), RUNTIME, enabled=True,
                         mlp_input_project=model.projector, capture=capture)
            self.assertEqual(model.events, [])

    def test_capture_requires_shared_path(self):
        model = f.Model()
        with self.assertRaises(ValueError):
            c.verify(model, f.tokens(4), f.f.seed(), RUNTIME,
                     enabled=True, capture=c.CAPTURE)
        self.assertEqual(model.events, [])

    def test_every_first_rejection_and_all_accepted(self):
        for width in (2, 4, 8):
            output, ctx, _ = self.run_candidate(width)
            before = [f.f.state_value(p) for p in output.prefixes]
            drafts = list(range(100, 100+width-1))
            for accepted in range(width):
                posterior = drafts + [999]
                posterior[accepted] = 999
                chosen = select_greedy(drafts, posterior, output)
                self.assertEqual(chosen.accepted_drafts, accepted)
                self.assertEqual(chosen.produced, tuple(drafts[:accepted])+ (999,))
                self.assertEqual(chosen.computed_count, accepted+1)
                self.assertEqual(chosen.next_anchor, 999)
                self.assertIs(chosen.layers, output.prefixes[accepted])
                self.assertEqual(chosen.features.rows, output.features.rows[:accepted+1])
            self.assertEqual([f.f.state_value(p) for p in output.prefixes], before)
            self.assertEqual(f.f.state_value(ctx.caches), f.f.state_value(f.f.seed(1023).caches))

    def test_invalid_round_contract_rejected(self):
        output, _, _ = self.run_candidate(4)
        for drafts, post in (([1, 2], [1, 2, 3]), ([1, 2, 3], [1, 2]),
                             ([1, True, 3], [1, 2, 3, 4]), ([1, -1, 3], [1, 2, 3, 4])):
            with self.assertRaises(ValueError):
                select_greedy(drafts, post, output)
        with self.assertRaises(ValueError):
            select_greedy([1, 2, 3], [1, 2, 3, 4],
                          SimpleNamespace(prefixes=output.prefixes, features=None))


if __name__ == '__main__':
    unittest.main()
