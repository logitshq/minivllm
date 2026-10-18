"""Small independent mechanism/negative checks, no optional dependencies."""
import contextlib
import io
import unittest
from . import toy as t


class ToyTests(unittest.TestCase):
    def test_binary32_rounding_and_algebraic_counterexample(self):
        self.assertEqual(t.f32(2**24 + 1), 2**24)
        result = t.counterexample()
        self.assertEqual(result['sequential'], 0)
        self.assertEqual(result['combined_update_sum'], 1)
        oracle = t.Dense(((result['initial'],),))
        for value in result['values']:
            output = oracle.step((1., 1., (.5,), 1., (value,), 1., (1.,)))
        self.assertEqual(output, (result['sequential'],))

    def test_dense_delta_uses_undecayed_state_dot(self):
        # Original kv=3*2=6, delta=(5-.5*6)*.25=.5, state=.5*2+3*.5=2.5.
        x = (.5, 1., (3.,), .25, (5.,), 1., (1.,))
        d, log = t.Dense(((2.,),)), t.Deferred(((2.,),), 8)
        self.assertEqual(d.step(x), (2.5,))
        self.assertEqual(log.step(x), (2.5,))
        self.assertEqual(log.state(), ((2.5,),))

    def test_preserved_scaled_key_order(self):
        state = ((t.f32(1 + 2**-23),),)
        x = (1., t.f32(1+2**-23), (t.f32(1-2**-24),), .5, (.25,), 1., (1.,))
        d, log = t.Dense(state), t.Deferred(state, 2)
        t.equal_bits('output', d.step(x), log.step(x))
        t.equal_bits('state', t.flat(d.state()), t.flat(log.state()))

    def test_pending_corruption_is_detected(self):
        initial = ((1.,),)
        d, log = t.Dense(initial), t.Deferred(initial, 8)
        x = (1., 1., (.5,), .5, (1.,), 1., (1.,))
        d.step(x); log.step(x)
        ex, ks, key, delta = log.log[0]
        log.log = ((ex, ks, key, (t.f32(delta[0] + .125),)),)
        with self.assertRaises(t.ComparisonFailure) as caught:
            t.equal_bits('corrupt', t.flat(d.state()), t.flat(log.state()))
        self.assertEqual(caught.exception.detail['fp32_index'], 0)

    def test_clone_reset_and_retire_pending_log(self):
        log = t.Deferred(((2.,),), 8)
        log.step((1., 1., (.5,), .5, (1.,), 1., (1.,)))
        original = t.raw(t.flat(log.state()))
        clone = log.clone()
        clone.reset(((7.,),))
        self.assertEqual(clone.log, ())
        self.assertEqual(t.raw(t.flat(log.state())), original)
        log.retire()
        self.assertEqual(clone.state(), ((7.,),))
        with self.assertRaises(RuntimeError):
            log.state()

    def test_bounded_log_and_write_count(self):
        x = (1., 1., (.25,), .25, (1.,), 1., (1.,))
        for merge in (1, 2, 4, 8):
            log = t.Deferred(((0.,),), merge)
            for _ in range(16):
                log.step(x)
                self.assertLess(len(log.log), merge)
            self.assertEqual(log.base_writes, 16 // merge)
            self.assertEqual(log.log_writes, (16 - 16 // merge) * 4)

    def test_default_cli_does_not_run(self):
        original = t.experiment
        try:
            t.experiment = lambda: self.fail('default CLI ran toy')
            with contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(t.main([]), 0)
            self.assertIn('"enabled": false', output.getvalue())
        finally:
            t.experiment = original

    def test_all_merge_intervals_short_exact(self):
        for merge in (1, 2, 4, 8):
            self.assertEqual(t.run_case(merge, 16, 'rounding_edges')['status'], 'PASS')


if __name__ == '__main__':
    unittest.main()
