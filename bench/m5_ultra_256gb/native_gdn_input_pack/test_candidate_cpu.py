"""Fake metadata/operator tests only; no MLX or numerical qualification.

The fake stores symbolic output-row ranges, not model tensors. It checks native
operator routing, row order, preparation ownership and failure boundaries.
"""
import unittest

from bench.m5_ultra_256gb.native_gdn_input_pack import candidate as c


class Array:
    def __init__(self, shape, dtype, rows=()):
        self.shape, self.dtype, self.rows = shape, dtype, tuple(rows)


class Linear:
    def __init__(self, label, n):
        self.in_features, self.out_features = 5120, n
        self.q = dict(c.QUANT)
        self.weight = Array((n, 640), "u32", ((n, label + "-w"),))
        self.scales = Array((n, 320), "u8", ((n, label + "-s"),))

    def __call__(self, x):
        raise AssertionError("candidate must not invoke or replace original Linears")


class FakeMX:
    bfloat16, uint32, uint8 = "bf16", "u32", "u8"

    def __init__(self):
        self.events = []
        self.output_dtype = self.bfloat16
        self.eval_error = self.qmm_error = None

    def concatenate(self, arrays, *, axis):
        self.events.append(("concat", tuple(arrays), axis))
        if axis != 0 or len({a.dtype for a in arrays}) != 1:
            raise AssertionError("unexpected concat")
        return Array((sum(a.shape[0] for a in arrays), arrays[0].shape[1]),
                     arrays[0].dtype, tuple(r for a in arrays for r in a.rows))

    def eval(self, *arrays):
        self.events.append(("eval", arrays))
        if self.eval_error:
            raise self.eval_error

    def quantized_matmul(self, *args, **kwargs):
        self.events.append(("qmm", args, kwargs))
        if self.qmm_error:
            raise self.qmm_error
        x, weight, scales, biases = args
        return Array((x.shape[0], weight.shape[0]), self.output_dtype, weight.rows)

    def split(self, array, boundaries, *, axis):
        self.events.append(("split", boundaries, axis))
        result = []
        for lo, hi in zip([0] + boundaries, boundaries + [array.shape[-1]]):
            cursor, rows = 0, []
            for count, label in array.rows:
                overlap = max(0, min(hi, cursor + count) - max(lo, cursor))
                if overlap:
                    rows.append((overlap, label))
                cursor += count
            result.append(Array((array.shape[0], hi - lo), array.dtype, rows))
        return result


class Bomb:
    def __getattribute__(self, name):
        raise AssertionError("OFF inspected an operand")


class Tests(unittest.TestCase):
    def setUp(self):
        self.mx = FakeMX()
        self.ops = tuple(Linear(label, n) for label, n in zip(("qkv", "z", "b", "a"), c.WIDTHS))
        self.x = Array((1, 5120), "bf16")

    def prepare(self):
        return c.prepare(self.mx, self.ops, enabled=True, linear_type=Linear)

    def test_default_off_does_not_inspect_operands(self):
        self.assertIsNone(c.prepare(Bomb(), Bomb(), linear_type=Bomb()))
        self.assertIsNone(c.prepare(Bomb(), Bomb(), enabled=False))
        with self.assertRaises(TypeError):
            c.prepare(Bomb(), Bomb(), enabled=1)

    def test_idle_materialization_preserves_original_arrays_and_row_order(self):
        before = tuple((op.weight, op.scales, dict(vars(op))) for op in self.ops)
        packed = self.prepare()
        self.assertEqual([e[0] for e in self.mx.events], ["concat", "concat", "eval"])
        self.assertEqual(packed.weight.rows, tuple((n, label + "-w") for label, n in
                                                 zip(("qkv", "z", "b", "a"), c.WIDTHS)))
        self.assertEqual(packed.scales.rows, tuple((n, label + "-s") for label, n in
                                                 zip(("qkv", "z", "b", "a"), c.WIDTHS)))
        for op, (w, s, attrs) in zip(self.ops, before):
            self.assertIs(op.weight, w)
            self.assertIs(op.scales, s)
            self.assertEqual(vars(op), attrs)
        self.assertEqual(self.mx.events[-1][1], (packed.weight, packed.scales))

    def test_one_native_qmm_none_bias_exact_quant_and_splits_without_eval(self):
        packed = self.prepare()
        self.mx.events.clear()
        outputs = packed.project(self.x)
        self.assertEqual([e[0] for e in self.mx.events], ["qmm", "split"])
        call = self.mx.events[0]
        self.assertEqual(call[1], (self.x, packed.weight, packed.scales, None))
        self.assertEqual(call[2], dict(transpose=True, group_size=16, bits=4, mode="nvfp4"))
        self.assertEqual(self.mx.events[1], ("split", [10240, 16384, 16432], -1))
        self.assertEqual([a.shape for a in outputs], [(1, n) for n in c.WIDTHS])
        self.assertEqual([a.rows for a in outputs], [((n, label + "-w"),) for label, n in
                                                   zip(("qkv", "z", "b", "a"), c.WIDTHS)])

    def test_unsupported_batch_shape_dtype_requests_fallback_without_operations(self):
        packed = self.prepare()
        self.mx.events.clear()
        for x in (Array((2, 5120), "bf16"), Array((8, 5120), "bf16"),
                  Array((5120,), "bf16"), Array((1, 5119), "bf16"), Array((1, 5120), "fp32")):
            with self.subTest(shape=x.shape, dtype=x.dtype):
                self.assertIsNone(packed.project(x))
        self.assertEqual(self.mx.events, [])

    def test_all_operands_validated_before_preparing_any_array(self):
        cases = (("weight", Array((48, 640), "u8")), ("scales", Array((48, 320), "bf16")),
                 ("biases", Array((48, 320), "bf16")), ("in_features", 5120.0),
                 ("out_features", 48.0), ("q", dict(group_size=64, bits=4, mode="affine")))
        for field, value in cases:
            with self.subTest(field=field):
                ops = tuple(Linear(str(i), n) for i, n in enumerate(c.WIDTHS))
                setattr(ops[-1], field, value)
                with self.assertRaises(ValueError):
                    c.prepare(self.mx, ops, enabled=True, linear_type=Linear)
                self.assertEqual(self.mx.events, [])

    def test_wrong_order_or_linear_subclass_rejected(self):
        class SubLinear(Linear):
            pass
        for ops in (self.ops[::-1], self.ops[:3],
                    self.ops[:3] + (SubLinear("a", 48),)):
            with self.assertRaises(ValueError):
                c.prepare(self.mx, ops, enabled=True, linear_type=Linear)
        self.assertEqual(self.mx.events, [])

    def test_reprepare_reads_current_replacements_without_mutating_sources(self):
        first = self.prepare()
        replacement = Array((10240, 640), "u32", ((10240, "changed-w"),))
        self.ops[0].weight = replacement
        self.ops[0].scales = Array((10240, 320), "u8", ((10240, "changed-s"),))
        second = self.prepare()
        self.assertEqual(first.weight.rows[0], (10240, "qkv-w"))
        self.assertEqual(second.weight.rows[0], (10240, "changed-w"))
        self.assertEqual(second.scales.rows[0], (10240, "changed-s"))
        self.assertIs(self.ops[0].weight, replacement)
        # This proves explicit rebuild reads current operands, not hot freshness
        # detection or MLX __setitem__/compiled-capture correctness.

    def test_prepare_failure_leaves_original_attributes_untouched(self):
        before = [dict(vars(op)) for op in self.ops]
        self.mx.eval_error = RuntimeError("fake allocation/materialization failure")
        with self.assertRaisesRegex(RuntimeError, "materialization failure"):
            self.prepare()
        self.assertEqual([dict(vars(op)) for op in self.ops], before)

    def test_qmm_failure_propagates_without_fallback(self):
        packed = self.prepare()
        self.mx.events.clear()
        self.mx.qmm_error = RuntimeError("fake kernel failure")
        with self.assertRaisesRegex(RuntimeError, "kernel failure"):
            packed.project(self.x)
        self.assertEqual([e[0] for e in self.mx.events], ["qmm"])

    def test_output_dtype_cannot_be_silently_cast_into_passing_contract(self):
        packed = self.prepare()
        self.mx.events.clear()
        self.mx.output_dtype = "fp32"
        with self.assertRaisesRegex(ValueError, "no output cast"):
            packed.project(self.x)
        self.assertEqual([e[0] for e in self.mx.events], ["qmm"])


if __name__ == "__main__":
    unittest.main()
