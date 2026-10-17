"""Screen contract tests using metadata only, with all real runtimes blocked."""
import builtins
import importlib.util
import math
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch


HERE = Path(__file__).resolve().parent


class ArrayMetadata:
    def __init__(self, shape, strides=None, offset=0):
        self.shape = shape
        self.strides = strides or ((1,) if len(shape) == 1 else (shape[1], 1))
        self.offset = offset

    def __getitem__(self, selection):
        selections = selection if isinstance(selection, tuple) else (selection,)
        selections += (slice(None),) * (len(self.shape) - len(selections))
        shape, strides, offset = [], [], self.offset
        for selected, length, stride in zip(selections, self.shape, self.strides):
            lo, hi, step = selected.indices(length)
            shape.append(len(range(lo, hi, step)))
            strides.append(stride * step)
            offset += lo * stride
        return ArrayMetadata(tuple(shape), tuple(strides), offset)

    def reshape(self, *shape):
        assert math.prod(shape) == math.prod(self.shape)
        assert len(self.shape) == 1 and self.strides == (1,)
        return ArrayMetadata(shape, offset=self.offset)


class MetadataBackend:
    def __init__(self):
        self.calls = []

    def concatenate(self, pieces):
        self.calls.append([p.shape for p in pieces])
        return ArrayMetadata((sum(p.shape[0] for p in pieces), pieces[0].shape[1]))


class ScreenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        original_import = builtins.__import__
        def no_runtime(name, *args, **kwargs):
            if name.split('.')[0] in {"mlx", "numpy", "torch"}:
                raise AssertionError("forbidden runtime import: " + name)
            return original_import(name, *args, **kwargs)
        spec = importlib.util.spec_from_file_location("e019_screen_cpu", HERE / "screen.py")
        cls.screen = importlib.util.module_from_spec(spec)
        with patch.object(builtins, "__import__", no_runtime):
            spec.loader.exec_module(cls.screen)

    def test_fixed_matrix_and_group_counts(self):
        cases = self.screen.build_cases()
        self.assertEqual(len(cases), 150)
        self.assertEqual(len({c["id"] for c in cases}), 150)
        self.assertEqual(sum(c["timed"] for c in cases), 120)
        self.assertEqual(sum(c["gate_group"] == "long_tail" for c in cases), 48)
        self.assertEqual(sum(c["gate_group"] == "other" for c in cases), 72)
        for n in self.screen.LENGTHS:
            selected = [c for c in cases if c["N"] == n]
            self.assertEqual(len(selected), 30 if n == 19 else 40)
            self.assertEqual({c["role"] for c in selected}, {"tail1", "tail3", "tail7", "reset", "caught_up"})

    def test_layout_shapes_strides_and_existing_compression_condition(self):
        seen = set()
        for case in self.screen.build_cases():
            n, layout = case["N"], case["layout"]
            if (n, layout) in seen:
                continue
            seen.add((n, layout))
            mx = MetadataBackend()
            history = self.screen.build_history(mx, ArrayMetadata(((2 * n + 4) * self.screen.WIDTH,)), n, layout)
            self.assertEqual(sum(p.shape[0] for p in history), n)
            self.assertTrue(all(p.shape[1] == self.screen.WIDTH for p in history))
            if layout == "old64_then_append":
                self.assertEqual(len(history), 65)
                self.assertEqual(mx.calls, [])
            elif layout == "old65_compress_then_append":
                self.assertEqual(len(history), 2)
                self.assertEqual(len(mx.calls), 1)
                self.assertEqual(len(mx.calls[0]), 65)
            else:
                self.assertEqual(mx.calls, [])
            if layout == "row_stride_cross":
                self.assertTrue(all(p.strides == (2 * self.screen.WIDTH, 1) for p in history))
            elif layout == "column_stride_cross":
                self.assertTrue(all(p.strides == (2 * self.screen.WIDTH, 2) for p in history))
            elif layout == "offset_single":
                self.assertEqual(history[0].offset, self.screen.WIDTH + 1)

    def test_concat_row_contract(self):
        self.assertEqual(self.screen.expected_concat_rows([19, 2, 1], 0), [])
        self.assertEqual(self.screen.expected_concat_rows([19, 2, 1], 1), [])
        self.assertEqual(self.screen.expected_concat_rows([19, 2, 1], 3), [[2, 1]])
        self.assertEqual(self.screen.expected_concat_rows([19, 2, 1], 7), [[4, 2, 1]])
        self.assertEqual(self.screen.expected_concat_rows([19, 2, 1], 22), [[19, 2, 1]])

    def test_gate_keeps_single_and_reset_regressions(self):
        cases = self.screen.build_cases()
        timing = [{"id": c["id"], "speedup": 1.5 if c["gate_group"] == "long_tail" else 1.0}
                  for c in cases if c["timed"]]
        self.assertTrue(self.screen.evaluate_gate(cases, timing)["advance"])
        for suffix in ("single-tail1", "single-reset"):
            changed = [dict(row) for row in timing]
            bad = next(row for row in changed if row["id"] == "N19-" + suffix)
            bad["speedup"] = 0.96
            gate = self.screen.evaluate_gate(cases, changed)
            self.assertFalse(gate["advance"])
            self.assertEqual(gate["other_below_097"], [bad])
        with self.assertRaises(AssertionError):
            self.screen.evaluate_gate(cases, timing[:-1])

    def test_gate_rejects_slow_long_group(self):
        cases = self.screen.build_cases()
        timing = [{"id": c["id"], "speedup": 1.24 if c["gate_group"] == "long_tail" else 1.0}
                  for c in cases if c["timed"]]
        self.assertFalse(self.screen.evaluate_gate(cases, timing)["advance"])

    def test_help_does_not_import_runtime_or_create_output(self):
        code = """import builtins, runpy, sys
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name.split('.')[0] in {'mlx', 'numpy', 'torch'}:
        raise RuntimeError('forbidden import: ' + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
sys.argv = [sys.argv[1], '--help']
runpy.run_path(sys.argv[0], run_name='__main__')
"""
        result = subprocess.run([sys.executable, "-c", code, str(HERE / "screen.py")], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--out", result.stdout)


if __name__ == "__main__":
    unittest.main()
