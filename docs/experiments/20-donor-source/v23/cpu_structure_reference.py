"""CPU metadata/source checks only; never import MLX or evaluate tensor values."""
import ast
import hashlib
import importlib.util
import inspect
from pathlib import Path
import re
import sys
import unittest

ROOT = Path(__file__).resolve().parent
P = ROOT.parent
V18 = P / "candidate-v18-gdn-postconv-build2-root/tutorial_code/miniengine/kernels/gdn_postconv_metal.py"
V18_SHA = "aa12ff8500194337c0b58f9621e01177272d382d5361f1071531db15d267c64f"
HEADER_SHA = "963a9408cc963262146461b7dc23bf00ff7eb19bd95c19f661b850cd9ff1e026"
BF16 = object()


class Metadata:
    def __init__(self, shape, dtype=BF16):
        self.shape = shape
        self.dtype = dtype


class CPUChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.before_mlx = {n for n in sys.modules if n == "mlx" or n.startswith("mlx.")}
        spec = importlib.util.spec_from_file_location("v23_cpu_source_only", ROOT / "conv_silu_raw_tail.py")
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def good(self):
        return Metadata((1, 10240)), Metadata((1, 3, 10240)), Metadata((10240, 4))

    def test_import_has_no_mlx_or_kernel_creation(self):
        self.assertEqual(self.before_mlx, {n for n in sys.modules if n == "mlx" or n.startswith("mlx.")})
        self.assertEqual(self.module._kernel.cache_info().currsize, 0)
        tree = ast.parse((ROOT / "conv_silu_raw_tail.py").read_text())
        names = [n.name for n in ast.walk(tree) if isinstance(n, ast.alias)]
        self.assertFalse(any(n == "mlx" or n.startswith("mlx.") for n in names))

    def test_sum_y_is_required_keyword_without_default(self):
        signature = inspect.signature(self.module.conv_silu_raw_tail)
        param = signature.parameters["sum_y"]
        self.assertEqual(param.kind, inspect.Parameter.KEYWORD_ONLY)
        self.assertIs(param.default, inspect.Parameter.empty)
        with self.assertRaises(TypeError):
            self.module.validate_metadata(*self.good(), bfloat16=BF16)

    def test_valid_metadata_for_all_predeclared_y(self):
        for y in (1, 2, 3, 4):
            self.module.validate_metadata(*self.good(), sum_y=y, bfloat16=BF16)

    def test_bad_y_rejected(self):
        for y in (True, False, 0, -1, 5, 1.0, "4", None):
            with self.subTest(y=y), self.assertRaises(ValueError):
                self.module.validate_metadata(*self.good(), sum_y=y, bfloat16=BF16)

    def test_bad_shape_rejected_each_input(self):
        for index in range(3):
            values = list(self.good())
            values[index] = Metadata((10240,))
            with self.subTest(index=index), self.assertRaises(ValueError):
                self.module.validate_metadata(*values, sum_y=4, bfloat16=BF16)

    def test_wrong_dtype_and_missing_metadata_rejected(self):
        for index in range(3):
            values = list(self.good())
            values[index].dtype = "float32"
            with self.subTest(index=index), self.assertRaises(ValueError):
                self.module.validate_metadata(*values, sum_y=4, bfloat16=BF16)
        with self.assertRaises(ValueError):
            self.module.validate_metadata(object(), *self.good()[1:], sum_y=4, bfloat16=BF16)

    def test_rejection_occurs_before_backend_compile(self):
        class NoGPUBackend:
            bfloat16 = BF16
            @property
            def fast(self):
                raise AssertionError("CPU test must never access a kernel backend")
        with self.assertRaises(ValueError):
            self.module.conv_silu_raw_tail(*self.good(), sum_y=True, backend=NoGPUBackend())
        self.assertEqual(self.module._kernel.cache_info().currsize, 0)

    def test_single_owner_and_disjoint_output_indices(self):
        s = self.module.SOURCE
        self.assertIn("uint channel = thread_position_in_grid.x;", s)
        self.assertEqual(self.module.GRID, (10240, 1, 1))
        owners = list(range(self.module.GRID[0]))
        self.assertEqual(len(set(owners)), 10240)
        indices = [row * 10240 + c for row in range(3) for c in owners]
        self.assertEqual(len(indices), len(set(indices)))
        self.assertEqual(set(indices), set(range(30720)))
        self.assertIsNone(re.search(r"(?:raw|old_tail|w)\[[^\]]+\]\s*=", s))
        self.assertEqual(self.module.OUTPUT_SHAPES, ((1, 10240), (1, 3, 10240)))
        self.assertIn('output_names=["act", "new_tail"]', (ROOT / "conv_silu_raw_tail.py").read_text())

    def test_raw_tail_source_mapping(self):
        s = self.module.SOURCE
        for statement in (
            "new_tail[channel] = old_tail[10240 + channel];",
            "new_tail[10240 + channel] = old_tail[20480 + channel];",
            "new_tail[20480 + channel] = raw[channel];",
        ):
            self.assertIn(statement, s)
        for c in (0, 127, 128, 2047, 2048, 4095, 4096, 10239):
            self.assertEqual([10240+c, 20480+c, c], [1*10240+c, 2*10240+c, c])

    def test_predeclared_order_and_bf16_boundaries(self):
        expected = {1: [[0, 1, 2, 3]], 2: [[0, 2], [1, 3]], 3: [[0, 3], [1], [2]], 4: [[0], [1], [2], [3]]}
        for y, groups in expected.items():
            self.assertEqual([list(range(j, 4, y)) for j in range(y)], groups)
        s = self.module.SOURCE
        for statement in (
            "for (uint y = 0; y < SUM_Y; ++y)",
            "bfloat16_t total = bfloat16_t(0);",
            "for (uint tap = y; tap < 4; tap += SUM_Y)",
            "bfloat16_t product = V18NativeMultiply()(input_value, w[channel * 4 + tap]);",
            "total = bfloat16_t(product + total);",
            "bfloat16_t conv = partial[0];",
            "for (uint y = 1; y < SUM_Y; ++y)",
            "conv = bfloat16_t(partial[y] + conv);",
        ):
            self.assertIn(statement, s)
        self.assertLess(s.index("bfloat16_t product"), s.index("total = bfloat16_t(product + total)"))
        self.assertLess(s.index("partial[y] = total"), s.index("bfloat16_t conv"))
        self.assertIsNone(re.search(r"\bfloat\b", s))
        self.assertEqual(self.module.COMPILE_OPTIONS, {"math_mode": "safe"})

    def test_activation_header_is_byte_identical_to_prior_qualified_source(self):
        self.assertEqual(hashlib.sha256(V18.read_bytes()).hexdigest(), V18_SHA)
        header = next(ast.literal_eval(n.value) for n in ast.parse(V18.read_text()).body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "HEADER" for t in n.targets))
        copied = (ROOT / "native_activation.metal").read_bytes()
        self.assertEqual(copied, header.encode())
        self.assertEqual(hashlib.sha256(copied).hexdigest(), HEADER_SHA)
        self.assertIn("act[channel] = v18_bf16_activation(conv).activated;", self.module.SOURCE)

    def test_mit_notice_and_qualification_limit_present(self):
        license_text = (ROOT / "LICENSE-MLX-MIT").read_text()
        for text in ("MIT License", "Copyright © 2023 Apple Inc.", "Permission is hereby granted", "THE SOFTWARE IS PROVIDED"):
            self.assertIn(text, license_text)
        notes = (ROOT / "source-notes.md").read_text()
        self.assertIn("NOT V23 qualification", notes)
        self.assertLessEqual(len(notes.splitlines()), 35)


if __name__ == "__main__":
    unittest.main(verbosity=2)
