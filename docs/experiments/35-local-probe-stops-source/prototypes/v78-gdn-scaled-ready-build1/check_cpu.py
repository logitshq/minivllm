"""V78 pure stdlib source/API/proof checks; fake backend, no MLX/GPU import."""
import ast
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
D = Path(__file__).resolve().parent
P = D.parent
HELPER = D / "gdn_activated_scaled_ready_metal.py"
OLD = P / "v44-gdn-activated-normreuse-build1/gdn_activated_norm_reuse_cached_metal.py"


def vals(path):
    return {target.id: ast.literal_eval(node.value) for node in ast.parse(path.read_text()).body if isinstance(node, ast.Assign) and isinstance(node.value, (ast.Constant, ast.Dict)) for target in node.targets if isinstance(target, ast.Name)}


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


class Check(unittest.TestCase):
    def test_exact_reverse_source_proof(self):
        old, new = vals(OLD), vals(HELPER)
        for key in ("HEADER", "_NORM_BODY", "COMPILE_OPTIONS", "SPLIT", "VALUE_TILE"):
            self.assertEqual(old[key], new[key])
        source = new["_PREWORK_SOURCE"]
        self.assertEqual(source.count(new["_NORM_BODY"]), 1)
        source = source.replace("    threadgroup float ready_scales[2];\n", "", 1)
        source = source.replace("            ready_scales[0] = qs;\n            ready_scales[1] = ks;\n", "", 1)
        source = source.replace("        threadgroup_barrier(mem_flags::mem_threadgroup);\n        q_ready[h * 128 + lane] = ready_scales[0] * qh[lane];\n        k_ready[h * 128 + lane] = ready_scales[1] * kh[lane];", "        q_ready[h * 128 + lane] = qh[lane];\n        k_ready[h * 128 + lane] = kh[lane];", 1)
        self.assertEqual(source, old["_PREWORK_SOURCE"])
        source = new["_STATE_SOURCE"]
        self.assertNotIn("scales[", source)
        self.assertNotIn("float qs", source)
        self.assertNotIn("float ks", source)
        source = source.replace("    float ex =", old["_SCALE_LOAD"] + "    float ex =", 1)
        source = source.replace("kv += kh[d] * held[j];", "kv += ks * kh[d] * held[j];", 1)
        source = source.replace("float s = ex * held[j] + kh[d] * delta;", "float s = ex * held[j] + ks * kh[d] * delta;", 1)
        source = source.replace("acc += qh[d] * s;", "acc += qs * qh[d] * s;", 1)
        self.assertEqual(source, old["_STATE_SOURCE"])
        self.assertNotIn("v18_bf16_activation(", new["_PREWORK_SOURCE"])
        self.assertEqual((D / "scaled-prework.metal").read_text(), new["_PREWORK_SOURCE"])
        self.assertEqual((D / "scaled-state.metal").read_text(), new["_STATE_SOURCE"])

    def test_uniform_barrier_and_unique_ownership(self):
        source = vals(HELPER)["_PREWORK_SOURCE"]
        self.assertEqual(source.count("threadgroup_barrier(mem_flags::mem_threadgroup)"), 2)
        self.assertIn("ready_scales[1] = ks;\n        }\n        threadgroup_barrier", source)
        self.assertEqual([h * 128 + lane for h in range(16) for lane in range(128)], list(range(2048)))
        self.assertEqual([(group - 16) * 128 + lane for group in range(16, 64) for lane in range(128)], list(range(6144)))
        self.assertEqual([sp * 32 + j for sp in range(4) for j in range(32)], list(range(128)))
        self.assertEqual([group * 32 + lane for group in range(4) for lane in range(32)], list(range(128)))

    def test_wrapper_ast_is_unchanged_except_pso_names(self):
        def functions(path):
            return {node.name: node for node in ast.parse(path.read_text()).body if isinstance(node, ast.FunctionDef)}
        old, new = functions(OLD), functions(HELPER)
        self.assertNotIn("_norm_probe", new)
        self.assertEqual(set(new), set(old) - {"_norm_probe"})
        for name, node in new.items():
            source = ast.unparse(node)
            source = source.replace("gdn_activated_scaled_ready_prework_v78_build1", "gdn_activated_norm_reuse_prework_v44_build1")
            source = source.replace("gdn_scaled_ready_cached_state_v78_build1", "gdn_norm_reuse_cached_state_v40_build1")
            self.assertEqual(ast.dump(ast.parse(source)), ast.dump(ast.parse(ast.unparse(old[name]))))

    def test_fake_launch_api_slots_and_safe_policy(self):
        candidate = load("v78_cpu_helper", HELPER)
        class Array:
            def __init__(self, shape, dtype):
                self.shape, self.dtype = shape, dtype
        factories, launches = [], []
        def factory(**metadata):
            factories.append(metadata)
            def kernel(**args):
                launches.append(args)
                return [Array(shape, dtype) for shape, dtype in zip(args["output_shapes"], args["output_dtypes"])]
            return kernel
        mx = NS(bfloat16="bf16", float32="f32", fast=NS(metal_kernel=factory))
        candidate._runtime = lambda: mx
        activated, g, S = Array((1, 10240), "bf16"), Array((1, 48), "f32"), Array((1, 48, 128, 128), "f32")
        result = candidate.gated_delta_step_activated_norm_reuse_metal(activated, g, g, S)
        self.assertEqual(len(result), 2)
        self.assertEqual(factories[0]["name"], "gdn_activated_scaled_ready_prework_v78_build1")
        self.assertEqual(factories[1]["name"], "gdn_scaled_ready_cached_state_v78_build1")
        self.assertEqual(factories[1]["input_names"], ["q_ready", "k_ready", "v", "scales", "g", "beta", "S"])
        self.assertIs(launches[1]["inputs"][6], S)
        self.assertEqual(launches[1]["inputs"][3].shape, (1, 16, 2))
        self.assertEqual(launches[0]["grid"], (8192, 1, 1))
        self.assertEqual(launches[0]["threadgroup"], (128, 1, 1))
        self.assertEqual(launches[1]["grid"], (128, 192, 1))
        self.assertEqual(launches[1]["threadgroup"], (32, 4, 1))
        self.assertTrue(all(item["compile_options"] == {"math_mode": "safe"} for item in factories))
        with self.assertRaises(ValueError):
            candidate.gated_delta_step_activated_norm_reuse_metal(Array((1, 10240), "f32"), g, g, S)
        candidate._runtime = lambda: self.fail("shape rejection must precede runtime")
        with self.assertRaises(ValueError):
            candidate.gated_delta_step_activated_norm_reuse_metal(Array((2, 10240), "bf16"), g, g, S)

    def test_failure_counts_and_strict_reference(self):
        numeric = load("v78_cpu_numeric", D / "numeric.py")
        self.assertEqual(numeric.completed_counts([]), {"strict_output_checks": 0, "strict_ready_checks": 0, "strict_diagnostic_checks": 0, "input_readonly_cases": 0})
        cases = [{"outputs": [{"strict_equal": True}, {"strict_equal": False}], "ready": [{"strict_equal": True}], "input_readonly": True}, {}]
        self.assertEqual(numeric.completed_counts(cases), {"strict_output_checks": 1, "strict_ready_checks": 1, "strict_diagnostic_checks": 0, "input_readonly_cases": 1})
        source = (D / "reference-scaled-ready-probe.metal").read_text()
        self.assertIn("q_expected[i] = qs * q_raw[i];", source)
        self.assertIn("k_expected[i] = ks * k_raw[i];", source)
        self.assertNotIn("bfloat", source)
        tree = ast.parse((D / "numeric.py").read_text())
        handler = next(node for node in ast.walk(tree) if isinstance(node, ast.ExceptHandler) and node.name == "error")
        self.assertFalse(any(isinstance(node, ast.Assert) for statement in handler.body for node in ast.walk(statement)))
        self.assertIsInstance(handler.body[-1], ast.Raise)
        self.assertEqual(numeric.BASE, OLD)
        self.assertEqual(len(numeric.REAL), 9)
        self.assertEqual(len(numeric.SYNTHETIC), 6)

    def test_syntax_inert_and_license(self):
        for path in (HELPER, D / "numeric.py", D / "check_cpu.py"):
            tree = ast.parse(path.read_text())
            compile(tree, str(path), "exec")
            for node in tree.body:
                if isinstance(node, ast.Import):
                    self.assertFalse(any(alias.name.startswith(("mlx", "numpy")) for alias in node.names))
                if isinstance(node, ast.ImportFrom):
                    self.assertFalse((node.module or "").startswith(("mlx", "numpy")))
        self.assertIn("Permission is hereby granted", (D / "LICENSE-MLX-MIT").read_text())
        self.assertFalse(any(name == "mlx" or name.startswith(("mlx.", "numpy")) for name in sys.modules))


if __name__ == "__main__":
    unittest.main()
