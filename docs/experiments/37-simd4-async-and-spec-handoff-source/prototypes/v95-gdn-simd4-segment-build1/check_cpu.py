"""V95 stdlib-only source, lane ownership and fake backend checks; no GPU/MLX."""
import ast
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
sys.dont_write_bytecode = True
D = Path(__file__).resolve().parent
P = D.parent
HELPER = D / "gdn_activated_simd4_segment_metal.py"
OLD = P / "v44-gdn-activated-normreuse-build1/gdn_activated_norm_reuse_cached_metal.py"


def vals(path):
    return {node.targets[0].id: ast.literal_eval(node.value)
            for node in ast.parse(path.read_text()).body
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, (ast.Constant, ast.Dict))}


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


class Check(unittest.TestCase):
    def test_reverse_state_source_and_unchanged_prework(self):
        old, new = vals(OLD), vals(HELPER)
        for key in ("HEADER", "_NORM_BODY", "_PREWORK_SOURCE", "_NORM_PROBE_SOURCE", "COMPILE_OPTIONS", "SPLIT", "VALUE_TILE"):
            self.assertEqual(old[key], new[key], key)
        source = new["_STATE_SOURCE"]
        self.assertNotIn("threadgroup_barrier", source)
        self.assertNotIn("threadgroup float", source)
        self.assertNotIn("simd_sum", source)
        self.assertEqual(source.count("simd_shuffle("), 2)
        self.assertIn("float held[32];", source)
        self.assertEqual(source.count("for (uint j = 0; j < 32; j++)"), 2)
        self.assertEqual(source.count("for (uint i = 0; i < SPLIT; i++)"), 2)
        self.assertEqual(source.count("if ("), 1)
        self.assertLess(source.rindex("simd_shuffle("), source.index("if (sp == 0)"))
        changes = [
            ("    uint lane = thread_index_in_simdgroup;\n    uint dv_in_simd = lane % 8;\n    uint dv_local = thread_position_in_threadgroup.y * 8 + dv_in_simd;\n    uint dv = threadgroup_position_in_grid.x * VALUE_TILE + dv_local;\n    uint sp = lane / 8;", "    uint dv_local = thread_position_in_threadgroup.x;\n    uint dv = threadgroup_position_in_grid.x * VALUE_TILE + dv_local;\n    uint sp = thread_position_in_threadgroup.y;"),
            ("    float qs = scales[", "    threadgroup float red[SPLIT * VALUE_TILE];\n    float qs = scales["),
            ("    float kv_part = kv;\n    kv = 0.0f;\n    for (uint i = 0; i < SPLIT; i++) kv += simd_shuffle(kv_part, dv_in_simd + i * 8);", "    red[sp * VALUE_TILE + dv_local] = kv;\n    threadgroup_barrier(mem_flags::mem_threadgroup);\n    kv = 0.0f;\n    for (uint i = 0; i < SPLIT; i++) kv += red[i * VALUE_TILE + dv_local];"),
            ("    float acc_part = acc;\n    acc = 0.0f;\n    for (uint i = 0; i < SPLIT; i++) acc += simd_shuffle(acc_part, dv_in_simd + i * 8);\n    if (sp == 0) {", "    threadgroup_barrier(mem_flags::mem_threadgroup);\n    red[sp * VALUE_TILE + dv_local] = acc;\n    threadgroup_barrier(mem_flags::mem_threadgroup);\n    if (sp == 0) {\n        acc = 0.0f;\n        for (uint i = 0; i < SPLIT; i++) acc += red[i * VALUE_TILE + dv_local];"),
        ]
        for a, b in changes:
            self.assertEqual(source.count(a), 1)
            source = source.replace(a, b, 1)
        self.assertEqual(source, old["_STATE_SOURCE"])
        self.assertEqual((D / "prework.metal").read_text(), new["_PREWORK_SOURCE"])
        self.assertEqual((D / "simd4-state.metal").read_text(), new["_STATE_SOURCE"])

    def test_expected_lane_mapping_bijection_and_source_lanes(self):
        # A CPU model of the required x-fast mapping; actual mapping is gated by the GPU integer probe.
        owners = [(x // 8, y * 8 + x % 8) for y in range(4) for x in range(32)]
        self.assertEqual(set(owners), {(sp, dv) for sp in range(4) for dv in range(32)})
        for y in range(4):
            for lane in range(32):
                sources = [lane % 8 + i * 8 for i in range(4)]
                self.assertEqual([(s // 8, y * 8 + s % 8) for s in sources], [(sp, y * 8 + lane % 8) for sp in range(4)])
                self.assertTrue(all(0 <= s < 32 for s in sources))
        self.assertEqual([sp * 32 + j for sp in range(4) for j in range(32)], list(range(128)))
        covered = {(tile * 32 + dv, sp * 32 + j) for tile in range(4) for sp, dv in owners for j in range(32)}
        self.assertEqual(covered, {(dv, d) for dv in range(128) for d in range(128)})
        self.assertEqual(sum(sp == 0 for sp, dv in owners), 32)

    def test_wrapper_ast_only_state_pso_name_changes(self):
        def functions(path):
            return {node.name: node for node in ast.parse(path.read_text()).body if isinstance(node, ast.FunctionDef)}
        old, new = functions(OLD), functions(HELPER)
        self.assertEqual(set(old), set(new))
        for name, node in new.items():
            source = ast.unparse(node).replace("gdn_simd4_segment_cached_state_v95_build1", "gdn_norm_reuse_cached_state_v40_build1")
            self.assertEqual(ast.dump(ast.parse(source)), ast.dump(ast.parse(ast.unparse(old[name]))), name)

    def test_fake_backend_launch_contract(self):
        helper = load("v95_cpu_helper", HELPER)
        class Array:
            def __init__(self, shape, dtype):
                self.shape, self.dtype = shape, dtype
        factories, launches = [], []
        def factory(**metadata):
            factories.append(metadata)
            def kernel(**args):
                launches.append(args)
                return [Array(s, t) for s, t in zip(args["output_shapes"], args["output_dtypes"])]
            return kernel
        mx = NS(bfloat16="bf16", float32="f32", fast=NS(metal_kernel=factory))
        helper._runtime = lambda: mx
        act, g, S = Array((1, 10240), "bf16"), Array((1, 48), "f32"), Array((1, 48, 128, 128), "f32")
        result = helper.gated_delta_step_activated_norm_reuse_metal(act, g, g, S)
        self.assertEqual([a.shape for a in result], [(1, 48, 128), (1, 48, 128, 128)])
        self.assertEqual([f["name"] for f in factories], ["gdn_activated_norm_reuse_prework_v44_build1", "gdn_simd4_segment_cached_state_v95_build1"])
        self.assertEqual(factories[1]["input_names"], ["q_ready", "k_ready", "v", "scales", "g", "beta", "S"])
        self.assertEqual(factories[1]["output_names"], ["o", "S_new"])
        self.assertIs(launches[1]["inputs"][6], S)  # Expected ABI only; no actual native reflection here.
        self.assertEqual(launches[0]["grid"], (8192, 1, 1))
        self.assertEqual(launches[0]["threadgroup"], (128, 1, 1))
        self.assertEqual(launches[1]["grid"], (128, 192, 1))
        self.assertEqual(launches[1]["threadgroup"], (32, 4, 1))
        self.assertTrue(all(f["compile_options"] == {"math_mode": "safe"} for f in factories))
        with self.assertRaises(ValueError):
            helper.gated_delta_step_activated_norm_reuse_metal(Array((1, 10240), "f32"), g, g, S)
        helper._runtime = lambda: self.fail("shape rejection precedes runtime")
        with self.assertRaises(ValueError):
            helper.gated_delta_step_activated_norm_reuse_metal(Array((2, 10240), "bf16"), g, g, S)

    def test_numeric_failure_counts_and_gate(self):
        numeric = load("v95_cpu_numeric", D / "numeric.py")
        self.assertEqual(numeric.BASE, OLD)
        self.assertEqual((len(numeric.REAL), len(numeric.SYNTHETIC)), (9, 6))
        self.assertEqual(numeric.completed_counts([]), {"strict_output_checks": 0, "input_readonly_cases": 0})
        self.assertEqual(numeric.completed_counts([{"outputs": [{"strict_equal": True}, {"strict_equal": False}], "input_readonly": True}, {}]), {"strict_output_checks": 1, "input_readonly_cases": 1})
        source = (D / "numeric.py").read_text()
        self.assertEqual(source.count('activated = nn.silu(bank["u"])'), 1)
        self.assertIn('assert rows == expected', source)
        self.assertIn('"actual_state_reflection_observed": False', source)
        self.assertIn('"output_feedback": False', source)
        tree = ast.parse(source)
        handler = next(node for node in ast.walk(tree) if isinstance(node, ast.ExceptHandler) and node.name == "error")
        self.assertIsInstance(handler.body[-1], ast.Raise)
        self.assertFalse(any(isinstance(n, ast.Assert) for stmt in handler.body for n in ast.walk(stmt)))

    def test_syntax_inert_and_license(self):
        for path in (HELPER, D / "numeric.py", D / "check_cpu.py"):
            tree = ast.parse(path.read_text())
            compile(tree, str(path), "exec")
            for node in tree.body:
                if isinstance(node, ast.Import):
                    self.assertFalse(any(a.name.startswith(("mlx", "numpy")) for a in node.names))
                if isinstance(node, ast.ImportFrom):
                    self.assertFalse((node.module or "").startswith(("mlx", "numpy")))
        self.assertIn("Permission is hereby granted", (D / "LICENSE-MLX-MIT").read_text())
        self.assertFalse(any(n == "mlx" or n.startswith(("mlx.", "numpy")) for n in sys.modules))


if __name__ == "__main__":
    unittest.main()
