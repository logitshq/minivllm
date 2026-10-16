"""V85 CPU-only identity, original numeric AST and timing/gate checks."""
import ast
import copy
import importlib.util
import json
from pathlib import Path
import sys
import unittest
sys.dont_write_bytecode = True
D = Path(__file__).resolve().parent
P = D.parent
OLD = P / "v81-gdn-two-kernel-native-range-build1/qualify.py"


def functions(path):
    return {node.name: node for node in ast.parse(path.read_text()).body if isinstance(node, ast.FunctionDef)}


def load():
    spec = importlib.util.spec_from_file_location("v85_cpu", D / "micro.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Check(unittest.TestCase):
    def test_original_numeric_gpu_AST_unchanged_except_one_handoff(self):
        old, new = functions(OLD), functions(D / "micro.py")
        body = copy.deepcopy(new["gpu"])
        removed = []
        class Strip(ast.NodeTransformer):
            def visit_If(self, node):
                if ast.unparse(node.test) == "args.performance":
                    self_test = ast.unparse(node)
                    assert self_test == "if args.performance:\n    return performance(args, result, pins, save, probe, call, report, expected, readonly)"
                    removed.append(node)
                    return None
                return self.generic_visit(node)
        body = Strip().visit(body)
        self.assertEqual(len(removed), 1)
        self.assertEqual(ast.dump(body), ast.dump(old["gpu"]))
        for name in ("sha", "write", "module"):
            self.assertEqual(ast.dump(new[name]), ast.dump(old[name]))

    def test_plan_exact_frozen81_and_single_clocked_call(self):
        module = load()
        old = json.loads((P / "v81-gdn-two-kernel-native-range-build1/protocol.json").read_text())["timing_plan_frozen_not_implemented"]
        self.assertEqual(module.PRIME, old["prime_groups"])
        self.assertEqual(module.REPEATS, old["timed_groups"])
        self.assertEqual([list(order) for order in module.SCHEDULE], old["schedule"])
        self.assertEqual(module.LIB, P / "v81-gdn-two-kernel-native-range-build1/libv81replay.dylib")
        self.assertEqual(module.REPEATS * 30, 30720)
        tree = functions(D / "micro.py")["performance"]
        block = next(node for node in ast.walk(tree) if isinstance(node, ast.For) and ast.unparse(node.target) == "(block_index, arm_name)")
        i = next(i for i, node in enumerate(block.body) if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "start")
        self.assertEqual(ast.unparse(block.body[i:i + 3]), "start = time.perf_counter_ns()\ncall('v81_loop_range', arm, REPEATS)\nend = time.perf_counter_ns()")
        self.assertEqual(ast.unparse(block.body[i - 1]), "call('v81_loop_range', arm, PRIME)")
        source = ast.unparse(tree)
        self.assertLess(source.index("end = time.perf_counter_ns()"), source.index("v81_copy_output"))
        self.assertIn("== 600", source)
        self.assertEqual(sum(a != b for a, b in module.SCHEDULE), 6)
        self.assertEqual(sum(a == b for a, b in module.SCHEDULE), 4)

    def test_gate_meaningful_boundary_cases(self):
        module = load()
        def pairs(candidate=94):
            result = []
            for i, order in enumerate(module.SCHEDULE):
                blocks = [{"arm": arm, "ns_per_dispatch": 100 if arm == "A" else candidate} for arm in order]
                row = {"order": order, "blocks": blocks, "kind": "control" if order[0] == order[1] else "AB"}
                if row["kind"] == "control":
                    row["slot2_div_slot1"] = 1.
                else:
                    row["reduction"] = 1 - candidate / 100
                result.append(row)
            return result
        self.assertEqual(module.score(pairs())["status"], "EXPLORATORY_PASS_NOT_ADOPTED")
        self.assertFalse(module.score(pairs())["adopted"])
        self.assertEqual(module.score(pairs(96))["status"], "STOP_MICRO_GATE")
        self.assertEqual(module.score(pairs(101))["wins"], 0)
        p = pairs()
        p[0]["slot2_div_slot1"] = 1.06
        self.assertEqual(module.score(p)["status"], "CONTROL_DRIFT_STOP")
        p = pairs()
        for row in p:
            if row["kind"] == "control" and row["order"][0] == "A":
                row["slot2_div_slot1"] = 1.03
        self.assertEqual(module.score(p)["status"], "CONTROL_DRIFT_STOP")
        p = pairs(80)
        for row in [row for row in p if row["kind"] == "AB"][:2]:
            row["reduction"] = -.1
        self.assertEqual(module.score(p)["status"], "STOP_MICRO_GATE")

    def test_inert_import_and_receipt_before_runtime(self):
        path = D / "micro.py"
        tree = ast.parse(path.read_text())
        compile(tree, str(path), "exec")
        for node in tree.body:
            if isinstance(node, ast.Import):
                self.assertFalse(any(alias.name.startswith(("mlx", "numpy")) for alias in node.names))
        source = ast.unparse(functions(path)["source_check"])
        self.assertIn("pins.get(path) == expected", source)
        self.assertIn("actual['pins'] == old_pins", source)
        self.assertNotIn("actual['pins'] == pins", source)
        main = ast.unparse(functions(path)["main"])
        self.assertLess(main.index("source_check()"), main.index("gpu(args, pins, support)"))
        self.assertLess(main.index("args.qualified_sha == V81_SHA"), main.index("gpu(args, pins, support)"))
        self.assertFalse(any(name == "mlx" or name.startswith(("mlx.", "numpy")) for name in sys.modules))


if __name__ == "__main__":
    unittest.main()
