"""V81 stdlib-only AST, source/access, stage and inert contracts; no native load."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
import unittest
sys.dont_write_bytecode = True
D = Path(__file__).resolve().parent
P = D.parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Check(unittest.TestCase):
    def test_ast_inert_and_numeric_only(self):
        path = D / "qualify.py"
        tree = ast.parse(path.read_text())
        compile(tree, str(path), "exec")
        for node in tree.body:
            if isinstance(node, ast.Import):
                self.assertFalse(any(alias.name.startswith(("mlx", "numpy")) for alias in node.names))
            if isinstance(node, ast.ImportFrom):
                self.assertFalse((node.module or "").startswith(("mlx", "numpy")))
        spec = importlib.util.spec_from_file_location("v81_inert_cpu", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertEqual(module.GROUPS, (1, 2, 3, 4, 5, 7, 8, 16))
        self.assertEqual(module.ARRAY_BYTES, (20480, 192, 192, 3145728, 8192, 8192, 24576, 128, 24576, 3145728))
        gpu = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "gpu")
        source = ast.unparse(gpu)
        self.assertLess(source.index("DYLD_INSERT_LIBRARIES"), source.index("import mlx.core"))
        self.assertIn("helper._prepare", source)
        self.assertIn("helper._state", source)
        self.assertNotIn("mx.compile", source)
        self.assertNotIn("perf_counter", source)
        self.assertNotIn("v33_restore_replay", source)
        self.assertFalse(any(name == "mlx" or name.startswith(("mlx.", "numpy")) for name in sys.modules))

    def test_recompute_store_only_source_proof(self):
        proof = json.loads((D / "source-access-proof.json").read_text())
        self.assertFalse(proof["actual_reflection_observed"])
        self.assertEqual(len(proof["source_entries"]), 4)
        for entry in proof["source_entries"]:
            path = Path(entry["helper"])
            self.assertEqual(sha(path), entry["helper_sha256"])
            vals = {target.id: ast.literal_eval(node.value) for node in ast.parse(path.read_text()).body if isinstance(node, ast.Assign) and isinstance(node.value, (ast.Constant, ast.Dict)) for target in node.targets if isinstance(target, ast.Name)}
            source = vals[entry["literal"]]
            self.assertEqual(vals["COMPILE_OPTIONS"], {"math_mode": "safe"})
            self.assertEqual(hashlib.sha256(source.encode()).hexdigest(), entry["literal_sha256"])
            for name, policy in entry["output_uses"].items():
                lines = [line.strip() for line in source.splitlines() if re.search(r"\b" + re.escape(name) + r"\b", line)]
                self.assertEqual(lines, policy["all_source_uses"])
                for line in lines:
                    self.assertRegex(line, r"^" + name + r"\[[^;\n]+\]\s*=\s*[^;]+;$")
                    self.assertNotRegex(line.split("=", 1)[1], r"\b" + name + r"\b")
            if entry["stage"] == 1:
                uses = [line.strip() for line in source.splitlines() if re.search(r"\bS\b", line)]
                self.assertEqual(uses, ["held[j] = S[base + d * Dv + dv];"])
        self.assertEqual(sha(P / "v44-gdn-activated-normreuse-build1/gdn_activated_norm_reuse_cached_metal.py"), "5f05383ccd697ee39bb92b2a5cf26254e9c061c0617184139074fa36b4bca952")
        self.assertEqual(sha(P / "v78-gdn-scaled-ready-build1/gdn_activated_scaled_ready_metal.py"), "80fbf529f7cc087437d9ab8cf87f6320567e29ef31cc2007703fd622cadeb569")

    def test_native_boundary_and_alias_guards(self):
        native = (D / "replay.mm").read_text()
        self.assertIn('#include "' + str(P / "v40-native-icb-build2/replay.mm") + '"', native)
        self.assertEqual(native.count('extern "C" int v81_'), 5)
        self.assertEqual(native.count('extern "C" const char*v81_'), 1)
        self.assertIn("if(i%2)[cmd setBarrier];else[cmd clearBarrier]", native)
        self.assertIn("encoded+=30;[e memoryBarrierWithScope:MTLBarrierScopeBuffers]", native)
        self.assertEqual(native.count("restoreArena()"), 1)
        self.assertIn("v81_disjoint_allocations()", native)
        self.assertIn("v81::dependencies(accesses,banks)", native)
        self.assertIn("else if(a.access!=MTLArgumentAccessReadOnly)", native)
        self.assertIn("slot.allocation[a]&&slot.allocation[a]!=id", native)
        self.assertIn("S().pins.size()!=60||S().dispatches!=60||S().outputs.size()!=180", native)
        self.assertIn("bi.zero!=immutable", native)
        self.assertIn("arm!=v.lastArm", native)
        self.assertIn("size!=slot.size[a]", native)
        self.assertIn("!v.capturedBefore[ordinal]", native)

    def test_frozen_later_timing_plan_is_inert(self):
        protocol = json.loads((D / "protocol.json").read_text())
        timing = protocol["timing_plan_frozen_not_implemented"]
        schedule = timing["schedule"]
        self.assertEqual(sum(a != b for a, b in schedule), 6)
        self.assertEqual(sum(a == b for a, b in schedule), 4)
        self.assertEqual(timing["minimum_arm_median_reduction"], .05)
        self.assertEqual(timing["minimum_pair_wins"], 5)
        self.assertEqual(timing["each_control_ratio"], [.95, 1.05])
        self.assertEqual(timing["per_arm_control_median_ratio"], [.98, 1.02])
        self.assertFalse(timing["performance_entrypoint_enabled"])
        self.assertEqual(timing["timed_groups"] * 30, timing["dispatches_per_timed_block"])
        self.assertEqual(protocol["S_payload_bytes"], 15 * 48 * 128 * 128 * 4)
        self.assertIn("Permission is hereby granted", (D / "LICENSE-MLX-MIT").read_text())


if __name__ == "__main__":
    unittest.main()
