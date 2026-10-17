"""No MLX import or GPU work: pinned-domain guard and CLI isolation checks."""
import importlib.util
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch


HERE = Path(__file__).resolve().parent


class FakeArray:
    def __init__(self, shape, dtype="float32"):
        self.shape, self.dtype, self.ndim = shape, dtype, len(shape)

    def astype(self, dtype):
        return FakeArray(self.shape, dtype)


class CandidateProtocolTest(unittest.TestCase):
    def setUp(self):
        self.launches = []
        self.definitions = []
        mlx = ModuleType("mlx")
        core = ModuleType("mlx.core")
        core.float32, core.float16, core.bfloat16 = "float32", "float16", "bfloat16"

        def metal_kernel(**definition):
            self.definitions.append(definition)
            def launch(**kwargs):
                self.launches.append(kwargs)
                return [FakeArray(shape) for shape in kwargs["output_shapes"]]
            return launch

        core.fast = SimpleNamespace(metal_kernel=metal_kernel)
        mlx.core = core
        spec = importlib.util.spec_from_file_location("e018_candidate_cpu_check", HERE / "candidate.py")
        self.module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"mlx": mlx, "mlx.core": core}):
            spec.loader.exec_module(self.module)

    def inputs(self, length=3):
        return [FakeArray((length, 1, 48, 128), "bfloat16") for _ in range(3)] + [
            FakeArray((length, 1, 48)) for _ in range(2)] + [FakeArray((1, 48, 128, 128))]

    def test_pinned_shapes_and_launch(self):
        for length in (1, 3, 5, 8):
            output, history = self.module.verify_history(*self.inputs(length))
            self.assertEqual(output.shape, (length, 1, 48, 128))
            self.assertEqual(history.shape, (length, 1, 48, 128, 128))
            launch = self.launches[-1]
            self.assertEqual(launch["template"], [("T", length)])
            self.assertEqual(launch["grid"], (128, 192, 1))
            self.assertEqual(launch["threadgroup"], (32, 4, 1))
            self.assertTrue(all(x.dtype == "float32" for x in launch["inputs"]))

    def test_unsupported_length_rejected_before_launch(self):
        with self.assertRaises(ValueError):
            self.module.verify_history(*self.inputs(4))
        self.assertEqual(self.launches, [])

    def test_state_dtype_rejected_before_launch(self):
        inputs = self.inputs()
        inputs[-1] = FakeArray((1, 48, 128, 128), "bfloat16")
        with self.assertRaises(ValueError):
            self.module.verify_history(*inputs)
        self.assertEqual(self.launches, [])

    def test_gate_shape_rejected_before_launch(self):
        inputs = self.inputs()
        inputs[3] = FakeArray((3, 48))
        with self.assertRaises(ValueError):
            self.module.verify_history(*inputs)
        self.assertEqual(self.launches, [])

    def test_integer_inputs_rejected_before_launch(self):
        inputs = self.inputs()
        inputs[0].dtype = "int32"
        with self.assertRaises(TypeError):
            self.module.verify_history(*inputs)
        self.assertEqual(self.launches, [])

    def test_help_never_imports_gpu_or_numpy(self):
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
        self.assertIn("--workload", result.stdout)


if __name__ == "__main__":
    unittest.main()
