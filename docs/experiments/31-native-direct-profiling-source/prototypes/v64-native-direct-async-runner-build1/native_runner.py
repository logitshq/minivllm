"""Explicit direct-feedback opt-in over frozen, qualified V56 FIFO policy.

Diagnostic mode snapshots all outputs; production mode snapshots predictions
only. Production full-state reads require a completed, quiescent request.
"""
import ctypes
import hashlib
import importlib.util
from pathlib import Path
import types

P = Path(__file__).resolve().parent.parent
PARENT = P / "v56-native-async-runner-build1/native_runner.py"
PARENT_SHA = "af252476a377e9624173a8dbb028650c283b56d5beb0248639b0f00c271dbd59"
assert hashlib.sha256(PARENT.read_bytes()).hexdigest() == PARENT_SHA
spec = importlib.util.spec_from_file_location("v64_frozen_parent56", PARENT)
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)


class NativeRunner(old.NativeRunner):
    def __init__(self, engine, library, library_sha256, *, diagnostic=False):
        assert type(diagnostic) is bool
        self.diagnostic = diagnostic
        super().__init__(engine, library, library_sha256)
        self.lib.v62_enable_direct_pipeline.argtypes = [ctypes.c_uint32, ctypes.c_uint32]
        self.lib.v62_enable_direct_pipeline.restype = ctypes.c_int
        self.lib.v62_copy_quiescent_output.argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_void_p]
        self.lib.v62_copy_quiescent_output.restype = ctypes.c_int

    def call(self, name, *args):
        if name == "v55_enable_pipeline":
            assert args == (2,)
            return super().call("v62_enable_direct_pipeline", 2, int(self.diagnostic))
        return super().call(name, *args)


def install(engine, library, library_sha256, *, diagnostic=False):
    adapter = NativeRunner(engine, library, library_sha256, diagnostic=diagnostic)
    engine.runner.run = types.MethodType(lambda runner, out: adapter.run(out), engine.runner)
    engine.runner.drop = types.MethodType(lambda runner, rid: adapter.drop(rid), engine.runner)
    return adapter
