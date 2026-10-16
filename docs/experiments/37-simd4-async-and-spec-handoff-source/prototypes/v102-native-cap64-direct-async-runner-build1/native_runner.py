"""Explicit startup CAP64 + old44 direct FIFO adapter; default engine unchanged.

Call configure_physical_cap64(enabled=True) before engine construction/ANY token,
then ordinary warm, then install(). Frozen64/56/41 parents remain immutable.
"""
import ctypes
import hashlib
import importlib.util
from pathlib import Path
import types

P = Path(__file__).resolve().parent.parent
PARENT = P / "v64-native-direct-async-runner-build1/native_runner.py"
PARENT_SHA = "acd59a4d817cb73fba07f6757b3a170a057e3c78ea707710624dda0a65e4bc07"
assert hashlib.sha256(PARENT.read_bytes()).hexdigest() == PARENT_SHA
spec = importlib.util.spec_from_file_location("v102_frozen_parent64", PARENT)
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)


def configure_physical_cap64(*, enabled=False):
    """Explicit startup metadata-only opt-in; caller invokes BEFORE engine/tokens.

    The returned original class steps let a qualification owner restore them in
    finally. This function does not inspect or migrate live caches.
    """
    if type(enabled) is not bool:
        raise TypeError("CAP64 startup switch requires strict bool")
    if not enabled:
        return {"enabled": False, "physical_capacity": None}
    from tutorial_code.miniengine.cache import AttnCache, DecodeBatch
    original = (AttnCache.step, DecodeBatch.step)
    assert all(v in (64, 256) for v in original)
    AttnCache.step = 64
    DecodeBatch.step = 64
    return {"enabled": True, "physical_capacity": 64,
            "original_steps": list(original), "effective_steps": [64, 64],
            "required_call_phase": "before engine construction and ANY token",
            "live_cache_migration": False}


class NativeRunner(old.NativeRunner):
    def __init__(self, engine, library, library_sha256, *, diagnostic=False):
        from tutorial_code.miniengine.cache import AttnCache, DecodeBatch
        assert (AttnCache.step, DecodeBatch.step) == (64, 64), "explicit startup CAP64 preparation required"
        super().__init__(engine, library, library_sha256, diagnostic=diagnostic)
        self.capacity = 64
        self.capacity_metadata = {"physical_capacity": 64, "ordinary_cache_steps": [64, 64],
                                  "startup_opt_in_required": True, "new_model_parameters": 0}

    def build(self, values):
        assert self.capacity == 64 and len(values) == 131
        for i, layer in enumerate(self.runner.model.layers):
            if layer.is_attn:
                assert all(v.shape == (1, 4, 64, 256) and v.dtype == self.mx.bfloat16
                           for v in values[3+2*i:5+2*i]), "physical KV capacity is not64"
        mx = self.mx
        for name, value in self.flatten(self.runner.model.parameters()):
            h = self.shared(value, "parameter." + name); self.parameters.append(h)
            self.call("v33_pin", h.address, h.nbytes, h.label.encode())
        assert len(self.parameters) == 1349
        labels_in = ("tokens", "positions", "lens") + tuple(f"state.{i}" for i in range(128))
        labels_out = ("logits",) + tuple(f"state.{i}" for i in range(128)) + ("prediction", "next_positions", "next_lens")
        self.inputs = [self.shared(v, n) for v, n in zip(values, labels_in)]
        ctx = self.Context("decode", self.layers(values[3:]), positions=values[1], lens=values[2], max_len=self.capacity)
        mx.eval(*(v for cache, layer in zip(ctx.caches, self.runner.model.layers)
                  for v in ((cache.K, cache.V) if layer.is_attn else (cache.conv, cache.S))))
        mx.synchronize()
        self.call("v33_begin")
        logits = self.runner.model(values[0], ctx)
        state = tuple(v for layer, cache in zip(self.runner.model.layers, ctx.caches)
                      for v in ((cache.K, cache.V) if layer.is_attn else (cache.conv, cache.S)))
        captured = (logits, *state, mx.argmax(logits, axis=-1).astype(mx.int32), values[1]+1, values[2]+1)
        mx.eval(*captured); mx.synchronize(); self.call("v33_end_build")
        self.outputs = [self.shared(v, n) for v, n in zip(captured, labels_out)]
        assert len(self.outputs) == 132
        for h in self.outputs:
            self.call("v33_register_output", h.address, h.nbytes, h.label.encode())
        for i, h in enumerate(self.inputs):
            self.call("v34_register_input", h.address, h.nbytes, (1, 2, 3)[i] if i < 3 else 0, h.label.encode())
            other = self.outputs[(129, 130, 131)[i] if i < 3 else i-2]
            assert h.nbytes == other.nbytes
            self.call("v34_link", h.address, other.address, h.nbytes)
        for i, layer in enumerate(self.runner.model.layers):
            if layer.is_attn: continue
            a, b = self.inputs[4+2*i], self.outputs[2+2*i]
            assert a.shape == b.shape == (1, 48, 128, 128) and a.dtype == b.dtype == mx.float32
            assert a.nbytes == b.nbytes == 3145728
            self.call("v36_register_pingpong", a.address, b.address, a.nbytes)
        self.call("v34_seal_feedback", 128, self.capacity); self.call("v38_enable_icb")
        report = self.report(); assert report["status"] == "OK"
        f = report["feedback"]; pp = f["pingpong"]
        assert (f["input_count"], f["output_count"], f["link_count"], f["state_count"], f["capacity"]) == (131, 132, 131, 128, self.capacity)
        assert (pp["requested_count"], pp["eligible_count"], pp["staged_link_count"]) == (48, 48, 83)
        assert f["icb"]["enabled"] and f["icb"]["banks"] == 2
        self.built = True
        self.call("v55_enable_pipeline", 2)
        r = self.report()
        assert r["status"] == "OK" and r["pipeline"]["enabled"]
        assert r["pipeline"]["depth"] == 2


def install(engine, library, library_sha256, *, diagnostic=False):
    adapter = NativeRunner(engine, library, library_sha256, diagnostic=diagnostic)
    engine.runner.run = types.MethodType(lambda runner, out: adapter.run(out), engine.runner)
    engine.runner.drop = types.MethodType(lambda runner, rid: adapter.drop(rid), engine.runner)
    return adapter
