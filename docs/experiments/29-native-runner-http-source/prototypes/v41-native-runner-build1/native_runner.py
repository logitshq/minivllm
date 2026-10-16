"""Experimental C1-only adapter for the qualified fixed-cap native ICB graph.

The ordinary runner handles prefill and every ineligible request. A request that
enters native decode stays there until drop; its live cache is owned by native.
This adapter deliberately requires scheduler capacity one, so no batch transition
can consume the ordinary runner's stale seed cache. It is default OFF.
"""
import ctypes
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import types


class Shared:
    def __init__(self, mx, np, value, label):
        self.original, self.label = value, label
        self.alias = value.view(mx.uint16) if value.dtype == mx.bfloat16 else value
        mx.eval(self.alias)
        self.buffer = memoryview(self.alias)
        if not self.buffer.c_contiguous:
            raise ValueError("native ICB requires contiguous input: " + label)
        self.bytes_view = self.buffer.cast("B")
        self.numpy_view = np.frombuffer(self.bytes_view, dtype=np.uint8)
        self.address = int(self.numpy_view.__array_interface__["data"][0])
        self.nbytes = int(value.nbytes)
        self.shape, self.dtype = value.shape, value.dtype
        assert self.address and self.nbytes == self.buffer.nbytes


class NativeRunner:
    def __init__(self, engine, library, library_sha256):
        import mlx.core as mx
        import numpy as np
        from tutorial_code.miniengine.cache import BatchKV, BatchState
        from tutorial_code.miniengine.context import ForwardContext
        from mlx.utils import tree_flatten

        library = Path(library).resolve()
        assert hashlib.sha256(library.read_bytes()).hexdigest() == library_sha256
        assert os.environ.get("DYLD_INSERT_LIBRARIES") == str(library)
        assert {k: os.environ.get(k) for k in (
            "MLX_MAX_OPS_PER_BUFFER", "MLX_MAX_MB_PER_BUFFER",
            "MLX_BFS_MAX_WIDTH", "MLX_METAL_FAST_SYNCH")} == {
                "MLX_MAX_OPS_PER_BUFFER": "50", "MLX_MAX_MB_PER_BUFFER": "50",
                "MLX_BFS_MAX_WIDTH": "20", "MLX_METAL_FAST_SYNCH": "0"}
        assert all(importlib.metadata.version(k) == "0.32.3" for k in ("mlx", "mlx-metal"))
        info = mx.device_info()
        assert mx.default_device() == mx.gpu
        assert info["device_name"] == "Apple M4 Max" and info["architecture"] == "applegpu_g16s"
        runner = engine.runner
        assert engine.scheduler.max_seqs == 1 and runner.spec is None
        assert runner._ahead is None and runner.batch.size == 0 and not engine.requests
        assert getattr(runner, "_functional_decode_helper", None) is None
        assert len(engine.model.layers) == 64
        gdns = [x.linear_attn for x in engine.model.layers if not x.is_attn]
        assert len(gdns) == 48 and all(x._gates_compile_enabled is True for x in gdns)
        self.mx, self.np, self.BKV, self.BS, self.Context = mx, np, BatchKV, BatchState, ForwardContext
        self.flatten, self.runner = tree_flatten, runner
        self.lib = ctypes.CDLL(None)
        U, V, S = ctypes.c_uint64, ctypes.c_void_p, ctypes.c_char_p
        specs = {"v33_pin": [U, U, S], "v33_begin": [], "v33_end_build": [],
                 "v33_register_output": [U, U, S],
                 "v34_register_input": [U, U, ctypes.c_uint32, S],
                 "v34_link": [U, U, U], "v34_seal_feedback": [ctypes.c_uint32, ctypes.c_uint32],
                 "v34_reset": [], "v34_write_input": [U, U, V], "v34_replay_step": [],
                 "v34_compare_output": [U, U, V], "v34_copy_output": [U, U, V],
                 "v36_register_pingpong": [U, U, U], "v38_enable_icb": []}
        for name, args in specs.items():
            fn = getattr(self.lib, name); fn.argtypes = args; fn.restype = ctypes.c_int
        self.lib.v34_report_json.argtypes = []; self.lib.v34_report_json.restype = S
        self.prediction = ctypes.c_int32()
        self.parameters, self.inputs, self.outputs = [], [], []
        self.active = None; self.native_steps = 0; self.request_count = 0
        self.built = False; self.capacity = 256
        self.original_run, self.original_drop = runner.run, runner.drop

    def report(self):
        value = self.lib.v34_report_json()
        assert value is not None
        return json.loads(value.decode())

    def call(self, name, *args):
        code = getattr(self.lib, name)(*args)
        if code:
            raise RuntimeError(f"{name} failed rc={code}: " + json.dumps(self.report()))

    def shared(self, value, label):
        return Shared(self.mx, self.np, value, label)

    def layers(self, flat):
        return [self.BKV(flat[2*i][:], flat[2*i+1][:]) if layer.is_attn
                else self.BS(flat[2*i], flat[2*i+1])
                for i, layer in enumerate(self.runner.model.layers)]

    def seed(self, request):
        mx = self.mx
        p = request.num_computed_tokens
        values = (mx.array([request.output_token_ids[-1]], mx.int32),
                  mx.array([p], mx.int32), mx.array([p+1], mx.int32),
                  *(v[:] for v in self.runner.batch.arrays()))
        assert len(values) == 131
        mx.eval(*values); mx.synchronize()
        return values

    def build(self, values):
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
        assert (f["input_count"], f["output_count"], f["link_count"], f["state_count"], f["capacity"]) == (131, 132, 131, 128, 256)
        assert (pp["requested_count"], pp["eligible_count"], pp["staged_link_count"]) == (48, 48, 83)
        assert f["icb"]["enabled"] and f["icb"]["banks"] == 2
        self.built = True

    def begin(self, request):
        values = self.seed(request)
        if not self.built: self.build(values)
        current = [self.shared(v, str(i)) for i, v in enumerate(values)]
        self.call("v34_reset")
        for original, value in zip(self.inputs, current):
            assert original.shape == value.shape and original.dtype == value.dtype and original.nbytes == value.nbytes
            self.call("v34_write_input", original.address, original.nbytes, ctypes.c_void_p(value.address))
        self.active = (request.req_id, request.num_computed_tokens, request.output_token_ids[-1])
        self.request_count += 1

    def eligible_new(self, out):
        if len(out.decode) != 1 or out.prefill: return False
        r = out.decode[0]
        s = r.sampling
        return (s.temperature == 0 and len(r.output_token_ids) == 1
                and r.num_prompt_tokens + s.max_tokens <= self.capacity
                and r.num_computed_tokens == r.num_prompt_tokens)

    def run(self, out):
        if self.active is None:
            if not self.eligible_new(out): return self.original_run(out)
            self.runner._sync_batch(out.decode)
            assert self.runner._ahead is None and self.runner.batch.size == 1
            if self.runner.batch.layers_len() != self.capacity:
                return self.original_run(out)
            self.begin(out.decode[0])
        assert len(out.decode) == 1 and not out.prefill
        r = out.decode[0]; rid, position, token = self.active
        assert r.req_id == rid and r.num_computed_tokens == position and r.output_token_ids[-1] == token
        assert r.sampling.temperature == 0 and position < self.capacity
        assert self.runner.batch.req_ids == [rid] and self.runner._ahead is None
        self.call("v34_replay_step")
        h = self.outputs[129]
        assert h.shape == (1,) and h.dtype == self.mx.int32 and h.nbytes == 4
        self.call("v34_copy_output", h.address, h.nbytes, ctypes.byref(self.prediction))
        prediction = int(self.prediction.value)
        self.runner.batch.lens = [position+1]
        self.active = (rid, position+1, prediction); self.native_steps += 1
        return {rid: prediction}

    def drop(self, request_id):
        if self.active is not None and self.active[0] == request_id:
            self.active = None
        self.original_drop(request_id)


def install(engine, library, library_sha256):
    adapter = NativeRunner(engine, library, library_sha256)
    # Bind only this runner instance; the original class and other engines remain intact.
    engine.runner.run = types.MethodType(lambda runner, out: adapter.run(out), engine.runner)
    engine.runner.drop = types.MethodType(lambda runner, rid: adapter.drop(rid), engine.runner)
    return adapter
