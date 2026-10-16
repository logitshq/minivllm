"""Opt-in C1 FIFO depth-two adapter; old qualified V41 class stays immutable.

Native owns both the live recurrent cache and token feedback until request drop.
Only the current delivered prediction crosses the CPU boundary. One future
prediction may already be computing; drop drains and discards that future work.
"""
from collections import deque
import ctypes
import hashlib
import importlib.util
from pathlib import Path
import types

P = Path(__file__).resolve().parent.parent
PARENT = P / "v41-native-runner-build1/native_runner.py"
PARENT_SHA = "bf7bea08b09dfc2905966b95e77d244ec25947a0c3a0d9506cddd05ca81cfea0"
assert hashlib.sha256(PARENT.read_bytes()).hexdigest() == PARENT_SHA
spec = importlib.util.spec_from_file_location("v56_frozen_parent41", PARENT)
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)


class NativeRunner(old.NativeRunner):
    def __init__(self, engine, library, library_sha256):
        super().__init__(engine, library, library_sha256)
        U = ctypes.c_uint64
        for name, args in {
            "v55_enable_pipeline": [ctypes.c_uint32], "v55_reset": [],
            "v55_submit_step": [ctypes.POINTER(U)], "v55_wait_step": [U],
            "v55_drain": [],
        }.items():
            fn = getattr(self.lib, name)
            fn.argtypes, fn.restype = args, ctypes.c_int
        self.lib.v55_report_json.argtypes = []
        self.lib.v55_report_json.restype = ctypes.c_char_p
        self.pending = deque()
        self.submitted = 0
        self.budget = 0
        self.total_submitted = 0
        self.total_discarded = 0
        self.drops = []

    def report(self):
        # Parent build validates the old schema before pipeline enable.
        import json
        value = self.lib.v34_report_json()
        assert value is not None
        return json.loads(value.decode())

    def build(self, values):
        super().build(values)
        self.call("v55_enable_pipeline", 2)
        r = self.report()
        assert r["status"] == "OK" and r["pipeline"]["enabled"]
        assert r["pipeline"]["depth"] == 2

    def begin(self, request):
        assert not self.pending
        values = self.seed(request)
        if not self.built:
            self.build(values)
        current = [self.shared(v, str(i)) for i, v in enumerate(values)]
        self.call("v55_reset")
        for original, value in zip(self.inputs, current):
            assert original.shape == value.shape and original.dtype == value.dtype
            assert original.nbytes == value.nbytes
            self.call("v34_write_input", original.address, original.nbytes,
                      ctypes.c_void_p(value.address))
        self.active = (request.req_id, request.num_computed_tokens,
                       request.output_token_ids[-1])
        self.submitted = 0
        self.budget = request.sampling.max_tokens - len(request.output_token_ids)
        assert self.budget >= 1
        self.request_count += 1

    def run(self, out):
        if self.active is None:
            if not self.eligible_new(out):
                return self.original_run(out)
            self.runner._sync_batch(out.decode)
            assert self.runner._ahead is None and self.runner.batch.size == 1
            if self.runner.batch.layers_len() != self.capacity:
                return self.original_run(out)
            self.begin(out.decode[0])
        assert len(out.decode) == 1 and not out.prefill
        r = out.decode[0]
        rid, position, token = self.active
        assert r.req_id == rid and r.num_computed_tokens == position
        assert r.output_token_ids[-1] == token
        assert r.sampling.temperature == 0 and position < self.capacity
        assert self.runner.batch.req_ids == [rid] and self.runner._ahead is None
        # Submit ahead of the CPU wait; native's GPU feedback supplies all
        # successor token/state inputs. No future reference token is uploaded.
        while len(self.pending) < 2 and self.submitted < self.budget:
            ticket = ctypes.c_uint64()
            self.call("v55_submit_step", ctypes.byref(ticket))
            self.submitted += 1
            assert ticket.value == self.submitted
            self.pending.append(ticket.value)
            self.total_submitted += 1
        assert self.pending
        self.call("v55_wait_step", self.pending.popleft())
        h = self.outputs[129]
        assert h.shape == (1,) and h.dtype == self.mx.int32 and h.nbytes == 4
        self.call("v34_copy_output", h.address, h.nbytes, ctypes.byref(self.prediction))
        prediction = int(self.prediction.value)
        self.runner.batch.lens = [position + 1]
        self.active = (rid, position + 1, prediction)
        self.native_steps += 1
        return {rid: prediction}

    def drop(self, request_id):
        if self.active is not None and self.active[0] == request_id:
            discarded = len(self.pending)
            self.call("v55_drain")
            self.pending.clear()
            self.total_discarded += discarded
            self.drops.append({"request_id": request_id,
                               "submitted": self.submitted,
                               "discarded": discarded})
            self.active = None
        self.original_drop(request_id)


def install(engine, library, library_sha256):
    adapter = NativeRunner(engine, library, library_sha256)
    engine.runner.run = types.MethodType(lambda runner, out: adapter.run(out), engine.runner)
    engine.runner.drop = types.MethodType(lambda runner, rid: adapter.drop(rid), engine.runner)
    return adapter
