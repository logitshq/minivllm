"""Root-only tiny replay test: large readonly parameter and interior output view."""
import ctypes
import hashlib
import json
from pathlib import Path
import sys

out = Path(sys.argv[1]); out.mkdir()
lib = ctypes.CDLL(None)
for name, types in {
    'v33_pin': [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_char_p],
    'v33_begin': [], 'v33_end_build': [], 'v33_restore_replay': [],
    'v33_register_output': [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_char_p],
    'v33_copy_output': [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_void_p],
}.items():
    getattr(lib, name).argtypes = types; getattr(lib, name).restype = ctypes.c_int
lib.v33_report_json.argtypes = []; lib.v33_report_json.restype = ctypes.c_char_p

def report(name):
    value = json.loads(lib.v33_report_json().decode())
    (out / (name + '.json')).write_text(json.dumps(value, indent=2) + '\n')
    return value

def call(name, *args):
    rc = getattr(lib, name)(*args)
    if rc:
        report('native-first-failure')
        raise RuntimeError(f'{name} rc={rc}')

import mlx.core as mx
import numpy as np

class Shared:
    def __init__(self, value):
        self.owner = value; self.view = memoryview(value)
        assert self.view.c_contiguous
        self.raw = self.view.cast('B'); self.np = np.frombuffer(self.raw, dtype=np.uint8)
        self.address = int(self.np.__array_interface__['data'][0]); self.nbytes = self.np.nbytes

x = mx.arange(4096, dtype=mx.float32) / 1024
w = mx.arange(8 * 1024 * 1024, dtype=mx.float32) / (8 * 1024 * 1024) + 1
idx = mx.arange(4096, dtype=mx.int32)
mx.eval(x, w, idx); mx.synchronize()
xh, wh, ih = Shared(x), Shared(w), Shared(idx)
seed, weight, indices = bytes(xh.raw), bytes(wh.raw), bytes(ih.raw)
call('v33_pin', wh.address, wh.nbytes, b'weight32MiB')
call('v33_begin')
y = mx.sin(x) * w[idx] + x * 2
mx.eval(y); mx.synchronize()
call('v33_end_build')
tail = y[1:]; mx.eval(tail); mx.synchronize()
yh, th = Shared(y), Shared(tail)
assert th.address == yh.address + 4
holders = [yh, th]; expected = [bytes(h.raw) for h in holders]
for name, holder in zip([b'y', b'tail'], holders):
    call('v33_register_output', holder.address, holder.nbytes, name)
for replay in range(2):
    call('v33_restore_replay')
    for holder, reference in zip(holders, expected):
        target = ctypes.create_string_buffer(holder.nbytes)
        call('v33_copy_output', holder.address, holder.nbytes, ctypes.cast(target, ctypes.c_void_p))
        assert target.raw == reference and bytes(holder.raw) == reference
    assert bytes(xh.raw) == seed and bytes(wh.raw) == weight and bytes(ih.raw) == indices
r = report('native-two-replays')
assert r['zero_copy_buffer_bytes'] == 32 * 1024 * 1024
receipt = {'status': 'PASS_TINY_TWO_REPLAY_INTERIOR_OUTPUT_AND_LARGE_WEIGHT',
           'checkpoint_loads': 0, 'performance_point': False,
           'output_sha256': [hashlib.sha256(v).hexdigest() for v in expected],
           'dispatches': r['dispatches'], 'zero_copy_buffer_bytes': r['zero_copy_buffer_bytes']}
(out / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt), flush=True)
