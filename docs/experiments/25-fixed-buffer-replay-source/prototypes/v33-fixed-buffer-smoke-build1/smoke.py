"""Root-only minimal Metal replay ABI smoke; no model/checkpoint load."""
import ctypes
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

P = Path('/private/tmp/minivllm-engine-optimization-20261014')
out = Path(sys.argv[1]); out.mkdir()
spec = importlib.util.spec_from_file_location('v33_worker_helpers', P / 'v33-fixed-buffer-replay-worker-build1/fullqual.py')
helpers = importlib.util.module_from_spec(spec); spec.loader.exec_module(helpers)
native = helpers.Native(out)
import mlx.core as mx
import numpy as np

x = mx.arange(4096, dtype=mx.float32) / 1024
w = mx.arange(4096, dtype=mx.float32) / 4096 + 1
mx.eval(x, w); mx.synchronize()
xh, wh = helpers.Shared(mx, np, x, 'x'), helpers.Shared(mx, np, w, 'weight')
seed, weight = xh.raw(), wh.raw()
native.call('v33_pin', wh.address, wh.nbytes, b'weight')
native.call('v33_begin')
y = mx.sin(x) * w + x * 2
mx.eval(y); mx.synchronize()
native.call('v33_end_build')
yh = helpers.Shared(mx, np, y, 'y'); expected = yh.raw()
native.call('v33_register_output', yh.address, yh.nbytes, b'y')
for i in range(2):
    native.call('v33_restore_replay')
    target = ctypes.create_string_buffer(yh.nbytes)
    native.call('v33_copy_output', yh.address, yh.nbytes, ctypes.cast(target, ctypes.c_void_p))
    assert target.raw == expected
    assert xh.raw() == seed and wh.raw() == weight and yh.raw() == expected
report = native.report('native-two-replays')
receipt = {'status': 'PASS_TINY_ABI_STRICT_TWO_REPLAY', 'checkpoint_loads': 0,
           'performance_point': False, 'sha256': hashlib.sha256(expected).hexdigest(),
           'dispatches': report['dispatches']}
(out / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt), flush=True)
