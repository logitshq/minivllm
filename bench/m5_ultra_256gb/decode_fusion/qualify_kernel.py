"""Exact fused convolution/state checks over diverse synthetic scales and states."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from tutorial_code.miniengine.kernels.gdn_metal import gated_delta_step_metal
from tutorial_code.miniengine.kernels.gdn_decode_fused import fused

ap = argparse.ArgumentParser()
ap.add_argument('--out', type=Path, required=True)
a = ap.parse_args()
a.out.mkdir()
rows = []
def digest(x):
    return hashlib.sha256(np.asarray(x.astype(mx.float32)).tobytes()).hexdigest()
for seed in range(4):
    for scale in (0.001, 0.125, 1, 16):
        mx.random.seed(1400 + seed)
        # Slicing exercises noncontiguous inputs accepted by the kernel wrapper.
        raw = (mx.random.normal((1, 20480)) * scale).astype(mx.bfloat16)[:, ::2]
        conv = (mx.random.normal((1, 3, 20480)) * scale).astype(mx.bfloat16)[:, :, ::2]
        w = mx.random.normal((10240, 4)).astype(mx.bfloat16)
        S = mx.random.normal((1, 48, 128, 128)) * (seed + 1)
        g = -mx.random.uniform(0, 10, (1, 48))
        beta = mx.sigmoid(mx.random.normal((1, 48)))
        xp = mx.concatenate([conv, raw[:, None]], axis=1)
        u = (xp * w.T[None]).sum(axis=1)
        q, k, v = mx.split(nn.silu(u), [2048, 4096], axis=-1)
        q = mx.repeat(q.reshape(1, 16, 128), 3, axis=1)
        k = mx.repeat(k.reshape(1, 16, 128), 3, axis=1)
        o, state = gated_delta_step_metal(q, k, v.reshape(1, 48, 128), g, beta, S)
        reference = (o, state, xp[:, 1:])
        candidate = fused(raw, conv, w, g, beta, S)
        mx.eval(reference, candidate)
        h1, h2 = [digest(x) for x in reference], [digest(x) for x in candidate]
        rows.append({'seed': seed, 'scale': scale, 'reference': h1, 'candidate': h2, 'exact': h1 == h2})
        (a.out / 'checks.json').write_text(json.dumps(rows, indent=2))
        assert h1 == h2, rows[-1]
(a.out / 'complete.json').write_text(json.dumps({'status': 'PASS', 'cases': len(rows)}, indent=2))
print('PASS', len(rows), flush=True)
