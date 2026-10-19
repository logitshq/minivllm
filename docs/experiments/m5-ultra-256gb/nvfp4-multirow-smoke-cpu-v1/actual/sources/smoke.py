"""Explicit standalone-child numerical smoke; default is description only."""
import argparse
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import signal
import struct
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
CANDIDATE = Path(__file__).with_name('candidate.py')
B1 = ROOT/'tutorial_code/miniengine/kernels/nvfp4_qmv.py'
PINS = {str(CANDIDATE): 'ee9c5eef0e1282125310ee6661be1ec5ce64dc93a82add48a6d6c17f8958816f',
        str(B1): 'd79f6b3c44eb0003f9c6a4b5ece79c1d8bdd896fc6095c4458e4ea400b1c9c32'}
SHAPES = ((512, 64), (5120, 256), (5120, 6144))
LIMIT, DEADLINE = 256 << 20, 120
Q = dict(transpose=True, group_size=16, bits=4, mode='nvfp4')


def need(ok, message):
    if not ok: raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for raw in iter(lambda: f.read(1 << 20), b''): h.update(raw)
    return h.hexdigest()


def write(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2); f.write('\n'); f.flush()


def load(path, name):
    need(sha(path) == PINS[str(path)], 'changed pinned source: '+str(path))
    sp = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
    return m


def memory(mx, *arrays):
    mx.synchronize()
    values = dict(active=mx.get_active_memory(), peak=mx.get_peak_memory(),
                  cache=mx.get_cache_memory(), buffers=mx.get_array_buffer_size(arrays))
    need(max(values.values()) < LIMIT, 'MLX buffer/peak bound exceeded')
    return values


def save(mx, path, arrays):
    mx.eval(*arrays.values()); memory(mx, arrays)
    mx.save_safetensors(str(path), arrays)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path))


def original_bytes(path):
    raw = path.read_bytes(); n = struct.unpack('<Q', raw[:8])[0]
    header = json.loads(raw[8:8+n]); result = {}
    for name, meta in header.items():
        if name == '__metadata__': continue
        need(meta['dtype'] == 'BF16', 'original output BF16 required')
        lo, hi = meta['data_offsets']; result[name] = raw[8+n+lo:8+n+hi]
        need(len(result[name]) == 2*meta['shape'][0]*meta['shape'][1], 'raw output length')
    return result


def weights(mx, folder, K, N):
    """Only <=256-row dense BF16 bands; no full dense or FP32 weight matrix."""
    wp, sp, source = [], [], []
    for offset in range(0, N, 256):
        rows = min(256, N-offset)
        dense = (((mx.arange(K)[None, :] + (mx.arange(rows)[:, None]+offset)*13) % 127-63)/16).astype(mx.bfloat16)
        pair = mx.quantize(dense, group_size=16, bits=4, mode='nvfp4')
        need(len(pair) == 2, 'NVFP4 must return weight/scales only')
        mx.eval(dense, *pair); memory(mx, dense, pair, wp, sp)
        source.append(save(mx, folder/f'source-band-{offset:05d}.safetensors', {'dense': dense}))
        wp.append(pair[0]); sp.append(pair[1])
        del dense, pair
    w, s = mx.concatenate(wp), mx.concatenate(sp)
    mx.eval(w, s); memory(mx, w, s, wp, sp)
    write(folder/'quantization.json', dict(recipe='BF16 (((column+row*13)%127)-63)/16; quantize NVFP4 group16 bits4, no global scale', source=source))
    return w, s


def case(mx, project, b1, folder, w, s, T, K, N, kind, strided, weight_ref):
    folder.mkdir(); write(folder/'intent.json', dict(T=T, K=K, N=N, kind=kind, column_stride2=strided, weights=weight_ref))
    if kind in ('zero', 'negative-zero'):
        x = mx.full((T, K), 0 if kind == 'zero' else 0x8000, dtype=mx.uint16).view(mx.bfloat16)
    else:
        x = (((mx.arange(T*K).reshape(T, K) % 37)-18)*(128 if kind == 'stress' else 0.125)).astype(mx.bfloat16)
    if strided:
        x, w, s = (mx.repeat(a, 2, axis=1)[:, ::2] for a in (x, w, s))
    inputs = save(mx, folder/'inputs.safetensors', {'x': x})
    # Both oracles issue independent M1 calls. Never native M=T QMM.
    def oracle(fn):
        rows = []
        for t in range(T):
            value = fn(x[t:t+1]); mx.eval(value); memory(mx, value, x, w, s)
            rows.append(value); mx.clear_cache()  # Sequential numerical checks bound stride-staging copies.
        return mx.concatenate(rows)
    native = oracle(lambda row: mx.quantized_matmul(row, w, s, **Q))
    custom = oracle(lambda row: b1.qmv(row, w, s))
    actual = project(x, w, s); need(actual is not None, 'candidate rejected fixed smoke case')
    out = dict(native_b1=native, custom_b1=custom, candidate=actual)
    need(all(a.shape == (T, N) and a.dtype == mx.bfloat16 for a in out.values()), 'output layout/dtype')
    receipt = save(mx, folder/'outputs.safetensors', out); raw = original_bytes(Path(receipt['path']))
    comparisons = {}
    for left, right in (('candidate', 'native_b1'), ('candidate', 'custom_b1'), ('custom_b1', 'native_b1')):
        a, b = raw[left], raw[right]; first = next((i for i in range(0, len(a), 2) if a[i:i+2] != b[i:i+2]), None)
        comparisons[left+'__'+right] = dict(equal=a == b, first_mismatch=None if first is None else
            dict(row=first//(2*N), column=first//2 % N, left=a[first:first+2].hex(), right=b[first:first+2].hex()))
    finite = all(all(word[0] & 0x7f80 != 0x7f80 for word in struct.iter_unpack('<H', data)) for data in raw.values())
    result = dict(status='PASS' if finite and all(r['equal'] for r in comparisons.values()) else 'FAIL',
                  finite=finite, comparisons=comparisons, inputs=inputs, outputs=receipt,
                  original_dtype_sha256={k:hashlib.sha256(v).hexdigest() for k,v in raw.items()}, memory=memory(mx, x, w, s, out))
    write(folder/'result.json', result); need(result['status'] == 'PASS', 'raw BF16 mismatch/nonfinite: '+folder.name)


def device(out):
    import mlx.core as mx
    identity = dict(version=importlib.metadata.version('mlx'), name=mx.device_info().get('device_name'),
                    architecture=mx.metal.device_info().get('architecture'), core_path=mx.__file__)
    write(out/'runtime.json', identity)
    need(identity['version'] == '0.32.3' and identity['name'] == 'Apple M5 Ultra'
         and identity['architecture'] == 'applegpu_g17d' and mx.default_device() == mx.gpu, 'fixed runtime/device')
    mx.set_cache_limit(0); mx.set_memory_limit(LIMIT)  # Guideline, not a process RSS cap.
    candidate = load(CANDIDATE, '_reuse_smoke_candidate'); b1 = load(B1, '_reuse_smoke_b1')
    project = candidate.build(mx, enabled=True); need(project is not None, 'candidate build rejected')
    completed = 0
    for K, N in SHAPES:
        folder = out/f'K{K}-N{N}'; folder.mkdir()
        w, s = weights(mx, folder, K, N); gc.collect(); mx.synchronize(); mx.clear_cache()
        weight_ref = save(mx, folder/'packed.safetensors', {'w':w, 'scales':s})
        for T in (2, 4, 8):
            for kind in (('finite', 'zero', 'negative-zero', 'stress') if K == 512 else ('finite',)):
                for strided in (False, True):
                    case(mx, project, b1, folder/f'T{T}-{kind}-stride{2 if strided else 1}', w, s, T, K, N, kind, strided, weight_ref)
                    completed += 1; gc.collect(); mx.synchronize(); mx.clear_cache()
        del w, s
        gc.collect(); mx.synchronize(); mx.clear_cache(); memory(mx)
    need(completed == 36 and all(sha(Path(p)) == h for p,h in PINS.items()), 'all cases/source closure')
    return dict(status='PASS_SYNTHETIC_BYTES_ONLY',cases=completed,memory=memory(mx),
                full_model_qualified=False,performance_qualified=False)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--run', action='store_true'); p.add_argument('--out', type=Path)
    a = p.parse_args(argv)
    if not a.run:
        need(a.out is None, '--out requires --run'); print('DESCRIPTION ONLY: 36 synthetic byte checks; explicit --run in a standalone child, no service/model load.'); return 0
    need(a.out is not None and a.out.is_absolute() and a.out.parent.resolve() == a.out.parent and not a.out.exists(), 'fresh absolute output required')
    need(__name__ == '__main__', 'run only as a standalone CLI child')
    need(signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0), 'existing timer')
    a.out.mkdir(); write(a.out/'invocation.json', dict(argv=sys.argv, pins=PINS, smoke_sha256=sha(Path(__file__)),deadline_seconds=DEADLINE,limit_bytes=LIMIT))
    need(all(sha(Path(path)) == h for path,h in PINS.items()), 'source pins before any backend import')
    write(a.out/'incomplete.json', dict(status='INCOMPLETE_UNTIL_RESULT',hard_deadline='Default SIGALRM terminates this child at 120s; retain nonzero child exit and partial artifacts.',start_monotonic=time.monotonic()))
    signal.signal(signal.SIGALRM, signal.SIG_DFL); signal.setitimer(signal.ITIMER_REAL, DEADLINE)
    try:
        result = device(a.out); write(a.out/'result.json', result); print(json.dumps(result), flush=True); return 0
    except BaseException as error:
        write(a.out/'failure.json', dict(status='FAIL',type=type(error).__name__,message=str(error))); raise


if __name__ == '__main__': raise SystemExit(main())
