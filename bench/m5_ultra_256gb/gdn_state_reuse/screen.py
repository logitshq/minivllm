"""E009: paired retained-state GDN screen; not an HTTP benchmark."""
import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import mlx.core as mx
import mlx.nn as nn
from candidate import enabled, fused
from tutorial_code.miniengine.cache import BatchKV, BatchState, DecodeBatch, make_cache
from tutorial_code.miniengine.context import ForwardContext
from tutorial_code.miniengine.engine.llm_engine import LLMEngine
from tutorial_code.miniengine.kernels.gdn_metal import gated_delta_step_metal
from tutorial_code.miniengine.model.gdn import _m5_ultra_decode_identity
from tutorial_code.miniengine.sampling_params import SamplingParams


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def exact(left, right):
    assert len(left) == len(right)
    results = []
    for a, b in zip(left, right):
        assert a.shape == b.shape and a.dtype == b.dtype
        dtype = mx.uint16 if a.dtype == mx.bfloat16 else mx.uint32
        results.append(mx.array_equal(a.view(dtype), b.view(dtype)))
    mx.eval(results)
    return all(bool(x.item()) for x in results)


def timed(fn):
    mx.synchronize()
    start = time.perf_counter_ns()
    result = fn()
    mx.eval(result)
    mx.synchronize()
    return (time.perf_counter_ns() - start) / 1e6


def paired(reference, candidate):
    cold = {'reference': timed(reference), 'candidate': timed(candidate)}
    for _ in range(2):
        timed(reference)
        timed(candidate)
    rows = []
    for block in range(3):
        for label in ('reference', 'candidate', 'candidate', 'reference'):
            rows.append({'block': block, 'variant': label,
                         'ms': timed(reference if label == 'reference' else candidate)})
    medians = {k: statistics.median(r['ms'] for r in rows if r['variant'] == k)
               for k in ('reference', 'candidate')}
    return {'first_timed_ms_after_qualification': cold, 'samples': rows, 'median_ms': medians,
            'speedup': medians['reference'] / medians['candidate']}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workload', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir()
    assert _m5_ultra_decode_identity(), 'This prototype is qualified on M5 Ultra only'
    mx.set_cache_limit(2 * 1024 ** 3)
    files = list((ROOT / 'tutorial_code/miniengine').rglob('*.py')) + list(Path(__file__).parent.glob('*.py'))
    pins = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    plan = {'hypothesis': 'Retain 32 FP32 state elements per lane; avoid the second source-level state read without changing arithmetic. Register allocation and actual traffic are unproven.', 'baseline_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            'pid': os.getpid(), 'source_sha256': pins,
            'workload_sha256': hashlib.sha256(args.workload.read_bytes()).hexdigest(),
            'batches': [9, 16, 32, 64], 'seeds': [11, 29], 'scales': [0.125, 8],
            'timing': '2 warm rounds, 3 ABBA blocks; all raw values and first timed calls after qualification retained; not cold compilation',
            'advance': 'Require every exact check and >= 5% full-forward median gain at B32 and B64; then separate HTTP qualification',
            'scope': 'single fresh process screen; direct forwards exclude scheduler and HTTP; never add to Pareto plot',
            'device': mx.device_info(), 'mlx': importlib.metadata.version('mlx'),
            'environment': {k: v for k, v in os.environ.items() if k.startswith(('MLX_', 'METAL_'))}}
    write(args.out / 'plan.json', plan)
    checks, kernels, forwards = [], [], []
    for B in plan['batches']:
        for seed in plan['seeds']:
            for scale in plan['scales']:
                mx.random.seed(seed)
                raw = (mx.random.normal((B, 20480)) * scale).astype(mx.bfloat16)[:, ::2]
                conv = (mx.random.normal((B, 3, 20480)) * scale).astype(mx.bfloat16)[:, :, ::2]
                w = mx.random.normal((10240, 4)).astype(mx.bfloat16)
                state = mx.random.normal((B, 48, 128, 128))
                g = -mx.random.uniform(0, 10, (B, 48))
                beta = mx.sigmoid(mx.random.normal((B, 48)))
                mx.eval(raw, conv, w, state, g, beta)
                def reference():
                    xp = mx.concatenate([conv, raw[:, None]], axis=1)
                    u = (xp * w.T[None]).sum(axis=1)
                    q, k, v = mx.split(nn.silu(u), [2048, 4096], axis=-1)
                    q = mx.repeat(q.reshape(B, 16, 128), 3, axis=1)
                    k = mx.repeat(k.reshape(B, 16, 128), 3, axis=1)
                    o, s = gated_delta_step_metal(q, k, v.reshape(B, 48, 128), g, beta, state)
                    return o, s, xp[:, 1:]
                def candidate():
                    return fused(raw, conv, w, g, beta, state)
                equal = exact(reference(), candidate())
                checks.append({'B': B, 'seed': seed, 'scale': scale, 'exact': equal})
                write(args.out / 'kernel-checks.json', checks)
                assert equal, checks[-1]
                if seed == 11 and scale == 0.125:
                    kernels.append({'B': B, **paired(reference, candidate)})
                    write(args.out / 'kernel-timing.json', kernels)
        print('PASS kernel', B, flush=True)
    del raw, conv, w, state, g, beta
    gc.collect()
    mx.clear_cache()
    workload = json.loads(args.workload.read_text())
    engine = LLMEngine(workload['model_dir'], max_seqs=64, max_len=2048,
                       max_batched_tokens=4096, cache_limit_gb=2)
    model = engine.model
    caches = make_cache(model)
    logits = model(mx.array(workload['prompt_ids']), ForwardContext('prefill', caches), last_only=True)
    mx.eval(logits, [a for c in caches for a in
                    ((c.keys, c.values) if hasattr(c, 'keys') else (c.conv_state, c.S))])
    def arrays(layers):
        return [a for c in layers for a in ((c.K, c.V) if isinstance(c, BatchKV) else (c.conv, c.S))]
    def copies(layers):
        return [BatchKV(c.K[:], c.V[:]) if isinstance(c, BatchKV) else BatchState(c.conv, c.S) for c in layers]
    full_checks = []
    for B in [8] + plan['batches']:
        batch = DecodeBatch(model)
        batch.rebuild([], [(str(i), caches, 256) for i in range(B)])
        mx.eval(batch.arrays())
        left, right = copies(batch.layers), copies(batch.layers)
        tokens = mx.array([workload['continuation_ids'][i % 32] for i in range(B)])
        def forward(active, layers, pos=19):
            ctx = ForwardContext('decode', layers, positions=mx.array([pos] * B),
                                 lens=mx.array([pos + 1] * B), max_len=pos + 1)
            with enabled(active):
                output = model(tokens, ctx)
            return [output] + arrays(layers)
        for step in range(3):
            old, new = forward(False, left, 19 + step), forward(True, right, 19 + step)
            equal = exact(old, new)
            full_checks.append({'B': B, 'step': step, 'exact_logits_and_128_cache_arrays': equal})
            write(args.out / 'full-checks.json', full_checks)
            assert equal, full_checks[-1]
            tokens = mx.argmax(old[0], axis=-1).astype(mx.int32)
            mx.eval(tokens)
        del old, new, left, right
        forwards.append({'B': B, **paired(lambda: forward(False, copies(batch.layers)),
                                          lambda: forward(True, copies(batch.layers)))})
        write(args.out / 'forward-timing.json', forwards)
        print('PASS full forward', B, forwards[-1]['speedup'], flush=True)
        del batch
        gc.collect()
        mx.clear_cache()
    def schedule(active):
        trace = []
        for i in range(12):
            engine.add_request(workload['prompt_ids'], SamplingParams(temperature=0, max_tokens=4 + i % 3), req_id=str(i))
        step = 0
        with enabled(active):
            while engine.has_work():
                if step == 2:
                    engine.add_request(workload['prompt_ids'], SamplingParams(temperature=0, max_tokens=4), req_id='late')
                output = engine.step()
                mx.synchronize()
                # Stream integer-view cache bytes into hashes, never serialize model data.
                import numpy as np
                hashes = [hashlib.sha256(np.asarray(a.view(mx.uint16 if a.dtype == mx.bfloat16 else mx.uint32)).tobytes()).hexdigest()
                          for a in engine.runner.batch.arrays()]
                trace.append({'outputs': output, 'batch': list(engine.runner.batch.req_ids), 'cache_sha256': hashes})
                step += 1
        assert not engine.runner.staging and not engine.runner.batch.size and engine.runner._ahead is None
        return trace
    before, after = schedule(False), schedule(True)
    write(args.out / 'scheduler.json', {'reference': before, 'candidate': after, 'exact': before == after})
    assert before == after
    assert all(hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == h for p, h in pins.items())
    advance = all(r['speedup'] >= 1.05 for r in forwards if r['B'] in (32, 64))
    write(args.out / 'complete.json', {'status': 'PASS', 'advance_to_http': advance,
          'kernel_cases': len(checks), 'full_forwards': len(full_checks), 'scheduler_exact': True,
          'forward_results': forwards, 'peak_gib': mx.get_peak_memory() / 1024 ** 3,
          'decision': 'HTTP qualification required' if advance else 'Do not integrate: screen failed preregistered performance threshold'})
    print('COMPLETE advance_to_http', advance, flush=True)


if __name__ == '__main__':
    main()
