"""E010: layer/state exactness before any prefill timing. No HTTP claims."""
import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import mlx.core as mx
from candidate import enabled
from tutorial_code.miniengine.cache import make_cache, DecodeBatch
from tutorial_code.miniengine.context import ForwardContext
from tutorial_code.miniengine.engine.llm_engine import LLMEngine
from tutorial_code.miniengine.model.gdn import _m5_ultra_decode_identity


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def arrays(cache):
    return [a for c in cache for a in ((c.keys, c.values) if hasattr(c, 'keys') else (c.conv_state, c.S))]


def exact(a, b):
    assert a.shape == b.shape and a.dtype == b.dtype, (a.shape, b.shape, a.dtype, b.dtype)
    dt = mx.uint16 if a.dtype == mx.bfloat16 else mx.uint32
    return bool(mx.array_equal(a.view(dt), b.view(dt)).item())


def timed(fn):
    mx.synchronize()
    start = time.perf_counter_ns()
    result = fn()
    mx.eval(result)
    mx.synchronize()
    return (time.perf_counter_ns() - start) / 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workload', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    args = ap.parse_args()
    args.out.mkdir()
    assert _m5_ultra_decode_identity()
    files = list((ROOT / 'tutorial_code/miniengine').rglob('*.py')) + list(Path(__file__).parent.glob('*.py'))
    pins = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    cases = [(2, 19), (8, 19), (26, 19), (32, 19), (2, 128), (8, 128), (2, 512)]
    plan = {'baseline_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            'source_sha256': pins, 'pid': os.getpid(), 'cases_B_T': cases,
            'device': mx.device_info(), 'mlx': importlib.metadata.version('mlx'),
            'workload_sha256': hashlib.sha256(args.workload.read_bytes()).hexdigest(),
            'environment': {k: v for k, v in os.environ.items() if k.startswith(('MLX_', 'METAL_'))},
            'qualification': 'Each layer output + both cache arrays per sequence bit exact. Final logits + three batched decode steps exact. Distinct rotated token IDs per row.',
            'timing': 'After all qualifications: 2 warm pairs then 3 ABBA blocks at B8/26/32 T19 and B8 T128. Fresh cache each call, all state evaluated.',
            'advance': 'Every exact check passes; >=10% prefill median reduction at B26 T19. Then independent HTTP qualification, not a Pareto point.',
            'scope': 'Single process screen; differing long-input cases are safety coverage, not HTTP workload.'}
    write(args.out / 'plan.json', plan)
    work = json.loads(args.workload.read_text())
    engine = LLMEngine(work['model_dir'], max_seqs=64, max_len=2048, max_batched_tokens=4096, cache_limit_gb=2)
    model = engine.model
    checks = []
    for B, T in cases:
        ids = mx.array([[work['prompt_ids'][(i + row * 3) % 19] for i in range(T)] for row in range(B)])
        left = [make_cache(model) for _ in range(B)]
        right = [make_cache(model) for _ in range(B)]
        contexts = [ForwardContext('prefill', c) for c in left]
        grouped = ForwardContext('prefill_batch', list(map(list, zip(*right))), prefill_shape=(B, T))
        xs = [model.embed_tokens(row) for row in ids]
        xg = model.embed_tokens(ids.reshape(-1))
        for li, layer in enumerate(model.layers):
            xs = [layer(x, c, li) for x, c in zip(xs, contexts)]
            with enabled():
                xg = layer(xg, grouped, li)
            mx.eval(xs, xg)
            output_exact = exact(mx.stack(xs), xg.reshape(B, T, -1))
            cache_exact = all(exact(a, b) for row in range(B)
                              for a, b in zip(arrays([left[row][li]]), arrays([right[row][li]])))
            item = {'B': B, 'T': T, 'layer': li, 'output_exact': output_exact, 'cache_exact': cache_exact}
            if not output_exact:
                item['max_abs_error'] = float(mx.max(mx.abs(mx.stack(xs).astype(mx.float32) - xg.reshape(B, T, -1).astype(mx.float32))).item())
            checks.append(item)
            write(args.out / 'layer-checks.json', checks)
            if not (output_exact and cache_exact):
                write(args.out / 'complete.json', {'status': 'REJECT_EXACT', 'first_failure': item})
                print('REJECT', item, flush=True)
                return
        ref = mx.concatenate([model.lm_head_fn(model.norm(x[-1:])) for x in xs])
        with enabled():
            from candidate import prefill_linear
            output = model.embed_tokens.as_linear if model.cfg.tie_word_embeddings else model.lm_head
            cand = prefill_linear(output, model.norm(xg.reshape(B, T, -1)[:, -1]), (B, 1))
        assert exact(ref, cand)
        batches = []
        for caches in (left, right):
            batch = DecodeBatch(model)
            batch.rebuild([], [(str(i), c, T + 32) for i, c in enumerate(caches)])
            batches.append(batch)
        token = mx.argmax(ref, axis=-1).astype(mx.int32)
        for step in range(3):
            outputs = []
            for batch in batches:
                ctx = ForwardContext('decode', batch.layers, positions=mx.array([T + step] * B),
                                     lens=mx.array([T + step + 1] * B), max_len=T + step + 1)
                outputs.append(model(token, ctx))
            assert exact(*outputs)
            assert all(exact(a, b) for a, b in zip(batches[0].arrays(), batches[1].arrays()))
            token = mx.argmax(outputs[0], axis=-1).astype(mx.int32)
        print('PASS layer/cache/logit/decode', B, T, flush=True)
        del left, right, contexts, grouped, xs, xg, batches, batch, outputs, ref, cand, token
        gc.collect()
        mx.clear_cache()
    results = []
    for B, T in [(8, 19), (26, 19), (32, 19), (8, 128)]:
        ids = mx.array([[work['prompt_ids'][(i + row * 3) % 19] for i in range(T)] for row in range(B)])
        def forward(active):
            caches = [make_cache(model) for _ in range(B)]
            if active:
                ctx = ForwardContext('prefill_batch', list(map(list, zip(*caches))), prefill_shape=(B, T))
                with enabled():
                    logits = model(ids, ctx, last_only=True)
            else:
                logits = [model(row, ForwardContext('prefill', c), last_only=True) for row, c in zip(ids, caches)]
            return [logits] + [arrays(c) for c in caches]
        for _ in range(2):
            timed(lambda: forward(False))
            timed(lambda: forward(True))
        samples = []
        for block in range(3):
            for label in ('reference', 'candidate', 'candidate', 'reference'):
                samples.append({'block': block, 'variant': label, 'ms': timed(lambda: forward(label == 'candidate'))})
        median = {v: statistics.median(r['ms'] for r in samples if r['variant'] == v) for v in ('reference', 'candidate')}
        results.append({'B': B, 'T': T, 'samples': samples, 'median_ms': median, 'speedup': median['reference'] / median['candidate']})
        write(args.out / 'timing.json', results)
        print('TIMING', B, T, results[-1]['speedup'], flush=True)
    assert all(hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == h for p, h in pins.items())
    advance = next(r for r in results if r['B'] == 26)['speedup'] >= 1 / 0.9
    write(args.out / 'complete.json', {'status': 'PASS', 'advance_to_http': advance, 'peak_gib': mx.get_peak_memory() / 1024 ** 3, 'results': results})
    print('COMPLETE advance_to_http', advance, flush=True)


if __name__ == '__main__':
    main()
