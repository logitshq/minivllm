"""E011 full-model exact checks followed by an ABBA forward screen."""
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
import traceback

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import mlx.core as mx
from candidate import candidate_class
from tutorial_code.miniengine.cache import BatchKV, BatchState, DecodeBatch, make_cache
from tutorial_code.miniengine.context import ForwardContext
from tutorial_code.miniengine.engine.compiled_decode import CompiledDecode
from tutorial_code.miniengine.engine.llm_engine import LLMEngine
from tutorial_code.miniengine.model.gdn import _m5_ultra_decode_identity


def write(p, value):
    p.write_text(json.dumps(value, indent=2) + '\n')


def arrays(layers):
    return [a for c in layers for a in ((c.K, c.V) if isinstance(c, BatchKV) else (c.conv, c.S))]


def copies(layers):
    return [BatchKV(c.K[:], c.V[:]) if isinstance(c, BatchKV) else BatchState(c.conv, c.S) for c in layers]


def timed(fn):
    mx.synchronize()
    start = time.perf_counter_ns()
    output = fn()
    mx.eval(output)
    mx.synchronize()
    return (time.perf_counter_ns() - start) / 1e6


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--workload', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    a = p.parse_args()
    a.out.mkdir()
    checks, timing = [], []
    try:
        cls, candidate_source = candidate_class()
        (a.out / 'compiled_decode_candidate.py').write_text(candidate_source)
        files = sorted((ROOT / 'tutorial_code/miniengine').rglob('*.py')) + sorted(Path(__file__).parent.glob('*.py'))
        pins = {str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
        plan = {'experiment': 'E011', 'baseline_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                'source_sha256': pins, 'candidate_sha256': hashlib.sha256(candidate_source.encode()).hexdigest(),
                'batches': [8, 9, 16, 32, 64], 'correctness': 'Each full logits and 128 cache arrays byte-equal, 3 consecutive steps plus positions255/256; stop at first mismatch',
                'timing': 'first timed A/B after numerical qualification retained (not cold compilation); two A/B warm rounds; three ABBA blocks; one fresh process screen only',
                'advance': 'All exact checks; B16 full forward >=1.10x, B32 and B64 >=0.97x, B8 control >=0.97x; then independent scheduler and paired HTTP qualification',
                'change': 'Graph-only B9..64; original per-projection arithmetic and kernels; B1..8 existing fusion unchanged',
                'hypothesis': 'Avoid rebuilding large decode graphs without the large-batch packed QMM route that previously regressed',
                'scope': 'Forward screen, not HTTP or new Pareto points',
                'device': mx.device_info(), 'mlx': importlib.metadata.version('mlx'),
                'workload_sha256': hashlib.sha256(a.workload.read_bytes()).hexdigest(),
                'environment': {k: v for k, v in os.environ.items() if k.startswith(('MLX_', 'METAL_'))}}
        write(a.out / 'plan.json', plan)
        assert _m5_ultra_decode_identity()
        w = json.loads(a.workload.read_text())
        engine = LLMEngine(w['model_dir'], max_seqs=64, max_len=2048, cache_limit_gb=2)
        model = engine.model
        for layer in model.layers:
            if not layer.is_attn:
                layer.linear_attn.set_gates_compile(True)
        cs = make_cache(model)
        output = model(mx.array(w['prompt_ids']), ForwardContext('prefill', cs), last_only=True)
        mx.eval(output, [x for c in cs for x in ((c.keys, c.values) if hasattr(c, 'keys') else (c.conv_state, c.S))])
        for B in plan['batches']:
            reference, candidate = CompiledDecode(model, fuse_decode=True), cls(model, fuse_decode=True)
            batch = DecodeBatch(model)
            batch.rebuild([], [(str(i), cs, 512) for i in range(B)])
            mx.eval(batch.arrays())
            ids = mx.array([w['continuation_ids'][i % 32] for i in range(B)])
            left, right = copies(batch.layers), copies(batch.layers)
            def forward(function, state, pos=19):
                ctx = ForwardContext('decode', state, positions=mx.array([pos] * B),
                                     lens=mx.array([pos + 1] * B), max_len=pos + 1)
                return [function(ids, ctx)] + arrays(state)
            for pos in (19, 20, 21, 255, 256):
                old, new = forward(reference, left, pos), forward(candidate, right, pos)
                eq = [mx.array_equal(x.view(mx.uint16 if x.dtype == mx.bfloat16 else mx.uint32),
                                     y.view(mx.uint16 if y.dtype == mx.bfloat16 else mx.uint32)) for x, y in zip(old, new)]
                mx.eval(eq)
                row = {'B': B, 'position': pos, 'bit_equal': [bool(x.item()) for x in eq]}
                checks.append(row)
                write(a.out / 'checks.json', checks)
                assert len(row['bit_equal']) == 129 and all(row['bit_equal']), ('numerical mismatch', B, pos)
                ids = mx.argmax(old[0], axis=-1).astype(mx.int32)
                mx.eval(ids)
            assert candidate._functions
            if B > 8:
                assert not reference._functions and reference._packed is None and candidate._packed is None
            # Frozen input/cache for timing; cache outputs are evaluated, not discarded.
            def run(v):
                return forward(reference if v == 'reference' else candidate, copies(batch.layers))
            cold = {v: timed(lambda: run(v)) for v in ('reference', 'candidate')}
            for _ in range(2):
                for v in ('reference', 'candidate'):
                    timed(lambda: run(v))
            samples = []
            for block in range(3):
                for v in ('reference', 'candidate', 'candidate', 'reference'):
                    samples.append({'block': block, 'variant': v, 'ms': timed(lambda: run(v))})
            medians = {v: statistics.median(r['ms'] for r in samples if r['variant'] == v) for v in ('reference', 'candidate')}
            row = {'B': B, 'first_timed_ms_after_qualification': cold, 'samples': samples, 'median_ms': medians,
                   'speedup': medians['reference'] / medians['candidate'],
                   'candidate_graph_count': len(candidate._functions), 'large_batch_packs': candidate._packed is not None}
            timing.append(row)
            write(a.out / 'forward-timing.json', timing)
            print('PASS B', B, 'speedup', row['speedup'], flush=True)
            del old, new, left, right, batch, reference, candidate
            gc.collect()
            mx.clear_cache()
        assert all(hashlib.sha256((ROOT / f).read_bytes()).hexdigest() == sha for f, sha in pins.items())
        by_b = {r['B']: r['speedup'] for r in timing}
        advance = by_b[16] >= 1.10 and min(by_b[b] for b in (8, 32, 64)) >= .97
        write(a.out / 'complete.json', {'status': 'PASS', 'checks': len(checks), 'advance': advance,
              'decision': 'Continue scheduler and HTTP qualification' if advance else 'Do not integrate: preregistered screen gate failed',
              'peak_active_gib': mx.get_peak_memory() / 1024**3, 'timing': timing})
        print('COMPLETE advance', advance, flush=True)
    except BaseException:
        write(a.out / 'failure.json', {'status': 'STOP', 'error': traceback.format_exc(), 'completed_checks': len(checks),
              'completed_timing_configs': len(timing), 'all_raw_preserved': True})
        raise


if __name__ == '__main__':
    main()
