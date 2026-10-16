"""Compare retiring rows against sequential retirement, including surviving state."""
import argparse, hashlib, importlib.util, json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import mlx.core as mx
import numpy as np
from tutorial_code.miniengine.engine.llm_engine import LLMEngine
from tutorial_code.miniengine.engine.model_runner import ModelRunner
from tutorial_code.miniengine.sampling_params import SamplingParams

ap = argparse.ArgumentParser()
ap.add_argument('--source', type=Path, required=True)
ap.add_argument('--baseline', type=Path, default=ROOT / '.artifacts/m5-ultra-256gb/pareto-batch-source-v1')
ap.add_argument('--out', type=Path, required=True)
a = ap.parse_args()
a.out.mkdir()

def candidate(name, root):
    path = root / 'tutorial_code/miniengine/engine' / (name + '.py')
    spec = importlib.util.spec_from_file_location('tutorial_code.miniengine.engine.retire_' + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

OldEngine = candidate('llm_engine', a.baseline).LLMEngine
OldRunner = candidate('model_runner', a.baseline).ModelRunner
NewEngine = candidate('llm_engine', a.source).LLMEngine
NewRunner = candidate('model_runner', a.source).ModelRunner
w = json.loads((ROOT / '.artifacts/m5-ultra-256gb/fullqual-build2/workload.json').read_text())
e = LLMEngine(w['model_dir'], max_seqs=64, max_len=2048, max_batched_tokens=4096, cache_limit_gb=2,
              compile_decode=True, fuse_decode=True)
for layer in e.model.layers:
    if not layer.is_attn:
        layer.linear_attn.set_gates_compile(True)

def signature(x):
    dtype = mx.uint16 if x.dtype == mx.bfloat16 else mx.uint32
    return {'shape': list(x.shape), 'dtype': str(x.dtype),
            'sha256': hashlib.sha256(np.asarray(x.view(dtype)).tobytes()).hexdigest()}

def run(new, B, staggered=False):
    e.__class__ = NewEngine if new else OldEngine
    e.runner.__class__ = NewRunner if new else OldRunner
    assert not e.has_work() and not e.requests and not e.runner.batch.size
    outputs = {}
    counts = {'rebuilds': 0, 'clears': 0}
    batch = e.runner.batch
    rebuild, clear = batch.rebuild, batch.clear
    def counted_rebuild(*args):
        counts['rebuilds'] += 1
        return rebuild(*args)
    def counted_clear():
        counts['clears'] += 1
        return clear()
    batch.rebuild, batch.clear = counted_rebuild, counted_clear
    def add(rid, limit, stop=False):
        e.add_request(w['prompt_ids'], SamplingParams(temperature=0, max_tokens=limit,
                      stop_token_ids=[79852] if stop else []), req_id=rid)
    for i in range(B):
        add('r' + str(i), (4 + 2 * (i % 3)) if staggered else 4)
    rows, step = [], 0
    try:
        while e.has_work():
            if staggered and step == 2:
                add('late-a', 3)
                add('late-b', 5)
                add('late-stop', 6, True)
            before = dict(counts)
            start = time.perf_counter()
            result = e.step()
            mx.synchronize()
            elapsed = time.perf_counter() - start
            for rid, tok, done in result:
                outputs.setdefault(rid, []).append(tok)
            rows.append({'outputs': result, 'batch': list(batch.req_ids), 'lens': list(batch.lens),
                         'requests': list(e.requests), 'states': [signature(x) for x in batch.arrays()],
                         'rebuilds': counts['rebuilds'] - before['rebuilds'],
                         'clears': counts['clears'] - before['clears'], 'elapsed_s': elapsed})
            step += 1
        assert not e.runner.staging and not e.runner.feats and e.runner._ahead is None
        assert not e.requests and not batch.size and not e.has_work()
    finally:
        batch.rebuild, batch.clear = rebuild, clear
    return {'rows': rows, 'ids': outputs, 'counts': counts}

checks = []
for B, staggered in ((2, False), (8, False), (32, False), (64, False), (8, True)):
    ref, new = run(False, B, staggered), run(True, B, staggered)
    assert ref['ids'] == new['ids']
    assert len(ref['rows']) == len(new['rows'])
    for old, now in zip(ref['rows'], new['rows']):
        for key in ('outputs', 'batch', 'lens', 'requests', 'states'):
            assert old[key] == now[key], (B, staggered, key)
    if not staggered:
        assert ref['rows'][-1]['rebuilds'] == B - 1
        assert new['rows'][-1]['rebuilds'] == 0 and new['rows'][-1]['clears'] == 1
    checks.append({'B': B, 'staggered': staggered, 'exact': True, 'reference': ref, 'candidate': new})
    (a.out / 'checks.json').write_text(json.dumps(checks, indent=2))
    print('PASS', B, 'staggered', staggered, ref['counts'], new['counts'], flush=True)
(a.out / 'complete.json').write_text(json.dumps({'status': 'PASS', 'cases': len(checks),
    'source_files': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in a.source.rglob('*.py')},
    'baseline_files': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in a.baseline.rglob('*.py')},
    'scope': 'Same outputs and surviving cache bytes after every engine step; simultaneous and staggered retirement.'}, indent=2))
