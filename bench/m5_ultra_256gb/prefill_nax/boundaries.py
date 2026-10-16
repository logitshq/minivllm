"""Additional E010 gates before expanding beyond the 19-token workload."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import mlx.core as mx
from candidate import enabled
from screen import arrays, exact, write
from tutorial_code.miniengine.cache import make_cache
from tutorial_code.miniengine.context import ForwardContext
from tutorial_code.miniengine.engine.llm_engine import LLMEngine


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--workload', required=True, type=Path)
    a = p.parse_args()
    a.out.mkdir()
    files = list((ROOT / 'tutorial_code/miniengine').rglob('*.py')) + list(Path(__file__).parent.glob('*.py'))
    pins = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    cases = [(2, t) for t in (1, 7, 8, 9, 13, 14, 16, 31, 32, 33, 63, 64, 65, 129, 257)]
    cases += [(b, 19) for b in (3, 7, 15, 25, 31)] + [(8, 512), (32, 128)]
    write(a.out / 'plan.json', {'pid': os.getpid(), 'source_sha256': pins, 'cases_B_T': cases,
                              'gate': 'Every full-model logit and per-layer cache array must be bit exact; no tolerance.'})
    w = json.loads(a.workload.read_text())
    engine = LLMEngine(w['model_dir'], cache_limit_gb=2)
    model = engine.model
    checks = []
    for B, T in cases:
        ids = mx.array([[w['prompt_ids'][(i + row * 3) % 19] for i in range(T)] for row in range(B)])
        left, right = [make_cache(model) for _ in range(B)], [make_cache(model) for _ in range(B)]
        ref = mx.concatenate([model(ids[r], ForwardContext('prefill', left[r]), last_only=True) for r in range(B)])
        mx.eval(ref, [arrays(c) for c in left])
        with enabled():
            cand = model(ids, ForwardContext('prefill_batch', list(map(list, zip(*right))), prefill_shape=(B, T)), last_only=True)
        logits_exact = exact(ref, cand)
        caches_exact = all(exact(x, y) for lc, rc in zip(left, right) for x, y in zip(arrays(lc), arrays(rc)))
        checks.append({'B': B, 'T': T, 'logits_exact': logits_exact, 'cache_exact': caches_exact})
        write(a.out / 'checks.json', checks)
        if not (logits_exact and caches_exact):
            write(a.out / 'complete.json', {'status': 'REJECT_EXACT', 'first_failure': checks[-1]})
            print('REJECT', checks[-1], flush=True)
            return
        print('PASS', B, T, flush=True)
        del left, right, ref, cand
        gc.collect()
        mx.clear_cache()
    assert all(hashlib.sha256((ROOT / f).read_bytes()).hexdigest() == h for f, h in pins.items())
    write(a.out / 'complete.json', {'status': 'PASS', 'cases': len(checks), 'peak_gib': mx.get_peak_memory() / 1024 ** 3})


if __name__ == '__main__':
    main()
