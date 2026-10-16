"""E010 engine lifecycle qualification, with cache byte signatures per step."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import os

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import mlx.core as mx
import numpy as np
from candidate import enabled
from tutorial_code.miniengine.engine.llm_engine import LLMEngine
from tutorial_code.miniengine.sampling_params import SamplingParams


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def digest(a):
    return hashlib.sha256(np.asarray(a.view(mx.uint16 if a.dtype == mx.bfloat16 else mx.uint32)).tobytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--workload', required=True, type=Path)
    a = ap.parse_args()
    a.out.mkdir()
    files = list((ROOT / 'tutorial_code/miniengine').rglob('*.py')) + list(Path(__file__).parent.glob('*.py'))
    pins = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    write(a.out / 'plan.json', {'pid': os.getpid(), 'source_sha256': pins,
          'cases': ['C64 with 512-token budget, partial prefill, staggered lengths and first-token stop',
                    'Mixed 19/7/130/601 input lengths, partial prefill, late arrivals'],
          'checks': 'Every delivered ID, live batch/state and staging cache byte signature, empty cleanup; graph/fuse decode enabled on both arms'})
    w = json.loads(a.workload.read_text())
    engine = LLMEngine(w['model_dir'], max_seqs=64, max_len=2048, max_batched_tokens=512,
                       cache_limit_gb=2, compile_decode=True, fuse_decode=True)
    for layer in engine.model.layers:
        if not layer.is_attn:
            layer.linear_attn.set_gates_compile(True)
            layer.linear_attn.set_norm_reuse(False)

    def run(active, case):
        def add(rid, T, max_tokens, stop=False):
            prompt = [w['prompt_ids'][i % 19] for i in range(T)]
            engine.add_request(prompt, SamplingParams(temperature=0, max_tokens=max_tokens,
                               stop_token_ids=[79852] if stop else []), req_id=rid)
        if case == 'C64':
            for i in range(64):
                add(str(i), 19, 4 + i % 4, i == 0)
        else:
            for i, T in enumerate([19] * 12 + [7] * 4 + [130] * 2 + [601]):
                add(str(i), T, 4 + i % 3)
        trace = []
        step = 0
        with enabled(active):
            while engine.has_work():
                if case == 'mixed' and step == 2:
                    for i in range(4):
                        add('late' + str(i), 19, 4 + i % 2)
                outputs = engine.step()
                batch = engine.runner.batch
                staging = {}
                for rid, caches in engine.runner.staging.items():
                    staging[rid] = [{'offset': c.offset, 'hashes': [digest(v) for v in
                                     ((c.keys, c.values) if hasattr(c, 'keys') else (c.conv_state, c.S))]}
                                    for c in caches]
                trace.append({'step': step, 'outputs': outputs, 'batch_ids': list(batch.req_ids),
                              'batch_lengths': list(batch.lens), 'batch_hashes': [digest(v) for v in batch.arrays()],
                              'staging': staging})
                step += 1
        assert not engine.runner.staging and not engine.runner.batch.size and engine.runner._ahead is None and not engine.requests
        return trace
    results = []
    for case in ('C64', 'mixed'):
        ref, cand = run(False, case), run(True, case)
        same = ref == cand
        write(a.out / (case + '.json'), {'reference': ref, 'candidate': cand, 'exact': same})
        assert same, case
        results.append({'case': case, 'steps': len(ref), 'exact': same})
        print('PASS', results[-1], flush=True)
    assert all(hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == h for p, h in pins.items())
    write(a.out / 'complete.json', {'status': 'PASS', 'cases': results, 'peak_gib': mx.get_peak_memory() / 1024 ** 3})


if __name__ == '__main__':
    main()
