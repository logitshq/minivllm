"""E012 modular selector/draft checks and timings; no target verify claim."""
import argparse
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
from candidate import ORIGINAL, enabled, select
from tutorial_code.miniengine.cache import make_cache
from tutorial_code.miniengine.context import ForwardContext
from tutorial_code.miniengine.engine.llm_engine import LLMEngine
from tutorial_code.miniengine.model.gdn import _m5_ultra_decode_identity
from tutorial_code.miniengine.spec.dflash2 import load_drafter


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def file_hash(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def bits_equal(a, b):
    assert len(a) == len(b)
    checks = [mx.array_equal(x.view(mx.uint16 if x.dtype == mx.bfloat16 else mx.uint32),
                            y.view(mx.uint16 if y.dtype == mx.bfloat16 else mx.uint32)) for x, y in zip(a, b)]
    mx.eval(checks)
    return [bool(x.item()) for x in checks]


def timed(fn):
    mx.synchronize()
    start = time.perf_counter_ns()
    fn()
    mx.synchronize()
    return (time.perf_counter_ns() - start) / 1e6


def paired(fn):
    first = {v: timed(lambda: fn(v)) for v in ('reference', 'candidate')}
    for _ in range(2):
        for v in ('reference', 'candidate'):
            timed(lambda: fn(v))
    rows = []
    for block in range(3):
        for v in ('reference', 'candidate', 'candidate', 'reference'):
            rows.append({'block': block, 'variant': v, 'ms': timed(lambda: fn(v))})
    med = {v: statistics.median(r['ms'] for r in rows if r['variant'] == v) for v in ('reference', 'candidate')}
    return {'first_timed_ms': first, 'samples': rows, 'median_ms': med, 'speedup': med['reference'] / med['candidate']}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--workload', required=True, type=Path)
    p.add_argument('--drafter', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    a = p.parse_args()
    a.out.mkdir()
    checks, timings = [], []
    try:
        pins = {str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest()
                for f in sorted((ROOT / 'tutorial_code/miniengine').rglob('*.py')) + sorted(Path(__file__).parent.glob('*.py'))}
        prompts = {'math': 'Compute 17 * 23 and explain briefly.', 'code': 'Write a Python function that reverses a list.', 'chinese': '请解释为什么月亮会有阴晴圆缺。'}
        write(a.out / 'plan.json', {'experiment': 'E012', 'baseline_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              'source_sha256': pins, 'drafter': str(a.drafter), 'drafter_revision': a.drafter.name,
              'drafter_config_sha256': hashlib.sha256((a.drafter / 'config.json').read_bytes()).hexdigest(),
              'drafter_weights_sha256': file_hash(a.drafter / 'model.safetensors'),
              'workload_sha256': file_hash(a.workload),
              'blocks': [3, 5, 8], 'prompts': prompts, 'synthetic_lengths': [1, 2, 4, 7], 'seeds': [11, 29], 'scales': [0, .125, 8],
              'correctness': 'Actual original selector IDs, instrumented reference scores, and candidate scores byte-equal; full draft IDs and all draft KV exact',
              'timing': 'Two warm rounds, three ABBA blocks, first post-qualification timed calls also retained',
              'advance': 'All exact checks, geometric mean full-draft speedup >=1.05 across all 9 contexts/blocks, every case >=0.97; then separate integration qualification',
              'scope': 'Selector and draft modules only. Does not verify or repair target GDN rollback, does not establish generation/HTTP speedup',
              'device': mx.device_info(), 'mlx': importlib.metadata.version('mlx'),
              'environment': {k: v for k, v in os.environ.items() if k.startswith(('MLX_', 'METAL_'))}})
        assert _m5_ultra_decode_identity()
        mx.set_cache_limit(2 * 1024**3)
        draft = load_drafter(str(a.drafter))
        def check_selector(label, h, logits, anchor):
            original = ORIGINAL(draft, h, logits, anchor)
            before, old_scores = select(draft, h, logits, anchor, device=False, trace=True)
            after, new_scores = select(draft, h, logits, anchor, device=True, trace=True)
            exact = bits_equal(old_scores, new_scores)
            row = {'case': label, 'original_ids': original, 'instrumented_reference_ids': before, 'candidate_ids': after, 'scores_bit_equal': exact}
            checks.append(row)
            write(a.out / 'selector-checks.json', checks)
            assert original == before == after and all(exact), row
        for length in (1, 2, 4, 7):
            for seed in (11, 29):
                for scale in (0, .125, 8):
                    mx.random.seed(seed)
                    h = (mx.random.normal((length, 10240)) * scale).astype(mx.bfloat16)[:, ::2]
                    logits = mx.random.normal((length, draft.cfg.vocab_size)) * scale
                    mx.eval(h, logits)
                    check_selector(f'synthetic-L{length}-seed{seed}-scale{scale}', h, logits, 79852)
            print('PASS synthetic length', length, flush=True)
        w = json.loads(a.workload.read_text())
        engine = LLMEngine(w['model_dir'], max_seqs=1, max_len=2048, cache_limit_gb=2)
        model = engine.model
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(w['model_dir'], local_files_only=True)
        full_checks = []
        for name, text in prompts.items():
            prompt = tok.apply_chat_template([{'role': 'user', 'content': text}], tokenize=True, add_generation_prompt=True, enable_thinking=False, return_dict=False)
            caches = make_cache(model)
            logits, features = model(mx.array(prompt), ForwardContext('prefill', caches), last_only=True, capture=tuple(draft.cfg.target_layer_ids))
            mx.eval(logits, features)
            anchor = int(mx.argmax(logits[0]).item())
            # Seed only already-computed past context; then freeze it for every arm.
            cut = len(prompt) - 4
            draft.reset()
            draft.draft(features[:cut], prompt[cut], cut, model.embed_tokens, model.lm_head_fn, 3)
            seed = [(l.self_attn.ctx_k, l.self_attn.ctx_v, l.self_attn.ctx_len) for l in draft.layers]
            mx.eval([x for k, v, _ in seed for x in (k, v)])
            def restore():
                for layer, (k, v, n) in zip(draft.layers, seed):
                    layer.self_attn.ctx_k, layer.self_attn.ctx_v, layer.self_attn.ctx_len = k, v, n
            def kv():
                return [x for l in draft.layers for x in (l.self_attn.ctx_k, l.self_attn.ctx_v)]
            for block in (3, 5, 8):
                inputs = []
                def run(v, observe=None):
                    restore()
                    with enabled(v == 'candidate', observe):
                        return draft.draft(features[cut:], anchor, len(prompt), model.embed_tokens, model.lm_head_fn, block)
                old = run('reference', lambda *values: inputs.append(values))
                old_kv = kv(); mx.eval(old_kv)
                new = run('candidate')
                new_kv = kv(); mx.eval(new_kv)
                exact = bits_equal(old_kv, new_kv)
                row = {'prompt': name, 'prompt_ids': prompt, 'block': block, 'old_ids': old, 'new_ids': new, 'draft_kv_bit_equal': exact}
                full_checks.append(row)
                write(a.out / 'full-draft-checks.json', full_checks)
                assert old == new and all(exact)
                h, output, saved_anchor = inputs[0]
                check_selector(f'real-{name}-block{block}', h, output, saved_anchor)
                local = paired(lambda v: ORIGINAL(draft, h, output, saved_anchor) if v == 'reference' else select(draft, h, output, saved_anchor))
                full = paired(run)
                timings.append({'prompt': name, 'block': block, 'selector': local, 'full_draft': full})
                write(a.out / 'timing.json', timings)
                print('PASS real', name, block, 'selector', local['speedup'], 'draft', full['speedup'], flush=True)
            draft.reset()
        assert all(hashlib.sha256((ROOT / f).read_bytes()).hexdigest() == sha for f, sha in pins.items())
        speedups = [r['full_draft']['speedup'] for r in timings]
        gain = statistics.geometric_mean(speedups)
        advance = gain >= 1.05 and min(speedups) >= .97
        write(a.out / 'complete.json', {'status': 'PASS', 'selector_cases': len(checks), 'full_draft_cases': len(full_checks),
              'full_draft_geomean_speedup': gain, 'advance': advance, 'peak_active_gib': mx.get_peak_memory() / 1024**3,
              'decision': 'Independent draft-module integration qualification next' if advance else 'Do not integrate: insufficient full-draft gain',
              'scope': 'No target verification, rollback, generation or HTTP speedup claimed'})
        print('COMPLETE advance', advance, 'geomean', gain, flush=True)
    except BaseException:
        write(a.out / 'failure.json', {'status': 'STOP', 'error': traceback.format_exc(), 'completed_checks': len(checks), 'timing_configs': len(timings), 'all_raw_preserved': True})
        raise


if __name__ == '__main__':
    main()
