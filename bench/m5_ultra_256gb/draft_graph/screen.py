"""E016: explicit-state draft backbone graph, strict numerics before timing."""
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
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import mlx.core as mx
from candidate import BackboneGraph, restore, states
from tutorial_code.miniengine.cache import make_cache
from tutorial_code.miniengine.context import ForwardContext
from tutorial_code.miniengine.engine.llm_engine import LLMEngine
from tutorial_code.miniengine.model.gdn import _m5_ultra_decode_identity
from tutorial_code.miniengine.spec.dflash2 import DFlash2Drafter, DraftLayer, load_drafter


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def file_hash(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            value.update(block)
    return value.hexdigest()


def flatten(cache):
    return [value for pair in cache for value in pair]


def measure(fn):
    mx.synchronize()
    start = time.perf_counter_ns()
    result = fn()
    mx.eval(result)
    mx.synchronize()
    return result, (time.perf_counter_ns() - start) / 1e6


def paired(run):
    first = {arm: measure(lambda: run(arm))[1] for arm in ('reference', 'candidate')}
    warm = []
    for repeat in range(2):
        for arm in ('reference', 'candidate'):
            warm.append({'repeat': repeat, 'arm': arm, 'ms': measure(lambda: run(arm))[1]})
    samples = []
    for repeat in range(3):
        for arm in ('reference', 'candidate', 'candidate', 'reference'):
            samples.append({'repeat': repeat, 'arm': arm, 'ms': measure(lambda: run(arm))[1]})
    medians = {arm: statistics.median(row['ms'] for row in samples if row['arm'] == arm)
               for arm in ('reference', 'candidate')}
    return {'first_post_qualification_ms': first, 'warm': warm, 'samples': samples,
            'median_ms': medians, 'speedup': medians['reference'] / medians['candidate']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workload', type=Path, required=True)
    parser.add_argument('--drafter', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir()
    checks, cold, timing = [], [], []
    graph = None
    try:
        workload = json.loads(args.workload.read_text())
        target = Path(workload['model_dir'])
        sources = sorted((ROOT / 'tutorial_code/miniengine').rglob('*.py')) + sorted(Path(__file__).parent.glob('*.py'))
        pins = {str(path.relative_to(ROOT)): file_hash(path) for path in sources}
        prompts = {'math': 'Compute 17 * 23 and explain briefly.',
                   'code': 'Write a Python function that reverses a list.',
                   'chinese': '请解释为什么月亮会有阴晴圆缺。'}
        plan = {'experiment': 'E016', 'head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                'source_sha256': pins, 'workload_sha256': file_hash(args.workload),
                'target_dir': str(target), 'target_config_sha256': file_hash(target / 'config.json'),
                'target_weights_sha256': {path.name: file_hash(path) for path in sorted(target.glob('*.safetensors'))},
                'drafter_dir': str(args.drafter), 'drafter_config_sha256': file_hash(args.drafter / 'config.json'),
                'drafter_weights_sha256': file_hash(args.drafter / 'model.safetensors'),
                'prompts': prompts, 'blocks': [3, 5, 8], 'new_context_tokens': 4,
                'selector': 'Production E012 ON for both arms',
                'correctness': 'Original backbone hidden/normalized hidden/logits/all ten KV arrays bit-equal and IDs equal. Same shape with +7 position probe; two real histories interleaved for three growing-KV steps.',
                'timing': 'All first graph calls retained with trace counts; then two warm rounds and three ABBA blocks per fixed-shape full draft. Parameter identity checks and cache restoration included.',
                'advance': 'Every strict check; full-draft geometric mean >=1.10x and every case >=0.97x. Advance only to shape-growth qualification, never production from this screen.',
                'limitations': 'Frozen layer parameters; replacement rejected. No bucket padding, no target verification, rollback, acceptance, generation or HTTP claim.',
                'device': mx.device_info(), 'mlx': importlib.metadata.version('mlx'),
                'environment': {key: value for key, value in os.environ.items() if key.startswith(('MLX_', 'METAL_'))}}
        write(args.out / 'plan.json', plan)
        assert _m5_ultra_decode_identity()
        mx.set_cache_limit(2 * 1024**3)
        engine = LLMEngine(str(target), max_seqs=1, max_len=2048, cache_limit_gb=2)
        model = engine.model
        draft = load_drafter(str(args.drafter), device_selection=True)
        graph = BackboneGraph(draft)
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(target), local_files_only=True)
        original_select = DFlash2Drafter._select
        original_layer = DraftLayer.__call__

        def reference(features, anchor, start, block, capture):
            if not capture:
                return draft.draft(features, anchor, start, model.embed_tokens, model.lm_head_fn, block)
            values, backbone = [], []
            def observe(self, hidden, logits, current_anchor):
                values.extend([hidden, logits])
                return original_select(self, hidden, logits, current_anchor)
            def observe_layer(self, *arguments):
                result = original_layer(self, *arguments)
                if self is draft.layers[-1]:
                    backbone.append(result)
                return result
            with patch.object(DFlash2Drafter, '_select', observe), patch.object(DraftLayer, '__call__', observe_layer):
                ids = draft.draft(features, anchor, start, model.embed_tokens, model.lm_head_fn, block)
            return ids, values[0], values[1], states(draft), backbone[0]

        def check(label, features, anchor, start, block, ref_seed, candidate_seed):
            restore(draft, ref_seed)
            before = reference(features, anchor, start, block, True)
            restore(draft, candidate_seed)
            traces_before = len(graph.traces)
            after, elapsed = measure(lambda: graph.draft(features, anchor, start, model.embed_tokens, model.lm_head_fn, block))
            cold.append({'case': label, 'ms': elapsed, 'traces_before': traces_before, 'traces_after': len(graph.traces),
                         'new_shapes': graph.traces[traces_before:], 'note': 'First candidate call for this case; may be cached shape'})
            write(args.out / 'first-calls.json', cold)
            left = [before[4], before[1], before[2]] + flatten(before[3])
            right = [after[4], after[1], after[2]] + flatten(after[3])
            exact = []
            for old, new in zip(left, right, strict=True):
                exact.append(old.shape == new.shape and old.dtype == new.dtype and
                             bool(mx.array_equal(old.view(mx.uint8), new.view(mx.uint8)).item()))
            row = {'case': label, 'start': start, 'block': block, 'reference_ids': before[0], 'candidate_ids': after[0],
                   'array_order': ['backbone_hidden', 'normalized_hidden', 'logits'] + [f'layer_{i}_{kind}' for i in range(len(draft.layers)) for kind in ('K', 'V')],
                   'bit_equal': exact}
            checks.append(row)
            write(args.out / 'checks.json', checks)
            if before[0] != after[0] or not all(exact):
                mx.save_safetensors(str(args.out / 'failed-arrays.safetensors'),
                                    {f'{arm}_{i}': array for arm, arrays in (('reference', left), ('candidate', right)) for i, array in enumerate(arrays)})
                raise AssertionError(row)
            return before[3], after[3]

        histories = {}
        for name, text in prompts.items():
            prompt = tokenizer.apply_chat_template([{'role': 'user', 'content': text}], tokenize=True,
                        add_generation_prompt=True, enable_thinking=False, return_dict=False)
            logits, features = model(mx.array(prompt), ForwardContext('prefill', make_cache(model)),
                                    last_only=True, capture=tuple(draft.cfg.target_layer_ids))
            mx.eval(logits, features)
            anchor = int(mx.argmax(logits[0]).item())
            histories[name] = (prompt, features, anchor)
            cut = len(prompt) - 4
            draft.reset()
            draft.draft(features[:cut], prompt[cut], cut, model.embed_tokens, model.lm_head_fn, 3)
            seed = states(draft)
            mx.eval(seed)
            for block in (3, 5, 8):
                label = f'{name}-B{block}'
                check(label, features[cut:], anchor, len(prompt), block, seed, seed)
                check(label + '-position-plus7', features[cut:], anchor, len(prompt) + 7, block, seed, seed)
                def run(arm):
                    restore(draft, seed)
                    if arm == 'reference':
                        return reference(features[cut:], anchor, len(prompt), block, False)
                    # Retain identical public work including IDs materialization.
                    return graph.draft(features[cut:], anchor, len(prompt), model.embed_tokens, model.lm_head_fn, block)[0]
                timing.append({'prompt': name, 'prompt_ids': prompt, 'block': block, 'full_draft': paired(run)})
                write(args.out / 'timing.json', timing)
                print('PASS', label, timing[-1]['full_draft']['speedup'], flush=True)

        # Equal shapes, distinct real historical features, interleaved requests.
        # Both state chains grow independently; every new shape cost is retained.
        chains = {}
        seed_length = min(len(histories[name][0]) for name in ('math', 'code')) - 7
        assert seed_length >= 4
        for name in ('math', 'code'):
            prompt, features, _ = histories[name]
            draft.reset()
            draft.draft(features[:seed_length], prompt[seed_length], seed_length, model.embed_tokens, model.lm_head_fn, 3)
            seed = states(draft)
            mx.eval(seed)
            chains[name] = (seed, seed)
        seed_equal = [bool(mx.array_equal(left.view(mx.uint8), right.view(mx.uint8)).item())
                      for left, right in zip(flatten(chains['math'][0]), flatten(chains['code'][0]), strict=True)]
        write(args.out / 'interleaved-seeds.json', {'seed_length': seed_length, 'seed_kv_bit_equal': seed_equal})
        assert not all(seed_equal), 'Interleaved request seeds must actually differ'
        for step in range(3):
            for name in ('math', 'code'):
                prompt, features, _ = histories[name]
                low, high = seed_length + step * 2, seed_length + 2 + step * 2
                chains[name] = check(f'interleaved-{name}-step{step}', features[low:high], prompt[high], high, 3, *chains[name])
        graph.assert_frozen()
        assert all(file_hash(ROOT / filename) == sha for filename, sha in pins.items())
        speedups = [row['full_draft']['speedup'] for row in timing]
        mean = statistics.geometric_mean(speedups)
        advance = mean >= 1.10 and min(speedups) >= .97
        write(args.out / 'complete.json', {'status': 'PASS', 'checks': len(checks), 'timing_cases': len(timing),
              'full_draft_geomean_speedup': mean, 'minimum_speedup': min(speedups), 'advance': advance,
              'trace_count': len(graph.traces), 'peak_active_gib': mx.get_peak_memory() / 1024**3,
              'decision': 'Consider bounded shape-growth qualification' if advance else 'Reject fixed-shape graph as insufficient gain',
              'scope': 'Fixed-shape draft upper bound only; no long-generation or HTTP qualification'})
    except BaseException:
        write(args.out / 'failure.json', {'status': 'STOP', 'error': traceback.format_exc(),
              'checks_completed': len(checks), 'timing_cases_completed': len(timing),
              'trace_history': graph.traces if graph is not None else [], 'all_raw_preserved': True})
        raise


if __name__ == '__main__':
    main()
