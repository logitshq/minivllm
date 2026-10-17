"""E014 bounded draft KV: exact cache-state replay and full-draft timing.

Long caches repeat short real target-derived seed KV. This is NOT long-context
generation. Parent owns the single GPU queue and the suspended-service lease.
"""
import argparse
from contextlib import contextmanager
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
from candidate import enabled
from tutorial_code.miniengine.cache import make_cache
from tutorial_code.miniengine.context import ForwardContext
from tutorial_code.miniengine.engine.llm_engine import LLMEngine
from tutorial_code.miniengine.model.gdn import _m5_ultra_decode_identity
from tutorial_code.miniengine.spec.dflash2 import DFlash2Drafter, load_drafter

CHECK_LENGTHS = [0, 31, 2040, 2047, 2048, 2049, 4095, 4096, 4097, 32768]
TIMING_LENGTHS = [32, 2048, 8192, 32768, 131072]
BLOCK_CHAIN = [(8, 1), (3, None), (5, 0), (8, 1), (3, 7)]
PROMPT = 'Compute 17 * 23 and explain briefly.'


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def file_hash(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def bit_equal(a, b):
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    equal = mx.array_equal(a.view(mx.uint8), b.view(mx.uint8))
    mx.eval(equal)
    return bool(equal.item())


def snapshot(draft):
    state = [(l.self_attn.ctx_k, l.self_attn.ctx_v, l.self_attn.ctx_len) for l in draft.layers]
    mx.eval([x for k, v, _ in state for x in (k, v) if x is not None])
    return state


def restore(draft, state):
    for layer, (k, v, n) in zip(draft.layers, state):
        layer.self_attn.ctx_k, layer.self_attn.ctx_v, layer.self_attn.ctx_len = k, v, n


def repeated_state(seed, length):
    result = []
    for k, v, _ in seed:
        copies = (length + k.shape[2] - 1) // k.shape[2]
        arrays = [mx.zeros((1, x.shape[1], 0, x.shape[3]), dtype=x.dtype) if not length
                  else mx.contiguous(mx.tile(x, (1, 1, copies, 1))[:, :, :length]) for x in (k, v)]
        result.append((*arrays, length))
    mx.eval([x for k, v, _ in result for x in (k, v)])
    return result


def bounded_state(state, window):
    result = [(mx.contiguous(k[:, :, -window:]), mx.contiguous(v[:, :, -window:]), n)
              if k.shape[2] > 2 * window else (k, v, n) for k, v, n in state]
    mx.eval([x for k, v, _ in result for x in (k, v)])
    return result


def logical_bytes(state):
    return sum(x.nbytes for k, v, _ in state for x in (k, v) if x is not None)


@contextmanager
def observe_selector(destination):
    original = DFlash2Drafter._select
    def capture(self, h, logits, anchor):
        destination.append((h, logits, anchor))
        return original(self, h, logits, anchor)
    DFlash2Drafter._select = capture
    try:
        yield
    finally:
        DFlash2Drafter._select = original


def paired(sample, persist):
    record = {'first_post_qualification': {}, 'warmup_samples': [], 'samples': []}
    for variant in ('reference', 'candidate'):
        record['first_post_qualification'][variant] = sample(variant)
        persist(record)
    for warm in range(2):
        for variant in ('reference', 'candidate'):
            record['warmup_samples'].append({'warm': warm, 'variant': variant, **sample(variant)})
            persist(record)
    for block in range(3):
        for arm, variant in enumerate(('reference', 'candidate', 'candidate', 'reference')):
            record['samples'].append({'block': block, 'arm': arm, 'variant': variant, **sample(variant)})
            persist(record)
    record['median_ms'] = {metric: {v: statistics.median(r[metric] for r in record['samples'] if r['variant'] == v)
                                   for v in ('reference', 'candidate')}
                           for metric in ('prepare_ms', 'draft_ms', 'teardown_ms', 'owned_lifecycle_ms')}
    record['speedup'] = {metric: med['reference'] / med['candidate'] for metric, med in record['median_ms'].items()
                         if metric in ('draft_ms', 'owned_lifecycle_ms')}
    persist(record)
    return record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workload', required=True, type=Path)
    p.add_argument('--drafter', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--cache-gib', type=int, choices=(2, 16), default=2)
    a = p.parse_args()
    a.out.mkdir()  # Never overwrite a previous attempt, including a failed attempt.
    checks, timings = [], []
    try:
        w = json.loads(a.workload.read_text())
        target_dir = Path(w['model_dir'])
        source_files = sorted((ROOT / 'tutorial_code/miniengine').rglob('*.py')) + sorted(Path(__file__).parent.glob('*.py'))
        pins = {str(f.relative_to(ROOT)): file_hash(f) for f in source_files}
        plan = {
            'experiment': 'E014-v2', 'baseline_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            'source_sha256': pins, 'workload_sha256': file_hash(a.workload), 'target_model': str(target_dir),
            'target_revision': target_dir.name, 'drafter': str(a.drafter), 'drafter_revision': a.drafter.name,
            'model_hash_status': 'pending; filled before any model load',
            'prompt': PROMPT, 'check_lengths': CHECK_LENGTHS, 'check_blocks': [3, 5, 8],
            'chained_input_rows': [4, None, 0, 1, 7], 'additional_projection_input_boundary': {'seed_length': 31, 'input_rows': [2049, 4097]},
            'variable_block_chain': {'lengths': [4095, 4096, 4097], 'block_and_input_rows': BLOCK_CHAIN},
            'residency_policy': 'When physical rows exceed 2W=4096, compact to W=2048; never crop projection inputs; logical ctx_len independent',
            'cold_initialization': 'None KV plus four real target feature rows, for all three blocks',
            'timing_lengths': TIMING_LENGTHS, 'timing_blocks': [3, 8], 'timing_input_rows': 4,
            'selector': 'E012 device selection ON in both arms; unchanged actual original selector',
            'correctness': 'Byte-exact selector input hidden/logits, identical draft IDs, all ten retained KV suffixes and logical ctx_len; chained states independently advance',
            'timing': 'Separate prepare (restore+sync), draft (complete draft+persistent KV materialization+sync), teardown (restore same-arm seed, release own result, sync). No observer. Report every phase; main metric draft+teardown owned lifecycle; 2 warm rounds, 3 ABBA blocks, every sample persisted',
            'cache_limit_gib': a.cache_gib, 'predeclared_process_matrix_cache_gib': [2, 16],
            'advance': 'All numeric checks; owned lifecycle every N32768/N131072 point >=1.10x and every N32/N2048 point >=0.97x; N8192 descriptive. Adoption requires the predeclared fresh 16GiB process gate, with 2GiB compatibility reported independently. No selection between pools.',
            'scope': 'Synthetic cache-state replay from repeated real short-history KV, NOT actual long-context generation, target verification, rollback or HTTP. Memory is tensor logical bytes only, not allocator evidence.',
            'mode_switch': 'Scoped benchmark only; each arm restores its own complete state. Cannot resume full-history mode from cropped state.',
            'environment': {k: v for k, v in os.environ.items() if k.startswith(('MLX_', 'METAL_'))},
            'mlx': importlib.metadata.version('mlx'),
        }
        write(a.out / 'plan.json', plan)
        model_files = sorted(target_dir.glob('*.safetensors')) + [target_dir / 'config.json', a.drafter / 'config.json', a.drafter / 'model.safetensors']
        assert len(list(target_dir.glob('*.safetensors'))) > 0
        plan['model_sha256'] = {str(f): file_hash(f) for f in model_files}
        plan['model_hash_status'] = 'complete'
        plan['device'] = mx.device_info()
        write(a.out / 'plan.json', plan)
        assert _m5_ultra_decode_identity(), 'This preregistered screen is M5 Ultra only'
        mx.set_cache_limit(a.cache_gib * 1024**3)
        draft = load_drafter(str(a.drafter), device_selection=True)
        assert draft.cfg.sliding_window == 2048 and draft._device_selection_enabled
        engine = LLMEngine(str(target_dir), max_seqs=1, max_len=2048, cache_limit_gb=a.cache_gib)
        model = engine.model
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(str(target_dir), local_files_only=True)
        prompt = tok.apply_chat_template([{'role': 'user', 'content': PROMPT}], tokenize=True, add_generation_prompt=True, enable_thinking=False, return_dict=False)
        logits, features = model(mx.array(prompt), ForwardContext('prefill', make_cache(model)), last_only=True, capture=tuple(draft.cfg.target_layer_ids))
        mx.eval(logits, features)
        anchor = int(mx.argmax(logits[0]).item())
        cut = len(prompt) - 4
        draft.draft(features[:cut], prompt[cut], cut, model.embed_tokens, model.lm_head_fn, 3)
        seed = snapshot(draft)
        write(a.out / 'seed.json', {'prompt_ids': prompt, 'real_seed_rows': cut, 'anchor': anchor,
              'real_feature_rows': int(features.shape[0]), 'seed_kv_shapes': [list(x.shape) for k, v, _ in seed for x in (k, v)],
              'long_cache_construction': 'Tile each already-projected real seed K/V along history; no new long target prefix or position-correct long RoPE history claimed'})

        def invoke(variant, state, incoming, start, block, observe=False):
            restore(draft, state)
            captured = []
            with enabled(variant == 'candidate'):
                if observe:
                    with observe_selector(captured):
                        ids = draft.draft(incoming, anchor, start, model.embed_tokens, model.lm_head_fn, block)
                else:
                    ids = draft.draft(incoming, anchor, start, model.embed_tokens, model.lm_head_fn, block)
            return ids, snapshot(draft), captured

        def check(label, old_state, new_state, incoming, start, block):
            old_ids, old_after, old_input = invoke('reference', old_state, incoming, start, block, True)
            new_ids, new_after, new_input = invoke('candidate', new_state, incoming, start, block, True)
            assert len(old_input) == len(new_input) == 1
            inputs_exact = [bit_equal(x, y) for x, y in zip(old_input[0][:2], new_input[0][:2])]
            suffix_exact, visible_exact = [], []
            for (ok, ov, on), (nk, nv, nn) in zip(old_after, new_after):
                for ox, nx in ((ok, nk), (ov, nv)):
                    suffix_exact.append(bit_equal(ox[:, :, ox.shape[2] - nx.shape[2]:], nx))
                    old_lo = max(0, ox.shape[2] - draft.cfg.sliding_window + block)
                    new_lo = max(0, nx.shape[2] - draft.cfg.sliding_window + block)
                    visible_exact.append(bit_equal(ox[:, :, old_lo:], nx[:, :, new_lo:]))
            expected_len = old_state[0][2] + (0 if incoming is None else incoming.shape[0])
            logical = [on == nn == expected_len for (_, _, on), (_, _, nn) in zip(old_after, new_after)]
            bounded = [k.shape[2] <= 2 * draft.cfg.sliding_window for k, _, _ in new_after]
            row = {'case': label, 'block': block, 'start': start, 'new_rows': None if incoming is None else incoming.shape[0],
                   'reference_ids': old_ids, 'candidate_ids': new_ids, 'selector_hidden_logits_bit_equal': inputs_exact,
                   'retained_suffix_bit_equal': suffix_exact, 'attention_visible_suffix_bit_equal': visible_exact,
                   'logical_length_equal': logical, 'logical_ctx_len': expected_len, 'physical_at_most_2W': bounded,
                   'reference_physical_rows': [k.shape[2] for k, _, _ in old_after],
                   'candidate_physical_rows': [k.shape[2] for k, _, _ in new_after],
                   'tensor_logical_bytes': {'reference': logical_bytes(old_after), 'candidate': logical_bytes(new_after)}}
            checks.append(row)
            write(a.out / 'checks.json', checks)
            assert old_ids == new_ids and old_input[0][2] == new_input[0][2] and all(inputs_exact + suffix_exact + visible_exact + logical + bounded), row
            return old_after, new_after

        cold = [(None, None, 0) for _ in draft.layers]
        for block in (3, 5, 8):
            check('cold-None-plus-real-context', cold, cold, features[:4], 4, block)
        for length in CHECK_LENGTHS:
            initial = repeated_state(seed, length)
            cropped = bounded_state(initial, draft.cfg.sliding_window)
            for block in (3, 5, 8):
                old_state, new_state = initial, cropped
                position = length
                for step, n in enumerate((4, None, 0, 1, 7)):
                    incoming = None if n is None else features[:n]
                    position += 0 if n is None else n
                    old_state, new_state = check(f'replay-N{length}-step{step}', old_state, new_state, incoming, position, block)
                if length == 31:
                    for rows in (2049, 4097):
                        big = mx.tile(features, ((rows + features.shape[0] - 1) // features.shape[0], 1))[:rows]
                        check(f'projection-input-{rows}', initial, cropped, big, length + rows, block)
            print('PASS chained cache replay', length, flush=True)

        for length in (4095, 4096, 4097):
            old_state = repeated_state(seed, length)
            new_state = bounded_state(old_state, draft.cfg.sliding_window)
            position = length
            for step, (block, n) in enumerate(BLOCK_CHAIN):
                incoming = None if n is None else features[:n]
                position += 0 if n is None else n
                old_state, new_state = check(f'variable-block-N{length}-step{step}', old_state, new_state, incoming, position, block)
            print('PASS variable-block boundary', length, flush=True)

        for length in TIMING_LENGTHS:
            initial = repeated_state(seed, length)
            cropped = bounded_state(initial, draft.cfg.sliding_window)
            for block in (3, 8):
                check(f'timing-replay-N{length}', initial, cropped, features[-4:], length + 4, block)
                states = {'reference': initial, 'candidate': cropped}
                incoming = features[-4:]
                def sample(variant):
                    prepared = states[variant]
                    after = ids = None
                    times = {}
                    try:
                        begin = time.perf_counter_ns()
                        restore(draft, prepared)
                        mx.synchronize()
                        times['prepare_ms'] = (time.perf_counter_ns() - begin) / 1e6
                        with enabled(variant == 'candidate'):
                            begin = time.perf_counter_ns()
                            ids = draft.draft(incoming, anchor, length + 4, model.embed_tokens, model.lm_head_fn, block)
                            after = snapshot(draft)
                            mx.synchronize()
                            times['draft_ms'] = (time.perf_counter_ns() - begin) / 1e6
                    finally:
                        # Destruction of THIS arm's output is charged to this arm.
                        # No result tensor escapes sample; next arm sees only seed.
                        begin = time.perf_counter_ns()
                        restore(draft, prepared)
                        after = ids = None
                        mx.synchronize()
                        times['teardown_ms'] = (time.perf_counter_ns() - begin) / 1e6
                        for layer, (k, v, n) in zip(draft.layers, prepared):
                            assert layer.self_attn.ctx_k is k and layer.self_attn.ctx_v is v and layer.self_attn.ctx_len == n
                    times['owned_lifecycle_ms'] = times['draft_ms'] + times['teardown_ms']
                    times['own_products_released_before_next_sample'] = True
                    return times
                row = {'history_rows': length, 'block': block,
                       'seed_tensor_logical_bytes': {v: logical_bytes(s) for v, s in states.items()}}
                timings.append(row)
                def persist(record):
                    row['full_draft'] = record
                    write(a.out / 'timing.json', timings)
                result = paired(sample, persist)
                print('TIMING', length, block, result['speedup'], flush=True)
        assert all(file_hash(ROOT / f) == sha for f, sha in pins.items()), 'Source changed during experiment'
        long = [r['full_draft']['speedup']['owned_lifecycle_ms'] for r in timings if r['history_rows'] in (32768, 131072)]
        short = [r['full_draft']['speedup']['owned_lifecycle_ms'] for r in timings if r['history_rows'] in (32, 2048)]
        advance = min(long) >= 1.10 and min(short) >= .97
        write(a.out / 'complete.json', {'status': 'PASS', 'numeric_cases': len(checks), 'timing_cases': len(timings),
              'cache_limit_gib': a.cache_gib, 'main_metric': 'owned_lifecycle_ms',
              'advance': advance, 'strong_reference_adoption_gate_passed': advance if a.cache_gib == 16 else None,
              'two_gib_compatibility_gate_passed': advance if a.cache_gib == 2 else None,
              'long_min_speedup': min(long), 'short_min_speedup': min(short),
              'scope': plan['scope'], 'decision': 'This pool gate passed; combine both predeclared fresh runs before adoption' if advance else 'This pool gate failed; preserve all points without selecting another pool post hoc'})
        print('COMPLETE advance', advance, flush=True)
    except BaseException:
        write(a.out / 'failure.json', {'status': 'STOP', 'error': traceback.format_exc(), 'completed_checks': len(checks),
              'timing_configs_started': len(timings), 'all_raw_preserved': True})
        raise


if __name__ == '__main__':
    main()
