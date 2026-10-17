"""E014 actual public switches: four-combination numerics and owned lifecycle."""
import argparse
import ast
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workload', required=True, type=Path)
    parser.add_argument('--drafter', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--cache-gib', required=True, choices=(2, 16), type=int)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    # Imported only in the sole GPU-owner child, after --help and argument checks.
    import mlx.core as mx
    import tutorial_code.miniengine.spec.dflash2 as spec
    from tutorial_code.miniengine.engine.llm_engine import LLMEngine
    from tutorial_code.miniengine.context import ForwardContext
    from tutorial_code.miniengine.cache import make_cache
    from tutorial_code.miniengine.model.gdn import _m5_ultra_decode_identity
    from screen import (write, file_hash, bit_equal, repeated_state, bounded_state,
                        snapshot, restore, observe_selector, paired, logical_bytes,
                        CHECK_LENGTHS, TIMING_LENGTHS, BLOCK_CHAIN, PROMPT)
    checks, timings = [], []
    try:
        workload = json.loads(args.workload.read_text())
        target = Path(workload['model_dir'])
        source_paths = sorted((ROOT / 'tutorial_code/miniengine').rglob('*.py')) + sorted(Path(__file__).parent.glob('*.py'))
        pins = {str(p.relative_to(ROOT)): file_hash(p) for p in source_paths}
        # Freeze the prior implementation's actual function, not a retyped oracle.
        old_path = ROOT / '.artifacts/m5-ultra-256gb/draft-window-v2-cache16/frozen-source/tutorial_code/miniengine/spec/dflash2.py'
        old_plan = json.loads((ROOT / '.artifacts/m5-ultra-256gb/draft-window-v2-cache16/plan.json').read_text())
        assert file_hash(old_path) == old_plan['source_sha256']['tutorial_code/miniengine/spec/dflash2.py']
        cls = next(n for n in ast.parse(old_path.read_text()).body if isinstance(n, ast.ClassDef) and n.name == 'DraftAttention')
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '__call__')
        namespace = dict(vars(spec))
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(old_path), 'exec'), namespace)
        old_call, current_call = namespace['__call__'], spec.DraftAttention.__call__
        @contextmanager
        def old_attention():
            spec.DraftAttention.__call__ = old_call
            try:
                yield
            finally:
                spec.DraftAttention.__call__ = current_call
        model_files = sorted(target.glob('*.safetensors')) + [target / 'config.json', args.drafter / 'config.json', args.drafter / 'model.safetensors']
        plan = {'experiment': 'E014-integrated', 'pid': __import__('os').getpid(),
                'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                'source_sha256': pins, 'old_attention_source': str(old_path), 'old_attention_sha256': file_hash(old_path),
                'workload_sha256': file_hash(args.workload), 'target': str(target), 'drafter': str(args.drafter),
                'cache_gib': args.cache_gib, 'predeclared_process_matrix': [16, 16, 16, 2],
                'numeric_reference': 'Frozen pre-integration actual DraftAttention.__call__, selector OFF; same original projections/weights',
                'numeric_variants': ['selectorOFF/kvOFF', 'selectorOFF/kvON', 'selectorON/kvOFF', 'selectorON/kvON'],
                'timing': 'Actual production selector ON, KV OFF vs ON; both own teardown; 2 warm rounds, 3 ABBA; four independent loaded drafter weights resident, all qualification KV reset before first sample',
                'gate': 'Every fresh16GiB run long N32768/131072 each >=1.10, short N32/2048 each >=.97, all strict checks; fixed2GiB compatibility separately',
                'scope': 'Actual module, synthetic long KV replay. No verify, rollback, generation or HTTP qualification',
                'model_hash_status': 'pending'}
        write(args.out / 'plan.json', plan)
        plan['model_sha256'] = {str(p): file_hash(p) for p in model_files}
        plan['model_hash_status'] = 'complete'
        plan['device'] = mx.device_info()
        write(args.out / 'plan.json', plan)
        assert _m5_ultra_decode_identity()
        mx.set_cache_limit(args.cache_gib * 1024**3)
        drafts = {(s, k): spec.load_drafter(str(args.drafter), device_selection=s, bounded_kv=k)
                  for s in (False, True) for k in (False, True)}
        for (selector, bounded), draft in drafts.items():
            assert draft._device_selection_enabled is selector
            assert all(layer.self_attn._bounded_kv_enabled is bounded for layer in draft.layers)
        engine = LLMEngine(str(target), max_seqs=1, max_len=2048, cache_limit_gb=args.cache_gib)
        model = engine.model
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(target), local_files_only=True)
        prompt = tokenizer.apply_chat_template([{'role': 'user', 'content': PROMPT}], tokenize=True,
                    add_generation_prompt=True, enable_thinking=False, return_dict=False)
        logits, features = model(mx.array(prompt), ForwardContext('prefill', make_cache(model)),
                    last_only=True, capture=tuple(drafts[False, False].cfg.target_layer_ids))
        mx.eval(logits, features)
        anchor = int(mx.argmax(logits[0]).item())
        cut = len(prompt) - 4
        with old_attention():
            drafts[False, False].draft(features[:cut], prompt[cut], cut, model.embed_tokens, model.lm_head_fn, 3)
        seed = snapshot(drafts[False, False])
        write(args.out / 'seed.json', {'prompt_ids': prompt, 'real_seed_rows': cut, 'anchor': anchor,
                                     'four_loaded_drafters': True, 'replay_scope': plan['scope']})

        def invoke(draft, state, incoming, position, block, old=False):
            restore(draft, state)
            captured = []
            with observe_selector(captured):
                if old:
                    with old_attention():
                        ids = draft.draft(incoming, anchor, position, model.embed_tokens, model.lm_head_fn, block)
                else:
                    ids = draft.draft(incoming, anchor, position, model.embed_tokens, model.lm_head_fn, block)
            return ids, snapshot(draft), captured

        def check(label, states, incoming, position, block):
            ids, oracle, observed = invoke(drafts[False, False], states['old'], incoming, position, block, True)
            next_states = {'old': oracle}
            for key, draft in drafts.items():
                got_ids, after, captured = invoke(draft, states[key], incoming, position, block)
                exact = [bit_equal(a, b) for a, b in zip(observed[0][:2], captured[0][:2])]
                suffix, visible, logical, bounds = [], [], [], []
                for (ok, ov, on), (nk, nv, nn) in zip(oracle, after):
                    for ox, nx in ((ok, nk), (ov, nv)):
                        suffix.append(bit_equal(ox[:, :, ox.shape[2] - nx.shape[2]:], nx))
                        visible.append(bit_equal(ox[:, :, max(0, ox.shape[2] - 2048 + block):],
                                                 nx[:, :, max(0, nx.shape[2] - 2048 + block):]))
                    logical.append(on == nn)
                    bounds.append(nk.shape[2] <= 4096 if key[1] else nk.shape[2] == on)
                row = {'case': label, 'selector': key[0], 'bounded': key[1], 'block': block,
                       'position': position, 'new_rows': None if incoming is None else incoming.shape[0],
                       'oracle_ids': ids, 'actual_ids': got_ids, 'hidden_logits_exact': exact,
                       'kv_suffix_exact': suffix, 'visible_exact': visible, 'logical_exact': logical,
                       'physical_bound': bounds}
                checks.append(row)
                write(args.out / 'checks.json', checks)
                assert len(observed) == len(captured) == 1 and ids == got_ids and all(exact + suffix + visible + logical + bounds), row
                next_states[key] = after
            return next_states

        def initial_states(state):
            return {'old': state, **{key: bounded_state(state, 2048) if key[1] else state for key in drafts}}

        for block in (3, 5, 8):
            cold = [(None, None, 0) for _ in seed]
            check('cold', {'old': cold, **{key: cold for key in drafts}}, features[:4], 4, block)
        for length in CHECK_LENGTHS:
            initial = repeated_state(seed, length)
            for block in (3, 5, 8):
                states, position = initial_states(initial), length
                for step, rows in enumerate((4, None, 0, 1, 7)):
                    incoming = None if rows is None else features[:rows]
                    position += rows or 0
                    states = check(f'N{length}-step{step}', states, incoming, position, block)
                if length == 31:
                    for rows in (2049, 4097):
                        incoming = mx.tile(features, ((rows + len(prompt) - 1) // len(prompt), 1))[:rows]
                        check(f'input{rows}', initial_states(initial), incoming, length + rows, block)
        for length in (4095, 4096, 4097):
            states, position = initial_states(repeated_state(seed, length)), length
            for step, (block, rows) in enumerate(BLOCK_CHAIN):
                position += rows or 0
                states = check(f'variable-N{length}-step{step}', states, None if rows is None else features[:rows], position, block)
        print('PASS four public switch combinations', len(checks), flush=True)
        states = incoming = None
        for length in TIMING_LENGTHS:
            initial = repeated_state(seed, length)
            cropped = bounded_state(initial, 2048)
            for block in (3, 8):
                check(f'timing-N{length}', initial_states(initial), features[-4:], length + 4, block)
                # No inactive/oracle drafter retains a full-history qualification
                # product during timing. Only declared seeds and weights remain.
                for drafter in drafts.values():
                    drafter.reset()
                mx.synchronize()
                mapping = {'reference': ((True, False), initial), 'candidate': ((True, True), cropped)}
                def sample(variant):
                    key, prepared = mapping[variant]
                    draft, after, ids, times = drafts[key], None, None, {}
                    try:
                        begin = time.perf_counter_ns()
                        restore(draft, prepared)
                        mx.synchronize()
                        times['prepare_ms'] = (time.perf_counter_ns() - begin) / 1e6
                        begin = time.perf_counter_ns()
                        ids = draft.draft(features[-4:], anchor, length + 4, model.embed_tokens, model.lm_head_fn, block)
                        after = snapshot(draft)
                        mx.synchronize()
                        times['draft_ms'] = (time.perf_counter_ns() - begin) / 1e6
                    finally:
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
                       'seed_logical_bytes': {'reference': logical_bytes(initial), 'candidate': logical_bytes(cropped)}}
                timings.append(row)
                def persist(record):
                    row['full_draft'] = record
                    write(args.out / 'timing.json', timings)
                record = paired(sample, persist)
                print('TIMING', length, block, record['speedup'], flush=True)
        assert all(file_hash(ROOT / p) == h for p, h in pins.items())
        assert len(checks) == 736 and len(timings) == 10
        longs = [r['full_draft']['speedup']['owned_lifecycle_ms'] for r in timings if r['history_rows'] in (32768, 131072)]
        shorts = [r['full_draft']['speedup']['owned_lifecycle_ms'] for r in timings if r['history_rows'] in (32, 2048)]
        write(args.out / 'complete.json', {'status': 'PASS', 'numeric_checks': len(checks), 'timing_cases': len(timings),
              'cache_gib': args.cache_gib, 'gate': min(longs) >= 1.10 and min(shorts) >= .97,
              'long_min': min(longs), 'short_min': min(shorts), 'scope': plan['scope']})
    except BaseException:
        (args.out / 'failure.json').write_text(json.dumps({'traceback': traceback.format_exc(),
            'completed_checks': len(checks), 'timing_cases_started': len(timings)}, indent=2) + '\n')
        raise


if __name__ == '__main__':
    main()
