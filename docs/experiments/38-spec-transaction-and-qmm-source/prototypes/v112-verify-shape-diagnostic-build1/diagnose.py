"""Root-only V112 same-input operator diagnostic; default CPU inert, no timing."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import traceback
from op_audit import Hooks, summarize
sys.dont_write_bytecode = True
D = Path(__file__).resolve().parent
P = D.parent
OLD = P/'v108-dflash2-transaction-build1'
ROOT = P/'candidate-v108-dflash2-transaction-build1-root'
sys.path.insert(0, str(OLD))
from runtime_support import identity as old_identity, signature, runtime, memory, write, provenance


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity():
    pins = old_identity()
    extra = json.loads((D/'source-pins.json').read_text())
    for path, digest in extra.items():
        assert sha(path) == digest, ('V112 source changed', path)
    assert all(extra.get(path) == digest for path, digest in pins.items())
    return extra


def compare(mx, np, actual, reference):
    assert actual.shape == reference.shape and actual.dtype == reference.dtype
    mx.eval(actual, reference)
    bf16 = actual.dtype == mx.bfloat16
    unsigned = mx.uint16 if bf16 or actual.dtype == mx.float16 else mx.uint32
    assert actual.dtype in (mx.bfloat16, mx.float16, mx.float32)
    a = np.ascontiguousarray(np.asarray(actual.view(unsigned)))
    b = np.ascontiguousarray(np.asarray(reference.view(unsigned)))
    mask = a != b;different = int(np.count_nonzero(mask))
    af = np.asarray(actual.astype(mx.float32));bf = np.asarray(reference.astype(mx.float32))
    assert np.isfinite(af).all() and np.isfinite(bf).all()
    first = None
    if different:
        idx = tuple(int(i) for i in np.argwhere(mask)[0])
        first = {'index': list(idx), 'batch_bits': int(a[idx]), 'M1_bits': int(b[idx]),
                 'batch_value': float(af[idx]), 'M1_value': float(bf[idx])}
    return {'shape': list(actual.shape), 'dtype': str(actual.dtype), 'strict_equal': not different,
            'different_elements': different, 'total_elements': a.size, 'first_difference': first,
            'max_abs_diagnostic_only': float(np.max(np.abs(af-bf))),
            'batch_sha256': hashlib.sha256(memoryview(a).cast('B')).hexdigest(),
            'M1_sha256': hashlib.sha256(memoryview(b).cast('B')).hexdigest()}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    mode = ap.add_mutually_exclusive_group();mode.add_argument('--inert', action='store_true');mode.add_argument('--execute-gpu', action='store_true')
    ap.add_argument('--out', type=Path);args = ap.parse_args(argv)
    pins = identity();block = json.loads((D/'actual-block.json').read_text())
    if not args.execute_gpu:
        assert not any(n == 'mlx' or n.startswith(('mlx.', 'numpy')) for n in sys.modules)
        print(json.dumps({'status': 'CPU_READY_V112_DIAGNOSTIC_GPU_UNRUN', 'pins': len(pins),
                          'model_loads': 0, 'saved_actual_draft_inputs': block['drafts'], 'future_gold_used': False,
                          'candidate_changed': False, 'performance_prepared': False}));return 0
    assert args.out and args.out.is_absolute();args.out.mkdir(exist_ok=False)
    r = {'status': 'STARTED', 'GPU_used': False, 'model_loads': 0, 'pins': pins, 'timing_run': False,
         'future_gold_used': False, 'candidate_changed': False, 'adopted': False, 'comparisons': [],
         'source_block': block, 'draft_model_loaded': False, 'diagnostic_differences_are_not_toleranced': True}
    save = lambda: write(args.out/'receipt.json', r)
    save();hooks = None
    try:
        import mlx.core as mx
        import mlx.nn as nn
        import numpy as np
        from mlx.utils import tree_flatten
        r['runtime'] = runtime(mx);mx.set_memory_limit(34<<30);mx.set_cache_limit(2<<30)
        assert not any(n.startswith('tutorial_code.miniengine') for n in sys.modules)
        sys.path.insert(0, str(ROOT))
        from tutorial_code.miniengine.engine.llm_engine import LLMEngine
        from tutorial_code.miniengine.cache import AttnCache, DecodeBatch, make_cache
        from tutorial_code.miniengine.context import ForwardContext
        from tutorial_code.miniengine.model import layers
        from tutorial_code.miniengine.model import gdn as gdn_module
        from tutorial_code.miniengine.spec.transaction import VerifyTransaction
        protocol = json.loads((OLD/'protocol.json').read_text())
        AttnCache.step = DecodeBatch.step = 64
        engine = LLMEngine(protocol['target_model_dir'], max_seqs=1, max_len=64, max_batched_tokens=512)
        model = engine.model;r.update(GPU_used=True, model_loads=1)
        gdns = [l.linear_attn for l in model.layers if not l.is_attn]
        assert len(gdns) == 48
        for gdn in gdns:gdn.set_gates_compile(True);gdn.set_conv_norm_reuse_cache(True)
        parameters = list(tree_flatten(model.parameters()));assert len(parameters) == 1349
        expected_linear = {name for name, op in model.named_modules() if type(op) is layers.Linear}
        expected_embeddings = {name for name, op in model.named_modules() if type(op) is layers.Embedding}
        r['model_operator_inventory'] = {'Linear': sorted(expected_linear), 'Embedding': sorted(expected_embeddings),
                                         'embedding_as_linear_may_be_zero_for_untied_target': True}
        identities = [(name, id(value)) for name, value in parameters]
        def parameter_signatures():
            assert identities == [(name, id(value)) for name, value in tree_flatten(model.parameters())]
            return {name: signature(mx, np, value) for name, value in parameters}
        before = parameter_signatures();r['parameters_before'] = before;r['parameter_identity_before'] = identities
        r['actual_imported_sources_before_token'] = provenance(pins)
        write(args.out/'model-ready.json', {'pid': __import__('os').getpid(), 'model_loads': 1, 'token_executed': False, 'draft_loaded': False})
        print(json.dumps({'event': 'MODEL_READY', 'pid': __import__('os').getpid()}), flush=True);save()
        caches = make_cache(model)
        # The only prefill inputs are prompt IDs. Anchor is generated, not fed from saved gold.
        logits, features = model(mx.array(block['prompt_ids']), ForwardContext('prefill', caches), last_only=True, capture=(5,19,33,47,61))
        anchor_array = mx.argmax(logits, axis=-1);mx.eval(anchor_array, features)
        anchor = int(anchor_array.item());r['live_anchor'] = anchor
        assert anchor == block['assert_live_anchor']
        initial = []
        for cache in caches:
            attention = hasattr(cache, 'keys');values = (cache.keys, cache.values) if attention else (cache.conv_state, cache.S)
            initial.append({'kind': 'attention' if attention else 'gdn', 'offset': cache.offset,
                            'values': [signature(mx, np, v) for v in values]})
        r['initial_cache_matches_actual108_all128'] = initial == block['initial_cache']
        assert r['initial_cache_matches_actual108_all128'] and all(c.offset == block['assert_start'] for c in caches)
        hooks = Hooks(mx, nn, model, layers, gdn_module);hooks.T = block['verify_inputs'];hooks.install()
        with VerifyTransaction(caches, block['verify_inputs']) as transaction:
            hooks.active = True
            try:
                # Only these actual saved DFlash proposals enter verify; no posterior/future gold enters it.
                logits, new_features = model(mx.array([anchor]+block['drafts']), ForwardContext('prefill', caches), capture=(5,19,33,47,61))
            finally:
                hooks.active = False
            posterior = mx.argmax(logits, axis=-1)
            # Complete ORIGINAL graph before building any alternative M1 operation.
            mx.eval(posterior, new_features, *transaction.eval_arrays())
            r['observed_posterior'] = posterior.tolist()
            r['original_posterior_reproduced'] = r['observed_posterior'] == block['assert_posterior']
            assert r['original_posterior_reproduced'], 'diagnostic did not reproduce saved V108 trajectory'
            records = hooks.records;r['captured_operation_count'] = len(records);save()
            actual_linear = [rec['name'] for rec in records if rec['kind'] == 'Linear']
            assert set(actual_linear) == expected_linear and len(actual_linear) == len(expected_linear)
            r['all_target_linear_operators_observed_once'] = True
            assert sum(rec['kind'] == 'GDN.gates' for rec in records) == 48
            assert sum(rec['kind'] == 'attention.SDPA' for rec in records) == 16
            first_qkv = next(rec for rec in records if rec['name'] == 'layers.0.linear_attn.in_proj_qkv')
            x = first_qkv['inputs'][0]
            raw = np.ascontiguousarray(np.asarray(x.view(mx.uint16)), dtype='<u2').tobytes()
            (args.out/'first-layer-projection-input.bf16').write_bytes(raw)
            r['first_layer_projection_input'] = {**signature(mx,np,x), 'path': str(args.out/'first-layer-projection-input.bf16'), 'encoding': 'little-endian BF16 raw bits', 'sha256': hashlib.sha256(raw).hexdigest()}
            for rec in records:
                inputs_before = [signature(mx,np,a) for a in rec['inputs']]
                outputs = rec['output'] if isinstance(rec['output'], tuple) else (rec['output'],)
                actual_before = [signature(mx,np,a) for a in outputs]
                for variant, value in rec['build']():
                    refs = value if isinstance(value, tuple) else (value,)
                    assert len(refs) == len(outputs)
                    for component, (actual, reference) in enumerate(zip(outputs, refs)):
                        row = {'order': rec['order'], 'kind': rec['kind'], 'name': rec['name'], 'variant': variant,
                               'component': component, 'metadata': rec['metadata'], 'input_signatures': inputs_before,
                               **compare(mx,np,actual,reference)}
                        r['comparisons'].append(row)
                assert inputs_before == [signature(mx,np,a) for a in rec['inputs']], 'diagnostic mutated actual input'
                assert actual_before == [signature(mx,np,a) for a in outputs], 'diagnostic mutated actual output'
                if rec['order'] % 16 == 0:save()
            r['summary'] = summarize(r['comparisons'])
            transaction.commit(0)
        hooks.close();hooks = None
        restored = []
        for cache in caches:
            attention = hasattr(cache,'keys');values = (cache.keys,cache.values) if attention else (cache.conv_state,cache.S)
            restored.append({'kind': 'attention' if attention else 'gdn', 'offset': cache.offset,
                             'values': [signature(mx,np,v) for v in values]})
        assert restored == initial;r['keep0_restored_all128'] = True
        after = parameter_signatures();assert after == before
        r['target1349_identity_readonly'] = True;r['actual_imported_sources_final'] = provenance(pins)
        r.update(status='DIAGNOSTIC_COMPLETED_V112_SAME_INPUT_BATCH_VS_M1_NO_ADOPTION', memory_final=memory(mx),
                 source_identity_final=True, old96_ID_gate_reinterpreted=False,
                 conclusion_limit='local operator differences under the original batch trajectory; not an ablation proving full96-ID repair, no observed PSO/dispatch claim')
        identity();return 0
    except BaseException as error:
        r.update(status='FAIL_STOP', error=repr(error), traceback=traceback.format_exc());raise
    finally:
        if hooks is not None:hooks.close()
        save()


if __name__ == '__main__':
    raise SystemExit(main())
