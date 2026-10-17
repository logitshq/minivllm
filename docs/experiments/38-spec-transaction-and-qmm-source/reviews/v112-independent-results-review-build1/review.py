"""Independent saved-evidence audit. Standard library only; no live probes or GPU.

Run normally to verify the sealed review, or --seal once while authoring it.
Original fixtures/sources remain at their recorded paths; no tensor is copied.
"""
import argparse
import ast
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import sys

HERE = Path(__file__).resolve().parent
P = HERE.parent
D = P / 'v112-verify-shape-diagnostic-build1'
RUN = P / 'v23-root-control-build1/diagnostic-v112-attempt1'
STATUS = 'DIAGNOSTIC_COMPLETED_V112_SAME_INPUT_BATCH_VS_M1_NO_ADOPTION'
RECEIPT_SHA = '09c7f55cfbd46a803b4bd014eb24f694573f62de48e2a3fdee3568ec1910992a'


def load(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def signature_valid(s):
    size = {'mlx.core.bfloat16': 2, 'mlx.core.float16': 2,
            'mlx.core.float32': 4, 'mlx.core.uint32': 4, 'mlx.core.uint8': 1}[s['dtype']]
    assert s['bytes'] == math.prod(s['shape']) * size
    assert re.fullmatch('[0-9a-f]{64}', s['sha256'])


def independent_summary(rows):
    counts, firsts, geometries = {}, {}, {}
    for r in rows:
        key = r['kind'] + ':' + r['variant']
        c = counts.setdefault(key, dict(calls=0, strict_equal=0, strict_different=0, different_elements=0))
        c['calls'] += 1
        c['strict_equal'] += int(r['strict_equal'])
        c['strict_different'] += int(not r['strict_equal'])
        c['different_elements'] += r['different_elements']
        detail = {k: r[k] for k in ('order', 'name', 'component', 'first_difference')}
        if not r['strict_equal']:
            firsts.setdefault(key, detail)
        if r['kind'] in ('Linear', 'Embedding.as_linear'):
            m = r['metadata']
            q = m['quantization']
            gkey = f"{r['kind']}:{q['mode'] if q else 'unquantized'}:K{m['input_features']}:N{m['output_features']}"
            g = geometries.setdefault(gkey, dict(calls=0, strict_equal=0, strict_different=0, first_difference=None))
            g['calls'] += 1
            g['strict_equal'] += int(r['strict_equal'])
            g['strict_different'] += int(not r['strict_equal'])
            if not r['strict_equal'] and g['first_difference'] is None:
                g['first_difference'] = detail
    return dict(by_kind=counts, first_difference_by_kind=firsts,
                linear_by_quantization_geometry=geometries,
                first_differing_captured_operation=next((r for r in rows if not r['strict_equal']), None))


def resource_audit(raw):
    for name in ('memory_pressure', 'pressure_level', 'swap', 'vm_stat'):
        assert raw[name]['returncode'] == 0
    assert int(raw['pressure_level']['text']) == raw['pressure_level_value'] == 1
    swap = float(re.search(r'used = ([\d.]+)M', raw['swap']['text'])[1])
    assert swap == raw['swap_used_mb']
    for k, v in raw['vm_counters'].items():
        assert int(re.search(r'^' + re.escape(k) + r':\s*(\d+)\.', raw['vm_stat']['text'], re.M)[1]) == v
    pm = raw['power_metadata']
    assert pm['raw']['returncode'] == 0 and pm['source'] == 'AC Power'
    assert "Now drawing from 'AC Power'" in pm['raw']['stdout']
    th = raw['processinfo_thermal_metadata']
    assert th['raw']['returncode'] == 0 and int(th['raw']['stdout']) == th['code'] == 0
    assert raw['process_inventory_ok'] and raw['only_owned_known_inference_processes']
    return dict(swap_used_mb=swap, Swapouts=raw['vm_counters']['Swapouts'])


def audit():
    receipt_path = RUN / 'child-output/receipt.json'
    assert digest(receipt_path) == RECEIPT_SHA
    r = load(receipt_path)
    assert r['status'] == STATUS and r['GPU_used'] and r['model_loads'] == 1
    assert not any(r[k] for k in ('timing_run', 'future_gold_used', 'candidate_changed', 'adopted', 'draft_model_loaded', 'old96_ID_gate_reinterpreted'))
    assert r['diagnostic_differences_are_not_toleranced']
    pins = load(D / 'source-pins.json')
    assert r['pins'] == pins and len(pins) == 76
    for path, h in pins.items():
        assert digest(path) == h, path
    oldpins = load(P / 'v108-dflash2-transaction-build1/source-pins.json')
    assert len(oldpins) == 61 and all(pins.get(p) == h for p, h in oldpins.items())
    for row in load(D / 'source-map.json'):
        assert digest(row['path']) == row['sha256']
    ready = load(D / 'CPU-READY.json')
    assert ready['pins'] == 76 and ready['preserved_all_original108_pins'] == 61
    for name, h in ready['identity'].items():
        assert digest(D / name) == h
    for key in ('actual_imported_sources_before_token', 'actual_imported_sources_final'):
        for name, path in r[key].items():
            assert name.startswith('tutorial_code.miniengine') and path in pins
            assert Path(path).is_relative_to(P / 'candidate-v108-dflash2-transaction-build1-root')
    assert len(r['actual_imported_sources_before_token']) == 23
    assert len(r['actual_imported_sources_final']) == 26
    assert all(r['actual_imported_sources_final'].get(k) == v for k, v in r['actual_imported_sources_before_token'].items())

    block = load(D / 'actual-block.json')
    assert block == r['source_block'] and digest(block['source_receipt']) == block['source_receipt_sha256']
    old = load(block['source_receipt'])
    assert old['status'] == 'FAIL_STOP' and old['A_old96_IDs_passed']
    tx = old['transactions'][0]
    assert tx['request'] == 'B.base' and tx['start'] == 19 and tx['verify_inputs'] == 5
    assert tx['drafts'] == block['drafts'] == [6297, 220, 95761, 95779]
    assert tx['posterior'] == block['assert_posterior'] == r['observed_posterior'] == [18887, 220, 98435, 100606, 98277]
    assert old['requests'][0]['prompt_ids'] == block['prompt_ids'] and len(block['prompt_ids']) == 19
    assert old['requests'][0]['IDs'][0] == block['assert_live_anchor'] == r['live_anchor'] == 79852
    assert len(tx['prefix_history']) == len(block['initial_cache']) == 64
    original_initial = [{'kind': c['kind'], **h[0]} for c, h in zip(tx['actual'], tx['prefix_history'])]
    assert original_initial == block['initial_cache']
    assert sum(len(c['values']) for c in block['initial_cache']) == 128
    assert all(c['offset'] == block['assert_start'] == 19 for c in block['initial_cache'])
    assert r['initial_cache_matches_actual108_all128'] and r['keep0_restored_all128'] and r['original_posterior_reproduced']

    rows = r['comparisons']
    assert len(rows) == 1138
    orders = defaultdict(list)
    seen = set()
    for x in rows:
        k = (x['order'], x['variant'], x['component'])
        assert k not in seen
        seen.add(k); orders[x['order']].append(x)
        assert x['total_elements'] == math.prod(x['shape'])
        assert type(x['strict_equal']) is bool and x['strict_equal'] == (x['different_elements'] == 0)
        assert 0 <= x['different_elements'] <= x['total_elements']
        assert (x['batch_sha256'] == x['M1_sha256']) == x['strict_equal']
        assert math.isfinite(x['max_abs_diagnostic_only']) and x['max_abs_diagnostic_only'] >= 0
        for s in x['input_signatures']: signature_valid(s)
        if x['strict_equal']:
            assert x['first_difference'] is None and x['max_abs_diagnostic_only'] == 0
        else:
            d = x['first_difference']
            assert x['kind'] == 'Linear' and x['dtype'] == 'mlx.core.bfloat16'
            assert len(d['index']) == len(x['shape']) and all(0 <= i < n for i, n in zip(d['index'], x['shape']))
            for side in ('batch', 'M1'):
                value = struct.unpack('<f', struct.pack('<I', d[side + '_bits'] << 16))[0]
                assert math.isfinite(value) and value == d[side + '_value']
            assert abs(d['batch_value'] - d['M1_value']) <= x['max_abs_diagnostic_only']
    assert set(orders) == set(range(978)) and r['captured_operation_count'] == 978
    for order, rs in orders.items():
        head = rs[0]
        assert all(all(x[k] == head[k] for k in ('kind', 'name', 'metadata', 'input_signatures')) for x in rs)
        expected = 4 if head['kind'] == 'GDN.gates' else 2 if head['kind'] == 'attention.SDPA' else 1
        assert len(rs) == expected
    summary = independent_summary(rows)
    assert summary == r['summary']
    linear = [x for x in rows if x['kind'] == 'Linear']
    assert len(linear) == len({x['name'] for x in linear}) == 497
    assert sorted(x['name'] for x in linear) == r['model_operator_inventory']['Linear']
    assert r['all_target_linear_operators_observed_once']
    assert (sum(x['strict_equal'] for x in linear), sum(not x['strict_equal'] for x in linear), sum(x['different_elements'] for x in linear)) == (146, 351, 1281)
    assert all(x['strict_equal'] for x in rows if x['kind'] != 'Linear')
    first = summary['first_differing_captured_operation']
    assert (first['order'], first['name'], first['max_abs_diagnostic_only']) == (9, 'layers.0.linear_attn.out_proj', 4.76837158203125e-7)

    fs = r['first_layer_projection_input']; signature_valid(fs)
    assert fs['shape'] == [5, 5120] and fs['bytes'] == 51200 and fs['dtype'] == 'mlx.core.bfloat16'
    raw = Path(fs['path']).read_bytes()
    assert len(raw) == 51200 and hashlib.sha256(raw).hexdigest() == fs['sha256']
    assert all((u & 0x7f80) != 0x7f80 for (u,) in struct.iter_unpack('<H', raw))
    qkv = next(x for x in rows if x['name'] == 'layers.0.linear_attn.in_proj_qkv')
    assert {k: fs[k] for k in ('shape', 'dtype', 'bytes', 'sha256')} == qkv['input_signatures'][0]
    assert fs['sha256'] == rows[0]['batch_sha256'] == rows[0]['M1_sha256']
    params = r['parameters_before']
    assert len(params) == len(r['parameter_identity_before']) == 1349
    assert set(params) == {x[0] for x in r['parameter_identity_before']}
    assert len({x[1] for x in r['parameter_identity_before']}) == 1349
    for s in params.values(): signature_valid(s)
    assert r['target1349_identity_readonly'] and r['source_identity_final']

    # Source-backed semantics, never execute or import candidate/GPU code.
    src = (D / 'diagnose.py').read_text(); hsrc = (D / 'op_audit.py').read_text()
    ast.parse(src); ast.parse(hsrc)
    assert src.index('mx.eval(posterior, new_features, *transaction.eval_arrays())') < src.index("rec['build']()")
    assert 'mx.array([anchor]+block[\'drafts\'])' in src and 'anchor = int(anchor_array.item())' in src
    assert "inputs_before == [signature(mx,np,a) for a in rec['inputs']]" in src
    assert 'actual_before == [signature(mx,np,a) for a in outputs]' in src
    assert 'assert restored == initial' in src and 'after = parameter_signatures();assert after == before' in src
    assert 'transaction.commit(0)' in src and 'hooks.close();hooks = None' in src
    htree = ast.parse(hsrc)
    wrappers = [n for n in ast.walk(htree) if isinstance(n, ast.FunctionDef) and n.name in ('wrapped', 'gates', 'unary', 'sdpa', 'rope')]
    assert len(wrappers) == 5
    for f in wrappers:
        assert isinstance(f.body[-1], ast.Return) and isinstance(f.body[-1].value, ast.Name) and f.body[-1].value.id == 'y'
        assert isinstance(f.body[0], ast.Assign) and isinstance(f.body[0].targets[0], ast.Name) and f.body[0].targets[0].id == 'y'
    assert 'x[i:i+1]' in hsrc and 'op.A_log, op.dt_bias' in hsrc
    assert 'k[:, :, :end, :], v[:, :, :end, :]' in hsrc

    resources = load(RUN / 'resources.json')
    assert len(resources) == 11
    audits = []
    base = resources[0]
    for i, derived in enumerate(resources):
        rawr = load(RUN / f'resource-{i:03d}-raw.json')
        inventory = list(rawr['known_inference_processes'])
        if rawr['owned_model_pid'] is not None and not any(x['pid'] == rawr['owned_model_pid'] for x in inventory):
            inventory.append(dict(pid=rawr['owned_model_pid'], owned=True, engine='explicit-owned-numeric-child'))
        expected = {**rawr, 'known_inference_processes': inventory,
                    'known_inference_process_count': len(inventory),
                    'only_owned_known_inference_processes': all(x['owned'] for x in inventory)}
        assert all(derived[k] == v for k, v in expected.items())
        audits.append(resource_audit(rawr))
        if i:
            previous = resources[i - 1]
            assert derived['swap_used_mb_delta'] == rawr['swap_used_mb'] - previous['swap_used_mb']
            assert derived['vm_counter_delta'] == {k: v - previous['vm_counters'][k] for k, v in rawr['vm_counters'].items()}
    assert {x['swap_used_mb'] for x in audits} == {4470.88}
    assert {x['Swapouts'] for x in audits} == {17460761}
    ext = load(RUN / 'external-resource-anypower-00-raw.json')
    extderived = load(RUN / 'external-resource-anypower.json')
    assert len(extderived) == 1 and all(extderived[0][k] == v for k, v in ext.items())
    extres = resource_audit(ext)
    assert extres == dict(swap_used_mb=4462.88, Swapouts=17460761)
    owner = load(RUN / 'owner-receipt.json'); release = load(RUN / 'external-release.json')
    assert (owner['owner_pid'], owner['driver_pid']) == (20374, 20383)
    assert owner['stage'] == 'reaped' and owner['returncode'] == 0 and owner['waited_reaped']
    assert owner['driver_sha256'] == digest(D / 'diagnose.py')
    assert owner['absence']['absent'] and owner['absence']['returncode'] == 1 and not owner['absence']['stdout']
    assert release['status'] == 'PASS_EXTERNAL_RELEASE' and release['GPU_released'] and release['tool_exit'] == 0 and release['result'] == STATUS
    assert {x['pid'] for x in release['strict_absence']} == {20374, 20383}
    assert all(x['returncode'] == 1 and not x['stdout'] and not x['stderr'] for x in release['strict_absence'])
    assert release['release_expected_power'] == 'any' and release['release_observed_power'] == 'AC Power'
    assert not (RUN / 'stderr.log').read_bytes()
    assert load(RUN / 'child-output/model-ready.json') == dict(pid=20383, model_loads=1, token_executed=False, draft_loaded=False)
    return dict(status='PASS_INDEPENDENT_SAVED_EVIDENCE_AUDIT_V112', GPU_used_by_reviewer=False,
                receipt_sha256=RECEIPT_SHA, source_pins_verified=76, original108_pins_preserved=61,
                actual_imports_before=23, actual_imports_final=26, captured_operations=978, comparisons=1138,
                linear_calls=497, linear_exact=146, linear_different=351, linear_differing_elements=1281,
                non_linear_comparisons_exact=641, by_kind=summary['by_kind'], linear_geometry=summary['linear_by_quantization_geometry'],
                earliest_difference=first, first_layer_input=fs,
                initial_cache_signatures=128, initial_matches_original108=True, keep0_restored_all128=True,
                target_parameter_count=1349, target_readonly='saved final assertion; before signatures/identities retained, after signatures not separately retained',
                deferred_original_outputs='source wrappers return y; alternatives built after original eval; saved actual input/output readonly assertions completed',
                live_anchor=79852, saved_actual_draft_proposals=block['drafts'], reproduced_posterior=r['observed_posterior'],
                future_gold_injected=False, owner=dict(owner_pid=20374, child_pid=20383, returncode=0, reaped=True, saved_strict_absence=True),
                resources=dict(owner_raw_samples=11, external_raw_samples=1, power='AC Power', thermal=0, pressure=1,
                               owner_swap_mb=4470.88, external_swap_mb=4462.88, Swapouts=17460761,
                               inventory='raw benchmark matcher empty; 9 during rows add explicit owned child PID20383 in derived inventory, exactly as owner source lines20-24',
                               policy='numerical any-power; saved release only, not live GPU lease probing or timing qualification'),
                limits=['1138 output raw pairs are not retained: audit recomputes counts/summary and checks saved hashes, not absent tensor payloads',
                        '978 is Python captured-call completion order, not actual dispatch count or execution order',
                        'same-input local batch versus M1 comparisons keep original batch outputs upstream; no counterfactual full trajectory',
                        'no observed PSO/reduction implementation, full96-ID repair, performance, or adoption claim'])


def expected_sources():
    paths = {str(p.resolve()) for p in D.iterdir() if p.is_file()}
    paths |= {str(p.resolve()) for p in RUN.rglob('*') if p.is_file()}
    paths |= set(load(D / 'source-pins.json'))
    paths.add(load(D / 'actual-block.json')['source_receipt'])
    paths.add(str(P / 'v23-root-control-build1/v112_operator_diagnostic_owner_build1.py'))
    return {p: dict(bytes=Path(p).stat().st_size, sha256=digest(p),
                    role='original tensor fixture identity; payload not copied' if p.endswith('.bf16') else 'source or saved evidence identity')
            for p in sorted(paths)}


def main():
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('--seal', action='store_true'); args = ap.parse_args()
    assert not any(n == 'mlx' or n.startswith(('mlx.', 'numpy')) for n in sys.modules)
    result = audit()
    seals = expected_sources()
    if args.seal:
        write(HERE / 'raw-source-seals.json', seals)
        write(HERE / 'receipt.json', result)
    else:
        assert seals == load(HERE / 'raw-source-seals.json')
        assert result == load(HERE / 'receipt.json')
        manifest = load(HERE / 'manifest.json')
        for name, row in manifest['files'].items():
            f = HERE / name
            assert f.stat().st_size == row['bytes'] and digest(f) == row['sha256'], name
    print(json.dumps(dict(status=result['status'], originals_verified=len(seals), captured=978, comparisons=1138,
                          Linear='497 = 146 exact + 351 different', GPU_used_by_reviewer=False)))


if __name__ == '__main__':
    main()
