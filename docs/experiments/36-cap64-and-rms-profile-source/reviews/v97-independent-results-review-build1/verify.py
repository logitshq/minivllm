"""Independent saved-evidence audit: stdlib only, never imports executed workers."""
import ast
import hashlib
import json
from pathlib import Path
import sys

P = Path('/private/tmp/minivllm-engine-optimization-20261014')
D = Path(__file__).resolve().parent
R = P / 'v23-root-control-build1/fullmodel-v97native-attempt1'
S = P / 'v97-vendor-rms-native-diagnostic-build1'
EXPECTED = '6c9621109e8f23e20b88244324b7b104e7672ca67a89342af52c4ebfe0b65947'
refs = {}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    refs[str(path)] = sha(path)
    return json.loads(path.read_text())


def save(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def main():
    x = read(R / 'child-output/receipt.json')
    assert refs[str(R / 'child-output/receipt.json')] == EXPECTED
    assert x['status'] == 'DIAGNOSTIC_COMPLETED_NEW_REFERENCE_STRICT_NO_ADOPTION'
    assert not x['adopted'] and not x['performance_point'] and not x['old_profile_pass'] and x['diagnostic_only']
    pins = read(S / 'source-pins.json')
    assert x['source_pins'] == pins
    for path, expected in pins.items():
        actual = sha(Path(path))
        assert actual == expected, ('source pin', path)
        refs[path] = actual
    assert x['source_pins_verified_final']
    for path in x['actual_imported_sources'].values():
        assert path in pins
    old93 = read(Path(x['preserved_V93_failure_receipt']))
    assert refs[x['preserved_V93_failure_receipt']] == x['preserved_V93_failure_sha256']
    assert old93['status'] == 'FAIL_STOP' and x['V93_original_FAIL_preserved']
    old38 = read(Path(x['old_profile_ID_and_hash_diagnostic_receipt']))
    assert refs[x['old_profile_ID_and_hash_diagnostic_receipt']] == x['old_profile_ID_and_hash_diagnostic_receipt_sha256']
    gold44paths = [Path(k) for k, v in pins.items() if v == x['actual44_gold_sha256']]
    assert len(gold44paths) == 1
    old44 = read(gold44paths[0])
    archive38 = {q['name']: q for q in old38['requests']}
    archive44 = {q['name']: q for q in old44['requests']}
    assert x['fixed_cap'] == 64 and x['canonical_capacity'] == 256
    assert x['same_parameter_identity_count'] == 1349 and x['same_parameter_identity_preserved']
    assert x['native_parameter_pins'] == {'model_parameter_arrays': 1349, 'derived_cachedgamma_arrays': 161, 'total_pins': 1510, 'reference_rawgamma_pinned_into_B': False}
    prep = x['vendor_rms_profile_preparation']
    assert prep['prepared_before_any_token'] and prep['same_parameter_identity']
    assert (prep['plain_norms'], prep['parameter_arrays'], prep['derived_cachedgamma_arrays'], prep['independent_raw_gamma_arrays']) == (161, 1349, 161, 161)
    assert len(prep['all161_gamma_pairs_strict']) == 161
    for q in prep['all161_gamma_pairs_strict']:
        assert q['strict_bytes_equal'] and q['separate_objects'] and q['separate_addresses'] and q['A_B_ranges_disjoint']
        a, b = q['A_raw'], q['B_cached']
        assert a['sha256'] == b['sha256'] and a['finite'] and b['finite']
        assert a['address'] + a['nbytes'] <= b['address'] or b['address'] + b['nbytes'] <= a['address']
        assert a['dtype'] == b['dtype'] == 'mlx.core.bfloat16'
        assert a['shape'] == b['shape'] and a['nbytes'] == b['nbytes']
    capture = x['capture']
    assert len(capture['inputs']) == 131 and len(capture['outputs']) == 132
    kv_labels = {q['label'] for q in capture['outputs'] if q['shape'] == [1, 4, 64, 256]}
    assert len(kv_labels) == 32
    assert all(q['shape'] != [1, 4, 256, 256] for q in capture['inputs'] + capture['outputs'])
    proofs = x['derived_cachedgamma161_native_readonly_proof']
    assert len(proofs) == 161 and len({q['label'] for q in proofs}) == 161
    for q in proofs:
        assert q['full_range_readonly'] and ' readonly:' in q['role']
        rows = [b for b in capture['native']['buffers'] if b['used'] and b['zero_copy']
                and b['original_buffer'] == q['original_buffer']
                and b['original_contents'] <= q['address']
                and q['address'] + q['bytes'] <= b['original_contents'] + b['length']
                and b['role'] == q['role']]
        assert len(rows) == 1
    assert x['A_reference_context_calls'] == 96 and x['A_reference_restore_final']
    norm = x['B_vendor_plain_rms_capture_engagement']
    assert norm['calls'] == norm['distinct_plain_norm_objects'] == 161
    assert norm['all161_prepared_enabled'] and norm['counter_removed_before_replay'] and not norm['GatedRMSNorm_hooked']
    assert len(norm['actual_branch_rows']) == 161 and all(q['actual_branch_predicates_passed'] for q in norm['actual_branch_rows'])
    assert x['B_conv_normreuse_cache_capture_engagement']['calls'] == 48
    prefill_count = logits_count = 0
    for q in x['independent_prefill_qualifications']:
        assert len(q['seed131']) == 131 and q['seed131_strict_bytes_equal']
        assert q['A_B_states_retained_separately'] and q['B_seed_origin'] == 'own B prefill only'
        for seed in q['seed131']:
            assert seed['strict_bytes_equal'] and seed['A_sha256'] == seed['B_sha256'] and seed['separate_array_handles']
            prefill_count += 1
        a, b = q['last_logits']['A'], q['last_logits']['B']
        assert q['last_logits']['strict_bytes_equal'] and a['sha256'] == b['sha256'] and a['finite'] and b['finite']
        assert a['address'] != b['address']
        logits_count += 1
    assert (prefill_count, logits_count) == (393, 3)
    counts = {'steps': 0, 'strict_outputs': 0, 'canonical_KV_tail_records': 0, 'old_equal': 0, 'old_different': 0,
              'reference_input_readonly_steps': 0, 'saved_readonly_array_checks': 0, 'seed_upload_calls': 0}
    ids = {}
    for req_index, q in enumerate(x['requests'], 1):
        assert len(q['steps']) == 31 and len(q['A_ids']) == len(q['B_ids']) == 32
        assert q['A_ids'] == q['B_ids'] and not (set(q['A_ids']) & set(x['eos_ids']))
        assert q['seed_upload_calls'] == q['seed_arrays'] == 131
        counts['seed_upload_calls'] += q['seed_upload_calls']
        assert q['B_seed_source'] == 'own independent B prefill'
        if q['name'] in ('base', 'reset-base'):
            assert not q['changed_seed_inputs_vs_capture']
        else:
            assert any(i >= 3 for i in q['changed_seed_inputs_vs_capture'])
        n = q['native']; f = n['feedback']; icb = f['icb']; pp = f['pingpong']
        assert n['error'] == '' and n['parameter_ranges_registered'] == 1510
        assert f['capacity'] == 64 and f['sealed'] and f['request_seed_uploaded_ranges'] == 131
        assert not f['CPU_full_state_restore_per_step'] and f['resets'] == req_index
        assert f['total_steps'] == 31 * req_index and f['request_steps'] == 31
        assert f['CPU_seed_upload_bytes_total'] == 158138380 * req_index
        assert (pp['requested_count'], pp['eligible_count'], pp['staged_link_count']) == (48, 48, 83)
        assert icb['enabled'] and icb['banks'] == 2 and icb['execution_count'] == 31 * req_index
        assert icb['commands_per_bank'] == n['dispatches'] == 1609
        assert sum(icb['bank_executions']) == icb['execution_count']
        assert len(icb['ICB_records']) == 2
        assert all(p['actual_supports_ICB'] and p['recreated'] for p in icb['PSO_records'])
        for step_index, step in enumerate(q['steps'], 1):
            assert step['step'] == step_index and step['B_fixed_L'] == 64
            assert step['A_exact_L'] == q['prompt_length'] + step_index
            assert step['A_token'] == step['B_token'] == q['A_ids'][step_index]
            assert step['seed_uploads_this_step'] == 0 and step['native_prediction_copied_bytes'] == 4
            assert step['reference_input_digest_readonly']
            counts['reference_input_readonly_steps'] += 1
            for ro in step['originals_readonly'].values():
                assert ro['strict_bytes_equal']
                counts['saved_readonly_array_checks'] += ro['arrays']
            olda = {a['label']: a for a in archive38[q['name']]['steps'][step_index - 1]['outputs']}
            oldb = {a['label']: a for a in archive44[q['name']]['steps'][step_index - 1]['outputs']}
            assert olda == oldb
            assert len(step['outputs']) == 132 and {r['label'] for r in step['outputs']} == set(olda)
            for output in step['outputs']:
                assert output['strict_bytes_equal'] and output['finite']
                assert output['A_sha256'] == output['canonical_B_sha256']
                old = olda[output['label']]
                assert (output['shape'], output['bytes'], output['dtype']) == (old['shape'], old['bytes'], old['dtype'])
                assert output['old_profile_sha256'] == old['A_sha256']
                equal = output['A_sha256'] == old['A_sha256']
                assert equal == output['old_profile_strict_equal']
                counts['old_equal' if equal else 'old_different'] += 1
                if output['label'] in kv_labels:
                    assert output['physical_shape'] == [1, 4, 64, 256] and output['shape'] == [1, 4, 256, 256]
                    assert output['physical_bytes'] == 131072 and output['bytes'] == 524288
                    assert output['physical_tail_zero_checked'] and output['canonical_CPU_zero_fill']
                    counts['canonical_KV_tail_records'] += 1
                else:
                    assert output['shape'] == output['physical_shape'] and output['bytes'] == output['physical_bytes']
                    assert output['physical_A_sha256'] == output['A_sha256']
                counts['strict_outputs'] += 1
            counts['steps'] += 1
        gamma = q['gamma322_readonly']
        assert gamma['arrays'] == 322 and gamma['strict_bytes_equal']
        old_equal = q['A_ids'] == archive38[q['name']]['A_ids']
        assert old_equal == (q['name'] == 'changed-prefill')
        if q['name'] != 'changed-prefill':
            assert q['A_ids'] == x['donor_visible_reference']['ids']
        ids[q['name']] = {'new_A_B_equal': True, 'old38_44_equal': old_equal,
                          'old_aligned_differences': sum(a != b for a, b in zip(q['A_ids'], archive38[q['name']]['A_ids'])),
                          'donor_visible_same_prompt_equal': None if q['name'] == 'changed-prefill' else True}
    assert counts['steps'] == 93 and counts['strict_outputs'] == 12276
    assert counts['canonical_KV_tail_records'] == 2976 and counts['seed_upload_calls'] == 393
    assert (counts['old_equal'], counts['old_different']) == (241, 12035)
    assert x['strict_output_checks'] == 12276 and x['canonical_KV_tail_zero_checks'] == 2976
    assert x['gamma322_readonly_final']['strict_bytes_equal'] and x['gamma322_readonly_final']['arrays'] == 322
    snapshot = x['vendor_rms_profile_final_snapshot']
    assert not snapshot['whole_omlx_semantics'] and not snapshot['old_gold_pass']
    assert snapshot['plain_norms'] == snapshot['enabled_count'] == 161 and snapshot['parameter_count'] == 1349
    warm = read(R / 'child-output/ordinary-warm32-ids-diagnostic.json')
    assert x['ordinary_warm32_diagnostic']['new_B_ids'] == x['requests'][0]['A_ids']
    assert not x['ordinary_warm32_diagnostic']['old_workload']['equal']
    assert x['ordinary_warm32_diagnostic']['saved_donor_visible']['equal']
    assert not x['donor_visible_reference']['source_kind'].startswith('raw')
    donor = Path(x['donor_visible_reference']['path'])
    assert sha(donor) == x['donor_visible_reference']['sha256']; refs[str(donor)] = sha(donor)
    # Source-only inverse audit: all GDN files, including activated helper, remain the current44 bytes.
    c44 = P / 'candidate-v44-conv-normreuse-build1-root/tutorial_code/miniengine'
    c92 = P / 'candidate-v92-vendor-rms-leaf-build1-root/tutorial_code/miniengine'
    unchanged_gdn = ['model/gdn.py', 'kernels/gdn.py', 'kernels/gdn_metal.py', 'kernels/gdn_conv_silu_metal.py', 'kernels/gdn_activated_norm_reuse_cached_metal.py']
    for rel in unchanged_gdn:
        assert (c44 / rel).read_bytes() == (c92 / rel).read_bytes(), rel
    def class_ast(path, name):
        return next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef) and n.name == name)
    assert ast.dump(class_ast(c44 / 'model/layers.py', 'GatedRMSNorm')) == ast.dump(class_ast(c92 / 'model/layers.py', 'GatedRMSNorm'))
    # Pure CPU extraction tests raw BF16 signed-zero preservation and rejects a nonzero physical tail.
    tree = ast.parse((S / 'fullqual.py').read_text())
    canonical = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'canonical_output')
    ns = {'CAP': 64, 'CANONICAL_CAP': 256}
    exec(compile(ast.Module(body=[canonical], type_ignores=[]), '<extracted canonical>', 'exec'), ns)
    physical = b''.join((b'\x00\x80' + bytes(64 * 256 * 2 - 2)) for _ in range(4))
    converted, shape = ns['canonical_output'](physical, [1, 4, 64, 256], 'mlx.core.bfloat16', True, 1)
    assert shape == [1, 4, 256, 256]
    assert all(converted[h * 256 * 256 * 2:h * 256 * 256 * 2 + 2] == b'\x00\x80' for h in range(4))
    bad = bytearray(physical); bad[256 * 2] = 1
    try:
        ns['canonical_output'](bad, [1, 4, 64, 256], 'mlx.core.bfloat16', True, 1)
    except AssertionError:
        pass
    else:
        raise AssertionError('nonzero physical KV tail accepted')
    owner = read(R / 'owner-receipt.json'); release = read(R / 'external-release.json')
    assert owner['owner_pid'] == 1104 and owner['child_pid'] == x['pid'] == 1119
    assert owner['returncode'] == 0 and owner['waited_reaped'] and owner['stage'] == 'reaped'
    assert owner['script_sha256'] == sha(S / 'fullqual.py')
    assert owner['native_library_sha256'] == sha(Path(x['native_library']))
    assert owner['absence'] == {'returncode': 1, 'stdout': '', 'stderr': ''}
    assert release['status'] == 'PASS_EXTERNAL_RELEASE' and release['GPU_released'] and release['tool_exit'] == 0
    assert {row['pid'] for row in release['strict_absence']} == {1104, 1119}
    assert all(row['returncode'] == 1 and row['stdout'] == row['stderr'] == '' for row in release['strict_absence'])
    snapshots = []
    outer = sorted(R.glob('resource-*-raw.json')); inner = sorted((R / 'child-output').glob('resource-*-raw.json'))
    for path in outer + inner:
        row = read(path); snapshots.append(row)
        assert row['process_inventory_ok'] and row['only_owned_known_inference_processes']
        assert row['pressure_level_value'] == 1 and row['swap_used_mb'] == 5006.88
        assert row['vm_counters']['Swapouts'] == 17460761
        assert row['power_metadata']['source'] == 'AC Power' and row['power_metadata']['percent'] == 100
        assert row['processinfo_thermal_metadata']['code'] == 0
    external = read(R / 'external-resource.json')
    assert len(external) == 1 and external[0]['known_inference_process_count'] == 0
    assert read(R / 'external-resource-00-raw.json') == external[0]
    model_ready = read(R / 'child-output/model-ready.json')
    assert model_ready == {'pid': 1119, 'token_executed': False, 'model_loads': 1}
    assert x['model_loads'] == 1 and x['allocation_bucket_restored_finally']
    derived = read(R / 'resource-derived.json')
    raw_outer = [json.loads(p.read_text()) for p in outer]
    assert len(derived) == len(raw_outer)
    for row, original in zip(derived, raw_outer):
        # The saved owner adds its explicitly owned child to the broad matcher inventory.
        for key, value in original.items():
            if key not in ('known_inference_processes', 'known_inference_process_count'):
                assert row[key] == value, ('raw/derived resource field changed', key)
        assert row['only_owned_known_inference_processes']
        assert all(p['pid'] == 1119 and p['owned'] for p in row['known_inference_processes'])
        assert row['known_inference_process_count'] == len(row['known_inference_processes'])
        if 'swap_used_mb_delta' in row:
            assert row['swap_used_mb_delta'] == 0.0
            assert row['vm_counter_delta']['Swapouts'] == 0
    afterload = R / 'child-output' / x['afterload']['raw_file']
    assert sha(afterload) == x['afterload']['raw_sha256']
    assert x['afterload']['exact_owned_pid_metadata']['pid'] == 1119
    assert owner['native_source_sha256'] == sha(Path(x['native_library']).parent / 'replay.mm')
    assert not any(n == 'mlx' or n.startswith(('mlx.', 'numpy')) for n in sys.modules)
    result = {'status': 'PASS_INDEPENDENT_SAVED_RESULT_AUDIT_NO_ADOPTION', 'receipt_SHA256': EXPECTED,
              'CPU_only': True, 'GPU_or_resource_probe_executed': False, 'MLX_or_numpy_imported': False,
              'source_pin_count_verified': len(pins), 'actual_imported_source_count_verified': len(x['actual_imported_sources']),
              'counts_recomputed': counts, 'prefill_seed_checks': prefill_count, 'prefill_logits_checks': logits_count,
              'gamma_pair_checks': 161, 'gamma_readonly_arrays': 322, 'native_cached_gamma_fullrange_readonly': len(proofs),
              'true_physical_KV_CAP': 64, 'CPU_canonical_CAP': 256, 'CPU_signed_zero_and_tail_guard_checks': True,
              'ICB': {'banks': 2, 'GDN_S_pingpong': 48, 'staged_links': 83, 'total_executions': 93, 'CPU_step_uploads': 0},
              'ids': ids, 'V93_FAIL_preserved': True, 'old_profile_PASS': False,
              'resource_samples': {'outer': len(outer), 'child': len(inner), 'external': len(external), 'pressure': 1,
                                   'swap_MB_constant': 5006.88, 'Swapouts_constant': 17460761, 'AC_percent': 100, 'thermal': 'nominal'},
              'owner_pid': 1104, 'child_pid': 1119, 'reaped_rc': 0, 'external_release': 'PASS',
              'source_scope': '161 plain RMSNorm leaves only; all five GDN source files and GatedRMSNorm AST equal44',
              'B_only_exploratory_timing_support': 'YES, separately declared new leaf profile and same qualified scope; no adoption, old-profile equivalence, whole-oMLX math or broad quality claim',
              'limitations': ['saved receipts/hash metadata rechecked; no raw tensor corpus retained to rerun native memcmp independently',
                              'visible donor reference is tokenizer roundtrip of visible text, not captured internal raw generation IDs',
                              'three prompts and32 tokens only; actual timing/protocol/control validation still unrun',
                              'top-level native scope string is inherited teacherforced wording; actual feedback.ICB evidence was checked instead',
                              'resource inventory covers known processes and point samples; unrelated GUI/unknown runtimes not proven absent']}
    save(D / 'verification.json', result); save(D / 'refsSHA.json', refs)
    print(json.dumps({'status': result['status'], 'pins': len(pins), 'steps': counts['steps'], 'outputs': counts['strict_outputs'],
                      'seed_checks': prefill_count, 'resource_samples': len(outer) + len(inner) + len(external),
                      'verification_SHA256': sha(D / 'verification.json')}))


if __name__ == '__main__':
    main()
