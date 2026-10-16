"""Independent CPU-only saved V82 source, oldgold, lifecycle, layout and native metadata review."""
from pathlib import Path
import hashlib,json,ast
P=Path('/private/tmp/minivllm-engine-optimization-20261014');D=P/'v82-native-fullqual-build1';R=P/'v23-root-control-build1/fullmodel-v82native-attempt1';O=R/'child-output';HERE=Path(__file__).resolve().parent
h=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
j=lambda p:json.loads(Path(p).read_text())
r=j(O/'receipt.json');assert h(O/'receipt.json')=='6553eae93b7f79bed92753749cb04b156bd4343db422ec88043a75c8072265af'
assert r['status']=='PASS_V82_BA_FP32_GATES_PREWORK_GDN_FULL_ICB_BASE_RESET_CHANGED_STRICT' and r['model_loads']==1 and r['stage']=='complete' and r['performance_point'] is False and r['adopted'] is False
pins=j(D/'source-pins.json');assert pins==r['source_pins']
for p,s in pins.items():assert h(p)==s,('pin',p)
m=j(D/'candidate-manifest.json');root=Path(m['source_root']);root=root.parent.parent if root.name=='miniengine' else root;expected={k.removeprefix('tutorial_code/miniengine/'):v for k,v in m['files'].items()};actual={str(f.relative_to(root/'tutorial_code/miniengine')):h(f) for f in (root/'tutorial_code/miniengine').rglob('*') if f.is_file()};assert actual==expected and sum(k.endswith('.py') for k in actual)==29
G=[]
for version,sha in [('38','c57e7625baa5a4ce450fc7306233e766f1efd71636713080d748780624542c8a'),('44','1dcfc41e7779c575602bd6c73bae4e4800664db42e1ee87e75414335bdc45b6d')]:
 f=P/f'v23-root-control-build1/fullmodel-v{version}native-attempt1/child-output/receipt.json';assert h(f)==sha;G.append({x['name']:x for x in j(f)['requests']})
count=0
for req in r['requests']:
 assert req['status']=='PASS_REQUEST_STRICT' and len(req['steps'])==31 and len(req['A_ids'])==32 and req['A_ids']==req['B_ids']
 assert req['seed_upload_calls']==req['seed_arrays']==131
 for g in G:assert req['prompt_ids']==g[req['name']]['prompt_ids'] and req['A_ids']==g[req['name']]['A_ids']
 for step in req['steps']:
  assert len(step['outputs'])==132 and step['B_fixed_L']==256 and step['A_token']==step['B_token']
  assert step['native_prediction_copied_bytes']==4 and step['seed_uploads_this_step']==0 and step['A_saved_V38_hash_checks']==132 and step['reference_input_digest_readonly']
  for role,arrays in [('seed',131),('capture_outputs',132),('request_seed',131)]:assert step['originals_readonly'][role]['strict_bytes_equal'] and step['originals_readonly'][role]['arrays']==arrays
  for g in G:
   prior=g[req['name']]['steps'][step['step']-1];assert prior['A_exact_L']==step['A_exact_L'] and prior['A_token']==step['A_token']
   assert len(prior['outputs'])==132
   for a,b in zip(step['outputs'],prior['outputs']):
    assert all(a[k]==b[k] for k in ['label','shape','dtype','bytes','A_sha256'])
    assert a['finite'] and a['strict_bytes_equal']
  count+=132
assert len(r['requests'])==3 and count==r['strict_output_checks']==12276 and r['native_steps']==93
assert r['requests'][0]['prompt_length']==r['requests'][1]['prompt_length']==19 and r['requests'][2]['prompt_length']==20 and len(r['requests'][2]['changed_seed_inputs_vs_capture'])==129
p=r['ba_gates_prework_preparation'];assert p['status']=='READY_ACTUAL_LAYOUT_PREPARED' and len(p['layout_rows'])==384 and len(p['prepared'])==48 and len(p['original288_target_arrays'])==288
assert p['readonly_after_prepare'] and p['target_payload_bytes']==13289472 and p['extra_packed_weight_bytes']==0 and p['parameter_owners']==1349 and p['same_parameter_identity_after_prepare'] and p['target_parameter_holders_retained'] and p['original_parameter_ranges_disjoint']
for row in p['layout_rows']:
 assert row['row_contiguous_derived_singleton_aware'] and row['internal_MLX_flags_observed'] is False and row['ensure_row_contiguous'] is False
 expected_stride=1
 for dim,stride in zip(reversed(row['shape']),reversed(row['actual_strides_elements'])):
  assert dim==1 or stride==expected_stride;expected_stride*=dim
for row in p['prepared']:assert row['same_original_handles'] and row['X_activated_readonly'] and row['mlx_active_before_prepare']==row['mlx_active_after_prepare'] and len(row['original_operand_ids'])==6
ranges=[(v['address'],v['address']+v['nbytes']) for v in p['original288_target_arrays']];assert all(b<=c or d<=a for i,(a,b) in enumerate(ranges) for c,d in ranges[i+1:])
eng=r['B_ba_gates_prework_capture_engagement'];assert eng['calls']==eng['distinct_GDN_objects']==48 and eng['all48_enabled'] and eng['observers_removed_before_replay']
assert len(eng['prework_calls'])==48 and {x['helper_id'] for x in eng['prework_calls']}=={x['helper_id'] for x in p['prepared']}
assert all(x['outputs']==6 and x['output_shapes']==[[1,16,128],[1,16,128],[1,48,128],[1,16,2],[1,48],[1,48]] and x['output_dtypes']==['mlx.core.float32']*6 for x in eng['prework_calls'])
bind=r['ba_gates_prework_native_binding_proof'];assert len(bind['original_target_ranges'])==288 and bind['Python_capture_calls']==48 and bind['buffer_argument_count_source_proof']==14 and 'NOT_OBSERVED' in bind['individual_argument_reflection'] and 'NOT_OBSERVED' in r['actual_individual_argument_reflection']
param={v['label']:v for v in p['original288_target_arrays']}
for row in bind['original_target_ranges']:
 before=param[row['parameter']];assert row['original_object_id']==before['original_object_id'] and row['address']==before['address'] and row['bytes']==before['nbytes'] and row['full_original_parameter_range_readonly']
 matches=[v for v in r['capture']['native']['buffers'] if v['original_buffer']==row['actual_original_buffer']];assert len(matches)==1
 b=matches[0];assert b['used'] and b['zero_copy'] and b['original_contents']<=row['address'] and row['address']+row['bytes']<=b['original_contents']+b['length'] and 'gdn_activated_BA_gates_prework_v76_build1' in b['role'] and b['role']==row['role']
assert r['capture']['parameter_arrays']==r['same_parameter_identity_count']==1349 and r['same_parameter_identity_preserved'] and r['ba_gates_target288_readonly_final']
assert len(r['capture']['inputs'])==131 and len(r['capture']['outputs'])==132
assert r['A_baseline_restore_final'] and len(r['A_all_features_OFF_legacy_dispatch_observations'])==r['A_baseline_temporary_bind_calls']==129
assert all(x['BA_conv_gates_all_OFF']==48 and x['legacy_dispatch_observed'] and x['all_flags_restored'] for x in r['A_all_features_OFF_legacy_dispatch_observations'])
for i,req in enumerate(r['requests'],1):
 n=req['native'];f=n['feedback'];icb=f['icb'];pp=f['pingpong'];assert n['status']=='OK' and n['dispatches']==1433
 assert f['total_steps']==icb['execution_count']==icb['encoded_count']==31*i and f['resets']==i and f['request_seed_uploaded_ranges']==131 and f['CPU_full_state_restore_per_step'] is False
 assert (pp['requested_count'],pp['eligible_count'],pp['staged_link_count'])==(48,48,83) and icb['banks']==2 and icb['commands_per_bank']==1433 and icb['bank_executions']==[16*i,15*i]
 assert f['input_count']==131 and f['output_count']==132 and f['state_count']==128 and f['capacity']==256
source=(D/'fullqual.py').read_text();ast.parse(source)
assert 'finally:\n                GDN._decode_ba_gates_prework=candidate_ba;PreparedBA.prework=original_prework' in source
assert source.index('seed_baselines = [h.raw()')<source.index('native.call("v33_begin")') and source.index('output_baselines=[h.raw()')<source.index('native.call("v34_replay_step")')
owner=j(R/'owner-receipt.json');release=j(R/'external-release.json');assert owner['script_sha256']==h(D/'fullqual.py') and owner['waited_reaped'] and owner['returncode']==0 and owner['stage']=='reaped' and owner['child_pid']==r['pid'] and owner['absence']['returncode']==1 and not owner['absence']['stdout']
assert release['status']=='PASS_EXTERNAL_RELEASE' and release['GPU_released'] and release['tool_exit']==0 and {x['pid'] for x in release['strict_absence']}=={owner['owner_pid'],owner['child_pid']} and all(x['returncode']==1 and not x['stdout'] for x in release['strict_absence'])
raws=[j(f) for f in sorted(O.glob('resource-*-raw.json'))];assert len(raws)==34
for x in raws:assert x['power_metadata']['source']=='AC Power' and x['processinfo_thermal_metadata']['code']==0 and x['pressure_level_value']==1 and x['swap_used_mb']==raws[0]['swap_used_mb'] and x['vm_counters']['Swapouts']==raws[0]['vm_counters']['Swapouts']
summary={'status':'PASS_CPU_INDEPENDENT_V82_SAVED_NUMERIC_SOURCE_LIFECYCLE_REVIEW','receipt_sha256':h(O/'receipt.json'),'fullqual_sha256':h(D/'fullqual.py'),'pins_checked':len(pins),'oldgold_receipts':2,'strict_output_checks':count,'steps':93,'requests':3,'all_IDs':96,'layout_rows':384,'prepared_helpers':48,'target_parameter_RO_ranges':288,'parameter_owner_identity':1349,'actual_capture_BA_calls':48,'actual_capture_prework_calls':48,'observers_removed_before_replay':True,'dispatches':1433,'S_eligible':48,'staged_links':83,'banks':[48,45],'declared_buffer_arguments':14,'actual_individual_argument_reflection':'NOT_OBSERVED','saved_child_resource_samples':len(raws),'owner_pid':owner['owner_pid'],'child_pid':owner['child_pid'],'external_release_sha256':h(R/'external-release.json'),'speedup_claim':False,'GPU_executed':False,'MLX_imported':False}
print(json.dumps(summary,indent=2));(HERE/'selected-fields.json').write_text(json.dumps(summary,indent=2)+'\n')
