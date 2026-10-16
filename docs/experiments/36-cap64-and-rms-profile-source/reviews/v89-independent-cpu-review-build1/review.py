"""Independent V89 stdlib CPU source and saved-evidence review. No MLX/GPU/probes."""
import ast
import hashlib
import json
from pathlib import Path
import struct
from types import SimpleNamespace

P=Path('/private/tmp/minivllm-engine-optimization-20261014')
D=Path(__file__).resolve().parent
W=P/'v89-cap64-native-fullqual-build1'
N=P/'v89-native-capacity64-build1'
R=P/'v23-root-control-build1/fullmodel-v89native-attempt1'
O=R/'child-output'
refs={};checks=[]
def sha(p):
 p=Path(p);v=hashlib.sha256(p.read_bytes()).hexdigest();refs[str(p)]={'sha256':v,'bytes':p.stat().st_size};return v
def read(p):sha(p);return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def ck(v,label):assert v,label;checks.append(label)
worker=W/'fullqual.py';native=N/'replay.mm'
ck(sha(worker)=='d1669d1bf0a0610af243764c0820d329ce63edcd479719e6be0c7355902c789c','exact_worker_SHA')
ck(sha(N/'libv89replay.dylib')=='ae5cfa2db175ed951fee6bf0b75b8c9fd04fd0149374c49b530076a58b256774','exact_native_library_SHA_no_load')
s=native.read_text();old=P/'v40-native-icb-build2/replay.mm';sha(native);sha(old)
ck(s.count('(capacity!=64&&capacity!=256)')==1 and s.replace('(capacity!=64&&capacity!=256)','capacity!=256').replace('fixed capacity64-or256 and complete input/output/link counts','fixed capacity256 and complete input/output/link counts')==old.read_text(),'exact_inverse_old40_except_capacity_guard_and_message')
tree=ast.parse(worker.read_text());compile(tree,str(worker),'exec')
canonical=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='canonical_output')
ck(not any(isinstance(n,ast.Name)and n.id in ('mx','np')for n in ast.walk(canonical)),'canonical_is_CPU_bytes_only')
ns={'CAP':64,'CANONICAL_CAP':256};exec(compile(ast.Module(body=[canonical],type_ignores=[]),str(worker),'exec'),ns)
valid=51;head=64*512;raw=b''.join((b'\x00\x80' if h%2 else b'\x00\x00')*(valid*256)+bytes((64-valid)*512)for h in range(4))
out,shape=ns['canonical_output'](raw,[1,4,64,256],'mlx.core.bfloat16',True,valid)
ck(shape==[1,4,256,256] and len(out)==524288,'canonical_complete256_shape_length')
for h in range(4):ck(out[h*256*512:h*256*512+head]==raw[h*head:(h+1)*head] and out[h*256*512+head:(h+1)*256*512]==bytes(192*512),'perhead_bits_signedzero_preserved_'+str(h))
bad=bytearray(raw);bad[valid*512:valid*512+2]=b'\x00\x80'
try:ns['canonical_output'](bad,[1,4,64,256],'mlx.core.bfloat16',True,valid)
except AssertionError:pass
else:raise AssertionError('negativezero physical tail was accepted')
checks.append('physical_tail_negativezero_rejected')
method=next(n for n in tree.body if isinstance(n,ast.ClassDef)and n.name=='Native')
call=next(n for n in method.body if isinstance(n,ast.FunctionDef)and n.name=='call');guard={};exec(compile(ast.Module(body=[call],type_ignores=[]),str(worker),'exec'),guard)
seen=[];fake=SimpleNamespace(lib=SimpleNamespace(v34_seal_feedback=lambda *a:seen.append(a)or 0))
for cap in (64,256):guard['call'](fake,'v34_seal_feedback',128,cap)
for state,cap in [(128,x)for x in (0,1,32,63,65,128,257,0xffffffff,-1,0x100000000,0x100000040,1<<80,64.0)]+[(True,64),(-1,64),(1<<32,64)]:
 try:guard['call'](fake,'v34_seal_feedback',state,cap)
 except AssertionError:continue
 raise AssertionError(('narrowing guard accepted invalid',state,cap))
ck(seen==[(128,64),(128,256)],'Python_seal_uint32_and_unknown_capacity_rejection_before_ctypes')
ck('pos<0||len!=pos+1||len>(int)F().capacity' in s,'inherited_pos_guard_identified')
ub=(D/'pos-guard-UBSan.log').read_text();sha(D/'pos-guard-UBSan.log');ck('signed integer overflow: 2147483647 + 1' in ub,'inherited_pos_overflow_CPU_UBSan_confirmed_outside_current_scope')

rp=O/'receipt.json';ck(sha(rp)=='40428c19e12c69c80209885f58246b585d0c0659c3890ae2396f5f7310143e86','exact_actual89_receipt_SHA');r=read(rp)
pins=read(W/'source-pins.json');ck(r['source_pins']==pins,'receipt_source_pins_match')
ck(all(sha(p)==h for p,h in pins.items()),'all_current_pinned_source_library_bytes')
manifest=read(W/'candidate-manifest.json')['files'];package=P/'candidate-v44-conv-normreuse-build1-root/tutorial_code/miniengine'
actual={str(p.relative_to(package)):sha(p)for p in package.rglob('*.py')};expected={k.removeprefix('tutorial_code/miniengine/'):v for k,v in manifest.items()}
ck(actual==expected and len(actual)==27,'all27_model44_package_files_unchanged')
g38=read(Path(r['A_saved_baseline_receipt']));g44=read(P/'v23-root-control-build1/fullmodel-v44native-attempt1/child-output/receipt.json')
ck(r['A_saved_baseline_receipt_sha256']==sha(Path(r['A_saved_baseline_receipt'])) and r['actual44_gold_sha256']==sha(P/'v23-root-control-build1/fullmodel-v44native-attempt1/child-output/receipt.json'),'both_actual_gold_SHA_bindings')
gold38={q['name']:q for q in g38['requests']};gold44={q['name']:q for q in g44['requests']}
ck(r['status']=='PASS_V89_PHYSICAL_CAP64_CANONICAL256_NATIVE_ICB_BASE_RESET_CHANGED_OLD38_44_STRICT' and r['native_steps']==93 and r['strict_output_checks']==12276 and r['canonical_KV_tail_zero_checks']==2976,'actual_PASS_93x132_2976_pairedKV_records')
ck(r['fixed_cap']==64 and r['canonical_capacity']==256 and not r['performance_point'] and not r['adopted'],'physical64_CPUcanonical256_no_perf_adoption')
ck(r['allocation_bucket']['effective']==[64,64] and r['allocation_bucket']['original']==[256,256] and r['allocation_bucket']['before_any_token'] and r['allocation_bucket_restored_finally'],'allocation_metadata64_before_any_token_finally256')
ck(r['model_loads']==1 and r['same_parameter_identity_count']==1349 and r['same_parameter_identity_preserved'] and r['A_baseline_restore_final'],'one_load_same1349_and_A_restore')
inputs=r['capture']['inputs'];outputs=r['capture']['outputs'];ck(len(inputs)==131 and len(outputs)==132,'physical_capture131_132')
kv={q['label']for q in outputs if q['shape']==[1,4,64,256]}
ck(len(kv)==32 and sum(q['shape']==[1,4,64,256]for q in inputs)==32,'32KV_physical64_inputs_and_outputs')
ck(not any(q['shape']==[1,4,256,256]for q in inputs+outputs),'no_captured_physical256_KV')
seedbytes=sum(q['nbytes']for q in inputs);outbytes=sum(q['nbytes']for q in outputs)
total=0;tail=0;canonical_checks=0
ck(r['declared_prompts']==g38['declared_prompts']==g44['declared_prompts'],'all_three_declared_prompts_match_gold')
for ci,q in enumerate(r['requests']):
 name=q['name'];a=gold38[name];b=gold44[name]
 ck(len(q['steps'])==31 and len(q['A_ids'])==32 and q['A_ids']==q['B_ids']==a['A_ids']==a['B_ids']==b['A_ids']==b['B_ids'],'all32IDs_match_both_gold_'+name)
 ck(q['seed_upload_calls']==q['seed_arrays']==131 and (q['changed_seed_inputs_vs_capture']==[] if ci<2 else any(i>=3 for i in q['changed_seed_inputs_vs_capture'])),'fresh_reset_or_changed_cache_seed_'+name)
 for si,e in enumerate(q['steps'],1):
  total+=1;length=q['prompt_length']+si
  ck(e['step']==si and e['A_exact_L']==length and e['B_fixed_L']==64 and 0<length<=51,'safe_position_length_'+name+str(si))
  ck(e['A_token']==e['B_token']==q['A_ids'][si] and e['seed_uploads_this_step']==0 and e['native_prediction_copied_bytes']==4 and e['numerical_KV_canonical_readback_bytes']==4194304,'GPU_feedback_no_step_seed_upload_'+name+str(si))
  ck(e['canonical132_oldCAP256_hash_checks']==132 and e['A_saved_V38_hash_checks']==e['A_saved_V44_hash_checks']==132 and e['KV_physical64_tail_zero_checks']==32,'canonical_and_tail_counts_'+name+str(si))
  old38={o['label']:o for o in a['steps'][si-1]['outputs']};old44={o['label']:o for o in b['steps'][si-1]['outputs']}
  ck(len(e['outputs'])==132 and set(old38)==set(old44)=={o['label']for o in e['outputs']},'all132_labels_'+name+str(si))
  for output in e['outputs']:
   label=output['label'];aa=old38[label];bb=old44[label]
   ck(aa==bb and all(output[k]==aa[k]for k in ('shape','dtype','bytes','A_sha256')) and output['canonical_B_sha256']==output['A_sha256'] and output['finite'] and output['strict_bytes_equal'],'canonical_dual_gold_'+name+str(si)+label)
   if label in kv:
    ck(output['physical_shape']==[1,4,64,256] and output['physical_bytes']==131072 and output['shape']==[1,4,256,256] and output['bytes']==524288 and output['physical_tail_zero_checked'] and output['canonical_CPU_zero_fill'],'paired_physicalKV_tail_zero_record_'+name+str(si)+label);tail+=1
   else:ck(output['physical_shape']==output['shape'] and output['physical_bytes']==output['bytes'] and output['physical_A_sha256']==output['A_sha256'] and not output['physical_tail_zero_checked'],'nonKV_unchanged_bytes_'+name+str(si)+label)
   canonical_checks+=1
  named={o['label']:o for o in e['outputs']}
  for label,value in [('next_positions',length),('next_lens',length+1),('prediction',q['A_ids'][si])]:ck(named[label]['A_sha256']==hashlib.sha256(struct.pack('<i',value)).hexdigest(),'int32_feedback_exact_'+name+str(si)+label)
  ck(e['reference_input_digest_readonly'] and all(v['strict_bytes_equal'] and v['arrays']==(132 if k=='capture_outputs'else 131)for k,v in e['originals_readonly'].items()),'reference_and_originals_readonly_'+name+str(si))
 fb=q['native']['feedback'];icb=fb['icb'];pp=fb['pingpong']
 ck(fb['capacity']==64 and fb['sealed'] and [fb[k]for k in ['input_count','output_count','link_count','state_count']]==[131,132,131,128],'native_actual_capacity64_counts_'+name)
 ck(fb['resets']==ci+1 and fb['request_steps']==31 and fb['total_steps']==(ci+1)*31 and fb['request_seed_uploaded_ranges']==131 and fb['CPU_seed_upload_bytes_total']==seedbytes*(ci+1) and fb['CPU_output_compare_bytes_total']==outbytes*(ci+1)*31 and not fb['CPU_full_state_restore_per_step'],'native_byte_counters_and_fresh_resets_'+name)
 ck(icb['enabled'] and icb['banks']==2 and icb['execution_count']==(ci+1)*31 and icb['bank_executions']==[(ci+1)*16,(ci+1)*15] and [pp[k]for k in ['requested_count','eligible_count','staged_link_count']]==[48,48,83],'twoICB48S83staged_bank_counts_'+name)
ck(total==93 and canonical_checks==12276 and tail==2976,'all93_12276_dualGold_2976_pair_tail_records')
resources=read(O/'resources-derived.json')
for i,sample in enumerate(resources):
 raw=read(O/f'resource-{i:02d}-raw.json');supplement={'known_inference_processes','known_inference_process_count','exactly_one_owned_known_model'}
 ck(all(sample.get(k)==v for k,v in raw.items()if k not in supplement),'resource_raw_fields_'+str(i))
 if any(sample.get(k)!=raw.get(k)for k in supplement):
  ck(raw['known_inference_processes']==[] and raw['known_inference_process_count']==0 and raw['exactly_one_owned_known_model'] is False and sample['model_loaded_by_worker'] and sample['known_inference_processes']==[{'pid':96714,'owned':True,'engine':'explicit-loaded-v24-fullmodel'}] and sample['known_inference_process_count']==1 and sample['exactly_one_owned_known_model'] is True,'explicit_loaded_worker_supplement_not_raw_matcher_'+str(i))
 ck(sample['process_inventory_ok'] and sample['only_owned_known_inference_processes'] and sample['pressure_level_value']==1 and sample['power_metadata']['source']=='AC Power' and sample['processinfo_thermal_metadata']['code']==0,'saved_resource_inventory_AC_thermal_pressure_'+str(i))
 if i:
  prior=resources[i-1];swap=sample['swap_used_mb']-prior['swap_used_mb'];counters={k:v-prior['vm_counters'][k]for k,v in sample['vm_counters'].items()if k in prior['vm_counters']}
  ck(sample['swap_used_mb_delta']==swap and sample['vm_counter_delta']==counters,'resource_saved_deltas_'+str(i))
  if i>=2:ck(not(swap>64 and resources[i-1]['swap_used_mb_delta']>64) and not(counters.get('Swapouts',0)>0 and resources[i-1]['vm_counter_delta'].get('Swapouts',0)>0),'no_sustained_swap_or_Swapouts_'+str(i))
owner=read(R/'owner-receipt.json');release=read(R/'external-release.json')
ck(owner['owner_pid']==96705 and owner['child_pid']==96714 and owner['returncode']==0 and owner['waited_reaped'] and owner['stage']=='reaped' and owner['script_sha256']==sha(worker) and owner['native_library_sha256']==sha(N/'libv89replay.dylib'),'owner_SHA_rc0_reaped')
ck(release['status']=='PASS_EXTERNAL_RELEASE' and release['GPU_released'] and release['tool_exit']==0 and {a['pid']for a in release['strict_absence']}=={96705,96714} and all(a['returncode']==1 and not a['stdout'] and not a['stderr']for a in release['strict_absence']),'saved_both_PID_absence_external_release')
sha(Path(__file__));sha(D/'pos-guard-probe.cpp')
selected={'status':'PASS_INDEPENDENT_CPU_REVIEW_DECLARED89_SCOPE','checks':len(checks),'new_CAP64_scope_blockers':[],'inherited_out_of_scope_finding':{'file':str(native),'line':183,'issue':'int32 pos+1 overflow before an explicit pos<capacity check for arbitrary INT32_MAX seed','CPU_UBSan_reproduced':True,'current19to50_positions_affected':False,'recommendation':'new future version: reject pos>=capacity before computing pos+1; preserve frozen89 evidence'},'source_pins':len(pins),'worker_SHA':sha(worker),'library_SHA':sha(N/'libv89replay.dylib'),'actual_receipt_SHA':sha(rp),'canonical_dual_gold_array_records':canonical_checks,'physical_AB_C_memcmp_recorded_bytes':outbytes*93,'paired_KV_tailzero_records':tail,'individual_A_plus_B_tail_checks_from_source':tail*2,'requests':3,'steps':total,'IDs_per_case':32,'actual_capacity':64,'canonical_capacity_CPU_only':256,'model_files_unchanged':27,'resource_samples':len(resources),'resource_inventory_note':'Raw matcher has no model match; loaded-worker derived records explicitly supplement exact ownedPID96714 as explicit-loaded-v24-fullmodel, separately verified; not relabeled as raw matcher observation','swap_used_MB_values':sorted({x['swap_used_mb']for x in resources}),'Swapouts_values':sorted({x['vm_counters']['Swapouts']for x in resources}),'saved_release_PASS':True,'GPU_or_probe_executed_by_reviewer':False,'evidence_limit':'No per-step A/B KV raw files are saved. This independently crosschecks all recorded canonical hashes against both immutable golds and native strict-compare byte counters/source; physical bytes and tail checks are source-bound execution evidence, not a fresh raw re-read.'}
write(D/'selected-fields.json',selected)
report=f'''# V89 independent CPU review

Declared physical64 / 32-token synchronous scope: **no new blocker found**. Actual saved full PASS remains valid; this is not an async/boundary/service/performance qualification.

- Exact worker d1669d1b…c789c, library ae5cfa2d…256774; inverse patch restores all frozen40 source bytes except the intended64/256 capacity guard and its message. All27 candidate44 Python files unchanged; all{len(pins)} pins match.
- Seal wrapper rejects negative/overflow/non-int values before uint32 narrowing and rejects capacities other than64/256. Native retains actual capacity64. Current positions19..50 and lengths20..51 satisfy its mathematical bounds.
- CPU canonical helper tests preserve signed-zero bits in valid rows, pad each head independently to full256, and reject even negative-zero bits in the physical tail. No MLX/NumPy/native library was loaded by the reviewer.
- Rechecked all93×132 =12,276 canonical metadata/hash records against **both** immutable V38 and V44 golds; all32 IDs/case match. Actual input/output KV shapes are[1,4,64,256], native reports capacity64 for all3requests, and canonical256 exists only as CPU bytes.
- 2,976 paired KV tail records cover both A and B source checks (5,952 individual sides); 93-step physical C-memcmp counter={outbytes*93:,}B. Three fresh resets/all131 uploads, changed-cache evidence, no per-step uploads, 48S/83staged links and two-ICB parity all match.
- Saved resource samples={len(resources)}; raw fields/deltas match, AC/thermal0/pressure1 pass; swap-used5006.88MB / Swapouts17460761 remain fixed. Raw matcher had no model match; derived records explicitly supplement loaded ownedPID96714, kept distinct from raw matcher evidence. Owner96705/child96714 rc0/wait/reap and external release/absence pass; these are saved snapshots, not continuous GPU observation.

**Inherited out-of-scope issue:** native replay.mm:183 computes int32 `pos+1` before an explicit upper bound on pos. A standalone CPU UBSan extraction confirms overflow for pos=INT32_MAX. Frozen40 has the same expression; current bounded89 inputs cannot trigger it. For a future expanded raw-seed/service API, reject pos>=capacity before the addition in a new version; do not alter frozen89 or its PASS.

Evidence limit: per-step KV raw arrays are not persisted. The independent saved audit verifies all gold hashes, physical-shape/strict/tail records and native compare-byte counters, rather than claiming a new raw re-read or GPU replay. Source map, {len(checks)} assertions and CPU-only UB reproduction are attached.
'''
(D/'REVIEW.md').write_text(report)
(D/'CHECKPOINT.md').write_text('COMPLETE. Independent CPU source + saved-result audit; no new89-scope blocker. Inherited INT32_MAX pos overflow documented, unrelated to current19..50 valid positions. No GPU/probe/rerun/source edits.\n')
write(D/'source-map.json',{'review_type':'INDEPENDENT_CPU_SOURCE_AND_SAVED_EVIDENCE','refs':dict(sorted(refs.items()))})
write(D/'verification.json',{'status':selected['status'],'checks':checks,'GPU_used':False,'native_library_loaded':False,'outputs':{n:hashlib.sha256((D/n).read_bytes()).hexdigest()for n in ['review.py','REVIEW.md','selected-fields.json','source-map.json','pos-guard-probe.cpp','pos-guard-UBSan.log']}})
print(json.dumps({'status':selected['status'],'checks':len(checks),'source_pins':len(pins),'canonical_dual_gold_checks':canonical_checks,'paired_tail_records':tail,'physical_compare_bytes':outbytes*93,'resources':len(resources),'swap':selected['swap_used_MB_values'],'Swapouts':selected['Swapouts_values'],'new_blockers':0,'inherited_out_of_scope_issue':True,'verification_SHA':hashlib.sha256((D/'verification.json').read_bytes()).hexdigest()}))
