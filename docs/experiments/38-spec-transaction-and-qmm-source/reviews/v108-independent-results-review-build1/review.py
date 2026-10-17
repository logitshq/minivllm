"""Independent stdlib replay of saved V108 evidence; no MLX/native/GPU or live process probes."""
from pathlib import Path
import collections,hashlib,json
P=Path('/private/tmp/minivllm-engine-optimization-20261014');R=P/'v23-root-control-build1';D=Path(__file__).resolve().parent
j=lambda p:json.loads(Path(p).read_text())
h=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
expected={'numeric-v108tiny-attempt1':'2a03d8ef97cbe0243b6c34c313d793e030e9dd64ff6610816dcc9f9bfde91b4b','fullmodel-v108transaction-attempt1':'7966209c6a956f71d0b688ff689df11c0f8d082661f8a39d5a52c51a3cd71d9d'}
S=P/'v108-dflash2-transaction-build1';pins=j(S/'source-pins.json');assert len(pins)==61 and h(S/'source-pins.json')=='485cd1fb76695295f6b2a926c7a52ba200f98e2a6f2cc6b367f3c611f3962c18'
for p,s in pins.items():assert h(p)==s,('pin changed',p)
tiny=j(R/'numeric-v108tiny-attempt1/child-output/receipt.json');full=j(R/'fullmodel-v108transaction-attempt1/child-output/receipt.json');assert tiny['pins']==full['pins']==pins
rawmap={};life={};summaries={}
for name,r in [('numeric-v108tiny-attempt1',tiny),('fullmodel-v108transaction-attempt1',full)]:
 out=R/name;assert h(out/'child-output/receipt.json')==expected[name]
 owner=j(out/'owner-receipt.json');release=j(out/'external-release.json');pid=owner['driver_pid'];code=0 if name.startswith('numeric') else 1
 assert owner['stage']=='reaped' and owner['returncode']==code and owner['waited_reaped'] and owner['absence']['absent']
 script=next(x for x in owner['command'] if x.endswith('.py'));assert h(script)==owner['driver_sha256']
 assert release['status']=='PASS_EXTERNAL_RELEASE' and release['GPU_released'] and release['tool_exit']==code and release['release_expected_power']=='any'
 assert {x['pid'] for x in release['strict_absence']}=={owner['owner_pid'],pid} and all(x['returncode']==1 and x['stdout']==x['stderr']=='' for x in release['strict_absence'])
 raws=[]
 for f in sorted(out.glob('resource-*-raw.json')):
  x=j(f);raws.append(x);rawmap[str(f)]=h(f)
 assert len(raws)==(3 if name.startswith('numeric') else 28)
 for x in raws:assert x['power_metadata']['source']=='AC Power' and x['processinfo_thermal_metadata']['code']==0 and x['pressure_level_value']==1 and x['swap_used_mb']==raws[0]['swap_used_mb'] and x['vm_counters']['Swapouts']==raws[0]['vm_counters']['Swapouts']
 finalraw=j(out/'external-resource-anypower-00-raw.json');assert finalraw['power_metadata']['source']==release['release_observed_power']
 for f in sorted(out.rglob('*')):
  if f.is_file():rawmap[str(f)]=h(f)
 life[name]={'owner':owner['owner_pid'],'child':pid,'waited_returncode':code,'external_release_sha256':h(out/'external-release.json'),'release_observed_power':release['release_observed_power'],'release_power_policy':'any, PID absence only; no timing qualification','owner_resource_samples':len(raws),'actual_owner_power_counts':dict(collections.Counter(x['power_metadata']['source'] for x in raws)),'actual_pressure_values':sorted({x['pressure_level_value'] for x in raws}),'actual_thermal_codes':sorted({x['processinfo_thermal_metadata']['code'] for x in raws}),'stable_swap_used_MB':raws[0]['swap_used_mb'],'stable_Swapouts':raws[0]['vm_counters']['Swapouts']}
 rows=r['transactions'];counts={'transactions':len(rows),'verify_input_tokens':0,'committed_inputs':0,'committed_cache_trace_rows':0,'committed_cache_array_comparisons':0,'literal_prefix_history_rows':0,'literal_prefix_array_signatures':0,'producer_owner_pairs_retained':0}
 for row in rows:
  T,keep=row['verify_inputs'],row['committed_inputs'];assert 1<=T<=8 and 0<=keep<=T
  assert row['same_invocation_prefix_strict'] and row['literal_history_readonly'] and row['invocation_inputs_readonly'] and row['actual']==row['expected']
  assert len(row['expected'])==len(row['actual'])==len(row['prefix_history'])==(2 if name.startswith('numeric') else 64)
  for expected_cache,actual_cache,history in zip(row['expected'],row['actual'],row['prefix_history']):
   assert len(history)==T+1 and all(h['offset']==history[0]['offset']+i and len(h['values'])==2 for i,h in enumerate(history))
   assert actual_cache['offset']==history[0]['offset']+keep and actual_cache['values']==history[keep]['values']
   for values in [actual_cache['values'],*[h['values'] for h in history]]:
    assert all(x is None or len(x['sha256'])==64 and x['bytes']>0 for x in values)
  kinds=collections.Counter(x['kind'] for x in row['actual']);assert kinds==({'attention':1,'gdn':1} if name.startswith('numeric') else {'attention':16,'gdn':48})
  assert row['producer_owners_retained']==kinds['attention']+kinds['gdn']*T
  counts['verify_input_tokens']+=T;counts['committed_inputs']+=keep;counts['committed_cache_trace_rows']+=len(row['actual']);counts['committed_cache_array_comparisons']+=sum(len(x['values']) for x in row['actual']);counts['literal_prefix_history_rows']+=sum(len(history) for history in row['prefix_history']);counts['literal_prefix_array_signatures']+=sum(len(h['values']) for history in row['prefix_history'] for h in history);counts['producer_owner_pairs_retained']+=row['producer_owners_retained']
 summaries[name]=counts
assert tiny['status']=='PASS_V108_TINY_LITERAL_PREFIX_TRANSACTION_21CASES' and tiny['strict_transactions']==tiny['completed_strict_transactions']==21 and tiny['abort_restored'] and tiny['source_inputs_readonly'] and tiny['source_before']==tiny['source_after']
assert {(r['verify_inputs'],r['committed_inputs']) for r in tiny['transactions']}=={(T,k) for T in [1,3,5,8] for k in range(T+1)}
assert summaries['numeric-v108tiny-attempt1']=={'transactions':21,'verify_input_tokens':116,'committed_inputs':58,'committed_cache_trace_rows':42,'committed_cache_array_comparisons':84,'literal_prefix_history_rows':274,'literal_prefix_array_signatures':548,'producer_owner_pairs_retained':137}
for name,path in tiny['actual_imported_sources'].items():assert path in pins and Path(path).is_relative_to(P/'candidate-v108-dflash2-transaction-build1-root') and h(path)==pins[path]
assert len(full['parameter_before'])==len(full['parameter_identity_before'])==1349 and full['target1349_identity_before']
assert full['status']=='FAIL_STOP' and full['A_old96_IDs_passed'] and full['completed_old_ID_checks']=={'A':96,'B':0} and full['completed_strict_transactions']==15 and full['model_loads']==1 and full['draft_parameter_arrays_observed']==175
assert full['tiny_receipt_sha256']==expected['numeric-v108tiny-attempt1'] and not full['timing_run'] and not full['performance_qualified'] and not full['adopted']
gold=j(S/'assertion-gold.json')['requests'];old=j(P/'v23-root-control-build1/fullmodel-v44native-attempt1/child-output/receipt.json');old={x['name']:x for x in old['requests']}
assert len(full['requests'])==4
for q in full['requests'][:3]:
 assert q['arm']=='A' and len(q['IDs'])==32 and q['IDs']==gold[q['name']]['expected_IDs']==old[q['name']]['A_ids'] and q['strict_old_ID_equal'] and q['runner_empty_after_finish'] and q['target1349_readonly'] and q['input_future_gold_used'] is False
b=full['requests'][-1];assert b['arm']=='B' and b['name']=='base' and len(b['IDs'])==32 and b['strict_old_ID_equal'] is False and b['input_future_gold_used'] is False
for key in ['runner_empty_after_finish','draft_reset_after_finish','target1349_readonly']:assert key not in b
for key in ['parameter_after','parameter_identity_after','actual_imported_sources_final','B_old96_IDs_passed','target1349_readonly']:assert key not in full
assert 'fullqual.py\", line 70' in full['traceback'] and 'assert emitted==expected' in full['traceback'] and 'old38/44/89 greedy IDs differ' in full['error']
assert next(i for i,(a,c) in enumerate(zip(b['IDs'],gold['base']['expected_IDs'])) if a!=c)==4
reference109=R/'omlx-mtp-v109-attempt1/child-output/receipt.json';r109=j(reference109);assert b['IDs']==r109['ordinary_control_raw32'] and all(b['IDs']==x['raw_output_ids'] for x in r109['MTP_per_request']);rawmap[str(reference109)]=h(reference109)
assert [x for batch in b['deliveries'] for x in batch]==b['IDs']
frontier=19;delivered=[];drafted=accepted=helpercalls=0
for row in full['transactions']:
 assert row['request']=='B.base' and row['start']==frontier and row['output_count_before']==len(delivered)+1 and row['remaining_before']==31-len(delivered) and row['verify_inputs']==min(5,row['remaining_before'],64-frontier)
 assert len(row['drafts'])==row['verify_inputs']-1 and len(row['posterior'])==row['verify_inputs']
 n=0
 while n<len(row['drafts']) and row['drafts'][n]==row['posterior'][n]:n+=1
 produced=(row['drafts'][:n]+[row['posterior'][n]])[:row['remaining_before']]
 assert row['accepted_before_clip']==n and row['delivered']==len(produced)==row['committed_inputs'] and row['next_anchor']==produced[-1] and row['next_frontier']==frontier+len(produced) and row['GDN_helper_calls']==48*row['verify_inputs'] and row['draft_calls']==1
 delivered.extend(produced);frontier+=len(produced);drafted+=len(row['drafts']);accepted+=n;helpercalls+=row['GDN_helper_calls']
assert delivered==b['IDs'][1:] and frontier==50 and len(delivered)==31 and drafted==58 and accepted==16 and helpercalls==3504
assert summaries['fullmodel-v108transaction-attempt1']=={'transactions':15,'verify_input_tokens':73,'committed_inputs':31,'committed_cache_trace_rows':960,'committed_cache_array_comparisons':1920,'literal_prefix_history_rows':5632,'literal_prefix_array_signatures':11264,'producer_owner_pairs_retained':3744}
diag=full['old128_state_hash_diagnostics'];assert len(diag)==1 and diag[0]['name']=='base' and len(diag[0]['states'])==128 and diag[0]['equal']==0 and diag[0]['different']==128
oldrows={x['label']:x for x in old['base']['steps'][-1]['outputs']}
for row in diag[0]['states']:assert row['old44_sha256']==oldrows[row['label']]['A_sha256'] and row['strict_equal']==(row['new_sha256']==row['old44_sha256']) is False
for name,path in full['actual_imported_sources_before_token'].items():assert path in pins and Path(path).is_relative_to(P/'candidate-v108-dflash2-transaction-build1-root') and h(path)==pins[path]
initial_external=j(R/'numeric-v108tiny-attempt1/external-resource-00-raw.json');assert initial_external['power_metadata']['source']=='Battery Power';rawmap[str(R/'numeric-v108tiny-attempt1/external-resource-00-raw.json')]=h(R/'numeric-v108tiny-attempt1/external-resource-00-raw.json')
result={'status':'PASS_CPU_SAVED_EVIDENCE_REVIEW_TINY_PASS_FULL_FAIL_PRESERVED','tiny_receipt_sha256':expected['numeric-v108tiny-attempt1'],'full_receipt_sha256':expected['fullmodel-v108transaction-attempt1'],'source_pins_SHA256':h(S/'source-pins.json'),'pins_verified':61,'same_pins_tiny_full':True,'counts':summaries,'failure':{'earliest_hard_failure':'fullqual.py:70 postgeneration B.base IDs gate called from108','old_A_IDs_completed':96,'old_B_IDs_completed':0,'B_base_actual_IDs':32,'first_zero_based_mismatch_index':4,'B_matches_actual109_official32':True,'actual109_receipt_sha256':h(reference109),'old128_state_diagnostic_equal':0,'old128_state_diagnostic_different':128,'B_postparameter_RO_assertion_completed':False,'final_parameter_after_present':False,'B_postcleanup_assertions_completed':False,'final_actual_import_provenance_completed':False,'full_qualification_PASS':False},'transactions':{'actual_draft_calls':15,'drafted_tokens':58,'accepted_before_clip_sum':16,'delivered_decode_tokens':31,'final_frontier':50,'actual44_GDN_helper_calls':3504,'history_vs_old132_equivalence_claim':False,'scope':'same invocation selected literal prefix + saved before/after readonly flags; no independent reconstruction of dead GPU arrays'},'lifecycle':life,'power_note':'Tinyowner3samplesAC; first externalrawBattery retained, finalanypowerrelease onlyabsentPID. Full28ownersamplesallACpressure1thermal0stableSO; anypowermetadata numericalpolicy, no timingqual.','tiny_abort_scope':'raises before producer;21completed-prefix casesinclude keep0 restore, not actualGPU partialproducer exception','parameter_scope':'1349before and threeA requestreadonly guards; afterdraftload guard by reached sourcecontrolflow. No B-post/finalreadonly assertion earned after IDFAIL.','actual_import_scope':{'tiny_final_modules':len(tiny['actual_imported_sources']),'full_before_token_modules':len(full['actual_imported_sources_before_token']),'full_final_modules':'NOT_REACHED'},'MLX_imported':False,'native_loaded':False,'GPU_executed':False,'source_modified':False}
(D/'receipt.json').write_text(json.dumps(result,indent=2)+'\n');(D/'raw-source-seals.json').write_text(json.dumps({'files':{p:{'bytes':Path(p).stat().st_size,'sha256':s} for p,s in {**pins,**rawmap}.items()},'scope':'full saved raw/source identities only, no source or tensor modifications'},indent=2)+'\n')
print(json.dumps({'status':result['status'],'tiny_transactions':21,'full_transactions':15,'full_committed_array_checks':1920,'full_prefix_signatures':11264,'full_qualification_PASS':False,'B_post_RO':'NOT_REACHED'}))
