"""Compact saved-result audit; Python stdlib only, no model/GPU/live probes."""
from collections import Counter
import hashlib,json,re,sys
from pathlib import Path
H=Path(__file__).resolve().parent;P=H.parent
D=P/'v117-verify-broadcast-m1-bounded-load-build1'
R=P/'v23-root-control-build1/fullmodel-v117broadcast-bounded-attempt1'
EXPECTED='bbc2067380d4ecac328427411a92853c8587a0a16ea11cf12bb8da2b370491e5'
def load(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def audit():
 assert sha(R/'child-output/receipt.json')==EXPECTED
 r=load(R/'child-output/receipt.json');pins=load(D/'source-pins.json');gold=load(D/'assertion-gold.json')['requests']
 assert r['status']=='PASS_V117_BOUNDED_LOAD_VERIFY_BROADCAST_M1_OLD96IDS_LITERAL_AND_ORDINARY_M1_COMMIT128_LASTLOGITS_STRICT'
 assert r['pins']==pins and len(pins)==216
 for p,h in pins.items():assert sha(p)==h,p
 assert r['model_loads']==1 and r['GPU_used'] and not r['timing_run'] and not r['performance_qualified'] and not r['adopted']
 assert r['parameter_before']==r['parameter_after'] and len(r['parameter_before'])==1349
 assert r['parameter_identity_before']==r['parameter_identity_after'] and len(r['parameter_identity_after'])==1349
 assert r['target1349_readonly'] and r['draft_parameter_arrays_observed']==175
 requests={x['arm']+'.'+x['name']:x for x in r['requests']};assert len(requests)==10
 for key,x in requests.items():
  expected=gold[x['name']]['expected_IDs'] if x['arm'] in ('A','B') else gold['base']['expected_IDs'][:x['budget']]
  assert x['IDs']==expected and len(x['IDs'])==x['budget']
  assert x['strict_old_ID_equal'] and not x['input_future_gold_used'] and x['runner_empty_after_finish'] and x['target1349_readonly']
  assert [v for vs in x['deliveries'] for v in vs]==x['IDs']
  if x['arm']!='A':assert x['draft_reset_after_finish']
 assert r['completed_old_ID_checks']=={'A':96,'B':96}
 refs={x['request']:x for x in r['ordinary_M1_reference_requests']};assert len(refs)==7
 for key,x in refs.items():
  q=requests[key];assert x['actual_IDs']==q['IDs'] and x['prompt_ids']==q['prompt_ids'] and not x['IDs_fed_to_B']
  assert x['ordinary_decode_steps']==q['budget']-1 and x['computed_frontier']==len(q['prompt_ids'])+q['budget']-1
 txs=r['transactions'];initial=r['same_frontier_initial_checks'];commits=r['same_frontier_commit_checks'];scopes=r['actual_verify_broadcast_rounds']
 assert len(txs)==len(initial)==len(commits)==len(scopes)==r['strict_transactions']==49
 prefix_pairs=0;computed=0
 for tx,begin,end,scope in zip(txs,initial,commits,scopes):
  key=tx['request'];q=requests[key];T=tx['verify_inputs'];keep=tx['committed_inputs'];start=tx['start']
  assert keep==tx['delivered'] and tx['next_frontier']==start+keep and 1<=keep<=T<=8
  assert tx['next_anchor']==q['IDs'][tx['output_count_before']+keep-1]
  assert tx['output_count_before']+tx['remaining_before']==q['budget']
  assert tx['GDN_helper_calls']==48*T and len(tx['drafts'])==T-1 and len(tx['posterior'])==T
  assert tx['same_invocation_prefix_strict'] and tx['literal_history_readonly'] and tx['invocation_inputs_readonly']
  assert tx['actual']==tx['expected'] and len(tx['actual'])==len(tx['prefix_history'])==64
  for c,hist in zip(tx['actual'],tx['prefix_history']):
   assert len(c['values'])==2 and len(hist)==T+1
   assert c['offset']==start+keep and c['values']==hist[keep]['values']
   assert [h['offset'] for h in hist]==list(range(start,start+T+1))
  assert tx['verify_broadcast_m1_114']==scope
  assert scope['transaction_owner']==tx['transaction'] and scope['verify_T']==T
  assert scope['original_Linear_instances']==scope['broadcast_M1_calls']==scope['distinct_Linear_calls']==497
  assert scope['every_Linear_once'] and scope['verify_lm_head_calls']==1 and scope['thread_owner_matches'] and scope['flags_restored_before_eval_or_draft'] and not scope['actual_PSO_observed']
  assert begin['request']==end['request']==key and end['transaction']==tx['transaction']
  assert begin['computed_frontier']==start and end['computed_frontier']==start+keep and end['keep']==keep
  for row in (begin,end):
   assert row['all128_strict_equal'] and row['all32_attention_tails_zero']
   assert len(row['cache_components'])==128 and len(row['attention_tail_checks'])==32
   assert [c['label'] for c in row['cache_components']]==[f'state.{i}' for i in range(128)]
   assert all(c['strict_equal'] and c['A']==c['B'] and c['first_difference'] is None for c in row['cache_components'])
   assert all(t['A_tail_zero'] and t['B_tail_zero'] and t['physical_shape']==[1,4,64,256] and t['used_prefix']==row['computed_frontier'] for t in row['attention_tail_checks'])
  score=end['committed_last_logits'];assert score['A']==score['B'] and score['strict_equal'] and score['first_difference'] is None
  assert score['A']['shape']==[1,248320] and end['hard_gate_pass'] and not end['B_received_A_IDs_or_hashes_as_input']
  prefix_pairs+=sum(len(h) for h in tx['prefix_history']);computed+=keep
 assert r['same_invocation_cache_array_checks']==128*49==6272
 assert r['delivered_decode_tokens']==computed==100 and sum(t['verify_inputs'] for t in txs)==211
 assert r['draft_calls']==sum(t['draft_calls'] for t in txs)==43 and r['drafted_tokens']==sum(t['drafted_tokens'] for t in txs)==162
 assert sum(t['producer_owners_retained'] for t in txs)==10912 and prefix_pairs==16640
 # Reconstruct each committed frontier/delivery chain; the final emitted token stays uncomputed.
 for key,q in requests.items():
  if q['arm']=='A':continue
  actual=[q['IDs'][0]];frontier=len(q['prompt_ids'])
  for tx in [t for t in txs if t['request']==key]:
   assert tx['start']==frontier and tx['output_count_before']==len(actual)
   n=0
   while n<len(tx['drafts']) and tx['drafts'][n]==tx['posterior'][n]:n+=1
   delivered=(tx['drafts'][:n]+[tx['posterior'][n]])[:tx['remaining_before']]
   assert n==tx['accepted_before_clip'] and len(delivered)==tx['delivered']
   actual.extend(delivered);frontier+=len(delivered)
  assert actual==q['IDs'] and frontier==len(q['prompt_ids'])+q['budget']-1
 stages=r['bounded_loader_stages'];released=[x['detail'] for x in stages if x['label']=='source_released_after_triple_eval']
 assert len(stages)==83 and len(released)==81 and sum(x['quantized'] for x in released)==47
 assert [x['index'] for x in released]==list(range(81)) and [x['unprocessed_lazy_sources'] for x in released]==list(range(80,-1,-1))
 for name in ('actual_imported_sources_before_token','actual_imported_sources_final'):
  for p in r[name].values():assert Path(p).is_relative_to(P/'candidate-v116-bounded-draftloader-build1-root') and p in pins
 resources=load(R/'resources.json');assert len(resources)==75
 for i,y in enumerate(resources):
  x=load(R/f'resource-{i:03d}-raw.json')
  assert x['pressure_level_value']==int(x['pressure_level']['text'])==1
  assert x['power_metadata']['source']=='AC Power' and x['processinfo_thermal_metadata']['code']==0
  assert float(re.search(r'used = ([\d.]+)M',x['swap']['text'])[1])==x['swap_used_mb']==5062.06
  assert int(re.search(r'Swapouts:\s*(\d+)',x['vm_stat']['text'])[1])==x['vm_counters']['Swapouts']==17519593
  inventory=list(x['known_inference_processes'])
  if x['owned_model_pid'] is not None and not any(z['pid']==x['owned_model_pid'] for z in inventory):inventory.append(dict(pid=x['owned_model_pid'],owned=True,engine='explicit-owned-full-model-child'))
  ignored={'known_inference_processes','known_inference_process_count','only_owned_known_inference_processes'}
  assert all(y[k]==v for k,v in x.items() if k not in ignored)
  assert y['only_owned_known_inference_processes'] and all(z['owned'] for z in y['known_inference_processes'])
  assert {z['pid'] for z in y['known_inference_processes']}==({39470} if x['owned_model_pid'] is not None else set())
  if i:
   assert y['swap_used_mb_delta']==x['swap_used_mb']-resources[i-1]['swap_used_mb']
   assert y['vm_counter_delta']=={k:v-resources[i-1]['vm_counters'][k] for k,v in x['vm_counters'].items()}
 ext=load(R/'external-resource-anypower-00-raw.json');exts=load(R/'external-resource-anypower.json')
 assert len(exts)==1 and all(exts[0][k]==v for k,v in ext.items())
 assert ext['power_metadata']['source']=='AC Power' and ext['processinfo_thermal_metadata']['code']==0 and ext['pressure_level_value']==1
 assert ext['swap_used_mb']==5062.06 and ext['vm_counters']['Swapouts']==17519593
 assert load(R/'child-output/model-ready.json')==dict(pid=39470,model_loads=1,token_executed=False,draft_loaded=False)
 o=load(R/'owner-receipt.json');e=load(R/'external-release.json')
 assert (o['owner_pid'],o['driver_pid'])==(39460,39470) and o['returncode']==0 and o['waited_reaped'] and o['absence']['absent']
 assert o['driver_sha256']==sha(D/'fullqual.py') and e['status']=='PASS_EXTERNAL_RELEASE' and e['GPU_released'] and e['tool_exit']==0
 assert {x['pid'] for x in e['strict_absence']}=={39460,39470} and all(x['returncode']==1 and not x['stdout'] and not x['stderr'] for x in e['strict_absence'])
 assert (R/'stderr.log').read_text()=='mx.metal.device_info is deprecated and will be removed in a future version. Use mx.device_info instead.\n'
 return dict(status='PASS_INDEPENDENT_V117_ACTUAL_SAVED_RESULTS_REVIEW',review_GPU_used=False,actual_receipt_sha256=EXPECTED,
             A_old_IDs=96,B_old_IDs=96,requests=10,ordinary_reference_requests=7,strict_transactions=49,
             literal_cache_components=6272,independent_initial_components=6272,independent_commit_components=6272,committed_last_logits=49,
             literal_prefix_pairs=prefix_pairs,producer_owner_pairs=10912,computed_verify_inputs=211,delivered_decode_tokens=100,
             parameters_before_after_exact=1349,scope_rounds_all497=49,bounded_loader_stages=83,source_pins_verified=216,
             actual_imports_before=len(r['actual_imported_sources_before_token']),actual_imports_final=len(r['actual_imported_sources_final']),
             resource_samples=75,all_AC_thermal0_pressure1=True,swap_MB=5062.06,Swapouts=17519593,owner39460_child39470_reaped_released=True,
             memory_final_allocator_only=r['memory_final'],limits='saved SHA/shape/flags and source assertions, no retained raw full tensors; no PSO/performance/HTTP/adoption proof')
if __name__=='__main__':
 assert not any(n=='mlx' or n.startswith(('mlx.','numpy')) for n in sys.modules)
 r=audit()
 if '--seal' in sys.argv:
  (H/'receipt.json').write_text(json.dumps(r,indent=2)+'\n')
  paths={str(p) for p in R.rglob('*') if p.is_file()}|set(load(D/'source-pins.json'))
  (H/'evidence-seals.json').write_text(json.dumps({p:dict(bytes=Path(p).stat().st_size,sha256=sha(p)) for p in sorted(paths)},indent=2)+'\n')
 else:
  assert r==load(H/'receipt.json')
  for p,x in load(H/'evidence-seals.json').items():assert Path(p).stat().st_size==x['bytes'] and sha(p)==x['sha256']
 print(json.dumps(r))
