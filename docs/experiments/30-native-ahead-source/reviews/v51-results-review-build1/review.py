"""Read only existing actual evidence; CPU stdlib, no framework/library execution."""
import bisect,hashlib,json
from pathlib import Path
P=Path('/private/tmp/minivllm-engine-optimization-20261014');D=Path(__file__).resolve().parent
Q=P/'v23-root-control-build1/fullmodel-v51native-attempt1/child-output/receipt.json';E=Q.parent.parent/'external-release.json'
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
assert sha(Q)=='6823e0fa31ca823d542cffbeb19bc07e84b49cddb1e4d1427d6367c9d16ca088'
q=json.loads(Q.read_text());ext=json.loads(E.read_text());assert ext['status']=='PASS_EXTERNAL_RELEASE' and ext['GPU_released'] is True and ext['tool_exit']==0
assert {x['pid'] for x in ext['strict_absence']}=={73088,73097} and all(x['returncode']==1 and x['stdout']=='' for x in ext['strict_absence'])
assert q['status']=='PASS_NATIVE_PREVIOUS_OP_HOIST_CONV_ACTIVATED_NORMREUSE_CACHE_ICB_BASE_RESET_CHANGED_VS_ACCEPTED25_STRICT' and q['native_steps']==93 and q['model_loads']==1
baseline_path=Path(q['A_saved_baseline_receipt']);assert sha(baseline_path)==q['A_saved_baseline_receipt_sha256']=='c57e7625baa5a4ce450fc7306233e766f1efd71636713080d748780624542c8a'
base={r['name']:r for r in json.loads(baseline_path.read_text())['requests']}
assert q['B_conv_normreuse_cache_capture_engagement']['calls']==q['B_conv_normreuse_cache_capture_engagement']['distinct_GDN_objects']==48
assert q['B_conv_normreuse_cache_capture_engagement']['counter_removed_after_capture']
checks=ro=0
for r in q['requests']:
 assert r['status']=='PASS_REQUEST_STRICT' and len(r['steps'])==31 and r['A_ids']==r['B_ids'] and len(r['B_ids'])==32 and r['seed_upload_calls']==131
 for i,s in enumerate(r['steps']):
  assert s['step']==i+1 and s['reference_input_digest_readonly'] and s['seed_uploads_this_step']==0 and s['native_prediction_copied_bytes']==4 and s['A_saved_V38_hash_checks']==132
  assert s['A_exact_L']==r['prompt_length']+i+1 and s['B_fixed_L']==256
  gold={a['label']:a for a in base[r['name']]['steps'][i]['outputs']}
  assert len(s['outputs'])==132
  for a in s['outputs']:
   g=gold[a['label']];assert a['strict_bytes_equal'] and a['finite'] and (a['A_sha256'],a['shape'],a['dtype'],a['bytes'])==(g['A_sha256'],g['shape'],g['dtype'],g['bytes']);checks+=1
  for name,n in [('seed',131),('capture_outputs',132),('request_seed',131)]:
   a=s['originals_readonly'][name];assert a['strict_bytes_equal'] and a['arrays']==n;ro+=n
n=q['capture']['native'];dep=n['feedback']['icb']['dependency_barriers'];rows=dep['dispatch_accesses_bank0'];h=dep['previous_op_hoist'];commands=[r['original_command_index'] for r in rows];sets=[(set(r['read_original_allocations']),set(r['write_original_allocations'])) for r in rows]
forced=set();candidate=set()
for b in dep['original_boundaries']:
 dest=bisect.bisect_right(commands,b['command_index']);(candidate if b['movable_exact_Buffers_scope'] else forced).add(dest)
flags=[];R=set();W=set()
for i,(r,w) in enumerate(sets):
 flag=i==0 or i in forced or bool(W&r or R&w or W&w);flags.append(flag)
 if flag:R.clear();W.clear()
 R|=r;W|=w
assert sum(flags)==h['pre_hoist_frontier_barriers']
edges=[];edge_counts=[0,0,0]
for j,(r,w) in enumerate(sets):
 for i,(pr,pw) in enumerate(sets[:j]):
  nums=[len(pw&r),len(pr&w),len(pw&w)]
  if any(nums):edges.append((i,j))
  for k in range(3):edge_counts[k]+=nums[k]
assert len(edges)==dep['proof']['conflicting_dispatch_pairs']==26376
assert edge_counts==[18679,16453,8147]
trials=sorted([(x['from_dispatch'],True,x) for x in h['moves']]+[(x['from_dispatch'],False,x) for x in h['rejected_moves']]);assert len(trials)==h['attempts']==225
for i,accepted,t in trials:
 assert i in candidate and i not in forced and flags[i] and t['to_dispatch']==i-1 and t['all_conflict_edges_checked']==len(edges)
 pr,pw=sets[i-1];r,w=sets[i];assert not(pw&r or pr&w or pw&w)
 trial=flags[:];trial[i-1]=True;trial[i]=False;prefix=[0]
 for f in trial:prefix.append(prefix[-1]+f)
 missing=[(a,b) for a,b in edges if prefix[b+1]==prefix[a+1]]
 assert bool(missing) != accepted
 if accepted:flags=trial
 else:assert missing[0]==(t['first_uncovered_edge_from_dispatch'],t['first_uncovered_edge_to_dispatch'])
assert flags==[r['barrier_before'] for r in rows] and all(flags[i] for i in forced if i<len(flags))
assert sum(flags)==1047 and h['accepted']==177 and h['rejected_by_all_edge_proof']==48 and h['all_conflict_edge_checks_across_trials']==225*26376
buffers=[b for b in n['buffers'] if b['used']];ids={b['original_buffer'] for b in buffers};assert len(ids)==dep['proof']['validated_disjoint_original_allocations']==1856
for ranges in (sorted((b['original_contents'],b['original_contents']+b['length']) for b in buffers),sorted((b['arena_offset'],b['arena_offset']+b['length']) for b in buffers if not b['zero_copy'])):
 assert all(start<end for start,end in ranges) and all(ranges[i-1][1]<=ranges[i][0] for i in range(1,len(ranges)))
swaps={};pp=n['feedback']['pingpong'];assert (pp['requested_count'],pp['eligible_count'],pp['staged_link_count'])==(48,48,83)
for pair in pp['pairs']:
 a,b=pair['A_original_buffer'],pair['B_original_buffer'];assert pair['eligible'] and pair['bytes']==3145728 and a!=b and a not in swaps and b not in swaps
 swaps[a]=b;swaps[b]=a
assert all(swaps.get(v)==k for k,v in swaps.items()) and {swaps.get(x,x) for x in ids}==ids
last=q['requests'][-1]['native']['feedback']['icb'];assert last['execution_count']==last['encoded_count']==93 and last['bank_executions']==[48,45]
assert checks==12276 and ro==36642 and dep['proof']['both_banks_all_conflict_edges_covered'] and dep['proof']['bank_plans_equal']
out={'status':'PASS_CPU_SELECTED_FIELDS_ACTUAL_V51','receipt_sha256':sha(Q),'external_release_sha256':sha(E),'requests':3,'native_steps':93,'strict_outputs_and_independent_saved_A_hash_checks':checks,'original_readonly_array_checks':ro,'retained_A_input_readonly_steps':93,'actual_dispatches':len(rows),'actual_hoist_attempts':225,'actual_accepted':177,'actual_rejected_witnesses_verified':48,'all_trial_conflict_edges_checked_independently':225*len(edges),'all_final_flags_independently_reproduced':True,'all_forced_boundaries_preserved':True,'disjoint_original_allocations_verified':1856,'both_bank_allocation_bijection_verified':True,'ICB_execution_count':93,'ICB_bank_executions':[48,45],'original_collapsed_barriers':1046,'planned_barriers':1047,'benefit_claim':False,'GPU_used':False}
(D/'selected-fields.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
