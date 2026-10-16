"""CPU-only analysis of existing V42 actual capture metadata; no framework imports."""
import bisect
from collections import Counter,defaultdict
import hashlib,json
from pathlib import Path
P=Path('/private/tmp/minivllm-engine-optimization-20261014');D=Path(__file__).resolve().parent
Q=P/'v23-root-control-build1/fullmodel-v42native-attempt1/child-output/receipt.json'
assert hashlib.sha256(Q.read_bytes()).hexdigest()=='0f4df3331bdf958132992b7086ef22a174685dd9147ce45c6e5c0a1a07bd8032'
q=json.loads(Q.read_text());assert q['status']=='PASS_NATIVE_DEPENDENCY_NORMREUSE_CACHE_ICB_BASE_RESET_CHANGED_VS_ACCEPTED25_STRICT' and q['native_steps']==93
n=q['capture']['native'];dep=n['feedback']['icb']['dependency_barriers'];rows=dep['dispatch_accesses_bank0'];boundaries=dep['original_boundaries'];buffers={b['original_buffer']:b for b in n['buffers']}
assert all(x['native']['feedback']['icb']['dependency_barriers']==dep for x in q['requests'])
assert sum(len(s['outputs']) for r in q['requests'] for s in r['steps'])==12276
assert sum(s['A_saved_V38_hash_checks'] for r in q['requests'] for s in r['steps'])==12276
commands=[r['original_command_index'] for r in rows];assert commands==sorted(commands)
by_destination=defaultdict(list)
for b in boundaries:by_destination[bisect.bisect_right(commands,b['command_index'])].append(b)
open_r=set();open_w=set();groups=[];hazard_counter=Counter();candidate_counter=Counter();candidate_patterns=Counter();alloc_conflicts=Counter();witness=[];recomputed=[];unique_conflict_ids=set()
for i,r in enumerate(rows):
 R=set(r['read_original_allocations']);W=set(r['write_original_allocations']);assert R|W <= buffers.keys()
 raw=R&open_w;war=W&open_r;waw=W&open_w
 assert (len(raw),len(war),len(waw))==(r['frontier_RAW_allocations'],r['frontier_WAR_allocations'],r['frontier_WAW_allocations'])
 forced=any(not b['movable_exact_Buffers_scope'] for b in by_destination[i]);original=bool(by_destination[i]);before=(i==0 or forced or bool(raw|war|waw))
 assert forced==r['forced_boundary_before'] and original==r['original_barrier_before'] and before==r['barrier_before']
 if before:open_r.clear();open_w.clear()
 open_r|=R;open_w|=W;recomputed.append(before)
 hazard_counter.update({'RAW':len(raw),'WAR':len(war),'WAW':len(waw)})
 for b in by_destination[i]:
  if not b['movable_exact_Buffers_scope']:continue
  candidate_counter['total']+=1;candidate_counter['forced_at_same_destination']+=forced;candidate_counter['has_allocation_hazard']+=bool(raw|war|waw)
  candidate_counter['has_RAW']+=bool(raw);candidate_counter['has_WAR']+=bool(war);candidate_counter['has_WAW']+=bool(waw)
  prev=rows[i-1] if i else {'read_original_allocations':[],'write_original_allocations':[]}
  adjacent=(R&set(prev['write_original_allocations']))|(W&(set(prev['read_original_allocations'])|set(prev['write_original_allocations'])))
  candidate_counter['has_immediately_previous_allocation_conflict']+=bool(adjacent)
  pattern='+'.join(k for k,s in [('RAW',raw),('WAR',war),('WAW',waw)] if s);candidate_patterns[pattern]+=1
  conflict=raw|war|waw;unique_conflict_ids|=conflict
  for k in conflict:alloc_conflicts[k]+=1
  if len(witness)<12:witness.append({'ordinal':i,'command_index':r['original_command_index'],'pattern':pattern,'original_allocation_ids':sorted(conflict),'lengths_bytes':[buffers[k]['length'] for k in sorted(conflict)],'immediately_previous_conflict':bool(adjacent)})
assert candidate_counter['total']==1059 and candidate_counter['has_allocation_hazard']==1059 and candidate_counter['forced_at_same_destination']==0
assert sum(recomputed)==1143 and sum(r['original_barrier_before'] for r in rows)==1142
# Independent all-pairs proof from the stored physical identity sets.
prefix=[0]
for b in recomputed:prefix.append(prefix[-1]+b)
edge=Counter()
sets=[(set(r['read_original_allocations']),set(r['write_original_allocations'])) for r in rows]
for j,(r,w) in enumerate(sets):
 for i,(pr,pw) in enumerate(sets[:j]):
  raw=len(pw&r);war=len(pr&w);waw=len(pw&w)
  edge.update({'RAW':raw,'WAR':war,'WAW':waw})
  if raw or war or waw:
   edge['conflicting_dispatch_pairs']+=1;assert prefix[j+1]>prefix[i+1]
assert (edge['RAW'],edge['WAR'],edge['WAW'],edge['conflicting_dispatch_pairs'])==(19353,17088,8779,27363)
assert not any(buffers[k]['zero_copy'] for k in unique_conflict_ids)
lengths=Counter(buffers[k]['length'] for k in unique_conflict_ids)
result={'status':'PASS_CPU_REVIEW_NO_BARRIER_REDUCTION','actual_receipt_sha256':hashlib.sha256(Q.read_bytes()).hexdigest(),'actual_status':q['status'],'requests':3,'native_steps':93,'strict_outputs':12276,'saved_V38_A_hash_checks':12276,'captured_dispatches':n['dispatches'],'origin_counts':dict(Counter(b['origin'] for b in boundaries)),'forced_raw_entries':sum(not b['movable_exact_Buffers_scope'] for b in boundaries),'forced_dispatch_locations':sum(r['forced_boundary_before'] for r in rows),'candidate_scope_classification':dict(candidate_counter),'candidate_hazard_patterns':dict(candidate_patterns),'candidate_unique_conflicting_original_allocations':len(unique_conflict_ids),'all_conflicting_allocations_mutable':True,'unique_conflict_allocation_length_histogram':{str(k):v for k,v in sorted(lengths.items())},'candidate_conflict_length_weighted_occurrences':dict(Counter(str(buffers[k]['length']) for k,v in alloc_conflicts.items() for _ in range(v))),'frontier_allocations':dict(hazard_counter),'independent_all_pairs_proof':dict(edge),'original_collapsed':1142,'planned':1143,'removed':0,'inserted':1,'inserted_only_ordinal': [i for i,r in enumerate(rows) if r['barrier_before'] and not r['original_barrier_before']],'candidate_deferred':0,'decision':'SKIP_NO_REDUCTION','no_performance_run':True,'byte_extent_overlap':'UNKNOWN: stored reflection/allocation identity proves allocation hazards; per-argument shader byte read/write extents are not captured','not_caused_by_shared_arena_identity':'planner and this audit use 1859 disjoint original MTLBuffer identities, never one arena ID','witnesses':witness,'GPU_used':False}
(D/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('witnesses','unique_conflict_allocation_length_histogram','candidate_conflict_length_weighted_occurrences')},indent=2))
