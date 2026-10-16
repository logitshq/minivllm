from pathlib import Path
from collections import Counter
import hashlib,json
P=Path('/private/tmp/minivllm-engine-optimization-20261014');O=P/'v23-root-control-build1/fullmodel-v81num-attempt1';C=O/'child-output';S=P/'v81-gdn-two-kernel-native-range-build1';D=Path(__file__).resolve().parent
files={}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):
 p=Path(p);files[str(p)]=sha(p);return json.loads(p.read_text())
r=read(C/'receipt.json');assert files[str(C/'receipt.json')]=='bd8e1b359462f210a5f539f0922a571b4f101eba730748b3a838b771f0f42f26'
assert r['status']=='PASS_V81_30_PUBLIC_60_CAPTURED_480_NATIVE_STRICT_NO_TIMING' and r['numeric_qualified'] and r['actual_reflection_verified']
assert not any(r[k] for k in ('timing_run','performance_enabled','model_loaded','HTTP_run','feedback','adopted'))
for p,h in r['pins'].items():assert sha(p)==h;files[p]=h
assert read(S/'source-pins.json')==r['pins']
assert sha(r['prior_v78_receipt'])==r['prior_v78_receipt_sha256']
assert len(r['cases'])==15 and [x['slot'] for x in r['cases']]==list(range(15)) and all(x['input_readonly_before_capture'] for x in r['cases'])
counts={};groups=(1,2,3,4,5,7,8,16)
for category,n in [('public_AB_checks',30),('captured_checks',60),('native_checks',480)]:
 rows=r[category];assert len(rows)==n and len({tuple((k,x.get(k)) for k in ('groups','arm','bank','output')) for x in rows})==n
 for x in rows:assert x['strict_equal'] and x['A_sha256']==x['B_sha256'] and x['bytes']==(24576 if x['output']=='o' else 3145728)
 counts[category]=n
assert Counter((x['groups'],x['arm']) for x in r['native_checks'])=={(g,a):30 for g in groups for a in (0,1)}
assert counts==r['completed_strict_counts'] and (r['public_strict_checks'],r['captured_strict_checks'],r['native_strict_checks'])==(30,60,480)
inputs=read(C/'inputs.json');original=read(C/'captured-originals.json');assert len(inputs)==75 and len(original)==180 and r['original_arrays_readonly']==255
assert len(r['readonly_checks'])==17 and all(x['original_arrays']==255 and x['all_raw_sha256_unchanged'] and x['native_shape_constants_unchanged'] for x in r['readonly_checks'])
for x in inputs+original:assert x['memoryview_c_contiguous'] and x['numpy_api']=='frombuffer; no copy fallback'
n=read(C/'native-built.json');assert n['status']=='OK' and n['enabled'] and n['capture_dispatches']==n['ICB_commands']==60 and n['fixed_saved_banks']==15
assert n['ranges']==[[0,30],[30,60]] and n['immutable_registered_ranges']==60 and n['mutable_registered_outputs']==180 and n['internal_state_barriers_per_range']==15
assert n['producer_consumer_edges_verified'] and n['original_and_arena_ranges_disjoint'] and n['arena_restore_once'] and not n['legacy_feedback_enabled'] and not n['legacy_ICB_enabled']
proof=read(S/'source-access-proof.json');assert proof['status']=='PASS_CPU_PINNED_GDN_STORE_ONLY_SOURCE_PROOF'
expected={(e['arm'],e['stage']):e for e in proof['source_entries']}
assert len(expected)==4 and len(n['PSOs'])==4 and all(x['supports_ICB'] for x in n['PSOs'])
state_buffers=set();psos=set()
for i,row in enumerate(n['dispatch_records']):
 arm=i//30;bank=(i%30)//2;stage=i%2
 assert (row['arm'],row['bank'],row['stage'],row['ordinal'])==(arm,bank,stage,i) and row['ICB_barrier_before']==bool(stage)
 args={x['index']:x for x in row['actual_active_arguments']};psos.add(row['function'])
 entry=expected[(arm,stage)]
 for name,p in entry['output_uses'].items():
  a=args[p['expected_index']];assert a['name']==name and a['reflection_access'] in (1,2) and a['source_store_only_RW_exception'] and not a['zero_copy']
 if stage:
  a=args[6];b=args[9];assert a['name']=='S' and a['reflection_access']==0 and a['zero_copy'] and not a['source_store_only_RW_exception'] and b['name']=='S_new' and a['original_buffer']!=b['original_buffer'];state_buffers.add(a['original_buffer'])
  pre={x['index']:x for x in n['dispatch_records'][i-1]['actual_active_arguments']}
  for src,dst in ((1,0),(2,1),(3,2),(4,3)):
   if dst not in args:assert arm==1 and dst==3;continue
   assert pre[src]['original_buffer']==args[dst]['original_buffer'] and args[dst]['reflection_access']==0
assert len(state_buffers)==15 and psos=={x['function'] for x in n['PSOs']} and r['actual_S_Snew_slots']==[6,9]
for bank in range(15):
 a,b=n['bank_mappings'][bank],n['bank_mappings'][15+bank]
 assert a['bank']==b['bank']==bank
 for j in range(4):assert a['arrays'][j]['logical_address']==b['arrays'][j]['logical_address'] and a['arrays'][j]['original_allocation']==b['arrays'][j]['original_allocation']
for g in groups:
 for a in (0,1):
  z=read(C/f'native-g{g}-a{a}.json');assert z['last_groups']==g and z['last_arm']==a and z['last_CB_completed'] and z['observed_encoded_dispatches']==30*g and z['trailing_Buffers_barriers_last_loop']==g and z['internal_state_barriers_per_range']==15
# Only terminal group16 raw arrays are retained: verify all60 hashes and their30 expected raw files.
terminal=[x for x in r['native_checks'] if x['groups']==16]
for x in terminal:
 b=x['bank'];oi=0 if x['output']=='o' else 1
 p=C/'numeric'/f'native-final-a{x["arm"]}-b{b:02d}-o{oi}.raw';q=C/'numeric'/f'expected-b{b:02d}-o{oi}.raw'
 assert p.stat().st_size==q.stat().st_size==x['bytes'] and sha(p)==sha(q)==x['A_sha256'];files[str(p)]=sha(p);files[str(q)]=sha(q)
owner=read(O/'owner-receipt.json');rel=read(O/'external-release.json');assert owner['owner_pid']==93282 and owner['child_pid']==93291 and owner['stage']=='reaped' and owner['returncode']==0 and owner['waited_reaped']
assert sha(owner['command'][2])==owner['script_sha256'] and sha(S/'replay.mm')==owner['native_source_sha256'] and sha(S/'libv81replay.dylib')==owner['native_library_sha256']
assert owner['absence']['returncode']==1 and owner['absence']['stdout']==owner['absence']['stderr']==''
assert rel['status']=='PASS_EXTERNAL_RELEASE' and rel['GPU_released'] and rel['tool_exit']==0 and {x['pid'] for x in rel['strict_absence']}=={93282,93291}
assert all(x['returncode']==1 and x['stdout']==x['stderr']=='' for x in rel['strict_absence'])
raws=[read(p) for p in sorted(set(O.glob('resource-*-raw.json'))|set(O.glob('external-resource-*-raw.json'))|set(C.glob('resource-*-raw.json')))]
assert raws and all(x['power_metadata']['source']=='AC Power' and x['processinfo_thermal_metadata']['code']==0 and x['pressure_level_value']==1 and all(y['owned'] for y in x['known_inference_processes']) for x in raws)
assert len({x['swap_used_mb'] for x in raws})==len({x['vm_counters']['Swapouts'] for x in raws})==1
receipt={'status':'PASS_INDEPENDENT_CPU_V81_SAVED_RESULT_REVIEW','counts':counts,'source_pins':len(r['pins']),'fixed_banks':15,'immutable_S_allocations':15,'S_input_output_indices':[6,9],'actual_PSOs':sorted(psos),'ICB_commands':60,'internal_barriers_per_range':15,'repeat_groups_each_arm':groups,'terminal_raw_arrays_verified':60,'expected_raw_arrays_verified':30,'readonly_arrays':255,'readonly_checkpoints':17,'raw_resource_samples':len(raws),'owner_child_reaped_released':[93282,93291],'no_feedback':True,'GPU_used_by_reviewer':False,'scope':'saved records + retained terminal group16 raw hashes; earlier group results are saved strict/hash records, not retained raw or independent GPU recomputation; no performance/fullmodel claim'}
(D/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');(D/'source-map.json').write_text(json.dumps(files,indent=2)+'\n')
(D/'REVIEW.md').write_text('''# V81 independent CPU saved-result review
- PASS: frozen source pins, owner script/native/library hashes and prior V78 proof match.
- Public30 + captured60 + native480 strict equality/hash/size records cover15 fixed banks; all declared counts recomputed.
- Native groups1,2,3,4,5,7,8,16 ×2arms ×15banks ×o/S=480; every completed report has30×groups dispatches and one trailing buffer barrier/group.
- Actual four PSOs and all60 dispatch records match source output roles; state S is ReadOnly arg6 and S_new is separate source-store-only arg9.
-15 distinct original S allocations are shared identically by A/B and remain fixed; no output feedback. Prework→state producer buffers match with15 internal barriers/range.
-60 terminal group16 raw outputs and30 expected raw arrays rehashed exactly; earlier groups are represented by saved equality/hash records.
-255 original input/captured-output arrays have17 saved readonly checkpoints; these flags do not constitute new GPU recapture.
- All saved raw resource probes are AC/thermal0/pressure1, swap-used/Swapouts unchanged;93282/93291 reaped0 and strict external absence pass.
- Local numeric proof only: no performance, autoregressive feedback, full-model, HTTP or adoption claim. Reviewer executed no GPU/runtime.
''')
print(json.dumps({'status':receipt['status'],'counts':counts,'pins':len(r['pins']),'resources':len(raws),'receipt_sha256':sha(D/'receipt.json'),'review_sha256':sha(D/'REVIEW.md'),'source_map_sha256':sha(D/'source-map.json')}))
