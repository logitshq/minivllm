"""CPU stdlib saved-result replay only; no runtime/probes or model imports."""
from pathlib import Path
import hashlib,json,statistics,math
P=Path('/private/tmp/minivllm-engine-optimization-20261014');R=P/'v23-root-control-build1';D=P/'v67-results-review-build1';D.mkdir(exist_ok=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
def release(d):
 p=d/'external-release.json';x=read(p)
 assert x['status']=='PASS_EXTERNAL_RELEASE' and x['GPU_released'] is True
 assert all(y['returncode']==1 and y['stdout']==y['stderr']=='' for y in x['strict_absence'])
 return {'sha256':sha(p),'pids':[y['pid'] for y in x['strict_absence']],'tool_exit':x['tool_exit']}
def resource(raw):
 assert raw['power_metadata']['source']=='AC Power'
 assert raw['processinfo_thermal_metadata']['code']==0 and raw['pressure_level_value']==1
 assert raw['known_inference_process_count']==0 and raw['only_owned_known_inference_processes'] is True
 return raw['swap_used_mb'],raw['vm_counters']['Swapouts']
d=R/'native-decode-v67-attempt1';out=d/'child-output';p=out/'receipt.json';r=read(p)
assert sha(p)=='a2dfb5593b698caf1be39193ee1e893f35f5a43b4784fafc075944152f4bc841'
assert r['status']=='PASS_PRODUCTION_DIRECT_ASYNC_B_ONLY_ENGINE_EXPLORATORY_SCREEN'
assert r['model_loads']==1 and r['schedule']==['warm']*3+['measure']*3
assert (r['native_delivered'],r['native_completed'],r['discarded'],r['resets'],r['ICB_banks'],r['final_output_hash_checks'])==(186,186,0,6,[96,90],792)
assert r['backend_API_recording_wrappers'] is False and r['source_pins_final_verified'] is True
assert r['capture']=={'build_input131_readonly':True,'baseline_before_first_replay':True}
for path,digest in r['source_pins'].items():assert sha(Path(path))==digest,(path,digest)
gold=read(R/'fullmodel-v44native-attempt1/child-output/receipt.json')['requests'][0]['steps'][-1]['outputs'];gold={x['label']:x for x in gold}
ids=read(P/'workload.json')['continuation_ids'];measure=[];all_resource=[]
for i,q in enumerate(r['requests'],1):
 assert q['index']==i and q['phase']==('warm' if i<=3 else 'measure') and q['ids']==ids
 dt=q['t2_ns']-q['t1_ns'];whole=q['t2_ns']-q['t0_ns']
 assert dt==q['post_first_token_decode31_including_seed_ns'] and whole==q['whole32_ns']
 assert dt/31==q['decode_ns_per_step'] and 31e9/dt==q['decode31_tokens_per_s'] and 32e9/whole==q['whole32_tokens_per_s']
 assert q['capture_owners263_readonly'] is True and len(q['final132_hashes'])==132
 for h in q['final132_hashes']:
  g=gold[h['label']];assert h['sha256']==h['qualified_A_sha256']==g['A_sha256']
  assert h['shape']==g['shape'] and h['dtype']==g['dtype'] and h['bytes']==g['bytes']
 n=q['native'];f=n['feedback'];a=n['pipeline'];x=n['direct_pipeline'];pp=f['pingpong'];icb=f['icb']
 assert n['status']=='OK' and a['enabled'] and a['depth']==2 and a['pending']==[] and a['inflight']==0
 assert (a['total_submitted'],a['total_completed'],a['total_delivered'],a['total_discarded'],a['total_failed'])==(31*i,31*i,31*i,0,0)
 assert (f['resets'],f['request_seed_uploaded_ranges'],icb['bank_executions'])==(i,131,[16*i,15*i])
 assert (pp['eligible_count'],pp['staged_link_count'])==(48,83)
 assert x['enabled'] and x['diagnostic_snapshot'] is False and x['legacy_F_stage_used'] is False
 assert (x['banks_checked'],x['input_output_range_pairs_checked'],x['input_input_range_pairs_checked'],x['direct_links'],x['direct_feedback_bytes_per_step'],x['prediction_slot_bytes'])==(2,34584,17030,83,19726348,4)
 for side in ['before','after']:
  z=q['resources'][side];f=out/z['raw_file'];assert sha(f)==z['raw_sha256'];all_resource.append(resource(read(f)))
 if i>3:measure.append(q)
assert len(set(all_resource))==1
m=r['measured_summary'];assert m['median_post_first_token_decode_ns_per_step_including_seed']==statistics.median(q['decode_ns_per_step'] for q in measure)
assert m['median_decode31_tokens_per_s']==statistics.median(q['decode31_tokens_per_s'] for q in measure)
assert m['median_whole32_tokens_per_s']==statistics.median(q['whole32_tokens_per_s'] for q in measure)
selected={'V67':{'receipt_sha256':sha(p),'release':release(d),'pins_rechecked':len(r['source_pins']),'final_hashes_rechecked':792,'readonly_scope':'saved runtime assertions, no postmortem original bytes reconstruction','resource_pairs_rechecked':6,'swap_and_SO_unchanged':all_resource[0],'summary':m},'micro':{}}
for version in [60,61]:
 d=R/f'kernel-micro-v{version}-attempt1';out=d/'child-output';res=read(out/'result.json');pairs=read(out/'pairs.json')
 assert res['status']=='STOP-MICRO-GATE' and res['control_passed'] is True and not res['adopted']
 assert len(pairs)==10 and [p['order'] for p in pairs]==[['A','A'],['B','B']]+[['A','B'],['B','A']]*3+[['A','A'],['B','B']]
 times={'A':[],'B':[]};wins=0;reductions=[];controls={'A':[],'B':[]}
 for pair in pairs:
  for b in pair['blocks']:
   assert b['end_ns']-b['start_ns']==b['total_ns'] and b['total_ns']/b['repeats']==b['ns_per_call']
   assert (b['prime_repeats'],b['repeats'],b['outputs_consumed_per_call'])==(2048,8192,2)
  if pair['order'][0]==pair['order'][1]:
   ratio=pair['blocks'][1]['ns_per_call']/pair['blocks'][0]['ns_per_call'];controls[pair['order'][0]].append(ratio);assert .98<=ratio<=1.02
   assert ratio==pair['slot2_div_slot1']
  else:
   q={b['variant']:b['ns_per_call'] for b in pair['blocks']};g=1-q['B']/q['A'];wins+=q['B']<q['A'];reductions.append(g)
   assert math.isclose(g,pair['reduction'],abs_tol=1e-15)
   for arm in times:times[arm].append(q[arm])
 med={arm:statistics.median(v) for arm,v in times.items()};gain=1-med['B']/med['A']
 assert med==res['medians_ns'] and math.isclose(gain,res['median_reduction'],abs_tol=1e-15) and wins==res['wins']==1
 assert controls==res['control_slot2_div_slot1']
 ro=read(out/'readonly.json');assert all(x['unchanged'] and x['before']==x['after'] for x in ro)
 rawresources=read(out/'resources.json');observed=[resource(x) for x in rawresources];assert len(set(observed))==1
 selected['micro'][str(version)]={'result_sha256':sha(out/'result.json'),'pairs_sha256':sha(out/'pairs.json'),'release':release(d),'medians_ns':med,'median_reduction_ratio_of_arm_medians':gain,'median_of_pair_reductions_descriptive':statistics.median(reductions),'wins':wins,'controls':controls,'resources_checked':len(observed),'status':'STOP-MICRO-GATE_NO_ADOPT'}
(D/'selected-fields.json').write_text(json.dumps(selected,indent=2)+'\n')
(D/'REVIEW.md').write_text('''# Saved V67 / V60 / V61 result review\nCPU stdlib only; no model import, GPU, resource probe or original evidence changes.\nV67 receipt a2dfb559…c841 and release bd54a8c9…b8acf match; owner82274/child82275 strict absence is recorded.\nAll145 bound source hashes,792 final132 hashes against actual44 A,186 delivered/completed,0discard,6reset and banks96/90 independently checked.\nAll six IDs match; original131+132 owner readonly is a saved runtime assertion, not recreated from dead-process bytes.\nTimestamps independently reproduce median34.691296ms/step,28.825675 decode tps and25.922431 whole32 tps (3measured requests).\nDecode begins after first prefill token and includes native reset/131seed uploads; hashes/resources are outside interval. No HTTP, paired gain or adoption claim.\nProduction direct proof is2banks,34584 i-o/17030 i-i comparisons,83links/19,726,348B plus4Bprediction snapshot; old20,223,012B stage retainedunused.\n12 request boundary raw resources hash-checked: AC/thermal0/pressure1, foreignknown0, swap-used/Swapouts unchanged.\nV60/V61 independently reconstruct six balanced AB/BA pairs and four same-arm controls each; all controls within2%.\nArm-median reductions are -1.200210% / -2.162110%, each1/6wins; result STOP-MICRO-GATE and noadopt are correct.\nThese are ratio-of-arm-medians, not median pair gain; descriptive pair-median gains differ and do not change failure.\nEach block is prime2048/repeat8192; both outputs consumed; immutable saved input, no feedback chain, no compile/fullmodel/E2E claim.\nBoth raw readonly dictionaries agree before/after;24 resource samples each satisfy AC/thermal0/pressure1 and stable swap/Swapouts.\nAll three release artifacts record reaped/exited owned PIDs with strict absence. selected-fields.json records exact SHAs and arithmetic.\n''')
print(json.dumps(selected,indent=2))
