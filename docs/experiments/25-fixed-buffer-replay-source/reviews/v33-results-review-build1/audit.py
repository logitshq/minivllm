"""Saved-evidence CPU audit only. Does not import MLX or load the replay library."""
from pathlib import Path
import hashlib, json, math, re, sys
import numpy as np
assert not any(x == 'mlx' or x.startswith('mlx.') for x in sys.modules)
P=Path('/private/tmp/minivllm-engine-optimization-20261014')
O=P/'v23-root-control-build1/fullmodel-v33-attempt1'; D=O/'child-output'; OUT=Path(__file__).parent

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
r=read(D/'receipt.json')
assert sha(D/'receipt.json')=='c6d34a20d8b526eff0f05afb58110a1768b35fc65cb96d4bb05a764a33998e85'
assert r['status']=='PASS_FIXED_L20_NATIVE_REPLAY_2_STRICT' and r['stage']=='complete'
assert r['capture_L']==20 and r['model_loads']==1 and r['teacherforced_fixed_seed']
assert all(r[k] is False for k in ('performance_point','adopted','autoregressive_replay','ICB_claim'))
assert r['gates_on'] and r['conv_off'] and r['functional_off']
assert r['runtime']['mlx_version']=='0.32.3' and r['runtime']['device']['device_name']=='Apple M4 Max'
assert r['runtime']['cache_limit_bytes']==2<<30
assert r['warm_ids']==read(P/'workload.json')['continuation_ids']
for path,s in r['source_pins'].items():assert sha(path)==s,(path,'source pin')
assert len(r['source_pins'])==42
for path in r['actual_imported_sources'].values():assert path in r['source_pins'] and 'candidate-v17-gdn-gates-build1-root/' in path
rawgroups={}
for key,count in (('seed_inputs',131),('expected_outputs',129)):
 rows=r[key];assert len(rows)==count and len({a['label'] for a in rows})==count
 actual=[]
 for a in rows:
  p=(D/a['raw_file']).resolve();assert p.is_relative_to(D.resolve())
  data=p.read_bytes();assert len(data)==a['nbytes'] and hashlib.sha256(data).hexdigest()==a['sha256']
  dtype={'mlx.core.bfloat16':'<u2','mlx.core.float32':'<f4','mlx.core.int32':'<i4'}[a['dtype']]
  x=np.frombuffer(data,dtype=dtype)
  assert len(data)==math.prod(a['shape'])*x.itemsize
  finite=bool(np.all((x&0x7f80)!=0x7f80)) if dtype=='<u2' else bool(np.isfinite(x).all())
  assert finite and a['finite'] and a['memoryview_c_contiguous'] and a['numpy_api']=='frombuffer; no copy fallback'
  assert a['bf16_equal_width_bit_view']==(dtype=='<u2')
  actual.append({k:a[k] for k in ('label','raw_file','nbytes','shape','dtype','sha256')})
 rawgroups[key]={'arrays':count,'bytes':sum(a['nbytes'] for a in rows),'raw_manifest_sha256':digest(actual)}
 del data,x
expected={a['label']:a for a in r['expected_outputs']}
def readonly(x,key):
 assert x['strict_bytes_equal'] is True and x['arrays']==rawgroups[key]['arrays'] and x['bytes']==rawgroups[key]['bytes']
readonly(r['seed_after_original_capture'],'seed_inputs')
assert len(r['replays'])==2
for i,rp in enumerate(r['replays'],1):
 assert rp['replay']==i and len(rp['outputs'])==129
 assert {x['label'] for x in rp['outputs']}==set(expected)
 for x in rp['outputs']:
  e=expected[x['label']]
  assert x['strict_equal'] and x['finite'] and x['bytes']==e['nbytes'] and x['sha256']==e['sha256']
 readonly(rp['seed_readonly'],'seed_inputs');readonly(rp['captured_reference_readonly'],'expected_outputs')
n=read(D/'native-output-mappings.json')
assert n['status']=='OK' and n['arena_built'] and n['error']=='' and n['parameter_ranges_registered']==1349
assert not n['original_events_resignaled_during_replay'] and n['native_replays']==0
for i in (1,2):
 z=read(D/f'native-after-replay-{i}.json');assert z['native_replays']==i
 assert {k:v for k,v in z.items() if k!='native_replays'}=={k:v for k,v in n.items() if k!='native_replays'}
built=read(D/'native-built-arena.json')
assert built['outputs']==[]
assert {k:v for k,v in built.items() if k!='outputs'}=={k:v for k,v in n.items() if k!='outputs'}
buffers={b['original_buffer']:b for b in n['buffers']};assert len(buffers)==len(n['buffers'])==1836
param={a['label']:a for a in r['parameter_ranges']};assert len(param)==1349
for a in param.values():assert a['address']>0 and a['memoryview_c_contiguous'] and a['numpy_api']=='frombuffer; no copy fallback'
large=[];small=[]
for b in buffers.values():
 assert b['used'] and b['length']>0 and b['original_contents']>0
 if b['large']:
  assert b['zero_copy'] and b['snapshot_bytes']==0 and b['length']>16<<20
  m=re.fullmatch(r'(parameter\..+) full-range=(\d+)\+(\d+) binding-delta=(\d+) readonly:(.+)/([^/]+)',b['role']);assert m,b
  label,addr,size,delta,ps,arg=m.groups();addr,size,delta=int(addr),int(size),int(delta)
  assert (addr,size)==(param[label]['address'],param[label]['nbytes'])
  assert b['original_contents']<=addr and addr+size<=b['original_contents']+b['length'] and delta>=0
  large.append(b)
 else:
  assert not b['zero_copy'] and b['snapshot_bytes']==b['length'] and b['arena_offset']%256==0
  assert b['arena_offset']+b['length']<=n['arena_bytes']
  small.append(b)
for ranges in (sorted((b['original_contents'],b['original_contents']+b['length']) for b in buffers.values()),sorted((b['arena_offset'],b['arena_offset']+b['length']) for b in small)):
 assert all(a[1]<=b[0] for a,b in zip(ranges,ranges[1:]))
assert sum(b['length'] for b in large)==n['zero_copy_buffer_bytes']
assert sum(b['length'] for b in small)==n['snapshot_bytes']
assert len(n['outputs'])==129 and {x['label'] for x in n['outputs']}==set(expected)
for x in n['outputs']:
 e=expected[x['label']];b=buffers[x['original_buffer']]
 assert x['address']==e['address'] and x['bytes']==e['nbytes']
 assert not b['large'] and x['address']==b['original_contents']+x['original_offset']
 assert 0<=x['original_offset'] and x['original_offset']+x['bytes']<=b['length']
 assert b['original_contents']<=x['captured_writable_binding_start']<=x['address']
cs=n['commit_records'];assert len(cs)==n['original_CB_boundaries']==85 and len({c['CB_id'] for c in cs})==85
assert len({c['queue'] for c in cs})==1
assert [c['CB_id'] for c in cs]==sorted(c['CB_id'] for c in cs)
for c in cs:
 assert not c['duplicate'] and c['commit_depth']==0 and c['parent_commit_CB_id']==c['parent_commit_CB_pointer']==0
 assert c['lifetime_observed_unknown_commands']==0
 if c['proven_empty_same_queue']:
  assert c['active_CB_id']==c['active_CB_pointer']==c['lifetime_observed_compute_encoders']==c['lifetime_observed_event_commands']==0
 else:
  assert c['active_CB_id']==c['CB_id'] and c['active_CB_pointer']==c['CB_pointer'] and c['lifetime_observed_compute_encoders']+c['lifetime_observed_event_commands']>0
assert sum(not c['proven_empty_same_queue'] for c in cs)==n['captured_CB_count']==84
ids={c['CB_id'] for c in cs}
assert len(n['fence_events'])==84 and sum(f['operation']=='update' for f in n['fence_events'])==83 and sum(f['operation']=='wait' for f in n['fence_events'])==1
assert len({f['original_fence'] for f in n['fence_events']})==83
fence_producers={}
for f in n['fence_events']:
 assert f['producer_CB_id'] in ids and not f['pre_capture_completed_proof'] and f['native_lowering']=='full buffer scope barrier'
 if f['operation']=='update':fence_producers[f['original_fence']]=f['producer_CB_id']
 else:assert fence_producers[f['original_fence']]==f['producer_CB_id']
assert len(n['event_commands'])==1
for e in n['event_commands']:
 assert e['operation']=='signal' and not e['pre_capture'] and e['actual_CB_id']==e['producer_CB_id'] in ids
 assert e['actual_queue']==cs[0]['queue'] and e['value']==1 and 'never re-signaled' in e['native_lowering']
allresources=[];resources={}
for base,pattern,derived in ((O,'resource-???-raw.json','resource-derived.json'),(D,'resource-??-raw.json','resources-derived.json'),(O,'external-resource-??-raw.json','external-resource.json')):
 ps=sorted(base.glob(pattern));ds=read(base/derived);assert len(ps)==len(ds)
 stats=[]
 for j,(p,d) in enumerate(zip(ps,ds)):
  raw=read(p)
  mutable={'known_inference_processes','known_inference_process_count','only_owned_known_inference_processes','exactly_one_owned_known_model'}
  for k,v in raw.items():
   if k not in mutable:assert d[k]==v,(p,k)
  assert int(raw['pressure_level']['text'])==d['pressure_level_value']==1
  assert int(re.search(r'free percentage: (\d+)%',raw['memory_pressure']['text'])[1])==d['free_percent']
  assert float(re.search(r'used = ([0-9.]+)M',raw['swap']['text'])[1])==d['swap_used_mb']
  for k in ('Swapouts','Swapins','Pageouts','Compressions','Decompressions'):
   assert int(re.search(r'^'+k+r':\s+(\d+)\.',raw['vm_stat']['text'],re.M)[1])==d['vm_counters'][k]
  assert d['process_inventory_ok'] and d['only_owned_known_inference_processes']
  assert all(x['owned'] and x['pid']==r['pid'] for x in d['known_inference_processes'])
  assert raw['power_metadata']['status']=='known' and raw['power_metadata']['source']=='AC Power' and "'AC Power'" in raw['power_metadata']['raw']['stdout']
  assert raw['processinfo_thermal_metadata']['status']=='known' and int(raw['processinfo_thermal_metadata']['raw']['stdout'])==raw['processinfo_thermal_metadata']['code']==0
  if j:
   prev=ds[j-1]
   assert d['swap_used_mb_delta']==d['swap_used_mb']-prev['swap_used_mb']==0
   assert d['vm_counter_delta']=={k:v-prev['vm_counters'][k] for k,v in d['vm_counters'].items() if k in prev['vm_counters']}
   assert d['vm_counter_delta']['Swapouts']==0
  stats.append({'file':str(p),'sha256':sha(p)})
  allresources.append(d)
 resources[str(base/derived)]={'samples':len(ps),'raw_manifest_sha256':digest(stats)}
assert sum(x['samples'] for x in resources.values())==19
assert sha(D/r['afterload']['raw_file'])==r['afterload']['raw_sha256']
own=read(O/'owner-receipt.json');rel=read(O/'external-release.json')
assert own['child_pid']==r['pid']==55580 and own['owner_pid']==55571
assert own['stage']=='reaped' and own['returncode']==0 and own['waited_reaped']
assert own['absence']=={'returncode':1,'stdout':'','stderr':''}
assert own['script_sha256']==r['source_pins'][own['command'][2]] and own['native_library_sha256']==r['source_pins'][r['native_library']]
assert own['native_source_sha256']==r['source_pins'][str(P/'v33-native-replay-build4/replay.mm')]
assert rel['status']=='PASS_EXTERNAL_RELEASE' and rel['tool_exit']==0 and rel['GPU_released']
assert {a['pid'] for a in rel['strict_absence']}=={55571,55580}
assert all(a['returncode']==1 and a['stdout']==a['stderr']=='' for a in rel['strict_absence'])
assert not any(x == 'mlx' or x.startswith('mlx.') for x in sys.modules)
result={'status':'PASS_SAVED_V33_FIXED_SEED_EVIDENCE_CPU_AUDIT','scope':'Independently hashed saved seed/GT raw; replay hashes/read-only are actual worker records, not a new GPU recapture; not autoregressive/ICB/performance qualification.','receipt_sha256':sha(D/'receipt.json'),'source_pins_verified':len(r['source_pins']),'actual_imported_sources_verified':len(r['actual_imported_sources']),'raw':rawgroups,'strict_replay_output_records':258,'replay_record_manifest_sha256':digest(r['replays']),'seed_and_reference_readonly_records_verified':5,'native':{'buffers':len(buffers),'mutable_buffers':len(small),'large_zero_copy_buffers':len(large),'snapshot_bytes':n['snapshot_bytes'],'arena_bytes':n['arena_bytes'],'zero_copy_buffer_bytes':n['zero_copy_buffer_bytes'],'output_mappings':129,'interior_outputs':sum(x['original_offset']>0 for x in n['outputs']),'fence_updates':83,'fence_waits':1,'compute_CB':83,'event_only_CB':1,'event_signals':1,'commit_records':85,'nonempty_CB':84,'proven_empty_CB':1,'dispatches':n['dispatches']},'resources':resources,'resource_summary':{'samples':19,'AC':True,'thermal_codes':sorted({x['processinfo_thermal_metadata']['code'] for x in allresources}),'pressure_levels':sorted({x['pressure_level_value'] for x in allresources}),'minimum_free_percent':min(x['free_percent'] for x in allresources),'swap_mb_growth':0,'Swapouts_growth':0,'Pageouts_growth_owner':read(O/'resource-derived.json')[-1]['vm_counters']['Pageouts']-read(O/'resource-derived.json')[0]['vm_counters']['Pageouts'],'compression_activity_present':True},'cleanup':{'owner':55571,'child':55580,'waited_reaped_exit0':True,'saved_external_strict_absence':True},'metadata_sha256':{str(p):sha(p) for p in sorted(list(D.glob('*.json'))+list(O.glob('*.json')))}}
(OUT/'receipt.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
print(json.dumps({k:result[k] for k in ('status','raw','strict_replay_output_records','native','resource_summary','cleanup')},indent=2))
