"""CPU-only audit of saved actual native-feedback receipts; no native library load."""
from pathlib import Path
import ast,hashlib,json,math,re,struct,sys
P=Path('/private/tmp/minivllm-engine-optimization-20261014');O=P/'v23-root-control-build1/fullmodel-v34native-attempt1';D=O/'child-output';OUT=Path(__file__).parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
r=read(D/'receipt.json')
assert sha(D/'receipt.json')=='ff67fbc7669244ec061d0203540caafcc9d9bcffb1789b1f6e412431980730d5'
assert r['status']=='PASS_NATIVE_FIXEDCAP_FEEDBACK_BASE_RESET_CHANGED_STRICT' and r['stage']=='complete'
assert r['native_steps']==93 and r['model_loads']==1 and r['source_pins_verified_final'] and r['fixed_cap']==256
assert not any(r[k] for k in ('performance_point','ICB_claim','adopted'))
assert r['gates_on']==r['gates_callable_count']==48 and r['conv_off'] and r['whole_functional_off']
assert len(r['source_pins'])==43
for p,s in r['source_pins'].items():assert sha(p)==s,p
for p in r['actual_imported_sources'].values():assert p in r['source_pins'] and 'candidate-v17-gdn-gates-build1-root/' in p
worker=P/'v34-native-fullqual-build1/fullqual.py';tree=ast.parse(worker.read_text())
loop=[x for x in ast.walk(tree) if isinstance(x,ast.For) and isinstance(x.target,ast.Name) and x.target.id=='step'][0]
calls=[x.args[0].value for x in ast.walk(loop) if isinstance(x,ast.Call) and isinstance(x.func,ast.Attribute) and x.func.attr=='call' and x.args and isinstance(x.args[0],ast.Constant)]
assert set(calls)=={'v34_replay_step','v34_compare_output','v34_copy_output'}
assert 'v33_restore_replay' not in worker.read_text()
cap=r['capture'];assert len(cap['inputs'])==131 and len(cap['outputs'])==132 and cap['parameter_arrays']==1349
byIn={x['address']:x for x in cap['inputs']};byOut={x['address']:x for x in cap['outputs']}
assert len(byIn)==131 and len(byOut)==132
for x in cap['inputs']+cap['outputs']:
 assert x['finite'] and x['memoryview_c_contiguous'] and x['numpy_api']=='frombuffer; no copy fallback'
 assert x['nbytes']==math.prod(x['shape'])*{'mlx.core.bfloat16':2,'mlx.core.float32':4,'mlx.core.int32':4}[x['dtype']]
inputbytes=sum(x['nbytes'] for x in byIn.values());outputbytes=sum(x['nbytes'] for x in byOut.values())
n=read(D/'native-sealed.json');assert cap['native']==n and n['status']=='OK' and n['arena_built'] and n['error']==''
buffers={x['original_buffer']:x for x in n['buffers']};assert len(buffers)==1854
small=[];zero=[]
for b in buffers.values():
 assert b['used'] and b['length']>0 and b['original_contents']>0
 if b['zero_copy']:
  assert b['snapshot_bytes']==0 and b['pinned_parameter_candidate']
  m=re.fullmatch(r'(parameter\..+) full-range=(\d+)\+(\d+) binding-delta=(\d+) readonly:(.+)/([^/]+)',b['role']);assert m,b
  _,addr,size,delta,_,_=m.groups();addr,size,delta=int(addr),int(size),int(delta)
  assert b['original_contents']<=addr and addr+size<=b['original_contents']+b['length'] and 0<=delta<size
  zero.append(b)
 else:
  assert not b['pinned_parameter_candidate'] and not b['large'] and b['snapshot_bytes']==b['length']
  assert b['arena_offset']%256==0 and b['arena_offset']+b['length']<=n['arena_bytes'];small.append(b)
assert len(zero)==n['parameter_ranges_registered']==1349 and len(small)==505
assert sum(x['length'] for x in zero)==n['zero_copy_buffer_bytes']==15133597696
assert sum(x['length'] for x in small)==n['snapshot_bytes']==366067053
for ranges in (sorted((x['original_contents'],x['original_contents']+x['length']) for x in buffers.values()),sorted((x['arena_offset'],x['arena_offset']+x['length']) for x in small)):
 assert all(a[1]<=b[0] for a,b in zip(ranges,ranges[1:]))
fb=n['feedback'];assert fb['sealed'] and (fb['input_count'],fb['output_count'],fb['link_count'],fb['state_count'],fb['capacity'])==(131,132,131,128,256)
ni={x['original_address']:x for x in fb['inputs']};no={x['address']:x for x in n['outputs']};ns={x['output_address']:x for x in fb['staged_outputs']}
assert set(ni)==set(byIn) and set(no)==set(ns)==set(byOut) and fb['input_output_aliases']==[]
assert sorted(x['role'] for x in ni.values())==[0]*128+[1,2,3]
for addr,x in ni.items():
 e=byIn[addr];b=buffers[x['original_buffer']]
 assert not b['zero_copy'] and x['bytes']==e['nbytes'] and x['label']==e['label']
 assert b['original_contents']+x['original_offset']==addr and b['arena_offset']+x['original_offset']==x['arena_offset']
 assert x['original_offset']>=0 and x['original_offset']+x['bytes']<=b['length']
for addr,x in no.items():
 e=byOut[addr];b=buffers[x['original_buffer']];t=ns[addr]
 assert not b['zero_copy'] and x['bytes']==e['nbytes']==t['bytes'] and x['label']==e['label']
 assert b['original_contents']+x['original_offset']==addr and x['original_offset']+x['bytes']<=b['length']
 assert b['original_contents']<=x['captured_writable_binding_start']<=addr and t['stage_offset']%16==0 and t['stage_offset']+t['bytes']<=fb['stage_bytes']
for ranges in (sorted((x['arena_offset'],x['arena_offset']+x['bytes']) for x in ni.values()),sorted((x['stage_offset'],x['stage_offset']+x['bytes']) for x in ns.values())):
 assert all(a[1]<=b[0] for a,b in zip(ranges,ranges[1:]))
links={x['input_address']:x for x in fb['links']};assert set(links)==set(byIn) and len({x['output_address'] for x in links.values()})==131
for i,x in enumerate(cap['inputs']):
 target=cap['outputs'][(129,130,131)[i] if i<3 else i-2]
 assert links[x['address']]=={'input_address':x['address'],'output_address':target['address'],'bytes':x['nbytes']}
commits=n['commit_records'];assert len(commits)==n['original_CB_boundaries']==85 and len({x['CB_id'] for x in commits})==85
assert len({x['queue'] for x in commits})==1 and sum(not x['proven_empty_same_queue'] for x in commits)==n['captured_CB_count']==84
for c in commits:
 assert not c['duplicate'] and c['commit_depth']==c['parent_commit_CB_id']==c['parent_commit_CB_pointer']==c['lifetime_observed_unknown_commands']==0
 if c['proven_empty_same_queue']:assert c['active_CB_id']==c['active_CB_pointer']==c['lifetime_observed_compute_encoders']==c['lifetime_observed_event_commands']==0
 else:assert c['active_CB_id']==c['CB_id'] and c['active_CB_pointer']==c['CB_pointer'] and c['lifetime_observed_compute_encoders']+c['lifetime_observed_event_commands']>0
seen={};fencecounts={'update':0,'wait':0}
for f in n['fence_events']:
 fencecounts[f['operation']]+=1
 assert not f['pre_capture_completed_proof'] and f['native_lowering']=='full buffer scope barrier'
 if f['operation']=='update':seen[f['original_fence']]=f['producer_CB_id']
 else:assert seen[f['original_fence']]==f['producer_CB_id']
assert fencecounts=={'update':84,'wait':10} and not n['original_events_resignaled_during_replay']
assert len(n['event_commands'])==1 and n['event_commands'][0]['operation']=='signal'
ids=read(P/'workload.json')['continuation_ids'];baseprompt=read(P/'workload.json')['prompt_ids']
assert r['declared_prompts']=={'base':baseprompt,'reset-base':baseprompt,'changed-prefill':baseprompt[:10]+[baseprompt[3]]+baseprompt[10:]}
assert [x['name'] for x in r['requests']]==['base','reset-base','changed-prefill']
strict=rocount=0
for j,req in enumerate(r['requests'],1):
 assert req['status']=='PASS_REQUEST_STRICT' and req['prompt_ids']==r['declared_prompts'][req['name']] and req['prompt_length']==len(req['prompt_ids'])
 assert req['A_ids']==req['B_ids'] and len(req['A_ids'])==32 and len(req['steps'])==31 and req['seed_upload_calls']==req['seed_arrays']==131
 assert req['first_token']==req['A_ids'][0] and not req['terminated_by_EOS'] and not set(req['A_ids'])&set(r['eos_ids'])
 if j<3:assert req['A_ids']==ids and req['changed_seed_inputs_vs_capture']==[]
 else:assert len(req['changed_seed_inputs_vs_capture'])==129 and len(set(req['changed_seed_inputs_vs_capture']))==129 and any(x>=3 for x in req['changed_seed_inputs_vs_capture'])
 after=read(D/f"native-after-{req['name']}.json");assert req['native']==after
 assert {k:v for k,v in after.items() if k not in ('native_replays','feedback')}=={k:v for k,v in n.items() if k not in ('native_replays','feedback')}
 f=after['feedback'];assert f['request_steps']==31 and f['total_steps']==after['native_replays']==j*31 and f['resets']==j and f['request_seed_uploaded_ranges']==131
 assert f['CPU_seed_upload_bytes_total']==inputbytes*j and f['CPU_output_compare_bytes_total']==outputbytes*31*j and f['CPU_full_state_restore_per_step'] is False
 assert f['GPU_blit_bytes_per_step']==inputbytes+outputbytes and all(v==0 for v in f['first_difference'].values())
 for i,s in enumerate(req['steps'],1):
  assert s['step']==i and s['A_exact_L']==len(req['prompt_ids'])+i and s['B_fixed_L']==256 and s['A_token']==s['B_token']==req['A_ids'][i]
  assert s['seed_uploads_this_step']==0 and s['native_prediction_copied_bytes']==4 and s['reference_input_digest_readonly']
  assert [x['label'] for x in s['outputs']]==[x['label'] for x in cap['outputs']]
  for x,c in zip(s['outputs'],cap['outputs']):
   assert x['finite'] and x['strict_bytes_equal'] and x['shape']==c['shape'] and x['dtype']==c['dtype'] and x['bytes']==c['nbytes'] and len(x['A_sha256'])==64;strict+=1
  for label,value in (('prediction',s['A_token']),('next_positions',s['A_exact_L']),('next_lens',s['A_exact_L']+1)):
   x=next(x for x in s['outputs'] if x['label']==label);assert hashlib.sha256(struct.pack('<i',value)).hexdigest()==x['A_sha256']
  for k,count,b in (('seed',131,inputbytes),('capture_outputs',132,outputbytes),('request_seed',131,inputbytes)):
   ro=s['originals_readonly'][k];assert ro['strict_bytes_equal'] and ro['arrays']==count and ro['bytes']==b;rocount+=1
assert strict==12276 and rocount==279
for a,b in zip(r['requests'][0]['steps'],r['requests'][1]['steps']):assert a['outputs']==b['outputs']
assert r['requests'][2]['steps'][0]['outputs'][0]['A_sha256']!=r['requests'][0]['steps'][0]['outputs'][0]['A_sha256']
allresources=[];resources={}
for base,pattern,derived in ((O,'resource-???-raw.json','resource-derived.json'),(D,'resource-??-raw.json','resources-derived.json'),(O,'external-resource-??-raw.json','external-resource.json')):
 ps=sorted(base.glob(pattern));ds=read(base/derived);assert len(ps)==len(ds)
 stats=[]
 for j,(p,d) in enumerate(zip(ps,ds)):
  raw=read(p);mutable={'known_inference_processes','known_inference_process_count','only_owned_known_inference_processes','exactly_one_owned_known_model'}
  for k,v in raw.items():
   if k not in mutable:assert d[k]==v,(p,k)
  assert int(raw['pressure_level']['text'])==d['pressure_level_value']==1
  assert int(re.search(r'free percentage: (\d+)%',raw['memory_pressure']['text'])[1])==d['free_percent']
  assert float(re.search(r'used = ([0-9.]+)M',raw['swap']['text'])[1])==d['swap_used_mb']
  for k in ('Swapouts','Swapins','Pageouts','Compressions','Decompressions'):assert int(re.search(r'^'+k+r':\s+(\d+)\.',raw['vm_stat']['text'],re.M)[1])==d['vm_counters'][k]
  assert d['process_inventory_ok'] and d['only_owned_known_inference_processes'] and all(x['owned'] and x['pid']==r['pid'] for x in d['known_inference_processes'])
  assert raw['power_metadata']['status']=='known' and raw['power_metadata']['source']=='AC Power' and "'AC Power'" in raw['power_metadata']['raw']['stdout']
  assert raw['processinfo_thermal_metadata']['status']=='known' and int(raw['processinfo_thermal_metadata']['raw']['stdout'])==raw['processinfo_thermal_metadata']['code']==0
  if j:
   prev=ds[j-1];assert d['swap_used_mb_delta']==d['swap_used_mb']-prev['swap_used_mb']==0
   assert d['vm_counter_delta']=={k:v-prev['vm_counters'][k] for k,v in d['vm_counters'].items() if k in prev['vm_counters']} and d['vm_counter_delta']['Swapouts']==0
  stats.append({'file':str(p),'sha256':sha(p)});allresources.append(d)
 resources[str(base/derived)]={'samples':len(ps),'raw_manifest_sha256':digest(stats)}
assert sum(x['samples'] for x in resources.values())==54
assert sha(D/r['afterload']['raw_file'])==r['afterload']['raw_sha256']
own=read(O/'owner-receipt.json');rel=read(O/'external-release.json')
assert own['owner_pid']==56896 and own['child_pid']==r['pid']==56905 and own['returncode']==0 and own['waited_reaped'] and own['stage']=='reaped'
assert own['absence']=={'returncode':1,'stdout':'','stderr':''}
assert own['script_sha256']==r['source_pins'][own['command'][2]] and own['native_library_sha256']==r['source_pins'][r['native_library']]
assert own['native_source_sha256']==r['source_pins'][str(P/'v34-native-replay-build1/replay.mm')]
assert rel['status']=='PASS_EXTERNAL_RELEASE' and rel['tool_exit']==0 and rel['GPU_released'] and {x['pid'] for x in rel['strict_absence']}=={56896,56905}
assert all(x['returncode']==1 and x['stdout']==x['stderr']=='' for x in rel['strict_absence'])
assert not any(x=='mlx' or x.startswith('mlx.') for x in sys.modules)
result={'status':'PASS_SAVED_V34_NATIVE_FEEDBACK_EVIDENCE_CPU_AUDIT','receipt_sha256':sha(D/'receipt.json'),'scope':'Saved source/counter/A-hash/Cmemcmp audit; no independent GPU recomputation or persisted full-output raw recapture. Actual native GPU feedback base/reset/changed prompt qualification only; not engine/service/performance/ICB.','source_pins':43,'requests':3,'generated_tokens':96,'decode_steps':93,'strict_output_records':strict,'original_readonly_records':rocount,'A_retained_state_digest_records':93,'input_ranges':131,'staged_output_ranges':132,'links':131,'changed_seed_inputs':129,'CPU_B_seed_upload_calls_total':393,'CPU_B_seed_upload_bytes_total':inputbytes*3,'CPU_B_seed_uploads_per_step':0,'B_prediction_observation_bytes_total':4*93,'CPU_compare_bytes_total':outputbytes*93,'GPU_blit_bytes_per_step':inputbytes+outputbytes,'native':{'buffer_count':1854,'zero_copy_parameter_allocations':1349,'small_zero_copy_allocations':sum(not b['large'] for b in zero),'mutable_allocations':505,'snapshot_bytes':n['snapshot_bytes'],'arena_bytes':n['arena_bytes'],'stage_bytes':fb['stage_bytes'],'zero_copy_parameter_bytes':n['zero_copy_buffer_bytes'],'actual_input_output_aliases':0,'dispatches':n['dispatches'],'commit_records':85,'fence_records':fencecounts},'resources':resources,'resource_summary':{'samples':54,'AC':True,'thermal_codes':sorted({x['processinfo_thermal_metadata']['code'] for x in allresources}),'pressure_levels':sorted({x['pressure_level_value'] for x in allresources}),'minimum_free_percent':min(x['free_percent'] for x in allresources),'swap_MB_growth':0,'Swapouts_growth':0,'Pageouts_growth_owner':read(O/'resource-derived.json')[-1]['vm_counters']['Pageouts']-read(O/'resource-derived.json')[0]['vm_counters']['Pageouts']},'cleanup':{'owner':56896,'child':56905,'waited_reaped_exit0':True,'saved_external_strict_absence':True},'metadata_sha256':{str(p):sha(p) for p in sorted(list(D.glob('*.json'))+list(O.glob('*.json')))}}
(OUT/'receipt.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('metadata_sha256','resources')},indent=2))
