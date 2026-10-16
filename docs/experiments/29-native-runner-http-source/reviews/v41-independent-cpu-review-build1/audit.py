"""Saved V41 runner evidence audit; CPU/std library only, no native or MLX load."""
from pathlib import Path
import ast,hashlib,json,re,sys
P=Path('/private/tmp/minivllm-engine-optimization-20261014');OUT=Path(__file__).parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
digest=lambda x:hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def resources(O,D,child):
 allres=[];files={}
 for base,pattern,derived in ((O,'resource-???-raw.json','resource-derived.json'),(D,'resource-??-raw.json','resources-derived.json'),(O,'external-resource-??-raw.json','external-resource.json')):
  ps=sorted(base.glob(pattern))
  if not ps:continue
  ds=read(base/derived);assert len(ps)==len(ds)
  for j,(p,d) in enumerate(zip(ps,ds)):
   raw=read(p);mutable={'known_inference_processes','known_inference_process_count','only_owned_known_inference_processes','exactly_one_owned_known_model'}
   for k,v in raw.items():
    if k not in mutable:assert d[k]==v,(p,k)
   assert int(raw['pressure_level']['text'])==d['pressure_level_value']==1 and int(re.search(r'free percentage: (\d+)%',raw['memory_pressure']['text'])[1])==d['free_percent']
   assert float(re.search(r'used = ([0-9.]+)M',raw['swap']['text'])[1])==d['swap_used_mb']
   for k in ('Swapouts','Swapins','Pageouts','Compressions','Decompressions'):assert int(re.search(r'^'+k+r':\s+(\d+)\.',raw['vm_stat']['text'],re.M)[1])==d['vm_counters'][k]
   assert d['process_inventory_ok'] and d['only_owned_known_inference_processes'] and all(x['owned'] and x['pid']==child for x in d['known_inference_processes'])
   assert raw['power_metadata']['status']=='known' and raw['power_metadata']['source']=='AC Power' and "'AC Power'" in raw['power_metadata']['raw']['stdout']
   assert raw['processinfo_thermal_metadata']['status']=='known' and int(raw['processinfo_thermal_metadata']['raw']['stdout'])==raw['processinfo_thermal_metadata']['code']==0
   if j:
    prev=ds[j-1];assert d['swap_used_mb_delta']==d['swap_used_mb']-prev['swap_used_mb']==0
    assert d['vm_counter_delta']=={k:v-prev['vm_counters'][k] for k,v in d['vm_counters'].items() if k in prev['vm_counters']} and d['vm_counter_delta']['Swapouts']==0
   allres.append(d)
  files[str(base/derived)]={'samples':len(ps),'raw_manifest_sha256':digest([{'file':str(p),'sha256':sha(p)} for p in ps]),'Pageouts_growth':ds[-1]['vm_counters']['Pageouts']-ds[0]['vm_counters']['Pageouts']}
 return {'samples':len(allres),'AC':True,'thermal_codes':sorted({x['processinfo_thermal_metadata']['code'] for x in allres}),'pressure_levels':sorted({x['pressure_level_value'] for x in allres}),'minimum_free_percent':min(x['free_percent'] for x in allres),'swap_MB_growth':0,'Swapouts_growth':0,'files':files}

def cleanup(O,owner,child):
 a=read(O/'owner-receipt.json');b=read(O/'external-release.json');assert a['owner_pid']==owner and a.get('child_pid',a.get('driver_pid'))==child and a['stage']=='reaped' and a['returncode']==0 and a['waited_reaped']
 assert a['absence']['returncode']==1 and a['absence']['stdout']==a['absence']['stderr']==''
 assert b['status']=='PASS_EXTERNAL_RELEASE' and b['GPU_released'] and b['tool_exit']==0 and {x['pid'] for x in b['strict_absence']}=={owner,child}
 assert all(x['returncode']==1 and x['stdout']==x['stderr']=='' for x in b['strict_absence'])
 return {'owner':owner,'child':child,'reaped_exit0':True,'saved_external_strict_absence':True}

O=P/'v23-root-control-build1/fullmodel-v41native-attempt1';D=O/'child-output';r=read(D/'receipt.json');A=P/'v41-native-runner-build1'
assert sha(D/'receipt.json')=='306d56d5bf2a66f2233f11f521d6016c6432982733ea2c95967d9534f52d3a49'
assert sha(A/'native_runner.py')=='bf7bea08b09dfc2905966b95e77d244ec25947a0c3a0d9506cddd05ca81cfea0'
assert sha(A/'fullqual.py')=='8822c32da9289ad0fd3c593449d9a5c09b1f3db3ab4bdefc19cc883af243804d'
assert r['status']=='PASS_NATIVE_RUNNER_C1_QUEUED_RESET_FALLBACK_STRICT' and r['stage']=='complete'
assert len(r['source_pins'])==83
for p,h in r['source_pins'].items():assert sha(p)==h,p
q=read(P/'v23-root-control-build1/fullmodel-v39native-attempt1/child-output/receipt.json')
assert r['prior_qualification_sha256']==sha(P/'v23-root-control-build1/fullmodel-v39native-attempt1/child-output/receipt.json')
assert r['model_loads']==1 and r['scheduler_max_seqs']==1 and not r['performance_point'] and not r['adopted']
assert len(r['requests'])==3 and [x['name'] for x in r['requests']]==[x['name'] for x in q['requests']]
checks=0
for row,ref in zip(r['requests'],q['requests']):
 assert row['B_ids']==ref['A_ids'] and len(row['B_ids'])==32 and row['status']=='PASS_32_GREEDY_AND_DROP' and row['captured_originals_readonly']
 assert len(row['steps'])==31
 for i,(step,expected) in enumerate(zip(row['steps'],ref['steps']),1):
  assert step['step']==i and len(step['checks'])==132
  for actual,e in zip(step['checks'],expected['outputs']):
   assert actual=={'label':e['label'],'bytes':e['bytes'],'sha256':e['A_sha256'],'qualified_A_sha256':e['A_sha256']};checks+=1
assert checks==12276 and r['core_all132_steps']==93 and r['first_prefill_tokens']==3
assert r['build_input131_readonly'] and r['captured_originals_readonly'] and r['terminal_cleanup']
core=r['core_native_after_three_queued_requests']['feedback'];assert core['resets']==3 and core['total_steps']==93 and core['icb']['execution_count']==93 and core['icb']['bank_executions']==[48,45]
ids=read(P/'workload.json')['continuation_ids']
assert r['whole_request_fallback_ids']==[ids[:2]] and r['prefill_only_ids']==[ids[:1]] and r['post_fallback_ids']==[ids]
assert r['native_steps']==124 and r['request_count']==4 and r['ordinary_warm_tokens']==32
assert len(r['post_fallback_final132'])==132
for actual,e in zip(r['post_fallback_final132'],q['requests'][0]['steps'][-1]['outputs']):
 assert actual=={'label':e['label'],'sha256':e['A_sha256'],'qualified_A_sha256':e['A_sha256']};checks+=1
assert checks==r['strict_output_checks']==12408
f=r['native']['feedback'];assert r['native']['status']=='OK' and f['resets']==4 and f['request_steps']==31 and f['total_steps']==124
assert f['icb']['execution_count']==f['icb']['encoded_count']==124 and f['icb']['bank_executions']==[64,60] and f['pingpong']['eligible_count']==48
assert f['CPU_seed_upload_bytes_total']==4*170721292 and f['CPU_full_state_restore_per_step'] is False
adapter=(A/'native_runner.py').read_text();at=ast.parse(adapter)
run=next(x for x in ast.walk(at) if isinstance(x,ast.FunctionDef) and x.name=='run')
calls=[x.args[0].value for x in ast.walk(run) if isinstance(x,ast.Call) and isinstance(x.func,ast.Attribute) and x.func.attr=='call' and x.args and isinstance(x.args[0],ast.Constant)]
assert calls==['v34_replay_step','v34_copy_output']
# Real runner API and terminal lifecycle paths were read, not replaced by CPU fake engines.
cache=P/'candidate-v39-conv-build1-root/tutorial_code/miniengine/cache.py';ct=ast.parse(cache.read_text());batch=next(x for x in ct.body if isinstance(x,ast.ClassDef) and x.name=='DecodeBatch');assert any(getattr(x,'name',None)=='layers_len' for x in batch.body)
assert not any(x=='mlx' or x.startswith('mlx.') for x in sys.modules)
result={'status':'PASS_SAVED_V41_C1_RUNNER_EVIDENCE_CPU_AUDIT_NO_CONCRETE_SOURCE_BLOCKER','scope':'Fixed C1 max_seqs1/non-spec/greedy native ownership until drop; no active handoff to ordinary stale seed cache. 93 core steps have all132 strict hashes; post-fallback fourth native request has all32IDs and final132 hashes only. No HTTP/performance/adoption qualification or independent GPU recapture.','receipt_sha256':sha(D/'receipt.json'),'source_pins_verified':83,'adapter_sha256':sha(A/'native_runner.py'),'worker_sha256':sha(A/'fullqual.py'),'core_queued_requests':3,'core_IDs':96,'core_all132_steps':93,'core_strict_hash_checks':12276,'post_fallback_native_steps':31,'post_fallback_final132_checks':132,'strict_hash_checks_total':12408,'native_steps_total':124,'native_requests':4,'ICB_bank_executions':[64,60],'whole_request_fallback':'temperature.7/top_k1,2IDs,0 native steps','prefill_only':'max_tokens1,1ID,0 native steps','readonly_scope':'131 captured inputs before original capture plus retained263 original input/output arrays checked at request completion','resources':resources(O,D,65871),'cleanup':cleanup(O,65862,65871),'metadata_sha256':{str(p):sha(p) for root in (O,D) for p in sorted(root.glob('*.json'))}}
(OUT/'receipt.json').write_text(json.dumps(result,indent=2,sort_keys=True)+chr(10))
print(json.dumps({k:v for k,v in result.items() if k not in ('metadata_sha256','resources')},indent=2));print('resources',result['resources'])
