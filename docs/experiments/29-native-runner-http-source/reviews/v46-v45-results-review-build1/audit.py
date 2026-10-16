"""CPU-only replay of saved V46 numerical and V45 stopped HTTP evidence."""
from pathlib import Path
import ast,hashlib,json,math,re,statistics
P=Path('/private/tmp/minivllm-engine-optimization-20261014');D=P/'v46-v45-results-review-build1';D.mkdir(exist_ok=True)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
digest=lambda x:hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
helper=ast.parse((P/'v38-results-review-build1/audit.py').read_text());exec(compile(ast.Module(body=[n for n in helper.body if isinstance(n,ast.FunctionDef) and n.name in ('resources','cleanup')],type_ignores=[]),'saved-resource-parser','exec'))
Q=P/'v23-root-control-build1/fullmodel-v46native-attempt1';QD=Q/'child-output';q=read(QD/'receipt.json');ref=read(P/'v23-root-control-build1/fullmodel-v44native-attempt1/child-output/receipt.json');w=read(P/'workload.json');ids=w['continuation_ids']
assert sha(QD/'receipt.json')=='ab6987a4d90c1faf28fe0fbc93c427bed0f50f0e0b94e04a829e2ae83f067252'
assert q['status']=='PASS_NATIVE_RUNNER_C1_QUEUED_RESET_FALLBACK_STRICT' and q['stage']=='complete' and q['prior_qualification_sha256']=='1dcfc41e7779c575602bd6c73bae4e4800664db42e1ee87e75414335bdc45b6d'
for p,h in q['source_pins'].items():assert sha(p)==h,p
assert len(q['source_pins'])==58 and q['strict_output_checks']==12408 and q['core_all132_steps']==93 and q['native_steps']==124 and q['request_count']==4
assert q['terminal_cleanup'] and q['captured_originals_readonly'] and q['build_input131_readonly']
assert q['whole_request_fallback_ids']==[ids[:2]] and q['prefill_only_ids']==[ids[:1]] and q['post_fallback_ids']==[ids]
checks=0
for req,old in zip(q['requests'],ref['requests']):
 assert req['name']==old['name'] and req['B_ids']==old['A_ids'] and len(req['steps'])==31 and req['captured_originals_readonly'] and req['status']=='PASS_32_GREEDY_AND_DROP'
 for step,rstep in zip(req['steps'],old['steps']):
  assert step['step']==rstep['step'] and len(step['checks'])==132
  for x,y in zip(step['checks'],rstep['outputs']):
   assert x=={'label':y['label'],'bytes':y['bytes'],'sha256':y['A_sha256'],'qualified_A_sha256':y['A_sha256']};checks+=1
for x,y in zip(q['post_fallback_final132'],ref['requests'][0]['steps'][-1]['outputs']):
 assert x=={'label':y['label'],'sha256':y['A_sha256'],'qualified_A_sha256':y['A_sha256']};checks+=1
assert checks==12408
for k in ('gates_on','conv_on','normreuse_cache_on','conv_normreuse_cache_on'):assert q[k]==48
assert q['whole_functional_off'] and not q['conv_off'] and q['adapter_sha256']=='bf7bea08b09dfc2905966b95e77d244ec25947a0c3a0d9506cddd05ca81cfea0'
f=q['native']['feedback'];assert q['native']['status']=='OK' and f['total_steps']==124 and f['resets']==4 and f['icb']['execution_count']==124 and f['icb']['bank_executions']==[64,60] and f['pingpong']['eligible_count']==48
full={'receipt_sha256':sha(QD/'receipt.json'),'source_pins':58,'strict_hash_records':checks,'core_native_steps':93,'total_native_steps':124,'resets':4,'banks':[64,60],'resources':resources(Q,QD,69505),'cleanup':cleanup(Q,69496,69505),'coverage':'queued3 every132 each step; fallback+max1; reentry32 IDs and final132 only'}
V=P/'v45-native-http-screen-build1';O=P/'v23-root-control-build1/native-http-v45-attempt1';C=O/'child-output';pins=read(V/'pins.json');config=read(V/'runtime-config.json');failure=read(C/'suite-failure.json');owner=read(O/'owner-receipt.json');release=read(O/'external-release.json');allabs=read(O/'external-all-server-absence.json')
assert sha(V/'pins.json')=='56564d419fcff6cc5873083bda9514cc1add1f0ceb7cb86d213e6edcb6326b7e'
assert sha(V/'driver.py')==owner['driver_sha256']=='212757fa69a62168b32e80f6a8968a1eac01179098d6f82d50b4cc95899a4321'
for p,h in pins['files'].items():assert sha(p)==h,p
for p,h in config['source_pins'].items():assert sha(p)==h,p
assert len(pins['files'])==14 and len(config['source_pins'])==61 and config['fullqual_receipt_sha256']==full['receipt_sha256'] and config['mode']=='conv_normreuse_cache'
assert config['native_library_sha256']=='8ef20b52e80cd0798e80abf2037c98211f13437dbf5ea4ddb47dce1b00ff471b'
old=(P/'m4-final-native15-official-attempt2/client.py').read_text();new=(V/'driver.py').read_text()
for name in ('request','group'):
 fn=lambda s:next(n for n in ast.walk(ast.parse(s)) if isinstance(n,ast.FunctionDef) and n.name==name)
 assert ast.dump(fn(old),include_attributes=False)==ast.dump(fn(new),include_attributes=False)
assert 'signal.SIGTERM' in new and 'KeyboardInterrupt' in new
assert not (C/'suite-complete.json').exists() and not (C/'owner-cleanup.json').exists()
assert failure['error']['message']=='unexpected power source: Battery Power' and len(failure['completed'])==4
completed=[];groupcount=0;native_raw_ids=0;allresources=[];powerchanges=[]
for A in sorted(p for p in C.iterdir() if p.is_dir()):
 for gp in sorted(A.glob('C1-*.json')):
  g=read(gp);assert g['actual_request_count']==1 and len(g['requests'])==1 and g['total_completion_tokens']==32
  r=g['requests'][0];start=r['start_perf_counter_s'];end=r['done_perf_counter_s'];first=r['first_text_perf_counter_s'];assert start<first<end
  assert r['client_total_s']==end-start and r['client_ttft_text_s']==first-start and r['client_post_first_chunk_s']==end-first
  assert r['usage']['prompt_tokens']==19 and r['usage']['completion_tokens']==32 and r['usage'].get('prompt_tokens_details',{}).get('cached_tokens',0)==0 and r['finish_reason']=='length'
  assert len(r['visible_roundtrip_ids'])==32 and r['full_visible_roundtrip_count_matches_usage'] and r['first_chunk_prefix_verified']
  assert g['group_window_s']==end-start and g['y_total_output_throughput_tok_s_mac']==32/(end-start)
  assert r['x_client_estimated_interactivity_tok_s_user']==(32-r['first_chunk_reconstructed_tokens'])/(end-first)==g['x_client_estimated_interactivity_tok_s_user']
  if 'native' in A.name:assert r['output_ids_when_available']==r['visible_roundtrip_ids']==ids;native_raw_ids+=32
  groupcount+=1
 if (A/'result.json').exists():
  r=read(A/'result.json');assert len(r['warmups'])==len(r['runs'])==3
  stats={'x':statistics.median(x['x_client_estimated_interactivity_tok_s_user'] for x in r['runs']),'y':statistics.median(x['y_total_output_throughput_tok_s_mac'] for x in r['runs']),'ttft_s':statistics.median(x['requests'][0]['client_ttft_text_s'] for x in r['runs']),'E2E_s':statistics.median(x['requests'][0]['client_total_s'] for x in r['runs'])}
  for k,v in stats.items():assert r[k]==v
  completed.append({'label':A.name,'engine':r['engine'],**stats})
 else:assert A.name=='C1-B2-lithos' and len(list(A.glob('C1-measure-*.json')))==2
 cleanuprow=read(A/'cleanup.json');exitrow=read(A/'owned-process-exit.json');assert exitrow==cleanuprow['owned_exit'] and exitrow['wait_completed'] and exitrow['returncode']==-15 and exitrow['exact_pid_absence']['returncode']==1
 assert exitrow['exact_pid_absence']['stdout']==exitrow['exact_pid_absence']['stderr']=='' and all(x['decision']=='SAME_IDENTITY_SIGNAL' for x in exitrow['signal_identity_decisions'])
 assert read(A/'port-absence.json')['absent']
 rs=read(A/'resources.json');raws=sorted(A.glob('resources-??-raw.json'));assert len(rs)==len(raws)
 for d,p in zip(rs,raws):
  raw=read(p)
  assert raw['label']==d['label'] and raw['power_metadata']==d['power_metadata'] and raw['processinfo_thermal_metadata']==d['processinfo_thermal_metadata']
  assert int(raw['pressure_level']['text'])==d['pressure_level_value']==1
  assert int(re.search(r'free percentage: (\d+)%',raw['memory_pressure']['text'])[1])==d['free_percent']
  assert float(re.search(r'used = ([0-9.]+)M',raw['swap']['text'])[1])==d['swap_used_mb']
  for k in ('Swapouts','Swapins','Pageouts','Compressions','Decompressions'):assert int(re.search(r'^'+k+r':\s+(\d+)\.',raw['vm_stat']['text'],re.M)[1])==d['vm_counters'][k]
  assert int(raw['processinfo_thermal_metadata']['raw']['stdout'])==d['processinfo_thermal_metadata']['code']==0
  assert d['process_inventory_ok'] and d['only_owned_known_inference_processes'] and all(x['owned'] and x['pid']==exitrow['pid'] for x in d['known_inference_processes'])
  if d['power_metadata']['source']=='Battery Power':powerchanges.append({'arm':A.name,'label':d['label'],'raw':str(p),'sha256':sha(p)})
  else:assert d['power_metadata']['source']=='AC Power'
  allresources.append(d)
 assert rs[-1]['swap_used_mb']==rs[0]['swap_used_mb'] and rs[-1]['vm_counters']['Swapouts']==rs[0]['vm_counters']['Swapouts']
assert sorted(completed,key=lambda x:x['label'])==sorted(failure['completed'],key=lambda x:x['label']) and groupcount==29 and native_raw_ids==192
assert powerchanges[0]['arm']=='C1-B2-lithos' and powerchanges[0]['label']=='after-C1-measure-1' and len(powerchanges)==2
nf=read(C/'C1-B1-native/native-final-metadata.json');warm=read(C/'C1-B1-native/warm-metadata.json');assert nf['active'] is None and nf['native_steps']==186 and nf['request_count']==6 and warm['native_steps']==93
f=nf['native']['feedback'];assert f['resets']==6 and f['icb']['execution_count']==186 and f['icb']['bank_executions']==[96,90] and f['pingpong']['eligible_count']==48
assert owner['owner_pid']==70057 and owner['driver_pid']==70059 and owner['returncode']==1 and owner['stage']=='reaped' and owner['waited_reaped']
assert release['status']=='PASS_EXTERNAL_RELEASE' and release['GPU_released'] and release['tool_exit']==1 and release['result']=='STOP_AC_TO_BATTERY_C1_CAMPAIGN_4_OF_9_COMPLETE'
assert {x['pid'] for x in release['strict_absence']}=={70057,70059} and all(x['returncode']==1 and x['stdout']==x['stderr']=='' for x in release['strict_absence'])
assert len(allabs['rows'])==5 and {x['pid'] for x in allabs['rows']}=={70078,70257,70441,70625,70811} and len({x['port'] for x in allabs['rows']})==5 and not allabs['formal_campaign_qualified']
assert all(x['waited_returncode']==-15 and x['external_absence']=={'returncode':1,'stdout':'','stderr':''} and x['port_connect_ex']==61 for x in allabs['rows'])
result={'status':'PASS_SAVED_V46_NUMERICAL_AND_V45_STOP_EVIDENCE_CPU_AUDIT','scope':'Saved records/source/time/resource arithmetic only; no independent GPU recomputation. V45 STOP remains unqualified; completed4 descriptive only, incomplete B2 Lithos two measured groups excluded. No AC/Battery pooling or C1/Pareto victory.','V46':full,'V45':{'controller_sha256':sha(V/'driver.py'),'pins_sha256':sha(V/'pins.json'),'controller_pins':14,'runtime_source_pins':61,'request_group_AST_equal':True,'completed4_descriptive':failure['completed'],'completed_arms':4,'planned_arms':9,'saved_HTTP_groups':groupcount,'saved_HTTP_tokens':groupcount*32,'native_raw_ID_tokens':native_raw_ids,'native_completed_steps':186,'native_resets':6,'first_bad_power':powerchanges[0],'saved_Battery_samples_in_arm':len(powerchanges),'per_arm_raw_samples':len(allresources),'min_free_percent':min(x['free_percent'] for x in allresources),'thermal_codes':[0],'pressure_levels':[1],'swap_MB_growth_each_arm':0,'Swapouts_growth_each_arm':0,'owner_cleanup_json_exists':False,'suite_complete_exists':False,'seven_PID_five_port_strict_absence':True,'controller_exit':1,'release_sha256':sha(O/'external-release.json'),'all_server_release_sha256':sha(O/'external-all-server-absence.json')}}
(D/'receipt.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
(D/'audit.py').write_bytes(Path(__file__).read_bytes())
print(json.dumps({'V46resources':full['resources'],'V45':result['V45'],'receipt_sha256':sha(D/'receipt.json')},indent=2))
