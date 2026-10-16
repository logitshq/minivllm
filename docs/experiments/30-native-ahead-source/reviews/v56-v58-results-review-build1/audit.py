"""CPU-only saved V56 qualification and V58 power STOP review; no native import."""
from pathlib import Path
import ast,hashlib,json,re,statistics
P=Path('/private/tmp/minivllm-engine-optimization-20261014');D=Path(__file__).parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();read=lambda p:json.loads(Path(p).read_text());digest=lambda x:hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
helper=ast.parse((P/'v38-results-review-build1/audit.py').read_text());exec(compile(ast.Module(body=[n for n in helper.body if isinstance(n,ast.FunctionDef) and n.name in ('resources','cleanup')],type_ignores=[]),'saved-resource-parser','exec'))
Q=P/'v23-root-control-build1/fullmodel-v56native-attempt1';QD=Q/'child-output';q=read(QD/'receipt.json');ref=read(P/'v23-root-control-build1/fullmodel-v44native-attempt1/child-output/receipt.json');w=read(P/'workload.json');ids=w['continuation_ids']
assert sha(QD/'receipt.json')=='3405239d749a876a09f896a3ef06fc1f6603a317f24ba87c7711b53782435089'
assert q['status']=='PASS_ASYNC_NATIVE_RUNNER_C1_QUEUED_RESET_EARLYSTOP_FALLBACK_STRICT' and q['stage']=='complete' and q['strict_output_checks']==12540
for p,h in q['source_pins'].items():assert sha(p)==h,p
assert len(q['source_pins'])==79 and q['terminal_cleanup'] and q['captured_originals_readonly'] and q['build_input131_readonly']
checks=0
for row,gold in zip(q['requests'],ref['requests']):
 assert row['name']==gold['name'] and row['B_ids']==gold['A_ids'] and len(row['steps'])==31 and row['captured_originals_readonly']
 for step,old in zip(row['steps'],gold['steps']):
  assert step['step']==old['step'] and len(step['checks'])==132
  for x,y in zip(step['checks'],old['outputs']):assert x=={'label':y['label'],'bytes':y['bytes'],'sha256':y['A_sha256'],'qualified_A_sha256':y['A_sha256']};checks+=1
for key,withbytes in [('post_fallback_final132',False),('post_early_stop_final132',True)]:
 assert len(q[key])==132
 for x,y in zip(q[key],ref['requests'][0]['steps'][-1]['outputs']):
  e={'label':y['label'],'sha256':y['A_sha256'],'qualified_A_sha256':y['A_sha256']}
  if withbytes:e['bytes']=y['bytes']
  assert x==e;checks+=1
assert checks==12540 and q['whole_request_fallback_ids']==[ids[:2]] and q['prefill_only_ids']==[ids[:1]] and q['post_fallback_ids']==q['post_early_stop_ids']==[ids]
def pipe(n,d,c,r,discard,banks):
 p=n['pipeline'];f=n['feedback'];assert n['status']=='OK' and not n['error'] and p['enabled'] and p['depth']==p['maximum_outstanding']==2
 assert p['total_submitted']==p['total_completed']==c and p['total_delivered']==d and p['total_discarded']==discard and p['total_failed']==0
 assert p['inflight']==0 and not p['pending'] and all(not s['busy'] for s in p['slots'])
 assert f['resets']==r and f['icb']['execution_count']==f['total_steps']==c and f['icb']['bank_executions']==banks and f['pingpong']['eligible_count']==48
pipe(q['normal_four_native'],124,124,4,0,[64,60]);pipe(q['early_stop']['native'],130,131,5,1,[68,63]);pipe(q['native'],161,162,6,1,[84,78])
e=q['early_stop'];assert e['ids']==[ids[:7]] and e['stop_token_index']==6 and e['stop_token_id']==ids[6]==103725 and ids[6] not in ids[:6]
assert e['max_tokens']==32 and e['native_delivered_delta']==6 and e['native_completed_delta']==7 and e['discarded_delta']==1 and e['all132_output_checks']==0 and e['captured_originals_readonly']
assert not e['native']['pipeline']['last_result_readable'] and q['native']['pipeline']['last_result_readable']
assert [(r['submitted'],r['discarded']) for r in q['all_drop_records']]==[(31,0)]*4+[(7,1),(31,0)]
t=q['native']['pipeline']['timing'];assert len(t)==162 and all(x['success'] for x in t)
assert [(x['request'],x['ticket']) for x in t if x['discarded']]==[(5,7)]
for req in range(1,7):assert [x['ticket'] for x in t if x['request']==req]==list(range(1,(7 if req==5 else 31)+1))
full={'receipt_sha256':sha(QD/'receipt.json'),'source_pins':79,'strict_hashes':12540,'core_steps_every132':93,'other_full_checks':'postfallback final132; postearlystop final132','early_all132_checks':0,'delivered':161,'completed_submitted':162,'discarded':1,'resets':6,'banks':[84,78],'readonly':True,'resources':resources(Q,QD,75478),'cleanup':cleanup(Q,75469,75478)}
V=P/'v58-native-http-screen-build1';O=P/'v23-root-control-build1/native-http-v58-attempt1';C=O/'child-output';config=read(V/'runtime-config.json');pins=read(V/'pins.json');failure=read(C/'suite-failure.json');owner=read(O/'owner-receipt.json');release=read(O/'external-release.json')
assert sha(V/'pins.json')=='6eabc41bbf3c444b8db4b70fb7f6aae0fe7f793bc0ca896c6042504e5c4af1ba' and sha(V/'driver.py')==owner['driver_sha256']=='032963f74dfe0d495ca0f18213d7f3165e8ab2ff9c80c1bdfc2bfe8a7c2654de'
for ps in (pins['files'],config['source_pins']):
 for p,h in ps.items():assert sha(p)==h,p
assert len(pins['files'])==14 and len(config['source_pins'])==83 and config['fullqual_receipt_sha256']==full['receipt_sha256']
for name in ('request','group'):
 fn=lambda s:next(n for n in ast.walk(ast.parse(s)) if isinstance(n,ast.FunctionDef) and n.name==name)
 assert ast.dump(fn((V/'driver.py').read_text()),include_attributes=False)==ast.dump(fn((P/'m4-final-native15-official-attempt2/client.py').read_text()),include_attributes=False)
assert len(failure['completed'])==2 and failure['error']['message']=='unexpected power source: Battery Power' and not (C/'suite-complete.json').exists() and not (C/'owner-cleanup.json').exists()
completed=[];groups=0;rawids=0;servers=[];rawresources=[];bads=[]
for A in sorted(p for p in C.iterdir() if p.is_dir()):
 for p in sorted(A.glob('C1-*.json')):
  g=read(p);r=g['requests'][0];assert g['actual_request_count']==len(g['requests'])==1
  start,first,end=[r[k] for k in ('start_perf_counter_s','first_text_perf_counter_s','done_perf_counter_s')];assert start<first<end
  assert r['client_total_s']==end-start and r['client_ttft_text_s']==first-start and g['y_total_output_throughput_tok_s_mac']==32/(end-start)
  assert r['usage']['prompt_tokens']==19 and r['usage']['completion_tokens']==32 and r['usage'].get('prompt_tokens_details',{}).get('cached_tokens',0)==0 and r['finish_reason']=='length'
  assert len(r['visible_roundtrip_ids'])==32 and r['full_visible_roundtrip_count_matches_usage'] and r['first_chunk_prefix_verified']
  assert r['x_client_estimated_interactivity_tok_s_user']==(32-r['first_chunk_reconstructed_tokens'])/(end-first)==g['x_client_estimated_interactivity_tok_s_user']
  if 'native' in A.name:assert r['output_ids_when_available']==r['visible_roundtrip_ids']==ids;rawids+=32
  groups+=1
 if (A/'result.json').exists():
  r=read(A/'result.json');stats={'x':statistics.median(x['x_client_estimated_interactivity_tok_s_user'] for x in r['runs']),'y':statistics.median(x['y_total_output_throughput_tok_s_mac'] for x in r['runs']),'ttft_s':statistics.median(x['requests'][0]['client_ttft_text_s'] for x in r['runs']),'E2E_s':statistics.median(x['requests'][0]['client_total_s'] for x in r['runs'])}
  for k,v in stats.items():assert r[k]==v
  completed.append({'label':A.name,'engine':r['engine'],**stats})
 else:assert A.name=='C1-B1-lithos' and len(list(A.glob('C1-measure-*.json')))==2
 ex=read(A/'owned-process-exit.json');cl=read(A/'cleanup.json');port=read(A/'port-absence.json');listener=read(A/'server-ready-exact-listener.json')
 assert ex==cl['owned_exit'] and ex['wait_completed'] and ex['returncode']==-15 and ex['exact_pid_absence']['returncode']==1 and ex['exact_pid_absence']['stdout']==ex['exact_pid_absence']['stderr']==''
 assert all(x['decision']=='SAME_IDENTITY_SIGNAL' for x in ex['signal_identity_decisions']) and port['absent'] and port['checks'][-1]['connect_ex']==61
 assert listener['owned_pid']==ex['pid'];servers.append({'arm':A.name,'pid':ex['pid'],'waited_returncode':-15,'saved_strict_absence':ex['exact_pid_absence'],'port':port['port'],'saved_ECONNREFUSED':61})
 rs=read(A/'resources.json');ps=sorted(A.glob('resources-??-raw.json'));assert len(rs)==len(ps)
 for d,p in zip(rs,ps):
  v=read(p);assert v['label']==d['label'] and v['power_metadata']==d['power_metadata'] and int(v['pressure_level']['text'])==d['pressure_level_value']==1
  assert int(re.search(r'free percentage: (\d+)%',v['memory_pressure']['text'])[1])==d['free_percent']
  assert float(re.search(r'used = ([0-9.]+)M',v['swap']['text'])[1])==d['swap_used_mb']
  for k in ('Swapouts','Swapins','Pageouts'):assert int(re.search(r'^'+k+r':\s+(\d+)\.',v['vm_stat']['text'],re.M)[1])==d['vm_counters'][k]
  assert int(v['processinfo_thermal_metadata']['raw']['stdout'])==d['processinfo_thermal_metadata']['code']==0
  if d['power_metadata']['source']=='Battery Power':bads.append({'arm':A.name,'label':d['label'],'raw_file':str(p),'sha256':sha(p)})
  else:assert d['power_metadata']['source']=='AC Power'
  rawresources.append(d)
 assert rs[-1]['swap_used_mb']==rs[0]['swap_used_mb'] and rs[-1]['vm_counters']['Swapouts']==rs[0]['vm_counters']['Swapouts']
assert len(servers)==3 and len({s['pid'] for s in servers})==len({s['port'] for s in servers})==3
assert sorted(completed,key=lambda x:x['label'])==sorted(failure['completed'],key=lambda x:x['label']) and groups==17 and rawids==192
assert bads[0]['arm']=='C1-B1-lithos' and bads[0]['label']=='after-C1-measure-1'
nf=read(C/'C1-B1-native/native-final-metadata.json');pipe(nf['native'],186,186,6,0,[96,90]);assert nf['native_steps']==186 and nf['request_count']==6 and nf['active'] is None
assert owner['owner_pid']==76107 and owner['driver_pid']==76108 and owner['returncode']==1 and owner['waited_reaped'] and owner['stage']=='reaped'
assert release['status']=='PASS_EXTERNAL_RELEASE' and release['tool_exit']==1 and release['GPU_released'] and {x['pid'] for x in release['strict_absence']}=={76107,76108}
assert all(x['returncode']==1 and x['stdout']==x['stderr']=='' for x in release['strict_absence'])
extra=[]
for p in sorted(C.glob('owner-resources-??-raw.json'))+sorted(O.glob('external-resource-??-raw.json')):
 v=read(p);assert v['processinfo_thermal_metadata']['code']==0 and int(v['pressure_level']['text'])==1;extra.append({'file':str(p),'sha256':sha(p),'label':v['label'],'power':v['power_metadata']['source']})
result={'status':'PASS_SAVED_V56_NUMERIC_AND_V58_POWER_STOP_CPU_AUDIT','scope':'Saved hash/counter/source/stat/resource/absence evidence only; no independent GPU recomputation. V58 STOP remains unqualified; two complete AC arms descriptive only; incomplete Lithium arm has two measure files excluded; no pooled AC/Battery curve or adoption.','V56':full,'V58':{'source_pins_sha256':sha(V/'pins.json'),'runtime_source_pins':83,'controller_pins':14,'completed2_descriptive':failure['completed'],'planned9_completed2':True,'saved_HTTP_groups':groups,'saved_HTTP_tokens':groups*32,'native_raw_ID_tokens':rawids,'native_pipeline':{'delivered_completed':186,'discarded':0,'depth':2,'resets':6,'banks':[96,90]},'first_bad_power':bads[0],'per_arm_raw_resource_samples':len(rawresources),'owner_and_external_samples':extra,'min_free_percent':min(r['free_percent'] for r in rawresources),'thermal_codes':[0],'pressure_levels':[1],'per_arm_swap_MB_and_SO_growth':0,'owner_driver_external_absence':release['strict_absence'],'servers_perarm_wait_strict_absence_ports':servers,'external_all_server_aggregate_exists':(O/'external-all-server-absence.json').exists(),'suite_complete_exists':False,'owner_cleanup_exists':False,'driver_exit':1}}
(D/'receipt.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps({'status':result['status'],'V56resources':full['resources'],'V58':result['V58'],'receipt_sha256':sha(D/'receipt.json')},indent=2))
