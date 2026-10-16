from pathlib import Path
import json,hashlib,statistics,math,subprocess,re,collections
from types import SimpleNamespace
P=Path('/private/tmp/minivllm-engine-optimization-20261014');R=P/'v23-root-control-build1/native-http-v105-attempt1';O=R/'child-output';S=P/'v105-cap64-direct-http-screen-build1';D=P/'v105-independent-results-review-build1';D.mkdir(exist_ok=True)
def sha(q):
 h=hashlib.sha256()
 with Path(q).open('rb') as f:
  while b:=f.read(1<<20):h.update(b)
 return h.hexdigest()
def load(q):return json.loads(Path(q).read_text())
def write(q,a):Path(q).write_text(json.dumps(a,indent=2,ensure_ascii=False)+'\n')
def equal(a,b):assert math.isclose(a,b,rel_tol=1e-11,abs_tol=1e-10),(a,b)
config=load(S/'runtime-config.json');controller=load(S/'pins.json')['files'];runtime=config['source_pins'];assert len(controller)==17 and len(runtime)==200
assert all(sha(q)==h for q,h in controller.items()) and all(sha(q)==h for q,h in runtime.items())
assert config['fixed_cap']==64 and config['native_library_sha256']=='bf3d99c4979fcbd50562abd2e2520b549b897930414c21be6760f67918dcbcdf'
assert config['fullqual_receipt_sha256']=='551ccb7d44965fe29b3c9f5c2cb2eaf7f8b2c43c65ba5ab9aefc6d90a589bba1'
w=load(P/'workload.json');failure=load(O/'suite-failure.json');assert len(failure['completed'])==8 and failure['error']['message']=='nominal thermal required'
release=load(R/'external-release.json');assert release['status']=='PASS_EXTERNAL_RELEASE' and release['tool_exit']==1 and release['GPU_released']
assert [x['pid'] for x in release['strict_absence']]==[7855,7856] and all(x['returncode']==1 and x['stdout']==x['stderr']=='' for x in release['strict_absence'])
arms=sorted(q for q in O.iterdir() if q.is_dir() and re.fullmatch(r'C1-B[123]-(native|omlx|lithos)',q.name));assert len(arms)==9
metrics=[];streams=[];resources=[];nested=[];native_proofs=[];seals={};groups=collections.Counter();total_tokens=0
for arm in arms:
 label=arm.name;engine=label.rsplit('-',1)[1];complete=(arm/'result.json').exists();rows=[]
 for q in sorted(arm.glob('C1-*-*.json')):
  g=load(q);assert g['actual_request_count']==1 and len(g['requests'])==1 and g['total_completion_tokens']==32
  phase='warm' if '-warm-' in g['group_id'] else 'measure';groups[phase]+=1;groups[engine]+=1
  req=g['requests'][0];assert req['usage']['completion_tokens']==32 and req['usage']['prompt_tokens']==19 and req['finish_reason']=='length'
  assert len(req['visible_roundtrip_ids'])==32 and req['first_chunk_prefix_verified'] and req['full_visible_roundtrip_count_matches_usage']
  if engine=='native':assert req['output_ids_when_available']==w['continuation_ids']
  else:assert req['output_ids_when_available'] is None
  equal(req['client_total_s'],req['done_perf_counter_s']-req['start_perf_counter_s']);equal(req['client_ttft_text_s'],req['first_text_perf_counter_s']-req['start_perf_counter_s']);equal(req['client_post_first_chunk_s'],req['done_perf_counter_s']-req['first_text_perf_counter_s'])
  x=(32-req['first_chunk_reconstructed_tokens'])/req['client_post_first_chunk_s'];y=32/req['client_total_s']
  equal(x,req['x_client_estimated_interactivity_tok_s_user']);equal(x,g['x_client_estimated_interactivity_tok_s_user']);equal(y,g['y_total_output_throughput_tok_s_mac']);equal(g['group_window_s'],req['client_total_s'])
  total_tokens+=32;rows.append((phase,g,req));streams.append({'arm':label,'phase':phase,'group':g['group_id'],'request_count':1,'completion_tokens':32,'x':x,'y':y,'ttft_s':req['client_ttft_text_s'],'E2E_s':req['client_total_s'],'raw_IDs_available':engine=='native','raw_IDs_match_workload':True if engine=='native' else None,'visible_roundtrip_count':32})
 assert len(rows)==(6 if complete else 1)
 if complete:
  result=load(arm/'result.json');m=[(g,q) for phase,g,q in rows if phase=='measure'];assert len(m)==3
  vals={'x':statistics.median(g['x_client_estimated_interactivity_tok_s_user'] for g,q in m),'y':statistics.median(g['y_total_output_throughput_tok_s_mac'] for g,q in m),'ttft_s':statistics.median(q['client_ttft_text_s'] for g,q in m),'E2E_s':statistics.median(q['client_total_s'] for g,q in m)}
  for k,v in vals.items():equal(v,result[k])
  suite=next(r for r in failure['completed'] if r['label']==label)
  for k,v in vals.items():equal(v,suite[k])
  metrics.append({'label':label,'engine':engine,'completed':True,'warm_groups':3,'measured_groups':3,**vals,'scope':'one completed arm, no pooled campaign result'})
 else:
  assert label=='C1-B3-omlx' and not list(arm.glob('C1-measure-*.json'))
  assert load(arm/'first-failure.json')['message']=='nominal thermal required';metrics.append({'label':label,'engine':engine,'completed':False,'warm_groups':1,'measured_groups':0,'status':'STOP_THERMAL_AFTER_WARM0','x':None,'y':None})
 rr=load(arm/'resources.json');rawfiles=sorted(arm.glob('resources-*-raw.json'));assert len(rr)==len(rawfiles)==(17 if complete else 5)
 for i,q in enumerate(rawfiles):
  a=load(q);assert a['label']==rr[i]['label'] and a['recorded_at']==rr[i]['recorded_at']
  assert a['pressure_level_value']==int(a['pressure_level']['text'].strip())==1
  assert a['processinfo_thermal_metadata']['code']==int(a['processinfo_thermal_metadata']['raw']['stdout'].strip())
  assert a['power_metadata']['source']=='AC Power' and "Now drawing from 'AC Power'" in a['power_metadata']['raw']['stdout']
  assert a['vm_counters']['Swapouts']==int(re.search(r'Swapouts:\s+(\d+)\.',a['vm_stat']['text']).group(1))
  assert a['swap_used_mb']==float(re.search(r'used = ([\d.]+)M',a['swap']['text']).group(1))
  if complete:assert a['processinfo_thermal_metadata']['code']==0
  resources.append({'arm':label,'index':i,'label':a['label'],'recorded_at':a['recorded_at'],'power':a['power_metadata']['source'],'thermal':a['processinfo_thermal_metadata']['code'],'pressure':a['pressure_level_value'],'Swapouts':a['vm_counters']['Swapouts'],'swap_MB':a['swap_used_mb'],'raw_sha256':sha(q)})
 if not complete:assert [(a['label'],a['processinfo_thermal_metadata']['code']) for a in rr]==[('before-model-launch',0),('server-ready-before-warmup',0),('before-C1-warm-0',0),('after-C1-warm-0',1),('after-owned-unload',1)]
 owner=load(arm/'owned-pid.json');birth=load(arm/'owned-birth.json')['identity'];ready=load(arm/'owned-ready-identity.json');pre=load(arm/'owned-pre-cleanup-identity.json');exit=load(arm/'owned-process-exit.json');cleanup=load(arm/'cleanup.json');signals=load(arm/'cleanup-signal-identities.json');port=load(arm/'port-absence.json');listener=load(arm/'server-ready-exact-listener.json');pid=owner['pid']
 assert owner['ppid']==7856 and ready['pid']==pre['pid']==exit['pid']==pid and ready['owner_pid']==pre['owner_pid']==7856
 assert ready['initial_identity_words']==birth and birth[:5]==ready['current_identity_words'][:5]==pre['current_identity_words'][:5]
 assert ready['direct_Popen_child_live'] and pre['direct_Popen_child_live'] and ready['same_start'] and pre['same_start']
 if engine=='omlx':
  title=load(arm/'owned-title-transition.json');assert title['decision']=='ONE_PINNED_OMLX_TITLE_TRANSITION' and title['initial_identity_words']==birth and title['current_identity_words'][5:]==['omlx-server']
  assert sha(ready['pinned_title_source'])==ready['pinned_title_source_sha256']
  preimport=load(ready['preimport_path']);assert preimport['pid']==pid and preimport['ppid']==7856
 current=pre['current_identity_words'];locked=hashlib.sha256(json.dumps(current).encode()).hexdigest();assert locked==pre['locked_identity_sha256']==exit['recorded_birth_identity_sha256']==signals['recorded_birth_sha256']
 assert all(x['decision']=='SAME_IDENTITY_SIGNAL' and x['current_identity_sha256']==locked for x in signals['decisions'])
 assert exit['wait_completed'] and exit['exact_pid_absence']['returncode']==1 and exit['exact_pid_absence']['stdout']==exit['exact_pid_absence']['stderr']=='' and exit['exact_pid_absence']['absent']
 assert cleanup['owned_exit']==exit and all(x['phase']=='resource' for x in cleanup['cleanup_errors'])
 assert port['absent'] and all(x['connect_ex']!=0 for x in port['checks']) and listener['owned_pid']==pid and listener['port']==port['port']
 assert listener['returncode']==0 and listener['stderr']=='' and {int(x[1:]) for x in listener['stdout'].splitlines() if x.startswith('p')}=={pid}
 fresh=next(x for x in load(D/'current-readonly-absence.json')['checks'] if x['pid']==pid and x['port']==port['port'])
 ps=SimpleNamespace(**fresh['ps']);pcmd=ps.command;assert ps.returncode==1 and ps.stdout==ps.stderr==''
 ls=SimpleNamespace(**fresh['lsof']);lcmd=ls.command;assert ls.returncode==1 and ls.stdout==ls.stderr==''
 nested.append({'arm':label,'pid':pid,'parent_pid':7856,'birth_start_words':birth[:5],'original_birth_words_sha256':hashlib.sha256(json.dumps(birth).encode()).hexdigest(),'locked_cleanup_identity_sha256':locked,'pinned_omlx_title_transition':engine=='omlx','wait_completed':True,'saved_returncode':exit['returncode'],'saved_PID_absent':True,'current_readonly_ps':{'command':pcmd,'returncode':ps.returncode,'stdout':ps.stdout,'stderr':ps.stderr},'port':port['port'],'saved_port_absent':True,'current_readonly_port_lsof':{'command':lcmd,'returncode':ls.returncode,'stdout':ls.stdout,'stderr':ls.stderr},'cleanup_resource_error_only':bool(cleanup['cleanup_errors'])})
 if engine=='native':
  st=load(arm/'startup-metadata.json');wm=load(arm/'warm-metadata.json');fm=load(arm/'native-final-metadata.json');assert st['parameter_count']==1349 and st['parameter_tree_unchanged'] and st['manifest_all_sources_match'] and st['fixed_cap']==64 and st['physical_CAP64_before_engine_and_ANYtoken'] and st['cache_allocation_steps']==[64,64] and st['direct_production'] and not st['measurement_hooks']
  assert st['startup_ordinary_warm_requests']==1 and st['startup_ordinary_warm_tokens']==32 and st['native_library_sha256']==config['native_library_sha256'] and st['prior_fullqual_sha256']==config['fullqual_receipt_sha256']
  assert len(st['actual_adapter_chain_sources'])==4 and all(sha(x)==runtime[x] for x in [*st['actual_adapter_chain_sources'],*st['actual_imported_sources'].values()])
  assert wm['completed_requests']==3 and wm['native_steps']==93 and wm['parameter_tree_unchanged'] and wm['terminal_cleanup'] and not wm['measurement_hooks']
  assert fm['native_steps']==186 and fm['request_count']==6 and fm['active'] is None
  for obj,steps,resets,banks in [(wm['native'],93,3,[48,45]),(fm['native'],186,6,[96,90])]:
   f=obj['feedback'];pipe=obj['pipeline'];dd=obj['direct_pipeline'];assert obj['status']=='OK' and not obj['error'] and obj['parameter_ranges_registered']==1349
   assert (f['capacity'],f['total_steps'],f['resets'])==(64,steps,resets) and f['icb']['banks']==2 and f['icb']['execution_count']==steps and f['icb']['bank_executions']==banks
   assert (f['pingpong']['eligible_count'],f['pingpong']['staged_link_count'])==(48,83)
   assert pipe['total_submitted']==pipe['total_completed']==pipe['total_delivered']==steps and pipe['total_discarded']==pipe['total_failed']==0 and pipe['pending']==[] and pipe['inflight']==0
   assert pipe['depth']==pipe['maximum_outstanding']==2 and len(pipe['slots'])==2 and all(not x['busy'] and x['stage_bytes']==4 for x in pipe['slots'])
   assert dd['enabled'] and not dd['diagnostic_snapshot'] and dd['prediction_slot_bytes']==4 and dd['quiescent_output_copies']==0 and f['input_output_aliases']==[]
  native_proofs.append({'arm':label,'model_loads':st['model_loads'],'parameter_identity_count':1349,'startup_tokens':32,'warm_steps':93,'final_steps':186,'resets':6,'bank_counts':[96,90],'S_eligible':48,'direct_links':83,'slot_bytes':[4,4],'production_no_fullstate_output_copies':True,'parameter_tree_identity_observed':True,'no_new_full_parameter_byte_readonly_test':True,'measurement_hooks':False,'actual_adapter_chain_sources':st['actual_adapter_chain_sources']})
 for q in arm.iterdir():
  if q.is_file() and not q.is_symlink():seals[str(q)]=sha(q)
assert groups['warm']==25 and groups['measure']==24 and sum(1 for x in metrics if x['completed'])==8 and len(streams)==49 and total_tokens==1568
for q in O.glob('*-raw.json'):
 a=load(q);resources.append({'arm':'campaign-owner','label':a['label'],'power':a['power_metadata']['source'],'thermal':a['processinfo_thermal_metadata']['code'],'pressure':a['pressure_level_value'],'Swapouts':a['vm_counters']['Swapouts'],'swap_MB':a['swap_used_mb'],'raw_sha256':sha(q)})
for q in [R/'external-resource-00-raw.json']:
 a=load(q);resources.append({'arm':'root-external-release','label':a['label'],'power':a['power_metadata']['source'],'thermal':a['processinfo_thermal_metadata']['code'],'pressure':a['pressure_level_value'],'Swapouts':a['vm_counters']['Swapouts'],'swap_MB':a['swap_used_mb'],'raw_sha256':sha(q)})
for base in [R,O]:
 for q in base.iterdir():
  if q.is_file() and not q.is_symlink():seals[str(q)]=sha(q)
write(D/'nested-server-release-proof.json',{'status':'PASS_ALL_9_SAVED_WAIT_BIRTH_AND_CURRENT_READONLY_ABSENCE','signals_sent_by_reviewer':0,'server_count':9,'servers':nested,'root_external_release_sha256':sha(R/'external-release.json'),'root_owner_and_driver_saved_absent':[7855,7856]})
write(D/'per-arm-metrics.json',{'status':'INCOMPLETE_THERMAL_STOP_DESCRIPTIVE_ARMS_ONLY','completed_arms':8,'attempted_arms':9,'arms':metrics,'pooled_statistics_computed':False,'complete_curve':False,'adopted':False})
write(D/'request-metrics.json',{'HTTP_requests':49,'HTTP_completion_tokens':1568,'HTTP_warm_requests':25,'HTTP_measure_requests':24,'native_startup_requests':3,'native_startup_tokens':96,'total_including_native_startup_requests':52,'total_including_native_startup_tokens':1664,'requests':streams})
write(D/'resource-samples.json',resources);write(D/'native-runtime-proofs.json',native_proofs);write(D/'raw-seals.json',seals)
resource_summary={}
for arm in sorted({x['arm'] for x in resources}):
 vals=[x for x in resources if x['arm']==arm];resource_summary[arm]={'samples':len(vals),'power':sorted(set(x['power'] for x in vals)),'thermal_codes':sorted(set(x['thermal'] for x in vals)),'pressure_levels':sorted(set(x['pressure'] for x in vals)),'Swapouts_delta':vals[-1]['Swapouts']-vals[0]['Swapouts'],'swap_MB_delta':vals[-1]['swap_MB']-vals[0]['swap_MB']}
write(D/'receipt.json',{'status':'PASS_SAVED_AUDIT_OF_INCOMPLETE_STOP_CAMPAIGN_NOT_PERFORMANCE_PASS','source_controller_pins':len(controller),'source_runtime_pins':len(runtime),'source_verified':True,'completed_arms':8,'attempted_arms':9,'HTTP_requests':49,'HTTP_tokens':1568,'HTTP_measured_requests':24,'HTTP_measured_tokens':768,'HTTP_warm_requests':25,'HTTP_warm_tokens':800,'startup_requests':3,'startup_tokens':96,'total_requests_including_startup':52,'total_tokens_including_startup':1664,'thermal_STOP_phase':'C1-B3-omlx after-C1-warm-0, before remaining2warm/allmeasure','nested_server_count':9,'nested_all_saved_waited_and_current_absent':True,'nested_all9_ports_saved_and_current_absent':True,'resource_summary':resource_summary,'native_runtime_proofs':native_proofs,'raw_seals_sha256':sha(D/'raw-seals.json'),'raw_files_sealed':len(seals),'suite_failure_sha256':sha(O/'suite-failure.json'),'no_pooling_or_retry':True,'full_C1_C2_C4_curve':False,'HTTP_or_Pareto_win':False,'adopted':False,'reviewer_GPU_used':False,'source_or_raw_modified':False})
print(json.dumps({'completed':8,'attempted':9,'groups':dict(groups),'tokens':total_tokens,'startup_tokens':96,'nested_PID':[x['pid'] for x in nested],'ports':[x['port'] for x in nested],'resources':resource_summary,'metrics':metrics,'rawfiles_sealed':len(seals)},indent=2))
