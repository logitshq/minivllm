"""Saved JSON/source/hash replay only. No MLX/GPU/live resource queries."""
from pathlib import Path
import json,hashlib,collections,re
D=Path(__file__).resolve().parent;P=D.parent;R=P/'v23-root-control-build1/omlx-mtp-v109-attempt1';O=R/'child-output';S=P/'v109-omlx-external-mtp-qual-build1'
def load(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n')
assert sha(O/'receipt.json')=='b9337fbc71bef73dfef4190834ba83c64b4483afa61b7d4cc1145d8e8c7b3ec5'
receipt=load(O/'receipt.json');cfg=load(S/'config.json');pin_sha=sha(S/'pins.json');assert pin_sha=='72856f12da32bc6438ef7aa050462e0bfbac35f0a074e718c07723cf28d55871'
pins=load(S/'pins.json')['files'];assert len(pins)==50 and all(sha(p)==h for p,h in pins.items())
assert receipt['status']=='PASS_EXTERNAL_MTP_GREEDY32_RAW_TARGET_CONTROL_RESOURCES' and receipt['actual_fresh_servers']==2 and receipt['actual_requests']==6 and receipt['terminal_cleanup'] and not receipt['performance_qualified'] and not receipt['global_best_claim']
release=load(R/'external-release.json');assert release['status']=='PASS_EXTERNAL_RELEASE' and release['tool_exit']==0 and release['GPU_released']
assert [x['pid'] for x in release['strict_absence']]==[12102,12103] and all(x['returncode']==1 and x['stdout']==x['stderr']=='' for x in release['strict_absence'])
canonical=receipt['ordinary_control_raw32'];assert len(canonical)==32 and not set(canonical)&set(cfg['eos_token_ids'])
perrequests=[];mode_summary={};nested=[];resource_rows=[];rawseals={};seqs={}
for mode,pid,port in [('none',12114,55521),('mtp',12238,55556)]:
 arm=O/mode;result=load(arm/'result.json');events=[json.loads(x) for x in (arm/'observer.jsonl').read_text().splitlines()];counter=collections.Counter(x['kind'] for x in events)
 assert result['source_pins_sha256']==pin_sha and result['observer_sha256']==sha(arm/'observer.jsonl') and result['mode']==mode and result['timings_are_instrumented_and_not_performance_evidence']
 assert all(e['pid']==pid for e in events) and counter['observer_installed']==1 and counter['terminal']==3
 assert events[0]['kind']=='observer_installed' and events[0]['mode']==mode and events[0]['no_added_array_eval'] and not events[0]['timing_eligible']
 terminals=[x for x in events if x['kind']=='terminal'];assert len({x['request_id'] for x in terminals})==3
 assert len(result['requests'])==len(result['observations'])==3
 if mode=='none':assert counter=={'observer_installed':1,'terminal':3}
 else:
  assert counter=={'observer_installed':1,'load':1,'attach':1,'route':3,'verify':51,'round':51,'stats':3,'terminal':3}
  loaded=next(x for x in events if x['kind']=='load');attached=next(x for x in events if x['kind']=='attach');flags=attached['flags']
  assert loaded['loaded'] and loaded['requested_path']==loaded['source_path']==cfg['draft_model_dir'] and loaded['draft_kind']=='mtp' and loaded['config_model_type']=='qwen3_5_mtp' and loaded['block_size']==3
  assert attached['drafter_source']==cfg['draft_model_dir'] and attached['draft_kind']=='mtp' and attached['engine_same_object']
  assert flags['vlm_mtp_enabled'] and flags['vlm_mtp_draft_model']==cfg['draft_model_dir'] and flags['vlm_mtp_draft_block_size'] is None and all(flags[x] is False for x in ['mtp_enabled','dflash_enabled','specprefill_enabled'])
 for terminal,http,observed in zip(terminals,result['requests'],result['observations']):
  rid=terminal['request_id'];assert observed['request_id']==rid and terminal['raw_output_ids']==http['visible_roundtrip_ids']==observed['raw_output_ids']==canonical
  assert terminal['prompt_tokens']==19 and terminal['completion_tokens']==32 and terminal['cached_tokens']==0 and terminal['finish_reason']==http['finish_reason']=='length' and terminal['error'] is None
  assert http['usage']['prompt_tokens']==19 and http['usage']['completion_tokens']==32 and http['usage'].get('prompt_tokens_details',{}).get('cached_tokens',0)==0
  assert not set(terminal['raw_output_ids'])&set(cfg['eos_token_ids'])
  if mode=='none':continue
  rr=[x for x in events if x.get('request_id')==rid];route=[x for x in rr if x['kind']=='route'];ver=[x for x in rr if x['kind']=='verify'];rnd=[x for x in rr if x['kind']=='round'];stat=[x for x in rr if x['kind']=='stats']
  assert len(route)==len(stat)==1 and len(ver)==len(rnd)==17 and [x['kind'] for x in rr]==['route']+['verify','round']*17+['stats','terminal']
  route=route[0];stat=stat[0];assert route['routed'] and route['rollback_capability'] and route['prompt_ids']==cfg['prompt_ids'] and route['max_tokens']==32 and route['sampler_temperature']==0
  assert route['scheduler_drafter_source']==cfg['draft_model_dir'] and route['stop_ids']==cfg['eos_token_ids']
  assert all(x['input_shape']==[1,3] and x['actual_draft_token_count']==2 for x in ver)
  drafted=[x['drafted'] for x in rnd];accepted=[x['accepted'] for x in rnd]
  assert drafted==stat['draft_lens']==observed['round_drafts']==[2]*17 and accepted==stat['accept_lens']==observed['round_accepts']
  assert sum(drafted)==observed['drafted']==34 and sum(accepted)==observed['accepted']==15 and observed['verify_calls']==17 and collections.Counter(accepted)=={0:7,1:5,2:5}
  assert all(0<=a<=b for a,b in zip(accepted,drafted)) and stat['emitted']==32 and stat['finish_reason']=='length' and stat['actual_config_block_size']==3 and stat['configured_block_size'] is None
  perrequests.append({'request_id':rid,'verify_calls':17,'verify_shapes':[[1,3]]*17,'verify_input_dtype_observed':sorted(set(x['input_dtype'] for x in ver)),'draft_proposals':34,'native_accepted_counter':15,'delivered_IDs':32,'accepted_per_round':accepted,'accepted_histogram':dict(collections.Counter(accepted)),'lifetime_fields_not_used_as_crossrequest_proof':True})
 rawfiles=sorted(arm.glob('resources-*-raw.json'));samples=load(arm/'resources.json');assert len(rawfiles)==len(samples)==len(result['resource_snapshots'])==9
 for q,record in zip(rawfiles,samples):
  a=load(q);assert a['label']==record['label'] and a['recorded_at']==record['recorded_at'] and a['power_metadata']['source']=='AC Power' and a['processinfo_thermal_metadata']['code']==0 and a['pressure_level_value']==1
  assert a['pressure_level']['text'].strip()=='1' and a['processinfo_thermal_metadata']['raw']['stdout'].strip()=='0' and "Now drawing from 'AC Power'" in a['power_metadata']['raw']['stdout']
  assert int(re.search(r'Swapouts:\s+(\d+)\.',a['vm_stat']['text']).group(1))==a['vm_counters']['Swapouts']
  assert float(re.search(r'used = ([\d.]+)M',a['swap']['text']).group(1))==a['swap_used_mb']
  resource_rows.append({'mode':mode,'label':a['label'],'power':'AC Power','thermal':0,'pressure':1,'Swapouts':a['vm_counters']['Swapouts'],'swap_MB':a['swap_used_mb'],'exact_owned_pid_RSS_KiB':a.get('owned_pid_rss_kib'),'raw_sha256':sha(q)})
 assert len({x['vm_counters']['Swapouts'] for x in samples})==1 and len({x['swap_used_mb'] for x in samples})==1
 memory=[x['memory'] for x in events if 'memory' in x];mode_summary[mode]={'event_counts':dict(counter),'requests':3,'raw_IDs':96,'resource_samples':9,'Swapouts_delta':0,'swap_MB_delta':0,'max_observed_MLX_allocator_peak_bytes':max(x['get_peak_memory'] for x in memory),'max_observed_MLX_active_bytes':max(x['get_active_memory'] for x in memory),'max_observed_MLX_cache_bytes':max(x['get_cache_memory'] for x in memory),'allocator_is_not_RSS_or_physical_footprint':True,'max_sampled_exact_owned_RSS_KiB':max(x.get('owned_pid_rss_kib') or 0 for x in samples)}
 birth=load(arm/'owned-birth.json')['identity'];ready=load(arm/'owned-ready-identity.json');pre=load(arm/'owned-pre-cleanup-identity.json');ex=load(arm/'owned-process-exit.json');cleanup=load(arm/'cleanup.json');sig=load(arm/'cleanup-signal-identities.json');pa=load(arm/'port-absence.json');li=load(arm/'server-ready-exact-listener.json');fresh=next(x for x in load(D/'current-readonly-absence.json')['checks'] if x['pid']==pid)
 assert birth[:5]==ready['current_identity_words'][:5]==pre['current_identity_words'][:5] and ready['same_start'] and pre['same_start'] and ready['direct_Popen_child_live'] and pre['direct_Popen_child_live'] and pre['owner_pid']==12103
 assert ready['initial_identity_words']==birth and ready['current_identity_words'][5:]==['omlx-server'] and sha(ready['pinned_title_source'])==ready['pinned_title_source_sha256']
 locked=hashlib.sha256(json.dumps(pre['current_identity_words']).encode()).hexdigest();assert locked==ex['recorded_birth_identity_sha256']==sig['recorded_birth_sha256']
 assert all(x['decision']=='SAME_IDENTITY_SIGNAL' and x['current_identity_sha256']==locked for x in sig['decisions'])
 assert ex['pid']==pid and ex['wait_completed'] and ex['returncode']==-15 and ex['exact_pid_absence']['returncode']==1 and ex['exact_pid_absence']['stdout']==ex['exact_pid_absence']['stderr']==''
 assert cleanup['cleanup_errors']==[] and cleanup['first_error'] is None and cleanup['owned_exit']==ex
 assert pa['port']==port and pa['absent'] and all(x['connect_ex']==61 for x in pa['checks']) and li['owned_pid']==pid and li['port']==port and li['returncode']==0 and li['stderr']==''
 assert {int(x[1:]) for x in li['stdout'].splitlines() if x.startswith('p')}=={pid} and fresh['port']==port
 assert all(fresh[x]['returncode']==1 and fresh[x]['stdout']==fresh[x]['stderr']=='' for x in ['ps','lsof'])
 preimport=load(arm/'official-preimport.json');assert preimport['pid']==pid and preimport['ppid']==12103 and preimport['vendor_sources_unchanged'] and preimport['qualification_observers_added'] and preimport['versions']==cfg['versions'] and preimport['mlx_not_imported']
 nested.append({'mode':mode,'pid':pid,'parent':12103,'port':port,'birth_start_words':birth[:5],'locked_cleanup_identity_sha256':locked,'pinned_title_transition':True,'saved_wait_completed':True,'saved_returncode':-15,'saved_PID_and_port_absent':True,'current_PID_and_port_absent':True,'no_signals_sent_by_reviewer':True})
 for q in arm.rglob('*'):
  if q.is_file() and not q.is_symlink() and q.suffix in ['.json','.jsonl','.log','.txt','.md']:rawseals[str(q)]=sha(q)
 seqs[mode]=[x['raw_output_ids'] for x in terminals]
assert seqs['none']==seqs['mtp']==[canonical]*3 and len(perrequests)==3 and sum(x['draft_proposals'] for x in perrequests)==102 and sum(x['native_accepted_counter'] for x in perrequests)==45
assert [x['request_id'] for x in perrequests]==[x['request_id'] for x in receipt['MTP_per_request']]
for q in R.iterdir():
 if q.is_file() and not q.is_symlink():rawseals[str(q)]=sha(q)
rawseals[str(O/'receipt.json')]=sha(O/'receipt.json')
write(D/'per-request-rounds.json',perrequests);write(D/'resource-samples.json',resource_rows);write(D/'nested-server-release-proof.json',nested);write(D/'raw-seals.json',rawseals);write(D/'source-seals.json',pins)
write(D/'receipt.json',{'status':'PASS_INDEPENDENT_V109_SAVED_CORRECTNESS_PARTICIPATION_RESOURCE_REPLAY','actual_receipt_sha256':sha(O/'receipt.json'),'root_external_release_sha256':sha(R/'external-release.json'),'source_pins_sha256':pin_sha,'source_pins':50,'modes':mode_summary,'six_raw_streams_equal_fresh_control':True,'delivered_IDs_total':192,'MTP_requests':3,'MTP_verify_calls_total':51,'MTP_proposals_total':102,'MTP_native_accepted_count_total':45,'MTP_delivered_IDs_total':96,'accepted_per_request_histogram':{'0':7,'1':5,'2':5},'no_claim_proposals_are_delivered':True,'nested2_current_absent':True,'nested2_ports_current_absent':True,'raw_files_sealed':len(rawseals),'raw_seals_sha256':sha(D/'raw-seals.json'),'source_seals_sha256':sha(D/'source-seals.json'),'allocator_scope':'MLX allocator counters; not RSS/physical footprint and not whole-process memory','instrumented_timings_not_performance':True,'performance_or_global_best_claim':False,'GPU_used':False,'source_or_raw_modified':False})
print(json.dumps({'modes':mode_summary,'per_request_counts':[{'verify':x['verify_calls'],'proposed':x['draft_proposals'],'accepted':x['native_accepted_counter'],'delivered':32,'histogram':x['accepted_histogram']} for x in perrequests],'raw_files_sealed':len(rawseals)},indent=2))
