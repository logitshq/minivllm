"""V128 independent saved-clock/route/resource replay; stdlib only, no live probes."""
from pathlib import Path
import ast,collections,csv,hashlib,json,math,statistics,types
P=Path('/private/tmp/minivllm-engine-optimization-20261014');D=P/'v128-fresh-full-pareto-build1';R=P/'v23-root-control-build1/full-curve-v128-attempt1';C=R/'child-output';O=Path(__file__).parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text())
def write(n,x):(O/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def near(a,b):assert math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-8),(a,b)
assert sha(D/'pins.json')=='94c395fe04d173734431c0bd41b09b8b38007d0479dabc7171a6e39617960b3e'
pins=load(D/'pins.json')['files'];cfg=load(D/'runtime-config.json')
for p,h in [*pins.items(),*cfg['source_pins'].items()]:assert sha(p)==h,p
suite=load(C/'suite-complete.json');assert sha(C/'suite-complete.json')=='3232911757dd8f16b50c1e2db02bc8419b812e6dfa92dc917e78bc5fe4091215'
plan=load(D/'campaign-plan.json')['configs'];assert [x['label']for x in plan]==[x['label']for x in suite['arms']]
reg=load(P/'m4-final-native15-official-attempt2/registry.json');allowed=load(cfg['same_C_ID_assertions'])['variants'];work=load(P/'workload.json')
tree=ast.parse((D/'driver.py').read_text());ns={}
for name in ('assert_pipeline',):
 f=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name==name);exec(compile(ast.Module(body=[f],type_ignores=[]),'actual_frozen_dict_gate','exec'),ns)
arm_rows=[];group_rows=[];request_rows=[];resource_rows=[];native_rows=[];official_rows=[];intervals=[];lithos_metric_count=0
for item in plan:
 A=C/item['label'];e=item['engine'];b=item['batch'];launch=load(A/'launch.json');expected_controls={'MLX_MAX_OPS_PER_BUFFER':'50','MLX_MAX_MB_PER_BUFFER':'50','MLX_BFS_MAX_WIDTH':'20','MLX_METAL_FAST_SYNCH':'0'};selected=dict(expected_controls)
 if e=='native'and b==1:selected['DYLD_INSERT_LIBRARIES']=cfg['native_library']
 if e=='lithos':selected['MONOLITH_ACCELERATOR']='on'
 assert launch['selected_environment']==selected and launch['environment']['knobs']==expected_controls and launch['environment']['gpu_arch_is_unset'] and not launch['environment']['other_mlx_key_value_sha256']
 res=load(A/'result.json');assert res['engine']==e and res['concurrency']==b and res['source_config_sha256']==sha(D/'runtime-config.json')
 phase_groups={}
 for phase in ('warm','measure'):
  gg=[]
  for i in range(3):
   gp=A/f'C{b}-{phase}-{i}.json';g=load(gp);assert g==res['warmups'if phase=='warm'else 'runs'][i];assert g['actual_request_count']==len(g['requests'])==b
   ds=[];fts=[];ct=0
   for q in g['requests']:
    st,en,ft=q['start_perf_counter_s'],q['done_perf_counter_s'],q['first_text_perf_counter_s'];assert st<ft<=en<=q['cleanup_complete_perf_counter_s'];near(en-st,q['client_total_s']);near(ft-st,q['client_ttft_text_s']);near(en-ft,q['client_post_first_chunk_s']);ds.append(en-st);fts.append(ft-st);intervals.append((st,en));u=q['usage'];assert u['prompt_tokens']==19 and u['completion_tokens']==32 and q['finish_reason']=='length';ct+=u['completion_tokens']
    assert q['full_visible_roundtrip_count_matches_usage'] and len(q['visible_roundtrip_ids'])==32 and ''.join(x['text']for x in q['events'])==q['text']
    raw=q['output_ids_when_available'];raw_hash=None
    if e=='native':
     assert isinstance(raw,list) and len(raw)==32 and raw==q['visible_roundtrip_ids'];raw_hash=hashlib.sha256(json.dumps(raw,separators=(',',':')).encode()).hexdigest()
     if b==1:assert raw==work['continuation_ids']
     else:assert allowed[str(b)].get(raw_hash)==raw
    else:assert raw is None
    details=u.get('prompt_tokens_details');cached=details.get('cached_tokens')if isinstance(details,dict)else None
    if cached is not None:assert cached==0
    else:assert e=='lithos'
    request_rows.append({'arm':item['label'],'engine':e,'C':b,'replica':item['replica'],'phase':phase,'group':i,'request_id':q['request_id'],'start_s':st,'done_s':en,'first_text_s':ft,'latency_ms':(en-st)*1000,'ttft_ms':(ft-st)*1000,'completion_tokens':32,'raw_ids_available':raw is not None,'raw32_json_sha256':raw_hash,'visible32_json_sha256':hashlib.sha256(json.dumps(q['visible_roundtrip_ids'],separators=(',',':')).encode()).hexdigest(),'usage_cached_tokens':cached,'usage_cached_observed':cached is not None})
   start=min(q['start_perf_counter_s']for q in g['requests']);end=max(q['done_perf_counter_s']for q in g['requests']);window=end-start;near(window,g['group_window_s']);assert ct==g['total_completion_tokens']==b*32;throughput=ct/window;near(throughput,g['y_total_output_throughput_tok_s_mac']);p50=statistics.median(ds)*1000
   group_rows.append({'arm':item['label'],'engine':e,'C':b,'replica':item['replica'],'phase':phase,'group':i,'request_count':b,'output_tokens':ct,'window_s':window,'request_latency_p50_ms':p50,'total_output_tok_s':throughput});gg.append((p50,throughput))
  phase_groups[phase]=gg
 x=statistics.median(z[0]for z in phase_groups['measure']);y=statistics.median(z[1]for z in phase_groups['measure']);near(x,res['latency_p50_ms']);near(y,res['y']);saved=next(z for z in suite['arms']if z['label']==item['label']);near(x,saved['latency_p50_ms']);near(y,saved['y'])
 arm_rows.append({'arm':item['label'],'engine':e,'C':b,'replica':item['replica'],'latency_p50_ms':x,'throughput_tok_s':y,'ttft_ms':res['ttft_s']*1000,'formal_requests':3*b,'formal_tokens':96*b})
 if e=='native':
  rt=res['runtime_identity'];warm=res['warm_metadata'];final=res['final_native_metadata'];assert rt['manifest_all_sources_match'] and warm['parameter_tree_unchanged']
  if b==1:
   assert rt['fixed_cap']==64 and rt['physical_CAP64_before_engine_and_ANYtoken'] and rt['cache_allocation_steps']==[64,64] and rt['model_loads']==1 and rt['gates_on']==48 and rt['startup_ordinary_warm_requests']==1 and rt['startup_ordinary_warm_tokens']==32
   assert rt['prior_fullqual_sha256']==cfg['fullqual_receipt_sha256'] and not rt['measurement_hooks'] and rt['direct_production']
   for path in [*rt['actual_imported_sources'].values(),*rt['actual_adapter_chain_sources']]:assert cfg['source_pins'][path]==sha(path)
   assert warm['native_steps']==93 and warm['completed_requests']==3 and warm['terminal_cleanup'] and not warm['measurement_hooks'];ns['assert_pipeline'](warm['native'],93,3);ns['assert_pipeline'](final['native'],186,6);assert final['active']is None and final['request_count']==6 and final['native_steps']==186
   native_rows.append({'arm':item['label'],'C':b,'profile':'direct_c1_44_101_102_CAP64','warm_steps':93,'final_steps':186,'resets':6,'bank_executions':[96,90],'startup_ordinary_requests':1,'startup_ordinary_tokens':32,'metadata_1349_parameter_unchanged':True,'full_weight_bytes_newly_hashed':False})
  else:
   assert rt['actual_source_files_sha256']==cfg['legacy_package_sources'] and len(rt['actual_source_files_sha256'])==25 and rt['server_max_seqs']==4 and rt['physical_capacity_steps']==[256,256] and not rt['native_adapter_installed'] and not rt['dynamic_native_batch_migration'] and rt['parameter_tree_array_count']==1349
   route=warm['routes'];counts={int(k):n for k,n in route['decode_batch_counts'].items()};assert counts[b]>0 and sum(k*n for k,n in counts.items())==48*31*3*b and route['hooks_restored_before_measure'] and route['setter_calls']==48
   assert route['compiled_eligible_calls']==counts.get(1,0) and route['fallback_calls']==sum(n for k,n in counts.items()if k!=1) and route['normal_decode_calls']==route['gates_route_calls']
   start_meta=load(A/'startup-gates-metadata.json');assert start_meta['parameter_metadata']==warm['parameter_metadata'] and len(start_meta['parameter_metadata'])==1349
   trace=load(A/'engine-trace.json');n=final['formal_trace_steps'];before,after=trace['steps'][:-n],trace['steps'][-n:];assert n>0 and sum(z['delivered_ids']for z in before)==3*b*32 and sum(z['delivered_ids']for z in after)==3*b*32 and max(z['active_decode_batch']for z in after)==b
   assert final['warm_hooks_removed_before_measure'] and final['idle_source_cleanup_assertions_completed'] and final['raw_same_C_assertions_passed'] and not final['native_adapter_installed'] and not final['pipeline186_assertion_applicable']
   native_rows.append({'arm':item['label'],'C':b,'profile':'legacy_accepted25_CAP256','warm_GDN_counts':counts,'warm_weighted_rows':sum(k*n for k,n in counts.items()),'formal_trace_steps':n,'formal_max_actual_B':b,'warm_hooks_restored':True,'warm_1349_parameter_metadata_unchanged':True,'full_tensor_check_newly_claimed':False})
 else:
  pre=load(A/'official-preimport.json');assert pre['engine']==e and pre['source_native_tree_sha256']==reg[e]['tree_sha256'] and pre['python_executable']==reg[e]['actual_environment']['python_executable'] and pre['files_checked']==len(reg[e]['source_native_files']) and pre['mlx_not_imported']
  row={'arm':item['label'],'engine':e,'C':b,'actual_preimport_tree_sha256':pre['source_native_tree_sha256'],'actual_files_checked_at_preimport':pre['files_checked'],'raw_IDs_available':False,'visible32_usage_policy_only':True}
  if e=='lithos':
   proof=res['lithos_proof'];assert proof['backend']=='m4_max' and proof['draft']is None and len(proof['metrics'])>=6 and all(z['cached_prompt_tokens']==0 for z in proof['metrics']);lithos_metric_count+=len(proof['metrics']);row['engine_metric_cached0_records']=len(proof['metrics']);row['usage_cached_observation']='not exposed, remains null'
  official_rows.append(row)
 for rp in sorted(A.glob('resources-*-raw.json')):
  z=load(rp);resource_rows.append({'arm':item['label'],'source':str(rp),'label':z['label'],'time':z['monotonic_s'],'power':z['power_metadata']['source'],'thermal':z['processinfo_thermal_metadata']['code'],'pressure':z['pressure_level_value'],'swap_used_mb':z['swap_used_mb'],'Swapouts':z['vm_counters']['Swapouts'],'known_foreign_count':sum(not p['owned']for p in z['known_inference_processes'])})
 exit=load(A/'owned-process-exit.json');clean=load(A/'cleanup.json');assert clean['error']is None and not clean['cleanup_errors'] and exit['wait_completed'] and exit['returncode']==-15 and exit['exact_pid_absence']['absent'] and load(A/'port-absence.json')['absent']
for rp in sorted(C.glob('owner-resources-*-raw.json')):
 z=load(rp);resource_rows.append({'arm':'campaign','source':str(rp),'label':z['label'],'time':z['monotonic_s'],'power':z['power_metadata']['source'],'thermal':z['processinfo_thermal_metadata']['code'],'pressure':z['pressure_level_value'],'swap_used_mb':z['swap_used_mb'],'Swapouts':z['vm_counters']['Swapouts'],'known_foreign_count':sum(not p['owned']for p in z['known_inference_processes'])})
assert len(request_rows)==270 and sum(z['phase']=='measure'for z in request_rows)==135 and len(group_rows)==126 and sum(z['completion_tokens']for z in request_rows)==8640
assert sum(z['raw_ids_available']for z in request_rows)==126 and sum(not z['raw_ids_available']for z in request_rows)==144 and sum(not z['usage_cached_observed']for z in request_rows)==18
assert len(resource_rows)==359
for z in resource_rows:assert z['power']=='AC Power' and z['thermal']==0 and z['pressure']==1 and z['known_foreign_count']==0;assert not any(st<=z['time']<=en for st,en in intervals)
for arm in set(z['arm']for z in resource_rows):
 rows=sorted((z for z in resource_rows if z['arm']==arm),key=lambda z:z['time']);assert len(rows)==(2 if arm=='campaign'else 17) and max(z['swap_used_mb']for z in rows)<=rows[0]['swap_used_mb'] and len({z['Swapouts']for z in rows})==1
ordered=sorted(resource_rows,key=lambda z:z['time']);assert len({z['Swapouts']for z in ordered})==1 and max(z['swap_used_mb']for z in ordered)<=ordered[0]['swap_used_mb']
cooling=[]
for p in sorted(C.glob('cool-gap-*.json')):
 z=load(p);assert z['requested_seconds']==60 and z['outside_HTTP_clocks'] and z['prior_owned_cleanup_passed'] and not z['adaptive'] and z['end_monotonic_s']-z['start_monotonic_s']>=60;assert all(en<=z['start_monotonic_s']or st>=z['end_monotonic_s']for st,en in intervals);cooling.append(z)
assert len(cooling)==20 and sum(x['requested_seconds']for x in cooling)==1200
configs=[]
for key in suite['medians']:
 e,c=key.split(':C');aa=[z for z in arm_rows if z['engine']==e and z['C']==int(c)];assert len(aa)==3;x=statistics.median(z['latency_p50_ms']for z in aa);y=statistics.median(z['throughput_tok_s']for z in aa);near(x,suite['medians'][key]['latency_p50_ms']);near(y,suite['medians'][key]['y']);configs.append({'point_id':key,'engine':e,'C':int(c),'latency_p50_ms':x,'throughput_tok_s':y,'fresh_arms':3})
def dominates(a,b):return a['latency_p50_ms']<=b['latency_p50_ms']and a['throughput_tok_s']>=b['throughput_tok_s']and(a['latency_p50_ms']<b['latency_p50_ms']or a['throughput_tok_s']>b['throughput_tok_s'])
for z in configs:z['dominated_by']=[a['point_id']for a in configs if dominates(a,z)];z['joint_non_dominated']=not z['dominated_by']
frontier=[x['point_id']for x in sorted(configs,key=lambda x:x['latency_p50_ms'])if x['joint_non_dominated']];assert frontier==['native:C1','native:C2','native:C4']
pairs=[]
for z in suite['signed_pairs']:
 b=z['concurrency'];rep=z['replica'];e=z['comparison'].removeprefix('native_vs_');n=next(a for a in arm_rows if(a['engine'],a['C'],a['replica'])==('native',b,rep));v=next(a for a in arm_rows if(a['engine'],a['C'],a['replica'])==(e,b,rep));lp=100*(1-n['latency_p50_ms']/v['latency_p50_ms']);yp=100*(n['throughput_tok_s']/v['throughput_tok_s']-1);near(lp,z['latency_reduction_percent']);near(yp,z['throughput_gain_percent']);assert z['negative_values_retained'];pairs.append({'C':b,'replica':rep,'rival':e,'native_latency_reduction_pct':lp,'native_throughput_gain_pct':yp,'native_throughput_ratio':n['throughput_tok_s']/v['throughput_tok_s'],'both_axes_improve':lp>0 and yp>0})
assert len(pairs)==12
pair_summaries=[]
for rival,b in [('omlx',1),('lithos',1),('omlx',2),('omlx',4)]:
 q=[z for z in pairs if(z['rival'],z['C'])==(rival,b)];assert len(q)==3
 pair_summaries.append({'rival':rival,'C':b,'pairs':3,'both_axes_wins':sum(z['both_axes_improve']for z in q),'latency_reduction_pct_range':[min(z['native_latency_reduction_pct']for z in q),max(z['native_latency_reduction_pct']for z in q)],'throughput_gain_pct_range':[min(z['native_throughput_gain_pct']for z in q),max(z['native_throughput_gain_pct']for z in q)],'geomean_throughput_ratio':math.exp(statistics.mean(math.log(z['native_throughput_ratio'])for z in q))})
owner=load(R/'owner-receipt.json');assert owner['returncode']==0 and owner['waited_reaped'] and owner['absence']['absent'] and owner['pins_sha256']==sha(D/'pins.json')
release=load(R/'external-release.json');assert sha(R/'external-release.json')=='d1563d894b01b5b0b5a7880a698ddf398f3ef5761cf372bdb3955117c7c551f8'and release['status']=='PASS_EXTERNAL_RELEASE_V128_FULL_CURVE'and len(release['pid_checks'])==23 and len(release['port_checks'])==21 and all(z['absent']for z in release['pid_checks']+release['port_checks'])
expected_pids={owner['owner_pid'],owner['driver_pid']}|{load(C/z['label']/'owned-pid.json')['pid']for z in plan};assert len(expected_pids)==23 and {z['pid']for z in release['pid_checks']}==expected_pids
assert len({z['port']for z in release['port_checks']})==21
write('reviewed_rows.json',{'scope':'V128 short19-to32/Qwen3.8-27B NVFP4/M4/frozen static configuration set only','primary_X':'nested median request POST-to-DONE latency p50 milliseconds','primary_Y':'group total output tokens/(lastDONE-firstPOST), nested medians','arms':arm_rows,'configurations':configs,'joint_frontier':frontier,'signed_pairs':pairs,'pair_summaries':pair_summaries,'observed_pair_both_axes_wins':sum(z['both_axes_improve']for z in pairs),'raw_suite_full_frontier_win_field':suite['full_frontier_win'],'field_interpretation':'pre-independent-audit conservative marker; not a computed dominance conclusion','all_tested_official_points_dominated':all(not z['joint_non_dominated']for z in configs if z['engine']!='native'),'global_all_workload_or_statistical_significance_claim':False})
write('request-clock-rows.json',request_rows);write('group-clock-rows.json',group_rows);write('resource-rows.json',resource_rows);write('native-route-proof.json',native_rows);write('official-source-proof.json',official_rows)
files={str(p):{'sha256':sha(p),'bytes':p.stat().st_size}for p in sorted(R.rglob('*'))if p.is_file()};write('evidence-identities.json',{'scope':'identities only; original launch/env/argv/auth/log bodies not copied','files':files})
verification={'status':'PASS_INDEPENDENT_V128_ACTUAL_CLOCKS_ROUTES_RESOURCES_AND_MEASURED_PARETO','suite_sha256':sha(C/'suite-complete.json'),'external_release_sha256':sha(R/'external-release.json'),'child_control_metadata_all21_exact50_50_20_0':True,'exact_saved_owned_PID_set_matched_release':23,'saved_port_absences':21,'controller_pins_rehashed':len(pins),'runtime_pins_rehashed':len(cfg['source_pins']),'HTTP_requests':270,'HTTP_outputs':8640,'formal_requests':135,'formal_outputs':4320,'native_startup_source_asserted_requests':3,'native_startup_source_asserted_outputs':96,'total_model_requests_including_startup':273,'total_outputs_including_startup':8736,'native_HTTP_raw32_requests':126,'official_visible_only_HTTP_requests':144,'usage_cached_field_missing':18,'lithos_engine_metrics_cached0_count':lithos_metric_count,'resource_arm_campaign_samples':359,'all_AC_nominal0_pressure1_knownforeign0_samples':True,'resource_swap_start_mb':ordered[0]['swap_used_mb'],'resource_swap_end_mb':ordered[-1]['swap_used_mb'],'Swapouts':ordered[0]['Swapouts'],'cooling_gaps':20,'fixed_cooling_requested_s':1200,'signed_pairs':12,'both_axes_paired_wins':sum(z['both_axes_improve']for z in pairs),'joint_frontier':frontier,'measured_configuration_set_frontier_all_native':True,'all_tested_official_points_dominated':True,'raw_full_frontier_win_false_is_pre_audit_marker':True,'new_full_tensor_or_parameter_byteRO_proof':False,'reviewer_GPU_or_live_probe':False,'W_Git_mutated':False};write('verification.json',verification);print(json.dumps(verification,indent=2))
