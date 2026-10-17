"""CPU-only allowlisted publication from immutable saved V128 evidence; no inference."""
from pathlib import Path
import ast,csv,hashlib,json,statistics
P=Path('/private/tmp/minivllm-engine-optimization-20261014')
O=Path(__file__).resolve().parent;S=O/'44-fresh-full-source'
D=P/'v128-fresh-full-pareto-build1';E=P/'v126-native-full-curve-prework-build1'
R=P/'v23-root-control-build1/full-curve-v128-attempt1';C=R/'child-output'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
originals={}
def original(p):
 p=Path(p);v={'sha256':sha(p),'bytes':p.stat().st_size};originals[str(p)]=v;return {'path':str(p),**v}
def write(n,x):
 p=S/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def exact(p,n):
 p=Path(p);dest=S/n;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(p.read_bytes());original(p);assert sha(dest)==sha(p)
def select(d,names):return {k:d[k] for k in names if k in d}
def resource(p,arm=None):
 x=json.loads(Path(p).read_text());power=x['power_metadata'];thermal=x['processinfo_thermal_metadata']
 assert power['source']=='AC Power' and thermal['code']==0 and x['pressure_level_value']==1
 return {'arm':arm,'source':original(p),**select(x,('label','recorded_at','monotonic_s','owned_model_pid','pressure_level_value','swap_used_mb','free_percent','known_inference_process_count','only_owned_known_inference_processes','exactly_one_owned_known_model')),'Swapouts':x['vm_counters']['Swapouts'],'power':select(power,('status','percent','source','state_text')),'thermal':select(thermal,('status','code','name'))}
# Exact source snapshots contain code or immutable source identities, never runtime environment/argv/config secrets.
for n in ('driver.py','owner.py','prepare_config.py','check_cpu.py','protocol.json','campaign-plan.json','pins.json','source-map.json','parent125.diff','CPU-READY.json','cpu-check.json','cpu-dry-run.json'):
 exact(D/n,'source-v128/'+n)
for n in ('server.py','c1_server.py','batched_server.py','entry_policy.py','profiles.json','qualify_batch.py','check_cpu.py','protocol.json','source-pins.json','source-map.json','old-same-C-ID-assertions.json','cpu-check.json','CPU-READY.json'):
 exact(E/n,'source-v126/'+n)
# Preserve complete source pin identities without exporting runtime-config values.
config=json.loads((D/'runtime-config.json').read_text());runtime_refs=config['source_pins']
assert len(runtime_refs)==234
for p,h in runtime_refs.items():assert sha(p)==h,p
controls=json.loads((D/'pins.json').read_text())['files'];assert len(controls)==19
for p,h in controls.items():assert sha(p)==h,p
write('source-identities.json',{'runtime_config_original':original(D/'runtime-config.json'),'runtime_sources':runtime_refs,'controller_sources':controls,'identities_checked_now':True,'source_snapshot_scope':'V128 harness and V126 static selector/batching; older mathematical/runtime sources are exact hash dependencies; no dylib, weights or runtime config values exported'})
# The suite is scalar-only and copied byte for byte. Its conservative pre-audit boolean is not a calculated frontier.
suite=json.loads((C/'suite-complete.json').read_text());assert len(suite['arms'])==21
assert suite['HTTP_requests']==270 and suite['measured_requests']==135 and suite['HTTP_completion_tokens']==8640
exact(C/'suite-complete.json','suite-complete.json')
plan=json.loads((D/'campaign-plan.json').read_text())['configs'];lookup={p['label']:p for p in plan}
requests=[];groups=[];arms=[];resources=[];pointrows=[]
for arm in suite['arms']:
 label=arm['label'];engine=arm['engine'];batch=arm['concurrency'];A=C/label
 summary=json.loads((A/'result.json').read_text());original(A/'result.json');formal=[]
 for phase in ('warm','measure'):
  for i in range(3):
   p=A/f'C{batch}-{phase}-{i}.json';g=json.loads(p.read_text());ref=original(p)
   assert len(g['requests'])==batch and g['total_completion_tokens']==32*batch
   ds=[];xs=[]
   for q in g['requests']:
    usage=q['usage'];assert usage['prompt_tokens']==19 and usage['completion_tokens']==32 and q['finish_reason']=='length'
    start=q['start_perf_counter_s'];done=q['done_perf_counter_s'];dur=done-start
    assert dur==q['client_total_s'] and q['cleanup_complete_perf_counter_s']>=done>start
    visible=q['visible_roundtrip_ids'];raw=q['output_ids_when_available'];assert len(visible)==32
    assert raw is None or raw==visible
    detail=usage.get('prompt_tokens_details');cached=detail.get('cached_tokens') if detail else None
    assert cached in (None,0)
    rr={'arm':label,'engine':engine,'concurrency':batch,'replica':lookup[label]['replica'],'phase':phase,'repeat':i,'request_id':q['request_id'],'source':ref,
     **select(q,('start_perf_counter_s','done_perf_counter_s','first_text_perf_counter_s','cleanup_complete_perf_counter_s','stream_cleanup_after_done_s','client_total_s','client_ttft_text_s','client_post_first_chunk_s','finish_reason','output_ids_when_available','visible_roundtrip_ids','first_chunk_reconstructed_tokens','first_chunk_prefix_verified','full_visible_roundtrip_count_matches_usage','x_client_estimated_interactivity_tok_s_user')),
     'usage':{'prompt_tokens':usage['prompt_tokens'],'completion_tokens':usage['completion_tokens'],'total_tokens':usage['total_tokens'],'cached_tokens_observed':cached,'cached_tokens_observation':'reported_in_usage' if cached is not None else 'not_exposed_in_usage'},'duration_ms':1000*dur}
    requests.append(rr);ds.append(dur);xs.append(q['x_client_estimated_interactivity_tok_s_user'])
   lo=min(q['start_perf_counter_s'] for q in g['requests']);hi=max(q['done_perf_counter_s'] for q in g['requests']);y=32*batch/(hi-lo);lat=1000*statistics.median(ds)
   assert y==g['y_total_output_throughput_tok_s_mac'] and hi-lo==g['group_window_s']
   gr={'arm':label,'engine':engine,'concurrency':batch,'replica':lookup[label]['replica'],'phase':phase,'repeat':i,'source':ref,'request_ids':[q['request_id'] for q in g['requests']],
    **select(g,('group_id','actual_request_count','arrival_spread_s','group_window_s','total_completion_tokens','y_total_output_throughput_tok_s_mac','x_client_estimated_interactivity_tok_s_user')),'latency_p50_ms':lat,'earliest_POST_perf_counter_s':lo,'last_DONE_perf_counter_s':hi}
   groups.append(gr)
   if phase=='measure':formal.append(gr)
 lat=statistics.median(g['latency_p50_ms'] for g in formal);y=statistics.median(g['y_total_output_throughput_tok_s_mac'] for g in formal)
 assert lat==arm['latency_p50_ms'] and y==arm['y']
 pointrows.append({'config':f'{engine}:C{batch}','arm':label,'engine':engine,'concurrency':batch,'replica':lookup[label]['replica'],'latency_ms':lat,'throughput_tok_s':y,'aux_interactivity_tok_s':arm['x'],'aux_ttft_ms':1000*arm['ttft_s'],'static_native_profile':summary['static_native_profile']})
 for p in sorted(A.glob('resources-*-raw.json')):resources.append(resource(p,label))
 cleanup=json.loads((A/'cleanup.json').read_text());exit=cleanup['owned_exit'];assert cleanup['error'] is None and not cleanup['cleanup_errors'] and exit['wait_completed'] and exit['exact_pid_absence']['absent']
 ports=json.loads((A/'port-absence.json').read_text());assert ports['absent']
 rec={'arm':label,'engine':engine,'concurrency':batch,'cleanup':{'error':None,'cleanup_errors':[],'owned_exit':{**select(exit,('pid','wait_completed','returncode','recorded_birth_identity_sha256')),'exact_pid_absence':select(exit['exact_pid_absence'],('pid','absent','returncode'))}},'port_absence':ports,'source':original(A/'result.json')}
 if engine=='native':
  r=summary['runtime_identity'];w=summary['warm_metadata'];n=json.loads((A/'native-final-metadata.json').read_text())
  rec['identity']=select(r,('manifest_all_sources_match','mode','scheduler_max_seqs','server_max_seqs','model_loads','parameter_count','parameter_tree_unchanged','gates_on','fixed_cap','native_library_sha256','adapter_sha256','prior_fullqual_sha256','prefix_cache','startup_ordinary_warm_requests','startup_ordinary_warm_tokens','async_pipeline_depth','direct_production','physical_CAP64_before_engine_and_ANYtoken','measurement_hooks','cache_policy','static_profile','dynamic_native_batch_migration','native_adapter_installed','physical_capacity_steps','allocator_cache_limit_bytes','required_client_concurrency','parameter_tree_array_count'))
  if batch==1:
   v=n['native'];f=v['feedback'];pp=v['pipeline'];dp=v['direct_pipeline'];icb=f['icb']
   rec['warm']={'completed_requests':w['completed_requests'],'native_steps':w['native_steps'],'parameter_tree_unchanged':w['parameter_tree_unchanged'],'terminal_cleanup':w['terminal_cleanup'],'measurement_hooks':w['measurement_hooks'],'pipeline':select(w['native']['pipeline'],('depth','total_submitted','total_completed','total_delivered','total_discarded','total_failed'))}
   rec['native_final']={'native_steps':n['native_steps'],'request_count':n['request_count'],'active':n['active'],'parameter_ranges_registered':v['parameter_ranges_registered'],'feedback':select(f,('capacity','resets','total_steps','CPU_full_state_restore_per_step')),'pipeline':select(pp,('enabled','depth','total_submitted','total_completed','total_delivered','total_discarded','total_failed','pending','inflight')),'icb':select(icb,('enabled','banks','execution_count','bank_executions')),'pingpong':select(f['pingpong'],('eligible_count','staged_link_count')),'direct_pipeline':select(dp,('enabled','diagnostic_snapshot','direct_links','prediction_slot_bytes'))}
  else:
   rec['warm']={'completed_requests':w['completed_requests'],'parameter_tree_unchanged':w['parameter_tree_unchanged'],'gates':select(w['gates'],('enabled','count','compiled_callable_count')),'routes':select(w['routes'],('decode_batch_counts','normal_decode_calls','gates_route_calls','compiled_eligible_calls','fallback_calls','setter_calls','hooks_restored_before_measure')),'weighted_GDN_rows':sum(int(b)*n for b,n in w['routes']['decode_batch_counts'].items())}
   rec['native_final']=n
   trace=json.loads((A/'engine-trace.json').read_text());rec['batch_trace_selected']={'source':original(A/'engine-trace.json'),'steps':[select(x,('active_decode_batch','delivered_ids','prefill_batch','prefill_count','decode_count','request_count')) for x in trace['steps']]}
  original(A/'warm-metadata.json');original(A/'native-final-metadata.json')
 if engine=='lithos':
  proof=summary['lithos_proof'];rec['Lithos_participation']={'backend':proof['backend'],'draft':proof['draft'],'Generation_cached_prompt_tokens':[m['cached_prompt_tokens'] for m in proof['metrics']],'server_log_original_sha256':proof['precleanup_server_log_sha256']}
 original(A/'cleanup.json');original(A/'port-absence.json');arms.append(rec)
for p in sorted(C.glob('owner-resources-*-raw.json')):resources.append(resource(p,'campaign'))
for p in sorted(C.glob('cool-gap-*.json')):exact(p,'cooling/'+p.name)
assert len(requests)==270 and len(groups)==126 and len(pointrows)==21
assert sum(x['usage']['completion_tokens'] for x in requests)==8640
configs=[]
for engine,batch in [('native',1),('native',2),('native',4),('omlx',1),('omlx',2),('omlx',4),('lithos',1)]:
 rows=[a for a in pointrows if a['engine']==engine and a['concurrency']==batch];assert len(rows)==3
 configs.append({'config':f'{engine}:C{batch}','engine':engine,'concurrency':batch,'fresh_arms':3,'latency_ms':statistics.median(a['latency_ms'] for a in rows),'throughput_tok_s':statistics.median(a['throughput_tok_s'] for a in rows),'static_native_profile':rows[0]['static_native_profile']})
def dominates(a,b):return a['latency_ms']<=b['latency_ms'] and a['throughput_tok_s']>=b['throughput_tok_s'] and (a['latency_ms']<b['latency_ms'] or a['throughput_tok_s']>b['throughput_tok_s'])
for c in configs:
 c['dominated_by']=[a['config'] for a in configs if dominates(a,c)];c['joint_non_dominated']=not c['dominated_by'];c['engine_non_dominated']=not any(dominates(a,c) for a in configs if a['engine']==c['engine'])
 assert c['latency_ms']==suite['medians'][c['config']]['latency_p50_ms'] and c['throughput_tok_s']==suite['medians'][c['config']]['y']
pairs=[]
for batch,engines in [(1,('omlx','lithos')),(2,('omlx',)),(4,('omlx',))]:
 for replica in (1,2,3):
  n=next(a for a in pointrows if (a['engine'],a['concurrency'],a['replica'])==('native',batch,replica))
  for e in engines:
   a=next(x for x in pointrows if (x['engine'],x['concurrency'],x['replica'])==(e,batch,replica));pair={'concurrency':batch,'replica':replica,'comparison':'native_vs_'+e,'latency_reduction_percent':100*(1-n['latency_ms']/a['latency_ms']),'throughput_gain_percent':100*(n['throughput_tok_s']/a['throughput_tok_s']-1)}
   old=next(x for x in suite['signed_pairs'] if (x['concurrency'],x['replica'],x['comparison'])==(batch,replica,pair['comparison']))
   assert all(pair[k]==old[k] for k in ('latency_reduction_percent','throughput_gain_percent'));pairs.append(pair)
ratios=[]
for batch,e in [(1,'omlx'),(1,'lithos'),(2,'omlx'),(4,'omlx')]:
 n=next(a for a in configs if a['config']==f'native:C{batch}');b=next(a for a in configs if a['config']==f'{e}:C{batch}');ps=[p for p in pairs if p['concurrency']==batch and p['comparison']=='native_vs_'+e]
 ratios.append({'concurrency':batch,'comparison':'native_vs_'+e,'ratio_of_config_medians':{'latency_reduction_percent':100*(1-n['latency_ms']/b['latency_ms']),'throughput_gain_percent':100*(n['throughput_tok_s']/b['throughput_tok_s']-1)},'median_of_signed_pair_percentages':{k:statistics.median(p[k] for p in ps) for k in ('latency_reduction_percent','throughput_gain_percent')},'paired_both_axis_wins':sum(p['latency_reduction_percent']>0 and p['throughput_gain_percent']>0 for p in ps),'pair_count':3})
owner=json.loads((R/'owner-receipt.json').read_text());assert owner['returncode']==0 and owner['waited_reaped']
write('owner-selected.json',{**select(owner,('owner_pid','driver_pid','deadline_seconds','driver_sha256','pins_sha256','resource_policy','stage','returncode','waited_reaped')),'absence':select(owner['absence'],('pid','absent','returncode')),'source':original(R/'owner-receipt.json')})
release=json.loads((R/'external-release.json').read_text());assert release['status']=='PASS_EXTERNAL_RELEASE_V128_FULL_CURVE' and len(release['pid_checks'])==23 and len(release['port_checks'])==21
assert all(p['absent'] for p in release['pid_checks']) and all(p['absent'] for p in release['port_checks'])
write('external-release-selected.json',{'status':release['status'],'source':original(R/'external-release.json'),'pid_checks':[select(p,('pid','returncode','absent')) for p in release['pid_checks']],'port_checks':release['port_checks'],'post_resource_selected':{'label':release['post_resource']['label'],'power':select(release['post_resource']['power_metadata'],('status','source','percent')),'thermal':select(release['post_resource']['processinfo_thermal_metadata'],('status','code','name')),'pressure_level_value':release['post_resource']['pressure_level_value']}})
write('requests-normalized.json',{'schema':'V128 safe numeric/token whitelist; no text, event payloads, headers, launch, env or process argv; timestamps are saved client perf_counter seconds, not wire capture','requests':requests})
write('groups-normalized.json',{'schema':'126 saved groups, 63 warm/63 formal; request rows joined by arm+request_id','groups':groups})
write('resources-whitelisted.json',{'scope':'outside-group saved metadata projections, raw command/process inventory omitted','samples':resources})
write('arms-selected.json',arms)
write('plot-data.json',{'scope':'one fresh V128 campaign only; correctness is an entry prerequisite, not a plot axis','metric_definitions':{'latency':'1000*median_3fresh(median_3formal_groups(median_requests(DONE-POST)))','throughput':'median_3fresh(median_3formal_groups(sum_completion_tokens/(last_DONE-earliest_POST)))'},'arms':pointrows,'configurations':configs,'joint_frontier':[c['config'] for c in configs if c['joint_non_dominated']],'pairs':pairs,'comparison_summary':ratios,'Lithos_C2_C4':None,'missing_reason':'pinned endpoint admission does not support C2/C4; no zero or serial surrogate point','connections':'guides joining observed engine-frontier configurations only; no interpolation or additional feasible points'})
with (S/'plot-data.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(pointrows[0]));w.writeheader();w.writerows(pointrows)
write('originals.json',originals)
print(json.dumps({'status':'PREPARED_SAFE_V128_EVIDENCE','groups':len(groups),'requests':len(requests),'formal_requests':sum(q['phase']=='measure' for q in requests),'resources':len(resources),'configs':configs,'joint_frontier':[c['config'] for c in configs if c['joint_non_dominated']],'comparison_summary':ratios},indent=2))
