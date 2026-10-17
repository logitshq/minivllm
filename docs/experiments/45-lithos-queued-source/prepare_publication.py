"""CPU-only safe V130 publication projection; no runtime/model imports."""
from pathlib import Path
import hashlib,json,statistics
P=Path('/private/tmp/minivllm-engine-optimization-20261014');O=Path(__file__).resolve().parent;S=O/'45-lithos-queued-source';R=P/'v23-root-control-build1';D=P/'v130-lithos-load-probe-prework-build1';A=P/'v130-lithos-concurrency-source-audit-build1';PERF=R/'lithos-ollama-v130perf-attempt1';QUAL=R/'lithos-ollama-v130qual-attempt1'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();originals={}
def orig(p):
 p=Path(p);v={'sha256':sha(p),'bytes':p.stat().st_size};originals[str(p)]=v;return {'path':str(p),**v}
def write(n,x):
 p=S/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def exact(p,n):
 p=Path(p);dest=S/n;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(p.read_bytes());assert sha(dest)==sha(p);orig(p)
def selected(v,keys):return {k:v[k] for k in keys if k in v}
def absence(v):return selected(v,('pid','returncode','absent','strict_absent'))
def exit_selected(v):return {**selected(v,('pid','wait_completed','returncode','recorded_birth_identity_sha256')),'exact_pid_absence':absence(v['exact_pid_absence'])}
for n in ('client.py','driver.py','qualify.py','README.md','SOURCE-REVIEW.md','plan-nine-arms.json','check_cpu.py'):
 exact(D/n,'source-v130/'+n)
for n in ('receipt.json','REVIEW.md','request-template.json','cpu-result.json'):
 exact(A/n,'source-audit/'+n)
# Verify all original pins without copying the vendor tree or its large manifest into the public bundle.
pins=json.loads((D/'pins.json').read_text())
for path,h in pins['files'].items():assert sha(path)==h,path
critical={p:h for p,h in pins['files'].items() if any(p.endswith('/'+n) for n in ('monolith/serve.py','monolith/serving/lifecycle.py','monolith/serving/protocol.py','monolith/serving/events.py','m4-final-native15-official-attempt2/official_entry.py','m4-final-native15-official-attempt2/client.py','m4-final-native15-official-attempt2/common.py'))}
write('source-identities.json',{'source_pin_original':orig(D/'pins.json'),'original_pin_count':len(pins['files']),'all_original_pins_rehashed':True,'critical_source_identities':critical,'source_audit_receipt':orig(A/'receipt.json'),'full_vendor_tree_copied':False,'scope':'original installed Lithos0.1.2/commit5485; no vendor patch or custom queue; snapshots are source code, not actual process argv/env'})
qs=json.loads((QUAL/'driver-output/receipt.json').read_text());suite=json.loads((PERF/'driver-output/suite-complete.json').read_text());assert qs['fixed_power']==suite['fixed_power']=='AC'
assert qs['HTTP_requests']==8 and suite['HTTP_requests']==126 and suite['formal_requests']==63
exact(PERF/'driver-output/suite-complete.json','suite-complete.json')
requests=[];groups=[];arms=[];resources=[];release_rows=[]
def extract_group(path,run_kind,arm,replica=None,phase=None):
 g=json.loads(path.read_text());ref=orig(path);C=g['requested_concurrency'];qq=[]
 for q in g['requests']:
  assert q['HTTP_status']==200 and q['error'] is None and q['retry_count']==0 and q['all_correctness_checks_passed']
  assert q['finish_reason']=='length' and (q['usage']['prompt_tokens'],q['usage']['completion_tokens'])==(19,32)
  assert len(q['visible_roundtrip_ids'])==32 and q['internal_raw_ids'] is None
  assert q['terminal']=='NDJSON_DONE_TRUE' if q['protocol']=='ollama' else q['terminal']=='SSE_DONE'
  terminal=q['events'][-1]['value']
  record={'run_kind':run_kind,'arm':arm,'replica':replica,'phase':phase,'formal_eligible':run_kind=='performance' and phase=='measure','group_id':g['group_id'],'concurrency':C,'source':ref,**selected(q,('request_id','protocol','endpoint','HTTP_status','start_perf_counter_s','headers_perf_counter_s','first_text_perf_counter_s','terminal_perf_counter_s','cleanup_perf_counter_s','complete_latency_s','terminal','finish_reason','usage','visible_roundtrip_ids','internal_raw_ids','all_correctness_checks_passed','retry_count','server_queue_wait_s_observed')),'output_text_sha256':hashlib.sha256(q['text'].encode()).hexdigest(),'raw_event_count':len(q['events']),'server_terminal_fields':selected(terminal,('done','done_reason','prompt_eval_count','eval_count','total_duration','load_duration','prompt_eval_duration','eval_duration'))}
  requests.append(record);qq.append(record)
 lo=min(q['start_perf_counter_s'] for q in qq);hi=max(q['terminal_perf_counter_s'] for q in qq);lat=1000*statistics.median(q['terminal_perf_counter_s']-q['start_perf_counter_s'] for q in qq);y=32*C/(hi-lo)
 assert lat==g['latency_p50_ms'] and y==g['total_throughput_tok_s'] and g['all_correctness_checks_passed']
 gr={'run_kind':run_kind,'arm':arm,'replica':replica,'phase':phase,'formal_eligible':run_kind=='performance' and phase=='measure','source':ref,**selected(g,('group_id','protocol','requested_concurrency','actual_request_count','successful_requests','rejected_requests','all_requested_POSTs_before_first_terminal','arrival_spread_s','all_attempts_window_s','latency_p50_ms','total_throughput_tok_s','completed_token_count','all_correctness_checks_passed','retries','queue_wait_in_complete_client_latency','boundary')),'earliest_pre_POST_perf_counter_s':lo,'last_terminal_perf_counter_s':hi,'request_ids':[q['request_id'] for q in qq]};groups.append(gr);return gr

def resource(path,kind,arm):
 x=json.loads(path.read_text());power=x['power_metadata'];thermal=x['processinfo_thermal_metadata'];assert power['source']=='AC Power' and thermal['code']==0 and x['pressure_level_value']==1
 resources.append({'run_kind':kind,'arm':arm,'source':orig(path),**selected(x,('label','recorded_at','monotonic_s','owned_model_pid','pressure_level_value','free_percent','swap_used_mb','known_inference_process_count','only_owned_known_inference_processes')),'power':selected(power,('status','source','percent','state_text')),'thermal':selected(thermal,('status','code','name')),'vm_counters':{k:v for k,v in x['vm_counters'].items() if isinstance(v,(int,float))}})

def cleanup_record(path):
 q=json.loads(path.read_text());assert q['failure'] is None and not q['cleanup_errors'] and q['owned_exit']['wait_completed'] and q['owned_exit']['exact_pid_absence']['absent'];orig(path)
 return {'failure':None,'cleanup_errors':[],'owned_exit':exit_selected(q['owned_exit'])}
for entry in qs['groups']:extract_group(QUAL/'driver-output'/(entry['group_id']+'.json'),'qualification','qualification',phase='qualification')
for path in sorted((QUAL/'driver-output').glob('resources-*-raw.json')):resource(path,'qualification','qualification')
write('qualification-selected.json',{**selected(qs,('status','fixed_power','HTTP_requests','completion_tokens','groups','source_pins_sha256','gold_is_saved_visible32_not_internal_raw','no_automatic_retry','no_new_performance_point','custom_queue','GPU_batch_claim','HTTP_arrival_fairness_claim','old_429_raw_observed')),'source':orig(QUAL/'driver-output/receipt.json'),'cleanup':cleanup_record(QUAL/'driver-output/cleanup.json')})
qpart=json.loads((QUAL/'driver-output/participation.json').read_text());assert qpart['generation_count']==qpart['cached_prompt_tokens0_count']==qpart['prefix_source_None_count']==8
write('qualification-participation-selected.json',{**selected(qpart,('backend','draft','original_log_sha256','generation_count','cached_prompt_tokens0_count','prefix_source_None_count','GPU_batch_observed','queue_policy','HTTP_arrival_fairness_claim')),'source':orig(QUAL/'driver-output/participation.json'),'Generation_cache':[{k:m[k] for k in ('cached_prompt_tokens','prefix_source')} for m in qpart['Generation_records']]})
for arm in suite['arms']:
 label=arm['label'];C=arm['concurrency'];replica=arm['replica'];root=PERF/'driver-output'/label;formal=[]
 for phase in ('warm','measure'):
  for i in range(3):
   g=extract_group(root/f'C{C}-{phase}-{i}.json','performance',label,replica,phase)
   if phase=='measure':formal.append(g)
 lat=statistics.median(g['latency_p50_ms'] for g in formal);y=statistics.median(g['total_throughput_tok_s'] for g in formal);assert lat==arm['latency_p50_ms'] and y==arm['total_throughput_tok_s']
 r=json.loads((root/'receipt.json').read_text());part=json.loads((root/'participation.json').read_text());assert part['generation_count']==part['cached_prompt_tokens0_count']==6*C and part['prefix_source_None_count']==6*C
 port=json.loads((root/'port-absence.json').read_text());assert port['absent']
 owner_cleanup=cleanup_record(root/'cleanup.json')
 ar={**arm,'run_kind':'performance','source':orig(root/'receipt.json'),'protocol':'official Ollama NDJSON','cold_and_qualification_excluded':True,'inherited_no_new_performance_point_flag':r['no_new_performance_point'],'flag_interpretation':'reused qualifier metadata, not a denial of completed formal driver measurements','cleanup':owner_cleanup,'port_absence':port,'participation':{**selected(part,('backend','draft','original_log_sha256','generation_count','cached_prompt_tokens0_count','prefix_source_None_count','GPU_batch_observed','queue_policy','HTTP_arrival_fairness_claim')),'Generation_cache':[{k:m[k] for k in ('cached_prompt_tokens','prefix_source')} for m in part['Generation_records']]}}
 arms.append(ar);orig(root/'participation.json');orig(root/'port-absence.json')
 for path in sorted(root.glob('resources-*-raw.json')):resource(path,'performance',label)
for path in sorted((PERF/'driver-output').glob('campaign-resources-*-raw.json')):resource(path,'performance-campaign','campaign')
for path in sorted((PERF/'driver-output').glob('cool-gap-*.json')):exact(path,'cooling/'+path.name)
for kind,root in [('qualification',QUAL),('performance',PERF)]:
 owner=json.loads((root/'owner-receipt.json').read_text());assert owner['returncode']==0 and owner['waited_reaped']
 write(f'{kind}-owner-selected.json',{**selected(owner,('owner_pid','driver_pid','deadline_seconds','script_sha256','pins_sha256','fixed_power','stage','returncode','waited_reaped')),'driver_absence':absence(owner['driver_absence']),'source':orig(root/'owner-receipt.json')})
 release=json.loads((root/'external-release.json').read_text());assert release['status'].startswith('PASS') and all(p['absent'] for p in release['strict_exact_pid_absences']) and all(p['absent'] for p in release['ports'])
 release_rows.append({'run_kind':kind,'source':orig(root/'external-release.json'),'status':release['status'],'pid_checks':[absence(p) for p in release['strict_exact_pid_absences']],'port_checks':release['ports']})
 resource(root/'external-release-resource-raw.json',kind+'-external-release','external-release')
configs=[]
for C in (1,2,4):
 aa=[a for a in arms if a['concurrency']==C];assert len(aa)==3
 configs.append({'engine':'lithos','protocol':'official Ollama NDJSON','concurrency':C,'fresh_arms':3,'latency_p50_ms':statistics.median(a['latency_p50_ms'] for a in aa),'total_throughput_tok_s':statistics.median(a['total_throughput_tok_s'] for a in aa)})
 assert configs[-1]['latency_p50_ms']==suite['medians'][str(C)]['latency_p50_ms'] and configs[-1]['total_throughput_tok_s']==suite['medians'][str(C)]['total_throughput_tok_s']
assert len(requests)==134 and len(groups)==58 and len(arms)==9 and sum(q['formal_eligible'] for q in requests)==63
assert sum(q['usage']['completion_tokens'] for q in requests)==4288
write('requests-normalized.json',{'schema':'134 numeric/token whitelist rows; qualification8 and performance126 remain separate; raw text/events/headers/auth/env omitted','requests':requests})
write('groups-normalized.json',{'schema':'58 groups=54 performance (27 warm+27 formal)+4 qualification; only63 measured requests ranked','groups':groups})
write('arms-selected.json',arms);write('resources-whitelisted.json',{'scope':'all saved selected-resource metadata; power domains and counters retained; raw processes/commands omitted','samples':resources});write('releases-selected.json',release_rows)
write('lithos-perf-data.json',{'scope':'one separate fresh V130 official Ollama queued-load slice, not paired/pool with old V128','fixed_power':suite['fixed_power'],'arms':[{k:a[k] for k in ('label','concurrency','replica','latency_p50_ms','total_throughput_tok_s')} for a in arms],'configurations':configs,'formal_requests':63,'formal_tokens':2016,'qualification_or_warm_ranked':False,'GPU_batch_claim':False,'HTTP_arrival_fairness_claim':False,'cross_campaign_pooled':False})
write('originals.json',originals)
print(json.dumps({'status':'PREPARED_SAFE_V130_DATA','qualification_requests':8,'performance_requests':126,'formal_requests':63,'groups':58,'arms':9,'resources':len(resources),'configurations':configs},indent=2))
