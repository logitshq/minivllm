"""CPU saved-evidence arithmetic/provenance replay. No MLX/GPU/live probes."""
from pathlib import Path
import argparse,hashlib,json,math,statistics
D=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--originals',action='store_true');a=p.parse_args()
def load(n):return json.loads((D/n).read_text())
def near(x,y):assert math.isclose(x,y,rel_tol=1e-12,abs_tol=1e-9),(x,y)
for name,x in load('manifest.json')['files'].items():
 b=(D.parent/name).read_bytes();assert len(b)==x['bytes'] and hashlib.sha256(b).hexdigest()==x['sha256'],name
groups=load('requests-whitelisted.json')['groups'];assert len(groups)==54;assert sum(x['phase']=='measure' for x in groups)==27
canonical=next(x for x in groups if x['engine']=='native')['raw_whitelisted']['requests'][0]['output_ids_when_available'];assert len(canonical)==32
visible={}
for x in groups:
 g=x['raw_whitelisted'];q=g['requests'][0];assert g['actual_request_count']==1 and g['total_completion_tokens']==32;assert len(q['visible_roundtrip_ids'])==32;assert q['usage']['prompt_tokens']==19 and q['usage']['completion_tokens']==32 and q['finish_reason']=='length'
 if 'prompt_tokens_details' in q['usage']:assert q['usage']['prompt_tokens_details']['cached_tokens']==0
 else:assert x['engine']=='lithos' # missing usage field is unknown, not a measured zero
 if x['engine']=='native':assert q['output_ids_when_available']==canonical==q['visible_roundtrip_ids']
 else:assert q['output_ids_when_available'] is None
 visible.setdefault(x['engine'],q['visible_roundtrip_ids']);assert visible[x['engine']]==q['visible_roundtrip_ids']
 assert ''.join(e['text'] for e in q['events'])==q['text']
 near(q['done_perf_counter_s']-q['start_perf_counter_s'],q['client_total_s']);near(q['client_total_s'],g['group_window_s']);near(q['first_text_perf_counter_s']-q['start_perf_counter_s'],q['client_ttft_text_s']);near(32/g['group_window_s'],g['y_total_output_throughput_tok_s_mac'])
suite=load('suite-complete.json');assert suite['status']=='PASS_C1_SCREEN' and len(suite['arms'])==9;assert suite['HTTP_requests']==54 and suite['HTTP_completion_tokens']==1728 and suite['measured_requests']==27 and suite['measured_tokens']==864 and suite['native_startup_ordinary_requests']==3 and suite['native_startup_ordinary_tokens']==96
plot=load('plot-data.json');assert len(plot['arms'])==9 and len(plot['configurations'])==3
for arm in suite['arms']:
 gg=[x['raw_whitelisted'] for x in groups if x['arm']==arm['label'] and x['phase']=='measure'];assert len(gg)==3
 near(statistics.median(q['requests'][0]['client_total_s'] for q in gg)*1000,arm['latency_p50_ms']);near(arm['E2E_s']*1000,arm['latency_p50_ms']);near(statistics.median(q['y_total_output_throughput_tok_s_mac'] for q in gg),arm['y']);near(statistics.median(q['requests'][0]['client_ttft_text_s'] for q in gg),arm['ttft_s'])
 point=next(q for q in plot['arms'] if q['arm']==arm['label']);near(point['latency_ms'],arm['latency_p50_ms']);near(point['throughput_tok_s'],arm['y'])
for c in plot['configurations']:
 aa=[x for x in plot['arms'] if x['engine']==c['engine']];assert len(aa)==3;near(c['latency_ms'],statistics.median(x['latency_ms'] for x in aa));near(c['throughput_tok_s'],statistics.median(x['throughput_tok_s'] for x in aa));near(c['ttft_ms'],statistics.median(x['ttft_ms'] for x in aa));near(c['latency_ms'],suite['medians'][c['engine']]['latency_p50_ms']);near(c['throughput_tok_s'],suite['medians'][c['engine']]['y'])
 dom=[x['engine'] for x in plot['configurations'] if x['latency_ms']<=c['latency_ms'] and x['throughput_tok_s']>=c['throughput_tok_s'] and(x['latency_ms']<c['latency_ms'] or x['throughput_tok_s']>c['throughput_tok_s'])];assert dom==c['dominated_by'] and c['joint_non_dominated']==(not dom)
assert [c['engine'] for c in plot['configurations'] if c['joint_non_dominated']]==['native']
for block in (1,2,3):
 n=next(x for x in plot['arms'] if x['engine']=='native' and x['block']==block);o=next(x for x in plot['arms'] if x['engine']=='omlx' and x['block']==block);assert n['latency_ms']<o['latency_ms'] and n['throughput_tok_s']>o['throughput_tok_s']
reviewed=load('reviewed_rows.json')
for point in plot['arms']:
 r=next(x for x in reviewed['fresh_arms'] if x['arm']==point['arm']);near(point['latency_ms'],r['latency_p50_ms']);near(point['throughput_tok_s'],r['throughput_tok_s'])
for point in plot['configurations']:
 r=next(x for x in reviewed['configuration_medians'] if x['engine']==point['engine']);near(point['latency_ms'],r['latency_p50_ms']);near(point['throughput_tok_s'],r['throughput_tok_s'])
assert not reviewed['full_frontier_win'] and not reviewed['statistical_significance_claim'] and not reviewed['legacy_first_text_proxy_primary']
capture=load('capture-scope.json');assert not capture['actual_wire_POST_bodies_captured'] and not capture['actual_wire_SSE_bytes_captured'] and not capture['authentication_headers_or_argv_exported']
resources=load('resources-whitelisted.json')['samples'];assert len(resources)==155 and sum(x['engine']=='campaign' for x in resources)==2
for r in resources:
 assert r['power_metadata']['source']=='AC Power' and r['processinfo_thermal_metadata']['code']==0 and r['pressure_level_value']==1
 for g in groups:
  q=g['raw_whitelisted']['requests'][0];assert not q['start_perf_counter_s']<=r['monotonic_s']<=q['done_perf_counter_s'],'probe inside request'
assert len({r['Swapouts'] for r in resources})==1
ordered=sorted(resources,key=lambda x:x['monotonic_s']);assert max(r['swap_used_mb'] for r in resources)<=ordered[0]['swap_used_mb'];assert ordered[-1]['swap_used_mb']<=ordered[0]['swap_used_mb']
for arm in load('arms-selected.json'):
 assert arm['cleanup']['owned_exit']['wait_completed'] and arm['cleanup']['owned_exit']['exact_pid_absence']['absent'] and arm['port_absence']['absent']
 if arm['engine']=='native':
  n=arm['native_selected'];r=arm['runtime_identity_selected'];assert r['fixed_cap']==64 and r['gates_on']==48 and r['model_loads']==1 and r['startup_ordinary_warm_requests']==1 and r['startup_ordinary_warm_tokens']==32
  assert n['native_steps']==186 and n['request_count']==6 and n['active'] is None and n['feedback']['resets']==6
  assert n['pipeline']['total_completed']==n['pipeline']['total_delivered']==186 and n['pipeline']['total_discarded']==n['pipeline']['total_failed']==0 and n['icb']['bank_executions']==[96,90]
  assert n['pingpong']['eligible_count']==48 and n['direct_pipeline']['direct_links']==83 and not n['direct_pipeline']['diagnostic_snapshot']
for path in sorted(D.glob('cool-gap-*.json')):
 c=json.loads(path.read_text());assert c['requested_seconds']==60 and c['end_monotonic_s']-c['start_monotonic_s']>=60 and c['outside_HTTP_clocks'] and c['prior_owned_cleanup_passed'] and not c['adaptive']
 for g in groups:
  q=g['raw_whitelisted']['requests'][0];assert q['done_perf_counter_s']<=c['start_monotonic_s'] or q['start_perf_counter_s']>=c['end_monotonic_s']
assert len(list(D.glob('cool-gap-*.json')))==8
r=load('external-release.json');assert r['GPU_released'] and r['tool_exit_code']==0 and len(r['strict_absence'])==11 and len(r['ports'])==9 and all(x['absent'] for x in r['strict_absence']+r['ports'])
r=load('owner-selected.json');assert r['returncode']==0 and r['waited_reaped'] and r['absence']['absent']
prior=load('prior105-stop-reference.json');assert prior['completed_arms']==8 and prior['error_message']=='nominal thermal required'
if a.originals:
 for name,x in load('originals.json').items():
  b=Path(name).read_bytes();assert len(b)==x['bytes'] and hashlib.sha256(b).hexdigest()==x['sha256'],name
print('PASS: fresh C1 9 arms /3 configurations; 54 HTTP1728 +3 startup96; 27 measured864; 155 outside-request AC0pressure1 samples; saved11PID/9port release; native-only non-dominated C1 configuration'+('; original identities checked' if a.originals else ''))
