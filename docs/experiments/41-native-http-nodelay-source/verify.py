"""Recompute saved pilot statistics and payload identities; no MLX/GPU/socket probes."""
from pathlib import Path
import argparse,hashlib,json,math,statistics
D=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--originals',action='store_true');args=p.parse_args()
def load(n):return json.loads((D/n).read_text())
def near(a,b):assert math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-10),(a,b)
manifest=load('manifest.json')
for name,item in manifest['files'].items():
 b=(D.parent/name).read_bytes();assert len(b)==item['bytes'];assert hashlib.sha256(b).hexdigest()==item['sha256'],name
g=load('requests-whitelisted.json')['groups'];assert len(g)==12
canonical=g[0]['raw_whitelisted']['requests'][0]['output_ids_when_available'];assert len(canonical)==32
for r in g:
 x=r['raw_whitelisted'];assert x['actual_request_count']==1;assert x['total_completion_tokens']==32
 q=x['requests'][0];assert q['output_ids_when_available']==canonical==q['visible_roundtrip_ids'];assert q['usage']['prompt_tokens']==19;assert q['usage']['completion_tokens']==32;assert q['usage']['prompt_tokens_details']['cached_tokens']==0;assert q['finish_reason']=='length';assert len(q['events'])==32
 assert [e['engine_delivery']['id'] for e in q['events']]==canonical
 near(q['done_perf_counter_s']-q['start_perf_counter_s'],q['client_total_s']);near(x['group_window_s'],q['client_total_s']);near(32/x['group_window_s'],x['y_total_output_throughput_tok_s_mac'])
result=load('result.json');assert result['status']=='STOP_FIRST_PAIR_NO_E2E_GAIN' and result['complete_pairs']==1 and len(result['arms'])==2
points=load('plot-data.json');assert len(points)==6
for arm in ('off','on'):
 r=[x['raw_whitelisted'] for x in g if x['arm']==arm and x['phase']=='measure'];assert len(r)==3
 a=next(x for x in result['arms'] if x['tcp_nodelay']==arm)
 near(statistics.median(x['group_window_s'] for x in r),a['E2E_s']);near(statistics.median(x['y_total_output_throughput_tok_s_mac'] for x in r),a['y']);near(statistics.median(x['requests'][0]['client_ttft_text_s'] for x in r),a['ttft_s'])
 for x in [x for x in points if x['arm']==arm]:
  rr=r[x['repeat']];near(x['latency_ms'],rr['group_window_s']*1000);near(x['throughput_tok_s'],rr['y_total_output_throughput_tok_s_mac'])
a,b=result['arms'];near(a['E2E_s']/b['E2E_s'],result['pairs'][0]['E2E_speedup']);assert b['E2E_s']>=a['E2E_s']
resources=load('resources-whitelisted.json')['samples'];assert len(resources)==34
for x in resources:assert x['pressure_level_value']==1 and x['power_metadata']['source']=='AC Power' and x['processinfo_thermal_metadata']['code']==0
assert len({x['swap_used_mb'] for x in resources})==1 and len({x['Swapouts'] for x in resources})==1
for a in load('arms-selected.json'):
 assert a['socket']['tcp_nodelay_actual']==(0 if a['arm']=='off' else 4);assert a['socket']['tcp_nodelay_enabled']==(a['arm']=='on')
 n=a['native_selected'];assert n['native_steps']==186 and n['request_count']==6 and n['active'] is None
 assert n['feedback']['resets']==6 and n['feedback']['capacity']==64
 assert n['pipeline']['total_delivered']==186 and n['pipeline']['total_completed']==186 and n['pipeline']['total_discarded']==0
 assert n['icb']['execution_count']==186 and n['icb']['bank_executions']==[96,90]
 assert a['cleanup']['owned_exit']['wait_completed'] and a['cleanup']['owned_exit']['exact_pid_absence']['absent']
release=load('external-release.json');assert release['GPU_released'] and release['actual_HTTP_requests']==12 and release['formal_HTTP_requests']==6
assert all(x['absent'] for x in release['owned_pids']+release['ports'])
owner=load('owner.json');assert owner['returncode']==2 and owner['waited_reaped'] and owner['absence']['absent']
if args.originals:
 for name,item in load('originals.json').items():
  b=Path(name).read_bytes();assert len(b)==item['bytes'] and hashlib.sha256(b).hexdigest()==item['sha256'],name
print('PASS: 12 groups / 6 measured / 384 HTTP tokens; medians, 0.997538352 ratio, sockets0/4, 34 resource samples, saved cleanup and payload hashes'+('; original paths checked' if args.originals else ''))
