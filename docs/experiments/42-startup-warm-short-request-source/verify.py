"""Saved V123 pilot identities/arithmetic/clock checks; standard library only."""
from pathlib import Path
import argparse,hashlib,json,math,statistics,ast
D=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--originals',action='store_true');a=p.parse_args()
def load(n):return json.loads((D/n).read_text())
def near(x,y):assert math.isclose(x,y,rel_tol=1e-12,abs_tol=1e-9),(x,y)
for name,v in load('manifest.json')['files'].items():
 b=(D.parent/name).read_bytes();assert len(b)==v['bytes'] and hashlib.sha256(b).hexdigest()==v['sha256'],name
g=load('requests-whitelisted.json')['groups'];assert len(g)==12;canonical=g[0]['raw_whitelisted']['requests'][0]['output_ids_when_available'];assert len(canonical)==32
points=load('plot-data.json');assert len(points)==6
for x in g:
 q=x['raw_whitelisted'];assert q['actual_request_count']==1 and q['total_completion_tokens']==32;r=q['requests'][0]
 assert r['output_ids_when_available']==canonical==r['visible_roundtrip_ids'];assert r['finish_reason']=='length';assert r['usage']['prompt_tokens']==19 and r['usage']['completion_tokens']==32 and r['usage']['prompt_tokens_details']['cached_tokens']==0
 assert ''.join(e['text'] for e in r['events'])==r['text'];assert all(e['engine_delivery'] is None for e in r['events'])
 near(r['done_perf_counter_s']-r['start_perf_counter_s'],r['client_total_s']);near(r['first_text_perf_counter_s']-r['start_perf_counter_s'],r['client_ttft_text_s']);near(r['client_total_s'],q['group_window_s']);near(32/q['group_window_s'],q['y_total_output_throughput_tok_s_mac'])
arms=load('arms-selected.json');assert len(arms)==2
for arm in arms:
 mode=arm['mode'];q=[x['raw_whitelisted'] for x in g if x['mode']==mode and x['phase']=='measure'];assert len(q)==3
 s=arm['summary'];assert s['all_raw32_match_old_fixture'];near(statistics.median(x['group_window_s'] for x in q)*1000,s['latency_p50_ms']);near(statistics.median(x['y_total_output_throughput_tok_s_mac'] for x in q),s['total_output_tok_s']);near(statistics.median(x['requests'][0]['client_ttft_text_s'] for x in q)*1000,s['ttft_p50_ms'])
 for x in [x for x in points if x['mode']==mode]:near(x['latency_ms'],q[x['repeat']]['group_window_s']*1000);near(x['throughput_tok_s'],q[x['repeat']]['y_total_output_throughput_tok_s_mac'])
 r=arm['startup_metadata'];assert r['model_loads']==1 and r['target_parameter_count']==1349 and r['draft_loads']==(mode=='speculative');assert r['capacity']==64 and r['cache_allocation_step']==64 and r['cache_policy']=='low64' and r['engine_cache_limit_gb']==0 and r['runtime_cache_limit_bytes']==64<<20
 warm=r['startup_ordinary_warm'];assert warm['raw_output_ids']==canonical and warm['raw32_equal_old_fixture'];assert warm['requests']==1 and warm['outside_HTTP_timing'] and warm['runner_idle_before_attach'] and warm['drafter_absent_during_warm'] and not warm['cause_of_resource_change_claimed']
 labels=[x['label'] for x in r['startup_allocator_stages']];assert labels.index('after_startup_ordinary_warm')>labels.index('before_startup_ordinary_warm')
 if mode=='speculative':assert labels.index('after_startup_ordinary_warm')<labels.index('before_draft_load')<labels.index('after_draft_load')
 else:assert 'before_draft_load' not in labels
 assert arm['owned_exit']['wait_completed'] and arm['owned_exit']['exact_pid_absence']['absent'] and arm['port_absence']['absent']
resources=load('resources-whitelisted.json')['samples'];assert len(resources)==38
for r in resources:
 assert r['power_metadata']['source']=='AC Power' and r['processinfo_thermal_metadata']['code']==0 and r['pressure_level_value']==1
 for x in g:
  if x['mode']==r['mode']:
   q=x['raw_whitelisted']['requests'][0];assert not q['start_perf_counter_s']<=r['monotonic_s']<=q['done_perf_counter_s'],'probe inside request'
assert len({r['swap_used_mb'] for r in resources})==1 and len({r['Swapouts'] for r in resources})==1
owner=load('owner.json');assert owner['status']=='COMPLETED_C1_SCREEN' and owner['stage']=='finished'
release=load('external-release.json');assert release['GPU_released'] and release['tool_exit_code']==0;assert all(x['absent'] for x in release['strict_absence']+release['ports'])
spec=next(x['summary'] for x in arms if x['mode']=='speculative');ordinary=next(x['summary'] for x in arms if x['mode']=='ordinary');ratio=spec['total_output_tok_s']/ordinary['total_output_tok_s'];near(ratio,.4880478012566465);assert ratio<1
script=(D/'corrected-launch.py').read_text();tree=ast.parse(script);assert 'for key in c.selected(env):env.pop(key,None)' in script and 'os.execve(' in script and '--execute-gpu' in script
if a.originals:
 for name,v in load('originals.json').items():
  b=Path(name).read_bytes();assert len(b)==v['bytes'] and hashlib.sha256(b).hexdigest()==v['sha256'],name
print('PASS: 12 HTTP / 384 tokens +2 startup /64; 6 measured /192; 38 outside-request AC/nominal/pressure1 samples; saved release; spec/ordinary throughput ratio %.12f'%ratio+('; original identities checked' if a.originals else ''))
