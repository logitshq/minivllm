"""Portable standard-library replay of public V130 rows; no live/model/GPU operations."""
from pathlib import Path
import hashlib,json,statistics
D=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 rows=json.loads((D/'requests-normalized.json').read_text())['requests'];groups=json.loads((D/'groups-normalized.json').read_text())['groups'];data=json.loads((D/'lithos-perf-data.json').read_text());suite=json.loads((D/'suite-complete.json').read_text())
 assert len(rows)==134 and len(groups)==58
 assert sum(r['run_kind']=='qualification' for r in rows)==8 and sum(r['run_kind']=='performance' for r in rows)==126
 assert sum(r['formal_eligible'] for r in rows)==63 and sum(r['usage']['completion_tokens'] for r in rows if r['formal_eligible'])==2016
 qmap={(q['run_kind'],q['arm'],q['request_id']):q for q in rows};assert len(qmap)==134
 rebuilt=[]
 for g in groups:
  qq=[qmap[g['run_kind'],g['arm'],i] for i in g['request_ids']];C=g['requested_concurrency'];assert len(qq)==C
  for q in qq:
   assert q['HTTP_status']==200 and q['finish_reason']=='length' and q['all_correctness_checks_passed'] and q['retry_count']==0
   assert q['internal_raw_ids'] is None and len(q['visible_roundtrip_ids'])==32
   assert (q['usage']['prompt_tokens'],q['usage']['completion_tokens'])==(19,32)
   assert q['complete_latency_s']==q['terminal_perf_counter_s']-q['start_perf_counter_s']
   assert q['cleanup_perf_counter_s']>=q['terminal_perf_counter_s']
   if q['protocol']=='ollama':
    assert q['terminal']=='NDJSON_DONE_TRUE' and q['server_terminal_fields']['done'] is True and q['server_terminal_fields']['done_reason']=='length'
    assert (q['server_terminal_fields']['prompt_eval_count'],q['server_terminal_fields']['eval_count'])==(19,32)
   else:assert q['terminal']=='SSE_DONE' and q['run_kind']=='qualification'
  lo=min(q['start_perf_counter_s'] for q in qq);hi=max(q['terminal_perf_counter_s'] for q in qq)
  lat=1000*statistics.median(q['terminal_perf_counter_s']-q['start_perf_counter_s'] for q in qq);y=sum(q['usage']['completion_tokens'] for q in qq)/(hi-lo)
  assert lat==g['latency_p50_ms'] and y==g['total_throughput_tok_s']
  assert max(q['start_perf_counter_s'] for q in qq)<min(q['terminal_perf_counter_s'] for q in qq)
  assert g['formal_eligible']==(g['run_kind']=='performance' and g['phase']=='measure')
  rebuilt.append(dict(g,latency=lat,throughput=y))
 arms=[]
 for arm in data['arms']:
  gg=[g for g in rebuilt if g['arm']==arm['label'] and g['formal_eligible']];assert len(gg)==3
  lat=statistics.median(g['latency'] for g in gg);y=statistics.median(g['throughput'] for g in gg)
  assert lat==arm['latency_p50_ms'] and y==arm['total_throughput_tok_s'];arms.append(arm)
 assert len(arms)==9 and sum(a['concurrency']*6 for a in arms)==126
 for c in data['configurations']:
  aa=[a for a in arms if a['concurrency']==c['concurrency']];assert len(aa)==3
  assert c['latency_p50_ms']==statistics.median(a['latency_p50_ms'] for a in aa)==suite['medians'][str(c['concurrency'])]['latency_p50_ms']
  assert c['total_throughput_tok_s']==statistics.median(a['total_throughput_tok_s'] for a in aa)==suite['medians'][str(c['concurrency'])]['total_throughput_tok_s']
 assert len({tuple(q['visible_roundtrip_ids']) for q in rows})==1
 qs=json.loads((D/'qualification-selected.json').read_text());assert qs['HTTP_requests']==8 and qs['completion_tokens']==256
 qp=json.loads((D/'qualification-participation-selected.json').read_text());assert qp['generation_count']==qp['cached_prompt_tokens0_count']==qp['prefix_source_None_count']==8
 selected=json.loads((D/'arms-selected.json').read_text())
 for arm in selected:
  C=arm['concurrency'];assert arm['participation']['generation_count']==arm['participation']['cached_prompt_tokens0_count']==arm['participation']['prefix_source_None_count']==6*C
  assert all(q['cached_prompt_tokens']==0 and q['prefix_source'] is None for q in arm['participation']['Generation_cache'])
  assert arm['cleanup']['owned_exit']['wait_completed'] and arm['cleanup']['owned_exit']['exact_pid_absence']['absent'] and arm['port_absence']['absent']
 resources=json.loads((D/'resources-whitelisted.json').read_text())['samples'];assert len(resources)==175
 for r in resources:assert r['power']['source']=='AC Power' and r['thermal']['code']==0 and r['pressure_level_value']==1
 for kind in ('qualification','performance'):
  rr=sorted((r for r in resources if r['run_kind'].startswith(kind)),key=lambda x:x['monotonic_s'])
  assert all(r['vm_counters']['Swapouts']==rr[0]['vm_counters']['Swapouts'] and r['swap_used_mb']<=rr[0]['swap_used_mb'] for r in rr)
 for r in resources:
  for g in rebuilt:
   if r['arm']==g['arm']:assert not(g['earliest_pre_POST_perf_counter_s']<=r['monotonic_s']<=g['last_terminal_perf_counter_s'])
 cool=[json.loads(p.read_text()) for p in sorted((D/'cooling').glob('*.json'))];assert len(cool)==8
 assert all(c['requested_seconds']==60 and c['end_monotonic_s']-c['start_monotonic_s']>=60 and c['outside_HTTP_clocks'] and c['prior_owned_cleanup_passed'] and not c['adaptive'] for c in cool)
 releases=json.loads((D/'releases-selected.json').read_text());assert len(releases)==2
 for r in releases:assert all(x['absent'] for x in r['pid_checks']) and all(x['absent'] for x in r['port_checks'])
 perf=next(r for r in releases if r['run_kind']=='performance');assert len(perf['pid_checks'])==11 and len(perf['port_checks'])==9
 checked=0
 if (D/'manifest.json').exists():
  manifest=json.loads((D/'manifest.json').read_text())
  for name,v in manifest['files'].items():assert sha(D/name)==v['sha256'] and (D/name).stat().st_size==v['bytes'],name;checked+=1
 return {'status':'PASS_PORTABLE_V130_SAVED_ROWS_REPLAY','scope':'author offline replay, not independent raw/GPU audit','qualification_requests':8,'performance_requests':126,'formal_requests':63,'formal_tokens':2016,'groups':58,'performance_groups':54,'performance_arms':9,'configuration_medians':data['configurations'],'all_visible_streams_same':True,'resources':175,'fixed_power':'AC','cooling_gaps':8,'performance_PID_absences':11,'performance_port_absences':9,'manifest_payloads_checked':checked,'qualification_warm_ranked':False,'GPU_used':False}
if __name__=='__main__':print(json.dumps(main(),indent=2))
