"""Offline V128 publication replay using only allowlisted saved rows (stdlib)."""
from pathlib import Path
import hashlib,json,math,statistics
D=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
 rows=json.loads((D/'requests-normalized.json').read_text())['requests']
 groups=json.loads((D/'groups-normalized.json').read_text())['groups']
 plot=json.loads((D/'plot-data.json').read_text());suite=json.loads((D/'suite-complete.json').read_text())
 assert len(rows)==270 and len(groups)==126 and sum(r['phase']=='measure' for r in rows)==135
 assert sum(r['usage']['completion_tokens'] for r in rows)==8640
 assert sum(r['engine']=='native' for r in rows)==126 and sum(r['engine']!='native' for r in rows)==144
 qmap={(q['arm'],q['request_id']):q for q in rows};assert len(qmap)==270
 rebuilt=[]
 for g in groups:
  qq=[qmap[g['arm'],i] for i in g['request_ids']];assert len(qq)==g['concurrency']
  lo=min(q['start_perf_counter_s'] for q in qq);hi=max(q['done_perf_counter_s'] for q in qq)
  lat=1000*statistics.median(q['done_perf_counter_s']-q['start_perf_counter_s'] for q in qq)
  y=sum(q['usage']['completion_tokens'] for q in qq)/(hi-lo)
  assert lat==g['latency_p50_ms'] and y==g['y_total_output_throughput_tok_s_mac']
  for q in qq:
   assert q['duration_ms']==1000*(q['done_perf_counter_s']-q['start_perf_counter_s'])
   assert q['finish_reason']=='length' and (q['usage']['prompt_tokens'],q['usage']['completion_tokens'])==(19,32)
   assert q['usage']['cached_tokens_observed'] in (0,None)
   assert len(q['visible_roundtrip_ids'])==32
   if q['engine']=='native':assert q['output_ids_when_available']==q['visible_roundtrip_ids']
   else:assert q['output_ids_when_available'] is None
  rebuilt.append(dict(g,latency_ms=lat,throughput_tok_s=y))
 ar=[]
 for a in plot['arms']:
  gg=[g for g in rebuilt if g['arm']==a['arm'] and g['phase']=='measure'];assert len(gg)==3
  lat=statistics.median(g['latency_ms'] for g in gg);y=statistics.median(g['throughput_tok_s'] for g in gg)
  assert lat==a['latency_ms'] and y==a['throughput_tok_s']
  old=next(s for s in suite['arms'] if s['label']==a['arm']);assert old['latency_p50_ms']==lat and old['y']==y
  ar.append(dict(a,latency_ms=lat,throughput_tok_s=y))
 cc=[]
 for c in plot['configurations']:
  aa=[a for a in ar if a['config']==c['config']];assert len(aa)==3
  lat=statistics.median(a['latency_ms'] for a in aa);y=statistics.median(a['throughput_tok_s'] for a in aa)
  assert lat==c['latency_ms'] and y==c['throughput_tok_s']
  cc.append(dict(c,latency_ms=lat,throughput_tok_s=y))
 def dominates(a,b):return a['latency_ms']<=b['latency_ms'] and a['throughput_tok_s']>=b['throughput_tok_s'] and (a['latency_ms']<b['latency_ms'] or a['throughput_tok_s']>b['throughput_tok_s'])
 frontier=[]
 for c in cc:
  d=[a['config'] for a in cc if dominates(a,c)];assert d==c['dominated_by'] and (not d)==c['joint_non_dominated']
  own=not any(dominates(a,c) for a in cc if a['engine']==c['engine']);assert own==c['engine_non_dominated']
  if not d:frontier.append(c['config'])
 assert frontier==plot['joint_frontier']==['native:C1','native:C2','native:C4']
 for p in plot['pairs']:
  c=p['concurrency'];r=p['replica'];other=p['comparison'].removeprefix('native_vs_')
  a=next(x for x in ar if (x['engine'],x['concurrency'],x['replica'])==('native',c,r))
  b=next(x for x in ar if (x['engine'],x['concurrency'],x['replica'])==(other,c,r))
  assert p['latency_reduction_percent']==100*(1-a['latency_ms']/b['latency_ms'])
  assert p['throughput_gain_percent']==100*(a['throughput_tok_s']/b['throughput_tok_s']-1)
 for s in plot['comparison_summary']:
  n=next(c for c in cc if c['config']==f"native:C{s['concurrency']}");e=s['comparison'].removeprefix('native_vs_');o=next(c for c in cc if c['config']==f"{e}:C{s['concurrency']}")
  assert s['ratio_of_config_medians']=={'latency_reduction_percent':100*(1-n['latency_ms']/o['latency_ms']),'throughput_gain_percent':100*(n['throughput_tok_s']/o['throughput_tok_s']-1)}
  ps=[p for p in plot['pairs'] if (p['concurrency'],p['comparison'])==(s['concurrency'],s['comparison'])]
  assert s['median_of_signed_pair_percentages']=={k:statistics.median(p[k] for p in ps) for k in ('latency_reduction_percent','throughput_gain_percent')}
  assert s['paired_both_axis_wins']==3
 resources=json.loads((D/'resources-whitelisted.json').read_text())['samples'];assert len(resources)==359
 for r in resources:
  assert r['power']['source']=='AC Power' and r['thermal']['code']==0 and r['pressure_level_value']==1
  for g in groups:
   if g['arm']==r['arm']:assert not (g['earliest_POST_perf_counter_s']<=r['monotonic_s']<=g['last_DONE_perf_counter_s'])
 byarm={x['arm'] for x in resources}
 for arm in byarm:
  rr=[x for x in resources if x['arm']==arm];assert len({r['Swapouts'] for r in rr})==1 and all(r['swap_used_mb']<=rr[0]['swap_used_mb'] for r in rr)
 cooldowns=[json.loads(p.read_text()) for p in sorted((D/'cooling').glob('*.json'))];assert len(cooldowns)==20
 assert all(c['requested_seconds']==60 and c['end_monotonic_s']-c['start_monotonic_s']>=60 and c['outside_HTTP_clocks'] and c['prior_owned_cleanup_passed'] and not c['adaptive'] for c in cooldowns)
 arms=json.loads((D/'arms-selected.json').read_text());assert len(arms)==21
 for a in arms:
  z=a['cleanup'];assert z['error'] is None and not z['cleanup_errors'] and z['owned_exit']['wait_completed'] and z['owned_exit']['exact_pid_absence']['absent'] and a['port_absence']['absent']
  if a['engine']=='native':
   if a['concurrency']==1:
    n=a['native_final'];assert n['native_steps']==186 and n['request_count']==6 and n['active'] is None
    assert n['feedback']['resets']==6 and n['icb']['bank_executions']==[96,90]
    assert n['pipeline']['total_submitted']==n['pipeline']['total_completed']==n['pipeline']['total_delivered']==186
    assert n['pipeline']['total_discarded']==n['pipeline']['total_failed']==0
   else:
    n=a['native_final'];assert n['actual_formal_max_batch']==a['concurrency'] and n['raw_same_C_assertions_passed'] and n['warm_hooks_removed_before_measure'] and not n['native_adapter_installed']
    assert a['warm']['weighted_GDN_rows']==48*31*3*a['concurrency']
    assert a['warm']['routes']['hooks_restored_before_measure']
 release=json.loads((D/'external-release-selected.json').read_text());assert len(release['pid_checks'])==23 and len(release['port_checks'])==21
 assert all(p['absent'] for p in release['pid_checks']) and all(p['absent'] for p in release['port_checks'])
 owner=json.loads((D/'owner-selected.json').read_text());assert owner['returncode']==0 and owner['waited_reaped'] and owner['absence']['absent']
 # Manifest is written only after independent audit and visual QA; replay also works during assembly.
 checked=0
 if (D/'manifest.json').exists():
  manifest=json.loads((D/'manifest.json').read_text())
  for name,v in manifest['files'].items():
   p=D/name;assert sha(p)==v['sha256'] and p.stat().st_size==v['bytes'],name;checked+=1
 return {'status':'PASS_OFFLINE_PUBLICATION_REPLAY','scope':'Author replay of safe saved rows, not independent GPU recomputation','groups':126,'HTTP_requests':270,'formal_requests':135,'arms':21,'configurations':7,'signed_pairs':12,'joint_frontier':frontier,'resource_samples':359,'cooling_gaps':20,'PID_absences':23,'port_absences':21,'cached_tokens_usage_reported':sum(q['usage']['cached_tokens_observed'] is not None for q in rows),'cached_tokens_usage_not_exposed':sum(q['usage']['cached_tokens_observed'] is None for q in rows),'aux_interactivity_null_requests':sum(q['x_client_estimated_interactivity_tok_s_user'] is None for q in rows),'manifest_payloads_checked':checked,'GPU_used':False}
if __name__=='__main__':print(json.dumps(main(),indent=2))
