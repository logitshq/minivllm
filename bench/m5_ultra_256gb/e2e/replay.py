"""CPU-only raw request replay. Client delivery proxies; all signs/groups/VM remain visible."""
import argparse,hashlib,json,statistics
from pathlib import Path
D=Path(__file__).resolve().parent
CONFIGS=[('pair1-A-off',False),('pair1-B-on',True),('pair2-B-on',True),('pair2-A-off',False),('pair3-A-off',False),('pair3-B-on',True)]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def verify(output):
 output=Path(output);suite=json.loads((output/'suite-complete.json').read_text());all_ids=None;rows=[];all_groups=[]
 assert [(r['config'],r['enabled']) for r in suite['configs']]==CONFIGS,'fixed six-fresh order differs'
 env=None;power=None
 for entry in suite['configs']:
  path=Path(entry['report']);assert path.parent==output/entry['config'] and sha(path)==entry['report_sha256']
  r=json.loads(path.read_text());assert r['concurrency']==1 and r['warmup_groups']==r['repeat']==3 and r['gates_compile_enabled'] is True and r['norm_reuse_enabled']==entry['enabled']
  assert r['runtime_source_identity']['actual_setter_calls']=={'gates':48,'norm_reuse':48}
  if env is None:env=r['environment'];power=r['expected_power']
  assert r['environment']==env and r['expected_power']==power,'power/environment pooling forbidden'
  process=[]
  for phase,key in [('warm','warmups'),('measure','runs')]:
   assert len(r[key])==3
   for i,g in enumerate(r[key]):
    raw=json.loads((path.parent/(g['group_id']+'.json')).read_text());assert raw==g and g['group_id']==f'C1-{phase}-{i}'
    assert len(g['requests'])==g['actual_request_count']==1
    q=g['requests'][0];assert q['usage']['completion_tokens']==32 and q['usage']['prompt_tokens']==19 and q['finish_reason']=='length' and q['usage'].get('prompt_tokens_details',{}).get('cached_tokens',0)==0
    ids=q['output_ids_when_available'];assert ids==q['visible_roundtrip_ids'] and len(ids)==32
    if all_ids is None:all_ids=ids
    assert ids==all_ids
    x=(32-q['first_chunk_reconstructed_tokens'])/(q['done_perf_counter_s']-q['first_text_perf_counter_s']);y=32/(q['done_perf_counter_s']-q['start_perf_counter_s'])
    ttft=q['first_text_perf_counter_s']-q['start_perf_counter_s'];e2e=q['done_perf_counter_s']-q['start_perf_counter_s']
    assert x==g['x_client_estimated_interactivity_tok_s_user'] and y==g['y_total_output_throughput_tok_s_mac'] and ttft==q['client_ttft_text_s'] and e2e==q['client_total_s']
    row={'config':entry['config'],'phase':phase,'group_id':g['group_id'],'x':x,'y':y,'TTFT_s':ttft,'E2E_s':e2e};all_groups.append(row)
    if phase=='measure':process.append(row)
  result={'config':entry['config'],'enabled':entry['enabled'],**{k:statistics.median(g[k] for g in process) for k in ('x','y','TTFT_s','E2E_s')}}
  assert result['x']==entry['x']==r['median_x_client_estimated_interactivity_tok_s_user'] and result['y']==entry['y']==r['median_y_total_output_throughput_tok_s_mac']
  warm=r['post_warm_metadata'];assert warm['gates']['compiled_callable_count']==48 and warm['norm_reuse']['enabled']==entry['enabled'] and warm['routes']['hooks_restored_before_measure'] is True
  from warm_routes import validate_metadata
  validate_metadata(warm['routes'],entry['enabled'])
  exit=r['owned_process_exit'];assert exit['wait_completed'] is True and exit['exact_pid_absence']['absent'] is True
  assert json.loads((path.parent/'port-absence.json').read_text())['absent'] is True
  # Authenticate each original raw resource before any derived inventory/delta view.
  resources=[json.loads(p.read_text()) for p in sorted(path.parent.glob('resource-snapshots-*-raw.json'))]
  assert len(resources)==len(r['resource_snapshots'])==17
  for raw,derived in zip(resources,r['resource_snapshots']):
   for key,value in raw.items():
    if key not in ('known_inference_processes','known_inference_process_count','only_owned_known_inference_processes'):assert derived[key]==value
   assert raw['power_metadata']['source']=={'AC':'AC Power','Battery':'Battery Power'}[power] and raw['pressure_level_value']==1 and raw['processinfo_thermal_metadata']['code'] in (0,1)
  result['all_raw_VM_and_derived_phase_deltas']=r['resource_snapshots'];rows.append(result)
 pairs=[]
 for i in range(3):
  pair=rows[2*i:2*i+2];a=next(r for r in pair if not r['enabled']);b=next(r for r in pair if r['enabled'])
  pairs.append({'pair':i+1,'order':'AB' if not pair[0]['enabled'] else 'BA','on_vs_off_percent':{k:(b[k]/a[k]-1)*100 for k in ('x','y','TTFT_s','E2E_s')}})
 aggregate={str(enabled):{k:statistics.median(r[k] for r in rows if r['enabled']==enabled) for k in ('x','y','TTFT_s','E2E_s')} for enabled in (False,True)}
 return {'status':'PASS_CPU_RAW_REPLAY','power':power,'all_36_groups':all_groups,'processes':rows,'pairs_all_signs_retained':pairs,'aggregate_median3_process_medians':aggregate,'aggregate_on_vs_off_percent':{k:(aggregate['True'][k]/aggregate['False'][k]-1)*100 for k in ('x','y','TTFT_s','E2E_s')},'scope':'native internal norm_reuse effect only; x/y client text delivery proxies, no GPU tps or official win; known matcher cannot exclude GUI/unrecognized work'}
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('output',type=Path);a=ap.parse_args();print(json.dumps(verify(a.output),indent=2))
