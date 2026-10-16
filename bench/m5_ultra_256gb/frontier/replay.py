"""Recompute all frontend timings from immutable request records, retaining every pair."""
import argparse,hashlib,json,statistics
from pathlib import Path

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def verify(root):
 root=Path(root).resolve();complete=json.loads((root/'complete.json').read_text());processes=[];request_count=0;resource_count=0
 expected=[(c,pair,e) for c in [1,2,4] for pair in [1,2,3] for e in (['tutorial','omlx'] if pair!=2 else ['omlx','tutorial'])]
 assert [(r['C'],r['pair'],r['engine']) for r in complete['rows']]==expected
 for entry in complete['rows']:
  p=Path(entry['path']);assert p.parent.parent==root and digest(p)==entry['sha256'];r=json.loads(p.read_text());c=r['concurrency'];measured=[]
  for phase,groups in r['groups'].items():
   assert phase in ('warm','measure') and len(groups)==3
   for i,g in enumerate(groups):
    assert g['group_id']==f'C{c}-{phase}-{i}' and json.loads((p.parent/(g['group_id']+'.json')).read_text())==g and len(g['requests'])==c
    xs=[]
    for q in g['requests']:
     assert q['usage']['completion_tokens']==32 and q['usage']['prompt_tokens']==19 and q['finish_reason']=='length' and q['usage'].get('prompt_tokens_details',{}).get('cached_tokens',0)==0
     assert len(q['visible_roundtrip_ids'])==32 and q['first_chunk_prefix_verified'] and q['full_visible_roundtrip_count_matches_usage']
     if r['engine']=='tutorial':assert q['output_ids_when_available']==q['visible_roundtrip_ids']
     assert q['start_perf_counter_s']<q['first_text_perf_counter_s']<q['done_perf_counter_s']
     x=(32-q['first_chunk_reconstructed_tokens'])/(q['done_perf_counter_s']-q['first_text_perf_counter_s']);assert x==q['x_client_estimated_interactivity_tok_s_user'];xs.append(x);request_count+=1
    x=statistics.median(xs);y=32*c/(max(q['done_perf_counter_s'] for q in g['requests'])-min(q['start_perf_counter_s'] for q in g['requests']));assert x==g['x_client_estimated_interactivity_tok_s_user'] and y==g['y_total_output_throughput_tok_s_mac']
    if phase=='measure':measured.append({'x':x,'y':y,'TTFT_s':statistics.median(q['first_text_perf_counter_s']-q['start_perf_counter_s'] for q in g['requests'])})
  row={'engine':r['engine'],'C':c,'pair':r['pair'],**{k:statistics.median(q[k] for q in measured) for k in ['x','y','TTFT_s']}}
  for k in ('x','y','TTFT_s'):assert row[k]==r[k]==entry[k]
  for rawfile in sorted(p.parent.glob('resource-*-raw.json')):
   s=json.loads(rawfile.read_text());assert s['pressure_level_value']==1 and s['processinfo_thermal_metadata']['code'] in (0,1) and s['power_metadata']['source']=='AC Power';resource_count+=1
  ex=json.loads((p.parent/'process-exit.json').read_text());assert ex['waited_reaped'] and ex['absence']['returncode']==1 and ex['absence']['stdout']==ex['absence']['stderr']==''
  cleanup=json.loads((p.parent/'cleanup.json').read_text());assert cleanup['first_failure'] is None and not cleanup['cleanup_errors']
  processes.append(row)
 points=[];pairs=[]
 for c in (1,2,4):
  for engine in ('tutorial','omlx'):
   points.append({'engine':engine,'C':c,**{k:statistics.median(q[k] for q in processes if q['C']==c and q['engine']==engine) for k in ['x','y','TTFT_s']}})
  for pair in (1,2,3):
   a=next(q for q in processes if q['C']==c and q['pair']==pair and q['engine']=='tutorial');b=next(q for q in processes if q['C']==c and q['pair']==pair and q['engine']=='omlx')
   pairs.append({'C':c,'pair':pair,'native_vs_omlx_percent':{k:(a[k]/b[k]-1)*100 for k in ['x','y','TTFT_s']}})
 return {'status':'PASS_CPU_RAW_REPLAY','requests':request_count,'tokens':request_count*32,'resources':resource_count,'processes':processes,'points_median3_process_medians':points,'pairs_all_signs':pairs,'scope':'M5 Ultra 256 GiB AC; x is client delivery proxy, y includes queue/prefill/HTTP; official raw IDs and GPU batching are not asserted'}
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('root');a=ap.parse_args();print(json.dumps(verify(a.root),indent=2))
