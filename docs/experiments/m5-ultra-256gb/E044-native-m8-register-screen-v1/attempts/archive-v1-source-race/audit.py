"""Offline raw-byte and timing audit. Standard library only; never loads a backend."""
import argparse,hashlib,json,math,re,statistics,struct,traceback
from pathlib import Path

def need(x,m):
 if not x:raise ValueError(m)
def read(p):return json.loads(p.read_bytes())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def tensors(p,expected):
 raw=p.read_bytes();n=struct.unpack('<Q',raw[:8])[0];h=json.loads(raw[8:8+n]);need(h.pop('__metadata__',None) is None,'only observed null metadata');need(set(h)==set(expected),'tensor keys');result={};spans=[]
 for key,shape in expected.items():
  m=h[key];lo,hi=m['data_offsets'];need(m['dtype']=='BF16' and m['shape']==shape and 0<=lo<=hi and hi-lo==math.prod(shape)*2,'dtype/shape/span')
  result[key]=raw[8+n+lo:8+n+hi];need(len(result[key])==hi-lo,'complete tensor');spans.append((lo,hi))
 pos=0
 for lo,hi in sorted(spans):need(lo==pos,'gap/overlap');pos=hi
 need(len(raw)==8+n+pos,'complete payload');return result
def finite(b):return all((v[0]&0x7f80)!=0x7f80 for v in struct.iter_unpack('<H',b))
def run(E):
 P=E/'probe';X=E/'execution';L=E/'lease';C=E/'cpu';plan=read(X/'plan.json');inv=read(P/'invocation.json');producer=read(P/'result.json');final=read(X/'final.json');lease=read(L/'lease.json');resume=read(L/'watchdog-resume.json');job=read(L/'job-0.json')
 need(final['status']=='FAIL' and final['lease_returncode']==-2 and final['lease_closed'] is False and final['errors']==[dict(type='AssertionError',message='owned numerical child identity'),dict(type='AssertionError',message='lease did not close')],'original failure preserved')
 need(lease['closed'] is False and lease['service_pid']==plan['service_pid']==52818 and ' '.join(lease['service_identity'].split())==final['service_identity_after']==plan['service_identity'],'original service identity')
 need(job['returncode']==0 and job['waited'] is True and job['pid']==lease['child_pid'] and job['command']==plan['child']==lease['jobs'][0],'direct child exit/command')
 need(inv['argv']==plan['child'][3:] and inv['deadline_seconds']==120 and inv['limit_bytes']==4<<30 and inv['cache_limit_bytes']==256<<20,'child invocation contract')
 need(resume['identity_match'] is True and resume['state_before']=='T' and resume['state_after']=='S','pause/resume receipt')
 need(producer['status']=='PASS_LOCAL_PROJECTION_ONLY' and len(producer['cases'])==2 and producer['full_model_qualified'] is False and producer['engine_performance_qualified'] is False,'producer scope')
 need(read(P/'runtime.json')==dict(version='0.32.3',name='Apple M5 Ultra',architecture='applegpu_g17d'),'runtime receipt')
 pins={**plan['pins'],**inv['pins']};need(all(plan['pins'].get(k,h)==h for k,h in inv['pins'].items()),'pin conflict')
 root=Path('/Users/ryanli/code/minivllm')
 for original,h in pins.items():need(sha(E/'sources'/Path(original).relative_to(root))==h,'source pin '+original)
 need(inv['source_sha256']==pins[plan['child'][3]],'entry pin')
 for label,count in (('cpu-v1',11),('future-cleanup-v2/cpu-v1',5)):
  folder=E/label;cpu=read(folder/'result.json');before=read(folder/'source-before.json')
  need(before==read(folder/'source-after.json') and cpu['status']=='PASS' and cpu['tests']==count and cpu['sources_unchanged'] is True and cpu['failures']==cpu['errors']==cpu['skips']==0 and cpu['forbidden_imports_present']==[],'CPU closure')
  for original,h in before.items():need(sha(E/'sources'/Path(original).relative_to(root))==h,'CPU source pin')
 need(read(X/'invocation.json')['plan_sha256']==sha(X/'plan.json')=='2c6518d2de4392c91cd4481011d7bfd999888739ced5c3e005b29b980382ac9f','prepared/executed plan SHA')
 reconciliation=read(X/'post-failure-reconciliation.json')
 need(sha(X/'post-failure-reconciliation.json')=='bee0a1ca402151f9f30dd975e8fe194efe64aae4715c71655f0cc90c80607d0a','sealed later reconciliation')
 need(reconciliation['status']=='PASS_RESTORATION_ONLY' and reconciliation['original_parent_status']=='FAIL' and reconciliation['original_lease_closed'] is False and reconciliation['original_lease_returncode']==-2 and reconciliation['performance_qualified'] is False,'restoration scope only')
 need(reconciliation['service_pid']==52818 and reconciliation['service_identity']==plan['service_identity'] and reconciliation['service_state']=='S' and reconciliation['process_records']=={'57173':None,'57165':None},'later identity/absence receipt')
 for original,h in reconciliation['original_evidence_sha256'].items():
  p=Path(original);folder=X if p.parent.name=='m8-register-projection-execution-v1' else L
  need(sha(folder/p.name)==h,'original failed evidence retained')
 need(not (L/'resume.json').exists(),'no invented normal resume receipt')
 need((X/'stderr.txt').read_text().endswith('KeyboardInterrupt: 15\n'),'retained lease interruption trace')
 manifest=read(E/'sources/bench/m5_ultra_256gb/omlx_d512_formal/prereg-formal-v1/inputs/model-manifest.json')
 rows=[];cases=[];word_count=0
 for label,N,base in (('head',248320,'language_model.lm_head'),('up',17408,'language_model.model.layers.0.mlp.up_proj')):
  folder=P/label;weights=read(folder/'weights.json');header_path=folder/'original-header.json';header=read(header_path)
  need(sha(header_path)==weights['header_sha256'] and weights['source_full_sha256']==manifest['files'][Path(weights['source']).name] and weights['full_sha_checked'] is True,'weight source receipt')
  for part,suffix,dtype,width,size in zip(weights['slices'],('weight','scales'),('U32','U8'),(640,320),(4,1)):
   key=base+'.'+suffix;m=header[key];lo,hi=m['data_offsets']
   need(part['name']==key and part['dtype']==dtype and part['shape']==[N,width] and m['dtype']==dtype and m['shape']==[N,width] and part['start']==8+header_path.stat().st_size+lo and part['bytes']==hi-lo==N*width*size and re.fullmatch('[0-9a-f]{64}',part['sha256']),'retained weight range identity')
  for T in (8,):
   case=folder/f'T{T}';num=read(case/'numerical.json');summary=read(case/'timing-summary.json');x=tensors(case/'inputs.safetensors',{'x':[T,5120]})['x'];out=tensors(case/'outputs.safetensors',{k:[T,N] for k in ('native_m8','candidate')})
   need(finite(x) and all(finite(b) for b in out.values()) and out['native_m8']==out['candidate'],'raw full BF16 equality/finite')
   need(num['status']=='PASS' and num['finite'] is True and all(c==dict(equal=True,first_mismatch=None) for c in num['comparisons'].values()) and num['comparisons']=={'candidate__native_m8':dict(equal=True,first_mismatch=None)},'producer numerical receipt')
   need(num['original_dtype_sha256']=={k:hashlib.sha256(b).hexdigest() for k,b in out.items()} and num['inputs_sha256']==sha(case/'inputs.safetensors') and num['outputs_sha256']==sha(case/'outputs.safetensors'),'raw input/output hashes')
   data=(case/'timings.jsonl').read_bytes();need(data.endswith(b'\n'),'complete JSONL');samples=[json.loads(line) for line in data.splitlines()];need(len(samples)==200,'200 samples per case');cursor=0
   for baseline in ('native_m8',):
    groups=[]
    for group in range(5):
     values={baseline:[],'candidate':[]}
     for pair in range(20):
      order=(baseline,'candidate') if pair%2==0 else ('candidate',baseline)
      for position,route in enumerate(order):
       sample=samples[cursor];cursor+=1
       need(set(sample)=={'baseline','group','pair','position','route','elapsed_ns'} and {k:sample[k] for k in ('baseline','group','pair','position','route')}==dict(baseline=baseline,group=group,pair=pair,position=position,route=route) and type(sample['elapsed_ns']) is int and sample['elapsed_ns']>0,'exact order/coverage/positive timing')
       values[route].append(sample['elapsed_ns'])
     med={k:statistics.median(v) for k,v in values.items()};ratio=med[baseline]/med['candidate'];record=summary[baseline]['groups'][group]
     need(record['median_ns']==med and record['ratio_baseline_over_candidate']==ratio,'producer group recomputation')
     groups.append(dict(group=group,median_ns=med,ratio_baseline_over_candidate=ratio))
    median=statistics.median(g['ratio_baseline_over_candidate'] for g in groups)
    need(summary[baseline]['median_group_ratio']==median,'producer median group ratio')
    rows.append(dict(projection=label,T=T,K=5120,N=N,baseline=baseline,groups=groups,median_group_ratio=median,candidate_latency_increase_percent=(1/median-1)*100))
   need(cursor==200 and next(c for c in producer['cases'] if c['projection']==label and c['T']==T)==dict(projection=label,T=T,K=5120,N=N,timing=summary),'producer aggregate case')
   cases.append(dict(projection=label,T=T,elements_per_route=T*N,output_matrices=2,status='PASS'));word_count+=T*N
 observations=[]
 for p in [*sorted(X.glob('resource-*.json')),X/'after-resource.json']:
  r=read(p);pressure=int(r['pressure_level']['text']);swap=float(re.search(r'used = ([0-9.]+)M',r['swap']['text'])[1]);swapouts=int(re.search(r'^Swapouts:\s*(\d+)\.',r['vm_stat']['text'],re.M)[1])
  need(pressure==r['pressure_level_value']==1 and swap==r['swap_used_mb'] and swapouts==r['vm_counters']['Swapouts'],'raw resource parse')
  if r['label']=='prefix-running':need(any(a['pid']==52818 and a['state']=='T' for a in r['native_inference_processes']) and r['owned_model_pid']==lease['child_pid'],'observed pause/child')
  observations.append(dict(file=p.name,pressure=pressure,swap_used_mib=swap,Swapouts=swapouts))
 need(len(observations)==6 and final['samples']==5 and len({o['swap_used_mib'] for o in observations})==len({o['Swapouts'] for o in observations})==1,'resource sample consistency')
 baseline=read(X/'physical-before/sample.json')['parsed'];physical=[]
 for p in sorted(X.glob('physical-[0-9]*/sample.json')):
  r=read(p);v=r['parsed'];need(r['status']=='MEASURED' and v['pid']==lease['child_pid'] and v['pressure']==1,'physical sample identity')
  need(max(v[k] for k in ('top_bytes','aux_current_bytes','aux_peak_bytes'))<=plan['physical']['child_limit_bytes'] and v['system_bytes']<=plan['physical']['system_limit_bytes'],'physical observation ceilings')
  need(v['swap_bytes']<=baseline['swap_bytes'] and v['swapouts']==baseline['swapouts'],'physical no swap growth')
  if physical:need(r['started_monotonic']-physical[-1]['ended_monotonic']<=5,'physical sample gap')
  physical.append(r)
 need(len(physical)==final['physical_samples']==4,'all physical observations')
 for r in (read(X/'before-health.json'),read(X/'after-health.json'),reconciliation['health']):
  s=r['/api/status'];target='Qwen3.8-Flash-Next-oQ6e-mtp';models=r['/v1/models/status']['models'];loaded=[m for m in models if m['loaded']]
  need(r['/health']['status']=='healthy' and s['version']=='0.7.0' and s['default_model']==target and s['loaded_models']==[target] and s['active_requests']==s['waiting_requests']==s['models_loading']==0 and len(loaded)==1 and loaded[0]['id']==target and not any(m['is_loading'] for m in models),'before/after saved health')
 return dict(status='PASS_OFFLINE_AUDIT',candidate_decision='REJECT_NATIVE_M8_REGISTER_FOR_SPEED' if all(r['median_group_ratio']<1 for r in rows) else 'MIXED_LOCAL_SCREEN',cases=cases,case_count=2,output_matrices=4,elements_per_route=word_count,pairwise_BF16_word_comparisons=word_count,raw_timing_samples=400,rows=rows,resource_observations=observations,physical_observations=[dict(file=f"physical-{i:04d}/sample.json",**r["parsed"]) for i,r in enumerate(physical)],
  execution_complete=False,numerical_child_complete=True,original_parent_status='FAIL',original_lease_closed=False,original_lease_returncode=-2,saved_watchdog_restore_and_reconciliation_consistent=True,restoration_status='PASS_RESTORATION_ONLY',median_ratios_above_one=sum(r['median_group_ratio']>1 for r in rows),median_ratios_below_one=sum(r['median_group_ratio']<1 for r in rows),full_model_qualified=False,engine_pareto_point=False,
  limits=['Parent execution remains FAIL and lease remains unclosed; this independent PASS covers saved numerics/timing and restoration receipt consistency only.','Original BF16 saved outputs independently checked; not a new GPU computation or per-timed-sample output check.','Timing order and medians recomputed from raw JSONL; clocks/5 warmups/fresh dispatch/sync follow frozen source and execution records.','Real weight payloads excluded; original source/slice hashes are producer evidence, not rehashed by this audit.','Only synthetic BF16 activations with real projection weights; no full hidden-state/model qualification.','Five sequential within-process groups show observed variability, not independent-run confidence intervals.','Resource and service checks reparse saved observations; no continuous isolation or independent present-time service probe.'])
def main():
 p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 try:r=run(a.evidence)
 except BaseException as e:r=dict(status='FAIL',error=repr(e),traceback=traceback.format_exc())
 with a.out.open('x') as f:json.dump(r,f,indent=2);f.write('\n')
 print(json.dumps({k:v for k,v in r.items() if k not in ('rows','cases','limits')},sort_keys=True));return 0 if r['status']=='PASS_OFFLINE_AUDIT' else 1
if __name__=='__main__':raise SystemExit(main())
