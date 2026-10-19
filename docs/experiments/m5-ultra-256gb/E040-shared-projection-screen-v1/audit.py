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
 P=E/'probe';X=E/'execution';L=E/'lease';C=E/'cpu';plan=read(X/'plan.json');inv=read(P/'invocation.json');producer=read(P/'result.json');final=read(X/'final.json');lease=read(L/'lease.json');resume=read(L/'resume.json');job=read(L/'job-0.json')
 need(final['status']=='PASS_LOCAL_PROJECTION_ONLY' and final['errors']==[] and final['lease_returncode']==0 and final['lease_closed'] is True,'execution closure')
 need(lease['closed'] is True and lease['service_pid']==plan['service_pid']==52818 and ' '.join(lease['service_identity'].split())==final['service_identity_after']==plan['service_identity'],'original service identity')
 need(job['returncode']==0 and job['waited'] is True and job['pid']==lease['child_pid'] and job['command']==plan['child']==lease['jobs'][0],'direct child exit/command')
 need(inv['argv']==plan['child'][3:] and inv['deadline_seconds']==120 and inv['limit_bytes']==4<<30 and inv['cache_limit_bytes']==256<<20,'child invocation contract')
 need(resume['identity_match'] is True and resume['state_before']=='T' and resume['state_after']=='R','pause/resume receipt')
 need(producer['status']=='PASS_LOCAL_PROJECTION_ONLY' and len(producer['cases'])==6 and producer['full_model_qualified'] is False and producer['engine_performance_qualified'] is False,'producer scope')
 need(read(P/'runtime.json')==dict(version='0.32.3',name='Apple M5 Ultra',architecture='applegpu_g17d'),'runtime receipt')
 pins={**plan['pins'],**inv['pins']};need(all(plan['pins'].get(k,h)==h for k,h in inv['pins'].items()),'pin conflict')
 root=Path('/Users/ryanli/code/minivllm')
 for original,h in pins.items():need(sha(E/'sources'/Path(original).relative_to(root))==h,'source pin '+original)
 need(inv['source_sha256']==pins[plan['child'][3]],'entry pin')
 cpu=read(C/'execution.json');mechanism=read(E/'mechanism-cpu/result.json')
 need(cpu['before']==cpu['after'] and cpu['sources_unchanged'] is True and cpu['returncode']==0 and 'Ran 1 test' in (C/'stderr.txt').read_text() and (C/'stderr.txt').read_text().endswith('\nOK\n'),'entry CPU closure')
 need(read(E/'mechanism-cpu/source-before.json')==read(E/'mechanism-cpu/source-after.json') and mechanism['status']=='PASS' and mechanism['tests']==7 and mechanism['failures']==mechanism['errors']==mechanism['skips']==0 and mechanism['sources_unchanged'] is True and mechanism['forbidden_imports_present']==[],'mechanism CPU closure')
 for original,h in {**cpu['before'],**read(E/'mechanism-cpu/source-before.json')}.items():need(sha(E/'sources'/Path(original).relative_to(root))==h,'CPU source pin')
 need(read(X/'invocation.json')['plan_sha256']==sha(X/'plan.json')=='a2aa9dfc57fe9b235e7345af3a7b2c150e4054805b29b9d711f74682c60028e7','prepared/executed plan SHA')
 need((X/'stderr.txt').read_text()=='mx.metal.device_info is deprecated and will be removed in a future version. Use mx.device_info instead.\n','only observed exact deprecation warning')
 manifest=read(E/'sources/bench/m5_ultra_256gb/omlx_d512_formal/prereg-formal-v1/inputs/model-manifest.json')
 rows=[];cases=[];word_count=0
 for label,N,base in (('head',248320,'language_model.lm_head'),('up',17408,'language_model.model.layers.0.mlp.up_proj')):
  folder=P/label;weights=read(folder/'weights.json');header_path=folder/'original-header.json';header=read(header_path)
  need(sha(header_path)==weights['header_sha256'] and weights['source_full_sha256']==manifest['files'][Path(weights['source']).name] and weights['full_sha_checked'] is True,'weight source receipt')
  for part,suffix,dtype,width,size in zip(weights['slices'],('weight','scales'),('U32','U8'),(640,320),(4,1)):
   key=base+'.'+suffix;m=header[key];lo,hi=m['data_offsets']
   need(part['name']==key and part['dtype']==dtype and part['shape']==[N,width] and m['dtype']==dtype and m['shape']==[N,width] and part['start']==8+header_path.stat().st_size+lo and part['bytes']==hi-lo==N*width*size and re.fullmatch('[0-9a-f]{64}',part['sha256']),'retained weight range identity')
  for T in (2,4,8):
   case=folder/f'T{T}';num=read(case/'numerical.json');summary=read(case/'timing-summary.json');x=tensors(case/'inputs.safetensors',{'x':[T,5120]})['x'];out=tensors(case/'outputs.safetensors',{k:[T,N] for k in ('native_b1','custom_b1','candidate')})
   need(finite(x) and all(finite(b) for b in out.values()) and out['native_b1']==out['custom_b1']==out['candidate'],'raw full BF16 equality/finite')
   need(num['status']=='PASS' and num['finite'] is True and all(c==dict(equal=True,first_mismatch=None) for c in num['comparisons'].values()) and len(num['comparisons'])==3,'producer numerical receipt')
   need(num['original_dtype_sha256']=={k:hashlib.sha256(b).hexdigest() for k,b in out.items()} and num['inputs_sha256']==sha(case/'inputs.safetensors') and num['outputs_sha256']==sha(case/'outputs.safetensors'),'raw input/output hashes')
   data=(case/'timings.jsonl').read_bytes();need(data.endswith(b'\n'),'complete JSONL');samples=[json.loads(line) for line in data.splitlines()];need(len(samples)==400,'400 samples per case');cursor=0
   for baseline in ('native_b1','custom_b1'):
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
   need(cursor==400 and next(c for c in producer['cases'] if c['projection']==label and c['T']==T)==dict(projection=label,T=T,K=5120,N=N,timing=summary),'producer aggregate case')
   cases.append(dict(projection=label,T=T,elements_per_route=T*N,output_matrices=3,status='PASS'));word_count+=T*N
 observations=[]
 for p in [*sorted(X.glob('resource-*.json')),X/'after-resource.json']:
  r=read(p);pressure=int(r['pressure_level']['text']);swap=float(re.search(r'used = ([0-9.]+)M',r['swap']['text'])[1]);swapouts=int(re.search(r'^Swapouts:\s*(\d+)\.',r['vm_stat']['text'],re.M)[1])
  need(pressure==r['pressure_level_value']==1 and swap==r['swap_used_mb'] and swapouts==r['vm_counters']['Swapouts'],'raw resource parse')
  if r['label']=='projection-running':need(any(a['pid']==52818 and a['state']=='T' for a in r['native_inference_processes']) and r['owned_model_pid']==lease['child_pid'],'observed pause/child')
  observations.append(dict(file=p.name,pressure=pressure,swap_used_mib=swap,Swapouts=swapouts))
 need(len(observations)==5 and final['samples']==4 and len({o['swap_used_mib'] for o in observations})==len({o['Swapouts'] for o in observations})==1,'resource sample consistency')
 for p in (X/'before-health.json',X/'after-health.json'):
  r=read(p);s=r['/api/status'];target='Qwen3.8-Flash-Next-oQ6e-mtp';models=r['/v1/models/status']['models'];loaded=[m for m in models if m['loaded']]
  need(r['/health']['status']=='healthy' and s['version']=='0.7.0' and s['default_model']==target and s['loaded_models']==[target] and s['active_requests']==s['waiting_requests']==s['models_loading']==0 and len(loaded)==1 and loaded[0]['id']==target and not any(m['is_loading'] for m in models),'before/after saved health')
 return dict(status='PASS_OFFLINE_AUDIT',candidate_decision='HEAD_T4_T8_PROMISING_LOCAL_ONLY' if all(r['median_group_ratio']>1 for r in rows if r['projection']=='head' and r['T'] in (4,8)) else 'MIXED_LOCAL_SCREEN',cases=cases,case_count=6,output_matrices=18,elements_per_route=word_count,pairwise_BF16_word_comparisons=word_count*3,raw_timing_samples=2400,rows=rows,resource_observations=observations,
  execution_complete=True,saved_pause_resume_health_consistent=True,median_ratios_above_one=sum(r['median_group_ratio']>1 for r in rows),median_ratios_below_one=sum(r['median_group_ratio']<1 for r in rows),full_model_qualified=False,engine_pareto_point=False,
  limits=['Original BF16 saved outputs independently checked; not a new GPU computation or per-timed-sample output check.','Timing order and medians recomputed from raw JSONL; clocks/5 warmups/fresh dispatch/sync follow frozen source and execution records.','Real weight payloads excluded; original source/slice hashes are producer evidence, not rehashed by this audit.','Only synthetic BF16 activations with real projection weights; no full hidden-state/model qualification.','Five sequential within-process groups show observed variability, not independent-run confidence intervals.','Resource and service checks reparse saved observations; no continuous isolation or independent present-time service probe.'])
def main():
 p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 try:r=run(a.evidence)
 except BaseException as e:r=dict(status='FAIL',error=repr(e),traceback=traceback.format_exc())
 with a.out.open('x') as f:json.dump(r,f,indent=2);f.write('\n')
 print(json.dumps({k:v for k,v in r.items() if k not in ('rows','cases','limits')},sort_keys=True));return 0 if r['status']=='PASS_OFFLINE_AUDIT' else 1
if __name__=='__main__':raise SystemExit(main())
