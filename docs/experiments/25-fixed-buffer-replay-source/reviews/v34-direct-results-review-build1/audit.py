"""CPU replay of saved direct-fixedcap numerical evidence, without MLX imports."""
from pathlib import Path
import hashlib,json,math,re,sys
import numpy as np
P=Path('/private/tmp/minivllm-engine-optimization-20261014');O=P/'v23-root-control-build1/fullmodel-v34-attempt1';D=O/'child-output';OUT=Path(__file__).parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
r=read(D/'receipt.json')
assert sha(D/'receipt.json')=='37d5a808dc4fd979f612d776d42f3e2f42913c66c6f4ae2662e8b675c7d3c5db'
assert r['status']=='PASS_FIXEDCAP256_32_GREEDY_STRICT_NUMERICAL_ONLY' and r['stage']=='complete'
assert r['no_performance_claim'] and not r['engine_integrated'] and r['model_loads']==r['model_load_attempts']==1
assert r['gates_callable_count']==48 and r['modes']['gates_ON']==48 and r['modes']['conv_OFF'] and r['modes']['whole_functional_OFF']
assert r['cache_limit_bytes']==2<<30 and r['runtime']['versions']=={'mlx':'0.32.3','mlx-metal':'0.32.3'}
assert r['runtime']['device_info']['device_name']=='Apple M4 Max'
assert len(r['source_pins'])==37
for p,s in r['source_pins'].items():assert sha(p)==s,p
for k in ('imported_sources_before_load','imported_sources_final'):
 for x in r[k].values():assert x['path'] in r['source_pins'] and r['source_pins'][x['path']]==x['sha256']
ids=read(P/'workload.json')['continuation_ids'];assert r['A_ids']==r['B_ids']==ids and len(ids)==32 and not set(ids)&set(r['eos_ids'])
assert len(r['steps'])==r['expected_decode_calls']==31 and r['expected_outputs']==32
raw=[];strict=0;readonly=0

def sig(x):
 assert x['finite'] and len(x['sha256'])==64
 assert x['bytes']==math.prod(x['shape'])*{'mlx.core.bfloat16':2,'mlx.core.float32':4,'mlx.core.int32':4}[x['dtype']]
def rawcheck(path,x):
 p=(D/path).resolve();assert p.is_relative_to(D.resolve())
 b=p.read_bytes();assert len(b)==x['bytes'] and hashlib.sha256(b).hexdigest()==x['sha256']
 a=np.frombuffer(b,dtype='<u2');assert x['dtype']=='mlx.core.bfloat16' and np.all((a&0x7f80)!=0x7f80)
 raw.append({'file':path,'bytes':len(b),'sha256':x['sha256']})
sig(r['prefill']['logits']);rawcheck(r['prefill']['raw_logits'],r['prefill']['logits'])
prevA=prevB=r['prefill']['states'];assert len(prevA)==r['prefill']['array_count']==128
for i,s in enumerate(r['steps'],1):
 assert s['step']==i and s['position']==18+i and s['exact_L']==19+i and s['B_max_len']==256
 assert s['A_token']==s['B_token']==ids[i]
 assert len(s['states'])==128
 for pair in [s['logits']]+s['states']:
  assert pair['strict_equal'] and pair['A']==pair['B'];sig(pair['A']);sig(pair['B']);strict+=1
 rawcheck(s['logits']['raw_A'],s['logits']['A']);rawcheck(s['logits']['raw_B'],s['logits']['B'])
 for side,prev in (('A',prevA),('B',prevB)):
  ro=s['retained_input_readonly'][side]
  assert ro['equal'] and ro['strict_bytes_equal'] and ro['count']==128 and ro['before']==ro['after']==prev
  readonly+=1
 ro=s['scalar_inputs_readonly'];assert ro['equal'] and ro['strict_bytes_equal'] and ro['count']==4 and ro['before']==ro['after']
 for x in ro['before']:sig(x);assert x['dtype']=='mlx.core.int32' and x['shape']==[1]
 readonly+=1
 prevA=[x['A'] for x in s['states']];prevB=[x['B'] for x in s['states']]
assert r['final_cache']['A']==prevA and r['final_cache']['B']==prevB and strict==3999 and readonly==93
assert len(raw)==len(list(D.glob('*.bin')))==63
for x in r['raw_resources']:assert sha(D/x['file'])==x['sha256']
allresources=[];resources={}
for base,pattern,derived in ((O,'resource-???-raw.json','resource-derived.json'),(D,'resource-??-raw.json','resources-derived.json'),(O,'external-resource-??-raw.json','external-resource.json')):
 ps=sorted(base.glob(pattern));ds=read(base/derived);assert len(ps)==len(ds)
 stats=[]
 for j,(p,d) in enumerate(zip(ps,ds)):
  raw_sample=read(p)
  mutable={'known_inference_processes','known_inference_process_count','only_owned_known_inference_processes','exactly_one_owned_known_model'}
  for k,v in raw_sample.items():
   if k not in mutable:assert d[k]==v,(p,k)
  assert int(raw_sample['pressure_level']['text'])==d['pressure_level_value']==1
  assert int(re.search(r'free percentage: (\d+)%',raw_sample['memory_pressure']['text'])[1])==d['free_percent']
  assert float(re.search(r'used = ([0-9.]+)M',raw_sample['swap']['text'])[1])==d['swap_used_mb']
  for k in ('Swapouts','Swapins','Pageouts','Compressions','Decompressions'):
   assert int(re.search(r'^'+k+r':\s+(\d+)\.',raw_sample['vm_stat']['text'],re.M)[1])==d['vm_counters'][k]
  assert d['process_inventory_ok'] and d['only_owned_known_inference_processes'] and all(x['owned'] and x['pid']==r['pid'] for x in d['known_inference_processes'])
  assert raw_sample['power_metadata']['status']=='known' and raw_sample['power_metadata']['source']=='AC Power' and "'AC Power'" in raw_sample['power_metadata']['raw']['stdout']
  assert raw_sample['processinfo_thermal_metadata']['status']=='known' and int(raw_sample['processinfo_thermal_metadata']['raw']['stdout'])==raw_sample['processinfo_thermal_metadata']['code']==0
  if j:
   prev=ds[j-1]
   assert d['swap_used_mb_delta']==d['swap_used_mb']-prev['swap_used_mb']==0
   assert d['vm_counter_delta']=={k:v-prev['vm_counters'][k] for k,v in d['vm_counters'].items() if k in prev['vm_counters']}
   assert d['vm_counter_delta']['Swapouts']==0
  stats.append({'file':str(p),'sha256':sha(p)});allresources.append(d)
 resources[str(base/derived)]={'samples':len(ps),'raw_manifest_sha256':digest(stats)}
assert sum(x['samples'] for x in resources.values())==25
own=read(O/'owner-receipt.json');rel=read(O/'external-release.json')
assert own['owner_pid']==55951 and own['child_pid']==r['pid']==55960 and own['stage']=='reaped' and own['returncode']==0 and own['waited_reaped']
assert own['absence']=={'returncode':1,'stdout':'','stderr':''}
assert own['script_sha256']==r['source_pins'][own['command'][2]] and not own['performance_qualified']
assert rel['status']=='PASS_EXTERNAL_RELEASE' and rel['tool_exit']==0 and rel['GPU_released']
assert {a['pid'] for a in rel['strict_absence']}=={55951,55960} and all(a['returncode']==1 and a['stdout']==a['stderr']=='' for a in rel['strict_absence'])
assert not any(x=='mlx' or x.startswith('mlx.') for x in sys.modules)
result={'status':'PASS_SAVED_V34_DIRECT_FIXEDCAP_EVIDENCE_CPU_AUDIT','receipt_sha256':sha(D/'receipt.json'),'scope':'Direct Python model A exactL20..50 vs B fixedcap256, independent greedy/state chains; no native replay/engine integration/service/performance qualification. State/readonly records are actual worker evidence, not new GPU raw recapture.','source_pins_verified':37,'steps':31,'strict_logit_state_records':strict,'readonly_records':readonly,'IDs':ids,'raw_logit_files':len(raw),'raw_logit_bytes':sum(x['bytes'] for x in raw),'raw_logit_manifest_sha256':digest(raw),'resources':resources,'resource_summary':{'samples':25,'thermal_codes':sorted({x['processinfo_thermal_metadata']['code'] for x in allresources}),'pressure_levels':sorted({x['pressure_level_value'] for x in allresources}),'minimum_free_percent':min(x['free_percent'] for x in allresources),'swap_MB_growth':0,'Swapouts_growth':0,'Pageouts_growth_owner':read(O/'resource-derived.json')[-1]['vm_counters']['Pageouts']-read(O/'resource-derived.json')[0]['vm_counters']['Pageouts']},'cleanup':{'owner':55951,'child':55960,'waited_reaped_exit0':True,'saved_external_strict_absence':True},'metadata_sha256':{str(p):sha(p) for p in sorted(list(D.glob('*.json'))+list(O.glob('*.json')))}}
(OUT/'receipt.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('metadata_sha256','IDs','resources')},indent=2))
