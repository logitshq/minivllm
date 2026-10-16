"""Saved native B-only performance evidence audit, CPU/stdlib only."""
from pathlib import Path
import ast,hashlib,json,math,re,statistics,sys
P=Path('/private/tmp/minivllm-engine-optimization-20261014');OUT=Path(__file__).parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
digest=lambda x:hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
results={};loopAST=[];sampleAST=[]
for version,owner,child,expectedsha,expectedstatus,pincount in [('v36',60569,60570,'3d932f1118e2d47c17bc08897f54a930958699b3968f89488b86de6b320d1989','PASS_S_PINGPONG_B_ONLY_EXPLORATORY_SCREEN',49),('v37',60339,60340,'ba321f243513d4f5f70ca66d4b75b0682bb6586459e4dda5588b2c4bbdc50600','PASS_B_ONLY_EXPLORATORY_SCREEN',46)]:
 O=P/f'v23-root-control-build1/native-decode-{version}-attempt1';D=O/'child-output';r=read(D/'receipt.json')
 assert sha(D/'receipt.json')==expectedsha and r['status']==expectedstatus and r['stage']=='complete'
 assert r['pid']==child and r['model_loads']==1 and r['fixed_cap']==256 and r['gates_on']==48 and r['conv_off'] and r['whole_functional_off']
 assert r['runtime']['cache_limit_bytes']==2<<30 and r['runtime']['versions']=={'mlx':'0.32.3','mlx-metal':'0.32.3'}
 assert r['runtime']['device']['device_name']=='Apple M4 Max' and r['schedule']==['warm']*3+['measure']*3
 assert len(r['source_pins'])==pincount
 for p,h in r['source_pins'].items():assert sha(p)==h,p
 assert sha(r['qualification_receipt'])==r['qualification_sha256']
 q=read(r['qualification_receipt']);assert q['native_steps']==93 and len(q['requests'])==3 and q['gates_callable_count']==48
 expected=q['requests'][0]['steps'][-1]['outputs'];assert len(expected)==132
 assert q['requests'][0]['steps'][-1]['outputs']==q['requests'][1]['steps'][-1]['outputs']
 for p,h in q['source_pins'].items():assert sha(p)==h,p
 own=read(O/'owner-receipt.json');rel=read(O/'external-release.json');script=Path(own['command'][2]);assert sha(script)==own['driver_sha256']==r['source_pins'][str(script)]
 tree=ast.parse(script.read_text());loops=[n for n in ast.walk(tree) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='_'];assert len(loops)==1
 loop=loops[0];loopAST.append(ast.dump(loop,include_attributes=False));assert isinstance(loop.iter,ast.Call) and loop.iter.args[0].value==31
 called=[ast.unparse(n.func) for n in ast.walk(loop) if isinstance(n,ast.Call)];assert set(called)=={'range','step_fn','copy_fn','RuntimeError','ids.append'}
 sample=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='sample');sampleAST.append(ast.dump(sample,include_attributes=False))
 ids=read(P/'workload.json')['continuation_ids'];rows=r['requests'];assert len(rows)==6
 sourcebylib=q['native_library'];assert r['source_pins'][sourcebylib]==q['source_pins'][sourcebylib]
 measures=[]
 for i,row in enumerate(rows,1):
  assert row['index']==i and row['phase']==r['schedule'][i-1] and row['ids']==ids
  t0,t1,t2=[row[k] for k in ('t0_ns','t1_ns','t2_ns')];assert t0<t1<t2
  assert row['prefill_seed_ready_ns']==t1-t0 and row['decode31_ns']==t2-t1 and row['whole32_ns']==t2-t0
  assert row['decode_ns_per_step']==(t2-t1)/31 and row['decode31_tokens_per_s']==31e9/(t2-t1) and row['whole32_tokens_per_s']==32e9/(t2-t0)
  assert len(row['final132_hash_checks'])==132
  for x,e in zip(row['final132_hash_checks'],expected):assert x=={'label':e['label'],'bytes':e['bytes'],'sha256':e['A_sha256'],'qualified_A_sha256':e['A_sha256']}
  assert row['final_hash_cpu_copy_bytes_outside_timing']==sum(e['bytes'] for e in expected)==171217932
  f=row['native']['feedback'];assert f['sealed'] and f['request_steps']==31 and f['total_steps']==row['native']['native_replays']==i*31 and f['resets']==i and f['request_seed_uploaded_ranges']==131
  assert f['CPU_output_compare_bytes_total']==0 and f['CPU_full_state_restore_per_step'] is False and f['CPU_seed_upload_bytes_total']==170721292*i
  if version=='v36':
   pp=f['pingpong'];assert (pp['requested_count'],pp['eligible_count'],pp['staged_link_count'])==(48,48,83)
  else:assert row['native']['last_resource_barriers_encoded']==0 and row['native']['resource_barrier_mappings']==[]
  if row['phase']=='measure':measures.append(row)
 summary={'n':3,'median_decode_ns_per_step':statistics.median(x['decode_ns_per_step'] for x in measures),'median_decode31_tokens_per_s':statistics.median(x['decode31_tokens_per_s'] for x in measures),'median_whole32_tokens_per_s':statistics.median(x['whole32_tokens_per_s'] for x in measures)}
 for k,v in summary.items():assert r['measured_summary'][k]==v
 times=None
 if version=='v36':
  previous=[]
  for i,row in enumerate(rows,1):
   t=row['native_timing'];assert sha(D/t['raw_file'])==t['raw_sha256']
   ts=read(D/t['raw_file']);assert ts['record_count']==len(ts['records'])==i*31 and ts['records'][:len(previous)]==previous
   current=ts['records'][len(previous):];assert [x['step'] for x in current]==list(range(1,32)) and {x['request'] for x in current}=={i}
   assert t['current_request_records']==31 and t['total_records']==i*31
   for x in current:
    assert x['CB_GPU_envelope_valid'] and x['CB_GPUStartTime_seconds']>0 and x['CB_GPUEndTime_seconds']>=x['CB_GPUStartTime_seconds']
    assert math.isclose(x['CB_GPU_envelope_ns'],(x['CB_GPUEndTime_seconds']-x['CB_GPUStartTime_seconds'])*1e9,rel_tol=0,abs_tol=1e-6)
    assert x['host_encode_ns']==x['graph_encode_ns']+x['feedback_encode_ns']
    assert x['host_total_ns']==x['host_encode_ns']+x['host_commit_ns']+x['host_wait_ns']
   assert sum(x['host_total_ns'] for x in current)<=row['decode31_ns']
   previous=ts['records']
  measuredtimings=[x for x in previous if x['request']>=4];assert len(measuredtimings)==93
  times={k:statistics.median(x[k] for x in measuredtimings) for k in ('host_encode_ns','graph_encode_ns','feedback_encode_ns','host_commit_ns','host_wait_ns','host_total_ns','CB_GPU_envelope_ns')}
 allres=[];resources={}
 for base,pattern,derived in ((D,'resource-??-raw.json','resources-derived.json'),(O,'external-resource-??-raw.json','external-resource.json')):
  ps=sorted(base.glob(pattern));ds=read(base/derived);assert len(ps)==len(ds)
  for j,(p,d) in enumerate(zip(ps,ds)):
   raw=read(p);mutable={'known_inference_processes','known_inference_process_count','only_owned_known_inference_processes','exactly_one_owned_known_model'}
   for k,v in raw.items():
    if k not in mutable:assert d[k]==v,(p,k)
   assert int(raw['pressure_level']['text'])==d['pressure_level_value']==1
   assert int(re.search(r'free percentage: (\d+)%',raw['memory_pressure']['text'])[1])==d['free_percent']
   assert float(re.search(r'used = ([0-9.]+)M',raw['swap']['text'])[1])==d['swap_used_mb']
   for k in ('Swapouts','Swapins','Pageouts','Compressions','Decompressions'):assert int(re.search(r'^'+k+r':\s+(\d+)\.',raw['vm_stat']['text'],re.M)[1])==d['vm_counters'][k]
   assert d['process_inventory_ok'] and d['only_owned_known_inference_processes'] and all(x['owned'] and x['pid']==child for x in d['known_inference_processes'])
   assert raw['power_metadata']['status']=='known' and raw['power_metadata']['source']=='AC Power' and "'AC Power'" in raw['power_metadata']['raw']['stdout']
   assert raw['processinfo_thermal_metadata']['status']=='known' and int(raw['processinfo_thermal_metadata']['raw']['stdout'])==raw['processinfo_thermal_metadata']['code']==0
   if j:
    prev=ds[j-1];assert d['swap_used_mb_delta']==d['swap_used_mb']-prev['swap_used_mb']<=0
    assert d['vm_counter_delta']=={k:v-prev['vm_counters'][k] for k,v in d['vm_counters'].items() if k in prev['vm_counters']} and d['vm_counter_delta']['Swapouts']==0
   allres.append(d)
  resources[str(base/derived)]={'samples':len(ps),'raw_manifest_sha256':digest([{'file':str(p),'sha256':sha(p)} for p in ps])}
 assert len(allres)==19
 assert own['owner_pid']==owner and own['driver_pid']==child and own['returncode']==0 and own['waited_reaped'] and own['stage']=='reaped'
 assert own['absence']['returncode']==1 and own['absence']['stdout']==own['absence']['stderr']=='' and own['absence']['absent']
 assert rel['status']=='PASS_EXTERNAL_RELEASE' and rel['tool_exit']==0 and rel['GPU_released'] and {x['pid'] for x in rel['strict_absence']}=={owner,child}
 assert all(x['returncode']==1 and x['stdout']==x['stderr']=='' for x in rel['strict_absence'])
 derived=read(D/'resources-derived.json')
 results[version]={'receipt_sha256':expectedsha,'source_pins_verified':pincount,'qualification_sha256':r['qualification_sha256'],'requests':6,'warm_requests':3,'measured_requests':3,'IDs_per_request':32,'final_strict_hash_records':792,'summary':summary,'measured_native_timing_medians_93_steps':times,'GPU_blit_bytes_per_step':rows[-1]['native']['feedback']['GPU_blit_bytes_per_step'],'resources':resources,'resource_summary':{'samples':19,'AC':True,'thermal_codes':sorted({x['processinfo_thermal_metadata']['code'] for x in allres}),'pressure_levels':sorted({x['pressure_level_value'] for x in allres}),'minimum_free_percent':min(x['free_percent'] for x in allres),'swap_MB_growth':derived[-1]['swap_used_mb']-derived[0]['swap_used_mb'],'Swapouts_growth':derived[-1]['vm_counters']['Swapouts']-derived[0]['vm_counters']['Swapouts'],'Pageouts_growth':derived[-1]['vm_counters']['Pageouts']-derived[0]['vm_counters']['Pageouts']},'cleanup':{'owner':owner,'child':child,'reaped_exit0':True,'saved_external_strict_absence':True},'metadata_sha256':{str(p):sha(p) for p in sorted(list(D.glob('*.json'))+list(O.glob('*.json')))}}
assert loopAST[0]==loopAST[1] and sampleAST[0]==sampleAST[1]
assert not any(x=='mlx' or x.startswith('mlx.') for x in sys.modules)
result={'status':'PASS_SAVED_V36_V37_B_ONLY_SCREEN_EVIDENCE_CPU_AUDIT','scope':'Independent saved receipt/hash/calculation replay; no new GPU execution. Separate B-only exploratory runs, no paired A/B/HTTP/Pareto/adoption claim. GPU envelope overlaps host wait and is not an exclusive-kernel duration.','timed_loop_AST_equal':True,'resource_guard_AST_equal':True,'results':results}
(OUT/'receipt.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
print(json.dumps({v:{k:x[k] for k in ('summary','measured_native_timing_medians_93_steps','resource_summary','GPU_blit_bytes_per_step')} for v,x in results.items()},indent=2))
