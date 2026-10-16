"""Independent saved V38 evidence audit, CPU/std library only; no native load."""
from pathlib import Path
import ast,hashlib,json,math,re,statistics,struct,sys
P=Path('/private/tmp/minivllm-engine-optimization-20261014');OUT=Path(__file__).parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
digest=lambda x:hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def pins(r,count):
 assert len(r['source_pins'])==count
 for p,h in r['source_pins'].items():assert sha(p)==h,p
 assert r['source_pins'][str(P/'source-manifest-v17-gdn-gates-build1.json')]=='31fa053925bb732ce2a68b4feb8e4d2351d73939138f6a6c4a7d56df2b5cb41c'
def icb(n,requests):
 f=n['feedback'];i=f['icb'];pp=f['pingpong'];steps=requests*31
 assert n['status']=='OK' and n['error']=='' and n['arena_built'] and n['native_replays']==steps
 assert i['enabled'] and i['banks']==2 and i['commands_per_bank']==n['dispatches']==2009
 assert i['dispatchThreads']+i['dispatchThreadgroups']==2009 and i['encoded_count']==i['execution_count']==steps and i['bank_executions']==[requests*16,requests*15]
 assert len(i['ICB_records'])==2 and [x['bank'] for x in i['ICB_records']]==[0,1] and len({x['ICB'] for x in i['ICB_records']})==2
 assert all(x['commands']==2009 and x['S_binding_bank']==x['bank'] for x in i['ICB_records'])
 assert i['unique_original_PSOs']==i['recreated_PSOs']==len(i['PSO_records'])==33
 assert all(x['resolved_original_function']>0 and x['original_descriptor']>0 and x['actual_supports_ICB'] and not x['original_supports_ICB'] and x['recreated'] and x['original_PSO']!=x['ICB_PSO'] for x in i['PSO_records'])
 assert i['constant_alignment']==256 and i['constant_bank_bytes']>0 and i['constant_bank_allocation']>=i['constant_bank_bytes'] and i['constant_bank_allocation']%256==0 and i['setBytes_payload_count']>0
 assert i['trailing_barrier_via_graph_fence'] and i['barrier_commands_per_bank']==1238 and i['original_barrier_count']>=1238
 rr=i['resources'];assert rr['read_only_count']==1350 and rr['read_only_parameter_buffers']==1349 and rr['constant_bank_read_declarations_per_CB']==rr['mutable_arena_count']==rr['icb_read_declarations']==1
 assert (pp['requested_count'],pp['eligible_count'],pp['staged_link_count'])==(48,48,83)
 assert f['request_steps']==31 and f['total_steps']==steps and f['resets']==requests and f['request_seed_uploaded_ranges']==131 and f['CPU_seed_upload_bytes_total']==170721292*requests and f['CPU_full_state_restore_per_step'] is False
 assert f['GPU_blit_bytes_per_step']==39949336 and all(v==0 for v in f['first_difference'].values())
 return {k:i[k] for k in ('banks','commands_per_bank','encoded_count','execution_count','bank_executions','unique_original_PSOs','recreated_PSOs','constant_bank_bytes','constant_bank_allocation','setBytes_payload_count','original_barrier_count','barrier_commands_per_bank','resources')}
def resources(O,D,child):
 allres=[];files={}
 for base,pattern,derived in ((O,'resource-???-raw.json','resource-derived.json'),(D,'resource-??-raw.json','resources-derived.json'),(O,'external-resource-??-raw.json','external-resource.json')):
  ps=sorted(base.glob(pattern))
  if not ps:continue
  ds=read(base/derived);assert len(ps)==len(ds)
  for j,(p,d) in enumerate(zip(ps,ds)):
   raw=read(p);mutable={'known_inference_processes','known_inference_process_count','only_owned_known_inference_processes','exactly_one_owned_known_model'}
   for k,v in raw.items():
    if k not in mutable:assert d[k]==v,(p,k)
   assert int(raw['pressure_level']['text'])==d['pressure_level_value']==1 and int(re.search(r'free percentage: (\d+)%',raw['memory_pressure']['text'])[1])==d['free_percent']
   assert float(re.search(r'used = ([0-9.]+)M',raw['swap']['text'])[1])==d['swap_used_mb']
   for k in ('Swapouts','Swapins','Pageouts','Compressions','Decompressions'):assert int(re.search(r'^'+k+r':\s+(\d+)\.',raw['vm_stat']['text'],re.M)[1])==d['vm_counters'][k]
   assert d['process_inventory_ok'] and d['only_owned_known_inference_processes'] and all(x['owned'] and x['pid']==child for x in d['known_inference_processes'])
   assert raw['power_metadata']['status']=='known' and raw['power_metadata']['source']=='AC Power' and "'AC Power'" in raw['power_metadata']['raw']['stdout']
   assert raw['processinfo_thermal_metadata']['status']=='known' and int(raw['processinfo_thermal_metadata']['raw']['stdout'])==raw['processinfo_thermal_metadata']['code']==0
   if j:
    prev=ds[j-1];assert d['swap_used_mb_delta']==d['swap_used_mb']-prev['swap_used_mb']==0
    assert d['vm_counter_delta']=={k:v-prev['vm_counters'][k] for k,v in d['vm_counters'].items() if k in prev['vm_counters']} and d['vm_counter_delta']['Swapouts']==0
   allres.append(d)
  files[str(base/derived)]={'samples':len(ps),'raw_manifest_sha256':digest([{'file':str(p),'sha256':sha(p)} for p in ps]),'Pageouts_growth':ds[-1]['vm_counters']['Pageouts']-ds[0]['vm_counters']['Pageouts']}
 return {'samples':len(allres),'AC':True,'thermal_codes':sorted({x['processinfo_thermal_metadata']['code'] for x in allres}),'pressure_levels':sorted({x['pressure_level_value'] for x in allres}),'minimum_free_percent':min(x['free_percent'] for x in allres),'swap_MB_growth':0,'Swapouts_growth':0,'files':files}
def cleanup(O,owner,child):
 a=read(O/'owner-receipt.json');b=read(O/'external-release.json');assert a['owner_pid']==owner and a.get('child_pid',a.get('driver_pid'))==child and a['stage']=='reaped' and a['returncode']==0 and a['waited_reaped']
 assert a['absence']['returncode']==1 and a['absence']['stdout']==a['absence']['stderr']==''
 assert b['status']=='PASS_EXTERNAL_RELEASE' and b['GPU_released'] and b['tool_exit']==0 and {x['pid'] for x in b['strict_absence']}=={owner,child}
 assert all(x['returncode']==1 and x['stdout']==x['stderr']=='' for x in b['strict_absence'])
 return {'owner':owner,'child':child,'reaped_exit0':True,'saved_external_strict_absence':True}
Q=P/'v23-root-control-build1/fullmodel-v38native-attempt1';QD=Q/'child-output';q=read(QD/'receipt.json')
assert sha(QD/'receipt.json')=='c57e7625baa5a4ce450fc7306233e766f1efd71636713080d748780624542c8a'
assert q['status']=='PASS_NATIVE_ICB_S_PINGPONG_BASE_RESET_CHANGED_STRICT' and q['native_steps']==93 and q['source_pins_verified_final'] and q['ICB_enable_called_after_feedback_seal']
assert q['gates_on']==q['gates_callable_count']==48 and q['conv_off'] and q['whole_functional_off'] and not q['adopted'] and not q['performance_point']
pins(q,47)
for p in q['actual_imported_sources'].values():assert p in q['source_pins'] and 'candidate-v17-gdn-gates-build1-root/' in p
assert len(q['capture']['inputs'])==131 and len(q['capture']['outputs'])==132 and q['capture']['parameter_arrays']==1349
ids=read(P/'workload.json')['continuation_ids'];base=read(P/'workload.json')['prompt_ids'];strict=readonly=0
assert q['declared_prompts']=={'base':base,'reset-base':base,'changed-prefill':base[:10]+[base[3]]+base[10:]}
for j,req in enumerate(q['requests'],1):
 assert req['status']=='PASS_REQUEST_STRICT' and req['A_ids']==req['B_ids'] and len(req['A_ids'])==32 and len(req['steps'])==31 and req['seed_upload_calls']==req['seed_arrays']==131
 assert req['prompt_ids']==q['declared_prompts'][req['name']] and req['prompt_length']==len(req['prompt_ids'])
 if j<3:assert req['A_ids']==ids and req['changed_seed_inputs_vs_capture']==[]
 else:assert len(req['changed_seed_inputs_vs_capture'])==129
 icb(req['native'],j);assert req['native']['feedback']['CPU_output_compare_bytes_total']==171217932*31*j
 assert req['native']==read(QD/f"native-after-{req['name']}.json")
 for k,s in enumerate(req['steps'],1):
  assert s['step']==k and s['A_exact_L']==len(req['prompt_ids'])+k and s['B_fixed_L']==256 and s['A_token']==s['B_token']==req['A_ids'][k]
  assert s['seed_uploads_this_step']==0 and s['native_prediction_copied_bytes']==4 and s['reference_input_digest_readonly'] and len(s['outputs'])==132
  for x,c in zip(s['outputs'],q['capture']['outputs']):
   assert x['label']==c['label'] and x['finite'] and x['strict_bytes_equal'] and x['shape']==c['shape'] and x['dtype']==c['dtype'] and x['bytes']==c['nbytes'] and len(x['A_sha256'])==64;strict+=1
  for label,value in (('prediction',s['A_token']),('next_positions',s['A_exact_L']),('next_lens',s['A_exact_L']+1)):
   assert next(x for x in s['outputs'] if x['label']==label)['A_sha256']==hashlib.sha256(struct.pack('<i',value)).hexdigest()
  for key,n,nb in (('seed',131,170721292),('capture_outputs',132,171217932),('request_seed',131,170721292)):
   ro=s['originals_readonly'][key];assert ro['strict_bytes_equal'] and ro['arrays']==n and ro['bytes']==nb;readonly+=1
assert strict==12276 and readonly==279
assert all(a['outputs']==b['outputs'] for a,b in zip(q['requests'][0]['steps'],q['requests'][1]['steps']))
assert q['requests'][0]['steps'][0]['outputs'][0]['A_sha256']!=q['requests'][2]['steps'][0]['outputs'][0]['A_sha256']
S=P/'v23-root-control-build1/native-decode-v38-attempt1';SD=S/'child-output';s=read(SD/'receipt.json')
assert sha(SD/'receipt.json')=='abc40171167e180a6c9d8cc559d2fb15113e6908861debc3d390a3a76e0b1755'
assert s['status']=='PASS_ICB_S_PINGPONG_B_ONLY_EXPLORATORY_SCREEN' and s['qualification_sha256']==sha(QD/'receipt.json') and s['qualification_receipt']==str(QD/'receipt.json') and s['schedule']==['warm']*3+['measure']*3
pins(s,51);expected=q['requests'][0]['steps'][-1]['outputs'];previous=[]
for j,row in enumerate(s['requests'],1):
 assert row['index']==j and row['phase']==s['schedule'][j-1] and row['ids']==ids and len(row['final132_hash_checks'])==132
 for x,e in zip(row['final132_hash_checks'],expected):assert x=={'label':e['label'],'bytes':e['bytes'],'sha256':e['A_sha256'],'qualified_A_sha256':e['A_sha256']}
 t0,t1,t2=[row[k] for k in ('t0_ns','t1_ns','t2_ns')];assert t0<t1<t2
 assert row['prefill_seed_ready_ns']==t1-t0 and row['decode31_ns']==t2-t1 and row['whole32_ns']==t2-t0
 assert row['decode_ns_per_step']==(t2-t1)/31 and row['decode31_tokens_per_s']==31e9/(t2-t1) and row['whole32_tokens_per_s']==32e9/(t2-t0)
 icb(row['native'],j);assert row['native']['feedback']['CPU_output_compare_bytes_total']==0 and row['final_hash_cpu_copy_bytes_outside_timing']==171217932
 t=row['native_timing'];assert sha(SD/t['raw_file'])==t['raw_sha256'];ts=read(SD/t['raw_file']);assert ts['record_count']==len(ts['records'])==j*31 and ts['records'][:len(previous)]==previous
 current=ts['records'][len(previous):];assert {x['request'] for x in current}=={j} and [x['step'] for x in current]==list(range(1,32))
 for x in current:
  assert x['CB_GPU_envelope_valid'] and math.isclose(x['CB_GPU_envelope_ns'],(x['CB_GPUEndTime_seconds']-x['CB_GPUStartTime_seconds'])*1e9,abs_tol=1e-6,rel_tol=0)
  assert x['host_encode_ns']==x['graph_encode_ns']+x['feedback_encode_ns'] and x['host_total_ns']==x['host_encode_ns']+x['host_commit_ns']+x['host_wait_ns']
 assert sum(x['host_total_ns'] for x in current)<=row['decode31_ns'];previous=ts['records']
measures=[x for x in s['requests'] if x['phase']=='measure'];assert len(measures)==3
summary={k:statistics.median(x[field] for x in measures) for k,field in [('median_decode_ns_per_step','decode_ns_per_step'),('median_decode31_tokens_per_s','decode31_tokens_per_s'),('median_whole32_tokens_per_s','whole32_tokens_per_s')]}
for k,v in summary.items():assert s['measured_summary'][k]==v
measured=[x for x in previous if x['request']>=4];assert len(measured)==93
spans={k:statistics.median(x[k] for x in measured) for k in ('graph_encode_ns','feedback_encode_ns','host_encode_ns','host_commit_ns','host_wait_ns','host_total_ns','CB_GPU_envelope_ns')}
# Actual source only: no CPU seed writes in either qualification or timed screen loop.
for path,target,allowed in [(P/'v38-native-fullqual-build1/fullqual.py','step',{'v34_replay_step','v34_compare_output','v34_copy_output'}),(P/'v38-native-decode-screen-build1/screen.py','_',set())]:
 tree=ast.parse(path.read_text());loop=next(x for x in ast.walk(tree) if isinstance(x,ast.For) and isinstance(x.target,ast.Name) and x.target.id==target)
 calls=[x.args[0].value for x in ast.walk(loop) if isinstance(x,ast.Call) and isinstance(x.func,ast.Attribute) and x.func.attr=='call' and x.args and isinstance(x.args[0],ast.Constant)]
 assert set(calls)==allowed
 if target=='_':assert {ast.unparse(x.func) for x in ast.walk(loop) if isinstance(x,ast.Call)}=={'range','step_fn','copy_fn','RuntimeError','ids.append'}
assert not any(x=='mlx' or x.startswith('mlx.') for x in sys.modules)
result={'status':'PASS_SAVED_V38_FULLQUAL_AND_B_SCREEN_EVIDENCE_CPU_AUDIT','scope':'Saved source/counter/A-hash/Cmemcmp/finalhash/time/resource replay; no independent GPU recomputation. Native ICB qualification and B-only absolute timing, no engine/service/HTTP/Pareto/adoption claim. GPU envelope overlaps wait.','fullqual':{'receipt_sha256':sha(QD/'receipt.json'),'source_pins':47,'reference_manifest_sha256':'31fa053925bb732ce2a68b4feb8e4d2351d73939138f6a6c4a7d56df2b5cb41c','requests':3,'IDs':96,'native_steps':93,'strict_output_records':12276,'original_readonly_records':279,'reference_readonly_records':93,'changed_seed_inputs':129,'icb':icb(q['requests'][-1]['native'],3),'resources':resources(Q,QD,61332),'cleanup':cleanup(Q,61323,61332)},'screen':{'receipt_sha256':sha(SD/'receipt.json'),'source_pins':51,'requests':6,'warm':3,'measure':3,'final_hash_records':792,'summary':summary,'measured_93_native_span_medians_ns':spans,'icb':icb(s['requests'][-1]['native'],6),'resources':resources(S,SD,62071),'cleanup':cleanup(S,62070,62071)},'metadata_sha256':{str(p):sha(p) for root in (Q,QD,S,SD) for p in sorted(root.glob('*.json'))}}
(OUT/'receipt.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
print(json.dumps({k:{n:v[n] for n in v if n not in ('resources','icb')}|{'resource_summary':{x:v['resources'][x] for x in v['resources'] if x!='files'}} for k,v in result.items() if k in ('fullqual','screen')},indent=2))
