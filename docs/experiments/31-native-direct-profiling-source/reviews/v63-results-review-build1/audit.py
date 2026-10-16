"""Independent saved V63 source/numeric/timestamp evidence audit. CPU stdlib only."""
from pathlib import Path
import ast,base64,collections,csv,hashlib,json,re,struct,sys
P=Path('/private/tmp/minivllm-engine-optimization-20261014');D=Path(__file__).resolve().parent
O=P/'v23-root-control-build1/fullmodel-v63native-attempt1';C=O/'child-output'
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
digest=lambda x:hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
r=read(C/'receipt.json');assert sha(C/'receipt.json')=='a4459b8841b605eb76b70fc29afd570bcc0902fbd8da1095b8d9735db5ecd4e5'
assert r['status']=='PASS_V63_PERTURBED_STAGE_PROFILE_93STEP_12276STRICT' and r['native_steps']==93 and r['strict_output_checks']==12276
assert r['profiled_steps']==3 and r['ordinary_steps']==90 and r['sampled_profile_steps']==2 and r['invalid_sampled_rows']==0
assert r['gates_on']==r['conv_on']==r['normreuse_cache_on']==48 and r['whole_functional_off'] and not r['performance_point'] and not r['adopted']
assert len(r['source_pins'])==69
for path,h in r['source_pins'].items():assert sha(path)==h,path
assert all(x in r['source_pins'] for x in r['actual_imported_sources'].values())
golds=[read(P/f'v23-root-control-build1/fullmodel-v{v}native-attempt1/child-output/receipt.json') for v in (38,44)]
assert r['actual44_gold_sha256']==sha(P/'v23-root-control-build1/fullmodel-v44native-attempt1/child-output/receipt.json')
ids=read(P/'workload.json')['continuation_ids'];strict=ro=0
for j,req in enumerate(r['requests'],1):
 assert req['status']=='PASS_REQUEST_STRICT' and len(req['steps'])==31 and req['A_ids']==req['B_ids'] and len(req['A_ids'])==32 and req['seed_upload_calls']==req['seed_arrays']==131
 if j<3:assert req['A_ids']==ids and req['changed_seed_inputs_vs_capture']==[]
 else:assert len(req['changed_seed_inputs_vs_capture'])==129
 n=req['native'];assert n==read(C/f"native-after-{req['name']}.json") and n['status']=='OK' and n['error']==''
 f=n['feedback'];i=f['icb'];assert f['total_steps']==n['native_replays']==31*j and f['request_steps']==31 and f['resets']==j and f['request_seed_uploaded_ranges']==131 and f['CPU_full_state_restore_per_step'] is False
 assert f['CPU_seed_upload_bytes_total']==170721292*j and f['CPU_output_compare_bytes_total']==171217932*31*j
 assert i['banks']==2 and i['commands_per_bank']==n['dispatches']==1625 and i['encoded_count']==i['execution_count']==31*j and i['bank_executions']==[16*j,15*j]
 assert (f['pingpong']['requested_count'],f['pingpong']['eligible_count'],f['pingpong']['staged_link_count'])==(48,48,83)
 for k,s in enumerate(req['steps'],1):
  assert s['step']==k and s['A_exact_L']==req['prompt_length']+k and s['B_fixed_L']==256 and s['A_token']==s['B_token']==req['A_ids'][k]
  assert s['native_prediction_copied_bytes']==4 and s['seed_uploads_this_step']==0 and s['reference_input_digest_readonly'] and len(s['outputs'])==132
  for gold in golds:assert s['outputs']==gold['requests'][j-1]['steps'][k-1]['outputs']
  for x in s['outputs']:assert x['finite'] and x['strict_bytes_equal'] and len(x['A_sha256'])==64;strict+=1
  for key,n,nb in [('seed',131,170721292),('capture_outputs',132,171217932),('request_seed',131,170721292)]:
   x=s['originals_readonly'][key];assert x['strict_bytes_equal'] and x['arrays']==n and x['bytes']==nb;ro+=1
assert strict==12276 and ro==279 and r['same_parameter_identity_count']==1349 and r['same_parameter_identity_preserved']
n=read(C/'native-sealed.json');buffers={x['original_buffer']:x for x in n['buffers']};allmap=None;profiles={};csvrows=[];gdn_groups={}
for j,meta in enumerate(r['profiles'],1):
 p=C/meta['raw_file'];assert sha(p)==meta['raw_sha256'];v=read(p);rows=v['rows'];sampled=j>1
 assert v['sampled']==sampled and v['request']==j and v['step']==1 and v['segments']==len(rows)==1625
 assert v['CB_GPUEndTime_seconds']>=v['CB_GPUStartTime_seconds']>0
 mapping=[];cats=collections.defaultdict(list);shapes=collections.defaultdict(list);roles=collections.defaultdict(list);group=collections.defaultdict(dict)
 for ordinal,row in enumerate(rows):
  assert row['ICB_ordinal']==ordinal and row['sampled']==sampled
  mapping.append({k:row[k] for k in ('ICB_ordinal','PSO','category','bindings')})
  b={x['index']:x for x in row['bindings']};assert len(b)==len(row['bindings'])
  for x in b.values():
   assert x['index']>=0 and x['offset']>=0
   raw=base64.b64decode(x['bytes_payload_base64'],validate=True)
   if x['original_buffer']:assert x['original_buffer'] in buffers and raw==b''
   else:assert raw and x['offset']==0
  if sampled:
   assert row['valid'] and 0<row['raw_start']<=row['raw_end']<(1<<64)-1 and row['raw_delta']==row['raw_end']-row['raw_start'];cats[row['category']].append(row['raw_delta'])
  else:assert not row['valid'] and row['raw_start']==row['raw_end']==0 and row['raw_delta'] is None
  if row['category']!='NVFP4-QMV':continue
  assert row['PSO']=='nvfp4_qmv_fast_bfloat16_t_gs_16_b_4_batch_0' and set(b)=={0,1,3,4,5,6}
  K,N=[struct.unpack('<i',base64.b64decode(b[t]['bytes_payload_base64']))[0] for t in (5,6)];assert K>0 and N>0 and K%512==0 and N%8==0
  weight=buffers[b[0]['original_buffer']];scales=buffers[b[1]['original_buffer']]
  info=[]
  for index,buffer,suffix,expected in ((0,weight,'weight',N*K//2),(1,scales,'scales',N*K//16)):
   m=re.fullmatch(r'parameter\.(.+) full-range=(\d+)\+(\d+) binding-delta=(\d+) readonly:(.+)/(w|scales)',buffer['role']);assert m
   name,start,size,delta,pso,arg=m.groups();assert name.endswith('.'+suffix) and int(size)==expected and int(delta)==0 and pso==row['PSO']
   assert buffer['used'] and buffer['zero_copy'] and int(start)==buffer['original_contents']+b[index]['offset'] and buffer['length']>=expected+b[index]['offset'];info.append(name.rsplit('.',1)[0])
  assert info[0]==info[1];name=info[0];role=name.split('.',2)[2] if name.startswith('layers.') else name
  assert buffers[b[3]['original_buffer']]['length']>=b[3]['offset']+2*K and buffers[b[4]['original_buffer']]['length']>=b[4]['offset']+2*N
  if sampled:shapes[K,N].append(row['raw_delta']);roles[role].append(row['raw_delta'])
  csvrows.append({'profile':meta['request'],'ordinal':ordinal,'K':K,'N':N,'parameter':name,'weight_buffer':b[0]['original_buffer'],'scale_buffer':b[1]['original_buffer'],'x_buffer':b[3]['original_buffer'],'x_offset':b[3]['offset'],'output_buffer':b[4]['original_buffer'],'output_offset':b[4]['offset'],'weight_payload_bytes':N*K//2,'scale_payload_bytes':N*K//16,'raw_delta':row['raw_delta']})
  if role.startswith('linear_attn.in_proj_'):
   layer=int(name.split('.')[1]);group[layer][role.split('_')[-1]]=(b[3]['original_buffer'],b[3]['offset'],K,N)
 assert all(set(x)=={'qkv','z','b','a'} and len({v[:3] for v in x.values()})==1 and {k:v[3] for k,v in x.items()}=={'qkv':10240,'z':6144,'b':48,'a':48} for x in group.values()) and len(group)==48
 if allmap is None:allmap=mapping
 else:assert mapping==allmap
 assert hashlib.sha256(json.dumps(mapping,sort_keys=True).encode()).hexdigest()==meta['ordinal_map_sha256']
 result={'raw_sha256':sha(p),'sampled':sampled,'segments':1625,'valid_rows':1625 if sampled else 0,'QMV_decoded':497,'unknown_QMV':0,'GDN_groups_same_x':48,'CB_GPU_envelope_ms':(v['CB_GPUEndTime_seconds']-v['CB_GPUStartTime_seconds'])*1000}
 if sampled:
  assert sum(map(len,cats.values()))==1625 and len([x for x in csvrows if x['profile']==meta['request']])==497
  result.update(category_raw_delta=[{'category':k,'commands':len(x),'sum_raw_delta':sum(x)} for k,x in cats.items()],shapes_ranked=[{'K':k[0],'N':k[1],'commands':len(x),'sum_raw_delta':sum(x)} for k,x in sorted(shapes.items(),key=lambda x:-sum(x[1]))],roles_ranked=[{'role':k,'commands':len(x),'sum_raw_delta':sum(x)} for k,x in sorted(roles.items(),key=lambda x:-sum(x[1]))],sum_raw_delta=sum(row['raw_delta'] for row in rows),span_raw=rows[-1]['raw_end']-rows[0]['raw_start'],adjacent_overlaps=sum(rows[t]['raw_start']<rows[t-1]['raw_end'] for t in range(1,1625)),between_encoder_gaps_raw=sum(max(0,rows[t]['raw_start']-rows[t-1]['raw_end']) for t in range(1,1625)))
 profiles[meta['request']]=result
with (D/'qmv-bindings.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(csvrows[0]));w.writeheader();w.writerows(csvrows)
# Reuse only frozen pure-stdlib saved-resource/cleanup functions, never import the old audit's top-level work.
tree=ast.parse((P/'v38-results-review-build1/audit.py').read_text());scope=globals().copy();exec(compile(ast.Module(body=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name in ('resources','cleanup')],type_ignores=[]),'saved-resource-helpers','exec'),scope)
resource=scope['resources'](O,C,79055);release=scope['cleanup'](O,79046,79055)
assert not any(x=='mlx' or x.startswith('mlx.') or x=='numpy' for x in sys.modules)
source=[P/'v63-native-stage-profile-build1/profile.mm',P/'v63-native-stage-profile-build1/fullqual.py',Path('/Users/ryanli/code/minivllm/.venv/lib/python3.13/site-packages/mlx/include/mlx/backend/metal/kernels/fp_quantized.h'),P/'SUMMARY-GDN-PROJECTION-MICRO-ATTEMPT2.md',P/'DESIGN-GDN-PROJECTION-PROTOTYPE.md']
result={'status':'PASS_SAVED_V63_NUMERIC_AND_ALL_RAW_TIMESTAMP_CPU_AUDIT','scope':'Saved source/A hashes/Cmemcmp/timestamps/resource/PID evidence replay; no independent GPU recomputation or production time-share conversion.','receipt_sha256':sha(C/'receipt.json'),'source_pins':69,'strict_output_records':strict,'original_readonly_records':ro,'reference_readonly_records':93,'profiles':profiles,'QMV_csv_sha256':sha(D/'qmv-bindings.csv'),'resources':resource,'cleanup':release,'sources':{str(p):sha(p) for p in source},'timestamp_units':'Unconverted raw uint64 counter values; SDK MTLCounters timestamp field has no numeric unit, capture contains no sampleTimestamps calibration or frequency query. CB GPUStartTime/EndTime are documented host seconds.'}
(D/'receipt.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps({'status':result['status'],'strict':strict,'readonly':ro,'resources':resource,'cleanup':release,'profiles':profiles},indent=2))
