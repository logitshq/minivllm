"""CPU saved-evidence recheck by the V127 harness author; no independent-source claim."""
from pathlib import Path
import ast,collections,hashlib,json,math
D=Path(__file__).resolve().parent;P=D.parent;R=P/'v23-root-control-build1/numeric-v127-attempt1';S=P/'v127-real-input-shared-m1-qual-build1';C=R/'child-output'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text())
def write(n,v):(D/n).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def main():
 assert sha(C/'receipt.json')=='011ad4f71498ae2e4ab0ca6fc551b368242c0b2c1abc4591e63224ea6aad0d52'
 q=load(C/'receipt.json');protocol=load(S/'protocol.json');pins=load(S/'source-pins.json');manifest=load(S/'manifest.json')
 assert q['status']=='PASS_V127_497_REAL_T5_AND_DERIVED_T2_T4_SHARED_W_M1_LOCAL_ONLY' and q['final_errors']==[]
 assert sha(S/'qualify.py')=='0171c8161435deb5c4df42253d7b4a3ab44e893096f92332ddeba7b6c32e1e96'
 assert sha(S/'source-pins.json')==q['source_pins_sha256']=='a09781dfeaedc6f1218262959c62cd3109c5d793f05222f2d1405158f4c3ec68'
 assert load(C/'preimport-pins.json')==pins and len(pins)==77
 for p,h in pins.items():assert sha(p)==h,p
 for n,meta in manifest['files'].items():assert sha(S/n)==meta['sha256'] and (S/n).stat().st_size==meta['bytes'],n
 old=load(P/'v23-root-control-build1/diagnostic-v113-attempt2/child-output/receipt.json')
 oldrows={r['name']:r for r in old['comparisons'] if r['variant']=='batch_vs_rowM1'}
 gold=load(P/'v113-broadcast-m1-diagnostic-build1/actual-linear-gold.json')['operators']
 block=load(P/'v113-broadcast-m1-diagnostic-build1/actual-block.json')
 rows=q['comparisons'];routes=q['routes'];ro=q['readonly'];original=q['original_hash_checks'];layouts=q['layouts']
 assert len(rows)==len(routes)==1491 and len(ro)==len(original)==497 and len(layouts)==2485
 assert q['completed_matrix_checks']==q['strict_equal_matrix_checks']==1491 and q['captured_actual_T5_operator_count']==497 and q['fallback_calls']==0
 by_name=collections.defaultdict(dict)
 observed=collections.Counter()
 for row,route in zip(rows,routes):
  n,k,t=row['N'],row['K'],row['T'];name=row['name'];assert t in (2,4,5) and name not in ()
  assert row['input_scope']==('actual_saved_B_verify_T5' if t==5 else 'derived_prefix_of_actual_T5_X')
  assert row['route']==route['route']=='shared124' and row['same_original_weight_and_scales'] is route['same_original_weight_and_scales'] is True
  assert all(row[z]==route[z] for z in ('name','T','N','K'))
  assert row['shape_dtype_finite_bytes_strict'] is True
  a,b=row['A_native_M1'],row['B_shared_or_fallback'];assert a==b and a['shape']==[t,n] and a['dtype']=='mlx.core.bfloat16' and a['bytes']==t*n*2
  assert row['input_signature']['shape']==[t,k] and row['input_signature']['dtype']=='mlx.core.bfloat16' and row['input_signature']['bytes']==t*k*2
  assert len(a['sha256'])==64 and t not in by_name[name];by_name[name][t]=row;observed[(t,n,k)]+=1
  if t==5:
   assert a['sha256']==gold[name]['row_M1_sha256']==oldrows[name]['right_sha256']
   assert [row['input_signature']]==gold[name]['input_signatures']==oldrows[name]['input_signatures']
 assert len(by_name)==497 and all(set(v)=={2,4,5} for v in by_name.values())
 expected=collections.Counter({(t,*map(int,key.replace('N','').split(':K'))):count for t in (2,4,5) for key,count in protocol['expected_geometry_operator_counts'].items()})
 assert observed==expected and len(observed)==27
 assert collections.Counter({(r['T'],r['N'],r['K']):r['calls'] for r in q['shared_route_counts']})==expected
 for item,readonly in zip(original,ro):
  name=item['name'];assert name==readonly['name'] and item['old112_and113_equal'] is True
  assert [item['input']]==gold[name]['input_signatures']==oldrows[name]['input_signatures']
  assert item['batch_output']['sha256']==gold[name]['batch_sha256']==oldrows[name]['left_sha256']
  assert readonly['before']==readonly['after'] and readonly['owner_ids_before']==readonly['owner_ids_after'] and readonly['unchanged'] is True
  assert len(readonly['owner_ids_before'])==4
  b=readonly['before'];assert b['x']==item['input'] and b['y']==item['batch_output']
  assert b['weight']==q['parameters_before'][name+'.weight'] and b['scales']==q['parameters_before'][name+'.scales']
  assert b['x']==by_name[name][5]['input_signature']
 names={r['label']:r for r in layouts};assert len(names)==2485
 for name,trows in by_name.items():
  base=trows[5];n,k=base['N'],base['K']
  for suffix,shape,strides in [('weight',[n,k//8],[k//8,1]),('scales',[n,k//16],[k//16,1])]+[(f'X.T{t}',[t,k],[k,1]) for t in (5,2,4)]:
   row=names[name+'.'+suffix];assert row['shape']==shape and row['actual_strides_elements']==strides and row['internal_MLX_flags_observed'] is row['ensure_row_contiguous'] is False
 assert q['parameters_before']==q['parameters_after'] and len(q['parameters_before'])==1349 and len(q['parameter_identity_before'])==1349
 assert len({name for name,_ in q['parameter_identity_before']})==1349
 assert q['target1349_identity_readonly'] and q['keep0_restored_all128'] and q['initial_cache_matches_actual108_all128']
 assert q['live_anchor']==block['assert_live_anchor'] and q['observed_posterior']==block['assert_posterior']
 for flag in ('all497_original_input_and_batch_output_hashes_reproduced','all497_native_M1_output_hashes_reproduced'):assert q[flag] is True
 assert q['model_loads']==1 and q['draft_loads']==0 and not q['live_drafting'] and not q['future_A_output_used_as_proposal']
 assert not q['timing_run'] and not q['fullmodel_qualified'] and not q['adopted']
 warm=q['ordinary_warm'];assert warm['raw_output_ids']==load(P/'workload.json')['continuation_ids'] and warm['output_tokens']==32 and warm['runner_idle_before_attach'] and warm['drafter_absent_during_warm']
 assert len(q['actual_imported_candidate_sources'])==26
 for path in q['actual_imported_candidate_sources'].values():assert pins[path]==sha(path)
 # Recompute prefix hashes for the one original input whose raw payload was saved in113.
 raw_path=P/'v23-root-control-build1/diagnostic-v113-attempt2/child-output/first-layer-projection-input.bf16';data=raw_path.read_bytes();first=by_name['layers.0.linear_attn.in_proj_qkv']
 assert len(data)==51200
 for t in (5,2,4):assert hashlib.sha256(data[:t*5120*2]).hexdigest()==first[t]['input_signature']['sha256']
 groups={'worker':q['resources'],'owner':load(R/'resources.json'),'external_after_release':[load(R/'external-resource.json')]}
 assert [len(v) for v in groups.values()]==[21,20,1]
 for group,values in groups.items():
  assert all(a['monotonic_s']<=b['monotonic_s'] for a,b in zip(values,values[1:]))
  for row in values:
   assert row['power_metadata']['source']=='AC Power' and row['processinfo_thermal_metadata']['code']==0 and row['pressure_level_value']==1
   assert row['swap_used_mb']==5038.06 and row['vm_counters']['Swapouts']==17519593 and all(p['owned'] for p in row['known_inference_processes'])
  if group!='external_after_release':
   for i,row in enumerate(values):
    path=C/f'resource-{i:03d}-raw.json' if group=='worker' else R/f'resource-{i:03d}-raw.json';raw=load(path)
    assert all(row[k]==v for k,v in raw.items() if k not in ('known_inference_processes','known_inference_process_count')),path
    if row['known_inference_processes']!=raw['known_inference_processes']:
     assert group=='owner' and row['known_inference_process_count']==len(row['known_inference_processes'])
     assert all(v['pid']==50911 and v['owned'] for v in row['known_inference_processes'])
 worker=groups['worker'];peak=max(x['mlx_memory']['get_peak_memory'] for x in worker);assert peak==16952614436
 owner=load(R/'owner-receipt.json');release=load(R/'external-release.json');assert owner['returncode']==0 and owner['waited_reaped'] and owner['stage']=='reaped' and owner['driver_sha256']==sha(S/'qualify.py')
 assert release['tool_session']==82729 and release['tool_exit_code']==0 and release['GPU_released'] and release['source_receipt_sha256']==sha(C/'receipt.json')
 assert {v['pid'] for v in release['strict_absence']}=={50896,50911}=={owner['owner_pid'],owner['driver_pid']}
 for v in [owner['absence'],*release['strict_absence']]:assert v['returncode']==1 and v['stdout']==v['stderr']=='' and v['absent'] is True
 # Inspect exact source assertions; these are not fresh tensor recomputation.
 text=(S/'qualify.py').read_text();ast.parse(text)
 for fragment in ('ab==nb','signature(actual_x)==x_before','transaction.eval_arrays()','transaction.commit(0)','cache_signatures()==initial','parameter_signatures()','sa==sn'):assert fragment in text
 result={'status':'PASS_AUTHOR_SAVED_EVIDENCE_RECHECK_V127_LOCAL_ONLY','reviewer_role':'V127 harness/plan/pilot author; saved-results recheck, NOT independent source-author review','actual_receipt_sha256':sha(C/'receipt.json'),'source_pins_verified':77,'immutable_manifest_payloads_verified':len(manifest['files']),'matrix_checks':1491,'actual_original_T5_matrices':497,'derived_prefix_matrices':994,'geometry_T_routes':27,'original_input_batch_and_native_M1_hash_matches':497,'readonly_operator_records':497,'target_parameter_readonly_records':1349,'target_parameter_payload_bytes':sum(v['bytes'] for v in q['parameters_before'].values()),'GPU_layout_descriptors':2485,'internal_MLX_flags_observed':False,'source_enforced_keep0_components':128,'literal_scope':'Source requests all original transaction.eval_arrays before alternatives, commits0 and compares all128 signatures to initial fixture. No separate post-keep0 per-array hashes/raw files or full literal-history readonly records were saved; execution success flags are source-enforced evidence.','raw_tensor_recomputation':False,'raw_scope':'All1491 saved exact-comparator results/metadata/hash pairs replayed; complete output payloads not saved. Only one113 original input raw blob supports CPU prefix rehash.','runtime_resources':{'worker_samples':21,'outer_samples':20,'external_post_samples':1,'total':42,'all_recorded_AC_thermal0_pressure1':True,'Swapouts':17519593,'swap_used_MB':5038.06,'free_percent_min':min(x['free_percent'] for g in groups.values() for x in g),'recorded_MLX_allocator_peak_bytes':peak,'recorded_MLX_active_max_bytes':max(x['mlx_memory']['get_active_memory'] for x in worker),'configured_cache_bytes':q['runtime']['cache_limit_bytes'],'recorded_MLX_cache_max_bytes':max(x['mlx_memory']['get_cache_memory'] for x in worker),'memory_scope':'MLX allocator counters, not RSS/physical footprint; cache setting is not an observed hard maximum.'},'cleanup':{'root_saved_wait_reap_reused':True,'external_PID_absence_reused':[50896,50911],'tool_session':82729,'tool_exit_code':0,'fresh_liveps_by_reviewer':False},'performance_or_fullmodel_or_adoption_claim':False,'GPU_used_by_reviewer':False,'W_or_Git_mutation':False}
 write('verification.json',result)
 raw_files=sorted(p for p in R.rglob('*') if p.is_file());write('raw-seals.json',{'scope':'original saved byte identities only; raw bodies not copied','files':{str(p.relative_to(R)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in raw_files}})
 (D/'REVIEW.md').write_text('''# V127 保存结果复核\n\nPASS，仅限同输入局部数值。复核者是 V127 harness/计划/pilot 作者，这是 author recheck，不能称独立源码作者审查；实际 GPU 执行与实时释放证据由 root 完成。\n\n77 pins 与冻结 manifest 原字节一致；1491完整 BF16 矩阵严格比较记录通过：497原真实B轨迹T5、994同X派生T2/T4；九几何×三T均走共享W，无fallback。497旧输入、原batch输出、native-M1输出SHA全部复现；含lm_head。2485 actual GPU shape/stride descriptor一致，内部MLX flags未观察。\n\n497条X/原Y/W/scales owner与字节RO、1349参数字节前后及源内identity断言通过。literal histories与producer先同边界eval，keep0全部128签名回到原initial fixture由源内断言完成；未另存逐项post-keep0或完整literal-history RO快照，不能声称本次CPU重算了这些原tensor。完整1491输出raw也未保存，本审查重放比较旗标/shape/dtype/SHA与精确bytes比较源码。\n\n21worker+20owner+1post=42资源样本均AC/thermal0/pressure1，Swapouts17519593、swap5038.06MB不变。allocator峰值16,952,614,436B，非RSS；cache配置64MiB而sample最大69,883,654B，不声称硬上限未超。一次targetload、零draftload，一次普通raw32warm；固定原B proposal fixture，不是live drafting。\n\nroot tool82729 exit0，50896/50911已wait/reap并有严格externalabsence；复用root实时证明，未liveps。无整模spec、性能或采用结论，V129仍unfinished/unrun。\n''')
 write('receipt.json',{'status':result['status'],'reviewer_role':result['reviewer_role'],'actual_receipt_sha256':result['actual_receipt_sha256'],'source_pins_sha256':sha(S/'source-pins.json'),'files':{n:sha(D/n) for n in ('review.py','verification.json','raw-seals.json','REVIEW.md')},'review_GPU_used':False,'source_author_independence_claimed':False,'fullmodel_or_performance_qualified':False})
 print(json.dumps({'status':result['status'],'REVIEW_sha256':sha(D/'REVIEW.md'),'verification_sha256':sha(D/'verification.json'),'receipt_sha256':sha(D/'receipt.json')}))
if __name__=='__main__':main()
