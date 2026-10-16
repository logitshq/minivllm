"""CPU stdlib replay of saved evidence only. No live probes/runtime/model imports."""
from pathlib import Path
import array,hashlib,json,math,statistics,sys
P=Path('/private/tmp/minivllm-engine-optimization-20261014');R=P/'v23-root-control-build1';D=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
expected={'fullmodel-v72num-attempt1':'b4159e4d4758fa1ffd20f11165c6428e7301975b14178f7a88fba7722be27c20','fullmodel-v72num2-attempt1':'d1356c8915ca2dc5c07537e024154d60411a55ce0a23bd3778cdea2fa6916967','kernel-micro-v72b2-attempt1':'f7eeff33a9e5d2f3af8a6a46d1d66a00913953f8b26960696a0f50e9a2f46744','fullmodel-v74native-attempt1':'3ac28207286e645e6453cd2680115b6adac7fc70bd270e2bc3cc09a1ad045a51','native-decode-v77-attempt1':'7d4e7c37bd6e1e70ee19fab4101b3b5535868de6375ecb730a949099b8d30292'}
receipts={};selected={};rawmap={}
def resources(out):
 xs=[]
 for p in sorted(out.glob('resource-*-raw.json')):
  x=read(p);rawmap[str(p)]=sha(p)
  assert x['power_metadata']['source']=='AC Power' and x['processinfo_thermal_metadata']['code']==0 and x['pressure_level_value']==1
  assert all(y['owned'] for y in x['known_inference_processes'])
  xs.append((x['swap_used_mb'],x['vm_counters']['Swapouts']))
 assert xs and len(set(xs))==1
 return {'samples':len(xs),'AC_thermal0_pressure1':True,'swap_used_MB':xs[0][0],'Swapouts':xs[0][1],'stable':True}
for name,digest in expected.items():
 root=R/name;out=root/'child-output';p=out/'receipt.json';assert sha(p)==digest
 r=read(p);receipts[name]=r;rel=read(root/'external-release.json');owner=read(root/'owner-receipt.json')
 assert rel['status']=='PASS_EXTERNAL_RELEASE' and rel['GPU_released'] is True and owner['stage']=='reaped' and owner['waited_reaped'] is True
 assert owner['returncode']==rel['tool_exit'] and all(x['returncode']==1 and x['stdout']==x['stderr']=='' for x in rel['strict_absence'])
 for path,d in r['source_pins'].items():assert sha(Path(path))==d,('sourcepin',name,path)
 selected[name]={'receipt_sha256':digest,'release_sha256':sha(root/'external-release.json'),'owner_receipt_sha256':sha(root/'owner-receipt.json'),'exit':owner['returncode'],'strict_absent_PIDs':[x['pid'] for x in rel['strict_absence']],'source_pins_verified':len(r['source_pins']),'resources':resources(out),'status':r['status']}
fail=receipts['fullmodel-v72num-attempt1'];assert fail['status']=='FAIL_STOP' and fail['timing_run'] is False and fail['adopted'] is False
failure=read(R/'fullmodel-v72num-attempt1/child-output/native-failure.json');assert failure['error']=='V72: active readwrite/unknown access forbidden'
assert failure['enabled'] is False and failure['loops']==0
for slot in failure['slot_records']:
 rw=[a for a in slot['actual_active_arguments'] if a['access']==1];assert len(rw)==1 and rw[0]['index']==(4 if slot['arm']==0 else 3) and rw[0]['name']==('y' if slot['arm']==0 else 'output')
pre=read(R/'fullmodel-v72num-attempt1/child-output/numeric-checks.json');assert len(pre)==72 and all(x['strict'] and x['A_sha256']==x['B_sha256'] for x in pre)
selected['fullmodel-v72num-attempt1'].update(formal_qualification=False,pre_failure_MLX_same_hash_checks=72,native_loops_completed=0)
hist={r['layer']:r for r in read(P/'swiglu-actual-micro-build1/gpu-actual-micro-run1/receipt.json')['rows']}
def check_raw(out,row):
 p=out/row['raw_file'];b=p.read_bytes();assert len(b)==34816 and sha(p)==row['sha256']==row['expected_sha256'] and row['strict'] is True
 vals=array.array('H');vals.frombytes(b)
 if sys.byteorder!='little':vals.byteswap()
 assert all(v&0x7f80!=0x7f80 for v in vals)
 ref=out/'numeric'/f'expected-{row["arm"]*8+row["slot"]:02d}.raw';assert b==ref.read_bytes()
 assert sha(ref)==hist[(0,8,16,24,32,40,48,56)[row['slot']]]['gate']['raw_sha256']
 rawmap[str(p)]=sha(p)
def build_proof(out):
 n=read(out/'native-built.json');assert n['status']=='OK' and n['enabled'] and n['independent_allocation_proof']
 assert (n['capture_dispatches'],n['commands_in_single_ICB'],n['output_allocations'],n['unique_registered_RO_ranges'],n['resolved_W_scale_X_bindings'],n['maxKernelBufferBindCount_all_recorded_bindings'])==(16,16,16,24,48,7)
 assert (n['arena_bytes'],n['snapshot_bytes'],n['zero_copy_buffer_bytes'])==(786432,786432,401162240)
 assert n['arena_initial_restore_once'] and not n['legacy_feedback_enabled'] and not n['legacy_ICB_enabled'] and n['internal_ICB_barriers']==0
 readonly=set();outputs=set();addrreads=set();addrouts=set()
 for i,q in enumerate(n['slot_records']):
  assert (q['arm'],q['slot'])==(i//8,i%8)
  args=q['actual_active_arguments'];w=[a for a in args if a['access']==1];assert len(w)==1 and w[0]['source_store_only_output_exception']
  y=w[0];assert y['original_offset']==0 and y['allocation_bytes']==49152 and not y['zero_copy'];assert y['original_buffer'] not in outputs;outputs.add(y['original_buffer']);addrouts.add(y['original_binding_address'])
  for a in args:
   if a['kind']=='actual_MTLBuffer' and a['access']==0:assert a['zero_copy'];readonly.add(a['original_buffer']);addrreads.add(a['original_binding_address'])
 assert len(readonly)==24 and len(outputs)==16 and not readonly&outputs and not addrreads&addrouts
 return {'actual_ICB_commands':16,'input_RO_allocations':24,'output_exclusive_allocations':16,'RW_output_source_exceptions':16,'maxBind':7,'zero_copy_bytes':n['zero_copy_buffer_bytes'],'arena_snapshot_bytes':786432}
for name in ['fullmodel-v72num2-attempt1','kernel-micro-v72b2-attempt1']:
 out=R/name/'child-output';r=receipts[name];assert (r['MLX_strict_checks'],r['native_strict_checks'],r['weights_payload_bytes'])==(72,128,401080320)
 c=read(out/'numeric-checks.json');assert len(c)==72 and all(x['strict'] and x['A_sha256']==x['B_sha256'] and x['bytes']==34816 for x in c)
 native=read(out/'native128-checks.json');assert len(native)==128
 for row in native:check_raw(out,row)
 assert {x['raw_file'] for x in native}=={f'numeric/qual-g{g}-a{a}-slot{s}.raw' for g in (1,2,3,4,5,7,8,16) for a in (0,1) for s in range(8)} and r['readonly_after_native128']['arrays']==56 and r['readonly_after_native128']['all_sha256_unchanged']
 assert r['compiled_trace_counts']=={'A':[1]*8,'B':[1]*8} and r['MLX72_input_arrays_readonly']==40
 selected[name].update(MLX_same_hash_checks=72,native_raw_checks_replayed=128,actual_allocation_proof=build_proof(out),readonly_saved_assertions={'inputs':40,'inputs_plus_original_outputs':56})
num=receipts['fullmodel-v72num2-attempt1'];assert num['status']=='PASS_V72_72_MLX_128_NATIVE_STRICT_NO_TIMING' and not num['timing_run']
root=R/'kernel-micro-v72b2-attempt1';out=root/'child-output';r=receipts[root.name];ps=read(out/'pairs.json');assert len(ps)==10
cmd=read(root/'owner-receipt.json')['command'];assert cmd[cmd.index('--qualified-sha')+1]==expected['fullmodel-v72num2-attempt1'] and cmd[cmd.index('--qualified-receipt')+1]==str(R/'fullmodel-v72num2-attempt1/child-output/receipt.json')
assert [x['order'] for x in ps]==[['A','A'],['B','B']]+[['A','B'],['B','A']]*3+[['A','A'],['B','B']]
times={'A':[],'B':[]};controls={'A':[],'B':[]};wins=0;block=0
for pair in ps:
 assert pair['readonly']['arrays']==56 and pair['readonly']['all_sha256_unchanged']
 vals={}
 for b in pair['blocks']:
  block+=1;n=b['native'];assert (b['prime_groups'],b['timed_groups'],b['encoded_dispatches'])==(128,1024,8192)
  assert b['wall_ns']/8192==b['ns_per_dispatch'];assert n['status']=='OK' and n['last_CB_completed'] and n['CBGPU_timestamps_valid']
  assert (n['last_groups'],n['observed_encoded_dispatches'],n['buffer_barriers_per_last_loop'],n['loops'])==(1024,8192,1024,16+2*block)
  assert n['total_groups']==92+1152*block and n['total_encoded_dispatches']==8*n['total_groups']
  assert 0<n['CBGPUEndTime_seconds']-n['CBGPUStartTime_seconds']<=n['last_native_wall_ns']/1e9 and n['last_native_wall_ns']<=b['wall_ns']
  assert len(b['outputs'])==8
  for row in b['outputs']:check_raw(out,row)
  vals[b['arm']]=b['ns_per_dispatch']
 if pair['kind']=='control':
  ratio=pair['blocks'][1]['wall_ns']/pair['blocks'][0]['wall_ns'];assert ratio==pair['slot2_div_slot1'];controls[pair['order'][0]].append(ratio)
 else:
  reduction=1-vals['B']/vals['A'];assert math.isclose(reduction,pair['reduction'],abs_tol=1e-15);wins+=reduction>0
  for arm in times:times[arm].append(vals[arm])
med={k:statistics.median(v) for k,v in times.items()};gain=1-med['B']/med['A'];assert med==r['medians_ns_per_dispatch'] and gain==r['median_reduction'] and wins==r['wins']==0
assert controls==r['controls'] and r['control_passed'] and all(all(.95<=x<=1.05 for x in xs) and .98<=statistics.median(xs)<=1.02 for xs in controls.values())
assert r['status']=='STOP_MICRO_GATE' and not r['adopted'] and r['no_Python_retrace']
selected[root.name].update(medians_ns_per_dispatch=med,median_reduction=gain,wins=wins,controls=controls,timed_final_slot_raw_checks_replayed=160,completed_loops=56,groups=23132,encoded_dispatches=185056,qualification_binding_owner_argv=True,per_dispatch_state_validation=False,snapshot_scope='Only final8 outputslots after each completed loop. Each group8 has barrier; repeated8192 dispatches are not individually value-checked.')
gold=read(R/'fullmodel-v44native-attempt1/child-output/receipt.json');gold={q['name']:q for q in gold['requests']};old38=read(R/'fullmodel-v38native-attempt1/child-output/receipt.json');old38={q['name']:q for q in old38['requests']}
def cold(r):
 c=r['compiled_cold'];assert c['trace_count']==1 and c['inputs']==131 and c['outputs']==132 and c['seed_originals_readonly']['strict_bytes_equal']
 assert len(c['outputs_checked'])==132
 for a,g in zip(c['outputs_checked'],gold['base']['steps'][0]['outputs']):
  v=a['actual'];assert a['strict_raw_SHA_match'] and v['sha256']==a['expected_actual44']['A_sha256']==g['A_sha256'] and v['shape']==g['shape'] and v['dtype']==g['dtype'] and v['nbytes']==g['bytes']
 e=r['B_compiled_step_capture_engagement'];assert e['cold_trace_count']==1 and e['capture_trace_delta']==0 and e['cold_GDN_python_calls']==48 and e['capture_GDN_python_calls']==0 and e['trace_observer_detached_before_native_replay'] and e['GDN_observer_removed']
 assert r['compiled_trace_count_after_native_replay']==1
r=receipts['fullmodel-v74native-attempt1'];cold(r);assert r['native_steps']==93 and r['strict_output_checks']==12276 and r['additional_cold_hash_checks']==132 and r['same_parameter_identity_preserved']
for i,q in enumerate(r['requests'],1):
 assert q['A_ids']==q['B_ids']==gold[q['name']]['A_ids'] and len(q['steps'])==31
 for j,e in enumerate(q['steps']):
  assert len(e['outputs'])==132 and len({x['label'] for x in e['outputs']})==132
  assert e['reference_input_digest_readonly'] and e['seed_uploads_this_step']==0 and e['native_prediction_copied_bytes']==4
  for ro in e['originals_readonly'].values():assert ro['strict_bytes_equal']
  for a,g,h in zip(e['outputs'],gold[q['name']]['steps'][j]['outputs'],old38[q['name']]['steps'][j]['outputs']):
   assert a['strict_bytes_equal'] and a['finite'] and a['label']==g['label']==h['label'] and a['A_sha256']==g['A_sha256']==h['A_sha256'] and a['bytes']==g['bytes'] and a['shape']==g['shape'] and a['dtype']==g['dtype']
 n=q['native'];f=n['feedback'];assert n['dispatches']==1498 and (f['total_steps'],f['resets'],f['icb']['bank_executions'])==(31*i,i,[16*i,15*i]) and f['pingpong']['eligible_count']==48 and f['pingpong']['staged_link_count']==83
selected['fullmodel-v74native-attempt1'].update(cold_hash_checks_replayed=132,core_saved_oldA_hashes_replayed=12276,dispatches=1498,steps=93,bank_executions=[48,45],no_extra_compile_trace=True)
r=receipts['native-decode-v77-attempt1'];cold(r);assert r['qualification_sha256']==expected['fullmodel-v74native-attempt1'] and r['qualification_release_sha256']==selected['fullmodel-v74native-attempt1']['release_sha256']
assert r['schedule']==['warm']*3+['measure']*3 and r['same1349_parameter_owners_final']
ids=read(P/'workload.json')['continuation_ids'];measured=[]
for i,q in enumerate(r['requests'],1):
 assert q['ids']==ids and q['index']==i and q['phase']==('warm' if i<=3 else 'measure')
 dt=q['t2_ns']-q['t1_ns'];whole=q['t2_ns']-q['t0_ns'];assert q['decode31_ns']==dt and q['whole32_ns']==whole and q['decode_ns_per_step']==dt/31 and q['decode31_tokens_per_s']==31e9/dt and q['whole32_tokens_per_s']==32e9/whole
 assert len(q['final132_hash_checks'])==132
 for a,g in zip(q['final132_hash_checks'],gold['base']['steps'][-1]['outputs']):assert a['label']==g['label'] and a['sha256']==a['qualified_A_sha256']==g['A_sha256'] and a['bytes']==g['bytes']
 for a in q['captured263_original_owners_readonly'].values():assert a['strict_bytes_equal']
 f=q['native']['feedback'];assert q['native']['dispatches']==1498 and (f['total_steps'],f['resets'],f['icb']['bank_executions'])==(31*i,i,[16*i,15*i])
 t=q['native_timing'];assert sha(R/'native-decode-v77-attempt1/child-output'/t['raw_file'])==t['raw_sha256'] and t['current_request_records']==31 and t['total_records']==31*i
 if i>3:measured.append(q)
m=r['measured_summary'];assert m['median_decode_ns_per_step']==statistics.median(x['decode_ns_per_step'] for x in measured) and m['median_decode31_tokens_per_s']==statistics.median(x['decode31_tokens_per_s'] for x in measured) and m['median_whole32_tokens_per_s']==statistics.median(x['whole32_tokens_per_s'] for x in measured)
selected['native-decode-v77-attempt1'].update(cold_hash_checks_replayed=132,final_hash_checks_replayed=792,dispatches=1498,steps=186,bank_executions=[96,90],summary=m,clock_scope='rawnative t1 after131seeduploads; no paired comparison or HTTP win')
(D/'selected-fields.json').write_text(json.dumps(selected,indent=2)+'\n');(D/'raw-source-map.json').write_text(json.dumps(rawmap,indent=2)+'\n')
print(json.dumps(selected,indent=2))
