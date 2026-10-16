"""CPU-only archived-text/source/statistics verifier. No MLX/native import, live resource probe or GPU.
Default needs this and prior chapter archives. --originals additionally hashes external identities.
"""
from pathlib import Path
import argparse,ast,gzip,hashlib,io,json,re,tarfile,statistics,collections,math
HERE=Path(__file__).resolve().parent;article=HERE.parent/'34-native-ba-gates-prework.md'
ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--originals',action='store_true');ap.add_argument('--report',type=Path);ARGS=ap.parse_args()
filehash=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
J=lambda p:json.loads(Path(p).read_text())
source_rows=J(HERE/'source-map.json');dep_rows=J(HERE/'dependency-map.json')['files'];index=J(HERE/'evidence-index.json');external_rows=J(HERE/'external-fixture-identities.json')['files'];opened={};virtual={};original_checks=set()
def archive_bytes(location):
 fn,sep,member=location.partition('::');p=(HERE/fn).resolve()
 if not sep:return p.read_bytes()
 if p not in opened:opened[p]=tarfile.open(p,'r:gz')
 f=opened[p].extractfile(member);assert f is not None;return f.read()
for row in source_rows:
 b=archive_bytes(row['archive']);assert len(b)==row['bytes'] and hashlib.sha256(b).hexdigest()==row['sha256'];b.decode('utf-8');virtual[row['source']]=b
assert filehash(HERE/index['archive'])==index['sha256']
assert (HERE/index['archive']).read_bytes()[:10]==bytes.fromhex('1f8b08000000000000ff')
t=tarfile.open(HERE/index['archive'],'r:gz');opened[(HERE/index['archive']).resolve()]=t
members=t.getmembers();assert len(members)==len(index['files'])==127
runs={'numeric-v76-attempt1','fullmodel-v82native-attempt1','native-decode-v86-attempt1'}
assert {row['archive'].split('/')[0] for row in index['files']}==runs
rebuilt=io.BytesIO()
with gzip.GzipFile(filename='',fileobj=rebuilt,mode='wb',mtime=0,compresslevel=6) as gz:
 with tarfile.open(fileobj=gz,mode='w|') as tf:
  for member,row in zip(members,index['files']):
   assert member.isfile() and member.name==row['archive'] and member.mtime==member.uid==member.gid==0 and member.mode==0o644
   assert not Path(member.name).is_absolute() and '..' not in Path(member.name).parts and Path(member.name).suffix in {'.json','.log','.md','.txt'}
   b=t.extractfile(member).read();b.decode('utf-8');assert len(b)==row['bytes'] and hashlib.sha256(b).hexdigest()==row['sha256'];virtual[row['source']]=b
   ti=tarfile.TarInfo(member.name);ti.size=len(b);ti.mtime=0;ti.mode=0o644;tf.addfile(ti,io.BytesIO(b))
assert hashlib.sha256(rebuilt.getvalue()).hexdigest()==index['sha256']
external=[]
for row in dep_rows:
 if row.get('location'):
  b=archive_bytes(row['location']);assert len(b)==row['bytes'] and hashlib.sha256(b).hexdigest()==row['sha256'];b.decode('utf-8');virtual.setdefault(row['source'],b)
 else:external.append(row)
 if ARGS.originals:
  p=Path(row['source']);assert p.stat().st_size==row['bytes'] and filehash(p)==row['sha256'];original_checks.add(str(p))
assert len(external)==len(external_rows)==125 and all(not x['payload_archived'] for x in external_rows)
for row in source_rows+index['files']:
 if ARGS.originals:
  p=Path(row['source']);assert p.read_bytes()==virtual[row['source']];original_checks.add(str(p))
def payload(p):
 key=str(p);assert key in virtual,('unmapped text',key);return virtual[key]
def read(p):return json.loads(payload(p))
def sha(p):
 key=str(p)
 if key in virtual:return hashlib.sha256(virtual[key]).hexdigest()
 row=next(x for x in dep_rows if x['source']==key);assert not row.get('location');return row['sha256'] # external identity only; --originals was checked above
P=Path('/private/tmp/minivllm-engine-optimization-20261014');R=P/'v23-root-control-build1'
D=P/'v82-native-fullqual-build1';R=P/'v23-root-control-build1/fullmodel-v82native-attempt1';O=R/'child-output';j=read;h=sha
r=j(O/'receipt.json');assert h(O/'receipt.json')=='6553eae93b7f79bed92753749cb04b156bd4343db422ec88043a75c8072265af'
assert r['status']=='PASS_V82_BA_FP32_GATES_PREWORK_GDN_FULL_ICB_BASE_RESET_CHANGED_STRICT' and r['model_loads']==1 and r['stage']=='complete' and r['performance_point'] is False and r['adopted'] is False
pins=j(D/'source-pins.json');assert pins==r['source_pins']
for p,s in pins.items():assert h(p)==s,('pin',p)
m=j(D/'candidate-manifest.json');root=Path(m['source_root']);root=root.parent.parent if root.name=='miniengine' else root;expected={k.removeprefix('tutorial_code/miniengine/'):v for k,v in m['files'].items()};actual={str(Path(key).relative_to(root/'tutorial_code/miniengine')):sha(key) for key in virtual if key.startswith(str(root/'tutorial_code/miniengine')+'/')};assert actual==expected and sum(k.endswith('.py') for k in actual)==29
G=[]
for version,gold_sha in [('38','c57e7625baa5a4ce450fc7306233e766f1efd71636713080d748780624542c8a'),('44','1dcfc41e7779c575602bd6c73bae4e4800664db42e1ee87e75414335bdc45b6d')]:
 f=P/f'v23-root-control-build1/fullmodel-v{version}native-attempt1/child-output/receipt.json';assert h(f)==gold_sha;G.append({x['name']:x for x in j(f)['requests']})
count=0
for req in r['requests']:
 assert req['status']=='PASS_REQUEST_STRICT' and len(req['steps'])==31 and len(req['A_ids'])==32 and req['A_ids']==req['B_ids']
 assert req['seed_upload_calls']==req['seed_arrays']==131
 for g in G:assert req['prompt_ids']==g[req['name']]['prompt_ids'] and req['A_ids']==g[req['name']]['A_ids']
 for step in req['steps']:
  assert len(step['outputs'])==132 and step['B_fixed_L']==256 and step['A_token']==step['B_token']
  assert step['native_prediction_copied_bytes']==4 and step['seed_uploads_this_step']==0 and step['A_saved_V38_hash_checks']==132 and step['reference_input_digest_readonly']
  for role,arrays in [('seed',131),('capture_outputs',132),('request_seed',131)]:assert step['originals_readonly'][role]['strict_bytes_equal'] and step['originals_readonly'][role]['arrays']==arrays
  for g in G:
   prior=g[req['name']]['steps'][step['step']-1];assert prior['A_exact_L']==step['A_exact_L'] and prior['A_token']==step['A_token']
   assert len(prior['outputs'])==132
   for a,b in zip(step['outputs'],prior['outputs']):
    assert all(a[k]==b[k] for k in ['label','shape','dtype','bytes','A_sha256'])
    assert a['finite'] and a['strict_bytes_equal']
  count+=132
assert len(r['requests'])==3 and count==r['strict_output_checks']==12276 and r['native_steps']==93
assert r['requests'][0]['prompt_length']==r['requests'][1]['prompt_length']==19 and r['requests'][2]['prompt_length']==20 and len(r['requests'][2]['changed_seed_inputs_vs_capture'])==129
p=r['ba_gates_prework_preparation'];assert p['status']=='READY_ACTUAL_LAYOUT_PREPARED' and len(p['layout_rows'])==384 and len(p['prepared'])==48 and len(p['original288_target_arrays'])==288
assert p['readonly_after_prepare'] and p['target_payload_bytes']==13289472 and p['extra_packed_weight_bytes']==0 and p['parameter_owners']==1349 and p['same_parameter_identity_after_prepare'] and p['target_parameter_holders_retained'] and p['original_parameter_ranges_disjoint']
for row in p['layout_rows']:
 assert row['row_contiguous_derived_singleton_aware'] and row['internal_MLX_flags_observed'] is False and row['ensure_row_contiguous'] is False
 expected_stride=1
 for dim,stride in zip(reversed(row['shape']),reversed(row['actual_strides_elements'])):
  assert dim==1 or stride==expected_stride;expected_stride*=dim
for row in p['prepared']:assert row['same_original_handles'] and row['X_activated_readonly'] and row['mlx_active_before_prepare']==row['mlx_active_after_prepare'] and len(row['original_operand_ids'])==6
ranges=[(v['address'],v['address']+v['nbytes']) for v in p['original288_target_arrays']];assert all(b<=c or d<=a for i,(a,b) in enumerate(ranges) for c,d in ranges[i+1:])
eng=r['B_ba_gates_prework_capture_engagement'];assert eng['calls']==eng['distinct_GDN_objects']==48 and eng['all48_enabled'] and eng['observers_removed_before_replay']
assert len(eng['prework_calls'])==48 and {x['helper_id'] for x in eng['prework_calls']}=={x['helper_id'] for x in p['prepared']}
assert all(x['outputs']==6 and x['output_shapes']==[[1,16,128],[1,16,128],[1,48,128],[1,16,2],[1,48],[1,48]] and x['output_dtypes']==['mlx.core.float32']*6 for x in eng['prework_calls'])
bind=r['ba_gates_prework_native_binding_proof'];assert len(bind['original_target_ranges'])==288 and bind['Python_capture_calls']==48 and bind['buffer_argument_count_source_proof']==14 and 'NOT_OBSERVED' in bind['individual_argument_reflection'] and 'NOT_OBSERVED' in r['actual_individual_argument_reflection']
param={v['label']:v for v in p['original288_target_arrays']}
for row in bind['original_target_ranges']:
 before=param[row['parameter']];assert row['original_object_id']==before['original_object_id'] and row['address']==before['address'] and row['bytes']==before['nbytes'] and row['full_original_parameter_range_readonly']
 matches=[v for v in r['capture']['native']['buffers'] if v['original_buffer']==row['actual_original_buffer']];assert len(matches)==1
 b=matches[0];assert b['used'] and b['zero_copy'] and b['original_contents']<=row['address'] and row['address']+row['bytes']<=b['original_contents']+b['length'] and 'gdn_activated_BA_gates_prework_v76_build1' in b['role'] and b['role']==row['role']
assert r['capture']['parameter_arrays']==r['same_parameter_identity_count']==1349 and r['same_parameter_identity_preserved'] and r['ba_gates_target288_readonly_final']
assert len(r['capture']['inputs'])==131 and len(r['capture']['outputs'])==132
assert r['A_baseline_restore_final'] and len(r['A_all_features_OFF_legacy_dispatch_observations'])==r['A_baseline_temporary_bind_calls']==129
assert all(x['BA_conv_gates_all_OFF']==48 and x['legacy_dispatch_observed'] and x['all_flags_restored'] for x in r['A_all_features_OFF_legacy_dispatch_observations'])
for i,req in enumerate(r['requests'],1):
 n=req['native'];f=n['feedback'];icb=f['icb'];pp=f['pingpong'];assert n['status']=='OK' and n['dispatches']==1433
 assert f['total_steps']==icb['execution_count']==icb['encoded_count']==31*i and f['resets']==i and f['request_seed_uploaded_ranges']==131 and f['CPU_full_state_restore_per_step'] is False
 assert (pp['requested_count'],pp['eligible_count'],pp['staged_link_count'])==(48,48,83) and icb['banks']==2 and icb['commands_per_bank']==1433 and icb['bank_executions']==[16*i,15*i]
 assert f['input_count']==131 and f['output_count']==132 and f['state_count']==128 and f['capacity']==256
source=payload(D/'fullqual.py').decode();ast.parse(source)
assert 'finally:\n                GDN._decode_ba_gates_prework=candidate_ba;PreparedBA.prework=original_prework' in source
assert source.index('seed_baselines = [h.raw()')<source.index('native.call("v33_begin")') and source.index('output_baselines=[h.raw()')<source.index('native.call("v34_replay_step")')
owner=j(R/'owner-receipt.json');release=j(R/'external-release.json');assert owner['script_sha256']==h(D/'fullqual.py') and owner['waited_reaped'] and owner['returncode']==0 and owner['stage']=='reaped' and owner['child_pid']==r['pid'] and owner['absence']['returncode']==1 and not owner['absence']['stdout']
assert release['status']=='PASS_EXTERNAL_RELEASE' and release['GPU_released'] and release['tool_exit']==0 and {x['pid'] for x in release['strict_absence']}=={owner['owner_pid'],owner['child_pid']} and all(x['returncode']==1 and not x['stdout'] for x in release['strict_absence'])
raws=[j(f) for f in sorted(Path(key) for key in virtual if key.startswith(str(O)+'/resource-') and key.endswith('-raw.json'))];assert len(raws)==34
for x in raws:assert x['power_metadata']['source']=='AC Power' and x['processinfo_thermal_metadata']['code']==0 and x['pressure_level_value']==1 and x['swap_used_mb']==raws[0]['swap_used_mb'] and x['vm_counters']['Swapouts']==raws[0]['vm_counters']['Swapouts']
summary={'status':'PASS_CPU_INDEPENDENT_V82_SAVED_NUMERIC_SOURCE_LIFECYCLE_REVIEW','receipt_sha256':h(O/'receipt.json'),'fullqual_sha256':h(D/'fullqual.py'),'pins_checked':len(pins),'oldgold_receipts':2,'strict_output_checks':count,'steps':93,'requests':3,'all_IDs':96,'layout_rows':384,'prepared_helpers':48,'target_parameter_RO_ranges':288,'parameter_owner_identity':1349,'actual_capture_BA_calls':48,'actual_capture_prework_calls':48,'observers_removed_before_replay':True,'dispatches':1433,'S_eligible':48,'staged_links':83,'banks':[48,45],'declared_buffer_arguments':14,'actual_individual_argument_reflection':'NOT_OBSERVED','saved_child_resource_samples':len(raws),'owner_pid':owner['owner_pid'],'child_pid':owner['child_pid'],'external_release_sha256':h(R/'external-release.json'),'speedup_claim':False,'GPU_executed':False,'MLX_imported':False}
summary82=summary
R=P/'v23-root-control-build1'

num=read(R/'numeric-v76-attempt1/child-output/receipt.json')
assert num['status']=='PASS_V76_BA_FP32_GATES_PREWORK_LOCAL_STRICT_45CASES_720CHECKS' and len(num['cases'])==45 and len(num['checks'])==num['strict_output_checks']==720
assert num['model_loaded'] is False and num['engine_integrated'] is False and num['timing_run'] is False and num['performance_claim'] is False and num['fullmodel_qualified'] is False and num['readonly_all71'] and num['actual_compiled_gate_reference']
for p,s in num['source_pins'].items():assert next(x for x in dep_rows if x['source']==p)['sha256']==s
assert all(c['historical_saved_bank_not_new_capture'] and c['not_a_matched_model_activation_tuple'] for c in num['cases'])
assert collections.Counter(c['layer'] for c in num['cases'])=={0:15,32:15,62:15}
counts=collections.Counter('/'.join(c['label'].split('/')[:3]) for c in num['checks']);assert len(counts)==45 and set(counts.values())=={16}
for c in num['checks']:
 assert c['strict_bytes_equal'] and c['A']['finite'] and c['B']['finite']
 assert all(c['A'][k]==c['B'][k] for k in ['shape','dtype','bytes','sha256'])
assert len(num['readonly'])==3 and all(c['all71_originals_unchanged'] for c in num['readonly'])
proof=read(P/'v76-gdn-gates-prework-build1/source-proof.json')
assert proof['production_input_count']==8 and proof['production_output_count']==6 and proof['declared_production_buffer_count']==14 and proof['declared_diagnostic_buffer_count']==16 and proof['actual_Metal_argument_reflection_observed'] is False
assert proof['state_SOURCE_byte_exact_to44'] and proof['state_HEADER_byte_exact_to44'] and proof['qmv_header_exact_to66build2'] and proof['QK_branch_bytes_exact_to44'] and proof['S_input_index']==6
helper=payload(P/'v76-gdn-gates-prework-build1/gdn_gates_prework_metal.py').decode()
for piece in ['const bfloat16_t bv = static_cast<bfloat16_t>(result_b);','const bfloat16_t av = static_cast<bfloat16_t>(result_a);','V76NativeLogAddExp()(apd, 0.0f)','ensure_row_contiguous=False']:
 assert piece in helper
assert num['donor_scope'].startswith('actual oMLX Q35 prework keeps gates/recurrence separate')

screen=read(R/'native-decode-v86-attempt1/child-output/receipt.json');assert sha(R/'native-decode-v86-attempt1/child-output/receipt.json')=='4475314b5937ee60d9261b3c6475c17577255eeb79a8b2e4163e4262bd4c54ba'
assert screen['status']=='PASS_V86_BA_GATES_PREWORK_GDN_NATIVE_B_ONLY_EXPLORATORY_SCREEN' and screen['model_loads']==1 and screen['stage']=='complete'
assert screen['qualification_sha256']==summary82['receipt_sha256'] and screen['qualification_release_sha256']==summary82['external_release_sha256'] and screen['schedule']==['warm']*3+['measure']*3
for p,s in screen['source_pins'].items():assert next(x for x in dep_rows if x['source']==p)['sha256']==s
assert 'NOT_OBSERVED' in screen['individual_argument_reflection'] and 'after prefill' in screen['timing_definition']['t1'] and 'all131 seed uploads' in screen['timing_definition']['t1']
assert screen['target288_original_parameters_readonly_final'] and screen['same1349_parameter_owners_final']
eng=screen['B_ba_gates_prework_capture_engagement'];assert eng['calls']==eng['distinct_GDN_objects']==48 and len(eng['prework_calls'])==48 and eng['observers_removed_before_replay']
prep=screen['ba_gates_prework_preparation'];assert len(prep['prepared'])==48 and len(prep['layout_rows'])==384 and len(prep['original288_target_arrays'])==288 and prep['readonly_after_prepare'] and prep['parameter_owners']==1349 and prep['extra_packed_weight_bytes']==0
assert screen['captured263_original_owners_readonly_baselines']['seed_arrays']==131 and screen['captured263_original_owners_readonly_baselines']['captured_outputs']==132 and screen['captured263_original_owners_readonly_baselines']['bytes']==341939224
base=read(P/'v23-root-control-build1/fullmodel-v44native-attempt1/child-output/receipt.json')['requests'][0]
final=base['steps'][-1]['outputs'];terminal=0
for i,req in enumerate(screen['requests'],1):
 assert req['index']==i and req['phase']==('warm' if i<=3 else 'measure') and req['ids']==base['A_ids'] and len(req['ids'])==32
 assert req['t0_ns']<req['t1_ns']<req['t2_ns']
 assert req['prefill_seed_ready_ns']==req['t1_ns']-req['t0_ns'] and req['decode31_ns']==req['t2_ns']-req['t1_ns'] and req['whole32_ns']==req['t2_ns']-req['t0_ns']
 assert req['decode_ns_per_step']==req['decode31_ns']/31 and req['decode31_tokens_per_s']==31e9/req['decode31_ns'] and req['whole32_tokens_per_s']==32e9/req['whole32_ns']
 assert len(req['final132_hash_checks'])==132
 for a,b in zip(req['final132_hash_checks'],final):assert a['label']==b['label'] and a['bytes']==b['bytes'] and a['sha256']==a['qualified_A_sha256']==b['A_sha256']
 terminal+=132;fb=req['native']['feedback'];icb=fb['icb'];pp=fb['pingpong']
 assert req['native']['dispatches']==1433 and fb['total_steps']==icb['execution_count']==31*i and fb['resets']==i and icb['bank_executions']==[16*i,15*i] and icb['banks']==2
 assert (pp['requested_count'],pp['eligible_count'],pp['staged_link_count'])==(48,48,83) and fb['CPU_full_state_restore_per_step'] is False and fb['request_seed_uploaded_ranges']==131
 for k,n in [('capture_seed',131),('captured_reference',132)]:assert req['captured263_original_owners_readonly'][k]['arrays']==n and req['captured263_original_owners_readonly'][k]['strict_bytes_equal']
assert len(screen['requests'])==6 and terminal==792
measured=[x for x in screen['requests'] if x['phase']=='measure'];actual={k:statistics.median(x[k] for x in measured) for k in ['decode_ns_per_step','decode31_tokens_per_s','whole32_tokens_per_s']}
for k,v in actual.items():assert screen['measured_summary']['median_'+k]==v
assert screen['measured_summary']['n']==3

life={};resource_count=0
for run in sorted(runs):
 out=R/run;owner=read(out/'owner-receipt.json');release=read(out/'external-release.json');receipt=read(out/'child-output/receipt.json')
 assert owner['stage']=='reaped' and owner['returncode']==0 and owner['waited_reaped'] and owner['absence']['returncode']==1 and not owner['absence']['stdout']
 pid=owner.get('child_pid',owner.get('driver_pid'));assert pid==receipt['pid']
 script=next(p for p in owner['command'] if p.endswith('.py'));assert sha(script)==owner.get('script_sha256',owner.get('driver_sha256'))
 assert release['status']=='PASS_EXTERNAL_RELEASE' and release['GPU_released'] and release['tool_exit']==0 and {x['pid'] for x in release['strict_absence']}=={owner['owner_pid'],pid}
 assert all(x['returncode']==1 and x['stdout']==x['stderr']=='' for x in release['strict_absence'])
 rawpaths=sorted(k for k in virtual if k.startswith(str(out/'child-output')+'/resource-') and k.endswith('-raw.json'));raws=[read(k) for k in rawpaths]
 for x in raws:assert x['power_metadata']['source']=='AC Power' and x['processinfo_thermal_metadata']['code']==0 and x['pressure_level_value']==1 and x['swap_used_mb']==raws[0]['swap_used_mb'] and x['vm_counters']['Swapouts']==raws[0]['vm_counters']['Swapouts']
 resource_count+=len(raws);life[run]={'receipt_sha256':sha(out/'child-output/receipt.json'),'release_sha256':sha(out/'external-release.json'),'owner':owner['owner_pid'],'child':pid,'saved_resource_samples':len(raws)}
# Compare source AST clock boundaries with the original screen; never run the model.
stree=ast.parse(payload(P/'v86-ba-gates-native-screen-build1/screen.py').decode())
loops=[n for n in ast.walk(stree) if isinstance(n,ast.For) and any(isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='t0' for t in x.targets) for x in n.body)]
assert len(loops)==1
# Actual source excludes fulloutput checks, resources and native reports between t1/t2.
body=loops[0].body;clock=lambda n,target:isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==target for t in n.targets)
a=next(i for i,n in enumerate(body) if clock(n,'t1'));b=next(i for i,n in enumerate(body) if clock(n,'t2'))
inside=ast.unparse(ast.Module(body=body[a+1:b],type_ignores=[]))
assert all(s not in inside for s in ['model(', 'write(', 'hashlib.', '.report(', '.sample('])
links=0
for doc in [article,HERE/'README.md']:
 for target in re.findall(r'(?<!!)\[[^\]]+\]\(([^)]+)\)',doc.read_text()):
  if '://' in target:continue
  p=(doc.parent/target).resolve()
  if not p.exists():assert p.name in ['verification.json','verification-originals.json','manifest.json'] and ARGS.report
  links+=1
manifest=HERE/'manifest.json'
if manifest.exists():
 m=J(manifest);assert m['chapter_sha256']==filehash(article)
 actual_files={str(p.relative_to(HERE)):filehash(p) for p in sorted(HERE.rglob('*')) if p.is_file() and p.name!='manifest.json'};assert m['files']==actual_files
summary={'status':'PASS_CPU_ARCHIVE_TEXT_HASH_STATS_SAVED_V76_V82_V86','article_sha256':filehash(article),'mode':'originals_identity_plus_saved_text' if ARGS.originals else 'archive_only_text_hash_statistics','archive':{'source_rows':len(source_rows),'dependency_bindings':len(dep_rows),'external_identity_only':len(external_rows),'evidence_text_members':len(index['files']),'evidence_gzip_sha256':index['sha256'],'deterministic_gzip_reconstruction':True,'original_paths_checked':len(original_checks),'tar_member_accessor_supported':True,'compressed33_gold_member_references':sum('33-native-range-compiled-step-source/historical-gold-dependencies.tar.gz::' in x.get('location','') for x in dep_rows),'tensor_payload_archived':False},'V76':{'cases':45,'saved_strict_checks':720,'readonly_original_arrays':71,'matched_fullmodel_activation_tuples':False,'production_declared_buffers':14,'actual_argument_reflection_observed':False},'V82':summary82,'V86':{'requests':6,'warm':3,'measure':3,'saved_terminal_hash_checks':792,'decode_steps':186,'resets':6,'banks':[96,90],'measured':actual,'clock':'t1 after seed uploads; raw native B-only, no HTTP/fair paired/adoption claim'},'saved_resource_samples':resource_count,'lifecycle':life,'validated_links':links,'MLX_imported':False,'native_loaded':False,'GPU_executed':False,'readonly_scope':'Saved runtime assertions against original once-only backups; no dead-process state recreation; --originals validates fixture identities, not new arithmetic runs.'}
if ARGS.report:ARGS.report.write_text(json.dumps(summary,ensure_ascii=False,sort_keys=True,indent=2)+'\n')
print(json.dumps({'status':summary['status'],'mode':summary['mode'],'V76checks':720,'V82checks':12276,'V86terminal':terminal,'text_members':len(index['files']),'original_paths_checked':len(original_checks),'no_speedup_or_HTTP_win':True}))
