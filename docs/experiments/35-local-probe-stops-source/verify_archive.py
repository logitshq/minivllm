"""CPU-only archived-text/source/statistics verifier. No MLX/native import, live resource probe or GPU.
Default needs this and prior chapter archives. --originals additionally hashes external identities.
"""
from pathlib import Path
import argparse,ast,gzip,hashlib,io,json,re,tarfile,statistics,collections,math
HERE=Path(__file__).resolve().parent;article=HERE.parent/'35-local-probe-stops.md'
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
members=t.getmembers();assert len(members)==len(index['files'])==278
runs={'numeric-v78-attempt1','numeric-v79-attempt1','fullmodel-v80num-attempt1','kernel-micro-v80-attempt1','fullmodel-v81num-attempt1','numeric-v83-attempt1','fullmodel-v84num-attempt1','kernel-micro-v84-attempt1','kernel-micro-v85-attempt1'}
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
assert len(external)==len(external_rows)==105 and all(not x['payload_archived'] for x in external_rows)
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

import array,sys
fixture_rows=J(HERE/'terminal-fixture-index.json')['files'];fixture={x['source']:x for x in fixture_rows};assert len(fixture)==len(fixture_rows)==1076 and all(not x['payload_archived'] for x in fixture_rows)
if ARGS.originals:
 for row in fixture_rows:
  p=Path(row['source']);assert p.stat().st_size==row['bytes'] and filehash(p)==row['sha256'];original_checks.add(str(p))
expected={'numeric-v78-attempt1':'f5d8fc89eb16ad29f0efed700b1a6a8200a975fb145b2db99a12b3632ec7807a','numeric-v79-attempt1':'3c8ef7e23a136443e8dfb2252002b4940d1ec5304bda32a778013dbf13335dff','fullmodel-v80num-attempt1':'76dd72927227ea1cd13647c97fbffa68264380407b813f2faa9168ba022fb06c','kernel-micro-v80-attempt1':'63c731a2be642da1dd4c5fdfafe8495961a57c78d4b3d307abac8fb75a82b18b','fullmodel-v81num-attempt1':'bd8e1b359462f210a5f539f0922a571b4f101eba730748b3a838b771f0f42f26','numeric-v83-attempt1':'3c8ef7e23a136443e8dfb2252002b4940d1ec5304bda32a778013dbf13335dff','fullmodel-v84num-attempt1':'4c816b0dab6b96f4ca096e47a96a81e72d5d182d96cd41275eddd75542607d10','kernel-micro-v84-attempt1':'13d32f4ec4e69fc000370a518ad13b8b0c331f78168c0ea50120de2bfed1cfbd','kernel-micro-v85-attempt1':'83adff180b95e13e1aa9ba006399d7ab37e7a2d135c40abf8e4eba0cd7d6c083'}
receipts={};selected={};finite_seen=set();used_fixtures=set();resource_count=0
def bindpins(pins):
 for p,digest in pins.items():
  row=next(x for x in dep_rows if x['source']==p);assert row['sha256']==digest
  if p in virtual:assert sha(p)==digest

def terminal(path,reference,digest,nbytes,dtype):
 path=Path(path);reference=Path(reference);assert str(path) in fixture and str(reference) in fixture
 assert fixture[str(path)]['sha256']==fixture[str(reference)]['sha256']==digest and fixture[str(path)]['bytes']==fixture[str(reference)]['bytes']==nbytes
 used_fixtures.update([str(path),str(reference)])
 if ARGS.originals:
  a=path.read_bytes();assert a==reference.read_bytes()
  if digest not in finite_seen:
   vals=array.array('H' if dtype=='bfloat16' else 'f');vals.frombytes(a)
   if sys.byteorder!='little':vals.byteswap()
   assert all((v&0x7f80)!=0x7f80 for v in vals) if dtype=='bfloat16' else all(math.isfinite(v) for v in vals)
   finite_seen.add(digest)
for run in sorted(runs):
 out=R/run;rf=out/'child-output'/('result.json' if run in ['numeric-v79-attempt1','numeric-v83-attempt1'] else 'receipt.json');assert sha(rf)==expected[run];r=read(rf);receipts[run]=r
 bindpins(r.get('source_pins',r.get('pins',{})))
 owner=read(out/'owner-receipt.json');release=read(out/'external-release.json');pid=owner.get('child_pid',owner.get('driver_pid'))
 assert owner['stage']=='reaped' and owner['waited_reaped'] and owner['returncode']==(2 if run in ['kernel-micro-v80-attempt1','kernel-micro-v84-attempt1','kernel-micro-v85-attempt1'] else 0) and owner['absence']['returncode']==1 and not owner['absence']['stdout']
 script=next(p for p in owner['command'] if p.endswith('.py'));assert sha(script)==owner.get('driver_sha256',owner.get('script_sha256'))
 assert release['status']=='PASS_EXTERNAL_RELEASE' and release['GPU_released'] and release['tool_exit']==owner['returncode'] and all(x['returncode']==1 and x['stdout']==x['stderr']=='' for x in release['strict_absence'])
 released={x['pid'] for x in release['strict_absence']};assert {owner['owner_pid'],pid}<=released
 if run in ['numeric-v79-attempt1','numeric-v83-attempt1']:
  child=read(out/'child-output/child.json');clean=read(out/'child-output/cleanup.json');assert child['owner_pid']==pid and clean['pid']==child['pid'] and clean['returncode']==0 and clean['wait_completed'] and clean['exact_pid_absence']['absent'] and child['pid'] in released
 rawpaths=sorted(k for k in virtual if k.startswith(str(out/'child-output')+'/resource-') and k.endswith('-raw.json'));raws=[read(k) for k in rawpaths]
 for x in raws:assert x['power_metadata']['source']=='AC Power' and x['processinfo_thermal_metadata']['code'] in (0,1) and x['pressure_level_value']==1 and x['swap_used_mb']==raws[0]['swap_used_mb'] and x['vm_counters']['Swapouts']==raws[0]['vm_counters']['Swapouts']
 resource_count+=len(raws);selected[run]={'receipt_sha256':sha(rf),'status':r['status'],'owner':owner['owner_pid'],'driver':pid,'waited_returncode':owner['returncode'],'release_sha256':sha(out/'external-release.json'),'saved_resource_samples':len(raws),'thermal_codes':sorted(set(x['processinfo_thermal_metadata']['code'] for x in raws))}

r=receipts['numeric-v78-attempt1'];assert r['status']=='PASS_V78_SCALED_READY_LOCAL_STRICT_9REAL_6SYNTHETIC' and len(r['cases'])==15 and r['strict_output_checks']==r['strict_ready_checks']==r['strict_diagnostic_checks']==30 and r['input_readonly_cases']==15
for c in r['cases']:
 assert c['input_readonly'] and c['input_before']==c['input_after']
 for role in ['ready','diagnostics','outputs']:
  assert len(c[role])==2
  for row in c[role]:assert row['strict_equal'] and all(row['A'][k]==row['B'][k] for k in ['shape','dtype','bytes','sha256'])
selected['numeric-v78-attempt1'].update(ready_checks=30,diagnostic_checks=30,output_checks=30,fixed_S_no_autoregressive_feedback=True)
for version,label,dirname in [(79,'HALF2','v79-nvfp4-half2-decode-build1'),(83,'UNROLL2','v83-nvfp4-kloop-unroll2-build1')]:
 run=f'numeric-v{version}-attempt1';out=R/run/'child-output';r=receipts[run];identity=read(out/'identity.json');qual=read(out/'qualification.json');m=P/dirname
 assert r['status']=='STRICT-NUMERIC-QUALIFICATION-PASS' and r['checks']==36 and r['readonly'] and r['all_bytes_exact'] and r['timing_run'] is False and r['adopted'] is False
 assert identity['driver_sha256']==sha(m/'micro.py') and identity['prototype_manifest_sha256']==sha(m/'source-manifest.json') and identity['protocol_sha256']==sha(m/'protocol.json')
 assert qual['checks']==36 and qual['readonly'] and qual['same_immutable_weight_handles'] and qual['trace_counts_after_cold']==qual['trace_counts_after_qualification']=={'A':1,'B':1} and label in qual['status']
 rows=read(out/'bytes.json');assert len(rows)==36 and all(x['exact'] and x['finite'] and x['differing_bytes']==0 and x['left_sha256']==x['right_sha256'] for x in rows)
 ro=read(out/'readonly.json');assert all(x['before']==x['after'] for x in ro)
 bindpins(read(m/'source-pins.json'))
 selected[run].update(checks=36,driver_sha256=identity['driver_sha256'],prototype_manifest_sha256=identity['prototype_manifest_sha256'],identity_sha256=sha(out/'identity.json'),result_SHA_alone_sufficient=False)
assert selected['numeric-v83-attempt1']['driver_sha256']=='d70788542b11c484032395c58f50f0ed6939498fe49cf17fa8e3a775a5b954ab' and selected['numeric-v83-attempt1']['prototype_manifest_sha256']=='3d250d55e6d6e938cc102f5bd8895ff6f97a3c9ffdcfbc8adec2fe516de69ab9'
head=payload(P/'v83-nvfp4-kloop-unroll2-build1/kernel_header.metal').decode();pragma='#pragma clang loop unroll_count(2)\n';assert head.count(pragma)==1 and head.replace(pragma,'')==payload(P/'v66-gdn-four-projection-build2/kernel_header.metal').decode()

A_fn='nvfp4_qmv_fast_bfloat16_t_gs_16_b_4_batch_0'
B_fns={80:'custom_kernel_v79_nvfp4_half2_r4_sg2_n17408_k5120__bfloat16_t_4_uint32_t_uint8_t_bfloat16_t_bfloat16_t',84:'custom_kernel_v83_nvfp4_unroll2_r4_sg2_n17408_k5120__bfloat16_t_4_uint32_t_uint8_t_bfloat16_t_bfloat16_t'}
for version in [80,84]:
 for perf in [False,True]:
  run=f'kernel-micro-v{version}-attempt1' if perf else f'fullmodel-v{version}num-attempt1';out=R/run/'child-output';r=receipts[run];assert r['MLX_strict_checks']==72 and r['native_strict_checks']==128 and r['MLX72_input_arrays_readonly']==40 and r['readonly_after_native128']['all_sha256_unchanged'] and r['readonly_after_native128']['arrays']==56
  checks=read(out/'numeric-checks.json');assert len(checks)==72 and all(x['strict'] and x['A_sha256']==x['B_sha256'] for x in checks)
  n=read(out/'native-built.json');assert n['status']=='OK' and n['capture_dispatches']==n['commands_in_single_ICB']==16 and n['unique_registered_RO_ranges']==n['readonly_allocations']==24 and n['resolved_W_scale_X_bindings']==48 and n['actual_function_and_schema_matched'] and n['arena_initial_restore_once']
  assert len(n['slot_records'])==16 and all(x['function']==(A_fn if x['arm']==0 else B_fns[version]) for x in n['slot_records'])
  rows=read(out/'native128-checks.json');assert len(rows)==128
  for row in rows:
   assert row['strict'] and row['sha256']==row['expected_sha256'];ref=out/'numeric'/f"expected-{row['arm']*8+row['slot']:02d}.raw";terminal(out/row['raw_file'],ref,row['sha256'],34816,'bfloat16')
  if perf:
   pairs=read(out/'pairs.json')
   for pair in pairs:
    assert pair['readonly']['arrays']==56 and pair['readonly']['all_sha256_unchanged']
    for block in pair['blocks']:
     n=block['native'];assert n['observed_encoded_dispatches']==8192 and n['buffer_barriers_per_last_loop']==1024 and n['last_CB_completed']
     for row in block['outputs']:
      assert row['strict'] and row['sha256']==row['expected_sha256'];ref=out/'numeric'/f"expected-{row['arm']*8+row['slot']:02d}.raw";terminal(out/row['raw_file'],ref,row['sha256'],34816,'bfloat16')
  selected[run].update(MLX_checks=72,native_terminal_checks=128,timed_terminal_checks=160 if perf else 0,per_dispatch_value_checks=False)

for run in ['fullmodel-v81num-attempt1','kernel-micro-v85-attempt1']:
 out=R/run/'child-output';r=receipts[run];assert len(r['cases'])==15 and len(r['public_AB_checks'])==30 and len(r['captured_checks'])==60 and len(r['native_checks'])==480 and r['completed_strict_counts']=={'public_AB_checks':30,'captured_checks':60,'native_checks':480}
 for key in ['public_AB_checks','captured_checks','native_checks']:
  for row in r[key]:assert row['strict_equal'] and row['A_sha256']==row['B_sha256']
 assert len(r['readonly_checks'])==17 and all(x['original_arrays']==255 and x['all_raw_sha256_unchanged'] and x['native_shape_constants_unchanged'] for x in r['readonly_checks'])
 n=r['native_final'];assert n['status']=='OK' and n['capture_dispatches']==n['ICB_commands']==60 and len(n['PSOs'])==4 and len(n['dispatch_records'])==60 and n['producer_consumer_edges_verified'] and n['fixed_saved_banks']==15 and n['internal_state_barriers_per_range']==15 and n['max_buffer_bind_count']==10 and n['arena_restore_once']
 scaled=[x for x in n['dispatch_records'] if x['arm']==1 and x['stage']==1];assert len(scaled)==15 and all(not any(a['index']==3 or a['name']=='scales' for a in x['actual_active_arguments']) for x in scaled)
 states=[x for x in n['dispatch_records'] if x['stage']==1]
 for st in states:
  args={a['index']:a for a in st['actual_active_arguments']};assert args[6]['name']=='S' and args[6]['reflection_access']==0 and args[9]['name']=='S_new' and args[9]['reflection_access']==1
 for groups in [1,2,3,4,5,7,8,16]:
  for arm in [0,1]:
   k=read(out/f'native-g{groups}-a{arm}.json');assert k['observed_encoded_dispatches']==30*groups and k['trailing_Buffers_barriers_last_loop']==groups and k['last_CB_completed']
 for row in r['native_checks']:
  if row['groups']==16:
   oi=0 if row['output']=='o' else 1;raw=out/'numeric'/f"native-final-a{row['arm']}-b{row['bank']:02d}-o{oi}.raw";ref=out/'numeric'/f"expected-b{row['bank']:02d}-o{oi}.raw";terminal(raw,ref,row['A_sha256'],row['bytes'],'float32')
 selected[run].update(public_checks=30,captured_checks=60,native_saved_checks=480,raw_terminal_outputs=60,raw_references=30,actual_PSOs=4,actual_scaled_state_arg3='inactive',S_input6_ReadOnly_Snew9=True,per_dispatch_value_checks=False)
 if run.startswith('kernel'):
  rows=r['timed_terminal_checks'];assert len(rows)==r['timed_terminal_strict_checks']==600 and all(x['strict_equal'] and x['actual_sha256']==x['expected_sha256'] for x in rows);selected[run]['timed_terminal_checks']=600

for run,multiplier in [('kernel-micro-v80-attempt1',8),('kernel-micro-v84-attempt1',8),('kernel-micro-v85-attempt1',30)]:
 r=receipts[run];out=R/run/'child-output';pairs=read(out/'pairs.json') if run!='kernel-micro-v85-attempt1' else r['pairs'];assert len(pairs)==10
 assert [tuple(p['order']) for p in pairs]==[('A','A'),('B','B')]+[('A','B'),('B','A')]*3+[('A','A'),('B','B')]
 ab=[];controls={'A':[],'B':[]}
 for p in pairs:
  a,b=p['blocks'];assert p['kind']==('control' if p['order'][0]==p['order'][1] else 'AB')
  for block in [a,b]:
   n=block['native'];assert block['prime_groups']==128 and block['timed_groups']==1024 and block['encoded_dispatches']==multiplier*1024 and block['ns_per_dispatch']==block['wall_ns']/(multiplier*1024)
   assert n['last_CB_completed'] and n['CBGPU_timestamps_valid'] and 0<n['CBGPUStartTime_seconds']<=n['CBGPUEndTime_seconds'] and n['observed_encoded_dispatches']==multiplier*1024
   envelope=(n['CBGPUEndTime_seconds']-n['CBGPUStartTime_seconds'])*1e9;assert envelope<=n['last_native_wall_ns']+1e5<=block['wall_ns']+2e5
  if p['kind']=='control':ratio=b['wall_ns']/a['wall_ns'];assert p['slot2_div_slot1']==ratio;controls[p['order'][0]].append(ratio)
  else:ab.append(p)
 med={arm:statistics.median(next(b['ns_per_dispatch'] for b in p['blocks'] if b['arm']==arm) for p in ab) for arm in ['A','B']};gain=1-med['B']/med['A'];wins=sum(next(b['wall_ns'] for b in p['blocks'] if b['arm']=='B')<next(b['wall_ns'] for b in p['blocks'] if b['arm']=='A') for p in ab)
 passed=all(all(.95<=v<=1.05 for v in values) and .98<=statistics.median(values)<=1.02 for values in controls.values());assert passed and r['control_passed'] and r['controls']==controls and r['medians_ns_per_dispatch']==med and r['median_reduction']==gain and r['wins']==wins
 assert r['status']=='STOP_MICRO_GATE' and gain<.05 and r['adopted'] is False
 n=r['native_final'];assert n['loops']==56 and n['total_groups']==23132 and n['total_encoded_dispatches']==23132*multiplier
 selected[run].update(medians_ns_per_dispatch=med,median_reduction=gain,wins=wins,control_passed=True,predeclared_minimum_reduction=.05,status='STOP_MICRO_GATE',timed_dispatches=20*1024*multiplier)
assert used_fixtures==set(fixture)
for key in ['author_self_review_80','author_self_review_85']:
 assert 'author' in J(HERE/'review-provenance.json')[key]
links=0
for doc in [article,HERE/'README.md']:
 for target in re.findall(r'(?<!!)\[[^\]]+\]\(([^)]+)\)',doc.read_text()):
  if '://' in target:continue
  p=(doc.parent/target).resolve()
  if not p.exists():assert p.name in ['verification.json','verification-originals.json','manifest.json'] and ARGS.report
  links+=1
manifest=HERE/'manifest.json'
if manifest.exists():
 m=J(manifest);assert m['chapter_sha256']==filehash(article);assert m['files']=={str(p.relative_to(HERE)):filehash(p) for p in sorted(HERE.rglob('*')) if p.is_file() and p.name!='manifest.json'}
summary={'status':'PASS_CPU_ARCHIVE_TEXT_HASH_STATS_LOCAL9','mode':'originals_raw_terminal_bytes_plus_text' if ARGS.originals else 'archive_only_text_identity_statistics','article_sha256':filehash(article),'actual_runs':9,'evidence_text_members':len(index['files']),'evidence_gzip_sha256':index['sha256'],'deterministic_gzip_reconstruction':True,'source_rows':len(source_rows),'dependency_bindings':len(dep_rows),'terminal_fixture_identities':len(fixture_rows),'external_source_pin_identities':len(external_rows),'original_paths_checked':len(original_checks),'saved_resource_samples':resource_count,'scope_excludes_actual':[76,82,86,89,90],'selected':selected,'per_dispatch_value_checks':False,'tensor_payload_archived':False,'MLX_imported':False,'native_loaded':False,'GPU_executed':False,'review_role':'Archive-author CPU saved-text/hash/statistics verification;85 author-self NOT independent; no new GPU arithmetic or live readonly reconstruction.','validated_links':links}
if ARGS.report:ARGS.report.write_text(json.dumps(summary,ensure_ascii=False,sort_keys=True,indent=2)+'\n')
print(json.dumps({'status':summary['status'],'mode':summary['mode'],'runs':9,'text_members':len(index['files']),'terminal_fixtures':len(fixture_rows),'original_paths_checked':len(original_checks),'all3_micro_STOP':True,'no_GPU':True}))
