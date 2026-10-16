"""CPU-only archived-text/source/statistics verifier. No MLX/native import, live resource probe or GPU.
Default needs this and prior chapter archives. --originals additionally hashes external identities.
"""
from pathlib import Path
import argparse,ast,gzip,hashlib,io,json,re,tarfile,statistics,collections,math
HERE=Path(__file__).resolve().parent;article=HERE.parent/'36-cap64-and-rms-profile.md'
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
members=t.getmembers();assert len(members)==len(index['files'])==255
runs={'fullmodel-v89native-attempt1','native-decode-v90-attempt1','numeric-v91-attempt1','fullmodel-v93native-attempt1','fullmodel-v97native-attempt1','native-decode-v100-attempt1'}
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
assert len(external)==len(external_rows)==10 and all(not x['payload_archived'] for x in external_rows)
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


import array,itertools,struct,sys
fixtures=J(HERE/'terminal-fixture-index.json')['files'];assert len(fixtures)==108 and all(not x['payload_archived'] for x in fixtures)
if ARGS.originals:
 for row in fixtures:
  p=Path(row['source']);assert p.stat().st_size==row['bytes'] and filehash(p)==row['sha256'];original_checks.add(str(p))
expected={'fullmodel-v89native-attempt1':'40428c19e12c69c80209885f58246b585d0c0659c3890ae2396f5f7310143e86','native-decode-v90-attempt1':'c654e16957033b680827b89831a93d7f0eb0609f6a6bccd6224c29e569b67288','numeric-v91-attempt1':'8e147291526235589fa5f6e811abb9bdf71407003f8a2272c420f225299a45dc','fullmodel-v93native-attempt1':'0cb2058cb050b0c82f14ff67e2ee636ea17843fa814b6f136f2d84e00ddcc575','fullmodel-v97native-attempt1':'6c9621109e8f23e20b88244324b7b104e7672ca67a89342af52c4ebfe0b65947','native-decode-v100-attempt1':'a2eb1eaa79f60901be83aa119a701d8fa137653da8ac7d63906ffb2f2c2cbc74'}
receipts={};selected={};resources_count=0
for run in sorted(runs):
 out=R/run;r=read(out/'child-output/receipt.json');assert sha(out/'child-output/receipt.json')==expected[run];receipts[run]=r
 for p,h in r['source_pins'].items():
  binding=next(x for x in dep_rows if x['source']==p);assert binding['sha256']==h
  if p in virtual:assert sha(p)==h
 owner=read(out/'owner-receipt.json');release=read(out/'external-release.json');pid=owner.get('child_pid',owner.get('driver_pid'))
 assert owner['stage']=='reaped' and owner['waited_reaped'] and owner['returncode']==(1 if run=='fullmodel-v93native-attempt1' else 0) and owner['absence']['returncode']==1 and not owner['absence']['stdout']
 script=next(p for p in owner['command'] if p.endswith('.py'));assert sha(script)==owner.get('script_sha256',owner.get('driver_sha256'))
 assert release['status']=='PASS_EXTERNAL_RELEASE' and release['GPU_released'] and release['tool_exit']==owner['returncode'] and {x['pid'] for x in release['strict_absence']}=={owner['owner_pid'],pid} and all(x['returncode']==1 and x['stdout']==x['stderr']=='' for x in release['strict_absence'])
 rawpaths=sorted(k for k in virtual if k.startswith(str(out/'child-output')+'/resource-') and k.endswith('-raw.json'));raws=[read(p) for p in rawpaths]
 for x in raws:assert x['power_metadata']['source']=='AC Power' and x['processinfo_thermal_metadata']['code']==0 and x['pressure_level_value']==1 and x['swap_used_mb']==raws[0]['swap_used_mb'] and x['vm_counters']['Swapouts']==raws[0]['vm_counters']['Swapouts']
 resources_count+=len(raws);selected[run]={'receipt_sha256':expected[run],'status':r['status'],'source_pin_bindings':len(r['source_pins']),'owner':owner['owner_pid'],'child':pid,'returncode':owner['returncode'],'release_sha256':sha(out/'external-release.json'),'saved_child_resource_samples':len(raws)}
G=[]
for version,expected_sha in [(38,'c57e7625baa5a4ce450fc7306233e766f1efd71636713080d748780624542c8a'),(44,'1dcfc41e7779c575602bd6c73bae4e4800664db42e1ee87e75414335bdc45b6d')]:
 p=R/f'fullmodel-v{version}native-attempt1/child-output/receipt.json';assert sha(p)==expected_sha;G.append({x['name']:x for x in read(p)['requests']})
r89=receipts['fullmodel-v89native-attempt1'];r93=receipts['fullmodel-v93native-attempt1'];r97=receipts['fullmodel-v97native-attempt1']
for r,is_new in [(r89,False),(r97,True)]:
 assert r['model_loads']==1 and r['fixed_cap']==64 and r['canonical_capacity']==256 and r['adopted'] is False and r['performance_point'] is False and r['allocation_bucket']['effective']==[64,64] and r['allocation_bucket_restored_finally']
 assert len(r['requests'])==3 and r['strict_output_checks']==12276 and r['native_steps']==93 and r['same_parameter_identity_count']==1349 and r['same_parameter_identity_preserved']
 checks=0;tails=0;equal_old=0
 for i,req in enumerate(r['requests'],1):
  assert len(req['steps'])==31 and len(req['A_ids'])==32 and req['A_ids']==req['B_ids'] and req['seed_upload_calls']==req['seed_arrays']==131
  if not is_new:assert req['A_ids']==G[0][req['name']]['A_ids']==G[1][req['name']]['A_ids']
  for step in req['steps']:
   assert step['B_fixed_L']==64 and len(step['outputs'])==132 and step['A_token']==step['B_token'] and step['reference_input_digest_readonly'] and step['native_prediction_copied_bytes']==4 and step['seed_uploads_this_step']==0 and step['KV_physical64_tail_zero_checks']==32
   for role,narrays in [('seed',131),('capture_outputs',132),('request_seed',131)]:assert step['originals_readonly'][role]['arrays']==narrays and step['originals_readonly'][role]['strict_bytes_equal']
   if is_new:assert step['originals_readonly']['A_request_seed']['arrays']==131 and step['originals_readonly']['A_request_seed']['strict_bytes_equal']
   oldrows=[g[req['name']]['steps'][step['step']-1]['outputs'] for g in G]
   for idx,row in enumerate(step['outputs']):
    assert row['finite'] and row['strict_bytes_equal'] and row['A_sha256']==row['canonical_B_sha256']
    for old in oldrows:assert all(row[k]==old[idx][k] for k in ['label','shape','dtype','bytes'])
    assert oldrows[0][idx]['A_sha256']==oldrows[1][idx]['A_sha256']
    if is_new:
     assert row['old_profile_sha256']==oldrows[0][idx]['A_sha256'] and row['old_profile_strict_equal']==(row['A_sha256']==row['old_profile_sha256']);equal_old+=row['old_profile_strict_equal']
    else:assert row['A_sha256']==oldrows[0][idx]['A_sha256']
    if row['physical_tail_zero_checked']:
     assert row['physical_shape']==[1,4,64,256] and row['shape']==[1,4,256,256] and row['physical_bytes']==131072 and row['bytes']==524288 and row['canonical_CPU_zero_fill'];tails+=1
    else:assert row['physical_shape']==row['shape'] and row['physical_bytes']==row['bytes'] and row['physical_A_sha256']==row['A_sha256'] and row['canonical_CPU_zero_fill'] is False
    checks+=1
  n=req['native'];f=n['feedback'];icb=f['icb'];pp=f['pingpong'];assert f['capacity']==64 and f['total_steps']==icb['execution_count']==31*i and f['resets']==i and f['request_seed_uploaded_ranges']==131 and f['CPU_full_state_restore_per_step'] is False
  assert icb['banks']==2 and icb['bank_executions']==[16*i,15*i] and (pp['eligible_count'],pp['staged_link_count'])==(48,83) and n['dispatches']==(1609 if is_new else 1625)
 assert checks==12276 and tails==r['canonical_KV_tail_zero_checks']==2976
 if is_new:assert equal_old==r['old_profile_canonical132_hash_diagnostic']['equal']==241 and 12276-equal_old==r['old_profile_canonical132_hash_diagnostic']['different']==12035
 selected['fullmodel-v97native-attempt1' if is_new else 'fullmodel-v89native-attempt1'].update(strict_outputs=checks,tail_zero_records=tails,raw_IDs=96,physical_GPU_cap=64,canonical_CPU_cap=256,old_hash_equal=equal_old if is_new else 12276)

# Preserve the actual failed gate and zero-capture boundary, not a reconstructed warm trace.
assert r93['status']=='FAIL_STOP' and r93['stage']=='ready' and r93['requests']==[] and r93['independent_prefill_qualifications']==[] and 'warm_ids == w["continuation_ids"]' in r93['traceback']
fail_native=read(R/'fullmodel-v93native-attempt1/child-output/native-on-failure.json');assert fail_native['dispatches']==fail_native['native_replays']==0
failure=read(P/'v93-failure-output-id-evidence-build1/evidence.json');assert failure['failed_receipt_sha256']==expected['fullmodel-v93native-attempt1'] and failure['prepared_gamma_pairs_verified']==161 and failure['A_B_prefill_pairs_completed']==failure['native_dispatches']==failure['native_replays']==0 and failure['warm_ids_saved'] is False and failure['warm_divergence_index']=='UNKNOWN' and failure['donor_internal_raw_ids_saved'] is False and failure['old_failure_relabelled'] is False
assert failure['old_workload_continuation_ids']!=failure['historical_donor_visible_roundtrip_ids'] and failure['historical_first_difference_index0']==4
selected['fullmodel-v93native-attempt1'].update(gamma_pairs_prepared=161,A_B_prefill_pairs=0,native_dispatches=0,native_replays=0,warm_stream_saved=False,FAIL_preserved=True)
for r in [r93,r97,receipts['native-decode-v100-attempt1']]:
 prep=r['vendor_rms_profile_preparation'];assert prep['plain_norms']==161 and prep['parameter_arrays']==1349 and prep['same_parameter_identity'] and prep['prepared_before_any_token'] and len(prep['all161_gamma_pairs_strict'])==161
 for pair in prep['all161_gamma_pairs_strict']:
  a,b=pair['A_raw'],pair['B_cached'];assert pair['strict_bytes_equal'] and pair['separate_objects'] and pair['separate_addresses'] and pair['A_B_ranges_disjoint'] and a['address']!=b['address'] and a['finite'] and b['finite'] and all(a[k]==b[k] for k in ['nbytes','shape','dtype','sha256'])
 assert prep['metadata']['whole_omlx_semantics'] is False and prep['metadata']['new_registered_parameters']==0
assert r97['status']=='DIAGNOSTIC_COMPLETED_NEW_REFERENCE_STRICT_NO_ADOPTION' and r97['diagnostic_only'] and r97['old_profile_pass'] is False and r97['old_ID_equality_required'] is False
prefills=r97['independent_prefill_qualifications'];assert len(prefills)==3
for pf in prefills:
 assert len(pf['seed131'])==131 and pf['seed131_strict_bytes_equal'] and pf['A_B_states_retained_separately'] and pf['B_seed_origin']=='own B prefill only'
 for row in pf['seed131']:assert row['strict_bytes_equal'] and row['separate_array_handles'] and row['A_sha256']==row['B_sha256']
 last=pf['last_logits'];assert last['strict_bytes_equal'] and last['A']['finite'] and last['B']['finite'] and last['A']['sha256']==last['B']['sha256']
assert r97['independent_prefill_summary']['seed131_strict_checks']==393 and r97['independent_prefill_summary']['last_logits_strict_checks']==3
newbase=r97['requests'][0]['A_ids'];assert newbase==r97['requests'][1]['A_ids']==r97['donor_visible_reference']['ids'] and r97['requests'][2]['A_ids']==G[0]['changed-prefill']['A_ids'] and newbase!=G[0]['base']['A_ids']
assert r97['donor_visible_reference']['source_kind']=='visible text tokenizer roundtrip, NOT internal raw generation IDs'
for r in [r97,receipts['native-decode-v100-attempt1']]:
 eng=r['B_vendor_plain_rms_capture_engagement'];assert eng['calls']==eng['distinct_plain_norm_objects']==161 and eng['all161_prepared_enabled'] and eng['counter_removed_before_replay'] and eng['GatedRMSNorm_hooked'] is False
 assert len(r['derived_cachedgamma161_native_readonly_proof'])==161 and all(x['full_range_readonly'] for x in r['derived_cachedgamma161_native_readonly_proof'])
 assert r['gamma322_readonly_final']['arrays']==322 and r['gamma322_readonly_final']['strict_bytes_equal']
selected['fullmodel-v97native-attempt1'].update(independent_seed_checks=393,independent_last_logits=3,old_hash_equal=241,old_hash_different=12035,plainnorm_calls=161,cached_gamma_RO_ranges=161,whole_OMLX_semantics=False,old_profile_PASS=False)

# Diagnostic RMS rounding: offline metadata; optional exact108raw/162metric replay.
r91=receipts['numeric-v91-attempt1'];cases=read(R/'numeric-v91-attempt1/child-output/cases.json');assert r91['status']=='DIAGNOSTIC_COMPLETED_NO_ADOPTION' and r91['cases']==27 and r91['outputs']==108 and r91['pairwise_diagnostic_comparisons']==162 and len(cases)==27 and r91['all_finite'] and r91['all15_originals_readonly']
counts={a+'-'+b:0 for a,b in itertools.combinations('ABCD',2)};fixture_by_path={x['source']:x for x in fixtures}
for case in cases:
 assert case['all15_originals_readonly'];values={}
 for name,row in case['outputs'].items():
  p=R/'numeric-v91-attempt1/child-output'/row['raw'];assert row['finite'] and row['bytes']==10240 and fixture_by_path[str(p)]['sha256']==row['sha256']
  if ARGS.originals:
   data=p.read_bytes();bits=array.array('H',data)
   if sys.byteorder!='little':bits.byteswap()
   assert len(bits)==5120 and all(v&0x7f80!=0x7f80 for v in bits);values[name]=(data,bits,[struct.unpack('<f',struct.pack('<I',v<<16))[0] for v in bits])
 for a,b in itertools.combinations('ABCD',2):
  key=a+'-'+b;decl=case['comparisons'][key];counts[key]+=not decl['strict_bytes_equal']
  assert decl['strict_bytes_equal']==(case['outputs'][a]['sha256']==case['outputs'][b]['sha256'])
  if ARGS.originals:
   aa,bb=values[a],values[b];ordered=lambda u:(~u)&0xffff if u&0x8000 else u|0x8000
   metrics={'strict_bytes_equal':aa[0]==bb[0],'different_elements':sum(x!=y for x,y in zip(aa[1],bb[1])),'max_BF16_ULP_distance':max(abs(ordered(x)-ordered(y)) for x,y in zip(aa[1],bb[1])),'max_absolute_difference':max(abs(x-y) for x,y in zip(aa[2],bb[2])),'signed_zero_bit_differences':sum(x!=y and (x&0x7fff)==(y&0x7fff)==0 for x,y in zip(aa[1],bb[1]))};assert all(decl[k]==v for k,v in metrics.items())
assert counts==r91['unequal_pair_counts']=={'A-B':21,'A-C':0,'A-D':21,'B-C':21,'B-D':0,'C-D':21}
selected['numeric-v91-attempt1'].update(cases=27,output_fixture_identities=108,pairwise_diagnostics=162,two_local_rounding_families=True,universal_formula_or_performance_claim=False)
# Optional independent checkpoint-range identity, no full-shard hash/copy.
specs=read(P/'v92-vendor-rms-leaf-build1/raw-gamma-specs.json')['arrays'];assert len(specs)==161
if ARGS.originals:
 for row in specs:
  with Path(row['path']).open('rb') as f:
   headlen=int.from_bytes(f.read(8),'little');head=f.read(headlen);assert hashlib.sha256(head).hexdigest()==row['header_sha256'];entry=json.loads(head)[row['key']];assert entry['dtype']==row['dtype'] and entry['shape']==row['shape'] and [x+8+headlen for x in entry['data_offsets']]==[row['offset'],row['offset']+row['nbytes']];f.seek(row['offset']);raw=f.read(row['nbytes']);assert len(raw)==row['nbytes'] and hashlib.sha256(raw).hexdigest()==row['raw_sha256']
  original_checks.add(row['path']+'::'+row['key'])

for version,reference in [(90,r89['requests'][0]['steps'][-1]['outputs']),(100,r97['requests'][0]['steps'][-1]['outputs'])]:
 run=f'native-decode-v{version}-attempt1';r=receipts[run];assert r['model_loads']==1 and r['fixed_cap']==64 and r['schedule']==['warm']*3+['measure']*3 and len(r['requests'])==6 and r['terminal_canonical_hash_checks']==792
 assert r['qualification_sha256']==expected['fullmodel-v89native-attempt1' if version==90 else 'fullmodel-v97native-attempt1']
 if version==100:assert r['old_profile_pass'] is False and r['whole_omlx_semantics'] is False and r['adopted'] is False and r['HTTP_or_Pareto_point'] is False and r['fair_AB_win_claim'] is False
 for i,req in enumerate(r['requests'],1):
  assert req['ids']==(G[0]['base']['A_ids'] if version==90 else newbase) and len(req['ids'])==32 and req['phase']==('warm' if i<=3 else 'measure') and req['t0_ns']<req['t1_ns']<req['t2_ns']
  assert req['decode31_ns']==req['t2_ns']-req['t1_ns'] and req['whole32_ns']==req['t2_ns']-req['t0_ns'] and req['decode_ns_per_step']==req['decode31_ns']/31 and req['decode31_tokens_per_s']==31e9/req['decode31_ns'] and req['whole32_tokens_per_s']==32e9/req['whole32_ns']
  assert len(req['final132_hash_checks'])==132;tails=0
  for row,gold in zip(req['final132_hash_checks'],reference):
   assert row['label']==gold['label'] and row['bytes']==gold['bytes'] and row['sha256']==row['qualified_A_sha256']==gold['A_sha256']
   if row['physical_tail_zero_checked']:assert row['physical_shape']==[1,4,64,256] and row['CPU_canonical_zero_fill'];tails+=1
  assert tails==32
  f=req['native']['feedback'];icb=f['icb'];pp=f['pingpong'];assert f['total_steps']==icb['execution_count']==31*i and f['resets']==i and f['request_seed_uploaded_ranges']==131 and f['capacity']==64 and icb['bank_executions']==[16*i,15*i] and icb['banks']==2 and pp['eligible_count']==48 and pp['staged_link_count']==83
  for role,narrays in [('capture_seed',131),('captured_reference',132)]:assert req['captured263_original_owners_readonly'][role]['arrays']==narrays and req['captured263_original_owners_readonly'][role]['strict_bytes_equal']
  if version==100:assert req['gamma322_readonly']['arrays']==322 and req['gamma322_readonly']['strict_bytes_equal']
 measured=[x for x in r['requests'] if x['phase']=='measure'];med={k:statistics.median(x[k] for x in measured) for k in ['decode_ns_per_step','decode31_tokens_per_s','whole32_tokens_per_s']}
 for k,v in med.items():assert r['measured_summary']['median_'+k]==v
 selected[run].update(requests=6,terminal_hashes=792,KV_tail_records=192,steps=186,resets=6,banks=[96,90],measured=med,B_only_unpaired=True)
# The root100 audit is saved JSON only, not a fabricated separately-runnable author script.
audit=read(P/'v100-root-saved-result-audit-build1/audit.json');assert audit['receipt_sha256']==expected['native-decode-v100-attempt1'] and audit['terminal132_hashes']==792 and audit['native_steps']==186 and audit['resets']==6
review97=read(P/'v97-independent-results-review-build1/verification.json');assert receipts['native-decode-v100-attempt1']['independent97_review_sha256']==sha(P/'v97-independent-results-review-build1/verification.json')
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
summary={'status':'PASS_CPU_ARCHIVE_TEXT_HASH_STATS_CAP64_NEW_RMS_PROFILE6','mode':'originals_plus_91raw_pairmetrics_and_gamma_ranges' if ARGS.originals else 'archive_only_text_identity_statistics','article_sha256':filehash(article),'actual_runs':6,'text_members':len(index['files']),'source_rows':len(source_rows),'dependency_bindings':len(dep_rows),'external_source_pin_identities':len(external_rows),'91_output_fixture_identities':108,'original_paths_checked':len(original_checks),'deterministic_evidence_gzip':True,'evidence_gzip_sha256':index['sha256'],'selected':selected,'V93_FAIL_preserved':True,'V97_old_profile_PASS':False,'whole_OMLX_semantics':False,'HTTP_Pareto_or_speedup_claim':False,'saved_child_resource_samples':resources_count,'no_tensor_payload_archived':True,'MLX_imported':False,'native_loaded':False,'GPU_executed':False,'validated_links':links,'scope':'Only89/90/91/93/97/100. Savedtext/metadata/identity verification; no new GPU or dead-process readonly reconstruction.'}
if ARGS.report:ARGS.report.write_text(json.dumps(summary,ensure_ascii=False,sort_keys=True,indent=2)+'\n')
print(json.dumps({'status':summary['status'],'mode':summary['mode'],'runs':6,'text_members':len(index['files']),'91_output_fixtures':108,'original_paths_checked':len(original_checks),'V93_FAIL_preserved':True,'V97_old_profile_PASS':False,'no_GPU':True}))
