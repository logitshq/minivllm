"""CPU-only archived-text/source/statistics verifier. No MLX/native import, live resource probe or GPU.
Default needs this and prior chapter archives. --originals additionally hashes external identities.
"""
from pathlib import Path
import argparse,ast,gzip,hashlib,io,json,re,tarfile,statistics,collections,math
HERE=Path(__file__).resolve().parent;article=HERE.parent/'37-simd4-async-and-spec-handoff.md'
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
members=t.getmembers();assert len(members)==len(index['files'])==566
runs={'fullmodel-v103prod-attempt1', 'fullmodel-v99native-attempt1', 'numeric-v95-attempt1', 'native-http-v105-attempt1', 'fullmodel-v101diag-attempt1', 'fullmodel-v101prod-attempt1', 'fullmodel-v103diag-attempt1', 'native-decode-v104-attempt1'}
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
assert len(external)==len(external_rows)==20 and all(not x['payload_archived'] for x in external_rows)
assert {(x['source'],x['bytes'],x['sha256']) for x in external}=={(x['source'],x['bytes'],x['sha256']) for x in external_rows}
assert len({x['source'] for x in external_rows})==len(external_rows)
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



expected={
'numeric-v95-attempt1':'5e4cb9b849d7628dd1f1b799651e51c51c4bdf4825b16659ea8e15224ec4675e',
'fullmodel-v99native-attempt1':'11654508130ba3d880a29075ae608db2d5aa97c33763d69ff5b69f1bbb1ae2f5',
'native-decode-v104-attempt1':'8cf88a87ea15388b8aa213cb5c9f63d6e7181c9e3acc32ba1c3ecc1b1fe5759f',
'fullmodel-v101diag-attempt1':'98489937d4bcddb93db83c84de09bb89017a0b4ce328272e0509bd99595a9d7d',
'fullmodel-v101prod-attempt1':'0ffe1ad5626a012ca2069b8163efcfc2cd91d98ec712a50c17f2631d93605f78',
'fullmodel-v103diag-attempt1':'f39f01aaa9e085cf263adb664fe8ba89fe5de5c4a4bee864c3ac2f856cad5b36',
'fullmodel-v103prod-attempt1':'551ccb7d44965fe29b3c9f5c2cb2eaf7f8b2c43c65ba5ab9aefc6d90a589bba1'}
selected={};receipts={}
for run in sorted(runs):
 out=R/run;owner=read(out/'owner-receipt.json');release=read(out/'external-release.json');pid=owner.get('child_pid',owner.get('driver_pid'))
 rc=1 if run=='native-http-v105-attempt1' else 0
 assert owner['stage']=='reaped' and owner['waited_reaped'] and owner['returncode']==rc and owner['absence']['returncode']==1 and owner['absence']['stdout']==owner['absence']['stderr']==''
 assert release['status']=='PASS_EXTERNAL_RELEASE' and release['GPU_released'] and release['tool_exit']==rc and {x['pid'] for x in release['strict_absence']}=={owner['owner_pid'],pid}
 assert all(x['returncode']==1 and x['stdout']==x['stderr']=='' for x in release['strict_absence'])
 selected[run]={'owner':owner['owner_pid'],'child':pid,'returncode':rc,'release_sha256':sha(out/'external-release.json')}
 if run in expected:
  f=out/'child-output/receipt.json';assert sha(f)==expected[run];r=read(f);receipts[run]=r;assert r['status'].startswith('PASS_')
  for path,h in r.get('source_pins',{}).items():assert sha(path)==h
  selected[run].update(status=r['status'],receipt_sha256=expected[run])
r=receipts['numeric-v95-attempt1'];assert r['strict_output_checks']==30 and r['input_readonly_cases']==15 and not r['timing_run'] and not r['engine_connected']
r99=receipts['fullmodel-v99native-attempt1'];assert r99['strict_output_checks']==12276 and r99['native_steps']==93 and r99['canonical_KV_tail_zero_checks']==2976 and r99['fixed_cap']==64 and r99['canonical_capacity']==256 and r99['same_parameter_identity_count']==1349 and r99['same_parameter_identity_preserved'] and not r99['adopted']
checks=tails=0
for req in r99['requests']:
 assert len(req['A_ids'])==32 and req['A_ids']==req['B_ids'] and len(req['steps'])==31
 for step in req['steps']:
  assert len(step['outputs'])==132
  for row in step['outputs']:
   assert row['finite'] and row['strict_bytes_equal'] and row['A_sha256']==row['canonical_B_sha256'];checks+=1
   if row['physical_tail_zero_checked']:tails+=1
assert checks==12276 and tails==2976
for mode,full,pred,terminal in [('diag',75,0,0),('prod',0,15,15)]:
 r=receipts[f'fullmodel-v101{mode}-attempt1'];assert r['physical_capacity']==64 and r['perstep_all_output_checks']==full and r['prediction_only_checks']==pred and r['quiescent_final_output_checks']==terminal and not r['production_perstep_all_state_claim']
for mode,total,allstate in [('diag',21252,True),('prod',660,False)]:
 r=receipts[f'fullmodel-v103{mode}-attempt1']
 assert r['strict_output_checks']==total and r['native_steps']==r['native_delivered_prediction_ID_checks']==161 and r['native_completed_steps']==r['total_submitted_steps']==162 and r['total_discarded_steps']==1 and r['resets']==6
 assert r['physical_capacity']==64 and r['CPU_canonical_capacity']==256 and r['same1349_parameter_identities_preserved'] and r['captured_originals_readonly'] and r['perstep_full_state_claim']==allstate and r['discarded_future_all132_tensor_checks']==0
r104=receipts['native-decode-v104-attempt1'];assert r104['terminal_canonical_hash_checks']==792 and r104['same1349_parameter_owners_final'] and r104['qualification_sha256']==expected['fullmodel-v99native-attempt1']
measured=[r for r in r104['requests'] if r['phase']=='measure'];assert len(measured)==3 and len(r104['requests'])==6
for key in ['decode_ns_per_step','decode31_tokens_per_s','whole32_tokens_per_s']:
 assert r104['measured_summary']['median_'+key]==statistics.median(x[key] for x in measured)
rev=read(P/'v105-independent-results-review-build1/receipt.json')
assert rev['completed_arms']==8 and rev['attempted_arms']==9 and rev['HTTP_requests']==49 and rev['HTTP_tokens']==1568 and rev['HTTP_measured_requests']==24 and rev['HTTP_measured_tokens']==768 and rev['total_requests_including_startup']==52 and rev['total_tokens_including_startup']==1664
assert not rev['full_C1_C2_C4_curve'] and not rev['HTTP_or_Pareto_win'] and not rev['adopted'] and rev['nested_server_count']==9 and rev['nested_all_saved_waited_and_current_absent'] and rev['nested_all9_ports_saved_and_current_absent']
failure=read(R/'native-http-v105-attempt1/child-output/suite-failure.json');assert failure['error']['message']=='nominal thermal required' and failure['raw_preserved']
q=R/'native-http-v105-attempt1/child-output';warm=measure=tokens=0
for arm in sorted({Path(k).relative_to(q).parts[0] for k in virtual if k.startswith(str(q)+'/C1-')}):
 for gid in ['warm','measure']:
  for n in range(3):
   path=q/arm/f'C1-{gid}-{n}.json'
   if str(path) not in virtual:continue
   v=read(path);assert v['actual_request_count']==1 and v['total_completion_tokens']==32
   if gid=='warm':warm+=1
   else:measure+=1
   tokens+=v['total_completion_tokens']
assert (warm,measure,tokens)==(25,24,1568)
last=read(q/'C1-B3-omlx/resources.json');assert next(x for x in last if x['label']=='after-C1-warm-0')['processinfo_thermal_metadata']['code']==1
native3=read(q/'C1-B3-native/resources.json');assert len(native3)==17 and all(x['processinfo_thermal_metadata']['code']==0 for x in native3)
for doc in [article,HERE/'README.md']:
 for link in re.findall(r'(?<!!)\[[^\]]+\]\(([^)]+)\)',doc.read_text()):
  if '://' not in link:assert (doc.parent/link).resolve().exists(),link
manifest=HERE/'manifest.json'
if manifest.exists():
 m=J(manifest);assert m['chapter_sha256']==filehash(article) and m['files']=={str(p.relative_to(HERE)):filehash(p) for p in sorted(HERE.rglob('*')) if p.is_file() and p.name!='manifest.json'}
summary={'status':'PASS_CPU_ARCHIVE_TEXT_IDENTITIES_AND_SAVED_COUNTS_V37','mode':'originals' if ARGS.originals else 'archive_only','actual_runs':8,'source_rows':len(source_rows),'text_members':len(index['files']),'dependency_bindings':len(dep_rows),'external_identities_without_payload':len(external),'original_paths_checked':len(original_checks),'article_sha256':filehash(article),'selected':selected,'V105_HTTP_requests':49,'V105_HTTP_tokens':1568,'V105_complete_campaign':False,'Pareto_win':False,'default_adopted':False,'no_tensor_payload':True,'MLX_imported':False,'GPU_executed':False,'scope':'Offline exact text/hash/count replay only; no saved-state tensor recomputation or target-machine runtime qualification.'}
if ARGS.report:ARGS.report.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:summary[k] for k in ['status','mode','actual_runs','source_rows','text_members','dependency_bindings','original_paths_checked','Pareto_win','GPU_executed']}))
