"""CPU-only archived-text/source/statistics verifier. No MLX/native import, live resource probe or GPU.
Default needs this and prior chapter archives. --originals additionally hashes external identities.
"""
from pathlib import Path
import argparse,ast,gzip,hashlib,io,json,re,tarfile,statistics,collections,math
HERE=Path(__file__).resolve().parent;article=HERE.parent/'38-spec-transaction-and-qmm.md'
ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--originals',action='store_true');ap.add_argument('--report',type=Path);ARGS=ap.parse_args()
filehash=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
J=lambda p:json.loads(Path(p).read_text())
source_rows=J(HERE/'source-map.json');dep_rows=J(HERE/'dependency-map.json')['files'];index=J(HERE/'evidence-index.json');external_rows=J(HERE/'external-fixture-identities.json')['files'];opened={};virtual={};original_checks=set()
redactions=J(HERE/'publication-redactions.json');redacted={x['source']:x for x in redactions['files']}
assert len(redacted)==len(redactions['files'])==32 and redactions['exact_original_evidence_members']==417
original_redactions_checked=set()
def public_check(row,b):
 assert len(b)==row.get('published_bytes',row['bytes']) and hashlib.sha256(b).hexdigest()==row.get('published_sha256',row['sha256'])
 if row.get('publication_redacted'):
  x=redacted[row['source']];assert (row['sha256'],row['bytes'],row['published_sha256'],row['published_bytes'])==(x['original_sha256'],x['original_bytes'],x['published_sha256'],x['published_bytes'])
 else:assert row['source'] not in redacted

def transformed_original(source,b):
 x=redacted[source];assert len(b)==x['original_bytes'] and hashlib.sha256(b).hexdigest()==x['original_sha256']
 v=json.loads(b)
 for field in x['fields']:
  parts=field.split('/');cur=v
  for part in parts[:-1]:cur=cur[int(part)] if isinstance(cur,list) else cur[part]
  key=int(parts[-1]) if isinstance(cur,list) else parts[-1]
  assert isinstance(cur[key],str) and cur[key] and cur[key]!=x['placeholder'];cur[key]=x['placeholder']
 original_redactions_checked.add(source)
 return (json.dumps(v,indent=2,ensure_ascii=False)+'\n').encode()
def archive_bytes(location):
 fn,sep,member=location.partition('::');p=(HERE/fn).resolve()
 if not sep:return p.read_bytes()
 if p not in opened:opened[p]=tarfile.open(p,'r:gz')
 f=opened[p].extractfile(member);assert f is not None;return f.read()
for row in source_rows:
 b=archive_bytes(row['archive']);public_check(row,b);b.decode('utf-8');virtual[row['source']]=b
assert filehash(HERE/index['archive'])==index['sha256']
assert (HERE/index['archive']).read_bytes()[:10]==bytes.fromhex('1f8b08000000000000ff')
t=tarfile.open(HERE/index['archive'],'r:gz');opened[(HERE/index['archive']).resolve()]=t
members=t.getmembers();assert len(members)==len(index['files'])
runs={'fullmodel-v108transaction-attempt1', 'numeric-v108tiny-attempt1', 'diagnostic-v112-attempt1', 'omlx-mtp-v111-attempt1', 'omlx-mtp-v109-attempt1'}
assert {row['archive'].split('/')[0] for row in index['files']}==runs
rebuilt=io.BytesIO()
with gzip.GzipFile(filename='',fileobj=rebuilt,mode='wb',mtime=0,compresslevel=6) as gz:
 with tarfile.open(fileobj=gz,mode='w|') as tf:
  for member,row in zip(members,index['files']):
   assert member.isfile() and member.name==row['archive'] and member.mtime==member.uid==member.gid==0 and member.mode==0o644
   assert not Path(member.name).is_absolute() and '..' not in Path(member.name).parts and Path(member.name).suffix in {'.json','.jsonl','.log','.md','.txt'}
   b=t.extractfile(member).read();b.decode('utf-8');public_check(row,b);virtual[row['source']]=b
   ti=tarfile.TarInfo(member.name);ti.size=len(b);ti.mtime=0;ti.mode=0o644;tf.addfile(ti,io.BytesIO(b))
assert hashlib.sha256(rebuilt.getvalue()).hexdigest()==index['sha256']
external=[]
for row in dep_rows:
 if row.get('location'):
  b=archive_bytes(row['location']);public_check(row,b);b.decode('utf-8');virtual.setdefault(row['source'],b)
 else:external.append(row)
 if ARGS.originals:
  p=Path(row['source']);assert p.stat().st_size==row['bytes'] and filehash(p)==row['sha256'];original_checks.add(str(p))
assert len(external)==len(external_rows) and all(not x['payload_archived'] for x in external_rows)
assert {(x['source'],x['bytes'],x['sha256']) for x in external}=={(x['source'],x['bytes'],x['sha256']) for x in external_rows}
assert len({x['source'] for x in external_rows})==len(external_rows)
for row in source_rows+index['files']:
 if ARGS.originals:
  p=Path(row['source']);b=p.read_bytes();assert (transformed_original(str(p),b) if str(p) in redacted else b)==virtual[row['source']];original_checks.add(str(p))
assert sum(bool(x.get('publication_redacted')) for x in index['files'])==32 and len(index['files'])==449
assert {x['source'] for x in index['files'] if x.get('publication_redacted')}==set(redacted)
def payload(p):
 key=str(p);assert key in virtual,('unmapped text',key);return virtual[key]
def read(p):return json.loads(payload(p))
def sha(p):
 key=str(p)
 if key in virtual:return hashlib.sha256(virtual[key]).hexdigest()
 row=next(x for x in dep_rows if x['source']==key);assert not row.get('location');return row['sha256'] # external identity only; --originals was checked above
P=Path('/private/tmp/minivllm-engine-optimization-20261014');R=P/'v23-root-control-build1'




expected={
'numeric-v108tiny-attempt1':('receipt.json','2a03d8ef97cbe0243b6c34c313d793e030e9dd64ff6610816dcc9f9bfde91b4b'),
'fullmodel-v108transaction-attempt1':('receipt.json','7966209c6a956f71d0b688ff689df11c0f8d082661f8a39d5a52c51a3cd71d9d'),
'omlx-mtp-v109-attempt1':('receipt.json','b9337fbc71bef73dfef4190834ba83c64b4483afa61b7d4cc1145d8e8c7b3ec5'),
'omlx-mtp-v111-attempt1':('suite-complete.json','95b83086f63ef6f26ea4d4b6061a52f1602b4b19aaa910766dba38d2f7694401'),
'diagnostic-v112-attempt1':('receipt.json','09c7f55cfbd46a803b4bd014eb24f694573f62de48e2a3fdee3568ec1910992a')}
selected={};receipts={}
for run in sorted(runs):
 out=R/run;owner=read(out/'owner-receipt.json');release=read(out/'external-release.json');pid=owner.get('child_pid',owner.get('driver_pid'))
 rc=1 if run=='fullmodel-v108transaction-attempt1' else 0
 assert owner['stage']=='reaped' and owner['waited_reaped'] and owner['returncode']==rc and owner['absence']['returncode']==1 and owner['absence']['stdout']==owner['absence']['stderr']==''
 assert release['status']=='PASS_EXTERNAL_RELEASE' and release['GPU_released'] and release['tool_exit']==rc and {x['pid'] for x in release['strict_absence']}=={owner['owner_pid'],pid}
 assert all(x['returncode']==1 and x['stdout']==x['stderr']=='' for x in release['strict_absence'])
 fn,h=expected[run];f=out/'child-output'/fn;assert sha(f)==h;r=read(f);receipts[run]=r
 for path,hash_value in r.get('pins',{}).items():assert sha(path)==hash_value
 selected[run]={'status':r['status'],'receipt_sha256':h,'owner':owner['owner_pid'],'child':pid,'returncode':rc,'release_sha256':sha(out/'external-release.json')}
r=receipts['numeric-v108tiny-attempt1'];assert r['strict_transactions']==r['completed_strict_transactions']==len(r['transactions'])==21 and r['abort_restored'] and r['source_inputs_readonly'] and not r['fullmodel_qualified'] and not r['timing_run']
assert all(x['same_invocation_prefix_strict'] and x['literal_history_readonly'] and x['invocation_inputs_readonly'] and x['producer_owners_retained'] for x in r['transactions'])
r=receipts['fullmodel-v108transaction-attempt1'];assert r['status']=='FAIL_STOP' and r['A_old96_IDs_passed'] and r['old96_ID_hard_gate_each_arm'] and r['old_tensor_gold_not_relaxed'] and len(r['requests'])==4 and len(r['transactions'])==15 and not r['performance_qualified'] and not r['adopted']
assert all(x['strict_old_ID_equal'] for x in r['requests'][:3]) and r['requests'][3]['IDs'][4]==100606 and '60277' in r['error']
assert 'target1349_readonly' not in r and 'parameter_after' not in r
r109=receipts['omlx-mtp-v109-attempt1'];assert r109['actual_fresh_servers']==2 and r109['actual_requests']==6 and r109['terminal_cleanup'] and len(r109['ordinary_control_raw32'])==32 and len(r109['MTP_per_request'])==3 and not r109['performance_qualified']
assert r['requests'][3]['IDs']==r109['ordinary_control_raw32']
r=receipts['diagnostic-v112-attempt1'];assert r['captured_operation_count']==978 and len(r['comparisons'])==1138 and r['initial_cache_matches_actual108_all128'] and r['original_posterior_reproduced'] and r['keep0_restored_all128'] and r['target1349_identity_readonly'] and r['source_identity_final'] and not r['future_gold_used'] and not r['candidate_changed'] and not r['adopted']
groups=collections.defaultdict(list)
for row in r['comparisons']:groups[row['kind']].append(row)
linear=groups.pop('Linear');assert len(linear)==497 and sum(x['strict_equal'] for x in linear)==146 and sum(not x['strict_equal'] for x in linear)==351
assert all(x['strict_equal'] and x['batch_sha256']==x['M1_sha256'] for rows in groups.values() for x in rows)
first=min((x for x in linear if not x['strict_equal']),key=lambda x:x['order']);assert first['order']==9 and first['name']=='layers.0.linear_attn.out_proj' and first['different_elements']==5 and first['total_elements']==25600 and first['max_abs_diagnostic_only']==4.76837158203125e-7
v=receipts['omlx-mtp-v111-attempt1'];assert v['HTTP_requests']==36 and v['HTTP_completion_tokens']==1152 and v['measured_requests']==18 and v['measured_tokens']==576 and len(v['arms'])==6 and len(v['pairs'])==3 and not v['fullfrontier_win'] and not v['global_best_claim'] and not v['adopted'] and not v['negative_or_slower_arms_discarded']
assert [x['order'] for x in v['pairs']]==[['none','mtp'],['mtp','none'],['none','mtp']]
for p in v['pairs']:
 for key,signed in [('x','x_signed_MTP_pct'),('y','y_signed_MTP_pct')]:
  assert p[signed]==(p['mtp'][key]/p['none'][key]-1)*100 and p[signed]<0
for signed in ['x_signed_MTP_pct','y_signed_MTP_pct']:assert v['paired_signed_medians'][signed]==statistics.median(x[signed] for x in v['pairs'])
for mode in ['none','mtp']:
 for key in ['x','y','ttft_s','E2E_s']:assert v['arm_medians'][mode][key]==statistics.median(x[mode][key] for x in v['pairs'])
assert len(v['fixed_cooling'])==5 and all(x['fixed_seconds']==20 and x['outside_request_clocks'] and not x['adaptive'] for x in v['fixed_cooling'])
for doc in [article,HERE/'README.md']:
 for link in re.findall(r'(?<!!)\[[^\]]+\]\(([^)]+)\)',doc.read_text()):
  if '://' not in link:assert (doc.parent/link).resolve().exists(),link
manifest=HERE/'manifest.json'
if manifest.exists():
 m=J(manifest);assert m['chapter_sha256']==filehash(article) and m['files']=={str(p.relative_to(HERE)):filehash(p) for p in sorted(HERE.rglob('*')) if p.is_file() and p!=HERE/'manifest.json'}
summary={'status':'PASS_CPU_ARCHIVE_PUBLICATION_VIEWS_AND_SAVED_COUNTS_V38','mode':'originals' if ARGS.originals else 'archive_only','actual_runs':5,'exact_original_evidence_members':417,'credential_redacted_views':32,'private_original_redactions_verified':len(original_redactions_checked),'source_rows':len(source_rows),'text_members':len(index['files']),'dependency_bindings':len(dep_rows),'external_identities_without_payload':len(external),'original_paths_checked':len(original_checks),'article_sha256':filehash(article),'selected':selected,'V108_old_IDs':'FAIL_STOP','V111_MTP_slower':True,'V112_only_Linear_local_differences':True,'Pareto_win':False,'default_adopted':False,'no_tensor_payload':True,'MLX_imported':False,'GPU_executed':False,'scope':'Offline published-byte/hash/count replay:187 exact sources,417 exact original evidence+32 credential-redacted views. Original SHA metadata for32 views is declared private-original provenance; only --originals verifies private bodies and field transformation. No tensor recomputation or runtime qualification.'}
if ARGS.report:ARGS.report.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:summary[k] for k in ['status','mode','actual_runs','source_rows','text_members','exact_original_evidence_members','credential_redacted_views','dependency_bindings','original_paths_checked','Pareto_win','GPU_executed']}))
