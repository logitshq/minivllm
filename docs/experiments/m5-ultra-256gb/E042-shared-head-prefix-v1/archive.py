"""Archive E042 small evidence, preserving audited local raw payloads by reference only."""
import hashlib,json,re,shutil,tarfile,tempfile
from pathlib import Path
R=Path('/Users/ryanli/code/minivllm');A=R/'.artifacts/m5-ultra-256gb';B=R/'bench/m5_ultra_256gb';D=Path(__file__).resolve().parent
P=A/'rowwise-head-shared-v1-child';X=A/'shared-head-prefix-execution-v1';L=A/'shared-head-prefix-lease-v1';Q=B/'rowwise_native_verify/head-shared-prereg-v1'
def read(p):return json.loads(p.read_bytes())
def sha(p):
 assert p.suffix!='.safetensors' and p.stat().st_size<64<<20,'metadata/text only'
 return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,value):
 with (D/name).open('x') as f:json.dump(value,f,indent=2);f.write('\n')
c=read(P/'complete.json');audit=read(Q/'independent-audit.json');plan=read(X/'plan.json');final=read(X/'final.json');lease=read(L/'lease.json');job=read(L/'job-0.json');resume=read(L/'resume.json');ae=read(A/'shared-head-prefix-audit-execution-v1/execution.json')
assert audit['status']==c['status']=='PASS' and audit['forward_prefix_qualified'] is c['forward_prefix_qualified'] is True
assert audit['complete_sha256']==sha(P/'complete.json') and audit['ledger_sha256']==sha(P/'events.jsonl')
assert audit['raw_files']==96 and audit['output_tensor_pairs']==5418 and audit['candidate_vocabulary_rows']==42 and audit['head_projection_calls']==9
assert len(audit['cases'])==len(c['cases'])==9 and sum(v['prefixes'] for v in audit['cases'])==42
assert all(not v[k] for v in (c,audit) for k in ('compiled_qualified','rollback_qualified','full_model_qualified','http_qualified','performance_qualified'))
assert ae['returncode']==0 and ae['argv']==plan['audit'] and ae['audit_source_sha256']==audit['audit_sha256']
assert c['source_model_pins_rechecked'] and c['device_cleanup_completed'] and c['head_projection_calls']==9
assert final['status']=='PASS_FORWARD_PREFIX_PRODUCER_ONLY' and final['errors']==[] and final['lease_returncode']==0 and final['lease_closed'] is True
assert lease['closed'] is True and lease['service_pid']==plan['service_pid']==52818 and ' '.join(lease['service_identity'].split())==plan['service_identity']==final['service_identity_after']
assert job['waited'] is True and job['returncode']==0 and job['command']==plan['child']==lease['jobs'][0]
identity=read(X/'numerical-child-identity.json');assert identity['pid']==job['pid']==c['pid']==lease['child_pid'] and identity['ppid']==c['ppid']==lease['owner_pid']
assert resume['identity_match'] is True and resume['state_before']=='T' and resume['state_after']=='S'
assert read(X/'invocation.json')['plan_sha256']==sha(X/'plan.json')=='55dbd68bb64a80b3091e629a20b6915a19d3ab92fb92c0b29417700aaefa6b7a'
for name in ('before-health.json','after-health.json'):
 h=read(X/name);s=h['/api/status'];target='Qwen3.8-Flash-Next-oQ6e-mtp'
 assert h['/health']['status']=='healthy' and s['loaded_models']==[target] and s['default_model']==target
 assert s['active_requests']==s['waiting_requests']==s['models_loading']==0
raw=[]
for p in sorted(P.rglob('*.safetensors')):
 side=p.with_suffix('.json');v=read(side);assert v['path']==str(p.relative_to(P)) and re.fullmatch('[0-9a-f]{64}',v['sha256']) and v['finite'] is True
 st=p.stat();raw.append(dict(path=str(p),relative_path=v['path'],bytes=st.st_size,sha256=v['sha256'],mtime_ns=st.st_mtime_ns,sidecar=str(side),sidecar_sha256=sha(side),payload_sha_evidence='Validated by the completed independent raw-byte auditor; not reread by metadata archiver.'))
assert len(raw)==96
save('raw-file-references.json',dict(raw_files=raw,total_bytes=sum(r['bytes'] for r in raw),raw_payload_archived=False,independent_audit_sha256=sha(Q/'independent-audit.json'),complete_sha256=sha(P/'complete.json'),scope='Local original raw paths and observed sizes; SHA values from bound sidecars fully checked by saved independent audit. This archive does not reread or package raw payloads.'))
baseline=read(X/'physical-before/sample.json');physical=[]
for p in sorted(X.glob('physical-[0-9]*/sample.json')):
 r=read(p);v=r['parsed'];assert r['status']=='MEASURED' and v['pid']==c['pid'] and v['pressure']==1
 assert max(v[k] for k in ('top_bytes','aux_current_bytes','aux_peak_bytes'))<=plan['physical']['child_limit_bytes'] and v['system_bytes']<=plan['physical']['system_limit_bytes']
 assert v['swap_bytes']<=baseline['parsed']['swap_bytes'] and v['swapouts']==baseline['parsed']['swapouts']==807478
 if physical:assert r['started_monotonic']-physical[-1]['ended_monotonic']<=5
 physical.append(r)
assert len(physical)==final['physical_samples']==47 and final['samples']==48
resources=dict(samples=47,child_current_max_bytes=max(r['parsed']['current_bytes'] for r in physical),child_recorded_peak_max_bytes=max(r['parsed']['peak_bytes'] for r in physical),system_max_bytes=max(r['parsed']['system_bytes'] for r in physical),pressure_levels=sorted({r['parsed']['pressure'] for r in physical}),swap_baseline_bytes=baseline['parsed']['swap_bytes'],swap_min_bytes=min(r['parsed']['swap_bytes'] for r in physical),swap_max_bytes=max(r['parsed']['swap_bytes'] for r in physical),swapouts=807478,scope='Saved parsed physical observations and source-bound monitor gates; no new resource probe or hard allocation cap.')
S={'child-metadata':P,'execution':X,'lease':L,'audit-execution':A/'shared-head-prefix-audit-execution-v1','child-prereg':Q,'parent-v1-never-run':B/'nvfp4_multirow_shared/head-prereg-v1','parent-v2':B/'nvfp4_multirow_shared/head-prereg-v2','cpu':B/'rowwise_native_verify/cpu-head-shared-v1','prepare-child':B/'rowwise_native_verify/prepare-head-shared-attempt-v1'}
stage=Path(tempfile.mkdtemp(prefix='e042-metadata-',dir='/private/tmp'));members=[];seen=set()
def copy(p,rel):
 key=str(rel)
 if key in seen:return
 assert not p.is_symlink() and p.suffix!='.safetensors';h=sha(p);q=stage/rel;q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q);assert sha(q)==h==sha(p)
 members.append(dict(path=key,source=str(p),bytes=q.stat().st_size,sha256=h));seen.add(key)
for label,root in S.items():
 for p in sorted(root.rglob('*')):
  if p.is_file() and p.suffix!='.safetensors':copy(p,Path(label)/p.relative_to(root))
prep=read(Q/'preparation.json');pins={}
def pin(p,h):
 p=Path(p);key=str(p);assert key not in pins or pins[key]==h;pins[key]=h
for items in (plan['pins'],c['sources'],audit['source_pins'],prep['source_pins']):
 for p,h in items.items():pin(p,h)
for name in ('model_manifest','source_manifest','fixtures'):
 ref=prep[name];pin(ref['path'],ref['sha256'])
childplan=read(P/'plan.json');pin(childplan['e024_plan']['path'],childplan['e024_plan']['sha256'])
e024=read(P/'inherited-e024-plan.json');pin(e024['e023_plan']['path'],e024['e023_plan']['sha256'])
core=read(Path(prep['source_manifest']['path']))
for name,h in core['files'].items():pin(R/'tutorial_code/miniengine'/name,h)
for p,h in pins.items():
 p=Path(p);assert sha(p)==h;copy(p,Path('sources')/p.relative_to(R))
runtime=[]
for name,h in prep['runtime_pins'].items():
 p=Path(name);record=dict(path=name,sha256=h,bytes=p.stat().st_size,scope='Frozen producer runtime manifest; not a new binary/runtime qualification.')
 if p.suffix in ('.py','.json') or p.name=='METADATA':
  assert sha(p)==h;copy(p,Path('runtime-text')/Path(name.lstrip('/')));record['text_sha_reread']=True
 else:record['payload_archived']=False;record['sha_reread_by_archiver']=False
 runtime.append(record)
save('runtime-references.json',runtime);save('manifest.json',dict(members=members,exclusions=['All96 original safetensors raw snapshots remain local; see raw-file-references.json.','Model weights and installed runtime binaries are not copied.']))
with tarfile.open(D/'evidence.tar.xz','w:xz',preset=6) as t:
 for m in members:t.add(stage/m['path'],arcname=m['path'],recursive=False)
expected={m['path']:m for m in members};seen=set()
with tarfile.open(D/'evidence.tar.xz','r|xz') as t:
 for info in t:
  m=expected[info.name];assert info.isfile() and info.name not in seen and info.size==m['bytes']
  assert hashlib.sha256(t.extractfile(info).read()).hexdigest()==m['sha256']==sha(Path(m['source']));seen.add(info.name)
assert seen==set(expected)
summary=dict(status='PASS_FORWARD_PREFIX_ONLY',producer_status=c['status'],independent_audit_status=audit['status'],cases=9,prefixes=42,output_tensor_pairs=5418,raw_files=96,raw_bytes_local=sum(r['bytes'] for r in raw),head_calls=9,parameter_value_receipts=1349,producer_elapsed_seconds=c['elapsed_wall_s'],audit_elapsed_seconds=ae['ended_at']-ae['started_at'],parent_child_lease_closed=True,service_restored_identity=final['service_identity_after'],resources=resources,forward_prefix_qualified=True,compiled_qualified=False,rollback_qualified=False,full_model_qualified=False,http_qualified=False,performance_qualified=False,limits=audit['limits'])
save('summary.json',summary)
v=dict(status='PASS_METADATA_RETENTION',members=len(members),uncompressed_metadata_bytes=sum(m['bytes'] for m in members),archive_bytes=(D/'evidence.tar.xz').stat().st_size,evidence_sha256=sha(D/'evidence.tar.xz'),manifest_sha256=sha(D/'manifest.json'),all_archived_members_reread_and_source_SHA_match=True,raw_files_payload_archived=False,raw_payload_rehashed_by_archiver=False,retained_staging=str(stage))
save('archive-verification.json',v);print(json.dumps(v));print(json.dumps({k:v for k,v in summary.items() if k!='limits'}))
