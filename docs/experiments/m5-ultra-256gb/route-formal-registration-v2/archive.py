"""Snapshot actual PREPARED formal-v2 registration; no backend/service operation."""
from pathlib import Path, PurePosixPath
import hashlib,json,tarfile,sys
R=Path('/Users/ryanli/code/minivllm');A=R/'.artifacts/m5-ultra-256gb';P=R/'bench/m5_ultra_256gb/route_http_formal/prereg-formal-v2';J=A/'route-http-formal-v2-cold-journal';O=R/'docs/experiments/m5-ultra-256gb/route-formal-registration-v2'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_bytes())
def info(p):return dict(sha256=sha(p),bytes=p.stat().st_size)
def write(p,x):
 with p.open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def listing(root):
 files=[]
 for p in sorted(root.rglob('*')):
  assert not p.is_symlink() and (p.is_file() or p.is_dir())
  if p.is_file():files.append(p)
 return files
if sys.argv[1:]!=['--run']:raise SystemExit('DESCRIPTION_ONLY')
assert not O.exists()
r=read(P/'receipt.json');j=read(J/'journal.json');reg=read(A/'route-http-formal-v2-registration/result.json')
assert r['status']==j['state']=='PREPARED' and reg['status']=='PASS'
assert sha(P/'owner-plan.json')==r['owner_plan_sha256']==reg['owner_plan_sha256']=='0920aceae4e8bb6ef5c68577d3ae905a2b5976287790a7bb7a4a6e4ea36470ce'
assert sha(J/'journal.json')==r['prepared_journal_sha256']=='da0e9518daecf71fabe6bc3f1a4ab51c2a79fc2310912938787a12f05a50bf33'
assert j['original']['pid']==43565 and len(j['trials'])==12
assert reg['service_changed'] is reg['service_prepared'] is False and reg['source_unchanged']
for path,digest in r['source_pins'].items():assert sha(Path(path))==digest
roots={'registration':A/'route-http-formal-v2-registration','preregistration':P,'cpu-prepare':A/'route-http-formal-v2-cpu-prepare','prepared-journal':J}
sets={name:listing(root) for name,root in roots.items()}
inputs={name+'/'+str(p.relative_to(roots[name])):p for name,paths in sets.items() for p in paths}
O.mkdir();(O/'archive.py').write_bytes(Path(__file__).read_bytes())
for name,p in [('registration-result.json',A/'route-http-formal-v2-registration/result.json'),('prepared-journal.json',J/'journal.json'),('receipt.json',P/'receipt.json')]:
 (O/name).write_bytes(p.read_bytes())
summary=dict(status='PREPARED',owner_plan_sha256=r['owner_plan_sha256'],prepared_journal_sha256=r['prepared_journal_sha256'],source_pins=len(r['source_pins']),trials=12,warm_requests=96,formal_requests=768,original_pid=43565,device_executed=False,performance_qualified=False,prior_registration_failure_preserved=True)
write(O/'summary.json',summary)
(O/'README.md').write_text('# E024 formal-v2 actual registration\n\nThe corrected registration and owner CPU preparation actually passed after all 64 guarded CPU tests passed. The read-only service preparation bound restored PID 43565; the real PREPARED journal was then sealed. No service stop or timing occurred during this snapshot. Twelve original OFF/ON specifications retain 96 warm and 768 formal requests.\n\nThe first registration failure remains in the separate immutable failure archive. This snapshot retains the complete successful guarded registration, preregistration source copies, CPU prepare, and actual service PREPARED journal. The later execution has separate outputs. The archive is retention evidence, not a performance result.\n')
for p in listing(O):inputs['archive-metadata/'+p.name]=p
manifest={name:dict(source=str(p),**info(p)) for name,p in inputs.items()};write(O/'manifest.json',manifest)
target=O/'evidence.tar.xz'
with tarfile.open(target,'w:xz',dereference=True,preset=1) as tar:
 for name,p in sorted(inputs.items()):tar.add(p,arcname=name,recursive=False)
seen=set()
with tarfile.open(target,'r:xz') as tar:
 for m in tar:
  path=PurePosixPath(m.name);assert m.isfile() and not path.is_absolute() and '..' not in path.parts and m.name in manifest and m.name not in seen
  data=tar.extractfile(m).read();meta=manifest[m.name];assert len(data)==meta['bytes'] and hashlib.sha256(data).hexdigest()==meta['sha256'];seen.add(m.name)
assert seen==set(manifest)
for name,p in inputs.items():assert info(p)=={k:manifest[name][k] for k in ('sha256','bytes')}
for name,root in roots.items():assert listing(root)==sets[name]
v=dict(status='PASS',registration_status='PREPARED',members=len(seen),raw_bytes=sum(x['bytes'] for x in manifest.values()),archive=info(target),all_members_rechecked=True,all_originals_rechecked=True,performance_qualified=False)
write(O/'archive-verification.json',v);print(json.dumps(v))
