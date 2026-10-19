"""Archive E037 raw artifacts and run only the independent standard-library audit."""
import hashlib,json,shutil,subprocess,sys,tarfile,tempfile,time
from pathlib import Path
R=Path('/Users/ryanli/code/minivllm');A=R/'.artifacts/m5-ultra-256gb';D=Path(__file__).resolve().parent
SOURCES={'device':A/'nvfp4-multirow-reuse-device-v1','execution':A/'nvfp4-multirow-reuse-device-execution-v1',
         'smoke-cpu':A/'nvfp4-multirow-smoke-cpu-v1','mechanism-cpu':A/'nvfp4-multirow-reuse-cpu-v1'}
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def write(p,v):
 with p.open('x') as f:json.dump(v,f,indent=2);f.write('\n')
stage=Path(tempfile.mkdtemp(prefix='e037-evidence-',dir='/private/tmp'));members=[]
write(D/'archive-command.json',dict(argv=[sys.executable,'-I','-B',str(Path(__file__).resolve())],cwd=str(R),started_at=time.time(),staging=str(stage),sources={k:str(v) for k,v in SOURCES.items()}))
for label,source in SOURCES.items():
 for p in sorted(source.rglob('*')):
  assert not p.is_symlink(),p
  if not p.is_file():continue
  rel=Path(label)/p.relative_to(source);q=stage/rel;q.parent.mkdir(parents=True,exist_ok=True)
  h=sha(p);shutil.copyfile(p,q);assert sha(q)==h==sha(p)
  members.append(dict(path=str(rel),source=str(p),bytes=q.stat().st_size,sha256=h))
write(D/'manifest.json',dict(schema='E037-evidence-manifest-v1',members=members))
cmd=[sys.executable,'-I','-B',str(D/'audit.py'),'--evidence',str(stage),'--out',str(D/'independent-audit.json')]
write(D/'audit-command.json',dict(argv=cmd,cwd=str(R),auditor_sha256=sha(D/'audit.py')))
with (D/'audit-stdout.txt').open('x') as stdout,(D/'audit-stderr.txt').open('x') as stderr:
 result=subprocess.run(cmd,cwd=R,stdout=stdout,stderr=stderr)
write(D/'audit-execution.json',dict(returncode=result.returncode,scope='Independent CPU raw-byte reader only; no backend/model/GPU execution.'))
with tarfile.open(D/'evidence.tar.xz','w:xz',preset=6) as tar:
 for m in members:tar.add(stage/m['path'],arcname=m['path'],recursive=False)
expected={m['path']:m for m in members};seen=set();total=0
with tarfile.open(D/'evidence.tar.xz','r|xz') as tar:
 for info in tar:
  assert info.isfile() and info.name in expected and info.name not in seen
  m=expected[info.name];assert info.size==m['bytes'];h=hashlib.sha256()
  stream=tar.extractfile(info)
  for b in iter(lambda:stream.read(1<<20),b''):h.update(b)
  assert h.hexdigest()==m['sha256']==sha(Path(m['source']));seen.add(info.name);total+=info.size
assert seen==set(expected)
summary=json.loads((D/'independent-audit.json').read_text())
verification=dict(status='PASS' if result.returncode==0 and summary['status']=='PASS' else 'FAIL',
 members=len(seen),uncompressed_bytes=total,archive_bytes=(D/'evidence.tar.xz').stat().st_size,
 evidence_sha256=sha(D/'evidence.tar.xz'),manifest_sha256=sha(D/'manifest.json'),
 all_members_reread_and_source_hashes_match=True,audit_returncode=result.returncode,
 independent_audit_sha256=sha(D/'independent-audit.json'),retained_staging=str(stage),scope='No original artifact modified; no device rerun.')
write(D/'archive-verification.json',verification)
print(json.dumps(verification,sort_keys=True));raise SystemExit(0 if verification['status']=='PASS' else 1)
