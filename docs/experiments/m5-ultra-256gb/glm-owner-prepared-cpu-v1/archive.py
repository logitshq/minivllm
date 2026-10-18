"""Retain the actual fake-only GLM owner test run. No model/runtime operations."""
from pathlib import Path, PurePosixPath
import hashlib,json,tarfile,sys,io
R=Path('/Users/ryanli/code/minivllm');A=R/'.artifacts/m5-ultra-256gb/glm-owner-prepared-cpu-v1';O=R/'docs/experiments/m5-ultra-256gb/glm-owner-prepared-cpu-v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def main():
 if sys.argv[1:]!=['--run']:print('DESCRIPTION_ONLY');return
 result=json.loads((A/'result.json').read_bytes())
 assert result['status']=='PASS' and result['tests']==12 and result['failures']==result['errors']==result['skipped']==0
 assert result['source_unchanged'] and not result['denied_operations'] and not result['forbidden_modules_loaded']
 manifest=json.loads((A/'artifact-manifest.json').read_bytes())['files']
 actual={str(p.relative_to(A)):p for p in A.rglob('*') if p.is_file() and p.name!='artifact-manifest.json'}
 assert set(actual)==set(manifest)
 for name,p in actual.items():assert not p.is_symlink() and manifest[name]==dict(sha256=sha(p),bytes=p.stat().st_size)
 O.mkdir(exist_ok=False)
 (O/'archive.py').write_bytes(Path(__file__).read_bytes())
 for name in ('result.json','stdout.txt','stderr.txt'):(O/name).write_bytes((A/name).read_bytes())
 (O/'README.md').write_text('# GLM direct-child owner: prepared CPU evidence\n\nAll 12 fake-only tests passed on 2026-10-17, with real subprocess/network/signals and model/backend imports denied. Coverage includes failure before spawn, unknown spawn, returned child with identity-read failure, bounded cleanup ordering, resource-checkpoint raw evidence and exact diagnostic scope. This is not an actual GLM load, resource-fit, numerical or performance result. The separate controller and real registration remain required.\n\nThe earlier restored E024 HTTP receipt is retained as historical preparation context; the CPU test did not stop or restore any service. A separate GDN numerical experiment was live, and no extra GPU operation was performed.\n\nThe archive contains the runner, stdout/stderr, every test outcome and source snapshots. manifest.json and archive-verification.json bind each member.\n')
 inputs={'cpu/'+str(p.relative_to(A)):p for p in A.rglob('*') if p.is_file()}
 for name in ('archive.py','README.md'):inputs['archive-metadata/'+name]=O/name
 records={name:dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size) for name,p in inputs.items()}
 assert all(not p.is_symlink() for p in inputs.values())
 write(O/'manifest.json',records)
 target=O/'evidence.tar.xz'
 with tarfile.open(target,'w:xz',preset=1) as tar:
  for name,p in sorted(inputs.items()):tar.add(p,arcname=name,recursive=False)
 seen=set()
 with tarfile.open(target,'r:xz') as tar:
  for m in tar:
   name=PurePosixPath(m.name)
   assert m.isfile() and not name.is_absolute() and '..' not in name.parts and m.name in records and m.name not in seen
   raw=tar.extractfile(m).read();assert len(raw)==records[m.name]['bytes'] and hashlib.sha256(raw).hexdigest()==records[m.name]['sha256'];seen.add(m.name)
 assert seen==set(records)
 for name,p in inputs.items():assert sha(p)==records[name]['sha256'] and p.stat().st_size==records[name]['bytes']
 verification=dict(status='PASS',tests=12,members=len(records),logical_bytes=sum(x['bytes'] for x in records.values()),archive_bytes=target.stat().st_size,archive_sha256=sha(target),all_members_rechecked=True,originals_rechecked=True,model_payloads_included=False,performance_qualified=False)
 write(O/'archive-verification.json',verification);print(json.dumps(verification))
if __name__=='__main__':main()
