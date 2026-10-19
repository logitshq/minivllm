"""Explicit small evidence archive; exclude both real model tensor payload slices."""
import hashlib,json,shutil,subprocess,sys,tarfile,tempfile
from pathlib import Path
R=Path('/Users/ryanli/code/minivllm');A=R/'.artifacts/m5-ultra-256gb';D=Path(__file__).resolve().parent
S={'probe':A/'real-projection-probe-v1','execution':A/'real-projection-probe-execution-v1','lease':A/'real-projection-probe-lease-v1','cpu':A/'real-projection-probe-cpu-v1'}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,value):
 with (D/name).open('x') as f:json.dump(value,f,indent=2);f.write('\n')
stage=Path(tempfile.mkdtemp(prefix='e038-evidence-',dir='/private/tmp'));members=[];excluded=[]
def copy(p,rel):
 h=sha(p);q=stage/rel;q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q);assert sha(q)==h==sha(p)
 members.append(dict(path=str(rel),source=str(p),bytes=q.stat().st_size,sha256=h))
for label,root in S.items():
 for p in sorted(root.rglob('*')):
  if not p.is_file():continue
  assert not p.is_symlink()
  if p.name=='selected-weights.safetensors':excluded.append(dict(path=str(p),bytes=p.stat().st_size,reason='Real model tensor payload; hashes retained in weights.json, no payload reread/archive.'));continue
  copy(p,Path(label)/p.relative_to(root))
plan=json.loads((S['execution']/'plan.json').read_text());inv=json.loads((S['probe']/'invocation.json').read_text())
for original,h in {**plan['pins'],**inv['pins']}.items():
 p=Path(original);assert sha(p)==h;copy(p,Path('sources')/p.relative_to(R))
assert len(excluded)==2
save('manifest.json',dict(members=members,excluded=excluded));cmd=[sys.executable,'-I','-B',str(D/'audit.py'),'--evidence',str(stage),'--out',str(D/'independent-audit.json')]
save('audit-command.json',dict(argv=cmd,cwd=str(R),audit_sha256=sha(D/'audit.py'),archive_sha256=sha(Path(__file__))))
with (D/'audit-stdout.txt').open('x') as out,(D/'audit-stderr.txt').open('x') as err:result=subprocess.run(cmd,cwd=R,stdout=out,stderr=err)
save('audit-execution.json',dict(returncode=result.returncode,scope='Offline reader only; no backend/model/service execution.'))
with tarfile.open(D/'evidence.tar.xz','w:xz',preset=6) as t:
 for m in members:t.add(stage/m['path'],arcname=m['path'],recursive=False)
expected={m['path']:m for m in members};seen=set()
with tarfile.open(D/'evidence.tar.xz','r|xz') as t:
 for info in t:
  m=expected[info.name];assert info.isfile() and info.name not in seen and info.size==m['bytes']
  assert hashlib.sha256(t.extractfile(info).read()).hexdigest()==m['sha256']==sha(Path(m['source']));seen.add(info.name)
assert seen==set(expected)
v=dict(status='PASS' if result.returncode==0 else 'FAIL',members=len(members),uncompressed_bytes=sum(m['bytes'] for m in members),archive_bytes=(D/'evidence.tar.xz').stat().st_size,evidence_sha256=sha(D/'evidence.tar.xz'),manifest_sha256=sha(D/'manifest.json'),independent_audit_sha256=sha(D/'independent-audit.json'),all_members_reread_and_original_SHA_match=True,excluded_model_bytes=sum(x['bytes'] for x in excluded),retained_staging=str(stage))
save('archive-verification.json',v);print(json.dumps(v));raise SystemExit(result.returncode)
