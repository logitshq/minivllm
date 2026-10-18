import hashlib,json,tarfile
from pathlib import Path
R=Path('/Users/ryanli/code/minivllm');A=R/'.artifacts/m5-ultra-256gb';D=R/'docs/experiments/m5-ultra-256gb'
def info(p):
 b=Path(p).read_bytes();return dict(sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))
def write(p,d):
 with p.open('x') as f:json.dump(d,f,indent=2);f.write('\n')
out=D/'gdn-formal-continuation-v1';out.mkdir()
for name in ('audit-closed-gdn-formal-v1.py','archive-gdn-formal-v1.py','close-gdn-formal-after-run-v1.py'):
 p=Path('/tmp')/name;(out/name).write_bytes(p.read_bytes())
for path,name in ((A/'gdn-http-formal-v1-continuation/result.json','startup-continuation.json'),(A/'gdn-http-formal-v1-execution/owner-invocation.json','startup-owner-invocation.json'),(A/'gdn-http-formal-v1-registration/result.json','registration-result.json')):
 (out/name).write_bytes(path.read_bytes())
(out/'README.md').write_text('''# One-cycle GDN formal follow-through\n\nThese are exact source and point-in-time startup snapshots for the already running twelve-phase benchmark. They do not assert completion. Commit e64ded9 preregistered plan02337f0267a327ad7b025fd90ae18dec124cfe139811a767e9597c2b27af52f5 before the run started at2026-10-18 06:36:50 UTC.\n\nThe original execution driver owns stop/experiment/finally-recovery. The separate bounded continuation waits for that actual driver final RESTORED, then invokes frozen offline restoration/formal audits and preserves either successful or failed evidence. It starts no model, service, retry, or new experiment. A nonzero audit/pack return never becomes PASS. No timing point is granted by this startup package. The final result stays in .artifacts/m5-ultra-256gb/gdn-http-formal-v1-continuation; completed evidence will be under E029-gdn-http-formal-v1 only if recovery and retention checks permit it.\n''')
write(out/'manifest.json',{p.name:info(p) for p in sorted(out.iterdir()) if p.is_file()})
# Two small CPU test attempts, including original fixture error. No test rerun.
out=D/'glm-cycle-v2-prepared-cpu-history-v1';out.mkdir()
trees={f'attempt-{n}':A/f'glm-cycle-v2-prepared-cpu-v{n}' for n in (1,2)}
source={label+'/'+str(p.relative_to(root)):p for label,root in trees.items() for p in sorted(root.rglob('*')) if p.is_file()}
assert all(not p.is_symlink() and p.suffix not in ('.safetensors','.bin','.so','.dylib','.metallib') for p in source.values())
manifest={name:dict(source=str(p),**info(p)) for name,p in source.items()}
write(out/'manifest.json',dict(files=manifest))
target=out/'evidence.tar.xz'
with tarfile.open(target,'x:xz',preset=1,dereference=True) as tar:
 for name,p in source.items():tar.add(p,arcname=name,recursive=False)
seen=set()
with tarfile.open(target,'r|xz') as tar:
 for member in tar:
  assert member.isfile() and member.name in manifest and member.name not in seen
  b=tar.extractfile(member).read();assert dict(sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))=={k:manifest[member.name][k] for k in ('sha256','bytes')};seen.add(member.name)
assert seen==set(source)
assert all(info(p)=={k:manifest[n][k] for k in ('sha256','bytes')} for n,p in source.items())
assert set(source)=={label+'/'+str(p.relative_to(root)) for label,root in trees.items() for p in root.rglob('*') if p.is_file()}
write(out/'archive-verification.json',dict(status='PASS',archive=info(target),members=len(source),raw_bytes=sum(v['bytes'] for v in manifest.values()),all_members_verified=True,original_files_rechecked=True,tree_membership_rechecked=True,tests_rerun=False,model_run=False))
(out/'archive.py').write_bytes(Path(__file__).read_bytes())
(out/'README.md').write_text('''# GLM v2 adapter CPU test history\n\nAttempt1:15 PASS and1 fixture ERROR, retained unchanged. The negative fixture replaced an external plan with an incomplete dictionary, so validation raised KeyError earlier than the fixture expected. Only that fake plan was repaired to retain its original schema and add a changed value, allowing the intended SHA rejection to be exercised. No production code or validation gate changed. Attempt2:16/16 PASS,0 errors/failures/skips. Both actual guarded commands, stdout/stderr, source snapshots and before/after old-tree inventories are retained.\n\nThis archive was reopened member by member and checked against the original bytes/SHA, then original files and tree membership were rechecked. The archive step reran no tests. Neither attempt imported a device backend or performed real subprocess/socket/signal/model/service work. This evidence does not prove GLM load, model correctness, restoration, or speed.\n''')
print(json.dumps(dict(continuation='startup snapshots preserved',glm_cpu_archive=json.loads((out/'archive-verification.json').read_bytes()))))
