from pathlib import Path
import hashlib,json,shutil,tarfile
R=Path('/Users/ryanli/code/minivllm');A=R/'.artifacts/m5-ultra-256gb';H=R/'bench/m5_ultra_256gb/gdn_graph_combined_formal_retry_v2';P=A/'gdn-graph-combined-formal-prepare-v2';J=A/'gdn-graph-combined-formal-v2-cold-journal'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_bytes())
def ref(p):return dict(path=str(p),sha256=sha(p))
def write(p,x):
 with p.open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
pre=read(P/'result.json');plan=read(P/'owner-plan.json');journal=read(J/'journal.json');cpu=A/'combined-formal-retry-v2-cpu-v1/result.json'
assert pre['status']=='PASS' and pre['source_unchanged'] and pre['original_specs_unchanged'] and len(pre['specs'])==12 and pre['denied_operations']==[]
assert read(cpu)['status']=='PASS' and read(cpu)['tests_run']==19 and read(cpu)['source_unchanged']
assert read(P/'controller-prepare-result.json')['returncode']==0
assert journal['state']=='PREPARED' and journal['owner_plan']==ref(P/'owner-plan.json') and journal['original']['pid']==91373
assert journal['controller_sha256']==sha(H/'controller.py') and journal['owner_source']==plan['execution']['owner']
(P/'register.py').write_bytes(Path(__file__).read_bytes())
pins={**plan['execution']['pins'],str(H/'execute.py'):sha(H/'execute.py'),str(P/'owner-plan.json'):sha(P/'owner-plan.json'),str(P/'prepare.py'):sha(P/'prepare.py'),str(P/'register.py'):sha(P/'register.py')}
assert all(sha(p)==h for p,h in pins.items())
value=dict(schema='GDN-graph-combined-formal-prereg-receipt-v2',status='PREPARED',owner_plan_sha256=sha(P/'owner-plan.json'),prepared_journal_sha256=sha(J/'journal.json'),source_pins=pins,actual_cpu_prepare=ref(P/'result.json'),focused_cpu=ref(cpu),read_only_service_prepare=ref(P/'controller-prepare-result.json'),device_run=False,performance_qualified=False)
write(P/'receipt.json',value)
D=H/'prereg-formal-v2';D.mkdir()
for p in sorted(P.iterdir()):
 if p.is_file():shutil.copyfile(p,D/p.name)
shutil.copyfile(J/'journal.json',D/'prepared-journal.json')
manifest={p.name:dict(source=str(P/p.name) if p.name!='prepared-journal.json' else str(J/'journal.json'),sha256=sha(p),bytes=p.stat().st_size) for p in sorted(D.iterdir())}
write(D/'manifest.json',manifest)
(D/'README.md').write_text('# Combined formal retry v2 registration\n\nCPU input preparation and read-only controller preparation PASS; no device timing result at registration. Actual parser/driver focused CPU suite19/19PASS. The driver will invoke real owner --prepare-only before stopping the service.\n\nAll12 original specs are referenced unchanged by owner-plan.json, and are already retained under the original formal preregistration. The new lifecycle binds the independently restored E032 no-experiment failure. Do not rewrite v1. Actual next archive is E033-gdn-graph-combined-formal-v2.\n')
roots=[('combined-formal-retry-v2-cpu-v1','combined-formal-retry-v2-cpu-v1'),('glm-v3-registration-cpu-v1','glm-v3-registration-cpu-v1')]
for name,source in roots:
 origin=A/source;dest=R/'docs/experiments/m5-ultra-256gb'/name;dest.mkdir()
 files={p.relative_to(origin).as_posix():p for p in sorted(origin.rglob('*')) if p.is_file()}
 assert all(not p.is_symlink() for p in files.values())
 m={k:dict(source=str(p),sha256=sha(p),bytes=p.stat().st_size) for k,p in files.items()};write(dest/'manifest.json',m)
 shutil.copyfile(origin/'result.json',dest/'result.json');shutil.copyfile(__file__,dest/'retain.py')
 archive=dest/'evidence.tar.xz'
 with tarfile.open(archive,'w:xz',preset=1) as tar:
  for name2,p in files.items():tar.add(p,arcname=name2,recursive=False)
 seen=[]
 with tarfile.open(archive,'r|xz') as tar:
  for member in tar:
   assert member.isfile() and member.name in m
   data=tar.extractfile(member).read();meta=m[member.name]
   assert len(data)==meta['bytes'] and hashlib.sha256(data).hexdigest()==meta['sha256']==sha(files[member.name]);seen.append(member.name)
 assert len(seen)==len(m)
 write(dest/'archive-verification.json',dict(status='PASS',members=len(m),raw_bytes=sum(x['bytes'] for x in m.values()),archive=dict(sha256=sha(archive),bytes=archive.stat().st_size),scope='Retention only; CPU tests are not model/service/performance qualification.'))
 (dest/'README.md').write_text('# '+name+'\n\nFirst focused CPU run and exact source/output snapshots retained; see result.json. Every archive member and original SHA verified. No device or performance qualification.\n')
 print(json.dumps(dict(archive=str(dest),members=len(m),bytes=archive.stat().st_size)))
print(json.dumps(dict(receipt_sha256=sha(P/'receipt.json'),owner_plan_sha256=value['owner_plan_sha256'],pins=len(pins))))
