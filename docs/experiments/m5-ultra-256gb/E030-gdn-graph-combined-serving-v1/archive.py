"""Offline closed-cycle audit and exact evidence retention; no device work."""
from pathlib import Path
import concurrent.futures, hashlib, json, os, shutil, subprocess, tarfile, time
R=Path('/Users/ryanli/code/minivllm'); A=R/'.artifacts/m5-ultra-256gb'; B=R/'bench/m5_ultra_256gb'
OWNER=A/'gdn-graph-combined-v1-owner'; JOURNAL=A/'gdn-graph-combined-v1-cold-journal'; EXEC=A/'gdn-graph-combined-v1-execution'; PREP=A/'gdn-graph-combined-prepare-v1'
ANALYSIS=A/'gdn-graph-combined-v1-analysis'; OUT=R/'docs/experiments/m5-ultra-256gb/E030-gdn-graph-combined-serving-v1'
CHILD=OWNER/'phases/gdn-graph-combined-serving/qualification'
OWNER_SHA='a2af16324fb323dcd9a93e4b9f07d4f127415bcaa0a12f04ca01792caf2696d9'; CHILD_SHA='fa59abb503725b060dd7e26f87e178bb1770178c6094b359fb666c21cac76dcf'
def read(p):return json.loads(Path(p).read_bytes())
def write(p,x):
 with Path(p).open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def pin(p):
 p=Path(p);s=p.stat();h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 n=p.stat();assert (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)==(n.st_dev,n.st_ino,n.st_size,n.st_mtime_ns),str(p)+' changed'
 return {'sha256':h.hexdigest(),'bytes':s.st_size}
def audit_job(name,source,expected,tail):
 sp=B/source;assert pin(sp)['sha256']==expected
 target=ANALYSIS/(name+'.json');cmd=[str(R/'.venv/bin/python'),'-I','-B',str(sp),*tail,'--out',str(target)]
 row={'command':cmd,'source':{'path':str(sp),**pin(sp)},'started_at':time.time()};write(ANALYSIS/(name+'-invocation.json'),row)
 with (ANALYSIS/(name+'.stdout')).open('x') as out,(ANALYSIS/(name+'.stderr')).open('x') as err:
  row['returncode']=subprocess.run(cmd,stdout=out,stderr=err,timeout=300).returncode
 row.update(ended_at=time.time(),output={'path':str(target),**pin(target)},status=read(target)['status'])
 write(ANALYSIS/(name+'-execution.json'),row);return name,row
j=read(JOURNAL/'journal.json');f=read(EXEC/'final.json')
assert j['state']==f['journal_state']=='RESTORED','cycle not closed'
assert pin(PREP/'owner-plan.json')['sha256']==OWNER_SHA and pin(CHILD/'plan.json')['sha256']==CHILD_SHA
assert j['owner_plan']=={'path':str(PREP/'owner-plan.json'),'sha256':OWNER_SHA}
ANALYSIS.mkdir();shutil.copyfile(__file__,ANALYSIS/'close.py')
jobs=[('restoration-audit','cold_observed_cycle/audit_restore.py','f7e5e7e751b52520a47facfdb65c33db40fb4c103e8ee272b5d154607087ce56',['--journal',str(JOURNAL)]),
 ('resource-summary','resource_reparse.py','5948ccbf7baeb21d1e71d9e8f7169b1bd9f84e65eb492746aba86fe377806c23',['--owner',str(OWNER)]),
 ('independent-numerical','gdn_graph_combined/audit.py','7e7718ddf1e1b6d744ec374e19b5513248f0374306503aeb8908391edb403270',['--child',str(CHILD),'--plan-sha256',CHILD_SHA])]
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
 results=dict(pool.map(lambda x:audit_job(*x),jobs))
write(ANALYSIS/'execution-summary.json',results)
print(json.dumps({k:{'status':v['status'],'returncode':v['returncode']} for k,v in results.items()}),flush=True)
assert results['restoration-audit']['status']=='PASS','restoration audit did not pass'
OUT.mkdir();shutil.copyfile(__file__,OUT/'archive.py')
for name in ['restoration-audit.json','resource-summary.json','independent-numerical.json','execution-summary.json']:shutil.copyfile(ANALYSIS/name,OUT/name)
shutil.copyfile(JOURNAL/'journal.json',OUT/'restored-journal.json')
shutil.copyfile(CHILD/'plan.json',OUT/'qualification-plan.json')
shutil.copyfile(CHILD/'complete.json',OUT/'qualification-complete.json')
summary={'schema':'GDN-graph-combined-serving-archive-v1','owner_plan_sha256':OWNER_SHA,'child_plan_sha256':CHILD_SHA,
 'owner_status':read(OWNER/'complete.json')['status'],'child_status':read(CHILD/'complete.json')['status'],
 'audit_statuses':{k:v['status'] for k,v in results.items()},'http_qualified':False,'performance_qualified':False,
 'scope':'Original42 parameter diagnostics and37 same-schedule numerical receipts with independent route/GDN trace reconstruction. Successful full tensor bytes were compared on device, not retained for offline recomputation. Resource and restoration outcomes are separate. No HTTP or timing qualification.'}
write(OUT/'summary.json',summary)
(OUT/'README.md').write_text('# E030: GDN and long decode graph composition\n\n'+summary['scope']+'\n\nAll first-attempt owner, journal, execution, preparation and offline-audit artifacts are retained. Earlier integration/qualifier/recovery CPU evidence is in [prepared CPU](../gdn-graph-combined-prepared-cpu-2026-10-18/README.md); the nine independent-auditor CPU checks are also included here. The manifest records original paths and SHA256; the compressed stream is independently verified member by member.\n')
roots={'owner':OWNER,'cold-journal':JOURNAL,'execution':EXEC,'preparation':PREP,'analysis':ANALYSIS,'auditor-cpu':A/'gdn-graph-combined-audit-cpu-v1'}
files={}
for prefix,root in roots.items():
 for p in sorted(root.rglob('*')):
  assert not p.is_symlink(),str(p)
  if p.is_file():
   assert p.suffix not in {'.dylib','.so','.bin','.metallib'},str(p)
   if p.suffix=='.safetensors':assert p==CHILD/'base/failure-arrays.safetensors' and read(p.with_name('failure.json'))['array_artifact']=={'path':p.name,'sha256':pin(p)['sha256']}
   files[prefix+'/'+p.relative_to(root).as_posix()]=p
for name in ['audit.py','test_audit_cpu.py']:files['sources/'+name]=B/'gdn_graph_combined'/name
for p in sorted(OUT.iterdir()):files['archive-metadata/'+p.name]=p
manifest={name:{'source':str(p),**pin(p)} for name,p in files.items()};write(OUT/'manifest.json',manifest)
files['manifest.json']=OUT/'manifest.json'
archive=OUT/'evidence.tar.xz'
with tarfile.open(archive,'w:xz',preset=1) as tar:
 for name,p in files.items():tar.add(p,arcname=name,recursive=False)
checked=0
with tarfile.open(archive,'r|xz') as tar:
 for member in tar:
  assert member.isfile() and member.name in files
  src=tar.extractfile(member);h=hashlib.sha256();size=0
  for chunk in iter(lambda:src.read(8<<20),b''):h.update(chunk);size+=len(chunk)
  expected=pin(files[member.name]);assert {'sha256':h.hexdigest(),'bytes':size}==expected
  if member.name!='manifest.json':assert all(expected[k]==manifest[member.name][k] for k in expected)
  checked+=1
assert checked==len(files)
write(OUT/'archive-verification.json',{'status':'PASS','members':checked,'archive':pin(archive),'raw_bytes':sum(x['bytes'] for x in manifest.values()),'scope':'Retention only, every member and original SHA checked; no qualification promotion.'})
print(json.dumps({'archive':str(OUT),'members':checked,'archive_bytes':archive.stat().st_size}),flush=True)
