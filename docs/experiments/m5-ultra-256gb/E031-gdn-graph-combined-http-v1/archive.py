"""Close the actual combined HTTP cycle with independent audits and retained evidence."""
from pathlib import Path
import concurrent.futures, hashlib, json, shutil, subprocess, tarfile, time
R=Path('/Users/ryanli/code/minivllm'); A=R/'.artifacts/m5-ultra-256gb'; B=R/'bench/m5_ultra_256gb'
OWNER=A/'gdn-graph-combined-http-v1-owner'; JOURNAL=A/'gdn-graph-combined-http-v1-cold-journal'; EXEC=A/'gdn-graph-combined-http-v1-execution'; PREP=A/'gdn-graph-combined-http-prepare-v1'
ANALYSIS=A/'gdn-graph-combined-http-v1-analysis'; OUT=R/'docs/experiments/m5-ultra-256gb/E031-gdn-graph-combined-http-v1'
OWNER_SHA='7783b4c35798d590f1a326739c38a410ac59a3d0655cd384813d6e5b9f0a02e8'
CYCLE_SHA='45c98bf8d8eacbb07c060b05310c686320fd9cce3cd9bbaece0cd6fc887c492f'
NUMERIC_SHA='e41abc8e5073ca812ffdcfc398471a028a6bbd0e12c03a39ec75b0c0e4498717'
RESTORE_SHA='0946be64082a5a9264b04606b1a865bcaca1bf266aa3fb42888d2d21ad4bd184'
def read(p):return json.loads(Path(p).read_bytes())
def write(p,x):
 with Path(p).open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def pin(p):
 p=Path(p);s=p.stat();h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 n=p.stat();assert (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)==(n.st_dev,n.st_ino,n.st_size,n.st_mtime_ns),str(p)+' changed'
 return dict(sha256=h.hexdigest(),bytes=s.st_size)
def audit_job(name,source,expected,tail):
 sp=B/source;assert pin(sp)['sha256']==expected
 target=ANALYSIS/(name+'.json');cmd=[str(R/'.venv/bin/python'),'-I','-B',str(sp),*tail,'--out',str(target)]
 row=dict(command=cmd,source=dict(path=str(sp),**pin(sp)),started_at=time.time());write(ANALYSIS/(name+'-invocation.json'),row)
 with (ANALYSIS/(name+'.stdout')).open('x') as out,(ANALYSIS/(name+'.stderr')).open('x') as err:
  row['returncode']=subprocess.run(cmd,stdout=out,stderr=err,timeout=420).returncode
 row.update(ended_at=time.time(),output=dict(path=str(target),**pin(target)),status=read(target)['status'])
 write(ANALYSIS/(name+'-execution.json'),row);return name,row
j=read(JOURNAL/'journal.json');f=read(EXEC/'final.json')
assert j['state']==f['journal_state']=='RESTORED','cycle not closed'
assert pin(PREP/'owner-plan.json')['sha256']==OWNER_SHA
assert j['owner_plan']==dict(path=str(PREP/'owner-plan.json'),sha256=OWNER_SHA)
ANALYSIS.mkdir();shutil.copyfile(__file__,ANALYSIS/'close.py')
jobs=[('restoration-audit','gdn_graph_combined_http_cycle/audit_restore.py',RESTORE_SHA,['--journal',str(JOURNAL)])]
for C in (1,8):
 child=OWNER/'phases'/f'c{C}-replay/qualification'
 jobs.append((f'independent-c{C}','gdn_graph_combined/http_audit.py',NUMERIC_SHA,['--child',str(child),'--plan-sha256',pin(child/'plan.json')['sha256']]))
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:results=dict(pool.map(lambda x:audit_job(*x),jobs))
args=['--owner',str(OWNER),'--journal',str(JOURNAL),'--plan-sha256',OWNER_SHA]
for C in (1,8):
 path=ANALYSIS/f'independent-c{C}.json';args += [f'--c{C}-audit',str(path),f'--c{C}-audit-sha256',pin(path)['sha256']]
name,row=audit_job('independent-cycle','gdn_graph_combined_http_cycle/audit_cycle.py',CYCLE_SHA,args);results[name]=row
write(ANALYSIS/'execution-summary.json',results)
print(json.dumps({k:dict(status=v['status'],returncode=v['returncode']) for k,v in results.items()}),flush=True)
OUT.mkdir();shutil.copyfile(__file__,OUT/'archive.py')
for name in ['restoration-audit.json','independent-c1.json','independent-c8.json','independent-cycle.json','execution-summary.json']:shutil.copyfile(ANALYSIS/name,OUT/name)
shutil.copyfile(JOURNAL/'journal.json',OUT/'restored-journal.json')
summary=dict(schema='GDN-graph-combined-http-archive-v1',owner_plan_sha256=OWNER_SHA,
 owner_status=read(OWNER/'complete.json')['status'],audit_statuses={k:v['status'] for k,v in results.items()},
 status='PASS' if all(v['status']=='PASS' and v['returncode']==0 for v in results.values()) else 'FAIL',
 performance_qualified=False,scope='Actual C1/C8 HTTP requests followed by same-history numerical replay, independent raw HTTP/count/state/route, resource and restoration checks. Original successful full tensors were compared on device, not retained for offline recomputation. This is observer-enabled qualification, never a timing result.')
write(OUT/'summary.json',summary)
(OUT/'README.md').write_text('# E031: Combined GDN and decode graph HTTP qualification\n\n'+summary['scope']+'\n\nAll first-attempt owner, journal, execution, preparation and offline-audit artifacts are retained. Prepared CPU history is in ../gdn-graph-combined-http-prepared-cpu-2026-10-18/. The manifest records original paths and SHA256; the compressed stream is independently verified member by member.\n')
roots=dict(owner=OWNER,**{'cold-journal':JOURNAL},execution=EXEC,preparation=PREP,analysis=ANALYSIS)
files={}
for prefix,root in roots.items():
 for p in sorted(root.rglob('*')):
  assert not p.is_symlink(),str(p)
  if p.is_file():
   assert p.suffix not in {'.dylib','.so','.bin','.metallib'},str(p)
   if p.suffix=='.safetensors':
    assert p.name=='failure-arrays.safetensors' and read(p.with_name('failure.json'))['array_artifact']==dict(path=p.name,sha256=pin(p)['sha256'])
   files[prefix+'/'+p.relative_to(root).as_posix()]=p
for name in ['audit_cycle.py','test_audit_cycle_cpu.py']:files['sources/'+name]=B/'gdn_graph_combined_http_cycle'/name
for p in sorted(OUT.iterdir()):files['archive-metadata/'+p.name]=p
manifest={name:dict(source=str(p),**pin(p)) for name,p in files.items()};write(OUT/'manifest.json',manifest)
files['manifest.json']=OUT/'manifest.json';archive=OUT/'evidence.tar.xz'
with tarfile.open(archive,'w:xz',preset=1) as tar:
 for name,p in files.items():tar.add(p,arcname=name,recursive=False)
checked=0
with tarfile.open(archive,'r|xz') as tar:
 for member in tar:
  assert member.isfile() and member.name in files
  src=tar.extractfile(member);h=hashlib.sha256();size=0
  for chunk in iter(lambda:src.read(8<<20),b''):h.update(chunk);size+=len(chunk)
  expected=pin(files[member.name]);assert dict(sha256=h.hexdigest(),bytes=size)==expected
  if member.name!='manifest.json':assert all(expected[k]==manifest[member.name][k] for k in expected)
  checked+=1
assert checked==len(files)
write(OUT/'archive-verification.json',dict(status='PASS',members=checked,archive=pin(archive),raw_bytes=sum(x['bytes'] for x in manifest.values()),scope='Retention only; every member and original SHA checked, no qualification promotion.'))
print(json.dumps(dict(archive=str(OUT),members=checked,archive_bytes=archive.stat().st_size,status=summary['status'])),flush=True)
