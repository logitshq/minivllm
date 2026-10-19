"""Close actual paired combined timing with independent raw HTTP/resource/restoration audits."""
from pathlib import Path
import concurrent.futures, hashlib, json, shutil, subprocess, tarfile, time
R=Path('/Users/ryanli/code/minivllm'); A=R/'.artifacts/m5-ultra-256gb'; B=R/'bench/m5_ultra_256gb'
OWNER=A/'gdn-graph-combined-formal-v2-owner'; JOURNAL=A/'gdn-graph-combined-formal-v2-cold-journal'; EXEC=A/'gdn-graph-combined-formal-v2-execution'; PREP=A/'gdn-graph-combined-formal-prepare-v2'
ANALYSIS=A/'gdn-graph-combined-formal-v2-analysis'; OUT=R/'docs/experiments/m5-ultra-256gb/E033-gdn-graph-combined-formal-v2'
OWNER_SHA=hashlib.sha256((PREP/'owner-plan.json').read_bytes()).hexdigest()
FORMAL_SHA='2391e9d4eab39df75064ae6ce07f6a0328e04ff21cbc977342d3f420d8e20b34'
RESTORE_SHA='7ab477f0d449bf08f78259a820d067f902d2032314ff4ebd389157b2578ec57d'
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
assert read(PREP/'receipt.json')['owner_plan_sha256']==OWNER_SHA=='f4e19bd0a842b48ea26fb2d27f870fdf62409897c7f862067c7800de5949b6f1'
assert f['prepare_returncode']==0
assert j['owner_plan']==dict(path=str(PREP/'owner-plan.json'),sha256=OWNER_SHA)
ANALYSIS.mkdir();shutil.copyfile(__file__,ANALYSIS/'close.py')
jobs=[('restoration-audit','gdn_graph_combined_formal_retry_v2/audit_restore.py',RESTORE_SHA,['--journal',str(JOURNAL)]),
 ('independent-formal','gdn_graph_combined_formal_retry_v2/audit_formal.py',FORMAL_SHA,['--owner',str(OWNER),'--journal',str(JOURNAL),'--plan-sha256',OWNER_SHA])]
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:results=dict(pool.map(lambda x:audit_job(*x),jobs))
write(ANALYSIS/'execution-summary.json',results)
print(json.dumps({k:dict(status=v['status'],returncode=v['returncode']) for k,v in results.items()}),flush=True)
OUT.mkdir();shutil.copyfile(__file__,OUT/'archive.py')
for name in ['restoration-audit.json','independent-formal.json','execution-summary.json']:shutil.copyfile(ANALYSIS/name,OUT/name)
shutil.copyfile(JOURNAL/'journal.json',OUT/'restored-journal.json')
summary=dict(schema='GDN-graph-combined-formal-archive-v2',owner_plan_sha256=OWNER_SHA,
 owner_status=read(OWNER/'complete.json')['status'],audit_statuses={k:v['status'] for k,v in results.items()},
 status='PASS' if all(v['status']=='PASS' and v['returncode']==0 for v in results.values()) else 'FAIL',
 performance_qualified=read(ANALYSIS/'independent-formal.json').get('performance_qualified',False),scope='Twelve fresh processes, same combined server, graph always ON and GDN OFF/ON, three pairs each C1/C8, all observers OFF. Each process serves8warm+64formal requests. Independent audit rereads raw HTTP/count/window/latency, resource and restoration evidence. Numerical prerequisites are the prior captured histories, not every future schedule; formal graph hits are not observed.')
write(OUT/'summary.json',summary)
(OUT/'README.md').write_text('# E033: Incremental GDN on graph paired timing retry\n\n'+summary['scope']+'\n\nAll retry owner, journal, execution, preparation and offline-audit artifacts are retained. Failed original CLI invocation remains in ../E032-gdn-graph-combined-formal-v1/. Focused CPU history is in ../combined-formal-retry-v2-cpu-v1/. Original factory and12 specs are unchanged. The manifest records original paths and SHA256; the compressed stream is independently verified member by member.\n')
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
for name in ['audit_formal.py','test_audit_formal_cpu.py']:files['sources/'+name]=B/'gdn_graph_combined_formal_retry_v2'/name
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
