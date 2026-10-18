"""Preserve explicitly named closed CPU attempts, including failures; --run."""
import argparse,hashlib,json,os,re,shutil,tarfile
from pathlib import Path
ROOT=Path('/Users/ryanli/code/minivllm');ART=ROOT/'.artifacts/m5-ultra-256gb'
DEST=ROOT/'docs/experiments/m5-ultra-256gb/next-modules-prepared-cpu-v1'
def need(ok,msg):
 if not ok:raise ValueError(msg)
def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  while part:=f.read(1<<20):h.update(part)
 return h.hexdigest()
def meta(path):return dict(sha256=digest(path),bytes=path.stat().st_size)
def read(path):return json.loads(path.read_bytes())
def save(path,value):
 with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def run(names):
 need(names and len(set(names))==len(names) and not DEST.exists(),'explicit unique attempts and fresh output')
 files={};results=[];trees={}
 for name in names:
  need(re.fullmatch(r'next-modules-cpu-v[0-9]+',name),'explicit CPU attempt directory name')
  source=ART/name;need(source.is_dir() and not source.is_symlink(),'regular attempt directory')
  r=read(source/'result.json');need(r['schema']=='m5-next-modules-cpu-v1' and r['status'] in ('PASS','FAIL') and 'ended_at' in r,'closed CPU result')
  need(r['numerical_qualified'] is False if 'numerical_qualified' in r else True,'no numerical claim')
  for absolute,m in r['source_before'].items():need(meta(source/'frozen-source'/absolute.lstrip('/'))==m,'exact original source snapshot')
  old=read(source/'artifact-manifest.json');need(old['schema']=='cpu-artifact-inventory-v1','original inventory')
  paths=[]
  for p in sorted(source.rglob('*')):
   need(not p.is_symlink(),'no symlink payload')
   if p.is_file():
    need(p.suffix not in ('.safetensors','.dylib','.so','.bin','.pack') and p.stat().st_size<32*(1<<20),'small CPU evidence only')
    rel=p.relative_to(source).as_posix();paths.append(rel)
    if rel!='artifact-manifest.json':need(rel in old['files'] and meta(p)==old['files'][rel],'original member SHA')
    files[name+'/'+rel]=dict(source=str(p),**meta(p))
  need(set(paths)==set(old['files'])|{'artifact-manifest.json'},'exact attempt tree')
  trees[name]=paths;results.append(dict(attempt=name,**r))
 DEST.mkdir();shutil.copyfile(__file__,DEST/'archive.py');save(DEST/'manifest.json',dict(schema='next-modules-cpu-archive-manifest-v1',files=files))
 archive=DEST/'evidence.tar.xz'
 with tarfile.open(archive,'x:xz',preset=3) as out:
  for name,m in files.items():
   p=Path(m['source']);need(meta(p)=={k:m[k] for k in ('sha256','bytes')},'source changed before archive')
   info=tarfile.TarInfo(name);info.size=m['bytes'];info.mode=0o644;info.mtime=0
   with p.open('rb') as f:out.addfile(info,f)
 seen=set()
 with tarfile.open(archive,'r:xz') as t:
  for member in t:
   need(member.isfile() and member.name in files and member.name not in seen,'regular unique archive member');seen.add(member.name)
   m=files[member.name];h=hashlib.sha256();n=0
   f=t.extractfile(member)
   while part:=f.read(1<<20):h.update(part);n+=len(part)
   need(n==m['bytes']==member.size and h.hexdigest()==m['sha256'],'archive member hash')
   need(meta(Path(m['source']))=={k:m[k] for k in ('sha256','bytes')},'original member unchanged')
 need(seen==set(files),'complete archive')
 for name,rels in trees.items():need(sorted(p.relative_to(ART/name).as_posix() for p in (ART/name).rglob('*') if p.is_file())==rels,'original tree unchanged')
 for r in results:shutil.copyfile(ART/r['attempt']/'result.json',DEST/(r['attempt']+'-result.json'))
 v=dict(schema='next-modules-cpu-archive-verification-v1',status='PASS',attempts=[{k:r[k] for k in ('attempt','status','tests','failures','errors','skipped','source_unchanged')} for r in results],members=len(files),raw_bytes=sum(m['bytes'] for m in files.values()),archive=dict(path=archive.name,**meta(archive)),manifest_sha256=digest(DEST/'manifest.json'),every_member_reread_sha_verified=True,every_original_file_rechecked=True,tree_membership_verified=True,numerical_qualified=False,http_qualified=False,performance_qualified=False,scope='Retention verification only; each original CPU attempt retains its own status and complete test outcomes.')
 save(DEST/'archive-verification.json',v);print(json.dumps(v))
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--attempt',action='append');p.add_argument('--run',action='store_true');a=p.parse_args()
 if a.run:run(a.attempt)
 else:print('DESCRIPTION_ONLY: no reads/writes beyond this source; use explicit closed attempt names and --run.')
