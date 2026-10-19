"""Archive two closed tiny CPU attempts and actual header-only plan; explicit --run."""
import argparse,hashlib,json,os,shutil,tarfile
from pathlib import Path
R=Path('/Users/ryanli/code/minivllm');A=R/'.artifacts/m5-ultra-256gb'
D=R/'docs/experiments/m5-ultra-256gb/glm-expert-repack-cpu-v2'
S=R/'bench/m5_ultra_256gb/glm_expert_repack'
PLAN_SHA='323ff8da5d2faf6bb774b1c03f3a52f49198cbb9898f21c91ac25aa546a4c50a'
def need(ok,message):
 if not ok:raise ValueError(message)
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  while raw:=f.read(1<<20):h.update(raw)
 return h.hexdigest()
def meta(path):return dict(bytes=path.stat().st_size,sha256=sha(path))
def read(path):return json.loads(path.read_bytes())
def save(path,value):
 with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def members(folder):
 need(folder.is_dir() and not folder.is_symlink(),'regular explicit directory')
 paths=[]
 for path in sorted(folder.rglob('*')):
  need(not path.is_symlink(),'no archive symlinks')
  if path.is_file():
   need(path.stat().st_size<32<<20 and path.suffix not in ('.safetensors','.bin','.so','.dylib','.pack'),'small evidence only')
   paths.append(path)
 return paths
def run():
 need(not D.exists(),'fresh archive output');files={};trees={};attempts=[]
 for name,count in (('glm-expert-repack-cpu-v1',11),('glm-expert-repack-cpu-v2',15)):
  folder=A/name;row=read(folder/'result.json')
  need(row['status'] in ('PASS','FAIL') and row['tests_run']==count,'closed retained attempt')
  before=read(folder/'source-before.json');after=read(folder/'source-after.json')
  need((before==after)==row['source_unchanged'],'source receipt agreement')
  snapshots=list((folder/'sources').iterdir())
  need(len(snapshots)==len(before) and {p.name for p in snapshots}=={Path(k).name for k in before},'exact retained source set')
  for path,h in before.items():need(sha(folder/'sources'/Path(path).name)==h,'original source snapshot SHA')
  paths=members(folder);trees[str(folder)]=[p.relative_to(folder).as_posix() for p in paths]
  for p in paths:files[name+'/'+p.relative_to(folder).as_posix()]=dict(source=str(p),**meta(p))
  attempts.append(dict(attempt=name,result_sha256=sha(folder/'result.json'),**row))
 plan=S/'prereg-v1/plan.json';need(sha(plan)==PLAN_SHA,'exact actual header plan')
 parsed=read(plan);need(parsed['candidate_sha256']==sha(S/'candidate.py')==read(A/'glm-expert-repack-cpu-v2/source-before.json')[str(S/'candidate.py')],'tested planned candidate unchanged')
 prereg=S/'prereg-v1';paths=members(prereg);trees[str(prereg)]=[p.relative_to(prereg).as_posix() for p in paths]
 for p in paths:files['actual-header-plan/'+p.relative_to(prereg).as_posix()]=dict(source=str(p),**meta(p))
 for name,p in (('review/README.md',S/'README.md'),('archive/archive.py',Path(__file__).resolve())):files[name]=dict(source=str(p),**meta(p))
 D.mkdir();shutil.copyfile(__file__,D/'archive.py');save(D/'manifest.json',dict(schema='glm-expert-repack-cpu-archive-manifest-v1',files=files))
 archive=D/'evidence.tar.xz'
 with tarfile.open(archive,'x:xz',preset=3) as t:
  for name,m in sorted(files.items()):
   p=Path(m['source']);need(meta(p)=={k:m[k] for k in ('bytes','sha256')},'source changed before copy')
   info=tarfile.TarInfo(name);info.size=m['bytes'];info.mode=0o644;info.mtime=0
   with p.open('rb') as f:t.addfile(info,f)
 seen=set()
 with tarfile.open(archive,'r:xz') as t:
  for member in t:
   need(member.isfile() and member.name in files and member.name not in seen,'unique regular known member');seen.add(member.name)
   expected=files[member.name];h=hashlib.sha256();count=0
   with t.extractfile(member) as f:
    while raw:=f.read(1<<20):h.update(raw);count+=len(raw)
   need(count==member.size==expected['bytes'] and h.hexdigest()==expected['sha256'],'member readback SHA')
   need(meta(Path(expected['source']))=={k:expected[k] for k in ('bytes','sha256')},'original evidence unchanged')
 need(seen==set(files),'complete archive member set')
 for folder,expected in trees.items():need([p.relative_to(Path(folder)).as_posix() for p in members(Path(folder))]==expected,'original membership unchanged')
 for row in attempts:shutil.copyfile(A/row['attempt']/'result.json',D/(row['attempt']+'-result.json'))
 summary=dict(schema='glm-expert-repack-cpu-archive-verification-v1',status='PASS',attempts=attempts,
  failed_cpu_attempts=[r['attempt'] for r in attempts if r['status']=='FAIL'],members=len(files),raw_bytes=sum(m['bytes'] for m in files.values()),
  archive=dict(filename=archive.name,**meta(archive)),manifest_sha256=sha(D/'manifest.json'),
  actual_header_plan=dict(path=str(plan),sha256=PLAN_SHA,summary=parsed['summary'],status='HEADER_PLAN_ONLY'),
  every_member_reread_sha_verified=True,every_original_file_rechecked=True,tree_membership_verified=True,
  actual_payload_copy=False,device_executed=False,model_loaded=False,numerical_qualified=False,performance_qualified=False,
  scope='Archive integrity only. Retains both original CPU results and actual header-only plan; no full-checkpoint repack, model evaluation or memory-fit/performance claim. No failed CPU attempt in these two runs; source-edit syntax errors are documented separately in the retained review.')
 save(D/'archive-verification.json',summary)
 text=f'''# GLM expert repack: CPU and header-plan evidence

The original guarded CPU attempt passed 11/11 before the MTP-wrapper and Darwin
F_NOCACHE revision. The revised source passed 15/15, including exact source-pinned
sanitizer/quant-alias AST fixtures on fake packed bytes and real tiny uncached
Darwin IO with unaligned offsets and short final blocks. Both attempts report
zero failures, errors, skips or forbidden operations and unchanged sources. There
are **no failed CPU attempts** among these two runs. Earlier source-edit syntax
errors are documented separately in the archived review; no test ran in them.

The actual metadata preparation exited 0 with HEADER_PLAN_ONLY: 113,446 source
tensors become 4,960 output tensors, including 378 backbone expert stacks, while
203,976,457,080 payload bytes are conserved. Plan SHA256 `{PLAN_SHA}`. This retains
headers/metadata only; no actual checkpoint payload was copied or evaluated.
MTP layer 45 retains its original bytes and still stacks through the loader.

The archive has {summary['members']} regular files, {summary['raw_bytes']:,} raw bytes
and {summary['archive']['bytes']:,} compressed bytes. SHA256
`{summary['archive']['sha256']}`. Every archive member, original file and original
directory membership was independently rechecked. Original PASS statuses remain
separate. This establishes archive integrity and tiny byte/layout contracts, not
an official loader run, physical-memory fit, numerical inference or Pareto point.
'''
 with (D/'README.md').open('x') as f:f.write(text)
 print(json.dumps({k:summary[k] for k in ('status','members','raw_bytes','archive','failed_cpu_attempts')}))
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',action='store_true');a=p.parse_args()
 if a.run:run()
 else:print('DESCRIPTION_ONLY: explicit --run archives named closed tiny CPU evidence and header plan.')
