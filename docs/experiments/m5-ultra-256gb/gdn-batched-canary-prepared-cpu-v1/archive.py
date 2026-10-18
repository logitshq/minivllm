"""Archive existing fake-only CPU history; never executes tests or backends."""
import argparse,hashlib,json,shutil,tarfile
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
RUNS=('gdn-batched-canary-cpu-v1','gdn-batched-audit-cpu-v1')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,obj):
 with p.open('x') as f:json.dump(obj,f,indent=2,allow_nan=False);f.write('\n')
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',action='store_true');a=p.parse_args()
 if not a.run:print(json.dumps(dict(action='describe',runs=RUNS,no_tests=True)));return
 entries={};results={}
 def add(path,name):
  if path.is_symlink() or not path.is_file() or name in entries:raise ValueError('unsafe or duplicate file: '+str(path))
  entries[name]=dict(source=str(path),bytes=path.stat().st_size,sha256=sha(path))
 for run in RUNS:
  folder=ROOT/'.artifacts/m5-ultra-256gb'/run
  result=json.loads((folder/'result.json').read_bytes())
  if result['status']!='PASS' or result['blocked'] or not result['sources_unchanged']:raise ValueError('unexpected CPU history: '+run)
  results[run]={k:result[k] for k in ('status','expected','discovered','tests_run','elapsed_seconds','blocked','sources_unchanged')}
  for path in sorted(folder.rglob('*')):
   if path.is_symlink():raise ValueError('source tree symlink')
   if path.is_file():add(path,'history/'+run+'/'+path.relative_to(folder).as_posix())
 for name in ('qualify.py','test_qualify_cpu.py','audit.py','test_audit_cpu.py','CANARY.md'):
  path=ROOT/'bench/m5_ultra_256gb/gdn_prework_batched'/name
  add(path,'current-source/'+path.relative_to(ROOT).as_posix())
 for name in ('archive.py','verify.py'):add(HERE/name,'archive-tools/'+name)
 manifest=dict(schema='GDN-batched-canary-CPU-archive-manifest-v1',members=entries)
 write(HERE/'manifest.json',manifest)
 archive=HERE/'evidence.tar.xz'
 with tarfile.open(archive,'x:xz',dereference=True,preset=6) as tar:
  for name,row in entries.items():
   path=Path(row['source'])
   if path.stat().st_size!=row['bytes'] or sha(path)!=row['sha256']:raise ValueError('input changed')
   tar.add(path,arcname=name,recursive=False)
 for name,row in entries.items():
  path=Path(row['source'])
  if path.stat().st_size!=row['bytes'] or sha(path)!=row['sha256']:raise ValueError('input changed after archive')
 write(HERE/'summary.json',dict(schema='GDN-batched-canary-CPU-archive-summary-v1',status='PASS_FAKE_CPU_ONLY',results=results,
  files=len(entries),raw_bytes=sum(r['bytes'] for r in entries.values()),archive_bytes=archive.stat().st_size,archive_sha256=sha(archive),
  device_executed=False,service_actions=False,model_correctness_qualified=False,performance_qualified=False,
  device_registration='NOT_REGISTERED; no numerical run',
  scope='Preserves first complete qualifier12 and auditor8 fake CPU runs, commands, outputs and source snapshots; no model/device/full-model/HTTP/performance qualification.'))
 import importlib.util
 s=importlib.util.spec_from_file_location('gdn_batched_archive_verify',HERE/'verify.py');v=importlib.util.module_from_spec(s);s.loader.exec_module(v)
 checked=v.verify(HERE)
 write(HERE/'archive-verification.json',dict(schema='GDN-batched-canary-CPU-archive-verification-v1',status='PASS',**checked,
  manifest_sha256=sha(HERE/'manifest.json'),summary_sha256=sha(HERE/'summary.json'),verify_source_sha256=sha(HERE/'verify.py'),
  original_members_rechecked=True,safe_extraction_verified=True))
 print(json.dumps(checked));print(HERE)
if __name__=='__main__':main()
