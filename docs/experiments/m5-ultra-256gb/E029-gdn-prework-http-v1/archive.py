"""Retain the actual closed GDN HTTP cycle, including failure prefixes. No service/device action."""
import argparse,hashlib,json,shutil,tarfile,tempfile
from pathlib import Path,PurePosixPath
R=Path('/Users/ryanli/code/minivllm');B=R/'bench/m5_ultra_256gb';A=R/'.artifacts/m5-ultra-256gb';OUT=R/'docs/experiments/m5-ultra-256gb/E029-gdn-prework-http-v1'
P=B/'gdn_prework_http_cycle/prereg-http-v1';O=A/'gdn-http-v1-owner';J=A/'gdn-http-v1-cold-journal';E=A/'gdn-http-v1-execution';AN=A/'gdn-http-v1-analysis'
PLAN='5d7bf34bec68dca784ebc449e660d14bcc63a6ecd2ffd7cae63981c22ebe3024';MAX=48*(1<<20)
def need(ok,msg):
 if not ok:raise ValueError(msg)
def read(p):return json.loads(Path(p).read_bytes())
def write(p,v):
 with Path(p).open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
def info(p):
 p=Path(p);need(p.is_file() and not p.is_symlink(),'regular source file');before=p.stat();h=hashlib.sha256();size=0
 with p.open('rb') as f:
  while raw:=f.read(1<<20):h.update(raw);size+=len(raw)
 after=p.stat();need((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns) and size==before.st_size,'source changed while hashing')
 return dict(sha256=h.hexdigest(),bytes=size)
def files(root):
 paths=sorted(root.rglob('*'));need(not any(p.is_symlink() for p in paths),'no archive symlinks')
 result=[p for p in paths if p.is_file()]
 for p in result:
  if p.suffix=='.safetensors':
   allowed={O/f'phases/c{C}-replay/qualification/base/failure-arrays.safetensors' for C in (1,8)}
   need(p in allowed,'model payload forbidden; only exact diagnostic failure arrays allowed')
   receipt=read(p.with_name('failure.json')).get('array_artifact')
   need(receipt==dict(path=p.name,sha256=info(p)['sha256']),'failure arrays must match actual first-failure receipt')
  else:need(p.suffix not in ('.bin','.so','.dylib','.pack','.metallib'),'model/runtime payload forbidden')
 return result

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--run',action='store_true');args=parser.parse_args()
 if not args.run:print('DESCRIPTION_ONLY: archive after actual restoration and independent audits.');return
 final,journal,summary,restore=[read(p) for p in (E/'final.json',J/'journal.json',AN/'execution-summary.json',AN/'restoration-audit.json')]
 need(final['journal_state']==journal['state']=='RESTORED' and final['recovery_returncode']==0,'actual closed recovery')
 need(restore['status']=='PASS' and restore['qualified'] is True and restore['journal_sha256']==info(J/'journal.json')['sha256'] and restore['owner_plan_sha256']==PLAN,'actual independent restoration')
 need(summary['status'] in ('PASS','FAIL') and summary['restoration_qualified'] is True and summary['performance_qualified'] is False and summary['ended_at'],'closed actual audit attempt')
 need(summary['owner_plan_sha256']==info(P/'owner-plan.json')['sha256']==PLAN,'actual registered plan')
 for row in summary['jobs'].values():
  if 'output' in row:need(info(row['output']['path'])=={k:row['output'][k] for k in ('sha256','bytes')},'actual audit output changed')
 need(all(info(p)['sha256']==h for p,h in read(P/'receipt.json')['source_pins'].items()),'registered sources changed')
 OUT.mkdir();(OUT/'archive.py').write_bytes(Path(__file__).read_bytes())
 sidecars={'restored-journal.json':J/'journal.json','restoration-audit.json':AN/'restoration-audit.json','audit-execution-summary.json':AN/'execution-summary.json'}
 for name in ('independent-c1.json','independent-c8.json','independent-cycle.json'):
  if (AN/name).exists():sidecars[name]=AN/name
 for name,p in sidecars.items():shutil.copyfile(p,OUT/name);need(info(p)==info(OUT/name),'exact sidecar copy')
 complete=read(O/'complete.json') if (O/'complete.json').exists() else {}
 cycle=read(AN/'independent-cycle.json') if (AN/'independent-cycle.json').exists() else None
 value=dict(schema='GDN-http-archive-summary-v1',owner_plan_sha256=PLAN,owner_status=complete.get('status'),audit_status=summary['status'],restoration_qualified=True,http_history_qualified=summary['http_history_qualified'],performance_qualified=False,archive_grants_qualification=False,original_identity=journal['original'],replacement_identity=journal['replacement'])
 write(OUT/'summary.json',value)
 (OUT/'README.md').write_text('# E029: GDN-only actual HTTP capture and numerical replay\n\nThis archive retains the preregistered C1/C8 four-phase experiment and every actual outcome. The separate restoration audit must pass even if the experiment fails. Numerical audit PASS applies to the captured histories; it does not prove all future arrival schedules. This observer-enabled cycle grants no performance point. E024 graph and E028 refill are OFF.\n\nAll owner/capture/replay wire bytes, traces, dispatch events, numerical receipts, resources, child cleanup, recovery, registration and audit outputs are retained. `manifest.json` binds exact bytes; the archive is reopened and every regular member checked against its original. Weights and runtime binaries are excluded. If split, concatenate `evidence.tar.xz.part-*` in numeric order; `archive-stream.json` binds all parts.\n')
 trees={'owner':O,'cold-journal':J,'execution':E,'analysis':AN,'preregistration':P,'cpu-prepare':A/'gdn-http-v1-cpu-prepare'}
 memberships={key:[str(p.relative_to(root)) for p in files(root)] for key,root in trees.items()}
 sources={key+'/'+name:trees[key]/name for key,names in memberships.items() for name in names}
 sources.update({'archive-sidecars/'+name:OUT/name for name in sidecars})
 sources.update({'archive-metadata/'+name:OUT/name for name in ('archive.py','summary.json','README.md')})
 manifest={name:dict(source=str(path),**info(path)) for name,path in sources.items()}
 write(OUT/'manifest.json',dict(schema='GDN-http-archive-manifest-v1',files=manifest,model_payloads_included=False))
 target=OUT/'evidence.tar.xz'
 with target.open('xb') as stream,tarfile.open(fileobj=stream,mode='w:xz',preset=1,dereference=True) as tar:
  for name,path in sorted(sources.items()):tar.add(path,arcname=name,recursive=False)
 whole=info(target);checked=set()
 with tarfile.open(target,'r|xz') as tar:
  for member in tar:
   name=PurePosixPath(member.name);need(member.isfile() and not name.is_absolute() and '..' not in name.parts and member.name in manifest and member.name not in checked,'safe unique member')
   h=hashlib.sha256();size=0
   with tar.extractfile(member) as f:
    while raw:=f.read(1<<20):h.update(raw);size+=len(raw)
   need(dict(sha256=h.hexdigest(),bytes=size)=={k:manifest[member.name][k] for k in ('sha256','bytes')},'decoded member differs');checked.add(member.name)
 need(checked==set(manifest),'complete archive membership')
 for name,p in sources.items():need(info(p)=={k:manifest[name][k] for k in ('sha256','bytes')},'original changed during archive')
 for key,root in trees.items():need([str(p.relative_to(root)) for p in files(root)]==memberships[key],'closed tree changed')
 parts=[]
 if whole['bytes']>MAX:
  with target.open('rb') as source:
   number=0
   while raw:=source.read(MAX):
    part=OUT/f'evidence.tar.xz.part-{number:03d}'
    with part.open('xb') as f:f.write(raw)
    parts.append(part);number+=1
  h=hashlib.sha256();size=0
  for part in parts:
   with part.open('rb') as f:
    while raw:=f.read(1<<20):h.update(raw);size+=len(raw)
  need(dict(sha256=h.hexdigest(),bytes=size)==whole,'split stream changed');target.unlink()
 else:parts=[target]
 stream=dict(archive=whole,split=len(parts)>1,parts=[dict(path=p.name,**info(p)) for p in parts]);write(OUT/'archive-stream.json',stream)
 verified=dict(schema='GDN-http-archive-verification-v1',status='PASS',members=len(manifest),raw_bytes=sum(v['bytes'] for v in manifest.values()),archive=whole,all_regular_members_decoded_verified=True,all_original_files_rehashed=True,all_tree_members_rechecked=True,model_payloads_rehashed=False,packaging_grants_qualification=False)
 write(OUT/'archive-verification.json',verified);print(json.dumps(verified))
if __name__=='__main__':main()
