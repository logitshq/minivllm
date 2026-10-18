"""Archive the first failed GLM load and a separate read-only recovery observation.
No model/runtime binary read, service action, audit retry or original-file write.
The archive retains one model-directory symlink as link metadata only. Verification
never extracts any archive entry or follows that link.
"""
import argparse, hashlib, io, json, os, pathlib, stat, tarfile, time, traceback
R=pathlib.Path('/Users/ryanli/code/minivllm');A=R/'.artifacts/m5-ultra-256gb'
OUT=R/'docs/experiments/m5-ultra-256gb/GLM-flash-load-v1'
ROOTS={
 'preparation-execution':A/'glm-flash-load-v1-preparation-execution',
 'prereg':R/'bench/m5_ultra_256gb/glm_flash_cycle_v2/prereg-load-v1',
 'owner':A/'glm-flash-load-v1-owner',
 'cold-journal':A/'glm-flash-load-v1-cold-journal',
 'execution':A/'glm-flash-load-v1-execution',
 'analysis':A/'glm-flash-load-v1-analysis',
 'inputs':A/'glm-flash-inputs-v1',
 'input-request':A/'glm-flash-inputs-v1-request',
 'input-preparation-execution':A/'glm-flash-inputs-v1-execution'}
BASELINE='80654a8a05c863b2c57eb07fc6adf2c629fc725ad0e033749d04e6cdf93b3c32'
VERIFY='3af7fa650c0b1412ebd1c48a5ae0d9398d6942e1872fd633ff044ec7d6cfb8d6'
JOURNAL='d1910477a276aa7e48e9e668e1284d5dce5057cbae998483e29a1b36ea744e49'
MODEL_LINK='raw/owner/phases/glm-official-load/probe/models/GLM-5.3-Flash-4bit'
MODEL_TARGET=str(R/'.artifacts/models/huggingface/models--mlx-community--GLM-5.3-Flash-4bit/snapshots/fa04cbc3622b16aa7475404060b05e506d81bd4e')
def need(ok,message):
 if not ok:raise ValueError(message)
def digest(raw):return hashlib.sha256(raw).hexdigest()
def sha(path):
 h=hashlib.sha256()
 with pathlib.Path(path).open('rb') as f:
  while block:=f.read(1<<20):h.update(block)
 return h.hexdigest()
def read(path):return json.loads(pathlib.Path(path).read_bytes())
def save(path,value):
 with pathlib.Path(path).open('x') as f:json.dump(value,f,indent=2);f.write('\n')
def regular(path):
 s=path.lstat();need(stat.S_ISREG(s.st_mode) and s.st_size<64<<20,'only bounded regular evidence '+str(path))
 need(path.suffix not in {'.safetensors','.gguf','.dylib','.so','.a','.bin','.pyc'},'payload or binary forbidden '+str(path))
 raw=path.read_bytes();after=path.lstat()
 need((s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns),'source changed during read')
 return raw

def inventory():
 rows=[]
 def visit(path,name):
  s=path.lstat();base=dict(name=name,source=str(path),mode=stat.S_IMODE(s.st_mode))
  if stat.S_ISLNK(s.st_mode):
   target=os.readlink(path)
   need(name==MODEL_LINK and target==MODEL_TARGET,'unexpected link; never dereference')
   rows.append(dict(base,type='symlink',linkname=target,bytes=0,link_sha256=digest(os.fsencode(target))))
  elif stat.S_ISDIR(s.st_mode):
   rows.append(dict(base,type='directory',bytes=0))
   for child in sorted(path.iterdir(),key=lambda p:p.name):visit(child,name+'/'+child.name)
  else:
   raw=regular(path);rows.append(dict(base,type='file',bytes=len(raw),sha256=digest(raw)))
 for name,p in ROOTS.items():
  need(p.is_dir() and not p.is_symlink(),'actual evidence root required '+str(p));visit(p,'raw/'+name)
 return rows

def verify_archive(folder, originals=False):
 folder=pathlib.Path(folder);manifest=read(folder/'manifest.json')
 need(sha(folder/'raw.tar.xz')==manifest['archive_sha256'],'archive SHA changed')
 expected={r['name']:r for r in manifest['members']};need(len(expected)==len(manifest['members']),'duplicate manifest names')
 found=set();total=0;files=0;links=0
 with tarfile.open(folder/'raw.tar.xz','r:xz') as tar:
  for member in tar:
   name=member.name;parts=pathlib.PurePosixPath(name).parts
   need(name in expected and name not in found and not member.name.startswith('/') and '..' not in parts
        and len(parts)>=2 and parts[0]=='raw' and parts[1] in ROOTS,'unsafe/unexpected/duplicate member')
   found.add(name);row=expected[name]
   if row['type']=='file':
    need(member.isfile() and member.size==row['bytes'],'regular member type/size')
    h=hashlib.sha256();n=0
    with tar.extractfile(member) as stream:
     while block:=stream.read(1<<20):h.update(block);n+=len(block)
    need(h.hexdigest()==row['sha256'] and n==row['bytes'],'regular member bytes changed')
    total+=n;files+=1
   elif row['type']=='directory':need(member.isdir(),'directory member type')
   else:
    need(member.issym() and name==MODEL_LINK and member.linkname==MODEL_TARGET==row['linkname']
         and digest(os.fsencode(member.linkname))==row['link_sha256'],'explicit model-link metadata changed');links+=1
 need(found==set(expected),'missing archive member')
 if originals:need(inventory()==manifest['members'],'original membership/bytes changed after tar')
 return dict(status='PASS',members=len(found),regular_files=files,symlinks=links,regular_bytes=total,
             all_members_stream_verified=True,all_originals_rechecked=originals,weights_followed=False,extraction_performed=False)

def archive():
 need(not OUT.exists(),'fresh archive output required')
 basepath=ROOTS['analysis']/'service-baseline.json';verpath=ROOTS['analysis']/'operational-recovery-v1/result.json';jpath=ROOTS['cold-journal']/'journal.json'
 need(sha(basepath)==BASELINE and sha(verpath)==VERIFY and sha(jpath)==JOURNAL,'actual closed observation refs changed')
 baseline=read(basepath);v=read(verpath);j=read(jpath);final=read(ROOTS['execution']/'final.json');owner=read(ROOTS['owner']/'complete.json')
 need(baseline['status']==v['status']=='OPERATIONALLY_RESTORED_WITH_MONITOR_FAILURE' and v['original_evidence_unchanged'] is True,
      'independent read-only operational observation required')
 need(j['state']=='LOAD_INTENT' and j['post_monitor_failed'] is True and final['recovery_returncode']==1
      and final['owner_returncode']==1 and owner['status']=='FAIL' and owner['child_returncode']==-9,'actual failed attempt changed')
 for p,receipt in v['original_evidence'].items():need(sha(p)==receipt['sha256'] and pathlib.Path(p).stat().st_size==receipt['bytes'],'observed original changed')
 for row in v['current_resource_samples']:
  for pkey,hkey in [('path','sha256'),('commands_path','commands_sha256')]:need(sha(row['raw'][pkey])==row['raw'][hkey],'observed physical evidence changed')
 need(sha(v['log_window']['path'])==v['log_window']['sha256'],'observed recovery log changed')
 rows=inventory();OUT.mkdir()
 (OUT/'archive.py').write_bytes(pathlib.Path(__file__).read_bytes())
 for name,source in [('service-baseline.json',basepath),('operational-recovery.json',verpath),
       ('journal.json',jpath),('driver-final.json',ROOTS['execution']/'final.json'),
       ('worker-release.json',ROOTS['cold-journal']/'load-worker/release.json'),
       ('owner-complete.json',ROOTS['owner']/'complete.json'),('first-failure-forensics.json',ROOTS['analysis']/'first-failure-forensics.json')]:
  (OUT/name).write_bytes(regular(source))
 with tarfile.open(OUT/'raw.tar.xz','x:xz',preset=3,dereference=False) as tar:
  for row in rows:
   info=tarfile.TarInfo(row['name']);info.mode=row['mode'];info.uid=info.gid=0;info.mtime=0
   if row['type']=='directory':info.type=tarfile.DIRTYPE;tar.addfile(info)
   elif row['type']=='symlink':info.type=tarfile.SYMTYPE;info.linkname=row['linkname'];tar.addfile(info)
   else:
    raw=regular(pathlib.Path(row['source']));need(digest(raw)==row['sha256'],'source changed before tar')
    info.size=len(raw);tar.addfile(info,io.BytesIO(raw))
 manifest=dict(schema='GLM-first-load-failure-archive-v1',created_at=time.time(),source_roots={k:str(p) for k,p in ROOTS.items()},
  archive_sha256=sha(OUT/'raw.tar.xz'),archive_bytes=(OUT/'raw.tar.xz').stat().st_size,members=rows,
  scope='Full first attempt evidence and supplied small input records. One absolute model symlink retained as metadata only; no model weights/runtime binaries.',
  model_symlink_policy='Never extract/follow links; verifier reads tar records only. No extractall command is provided.',
  original_experiment='FAIL',original_recovery='LOAD_INTENT / monitor failed / returncode 1',
  independent_current_service='OPERATIONALLY_RESTORED_WITH_MONITOR_FAILURE',
  original_closed_audit=dict(executed=False,reason='Original RESTORED gate failed; root explicitly requested separate read-only observed-recovery verification instead.',
                            saved_command=read(ROOTS['prereg']/'commands.json')['closed_audit']),
  qualifications=dict(full_model=False,performance=False,physical_resource_safety=False))
 save(OUT/'manifest.json',manifest)
 checked=verify_archive(OUT,originals=True)
 checked.update(archive_sha256=manifest['archive_sha256'],manifest_sha256=sha(OUT/'manifest.json'),archive_bytes=manifest['archive_bytes'])
 save(OUT/'archive-verification.json',checked)
 summary=dict(schema='GLM-first-load-failure-summary-v1',status='FAILED_LOAD_WITH_INDEPENDENT_CURRENT_SERVICE_RECOVERY',
  owner_returncode=1,child_returncode=-9,live_physical_samples=37,completed_requests=0,
  recovery_returncode=1,original_journal_state='LOAD_INTENT',post_monitor_failed=True,http_worker_returncode=-15,
  original_post_swap_growth_bytes=849347,service_baseline_sha256=BASELINE,operational_recovery_sha256=VERIFY,
  original_journal_sha256=JOURNAL,archive=checked,
  performance_qualified=False,full_model_correctness_qualified=False,physical_resource_safety_qualified=False)
 save(OUT/'summary.json',summary)
 text='''# GLM-5.3-Flash first cold-load attempt\n\nThe GLM load did not complete. No request was generated and no model correctness, fit, quality or performance qualification is granted.\n\n- The owner retained 37 live physical observations, then stopped on `resource checkpoint exceeded observation cadence`. The final seven-command checkpoint took 1.598319 s, exceeding the fixed 1 s post-physical cadence. It passed its separate 5 s checkpoint budget. The preceding five-command physical probe took 0.244151 s. Individual command times were not recorded; attributing the delay to a particular command is unsupported.\n- The direct child 71812 was sent TERM, then KILL after the bounded wait, and strictly reaped with code −9. The original four-request diagnostic never reached its first request.\n- Restoring the original persisted Qwen service reached a real replacement, PID 72953. Its HTTP-load monitor failed after swap increased by 849,347 B from that monitor’s baseline. The HTTP worker was reaped with code −15. The original controller journal remains `LOAD_INTENT`, `post_monitor_failed=true`; driver recovery return code remains 1.\n- A separately authorized read-only check subsequently verified the actual replacement PID/job/port, idle loaded model, unchanged persisted settings/model metadata, resident PLE and MTP completion log markers, old child/worker absence, and two stable fresh physical observations. Its status is **OPERATIONALLY_RESTORED_WITH_MONITOR_FAILURE**. It does not convert the failed historical monitor or HTTP response into success.\n\n`service-baseline.json` and `operational-recovery.json` are exact copies of those separate receipts. Any future lifecycle must explicitly accept this observed-baseline contract; the original RESTORED prerequisite is not met. The saved old `closed_audit` command was not run because its original gate was not met.\n\n`raw.tar.xz` preserves preparation-execution, preregistration, owner, original cold journal, driver execution, analysis, and the three supplied input-preparation trees. `manifest.json` records every member and SHA; `archive-verification.json` records streamed tar comparison and a full original-tree recheck. The one `probe/models` symlink is retained only as link metadata. **Do not use unrestricted extraction or follow that link**: it points outside the archive to the model snapshot. No model weights or runtime binaries are included.\n\nVerify without extraction or service access:\n\n```sh\npython3 archive.py --verify .\n```\n\n`archive.py --run` is a one-shot creator bound to the original paths and refuses an existing destination. It never calls a model, service mutation or recovery command.\n'''
 (OUT/'README.md').write_text(text)
 print(json.dumps(summary,indent=2))

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',action='store_true');p.add_argument('--verify',type=pathlib.Path);a=p.parse_args()
 if a.verify:print(json.dumps(verify_archive(a.verify),indent=2));return
 if not a.run:print(json.dumps(dict(output=str(OUT),roots={k:str(p) for k,p in ROOTS.items()},default_inert=True)));return
 archive()
if __name__=='__main__':main()
