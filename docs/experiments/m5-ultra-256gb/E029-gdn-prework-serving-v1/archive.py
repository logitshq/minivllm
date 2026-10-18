"""Retain CLOSED E029 original37 serving evidence, including failures. --run is explicit.
No tests, model/runtime imports, service operations, or model payload reads.
Only independent restoration PASS permits packaging. Retention grants no gate.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tarfile
import tempfile

R=Path('/Users/ryanli/code/minivllm'); B=R/'bench/m5_ultra_256gb'; A=R/'.artifacts/m5-ultra-256gb'
NAME='gdn-prework-serving-v1'; PHASE='gdn-prework-serving'
P=B/'gdn_prework_unbound/prereg-serving-v1'
OWNER=A/(NAME+'-owner'); JOURNAL=A/(NAME+'-cold-journal'); EXEC=A/(NAME+'-execution')
CHILD=OWNER/'phases'/PHASE/'qualification'
OUT=R/'docs/experiments/m5-ultra-256gb/E029-gdn-prework-serving-v1'
RESTORE_SHA='34d608b54ed5779b97e85a0463c17f8796876be731f6f0aafcbe804ffc7b15ca'
NUMERIC_SHA='268cc336417a3d31dffc30c749701d132da610fbb41d9b0f4c85ad6adaa2f2bc'
RESOURCE_SHA='5948ccbf7baeb21d1e71d9e8f7169b1bd9f84e65eb492746aba86fe377806c23'
MAX_PART=48*(1<<20)
BINARY_SUFFIXES={'.bin','.pack','.dylib','.so','.metallib','.safetensors'}

def need(ok,msg):
 if not ok:raise ValueError(msg)
def read(path):
 def pairs(items):
  result={}
  for k,v in items:need(k not in result,'duplicate JSON key');result[k]=v
  return result
 return json.loads(Path(path).read_bytes(),object_pairs_hook=pairs,parse_constant=lambda v:(_ for _ in ()).throw(ValueError(v)))
def write(path,value):
 with Path(path).open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')
def info(path):
 path=Path(path);need(path.is_file() and not path.is_symlink(),'regular file: '+str(path))
 key=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)
 st=path.stat();digest=hashlib.sha256();size=0
 with path.open('rb') as stream:
  need(key(os.fstat(stream.fileno()))==key(st),'replaced before read')
  while chunk:=stream.read(1<<20):digest.update(chunk);size+=len(chunk)
  need(key(os.fstat(stream.fileno()))==key(st),'changed while read')
 need(key(path.stat())==key(st) and size==st.st_size,'changed after read')
 return dict(sha256=digest.hexdigest(),bytes=size)
def sha(p):return info(p)['sha256']
def listing(root):
 need(root.is_dir() and not root.is_symlink(),'real tree '+str(root));files=[]
 for folder,dirs,names in os.walk(root,followlinks=False):
  for name in sorted(dirs+names):
   p=Path(folder)/name;need(not p.is_symlink(),'unexpected symlink '+str(p))
   if p.is_file():
    allowed_tensor=p==CHILD/'base/failure-arrays.safetensors'
    need(p.suffix not in BINARY_SUFFIXES or allowed_tensor,'unexpected model/runtime payload '+str(p))
    if allowed_tensor:
     need(read(p.with_name('failure.json')).get('array_artifact')==dict(path=p.name,sha256=sha(p)),
          'first-difference original-dtype tensors must bind actual failure.json')
    files.append(p)
   else:need(p.is_dir(),'special file '+str(p))
 return sorted(files)
class Parts(io.RawIOBase):
 def __init__(self,paths):self.paths=iter(paths);self.current=None
 def readable(self):return True
 def readinto(self,buffer):
  while True:
   if self.current is None:
    try:self.current=next(self.paths).open('rb')
    except StopIteration:return 0
   n=self.current.readinto(buffer)
   if n:return n
   self.current.close();self.current=None
 def close(self):
  if self.current is not None:self.current.close()
  super().close()
def run(analysis,PLAN_SHA,CHILD_SHA):
 need(re.fullmatch("[0-9a-f]{64}",PLAN_SHA) and re.fullmatch("[0-9a-f]{64}",CHILD_SHA),"explicit expected registered plan hashes")
 analysis=Path(analysis).absolute()
 need(analysis.parent==A and re.fullmatch(NAME+r'-analysis(?:-repeat-[1-9][0-9]*)?',analysis.name),'fixed analysis path')
 need(not OUT.exists(),'fresh archive, never overwrite')
 receipt=read(P/'receipt.json');plan=read(P/'owner-plan.json');final=read(EXEC/'final.json');journal=read(JOURNAL/'journal.json')
 need(receipt['schema']=='GDN-prework-serving-prereg-receipt-v1' and receipt['status']=='PREPARED'
      and sha(P/'owner-plan.json')==receipt['owner_plan_sha256']==PLAN_SHA and sha(P/'child-plan.json')==receipt['child_plan_sha256']==CHILD_SHA,'registered exact plans')
 need(plan['outputs']==dict(journal=str(JOURNAL),out=str(OWNER)) and plan['experiment']['phase']==PHASE
      and plan['experiment']['plan']==dict(path=str(P/'child-plan.json'),sha256=CHILD_SHA),'registered real outputs')
 need(final['journal_state']==journal['state']=='RESTORED' and type(final['recovery_returncode']) is int and final['recovery_returncode']==0,'actual recovery not closed')
 need(journal['owner_plan']==dict(path=str(P/'owner-plan.json'),sha256=PLAN_SHA) and journal['owner_directory']==str(OWNER)
      and journal['trials']==[PHASE] and journal['lifecycle']==plan['lifecycle'],'journal exact owner binding')
 restore=read(analysis/'restoration-audit.json');summary=read(analysis/'execution-summary.json')
 need(restore['schema']=='cold-service-restoration-audit-v1' and restore['status']=='PASS' and restore['qualified'] is True
      and restore['journal_sha256']==sha(JOURNAL/'journal.json') and restore['owner_plan_sha256']==PLAN_SHA
      and restore['audit_source_sha256']==RESTORE_SHA and restore['controller_sha256']==journal['controller_sha256']
      and restore['service_restore']['replacement']==journal['replacement'],'bound independent restoration PASS')
 need(summary['schema']=='GDN-prework-serving-closed-audits-v1' and summary['restoration_qualified'] is True
      and summary['owner_plan_sha256']==PLAN_SHA and summary['child_plan_sha256']==CHILD_SHA and 'ended_at' in summary,'completed closed audit runner')
 jobs=summary['jobs'];need(set(jobs)=={'restoration','resources','numerical'},'all independent audit attempts accounted')
 expected={'restoration':('restoration-audit.json',B/'cold_cycle_v3/audit_restore.py',RESTORE_SHA),
 'resources':('resource-summary.json',B/'resource_reparse.py',RESOURCE_SHA),
 'numerical':('independent-numerical.json',B/'gdn_prework_unbound/audit_serving.py',NUMERIC_SHA)}
 audits={}
 for key,(filename,source,digest) in expected.items():
  job=jobs[key];target=analysis/filename
  if job['status']=='NOT_RUN':need(not target.exists(),'NOT_RUN fabricated output');audits[key]=None;continue
  need(read(analysis/(key+'-execution.json'))==job and job['status'] in ('RETURNED','ERROR')
       and job['source']==dict(path=str(source),**info(source)) and sha(source)==digest,'audit invocation/source '+key)
  middle=['--journal',str(JOURNAL)] if key=='restoration' else ['--owner',str(OWNER)] if key=='resources' else ['--child',str(CHILD),'--plan-sha256',CHILD_SHA]
  need(job['argv']==[str(R/'.venv/bin/python'),'-I','-B',str(source),*middle,'--out',str(target)],'real offline CLI '+key)
  if target.exists():
   need(job['output']==dict(path=str(target),**info(target)),'audit output bytes');value=read(target)
   need(job['audit_status']==value['status'] and (value['status']!='PASS' or job.get('returncode')==0),'audit status/returncode')
   audits[key]=value
  else:need(job['status']=='ERROR' or job.get('returncode')!=0,'missing successful output');audits[key]=None
 need(audits['restoration']==restore and jobs['restoration'].get('returncode')==0,'real restore invocation returned PASS')
 for filename,ref in summary['inputs'].items():
  need(info(ref['path'])=={k:ref[k] for k in ('sha256','bytes')},'runner bound input '+filename)
  if (analysis/filename).exists():need(info(analysis/filename)==info(ref['path']),'runner input copy')
 # Sources are small explicitly registered files, never follow a model manifest to its payload.
 pins=dict(receipt['source_pins'])
 for path,digest in summary['source_pins'].items():
  need(path not in pins or pins[path]==digest,'source pin conflict');pins[path]=digest
 sources={}
 for name,digest in pins.items():
  path=Path(name);need(path.is_absolute() and path.suffix in ('.py','.json','.jsonl','.md','.metal','.h') and path.stat().st_size<32*(1<<20),'small registered source')
  need(sha(path)==digest,'registered source changed '+name);sources['sources/'+name.lstrip('/')]=path
  copy=analysis/'frozen-source'/name.lstrip('/')
  need(copy.is_file() and info(copy)==info(path),'analysis frozen source copy '+name)
 numerical=audits['numerical'];resource=audits['resources'];complete=read(OWNER/'complete.json') if OWNER.exists() else None
 child_complete=read(CHILD/'complete.json') if (CHILD/'complete.json').exists() else None
 if complete is not None:need(complete['owner_plan_sha256']==PLAN_SHA and complete['planned_trials']==[PHASE],'owner actual terminal')
 if child_complete is not None:need(child_complete['plan_sha256']==CHILD_SHA,'child actual terminal')
 tensor_files=sorted(CHILD.rglob('*.safetensors')) if CHILD.exists() else []
 if numerical:
  need(numerical['schema']=='GDN-prework-serving-independent-v1' and numerical['audit_source_sha256']==NUMERIC_SHA
       and numerical['child_directory']==str(CHILD) and numerical['http_qualified'] is False
       and numerical['performance_qualified'] is False and numerical['offline_tensor_bytes_rechecked'] is False,
       'independent numerical source/child/scope')
  if numerical['status']=='PASS':
   need(numerical['qualified'] is True and numerical['same_schedule_qualified'] is True
        and numerical['plan_sha256']==CHILD_SHA and complete['status']==child_complete['status']=='PASS',
        'actual independent full original37 PASS')
   need([len(case['completed']) for case in numerical['ledger']['cases']]==[8,16,13]
        and numerical['ledger']['partial_cases']==[] and numerical['ledger']['truncated_final_line'] is None,
        'all original37 independent terminal accounting')
   need(child_complete['source_pins_rechecked'] is True and child_complete['same_schedule_checks_passed'] is True,
        'actual postrun source/canary recheck')
  else:need(numerical['qualified'] is False and numerical['same_schedule_qualified'] is False,'failed numerical receipt cannot qualify')
 if resource:
  need(resource['source_sha256']==RESOURCE_SHA and resource['owner_directory']==str(OWNER),'resource audit identity')
  for name,expected_pin in resource.get('raw_file_pins',{}).items():
   target=OWNER/name;need(target.resolve().is_relative_to(OWNER) and info(target)==expected_pin,'resource raw binding '+name)
 # Preserve every selected/repeated analysis attempt, successful or failed.
 attempts=sorted(p for p in A.glob(NAME+'-analysis*') if p.is_dir() and re.fullmatch(NAME+r'-analysis(?:-repeat-[1-9][0-9]*)?',p.name))
 need(analysis in attempts,'selected actual audit directory')
 trees={'owner':OWNER,'cold-journal':JOURNAL,'execution':EXEC,'cpu-prepare':A/(NAME+'-cpu-prepare'),
        'preregistration':P,'source-seal':A/'cold-cycle-v3-source-seal'}
 trees.update({'analysis/'+p.name:p for p in attempts});memberships={};inputs=dict(sources);missing=[]
 for prefix,root in trees.items():
  if prefix=='owner' and not root.exists():
   need(restore['owner_reap']['status']=='NO_EXPERIMENT','absent owner needs independent NO_EXPERIMENT');missing.append(prefix);continue
  files=listing(root);memberships[prefix]=[str(p.relative_to(root)) for p in files]
  for path in files:inputs[prefix+'/'+str(path.relative_to(root))]=path
 # Analysis manifest was emitted last by the closed runner and excludes itself.
 inv=read(analysis/'manifest.json')
 need(set(inv)=={str(p.relative_to(analysis)) for p in listing(analysis) if p!=analysis/'manifest.json'},'analysis exact inventory')
 for name,pin in inv.items():need(info(analysis/name)==pin,'analysis inventory '+name)
 OUT.mkdir();(OUT/'archive.py').write_bytes(Path(__file__).read_bytes())
 write(OUT/'pack-command.json',dict(argv=sys.argv,scope='offline retention only'))
 sidecars={'restored-journal.json':JOURNAL/'journal.json','restoration-audit.json':analysis/'restoration-audit.json',
           'audit-execution-summary.json':analysis/'execution-summary.json'}
 for name in ('resource-summary.json','independent-numerical.json'):
  if (analysis/name).exists():sidecars[name]=analysis/name
 for name,path in sidecars.items():shutil.copyfile(path,OUT/name);need(info(path)==info(OUT/name),'exact sidecar')
 synopsis=dict(schema='E029-gdn-serving-archive-summary-v1',owner_plan_sha256=PLAN_SHA,child_plan_sha256=CHILD_SHA,
  owner_status=complete['status'] if complete else 'NOT_STARTED',child_status=child_complete['status'] if child_complete else 'NOT_STARTED',
  audit_statuses={k:v['status'] if v else jobs[k]['status'] for k,v in audits.items()},
  actual_failure_tensor_files=len(tensor_files),actual_failure_tensor_bytes=sum(p.stat().st_size for p in tensor_files),
  same_schedule_qualified=bool(numerical and numerical.get('same_schedule_qualified')),
  coverage=numerical.get('coverage') if numerical else None,
  observed_eos_requests=numerical.get('observed_eos_requests') if numerical else None,
  source_model_postrecheck=read(CHILD/'base/complete.json').get('source_model_pins_rechecked') if (CHILD/'base/complete.json').exists() else None,
  restoration_verified=True,resources_verified=bool(resource and resource['status']=='PASS'),
  original_identity=journal['original'],replacement_identity=journal['replacement'],
  http_qualified=False,performance_qualified=False,combined_candidates_qualified=False,archive_grants_qualification=False,
  scope='Original37 same-schedule numerical receipts, all actual48-layer candidate/fallback dispatch and original state/ahead/EOS checks only if independent PASS. Success tensor payloads are not retained or recomputed offline. Failed prefixes remain failed. No HTTP/performance/composition/parameter-replacement qualification.')
 write(OUT/'summary.json',synopsis)
 (OUT/'README.md').write_text('# E029: GDN original37 serving qualification evidence\n\n'
 'Independent restoration PASS closes this package. Execution, numerical and resource outcomes remain separate in summary.json and untouched audit sidecars. All actual failures/prefixes are retained without promotion. When numerical PASS is recorded, the scope is the original 8 C1, 16 C8 and 13 lifecycle requests with independently reconstructed scheduling, all original logits/state/ahead finite-bit receipts, and actual 48-layer GDN dispatch accounting. Successful full tensor payloads are not stored; this is receipt verification, not offline arithmetic. This package grants no HTTP, performance, parameter replacement or E024/E028 composition qualification.\n\n'
 'All owner, journal, execution, CPU-prepare, analysis attempts, registration, source-seal and registered small sources are preserved. Any actual first-failure original-dtype tensor must exactly match base/failure.json. Model weights/runtime binaries are not copied or rehashed. manifest.json maps original paths to member SHA256 and sizes. Every regular member is safely streamed into an isolated temporary directory and verified; originals and tree membership are rechecked.\n\n'
 'For a split archive, concatenate evidence.tar.xz.part-* in numeric order. archive-stream.json binds the full stream and each part, at most 48 MiB. archive-verification.json certifies retention only.\n')
 for name in ('archive.py','pack-command.json','summary.json','README.md'):inputs['archive-metadata/'+name]=OUT/name
 for name in sidecars:inputs['archive-sidecars/'+name]=OUT/name
 manifest={name:dict(source=str(path),**info(path)) for name,path in sorted(inputs.items())}
 def bound(prefix,name,pin):
  p=PurePosixPath(name);need(not p.is_absolute() and '..' not in p.parts and str(p)==name,'safe audited relative file')
  key=prefix+'/'+name;need(key in manifest,'audited input absent from archive manifest '+key)
  need(set(pin)<= {'sha256','bytes'} and all(manifest[key][k]==v for k,v in pin.items()),'audit/raw binding '+key)
 if numerical and numerical['status']=='PASS':
  prefix='owner/phases/'+PHASE+'/qualification'
  need(set(numerical['terminal_sha256'])=={'plan.json','complete.json','base/plan.json','base/complete.json'},'all numerical terminal bindings')
  for name,digest in numerical['terminal_sha256'].items():bound(prefix,name,dict(sha256=digest))
  for name,key in (('base/steps.jsonl','ledger'),('dispatch.jsonl','dispatch')):
   bound(prefix,name,{k:numerical[key][k] for k in ('sha256','bytes')})
 if resource and resource['status'] in ('PASS','REJECT'):
  need(type(resource['raw_file_pins']) is dict and resource['raw_file_pins'],'actual resource raw inputs')
  for name,pin in resource['raw_file_pins'].items():
   need(set(pin)=={'sha256','bytes'},'resource exact raw metadata');bound('owner',name,pin)
 write(OUT/'manifest.json',dict(schema='E029-gdn-serving-archive-manifest-v1',files=manifest,source_pins=pins,missing_unstarted_trees=missing,
  analysis_attempts=[str(p) for p in attempts],model_payloads_included=False,runtime_payloads_included=False))
 target=OUT/'evidence.tar.xz'
 with tarfile.open(target,'w:xz',dereference=True,preset=1) as tar:
  for name,path in sorted(inputs.items()):tar.add(path,arcname=name,recursive=False)
 whole=info(target);parts=[]
 if whole['bytes']>MAX_PART:
  with target.open('rb') as src:
   number=0
   while data:=src.read(MAX_PART):
    p=OUT/f'evidence.tar.xz.part-{number:03d}';need(number<1000,'part count limit')
    with p.open('xb') as dest:dest.write(data)
    parts.append(p);number+=1
  # Confirm exact concatenated stream before removing only the temporary unsplit copy.
  h=hashlib.sha256();n=0
  for p in parts:
   with p.open('rb') as stream:
    while chunk:=stream.read(1<<20):h.update(chunk);n+=len(chunk)
  need(dict(sha256=h.hexdigest(),bytes=n)==whole,'split stream mismatch');target.unlink()
 else:parts=[target]
 write(OUT/'archive-stream.json',dict(schema='E029-gdn-serving-archive-stream-v1',archive=whole,split=len(parts)>1,max_part_bytes=MAX_PART,
  parts=[dict(path=p.name,**info(p)) for p in parts]))
 actual={}
 with tempfile.TemporaryDirectory(prefix='e029-tar-verification-') as temp, io.BufferedReader(Parts(parts)) as joined:
  base=Path(temp).resolve()
  with tarfile.open(fileobj=joined,mode='r|xz') as tar:
   for member in tar:
    name=PurePosixPath(member.name)
    need(member.isfile() and not name.is_absolute() and '..' not in name.parts and str(name)==member.name
         and member.name not in actual and member.name in manifest,'unsafe/unexpected member')
    extracted=base.joinpath(*name.parts);need(extracted.resolve().is_relative_to(base),'extraction outside temporary root')
    extracted.parent.mkdir(parents=True,exist_ok=True)
    with tar.extractfile(member) as src,extracted.open('xb') as dest:shutil.copyfileobj(src,dest,1<<20)
    actual[member.name]=info(extracted);need(actual[member.name]=={k:manifest[member.name][k] for k in ('sha256','bytes')},'extracted member differs')
    extracted.unlink()
 need(set(actual)==set(manifest),'complete extracted member set')
 for name,path in inputs.items():need(info(path)==actual[name],'original changed during archive '+name)
 for prefix,names in memberships.items():need([str(p.relative_to(trees[prefix])) for p in listing(trees[prefix])]==names,'tree changed '+prefix)
 need(sorted(p for p in A.glob(NAME+'-analysis*') if p.is_dir() and re.fullmatch(NAME+r'-analysis(?:-repeat-[1-9][0-9]*)?',p.name))==attempts,'audit attempts changed')
 need(all(info(p)['bytes']<=MAX_PART for p in parts),'Git part size')
 write(OUT/'archive-verification.json',dict(schema='E029-gdn-serving-archive-verification-v1',status='PASS',archive=whole,
  split=len(parts)>1,parts=[dict(path=p.name,**info(p)) for p in parts],members=len(actual),raw_bytes=sum(v['bytes'] for v in actual.values()),
  all_safe_extracted_members_verified=True,all_original_files_rehashed=True,all_tree_members_rechecked=True,
  actual_raw_tensor_files=len(tensor_files),model_payloads_rehashed=False,runtime_payloads_rehashed=False,packaging_grants_qualification=False))
 print(json.dumps(dict(output=str(OUT),**read(OUT/'archive-verification.json'))))
if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',action='store_true')
 parser.add_argument('--analysis',type=Path,default=A/(NAME+'-analysis'))
 parser.add_argument('--owner-plan-sha256');parser.add_argument('--child-plan-sha256');args=parser.parse_args()
 if args.run:
  need(args.owner_plan_sha256 is not None and args.child_plan_sha256 is not None,'both registered plan hashes are required')
  run(args.analysis,args.owner_plan_sha256,args.child_plan_sha256)
 else:print('DESCRIPTION_ONLY: requires actual closed audit and restoration PASS; no archive written.')
