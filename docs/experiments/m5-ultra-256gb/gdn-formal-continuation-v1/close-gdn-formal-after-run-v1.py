"""One bounded continuation of the already running registered experiment.
Wait only for its actual driver's terminal recovery. Never launches a GPU run,
service command, retry, new experiment, or recurring task. Preserve failures.
"""
import datetime,hashlib,json,subprocess,sys,time,traceback
from pathlib import Path
R=Path('/Users/ryanli/code/minivllm');A=R/'.artifacts/m5-ultra-256gb'
OUT=A/'gdn-http-formal-v1-continuation';E=A/'gdn-http-formal-v1-execution';J=A/'gdn-http-formal-v1-cold-journal'
PLAN='02337f0267a327ad7b025fd90ae18dec124cfe139811a767e9597c2b27af52f5'
SOURCES={Path('/tmp/audit-closed-gdn-formal-v1.py'):'85824f920873b84cca84c6fe87e48b7d70af1037d88cf96f45b9bc53e7cab086',Path('/tmp/archive-gdn-formal-v1.py'):'610d77ad81e2f79208785775cb7eb654e381e995a0759fe999da6ac39e29ad2a'}
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def need(v,m):
 if not v:raise ValueError(m)
def save(name,value):
 p=OUT/name;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');t.replace(p)
def invoke(label,path,args):
 need(sha(path)==SOURCES[path],'continuation source changed')
 cmd=[str(R/'.venv/bin/python'),'-I','-B',str(path),*args]
 row=dict(argv=cmd,cwd=str(R),source_sha256=sha(path),started_at=now());save(label+'-command.json',row)
 print(label+' START '+row['started_at'],flush=True)
 start=time.monotonic()
 try:
  with (OUT/(label+'.stdout')).open('xb') as out,(OUT/(label+'.stderr')).open('xb') as err:
   p=subprocess.run(cmd,cwd=R,stdout=out,stderr=err,timeout=3600,check=False)
  row.update(returncode=p.returncode,status='RETURNED')
 except BaseException as error:row.update(status='ERROR',error=dict(type=type(error).__name__,message=str(error)))
 row.update(ended_at=now(),elapsed_seconds=time.monotonic()-start);save(label+'-command.json',row)
 print(label+' END '+str(row.get('returncode',row.get('error'))),flush=True);return row

def main():
 if sys.argv[1:]!=['--run']:
  print('DESCRIPTION_ONLY: waits for existing formal execution, then offline audits/retention only.');return 0
 need(read(E/'execution-receipt.json')['owner_plan_sha256']==PLAN,'actual registered execution must already exist')
 need(all(sha(p)==h for p,h in SOURCES.items()),'exact reviewed closure sources')
 OUT.mkdir();(OUT/'runner.py').write_bytes(Path(__file__).read_bytes())
 for p in SOURCES:(OUT/p.name).write_bytes(p.read_bytes())
 state=dict(schema='GDN-formal-one-cycle-continuation-v1',status='WAITING',started_at=now(),owner_plan_sha256=PLAN,wait_deadline_seconds=46000,device_or_service_action=False,restoration_qualified=False,performance_qualified=False,archive_verified=False,jobs={})
 save('result.json',state)
 try:
  deadline=time.monotonic()+46000
  while True:
   need(time.monotonic()<deadline,'registered cycle did not close within bounded wait; no intervention or audit')
   if (E/'final.json').exists():
    try:
     final=read(E/'final.json');break
    except json.JSONDecodeError:
     # The unchanged frozen driver writes this small file directly. Wait for
     # that write to finish; malformed persistent output still hits deadline.
     time.sleep(1);continue
   time.sleep(15)
  journal=read(J/'journal.json')
  need(final['journal_state']==journal['state']=='RESTORED' and final['recovery_returncode']==0,'actual driver recovery not complete')
  need(journal['owner_plan']['sha256']==PLAN,'actual completed plan')
  state.update(status='AUDITING',driver_final=final);save('result.json',state)
  state['jobs']['audit']=invoke('audit',next(p for p in SOURCES if p.name.startswith('audit-')),['--run']);save('result.json',state)
  analysis=A/'gdn-http-formal-v1-analysis';summary=read(analysis/'execution-summary.json')
  need(summary['restoration_qualified'] is True and summary['status'] in ('PASS','FAIL'),'closed restoration required for retention')
  state.update(status='ARCHIVING',restoration_qualified=True);save('result.json',state)
  state['jobs']['archive']=invoke('archive',next(p for p in SOURCES if p.name.startswith('archive-')),['--run','--analysis',str(analysis)])
  archived=R/'docs/experiments/m5-ultra-256gb/E029-gdn-http-formal-v1'
  state.update(status='PASS' if summary['status']=='PASS' and state['jobs']['audit'].get('returncode')==0 and state['jobs']['archive'].get('returncode')==0 else 'FAIL',restoration_qualified=True,performance_qualified=summary['performance_qualified'],archive_verified=state['jobs']['archive'].get('returncode')==0)
  if state['archive_verified']:
   state['archive_verification']=read(archived/'archive-verification.json')
   need(state['archive_verification']['status']=='PASS' and state['archive_verification']['schema']=='GDN-prework-formal-archive-verification-v1','actual archive verification required')
 except BaseException as error:state.update(status='FAIL',error=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc()))
 finally:
  if state['status']!='PASS':state['performance_qualified']=False
  state['ended_at']=now();save('result.json',state)
 print(json.dumps({k:state[k] for k in ('status','restoration_qualified','performance_qualified','archive_verified') if k in state}),flush=True)
 return int(state['status']!='PASS')
if __name__=='__main__':raise SystemExit(main())
