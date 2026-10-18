"""Wait for the already running driver, then audit and retain its closed evidence.
No model/server/service actions. One fixed cycle; not a recurring task.
"""
import hashlib,json,subprocess,sys,time,traceback
from pathlib import Path
R=Path('/Users/ryanli/code/minivllm');A=R/'.artifacts/m5-ultra-256gb';N='route-http-formal-v2'
OUT=A/(N+'-closure');FINAL=A/(N+'-execution/final.json');JOURNAL=A/(N+'-cold-journal/journal.json')
ANALYSIS=A/(N+'-analysis');PY=R/'.venv/bin/python'
JOBS=[('audit',Path('/tmp/audit-closed-route-formal-v2.py'),'e653955cace058183a6211c677898dbc3ef8cbb8b4d149b94199fdf14c4e2188'),
      ('archive',Path('/tmp/archive-route-formal-v2.py'),'44b41d200881b3be6df29b1840da3673c60e43f87b2d10f4ef686352a2fe7477')]
def need(ok,msg):
 if not ok:raise RuntimeError(msg)
def read(p):return json.loads(p.read_bytes())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(value):(OUT/'status.json').write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def main():
 need(sys.argv[1:]==['--run'],'explicit --run required')
 need(not OUT.exists() and not ANALYSIS.exists(),'exclusive unstarted closure')
 for _,p,h in JOBS:need(sha(p)==h,'reviewed script changed')
 OUT.mkdir();(OUT/'runner.py').write_bytes(Path(__file__).read_bytes())
 value=dict(status='WAITING_FOR_DRIVER',started_at=time.time(),jobs=[],service_actions=False)
 save(value);deadline=time.monotonic()+43200
 try:
  while not FINAL.exists():
   need(time.monotonic()<deadline,'driver did not close within 12 hours')
   time.sleep(30)
  for attempt in range(3):
   try:
    final=read(FINAL);journal=read(JOURNAL);break
   except (json.JSONDecodeError,FileNotFoundError):
    if attempt==2:raise
    time.sleep(1)
  need(final['journal_state']==journal['state']=='RESTORED' and final['recovery_returncode']==0,'actual restoration required; no recovery action attempted by this helper')
  value.update(status='AUDITING',final_sha256=sha(FINAL),journal_sha256=sha(JOURNAL));save(value)
  for label,script,digest in JOBS:
   need(sha(script)==digest,'reviewed script changed before invocation')
   command=[str(PY),'-I','-B',str(script),'--run'];row=dict(label=label,argv=command,source_sha256=digest,started_at=time.time())
   value['jobs'].append(row);save(value)
   with (OUT/(label+'.stdout')).open('xb') as stdout,(OUT/(label+'.stderr')).open('xb') as stderr:
    child=subprocess.run(command,cwd=R,stdout=stdout,stderr=stderr,check=False,timeout=2400)
   row.update(returncode=child.returncode,ended_at=time.time());save(value)
   if label=='audit':
    result=read(ANALYSIS/'execution-summary.json')
    need(result['restoration_qualified'] is True,'independent restoration did not qualify')
    value['audit_status']=result['status'];value['audit_returncode']=child.returncode;value['performance_qualified']=result['performance_qualified']
   else:need(child.returncode==0,'archive failed; retain outputs for review')
  value['status']='PASS' if value['audit_status']=='PASS' and value['audit_returncode']==0 else 'EXPERIMENT_FAILED_EVIDENCE_RETAINED'
 except BaseException as e:value.update(status='FAIL',error=dict(type=type(e).__name__,message=str(e),traceback=traceback.format_exc()))
 finally:
  if value['status']!='PASS':value['performance_qualified']=False
  value['ended_at']=time.time();save(value)
 print(json.dumps(value),flush=True)
 return int(value['status']!='PASS')
if __name__=='__main__':raise SystemExit(main())
