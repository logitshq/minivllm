"""Single local GPU owner. User requested these experiments; no historical GO reuse."""
import argparse,json,os,subprocess,sys,traceback
from pathlib import Path
import driver as d

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--frozen-sha256',required=True);args=ap.parse_args()
 f=d.verify_frozen(args.frozen_sha256)
 if d.env_record()!=f['selected_environment']:raise ValueError('exact environment required')
 with (d.P/'external-owner-attempt1.consumed.json').open('x') as stream:json.dump({'pid':os.getpid(),'freeze':args.frozen_sha256},stream)
 out=d.P/'owner';out.mkdir();record={'pid':os.getpid(),'GPU_released':False,'status':'STARTED','cleanup_errors':[]};child=None;resource=None
 try:
  resource=d.Resources(out,f,{'expected_power':'AC'});resource.sample('owner-preflight')
  go=dict(f['GO_schema'],frozen_manifest_sha256=args.frozen_sha256,expected_power='AC',authorizing_owner_pid=os.getpid(),output_dir=str(d.P/'gpu-fullmodel-run1'))
  path=d.P/'GO.json';d.write(path,go)
  cmd=[d.PYTHON,'-B',str(d.D/'driver.py'),'--go',str(path),'--frozen-sha256',args.frozen_sha256,'--output-dir',go['output_dir']]
  with (out/'stdout.log').open('wb') as stdout,(out/'stderr.log').open('wb') as stderr:
   child=subprocess.Popen(cmd,stdout=stdout,stderr=stderr,start_new_session=True);record['driver_pid']=child.pid;d.write(out/'receipt.json',record)
   record['returncode']=child.wait(timeout=1800);record['waited_reaped']=True;record['absence']=d.exact_absence(child.pid,out/'driver-absence.json')
  if record['returncode']!=0:raise ValueError('fullmodel driver failed; see preserved child receipt')
  record['status']='PASS_OWNER_BOUNDED_SCOPE'
 except BaseException as e:record['status']='STOP_FIRST_FAILURE';record['first_failure']={'text':str(e),'traceback':traceback.format_exc()}
 finally:
  if child is not None and child.poll() is None:
   try:
    child.terminate()
    try:child.wait(timeout=15)
    except subprocess.TimeoutExpired:child.kill();child.wait(timeout=15)
    record['cleanup_child_absence']=d.exact_absence(child.pid,out/'cleanup-absence.json')
   except BaseException as e:record['cleanup_errors'].append(str(e))
  if resource:
   try:resource.sample('owner-postflight')
   except BaseException as e:record['cleanup_errors'].append(str(e));record['status']='STOP_FIRST_FAILURE'
   record['raw_resources']=resource.raws
  d.write(out/'receipt.json',record)
 print(json.dumps(record,indent=2));return 0 if record['status'].startswith('PASS') else 1
if __name__=='__main__':raise SystemExit(main())
