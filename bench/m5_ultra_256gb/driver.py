"""V19 fullmodel correctness only. CPU default is inert; future root GO is mandatory."""
import argparse, copy, hashlib, importlib.util, json, os
from pathlib import Path
import sys, traceback, types
D=Path(__file__).resolve().parent
P=Path(os.environ.get('MINIVLLM_EXPERIMENT_DOMAIN',str(D.parents[1]/'.artifacts/m5-ultra-256gb/fullqual-build1'))).resolve()
FROZEN=P/'frozen.json'
PYTHON=str(D.parents[1]/'.venv/bin/python')
SCHEMA='v19-fullmodel-qual-go-m5-ultra-build1'
ROLES=('accepted','candidate')

def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  while b:=f.read(1<<20):h.update(b)
 return h.hexdigest()

def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def pinned(name,path,frozen):
 data=Path(path).read_bytes()
 if hashlib.sha256(data).hexdigest()!=frozen['files'][str(path)]:raise ValueError('helper changed: '+str(path))
 m=types.ModuleType(name);m.__file__=str(path)
 exec(compile(data,str(path),'exec',dont_inherit=True),m.__dict__)
 return m

def verify_frozen(expected=None):
 if expected is not None and digest(FROZEN)!=expected:raise ValueError('frozen SHA differs')
 f=json.loads(FROZEN.read_text())
 for path,want in f['files'].items():
  if digest(path)!=want:raise ValueError('frozen file changed: '+path)
 for role in ROLES:
  root=Path(f[role+'_root']);actual={str(p.relative_to(root)):digest(p) for p in root.rglob('*.py')}
  if actual!=f[role+'_sources']:raise ValueError('complete source tree changed: '+role)
 return f

def validate_go(args,f):
 if not args.go or not args.frozen_sha256:raise ValueError('explicit GO and frozen SHA required')
 go=json.loads(args.go.read_text())
 required={'schema':SCHEMA,'stage':'fullmodel_correctness_no_timing','frozen_manifest_sha256':args.frozen_sha256,
  'sole_gpu_owner':True,'actual_unsandboxed_execution':True,'attempt':1,'serial_model_loads':2,
  'baseline_gates_compile':True,'candidate_default_off':True,'no_performance':True,
  'baseline_source_manifest_sha256':f['baseline_source_manifest_sha256'],
  'candidate_source_manifest_sha256':f['candidate_source_manifest_sha256']}
 if set(go)!=set(required)|{'expected_power','authorizing_owner_pid','output_dir'}:raise ValueError('GO exact field set differs')
 if not args.output_dir.is_absolute():raise ValueError('absolute fresh output directory required')
 for k,v in required.items():
  if type(go.get(k)) is not type(v) or go[k]!=v:raise ValueError('invalid GO field '+k)
 if go.get('expected_power') not in ('AC','Battery'):raise ValueError('fixed expected_power required')
 if type(go.get('authorizing_owner_pid')) is not int or go['authorizing_owner_pid']<=0:raise ValueError('outer owner PID required')
 if str(args.output_dir)!=go.get('output_dir'):raise ValueError('GO output directory differs')
 return go

def derived_snapshot(raw,pid,loaded):
 s=copy.deepcopy(raw)
 if loaded and not any(x['pid']==pid for x in s['known_inference_processes']):
  s['known_inference_processes'].append({'pid':pid,'owned':True,'engine':'explicit-loaded-fullmodel-qualification'})
 s['known_inference_process_count']=len(s['known_inference_processes'])
 s['only_owned_known_inference_processes']=all(x['owned'] for x in s['known_inference_processes'])
 return s

class Resources:
 def __init__(self,out,f,go):
  self.out,self.f,self.go=out,f,go;self.history=[];self.raws=[]
  self.resource=pinned('fullqual_resource',D/'resources.py',f)
 def sample(self,label,loaded=False):
  raw=self.resource.snapshot(label,os.getpid());path=self.out/('resource-%02d-raw.json'%len(self.raws))
  write(path,raw);self.raws.append({'file':path.name,'sha256':digest(path)})
  s=derived_snapshot(raw,os.getpid(),loaded);self.history.append(s)
  self.resource.deltas(self.history);write(self.out/'resource-derived.json',self.history)
  self.resource.guard(self.history)
  if s['pressure_level_value']!=1:raise ValueError('requires pressure1')
  if s['power_metadata']['status']!='known' or s['power_metadata']['source']!={'AC':'AC Power','Battery':'Battery Power'}[self.go['expected_power']]:raise ValueError('actual power mismatch')
  if s['processinfo_thermal_metadata']['status']!='known' or s['processinfo_thermal_metadata']['code'] not in (0,1):raise ValueError('thermal unknown/serious')
  if s['exact_owned_pid_metadata']['status']!='known' or s['known_inference_process_count']!=int(loaded):raise ValueError('process inventory differs')

def env_record():
 return {k:v for k,v in os.environ.items() if k.startswith(('MLX_','METAL_','DYLD_')) or k in ('PYTHONPATH','PYTHONHOME','PYTHONDONTWRITEBYTECODE')}

def environment_guard(f,result):
 if sys.executable!=PYTHON or sys.version!=f['python_version']:raise ValueError('pinned interpreter/version differs')
 if result['actual_environment']!=f['selected_environment']:raise ValueError('selected environment differs')

def exact_absence(pid,out):
 import subprocess
 r=subprocess.run(['/bin/ps','-p',str(pid),'-o','pid=,ppid=,rss=,comm='],capture_output=True,text=True,timeout=10)
 raw={'pid':pid,'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr};write(out,raw)
 if r.returncode!=1 or r.stdout!='' or r.stderr!='':raise ValueError('exact child absence not proved')
 return raw

def execute(args,f,go,result):
 import subprocess
 resource=None;active=None
 try:
  resource=Resources(args.output_dir,f,go);resource.sample('owner-before-first-child');environment_guard(f,result)
  for role in ROLES:
   out=args.output_dir/role;out.mkdir()
   command=[PYTHON,'-B',str(D/'driver.py'),'--go',str(args.go),'--frozen-sha256',args.frozen_sha256,
    '--output-dir',str(args.output_dir),'--worker-role',role,'--internal-owner-pid',str(os.getpid())]
   with (out/'stdout.log').open('wb') as stdout,(out/'stderr.log').open('wb') as stderr:
    active=subprocess.Popen(command,stdout=stdout,stderr=stderr,start_new_session=True)
    row={'role':role,'pid':active.pid,'command':command,'waited_reaped':False};result['children'].append(row);write(args.output_dir/'receipt.json',result)
    try:row['returncode']=active.wait(timeout=1800)
    except BaseException:
     active.terminate()
     try:active.wait(timeout=15)
     except subprocess.TimeoutExpired:active.kill();active.wait(timeout=15)
     raise
    finally:
     row['waited_reaped']=active.returncode is not None
     primary=sys.exc_info()[1]
     try:row['absence']=exact_absence(active.pid,out/'exact-absence.json')
     except BaseException as cleanup:
      if primary is None:raise
      result.setdefault('cleanup_errors',[]).append('child absence after primary error: '+str(cleanup))
     finally:active=None
   if row['returncode']!=0:raise ValueError('STOP child '+role)
   resource.sample('owner-after-reaped-'+role)
  replay=pinned('fullqual_replay',D/'replay.py',f)
  result['independent_replay']=replay.verify(args.output_dir,include_owner=False)
  verify_frozen(args.frozen_sha256)
  result['status']='PASS_FULLMODEL_BOUNDED_SCOPE'
 finally:
  if active is not None:
   try:
    if active.poll() is None:
     active.terminate()
     try:active.wait(timeout=15)
     except subprocess.TimeoutExpired:active.kill();active.wait(timeout=15)
    exact_absence(active.pid,args.output_dir/('finally-child-'+str(active.pid)+'-absence.json'))
   except BaseException as cleanup:result.setdefault('cleanup_errors',[]).append('owner child cleanup: '+str(cleanup));result['status']='STOP_FIRST_FAILURE'
  if resource is not None:
   try:resource.sample('owner-finally-postflight')
   except BaseException as e:result.setdefault('cleanup_errors',[]).append(str(e));result['status']='STOP_FIRST_FAILURE'
   result['raw_resources']=resource.raws

def main(argv=None):
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--inert',action='store_true');ap.add_argument('--go',type=Path);ap.add_argument('--frozen-sha256');ap.add_argument('--output-dir',type=Path)
 ap.add_argument('--worker-role',choices=ROLES,help=argparse.SUPPRESS);ap.add_argument('--internal-owner-pid',type=int,help=argparse.SUPPRESS)
 args=ap.parse_args(argv)
 if args.inert:
  f=verify_frozen(args.frozen_sha256);print(json.dumps({'CPU_only':True,'files':len(f['files']),'MLX_imported':any(k=='mlx' or k.startswith('mlx.') for k in sys.modules)}));return 0
 if not args.go or not args.frozen_sha256 or args.output_dir is None:ap.error('explicit --go, --frozen-sha256 and fresh --output-dir required before runtime')
 try:f=verify_frozen(args.frozen_sha256);go=validate_go(args,f)
 except (OSError,ValueError,KeyError) as e:ap.error(str(e))
 # Invalid GO cannot import helpers, snapshot resources, create directories, or consume attempt.
 if args.worker_role:
  if args.internal_owner_pid!=os.getppid():ap.error('worker must be exact child of owner')
  if not (args.output_dir/'attempt-consumed.json').is_file():ap.error('owner attempt missing')
  marker=json.loads((args.output_dir/'attempt-consumed.json').read_text())
  if marker!={'GO_sha256':digest(args.go),'owner_pid':args.internal_owner_pid}:ap.error('worker owner/GO marker differs')
  out=args.output_dir/args.worker_role
  with (out/'worker-attempt.consumed.json').open('x') as latch:json.dump({'owner_pid':args.internal_owner_pid,'role':args.worker_role},latch)
 else:
  if os.getppid()!=go['authorizing_owner_pid']:ap.error('outer authorized owner PID differs')
  if args.output_dir.exists():ap.error('fresh output path required; no retry')
  with (P/'attempt1.consumed.json').open('x') as latch:json.dump({'GO_sha256':digest(args.go),'output_dir':str(args.output_dir)},latch)
  args.output_dir.mkdir();out=args.output_dir;write(out/'attempt-consumed.json',{'GO_sha256':digest(args.go),'owner_pid':os.getpid()})
 result={'schema':'v19-fullmodel-qual-receipt-m5-ultra-build1','role':args.worker_role or 'owner','pid':os.getpid(),'status':'STARTED','children':[],
  'frozen_sha256':args.frozen_sha256,'GO_sha256':digest(args.go),'GO':go,'no_performance_claim':True,'GPU_released':False,'external_owner_exact_reap_required':True,'cleanup_errors':[]}
 try:
  (out/'ROOT-GO.raw.json').write_bytes(args.go.read_bytes())
  result['actual_environment']=env_record();write(out/'receipt.json',result)
  if args.worker_role:pinned('fullqual_worker',D/'worker.py',f).run(args,f,go,result,out,sys.modules[__name__])
  else:execute(args,f,go,result)
 except BaseException as e:
  result['status']='STOP_FIRST_FAILURE';result['error']={'type':type(e).__name__,'text':str(e),'traceback':traceback.format_exc()}
 finally:
  if 'raw_resources' not in result:
   try:
    postflight=Resources(out,f,go);postflight.sample('accepted-GO-earlyfailure-finally');result['raw_resources']=postflight.raws
   except BaseException as e:result.setdefault('cleanup_errors',[]).append(str(e));result['status']='STOP_FIRST_FAILURE'
  try:
   if digest(args.go)!=result['GO_sha256']:raise ValueError('root GO bytes changed during attempt')
  except BaseException as e:result['status']='STOP_FIRST_FAILURE';result.setdefault('cleanup_errors',[]).append(str(e))
  write(out/'receipt.json',result)
 print(json.dumps({'status':result['status'],'role':result['role'],'out':str(out)}))
 return 0 if result['status'].startswith('PASS_') else 1
if __name__=='__main__':raise SystemExit(main())
