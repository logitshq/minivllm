"""Root sole-GPU local numeric owner V95, explicit frozen source, monitored 300s, owned cleanup."""
import argparse,copy,hashlib,json,os,subprocess,sys,time
from pathlib import Path
P=Path('/private/tmp/minivllm-engine-optimization-20261014');R=P/'v23-root-control-build1'
sys.path.insert(0,str(P));import resource_snapshot_power_v1 as resources
sys.path.insert(0,str(P/'v25-c1-screen-build1'))
from support import identity,process_parent,strict_absence,write
ap=argparse.ArgumentParser();ap.add_argument('--variant',choices=('v95',),required=True);a=ap.parse_args()
assert sys.executable=='/Users/ryanli/code/minivllm/.venv/bin/python'
assert not any(k.startswith(('MLX_','METAL_','DYLD_','OMLX_','MONOLITH_','LITHOS_')) or k in ('PYTHONPATH','PYTHONHOME','PYTHONDONTWRITEBYTECODE') for k in os.environ)
variants={'v95':('v95-gdn-simd4-segment-build1/numeric.py','a810e589d87f5d4d0ecffeb0418820ad5ec5fd1ebab7380db333d1e4d2834980','--execute-gpu')}
relative,expected,flag=variants[a.variant];script=P/relative
assert hashlib.sha256(script.read_bytes()).hexdigest()==expected
O=R/('numeric-'+a.variant+'-attempt1');O.mkdir()
cmd=[sys.executable,'-B',str(script),flag,'--out',str(O/'child-output')]
record={'owner_pid':os.getpid(),'command':cmd,'driver_sha256':expected,'deadline_seconds':300,'stage':'before-spawn'}
history=[];proc=None;birth=None
write(O/'owner-receipt.json',record)
def sample(label,pid=None):
 raw=resources.snapshot(label,pid);write(O/('resource-%03d-raw.json'%len(history)),raw)
 row=copy.deepcopy(raw)
 if pid is not None and not any(x['pid']==pid for x in row['known_inference_processes']):row['known_inference_processes'].append({'pid':pid,'owned':True,'engine':'explicit-owned-numeric-child'})
 row['known_inference_process_count']=len(row['known_inference_processes']);row['only_owned_known_inference_processes']=all(x['owned'] for x in row['known_inference_processes'])
 history.append(row);resources.deltas(history);write(O/'resources.json',history);resources.guard(history)
 assert raw['pressure_level_value']==1
 assert raw['power_metadata']['status']=='known' and raw['power_metadata']['source']=='AC Power'
 assert raw['processinfo_thermal_metadata']['code'] in (0,1)
 assert row['only_owned_known_inference_processes']
try:
 sample('before-local-numeric')
 with (O/'stdout.log').open('x') as out,(O/'stderr.log').open('x') as err:
  proc=subprocess.Popen(cmd,env={**os.environ,'MLX_MAX_OPS_PER_BUFFER':'50','MLX_MAX_MB_PER_BUFFER':'50','MLX_BFS_MAX_WIDTH':'20','MLX_METAL_FAST_SYNCH':'0'},stdout=out,stderr=err,start_new_session=True)
  birth=identity(proc.pid);assert birth is not None and process_parent(proc.pid)['ppid']==os.getpid()
  record.update(driver_pid=proc.pid,driver_birth=birth,stage='running');write(O/'owner-receipt.json',record)
  print(json.dumps({'owner_pid':os.getpid(),'driver_pid':proc.pid,'output':str(O)}),flush=True)
  started=time.monotonic()
  while proc.poll() is None:
   assert time.monotonic()-started<300,'local numeric deadline'
   sample('during-local-numeric',proc.pid);time.sleep(2)
  record['returncode']=proc.wait(timeout=10)
except BaseException as e:
 record['owner_error']={'type':type(e).__name__,'text':str(e)}
 raise
finally:
 if proc is not None:
  if proc.poll() is None:
   assert identity(proc.pid)==birth and process_parent(proc.pid)['ppid']==os.getpid(),'ownership unknown; no signal'
   record['timeout_cleanup_signal']='SIGTERM';proc.terminate()
   try:proc.wait(timeout=90)
   except subprocess.TimeoutExpired:
    assert identity(proc.pid)==birth and process_parent(proc.pid)['ppid']==os.getpid()
    proc.kill();proc.wait(timeout=20);record['driver_force_killed']=True
  record.update(returncode=proc.returncode,waited_reaped=True,stage='reaped',absence=strict_absence(proc.pid));write(O/'owner-receipt.json',record)
 sample('after-local-numeric-reaped')
print(json.dumps(record),flush=True)
raise SystemExit(record['returncode'])
