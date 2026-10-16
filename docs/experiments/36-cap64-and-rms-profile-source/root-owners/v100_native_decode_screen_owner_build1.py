"""Root sole-GPU owner: V100 vendor-RMS profile physicalCAP64 B-only native decode exploratory screen, 240s deadline, exact driver identity cleanup."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
P=Path('/private/tmp/minivllm-engine-optimization-20261014')
D=P/'v25-c1-screen-build1'
sys.path.insert(0,str(D))
from support import identity,process_parent,strict_absence,write
O=P/'v23-root-control-build1/native-decode-v100-attempt1'
assert sys.executable=='/Users/ryanli/code/minivllm/.venv/bin/python'
assert not any(k.startswith(('MLX_','METAL_','DYLD_','OMLX_','MONOLITH_','LITHOS_')) or k in ('PYTHONPATH','PYTHONHOME','PYTHONDONTWRITEBYTECODE') for k in os.environ)
O.mkdir()
cmd=[sys.executable,'-B',str(P/'v100-vendor-rms-profile-native-screen-build1/screen.py'),'--out',str(O/'child-output'),'--tokens','32']
record={'owner_pid':os.getpid(),'command':cmd,'driver_sha256':hashlib.sha256((P/'v100-vendor-rms-profile-native-screen-build1/screen.py').read_bytes()).hexdigest(),'deadline_seconds':240,'stage':'before-spawn'}
write(O/'owner-receipt.json',record)
proc=None;birth=None
try:
 with (O/'stdout.log').open('x') as out,(O/'stderr.log').open('x') as err:
  proc=subprocess.Popen(cmd,env={**os.environ,'DYLD_INSERT_LIBRARIES':str(P/'v89-native-capacity64-build1/libv89replay.dylib'),'MLX_MAX_OPS_PER_BUFFER':'50','MLX_MAX_MB_PER_BUFFER':'50','MLX_BFS_MAX_WIDTH':'20','MLX_METAL_FAST_SYNCH':'0'},stdout=out,stderr=err,start_new_session=True)
  birth=identity(proc.pid);assert birth is not None and process_parent(proc.pid)['ppid']==os.getpid()
  record.update(driver_pid=proc.pid,driver_birth=birth,stage='running')
  write(O/'owner-receipt.json',record)
  print(json.dumps({'owner_pid':os.getpid(),'driver_pid':proc.pid,'output':str(O)}),flush=True)
  record['returncode']=proc.wait(timeout=240)
except BaseException as e:
 record['owner_error']={'type':type(e).__name__,'text':str(e)}
 raise
finally:
 if proc is not None:
  if proc.poll() is None:
   assert identity(proc.pid)==birth and process_parent(proc.pid)['ppid']==os.getpid(),'ownership unknown; no signal'
   record['timeout_cleanup_signal']='SIGTERM'
   proc.terminate()
   try:proc.wait(timeout=90)
   except subprocess.TimeoutExpired:
    assert identity(proc.pid)==birth and process_parent(proc.pid)['ppid']==os.getpid()
    proc.kill();proc.wait(timeout=20)
    record['driver_force_killed']=True
  record.update(returncode=proc.returncode,waited_reaped=True,stage='reaped',absence=strict_absence(proc.pid))
  write(O/'owner-receipt.json',record)
print(json.dumps(record),flush=True)
raise SystemExit(record['returncode'])
