"""C1 screen support; original native resource and exact-PID cleanup boundaries."""
import ast,copy,errno,hashlib,importlib.metadata,json,os,signal,socket,subprocess,sys,time
from pathlib import Path
from datetime import datetime
P=Path('/private/tmp/minivllm-engine-optimization-20261014')
D=P/'v25-c1-screen-build1'
sys.path.insert(0,str(P))
import metal_buffer_bootstrap_v1 as boot
from power_paired_support_v1 import phase_guard
PYTHON=boot.NATIVE_PYTHON
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  while b:=f.read(1<<20):h.update(b)
 return h.hexdigest()

def write_new(p,v):
 with Path(p).open('x') as f:f.write(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n')

def write(p,v):Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n')

def selected(env):return {k:v for k,v in env.items() if k.startswith(('MLX_','METAL_','DYLD_','OMLX_','MONOLITH_','LITHOS_')) or k in ('PYTHONPATH','PYTHONHOME','PYTHONDONTWRITEBYTECODE')}

def child_environment(inherited):
 assert not selected(inherited),'parent selected controls must be empty'
 return boot.environment({**inherited,'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','PYTHONDONTWRITEBYTECODE':'1','TOKENIZERS_PARALLELISM':'false'},'50')

def strict_absence(pid):
 cmd=['/bin/ps','-p',str(pid),'-o','pid=,ppid=,rss=,comm='];r=subprocess.run(cmd,capture_output=True,text=True,timeout=10)
 v={'pid':pid,'command':cmd,'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr,'absent':r.returncode==1 and r.stdout==r.stderr==''}
 if not v['absent']:raise RuntimeError('strict PID absence unknown/present: '+str(pid))
 return v

def identity(pid):
 cmd=['/bin/ps','-p',str(pid),'-o','lstart=,command='];r=subprocess.run(cmd,capture_output=True,text=True,timeout=10,env=dict(os.environ,LC_ALL='C'))
 if r.returncode==1 and r.stdout==r.stderr=='':return None
 assert r.returncode==0 and r.stderr=='' and len(r.stdout.split())>=6,'owned process identity unknown'
 words=r.stdout.split();datetime.strptime(' '.join(words[:5]),'%a %b %d %H:%M:%S %Y')
 return words

def process_parent(pid):
 cmd=['/bin/ps','-p',str(pid),'-o','ppid='];r=subprocess.run(cmd,capture_output=True,text=True,timeout=10)
 assert r.returncode==0 and r.stderr=='' and r.stdout.strip().isdigit(),'owned PPID unknown'
 return {'command':cmd,'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr,'ppid':int(r.stdout.strip())}

def cleanup_owned(proc,birth,evidence_path=None):
 history=[]
 def save():
  if evidence_path is not None:write(evidence_path,{'pid':proc.pid,'recorded_birth_sha256':hashlib.sha256(json.dumps(birth).encode()).hexdigest(),'decisions':history})
 if proc.poll() is None:
  assert birth is not None,'missing recorded ownership; no signal sent'
  for sig,timeout in ((signal.SIGTERM,10),(signal.SIGKILL,15)):
   if proc.poll() is not None:break
   try:current=identity(proc.pid)
   except BaseException:
    history.append({'signal':int(sig),'decision':'UNKNOWN_NO_SIGNAL'});save();raise
   row={'signal':int(sig),'current_identity_sha256':hashlib.sha256(json.dumps(current).encode()).hexdigest(),'decision':'ABSENT_NO_SIGNAL' if current is None else 'SAME_IDENTITY_SIGNAL' if current==birth else 'FOREIGN_REUSED_NO_SIGNAL'};history.append(row);save()
   if current is None:break
   assert current==birth,'foreign/reused/unknown identity; no signal sent'
   try:os.kill(proc.pid,sig)
   except ProcessLookupError:pass
   try:proc.wait(timeout=timeout);break
   except subprocess.TimeoutExpired:
    if sig==signal.SIGKILL:raise
 proc.wait(timeout=15)
 return {'pid':proc.pid,'wait_completed':True,'returncode':proc.returncode,'exact_pid_absence':strict_absence(proc.pid),'recorded_birth_identity_sha256':hashlib.sha256(json.dumps(birth).encode()).hexdigest(),'signal_identity_decisions':history}

def derived_snapshot(raw,pid=None):
 s=copy.deepcopy(raw)
 if pid is not None and not any(x['pid']==pid for x in s['known_inference_processes']):s['known_inference_processes'].append({'pid':pid,'owned':True,'engine':'explicit-c1-owned-model-server'})
 s['known_inference_process_count']=len(s['known_inference_processes']);s['only_owned_known_inference_processes']=all(x['owned'] for x in s['known_inference_processes']);return s

def probe(samples,path,label,power,pid=None):
 from resource_snapshot_power_v1 import snapshot,deltas
 raw=snapshot(label,pid)
 raw_path=path.parent/(path.stem+'-%02d-raw.json'%len(samples))
 write_new(raw_path,raw) # immutable unmodified snapshot precedes derived guards
 samples.append(derived_snapshot(raw,pid));deltas(samples);write(path,samples)
 phase_guard(samples,power,pid)
 assert samples[-1]['known_inference_process_count']==int(pid is not None),'unexpected known inference inventory'
 assert samples[-1]['pressure_level_value']==1,'requires pressure1'
 return raw_path

def wait_port_absence(port):
 deadline=time.monotonic()+30;checks=[]
 while True:
  with socket.socket() as sock:sock.settimeout(.2);rc=sock.connect_ex(('127.0.0.1',port))
  checks.append({'monotonic_s':time.monotonic(),'connect_ex':rc})
  if rc==errno.ECONNREFUSED:return {'port':port,'localhost_only':True,'absent':True,'checks':checks}
  if rc!=0:raise OSError(rc,'unknown loopback port absence; only ECONNREFUSED proves absent')
  if time.monotonic()>deadline:raise TimeoutError('owned loopback port remains open')
  time.sleep(.05)

def exact_listener(pid,port,path):
 command=['/usr/sbin/lsof','-nP','-iTCP:'+str(port),'-sTCP:LISTEN','-Fp']
 result=subprocess.run(command,capture_output=True,text=True,timeout=15)
 raw={'command':command,'owned_pid':pid,'port':port,'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr};write_new(path,raw)
 pids={int(line[1:]) for line in result.stdout.splitlines() if line.startswith('p')}
 assert result.returncode==0 and pids=={pid} and not result.stderr.strip(),'ready loopback listener not owned by exact server PID'
 return raw
