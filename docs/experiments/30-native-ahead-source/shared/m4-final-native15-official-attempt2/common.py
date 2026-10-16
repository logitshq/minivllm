"""Stdlib control and ownership helpers; imports never probe or launch."""
import ast,copy,errno,hashlib,importlib.metadata,json,os,shlex,signal,socket,subprocess,sys,time
from pathlib import Path
from datetime import datetime,timezone
D=Path(__file__).resolve().parent
P=Path('/private/tmp/minivllm-engine-optimization-20261014')
sys.path.insert(0,str(P))
assert hashlib.sha256((P/'metal_buffer_bootstrap_v1.py').read_bytes()).hexdigest()=='d7faae6c7b24e43dbbeba7a91a88c7cacae08d4c0ae5b6b4c9e4c4a67ecf48f1'
assert hashlib.sha256((P/'power_paired_support_v1.py').read_bytes()).hexdigest()=='cefe1b28e179561b9176a715e86e45ddcf821ec8260fc02ced12379b65c2f5a3'
import metal_buffer_bootstrap_v1 as boot
from power_paired_support_v1 import phase_guard
PYTHON=boot.NATIVE_PYTHON
CANDIDATE=P/'candidate-v17-gdn-gates-build1-root/tutorial_code/miniengine'
SOURCE=P/'source-manifest-v17-gdn-gates-build1.json'
CHILD=D/'child.py'
PLAN=json.loads((D/'plan.json').read_text())
REGISTRY=json.loads((D/'registry.json').read_text())
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
def verify_source():
 f=json.loads((D/'pins.json').read_text());actual={str(p.relative_to(CANDIDATE)):sha(p) for p in CANDIDATE.rglob('*.py')}
 assert actual==f['native_sources'] and len(actual)==25
 return {'root':str(CANDIDATE),'source_count':25,'source_files_sha256':actual,'source_manifest_sha256':sha(SOURCE)}
def verify_frozen(expected=None):
 if expected is not None:assert sha(D/'pins.json')==expected,'pins SHA differs'
 f=json.loads((D/'pins.json').read_text())
 for p,h in f['files'].items():assert sha(p)==h,'pinned input differs: '+p
 verify_source();return f

def actual_preimport_identity():
 assert not any(n=='mlx' or n.startswith('mlx.') for n in sys.modules),'MLX preimported'
 assert sys.executable==PYTHON,'native interpreter differs'
 f=json.loads((D/'pins.json').read_text());assert sys.version==f['native_python_version']
 versions={k:importlib.metadata.version(k) for k in ('mlx','mlx-metal')};assert versions=={'mlx':'0.32.3','mlx-metal':'0.32.3'}
 for p,h in f['native_binaries'].items():assert sha(p)==h
 return {'python_executable':sys.executable,'python_version':sys.version,'versions':versions,'native_binaries_sha256':f['native_binaries'],'mlx_not_imported':True,'source':verify_source()}
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

def lock_owned_identity(proc,birth,engine,run_dir,phase):
 # Popen/waitpid still owns this exact live child; immutable birth remains mandatory.
 if proc.poll() is not None:return birth
 current=identity(proc.pid)
 if current is None:return birth
 assert birth is not None and current[:5]==birth[:5],'birth changed/unknown; no signal authorized'
 proof={'pid':proc.pid,'owner_pid':os.getpid(),'phase':phase,'initial_identity_words':birth,'current_identity_words':current,'same_start':True,'direct_Popen_child_live':True}
 is_title=(current[5:]==['omlx-server'])
 if current!=birth or (engine=='omlx' and is_title):
  assert engine=='omlx' and is_title,'unexpected command transition; no signal authorized'
  parent=process_parent(proc.pid);assert parent['ppid']==os.getpid(),'foreign/reparented PID; no signal authorized'
  prepath=run_dir/'official-preimport.json';pre=json.loads(prepath.read_text())
  assert pre['pid']==proc.pid and pre['ppid']==os.getpid() and pre['engine']=='omlx' and pre['python_executable']==REGISTRY['omlx']['actual_environment']['python_executable'] and pre['source_native_tree_sha256']==REGISTRY['omlx']['tree_sha256'],'preimport ownership mismatch; no signal authorized'
  source=next(Path(p) for p in REGISTRY['omlx']['source_native_files'] if p.endswith('/omlx/process_title.py'))
  assert sha(source)==REGISTRY['omlx']['source_native_files'][str(source)]
  tree=ast.parse(source.read_text());decl=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='DEFAULT_PROCESS_TITLE' for t in n.targets));assert ast.literal_eval(decl.value)=='omlx-server'
  proof.update(parent_query=parent,preimport_path=str(prepath),preimport_sha256=sha(prepath),pinned_title_source=str(source),pinned_title_source_sha256=sha(source))
  if phase=='ready':
   listener=json.loads((run_dir/'server-ready-exact-listener.json').read_text());pids={int(x[1:]) for x in listener['stdout'].splitlines() if x.startswith('p')}
   assert listener['returncode']==0 and listener['stderr']=='' and listener['owned_pid']==proc.pid and pids=={proc.pid},'ready listener ownership mismatch'
   proof['ready_listener']=listener
  if current!=birth:
   proof['decision']='ONE_PINNED_OMLX_TITLE_TRANSITION';write_new(run_dir/'owned-title-transition.json',proof)
 else:proof['decision']='UNCHANGED_EXACT_IDENTITY'
 proof['locked_identity_sha256']=hashlib.sha256(json.dumps(current).encode()).hexdigest();write_new(run_dir/('owned-'+phase+'-identity.json'),proof)
 return current

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

def verify_previous_release():
 f=json.loads((D/'pins.json').read_text());refs=f['latest_attempt1_release']
 for x in refs.values():assert sha(x['path'])==x['sha256']
 release=json.loads(Path(refs['release']['path']).read_text());proof=json.loads(Path(refs['absence']['path']).read_text())
 assert release['status']=='GPU_RELEASED' and release['suite_qualified'] is False and release['actual_tool_exit_code']==1
 assert release['proof']==refs['absence'] and release['driver_wait_reap'] is True and release['native_child_wait_reap'] is True and release['omlx_child_wait_record_missing'] is True
 assert proof['tool_returned_reaped'] is True and proof['tool_exit_code']==1 and proof['actual_tool_session']==12897
 assert [v['pid'] for v in proof['strict_exact_pid_absence']]==[16126,16152,16361] and all(v['strict_absent'] is True and v['returncode']==1 and v['stdout']==v['stderr']=='' for v in proof['strict_exact_pid_absence'])
 assert len(proof['ports'])==2 and all(v['strict_refused'] is True and v['connect_ex']==errno.ECONNREFUSED for v in proof['ports'])
 assert proof['omlx_child_wait_record_missing'] is True and proof['no_retroactive_qualified_result'] is True
 q=f['fullqual_baseline_evidence'];assert sha(q['path'])==q['sha256'];qualified=json.loads(Path(q['path']).read_text());assert qualified['status']=='PASS_FULLMODEL_BOUNDED_SCOPE' and qualified['fullmodel_comparison_qualified'] is True
 raw=qualified['driver_receipt'];assert sha(raw['path'])==raw['sha256'];raw=json.loads(Path(raw['path']).read_text());assert raw['status']=='PASS_FULLMODEL_BOUNDED_SCOPE' and raw['GO']['baseline_source_manifest_sha256']==f['native_source_manifest_sha256']
 return refs

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
