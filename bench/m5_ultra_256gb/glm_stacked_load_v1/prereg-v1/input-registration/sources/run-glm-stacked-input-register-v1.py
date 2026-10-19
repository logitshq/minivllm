import contextlib,hashlib,importlib.abc,importlib.util,json,mmap,os,signal,socket,subprocess,sys,time,traceback,unittest.mock
from pathlib import Path
R=Path('/Users/ryanli/code/minivllm');S=R/'bench/m5_ultra_256gb/glm_stacked_load_v1';A=R/'.artifacts/m5-ultra-256gb';O=A/'glm-stacked-input-registration-attempt-v1';I=A/'glm-stacked-inputs-v1';G=A/'glm-stacked-registration-v1';AN=A/'glm-stacked-analysis-v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):
 with p.open('x') as f:json.dump(v,f,indent=2);f.write('\n')
def read(p):return json.loads(p.read_bytes())
def load(name):
 p=S/(name+'.py');sp=importlib.util.spec_from_file_location('_stacked_cpu_actual_'+name,p);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
assert not any(p.exists() for p in (O,I,G,AN));O.mkdir();AN.mkdir();(O/'sources').mkdir();files=sorted(S.glob('*.py'))+sorted(S.glob('*.json'))+[Path(__file__).resolve()];before={str(p):sha(p) for p in files}
for p in files:(O/'sources'/p.name).write_bytes(p.read_bytes())
write(O/'source-before.json',before);forbidden=[]
def deny(label):
 def f(*a,**k):forbidden.append(label);raise AssertionError('forbidden '+label)
 return f
class Guard(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0] in ('mlx','mlx_lm','mlx_vlm','omlx','torch','transformers','safetensors'):deny('backend '+name)()
g=Guard();sys.meta_path.insert(0,g);exception=None;steps=[];start=time.monotonic()
try:
 with contextlib.ExitStack() as stack:
  for obj,name in ((subprocess,'Popen'),(os,'kill'),(os,'killpg'),(os,'system'),(socket.socket,'connect'),(socket.socket,'bind'),(mmap,'mmap'),(signal,'signal'),(signal,'setitimer')):stack.enter_context(unittest.mock.patch.object(obj,name,deny(name)))
  with (O/'stdout.txt').open('x') as out,(O/'stderr.txt').open('x') as err,contextlib.redirect_stdout(out),contextlib.redirect_stderr(err):
   prompt=read(A/'glm-flash-inputs-v1/probe-plan.json')['prompt']
   args=['--run','--prompt',prompt['path'],'--prompt-sha256',prompt['sha256'],'--outer-python',str(R/'.venv/bin/python'),'--out',str(I)]
   write(O/'input-command.json',[str(R/'.venv/bin/python'),'-I','-B',str(S/'prepare_inputs.py'),*args]);rc=load('prepare_inputs').main(args);steps.append(dict(name='CPU_inputs',returncode=rc));assert rc==0
   b=read(S/'predecessor-manifest.json')['binding']
   args=['--run','--out',str(G),'--owner',str(A/'glm-stacked-owner-v1'),'--journal',str(A/'glm-stacked-cold-journal-v1'),'--analysis',str(AN/'closed-audit.json'),'--python',str(R/'.venv/bin/python'),'--probe-plan',str(I/'probe-plan.json'),'--probe-plan-sha256',sha(I/'probe-plan.json'),'--verification',str(I/'model-runtime-inventory.json'),'--verification-sha256',sha(I/'model-runtime-inventory.json'),'--predecessor-journal',b['path'],'--predecessor-journal-sha256',b['sha256'],'--predecessor-audit',b['audit']['path'],'--predecessor-audit-sha256',b['audit']['sha256'],'--child-limit-bytes','236223201280','--interval-seconds','1','--probe-seconds','5']
   write(O/'registration-command.json',[str(R/'.venv/bin/python'),'-I','-B',str(S/'preregister.py'),*args]);rc=load('preregister').main(args);steps.append(dict(name='CPU_registration',returncode=rc));assert rc==0
   # Actual inventory verification runs before a future child spawn and again
   # after cleanup; exercise it now without System, backend or payload hash.
   owner=load('owner');probe=owner.helper('../glm_flash/load_probe.py');plan=read(G/'owner-plan.json')
   proof=owner.verify_inventory(plan,probe,lambda:None);write(O/'actual-derived-inventory-proof.json',proof);steps.append(dict(name='actual_inventory',status=proof['status']))
except BaseException as e:exception=dict(type=type(e).__name__,message=str(e),traceback=traceback.format_exc())
finally:sys.meta_path.remove(g)
after={str(p):sha(p) for p in files};write(O/'source-after.json',after)
v=dict(status='PASS' if exception is None and not forbidden and before==after else 'FAIL',steps=steps,exception=exception,forbidden=forbidden,source_unchanged=before==after,elapsed_seconds=time.monotonic()-start,service_prepared=False,model_loaded=False,scope='Actual CPU inputs/registration and supplied runtime hashes/header-stat admission only; no process/network/service/backend/timer action or new payload hash.')
write(O/'result.json',v);print(json.dumps(v));raise SystemExit(0 if v['status']=='PASS' else 1)
