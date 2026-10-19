import contextlib,hashlib,importlib.abc,importlib.util,json,mmap,os,socket,subprocess,sys,time,traceback,unittest,unittest.mock
from pathlib import Path
R=Path('/Users/ryanli/code/minivllm');S=R/'bench/m5_ultra_256gb/nvfp4_multirow_reuse';O=R/'.artifacts/m5-ultra-256gb/nvfp4-multirow-reuse-cpu-v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):
 with p.open('x') as f:json.dump(v,f,indent=2);f.write('\n')
O.mkdir();(O/'sources').mkdir();files=[S/'candidate.py',S/'test_cpu.py',S/'README.md',Path(__file__).resolve()];before={str(p):sha(p) for p in files}
for p in files:(O/'sources'/p.name).write_bytes(p.read_bytes())
write(O/'source-before.json',before);write(O/'command.json',dict(argv=[sys.executable,'-I','-B',str(Path(__file__).resolve())],cwd=str(R),python=sys.version));forbidden=[]
def deny(label):
 def f(*a,**k):forbidden.append(label);raise AssertionError('forbidden '+label)
 return f
class Guard(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0] in ('mlx','mlx_lm','mlx_vlm','omlx','torch','transformers','safetensors','numpy','ctypes'):deny('backend '+name)()
def audit(event,args):
 if event.startswith(('socket.','subprocess.','ctypes.')) or event in ('os.system','os.fork','os.forkpty','os.posix_spawn','os.kill','os.killpg','mmap.__new__'):deny('audit '+event)()
sys.addaudithook(audit)
g=Guard();sys.meta_path.insert(0,g);result=None;error=None;t=time.monotonic()
try:
 with contextlib.ExitStack() as stack:
  for obj,name in ((subprocess,'Popen'),(os,'kill'),(os,'killpg'),(os,'system'),(socket,'socket'),(socket,'create_connection'),(mmap,'mmap')):stack.enter_context(unittest.mock.patch.object(obj,name,deny(name)))
  with (O/'stdout.txt').open('x') as out,(O/'stderr.txt').open('x') as err,contextlib.redirect_stdout(out),contextlib.redirect_stderr(err):
   sp=importlib.util.spec_from_file_location('nvfp4_multirow_cpu_test',S/'test_cpu.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);suite=unittest.defaultTestLoader.loadTestsFromModule(m);assert suite.countTestCases()==5;result=unittest.TextTestRunner(stream=err,verbosity=2).run(suite)
except BaseException as e:error=dict(type=type(e).__name__,message=str(e),traceback=traceback.format_exc())
finally:sys.meta_path.remove(g)
after={str(p):sha(p) for p in files};write(O/'source-after.json',after)
row=dict(status='PASS' if result and result.wasSuccessful() and result.testsRun==5 and before==after and not forbidden and error is None else 'FAIL',tests_run=result.testsRun if result else 0,failures=[(a.id(),b)for a,b in result.failures]if result else[],errors=[(a.id(),b)for a,b in result.errors]if result else[],skipped=list(result.skipped)if result else[],exception=error,forbidden=forbidden,source_unchanged=before==after,elapsed_seconds=time.monotonic()-t,scope='Five source/dispatch mechanism tests with injected fake backend only; no actual MLX/model/GPU/Metal compilation, mmap, child process, signals or network; no arithmetic, numerical or performance qualification.')
write(O/'result.json',row);print(json.dumps({k:row[k]for k in ('status','tests_run','exception','forbidden','source_unchanged','elapsed_seconds')}));raise SystemExit(0 if row['status']=='PASS' else 1)
