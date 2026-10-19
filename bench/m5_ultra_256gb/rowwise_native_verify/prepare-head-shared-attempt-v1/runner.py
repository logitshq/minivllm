import contextlib,hashlib,importlib.abc,json,pathlib,runpy,sys,time,traceback
R=pathlib.Path('/Users/ryanli/code/minivllm');N=R/'bench/m5_ultra_256gb/rowwise_native_verify';O=N/'prepare-head-shared-attempt-v1';O.mkdir()
class NoBackend(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in {'mlx','numpy','torch'}:raise AssertionError('forbidden CPU preparation import: '+fullname)
def deny(event,args):
 if event in {'subprocess.Popen','socket.__new__','socket.connect','socket.bind','os.kill','os.fork','os.system'}:raise AssertionError('forbidden CPU preparation operation: '+event)
sys.meta_path.insert(0,NoBackend());sys.addaudithook(deny)
def write(name,data):
 with (O/name).open('x') as f:json.dump(data,f,indent=2);f.write('\n')
argv=[str(N/'prepare_head.py'),'--run','--out',str(N/'head-shared-prereg-v1'),'--child',str(R/'.artifacts/m5-ultra-256gb/rowwise-head-shared-v1-child'),'--audit-out',str(N/'head-shared-prereg-v1/independent-audit.json')]
write('command.json',dict(argv=[sys.executable,'-I','-B',*argv],actual_wrapper_argv=sys.argv,execution='runpy.run_path with exact sys.argv and forbidden backend/process operations'))
(O/'runner.py').write_bytes(pathlib.Path(__file__).read_bytes());code=None;error=None;start=time.monotonic()
with (O/'stdout.txt').open('x') as out,(O/'stderr.txt').open('x') as err,contextlib.redirect_stdout(out),contextlib.redirect_stderr(err):
 try:
  sys.argv=argv;runpy.run_path(argv[0],run_name='__main__');code=0
 except SystemExit as e:code=e.code
 except BaseException:error=traceback.format_exc();err.write(error)
record=dict(status='PASS' if code==0 and error is None else 'FAIL',returncode=code,error=error,elapsed_s=time.monotonic()-start,forbidden_imports_present=[n for n in sys.modules if n.split('.')[0] in {'mlx','numpy','torch'}]);write('result.json',record);print(json.dumps(record));raise SystemExit(0 if record['status']=='PASS' else 1)
