import contextlib, hashlib, importlib.abc, json, pathlib, sys, time, traceback, unittest
R=pathlib.Path('/Users/ryanli/code/minivllm'); N=R/'bench/m5_ultra_256gb/nvfp4_m8_shared'; O=N/'cpu-v2';O.mkdir()
class NoBackend(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in {'mlx','numpy','torch'}: raise AssertionError('forbidden CPU test import: '+fullname)
def no_operations(event,args):
 if event in {'subprocess.Popen','socket.__new__','socket.connect','socket.bind','os.kill','os.fork','os.system'}: raise AssertionError('forbidden CPU test operation: '+event)
sys.meta_path.insert(0,NoBackend());sys.addaudithook(no_operations);sys.path.insert(0,str(R))
def write(name,value):
 with (O/name).open('x') as f:json.dump(value,f,indent=2);f.write('\n')
def pins():return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(N.glob('*.py'))}
write('command.json',dict(argv=sys.argv,executable=sys.executable,backend_imports_forbidden=True,process_network_operations_forbidden=True));before=pins();write('source-before.json',before)
(O/'runner.py').write_bytes(pathlib.Path(__file__).read_bytes());sources=O/'sources';sources.mkdir()
for p in [*N.glob('*.py'),N/'references.json']:(sources/p.name).write_bytes(p.read_bytes())
start=time.monotonic();result=None;error=None
with (O/'stdout.txt').open('x') as out,(O/'stderr.txt').open('x') as err,contextlib.redirect_stdout(out),contextlib.redirect_stderr(err):
 try:
  suite=unittest.defaultTestLoader.loadTestsFromNames(['bench.m5_ultra_256gb.nvfp4_m8_shared.test_cpu','bench.m5_ultra_256gb.nvfp4_m8_shared.test_probe_cpu']);result=unittest.TextTestRunner(verbosity=2).run(suite)
 except BaseException:
  error=traceback.format_exc();err.write(error)
after=pins();write('source-after.json',after);record=dict(status='PASS' if result and result.wasSuccessful() and before==after and not error else 'FAIL',tests=result.testsRun if result else 0,failures=len(result.failures) if result else 0,errors=len(result.errors) if result else 0,skips=len(result.skipped) if result else 0,runner_error=error,elapsed_s=time.monotonic()-start,sources_unchanged=before==after,forbidden_imports_present=[x for x in sys.modules if x.split('.')[0] in {'mlx','numpy','torch'}]);write('result.json',record);print(json.dumps(record));raise SystemExit(0 if record['status']=='PASS' else 1)
