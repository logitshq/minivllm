import contextlib, hashlib, importlib.abc, json, pathlib, sys, time, unittest
R=pathlib.Path('/Users/ryanli/code/minivllm'); N=R/'bench/m5_ultra_256gb/rowwise_native_verify'; O=N/'cpu-head-reuse-v1';O.mkdir()
class NoBackend(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in {'mlx','numpy','torch','subprocess','socket'}: raise AssertionError('forbidden CPU test import: '+fullname)
sys.meta_path.insert(0,NoBackend());sys.path.insert(0,str(R))
def write(name,value):
 with (O/name).open('x') as f:json.dump(value,f,indent=2);f.write('\n')
def pins():return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(N.glob('*.py'))}
write('command.json',dict(argv=sys.argv,executable=sys.executable,backend_imports_forbidden=True));before=pins();write('source-before.json',before)
(O/'runner.py').write_bytes(pathlib.Path(__file__).read_bytes());sources=O/'sources';sources.mkdir()
for p in N.glob('*.py'):(sources/p.name).write_bytes(p.read_bytes())
start=time.monotonic()
with (O/'stdout.txt').open('x') as out,(O/'stderr.txt').open('x') as err,contextlib.redirect_stdout(out),contextlib.redirect_stderr(err):
 suite=unittest.defaultTestLoader.loadTestsFromNames(['bench.m5_ultra_256gb.rowwise_native_verify.'+m for m in ('test_candidate_cpu','test_qualify_cpu','test_audit_cpu')]);result=unittest.TextTestRunner(verbosity=2).run(suite)
after=pins();write('source-after.json',after);record=dict(status='PASS' if result.wasSuccessful() and before==after else 'FAIL',tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),skips=len(result.skipped),elapsed_s=time.monotonic()-start,sources_unchanged=before==after,forbidden_imports_present=[x for x in sys.modules if x.split('.')[0] in {'mlx','numpy','torch'}]);write('result.json',record);print(json.dumps(record));raise SystemExit(0 if record['status']=='PASS' else 1)
