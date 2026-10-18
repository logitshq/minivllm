"""Second recorded fake-only route-formal run, including registration regression.

No real subprocess/network/process signals or model/backend imports. Module
names and expected counts are explicit command arguments, never auto-skipped.
"""
import argparse,contextlib,datetime,hashlib,importlib.abc,importlib.metadata,json,os
from pathlib import Path
import socket,subprocess,sys,time,traceback,unittest
from unittest.mock import patch
ROOT=Path('/Users/ryanli/code/minivllm'); ART=ROOT/'.artifacts/m5-ultra-256gb'
PYTHON=ROOT/'.artifacts/omlx-baseline/.venv/bin/python'
SUITES={
 'bench.m5_ultra_256gb.route_http_formal.test_owner_cpu':14,
 'bench.m5_ultra_256gb.route_http_formal.test_controller_cpu':12,
 'bench.m5_ultra_256gb.route_http_formal.test_audit_restore_cpu':11,
 'bench.m5_ultra_256gb.route_http_formal.test_audit_formal_cpu':13,
 'bench.m5_ultra_256gb.route_http_formal.test_execute_cpu':12,
 'bench.m5_ultra_256gb.route_http_formal.test_preregister_cpu':2,
}
# All six whole modules include the real registrar-write regression.
DRIVER_PENDING=False
SEALED_SOURCES={
 'owner.py':'455683cb8dc9d09b1e7c85a4ae586d8ffadf58bf6a637d23334975fabec3121d',
 'controller.py':'98d48eb4ad608be5d8d65f6a28622126483ee96ff9a312b3cbd16ee4c63817f3',
 'audit_restore.py':'90a980d0dbe6a61fc1bab2ac6b2edaa9bf012bb60726cd0742801d11410eb0b1',
 'audit_formal.py':'ed8ad64c3eb5fed052effd05981a386006f4ea1f752463bb0ada0c2f0e4ea5e0',
 'formal.py':'0e590191d32b12027bc8b5f4f76335ed42a633a4179d90431427aa42c6e18f30',
 'execute.py':'15deb40c8a72eaa08c677e9a68f30261e106b1c9efe0465b481656f075e9b9fa',
 'preregister.py':'bc91a33edfaf683753872e7c245a16488f4ae00264be9efda8533db313360e0f',
 'test_owner_cpu.py':'679548f1fdc05891d127f5982d01f82d9150ef6a0e53715ace2ed12601733c25',
 'test_controller_cpu.py':'8e1df8ed3c4babfc04e6622ab21b4d7a2aa692e8c87b85616611c9e93e34ca51',
 'test_audit_restore_cpu.py':'1bd1b91d7217e685d5c3576b0b0f5da76680adaeef177b65cff07f34ac617924',
 'test_audit_formal_cpu.py':'8d2612a0d91e09131a5f134392b0164297c53f3c23d848173226108414ad08ed',
 'test_execute_cpu.py':'a594ab8885b6ba24dd24e644a037b06c6af3a47846d02ae22796a797bc4e96f8',
 'test_preregister_cpu.py':'b9dfabfcbffb54e7df15e38bf5ed9c115f3f8d5aa0e8ff8171a1612aaa5931f6',
}


FORBIDDEN={'mlx','omlx','mlx_lm','mlx_vlm','torch','numpy','tutorial_code','monolith'}
PREVIOUS_CPU=ART/'route-formal-prepared-cpu-v1/result.json'
PREVIOUS_CPU_SHA='c8706e6f14e3adb92a2770f3333889a29be3c76b1b12ecf6512c06625293ce31'
PREVIOUS_REGISTRATION=ART/'route-http-formal-v1-registration/result.json'
PREVIOUS_REGISTRATION_SHA='fa9e4b860f246d09808fe01bfb8a64a41d4346098406659dc57e97abf38b6c14'
SOURCE_DIRS=('route_http_formal','route_http_cycle','route_decode_graph','identity_refill',
 'pareto_realistic','long_decode_graph','long_decode_isolation','gdn_prework_unbound',
 'omlx_d512_formal_retry','omlx_d512_formal','omlx_d512','omlx_cold_restart',
 'cold_cycle','cold_cycle_v2','cold_cycle_v3','lithos_d512_retry','lithos_d512_formal')

def need(ok,msg):
 if not ok:raise AssertionError(msg)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def read(p):return json.loads(p.read_bytes())
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--suite',action='append',required=True);parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
 need(Path(sys.executable)==PYTHON and Path(sys.prefix)==PYTHON.parent.parent and sys.flags.isolated==1 and sys.dont_write_bytecode is True,'actual vendor Python -I -B required')
 need(args.out.is_absolute() and args.out.parent==ART and not args.out.exists(),'fresh artifact output')
 suites=[]
 for item in args.suite:
  name,n=item.rsplit(':',1);need(name in SUITES and int(n)==SUITES[name] and len(name.split('.'))==4 and (ROOT/Path(*name.split('.'))).with_suffix('.py').is_file(),'whole registered test module and exact expected count required')
  suites.append(dict(module=name,expected_tests=int(n)))
 need(not DRIVER_PENDING,'driver/test source freeze remains pending; this runner is not executable yet')
 need(len({r['module'] for r in suites})==len(suites) and {r['module'] for r in suites}==set(SUITES),
      'all registered whole modules exactly once; no test filtering')
 for name,digest in SEALED_SOURCES.items():
  need(sha((ROOT/'bench/m5_ultra_256gb/route_http_formal'/name).read_bytes())==digest,'sealed source changed: '+name)
 final=ART/'gdn-prework-canary-v1-execution/final.json';prior=ROOT/'docs/experiments/m5-ultra-256gb/E029-gdn-prework-canary-v1';journal=prior/'restored-journal.json';audit=prior/'restoration-audit.json'
 f,j,a=map(read,(final,journal,audit));need(f['journal_state']==j['state']=='RESTORED' and f['recovery_returncode']==0 and a['status']=='PASS' and a['qualified'] is True and a['journal_sha256']==sha(journal.read_bytes()),'closed restored independent gate')
 previous_cpu,previous_registration=read(PREVIOUS_CPU),read(PREVIOUS_REGISTRATION)
 need(sha(PREVIOUS_CPU.read_bytes())==PREVIOUS_CPU_SHA and previous_cpu['status']=='PASS' and previous_cpu['tests']==62,
      'preserve exact original62 CPU PASS')
 need(sha(PREVIOUS_REGISTRATION.read_bytes())==PREVIOUS_REGISTRATION_SHA and previous_registration['status']=='FAIL'
      and previous_registration['error']['message']=='route formal plan SHA' and previous_registration['service_changed'] is False
      and previous_registration['service_prepared'] is False,'preserve exact original registration FAIL without service action')
 args.out.mkdir();(args.out/'runner.py').write_bytes(Path(__file__).read_bytes())
 (args.out/'previous-cpu-result.json').write_bytes(PREVIOUS_CPU.read_bytes())
 (args.out/'previous-registration-result.json').write_bytes(PREVIOUS_REGISTRATION.read_bytes())
 rec=dict(schema='E024-route-formal-prepared-cpu-v2',status='RUNNING',started_at=now(),argv=[str(PYTHON),'-I','-B',str(Path(__file__).resolve()),*sys.argv[1:]],suites=suites,expected_tests=sum(r['expected_tests'] for r in suites),python_executable=sys.executable,python_resolved=str(Path(sys.executable).resolve()),python_version=sys.version,tests=0,failures=0,errors=0,skipped=0,denied_operations=[],test_outcomes=[],source_before={},source_after={},scope='Fake CPU wiring/validator tests only. No GPU/model/service/physical-capacity/numerical/performance qualification.')
 rec['previous_attempts']=[dict(path=str(PREVIOUS_CPU),sha256=PREVIOUS_CPU_SHA,status='PASS',tests=62),dict(path=str(PREVIOUS_REGISTRATION),sha256=PREVIOUS_REGISTRATION_SHA,status='FAIL',scope='registration only, no service action')]
 rec['sealed_source_pins']={str(ROOT/'bench/m5_ultra_256gb/route_http_formal'/n):h for n,h in SEALED_SOURCES.items()}
 rec['restoration_gate_scope']='Historical independently closed canary evidence only; no live service observation or authorization to run devices.'
 for name,p in [('execution-final.json',final),('restored-journal.json',journal),('restoration-audit.json',audit)]:
  raw=p.read_bytes();(args.out/name).write_bytes(raw);rec.setdefault('restoration_gate',{})[name]=dict(path=str(p),sha256=sha(raw))
 paths={Path(__file__).resolve()};base=ROOT/'bench/m5_ultra_256gb'
 for name in SOURCE_DIRS:paths.update((base/name).glob('*.py'))
 paths.update((ROOT/'tutorial_code/miniengine').rglob('*.py'))
 paths.update(base/n for n in ('resources.py','resources_v2.py','resource_base.py','resource_reparse.py','long_decode_graph/model-source-pins.json','cold_cycle/lifecycle.json','cold_cycle_v2/lifecycle.json','cold_cycle_v3/lifecycle.json'))
 def save():write(args.out/'result.json',rec)
 def deny(kind,detail):
  rec['denied_operations'].append(dict(kind=kind,detail=str(detail)));raise AssertionError('forbidden real operation: '+kind)
 def hook(event,args):
  if event in ('subprocess.Popen','os.system','os.fork','os.forkpty','os.posix_spawn','os.exec','os.spawn','os.kill','os.killpg','socket.connect','socket.bind'):deny(event,args)
 class Imports(importlib.abc.MetaPathFinder):
  def find_spec(self,fullname,path=None,target=None):
   if fullname.split('.')[0] in FORBIDDEN:deny('backend import',fullname)
   return None
 class Result(unittest.TextTestResult):
  def startTest(self,test):rec['test_outcomes'].append(dict(id=test.id(),status='RUNNING'));super().startTest(test)
  def addSuccess(self,test):rec['test_outcomes'][-1]['status']='PASS';super().addSuccess(test)
  def addFailure(self,test,error):rec['test_outcomes'][-1].update(status='FAIL',detail=self._exc_info_to_string(error,test));super().addFailure(test,error)
  def addError(self,test,error):rec['test_outcomes'][-1].update(status='ERROR',detail=self._exc_info_to_string(error,test));super().addError(test,error)
  def addSkip(self,test,reason):rec['test_outcomes'][-1].update(status='SKIP',reason=reason);super().addSkip(test,reason)
  def addExpectedFailure(self,test,error):rec['test_outcomes'][-1]['status']='EXPECTED_FAILURE';super().addExpectedFailure(test,error)
  def addUnexpectedSuccess(self,test):rec['test_outcomes'][-1]['status']='UNEXPECTED_SUCCESS';super().addUnexpectedSuccess(test)
  def addSubTest(self,test,subtest,error):
   if error is not None:rec['test_outcomes'][-1].update(status='FAIL',subtest=str(subtest),detail=self._exc_info_to_string(error,test))
   super().addSubTest(test,subtest,error)
 guard=None;start=time.monotonic();save()
 try:
  for p in sorted(paths):
   raw=p.read_bytes();rec['source_before'][str(p)]=dict(sha256=sha(raw),bytes=len(raw));dest=args.out/'frozen-source'/str(p).lstrip('/');dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(raw)
  need(not any(n.split('.')[0] in FORBIDDEN for n in sys.modules),'no backend preloaded')
  os.chdir(ROOT);sys.path.insert(0,str(ROOT));guard=Imports();sys.meta_path.insert(0,guard);sys.addaudithook(hook);save()
  with (args.out/'stdout.txt').open('x') as out,(args.out/'stderr.txt').open('x') as err,contextlib.redirect_stdout(out),contextlib.redirect_stderr(err),patch.object(subprocess,'Popen',side_effect=lambda *a,**k:deny('Popen',a)),patch.object(subprocess,'run',side_effect=lambda *a,**k:deny('subprocess.run',a)),patch.object(socket.socket,'connect',side_effect=lambda *a,**k:deny('socket.connect',a)),patch.object(socket.socket,'connect_ex',side_effect=lambda *a,**k:deny('socket.connect_ex',a)),patch.object(socket.socket,'bind',side_effect=lambda *a,**k:deny('socket.bind',a)):
   loaded=[]
   for row in suites:
    suite=unittest.defaultTestLoader.loadTestsFromName(row['module']);row['discovered_tests']=suite.countTestCases();loaded.append(suite)
   result=unittest.TextTestRunner(stream=err,verbosity=2,resultclass=Result).run(unittest.TestSuite(loaded))
  rec.update(tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),expected_failures=len(result.expectedFailures),unexpected_successes=len(result.unexpectedSuccesses),loader_errors=unittest.defaultTestLoader.errors,failures_detail=[dict(test=t.id(),traceback=s) for t,s in result.failures],errors_detail=[dict(test=t.id(),traceback=s) for t,s in result.errors])
  rec['status']='PASS' if result.wasSuccessful() and result.testsRun==rec['expected_tests'] and not result.skipped and not result.expectedFailures and not result.unexpectedSuccesses and not rec['loader_errors'] and not rec['denied_operations'] and all(r['discovered_tests']==r['expected_tests'] for r in suites) else 'FAIL'
 except BaseException as exc:rec.update(status='FAIL',error=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()))
 finally:
  if guard in sys.meta_path:sys.meta_path.remove(guard)
  for p in sorted(paths):
   try:
    raw=p.read_bytes();rec['source_after'][str(p)]=dict(sha256=sha(raw),bytes=len(raw))
   except BaseException as exc:rec['source_after'][str(p)]=dict(error=str(exc))
  rec['source_unchanged']=bool(rec['source_before']) and rec['source_before']==rec['source_after'];rec['forbidden_modules_loaded']=sorted(n for n in sys.modules if n.split('.')[0] in FORBIDDEN)
  if not rec['source_unchanged'] or rec['forbidden_modules_loaded']:rec['status']='FAIL'
  rec.update(ended_at=now(),elapsed_seconds=time.monotonic()-start);save()
  files={str(p.relative_to(args.out)):dict(sha256=sha(p.read_bytes()),bytes=p.stat().st_size) for p in args.out.rglob('*') if p.is_file()}
  write(args.out/'artifact-manifest.json',dict(schema='cpu-artifact-inventory-v1',files=files))
 print(json.dumps({k:rec[k] for k in ('status','tests','failures','errors','skipped','source_unchanged')}));return 0 if rec['status']=='PASS' else 1
if __name__=='__main__':raise SystemExit(main())
