"""Guarded CPU-only prepared server/capture tests; retain complete evidence."""
import builtins
from contextlib import ExitStack, redirect_stdout, redirect_stderr
import datetime
import hashlib
import importlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import unittest
from unittest.mock import patch

ROOT=Path('/Users/ryanli/code/minivllm')
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
ART=ROOT/'.artifacts/m5-ultra-256gb'
closed=json.loads((ART/'route-decode-serving-v1-execution/final.json').read_bytes())
assert closed['journal_state']=='RESTORED' and closed['recovery_returncode']==0
MODULES=['bench.m5_ultra_256gb.route_decode_graph.test_http_server_cpu',
         'bench.m5_ultra_256gb.route_decode_graph.test_http_capture_cpu']
ROUTE=ROOT/'bench/m5_ultra_256gb/route_decode_graph'
DONOR=ROOT/'bench/m5_ultra_256gb/pareto_realistic'
OLD=ROOT/'bench/m5_ultra_256gb/long_decode_graph'
paths=[ROUTE/n for n in ('http_server_v2.py','test_http_server_cpu.py','http_capture.py',
    'test_http_capture_cpu.py','integration.py','candidate.py','serving_qualification.py')]
paths += [DONOR/n for n in ('native_server.py','http_capture_v2.py','test_http_capture_v2_cpu.py',
    'native_cleanup.py','client.py','http_qualification.py','http_join.py','http_replay.py','qualify_native.py')]
paths += [OLD/n for n in ('qualify.py','candidate.py','model-source-pins.json')]
paths += [DONOR/'prereg-qualification-v1/plan.json']
plan=json.loads(paths[-1].read_bytes())
paths += [Path(plan[k]['path']) for k in ('fixtures','source_manifest','model_manifest')]
paths=list(dict.fromkeys(paths))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name,value):
    with (OUT/name).open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
before={str(p):sha(p) for p in paths}
source_dir=OUT/'source-before'
for p in paths:
    target=source_dir/p.relative_to(ROOT)
    target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('xb') as f:f.write(p.read_bytes())
    assert sha(target)==before[str(p)]
assert before[str(ROUTE/'http_server_v2.py')]=='8ed952b4870465718a33e0af7e4ecddbf3fbd6265571e7efd3467b904c5621b7'
assert before[str(ROUTE/'test_http_server_cpu.py')]=='36b3413b5250f88c52dd5ba3a1d4d76b30a214a26b8279736c1f786e7e4b1a76'
assert before[str(ROUTE/'http_capture.py')]=='b331a02988413356e113b3a1283ab0eb8d40de17796b5623a17df312c214151f'
assert before[str(ROUTE/'test_http_capture_cpu.py')]=='052afe7b1c02e2b5ceaf7f58428f114f70e5fdf02329a9bcca6d0ac53027ea21'
record=dict(schema='E024-route-http-prepared-cpu-v1',command=[sys.executable,'-I','-B',str(Path(__file__).resolve())],
    cwd=str(ROOT),modules=MODULES,expected_tests=23,source_pins=before,runner_sha256=sha(__file__),
    started_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    scope='CPU mock mechanisms only; no real sockets/processes/model/GPU. No numerical/HTTP/performance qualification.')
save('command.json',record)
forbidden=('mlx','mlx_lm','mlx_vlm','torch','numpy','tutorial_code','omlx')
original_import=builtins.__import__
def guarded(name,*args,**kwargs):
    if name.split('.')[0] in forbidden:raise AssertionError('forbidden runtime import: '+name)
    return original_import(name,*args,**kwargs)
def denied(*args,**kwargs):raise AssertionError('real subprocess/socket/signal backend is forbidden')
def timeout(*args):raise TimeoutError('guarded CPU suite deadline')
signal.signal(signal.SIGALRM,timeout);signal.alarm(60)
started=time.monotonic();result=None;error=None;loaded_count=None
with (OUT/'stdout.txt').open('x') as stdout,(OUT/'stderr.txt').open('x') as stderr:
  with redirect_stdout(stdout),redirect_stderr(stderr):
    try:
      with ExitStack() as stack:
        for target in ('subprocess.run','subprocess.Popen','socket.create_connection',
                       'socket.socket.connect','socket.socket.connect_ex','socket.socket.bind',
                       'socket.socket.listen','os.kill'):
          stack.enter_context(patch(target,side_effect=denied))
        stack.enter_context(patch('builtins.__import__',guarded))
        modules=[importlib.import_module(n) for n in MODULES]
        for mod in modules:
          backend=getattr(mod.m,'System',None)
          if backend is not None:stack.enter_context(patch.object(backend,'__init__',side_effect=denied))
        suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromModule(m) for m in modules)
        loaded_count=suite.countTestCases()
        assert loaded_count==23,loaded_count
        result=unittest.TextTestRunner(verbosity=2,stream=stderr).run(suite)
    except BaseException as exc:
      import traceback
      traceback.print_exc()
      error=dict(type=type(exc).__name__,message=str(exc))
signal.alarm(0)
after={str(p):sha(p) for p in paths}
loaded_forbidden=sorted(n for n in sys.modules if n.split('.')[0] in forbidden)
passed=(error is None and result is not None and result.wasSuccessful() and result.testsRun==23
        and not result.skipped and not result.expectedFailures and not result.unexpectedSuccesses
        and before==after and not loaded_forbidden)
record.update(status='PASS' if passed else 'FAIL',elapsed_seconds=time.monotonic()-started,
    loaded_test_count=loaded_count,tests=None if result is None else result.testsRun,
    failures=None if result is None else len(result.failures),errors=None if result is None else len(result.errors),
    skipped=None if result is None else len(result.skipped),runner_error=error,
    source_pins_after=after,source_pins_unchanged=before==after,forbidden_modules_loaded=loaded_forbidden,
    stdout_sha256=sha(OUT/'stdout.txt'),stderr_sha256=sha(OUT/'stderr.txt'))
save('result.json',record)
print(json.dumps({k:record[k] for k in ('status','tests','failures','errors','elapsed_seconds','source_pins_unchanged','forbidden_modules_loaded')}))
raise SystemExit(0 if passed else 1)
