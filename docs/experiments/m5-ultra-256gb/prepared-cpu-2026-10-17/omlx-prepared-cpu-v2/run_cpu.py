"""One fixed fake-only test rerun; no device/vendor imports or real I/O backend."""
import contextlib
import datetime
import hashlib
import importlib.abc
import importlib.metadata
import json
from pathlib import Path
import socket
import subprocess
import sys
import time
import unittest
from unittest.mock import patch

ROOT=Path('/Users/ryanli/code/minivllm')
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
NAME='bench.m5_ultra_256gb.omlx_d512.test_v2_cpu'
PYTHON=ROOT/'.artifacts/omlx-baseline/.venv/bin/python'
assert Path(sys.executable)==PYTHON
assert Path(sys.prefix)==PYTHON.parent.parent
final=json.loads((ROOT/'.artifacts/m5-ultra-256gb/lithos-d512-formal-v1-execution/final.json').read_bytes())
journal=json.loads((ROOT/'.artifacts/m5-ultra-256gb/lithos-d512-formal-v1-cold-journal/journal.json').read_bytes())
assert final['journal_state']==journal['state']=='RESTORED' and final['recovery_returncode']==0
paths=list(sorted((ROOT/'bench/m5_ultra_256gb/omlx_d512').glob('*.py')))
paths += [ROOT/'bench/m5_ultra_256gb/pareto_realistic'/name for name in
          ('native_cleanup.py','client.py','http_capture_v2.py')]
paths += [ROOT/'.artifacts/omlx-baseline/.venv/lib/python3.13/site-packages/omlx/api/openai_models.py',Path(__file__).resolve()]
def pins():return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
before=pins()
record=dict(schema='omlx-prepared-cpu-review-v2',status='RUNNING',
    command=[str(PYTHON),'-I','-B',str(Path(__file__).resolve())],cwd=str(ROOT),
    python_executable=sys.executable,python_resolved=str(Path(sys.executable).resolve()),prefix=sys.prefix,
    python_version=sys.version,versions={n:importlib.metadata.version(n) for n in ('pydantic','httpx')},
    source_pins_before=before,expected_tests=14,started_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    scope='CPU fixtures, strict protocol validators and fake lifecycle only; no GPU/model/service qualification.')
def save():(OUT/'result.json').write_text(json.dumps(record,indent=2)+'\n')
save()
class DenyDeviceImports(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in ('mlx','omlx','mlx_lm','mlx_vlm','torch'):
            raise AssertionError('real device/vendor import forbidden: '+fullname)
        return None
guard=DenyDeviceImports();sys.meta_path.insert(0,guard)
started=time.monotonic()
try:
    with (OUT/'stdout.txt').open('x') as stdout,(OUT/'stderr.txt').open('x') as stderr, \
         contextlib.redirect_stdout(stdout),contextlib.redirect_stderr(stderr), \
         patch.object(subprocess,'Popen',side_effect=AssertionError('real child forbidden')), \
         patch.object(subprocess,'run',side_effect=AssertionError('real subprocess forbidden')), \
         patch.object(socket.socket,'connect',side_effect=AssertionError('real socket connection forbidden')), \
         patch.object(socket.socket,'connect_ex',side_effect=AssertionError('real socket connection forbidden')), \
         patch.object(socket.socket,'bind',side_effect=AssertionError('real listener forbidden')):
        suite=unittest.defaultTestLoader.loadTestsFromName(NAME)
        result=unittest.TextTestRunner(stream=stderr,verbosity=2).run(suite)
    after=pins()
    passed=result.wasSuccessful() and result.testsRun==14 and not result.skipped and before==after
    record.update(status='PASS' if passed else 'FAIL',tests=result.testsRun,failures=len(result.failures),
        errors=len(result.errors),skipped=len(result.skipped),source_pins_after=after,source_pins_unchanged=before==after)
except BaseException as exc:
    record.update(status='FAIL',error={'type':type(exc).__name__,'message':str(exc)})
finally:
    sys.meta_path.remove(guard)
    record['elapsed_seconds']=time.monotonic()-started
    record['forbidden_modules_loaded']=[n for n in sys.modules if n.split('.')[0] in ('mlx','omlx','mlx_lm','mlx_vlm','torch')]
    if record['forbidden_modules_loaded']:record['status']='FAIL'
    save()
print(json.dumps({k:record.get(k) for k in ('status','tests','failures','errors','elapsed_seconds','forbidden_modules_loaded')}))
raise SystemExit(0 if record['status']=='PASS' else 1)
