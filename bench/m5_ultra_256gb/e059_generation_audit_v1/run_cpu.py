"""Guarded CPU tests, fresh attempt directory, source snapshots and real exit."""
import builtins
import contextlib
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import time
import unittest

HERE=Path(__file__).resolve().parent
attempt=Path(sys.argv[1]);attempt.mkdir(exist_ok=False)
sources=[HERE/'audit.py',HERE/'test_audit.py',Path(__file__).resolve(),Path('/Users/ryanli/code/minivllm/bench/m5_ultra_256gb/e058_agent_audit_v2/audit.py')]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
before={str(p):sha(p) for p in sources}
(attempt/'sources').mkdir()
for i,p in enumerate(sources):shutil.copyfile(p,attempt/'sources'/f'{i}-{p.name}')
(attempt/'sources-before.json').write_text(json.dumps(before,indent=2)+'\n')
(attempt/'argv.json').write_text(json.dumps([sys.executable,*sys.argv],indent=2)+'\n')
forbidden=[]
def guard(event,args):
    if event in {'subprocess.Popen','os.system','os.fork','os.forkpty','os.posix_spawn','os.kill','os.killpg','socket.__new__','socket.connect','socket.bind'}:
        forbidden.append(event);raise RuntimeError('forbidden CPU side effect '+event)
    if event=='import' and args[0].split('.')[0] in {'mlx','torch','tensorflow','jax','cupy'}:
        forbidden.append('import '+args[0]);raise RuntimeError('forbidden backend '+args[0])
sys.addaudithook(guard)
started=time.monotonic();rc=1;count=0;failures=[];errors=[]
with (attempt/'stdout.txt').open('w') as stdout,(attempt/'stderr.txt').open('w') as stderr,contextlib.redirect_stdout(stdout),contextlib.redirect_stderr(stderr):
    try:
        spec=importlib.util.spec_from_file_location('tests',HERE/'test_audit.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        suite=unittest.defaultTestLoader.loadTestsFromModule(m);result=unittest.TextTestRunner(stream=stderr,verbosity=2).run(suite)
        count=result.testsRun;failures=[str(x[0]) for x in result.failures];errors=[str(x[0]) for x in result.errors];rc=0 if result.wasSuccessful() and not forbidden else 1
    except BaseException:
        import traceback;traceback.print_exc();rc=1
    after={str(p):sha(p) for p in sources};unchanged=before==after
    if not unchanged:rc=1
(attempt/'sources-after.json').write_text(json.dumps(after,indent=2)+'\n')
receipt=dict(status='PASS' if rc==0 else 'FAIL',tests=count,failures=failures,errors=errors,forbidden=forbidden,source_unchanged=unchanged,exit_code=rc,elapsed_s=time.monotonic()-started)
(attempt/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');(attempt/'exit-code.txt').write_text(str(rc)+'\n')
print(json.dumps(receipt));raise SystemExit(rc)
