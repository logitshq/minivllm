"""Root sole-GPU owner; wait-only during two serial official MTP numerical qualification arms."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

P = Path('/private/tmp/minivllm-engine-optimization-20261014')
D = P / 'v109-omlx-external-mtp-qual-build1'
O = P / 'v23-root-control-build1/omlx-mtp-v109-attempt1'
sys.path.insert(0, str(P / 'v25-c1-screen-build1'))
from support import identity, process_parent, strict_absence, write

assert sys.executable == '/Users/ryanli/code/minivllm/.venv/bin/python'
assert not any(k.startswith(('MLX_', 'METAL_', 'DYLD_', 'OMLX_', 'MONOLITH_', 'LITHOS_'))
               or k in ('PYTHONPATH', 'PYTHONHOME', 'PYTHONDONTWRITEBYTECODE') for k in os.environ)
pins = '72856f12da32bc6438ef7aa050462e0bfbac35f0a074e718c07723cf28d55871'
assert hashlib.sha256((D / 'pins.json').read_bytes()).hexdigest() == pins
O.mkdir()
command = [sys.executable, '-B', str(D / 'fullqual.py'), '--run', '--pins-sha256', pins,
           '--out', str(O / 'child-output')]
r = {'owner_pid': os.getpid(), 'command': command, 'deadline_seconds': 900,
     'driver_sha256': hashlib.sha256((D / 'fullqual.py').read_bytes()).hexdigest(),
     'pins_sha256': pins, 'resource_policy': 'driver samples only outside request/group windows',
     'stage': 'before-spawn'}
write(O / 'owner-receipt.json', r)
proc = None; birth = None
try:
    with (O / 'stdout.log').open('x') as out, (O / 'stderr.log').open('x') as err:
        proc = subprocess.Popen(command, env=dict(os.environ), stdout=out, stderr=err, start_new_session=True)
        birth = identity(proc.pid)
        assert birth is not None and process_parent(proc.pid)['ppid'] == os.getpid()
        r.update(driver_pid=proc.pid, driver_birth=birth, stage='running')
        write(O / 'owner-receipt.json', r)
        print(json.dumps({'owner_pid': os.getpid(), 'driver_pid': proc.pid, 'output': str(O)}), flush=True)
        r['returncode'] = proc.wait(timeout=900)
except BaseException as exc:
    r['owner_error'] = {'type': type(exc).__name__, 'message': str(exc)}
    raise
finally:
    if proc is not None:
        if proc.poll() is None:
            assert identity(proc.pid) == birth and process_parent(proc.pid)['ppid'] == os.getpid()
            proc.terminate()
            try: proc.wait(timeout=90)
            except subprocess.TimeoutExpired:
                assert identity(proc.pid) == birth and process_parent(proc.pid)['ppid'] == os.getpid()
                proc.kill(); proc.wait(timeout=20)
                r['driver_force_killed'] = True
        r.update(returncode=proc.returncode, waited_reaped=True, stage='reaped', absence=strict_absence(proc.pid))
        write(O / 'owner-receipt.json', r)
print(json.dumps(r), flush=True)
raise SystemExit(r['returncode'])
