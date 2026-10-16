"""Own one child, retain raw resource samples, wait/reap and record cleanup."""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import resources

ap = argparse.ArgumentParser()
ap.add_argument('--out', type=Path, required=True)
ap.add_argument('--cwd', type=Path, required=True)
ap.add_argument('command', nargs=argparse.REMAINDER)
a = ap.parse_args()
a.out.mkdir()
samples = []
def write(name, data):
    (a.out / name).write_text(json.dumps(data, indent=2) + '\n')
def probe(label, pid=None):
    raw = resources.snapshot(label, pid)
    write(f'resource-{len(samples):03d}.json', raw)
    samples.append(raw)
    resources.guard(samples)
    assert raw['pressure_level_value'] == 1
    assert raw['power_metadata']['source'] == 'AC Power'
    assert raw['processinfo_thermal_metadata']['code'] in (0, 1)
    state = Path('/Users/ryanli/Documents/Codex/2026-10-15/w/outputs/local-model-comparison.json')
    if state.exists():
        d = json.loads(state.read_text())
        if d['status'].startswith('benchmarking'):
            raise RuntimeError('Another chat began GPU benchmarking; stop our child')
write('attempt.json', {'owner_pid': os.getpid(), 'command': a.command, 'cwd': str(a.cwd)})
child = None
error = None
def stop(signum, frame):
    raise KeyboardInterrupt(str(signum))
signal.signal(signal.SIGTERM, stop)
try:
    probe('preflight')
    with (a.out / 'child.log').open('w') as log:
        child = subprocess.Popen(a.command, cwd=a.cwd, stdout=log, stderr=subprocess.STDOUT)
        write('child.json', {'pid': child.pid})
        while child.poll() is None:
            probe('child-running', child.pid)
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        child.wait()
        assert child.returncode == 0, f'child exit {child.returncode}'
except BaseException:
    error = traceback.format_exc()
    write('failure.json', {'error': error})
finally:
    if child:
        if child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=15)
        child.wait()
        p = subprocess.run(['/bin/ps', '-p', str(child.pid), '-o', 'pid='], capture_output=True, text=True)
        assert p.returncode == 1 and not p.stdout
        write('release.json', {'pid': child.pid, 'returncode': child.returncode, 'waited_reaped': True,
                              'strict_absence': True})
    probe('postflight')
    write('resources.json', resources.deltas(samples))
if error:
    raise RuntimeError(error)
write('complete.json', {'status': 'PASS', 'owner_pid': os.getpid()})
print('PASS', a.out, flush=True)
