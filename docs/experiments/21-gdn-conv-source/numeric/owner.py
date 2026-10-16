"""Root-owned small-array numeric run; no model load or performance claim."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import time

D = Path(__file__).resolve().parent
P = D.parent
sys.path.insert(0, str(P))
import resource_snapshot_power_v1 as resources


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    assert sys.executable == '/Users/ryanli/code/minivllm/.venv/bin/python'
    assert not any(k.startswith(('MLX_', 'METAL_', 'DYLD_')) for k in os.environ)
    out = D / 'numeric-attempt1'
    out.mkdir()
    history = []
    def sample(label):
        raw = resources.snapshot(label, os.getpid())
        write(out / ('resource-%02d.json' % len(history)), raw)
        history.append(raw)
        resources.deltas(history)
        resources.guard(history)
        assert raw['pressure_level_value'] == 1
        assert raw['power_metadata']['source'] == 'AC Power'
        assert raw['processinfo_thermal_metadata']['code'] in (0, 1)
        assert raw['known_inference_process_count'] == 0
    sample('before-small-numeric')
    command = [sys.executable, '-B', str(P / 'v23-conv-numeric-build1/runner.py'),
               '--execute-gpu', '--sum-y', '4', '--sum-evidence', str(D / 'native-pipeline.json'),
               '--checkpoint-dir', '/Users/ryanli/.cache/huggingface/hub/models--mlx-community--Qwen3.8-27B-nvfp4/snapshots/5ff8ef173ad0d7c3aae92f0be43031a6ab8067c6',
               '--output-dir', str(out / 'child-output')]
    env = {**os.environ, 'MLX_MAX_OPS_PER_BUFFER': '50', 'MLX_MAX_MB_PER_BUFFER': '50',
           'MLX_BFS_MAX_WIDTH': '20', 'MLX_METAL_FAST_SYNCH': '0',
           'PYTHONDONTWRITEBYTECODE': '1', 'HF_HUB_OFFLINE': '1'}
    record = {'owner_pid': os.getpid(), 'command': command, 'source_sha256': {
        str(f): hashlib.sha256(f.read_bytes()).hexdigest()
        for f in (P / 'v23-conv-numeric-build1').iterdir() if f.suffix in ('.py', '.metal', '.json')},
        'performance_qualified': False, 'fullmodel_qualified': False}
    active = None
    write(out / 'owner-receipt.json', record)
    try:
        with (out / 'stdout.log').open('x') as stdout, (out / 'stderr.log').open('x') as stderr:
            active = subprocess.Popen(command, env=env, stdout=stdout, stderr=stderr, start_new_session=True)
            record['child_pid'] = active.pid
            write(out / 'owner-receipt.json', record)
            try:
                record['returncode'] = active.wait(timeout=180)
            except subprocess.TimeoutExpired:
                record['timeout_seconds'] = 180
                raise
    finally:
        if active is not None:
            if active.poll() is None:
                active.terminate()
                try:
                    active.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    active.kill()
                    active.wait(timeout=10)
            record['returncode'] = active.returncode
            record['waited_reaped'] = True
            absent = subprocess.run(['/bin/ps', '-p', str(active.pid), '-o', 'pid='],
                                    text=True, capture_output=True, timeout=10)
            record['absence'] = {'returncode': absent.returncode, 'stdout': absent.stdout, 'stderr': absent.stderr}
            write(out / 'owner-receipt.json', record)
            assert absent.returncode == 1 and absent.stdout == absent.stderr == ''
        sample('after-small-numeric-reaped')
    print(json.dumps(record))
    return record['returncode']


if __name__ == '__main__':
    raise SystemExit(main())
