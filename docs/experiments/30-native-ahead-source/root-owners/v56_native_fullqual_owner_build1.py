"""Root-only serial correctness owner, with resource monitoring and owned cleanup."""
from pathlib import Path
import argparse
import copy
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
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', choices=('v56native',), required=True)
    ap.add_argument('--attempt', type=int, default=1)
    args = ap.parse_args()
    assert args.attempt > 0
    assert sys.executable == '/Users/ryanli/code/minivllm/.venv/bin/python'
    assert not any(k.startswith(('MLX_', 'METAL_', 'DYLD_')) for k in os.environ)
    script = P / 'v56-native-async-runner-build1/fullqual.py'
    assert script.is_file()
    out = D / ('fullmodel-%s-attempt%d' % (args.variant, args.attempt))
    out.mkdir()
    history, active = [], None
    def sample(label, pid=None):
        raw = resources.snapshot(label, pid)
        write(out / ('resource-%03d-raw.json' % len(history)), raw)
        derived = copy.deepcopy(raw)
        if pid is not None and not any(x['pid'] == pid for x in derived['known_inference_processes']):
            derived['known_inference_processes'].append({'pid': pid, 'owned': True, 'engine': 'explicit-owned-correctness-child'})
        derived['known_inference_process_count'] = len(derived['known_inference_processes'])
        derived['only_owned_known_inference_processes'] = all(x['owned'] for x in derived['known_inference_processes'])
        history.append(derived)
        resources.deltas(history)
        write(out / 'resource-derived.json', history)
        resources.guard(history)
        assert raw['pressure_level_value'] == 1, 'memory pressure warning'
        assert raw['power_metadata']['source'] == 'AC Power', 'power domain changed'
        assert raw['processinfo_thermal_metadata']['code'] in (0, 1), 'thermal guard'
        assert derived['only_owned_known_inference_processes'], 'foreign inference process'
    sample('before-fullmodel-correctness')
    command = [sys.executable, '-B', str(script), '--out', str(out / 'child-output')]
    env = {**os.environ, 'MLX_MAX_OPS_PER_BUFFER': '50', 'MLX_MAX_MB_PER_BUFFER': '50',
           'MLX_BFS_MAX_WIDTH': '20', 'MLX_METAL_FAST_SYNCH': '0',
           'DYLD_INSERT_LIBRARIES': str(P / 'v55-native-async-build2/libv55replay.dylib'),
           'PYTHONDONTWRITEBYTECODE': '1', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'}
    record = {'owner_pid': os.getpid(), 'command': command,
              'script_sha256': hashlib.sha256(script.read_bytes()).hexdigest(),
              'native_library_sha256': hashlib.sha256((P / 'v55-native-async-build2/libv55replay.dylib').read_bytes()).hexdigest(),
              'native_source_sha256': hashlib.sha256((P / 'v55-native-async-build2/replay.mm').read_bytes()).hexdigest(),
              'performance_qualified': False, 'stage': 'before-spawn'}
    write(out / 'owner-receipt.json', record)
    try:
        with (out / 'stdout.log').open('x') as stdout, (out / 'stderr.log').open('x') as stderr:
            active = subprocess.Popen(command, env=env, stdout=stdout, stderr=stderr, start_new_session=True)
            record.update(child_pid=active.pid, stage='monitoring')
            write(out / 'owner-receipt.json', record)
            print(json.dumps({'owner_pid': os.getpid(), 'child_pid': active.pid, 'output': str(out)}), flush=True)
            start = time.monotonic()
            while active.poll() is None:
                if time.monotonic() - start > 600:
                    raise TimeoutError('fullmodel correctness limit600s')
                sample('during-fullmodel-correctness', active.pid)
                time.sleep(2)
            record['returncode'] = active.wait(timeout=10)
    except BaseException as error:
        record['owner_error'] = {'type': type(error).__name__, 'message': str(error)}
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
            record['stage'] = 'reaped'
            write(out / 'owner-receipt.json', record)
            assert absent.returncode == 1 and absent.stdout == absent.stderr == ''
        sample('after-fullmodel-correctness-reaped')
    print(json.dumps(record))
    return record['returncode']


if __name__ == '__main__':
    raise SystemExit(main())
