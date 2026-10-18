"""Bounded E023 child owner; stdlib/help only, no model or service control.

The pinned plan adds execution={python, qualifier:{path,sha256},
pins:{absolute_source_or_runtime_file:sha256}, environment:{four MLX_* variables}}.
Deadline covers validation/preflight/load/exec. Cleanup adds at most 15s TERM,
15s KILL/reap, 5s absence probe and 30s postflight. The outer suspended_service
lease alone restores the unrelated service. Child evidence lives in qualification/.
"""
import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback


def write(out, name, value):
    (out / name).write_text(json.dumps(value, indent=2) + '\n')


def prepare(args, check):
    frozen = args.out / 'frozen-source'
    verified = {}
    def pin(path, expected):
        path = Path(path)
        if not path.is_absolute() or '..' in path.parts:
            raise ValueError('pin paths must be absolute without traversal')
        copied = path.suffix in ('.py', '.json', '.md')
        target = frozen / str(path).lstrip('/')
        if copied:
            target.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        with path.open('rb') as source, (target.open('wb') if copied else open(os.devnull, 'wb')) as dest:
            while chunk := source.read(8 << 20):
                check()
                digest.update(chunk)
                if copied:
                    dest.write(chunk)
        if digest.hexdigest() != expected:
            raise ValueError('pin mismatch: ' + str(path))
        verified[str(path)] = {'sha256': expected, 'bytes': path.stat().st_size, 'source_copied': copied}
        write(args.out, 'verified-pins.json', verified)
    def pinned_json(path, expected):
        pin(path, expected)
        return json.loads(Path(path).read_text())
    plan = pinned_json(args.plan, args.plan_sha256)
    if plan['schema'] != 'E023-native-qualification-plan-v1' or not 0 < plan['deadline_s'] <= min(1800, args.deadline_seconds):
        raise ValueError('invalid plan schema or child deadline')
    execution = plan['execution']
    qualifier = Path(execution['qualifier']['path'])
    if qualifier != Path(__file__).resolve().with_name('qualify_native.py'):
        raise ValueError('only the adjacent fixed qualifier may be launched')
    pin(qualifier, execution['qualifier']['sha256'])
    required = {Path(__file__).resolve(), qualifier} | {Path(__file__).resolve().parents[1] / n for n in ('resources_v2.py', 'resources.py', 'resource_base.py', 'suspended_service.py')}
    if not required <= {Path(p) for p in execution['pins']}:
        raise ValueError('execution pins omit owner/qualifier/resource/service source')
    for path, expected in execution['pins'].items():
        pin(path, expected)
    manifests = {key: pinned_json(plan[key]['path'], plan[key]['sha256']) for key in ('source_manifest', 'model_manifest', 'fixtures')}
    root = Path(plan['package_root']).resolve() / 'tutorial_code/miniengine'
    if set(manifests['source_manifest']['files']) != {str(p.relative_to(root)) for p in root.rglob('*.py')}:
        raise ValueError('source manifest must cover exactly the frozen core')
    for relative, expected in manifests['source_manifest']['files'].items():
        if Path(relative).is_absolute() or '..' in Path(relative).parts:
            raise ValueError('invalid source relative path')
        pin(root / relative, expected)
    env = execution['environment']
    if len(env) != 4 or any(not k.startswith('MLX_') or not isinstance(v, str) for k, v in env.items()):
        raise ValueError('explicit four MLX environment strings required')
    python = Path(execution['python'])
    if not python.is_absolute() or not os.access(python, os.X_OK):
        raise ValueError('explicit executable Python path required')
    write(args.out, 'plan.json', plan)
    return [str(python), '-I', '-B', str(qualifier), '--plan', str(args.plan), '--plan-sha256', args.plan_sha256,
            '--out', str(args.out / 'qualification')], {**{k: v for k, v in os.environ.items() if not k.startswith('MLX_')}, **env}


def run(args, resources=None):
    args.out.mkdir(parents=True, exist_ok=False)
    started, child, samples, errors = time.monotonic(), None, [], []
    def failure(phase):
        row = {'phase': phase, 'traceback': traceback.format_exc()}
        errors.append(row)
        if len(errors) == 1:
            write(args.out, 'first-error.json', row)
        write(args.out, 'errors.json', errors)
    def stop(*unused):
        raise TimeoutError('owner absolute deadline or cancellation')
    def check():
        if time.monotonic() - started >= args.deadline_seconds:
            stop()
    def probe(label, pid=None):
        raw = resources.snapshot(label, pid, suspended_pids=(args.suspended_pid,))
        samples.append(raw)
        write(args.out, f'resource-{len(samples)-1:03d}.json', raw)
        resources.guard(samples)
        if (raw['pressure_level_value'] != 1 or raw['power_metadata']['source'] != 'AC Power'
                or raw['processinfo_thermal_metadata']['code'] not in (0, 1)):
            raise RuntimeError('pressure/power/thermal guard failed')
        suspended = [p for p in raw['native_inference_processes'] if p['pid'] == args.suspended_pid]
        if len(suspended) != 1 or not suspended[0]['authorized_suspended']:
            raise RuntimeError('authorized service is not observed suspended')
    if signal.getitimer(signal.ITIMER_REAL) != (0.0, 0.0):
        raise RuntimeError('outer Python alarm already active')
    handlers = {s: signal.signal(s, stop) for s in (signal.SIGALRM, signal.SIGTERM, signal.SIGINT)}
    signal.setitimer(signal.ITIMER_REAL, args.deadline_seconds)
    try:
        command, env = prepare(args, check)
        if resources is None:
            sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
            resources = importlib.import_module('resources_v2')
        write(args.out, 'attempt.json', {'owner_pid': os.getpid(), 'command': command,
              'environment': {k: v for k, v in env.items() if k.startswith('MLX_')}, 'deadline_seconds': args.deadline_seconds})
        probe('preflight')
        check()
        with (args.out / 'child.log').open('x') as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
            write(args.out, 'child.json', {'pid': child.pid, 'owner_pid': os.getpid()})
            while child.poll() is None:
                check()
                probe('child-running', child.pid)
                check()
                try:
                    child.wait(timeout=min(5, args.deadline_seconds - (time.monotonic() - started)))
                except subprocess.TimeoutExpired:
                    pass
            child.wait(timeout=0)
        if child.returncode != 0:
            raise RuntimeError(f'qualifier exit {child.returncode}')
        if json.loads((args.out / 'qualification/complete.json').read_text())['status'] != 'PASS':
            raise RuntimeError('qualifier did not record PASS')
    except BaseException:
        failure('execution')
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        if child is not None:
            try:
                if child.poll() is None:
                    try:
                        child.terminate()
                        child.wait(timeout=15)
                    except BaseException:
                        failure('terminate/wait')
                    if child.poll() is None:
                        child.kill()
                child.wait(timeout=15)
                absent = subprocess.run(['/bin/ps', '-p', str(child.pid), '-o', 'pid='], capture_output=True, text=True, timeout=5)
                if absent.returncode != 1 or absent.stdout.strip():
                    raise RuntimeError('strict child PID absence not established')
                write(args.out, 'release.json', {'pid': child.pid, 'returncode': child.returncode, 'waited_reaped': True, 'strict_absence': True})
            except BaseException:
                failure('cleanup')
        try:
            signal.setitimer(signal.ITIMER_REAL, 30)
            if resources is not None:
                probe('postflight')
                write(args.out, 'resources.json', resources.deltas(samples))
        except BaseException:
            failure('postflight')
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
        write(args.out, 'complete.json', {'status': 'FAIL' if errors else 'PASS', 'owner_pid': os.getpid(),
              'errors': errors, 'samples': len(samples), 'elapsed_wall_s': time.monotonic() - started,
              'external_service_restoration': 'outer suspended_service sole owner'})
    return int(bool(errors))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--suspended-pid', type=int, required=True)
    parser.add_argument('--deadline-seconds', type=float, required=True)
    args = parser.parse_args(argv)
    if not math.isfinite(args.deadline_seconds) or not 0 < args.deadline_seconds <= 1830 or args.suspended_pid <= 0:
        parser.error('positive PID and finite deadline <=1830 required')
    args.plan, args.out = args.plan.absolute(), args.out.absolute()
    return run(args)


if __name__ == '__main__':
    raise SystemExit(main())
