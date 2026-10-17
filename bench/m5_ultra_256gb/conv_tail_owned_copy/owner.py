"""E026 numerical-only tiny matrix owner. Help/import are stdlib-only.

Plan E026-tiny-owner-v1: arms=[off,on-lazy,on-materialize], arm_seconds=30,
total_seconds=180, execution={python,screen:{path,sha256},pins:{absolute:sha256},
environment:{four fixed MLX variables}}. No model manifests or service control.
Work (including pin hashes) stops at 160s, reserving 20s for bounded cleanup and
postflight. Each direct child has a 30s hard alarm, TERM3s/KILL3s/ps2s cleanup.
Only the outer suspended_service lease owns stopping/restoring PID 74384.
"""
import argparse
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
ARMS = ['off', 'on-lazy', 'on-materialize']
ENV = {'MLX_MAX_OPS_PER_BUFFER': '50', 'MLX_MAX_MB_PER_BUFFER': '50',
       'MLX_BFS_MAX_WIDTH': '20', 'MLX_METAL_FAST_SYNCH': '0'}
REASON = ('Pressure 2 is permitted only for this isolated no-weight <64MiB-boundary '
          'bit/backing diagnostic, never performance or full-model qualification. '
          'The child checks active/peak at boundaries; this is not an OS hard memory cap.')


def write(out, name, value):
    (out / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def sha(path, check=lambda: None):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while data := stream.read(8 << 20):
            check(); result.update(data)
    check()
    return result.hexdigest()


def need(condition, message):
    if not condition:
        raise RuntimeError(message)


def prepare(args, check):
    need(args.plan.is_absolute() and sha(args.plan, check) == args.plan_sha256, 'plan SHA mismatch')
    plan = json.loads(args.plan.read_text())
    need(set(plan) == {'schema', 'arms', 'arm_seconds', 'total_seconds', 'execution'}
         and plan['schema'] == 'E026-tiny-owner-v1' and plan['arms'] == ARMS
         and plan['arm_seconds'] == 30 and plan['total_seconds'] == 180, 'fixed tiny owner plan required')
    write(args.out, 'plan.json', plan)
    ex = plan['execution']
    need(set(ex) == {'python', 'screen', 'pins', 'environment'} and ex['environment'] == ENV, 'fixed execution/environment required')
    need(set(ex['screen']) == {'path', 'sha256'} and Path(ex['screen']['path']) == HERE / 'screen.py', 'adjacent screen required')
    python = Path(ex['python'])
    need(python.is_absolute() and os.access(python, os.X_OK), 'absolute executable Python required')
    required = {Path(__file__).resolve(), HERE / 'candidate.py', HERE / 'screen.py', HERE / 'source-pins.json',
                HERE / 'test_cpu.py', HERE / 'test_screen_cpu.py', HERE / 'test_owner_cpu.py', python.resolve()}
    required |= {BASE / name for name in ('resources_v2.py', 'resources.py', 'resource_base.py', 'suspended_service.py')}
    pins = ex['pins']
    need(required <= {Path(p) for p in pins}, 'pins omit Python or experiment/resource/service source')
    need(pins[str(HERE / 'screen.py')] == ex['screen']['sha256'], 'screen pin conflict')
    for fragment in ('/mlx/core.', '/mlx/lib/libmlx.dylib', '/mlx/lib/mlx.metallib', '/numpy/__init__.py', '/numpy/_core/_multiarray_umath.'):
        need(any(fragment in p for p in pins), 'missing runtime pin: ' + fragment)
    verified = {}
    for name, expected in pins.items():
        path = Path(name)
        need(path.is_absolute() and '..' not in path.parts and path.suffix != '.safetensors', 'absolute non-weight pins required')
        need(isinstance(expected, str) and re.fullmatch('[0-9a-f]{64}', expected), 'invalid SHA')
        need(sha(path, check) == expected, 'pin mismatch: ' + name)
        copied = path.suffix in ('.py', '.json', '.md', '.h', '.cpp', '.txt')
        if copied:
            dest = args.out / 'frozen-source' / name.lstrip('/')
            dest.parent.mkdir(parents=True, exist_ok=True)
            with path.open('rb') as source, dest.open('wb') as target:
                while data := source.read(8 << 20):
                    check(); target.write(data)
            need(sha(dest, check) == expected, 'source changed while freezing')
        verified[name] = {'sha256': expected, 'bytes': path.stat().st_size, 'source_copied': copied}
    manifest = json.loads((HERE / 'source-pins.json').read_text())
    for item in manifest['sources'].values():
        path = (HERE / item['copy']).resolve()
        need(path.is_relative_to(HERE) and pins.get(str(path)) == item['sha256'], 'source-evidence copy missing from execution pins')
    write(args.out, 'verified-pins.json', verified)
    write(args.out, 'plan.json', plan)
    spec = importlib.util.spec_from_file_location('e026_pinned_screen', HERE / 'screen.py')
    screen = importlib.util.module_from_spec(spec); spec.loader.exec_module(screen)
    env = {k: v for k, v in os.environ.items() if not k.startswith('MLX_')}
    return plan, screen, {**env, **ENV}


def guard(samples, owned_pid=None):
    last, first = samples[-1], samples[0]
    pressure = last.get('pressure_level_value')
    need(first.get('pressure_level_value') in (1, 2) and pressure in (1, 2)
         and pressure <= min(s['pressure_level_value'] for s in samples), 'pressure worsened or unknown')
    for key in ('vm_stat', 'swap', 'pressure_level'):
        need(last.get(key, {}).get('returncode') == 0, 'resource command failed: ' + key)
    text = last['vm_stat']['text']
    page = re.search(r'page size of (\d+) bytes', text)
    free = re.search(r'^Pages free:\s*(\d+)\.', text, re.M)
    need(page and free and int(page[1]) * int(free[1]) >= 50 << 30, 'real free pages below 50GiB or unknown')
    last['real_free_bytes'] = int(page[1]) * int(free[1])
    swapouts, base_swapouts = last.get('vm_counters', {}).get('Swapouts'), first.get('vm_counters', {}).get('Swapouts')
    used, base_used = last.get('swap_used_mb'), first.get('swap_used_mb')
    need(type(swapouts) is int and type(base_swapouts) is int and swapouts == base_swapouts, 'new/reset/unknown Swapouts')
    need(type(used) in (int, float) and type(base_used) in (int, float)
         and math.isfinite(used) and math.isfinite(base_used) and used >= 0 and used <= base_used + 16, 'swap-used growth above 16MiB or unknown')
    need(last.get('power_metadata', {}).get('source') == 'AC Power'
         and last.get('processinfo_thermal_metadata', {}).get('code') in (0, 1), 'power/thermal guard failed')
    need(last.get('native_process_inventory_ok') and last.get('process_inventory_ok'), 'process inventory unavailable')
    native = last.get('native_inference_processes', [])
    suspended = [p for p in native if p['pid'] == 74384]
    need(len(suspended) == 1 and 'T' in suspended[0]['state'], 'lease PID 74384 not observed stopped')
    for p in native + last.get('known_inference_processes', []):
        need(p['pid'] in (74384, owned_pid), 'foreign inference PID')
        if p['pid'] == owned_pid:
            need(p.get('ppid') == os.getpid(), 'child is not directly owned')


def validate_child(screen, out, arm, pid, pins):
    value = json.loads((out / 'complete.json').read_text())
    need(value.get('status') == 'PASS' and value.get('arm') == arm and value.get('pid') == pid
         and value.get('ppid') == os.getpid() and value.get('device_run') is True
         and value.get('source_pins_rechecked') is True and value.get('contract') == screen.contract(arm), 'child incomplete/identity/contract mismatch')
    reported = value.get('source_pins', {})
    need(str(HERE / 'screen.py') in reported and all(pins.get(path) == digest for path, digest in reported.items()), 'child source pins differ from owner plan')
    resolved = {str(Path(path).resolve()) for path in pins}
    need(all(str(Path(value.get('runtime', {}).get(key, '')).resolve()) in resolved for key in ('mlx_file', 'numpy_file')), 'child loaded unpinned runtime path')
    expected = [f'T{n}-{kind}' for n in screen.LENGTHS for kind in screen.KINDS]
    need([c.get('case') for c in value.get('cases', [])] == expected, 'all 16 ordered cases required')
    for row in value['cases']:
        need(row.get('status') == 'PASS' and row.get('bit_equal') is True
             and row.get('parent_sha256_before') == row.get('parent_sha256_after')
             and row.get('finite') is (True if row['kind'] == 'finite' else None), 'failed numeric child case')
        need(row['expected_sha256'] == row['actual_sha256'], 'child raw bit hashes differ')
        for prefix in ('expected', 'actual'):
            path = out / row[prefix + '_file']
            need(path.resolve().is_relative_to(out.resolve()) and sha(path) == row[prefix + '_sha256'], 'child raw bit artifact missing/corrupt')
    phases = [json.loads(line) for line in (out / 'phases.jsonl').read_text().splitlines()]
    names = ['before_parent', 'after_parent_and_tail_eval', 'after_helper_graph_build', 'after_output_eval',
             'after_validation_before_drop', 'after_drop_sources', 'after_gc', 'after_sync', 'after_clear_cache',
             'after_drop_output', 'after_final_gc_sync_clear_cache']
    need([(p.get('case'), p.get('phase')) for p in phases] == [(case, name) for case in expected for name in names], 'partial/out-of-order child phases')
    for p in phases:
        need(p.get('pid') == pid and p.get('ppid') == os.getpid(), 'phase PID mismatch')
        screen.guard_snapshot(p)
    return {'arm': arm, 'pid': pid, 'status': 'PASS', 'cases': len(expected), 'phases': len(phases)}


def reap(child, out):
    notes = []
    if child.poll() is None:
        try:
            child.terminate(); child.wait(timeout=3)
        except Exception as error:
            notes.append(type(error).__name__ + ': ' + str(error))
        if child.poll() is None:
            child.kill()
    child.wait(timeout=3)
    raw = subprocess.run(['/bin/ps', '-p', str(child.pid), '-o', 'pid='], capture_output=True, text=True, timeout=2)
    row = {'pid': child.pid, 'returncode': child.returncode, 'waited_reaped': True,
           'strict_absence': raw.returncode == 1 and not raw.stdout.strip(), 'cleanup_notes': notes,
           'absence_probe': {'returncode': raw.returncode, 'stdout': raw.stdout, 'stderr': raw.stderr}}
    write(out, 'release.json', row)
    need(row['strict_absence'], 'strict PID absence not established')


def run(args, resources=None):
    args.out.mkdir(parents=True, exist_ok=False)
    started, child, child_out = time.monotonic(), None, None
    work_end, samples, errors, arms = started + 160, [], [], []
    def stop(*unused): raise TimeoutError('E026 absolute owner/arm deadline or cancellation')
    def until(deadline):
        need(deadline > time.monotonic(), 'owner work budget exhausted')
        signal.setitimer(signal.ITIMER_REAL, deadline - time.monotonic())
    def check():
        if time.monotonic() >= work_end: stop()
    def failed(phase):
        row = {'phase': phase, 'traceback': traceback.format_exc()}; errors.append(row)
        if len(errors) == 1: write(args.out, 'first-error.json', row)
        write(args.out, 'errors.json', errors)
    def probe(label, pid=None):
        row = resources.snapshot(label, pid, suspended_pids=(74384,)); samples.append(row)
        write(args.out, f'resource-{len(samples)-1:03d}.json', row)
        guard(samples, pid)
        write(args.out, f'resource-{len(samples)-1:03d}.json', row)
    need(signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0), 'outer alarm already active')
    previous = {sig: signal.signal(sig, stop) for sig in (signal.SIGALRM, signal.SIGTERM, signal.SIGINT)}
    try:
        until(work_end)
        plan, screen, env = prepare(args, check)
        if resources is None:
            sys.path.insert(0, str(BASE)); resources = importlib.import_module('resources_v2')
            need(Path(resources.__file__).resolve() == BASE / 'resources_v2.py', 'wrong resources module')
        write(args.out, 'attempt.json', {'owner_pid': os.getpid(), 'plan_sha256': args.plan_sha256,
              'environment': ENV, 'arm_seconds': 30, 'total_seconds': 180, 'work_seconds': 160,
              'pressure_policy_reason': REASON, 'service_restoration': 'outer suspended_service only'})
        probe('preflight')
        for arm in ARMS:
            check(); probe(arm + '/before')
            child_out = args.out / arm
            child_out.mkdir()
            command = [plan['execution']['python'], '-I', '-B', str(HERE / 'screen.py'), '--run', '--arm', arm, '--out', str(child_out / 'screen')]
            until(min(work_end, time.monotonic() + 30))
            with (child_out / 'child.log').open('x') as log:
                child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
                write(child_out, 'child.json', {'pid': child.pid, 'ppid': os.getpid(), 'command': command, 'arm': arm})
                while child.poll() is None:
                    probe(arm + '/running', child.pid)
                    try: child.wait(timeout=2)
                    except subprocess.TimeoutExpired: pass
                child.wait(timeout=0)
            need(child.returncode == 0, f'{arm} child exited {child.returncode}')
            arms.append(validate_child(screen, child_out / 'screen', arm, child.pid, plan['execution']['pins']))
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGTERM, signal.SIG_IGN); signal.signal(signal.SIGINT, signal.SIG_IGN)
            reap(child, child_out); child = None
            signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
            until(work_end); probe(arm + '/after')
    except BaseException:
        failed('execution')
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGTERM, signal.SIG_IGN); signal.signal(signal.SIGINT, signal.SIG_IGN)
        if child is not None:
            try: reap(child, child_out)
            except BaseException: failed('cleanup')
        try:
            signal.setitimer(signal.ITIMER_REAL, 10)
            if resources is not None: probe('postflight')
        except BaseException: failed('postflight')
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            for sig, handler in previous.items(): signal.signal(sig, handler)
        write(args.out, 'resources.json', samples)
        passed = not errors and len(arms) == 3
        write(args.out, 'complete.json', {'status': 'PASS' if passed else 'FAIL', 'owner_pid': os.getpid(),
              'arms': arms, 'errors': errors, 'samples': len(samples), 'elapsed_wall_s': time.monotonic() - started,
              'pressure_policy_reason': REASON, 'scope': 'numerical/backing only; no model or performance',
              'external_service_restoration': 'outer suspended_service sole owner'})
    return int(not passed)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--suspended-pid', type=int, required=True)
    args = parser.parse_args(argv)
    if args.suspended_pid != 74384: parser.error('only lease-authorized PID 74384 is allowed')
    args.plan, args.out = args.plan.absolute(), args.out.absolute()
    return run(args)


if __name__ == '__main__':
    raise SystemExit(main())
