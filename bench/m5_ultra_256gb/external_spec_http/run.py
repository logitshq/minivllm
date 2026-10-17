"""Scheduled E020 controller. Directly owns the actual listening server PID.

Do not run under guarded_v2: its child is a controller, not the GPU process.
Use suspended_service.py around this controller; each model server is its direct
child and must own the HTTP listener. --help imports neither MLX nor servers.
"""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import secrets
import signal
import socket
import statistics
import subprocess
import sys
import time
import traceback

from contract import (CONFIGS, ENV, HERE, LITHOS, LITHOS_HEAD, MODEL_ID, REPO, digest,
                      official_log_evidence, read, summarize, validate_request, write)
from launch import specification


@contextmanager
def probe_deadline(seconds):
    """Bound even the older resource probe's subprocesses without GPU hooks."""
    assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)
    def expired(signum, frame):
        raise TimeoutError(f'resource sampling exceeded {seconds} seconds')
    previous = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def ps(pid):
    result = subprocess.run(['/bin/ps', '-p', str(pid), '-o', 'pid=,ppid=,stat=,lstart=,comm='],
                            capture_output=True, text=True, timeout=5)
    return {'pid': pid, 'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr}


def listeners(port):
    result = subprocess.run(['/usr/sbin/lsof', '-nP', f'-iTCP:{port}', '-sTCP:LISTEN', '-Fpc'],
                            capture_output=True, text=True, timeout=5)
    assert result.returncode in (0, 1) and not result.stderr, 'listener inventory unavailable'
    pids = sorted({int(line[1:]) for line in result.stdout.splitlines() if line.startswith('p')})
    return {'port': port, 'pids': pids, 'returncode': result.returncode,
            'stdout': result.stdout, 'stderr': result.stderr}


def verify_pins(plan, *, full):
    for filename, sha in plan['pins'].items():
        path = Path(filename)
        stat = path.stat()
        expected = plan['pin_stats'][filename]
        assert stat.st_size == expected['bytes'] and stat.st_mtime_ns == expected['mtime_ns'], 'changed pinned file: ' + filename
        if full or stat.st_size < 4 << 20:
            assert digest(path) == sha, 'SHA mismatch: ' + filename
    assert subprocess.check_output(['git', '-C', str(LITHOS), 'rev-parse', 'HEAD'], text=True).strip() == LITHOS_HEAD
    assert not subprocess.check_output(['git', '-C', str(LITHOS), 'status', '--porcelain'], text=True).strip()


def lease_check(path, controller_pid, minimum_remaining):
    lease = read(path)
    assert lease['service_pid'] == 74384 and not lease.get('closed') and lease.get('paused_at')
    assert lease.get('child_pid') == controller_pid, 'lease must directly supervise this controller'
    assert lease['owner_pid'] == os.getppid(), 'lease owner is not controller parent'
    assert lease['deadline'] - time.time() > minimum_remaining, 'insufficient remaining suspension lease'
    from suspended_service import identity, state
    assert identity(74384) == lease['service_identity'] and 'T' in (state(74384) or ''), 'authorized service is not identically stopped'


def sample_resources(out, samples, label, server=None):
    import resources_v2
    with probe_deadline(30):
        raw = resources_v2.snapshot(label, server.pid if server is not None else None, suspended_pids=(74384,))
    write(out / f'resource-{len(samples):03d}.json', raw)
    samples.append(raw)
    resources_v2.deltas(samples)
    write(out / 'resources.json', samples)
    resources_v2.guard(samples)
    assert raw['power_metadata']['source'] == 'AC Power'
    assert raw['processinfo_thermal_metadata']['code'] in (0, 1)
    if server is not None:
        assert server.poll() is None, 'actual server exited'
        owner = raw['exact_owned_pid_metadata']
        assert owner['status'] == 'known' and owner['ppid'] == os.getpid(), 'GPU server is not a direct owned child'


def close_server(child, port):
    result = {'pid': child.pid, 'parent_pid': os.getpid(), 'port': port, 'kill_escalated': False}
    if child.poll() is None:
        child.terminate()
        try:
            child.wait(timeout=8)
        except subprocess.TimeoutExpired:
            result['kill_escalated'] = True
            child.kill()
            child.wait(timeout=4)
    else:
        child.wait()
    result.update(returncode=child.returncode, waited_reaped=True, absence=ps(child.pid), listeners=listeners(port))
    with socket.socket() as sock:
        sock.settimeout(1)
        result['tcp_connect_ex_after_reap'] = sock.connect_ex(('127.0.0.1', port))
    result['valid'] = (result['absence']['returncode'] == 1 and not result['absence']['stdout']
                       and not result['absence']['stderr'] and not result['listeners']['pids']
                       and result['tcp_connect_ex_after_reap'] != 0)
    return result


def run_one(plan, config, output, lease_path):
    import httpx
    from tokenizers import Tokenizer
    from client import body_for, request
    out = output / f'pair{config["pair"]}-{config["arm"]}'
    out.mkdir()
    arm, pair = config['arm'], config['pair']
    engine = arm.split('-')[0]
    expected = plan['oracle_ids'][engine]
    tok = Tokenizer.from_file(str(Path(plan['workload']['model_dir']) / 'tokenizer.json'))
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    assert not listeners(port)['pids']
    key = 'benchmark-' + secrets.token_urlsafe(24)
    spec = specification(plan, arm, out, port, key)
    public = specification(plan, arm, out, port, '<local-ephemeral-key>')
    write(out / 'launch.json', {**config, **public, 'controller_pid': os.getpid(),
                               'process_model': 'direct Popen of actual CLI/server; no fork wrapper'})
    for filename, value in spec['files'].items():
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        write(path, value)
    for filename, target in spec['symlinks'].items():
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.symlink_to(target)
    child_env = dict(os.environ, **spec['environment'])
    headers = {} if engine == 'native' else {'Authorization': 'Bearer ' + key}
    base = f'http://127.0.0.1:{port}'
    child, release, identity = None, None, None
    samples, requests, cleanup_errors = [], [], []
    first_failure, route = None, None
    log = (out / 'server.log').open('w')
    try:
        lease_check(lease_path, os.getpid(), 1020)
        sample_resources(out, samples, 'before-launch')
        lease_check(lease_path, os.getpid(), 1020)
        child = subprocess.Popen(spec['command'], cwd=spec['cwd'], env=child_env,
            stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        write(out / 'owned-pid.json', {'pid': child.pid, 'controller_pid': os.getpid(), 'direct_child': True,
                                      'ps_at_launch': ps(child.pid)})
        start = last_probe = time.monotonic()
        with httpx.Client(headers=headers, trust_env=False, timeout=1) as client:
            while True:
                assert child.poll() is None, 'server exited before ready'
                try:
                    response = client.get(base + '/v1/models')
                    if response.status_code == 200 and response.json().get('data'):
                        models = response.json()
                        break
                except httpx.HTTPError:
                    pass
                assert time.monotonic() - start < 900, 'readiness timeout'
                if time.monotonic() - last_probe >= 10:
                    lease_check(lease_path, os.getpid(), 120)
                    sample_resources(out, samples, 'readiness-wait', child)
                    last_probe = time.monotonic()
                time.sleep(.2)
        listener = listeners(port)
        write(out / 'ready.json', {'models': models, 'listeners': listener, 'ps': ps(child.pid)})
        assert listener['pids'] == [child.pid], 'listening PID must be the direct owned server, not a grandchild'
        sample_resources(out, samples, 'ready', child)
        assert len(models['data']) == 1
        model_id = models['data'][0]['id']
        if engine == 'omlx':
            assert model_id == MODEL_ID
        elif engine == 'native':
            identity = models['benchmark_identity']
            assert identity['manifest_all_sources_match'] and not identity['step_observer']
            assert identity['cache_limit_gib'] == 2 and identity['compile_decode'] and identity['fuse_decode']
            assert identity['drafter'] is None and identity['normal_eos']
            assert identity['source_files'] == read(plan['native_source_manifest'])['files']
        path, body = body_for(arm, model_id, plan['prompt_text'], plan['user_text'])
        for phase in ('warm', 'formal'):
            for index in range(3):
                label = f'{phase}-{index}'
                lease_check(lease_path, os.getpid(), 420)
                sample_resources(out, samples, 'before-' + label, child)
                lease_check(lease_path, os.getpid(), 420)
                row = request(base, headers, path, body, tok, f'pair{pair}-{arm}-{label}')
                row.update(phase=phase, index=index, arm=arm, pair=pair)
                # Save first even if an unexpected API schema breaks validation.
                write(out / (label + '.json'), row)
                row['contract_failures'] = validate_request(row, expected, native=engine == 'native')
                row['qualified_raw_ids_source'] = engine
                row['visible_ID_scope'] = ('live native raw IDs also captured' if engine == 'native' else
                                          'canonical visible text compared with prior engine raw-ID qualification; no live raw IDs')
                write(out / (label + '.json'), row)
                requests.append(row)
                if row.get('error') or not row.get('done'):
                    # Cancellation/timeout goes straight to server reap. Do not
                    # run another external resource probe while the GPU is live.
                    raise RuntimeError('transport/stream failure retained; no retry')
                sample_resources(out, samples, 'after-' + label, child)
                assert listeners(port)['pids'] == [child.pid]
                time.sleep(plan['cooldown_seconds_after_each_request'])
        # Stock log production may finish just after the final SSE; the fixed
        # cooldown above is outside HTTP timings. Log diagnostics never rank speed.
        log.flush()
        if engine != 'native':
            route = official_log_evidence(arm, (out / 'server.log').read_text(),
                target=plan['workload']['model_dir'], draft=plan['drafters'][engine],
                packs=plan['packs'].get(arm, {}))
        else:
            route = {'valid': True, 'failures': [], 'scope': 'native loader identity, no drafter, no step observer',
                     'identity': identity}
        write(out / 'route.json', route)
    except BaseException as error:
        first_failure = {'type': type(error).__name__, 'text': str(error), 'traceback': traceback.format_exc()}
        write(out / 'first-failure.json', first_failure)
    finally:
        # A repeated cancellation must not orphan the directly owned GPU server.
        # The original signal is delivered again only after bounded cleanup.
        previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM, signal.SIGINT})
        try:
            if child is not None:
                try:
                    release = close_server(child, port)
                    write(out / 'release.json', release)
                    if not release['valid']:
                        cleanup_errors.append('server PID/port remained after reap')
                except BaseException as error:
                    cleanup_errors.append({'type': type(error).__name__, 'text': str(error), 'traceback': traceback.format_exc()})
            log.close()
            try:
                sample_resources(out, samples, 'after-unload')
            except BaseException as error:
                cleanup_errors.append({'type': type(error).__name__, 'text': str(error)})
            write(out / 'cleanup.json', {'first_failure': first_failure, 'cleanup_errors': cleanup_errors,
                                        'release_valid': release and release['valid']})
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
    formal = [r for r in requests if r['phase'] == 'formal']
    failures = [f'{r["phase"]}-{r["index"]}: {message}' for r in requests for message in r['contract_failures']]
    failures += (route or {}).get('failures', [])
    valid = (len(requests) == 6 and len(formal) == 3 and not failures and not first_failure
             and not cleanup_errors and bool(route and route['valid']) and bool(release and release['valid']))
    def median(key):
        values = [r[key] for r in formal if r.get(key) is not None]
        return statistics.median(values) if len(values) == 3 else None
    raw_medians = {'median_wall_s': median('client_total_s'), 'median_TTFT_s': median('client_ttft_text_s'),
                   'median_output_tok_s': median('output_tok_s'),
                   'median_client_estimated_interactivity_tok_s': median('client_estimated_interactivity_tok_s')}
    result = {**config, 'valid': valid, 'contract_failures': failures, 'fatal_failure': first_failure,
        'cleanup_errors': cleanup_errors, 'request_count': len(requests), 'formal_count': len(formal),
        'diagnostic_medians_including_invalid_samples': raw_medians,
        **{k: v if valid else None for k, v in raw_medians.items()},
        'request_files': [str(out / f'{r["phase"]}-{r["index"]}.json') for r in requests],
        'resource_samples': len(samples), 'release': release, 'route_valid': bool(route and route['valid']),
        'scope': 'One fresh server; 3 warm + 3 formal serial requests. No observer timing. Actual server PID reaped.'}
    write(out / 'result.json', result)
    if first_failure or cleanup_errors:
        raise RuntimeError('fatal request/resource/lifecycle failure; process result retained')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--lease', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(HERE.parent))
    def stop(signum, frame):
        raise KeyboardInterrupt(f'controller received signal {signum}')
    signal.signal(signal.SIGTERM, stop)
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    rows = []
    try:
        assert digest(args.plan) == args.plan_sha256, 'reviewed plan hash mismatch'
        plan = read(args.plan)
        assert plan['schema'] == 'E020-v1' and plan['configs'] == CONFIGS
        selected = {k: v for k, v in os.environ.items()
                    if k.startswith(('MLX_', 'METAL_', 'OMLX_', 'MONOLITH_', 'LITHOS_', 'DYLD_'))
                    or k in ('PYTHONPATH', 'PYTHONHOME', 'PYTHONDONTWRITEBYTECODE')}
        assert selected == ENV, f'unpinned launch environment: {selected}'
        lease_check(args.lease, os.getpid(), 1020)
        write(args.out / 'attempt.json', {'pid': os.getpid(), 'parent_pid': os.getppid(),
            'plan': str(args.plan.resolve()), 'plan_sha256': args.plan_sha256, 'configs': CONFIGS,
            'lease': str(args.lease.resolve()), 'no_guarded_wrapper': True})
        write(args.out / 'plan.json', plan)
        verify_pins(plan, full=True)
        write(args.out / 'pins-before.json', {'all_full_sha_match': True, 'count': len(plan['pins'])})
        for config in CONFIGS:
            verify_pins(plan, full=False)
            print('START', config, flush=True)
            rows.append(run_one(plan, config, args.out, args.lease))
            write(args.out / 'progress.json', summarize(rows))
            print(json.dumps({k: rows[-1][k] for k in ('arm', 'pair', 'valid', 'median_output_tok_s', 'median_TTFT_s')}), flush=True)
        verify_pins(plan, full=True)
        write(args.out / 'pins-after.json', {'all_full_sha_match': True, 'count': len(plan['pins'])})
        result = summarize(rows)
        result.update(status='PASS_CONTRACT' if all(r['valid'] for r in rows) else 'FAIL_CONTRACT_RESULTS_RETAINED',
                      pins_full_sha_match=True,
                      service_restore='Must be audited from outer lease resume.json after this controller exits')
        write(args.out / 'complete.json', result)
        # Contract failure is a result, not a reason to replace a process or retry.
    except BaseException as error:
        write(args.out / 'failure.json', {'type': type(error).__name__, 'text': str(error),
            'traceback': traceback.format_exc(), 'completed': summarize(rows)})
        raise


if __name__ == '__main__':
    main()
