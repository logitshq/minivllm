"""E023 owner for one exact cold-removed service; stdlib only until child launch.

The wrapper plan pins this owner, controller_v2 and the unchanged original E023
numerical plan. It contains no runtime journal ID: a short journal lock binds a
STOPPED snapshot and durable attempt before preparation. A separately fsynced
spawn intent precedes Popen. Only failures before that intent can certify that no
child started. This owner never removes/restores a service or sends HTTP requests.
SIGKILL/power loss before terminal, or an unknown spawn after intent, requires
manual reconciliation; absence of a receipt is never a no-child certificate.

The existing 1830s work limit and 15s TERM + 15s KILL/reap + 5s absence + 30s
postflight bounds remain. Resource pressure must remain 1, with the original
power/thermal/swap guards. Recognized process inventory is not a global GPU lock.
Numerical PASS still requires independent ledger audit; it is not HTTP/performance
qualification. Help/import performs no probes, model imports or service action.
"""
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace


HERE = Path(__file__).resolve().parent
SUPPORT_SHA = '1efaf440e991fed64dedf15455be2d37223e709fec41bb07e8c0295a2975c1c2'
NUMERICAL_SHA = '21ba52e43ce54c2b463ab46bfd212c53d06d6303530c77939e3e6b801df547fb'
QUALIFIER_SHA = '26964fcbed9a2e5dcefaee4888244987897f3138d4a2b7add59bf63a7461b7bf'
CONTRACT = 'cold-restart-persisted-visible-v1'
PROPOSAL_SHA = 'e96a4a4d5be6f6a69d597387bbc566eadedae81065bfa879445352d986515cb8'
ORIGINAL = {'pid': 74384, 'ppid': 1, 'start': 'Fri Oct 16 08:01:58 2026', 'comm': 'omlx-server'}
JOB = 'gui/501/local.omlx.server'


def need(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def sha(path):
    return sha_bytes(Path(path).read_bytes())


def absolute(value):
    p = Path(value)
    need(p.is_absolute() and '..' not in p.parts and p == p.resolve(), 'canonical absolute path required')
    return p


def durable(path, value, *, exclusive=False, raw=False):
    """Intent uses O_EXCL; all receipts fsync data and parent directory."""
    path = Path(path)
    content = value if raw else (json.dumps(value, indent=2, allow_nan=False) + '\n').encode()
    tmp = None
    if exclusive:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    else:
        fd, tmp = tempfile.mkstemp(prefix='.owner-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        if tmp is not None:
            os.replace(tmp, path)
        dfd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    finally:
        if tmp is not None and os.path.exists(tmp):
            os.unlink(tmp)


@contextmanager
def journal_lock(directory):
    # Same cooperative lock as controller_v2. Never create a new journal/lock.
    with (directory / '.lock').open('r+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def validate_plan(args):
    raw = args.plan.read_bytes()
    need(sha_bytes(raw) == args.plan_sha256, 'owner plan SHA mismatch')
    plan = json.loads(raw)
    need(set(plan) == {'schema', 'execution', 'qualification', 'cold_restart', 'deadline_seconds'}, 'wrapper plan fields')
    need(plan['schema'] == 'E023-service-absent-owner-plan-v1', 'wrapper schema')
    seconds = plan['deadline_seconds']
    need(type(seconds) in (int, float) and math.isfinite(seconds) and 1800 <= seconds <= 1830, 'work deadline must be 1800..1830s')
    need(set(plan['execution']) == {'owner'}, 'execution fields')
    owner = plan['execution']['owner']
    need(set(owner) == {'path', 'sha256'} and absolute(owner['path']) == Path(__file__).resolve()
         and sha(owner['path']) == owner['sha256'], 'owner source pin')
    q = plan['qualification']
    need(set(q) == {'path', 'sha256'} and q['sha256'] == NUMERICAL_SHA, 'must reuse unchanged original E023 plan')
    numeric_raw = absolute(q['path']).read_bytes()
    need(sha_bytes(numeric_raw) == NUMERICAL_SHA, 'original numerical plan SHA')
    numeric = json.loads(numeric_raw)
    need(numeric['execution']['qualifier'] == {'path': str(HERE / 'qualify_native.py'), 'sha256': QUALIFIER_SHA}, 'fixed numerical qualifier')
    cold = plan['cold_restart']
    need(set(cold) == {'journal_path', 'controller', 'approved_contract', 'proposal_plan_sha256'}, 'cold contract fields')
    absolute(cold['journal_path'])
    controller = cold['controller']
    need(set(controller) == {'path', 'sha256'} and absolute(controller['path']) == HERE.parent / 'omlx_cold_restart/controller_v2.py'
         and sha(controller['path']) == controller['sha256'], 'fixed v2 controller source')
    need(cold['approved_contract'] == CONTRACT and cold['proposal_plan_sha256'] == PROPOSAL_SHA, 'approved contract/proposal binding')
    return plan, raw, numeric


def stopped(plan, args, binding=None):
    raw = (Path(plan['cold_restart']['journal_path']) / 'journal.json').read_bytes()
    data = json.loads(raw)
    expected = {'schema': 'omlx-cold-restart-journal-v2', 'state': 'STOPPED',
                'plan_sha256': PROPOSAL_SHA, 'controller_sha256': plan['cold_restart']['controller']['sha256'],
                'approved_contract': CONTRACT, 'original': ORIGINAL, 'owner_directory': str(args.out),
                'owner_plan': {'path': str(args.plan), 'sha256': args.plan_sha256},
                'owner_source': plan['execution']['owner']}
    need(all(data.get(k) == v for k, v in expected.items()), 'journal is not bound STOPPED; recovery or contract change')
    need(isinstance(data.get('id'), str) and data['id'] and type(data.get('removed_at')) in (int, float)
         and math.isfinite(data['removed_at']) and 0 < data['removed_at'] <= time.time(), 'journal runtime identity')
    stable = {k: data[k] for k in (*expected, 'id', 'removed_at')}
    need(binding is None or binding == stable, 'journal stable identity changed')
    # Events/resource samples may grow during recovery waiting; state may not.
    return raw, stable


def module_from(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def load_prepare():
    path = HERE / 'qualify_owner.py'
    need(sha(path) == SUPPORT_SHA, 'frozen numerical prepare changed')
    return module_from(path, '_absent_e023_prepare').prepare


def load_resources(numeric):
    # Import only these three verified stdlib modules, never ambient same-named modules.
    names = ('resource_base', 'resources', 'resources_v2')
    previous = {name: sys.modules.get(name) for name in names}
    try:
        for name in names:
            path = HERE.parent / (name + '.py')
            need(sha(path) == numeric['execution']['pins'][str(path)], 'resource source pin ' + name)
            sys.modules[name] = module_from(path, name)
        return sys.modules['resources_v2']
    finally:
        for name, old in previous.items():
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old


class System:
    """Real backend is never used by the CPU tests. Service actions are absent."""
    def read(self, command, timeout=5):
        r = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
        return {'command': command, 'returncode': r.returncode, 'stdout': r.stdout}

    def identity(self, pid):
        r = self.read(['/bin/ps', '-p', str(pid), '-o', 'pid=,ppid=,lstart=,comm='])
        need(r['returncode'] == 0, 'owner identity unknown')
        m = re.fullmatch(r'\s*(\d+)\s+(\d+)\s+([A-Za-z]{3} [A-Za-z]{3}\s+\d+ \d\d:\d\d:\d\d \d{4})\s+(.+?)\s*', r['stdout'])
        need(m is not None and int(m[1]) == pid, 'owner identity format')
        return {'pid': int(m[1]), 'ppid': int(m[2]), 'start': ' '.join(m[3].split()), 'comm': m[4]}

    def spawn(self, command, log, env):
        return subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)

    def start_deadline(self, seconds, stop):
        need(signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0), 'outer alarm already active')
        self.handlers = {s: signal.signal(s, stop) for s in (signal.SIGALRM, signal.SIGTERM, signal.SIGINT)}
        signal.setitimer(signal.ITIMER_REAL, seconds)

    def cleanup_mode(self):
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)

    def postflight_deadline(self):
        signal.setitimer(signal.ITIMER_REAL, 30)

    def restore_deadline(self):
        signal.setitimer(signal.ITIMER_REAL, 0)
        for sig, handler in self.handlers.items():
            signal.signal(sig, handler)


def absence(system, pid=74384):
    """Retain return codes and output hashes; launchctl output may contain secrets."""
    commands = [(['/bin/ps', '-p', str(pid), '-o', 'pid='], 1),
                (['/bin/launchctl', 'print', JOB], 113),
                (['/usr/sbin/lsof', '-nP', '-a', '-iTCP:8001', '-sTCP:LISTEN', '-Fpn'], 1)]
    rows = []
    for command, code in commands:
        raw = system.read(command)
        text = raw['stdout']
        rows.append({'command': command, 'returncode': raw['returncode'], 'stdout_sha256': sha_bytes(text.encode()),
                     'stdout_bytes': len(text.encode()), 'absent': raw['returncode'] == code and not text.strip()})
    return {'status': 'PASS' if all(r['absent'] for r in rows) else 'FAIL', 'checks': rows}


def run(args, *, system=None):
    system = System() if system is None else system
    plan, plan_raw, numeric = validate_plan(args)
    directory = Path(plan['cold_restart']['journal_path'])
    started = time.monotonic()
    child, resources, intent, attempt, binding = None, None, False, None, None
    samples, errors, absence_probes = [], [], []
    phase = 'attempt'
    deadline_active = False

    def failure(exc, where):
        row = {'phase': where, 'type': type(exc).__name__, 'error': str(exc), 'time': time.time()}
        errors.append(row)
        durable(args.out / 'errors.json', errors)
        if len(errors) == 1:
            durable(args.out / 'first-error.json', row, exclusive=True)

    def stop(*unused):
        raise TimeoutError('owner absolute deadline or cancellation')

    def check():
        if time.monotonic() - started >= plan['deadline_seconds']:
            stop()
        stopped(plan, args, binding)

    def probe(label, pid=None, *, cleanup=False):
        if not cleanup:
            check()
        else:
            stopped(plan, args, binding)
        proof = absence(system)
        n = len(samples)
        proof = {'label': label, 'time': time.time(), 'resource_index_if_sampled': n, **proof}
        absence_probes.append(proof)
        durable(args.out / f'absence-{len(absence_probes)-1:03d}.json', proof, exclusive=True)
        need(proof['status'] == 'PASS', 'removed service PID/job/8001 reappeared or absence unknown')
        raw = resources.snapshot(label, pid, suspended_pids=())
        samples.append(raw)
        durable(args.out / f'resource-{n:03d}.json', raw)
        resources.guard(samples)
        need(raw['authorized_suspended_pids'] == []
             and all(p['owned'] and p['pid'] == pid and not p['authorized_suspended'] for p in raw['native_inference_processes']),
             'no suspended exemption or foreign native process permitted')
        need(raw['pressure_level_value'] == 1 and raw['power_metadata']['source'] == 'AC Power'
             and raw['processinfo_thermal_metadata']['code'] in (0, 1), 'pressure/power/thermal guard failed')
        stopped(plan, args, binding)

    # Lock spans admission+durable attempt. Controller cannot choose no-experiment
    # between its own directory check and this producer's creation.
    with journal_lock(directory):
        snapshot, binding = stopped(plan, args)
        identity = system.identity(os.getpid())
        need(identity['pid'] == os.getpid(), 'wrong owner process identity')
        args.out.mkdir(parents=False, exist_ok=False)
        durable(args.out / 'stopped-journal.json', snapshot, exclusive=True, raw=True)
        attempt = {'schema': 'cold-service-owner-attempt-v1', 'owner_pid': os.getpid(), 'owner_identity': identity,
                   'owner_source': plan['execution']['owner'], 'owner_plan_sha256': args.plan_sha256,
                   'journal_id': binding['id'], 'journal_path': str(directory),
                   'controller_sha256': binding['controller_sha256'], 'removed_at': binding['removed_at'],
                   'created_at': time.time(), 'phase': 'PRESPAWN', 'stopped_journal_sha256': sha_bytes(snapshot)}
        durable(args.out / 'attempt.json', attempt, exclusive=True)
    try:
        remaining = plan['deadline_seconds'] - (time.monotonic() - started)
        need(remaining > 0, 'work deadline elapsed before preparation')
        system.start_deadline(remaining, stop)
        deadline_active = True
        durable(args.out / 'owner-plan.json', plan_raw, exclusive=True, raw=True)
        phase = 'prepare'
        source_dir = args.out / 'owner-source'
        source_dir.mkdir()
        for item in (plan['execution']['owner'], plan['cold_restart']['controller']):
            content = Path(item['path']).read_bytes()
            need(sha_bytes(content) == item['sha256'], 'wrapper source changed before preparation')
            durable(source_dir / Path(item['path']).name, content, exclusive=True, raw=True)
        resources = load_resources(numeric)
        probe('before-prepare')
        q = plan['qualification']
        command, env = load_prepare()(SimpleNamespace(out=args.out, plan=Path(q['path']), plan_sha256=q['sha256'],
                                                     deadline_seconds=plan['deadline_seconds']), check)
        durable(args.out / 'launch.json', {'command': command, 'environment': {k: v for k, v in env.items() if k.startswith('MLX_')},
                                         'numerical_plan_sha256': NUMERICAL_SHA}, exclusive=True)
        phase = 'preflight'
        probe('after-prepare-before-spawn')
        validate_plan(args)  # no source/plan replacement between prepare and spawn
        with (args.out / 'child.log').open('x') as log:
            with journal_lock(directory):
                check()
                final_absence = absence(system)
                durable(args.out / 'pre-spawn-absence.json', final_absence, exclusive=True)
                need(final_absence['status'] == 'PASS', 'service absence changed immediately before intent')
                phase = 'spawn-intent'
                # Set conservatively BEFORE write: failed fsync/O_EXCL is not proof
                # of no intent. A disk-visible partial intent also forbids no-child.
                intent = True
                durable(args.out / 'spawn-intent.json', {'schema': 'cold-service-owner-spawn-intent-v1',
                        'owner_pid': os.getpid(), 'journal_id': binding['id'], 'owner_plan_sha256': args.plan_sha256,
                        'attempt_sha256': sha(args.out / 'attempt.json'), 'time': time.time()}, exclusive=True)
                phase = 'spawn'
                child = system.spawn(command, log, env)
                durable(args.out / 'child.json', {'pid': child.pid, 'owner_pid': os.getpid(), 'spawned_at': time.time()}, exclusive=True)
            phase = 'running'
            while child.poll() is None:
                check()
                probe('child-running', child.pid)
                check()
                try:
                    child.wait(timeout=min(5, plan['deadline_seconds'] - (time.monotonic() - started)))
                except subprocess.TimeoutExpired:
                    pass
            child.wait(timeout=0)
        need(child.returncode == 0, f'qualifier exit {child.returncode}')
        complete = json.loads((args.out / 'qualification/complete.json').read_text())
        need(complete['status'] == 'PASS', 'qualifier did not record PASS')
    except BaseException as exc:
        failure(exc, phase)
    finally:
        if deadline_active:
            system.cleanup_mode()
        if child is not None:
            try:
                if child.poll() is None:
                    try:
                        child.terminate()
                        child.wait(timeout=15)
                    except BaseException as exc:
                        failure(exc, 'terminate/wait')
                    if child.poll() is None:
                        child.kill()
                child.wait(timeout=15)
                raw = system.read(['/bin/ps', '-p', str(child.pid), '-o', 'pid='])
                durable(args.out / 'child-absence.json', raw, exclusive=True)
                need(raw['returncode'] == 1 and not raw['stdout'].strip(), 'strict child PID absence not established')
                need(type(child.returncode) is int, 'child not waited/reaped')
                durable(args.out / 'release.json', {'pid': child.pid, 'returncode': child.returncode,
                        'waited_reaped': True, 'strict_absence': True, 'time': time.time()}, exclusive=True)
            except BaseException as exc:
                failure(exc, 'cleanup')
        try:
            if deadline_active:
                system.postflight_deadline()
            if resources is not None:
                probe('postflight', cleanup=True)
                durable(args.out / 'resources.json', resources.deltas(samples))
            validate_plan(args)
        except BaseException as exc:
            failure(exc, 'postflight')
        finally:
            if deadline_active:
                system.restore_deadline()
        if not intent and all(not (args.out / p).exists() for p in ('spawn-intent.json', 'child.json', 'release.json')):
            need(child is None and errors, 'NO_CHILD_STARTED requires failure strictly before intent')
            durable(args.out / 'terminal.json', {'schema': 'cold-service-owner-terminal-v1', 'status': 'NO_CHILD_STARTED',
                    **{k: attempt[k] for k in ('owner_pid', 'owner_identity', 'owner_source', 'owner_plan_sha256', 'journal_id', 'removed_at')},
                    'terminal_at': time.time(), 'phase': errors[0]['phase'], 'error': errors[0],
                    'attempt_sha256': sha(args.out / 'attempt.json'), 'spawn_intent_written': False, 'child_started': False}, exclusive=True)
        durable(args.out / 'complete.json', {'schema': 'E023-service-absent-owner-result-v1',
                'status': 'FAIL' if errors else 'PASS', 'owner_pid': os.getpid(), 'errors': errors,
                'samples': len(samples), 'absence_probes': len(absence_probes), 'elapsed_wall_s': time.monotonic() - started,
                'spawn_intent_written': intent, 'child_started': True if child is not None else None if intent else False,
                'numerical_plan_sha256': NUMERICAL_SHA, 'owner_plan_sha256': args.plan_sha256,
                'external_service_restoration': 'separate pinned cold-restart controller; not performed here',
                'scope': 'unchanged 37-request numerical workload; independent stream audit still required; no HTTP/performance claim'})
    return int(bool(errors))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    args.plan, args.out = absolute(args.plan), absolute(args.out)
    need(args.out.parent.is_dir() and not args.out.exists(), 'new output in existing parent required')
    return run(args)


if __name__ == '__main__':
    raise SystemExit(main())
