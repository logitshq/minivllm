"""Explicitly authorized temporary SIGSTOP lease, with automatic SIGCONT."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def write(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def identity(pid):
    result = subprocess.run(['/bin/ps', '-p', str(pid), '-o', 'lstart=,comm='], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def state(pid):
    result = subprocess.run(['/bin/ps', '-p', str(pid), '-o', 'state='], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def resume(lease):
    same = identity(lease['service_pid']) == lease['service_identity']
    before = state(lease['service_pid'])
    if same and before and 'T' in before:
        os.kill(lease['service_pid'], signal.SIGCONT)
    return {'identity_match': same, 'state_before': before, 'state_after': state(lease['service_pid']), 'time': time.time()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--watchdog', type=Path)
    ap.add_argument('--pid', type=int)
    ap.add_argument('--jobs', type=Path)
    ap.add_argument('--out', type=Path)
    a = ap.parse_args()
    if a.watchdog:
        while True:
            lease = json.loads(a.watchdog.read_text())
            if lease.get('closed'):
                return
            if identity(lease['owner_pid']) != lease['owner_identity'] or time.time() > lease['deadline']:
                write(a.watchdog.with_name('watchdog-resume.json'), resume(lease))
                return
            time.sleep(3)
    assert a.pid and a.jobs and a.out
    service_identity = identity(a.pid)
    assert service_identity and service_identity.endswith('omlx-server'), service_identity
    assert 'T' not in state(a.pid), 'Do not take ownership of an already-stopped service'
    a.out.mkdir()
    path = a.out / 'lease.json'
    lease = {'service_pid': a.pid, 'service_identity': service_identity,
             'owner_pid': os.getpid(), 'owner_identity': identity(os.getpid()),
             'deadline': time.time() + 45 * 60, 'closed': False,
             'authorization': 'User explicitly authorized temporary pause and restoration in this conversation',
             'jobs': json.loads(a.jobs.read_text())}
    write(path, lease)
    watchdog = subprocess.Popen([sys.executable, '-B', str(Path(__file__).resolve()), '--watchdog', str(path.resolve())],
                                start_new_session=True, stdin=subprocess.DEVNULL,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    child = None
    def stop(signum, frame):
        raise KeyboardInterrupt(str(signum))
    signal.signal(signal.SIGTERM, stop)
    try:
        os.kill(a.pid, signal.SIGSTOP)
        time.sleep(2)
        assert 'T' in state(a.pid) and identity(a.pid) == service_identity
        lease['paused_at'] = time.time()
        write(path, lease)
        print('PAUSED', a.pid, flush=True)
        for i, command in enumerate(lease['jobs']):
            print('JOB', i, command, flush=True)
            child = subprocess.Popen(command)
            lease['current_job'] = i
            lease['child_pid'] = child.pid
            write(path, lease)
            code = child.wait()
            write(a.out / f'job-{i}.json', {'command': command, 'pid': child.pid, 'returncode': code, 'waited': True})
            assert code == 0, (i, code)
            child = None
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        write(a.out / 'resume.json', resume(lease))
        lease['closed'] = True
        write(path, lease)
        watchdog.wait(timeout=10)
        print('RESUMED', a.pid, state(a.pid), flush=True)


if __name__ == '__main__':
    main()
