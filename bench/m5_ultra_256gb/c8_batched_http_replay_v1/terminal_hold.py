"""Default OFF: unchanged qualifier in this PID, followed by bounded exit ack."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import time
import traceback

R = Path('/Users/ryanli/code/minivllm')
QUALIFIER = R/'bench/m5_ultra_256gb/c8_batched_http_replay_v1/qualify.py'


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic(path, value):
    assert not path.exists(), 'fresh control receipt required'
    temporary = path.with_suffix(path.suffix+'.tmp')
    with temporary.open('x') as stream:
        json.dump(value, stream, indent=2); stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)


def run(command_path, command_sha, control):
    assert sha(command_path) == command_sha
    command = json.loads(command_path.read_text())
    assert command[:4] == [str(R/'.venv/bin/python'),'-I','-B',str(QUALIFIER)]
    assert len(command) == 11 and command[4] == '--run' and command[5::2] == ['--plan','--plan-sha256','--out'], 'fixed original qualifier argv'
    assert control.is_absolute() and control.is_dir() and not control.is_symlink()
    done_path = control/'numerical-done.json'; ack_path = control/'numerical-exit-ack.json'
    assert not done_path.exists() and not ack_path.exists(), 'fresh terminal protocol'
    parent = os.getppid(); previous = sys.argv; code = 1; errors = []
    try:
        sys.argv = command[3:]
        runpy.run_path(str(QUALIFIER), run_name='__main__')
        code = 0
    except SystemExit as exc:
        code = 0 if exc.code is None else exc.code if type(exc.code) is int else 1
        if type(exc.code) not in (int, type(None)): errors.append(str(exc.code))
    except BaseException:
        errors.append(traceback.format_exc())
    finally:
        sys.argv = previous
    complete_path = Path(command[-1])/'complete.json'; complete = None
    try:
        raw = json.loads(complete_path.read_text())
        complete = dict(path=str(complete_path), sha256=sha(complete_path), status=raw['status'])
    except BaseException:
        errors.append(traceback.format_exc())
    deadline = time.monotonic()+60
    done = dict(schema='E053-terminal-hold-v1', pid=os.getpid(), ppid=parent, command=command,
                qualifier_returncode=code, qualifier_complete=complete, wrapper_errors=errors,
                ended_unix=time.time(), ack_deadline_monotonic=deadline)
    atomic(done_path, done); digest = sha(done_path)
    try:
        while not ack_path.exists():
            assert os.getppid() == parent, 'numerical owner changed before ack'
            assert time.monotonic() < deadline, '60-second terminal ack deadline'
            time.sleep(.2)
        ack = json.loads(ack_path.read_text())
        assert ack['pid'] == os.getpid() and ack['owner_pid'] == parent and ack['done_sha256'] == digest
        assert ack['final_physical_and_metadata_complete'] is True and ack['command'] == command
        return code if not errors else 1
    except BaseException:
        atomic(control/'terminal-hold-error.json', dict(error=traceback.format_exc(), original_qualifier_returncode=code))
        return 1


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--run', action='store_true')
    p.add_argument('--command-json', type=Path); p.add_argument('--command-sha256'); p.add_argument('--control', type=Path)
    a=p.parse_args()
    if a.run: raise SystemExit(run(a.command_json, a.command_sha256, a.control))
    print('DESCRIPTION ONLY: explicit run of unchanged qualifier, then at most 60 seconds for final-sample exit ack.')
