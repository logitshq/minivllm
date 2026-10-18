"""Two independent observation lanes; no model/service action or CLI on import.

Callbacks retain the frozen physical_guard.measure/observe and owner.checkpoint
algorithms. They receive (index, commands, absolute_deadline); use commands.run
for measure and commands.read as the checkpoint System.read method. Callbacks
must obey that deadline and save their own raw evidence. This helper only closes
scheduling and read-command ownership. The caller still owns model-child reap,
service recovery, source pins and independent resource qualification.

Intervals are maximum end-to-next-start gaps, NOT maximum sample duration or
start-to-start periods. Schedule at half that gap to leave real scheduler margin.
Python threads are not hard real-time: exceeding either bound fails closed.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import threading
import time


def need(ok, message):
    if not ok:
        raise ValueError(message)


def error_record(error):
    return dict(type=type(error).__name__, message=str(error))


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


class MonitorError(RuntimeError):
    def __init__(self, report):
        self.report = report
        super().__init__('Live monitoring incomplete; see monitor.json and retained lane/command receipts')


class Commands:
    """Only returned read-command Popen handles; never the model process.

    communicate(timeout) is the same stdlib bounded pipe reader used by run().
    On failure, TERM then KILL are each allowed 0.5 seconds for communicate/wait.
    A constructor failure has unknown launch outcome, even if an empty query
    could later be obtained. It cannot become a successful monitor receipt.
    """
    def __init__(self, out, cancel, *, popen=subprocess.Popen):
        self.out = Path(out); self.cancel = cancel; self.popen = popen
        self.records = []; self.lock = threading.Lock(); self.deadline = None
        self.failed = 0; self.unsettled = 0

    def snapshot(self):
        with self.lock:
            return json.loads(json.dumps(self.records, allow_nan=False))

    def summary(self):
        """Constant-size live bookkeeping; raw history is copied only at exit."""
        with self.lock:
            return dict(count=len(self.records), failed=self.failed, unresolved=self.unsettled)

    def run(self, argv, *, capture_output=True, text=True, timeout):
        need(capture_output is True and text is True and type(argv) is list
             and argv and all(type(v) is str for v in argv) and Path(argv[0]).is_absolute(),
             'explicit read command/text capture required')
        need(type(timeout) in (int, float) and math.isfinite(timeout) and 0 < timeout <= 5,
             'read command timeout must be bounded by five seconds')
        if self.cancel.is_set():
            raise InterruptedError('monitor cancelled before read command')
        began = time.monotonic(); deadline = min(began + timeout, self.deadline or began + timeout)
        need(deadline > began, 'callback deadline before read command')
        index = len(self.records); stem = self.out / ('command-%04d' % index)
        row = dict(command=list(argv), started_monotonic=began, started_at=time.time(),
                   deadline_monotonic=deadline, pid=None, waited_reaped=False,
                   unknown_launch=False, actions=[], status='INCOMPLETE')
        save(stem.with_suffix('.intent.json'), row)
        with self.lock:
            self.records.append(dict(row))
            self.unsettled += 1
        child = None; output = errors = None
        try:
            row['unknown_launch'] = True
            child = self.popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            row.update(pid=child.pid, unknown_launch=False)
            save(stem.with_suffix('.child.json'), dict(pid=child.pid, owner_pid=os.getpid(),
                 ownership='direct-Popen-returned', command=list(argv)))
            while True:
                if self.cancel.is_set():
                    raise InterruptedError('monitor cancelled during read command')
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError('read command deadline')
                try:
                    output, errors = child.communicate(timeout=min(.05, remaining)); break
                except subprocess.TimeoutExpired:
                    pass
            code = child.wait(timeout=max(.001, deadline - time.monotonic()))
            row.update(returncode=code, waited_reaped=True)
            need(type(code) is int and child.returncode == code, 'actual read-command wait required')
            need(time.monotonic() <= deadline, 'read command completed after deadline')
            row['status'] = 'COMPLETE'
            return subprocess.CompletedProcess(argv, code, output, errors)
        except BaseException as error:
            row['error'] = error_record(error)
            if child is not None and not row['waited_reaped']:
                for action in ('terminate', 'kill'):
                    try:
                        if child.poll() is None:
                            row['actions'].append(action); getattr(child, action)()
                        output, errors = child.communicate(timeout=.5)
                        code = child.wait(timeout=.5)
                        need(type(code) is int and child.returncode == code, 'cleanup wait return code')
                        row.update(returncode=code, waited_reaped=True); break
                    except BaseException as cleanup_error:
                        row.setdefault('cleanup_errors', []).append(error_record(cleanup_error))
            raise
        finally:
            row.update(ended_monotonic=time.monotonic(), ended_at=time.time())
            row['elapsed_seconds'] = row['ended_monotonic'] - began
            # The frozen checkpoint intentionally redacts unrelated process args.
            # Return the text to that parser, but do not duplicate secrets on disk.
            if argv == ['/bin/ps', '-axo', 'pid=,ppid=,args='] and isinstance(output, str):
                row.update(stdout_bytes=len(output.encode()), stdout_sha256=hashlib.sha256(output.encode()).hexdigest())
            else:
                row['stdout'] = output
            row['stderr'] = errors
            with self.lock:
                self.records[index] = dict(row)
                self.failed += int(row['status'] != 'COMPLETE')
                self.unsettled -= int(row['waited_reaped'])
            save(stem.with_suffix('.json'), row)

    def read(self, argv, timeout=5):
        value = self.run(argv, timeout=timeout)
        return dict(command=argv, returncode=value.returncode, stdout=value.stdout, stderr=value.stderr)


def monitor(child, *, out, physical, metadata, work_deadline, physical_interval=1.0,
            metadata_interval=5.0, probe_seconds=5.0, join_seconds=7.0,
            popen=subprocess.Popen):
    """Block until the actual child exits or monitoring fails; never signal it.

    Each callback's complete execution, including its own evidence writes, must
    fit probe_seconds. No callbacks overlap within a lane. On normal child exit,
    in-flight callbacks finish under their original deadline; ANY failure still
    rejects. On error, stop new work and cancel outstanding read commands, then
    bounded-join both workers. An unjoined worker or unknown/unreaped read-command
    is explicit unresolved evidence, requiring the outer owner to fail closed.
    Popen.poll may internally reap an exited child; it is not the owner's full
    final wait/strict-absence release proof, which must still be retained.
    """
    for value, maximum in ((physical_interval, 1), (metadata_interval, 5), (probe_seconds, 5), (join_seconds, 7)):
        need(type(value) in (int, float) and math.isfinite(value) and 0 < value <= maximum, 'finite monitor bounds')
    need(type(child.pid) is int and child.pid > 1 and callable(physical) and callable(metadata), 'owned child/callbacks')
    need(type(work_deadline) in (int, float) and math.isfinite(work_deadline)
         and work_deadline > time.monotonic(), 'absolute future work deadline')
    out = Path(out); out.mkdir(parents=False, exist_ok=False)
    started = time.monotonic(); stop = threading.Event(); cancel = threading.Event(); lock = threading.Lock()
    lanes = {}; failures = []; threads = []; code = None; primary = None

    def failure(lane, error):
        with lock:
            failures.append(dict(lane=lane, **error_record(error)))
        cancel.set(); stop.set()

    def worker(name, callback, gap):
        lane = lanes[name]; commands = lane['commands']; previous_end = started; index = 0
        try:
            while not stop.is_set():
                begin = time.monotonic(); deadline = min(begin + probe_seconds, work_deadline)
                row = dict(index=index, started_monotonic=begin, started_at=time.time(),
                           prior_end_monotonic=previous_end, gap_seconds=begin-previous_end,
                           deadline_monotonic=deadline, status='INCOMPLETE')
                with lock:
                    lane['active'] = dict(row)
                try:
                    need(0 <= row['gap_seconds'] <= gap, name + ' end-to-start gap exceeded')
                    need(deadline > begin, name + ' work deadline')
                    commands.deadline = deadline
                    first_command = commands.summary()['count']
                    row['result'] = callback(index, commands, deadline)
                    # Serialization is part of the callback protocol, not deferred
                    # until the model may already have been granted completion.
                    json.dumps(row['result'], allow_nan=False)
                    summary = commands.summary()
                    row['command_range'] = dict(start=first_command, stop=summary['count'])
                    need(summary['failed'] == 0 and summary['unresolved'] == 0,
                         name + ' callback masked a read-command failure')
                    need(time.monotonic() <= deadline, name + ' callback deadline exceeded')
                    row['status'] = 'COMPLETE'
                except BaseException as error:
                    row['error'] = error_record(error); raise
                finally:
                    row.update(ended_monotonic=time.monotonic(), ended_at=time.time())
                    row['elapsed_seconds'] = row['ended_monotonic'] - begin
                    save(out/name/('sample-%04d.json' % index), row)
                    with lock:
                        lane['samples'].append(row); lane['active'] = None
                previous_end = row['ended_monotonic']; index += 1
                # Half-gap target leaves margin; the next iteration checks the
                # actual end-to-start gap rather than trusting sleep accuracy.
                stop.wait(gap * .5)
        except BaseException as error:
            failure(name, error)

    for name, callback, gap in (('physical', physical, physical_interval), ('metadata', metadata, metadata_interval)):
        (out/name).mkdir()
        lanes[name] = dict(samples=[], active=None, commands=Commands(out/name, cancel, popen=popen))
        threads.append(threading.Thread(target=worker, args=(name, callback, gap), name='glm-monitor-'+name, daemon=True))
    try:
        need(child.poll() is None, 'no live owned child at monitor start')
        for thread in threads:
            thread.start()
        while True:
            if cancel.is_set():
                break
            now = time.monotonic()
            need(now <= work_deadline, 'owned work deadline')
            with lock:
                active = [(name, lane['active']) for name, lane in lanes.items() if lane['active'] is not None]
            for name, row in active:
                need(now <= row['deadline_monotonic'], name + ' callback did not settle by deadline')
            code = child.poll()
            if code is not None:
                need(type(code) is int and code == 0, 'owned child exited nonzero/unknown')
                break
            stop.wait(.01)
    except BaseException as error:
        primary = error; failure('owner-monitor', error)
    finally:
        stop.set(); join_deadline = time.monotonic() + join_seconds
        for thread in threads:
            if thread.ident is not None:
                thread.join(timeout=max(0, join_deadline-time.monotonic()))
        unjoined = [thread.name for thread in threads if thread.is_alive()]
        if unjoined:
            cancel.set()
        with lock:
            snapshot = {name: dict(samples=list(lane['samples']), active=lane['active']) for name, lane in lanes.items()}
            errors = list(failures)
        commands = {name: lane['commands'].snapshot() for name, lane in lanes.items()}
        unresolved = [dict(lane=name, index=i, **row) for name, rows in commands.items()
                      for i, row in enumerate(rows) if row.get('unknown_launch') or not row.get('waited_reaped')]
        complete = (not errors and not unjoined and not unresolved and code == 0
                    and all(lane['samples'] and lane['active'] is None for lane in snapshot.values()))
        report = dict(schema='GLM-dual-cadence-monitor-v1', status='COMPLETE' if complete else 'INCOMPLETE',
            pid=child.pid, owner_pid=os.getpid(), child_returncode=code, started_monotonic=started,
            ended_monotonic=time.monotonic(), work_deadline=work_deadline,
            policy=dict(physical_max_gap_seconds=physical_interval, metadata_max_gap_seconds=metadata_interval,
                        target_gap_fraction=.5, callback_seconds=probe_seconds, join_seconds=join_seconds,
                        read_cleanup_communicate_seconds=.5, read_cleanup_wait_seconds=.5),
            lanes=snapshot, commands=commands, errors=errors, unjoined_workers=unjoined,
            unresolved_read_commands=unresolved, read_commands_settled=not(unjoined or unresolved),
            model_child_release_proven=False, physical_resource_safety_qualified=False, performance_qualified=False,
            scope='Independent observation cadences only. Caller must validate raw checks and reap the model child; no exit-race exemption.')
        save(out/'monitor.json', report)
    if report['status'] != 'COMPLETE':
        raise MonitorError(report) from primary
    return report
