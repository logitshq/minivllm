"""Prepared GDN-only HTTP capture; not numerical/HTTP qualification.

prepare/validate/capture retain the frozen native D512 C1x8/C8x16 client and
direct-child lifecycle. Only this new specification accepts the real GDN-only server,
explicit qualification/GDN flags and a separate GDN submission sidecar. No
model/runtime import, process or socket operation occurs during preparation.
The external owner verifies runtime/weights, resource admission and recovery.
Closed capture evidence still requires independent numerical replay and route
audit; startup identity is not evidence of an actual GDN Python submission.
"""
import asyncio
import copy
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
DONOR = HERE.parent / 'pareto_realistic/http_capture_v2.py'
DONOR_SHA = '536fe41d6c8e2bdffe83ba6f43fba87dd6265deb4c84f6204bc9c7220f8280e3'
SERVER = HERE / 'http_server.py'
SERVER_SHA = 'e4408d9df0c26a2f7fc5db93312902e4797b72a8958f67f1912b4b9bd628e3d4'
DISPATCH_SCHEMA = 'GDN-prework-http-events-v1'


def need(ok, message):
    if not ok: raise ValueError(message)


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_fixed(path, digest, name):
    need(sha(path) == digest, 'frozen capture support changed: ' + str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


donor = load_fixed(DONOR, DONOR_SHA, '_gdn_frozen_http_capture')
write, no_process, owned_listener, ready_identity = (
    donor.write, donor.no_process, donor.owned_listener, donor.ready_identity)
System = donor.System  # Original client, readiness, wait and strict OS probes.


def prepare(base_plan_path, base_plan_sha, concurrency, *, python, environment,
            transport, deadlines):
    """Build a new source-bound spec from unchanged frozen D512 inputs."""
    server = load_fixed(SERVER, SERVER_SHA, '_gdn_capture_real_server')
    original = donor.prepare(base_plan_path, base_plan_sha, concurrency, python=python,
        environment=environment, transport=transport, deadlines=deadlines)
    need(type(concurrency) is int and concurrency in (1, 8), 'fixed C1/C8 capture only')
    cfg = original['identity']['configuration']
    need(cfg['max_len'] == 2048 and cfg['max_seqs'] == concurrency
         and cfg['compile_decode'] == cfg['fuse_decode'] == cfg['gates_compile'] == 'on'
         and cfg['norm_reuse'] == 'off', 'unchanged D512 current stack/context required')
    pins = server.source_pins()
    need(pins[str(SERVER)] == SERVER_SHA, 'real GDN-only entry identity')
    sources = dict(original['sources'])
    for path, digest in pins.items():
        need(path not in sources or sources[path] == digest, 'conflicting server source pin')
        sources[path] = digest
    sources[str(Path(__file__).resolve())] = sha(__file__)
    command = list(original['command'])
    need(command[3] == str(donor.HERE / 'native_server.py'), 'original native argv entry changed')
    command[3] = str(SERVER)
    command += ['--phase', 'qualification', '--gdn-prework', 'on']
    args = SimpleNamespace(phase='qualification', gdn_prework='on', max_seqs=concurrency)
    identity = dict(original['identity'])
    identity.update(server.identity_fields(args, pins, installed=True, dispatch_observer=True))
    return {**original, 'schema': 'GDN-prework-http-capture-spec-v1',
        'sources': sources, 'command': command, 'identity': identity,
        'dispatch_events_schema': DISPATCH_SCHEMA,
        'qualification_scope': 'Instrumented capture only; independent dispatch/numerical replay and outer resource/restoration audit are required.'}


def validate(spec):
    need(spec.get('schema') == 'GDN-prework-http-capture-spec-v1', 'GDN capture spec required')
    expected = prepare(spec['base_plan']['path'], spec['base_plan']['sha256'], spec['concurrency'],
        python=spec['python'], environment=spec['environment'], transport=spec['transport'],
        deadlines=spec['deadlines'])
    need(json.dumps(spec, sort_keys=True, allow_nan=False) ==
         json.dumps(expected, sort_keys=True, allow_nan=False), 'GDN capture spec was changed')
    for path, digest in spec['sources'].items():
        need(sha(path) == digest, 'GDN capture source changed: ' + path)


def dispatch_startup(out, spec, pid):
    """Only bind the retained sidecar's start; later replay must check every call."""
    with (out / 'dispatch-events.jsonl').open('rb') as stream:
        first = stream.readline()
    expected = dict(event='dispatch-startup', schema=DISPATCH_SCHEMA, pid=pid,
                    phase=f"HTTP-C{spec['concurrency']}", identity=spec['identity'])
    need(type(pid) is int and pid > 1 and first.endswith(b'\n')
         and json.dumps(json.loads(first), sort_keys=True, allow_nan=False) ==
             json.dumps(expected, sort_keys=True, allow_nan=False),
         'closed dispatch sidecar lacks actual child/source/startup identity')

async def capture(spec, out, *, checkpoint, backend=None):
    """Return retained result; all failures remain failures after cleanup.

checkpoint is called before spawn, after spawn/ready, periodically, after the
client and after reap. It may write outer durable intent/ownership receipts.
No retry, replacement server, warmup inference or service restoration occurs.
    A cleanup timeout is unresolved FAIL: the outer owner must reconcile/reap
    child.json's exact PID before restoration; it must never start replay.
"""
    need(callable(checkpoint), 'external resource/service checkpoint is mandatory')
    validate(spec)
    spec = copy.deepcopy(spec); out = Path(out).resolve(); out.mkdir(parents=True, exist_ok=False)
    system = backend or System(); d = spec['deadlines']; port = spec['transport']['port']
    command = spec['command'] + ['--qualification-trace', str(out / 'trace.jsonl'),
                                 '--dispatch-events', str(out / 'dispatch-events.jsonl')]
    result = dict(schema='GDN-prework-http-capture-result-v1', status='INCOMPLETE',
        numerical_qualified=False, performance_qualified=False, cleanup_complete=False, server_reaped=False,
        started_at=time.time(), errors=[], scope='Observed GDN-only HTTP capture only; complete dispatch/numerical replay and outer resource/restore audit remain separate.')
    write(out / 'spec.json', spec); write(out / 'launch.json', dict(command=command, environment=spec['environment']))
    write(out / 'complete.json', result)
    child = task = None
    stdout = stderr = None
    events = (out / 'events.jsonl').open('x')
    def event(label, **fields):
        events.write(json.dumps(dict(label=label, time=time.time(), monotonic=time.monotonic(), **fields)) + '\n')
        events.flush()
    def error(exc, phase):
        result['errors'].append(dict(phase=phase, type=type(exc).__name__, message=str(exc)))
    async def check(label, pid):
        value = checkpoint(label, pid)
        if inspect.isawaitable(value):
            await value
        event('checkpoint', phase=label, pid=pid)
    async def run_client():
        try:
            group = await system.group(spec, out / 'client')
            write(out / 'group.json', group)
            return group
        except BaseException as exc:
            if hasattr(exc, 'result'):
                write(out / 'group.json', exc.result)
            raise
    phase = 'preflight'
    try:
        async with asyncio.timeout(d['overall']):
            await check('pre-spawn', None)
            port_before = await system.port(port); write(out / 'port-before.json', port_before)
            need(no_process(port_before), 'capture port already occupied or unknown')
            stdout = (out / 'server.stdout').open('xb'); stderr = (out / 'server.stderr').open('xb')
            phase = 'spawn'
            write(out / 'spawn-intent.json', dict(command=command, time=time.time()))
            child = system.spawn(command, spec['environment'], stdout, stderr)
            write(out / 'child.json', dict(pid=child.pid, time=time.time()))
            await check('spawned', child.pid)
            identity = await system.identity(child.pid); write(out / 'child-identity.json', identity)
            need(identity['returncode'] == 0 and identity['stdout'].split()[:2] == [str(child.pid), str(os.getpid())],
                 'direct server process identity unavailable')
            phase = 'readiness'
            async with asyncio.timeout(d['ready']):
                sampled = time.monotonic()
                while True:
                    need(child.poll() is None, 'server exited before readiness')
                    raw = await system.readiness(port); event('readiness', raw=raw)
                    if ready_identity(raw, spec['identity']):
                        listener = await system.port(port, child.pid); write(out / 'listener.json', listener)
                        need(owned_listener(listener, child.pid, port), 'ready listener is not the owned localhost server')
                        write(out / 'ready.json', raw)
                        break
                    if time.monotonic() - sampled >= d['resource_interval']:
                        await check('starting', child.pid); sampled = time.monotonic()
                    await asyncio.sleep(0.1)
            await check('ready', child.pid)
            phase = 'client'
            task = asyncio.create_task(run_client(), name='capture-http-group')
            while not task.done():
                await asyncio.wait([task], timeout=d['resource_interval'])
                need(child.poll() is None, 'server exited during capture')
                await check('capturing', child.pid)
            group = task.result()
            write(out / 'group.json', group)
            need(group['schema'] == 'E023-http-group-v1' and group['phase'] == 'qualification'
                 and group['qualification'] is None and group['concurrency'] == spec['concurrency']
                 and len(group['requests']) == len(spec['jobs'])
                 and all({k: row.get(k) for k in job} == job for row, job in zip(group['requests'], spec['jobs']))
                 and group['group_error'] is None and group['cleanup']['complete'] is True
                 and group['cleanup']['adapter'] == 'native-close-cancellation-v1'
                 and group['summary']['transport_complete'] is True
                 and group['summary']['formal_point_eligible'] is False,
                 'client capture or cleanup failed')
            await check('after-client', child.pid)
            raw = await system.readiness(port); write(out / 'ready-after-client.json', raw)
            need(ready_identity(raw, spec['identity']), 'server failed after HTTP completion')
            result['capture_complete'] = True
    except BaseException as exc:
        error(exc, phase)
    finally:
        async def finish_owned():
            # This task is not cancelled when its calling capture is cancelled.
            try:
                if task is not None:
                    if not task.done():
                        task.cancel()
                        await asyncio.wait([task], timeout=d['close'] + 1)
                    if not task.done():
                        task.cancel(); error(RuntimeError('HTTP client task unresolved; outer owner must stop'), 'client-cleanup')
                    elif not task.cancelled() and task.exception() is None:
                        write(out / 'group.json', task.result())
            except BaseException as exc:
                error(exc, 'client-cleanup')
            finally:
                # Receipt/write/client-close errors cannot skip the owned server.
                if child is not None:
                    try:
                        if child.poll() is None:
                            event('server-term', pid=child.pid)
                            try:
                                child.terminate()
                                await system.wait(child, d['term'])
                            except subprocess.TimeoutExpired:
                                event('server-term-timeout', pid=child.pid)
                            except BaseException as exc:
                                error(exc, 'server-term')
                            if child.poll() is None:
                                event('server-kill', pid=child.pid); child.kill()
                        await system.wait(child, d['kill'])
                        absence = await system.absent(child.pid); write(out / 'child-absence.json', absence)
                        port_after = await system.port(port); write(out / 'port-after.json', port_after)
                        need(type(child.returncode) is int and no_process(absence) and no_process(port_after),
                             'owned child/port not strictly reaped and absent')
                        write(out / 'release.json', dict(pid=child.pid, returncode=child.returncode,
                            waited_reaped=True, strict_absence=True, port_absent=True, time=time.time()))
                        result['server_reaped'] = True
                        client_closed = task is None
                        if task is not None and task.done() and (out / 'group.json').exists():
                            client_closed = json.loads((out / 'group.json').read_text())['cleanup']['complete'] is True
                        result['cleanup_complete'] = client_closed
                    except BaseException as exc:
                        error(exc, 'server-cleanup')
        cleanup_task = asyncio.create_task(finish_owned(), name='capture-owned-cleanup')
        # close + TERM + KILL/wait + two bounded 5s probes + scheduling margin.
        cleanup_end = asyncio.get_running_loop().time() + d['close'] + d['term'] + d['kill'] + 12
        while not cleanup_task.done() and asyncio.get_running_loop().time() < cleanup_end:
            try:
                await asyncio.wait([cleanup_task], timeout=max(0, cleanup_end - asyncio.get_running_loop().time()))
            except asyncio.CancelledError as exc:
                error(exc, 'cancel-during-cleanup')
        if not cleanup_task.done():
            cleanup_task.cancel()
            result['cleanup_complete'] = False
            error(TimeoutError('owned cleanup absolute deadline; outer owner must reconcile/reap'), 'cleanup-deadline')
        elif cleanup_task.cancelled():
            result['cleanup_complete'] = False
            error(RuntimeError('owned cleanup task cancelled unexpectedly'), 'cleanup-task')
        elif cleanup_task.exception() is not None:
            result['cleanup_complete'] = False
            error(cleanup_task.exception(), 'cleanup-task')
        result['cleanup_task_resolved'] = cleanup_task.done()
        result['client_task_resolved'] = task is None or task.done()
        try:
            await check('after-reap' if result['server_reaped'] else 'cleanup-incomplete',
                        None if result['server_reaped'] or child is None else child.pid)
            validate(spec)
            if result.get('capture_complete'):
                with (out / 'trace.jsonl').open('rb') as trace:
                    first = trace.readline()
                need(first.endswith(b'\n') and json.loads(first) == {'event': 'startup', 'identity': spec['identity']},
                     'closed trace lacks matching startup identity')
                dispatch_startup(out, spec, child.pid)
                result['dispatch_events_retained'] = True
            result['small_pins_rechecked'] = True
        except BaseException as exc:
            error(exc, 'postflight')
        if stdout is not None: stdout.close()
        if stderr is not None: stderr.close()
        events.close()
        result['ended_at'] = time.time()
        result['artifacts'] = {p.name: {'sha256': sha(p), 'bytes': p.stat().st_size}
            for p in sorted(out.iterdir()) if p.is_file() and p.name != 'complete.json'}
        result['status'] = 'PASS' if result.get('capture_complete') and result['cleanup_complete'] and not result['errors'] else 'FAIL'
        write(out / 'complete.json', result)
    return result
