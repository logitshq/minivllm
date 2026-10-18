"""One owned native HTTP capture phase, before independently owned replay.

No service control, model import, numerical/performance qualification or outer
resource policy lives here. ``prepare`` reads only small pinned source/metadata;
the outer owner verifies the Python/runtime/weights and authorizes execution.
``capture`` owns exactly one server child and always attempts bounded reap.
Its mandatory checkpoint(label, pid) callback supplies the external service,
resource and journal checks; an async callback must not block the event loop.
The outer owner must refuse replay unless capture and cleanup both succeeded.
"""
import asyncio
import base64
import copy
import hashlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import subprocess
import time

HERE = Path(__file__).resolve().parent
FROZEN = {
    'native_server.py': '9bcb37eff8bdecd79800afdedc82deacf9f7a0a9a7a1e0d62dc8ba49e70fb68f',
    'client.py': 'ef64a114f4f7d90ec5044d6bdcf058ca05f98dea445d6639db706a4fc75217dc',
    'native_cleanup.py': '6c292a9d3cb28c047d1dc92ac0fec4abe19fa8df84c172ef61d2f6ce41157ca6',
}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    target = Path(path)
    temporary = target.with_suffix(target.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(target)


def load(name):
    path = HERE / (name + '.py')
    spec = importlib.util.spec_from_file_location('_capture_private_' + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_sources(sources):
    expected = {str(HERE / name) for name in (*FROZEN, 'http_capture.py', 'http_qualification.py')}
    need(set(sources) == expected, 'exact capture source set required')
    for path, digest in sources.items():
        need(sha(path) == digest, 'capture source changed: ' + path)
    for name, digest in FROZEN.items():
        need(sources[str(HERE / name)] == digest, 'frozen capture dependency changed: ' + name)


def prepare(base_plan_path, base_plan_sha, concurrency, *, python, environment,
            transport, deadlines):
    """Build the fixed launch/request contract; full runtime/weight pins are external.

transport: {port, tcp_nodelay, io_timeout_seconds}; deadlines: positive seconds
for ready, request, group, close, term, kill, overall, resource_interval. Overall
covers launch/readiness/capture; close and TERM/KILL are separate cleanup bounds.
"""
    sources = {str(HERE / name): sha(HERE / name)
               for name in (*FROZEN, 'http_capture.py', 'http_qualification.py')}
    verify_sources(sources)
    q = load('http_qualification')
    base_plan_path = Path(base_plan_path).resolve()
    need(base_plan_sha == q.BASE_PLAN_SHA and sha(base_plan_path) == base_plan_sha,
         'unchanged fixed numerical plan required')
    numeric = q.read(base_plan_path)
    fixtures = q.read(q.pinned(numeric['fixtures']))['fixtures']
    source, model = [q.read(q.pinned(numeric[k])) for k in ('source_manifest', 'model_manifest')]
    need(str(Path(python).absolute()) == numeric['execution']['python'], 'pinned Python path required')
    need(set(transport) == {'port', 'tcp_nodelay', 'io_timeout_seconds'}
         and type(transport['port']) is int and 1024 <= transport['port'] <= 65535
         and transport['port'] != 8001 and type(transport['tcp_nodelay']) is bool,
         'explicit isolated port and TCP option required')
    names = {'ready', 'request', 'group', 'close', 'term', 'kill', 'overall', 'resource_interval'}
    need(set(deadlines) == names and all(type(v) in (int, float) and math.isfinite(v) and v > 0
                                      for v in deadlines.values()), 'all absolute budgets must be positive/finite')
    need(type(transport['io_timeout_seconds']) in (int, float)
         and math.isfinite(transport['io_timeout_seconds']) and transport['io_timeout_seconds'] > 0,
         'finite server IO timeout required')
    allowed = {'HOME', 'PATH', 'TMPDIR', 'LANG', 'LC_ALL', 'PYTHONDONTWRITEBYTECODE'} | set(numeric['execution']['environment'])
    need(isinstance(environment, dict) and set(environment) <= allowed
         and all(isinstance(k, str) and isinstance(v, str) for k, v in environment.items())
         and {k: environment.get(k) for k in numeric['execution']['environment']} == numeric['execution']['environment'],
         'explicit minimal environment with unchanged registered MLX flags required')
    cfg = numeric['configuration']
    command = [str(python), '-I', '-B', str(HERE / 'native_server.py'),
        '--model', numeric['model'], '--package-root', numeric['package_root'],
        '--source-manifest', numeric['source_manifest']['path'],
        '--source-manifest-sha256', numeric['source_manifest']['sha256'],
        '--port', str(transport['port']), '--max-len', str(cfg['max_len']),
        '--max-seqs', str(concurrency), '--max-batched-tokens', str(cfg['max_batched_tokens']),
        '--cache-limit-gib', str(cfg['cache_limit_gib'])]
    for key, enabled in numeric['candidate'].items():
        command += ['--' + key.replace('_', '-'), 'on' if enabled else 'off']
    command += ['--tcp-nodelay', 'on' if transport['tcp_nodelay'] else 'off',
                '--io-timeout-seconds', str(float(transport['io_timeout_seconds']))]
    return {'schema': 'E023-http-capture-spec-v1', 'sources': sources,
        'base_plan': {'path': str(base_plan_path), 'sha256': base_plan_sha},
        'python': str(python), 'environment': dict(environment), 'transport': dict(transport),
        'deadlines': dict(deadlines), 'concurrency': concurrency, 'command': command,
        'jobs': q.jobs_for(fixtures, concurrency),
        'identity': q.expected_identity(numeric, fixtures, source, model, concurrency, transport),
        'external_pin_scope': 'Outer owner must verify runtime/Python/all source and weight pins before and after.'}


def validate(spec):
    need(spec.get('schema') == 'E023-http-capture-spec-v1', 'capture spec required')
    verify_sources(spec['sources'])
    expected = prepare(spec['base_plan']['path'], spec['base_plan']['sha256'], spec['concurrency'],
        python=spec['python'], environment=spec['environment'], transport=spec['transport'],
        deadlines=spec['deadlines'])
    need(json.dumps(expected, sort_keys=True) == json.dumps(spec, sort_keys=True), 'capture spec was changed')


class System:
    """Only used by an explicitly authorized outer owner, never by CPU tests."""
    def spawn(self, command, environment, stdout, stderr):
        return subprocess.Popen(command, env=environment, stdout=stdout, stderr=stderr)

    async def read(self, command):
        def run():
            p = subprocess.run(command, capture_output=True, text=True, timeout=5)
            return dict(command=command, returncode=p.returncode, stdout=p.stdout, stderr=p.stderr)
        return await asyncio.to_thread(run)

    async def port(self, port, pid=None):
        command = ['/usr/sbin/lsof', '-nP', '-a']
        if pid is not None:
            command += ['-p', str(pid)]
        command += [f'-iTCP:{port}', '-sTCP:LISTEN', '-Fpn']
        return await self.read(command)

    async def absent(self, pid):
        return await self.read(['/bin/ps', '-p', str(pid), '-o', 'pid='])

    async def identity(self, pid):
        return await self.read(['/bin/ps', '-p', str(pid), '-o', 'pid=,ppid=,lstart=,comm='])

    async def readiness(self, port):
        import httpx
        started = time.monotonic()
        try:
            async with httpx.AsyncClient(timeout=2, trust_env=False, follow_redirects=False) as client:
                response = await client.get(f'http://127.0.0.1:{port}/v1/models')
                return dict(start_monotonic=started, end_monotonic=time.monotonic(),
                    status=response.status_code, headers=dict(response.headers),
                    body_base64=base64.b64encode(response.content).decode())
        except (httpx.TransportError, OSError) as exc:
            return dict(start_monotonic=started, end_monotonic=time.monotonic(),
                        error={'type': type(exc).__name__, 'message': str(exc)})

    async def wait(self, child, seconds):
        return await asyncio.to_thread(child.wait, timeout=seconds)

    async def group(self, spec, output):
        cleanup = load('native_cleanup')
        d = spec['deadlines']
        return await cleanup.run_group(f"http://127.0.0.1:{spec['transport']['port']}",
            spec['jobs'], spec['concurrency'], phase='qualification', qualification=None,
            request_timeout_s=d['request'], group_timeout_s=d['group'],
            cleanup_timeout_s=d['close'], output_dir=output)


def no_process(raw):
    return raw['returncode'] == 1 and not raw['stdout'].strip() and not raw['stderr'].strip()


def owned_listener(raw, pid, port):
    return (raw['returncode'] == 0 and not raw['stderr'].strip()
            and raw['stdout'].splitlines() == [f'p{pid}', f'n127.0.0.1:{port}'])


def ready_identity(raw, identity):
    if raw.get('status') != 200:
        return False
    data = json.loads(base64.b64decode(raw['body_base64'], validate=True))
    need(set(data) == {'data', 'benchmark_identity'} and data['data'] == [{'id': 'native-target-only'}],
         'unexpected readiness document')
    need(json.dumps(data['benchmark_identity'], sort_keys=True) == json.dumps(identity, sort_keys=True),
         'actual startup identity differs from capture contract')
    return True


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
    command = spec['command'] + ['--qualification-trace', str(out / 'trace.jsonl')]
    result = dict(schema='E023-http-capture-result-v1', status='INCOMPLETE',
        numerical_qualified=False, performance_qualified=False, cleanup_complete=False, server_reaped=False,
        started_at=time.time(), errors=[], scope='Observed HTTP schedule capture only; replay and outer resource/restore audit remain separate.')
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
