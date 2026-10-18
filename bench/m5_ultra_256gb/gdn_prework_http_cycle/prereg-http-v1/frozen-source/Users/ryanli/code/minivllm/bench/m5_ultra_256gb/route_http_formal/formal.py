"""Source-prepared paired route-graph timing; no device action on import or CLI.

OFF uses the unchanged E023 native entry. ON uses the frozen E024 entry in
formal mode, with step and route observers disabled. Both require the completed
C1/C8 route cycle plus their own recorded-history numerical prerequisite. These
receipts do not prove every future arrival schedule or an unobserved graph hit.

System, transport, count/EOS rules and raw metrics are the frozen native formal
functions. run() is a narrow copy of that source solely to use this module's
strict spec validator and observer-OFF selection; no imported globals are
patched. The future outer owner must await run in its own process, retain the
actual server Popen, enforce resource/deadline/reap, and restore the service.
There is deliberately no device --run CLI or owner/controller in this module.
"""
import argparse
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
NATIVE = HERE.parent / 'pareto_realistic'
ROUTE = HERE.parent / 'route_decode_graph'
CYCLE = HERE.parent / 'route_http_cycle'
DONOR = NATIVE / 'native_formal.py'
SERVER = ROUTE / 'http_server_v2.py'
PRODUCER = ROUTE / 'http_qualification_v2.py'
AUDITOR = ROUTE / 'http_audit_v2.py'
CYCLE_AUDITOR = CYCLE / 'audit_cycle.py'
BASE_SHA = '21ba52e43ce54c2b463ab46bfd212c53d06d6303530c77939e3e6b801df547fb'
FIXED = {
    DONOR: 'ec9cdc924f7987ad46eac6820b6264495c95fb182c726545a6e3ce283df7e654',
    SERVER: '8ed952b4870465718a33e0af7e4ecddbf3fbd6265571e7efd3467b904c5621b7',
    PRODUCER: 'd3e2a4f76b7c384f6c6c986d0838b68227618502b0552e5cdf74f149029e39e7',
    AUDITOR: '88f32e3b3f44206b48a1f581cbfe0bde0788060cbd2a0c5030a3ec472a0139fe',
    CYCLE_AUDITOR: 'bdcd32a06133bae37786eb148318a0d38847df2f2a20c94c984104394570cca4',
    CYCLE / 'owner.py': 'c279e9de954080894ed2b8f35f169cb30f485f93b08568991cd61a61e13bd4f9',
    CYCLE / 'controller.py': 'e6bc72007043ed0376e98ea7f28ee152587294dec200953e93a5c7c736a90694',
    CYCLE / 'audit_restore.py': 'a51c4a838230fb5d1e3dbbc788dea12df5f78eb360006e657e47f19ea56d83a7',
    NATIVE / 'audit_http_owner_v2.py': '8d8aee0372497fcf591d53e4bc02b3d1e7eac7eb1d2bf639168a72802e2395bd',
}
TRIALS = ('r1-c1-off', 'r1-c1-on', 'r1-c8-on', 'r1-c8-off',
          'r2-c8-off', 'r2-c8-on', 'r2-c1-on', 'r2-c1-off',
          'r3-c1-off', 'r3-c1-on', 'r3-c8-on', 'r3-c8-off')
TERMINALS = ('plan.json', 'complete.json', 'base/plan.json', 'base/complete.json', 'recording.json')


def need(ok, message):
    if not ok: raise ValueError(message)


def same(left, right, message):
    need(json.dumps(left, sort_keys=True, allow_nan=False) ==
         json.dumps(right, sort_keys=True, allow_nan=False), message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(1 << 20): digest.update(block)
    return digest.hexdigest()


def read(path):
    def pairs(items):
        result = {}
        for key, value in items:
            need(key not in result, 'duplicate JSON field: ' + key); result[key] = value
        return result
    def bad(value): raise ValueError('nonfinite JSON: ' + value)
    return json.loads(Path(path).read_bytes(), object_pairs_hook=pairs, parse_constant=bad)


def pinned(ref):
    need(type(ref) is dict and set(ref) == {'path', 'sha256'}, 'exact path/SHA reference')
    path = Path(ref['path'])
    need(path.is_absolute() and path.resolve() == path and not path.is_symlink()
         and path.is_file() and sha(path) == ref['sha256'], 'changed artifact: ' + str(path))
    return path


def write(path, value):
    path = Path(path); temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def module(path):
    need(path in FIXED and sha(path) == FIXED[path], 'frozen support changed: ' + str(path))
    spec = importlib.util.spec_from_file_location('_route_formal_' + path.parent.name + '_' + path.stem, path)
    result = importlib.util.module_from_spec(spec); spec.loader.exec_module(result); return result


def donor(): return module(DONOR)
def load(name): return donor().load(name)
def count_contract(spec, phase): return donor().count_contract(spec, phase)
def check_group(group, spec, phase, decode_text): return donor().check_group(group, spec, phase, decode_text)


class System:
    def __new__(cls): return donor().System()


def source_pins():
    native, route, producer = donor(), module(SERVER), module(PRODUCER)
    pins = {str(p): h for p, h in FIXED.items()}
    for more in ({str(NATIVE/name): h for name,h in native.FIXED.items()}, route.source_pins(), producer.source_pins()):
        for path, digest in more.items():
            need(path not in pins or pins[path] == digest, 'conflicting original/new source pin')
            pins[path] = digest
    pins[str(Path(__file__).resolve())] = sha(__file__)
    for path, digest in pins.items(): need(sha(path) == digest, 'source changed: ' + path)
    return pins


def bind_route(binding, concurrency, base_plan):
    """Read only closed small receipts; the recorded steps ledger is not opened.

    The source-pinned independent auditor already checked that ledger and raw
    route/wire evidence. This binder verifies their terminal/source references;
    it is neither a rerun of the numerical audit nor a performance certificate.
    """
    need(type(binding) is dict and set(binding) == {'audit', 'replay_plan', 'child_directory', 'cycle_audit'},
         'explicit route audit/replay/child/closed-cycle binding')
    audit = read(pinned(binding['audit'])); cycle = read(pinned(binding['cycle_audit']))
    need(cycle.get('schema') == 'E024-route-http-cycle-independent-audit-v1'
         and cycle.get('status') == 'PASS' and cycle.get('qualified') is True
         and cycle.get('http_history_qualified') is True and cycle.get('independent_resource_verified') is True
         and cycle.get('restoration_verified') is True and cycle.get('performance_qualified') is False
         and cycle.get('audit_source_sha256') == FIXED[CYCLE_AUDITOR], 'closed independently qualified route cycle required')
    need(audit.get('schema') == 'E024-recorded-route-http-independent-audit-v2'
         and audit.get('status') == 'PASS' and audit.get('qualified') is True
         and audit.get('http_history_qualified') is True and audit.get('concurrency') == concurrency
         and audit.get('audit_source_sha256') == FIXED[AUDITOR]
         and audit.get('performance_qualified') is False
         and audit.get('owner_resources_restore_qualified') is False, 'actual route numerical qualification required')
    child = Path(binding['child_directory'])
    need(child.is_absolute() and child.resolve() == child and child.is_dir() and not child.is_symlink(), 'canonical actual replay directory')
    need(audit['child_directory'] == str(child), 'audit/actual replay directory')
    replay_path = pinned(binding['replay_plan'])
    need(base_plan['sha256'] == BASE_SHA == audit['base_plan_sha256'], 'same original numeric workload')
    need(sha(child/'plan.json') == binding['replay_plan']['sha256'] == audit['plan_sha256'], 'actual route replay plan')
    same(audit['terminal_sha256'], {name: sha(child/name) for name in TERMINALS}, 'closed numerical terminal bytes')
    producer = module(PRODUCER)
    plan, numeric, capture, _ = producer.validate(replay_path, binding['replay_plan']['sha256'])
    same(plan['qualification'], base_plan, 'same source/model/config numeric plan')
    need(plan['concurrency'] == concurrency, 'same qualified concurrency')
    for left, right in (('sources', 'sources'), ('raw_inputs', 'inputs'), ('capture', 'capture'), ('serving_prerequisite', 'serving')):
        same(audit[left], plan[right], 'actual audit binding: ' + left)
    top = read(child/'complete.json'); base = read(child/'base/complete.json')
    need(top['schema'] == 'E024-recorded-route-http-result-v1' and top['status'] == base['status'] == 'PASS'
         and top['prepare_only'] is False and top['replay_numerical_checks_passed'] is True
         and top['input_source_pins_rechecked'] is True and top['independent_audit_required'] is True
         and top['numerical_qualified'] is top['http_qualified'] is top['performance_qualified'] is top['owner_resources_restore_qualified'] is False,
         'actual replay terminal scope, not an invented producer qualification')
    count = 8 if concurrency == 1 else 16
    need(type(top['total_requests']) is int and top['total_requests'] == count == audit['ledger']['requests']
         and top['concurrency'] == concurrency and audit['http']['max_http_outstanding'] == concurrency
         and audit['http']['fully_drained'] is True, 'all actual HTTP requests/drained concurrency')
    for key in ('sources', 'inputs', 'capture', 'serving'): same(top[key], plan[key], 'producer plan binding: ' + key)
    same(top['base_result'], dict(path='base/complete.json', sha256=sha(child/'base/complete.json'), returncode=0, status='PASS'), 'base terminal binding')
    need(set(cycle['numerical_audits']) == {'1', '8'}, 'both actual concurrency audits required')
    for C, ref in cycle['numerical_audits'].items():
        item = {key: ref[key] for key in ('path', 'sha256')}; other = read(pinned(item))
        need(other['status'] == 'PASS' and other['qualified'] is other['http_history_qualified'] is True
             and other['audit_source_sha256'] == FIXED[AUDITOR] and other['concurrency'] == int(C)
             and other['ledger']['requests'] == (8 if C == '1' else 16), 'closed two-C numerical audit set')
        if C == str(concurrency): same(item, binding['audit'], 'cycle audited this exact C receipt')
    restoration = cycle['restoration']
    need(restoration['status'] == 'PASS' and restoration['qualified'] is True
         and restoration['journal_sha256'] == cycle['journal_sha256']
         and restoration['owner_plan_sha256'] == cycle['owner_plan_sha256'], 'cycle restoration binding')
    owner = Path(cycle['owner_directory']); owner_plan = read(owner/'owner-plan.json')
    need(owner.is_absolute() and owner.resolve() == owner
         and sha(owner/'owner-plan.json') == cycle['owner_plan_sha256']
         and sha(owner/'complete.json') == cycle['owner_complete_sha256'], 'actual closed owner files')
    journal = Path(owner_plan['cold_restart']['journal_path'])/'journal.json'
    need(read(journal)['state'] == 'RESTORED' and sha(journal) == cycle['journal_sha256'], 'actual immutable restored journal')
    for path in (CYCLE_AUDITOR, AUDITOR, PRODUCER, CYCLE/'controller.py', CYCLE/'audit_restore.py', CYCLE/'owner.py'):
        need(cycle['source_pins'].get(str(path)) == FIXED[path], 'cycle audited actual adapter source')
    return plan, capture, audit


def select_entry(original, arm, server, pins):
    """Pure selection. OFF does not invoke/import the route entry or inspect a model."""
    need(arm in ('off', 'on'), 'arm is exactly off/on')
    command, identity = copy.deepcopy(original['command']), copy.deepcopy(original['identity'])
    need(command[3] == str(NATIVE/'native_server.py') and not any(
        flag in command for flag in ('--qualification-trace', '--route-events', '--route-decode-graph', '--phase')),
        'original observer-OFF native entry required')
    need(identity['step_observer'] is False and identity['qualification_trace_enabled'] is False,
         'original observers must be OFF')
    if arm == 'on':
        command[3] = str(SERVER); command += ['--phase', 'formal', '--route-decode-graph', 'on']
        identity.update(server.identity_fields(SimpleNamespace(phase='formal', route_decode_graph='on'),
                         pins, installed=True, route_observer=False))
    return command, identity


def prepare(base_plan, concurrency, repeat, *, arm, baseline_qualification, route_qualification,
            environment, transport, deadlines):
    native = donor()
    original = native.prepare(base_plan, concurrency, repeat, qualification=baseline_qualification,
                              environment=environment, transport=transport, deadlines=deadlines)
    plan, capture, audit = bind_route(route_qualification, concurrency, base_plan)
    server = module(SERVER); pins = server.source_pins()
    qualified_identity = copy.deepcopy(original['identity'])
    qualified_identity.update(step_observer=True, qualification_trace_enabled=True)
    qualified_identity.update(server.identity_fields(SimpleNamespace(phase='qualification', route_decode_graph='on'),
                              pins, installed=True, route_observer=True))
    same(capture['identity'], qualified_identity, 'qualified source/model/config differs from formal source selection')
    same(capture['environment'], environment, 'same qualified MLX environment')
    same({k:capture['transport'][k] for k in ('tcp_nodelay','io_timeout_seconds')},
         {k:transport[k] for k in ('tcp_nodelay','io_timeout_seconds')}, 'same qualified socket semantics')
    command, identity = select_entry(original, arm, server, pins)
    return {**original, 'schema':'E024-route-http-formal-spec-v1', 'arm':arm,
        'phase':f'r{repeat}-c{concurrency}-{arm}', 'command':command, 'identity':identity,
        'qualification':baseline_qualification if arm == 'off' else route_qualification,
        'baseline_qualification':baseline_qualification, 'route_qualification':route_qualification,
        'observer_enabled':False, 'sources':source_pins(),
        'scope':'Observer-OFF native paired timing. Prerequisites qualify captured histories, not future arrival schedules; install identity is not an invocation observer. No performance grant until independent outer audit.'}


def validate(spec):
    need(spec.get('schema') == 'E024-route-http-formal-spec-v1', 'route formal spec required')
    same(spec['sources'], source_pins(), 'exact original/new source closure')
    actual = prepare(spec['base_plan'], spec['concurrency'], spec['repeat'], arm=spec['arm'],
        baseline_qualification=spec['baseline_qualification'], route_qualification=spec['route_qualification'],
        environment=spec['environment'], transport=spec['transport'], deadlines=spec['deadlines'])
    same(spec, actual, 'formal spec changed')


async def run(spec, out, *, checkpoint, backend=None):
    """Single fresh trial. Mandatory outer checkpoint owns resources and service absence."""
    need(callable(checkpoint), 'external resource checkpoint required'); validate(spec)
    cap = load('http_capture_v2'); system = backend or System(); spec = copy.deepcopy(spec)
    out = Path(out).resolve(); out.mkdir(parents=True, exist_ok=False)
    d = spec['deadlines']; port = spec['transport']['port']; child = task = None
    stdout = stderr = None; phase = 'preflight'; last_group = None
    result = dict(schema='E024-route-http-formal-result-v1', status='INCOMPLETE', concurrency=spec['concurrency'],
        repeat=spec['repeat'], arm=spec['arm'], phase=spec['phase'], observer_enabled=False,
        full_state_qualified=False, count_eligible=False, performance_qualified=False, server_reaped=False,
        cleanup_complete=False, started_at=time.time(), errors=[],
        scope='One observer-OFF trial; measured point still requires outer runtime/resource/reap/restoration audit.')
    write(out / 'spec.json', spec); write(out / 'complete.json', result)
    write(out / 'launch.json', dict(command=spec['command'], environment=spec['environment']))
    events = (out / 'events.jsonl').open('x')
    def event(label, **fields):
        events.write(json.dumps(dict(label=label, time=time.time(), monotonic=time.monotonic(), **fields)) + '\n'); events.flush()
    def error(exc, where):
        result['errors'].append(dict(phase=where, type=type(exc).__name__, message=str(exc)))
    async def check(label, pid):
        value = checkpoint(label, pid)
        if inspect.isawaitable(value): await value
        event('checkpoint', phase=label, pid=pid)
    async def run_group(name):
        try:
            group = await system.group(spec, name, out / name)
            write(out / (name + '-group.json'), group); return group
        except BaseException as exc:
            if hasattr(exc, 'result'): write(out / (name + '-group.json'), exc.result)
            raise
    try:
        decode_text = system.decoder(spec)
        async with asyncio.timeout(d['overall']):
            await check('pre-spawn', None)
            before = await system.port(port); write(out / 'port-before.json', before)
            need(cap.no_process(before), 'formal port occupied or unknown')
            need(not any(flag in spec['command'] for flag in ('--qualification-trace', '--route-events')), 'both formal observers must be OFF')
            stdout = (out / 'server.stdout').open('xb'); stderr = (out / 'server.stderr').open('xb')
            phase = 'spawn'; write(out / 'spawn-intent.json', dict(command=spec['command'], time=time.time()))
            child = system.spawn(spec['command'], spec['environment'], stdout, stderr)
            write(out / 'child.json', dict(pid=child.pid, time=time.time())); await check('spawned', child.pid)
            identity = await system.identity(child.pid); write(out / 'child-identity.json', identity)
            need(identity['returncode'] == 0 and identity['stdout'].split()[:2] == [str(child.pid), str(os.getpid())], 'direct formal server identity')
            phase = 'readiness'
            async with asyncio.timeout(d['ready']):
                sampled = time.monotonic()
                while True:
                    need(child.poll() is None, 'formal server exited before ready')
                    raw = await system.readiness(port); event('readiness', raw=raw)
                    if cap.ready_identity(raw, spec['identity']):
                        listener = await system.port(port, child.pid); write(out / 'listener.json', listener)
                        need(cap.owned_listener(listener, child.pid, port), 'owned localhost listener')
                        write(out / 'ready.json', raw); break
                    if time.monotonic() - sampled >= d['resource_interval']:
                        await check('starting', child.pid); sampled = time.monotonic()
                    await asyncio.sleep(.1)
            await check('ready', child.pid)
            for phase in ('warm', 'formal'):
                task = asyncio.create_task(run_group(phase), name='native-' + phase)
                while not task.done():
                    await asyncio.wait([task], timeout=d['resource_interval'])
                    need(child.poll() is None, 'formal server exited during group')
                    await check(phase, child.pid)
                last_group = task.result()
                checked = check_group(last_group, spec, phase, decode_text)
                write(out / (phase + '-raw-check.json'), checked)
                if phase == 'formal':
                    result.update(count_eligible=True, point=checked['point'])
                    write(out / 'point.json', dict(**checked['point'], count_eligible=True, performance_qualified=False))
                await check('after-' + phase, child.pid)
                raw = await system.readiness(port); write(out / ('ready-after-' + phase + '.json'), raw)
                need(cap.ready_identity(raw, spec['identity']), 'server identity failed after group')
    except BaseException as exc:
        error(exc, phase)
    finally:
        async def finish_owned():
            try:
                if task is not None and not task.done():
                    task.cancel(); await asyncio.wait([task], timeout=d['close'] + 1)
                if task is not None and not task.done():
                    task.cancel(); error(TimeoutError('client unresolved'), 'client-cleanup')
            except BaseException as exc:
                error(exc, 'client-cleanup')
            finally:
                if child is not None:
                    try:
                        if child.poll() is None:
                            event('server-term', pid=child.pid)
                            try: child.terminate(); await system.wait(child, d['term'])
                            except subprocess.TimeoutExpired: event('server-term-timeout', pid=child.pid)
                            except BaseException as exc: error(exc, 'server-term')
                            if child.poll() is None:
                                event('server-kill', pid=child.pid); child.kill()
                        await system.wait(child, d['kill'])
                        absent = await system.absent(child.pid); write(out / 'child-absence.json', absent)
                        port_after = await system.port(port); write(out / 'port-after.json', port_after)
                        need(type(child.returncode) is int and cap.no_process(absent) and cap.no_process(port_after), 'strict server/port absence')
                        write(out / 'release.json', dict(pid=child.pid, returncode=child.returncode,
                            waited_reaped=True, strict_absence=True, port_absent=True, time=time.time()))
                        result['server_reaped'] = True
                        group_path = out / (phase + '-group.json')
                        result['cleanup_complete'] = task is None or (task.done() and group_path.exists()
                            and read(group_path)['cleanup']['complete'] is True)
                    except BaseException as exc: error(exc, 'server-cleanup')
        cleaner = asyncio.create_task(finish_owned(), name='native-formal-owned-cleanup')
        end = asyncio.get_running_loop().time() + d['close'] + d['term'] + d['kill'] + 12
        while not cleaner.done() and asyncio.get_running_loop().time() < end:
            try: await asyncio.wait([cleaner], timeout=max(0, end - asyncio.get_running_loop().time()))
            except asyncio.CancelledError as exc: error(exc, 'cancel-during-cleanup')
        if not cleaner.done():
            cleaner.cancel(); result['cleanup_complete'] = False; error(TimeoutError('owned cleanup deadline'), 'cleanup')
        elif cleaner.cancelled():
            result['cleanup_complete'] = False; error(RuntimeError('cleanup task cancelled'), 'cleanup')
        elif cleaner.exception() is not None:
            result['cleanup_complete'] = False; error(cleaner.exception(), 'cleanup')
        result['cleanup_task_resolved'] = cleaner.done()
        result['client_task_resolved'] = task is None or task.done()
        try:
            await check('after-reap' if result['server_reaped'] else 'cleanup-incomplete',
                        None if result['server_reaped'] or child is None else child.pid)
            validate(spec); result['small_pins_rechecked'] = True
        except BaseException as exc: error(exc, 'postflight')
        if stdout is not None: stdout.close()
        if stderr is not None: stderr.close()
        events.close(); result['ended_at'] = time.time()
        result['status'] = 'PASS' if result['count_eligible'] and result['cleanup_complete'] and not result['errors'] else 'FAIL'
        result['artifacts'] = {p.name: {'sha256': sha(p), 'bytes': p.stat().st_size}
            for p in sorted(out.iterdir()) if p.is_file() and p.name != 'complete.json'}
        write(out / 'complete.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path)
    parser.add_argument('--spec-sha256')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args(argv)
    if args.spec is None and args.spec_sha256 is None and args.out is None:
        print(json.dumps(dict(schema='E024-route-http-formal-plan-v1', trials=TRIALS,
            warm_requests_per_process=8, formal_requests_per_process=64,
            implementation='source-prepared; owner/controller and independent formal audit still required',
            device_run=False, performance_qualified=False), indent=2)); return 0
    need(all(value is not None for value in (args.spec,args.spec_sha256,args.out)), 'spec/SHA/out all required for CPU validation')
    spec = read(pinned(dict(path=str(args.spec.resolve()), sha256=args.spec_sha256)))
    validate(spec); args.out.mkdir(parents=True, exist_ok=False)
    write(args.out/'prepared.json', dict(status='PREPARED', spec_sha256=args.spec_sha256,
        device_run=False, performance_qualified=False, scope='Spec validated only; external direct-server owner required.'))
    return 0


if __name__ == '__main__': raise SystemExit(main())
