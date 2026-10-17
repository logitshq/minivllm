"""Fixed C1 capture/replay then C8 capture/replay while a v3 service is absent.

No service mutation occurs here. Preparation is CPU/source-only; execution needs
the exact STOPPED journal. Four sequential direct children have separate durable
intents and reap receipts. A failed/unknown capture never authorizes replay.
SIGKILL between intent and child receipt is deliberately unresolved, not no-child.
All timing is diagnostic until independent captured-schedule numerical audit.
"""
import argparse
import asyncio
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
PHASES = ('c1-capture', 'c1-replay', 'c8-capture', 'c8-replay')
WORK_SECONDS = 5700
REPLAY_SECONDS = 1830
POSTFLIGHT_SECONDS = 120
FIXED = {
    'qualify_absent_owner.py': '5beb6c9bcfa4e22041a386cdbea3f5ecee210cdcb9b345512c50b6c6f873c62d',
    'qualify_owner.py': '1efaf440e991fed64dedf15455be2d37223e709fec41bb07e8c0295a2975c1c2',
    'qualify_native.py': '26964fcbed9a2e5dcefaee4888244987897f3138d4a2b7add59bf63a7461b7bf',
    'http_capture_v2.py': '536fe41d6c8e2bdffe83ba6f43fba87dd6265deb4c84f6204bc9c7220f8280e3',
    'http_qualification.py': '471bebd8a114d3a7f5a190c20909f6896f17fd39f282656546c1366c3fa2c6e2',
    'http_join.py': 'a362c321c2329d7ab53ab6a1a5a96981bb95b44733739fdbb56dd3d38bef7b4e',
    'http_replay.py': '1304c58c66e56b3aeaa582ef9f7d61317280cf3107f8f600485426119970fb65',
    'native_cleanup.py': '6c292a9d3cb28c047d1dc92ac0fec4abe19fa8df84c172ef61d2f6ce41157ca6',
    'native_server.py': '9bcb37eff8bdecd79800afdedc82deacf9f7a0a9a7a1e0d62dc8ba49e70fb68f',
    'client.py': 'ef64a114f4f7d90ec5044d6bdcf058ca05f98dea445d6639db706a4fc75217dc',
}
NUMERICAL_SHA = '21ba52e43ce54c2b463ab46bfd212c53d06d6303530c77939e3e6b801df547fb'


def need(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(name):
    path = HERE / (name + '.py')
    need(sha(path) == FIXED[path.name], 'frozen source changed: ' + str(path))
    spec = importlib.util.spec_from_file_location('_http_owner_' + name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def source_pins():
    return {**{str(HERE / n): digest for n, digest in FIXED.items()},
            str(Path(__file__).resolve()): sha(__file__)}


def plan_template(qualification, cold_restart, http):
    """Call on CPU, save exact JSON, then bind its SHA in controller prepare."""
    return dict(schema='E023-http-service-absent-owner-plan-v1',
        execution={'owner': {'path': str(Path(__file__).resolve()), 'sha256': sha(__file__)},
                   'pins': source_pins()}, qualification=qualification,
        cold_restart=cold_restart, http=http, deadline_seconds=WORK_SECONDS)


def prior_replacement(plan):
    """Reuse the pinned controller's exact archived restoration validation."""
    item = plan['cold_restart']['controller']
    need(sha(item['path']) == item['sha256'], 'controller source changed')
    spec = importlib.util.spec_from_file_location('_http_owner_controller_v3', item['path'])
    controller = importlib.util.module_from_spec(spec); spec.loader.exec_module(controller)
    prior, _ = controller.predecessor(plan['cold_restart']['predecessor'])
    return prior['replacement']


def validate_plan(args):
    base = load('qualify_absent_owner')
    raw = base.absolute(args.plan).read_bytes()
    need(base.sha_bytes(raw) == args.plan_sha256, 'HTTP owner plan SHA')
    plan = json.loads(raw)
    need(set(plan) == {'schema', 'execution', 'qualification', 'cold_restart', 'http', 'deadline_seconds'}
         and plan['schema'] == 'E023-http-service-absent-owner-plan-v1', 'HTTP owner plan schema')
    need(type(plan['deadline_seconds']) is int and plan['deadline_seconds'] == WORK_SECONDS, 'fixed work budget')
    need(plan['execution'] == {'owner': {'path': str(Path(__file__).resolve()), 'sha256': sha(__file__)},
                              'pins': source_pins()}, 'exact owner and adapter pins')
    for path, digest in plan['execution']['pins'].items():
        need(sha(path) == digest, 'owner dependency changed: ' + path)
    q = plan['qualification']
    need(set(q) == {'path', 'sha256'} and q['sha256'] == NUMERICAL_SHA
         and sha(base.absolute(q['path'])) == NUMERICAL_SHA, 'unchanged numerical plan required')
    numeric = json.loads(Path(q['path']).read_bytes())
    cold = plan['cold_restart']
    need(set(cold) == {'journal_path', 'controller', 'approved_contract', 'proposal_plan_sha256', 'predecessor'},
         'v3 cold fields')
    base.absolute(cold['journal_path'])
    c = cold['controller']
    need(set(c) == {'path', 'sha256'} and base.absolute(c['path']) == HERE.parent / 'omlx_cold_restart/controller_v3.py'
         and sha(c['path']) == c['sha256'], 'actual v3 controller pin')
    need(cold['approved_contract'] == base.CONTRACT and cold['proposal_plan_sha256'] == base.PROPOSAL_SHA,
         'approved unchanged cold restart contract')
    predecessor = cold['predecessor']
    need(set(predecessor) == {'path', 'sha256', 'audit'} and set(predecessor['audit']) == {'path', 'sha256'},
         'immutable restored predecessor and audit required')
    for item in (predecessor, predecessor['audit']):
        need(sha(base.absolute(item['path'])) == item['sha256'], 'predecessor evidence changed')
    prior_replacement(plan)  # Restoration may PASS after a failed experiment.
    need(set(plan['http']) == {'transport', 'deadlines', 'environment'}, 'fixed capture config')
    d = plan['http']['deadlines']
    need(d.get('overall') == 900 and d.get('term') == 15 and d.get('kill') == 15
         and 0 < d.get('resource_interval', 0) <= 5 and 0 < d.get('close', 0) <= 15,
         'capture 900s/TERM15/KILL15 bounds required')
    capture = load('http_capture_v2')
    specs = [capture.prepare(q['path'], q['sha256'], concurrency,
        python=numeric['execution']['python'], **plan['http']) for concurrency in (1, 8)]
    return base, plan, raw, numeric, specs


def stopped(base, plan, args, binding=None):
    raw = (Path(plan['cold_restart']['journal_path']) / 'journal.json').read_bytes()
    j = json.loads(raw)
    expected = dict(schema='omlx-cold-restart-journal-v3', state='STOPPED',
        plan_sha256=base.PROPOSAL_SHA, controller_sha256=plan['cold_restart']['controller']['sha256'],
        approved_contract=base.CONTRACT, owner_directory=str(args.out),
        owner_plan={'path': str(args.plan), 'sha256': args.plan_sha256},
        owner_source=plan['execution']['owner'], predecessor=plan['cold_restart']['predecessor'])
    need(all(j.get(k) == v for k, v in expected.items()), 'bound v3 journal is not STOPPED')
    need(type(j.get('original')) is dict and type(j['original'].get('pid')) is int and j['original']['pid'] > 1,
         'v3 current original identity required')
    need(j['original'] == (prior_replacement(plan) if binding is None else binding['original']),
         'current original differs from verified predecessor replacement')
    need(isinstance(j.get('id'), str) and j['id'] and type(j.get('removed_at')) in (float, int)
         and math.isfinite(j['removed_at']) and 0 < j['removed_at'] <= time.time(), 'journal runtime identity')
    need(type(j.get('predecessor_copies')) is dict, 'controller retained predecessor copies required')
    stable = {k: j[k] for k in (*expected, 'id', 'removed_at', 'original', 'predecessor_copies')}
    need(binding is None or stable == binding, 'STOPPED journal identity changed')
    return raw, stable


def full_prepare(base, plan, numeric, out, check):
    """Frozen runtime/source preparation plus all model manifest files, no imports."""
    out.mkdir()
    q = plan['qualification']
    base.load_prepare()(SimpleNamespace(out=out, plan=Path(q['path']), plan_sha256=q['sha256'],
                                       deadline_seconds=WORK_SECONDS), check)
    manifest = json.loads(Path(numeric['model_manifest']['path']).read_bytes())
    pins = {}
    for relative, expected in manifest['files'].items():
        need(not Path(relative).is_absolute() and '..' not in Path(relative).parts, 'model manifest traversal')
        path = Path(numeric['model']) / relative
        digest = hashlib.sha256()
        before = path.stat()
        with path.open('rb') as stream:
            while chunk := stream.read(8 << 20):
                check(); digest.update(chunk)
        after = path.stat()
        need((before.st_ino, before.st_size, before.st_mtime_ns) == (after.st_ino, after.st_size, after.st_mtime_ns)
             and digest.hexdigest() == expected, 'model pin changed: ' + relative)
        pins[str(path)] = dict(sha256=expected, bytes=after.st_size, inode=after.st_ino, mtime_ns=after.st_mtime_ns)
        base.durable(out / 'verified-model-pins.json', pins)
    for path, digest in plan['execution']['pins'].items():
        check(); raw = Path(path).read_bytes(); need(base.sha_bytes(raw) == digest, 'adapter pin changed')
        dest = out / 'frozen-adapters' / Path(path).name
        dest.parent.mkdir(exist_ok=True)
        base.durable(dest, raw, exclusive=True, raw=True)


class System:
    """All real OS actions are execution-only; CPU tests inject a fake."""
    def __init__(self, base):
        self.base = base.System()
    def __getattr__(self, name):
        return getattr(self.base, name)
    def postflight_deadline(self):
        signal.setitimer(signal.ITIMER_REAL, POSTFLIGHT_SECONDS)
    def capture_backend(self, module):
        return module.System()


def run(args, *, system=None, resources=None, capture_module=None):
    base, plan, plan_raw, numeric, specs = validate_plan(args)
    system = system or System(base)
    capture_module = capture_module or load('http_capture_v2')
    directory = Path(plan['cold_restart']['journal_path'])
    started = time.monotonic()
    children, samples, errors, stages = [], [], [], []
    absence_count = 0
    reap_attempts = {}
    binding = attempt = None
    intent = deadline_active = cancelled = False
    capture_task = None
    phase = 'attempt'
    live = {}  # only direct Popen objects actually returned to this owner

    def failure(exc, where):
        row = dict(phase=where, type=type(exc).__name__, error=str(exc), time=time.time())
        errors.append(row); base.durable(args.out / 'errors.json', errors)
        if len(errors) == 1:
            base.durable(args.out / 'first-error.json', row, exclusive=True)

    def stop(*unused):
        nonlocal cancelled
        cancelled = True
        if capture_task is not None:
            capture_task.cancel()
        else:
            raise TimeoutError('HTTP owner absolute deadline or cancellation')

    def check(*, cleanup=False):
        if not cleanup:
            need(not cancelled and time.monotonic() - started < WORK_SECONDS, 'owner cancelled/deadline')
        return stopped(base, plan, args, binding)

    def probe(label, pid=None, *, cleanup=False):
        nonlocal absence_count
        check(cleanup=cleanup)
        absence = base.absence(system, binding['original']['pid'])
        n = len(samples)
        absence_count += 1
        base.durable(args.out / f'absence-{absence_count-1:03d}.json',
                     dict(label=label, time=time.time(), resource_index_if_sampled=n, **absence), exclusive=True)
        need(absence['status'] == 'PASS', 'removed service PID/job/8001 reappeared or unknown')
        sample = resources.snapshot(label, pid, suspended_pids=())
        samples.append(sample)
        base.durable(args.out / f'resource-{n:03d}.json', sample, exclusive=True)
        resources.guard(samples)
        need(sample['authorized_suspended_pids'] == [] and all(p['owned'] and p['pid'] == pid
             and not p['authorized_suspended'] for p in sample['native_inference_processes']), 'foreign process/exemption')
        need(sample['pressure_level_value'] == 1 and sample['power_metadata']['source'] == 'AC Power'
             and sample['processinfo_thermal_metadata']['code'] in (0, 1), 'pressure/power/thermal guard')
        check(cleanup=cleanup)

    def ref(path):
        return {'path': str(path.relative_to(args.out)), 'sha256': sha(path)}

    def index():
        base.durable(args.out / 'children.json', dict(schema='cold-service-owner-children-v1', children=children))

    def bound():
        return {k: attempt[k] for k in ('owner_pid', 'owner_identity', 'owner_source', 'owner_plan_sha256',
                                        'journal_id', 'removed_at')}

    def spawn_impl(stage, spawn_call):
        nonlocal intent
        folder = args.out / 'phases' / stage
        with base.journal_lock(directory):
            check()
            absent = base.absence(system, binding['original']['pid'])
            base.durable(folder / 'pre-spawn-absence.json', absent, exclusive=True)
            need(absent['status'] == 'PASS', 'service absence changed before spawn intent')
            evidence = dict(phase=stage, owner_pid=os.getpid(), journal_id=binding['id'],
                owner_plan_sha256=args.plan_sha256, attempt_sha256=sha(args.out / 'attempt.json'), time=time.time())
            if not intent:
                intent = True  # failed fsync is unknown, never NO_CHILD_STARTED
                base.durable(args.out / 'spawn-intent.json', evidence, exclusive=True)
            base.durable(folder / 'intent.json', evidence, exclusive=True)
            child = spawn_call()  # unknown outcome after intent deliberately fails closed
            live[stage] = child
            need(type(child.pid) is int and child.pid > 1 and child.pid != os.getpid()
                 and all(row['pid'] != child.pid for row in children), 'unique returned child PID required')
            child_receipt = dict(phase=stage, pid=child.pid, owner_pid=os.getpid(), identity=None,
                                 ownership='direct-Popen-returned', spawned_at=time.time())
            base.durable(folder / 'child.json', child_receipt, exclusive=True)
            entry = dict(phase=stage, pid=child.pid, identity=None,
                intent=ref(folder / 'intent.json'), child=ref(folder / 'child.json'), release=None)
            children.append(entry)
            index()
            # A failure here is still a known direct Popen child. It is reaped,
            # retained as identity:null, and never admitted to the next phase.
            identity = system.identity(child.pid)
            need(identity['pid'] == child.pid and identity['ppid'] == os.getpid(), 'direct owned child identity')
            child_receipt['identity'] = entry['identity'] = identity
            base.durable(folder / 'child.json', child_receipt)
            entry['child'] = ref(folder / 'child.json'); index()
            return child

    def spawn(stage, spawn_call):
        try:
            return spawn_impl(stage, spawn_call)
        except BaseException as exc:
            # Frozen capture writes its local intent immediately before invoking
            # this tracked backend. A failed lock/check here has not called Popen.
            # Record that positive fact; absence of a child file is insufficient.
            inner = args.out / 'phases/c1-capture/capture/spawn-intent.json'
            if stage == 'c1-capture' and not intent and not live and not children and inner.exists():
                raw = json.loads(inner.read_bytes())
                base.durable(args.out / 'phases/c1-capture/no-popen.json', dict(
                    schema='cold-service-no-popen-v1', phase=stage, owner_pid=os.getpid(),
                    journal_id=binding['id'], owner_plan_sha256=args.plan_sha256,
                    attempt_sha256=sha(args.out / 'attempt.json'),
                    ownership='tracked-backend-rejected-before-Popen', popen_called=False,
                    time=time.time(), inner_spawn_intent=ref(inner), command=raw['command'],
                    error={'type': type(exc).__name__, 'message': str(exc)}), exclusive=True)
            raise

    def release(stage):
        child = live[stage]
        folder = args.out / 'phases' / stage
        serial = reap_attempts.get(stage, 0)
        reap_attempts[stage] = serial + 1
        if child.poll() is None:
            try:
                child.terminate(); child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                pass
            except BaseException as exc:
                failure(exc, 'terminate:' + stage)
            if child.poll() is None:
                child.kill()
        child.wait(timeout=15)
        raw = system.read(['/bin/ps', '-p', str(child.pid), '-o', 'pid='])
        absent_path = folder / f'child-absence-{serial:02d}.json'
        base.durable(absent_path, raw, exclusive=True)
        need(type(child.returncode) is int and raw['returncode'] == 1 and not raw['stdout'].strip(),
             'strict direct child reap/absence missing')
        if stage.endswith('capture'):
            raw_port = system.read(['/usr/sbin/lsof', '-nP', '-a',
                f"-iTCP:{plan['http']['transport']['port']}", '-sTCP:LISTEN', '-Fpn'])
            base.durable(folder / f'port-absence-{serial:02d}.json', raw_port, exclusive=True)
            need(raw_port['returncode'] == 1 and not raw_port['stdout'].strip(), 'capture listener survives')
        receipt = dict(phase=stage, pid=child.pid, returncode=child.returncode,
                       waited_reaped=True, strict_absence=True, absence=ref(absent_path), time=time.time())
        release_path = folder / 'release.json'
        if release_path.exists():
            # A previous durable release may precede an index write failure. Do
            # not overwrite it or treat mere file presence as successful reap.
            old = json.loads(release_path.read_bytes())
            need(set(old) == set(receipt) and all(old[k] == receipt[k] for k in
                ('phase', 'pid', 'returncode', 'waited_reaped', 'strict_absence'))
                and type(old['time']) in (int, float) and math.isfinite(old['time'])
                and binding['removed_at'] <= old['time'] <= time.time(), 'existing release mismatch')
            previous_absence = args.out / old['absence']['path']
            need(previous_absence.parent == folder and previous_absence.name.startswith('child-absence-')
                 and sha(previous_absence) == old['absence']['sha256'], 'existing release absence pin')
            previous_raw = json.loads(previous_absence.read_bytes())
            need(previous_raw['command'] == ['/bin/ps', '-p', str(child.pid), '-o', 'pid=']
                 and previous_raw['returncode'] == 1 and not previous_raw['stdout'].strip(), 'existing strict absence')
        else:
            base.durable(release_path, receipt, exclusive=True)
        row = next(row for row in children if row['phase'] == stage)
        row['release'] = ref(release_path); index()
        del live[stage]

    async def capture_one(spec, folder, stage):
        nonlocal capture_task
        underlying = system.capture_backend(capture_module)
        class Tracked:
            def __getattr__(self, name):
                return getattr(underlying, name)
            def spawn(self, command, environment, stdout, stderr):
                return spawn(stage, lambda: underlying.spawn(command, environment, stdout, stderr))
        async def checkpoint(label, pid):
            # Keep synchronous probe processes off the HTTP loop; finish the bounded
            # probe before propagating cancellation so no thread writes after reap.
            task = asyncio.create_task(asyncio.to_thread(probe, stage + ':' + label, pid,
                cleanup=label in ('after-reap', 'cleanup-incomplete')))
            interrupted = False
            while not task.done():
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    interrupted = True
            task.result()
            if interrupted:
                raise asyncio.CancelledError()
        capture_task = asyncio.create_task(capture_module.capture(spec, folder / 'capture',
                                            checkpoint=checkpoint, backend=Tracked()))
        try:
            return await capture_task
        finally:
            capture_task = None

    with base.journal_lock(directory):
        snapshot, binding = stopped(base, plan, args)
        identity = system.identity(os.getpid())
        need(identity['pid'] == os.getpid(), 'owner identity')
        args.out.mkdir(exist_ok=False)
        (args.out / 'phases').mkdir()
        base.durable(args.out / 'stopped-journal.json', snapshot, exclusive=True, raw=True)
        attempt = dict(schema='cold-service-owner-attempt-v1', owner_pid=os.getpid(), owner_identity=identity,
            owner_source=plan['execution']['owner'], owner_plan_sha256=args.plan_sha256,
            journal_id=binding['id'], journal_path=str(directory), controller_sha256=binding['controller_sha256'],
            removed_at=binding['removed_at'], created_at=time.time(), phase='PRESPAWN',
            stopped_journal_sha256=base.sha_bytes(snapshot))
        base.durable(args.out / 'attempt.json', attempt, exclusive=True)
        index()
    try:
        system.start_deadline(WORK_SECONDS - (time.monotonic() - started), stop); deadline_active = True
        base.durable(args.out / 'owner-plan.json', plan_raw, exclusive=True, raw=True)
        resources = resources or base.load_resources(numeric)
        phase = 'prepare'
        probe('before-prepare')
        full_prepare(base, plan, numeric, args.out / 'preflight', check)
        base.durable(args.out / 'capture-specs.json', specs, exclusive=True)
        probe('after-prepare')
        for spec in specs:
            concurrency = spec['concurrency']
            phase = f'c{concurrency}-capture'
            folder = args.out / 'phases' / phase; folder.mkdir()
            result = asyncio.run(capture_one(spec, folder, phase))
            if phase in live:
                release(phase)
            entry = next((row for row in children if row['phase'] == phase), None)
            need(entry is not None and entry['release'] is not None
                 and json.loads((folder / 'capture/child.json').read_bytes())['pid'] == entry['pid'],
                 'capture did not return the actually owned/reaped server')
            need(result['status'] == 'PASS' and result['cleanup_complete'] is True
                 and result['server_reaped'] is True and result['cleanup_task_resolved'] is True
                 and result['client_task_resolved'] is True, 'capture/cleanup failed; replay forbidden')
            check(); probe(phase + ':released')
            stages.append(dict(phase=phase, status='PASS', result=ref(folder / 'capture/complete.json')))
            # Only closed, fully reaped capture artifacts enter the replay plan.
            q = load('http_qualification')
            replay_plan = dict(schema='E023-recorded-http-plan-v1', qualification=plan['qualification'],
                concurrency=concurrency,
                transport={k: plan['http']['transport'][k] for k in ('tcp_nodelay', 'io_timeout_seconds')},
                inputs={k: {'path': str(folder / 'capture' / filename), 'sha256': sha(folder / 'capture' / filename)}
                        for k, filename in (('group', 'group.json'), ('trace', 'trace.jsonl'))},
                sources={str(p): sha(p) for p in (*q.FIXED, *q.ADAPTERS)})
            phase = f'c{concurrency}-replay'
            folder = args.out / 'phases' / phase; folder.mkdir()
            base.durable(folder / 'plan.json', replay_plan, exclusive=True)
            replay_sha = sha(folder / 'plan.json')
            q.validate(folder / 'plan.json', replay_sha)
            command = [numeric['execution']['python'], '-I', '-B', str(HERE / 'http_qualification.py'),
                '--plan', str(folder / 'plan.json'), '--plan-sha256', replay_sha, '--out', str(folder / 'qualification')]
            base.durable(folder / 'launch.json', dict(command=command, environment=plan['http']['environment']), exclusive=True)
            probe(phase + ':pre-spawn')
            phase_started = time.monotonic()
            with (folder / 'child.log').open('x') as log:
                child = spawn(phase, lambda: system.spawn(command, log, plan['http']['environment']))
                try:
                    while child.poll() is None:
                        check(); need(time.monotonic() - phase_started < REPLAY_SECONDS, 'replay deadline')
                        probe(phase + ':running', child.pid)
                        try:
                            child.wait(timeout=min(5, max(.001, REPLAY_SECONDS - (time.monotonic() - phase_started))))
                        except subprocess.TimeoutExpired:
                            pass
                    child.wait(timeout=0)
                finally:
                    release(phase)
            need(child.returncode == 0, 'replay child exit ' + str(child.returncode))
            result = json.loads((folder / 'qualification/complete.json').read_bytes())
            need(result['status'] == 'PASS' and result['numerical_qualified'] is True
                 and result['concurrency'] == concurrency and result['total_requests'] == (8 if concurrency == 1 else 16)
                 and result['plan_sha256'] == replay_sha and result['input_source_pins_rechecked'] is True,
                 'complete captured-schedule replay missing')
            stages.append(dict(phase=phase, status='PASS', result=ref(folder / 'qualification/complete.json')))
            probe(phase + ':released')
        need([row['phase'] for row in children] == list(PHASES)
             and [row['phase'] for row in stages] == list(PHASES), 'complete four-phase ownership prefix required')
    except BaseException as exc:
        failure(exc, phase)
    finally:
        if deadline_active:
            system.cleanup_mode()
        for stage in list(live):
            try:
                release(stage)
            except BaseException as exc:
                failure(exc, 'cleanup:' + stage)
        try:
            if deadline_active:
                system.postflight_deadline()
            if resources is not None:
                probe('postflight', cleanup=True)
                base.durable(args.out / 'resources.json', resources.deltas(samples))
            validate_plan(args)
            full_prepare(base, plan, numeric, args.out / 'postflight', lambda: check(cleanup=True))
        except BaseException as exc:
            failure(exc, 'postflight')
        finally:
            if deadline_active:
                system.restore_deadline()
        deep_spawn = [p for p in (args.out / 'phases').rglob('*.json')
                      if p.name in ('intent.json', 'child.json', 'release.json', 'spawn-intent.json')]
        no_popen = args.out / 'phases/c1-capture/no-popen.json'
        only_rejected_inner = (no_popen.exists() and deep_spawn == [args.out / 'phases/c1-capture/capture/spawn-intent.json'])
        if not intent and (not deep_spawn or only_rejected_inner):
            need(not children and not live and errors, 'no-child requires pre-intent failure')
            base.durable(args.out / 'terminal.json', dict(schema='cold-service-owner-terminal-v1',
                status='NO_CHILD_STARTED', **bound(), terminal_at=time.time(), phase=errors[0]['phase'],
                error=errors[0], attempt_sha256=sha(args.out / 'attempt.json'),
                spawn_intent_written=False, child_started=False,
                no_popen_receipt=ref(no_popen) if only_rejected_inner else None), exclusive=True)
        elif intent:
            intended = [stage for stage in PHASES if (args.out / 'phases' / stage / 'intent.json').exists()]
            resolved = [row['phase'] for row in children if row['release'] is not None]
            if not live and intended == resolved and resolved == list(PHASES[:len(resolved)]):
                base.durable(args.out / 'release.json', dict(schema='cold-service-owner-release-v2',
                    **bound(), attempt_sha256=sha(args.out / 'attempt.json'), children=children,
                    children_sha256=sha(args.out / 'children.json'), all_children_reaped=True, time=time.time()), exclusive=True)
        else:
            failure(RuntimeError('nested spawn evidence without global intent; no no-child certificate'), 'terminal')
        result = dict(schema='E023-http-owner-result-v1', status='FAIL' if errors else 'PASS',
            owner_pid=os.getpid(), owner_plan_sha256=args.plan_sha256, errors=errors, absence_probes=absence_count,
            first_failure=errors[0] if errors else None, phases=stages, samples=len(samples),
            spawn_intent_written=intent,
            child_started=True if children else False if (args.out / 'terminal.json').exists() else None,
            all_children_reaped=(args.out / 'release.json').exists(), elapsed_wall_s=time.monotonic() - started,
            numerical_replay_complete=not errors and len(stages) == 4,
            independently_qualified=False, performance_qualified=False,
            scope='Fixed baseline C1/C8 captured schedules only; independent raw replay/resource/restore audit required.')
        base.durable(args.out / 'complete.json', result)
    return int(bool(errors))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--plan-sha256', required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--prepare-only', action='store_true')
    args = p.parse_args(argv)
    args.plan, args.out = args.plan.resolve(), args.out.resolve()
    need(args.out.parent.is_dir() and not args.out.exists(), 'fresh output in existing parent required')
    if args.prepare_only:
        base, plan, _, _, specs = validate_plan(args)
        args.out.mkdir()
        base.durable(args.out / 'prepared.json', dict(schema='E023-http-owner-prepare-v1', status='PREPARED',
            owner_plan_sha256=args.plan_sha256, capture_specs=specs, sources=plan['execution']['pins'],
            runtime_weights_verified=False, scope='CPU small source/metadata only; no service or model action'), exclusive=True)
        return 0
    return run(args)


if __name__ == '__main__':
    raise SystemExit(main())
