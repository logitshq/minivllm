"""Two or six fixed fresh native formal trials while a v4 service is absent.

No launchd/service mutation. Each rN-cC trial uses the frozen observer-OFF
runner, eight warm requests then 64 formal requests. Numerical eligibility
requires the pinned independent captured-history audit for that concurrency.
Direct Popen ownership, durable intents, pressure1/AC/thermal/swap guards and
full runtime/source/weight pre/post checks preserve the earlier owner contract.
The output contains individual count-eligible points; independent resource and
restoration audit is still required before claiming performance qualification.
Unknown spawn outcomes or SIGKILL receipt gaps never certify no child.
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
PHASES = ('r1-c1', 'r1-c8', 'r2-c8', 'r2-c1', 'r3-c1', 'r3-c8')
PREPARE_SECONDS = 300
POSTFLIGHT_SECONDS = 120
FIXED = {
    'http_owner_v2.py': 'f645bc1a32bee8fdaf57d1626a84ebce1ad6d1c775ecdc9b6de3ebbc12105bf1',
    'native_formal.py': 'ec9cdc924f7987ad46eac6820b6264495c95fb182c726545a6e3ce283df7e654',
    'audit_http_replay.py': '1e26e35bdde5cf7b81da37101326a4ef6f87ef7db2775c6324b56d053645c995',
    'audit_http_replay_v2.py': '7cdcf5a2f02768ef75686b0ff0b0da9d21f5f9c82ee3866e24c1f869c709f031',
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
    spec = importlib.util.spec_from_file_location('_native_formal_owner_' + name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def source_pins():
    return {**{str(HERE / n): digest for n, digest in FIXED.items()},
            str(Path(__file__).resolve()): sha(__file__)}


def trial_names(trials):
    need(type(trials) is list and (trials == list(PHASES[:2]) or trials == list(PHASES)),
         'only complete registered two- or six-trial cycles')
    return trials


def work_seconds(trials, deadlines):
    trial_names(trials)
    values = [deadlines[k] for k in ('overall', 'close', 'term', 'kill')]
    need(all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in values), 'finite trial budgets')
    return math.ceil(PREPARE_SECONDS + len(trials) * (sum(values) + 12))


def plan_template(qualification, cold_restart, http, qualifications, trials=None):
    """Small CPU recipe, saved/pinned before the new controller's prepare call."""
    trials = list(PHASES[:2]) if trials is None else trial_names(trials)
    return dict(schema='E023-native-formal-owner-plan-v1',
        execution={'owner': {'path': str(Path(__file__).resolve()), 'sha256': sha(__file__)},
                   'pins': source_pins()}, qualification=qualification, qualifications=qualifications,
        cold_restart=cold_restart, http=http, trials=trials,
        deadline_seconds=work_seconds(trials, http['deadlines']))


def prior_replacement(plan):
    """Reuse the pinned controller's exact archived restoration validation."""
    item = plan['cold_restart']['controller']
    need(sha(item['path']) == item['sha256'], 'controller source changed')
    spec = importlib.util.spec_from_file_location('_native_formal_owner_controller_v4', item['path'])
    controller = importlib.util.module_from_spec(spec); spec.loader.exec_module(controller)
    prior, _ = controller.predecessor(plan['cold_restart']['predecessor'])
    return prior['replacement']


def validate_plan(args):
    base = load('qualify_absent_owner')
    raw = base.absolute(args.plan).read_bytes()
    need(base.sha_bytes(raw) == args.plan_sha256, 'native formal owner plan SHA')
    plan = json.loads(raw)
    need(set(plan) == {'schema', 'execution', 'qualification', 'qualifications', 'cold_restart', 'http', 'trials', 'deadline_seconds'}
         and plan['schema'] == 'E023-native-formal-owner-plan-v1', 'native formal owner plan schema')
    trial_names(plan['trials'])
    need(type(plan['deadline_seconds']) is int and plan['deadline_seconds'] == work_seconds(plan['trials'], plan['http']['deadlines']), 'exact fixed-trial work budget')
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
         'v4 cold fields')
    base.absolute(cold['journal_path'])
    c = cold['controller']
    need(set(c) == {'path', 'sha256'} and base.absolute(c['path']) == HERE.parent / 'omlx_cold_restart/controller_v4.py'
         and sha(c['path']) == c['sha256'], 'actual v4 controller pin')
    need(cold['approved_contract'] == base.CONTRACT and cold['proposal_plan_sha256'] == base.PROPOSAL_SHA,
         'approved unchanged cold restart contract')
    predecessor = cold['predecessor']
    need(set(predecessor) == {'path', 'sha256', 'audit'} and set(predecessor['audit']) == {'path', 'sha256'},
         'immutable restored predecessor and audit required')
    for item in (predecessor, predecessor['audit']):
        need(sha(base.absolute(item['path'])) == item['sha256'], 'predecessor evidence changed')
    prior_replacement(plan)  # Restoration may PASS after a failed experiment.
    need(set(plan['http']) == {'transport', 'deadlines', 'environment'}, 'fixed formal config')
    d = plan['http']['deadlines']
    need(0 < d.get('overall', 0) <= 3600 and d.get('term') == d.get('kill') == 15
         and 0 < d.get('resource_interval', 0) <= 5 and 0 < d.get('close', 0) <= 15,
         'formal overall<=3600s/TERM15/KILL15/resource bounds required')
    need(set(plan['qualifications']) == {'1', '8'}, 'independent C1 and C8 numerical audit bindings')
    formal = load('native_formal')
    specs = [formal.prepare(q, int(stage[-1]), int(stage[1]),
        qualification=plan['qualifications'][stage[-1]], **plan['http']) for stage in plan['trials']]
    return base, plan, raw, numeric, specs


def stopped(base, plan, args, binding=None):
    raw = (Path(plan['cold_restart']['journal_path']) / 'journal.json').read_bytes()
    j = json.loads(raw)
    expected = dict(schema='omlx-cold-restart-journal-v4', state='STOPPED',
        plan_sha256=base.PROPOSAL_SHA, controller_sha256=plan['cold_restart']['controller']['sha256'],
        approved_contract=base.CONTRACT, owner_directory=str(args.out),
        owner_plan={'path': str(args.plan), 'sha256': args.plan_sha256},
        owner_source=plan['execution']['owner'], predecessor=plan['cold_restart']['predecessor'],
        trials=plan['trials'])
    need(all(j.get(k) == v for k, v in expected.items()), 'bound v4 journal is not STOPPED')
    need(type(j.get('original')) is dict and type(j['original'].get('pid')) is int and j['original']['pid'] > 1,
         'v4 current original identity required')
    need(j['original'] == (prior_replacement(plan) if binding is None else binding['original']),
         'current original differs from verified predecessor replacement')
    need(isinstance(j.get('id'), str) and j['id'] and type(j.get('removed_at')) in (float, int)
         and math.isfinite(j['removed_at']) and 0 < j['removed_at'] <= time.time(), 'journal runtime identity')
    need(type(j.get('predecessor_copies')) is dict, 'controller retained predecessor copies required')
    stable = {k: j[k] for k in (*expected, 'id', 'removed_at', 'original', 'predecessor_copies')}
    need(binding is None or stable == binding, 'STOPPED journal identity changed')
    return raw, stable


def full_prepare(base, plan, numeric, out, check):
    """Reuse original runtime/model SHA and stat checks without source spoofing."""
    return load('http_owner_v2').full_prepare(base, plan, numeric, out, check)


class System:
    """All real OS actions are execution-only; CPU tests inject a fake."""
    def __init__(self, base):
        self.base = base.System()
    def __getattr__(self, name):
        return getattr(self.base, name)
    def postflight_deadline(self):
        signal.setitimer(signal.ITIMER_REAL, POSTFLIGHT_SECONDS)
    def formal_backend(self, module):
        return module.System()


def run(args, *, system=None, resources=None, formal_module=None):
    base, plan, plan_raw, numeric, specs = validate_plan(args)
    system = system or System(base)
    formal_module = formal_module or load('native_formal')
    directory = Path(plan['cold_restart']['journal_path'])
    phases = tuple(plan['trials']); budget = plan['deadline_seconds']
    started = time.monotonic()
    children, samples, errors, stages = [], [], [], []
    absence_count = 0
    reap_attempts = {}
    phase_intent_started = set()
    binding = attempt = None
    intent = deadline_active = cancelled = False
    trial_task = None
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
        if trial_task is not None:
            trial_task.cancel()
        else:
            raise TimeoutError('formal owner absolute deadline or cancellation')

    def check(*, cleanup=False):
        if not cleanup:
            need(not cancelled and time.monotonic() - started < budget, 'owner cancelled/deadline')
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
            phase_intent_started.add(stage)  # failed durable write is not a no-Popen proof
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
            # Frozen formal runner writes its local intent immediately before invoking
            # this tracked backend. A failed lock/check here has not called Popen.
            # Record that positive fact; absence of a child file is insufficient.
            inner = args.out / 'phases' / stage / 'trial/spawn-intent.json'
            known_next = (len(children) < len(phases) and stage == phases[len(children)]
                          and all(row['release'] is not None for row in children))
            if (known_next and stage not in phase_intent_started and not live
                    and (not intent or bool(children)) and inner.exists()):
                raw = json.loads(inner.read_bytes())
                base.durable(args.out / 'phases' / stage / 'no-popen.json', dict(
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
        if stage in phases:
            raw_port = system.read(['/usr/sbin/lsof', '-nP', '-a',
                f"-iTCP:{plan['http']['transport']['port']}", '-sTCP:LISTEN', '-Fpn'])
            base.durable(folder / f'port-absence-{serial:02d}.json', raw_port, exclusive=True)
            need(raw_port['returncode'] == 1 and not raw_port['stdout'].strip(), 'formal listener survives')
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

    async def trial_one(spec, folder, stage):
        nonlocal trial_task
        underlying = system.formal_backend(formal_module)
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
        trial_task = asyncio.create_task(formal_module.run(spec, folder / 'trial',
                                            checkpoint=checkpoint, backend=Tracked()))
        try:
            return await trial_task
        finally:
            trial_task = None

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
        system.start_deadline(budget - (time.monotonic() - started), stop); deadline_active = True
        base.durable(args.out / 'owner-plan.json', plan_raw, exclusive=True, raw=True)
        resources = resources or base.load_resources(numeric)
        phase = 'prepare'
        probe('before-prepare')
        full_prepare(base, plan, numeric, args.out / 'preflight', check)
        base.durable(args.out / 'formal-specs.json', specs, exclusive=True)
        probe('after-prepare')
        for phase, spec in zip(phases, specs):
            folder = args.out / 'phases' / phase; folder.mkdir()
            result = asyncio.run(trial_one(spec, folder, phase))
            if phase in live:
                release(phase)
            entry = next((row for row in children if row['phase'] == phase), None)
            need(entry is not None and entry['release'] is not None
                 and json.loads((folder / 'trial/child.json').read_bytes())['pid'] == entry['pid'],
                 'formal runner did not return the actually owned/reaped server')
            need(result['status'] == 'PASS' and result['count_eligible'] is True
                 and result['performance_qualified'] is False and result['cleanup_complete'] is True
                 and result['server_reaped'] is True and result['cleanup_task_resolved'] is True
                 and result['client_task_resolved'] is True and result['small_pins_rechecked'] is True,
                 'formal trial/cleanup failed; following trial forbidden')
            check(); probe(phase + ':released')
            point = json.loads((folder / 'trial/point.json').read_bytes())
            need(point == dict(**result['point'], count_eligible=True, performance_qualified=False)
                 and point['concurrency'] == spec['concurrency'] and point['repeat'] == spec['repeat']
                 and point['requests'] == 64, 'trial point identity differs from closed result')
            stages.append(dict(phase=phase, status='PASS', result=ref(folder / 'trial/complete.json'),
                point=ref(folder / 'trial/point.json'), numerical_audit=spec['qualification']['audit']))
            # Publish each closed count-eligible point immediately, before next trial.
            base.durable(args.out / 'points.json', dict(schema='E023-native-formal-points-v1', trials=stages,
                performance_qualified=False, scope='Independent resource/restoration audit remains required.'))
        need([row['phase'] for row in children] == list(phases)
             and [row['phase'] for row in stages] == list(phases), 'complete registered trial cycle required')
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
        no_popen = args.out / 'phases/r1-c1/no-popen.json'
        only_rejected_inner = (no_popen.exists() and deep_spawn == [args.out / 'phases/r1-c1/trial/spawn-intent.json'])
        if not intent and (not deep_spawn or only_rejected_inner):
            need(not children and not live and errors, 'no-child requires pre-intent failure')
            base.durable(args.out / 'terminal.json', dict(schema='cold-service-owner-terminal-v1',
                status='NO_CHILD_STARTED', **bound(), terminal_at=time.time(), phase=errors[0]['phase'],
                error=errors[0], attempt_sha256=sha(args.out / 'attempt.json'),
                spawn_intent_written=False, child_started=False,
                no_popen_receipt=ref(no_popen) if only_rejected_inner else None), exclusive=True)
        elif intent:
            intended = [stage for stage in phases if (args.out / 'phases' / stage / 'intent.json').exists()]
            resolved = [row['phase'] for row in children if row['release'] is not None]
            extra_inner = [stage for stage in phases if stage not in resolved
                           and (args.out / 'phases' / stage / 'trial/spawn-intent.json').exists()]
            next_proof = (args.out / 'phases' / phases[len(resolved)] / 'no-popen.json') if len(resolved) < len(phases) else None
            known_rejection = (extra_inner == list(phases[len(resolved):len(resolved)+1])
                               and next_proof is not None and next_proof.exists())
            if (not live and intended == resolved and resolved == list(phases[:len(resolved)])
                    and (not extra_inner or known_rejection)):
                base.durable(args.out / 'release.json', dict(schema='cold-service-owner-release-v2',
                    **bound(), attempt_sha256=sha(args.out / 'attempt.json'), children=children,
                    children_sha256=sha(args.out / 'children.json'), all_children_reaped=True, time=time.time(),
                    **({'no_popen_receipt': ref(next_proof)} if known_rejection else {})), exclusive=True)
        else:
            failure(RuntimeError('nested spawn evidence without global intent; no no-child certificate'), 'terminal')
        result = dict(schema='E023-native-formal-owner-result-v1', status='FAIL' if errors else 'PASS',
            owner_pid=os.getpid(), owner_plan_sha256=args.plan_sha256, errors=errors, absence_probes=absence_count,
            first_failure=errors[0] if errors else None, phases=stages, samples=len(samples),
            spawn_intent_written=intent,
            child_started=True if children else False if (args.out / 'terminal.json').exists() else None,
            all_children_reaped=(args.out / 'release.json').exists(), elapsed_wall_s=time.monotonic() - started,
            formal_trials_complete=not errors and len(stages) == len(phases),
            planned_trials=list(phases),
            independently_qualified=False, performance_qualified=False,
            scope='Observer-OFF formal trials with captured-history numerical/count bindings; independent resource/restoration audit required.')
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
        base.durable(args.out / 'prepared.json', dict(schema='E023-native-formal-owner-prepare-v1', status='PREPARED',
            owner_plan_sha256=args.plan_sha256, formal_specs=specs, sources=plan['execution']['pins'],
            runtime_weights_verified=False, scope='CPU small source/metadata only; no service or model action'), exclusive=True)
        return 0
    return run(args)


if __name__ == '__main__':
    raise SystemExit(main())
