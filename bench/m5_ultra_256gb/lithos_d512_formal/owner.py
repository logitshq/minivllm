"""Six fixed fresh Lithos D512 formal trials; direct owned server per phase.

Narrow adaptation of frozen native_formal_owner.py, with its durable intents,
settled async probes and TERM/KILL/wait protocol. No model/service action on
import/help/prepare-only. Actual execution requires a separately reviewed cold
controller and exact STOPPED journal. No controller is implemented here.
Full supplied Lithos/runtime/target/pack/outer-runtime pins are verified before
and after; no native preparation receipt is substituted for those dependencies.
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
import sys
import time

HERE = Path(__file__).resolve().parent
PHASES = ('r1-c1', 'r1-c8', 'r2-c8', 'r2-c1', 'r3-c1', 'r3-c8')
TRIALS = ((1,1),(1,8),(2,8),(2,1),(3,1),(3,8))
JOURNAL_SCHEMA = 'E023-lithos-D512-formal-cold-journal-v1'
PREPARE_SECONDS = 300
POSTFLIGHT_SECONDS = 120
NUMERICAL_SHA = '21ba52e43ce54c2b463ab46bfd212c53d06d6303530c77939e3e6b801df547fb'
E020_PLAN = HERE.parent / 'external_spec_http/prereg-v1/plan.json'
E020_SHA = 'e3f0c3e24d4ea841d5e6c0b0094652e65e463c02367836e765ef6d9e3c6e2bdc'
LITHOS_ROOT = Path('/Users/ryanli/code/lithos-metal')
GROUPS = ('lithos_runtime', 'lithos_source', 'target_model', 'pack', 'outer_runtime')
FIXED = {'formal.py': 'fca180b4296e352e7b75db517ac897fe56d25ceea313d31b6a6c091a515c73ff', 'contract.py': '2c5b221608e8c53b39116b2e13b247b9fb78d83c74cbee069179efbea9210b32', '../lithos_d512_retry/owner.py': '97b0f596b8369e36b71fef838f4e290ae4f63b6a02cb47f13e01aa9e14c8f2af', '../lithos_d512_retry/audit_qualification.py': 'a932583f5a5234b0aff434d9b967a004fcdea05ff9b1c03a1d5d34fe86f6df08', '../lithos_d512_retry/audit_resources.py': 'e6468ac9bcc90fc2d4f3fcdfc55c5292e042a479905e38de9a85b5d5a0ba093f', '../lithos_d512_retry/audit_restore.py': '002cb1839b00a0d617404801dcb08cca873deb048da6c39bc726c67f1ef8aa33', '../lithos_d512_retry/trial.py': '32d1bdca8236c70129ba5113bfadbad377c8c925e40a925bafd027e8ecff5ebe', '../lithos_d512_retry/serve_probe.py': 'c6f4915a83d15cf5c527d71e73d4e54683c0de3881de4c2352b9e0d0abc14703', '../lithos_d512_retry/join.py': '2d4dfa12c9672a7a9deff7a962ea18dd671f53c4e7010eed3589fa4a0826cd0d', '../pareto_realistic/lithos_client.py': '13a629f096434bc6c422803bf7bfd8992e24396cb122880b493e893cc5ecd2e9', '../pareto_realistic/client.py': 'ef64a114f4f7d90ec5044d6bdcf058ca05f98dea445d6639db706a4fc75217dc', '../pareto_realistic/http_capture_v2.py': '536fe41d6c8e2bdffe83ba6f43fba87dd6265deb4c84f6204bc9c7220f8280e3', '../pareto_realistic/qualify_absent_owner.py': '5beb6c9bcfa4e22041a386cdbea3f5ecee210cdcb9b345512c50b6c6f873c62d', '../resource_base.py': '41e23aed50dca8782bc1e182237f2dfe63bf6a2b9bf8061932ba2df1cdf2bf26', '../resources.py': '5a743d8ba5e0f1ab211a044fb1a1006b940163bb62f652e93dd849afe05da200', '../resources_v2.py': '15a207240436e92500c76b2ae791ba896540e53cd52c1f9c6237b83976b072d0'}


def need(ok, message):
    if not ok: raise RuntimeError(message)


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(name):
    relative = 'formal.py' if name == 'trial' else '../pareto_realistic/qualify_absent_owner.py'
    need(name in ('trial', 'qualify_absent_owner'), 'only fixed stdlib modules')
    path = (HERE / relative).resolve()
    need(sha(path) == FIXED[relative], 'frozen source changed: ' + str(path))
    spec = importlib.util.spec_from_file_location('_lithos_owner_' + name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def source_pins():
    return {**{str((HERE / n).resolve()): digest for n, digest in FIXED.items()},
            str(Path(__file__).resolve()): sha(__file__)}


def trial_names(trials):
    need(trials == list(PHASES) and type(trials) is list, 'exact six alternating Lithos formal phases')
    return trials


def work_seconds(trials, deadlines):
    trial_names(trials)
    values = [deadlines[k] for k in ('overall', 'close', 'term', 'kill')]
    need(all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in values), 'finite trial budgets')
    return math.ceil(PREPARE_SECONDS + len(trials) * (sum(values) + 12))


def plan_template(qualification, qualifications, cold_restart, http, verification):
    return dict(schema='E023-lithos-D512-formal-owner-plan-v1', execution=dict(
        owner=dict(path=str(Path(__file__).resolve()), sha256=sha(__file__)), pins=source_pins()),
        qualification=qualification, qualifications=qualifications, cold_restart=cold_restart, http=http, verification=verification,
        trials=list(PHASES), deadline_seconds=work_seconds(list(PHASES), http['deadlines']))


def _reviewed_owner():
    path=HERE.parent/'lithos_d512_retry/owner.py'
    need(sha(path)==FIXED['../lithos_d512_retry/owner.py'],'frozen helper source changed')
    spec=importlib.util.spec_from_file_location('_lithos_formal_frozen_owner_helpers',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


_REUSED=_reviewed_owner()


prior_replacement = _REUSED.prior_replacement


def verification_contract(verification, numeric, specs):
    need(len(specs)==6 and all(s['pack_dir']==specs[0]['pack_dir'] for s in specs), 'one exact target pack across all six trials')
    return _REUSED.verification_contract(verification,numeric,specs)


def qualified_verification(verification, bindings):
    need(type(bindings) is dict and set(bindings)=={'1','8'} and bindings['1']==bindings['8'],
         'one independent C1/C8 qualification audit package')
    ref=bindings['1']['owner_plan'];path=Path(ref['path'])
    need(path.is_absolute() and sha(path)==ref['sha256'],'qualification owner plan source')
    qualified=json.loads(path.read_bytes())
    need(verification==qualified['verification'],
         'actual runtime/source/model/pack groups must match qualification')


def validate_plan(args):
    base = load('qualify_absent_owner'); raw = base.absolute(args.plan).read_bytes()
    need(base.sha_bytes(raw) == args.plan_sha256, 'Lithos owner plan SHA')
    plan = json.loads(raw)
    need(set(plan) == {'schema','execution','qualification','qualifications','cold_restart','http','verification','trials','deadline_seconds'}
         and plan['schema'] == 'E023-lithos-D512-formal-owner-plan-v1', 'Lithos owner plan schema')
    trial_names(plan['trials'])
    need(type(plan['deadline_seconds']) is int and plan['deadline_seconds'] == work_seconds(plan['trials'],plan['http']['deadlines']), 'exact work budget')
    need(plan['execution'] == dict(owner=dict(path=str(Path(__file__).resolve()),sha256=sha(__file__)),pins=source_pins()), 'exact owner/source pins')
    for path,digest in plan['execution']['pins'].items(): need(sha(path) == digest, 'source changed: ' + path)
    q = plan['qualification']
    need(set(q) == {'path','sha256'} and q['sha256'] == NUMERICAL_SHA and sha(base.absolute(q['path'])) == NUMERICAL_SHA, 'unchanged D fixture/resource plan')
    numeric = json.loads(Path(q['path']).read_bytes()); cold = plan['cold_restart']
    need(set(cold) == {'journal_path','controller','approved_contract','proposal_plan_sha256','predecessor'}, 'cold contract fields')
    base.absolute(cold['journal_path']); controller = cold['controller']
    need(set(controller) == {'path','sha256'} and base.absolute(controller['path']) == HERE / 'controller.py'
         and sha(controller['path']) == controller['sha256'], 'actual separate Lithos controller required')
    need(cold['approved_contract'] == base.CONTRACT and cold['proposal_plan_sha256'] == base.PROPOSAL_SHA, 'unchanged authorized contract')
    predecessor = cold['predecessor']
    need(set(predecessor) == {'path','sha256','audit'} and set(predecessor['audit']) == {'path','sha256'}, 'immutable restored predecessor')
    for item in (predecessor,predecessor['audit']): need(sha(base.absolute(item['path'])) == item['sha256'], 'predecessor changed')
    prior_replacement(plan)
    http = plan['http']; need(set(http) == {'port','api_key','deadlines','environment'}, 'Lithos HTTP fields')
    d = http['deadlines']
    need(0 < d.get('overall',0) <= 3600 and d.get('term') == d.get('kill') == 15
         and 0 < d.get('close',0) <= 15 and 0 < d.get('resource_interval',0) <= 5, 'original bounded cleanup/resource policy')
    bindings=plan['qualifications']
    qualified_verification(plan['verification'],bindings)
    trial = load('trial')
    specs = [trial.prepare(args.out / 'phases' / phase / 'trial', c, repeat, http['port'], http['api_key'],
             qualification=bindings[str(c)],deadlines=d,environment=http['environment'])
             for phase,(repeat,c) in zip(PHASES,TRIALS)]
    verification_contract(plan['verification'], numeric, specs)
    return base,plan,raw,numeric,specs


def stopped(base, plan, args, binding=None):
    raw = (Path(plan['cold_restart']['journal_path']) / 'journal.json').read_bytes(); j = json.loads(raw)
    expected = dict(schema=JOURNAL_SCHEMA,state='STOPPED',plan_sha256=base.PROPOSAL_SHA,
        controller_sha256=plan['cold_restart']['controller']['sha256'],approved_contract=base.CONTRACT,
        owner_directory=str(args.out),owner_plan=dict(path=str(args.plan),sha256=args.plan_sha256),
        owner_source=plan['execution']['owner'],predecessor=plan['cold_restart']['predecessor'],trials=plan['trials'])
    need(all(j.get(k) == v for k,v in expected.items()), 'bound Lithos cold journal is not STOPPED')
    need(type(j.get('original')) is dict and type(j['original'].get('pid')) is int and j['original']['pid'] > 1, 'original identity')
    need(j['original'] == (prior_replacement(plan) if binding is None else binding['original']), 'predecessor replacement identity')
    need(type(j.get('id')) is str and bool(j['id']) and type(j.get('removed_at')) in (int,float)
         and math.isfinite(j['removed_at']) and 0 < j['removed_at'] <= time.time(), 'journal runtime identity')
    need(type(j.get('predecessor_copies')) is dict, 'retained predecessor copies')
    stable = {k:j[k] for k in (*expected,'id','removed_at','original','predecessor_copies')}
    need(binding is None or binding == stable, 'STOPPED identity changed'); return raw,stable


full_prepare = _REUSED.full_prepare


System = _REUSED.System


def run(args, *, system=None, resources=None, trial_module=None):
    base, plan, plan_raw, numeric, specs = validate_plan(args)
    system = system or System(base)
    trial_module = trial_module or load('trial')
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
            raise TimeoutError('Lithos owner absolute deadline or cancellation')

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
            # Frozen Lithos trial writes its local intent immediately before invoking
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
                f"-iTCP:{plan['http']['port']}", '-sTCP:LISTEN', '-Fpn'])
            base.durable(folder / f'port-absence-{serial:02d}.json', raw_port, exclusive=True)
            need(raw_port['returncode'] == 1 and not raw_port['stdout'].strip(), 'Lithos listener survives')
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
        underlying = system.trial_backend(trial_module)
        class Tracked:
            def __getattr__(self, name):
                return getattr(underlying, name)
            def spawn(self, command, environment, stdout, stderr, *, cwd):
                return spawn(stage, lambda: underlying.spawn(command, environment, stdout, stderr, cwd=cwd))
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
        trial_task = asyncio.create_task(trial_module.run(spec, folder / 'trial',
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
        base.durable(args.out / 'trial-specs.json', specs, exclusive=True)
        probe('after-prepare')
        for phase, spec in zip(phases, specs):
            folder = args.out / 'phases' / phase; folder.mkdir()
            result = asyncio.run(trial_one(spec, folder, phase))
            if phase in live:
                release(phase)
            entry = next((row for row in children if row['phase'] == phase), None)
            need(entry is not None and entry['release'] is not None
                 and json.loads((folder / 'trial/child.json').read_bytes())['pid'] == entry['pid'],
                 'Lithos trial did not return the actually owned/reaped server')
            need(result['schema'] == 'E023-lithos-D512-formal-result-v1'
                 and result['concurrency'] == spec['concurrency'] and result['repeat']==spec['repeat']
                 and result['status'] == 'PASS' and result['count_eligible'] is True
                 and result['full_state_qualified'] is False
                 and result['performance_qualified'] is False and result['cleanup_complete'] is True
                 and result['server_reaped'] is True and result['cleanup_task_resolved'] is True
                 and result['client_task_resolved'] is True and result['small_pins_rechecked'] is True
                 and result['logs_rechecked'] is True,
                 'Lithos formal trial/cleanup failed; following trial forbidden')
            inner_release = json.loads((folder / 'trial/release.json').read_bytes())
            need(inner_release['pid'] == entry['pid'] and inner_release['waited_reaped'] is True
                 and inner_release['strict_absence'] is True and inner_release['port_absent'] is True,
                 'trial release does not match direct server')
            check(); probe(phase + ':released')
            need(json.loads((folder/'trial/point.json').read_bytes()) == dict(**result['point'],
                 count_eligible=True,performance_qualified=False),'retained trial point differs from result')
            stages.append(dict(phase=phase,status='PASS',concurrency=spec['concurrency'],repeat=spec['repeat'],
                result=ref(folder/'trial/complete.json'),point=ref(folder/'trial/point.json')))
            base.durable(args.out/'points.json',dict(schema='E023-lithos-D512-formal-cycle-v1',
                trials=stages,performance_qualified=False,full_state_qualified=False,
                scope='Per-trial count-eligible points only; independent full cycle/raw/resource/restoration audit required.'))
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
        result = dict(schema='E023-lithos-D512-formal-owner-result-v1', status='FAIL' if errors else 'PASS',
            owner_pid=os.getpid(), owner_plan_sha256=args.plan_sha256, errors=errors, absence_probes=absence_count,
            first_failure=errors[0] if errors else None, phases=stages, samples=len(samples),
            spawn_intent_written=intent,
            child_started=True if children else False if (args.out / 'terminal.json').exists() else None,
            all_children_reaped=(args.out / 'release.json').exists(), elapsed_wall_s=time.monotonic() - started,
            formal_trials_complete=not errors and len(stages) == len(phases),
            planned_trials=list(phases),
            independently_qualified=False, performance_qualified=False,
            scope='Six independent observer-OFF trials; count eligibility is separate from final performance qualification; independent raw/resource/restoration audit required.')
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
        base.durable(args.out / 'prepared.json', dict(schema='E023-lithos-D512-formal-owner-prepare-v1', status='PREPARED',
            owner_plan_sha256=args.plan_sha256, trial_specs=specs, sources=plan['execution']['pins'],
            runtime_weights_pack_verified=False, scope='CPU small source/metadata only; no service or model action'), exclusive=True)
        return 0
    return run(args)


if __name__ == '__main__':
    raise SystemExit(main())
