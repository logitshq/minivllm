"""Twelve paired OFF/ON direct native servers; source-only until registered.

The fixed donor's run body retains actual Popen ownership and cancellation/reap.
Only phase/spec/result bindings and the first-phase receipt path differ. No
imported globals or source paths are rewritten. Native157 pre/post preparation
and model hashing remain; extra adapter copies use absolute-path-relative names
because original/new formal.py share a basename. Each closed point is retained;
performance qualification belongs to independent resource/restoration auditing.
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

HERE=Path(__file__).resolve().parent
NATIVE=HERE.parent/'pareto_realistic'
DONOR_PATH=NATIVE/'native_formal_owner.py'
DONOR_SHA='e33faa8f211e905f40daabd00941f3fd016041c9705c6a7a4a2fe805fb0aceff'
FORMAL_SHA='e3fd332884975426487853108098933283e4b98705671cfe8dbccd37f91bfb7a'
NUMERICAL_SHA='21ba52e43ce54c2b463ab46bfd212c53d06d6303530c77939e3e6b801df547fb'
PHASES=('r1-c1-off','r1-c1-on','r1-c8-on','r1-c8-off',
        'r2-c8-off','r2-c8-on','r2-c1-on','r2-c1-off',
        'r3-c1-off','r3-c1-on','r3-c8-on','r3-c8-off')
PREPARE_SECONDS=300
POSTFLIGHT_SECONDS=120
DEADLINES=dict(ready=120,request=600,warm=720,formal=2400,close=10,
               term=15,kill=15,overall=3300,resource_interval=5)
TRANSPORT=dict(port=18023,tcp_nodelay=True,io_timeout_seconds=300.0)
REQUIRED_AUDIT_SOURCES=('audit_formal.py','audit_restore.py')
SUPPORT={
    '../route_http_formal/audit_formal_v2.py':'b2bc55b87043d66d79154fd32dc7579cc7c6fdb457844a1b1d3df932ff41224e',
    '../route_http_formal/audit_formal.py':'ed8ad64c3eb5fed052effd05981a386006f4ea1f752463bb0ada0c2f0e4ea5e0',
    '../omlx_d512_formal_retry/controller.py':'8023d0b114c02e7a25cf8bfc305ef0986db3d7086f84cefd6f9c50436a5ded30',
    '../cold_cycle/controller.py':'b5237c458afa8324ffa6d5e6a5c49167578daec7425ff036112ec8e108d61213',
    '../cold_cycle/lifecycle.json':'d9417e51a781275f18c296e20ac98874e61b8ce17b8bf0902ba63e27b5b2b0a8',
    '../omlx_cold_restart/controller_v4.py':'836a4c16534521441da8021df3f963210d885da867f79a33d86a3b6dd01101c0',
    '../omlx_cold_restart/controller.py':'aad297fb08515d6304fd3ae435301318650e8443ad22628c1ed2e4791887f0f3',
    '../omlx_cold_restart/audit_restore_v4.py':'af46699fcea1065c4fcf69a85225a916987fba484098d53ef2acb905eb4ae3a8',
    '../gdn_prework_http_cycle/controller.py':'04200ccde66bb6525f274282bd65338e222855201f38296bb46144abfc5372c5',
    '../gdn_prework_http_cycle/audit_restore.py':'40d07aabef5d3588b4eac57a2acadf4278eee2328a3304da2a041a234f66ff5e',
    '../gdn_prework_http_cycle/owner.py':'a4776f1125b9b8db82d83b0f2cebd78c57cf15894354d884d0e2cd07224a7ddd',
    '../cold_cycle_v3/controller.py':'674b21fbcafa0e79164dedb78875f513e9a24d478337c2aeeb67a439e69a4dc9',
    '../cold_cycle_v3/owner.py':'4c2cc6f305de3ecf0c6495e45838baae885efbf7a8f479ed2043dc0ffb67f31c',
    '../cold_cycle_v3/audit_restore.py':'34d608b54ed5779b97e85a0463c17f8796876be731f6f0aafcbe804ffc7b15ca',
    '../cold_cycle_v3/lifecycle.json':'c7a6afdcc7807a2b1e5629df628c716ff77278fe3d0bf8562715b8794c4e8dca',
    '../pareto_realistic/audit_absent_qualification.py':'1b7b526dac98c43f5b77f915aec3173d755d1efb28ed49b989e7601293ddaf6d',
    '../pareto_realistic/audit_native_formal.py':'b57e4e9657f3dddb04069290bdfbab4c99b0d6cbfb1427e454d0217d5463c870',
    '../lithos_d512_formal/audit_resources.py':'398ff7a52dba307418f4c2725f33988b1571d65b11e3f46c9e97759f3a167c9e',
}
PLAN_SCHEMA='GDN-prework-http-formal-owner-plan-v1'
JOURNAL_SCHEMA='GDN-prework-http-formal-cold-journal-v1'


def need(ok,message):
    if not ok: raise RuntimeError(message)


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def module(path,digest,name):
    need(sha(path)==digest,'frozen source changed: '+str(path))
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result


def donor(): return module(DONOR_PATH,DONOR_SHA,'gdn_formal_native_owner')


def load(name):
    if name=='formal':return module(HERE/'formal.py',FORMAL_SHA,'gdn_formal_runner')
    return donor().load(name)


def source_pins():
    pins={**donor().source_pins(),**load('formal').source_pins(),
          str(DONOR_PATH):DONOR_SHA,str(Path(__file__).resolve()):sha(__file__)}
    for name,digest in SUPPORT.items():
        path=(HERE/name).resolve();need(sha(path)==digest,'fixed support source changed: '+name);pins[str(path)]=digest
    for name in ('controller.py',*REQUIRED_AUDIT_SOURCES):
        path=HERE/name
        need(path.is_file() and not path.is_symlink(),'required actual source absent: '+name)
        pins[str(path)]=sha(path)
    return pins


def trial_names(value):
    need(type(value) is list and value==list(PHASES),'exact twelve alternating OFF/ON trials required')
    return value


def work_seconds(trials,deadlines):
    trial_names(trials)
    need(json.dumps(deadlines,sort_keys=True)==json.dumps(DEADLINES,sort_keys=True),'unchanged prepared deadlines')
    return PREPARE_SECONDS+len(trials)*(sum(deadlines[k] for k in ('overall','close','term','kill'))+12)


def plan_template(qualification,cold_restart,http,qualifications,gdn_qualifications,prepared_specs):
    return dict(schema=PLAN_SCHEMA,execution=dict(owner=dict(path=str(HERE/'owner.py'),sha256=sha(__file__)),pins=source_pins()),
        qualification=qualification,qualifications=qualifications,gdn_qualifications=gdn_qualifications,
        prepared_specs=prepared_specs,cold_restart=cold_restart,http=http,trials=list(PHASES),
        deadline_seconds=work_seconds(list(PHASES),http['deadlines']))


def prior_replacement(plan):
    ref=plan['cold_restart']['controller']
    c=module(Path(ref['path']),ref['sha256'],'gdn_formal_bound_controller')
    prior,_=c.predecessor(plan['cold_restart']['predecessor']);return prior['replacement']


def validate_plan(args):
    base=load('qualify_absent_owner');raw=base.absolute(args.plan).read_bytes()
    need(base.sha_bytes(raw)==args.plan_sha256,'GDN formal plan SHA')
    plan=json.loads(raw)
    need(set(plan)=={'schema','execution','qualification','qualifications','gdn_qualifications','prepared_specs',
                    'cold_restart','http','trials','deadline_seconds'} and plan['schema']==PLAN_SCHEMA,'exact GDN formal plan')
    trial_names(plan['trials'])
    need(type(plan['deadline_seconds']) is int and plan['deadline_seconds']==work_seconds(plan['trials'],plan['http']['deadlines']),
         'exact twelve-trial hard work bound')
    need(plan['execution']==dict(owner=dict(path=str(HERE/'owner.py'),sha256=sha(__file__)),pins=source_pins()),'actual owner/dependency pins')
    for p,h in plan['execution']['pins'].items():need(sha(p)==h,'owner dependency changed: '+p)
    q=plan['qualification']
    need(set(q)=={'path','sha256'} and q['sha256']==NUMERICAL_SHA and sha(base.absolute(q['path']))==NUMERICAL_SHA,'original numerical plan')
    numeric=json.loads(Path(q['path']).read_bytes());cold=plan['cold_restart']
    need(set(cold)=={'journal_path','controller','approved_contract','proposal_plan_sha256','predecessor'},'exact cold binding')
    base.absolute(cold['journal_path']);c=cold['controller']
    need(set(c)=={'path','sha256'} and base.absolute(c['path'])==HERE/'controller.py' and sha(c['path'])==c['sha256'],'actual adjacent controller')
    need(cold['approved_contract']==base.CONTRACT and cold['proposal_plan_sha256']==base.PROPOSAL_SHA,'unchanged approved contract')
    prior_replacement(plan)
    need(set(plan['http'])=={'transport','deadlines','environment'},'fixed HTTP configuration')
    load('formal').same(plan['http']['transport'],TRANSPORT,'unchanged prepared transport')
    need(set(plan['qualifications'])==set(plan['gdn_qualifications'])=={'1','8'},'both independent C1/C8 prerequisite bindings')
    need(set(plan['prepared_specs'])==set(PHASES),'all twelve original prepared spec references')
    formal=load('formal');specs=[]
    for phase in PHASES:
        r,c,arm=phase.split('-');C=int(c[1:]);repeat=int(r[1:])
        spec=formal.prepare(q,C,repeat,arm=arm,baseline_qualification=plan['qualifications'][str(C)],
            gdn_qualification=plan['gdn_qualifications'][str(C)],**plan['http'])
        ref=plan['prepared_specs'][phase]
        need(set(ref)=={'path','sha256'} and sha(base.absolute(ref['path']))==ref['sha256'],'prepared spec file SHA')
        formal.same(spec,json.loads(Path(ref['path']).read_bytes()),'actual prepared spec changed: '+phase)
        need(spec['phase']==phase and spec['arm']==arm and spec['observer_enabled'] is False,'actual arm/phase observer binding')
        specs.append(spec)
    return base,plan,raw,numeric,specs


def stopped(base,plan,args,binding=None):
    raw=(Path(plan['cold_restart']['journal_path'])/'journal.json').read_bytes();j=json.loads(raw)
    expected=dict(schema=JOURNAL_SCHEMA,state='STOPPED',plan_sha256=base.PROPOSAL_SHA,
        controller_sha256=plan['cold_restart']['controller']['sha256'],approved_contract=base.CONTRACT,
        owner_directory=str(args.out),owner_plan=dict(path=str(args.plan),sha256=args.plan_sha256),
        owner_source=plan['execution']['owner'],predecessor=plan['cold_restart']['predecessor'],trials=plan['trials'])
    need(all(j.get(k)==v for k,v in expected.items()),'bound GDN formal journal is not STOPPED')
    need(j['original']==(prior_replacement(plan) if binding is None else binding['original']),'predecessor replacement changed')
    need(type(j.get('id')) is str and j['id'] and type(j.get('removed_at')) in (int,float)
         and math.isfinite(j['removed_at']) and 0<j['removed_at']<=time.time(),'journal identity/time')
    need(type(j.get('predecessor_copies')) is dict,'retained predecessor copies required')
    stable={k:j[k] for k in (*expected,'id','removed_at','original','predecessor_copies')}
    need(binding is None or stable==binding,'STOPPED stable binding changed');return raw,stable


def full_prepare(base,plan,numeric,out,check):
    """Original157 native pins + actual model hashes; distinct adapter copy names."""
    out.mkdir();q=plan['qualification']
    base.load_prepare()(SimpleNamespace(out=out,plan=Path(q['path']),plan_sha256=q['sha256'],deadline_seconds=5700),check)
    manifest=json.loads(Path(numeric['model_manifest']['path']).read_bytes());pins={}
    for relative,expected in manifest['files'].items():
        need(not Path(relative).is_absolute() and '..' not in Path(relative).parts,'model manifest traversal')
        path=Path(numeric['model'])/relative;digest=hashlib.sha256();before=path.stat()
        with path.open('rb') as stream:
            while chunk:=stream.read(8<<20):check();digest.update(chunk)
        after=path.stat()
        need((before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns)
             and digest.hexdigest()==expected,'model pin changed: '+relative)
        pins[str(path)]=dict(sha256=expected,bytes=after.st_size,inode=after.st_ino,mtime_ns=after.st_mtime_ns)
        base.durable(out/'verified-model-pins.json',pins)
    adapters={}
    for source,digest in plan['execution']['pins'].items():
        check();path=Path(source);need(path.is_absolute() and path==path.resolve(),'absolute adapter source')
        raw=path.read_bytes();need(base.sha_bytes(raw)==digest,'adapter source changed')
        target=out/'frozen-adapters'/path.relative_to('/');target.parent.mkdir(parents=True,exist_ok=True)
        base.durable(target,raw,exclusive=True,raw=True)
        adapters[source]=dict(sha256=digest,bytes=len(raw),copy=str(target.relative_to(out)))
    base.durable(out/'adapter-pins.json',adapters,exclusive=True)


class System:
    def __init__(self,base):self.delegate=donor().System(base)
    def __getattr__(self,name):return getattr(self.delegate,name)


def run(args, *, system=None, resources=None, formal_module=None):
    base, plan, plan_raw, numeric, specs = validate_plan(args)
    system = system or System(base)
    formal_module = formal_module or load('formal')
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
            inner_release=json.loads((folder/'trial/release.json').read_bytes())
            outer_release=json.loads((args.out/entry['release']['path']).read_bytes())
            need(inner_release['pid']==outer_release['pid'] and inner_release['returncode']==outer_release['returncode']
                 and inner_release['waited_reaped'] is True and inner_release['strict_absence'] is True
                 and inner_release['port_absent'] is True,'inner/outer server release mismatch')
            need(result['schema']=='GDN-prework-http-formal-result-v1' and result['phase']==phase
                 and result['arm']==spec['arm'] and result['observer_enabled'] is False
                 and result['status'] == 'PASS' and result['count_eligible'] is True
                 and result['performance_qualified'] is False and result['cleanup_complete'] is True
                 and result['server_reaped'] is True and result['cleanup_task_resolved'] is True
                 and result['client_task_resolved'] is True and result['small_pins_rechecked'] is True,
                 'formal trial/cleanup failed; following trial forbidden')
            check(); probe(phase + ':released')
            point = json.loads((folder / 'trial/point.json').read_bytes())
            need(point == dict(**result['point'], count_eligible=True, performance_qualified=False)
                 and point['concurrency'] == spec['concurrency'] and point['repeat'] == spec['repeat']
                 and point['requests'] == 64, 'trial point identity differs from closed result')
            stages.append(dict(phase=phase, arm=spec['arm'], status='PASS', result=ref(folder / 'trial/complete.json'),
                point=ref(folder / 'trial/point.json'), numerical_audit=spec['qualification']['audit']))
            # Publish each closed count-eligible point immediately, before next trial.
            base.durable(args.out / 'points.json', dict(schema='GDN-prework-http-formal-points-v1', trials=stages,
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
        no_popen = args.out / 'phases' / phases[0] / 'no-popen.json'
        only_rejected_inner = (no_popen.exists() and deep_spawn == [args.out / 'phases' / phases[0] / 'trial/spawn-intent.json'])
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
            if (not live and resolved and set(resolved) == phase_intent_started
                    and intended == resolved and resolved == list(phases[:len(resolved)])
                    and (not extra_inner or known_rejection)):
                base.durable(args.out / 'release.json', dict(schema='cold-service-owner-release-v2',
                    **bound(), attempt_sha256=sha(args.out / 'attempt.json'), children=children,
                    children_sha256=sha(args.out / 'children.json'), all_children_reaped=True, time=time.time(),
                    **({'no_popen_receipt': ref(next_proof)} if known_rejection else {})), exclusive=True)
        else:
            failure(RuntimeError('nested spawn evidence without global intent; no no-child certificate'), 'terminal')
        result = dict(schema='GDN-prework-http-formal-owner-result-v1', status='FAIL' if errors else 'PASS',
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
        base.durable(args.out / 'prepared.json', dict(schema='GDN-prework-http-formal-owner-prepare-v1', status='PREPARED',
            owner_plan_sha256=args.plan_sha256, formal_specs=specs, sources=plan['execution']['pins'],
            runtime_weights_verified=False, scope='CPU small source/metadata only; no service or model action'), exclusive=True)
        return 0
    return run(args)


if __name__=='__main__':raise SystemExit(main())
