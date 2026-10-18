"""One direct official GLM load-probe child; no service mutations.

Prepared source only. A separately reviewed adjacent controller is required;
missing controller/registration fails closed. No Qwen preflight is executed.
Empty argv describes; --run with a complete plan/sha/out explicitly executes.
The external driver must always attempt controller recovery in finally. This
owner reports observed diagnostics, never physical fit, full correctness or TPS.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PHASE = 'glm-official-load'
SCHEMA = 'GLM-cold-load-owner-plan-v1'
CONTRACT = 'cold-restart-persisted-visible-v1'
PROPOSAL = 'e96a4a4d5be6f6a69d597387bbc566eadedae81065bfa879445352d986515cb8'
JOB = 'gui/501/local.omlx.server'
FIXED = {
    '../glm_flash/load_probe.py': 'cd9df050384af83d92f0321d95f5bba70dca349defbedf0bb7d1d1e8b57385fc',
    '../glm_flash/physical_guard.py': '908e0dc74812fd82b65c63a22e03ed8c5692e812bdbf3d6957e219aa52b14166',
    '../pareto_realistic/qualify_absent_owner.py': '5beb6c9bcfa4e22041a386cdbea3f5ecee210cdcb9b345512c50b6c6f873c62d',
}
NATIVE_NAMES = {'omlx-server', 'omlx', 'lithos-metal', 'llama-server', 'mlx_lm.server'}
LEGACY_MATCHES = ('core_http', '/bin/omlx serve', '-m monolith.serve', '-m minivllm.server',
                  '-m mlx_lm.server', '/lithos-metal serve', '/dflash benchmark', '/glm_flash/load_probe.py')


def need(ok, message):
    if not ok: raise ValueError(message)


def number(v): return type(v) in (int, float) and math.isfinite(v)
def read(path): return json.loads(Path(path).read_bytes())


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(1 << 20): value.update(block)
    return value.hexdigest()


def ref(path): return dict(path=str(Path(path).resolve()), sha256=sha(path))


def reference(item):
    need(type(item) is dict and set(item) == {'path', 'sha256'}, 'exact path/SHA reference')
    path = Path(item['path'])
    need(path.is_absolute() and path.is_file() and not path.is_symlink()
         and str(path.resolve()) == str(path) and type(item['sha256']) is str
         and re.fullmatch('[0-9a-f]{64}', item['sha256']) and sha(path) == item['sha256'], 'changed reference: '+str(path))
    return path


def module(path, digest):
    path = Path(path); need(sha(path) == digest, 'reviewed helper changed')
    spec = importlib.util.spec_from_file_location('_glm_owner_'+path.stem, path)
    result = importlib.util.module_from_spec(spec); spec.loader.exec_module(result); return result


def helper(relative): return module((HERE/relative).resolve(), FIXED[relative])


def stamp(path):
    s = Path(path).stat()
    return dict(bytes=s.st_size, mtime_ns=s.st_mtime_ns, inode=s.st_ino, device=s.st_dev)


def file_record(row):
    return (type(row) is dict and set(row) == {'sha256','stat'}
        and type(row['sha256']) is str and re.fullmatch('[0-9a-f]{64}',row['sha256']) is not None
        and type(row['stat']) is dict and set(row['stat']) == {'bytes','mtime_ns','inode','device'}
        and all(type(v) is int and v >= 0 for v in row['stat'].values()))


def source_pins(plan):
    pins = {str((HERE/p).resolve()): h for p, h in FIXED.items()}
    pins[str(Path(__file__).resolve())] = sha(__file__)
    controller = plan['cold_restart']['controller']; pins[controller['path']] = controller['sha256']
    probe = read(reference(plan['probe_plan']))
    for path, digest in probe['source_pins'].items():
        need(path not in pins or pins[path] == digest, 'conflicting probe/helper source')
        pins[path] = digest
    return pins


def verify_inventory(plan, probe, check):
    """Fresh runtime hashes + metadata/shard stat; retained payload hash is explicit.

    The registered file inventory must be created separately. This function does
    not pretend that a new stat proves a new full 204GB payload hash.
    """
    receipt = read(reference(plan['verification']))
    need(set(receipt) == {'schema','snapshot','download_receipt','payload_scope','shards','runtime'}
         and receipt['schema'] == 'GLM-model-runtime-inventory-v1'
         and receipt['snapshot'] == str(probe.SNAPSHOT)
         and receipt['payload_scope'] == 'retained-download-full-SHA256-and-registered-stat;no-new-payload-hash',
         'explicit GLM inventory scope')
    child = read(reference(plan['probe_plan']))
    need(receipt['download_receipt'] == child['download_receipt'], 'same full-download provenance')
    download = read(reference(receipt['download_receipt']))
    manifest = read(probe.HERE/'metadata-v1/manifest.json')
    expected = {r['name']: dict(bytes=r['bytes'], sha256=r['sha256']) for r in manifest['shards']}
    need(manifest['revision'] == probe.REVISION and len(expected) == 62
         and sum(r['bytes'] for r in expected.values()) == 203992076296
         and download['status'] == 'PASS_PINNED_DOWNLOAD_AND_SHA_ONLY'
         and download['revision'] == probe.REVISION and download['snapshot'] == str(probe.SNAPSHOT)
         and {r['name']: dict(bytes=r['bytes'], sha256=r['sha256']) for r in download['shards']} == expected
         and len(download['shards']) == 62 and all(r['sha256_verified'] is True for r in download['shards']),
         'exact 62-shard full-hash provenance')
    need(type(receipt['shards']) is dict and set(receipt['shards']) == set(expected), 'exact shard inventory')
    for name, row in receipt['shards'].items():
        check(); path = probe.SNAPSHOT/name
        need(file_record(row) and row['sha256'] == expected[name]['sha256']
             and row['stat']['bytes'] == expected[name]['bytes'] and stamp(path) == row['stat'], 'changed shard stat: '+name)
    runtime = receipt['runtime']
    need(type(runtime) is dict and bool(runtime), 'explicit runtime inventory')
    required = {str(p.resolve()) for p in probe.REQUIRED_SOURCES if p.is_relative_to(probe.SITE)}
    required.update(str(p.resolve()) for p in (probe.VENV/'bin/python', Path(sys.executable),
        probe.SITE/'mlx/core.cpython-313-darwin.so', probe.SITE/'mlx/lib/libmlx.dylib', probe.SITE/'mlx/lib/libjaccl.dylib'))
    need(required <= set(runtime), 'missing actual interpreter, GLM/vendor or MLX native runtime file')
    for name, row in runtime.items():
        check(); path = Path(name)
        need(path.is_absolute() and str(path.resolve()) == name and file_record(row)
             and stamp(path) == row['stat'] and sha(path) == row['sha256'] and stamp(path) == row['stat'],
             'runtime changed: '+name)
    metadata = probe.model_metadata()
    return dict(schema='GLM-owner-model-runtime-proof-v1', status='PASS', inventory=plan['verification'],
        snapshot=str(probe.SNAPSHOT), payload_scope=receipt['payload_scope'], weight_payload_SHA_rechecked=False,
        shard_count=62, runtime_files=len(runtime), runtime_SHA_rechecked=True, metadata=metadata,
        scope='Registered supplied runtime files and fixed GLM metadata; not a recursive dependency-closure proof.')


def validate_plan(args):
    path = reference(dict(path=str(args.plan), sha256=args.plan_sha256)); plan = read(path)
    need(set(plan) == {'schema','execution','probe_plan','verification','physical_policy','cold_restart','outputs','budgets'}
         and plan['schema'] == SCHEMA, 'exact GLM owner plan')
    need(plan['execution'] == dict(owner=ref(__file__), pins=source_pins(plan)), 'actual owner/probe/source pins')
    for p, h in plan['execution']['pins'].items(): need(sha(p) == h, 'source changed: '+p)
    probe = helper('../glm_flash/load_probe.py'); guard = helper('../glm_flash/physical_guard.py')
    child = read(reference(plan['probe_plan']))
    need(child['schema'] == 'GLM-official-load-plan-v1' and child['snapshot'] == str(probe.SNAPSHOT)
         and child['environment'] == probe.ENVIRONMENT, 'actual GLM probe plan/environment')
    d = child['deadlines']; limits = {'work_seconds':1800,'request_seconds':180,'shutdown_seconds':60}
    need(set(d) == set(limits) and all(number(d[k]) and 0 < d[k] <= limits[k] for k in limits)
         and d['request_seconds'] <= d['work_seconds'], 'finite original probe deadlines')
    probe.source_check(child)
    for k, (filename, digest) in probe.FIXED_RECEIPTS.items():
        need(child[k] == dict(path=str(ROOT/'.artifacts/m5-ultra-256gb/glm-flash-download-v1'/filename),sha256=digest), 'original model receipt')
        reference(child[k])
    raw = reference(child['prompt']).read_bytes(); raw.decode('utf-8', errors='strict')
    need(0 < len(raw) <= 4096 and b'\0' not in raw, 'bounded plain UTF8 prompt')
    policy = plan['physical_policy']
    need(policy == guard.policy(**{k:policy[k] for k in ('child_limit_bytes','work_seconds','interval_seconds','probe_seconds')})
         and policy['work_seconds'] == d['work_seconds'], 'same original physical/child deadline policy')
    need(plan['budgets'] == dict(work_seconds=1830, prepare_seconds=300, postflight_seconds=120,
         term_seconds=15, kill_seconds=15, checkpoint_seconds=5), 'fixed finite owner budgets')
    cold = plan['cold_restart']
    need(set(cold) == {'journal_path','controller','predecessor','approved_contract','proposal_plan_sha256'}
         and cold['approved_contract'] == CONTRACT and cold['proposal_plan_sha256'] == PROPOSAL,
         'same authorized cold-service contract')
    need(reference(cold['controller']) == HERE/'controller.py', 'separately reviewed real GLM controller required')
    controller = module(cold['controller']['path'], cold['controller']['sha256'])
    need(controller.CONTRACT == CONTRACT and controller.PLAN_SHA == PROPOSAL
         and controller.JOURNAL_SCHEMA == 'GLM-cold-load-journal-v1', 'actual controller protocol')
    prior, _ = controller.predecessor(cold['predecessor'])
    need(prior['state'] == 'RESTORED' and type(prior['replacement']['pid']) is int, 'independently bound restored predecessor')
    outputs = plan['outputs']
    need(outputs == dict(owner=str(args.out), probe=str(args.out/'phases'/PHASE/'probe')),
         'one exact direct child output')
    return plan, probe, guard, prior


def stopped(plan, args, prior, binding=None):
    raw = (Path(plan['cold_restart']['journal_path'])/'journal.json').read_bytes(); j = json.loads(raw)
    expected = dict(schema='GLM-cold-load-journal-v1',state='STOPPED',plan_sha256=PROPOSAL,
        approved_contract=CONTRACT,controller_sha256=plan['cold_restart']['controller']['sha256'],
        original=prior['replacement'],owner_directory=str(args.out),owner_source=plan['execution']['owner'],
        owner_plan=dict(path=str(args.plan),sha256=args.plan_sha256),predecessor=plan['cold_restart']['predecessor'],trials=[PHASE])
    need(all(j.get(k) == v for k,v in expected.items()) and type(j.get('id')) is str and bool(j['id'])
         and number(j.get('removed_at')) and 0 < j['removed_at'] <= time.time(), 'actual bound service must remain STOPPED')
    stable = {k:j[k] for k in (*expected,'id','removed_at')}
    need(binding is None or binding == stable, 'STOPPED identity changed/recovery began')
    return raw, stable


class System:
    """OS operations only when explicitly executing an approved future plan."""
    def read(self, argv, timeout=5):
        r = subprocess.run(argv,capture_output=True,text=True,timeout=timeout)
        return dict(command=argv,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)

    def identity(self, pid):
        row = self.read(['/bin/ps','-p',str(pid),'-o','pid=,ppid=,lstart=,comm='])
        m = re.fullmatch(r'\s*(\d+)\s+(\d+)\s+([A-Za-z]{3} [A-Za-z]{3}\s+\d+ \d\d:\d\d:\d\d \d{4})\s+(.+?)\s*',row['stdout'])
        need(row['returncode'] == 0 and not row['stderr'].strip() and m and int(m[1]) == pid,'unknown process identity')
        return dict(pid=int(m[1]),ppid=int(m[2]),start=' '.join(m[3].split()),comm=m[4])

    def spawn(self, command, log, env): return subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env)


def checkpoint(system, *, old_pid, child_pid, owner_pid, deadline, record=None):
    """Retain successful and failed raw checks; redact unrelated process argv."""
    result = dict(schema='GLM-owner-checkpoint-v1',status='FAIL',time=time.time(),
        monotonic=time.monotonic(),old_pid=old_pid,child_pid=child_pid,owner_pid=owner_pid,
        commands=[],recognized=[],authorized_suspended_pids=[],
        scope='Raw native titles; legacy argv matcher receipt only. Unrecognized GPU users remain unknown.')
    def call(argv):
        row = dict(command=argv); result['commands'].append(row)
        try:
            remaining = deadline-time.monotonic(); need(remaining > 0, 'checkpoint aggregate deadline')
            row.update(system.read(argv,timeout=remaining))
        except BaseException as exc:
            row.update(returncode=None,error_type=type(exc).__name__,error=str(exc),
                stdout=getattr(exc,'stdout',None),stderr=getattr(exc,'stderr',None))
            for key in ('stdout','stderr'):
                if isinstance(row[key],bytes): row[key]=row[key].decode(errors='replace')
            raise
        need(type(row.get('stdout')) is type(row.get('stderr')) is str,'raw command text fields')
        return row
    try:
        for argv, rc in ((['/bin/ps','-p',str(old_pid),'-o','pid='],1),
                         (['/bin/launchctl','print',JOB],113),
                         (['/usr/sbin/lsof','-nP','-a','-iTCP:8001','-sTCP:LISTEN','-Fpn'],1)):
            row = call(argv)
            # launchctl's absent-job diagnostic may be on stderr; retain it.
            need(type(row['returncode']) is int and row['returncode'] == rc and not row['stdout'].strip(),
                 'removed service PID/job/port returned')
            if rc == 1: need(not row['stderr'].strip(), 'absence command stderr')
        power = call(['/usr/bin/pmset','-g','batt'])
        thermal = call(['/usr/bin/osascript','-l','JavaScript','-e','ObjC.import("Foundation"); String($.NSProcessInfo.processInfo.thermalState)'])
        process = call(['/bin/ps','-axo','pid=,ppid=,comm='])
        legacy = call(['/bin/ps','-axo','pid=,ppid=,args='])
        need(all(type(r['returncode']) is int and r['returncode'] == 0 and not r['stderr'].strip()
                 for r in (power,thermal,process,legacy)), 'resource metadata command failed')
        need(re.findall(r"Now drawing from '([^']+)'",power['stdout']) == ['AC Power']
             and thermal['stdout'].strip() in ('0','1'), 'power/thermal gate')
        for kind,row in (('native',process),('legacy',legacy)):
            seen = set()
            for line in row['stdout'].splitlines():
                fields = line.split(None,2)
                need(len(fields)==3 and fields[0].isdigit() and fields[1].isdigit(),'process inventory format')
                pid,ppid = int(fields[0]),int(fields[1]); command = fields[2]
                need(pid not in seen,'duplicate process PID'); seen.add(pid)
                match = Path(command).name in NATIVE_NAMES if kind=='native' else any(s in command for s in LEGACY_MATCHES)
                if match or pid == child_pid:
                    need(pid == child_pid and ppid == owner_pid,'foreign recognized inference process')
                    result['recognized'].append(dict(kind=kind,pid=pid,ppid=ppid))
            need(child_pid is None or child_pid in seen,'live child missing from process inventory')
        need(time.monotonic() <= deadline,'checkpoint exceeded aggregate deadline')
        result['status']='PASS'; return result
    except BaseException as exc:
        result['error']=dict(type=type(exc).__name__,message=str(exc)); raise
    finally:
        # Redact on failure too: ps args can contain unrelated API keys.
        for row in result['commands']:
            if row['command'] == ['/bin/ps','-axo','pid=,ppid=,args=']:
                value=row.pop('stdout',None)
                if isinstance(value,str):
                    row['stdout_sha256']=hashlib.sha256(value.encode()).hexdigest()
                    row['stdout_bytes']=len(value.encode())
        result['ended_monotonic']=time.monotonic()
        if record is not None: record(result)


def child_command(plan, probe, admission, output):
    return [str(probe.VENV/'bin/python'),'-I','-B',str(probe.HERE/'load_probe.py'),'--run',
        '--plan',plan['probe_plan']['path'],'--plan-sha256',plan['probe_plan']['sha256'],
        '--admission',str(admission),'--admission-sha256',sha(admission),'--out',str(output)]


def probe_terminal(value, plan, admission_sha, pid, owner_pid, probe):
    need(value.get('schema') == 'GLM-official-load-result-v1' and value.get('status') == 'PASS'
         and value.get('pid') == pid and value.get('owner_pid') == owner_pid
         and value.get('plan_sha256') == plan['probe_plan']['sha256'] and value.get('admission_sha256') == admission_sha
         and value.get('probe_sha256') == FIXED['../glm_flash/load_probe.py']
         and value.get('snapshot') == str(probe.SNAPSHOT) and value.get('versions') == probe.EXPECTED
         and value.get('short_lifecycle_diagnostic_complete') is True and value.get('shutdown',{}).get('complete') is True
         and [r.get('cap') for r in value.get('completed_requests',[])] == [8,32,32,32], 'missing/wrong original GLM completion')
    need(all(value.get(k) is False for k in ('performance_qualified','full_model_correctness_qualified','chat_quality_qualified',
         'physical_resource_safety_qualified')), 'probe scope was broadened')


def run(args, *, system=None):
    plan,probe,guard,prior = validate_plan(args); base = helper('../pareto_realistic/qualify_absent_owner.py')
    system = system or System(); owner_pid = os.getpid(); directory = Path(plan['cold_restart']['journal_path'])
    folder = args.out/'phases'/PHASE; child_out = Path(plan['outputs']['probe'])
    child = None; intent = False; children = []; errors = []; live = []; cold_rows = []; checkpoints = []
    binding = attempt = None; began = time.monotonic(); child_started = None; phase = 'attempt'; release = None
    def write(path,value,exclusive=True): base.durable(path,value,exclusive=exclusive)
    def error(exc,where):
        errors.append(dict(phase=where,type=type(exc).__name__,message=str(exc),time=time.time()))
        write(args.out/'errors.json',errors,False)
    def check():
        need(time.monotonic()-began < 1830,'owner work deadline'); return stopped(plan,args,prior,binding)
    def point(label,pid=None,cleanup=False):
        if cleanup: stopped(plan,args,prior,binding)
        else: check()
        start=time.monotonic()
        def retain(row):
            row['label']=label
            write(args.out/f'checkpoint-{len(checkpoints):04d}.json',row);checkpoints.append(row)
        row=checkpoint(system,old_pid=prior['replacement']['pid'],child_pid=pid,owner_pid=owner_pid,
                       deadline=start+plan['budgets']['checkpoint_seconds'],record=retain)
        if cleanup: stopped(plan,args,prior,binding)
        else: check()
        return row
    def relative(path): return dict(path=str(path.relative_to(args.out)),sha256=sha(path))
    def index(): write(args.out/'children.json',dict(schema='cold-service-owner-children-v1',children=children),False)
    def stop(*unused): raise TimeoutError('GLM owner deadline or cancellation')
    need(signal.getitimer(signal.ITIMER_REAL)==(0.0,0.0),'outer alarm already active')
    handlers={s:signal.getsignal(s) for s in (signal.SIGALRM,signal.SIGINT,signal.SIGTERM)}
    with base.journal_lock(directory):
        snapshot,binding=stopped(plan,args,prior); identity=system.identity(owner_pid)
        need(identity['pid']==owner_pid,'actual owner identity')
        args.out.mkdir(exist_ok=False);folder.mkdir(parents=True)
        base.durable(args.out/'stopped-journal.json',snapshot,exclusive=True,raw=True)
        attempt=dict(schema='cold-service-owner-attempt-v1',owner_pid=owner_pid,owner_identity=identity,
            owner_source=plan['execution']['owner'],owner_plan_sha256=args.plan_sha256,journal_id=binding['id'],
            journal_path=str(directory),controller_sha256=binding['controller_sha256'],removed_at=binding['removed_at'],
            created_at=time.time(),phase='PRESPAWN',stopped_journal_sha256=hashlib.sha256(snapshot).hexdigest())
        write(args.out/'attempt.json',attempt);index()
    try:
        for s in handlers: signal.signal(s,stop)
        remaining=1830-(time.monotonic()-began);need(remaining>0,'owner work deadline before prepare')
        signal.setitimer(signal.ITIMER_REAL,remaining)
        base.durable(args.out/'owner-plan.json',args.plan.read_bytes(),exclusive=True,raw=True)
        phase='prepare';start=time.monotonic();point('before-prepare')
        def preparing(): check();need(time.monotonic()-start <= 300,'bounded GLM preparation')
        proof=verify_inventory(plan,probe,preparing);write(args.out/'model-runtime.json',proof)
        for name,digest in source_pins(plan).items():
            preparing();need(sha(name)==digest,'source changed during freeze')
            dest=args.out/'owner-source'/name.lstrip('/');dest.parent.mkdir(parents=True,exist_ok=True)
            base.durable(dest,Path(name).read_bytes(),exclusive=True,raw=True)
        write(args.out/'source-pins.json',source_pins(plan));point('after-prepare')
        phase='cold-physical'
        for i in range(2):
            if i: time.sleep(4.1)
            point('cold-'+str(i));cold_rows.append(guard.measure(args.out/f'cold-{i}',owner_pid,
                timeout_seconds=plan['physical_policy']['probe_seconds']))
        cold=guard.cold_admission(cold_rows,plan['physical_policy'],owner_pid=owner_pid)
        write(args.out/'cold-physical.json',dict(samples=[ref(args.out/f'cold-{i}'/'sample.json') for i in range(2)],admission=cold))
        point('before-spawn');snapshot,_=check()
        write(args.out/'cold-release.json',dict(stopped_journal=ref(args.out/'stopped-journal.json'),
            checkpoint=ref(args.out/f'checkpoint-{len(checkpoints)-1:04d}.json'),journal_id=binding['id'],removed_at=binding['removed_at']))
        issued=time.time();admission=dict(schema='GLM-official-load-admission-v1',owner_pid=owner_pid,
            plan_sha256=plan['probe_plan']['sha256'],probe_sha256=FIXED['../glm_flash/load_probe.py'],
            output_directory=str(child_out),snapshot=str(probe.SNAPSHOT),issued_at=issued,expires_at=issued+1800,
            cold_service_absent=True,exclusive_resources=True,hard_reap_owner=True,model_runtime_verified=True,
            evidence=dict(cold_release=ref(args.out/'cold-release.json'),resources=ref(args.out/'cold-physical.json'),model_runtime=ref(args.out/'model-runtime.json')))
        write(args.out/'admission.json',admission);admission_sha=sha(args.out/'admission.json')
        command=child_command(plan,probe,args.out/'admission.json',child_out)
        need(not any(k.startswith('OMLX_') for k in os.environ),'inherited OMLX environment forbidden')
        env=dict(os.environ);env.update(probe.ENVIRONMENT)
        write(args.out/'launch.json',dict(command=command,environment=probe.ENVIRONMENT,probe_plan=plan['probe_plan'],admission=ref(args.out/'admission.json')))
        with (folder/'child.log').open('x') as log:
            with base.journal_lock(directory):
                check();phase='spawn';intent=True
                evidence=dict(phase=PHASE,owner_pid=owner_pid,journal_id=binding['id'],owner_plan_sha256=args.plan_sha256,
                    attempt_sha256=sha(args.out/'attempt.json'),command=command,time=time.time())
                write(args.out/'spawn-intent.json',evidence);write(folder/'intent.json',evidence)
                child=system.spawn(command,log,env);child_started=time.monotonic()
                need(type(child.pid) is int and child.pid>1 and child.pid!=owner_pid,'returned child PID')
                payload=dict(phase=PHASE,pid=child.pid,owner_pid=owner_pid,identity=None,ownership='direct-Popen-returned',
                    spawned_at=time.time(),spawned_monotonic=child_started)
                write(folder/'child.json',payload);children.append(dict(phase=PHASE,pid=child.pid,identity=None,
                    intent=relative(folder/'intent.json'),child=relative(folder/'child.json'),release=None));index()
                ci=system.identity(child.pid);need(ci['pid']==child.pid and ci['ppid']==owner_pid,'direct child identity')
                payload['identity']=children[0]['identity']=ci;write(folder/'child.json',payload,False)
                children[0]['child']=relative(folder/'child.json');index()
            phase=PHASE
            while child.poll() is None:
                check();need(system.identity(child.pid)==ci,'child identity changed')
                row=guard.measure(args.out/f'physical-{len(live):04d}',child.pid,timeout_seconds=plan['physical_policy']['probe_seconds'])
                guard.observe(row,plan['physical_policy'],cold,child_pid=child.pid,child_started_monotonic=child_started,previous=live[-1] if live else None)
                live.append(row);point('running',child.pid)
                need(child.poll() is not None or system.identity(child.pid)==ci,'child identity changed after sample')
                remaining=min(1830-(time.monotonic()-began),plan['physical_policy']['work_seconds']-(time.monotonic()-child_started))
                need(remaining>0,'child elapsed deadline')
                gap_deadline=row['ended_monotonic']+plan['physical_policy']['interval_seconds']
                need(time.monotonic()<=gap_deadline,'resource checkpoint exceeded observation cadence')
                try:child.wait(timeout=min(0.1,remaining,max(0,gap_deadline-time.monotonic())))
                except subprocess.TimeoutExpired:pass
            child.wait(timeout=0)
        need(time.monotonic()-child_started<=plan['physical_policy']['work_seconds'],'final child elapsed deadline')
        need(child.returncode==0,'official GLM probe failed')
        probe_terminal(read(child_out/'complete.json'),plan,admission_sha,child.pid,owner_pid,probe)
    except BaseException as exc:error(exc,phase)
    finally:
        try:
            signal.setitimer(signal.ITIMER_REAL,0)
            for s in (signal.SIGINT,signal.SIGTERM):signal.signal(s,signal.SIG_IGN)
            if child is not None:
                try:
                    def absent(pid):
                        row=system.read(['/bin/ps','-p',str(pid),'-o','pid='],timeout=2)
                        write(folder/'child-absence.json',row);return row
                    release=guard.reap_owned(child,expected_pid=child.pid,owner_pid=owner_pid,absence=absent,out=folder/'reap.json')
                    need(release['status']=='REAPED' and len(children)==1,'unresolved known-child cleanup')
                    write(folder/'release.json',dict(phase=PHASE,pid=child.pid,returncode=child.returncode,
                        waited_reaped=True,strict_absence=True,absence=relative(folder/'child-absence.json'),reap=relative(folder/'reap.json'),time=time.time()))
                    children[0]['release']=relative(folder/'release.json');index()
                except BaseException as exc:error(exc,'reap')
            try:
                signal.signal(signal.SIGALRM,stop);signal.setitimer(signal.ITIMER_REAL,120)
                stopped(plan,args,prior,binding)
                point('postflight',cleanup=True)
                write(args.out/'model-runtime-after.json',verify_inventory(plan,probe,lambda:stopped(plan,args,prior,binding)))
                need(all(sha(p)==h for p,h in plan['execution']['pins'].items()),'postflight source changed')
            except BaseException as exc:error(exc,'postflight')
            finally:
                signal.setitimer(signal.ITIMER_REAL,0)
            bound={k:attempt[k] for k in ('owner_pid','owner_identity','owner_source','owner_plan_sha256','journal_id','removed_at')}
            if not intent:
                need(child is None and not children and errors,'pre-intent no-child proof only')
                write(args.out/'terminal.json',dict(schema='cold-service-owner-terminal-v1',status='NO_CHILD_STARTED',**bound,
                    attempt_sha256=sha(args.out/'attempt.json'),spawn_intent_written=False,child_started=False,
                    terminal_at=time.time(),phase=errors[0]['phase'],error=errors[0]))
            elif len(children)==1 and children[0]['release'] is not None:
                write(args.out/'release.json',dict(schema='cold-service-owner-release-v2',**bound,attempt_sha256=sha(args.out/'attempt.json'),
                    children=children,children_sha256=sha(args.out/'children.json'),all_children_reaped=True,time=time.time()))
            terminal={}
            try:
                if (child_out/'complete.json').is_file():
                    terminal=read(child_out/'complete.json');need(type(terminal) is dict,'probe completion must be an object')
            except BaseException as exc:
                terminal={};error(exc,'probe-terminal-read')
            try:
                result = guard.diagnostic_report(cold_rows,live,plan['physical_policy'],release or {},terminal,
                    child_started_monotonic=child_started,observation_error=errors[0] if errors else None)
            except BaseException as exc:
                error(exc,'diagnostic-read')
                result=dict(schema='GLM-owner-diagnostic-error-v1',status='FAIL',error=errors[-1],
                    physical_resource_safety_qualified=False,full_model_correctness_qualified=False,performance_qualified=False)
            write(args.out/'diagnostic.json',result)
            write(args.out/'complete.json',dict(schema='GLM-cold-load-owner-result-v1',status='PASS' if not errors and result['status']=='PASS' else 'FAIL',
                owner_pid=owner_pid,owner_plan_sha256=args.plan_sha256,planned_trials=[PHASE],errors=errors,
                spawn_intent_written=intent,child_started=True if child else False if not intent else None,
                all_children_reaped=(args.out/'release.json').exists(),child_returncode=child.returncode if child else None,
                live_physical_samples=len(live),checkpoints=len(checkpoints),diagnostic=ref(args.out/'diagnostic.json'),
                full_model_correctness_qualified=False,performance_qualified=False,physical_resource_safety_qualified=False,
                independent_raw_resource_and_restoration_audits_required=True))
        finally:
            signal.setitimer(signal.ITIMER_REAL,0)
            for signum,handler in handlers.items(): signal.signal(signum,handler)

    return int(bool(errors) or result['status']!='PASS')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',action='store_true')
    parser.add_argument('--plan',type=Path);parser.add_argument('--plan-sha256');parser.add_argument('--out',type=Path)
    args=parser.parse_args(argv)
    if not args.run:
        need(not any((args.plan,args.plan_sha256,args.out)),'explicit --run required with owner arguments')
        print('DESCRIPTION_ONLY: GLM direct-child owner; reviewed adjacent controller and real STOPPED admission required.');return 0
    need(all((args.plan,args.plan_sha256,args.out)),'complete explicit owner plan/SHA/out required')
    args.plan,args.out=args.plan.resolve(),args.out.resolve()
    need(args.out.parent.is_dir() and not args.out.exists(),'fresh owner output required')
    return run(args)


if __name__=='__main__':raise SystemExit(main())
