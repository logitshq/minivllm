"""Fixed E024 diagnostic service recovery v7 from a verified immutable RESTORED predecessor.

New journal only. The existing user authorization covers this same cold cycle.
Reuses frozen v1 OS/status parsers; no service action occurs on import or --help.
The parent samples physical memory while directly owning one fixed POST worker.
Stopping that worker cancels only the HTTP wait, never the service-side loader.
Every failed/unknown load retains LOAD_INTENT for operator reconciliation; no
duplicate POST, automatic service kill, inference, or experiment launch exists.
"""
import argparse
import contextlib
import hashlib
import http.client
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import time
import uuid

BASE_PATH=Path(__file__).with_name('controller.py')
BASE_SHA='aad297fb08515d6304fd3ae435301318650e8443ad22628c1ed2e4791887f0f3'
if hashlib.sha256(BASE_PATH.read_bytes()).hexdigest()!=BASE_SHA:
    raise RuntimeError('frozen v1 support changed')
_spec=importlib.util.spec_from_file_location('fixed_omlx_cold_v1_support',BASE_PATH)
b=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(b)
PLAN_SHA,CONTRACT,GIB=b.PLAN_SHA,b.CONTRACT,b.GIB
Refuse,NotReady,need=b.Refuse,b.NotReady,b.need
Journal=b.Journal
V6_SHA='9680bcfde32f80b03dc5ee887096678ff9ebba8b8b2a49b8d8580f95d6a77a92'
RESTORE_AUDIT_SHA='d11f9eb7757b12251d794e01c52c3aa76f6843d373f112b39af5cabfb60abffd'
ARCHIVE_ROOT=b.ROOT/'docs/experiments/m5-ultra-256gb'
PHASES=('e024-route-canary',)


def trial_names(value):
    need(type(value) is list and value==list(PHASES),
         'exact single E024 diagnostic phase required')
    return value


def registered_trials(j):
    ref=j.data['owner_plan'];need(b.sha(ref['path'])==ref['sha256'],'owner wrapper plan SHA')
    plan=b.read_json(ref['path'])
    need(plan.get('schema')=='E024-route-cold-owner-plan-v1','E024 cold owner plan required')
    trials=trial_names(plan.get('trials'))
    need(trials==j.data.get('trials'),'journal registered trials changed')
    return trials


def fixed_identity(value):
    need(type(value) is dict and set(value)=={'pid','ppid','start','comm'}
         and type(value['pid']) is int and value['pid']>1
         and type(value['ppid']) is int and value['ppid']==1
         and type(value['start']) is str and bool(value['start'])
         and value['comm']=='omlx-server','fixed service identity invalid')
    return value


def original_identity(s,expected):
    expected=fixed_identity(expected);pid=expected['pid']
    need(s.process(pid)==expected,'original service identity changed')
    need(s.job()==dict(pid=pid,state='running',path=b.PLIST) and s.listener()==pid,
         'original supervisor/listener mismatch')
    return dict(expected)


def absent_service(s,original):
    need(s.process(fixed_identity(original)['pid']) is None and s.job() is None
         and s.listener() is None,'service not fully removed')


def replacement(s,original):
    original=fixed_identity(original);job=s.job()
    need(job is not None and job.get('state')=='running' and job.get('path')==b.PLIST
         and type(job.get('pid')) is int and job['pid']>1 and job['pid']!=original['pid'],
         'replacement job not ready')
    identity=fixed_identity(s.process(job['pid']))
    need(identity['pid']==job['pid'] and s.listener()==identity['pid'],'replacement PID/listener mismatch')
    return identity


def predecessor(binding):
    """Accept one explicitly pinned, independently audited v6 restoration archive.

    Both immutable bytes are SHA-pinned by the new owner plan. Restoration audit
    PASS is required; the preceding numerical experiment may have failed.
    """
    need(type(binding) is dict and set(binding)=={'path','sha256','audit'},'predecessor binding fields')
    audit_ref=binding['audit'];need(type(audit_ref) is dict and set(audit_ref)=={'path','sha256'},'audit binding fields')
    raw=[]
    for item in (binding,audit_ref):
        path=Path(item['path'])
        need(path.is_absolute() and path.resolve().is_relative_to(ARCHIVE_ROOT.resolve()),'immutable archived predecessor required')
        data=path.read_bytes();need(hashlib.sha256(data).hexdigest()==item['sha256'],'predecessor/audit SHA changed')
        raw.append(data)
    prior,audit=map(json.loads,raw)
    need(prior.get('schema')=='omlx-cold-restart-journal-v6'
         and prior.get('controller_sha256')==V6_SHA
         and audit.get('schema')=='cold-service-restoration-audit-v1'
         and audit.get('status')=='PASS' and audit.get('qualified') is True
         and audit.get('audit_source_sha256')==RESTORE_AUDIT_SHA
         and b.sha(Path(__file__).with_name('audit_restore_v6.py'))==RESTORE_AUDIT_SHA
         and audit.get('journal_sha256')==binding['sha256']
         and audit.get('controller_sha256')==V6_SHA,
         'reviewed v6 restoration journal/controller/audit binding required')
    need(prior.get('state')=='RESTORED' and prior.get('approved_contract')==CONTRACT
         and prior.get('plan_sha256')==PLAN_SHA and prior.get('support_sha256')==BASE_SHA
         and not prior.get('post_monitor_failed')
         and not any(e.get('event') in ('post_monitor_incomplete','post_worker_cleanup_failed')
                     for e in prior.get('events',[])),'predecessor recovery incomplete')
    restored=audit.get('service_restore',{})
    need(restored.get('status')=='PERSISTED_VISIBLE_RECEIPTS_CONSISTENT'
         and restored.get('replacement')==prior.get('replacement')
         and restored.get('raw_log_sha256')==prior.get('log_window',{}).get('sha256')
         and restored.get('raw_log_bytes')==prior.get('log_window',{}).get('bytes')
         and restored.get('http_worker_returncode')==0
         and audit.get('owner_plan_sha256')==prior.get('owner_plan',{}).get('sha256'),
         'verified restoration receipt mismatch')
    fixed_identity(prior.get('replacement'))
    need(prior.get('final_status')==dict(version='0.7.0',default_model=b.TARGET,loaded=[b.TARGET],
         loading=0,target_loading=False,path=b.MODEL,pinned=False,engine_type='vlm',config_model_type='qwen4_exp'),
         'predecessor persisted-visible variant changed')
    need(prior.get('exact_runtime_signature_restored') is False and prior.get('warm_kv_restored') is False,
         'unsupported restoration claim')
    return prior,raw


def atomic_bytes(path,raw):
    path=Path(path);fd,tmp=tempfile.mkstemp(prefix='.raw-',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
        os.replace(tmp,path)
        dfd=os.open(path.parent,os.O_RDONLY)
        try:os.fsync(dfd)
        finally:os.close(dfd)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)


def exclusive_json(path,value):
    with Path(path).open('x') as stream:
        json.dump(value,stream);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    dfd=os.open(Path(path).parent,os.O_RDONLY)
    try:os.fsync(dfd)
    finally:os.close(dfd)


def parse_resources(footprint,pressure,swap,memory,now):
    need(footprint.get('unit')=='byte' and type(footprint.get('bytes per unit')) is int
         and footprint['bytes per unit']==1,'physical byte units required')
    need(footprint.get('errors')==[] and footprint.get('warnings')==[],'physical errors/warnings')
    value=footprint.get('auxiliary',{}).get('sys_footprint')
    need(type(value) is int and value>0,'system physical footprint invalid')
    need(re.fullmatch(r'\d+',pressure.strip()) and re.fullmatch(r'\d+',memory.strip()),'pressure/RAM format')
    match=re.search(r'used = ([\d.]+)M',swap)
    need(match is not None and math.isfinite(float(match[1])),'swap format')
    return dict(time=now,system_footprint=value,pressure=int(pressure),
                swap_bytes=round(float(match[1])*(1<<20)),physical_ram=int(memory))


class System(b.System):
    def __init__(self,directory):
        self.directory=Path(directory);self.evidence=self.directory/'evidence'
        self.evidence.mkdir(mode=0o700,exist_ok=True)

    def resources(self):
        folder=self.evidence/('physical-'+uuid.uuid4().hex);folder.mkdir(mode=0o700)
        commands=[['/usr/bin/footprint','-p',str(os.getpid()),'--noCategories','--sysFootprint',
                   '-f','bytes','-j',str(folder/'footprint.raw.json')],
                  ['/usr/sbin/sysctl','-n','kern.memorystatus_vm_pressure_level'],
                  ['/usr/sbin/sysctl','vm.swapusage'],['/usr/sbin/sysctl','-n','hw.memsize']]
        captures=[]
        try:
            for argv in commands:
                started=time.monotonic()
                try:
                    proc=subprocess.run(argv,capture_output=True,text=True,timeout=10)
                    row=dict(command=argv,returncode=proc.returncode,stdout=proc.stdout,stderr=proc.stderr)
                except subprocess.TimeoutExpired as error:
                    row=dict(command=argv,returncode=None,stdout=str(error.stdout or ''),stderr=str(error.stderr or ''))
                row['elapsed_s']=time.monotonic()-started;captures.append(row)
                b.atomic_write(folder/'commands.json',captures)
                need(row['returncode']==0,'physical probe failed; raw commands retained')
            path=folder/'footprint.raw.json'
            raw=path.read_bytes();atomic_bytes(path,raw)
            parsed=parse_resources(json.loads(raw),captures[1]['stdout'],captures[2]['stdout'],captures[3]['stdout'],time.time())
            parsed['raw']={'path':str(path),'sha256':hashlib.sha256(raw).hexdigest(),
                           'commands_path':str(folder/'commands.json'),'commands_sha256':b.sha(folder/'commands.json')}
            b.atomic_write(folder/'parsed.json',parsed)
            return parsed
        except BaseException as error:
            b.atomic_write(folder/'error.json',dict(type=type(error).__name__,message=str(error)))
            raise

    def log_window(self,before):
        folder=self.evidence/('log-'+uuid.uuid4().hex);folder.mkdir(mode=0o700)
        try:
            with b.LOG.open('rb') as stream:
                stat=os.fstat(stream.fileno());start=dict(dev=stat.st_dev,ino=stat.st_ino,size=stat.st_size,mtime_ns=stat.st_mtime_ns)
                need((start['dev'],start['ino'])==(before['dev'],before['ino']),'log rotated; manual recovery review')
                need(before['size']<=start['size']<=before['size']+(4<<20),'log window truncated/too large')
                stream.seek(before['size']);raw=stream.read(start['size']-before['size'])
                need(len(raw)==start['size']-before['size'],'short log window')
                after=os.fstat(stream.fileno())
                need(after.st_dev==stat.st_dev and after.st_ino==stat.st_ino and after.st_size>=stat.st_size,
                     'log identity/truncation changed during capture')
            path=folder/'window.raw.log';atomic_bytes(path,raw)
            receipt=dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),
                         source=str(b.LOG),before=before,captured=start,start_offset=before['size'],end_offset=start['size'])
            b.atomic_write(folder/'window.json',receipt)
            return raw.decode('utf-8',errors='strict').splitlines(),receipt
        except BaseException as error:
            b.atomic_write(folder/'error.json',dict(type=type(error).__name__,message=str(error)))
            raise


def owner_plan(path,digest,journal):
    path=Path(path).resolve();need(b.sha(path)==digest,'owner wrapper plan SHA')
    plan=b.read_json(path)
    need(plan.get('schema')=='E024-route-cold-owner-plan-v1','service-absent owner plan required')
    owner=plan['execution']['owner'];cold=plan['cold_restart']
    need(Path(owner['path']).is_absolute() and b.sha(owner['path'])==owner['sha256'],'owner source pin')
    need(Path(cold['journal_path']).resolve()==Path(journal).resolve(),'owner journal binding')
    need(Path(cold['controller']['path']).resolve()==Path(__file__).resolve()
         and cold['controller']['sha256']==b.sha(__file__),'owner v7 controller pin')
    need(cold['approved_contract']==CONTRACT and cold['proposal_plan_sha256']==PLAN_SHA,'owner approved-contract binding')
    need('predecessor' in cold,'owner immutable predecessor missing')
    trial_names(plan.get('trials'))
    need(type(plan.get('deadline_seconds')) is int and plan['deadline_seconds']==1830,'fixed E024 owner deadline')
    return plan


def prepare(j,s,owner_directory,wrapper,wrapper_sha):
    need(not j.data,'journal already exists');b.pins();need(os.getuid()==501,'wrong launchd user')
    owner=Path(owner_directory).resolve()
    need(not owner.exists() and owner.parent.is_dir(),'predeclare a fresh owner directory')
    plan=owner_plan(wrapper,wrapper_sha,j.directory)
    binding=plan['cold_restart']['predecessor'];prior,raw=predecessor(binding)
    identity,status=original_identity(s,prior['replacement']),b.service_status(s,[b.TARGET])
    need(status==prior['final_status'],'persisted-visible service status differs from predecessor')
    need(b.file_stat(b.GLOBAL_SETTINGS)==prior['global_settings_stat'],'global settings changed since predecessor')
    metadata={str(Path(b.MODEL)/name):b.sha(Path(b.MODEL)/name) for name in
              ('config.json','model.safetensors.index.json') if (Path(b.MODEL)/name).exists()}
    need(str(Path(b.MODEL)/'config.json') in metadata,'model config missing')
    need(metadata==prior['model_metadata'],'model metadata changed since predecessor')
    copies={}
    for name,data in zip(('predecessor.json','predecessor-audit.json'),raw):
        path=j.directory/name;atomic_bytes(path,data);copies[name]=hashlib.sha256(data).hexdigest()
    j.data=dict(schema='omlx-cold-restart-journal-v7',id=str(uuid.uuid4()),state='PREPARED',
        plan_sha256=PLAN_SHA,controller_sha256=b.sha(__file__),support_sha256=BASE_SHA,original=identity,
        original_status=status,global_settings_stat=b.file_stat(b.GLOBAL_SETTINGS),model_metadata=metadata,
        log_start=s.log_identity(),owner_directory=str(owner),owner_plan={'path':str(Path(wrapper).resolve()),'sha256':wrapper_sha},
        owner_source=plan['execution']['owner'],trials=plan['trials'],predecessor=binding,predecessor_copies=copies,resource_samples=[],events=[])
    j.save('prepared_v7_read_only')


def authorize(j,approved_contract,plan_sha):
    need(approved_contract==CONTRACT and plan_sha==PLAN_SHA,'explicit approved contract/fixed plan required')
    need(j.data.get('schema')=='omlx-cold-restart-journal-v7','fresh v7 journal required; never migrate old journals')
    need(j.data['controller_sha256']==b.sha(__file__) and j.data['plan_sha256']==PLAN_SHA
         and j.data['support_sha256']==BASE_SHA and b.sha(BASE_PATH)==BASE_SHA,'controller/source identity changed')
    need(j.data['global_settings_stat']==b.file_stat(b.GLOBAL_SETTINGS),'global settings stat changed')
    for path,digest in j.data['model_metadata'].items():need(b.sha(path)==digest,'model metadata changed')
    plan=owner_plan(j.data['owner_plan']['path'],j.data['owner_plan']['sha256'],j.directory)
    need(plan['execution']['owner']==j.data['owner_source'],'owner source binding changed')
    need(plan['trials']==j.data['trials'],'registered trials changed')
    need(plan['cold_restart']['predecessor']==j.data['predecessor'],'predecessor owner binding changed')
    prior,raw=predecessor(j.data['predecessor'])
    need(prior['replacement']==j.data['original'] and prior['global_settings_stat']==j.data['global_settings_stat']
         and prior['model_metadata']==j.data['model_metadata'],'original no longer matches predecessor')
    copies={name:hashlib.sha256(data).hexdigest() for name,data in zip(('predecessor.json','predecessor-audit.json'),raw)}
    need(j.data['predecessor_copies']==copies,'predecessor copy manifest changed')
    for name,digest in copies.items():need(b.sha(j.directory/name)==digest,'frozen predecessor copy changed')
    b.pins()


def stop(j,s,approved_contract,plan_sha):
    authorize(j,approved_contract,plan_sha)
    if j.data['state']=='STOPPED':absent_service(s,j.data['original']);return
    if j.data['state']=='STOP_INTENT':
        absent_service(s,j.data['original']);j.save('stop_reconciled',state='STOPPED',removed_at=time.time());return
    need(j.data['state']=='PREPARED','stop inadmissible')
    need(not Path(j.data['owner_directory']).exists(),'owner appeared before stop')
    original_identity(s,j.data['original']);b.service_status(s,[b.TARGET])
    j.save('bootout_intent',state='STOP_INTENT',approved_contract=approved_contract)
    j.save('bootout_returned',bootout_returncode=s.mutate('stop'))
    for _ in range(15):
        try:absent_service(s,j.data['original']);j.save('service_absent',state='STOPPED',removed_at=time.time());return
        except Refuse:s.sleep(2)
    raise Refuse('bootout unknown; never repeat blindly')


def phase_reaps(j,directory,attempt):
    """Validate the registered E024 phase' complete observed child prefix."""
    planned=registered_trials(j)
    paths=[directory/name for name in ('spawn-intent.json','children.json','release.json')]
    if not all(path.exists() for path in paths):raise NotReady('diagnostic owner spawn/reap outcome unresolved')
    intent,index,final=map(b.read_json,paths);pid=attempt['owner_pid']
    binding=dict(owner_pid=pid,owner_identity=attempt['owner_identity'],
        attempt_sha256=b.sha(directory/'attempt.json'),owner_plan_sha256=j.data['owner_plan']['sha256'],
        journal_id=j.data['id'],owner_source=j.data['owner_source'],removed_at=j.data['removed_at'])
    need(all(intent.get(k)==binding[k] for k in ('owner_pid','attempt_sha256','owner_plan_sha256','journal_id')),
         'diagnostic spawn intent binding')
    need(index.get('schema')=='cold-service-owner-children-v1' and type(index.get('children')) is list,
         'diagnostic children index required')
    children=index['children']
    need(final.get('schema')=='cold-service-owner-release-v2' and all(final.get(k)==v for k,v in binding.items())
         and final.get('children')==children and final.get('children_sha256')==b.sha(paths[1])
         and final.get('all_children_reaped') is True,'diagnostic final release binding')
    phases=[row.get('phase') for row in children]
    need(0<len(phases)<=len(planned) and phases==planned[:len(phases)],'diagnostic phase order/prefix')
    need(type(final.get('time')) in (int,float) and math.isfinite(final['time'])
         and final['time']>=attempt['created_at'],'diagnostic final release time')
    need(not (directory/'child.json').exists(),'legacy single-child evidence forbidden for diagnostic owner')
    expected=set();pids=[];last=attempt['created_at']
    for row in children:
        phase=row['phase'];payloads={}
        for kind in ('intent','child','release'):
            relative=f'phases/{phase}/{kind}.json';ref=row.get(kind,{})
            need(ref.get('path')==relative,'diagnostic phase receipt path')
            path=directory/relative;expected.add(relative)
            need(path.is_file() and not path.is_symlink() and b.sha(path)==ref.get('sha256'),'diagnostic phase receipt SHA')
            paths.append(path);payloads[kind]=b.read_json(path)
        pi,child,release=(payloads[k] for k in ('intent','child','release'));cpid=child.get('pid')
        need(all(pi.get(k)==binding[k] for k in ('owner_pid','attempt_sha256','owner_plan_sha256','journal_id'))
             and pi.get('phase')==phase,'diagnostic phase intent binding')
        need(type(pi.get('time')) in (int,float) and math.isfinite(pi['time']) and pi['time']>=last,
             'diagnostic phase intent time')
        identity=child.get('identity')
        identity_known=(type(identity) is dict and type(identity.get('pid')) is int and identity['pid']==cpid
             and type(identity.get('ppid')) is int and identity['ppid']==pid
             and isinstance(identity.get('start'),str) and bool(identity['start'])
             and isinstance(identity.get('comm'),str) and bool(identity['comm']))
        known_popen=(identity is None and 'identity' in child and child.get('ownership')=='direct-Popen-returned')
        need(type(cpid) is int and cpid>1 and cpid!=pid and cpid not in pids
             and child.get('owner_pid')==pid and child.get('phase')==phase
             and (identity_known or known_popen) and 'identity' in row
             and row.get('pid')==cpid and row['identity']==identity,'diagnostic direct child identity')
        need(release.get('pid')==cpid and type(release.get('returncode')) is int
             and release.get('waited_reaped') is True and release.get('strict_absence') is True
             and type(release.get('time')) in (int,float) and math.isfinite(release['time'])
             and pi['time']<=release['time']<=final['time'],'diagnostic phase child not proven reaped')
        last=release['time'];pids.append(cpid)
    need(final.get('no_popen_receipt') is None,'direct diagnostic owner has no nested spawn exception')
    found={str(path.relative_to(directory)) for path in (directory/'phases').rglob('*')
           if path.name in ('intent.json','child.json','release.json','spawn-intent.json','no-popen.json')}
    need(found==expected,'unresolved or omitted diagnostic phase evidence')
    return paths,pids


def reaped(j,s,no_experiment=False):
    directory=Path(j.data['owner_directory'])
    if no_experiment:
        need(not directory.exists(),'cannot bypass appeared owner directory');return {'no_experiment':True}
    if not directory.exists():raise NotReady('owner not started; explicit no-experiment needed')
    need(b.sha(j.data['owner_source']['path'])==j.data['owner_source']['sha256'],'pinned owner source changed')
    attempt_path=directory/'attempt.json';terminal_path=directory/'terminal.json'
    if not attempt_path.exists():raise NotReady('durable owner attempt missing')
    a=b.read_json(attempt_path);pid=a.get('owner_pid')
    need(a.get('schema')=='cold-service-owner-attempt-v1' and type(pid) is int and pid>1,'owner attempt identity')
    binding=dict(journal_id=j.data['id'],owner_plan_sha256=j.data['owner_plan']['sha256'],
                 owner_source=j.data['owner_source'],removed_at=j.data['removed_at'])
    need(all(a.get(k)==v for k,v in binding.items()),'owner attempt binding mismatch')
    need(a.get('journal_path')==str(j.directory) and a.get('controller_sha256')==j.data['controller_sha256'],
         'owner journal/controller mismatch')
    need(type(a.get('created_at')) in (int,float) and math.isfinite(a['created_at'])
         and a['created_at']>=j.data['removed_at'] and a.get('phase')=='PRESPAWN','owner attempt time/phase')
    need(type(a.get('owner_identity')) is dict and a['owner_identity'].get('pid')==pid,'owner process identity missing')
    snapshot=directory/'stopped-journal.json'
    need(snapshot.exists() and b.sha(snapshot)==a.get('stopped_journal_sha256'),'owner STOPPED snapshot SHA')
    snap=b.read_json(snapshot)
    need(snap.get('state')=='STOPPED' and all(snap.get(k)==j.data[k] for k in
         ('schema','id','removed_at','controller_sha256','owner_directory','owner_plan','owner_source',
          'original','predecessor','predecessor_copies','trials')),'owner STOPPED binding')
    files=[attempt_path,snapshot]
    nochild=terminal_path.exists() and b.read_json(terminal_path).get('status')=='NO_CHILD_STARTED'
    if nochild:
        terminal=b.read_json(terminal_path);files.append(terminal_path)
        need(terminal.get('schema')=='cold-service-owner-terminal-v1'
             and all(terminal.get(k)==v for k,v in binding.items())
             and terminal.get('owner_pid')==pid and terminal.get('owner_identity')==a['owner_identity']
             and terminal.get('attempt_sha256')==b.sha(attempt_path)
             and terminal.get('spawn_intent_written') is False and terminal.get('child_started') is False,
             'no-child terminal proof invalid')
        need(type(terminal.get('terminal_at')) in (int,float) and math.isfinite(terminal['terminal_at'])
             and terminal['terminal_at']>=a['created_at'],'no-child terminal time')
        need(all(not (directory/name).exists() for name in ('spawn-intent.json','child.json','release.json')),
             'spawn evidence forbids NO_CHILD_STARTED')
        index=directory/'children.json'
        if index.exists():
            need(b.read_json(index)==dict(schema='cold-service-owner-children-v1',children=[]),
                 'nonempty children index forbids NO_CHILD_STARTED')
            files.append(index)
        found={str(path.relative_to(directory)) for path in (directory/'phases').rglob('*')
               if path.name in ('intent.json','child.json','release.json','spawn-intent.json','no-popen.json')}
        registered_trials(j)
        need(not found and terminal.get('no_popen_receipt') is None,
             'direct diagnostic owner cannot resolve nested spawn evidence')
        identities=[pid]
    else:
        paths,children=phase_reaps(j,directory,a)
        files+=paths;identities=[pid]+children
    need(all(path.stat().st_mtime>=j.data['removed_at'] for path in files),'stale owner receipts')
    if any(s.process(p) is not None for p in identities):raise NotReady('owner or child exists, including PID reuse')
    return dict(no_experiment=False,no_child_started=nochild,owner_pid=pid,
                receipts=[dict(path=str(path),sha256=b.sha(path)) for path in files])


def sample(j,s,label):
    row=s.resources();row['label']=label
    rows=j.data.setdefault('resource_samples',[]);rows.append(row)
    j.save('resource_sample_retained')
    return row


def check_resource(row,limit,baseline_swap=None):
    need(all(type(row.get(k)) is int for k in ('system_footprint','physical_ram','pressure','swap_bytes')),
         'invalid physical sample types')
    if not (row['physical_ram']==256*GIB and 0<row['system_footprint']<=limit and row['pressure']==1
            and row['swap_bytes']>=0 and (baseline_swap is None or row['swap_bytes']<=baseline_swap)):
        raise NotReady('physical admission/monitor limit failed; no new mutation')


def resource_gate(j,s,loaded=False):
    first=sample(j,s,'loaded-stability-1' if loaded else 'cold-stability-1');s.sleep(5)
    second=sample(j,s,'loaded-stability-2' if loaded else 'cold-stability-2')
    limit=(256-16)*GIB if loaded else 40*GIB
    check_resource(first,limit);check_resource(second,limit,first['swap_bytes'])
    need(4<=second['time']-first['time']<=60,'resource interval invalid')
    if abs(second['system_footprint']-first['system_footprint'])>GIB:raise NotReady('physical footprint not stable')
    return second


def load_worker(journal,nonce):
    """Internal fixed-POST subprocess; durable claim forbids a second worker."""
    path=Path(journal).resolve();data=b.read_json(path/'journal.json');worker=data['load_worker']
    need(data.get('schema')=='omlx-cold-restart-journal-v7' and data.get('state')=='LOAD_INTENT'
         and data.get('approved_contract')==CONTRACT and data.get('controller_sha256')==b.sha(__file__)
         and worker['nonce']==nonce,'fixed worker authorization binding')
    prior,_=predecessor(data['predecessor'])
    need(data['original']==prior['replacement'],'fixed worker predecessor identity changed')
    backend=b.System()
    need(os.getppid()==worker['parent_identity']['pid']
         and backend.process(os.getppid())==worker['parent_identity'],'direct controller parent changed')
    folder=path/'load-worker';need(str(folder)==worker['directory'],'fixed worker directory')
    exclusive_json(folder/'claim.json',dict(pid=os.getpid(),ppid=os.getppid(),nonce=nonce,time=time.time()))
    def alarm(*unused):raise TimeoutError('absolute POST worker 600s deadline')
    old=signal.signal(signal.SIGALRM,alarm);signal.setitimer(signal.ITIMER_REAL,600)
    result=dict(status='UNKNOWN',pid=os.getpid(),nonce=nonce)
    try:
        need(replacement(backend,data['original'])==data['replacement'],'replacement changed before worker POST')
        status=b.service_status(backend)
        need(not status['loaded'] and status['loading']==0 and not status['target_loading'],
             'worker refuses duplicate/active load')
        result['response']=backend.http('POST','/v1/models/'+b.TARGET+'/load',timeout=600)
        result['status']='RETURNED'
    except BaseException as error:result.update(error_type=type(error).__name__,error=str(error))
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old)
        b.atomic_write(folder/'result.json',result)
    return 0 if result['status']=='RETURNED' else 1


def monitored_load(j,s,identity,baseline,popen=subprocess.Popen):
    folder=j.directory/'load-worker';folder.mkdir(mode=0o700,exist_ok=False)
    parent=s.process(os.getpid());need(parent is not None,'controller identity unavailable')
    nonce=uuid.uuid4().hex
    j.save('load_intent_monitored',state='LOAD_INTENT',load_baseline_swap_bytes=baseline['swap_bytes'],
           load_worker=dict(nonce=nonce,directory=str(folder),parent_identity=parent))
    child=None;primary=None
    started=time.monotonic()
    try:
        with (folder/'stdout.txt').open('xb') as stdout,(folder/'stderr.txt').open('xb') as stderr:
            child=popen([sys.executable,'-I','-B',str(Path(__file__).resolve()),'load-worker',
                         '--journal',str(j.directory),'--nonce',nonce],stdout=stdout,stderr=stderr)
            j.save('post_worker_started',post_worker_pid=child.pid)
            while True:
                need(time.monotonic()-started<600,'absolute monitored POST deadline')
                row=sample(j,s,'during-load-POST')
                check_resource(row,(256-16)*GIB,baseline['swap_bytes'])
                need(replacement(s,j.data['original'])==identity,'replacement changed during POST')
                need(time.monotonic()-started<600,'absolute monitored POST deadline')
                if child.poll() is not None:break
                s.sleep(2)
            child.wait(timeout=1)
            path=folder/'result.json'
            if path.exists():j.save('post_worker_returned',post_result=dict(path=str(path),sha256=b.sha(path)))
            else:j.save('post_worker_outcome_unknown',post_worker_returncode=child.returncode)
    except BaseException as error:
        primary=error;j.save('post_monitor_incomplete',monitor_error_type=type(error).__name__,post_monitor_failed=True,
                            cancellation_scope='HTTP wait only; service loader may remain active')
        raise
    finally:
        try:
            if child is not None:
                if child.poll() is None:
                    child.terminate()
                    try:child.wait(timeout=5)
                    except subprocess.TimeoutExpired:child.kill();child.wait(timeout=5)
                else:child.wait(timeout=1)
                absent=s.process(child.pid) is None
                b.atomic_write(folder/'release.json',dict(pid=child.pid,returncode=child.returncode,
                    waited_reaped=True,strict_absence=absent,scope='HTTP client worker only'))
                need(absent,'HTTP worker PID absence unresolved')
        except BaseException as error:
            j.save('post_worker_cleanup_failed',cleanup_error_type=type(error).__name__)
            if primary is None:raise


def recover(j,s,approved_contract,plan_sha,no_experiment=False):
    authorize(j,approved_contract,plan_sha)
    if j.data['state']=='STOP_INTENT':
        absent_service(s,j.data['original']);j.save('stop_reconciled',state='STOPPED',removed_at=time.time())
    need(j.data['state'] in ('STOPPED','BOOTSTRAP_INTENT','JOB_RESTORED','LOAD_INTENT','RESTORED'),'recovery state')
    need(not j.data.get('post_monitor_failed'),'prior POST monitor failure requires independent operator review')
    reap=reaped(j,s,no_experiment);j.save('experiment_terminal_checked',reap=reap)
    if j.data['state']=='STOPPED':
        absent_service(s,j.data['original']);resource_gate(j,s);reaped(j,s,no_experiment);absent_service(s,j.data['original'])
        j.save('bootstrap_intent',state='BOOTSTRAP_INTENT',recovery_log_start=s.log_identity())
        j.save('bootstrap_returned',bootstrap_returncode=s.mutate('bootstrap'))
    if j.data['state']=='BOOTSTRAP_INTENT':
        for _ in range(30):
            try:
                identity=replacement(s,j.data['original']);b.service_status(s)
                j.save('job_reconciled',state='JOB_RESTORED',replacement=identity);break
            except (Refuse,OSError,http.client.HTTPException):s.sleep(2)
        else:raise Refuse('bootstrap unknown; no repeated mutation')
    identity=replacement(s,j.data['original']);need(identity==j.data['replacement'],'replacement changed')
    status=b.service_status(s)
    if j.data['state']=='JOB_RESTORED':
        need(not status['loaded'] and status['loading']==0 and not status['target_loading'],'unexpected auto-load')
        baseline=resource_gate(j,s);reaped(j,s,no_experiment)
        need(replacement(s,j.data['original'])==identity,'replacement changed before POST')
        monitored_load(j,s,identity,baseline)
    # Resume an unknown LOAD_INTENT by observation only; never another POST.
    worker_folder=j.directory/'load-worker'
    if j.data['state']=='LOAD_INTENT':
        # A restarted controller must not overlap a surviving prior HTTP worker.
        claim=worker_folder/'claim.json'
        worker_pid=j.data.get('post_worker_pid')
        if claim.exists():
            info=b.read_json(claim)
            need(info.get('nonce')==j.data['load_worker']['nonce'],'old HTTP worker claim binding')
            need(worker_pid is None or info.get('pid')==worker_pid,'old HTTP worker PID binding')
            worker_pid=info.get('pid')
        need(type(worker_pid) is int and worker_pid>1,'unknown POST worker spawn outcome; manual review')
        if s.process(worker_pid) is not None:raise NotReady('prior HTTP worker still exists; no recovery completion')
    baseline_swap=j.data['load_baseline_swap_bytes']
    for _ in range(120):
        row=sample(j,s,'load-reconciliation');check_resource(row,(256-16)*GIB,baseline_swap)
        need(replacement(s,j.data['original'])==identity,'replacement changed while loading');status=b.service_status(s)
        if status['loaded']==[b.TARGET] and status['loading']==0 and not status['target_loading']:break
        need(status['loading']>0 or status['target_loading'],'no active/completed loader; no duplicate POST')
        s.sleep(5)
    else:raise Refuse('load incomplete at bounded deadline')
    lines,receipt=s.log_window(j.data['recovery_log_start'])
    j.save('raw_log_window_retained',log_window=receipt)
    markers=b.verify_logs(lines);resource_gate(j,s,loaded=True);authorize(j,approved_contract,plan_sha)
    need(replacement(s,j.data['original'])==identity,'replacement changed at completion');final=b.service_status(s,[b.TARGET])
    j.save('restored_persisted_visible_variant',state='RESTORED',final_status=final,log_markers=markers,
           exact_runtime_signature_restored=False,warm_kv_restored=False,
           monitoring_scope='observed physical bounds; no OS hard cap; HTTP cancellation does not cancel server loader')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=('prepare','stop','recover','load-worker'))
    p.add_argument('--journal',type=Path,required=True);p.add_argument('--plan-sha256')
    p.add_argument('--approved-contract');p.add_argument('--owner-directory',type=Path)
    p.add_argument('--owner-plan',type=Path);p.add_argument('--owner-plan-sha256')
    p.add_argument('--no-experiment',action='store_true');p.add_argument('--watchdog-seconds',type=int,default=0)
    p.add_argument('--nonce');a=p.parse_args(argv)
    if a.phase=='load-worker':return load_worker(a.journal,a.nonce)
    need(a.plan_sha256==PLAN_SHA,'fixed proposal SHA required')
    need(a.phase=='prepare' or a.approved_contract==CONTRACT,'fixed authorized contract required')
    need(a.phase!='prepare' or (a.owner_directory and a.owner_plan and a.owner_plan_sha256),'prepare owner binding missing')
    need(a.phase=='recover' or not a.no_experiment,'no-experiment recovery only')
    need(0<=a.watchdog_seconds<=3600 and (not a.watchdog_seconds or a.phase=='recover'),'watchdog bound')
    need(signal.getitimer(signal.ITIMER_REAL)==(0.0,0.0),'existing alarm')
    def alarm(*unused):raise TimeoutError('absolute controller deadline; durable intent retained')
    old=signal.signal(signal.SIGALRM,alarm)
    signal.setitimer(signal.ITIMER_REAL,1800+a.watchdog_seconds if a.phase=='recover' else 90)
    try:
        until=time.monotonic()+a.watchdog_seconds
        while True:
            waiting=False
            with contextlib.closing(Journal(a.journal,create=a.phase=='prepare')) as j:
                s=System(j.directory)
                try:
                    if a.phase=='prepare':prepare(j,s,a.owner_directory,a.owner_plan,a.owner_plan_sha256)
                    elif a.phase=='stop':stop(j,s,a.approved_contract,a.plan_sha256)
                    else:recover(j,s,a.approved_contract,a.plan_sha256,a.no_experiment)
                except NotReady as error:
                    if a.phase=='recover' and j.data.get('state')!='LOAD_INTENT' and time.monotonic()<until:
                        j.save('watchdog_waiting_no_mutation');waiting=True
                    else:
                        j.save('controller_incomplete',last_error_type=type(error).__name__,recovery_complete=False);raise
                except BaseException as error:
                    if j.data:j.save('controller_incomplete',last_error_type=type(error).__name__,
                                    recovery_complete=j.data.get('state')=='RESTORED')
                    raise
            if not waiting:break
            # Release the journal lock before the owner can publish spawn/terminal evidence.
            time.sleep(5)
    finally:signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old)
    return 0


if __name__=='__main__':raise SystemExit(main())
