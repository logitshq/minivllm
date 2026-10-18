"""Stable single-child cold-cycle controller, version2.

Only predecessor admission expands: frozen v1 single-child and frozen Lithos
formal recovery receipts can enter this lifecycle. OS/load/reap algorithms,
work1830/child1800 budgets and per-run immutable sources remain unchanged.

Per-run phase/qualifier/source pins live in an immutable run plan. The reviewed
lifecycle manifest binds these three adjacent sources without circular hashes.
This one-time migration retains the v7 OS/recovery/resource/reap algorithms.

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

HERE=Path(__file__).resolve().parent
BASE_PATH=HERE.parent/'omlx_cold_restart/controller.py'
BASE_SHA='aad297fb08515d6304fd3ae435301318650e8443ad22628c1ed2e4791887f0f3'
if hashlib.sha256(BASE_PATH.read_bytes()).hexdigest()!=BASE_SHA:
    raise RuntimeError('frozen v1 support changed')
_spec=importlib.util.spec_from_file_location('fixed_omlx_cold_v1_support',BASE_PATH)
b=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(b)
PLAN_SHA,CONTRACT,GIB=b.PLAN_SHA,b.CONTRACT,b.GIB
Refuse,NotReady,need=b.Refuse,b.NotReady,b.need
Journal=b.Journal
V7_SHA='c30aaffc4157e606b2027c98933cb97d0602aa5db6a614338ae0631dce70a1ab'
V7_AUDIT_SHA='0ab9fd49b4e2919d686845f587d291e78d24a75cbd20325fdc9ca78fe6149ad1'
MIGRATION_JOURNAL_SHA='b8a209b3ea6b888ccf4331b695881a85962f1097d51c46da84338ca084669ce2'
MIGRATION_AUDIT_SHA='732e5c19b4afde075387907cb2de89026ae42510392fb9ec260e643cee06561e'
INHERITED_OWNER_SHA='4dff5f46d0b35d038ffbb0e32ebb6197b047f4da6e847b3160f3ae8685caf6a4'
SUPPORT_PINS={
    '../long_decode_graph/owner.py':'d9469fe561ee5697068de1582d1e167d2a0e0da222cc03d4449028435c232134',
    '../long_decode_graph/qualify.py':'c24219f431b08502a26a4f7a83c002214bddd7ea3962d6b1869b313e4c13faf7',
    '../long_decode_graph/candidate.py':'eb14b671a66764fe677219762090acb495b89d384cf078f3f63b918a40aa2b84',
    '../long_decode_graph/model-source-pins.json':'7143538a168bfcd8b01292c9687472c8647db393c6d2a510d628f71c670260c5',
    '../pareto_realistic/qualify_absent_owner.py':'5beb6c9bcfa4e22041a386cdbea3f5ecee210cdcb9b345512c50b6c6f873c62d',
    '../pareto_realistic/qualify_owner.py':'1efaf440e991fed64dedf15455be2d37223e709fec41bb07e8c0295a2975c1c2',
}
ARCHIVE_ROOT=b.ROOT/'docs/experiments/m5-ultra-256gb'
JOURNAL_SCHEMA='cold-single-child-journal-v2'
RUN_SCHEMA='cold-single-child-run-plan-v2'
MANIFEST_SCHEMA='cold-single-child-lifecycle-v2'


def reference(item):
    need(type(item) is dict and set(item)=={'path','sha256'},'exact source/plan reference')
    path=Path(item['path'])
    need(path.is_absolute() and str(path.resolve())==str(path) and not path.is_symlink()
         and type(item['sha256']) is str and re.fullmatch(r'[0-9a-f]{64}',item['sha256'])
         and b.sha(path)==item['sha256'],'pinned real absolute file required')
    return path


def lifecycle(ref):
    path=reference(ref)
    need(path==HERE/'lifecycle.json','fixed reviewed lifecycle manifest path')
    value=b.read_json(path)
    need(type(value) is dict and set(value)=={'schema','sources'}
         and value['schema']==MANIFEST_SCHEMA and set(value['sources'])=={'controller','owner','audit_restore'},
         'exact three-source lifecycle manifest')
    for name,item in value['sources'].items():
        need(reference(item)==HERE/(name+'.py'),'only reviewed adjacent lifecycle modules allowed')
    need(value['sources']['controller']['sha256']==b.sha(__file__),'actual controller source')
    return value


def source_pins(plan,manifest):
    pins={str((HERE/name).resolve()):digest for name,digest in SUPPORT_PINS.items()}
    pins.update({row['path']:row['sha256'] for row in manifest['sources'].values()})
    for path,digest in plan['experiment']['source_pins'].items():
        need(path not in pins or pins[path]==digest,'experiment cannot replace fixed lifecycle/support sources')
        pins[path]=digest
    for item in (plan['lifecycle'],plan['experiment']['plan'],plan['inherited_owner_plan']):
        need(item['path'] not in pins or pins[item['path']]==item['sha256'],'conflicting source reference')
        pins[item['path']]=item['sha256']
    for path,digest in pins.items():need(b.sha(path)==digest,'fixed source changed: '+path)
    return pins


def run_plan(path,digest):
    path=reference(dict(path=str(Path(path).resolve()),sha256=digest));plan=b.read_json(path)
    need(type(plan) is dict and set(plan)=={'schema','lifecycle','predecessor','experiment','inherited_owner_plan','outputs','budgets'}
         and plan['schema']==RUN_SCHEMA,'exact stable single-child plan')
    manifest=lifecycle(plan['lifecycle']);experiment=plan['experiment'];budgets=plan['budgets']
    need(type(experiment) is dict and set(experiment)=={'phase','qualifier','plan','source_pins'}
         and type(experiment['phase']) is str and re.fullmatch(r'[a-z][a-z0-9-]{0,63}',experiment['phase']),
         'one safe explicit phase name')
    qualifier=reference(experiment['qualifier']);child_path=reference(experiment['plan'])
    need(qualifier.suffix=='.py' and qualifier.is_relative_to(b.ROOT/'bench/m5_ultra_256gb'),
         'reviewed repository Python qualifier only')
    child=b.read_json(child_path)
    need(type(budgets) is dict and set(budgets)=={'work_seconds','child_seconds','term_seconds','kill_seconds'}
         and type(budgets['work_seconds']) is int and budgets['work_seconds']==1830
         and type(budgets['term_seconds']) is int and budgets['term_seconds']==15
         and type(budgets['kill_seconds']) is int and budgets['kill_seconds']==15
         and type(budgets['child_seconds']) in (int,float) and math.isfinite(budgets['child_seconds'])
         and 0<budgets['child_seconds']<=1800,'unchanged work/child/reap budgets')
    need(type(child) is dict and child.get('qualifier_sha256')==experiment['qualifier']['sha256']
         and type(child.get('deadline_s')) in (int,float) and child['deadline_s']==budgets['child_seconds'],
         'child source/deadline binding; parent also enforces elapsed deadline')
    pins=experiment['source_pins'];need(type(pins) is dict and pins.get(str(qualifier))==experiment['qualifier']['sha256'],
         'explicit qualifier/dependency source pins required')
    for name,value in pins.items():
        p=reference(dict(path=name,sha256=value))
        need(p.is_relative_to(b.ROOT) and p.suffix in ('.py','.json','.metal','.h','.md'),
             'small reviewed repository source dependencies only')
    old=reference(plan['inherited_owner_plan'])
    need(plan['inherited_owner_plan']['sha256']==INHERITED_OWNER_SHA,'fixed inherited prepare plan')
    outputs=plan['outputs'];need(type(outputs) is dict and set(outputs)=={'journal','out'},'exact outputs')
    directories=[]
    for value in outputs.values():
        p=Path(value);need(type(value) is str and p.is_absolute() and str(p.resolve())==value
             and p.is_relative_to(b.ROOT/'.artifacts/m5-ultra-256gb') and p.parent.is_dir(),'fixed absolute experiment output')
        directories.append(p)
    need(not directories[0].is_relative_to(directories[1]) and not directories[1].is_relative_to(directories[0]),
         'separate journal and owner output directories')
    source_pins(plan,manifest)
    return plan,manifest



def trial_names(value):
    need(type(value) is list and len(value)==1 and type(value[0]) is str
         and re.fullmatch(r'[a-z][a-z0-9-]{0,63}',value[0]),'exact one safe phase required')
    return value


def registered_trials(j):
    ref=j.data['owner_plan'];plan,_=run_plan(ref['path'],ref['sha256'])
    trials=trial_names([plan['experiment']['phase']])
    need(trials==j.data.get('trials') and plan['lifecycle']==j.data.get('lifecycle'),'registered phase/lifecycle changed')
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


def predecessor(binding,lifecycle_ref):
    """Accept reviewed predecessor protocols or this identical lifecycle version.

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
    manifest=lifecycle(lifecycle_ref)
    migration=(binding['sha256']==MIGRATION_JOURNAL_SHA and audit_ref['sha256']==MIGRATION_AUDIT_SHA)
    if migration:
        need(prior.get('schema')=='omlx-cold-restart-journal-v7' and prior.get('controller_sha256')==V7_SHA
             and audit.get('audit_source_sha256')==V7_AUDIT_SHA
             and b.sha(HERE.parent/'omlx_cold_restart/audit_restore_v7.py')==V7_AUDIT_SHA,
             'one explicit v7 migration predecessor only')
        controller_sha=V7_SHA
    elif prior.get('schema') in ('cold-single-child-journal-v1','E023-lithos-D512-formal-cold-journal-v1'):
        # A source-reviewed restoration protocol, never a producer numerical PASS.
        # Both raw journal/audit bytes remain pinned by this new immutable plan.
        protocols={
            'cold-single-child-journal-v1':('cold_cycle','b5237c458afa8324ffa6d5e6a5c49167578daec7425ff036112ec8e108d61213','08f5a24dc9d100ca5e5b4ed469ec6587f7ba5b36b3910e6e580e7027f32b5d21'),
            'E023-lithos-D512-formal-cold-journal-v1':('lithos_d512_formal','5c13010c5645e93eb099592e32ee6192f8a2ec7ca81c5ab2ed67c06f7e4273c2','41d260e63ce13c5c5912217ae74169e5c3cd118af32f3667aefffc3637a2d66f'),
        }
        folder,controller_sha,auditor_sha=protocols[prior['schema']]
        need(b.sha(HERE.parent/folder/'controller.py')==controller_sha
             and b.sha(HERE.parent/folder/'audit_restore.py')==auditor_sha
             and prior.get('controller_sha256')==controller_sha
             and audit.get('audit_source_sha256')==auditor_sha,
             'only frozen reviewed predecessor restoration protocols accepted')
        if folder=='cold_cycle':
            previous_lifecycle=dict(path=str(HERE.parent/folder/'lifecycle.json'),sha256='d9417e51a781275f18c296e20ac98874e61b8ce17b8bf0902ba63e27b5b2b0a8')
            need(prior.get('lifecycle')==previous_lifecycle and audit.get('lifecycle')==previous_lifecycle
                 and b.sha(previous_lifecycle['path'])==previous_lifecycle['sha256'],'fixed v1 lifecycle predecessor')
    else:
        need(prior.get('schema')==JOURNAL_SCHEMA and prior.get('lifecycle')==lifecycle_ref
             and audit.get('lifecycle')==lifecycle_ref
             and prior.get('controller_sha256')==manifest['sources']['controller']['sha256']
             and audit.get('audit_source_sha256')==manifest['sources']['audit_restore']['sha256'],
             'same reviewed lifecycle predecessor required')
        controller_sha=manifest['sources']['controller']['sha256']
    need(audit.get('schema')=='cold-service-restoration-audit-v1' and audit.get('status')=='PASS'
         and audit.get('qualified') is True and audit.get('journal_sha256')==binding['sha256']
         and audit.get('controller_sha256')==controller_sha,'exact independent restoration receipt')
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
    plan,manifest=run_plan(path,digest)
    need(plan['outputs']['journal']==str(Path(journal).resolve()),'owner journal binding')
    return plan


def prepare(j,s,owner_directory,wrapper,wrapper_sha):
    need(not j.data,'journal already exists');b.pins();need(os.getuid()==501,'wrong launchd user')
    owner=Path(owner_directory).resolve()
    need(not owner.exists() and owner.parent.is_dir(),'predeclare a fresh owner directory')
    plan=owner_plan(wrapper,wrapper_sha,j.directory)
    binding=plan['predecessor'];prior,raw=predecessor(binding,plan['lifecycle'])
    need(plan['outputs']['out']==str(owner),'owner output binding')
    manifest=lifecycle(plan['lifecycle'])
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
    j.data=dict(schema=JOURNAL_SCHEMA,id=str(uuid.uuid4()),state='PREPARED',
        plan_sha256=PLAN_SHA,controller_sha256=b.sha(__file__),support_sha256=BASE_SHA,original=identity,
        original_status=status,global_settings_stat=b.file_stat(b.GLOBAL_SETTINGS),model_metadata=metadata,
        log_start=s.log_identity(),owner_directory=str(owner),owner_plan={'path':str(Path(wrapper).resolve()),'sha256':wrapper_sha},
        owner_source=manifest['sources']['owner'],trials=[plan['experiment']['phase']],lifecycle=plan['lifecycle'],predecessor=binding,predecessor_copies=copies,resource_samples=[],events=[])
    j.save('prepared_stable_read_only')


def authorize(j,approved_contract,plan_sha):
    need(approved_contract==CONTRACT and plan_sha==PLAN_SHA,'explicit approved contract/fixed plan required')
    need(j.data.get('schema')==JOURNAL_SCHEMA,'fresh stable journal required; never migrate old journal bytes')
    need(j.data['controller_sha256']==b.sha(__file__) and j.data['plan_sha256']==PLAN_SHA
         and j.data['support_sha256']==BASE_SHA and b.sha(BASE_PATH)==BASE_SHA,'controller/source identity changed')
    need(j.data['global_settings_stat']==b.file_stat(b.GLOBAL_SETTINGS),'global settings stat changed')
    for path,digest in j.data['model_metadata'].items():need(b.sha(path)==digest,'model metadata changed')
    plan=owner_plan(j.data['owner_plan']['path'],j.data['owner_plan']['sha256'],j.directory)
    manifest=lifecycle(plan['lifecycle'])
    need(manifest['sources']['owner']==j.data['owner_source'] and plan['lifecycle']==j.data['lifecycle'],'owner/lifecycle source binding changed')
    need([plan['experiment']['phase']]==j.data['trials'],'registered phase changed')
    need(plan['predecessor']==j.data['predecessor'],'predecessor owner binding changed')
    prior,raw=predecessor(j.data['predecessor'],j.data['lifecycle'])
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
          'original','predecessor','predecessor_copies','trials','lifecycle')),'owner STOPPED binding')
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
    need(data.get('schema')==JOURNAL_SCHEMA and data.get('state')=='LOAD_INTENT'
         and data.get('approved_contract')==CONTRACT and data.get('controller_sha256')==b.sha(__file__)
         and worker['nonce']==nonce,'fixed worker authorization binding')
    prior,_=predecessor(data['predecessor'],data['lifecycle'])
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
