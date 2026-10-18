"""Twelve direct-server GDN OFF/ON trials, unchanged cold recovery primitives.

Only source/protocol-bound functions are local. Frozen stop/reap/recover code
objects receive an explicit independent dependency dictionary; imported module
globals, __file__, HERE and evidence are never mutated. Actual load-worker argv
uses this real file. A failed experiment may restore only after actual strict
reap; unknown Popen never becomes a fabricated no-child proof.
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
import signal
import subprocess
import sys
import time
from types import FunctionType, SimpleNamespace
import uuid

HERE=Path(__file__).resolve().parent
DONOR_PATH=HERE.parent/'omlx_d512_formal_retry/controller.py'
DONOR_SHA='8023d0b114c02e7a25cf8bfc305ef0986db3d7086f84cefd6f9c50436a5ded30'
OWNER_SHA='0148335cd334524fe367aab00f3a5d2026d1848a928ab469f1d1ce975334c461'
ARCHIVE_ROOT=HERE.parents[2]/'docs/experiments/m5-ultra-256gb'
PREDECESSOR_ARCHIVE=ARCHIVE_ROOT/'E029-gdn-prework-http-v1'
PREDECESSOR_FOLDER='gdn_prework_http_cycle'
PREDECESSOR_SCHEMA='GDN-prework-http-cold-journal-v1'
PREDECESSOR_CONTROLLER_SHA='04200ccde66bb6525f274282bd65338e222855201f38296bb46144abfc5372c5'
PREDECESSOR_AUDITOR_SHA='40d07aabef5d3588b4eac57a2acadf4278eee2328a3304da2a041a234f66ff5e'
JOURNAL_SCHEMA='GDN-prework-http-formal-cold-journal-v1'
OWNER_SCHEMA='GDN-prework-http-formal-owner-plan-v1'
PHASES=('r1-c1-off','r1-c1-on','r1-c8-on','r1-c8-off',
        'r2-c8-off','r2-c8-on','r2-c1-on','r2-c1-off',
        'r3-c1-off','r3-c1-on','r3-c8-on','r3-c8-off')


def load(path,digest,name):
    if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:raise ValueError('frozen lifecycle support changed: '+str(path))
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


_donor=load(DONOR_PATH,DONOR_SHA,'gdn_formal_frozen_recovery')
b=_donor.b
PLAN_SHA,CONTRACT,GIB,BASE_PATH,BASE_SHA=_donor.PLAN_SHA,_donor.CONTRACT,_donor.GIB,_donor.BASE_PATH,_donor.BASE_SHA
Refuse,NotReady,need=_donor.Refuse,_donor.NotReady,_donor.need
Journal,System=_donor.Journal,_donor.System
fixed_identity,original_identity=_donor.fixed_identity,_donor.original_identity
absent_service,replacement=_donor.absent_service,_donor.replacement
atomic_bytes,exclusive_json=_donor.atomic_bytes,_donor.exclusive_json
parse_resources,sample,check_resource,resource_gate=_donor.parse_resources,_donor.sample,_donor.check_resource,_donor.resource_gate
no_popen_proof=_donor.no_popen_proof


def owner_module(): return load(HERE/'owner.py',OWNER_SHA,'gdn_formal_actual_owner')


def predecessor(binding):
    """Only the future immutable GDN HTTP archive; exact bytes are plan inputs.

    The owner plan freezes both bytes. A failed experiment is permissible; a
    missing/incomplete restoration, unknown source or live journal is not.
    """
    need(type(binding) is dict and set(binding)=={'path','sha256','audit'},'predecessor binding fields')
    need(type(binding['audit']) is dict and set(binding['audit'])=={'path','sha256'},'audit binding fields')
    raw=[]
    for item in (binding,binding['audit']):
        path=Path(item['path'])
        need(path.is_absolute() and path==path.resolve() and path.is_relative_to(ARCHIVE_ROOT.resolve())
             and path.is_file() and not path.is_symlink(),'regular immutable archived predecessor')
        need(type(item['sha256']) is str and len(item['sha256'])==64
             and all(x in '0123456789abcdef' for x in item['sha256']),'predecessor SHA syntax')
        data=path.read_bytes();need(hashlib.sha256(data).hexdigest()==item['sha256'],'predecessor/audit SHA changed');raw.append(data)
    prior,audit=map(json.loads,raw)
    need(Path(binding['path'])==PREDECESSOR_ARCHIVE/'restored-journal.json'
         and Path(binding['audit']['path'])==PREDECESSOR_ARCHIVE/'restoration-audit.json',
         'actual archived GDN HTTP restoration predecessor required')
    controller_sha,auditor_sha=PREDECESSOR_CONTROLLER_SHA,PREDECESSOR_AUDITOR_SHA
    need(prior.get('schema')==PREDECESSOR_SCHEMA
         and prior.get('phases')==['c1-capture','c1-replay','c8-capture','c8-replay']
         and b.sha(HERE.parent/PREDECESSOR_FOLDER/'controller.py')==controller_sha
         and b.sha(HERE.parent/PREDECESSOR_FOLDER/'audit_restore.py')==auditor_sha
         and prior.get('controller_sha256')==controller_sha and audit.get('audit_source_sha256')==auditor_sha,
         'exact reviewed predecessor controller/auditor required')
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


def trial_names(value):
    need(type(value) is list and value==list(PHASES),'exact twelve GDN formal phases required');return value


def source_pins(plan):
    owner=owner_module();pins=owner.source_pins()
    need(plan['execution']==dict(owner=dict(path=str(HERE/'owner.py'),sha256=OWNER_SHA),pins=pins),'actual owner/dependency bindings')
    support={str(DONOR_PATH):DONOR_SHA,
        str(HERE.parent/PREDECESSOR_FOLDER/'controller.py'):PREDECESSOR_CONTROLLER_SHA,
        str(HERE.parent/PREDECESSOR_FOLDER/'audit_restore.py'):PREDECESSOR_AUDITOR_SHA,
        str(_donor.STABLE_PATH):_donor.STABLE_SHA,str(_donor.FORMAL_PATH):_donor.FORMAL_SHA,
        _donor.STABLE_LIFECYCLE['path']:_donor.STABLE_LIFECYCLE['sha256'],str(BASE_PATH):BASE_SHA}
    for p,h in support.items():need(b.sha(p)==h,'frozen lifecycle source changed: '+p)
    pins.update(support);pins[str(Path(__file__).resolve())]=b.sha(__file__);return pins


def owner_plan(path,digest,journal):
    path=Path(path).resolve();owner=owner_module()
    _,plan,_,_,_=owner.validate_plan(SimpleNamespace(plan=path,plan_sha256=digest))
    need(plan['schema']==OWNER_SCHEMA and Path(plan['cold_restart']['journal_path']).resolve()==Path(journal).resolve(),'actual GDN plan/journal')
    need(plan['cold_restart']['controller']==dict(path=str(Path(__file__).resolve()),sha256=b.sha(__file__)),'real GDN controller source')
    source_pins(plan);return plan


def registered_trials(j):
    ref=j.data['owner_plan'];raw=b.read_json(ref['path'])
    plan=owner_plan(ref['path'],ref['sha256'],raw['cold_restart']['journal_path'])
    need(plan['trials']==j.data['trials'],'registered phase order changed');return plan['trials']


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
    j.data=dict(schema=JOURNAL_SCHEMA,id=str(uuid.uuid4()),state='PREPARED',
        plan_sha256=PLAN_SHA,controller_sha256=b.sha(__file__),support_sha256=BASE_SHA,original=identity,
        original_status=status,global_settings_stat=b.file_stat(b.GLOBAL_SETTINGS),model_metadata=metadata,
        log_start=s.log_identity(),owner_directory=str(owner),owner_plan={'path':str(Path(wrapper).resolve()),'sha256':wrapper_sha},
        owner_source=plan['execution']['owner'],trials=plan['trials'],predecessor=binding,predecessor_copies=copies,resource_samples=[],events=[])
    j.save('prepared_gdn_formal_read_only')


def authorize(j,approved_contract,plan_sha):
    need(approved_contract==CONTRACT and plan_sha==PLAN_SHA,'explicit approved contract/fixed plan required')
    need(j.data.get('schema')==JOURNAL_SCHEMA,'fresh GDN formal journal required; never migrate old journals')
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


def load_worker(journal,nonce):
    """Internal fixed-POST subprocess; durable claim forbids a second worker."""
    path=Path(journal).resolve();data=b.read_json(path/'journal.json');worker=data['load_worker']
    need(data.get('schema')==JOURNAL_SCHEMA and data.get('state')=='LOAD_INTENT'
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


def _bound(name,dependencies):
    original=getattr(_donor,name)
    # This separate dictionary is never written back to the imported donor.
    namespace=dict(original.__globals__);namespace.update(dependencies)
    result=FunctionType(original.__code__,namespace,original.__name__,original.__defaults__,original.__closure__)
    result.__kwdefaults__=original.__kwdefaults__;return result


stop=_bound('stop',dict(authorize=authorize))
phase_reaps=_bound('phase_reaps',dict(registered_trials=registered_trials))
reaped=_bound('reaped',dict(registered_trials=registered_trials,phase_reaps=phase_reaps))
recover=_bound('recover',dict(authorize=authorize,reaped=reaped,monitored_load=monitored_load))


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
