"""Six-phase oMLX D512 formal retry adapter; no actions on import or --help.

OS identity/physical/resource primitives are directly reused from the frozen
stable controller. Only protocol/source-bound orchestration bodies are local:
no imported module globals, __file__, HERE or journal bytes are rewritten.
The original authorization, MTP/resident-PLE restoration and unknown-load rules
remain. Six directly owned server phases are proved reaped before recovery.
CLI: prepare --journal NEW --owner-directory NEW --owner-plan PATH
--owner-plan-sha256 SHA --plan-sha256 PROPOSAL; stop/recover additionally require
--approved-contract CONTRACT. load-worker is the nonce-bound internal POST only.
Performance success is separate: even a killed/failed experiment must be
reaped before recovery. A request seal never substitutes for process termination.
"""
import argparse
import ast
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
import uuid

HERE=Path(__file__).resolve().parent
STABLE_PATH=HERE.parent/'cold_cycle/controller.py'
STABLE_SHA='b5237c458afa8324ffa6d5e6a5c49167578daec7425ff036112ec8e108d61213'
FORMAL_PATH=HERE.parent/'omlx_cold_restart/controller_v4.py'
FORMAL_SHA='836a4c16534521441da8021df3f963210d885da867f79a33d86a3b6dd01101c0'
STABLE_LIFECYCLE=dict(path=str(HERE.parent/'cold_cycle/lifecycle.json'),sha256='d9417e51a781275f18c296e20ac98874e61b8ce17b8bf0902ba63e27b5b2b0a8')
ARCHIVE_ROOT=HERE.parents[2]/'docs/experiments/m5-ultra-256gb'
PREDECESSOR_SCHEMA='E023-omlx-D512-formal-cold-journal-v1'
PREDECESSOR_FOLDER='omlx_d512_formal'
PREDECESSOR_CONTROLLER_SHA='5fcd7fd9fd3d85ca7e5de697971758e314474ca7f2548df54fe840e678fed333'
PREDECESSOR_AUDITOR_SHA='328a4241f034c69dcf18c97d1f95e22492cb1063b0f87c69174a8059e35fc538'
PREDECESSOR_JOURNAL_SHA='0b4886ab28a5e4c0a6bf3d4228ae9da267d89c566f3a4f4bdf43e49120966a64'
PREDECESSOR_AUDIT_SHA='831e8a1b3b07912f3bfe75dceb84cfaf6ae84c798564e4e40574126b972ffd1b'
PREDECESSOR_ARCHIVE=ARCHIVE_ROOT/'E023-omlx-d512-formal-v1'
PHASES=('r1-c1','r1-c8','r2-c8','r2-c1','r3-c1','r3-c8')
JOURNAL_SCHEMA='E023-omlx-D512-formal-cold-journal-v1'
OWNER_SCHEMA='E023-omlx-D512-formal-owner-plan-v1'
OWNER_SHA='fd9b0131cda5edb553b360ed2a124e878d12435d9540630454cbfa9b5f35f1f0'


def load(path,digest,name):
    if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:raise ValueError('frozen lifecycle support changed: '+str(path))
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


stable=load(STABLE_PATH,STABLE_SHA,'omlx_formal_retry_frozen_cold_primitives')
formal=load(FORMAL_PATH,FORMAL_SHA,'omlx_formal_retry_frozen_formal_proof')
b=stable.b
PLAN_SHA,CONTRACT,GIB,BASE_PATH,BASE_SHA=stable.PLAN_SHA,stable.CONTRACT,stable.GIB,stable.BASE_PATH,stable.BASE_SHA
Refuse,NotReady,need=stable.Refuse,stable.NotReady,stable.need
Journal,System=stable.Journal,stable.System
fixed_identity,original_identity=stable.fixed_identity,stable.original_identity
absent_service,replacement=stable.absent_service,stable.replacement
atomic_bytes,exclusive_json=stable.atomic_bytes,stable.exclusive_json
parse_resources,sample,check_resource,resource_gate=stable.parse_resources,stable.sample,stable.check_resource,stable.resource_gate
# Pure validator: all varying phase/identity values are arguments, no v4 enum or source identity is consulted.
no_popen_proof=formal.no_popen_proof


def predecessor(binding):
    """Only the exact independently restored failed formal-v1 archive.

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
         and Path(binding['audit']['path'])==PREDECESSOR_ARCHIVE/'restoration-audit.json'
         and binding['sha256']==PREDECESSOR_JOURNAL_SHA and binding['audit']['sha256']==PREDECESSOR_AUDIT_SHA,
         'exact archived oMLX formal-v1 restoration predecessor required')
    controller_sha,auditor_sha=PREDECESSOR_CONTROLLER_SHA,PREDECESSOR_AUDITOR_SHA
    need(prior.get('schema')==PREDECESSOR_SCHEMA
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
    need(type(value) is list and value==list(PHASES),'exact six alternating oMLX formal phases required')
    return value


def source_pins(plan):
    owner=plan['execution']['owner'];path=Path(owner['path'])
    need(set(owner)=={'path','sha256'} and path==HERE/'owner.py' and not path.is_symlink()
         and b.sha(path)==owner['sha256']==OWNER_SHA,'actual frozen adjacent oMLX owner source')
    assignments={target.id:node.value for node in ast.parse(path.read_text()).body
        if isinstance(node,ast.Assign) for target in node.targets if isinstance(target,ast.Name)}
    fixed=ast.literal_eval(assignments['FIXED']);local=ast.literal_eval(assignments['LOCAL'])
    auditors=ast.literal_eval(assignments['REQUIRED_AUDIT_SOURCES'])
    need(local==('formal.py',)
         and auditors==('audit_formal.py','audit_resources.py','audit_restore.py'),'actual reviewed local dependency roles')
    pins={str((HERE/name).resolve()):digest for name,digest in fixed.items()}
    for name in (*local,*auditors):
        dependency=HERE/name
        need(dependency.is_file() and not dependency.is_symlink(),'required real local source missing: '+name)
        if name in fixed: need(b.sha(dependency)==fixed[name],'fixed local source changed: '+name)
        pins[str(dependency)]=b.sha(dependency)
    pins[str(path)]=owner['sha256']
    need(plan['execution']==dict(owner=owner,pins=pins),'exact oMLX owner dependency set')
    for name,digest in pins.items():need(b.sha(name)==digest,'owner dependency changed: '+name)
    support={str(STABLE_PATH):STABLE_SHA,str(FORMAL_PATH):FORMAL_SHA,str(BASE_PATH):BASE_SHA,
             STABLE_LIFECYCLE['path']:STABLE_LIFECYCLE['sha256'],
             str(HERE.parent/PREDECESSOR_FOLDER/'controller.py'):PREDECESSOR_CONTROLLER_SHA,
             str(HERE.parent/PREDECESSOR_FOLDER/'audit_restore.py'):PREDECESSOR_AUDITOR_SHA}
    for name,digest in support.items():need(b.sha(name)==digest,'frozen lifecycle support changed: '+name)
    pins.update(support);pins[str(Path(__file__).resolve())]=b.sha(__file__)
    return pins


def owner_plan(path,digest,journal):
    path=Path(path).resolve();need(b.sha(path)==digest,'owner wrapper plan SHA')
    plan=b.read_json(path)
    need(set(plan)=={'schema','execution','qualification','cold_restart','http','verification','trials','deadline_seconds'}
         and plan['schema']==OWNER_SCHEMA,'exact oMLX owner plan required')
    source_pins(plan);cold=plan['cold_restart'];trial_names(plan['trials'])
    need(set(cold)=={'journal_path','controller','approved_contract','proposal_plan_sha256','predecessor'}
         and Path(cold['journal_path']).resolve()==Path(journal).resolve(),'owner journal binding')
    need(cold['controller']==dict(path=str(Path(__file__).resolve()),sha256=b.sha(__file__)),'real oMLX controller source')
    need(cold['approved_contract']==CONTRACT and cold['proposal_plan_sha256']==PLAN_SHA,'unchanged approved contract')
    predecessor(cold['predecessor'])
    q=plan['qualification']
    need(type(q) is dict and set(q)=={'owner_directory','owner_plan','audit','resource_audit','restoration_audit'},
         'actual C1/C8 qualification binding required')
    need(Path(q['owner_directory']).is_absolute(),'absolute qualification owner directory')
    for role in ('owner_plan','audit','resource_audit','restoration_audit'):
        ref=q[role]
        need(set(ref)=={'path','sha256'} and Path(ref['path']).is_absolute()
             and b.sha(ref['path'])==ref['sha256'],'qualification receipt/source SHA: '+role)
    qualified=b.read_json(q['owner_plan']['path']);numeric=qualified['qualification']
    need(qualified['schema']=='E023-omlx-D512-owner-plan-v1'
         and qualified['verification']==plan['verification']
         and numeric['sha256']==b.sha(numeric['path'])=='21ba52e43ce54c2b463ab46bfd212c53d06d6303530c77939e3e6b801df547fb',
         'same qualified dependency groups and original fixture/resource plan')
    d=plan['http']['deadlines'];values=[d[k] for k in ('overall','close','term','kill')]
    need(all(type(x) in (int,float) and math.isfinite(x) and x>0 for x in values)
         and d['overall']<=3600 and d['close']<=15 and d['term']==d['kill']==15
         and type(d['resource_interval']) in (int,float) and 0<d['resource_interval']<=5
         and type(plan['deadline_seconds']) is int and plan['deadline_seconds']==math.ceil(300+6*(sum(values)+12)),
         'registered bounded six-phase work/cleanup budget')
    need(type(plan['http']['port']) is int and 1024<=plan['http']['port']<=65535,'registered experiment port')
    return plan


def registered_trials(j):
    ref=j.data['owner_plan'];need(b.sha(ref['path'])==ref['sha256'],'owner wrapper plan SHA')
    raw=b.read_json(ref['path']);plan=owner_plan(ref['path'],ref['sha256'],raw['cold_restart']['journal_path'])
    need(plan['trials']==j.data['trials'],'registered phase order changed')
    return plan['trials']


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
    j.save('prepared_omlx_read_only')

def authorize(j,approved_contract,plan_sha):
    need(approved_contract==CONTRACT and plan_sha==PLAN_SHA,'explicit approved contract/fixed plan required')
    need(j.data.get('schema')==JOURNAL_SCHEMA,'fresh oMLX journal required; never migrate old journals')
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
    """Validate the registered formal trials' complete observed child prefix."""
    planned=registered_trials(j)
    paths=[directory/name for name in ('spawn-intent.json','children.json','release.json')]
    if not all(path.exists() for path in paths):raise NotReady('formal owner spawn/reap outcome unresolved')
    intent,index,final=map(b.read_json,paths);pid=attempt['owner_pid']
    binding=dict(owner_pid=pid,owner_identity=attempt['owner_identity'],
        attempt_sha256=b.sha(directory/'attempt.json'),owner_plan_sha256=j.data['owner_plan']['sha256'],
        journal_id=j.data['id'],owner_source=j.data['owner_source'],removed_at=j.data['removed_at'])
    need(all(intent.get(k)==binding[k] for k in ('owner_pid','attempt_sha256','owner_plan_sha256','journal_id')),
         'formal spawn intent binding')
    need(index.get('schema')=='cold-service-owner-children-v1' and type(index.get('children')) is list,
         'formal children index required')
    children=index['children']
    need(final.get('schema')=='cold-service-owner-release-v2' and all(final.get(k)==v for k,v in binding.items())
         and final.get('children')==children and final.get('children_sha256')==b.sha(paths[1])
         and final.get('all_children_reaped') is True,'formal final release binding')
    phases=[row.get('phase') for row in children]
    need(0<len(phases)<=len(planned) and phases==planned[:len(phases)],'formal phase order/prefix')
    need(type(final.get('time')) in (int,float) and math.isfinite(final['time'])
         and final['time']>=attempt['created_at'],'formal final release time')
    need(not (directory/'child.json').exists(),'legacy single-child evidence forbidden for formal owner')
    expected=set();pids=[];last=attempt['created_at']
    for row in children:
        phase=row['phase'];payloads={}
        for kind in ('intent','child','release'):
            relative=f'phases/{phase}/{kind}.json';ref=row.get(kind,{})
            need(ref.get('path')==relative,'formal phase receipt path')
            path=directory/relative;expected.add(relative)
            need(path.is_file() and not path.is_symlink() and b.sha(path)==ref.get('sha256'),'formal phase receipt SHA')
            paths.append(path);payloads[kind]=b.read_json(path)
        pi,child,release=(payloads[k] for k in ('intent','child','release'));cpid=child.get('pid')
        need(all(pi.get(k)==binding[k] for k in ('owner_pid','attempt_sha256','owner_plan_sha256','journal_id'))
             and pi.get('phase')==phase,'formal phase intent binding')
        need(type(pi.get('time')) in (int,float) and math.isfinite(pi['time']) and pi['time']>=last,
             'formal phase intent time')
        identity=child.get('identity')
        identity_known=(type(identity) is dict and type(identity.get('pid')) is int and identity['pid']==cpid
             and type(identity.get('ppid')) is int and identity['ppid']==pid
             and isinstance(identity.get('start'),str) and bool(identity['start'])
             and isinstance(identity.get('comm'),str) and bool(identity['comm']))
        known_popen=(identity is None and 'identity' in child and child.get('ownership')=='direct-Popen-returned')
        need(type(cpid) is int and cpid>1 and cpid!=pid and cpid not in pids
             and child.get('owner_pid')==pid and child.get('phase')==phase
             and (identity_known or known_popen) and 'identity' in row
             and row.get('pid')==cpid and row['identity']==identity,'formal direct child identity')
        need(release.get('pid')==cpid and type(release.get('returncode')) is int
             and release.get('waited_reaped') is True and release.get('strict_absence') is True
             and type(release.get('time')) in (int,float) and math.isfinite(release['time'])
             and pi['time']<=release['time']<=final['time'],'formal phase child not proven reaped')
        last=release['time'];pids.append(cpid)
        # The pinned trial helper records this same directly owned Popen child.
        # Its cleanup can fail before the outer owner successfully reaps it.
        for name in ('spawn-intent.json','child.json','release.json'):
            relative=f'phases/{phase}/trial/{name}';path=directory/relative
            if not path.exists():continue
            need(path.is_file() and not path.is_symlink(),'trial evidence must be a regular file')
            nested=b.read_json(path)
            if name!='spawn-intent.json':
                need(type(nested.get('pid')) is int and nested['pid']==cpid,'trial nested PID differs from owned child')
            else:
                need(type(nested.get('time')) in (int,float) and math.isfinite(nested['time'])
                     and attempt['created_at']<=nested['time']<=pi['time'],'trial spawn intent time')
            paths.append(path);expected.add(relative)
    proof=final.get('no_popen_receipt')
    if proof is not None:
        need(len(phases)<len(planned),'no trial remains for pre-Popen proof')
        extra=no_popen_proof(j,directory,attempt,planned[len(phases)],last,final['time'],proof)
        paths+=extra;expected.update(str(p.relative_to(directory)) for p in extra)
    found={str(path.relative_to(directory)) for path in (directory/'phases').rglob('*')
           if path.name in ('intent.json','child.json','release.json','spawn-intent.json','no-popen.json')}
    need(found==expected,'unresolved or omitted formal phase evidence')
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
        phase=registered_trials(j)[0]
        if found or terminal.get('no_popen_receipt') is not None:
            extra=no_popen_proof(j,directory,a,phase,a['created_at'],terminal['terminal_at'],
                                 terminal.get('no_popen_receipt'))
            need(found=={str(p.relative_to(directory)) for p in extra},
                 'phase spawn evidence forbids NO_CHILD_STARTED')
            files+=extra
        identities=[pid]
    else:
        paths,children=phase_reaps(j,directory,a)
        files+=paths;identities=[pid]+children
    need(all(path.stat().st_mtime>=j.data['removed_at'] for path in files),'stale owner receipts')
    if any(s.process(p) is not None for p in identities):raise NotReady('owner or child exists, including PID reuse')
    return dict(no_experiment=False,no_child_started=nochild,owner_pid=pid,
                receipts=[dict(path=str(path),sha256=b.sha(path)) for path in files])

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
