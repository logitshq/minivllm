"""Stable single-child fixed-service restoration audit; --journal DIR --out NEW_JSON.

No numerical success is required. Raw physical/log/child-absence evidence is
checked independently; final management HTTP, owner absence and launchd/listener
observations remain source-pinned controller receipts, never current live facts.
No model, ledger, service, process probe, HTTP or GPU operation is performed.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

HERE=Path(__file__).resolve().parent
DIAGNOSTIC_OWNER_PLAN_SHA='4dff5f46d0b35d038ffbb0e32ebb6197b047f4da6e847b3160f3ae8685caf6a4'
DIAGNOSTIC_PLAN_SHA='5025d37768a5fa74d6e3e256a96ff8b8e4e92b31837183d5ffe031d9473e7123'
FROZEN_V2_AUDIT_SHA='01b01117e1b4dfac346b1231bb0e36528604fcb2e0d54c9e61461d59b88c2dec'
RAW_AUDITOR_SHA='1b7b526dac98c43f5b77f915aec3173d755d1efb28ed49b989e7601293ddaf6d'
NUMERIC_PLAN_SHA='21ba52e43ce54c2b463ab46bfd212c53d06d6303530c77939e3e6b801df547fb'
SCOPE='Independent raw restoration evidence and stable single-child owner reap protocol only; no numerical or performance qualification, no current live observation.'


class Incomplete(Exception):pass


def need(condition,message):
    if not condition:raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        while part:=stream.read(1<<20):h.update(part)
    return h.hexdigest()


def read(path):
    path=Path(path)
    if not path.is_file():raise Incomplete('missing evidence: '+str(path))
    need(path.suffix=='.json','only small JSON evidence is read')
    return json.loads(path.read_bytes())


def finite(value):return type(value) in (int,float) and math.isfinite(value)


def load(path,digest,name):
    need(sha(path)==digest,'frozen support source changed: '+str(path))
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module);return module


def sources(lifecycle_ref):
    # This audited entrypoint only loads the three adjacent reviewed source paths.
    need(set(lifecycle_ref)=={'path','sha256'} and Path(lifecycle_ref['path'])==HERE/'lifecycle.json'
         and sha(lifecycle_ref['path'])==lifecycle_ref['sha256'],'fixed reviewed lifecycle manifest')
    manifest=read(lifecycle_ref['path'])
    need(set(manifest)=={'schema','sources'} and manifest['schema']=='cold-single-child-lifecycle-v3'
         and set(manifest['sources'])=={'controller','owner','audit_restore'},'exact lifecycle source set')
    for name,item in manifest['sources'].items():
        need(set(item)=={'path','sha256'} and item['path']==str(HERE/(name+'.py'))
             and not Path(item['path']).is_symlink() and sha(item['path'])==item['sha256'],'reviewed lifecycle source changed')
    need(manifest['sources']['audit_restore']['sha256']==sha(__file__),'actual restoration auditor source')
    controller=load(HERE/'controller.py',manifest['sources']['controller']['sha256'],'_stable_cold_protocol')
    controller.lifecycle(lifecycle_ref)
    raw=load(HERE.parent/'pareto_realistic/audit_absent_qualification.py',RAW_AUDITOR_SHA,'_stable_cold_raw_audit')
    return controller,raw


# The frozen helpers take their actual protocol module as an argument. Do not
# call the old sources()/audit() entrypoints or mutate that module's globals.
frozen_v2=load(HERE.parent/'cold_cycle_v2/audit_restore.py',FROZEN_V2_AUDIT_SHA,'_cold_v3_frozen_binding_helpers')
binding=frozen_v2.binding
no_child=frozen_v2.no_child


def owner_reap(data,journal,plan,c,a):
    owner=Path(data['owner_directory']);need(owner.is_absolute(),'absolute owner directory')
    if data['reap'].get('no_experiment') is True:
        need(data['reap']=={'no_experiment':True} and not owner.exists(),'no-experiment bypass')
        return data['removed_at'],dict(status='NO_EXPERIMENT',children=[])
    attempt=read(owner/'attempt.json');stopped=read(owner/'stopped-journal.json');owner_pid=attempt['owner_pid']
    a.identity(attempt['owner_identity'],pid=owner_pid)
    need(attempt['schema']=='cold-service-owner-attempt-v1' and attempt['phase']=='PRESPAWN'
         and attempt['journal_id']==data['id'] and attempt['journal_path']==str(journal)
         and attempt['controller_sha256']==data['controller_sha256'] and attempt['owner_source']==data['owner_source']
         and attempt['owner_plan_sha256']==data['owner_plan']['sha256'] and attempt['removed_at']==data['removed_at']
         and attempt['stopped_journal_sha256']==sha(owner/'stopped-journal.json')
         and finite(attempt['created_at']) and attempt['created_at']>=data['removed_at'],'durable owner attempt binding')
    stable=('schema','id','removed_at','controller_sha256','support_sha256','plan_sha256','approved_contract',
            'original','original_status','global_settings_stat','model_metadata','log_start',
            'owner_directory','owner_source','owner_plan','predecessor','predecessor_copies','trials','lifecycle')
    need(stopped['state']=='STOPPED' and all(stopped.get(k)==data[k] for k in stable)
         and data['events'][:len(stopped['events'])]==stopped['events'],'STOPPED/final immutable prefix')
    complete=read(owner/'complete.json')
    need(complete['schema']=='cold-single-child-owner-result-v1' and complete['status'] in ('PASS','FAIL')
         and complete['owner_pid']==owner_pid and complete['owner_plan_sha256']==data['owner_plan']['sha256']
         and complete.get('planned_trials')==data['trials'] and complete['lifecycle']==data['lifecycle'],
         'owner terminal identity (numeric success is not required)')
    files=[owner/'attempt.json',owner/'stopped-journal.json'];children=[]
    terminal=owner/'terminal.json';nochild=terminal.exists() and read(terminal).get('status')=='NO_CHILD_STARTED'
    plan_copy=owner/'owner-plan.json'
    # An early caught pre-spawn failure may precede this copy. Existing bytes
    # must still match; every branch with a child must retain the plan copy.
    if plan_copy.exists() or not nochild:
        need(sha(plan_copy)==data['owner_plan']['sha256'],'owner saved plan SHA')
    if nochild:
        released=no_child(data,owner,attempt,files,c)
    else:
        paths,pids=c.phase_reaps(SimpleNamespace(data=data),owner,attempt);files.extend(paths)
        children=read(owner/'children.json')['children'];released=read(owner/'release.json')['time']
        copies=c.source_pins(plan,c.lifecycle(plan['lifecycle']))
        need(read(owner/'owner-source/manifest.json')==copies,'pre-spawn owner source manifest')
        for source,digest in copies.items():
            need(sha(owner/'owner-source'/source.lstrip('/'))==digest,'pre-spawn frozen source copy changed')
        for row,pid in zip(children,pids):
            release=read(owner/row['release']['path']);ref=release['absence']
            folder=owner/'phases'/row['phase'];raw_path=a.within(str(owner/ref['path']),folder)
            need(raw_path.name=='child-absence.json' and sha(raw_path)==ref['sha256'],'raw child absence binding')
            raw=read(raw_path);need(raw['command']==['/bin/ps','-p',str(pid),'-o','pid=']
                and type(raw['returncode']) is int and raw['returncode']==1 and type(raw['stdout']) is str
                and not raw['stdout'].strip(),'raw child strict absence')
    expected=dict(no_experiment=False,no_child_started=nochild,owner_pid=owner_pid,
                  receipts=[dict(path=str(p),sha256=sha(p)) for p in files])
    need(data['reap']==expected,'controller bound reap receipts differ from actual terminal chain')
    need(all(p.stat().st_mtime>=data['removed_at'] for p in files),'stale owner terminal files')
    return released,dict(status='NO_CHILD_STARTED' if nochild else 'ALL_OBSERVED_CHILDREN_REAPED',
        owner_status=complete['status'],owner_pid=owner_pid,children=[dict(phase=r['phase'],pid=r['pid']) for r in children],
        scope='Phase intents/releases and retained child ps bytes checked; owner absence at restoration is a pinned-controller receipt.')


def audit(journal):
    journal=Path(journal).resolve();path=journal/'journal.json'
    result=dict(schema='cold-service-restoration-audit-v1',status='INCOMPLETE',qualified=False,
        performance_qualified=False,numerical_qualified=False,ledger_opened=False,scope=SCOPE,audit_source_sha256=sha(__file__),
        frozen_v2_audit_source_sha256=FROZEN_V2_AUDIT_SHA)
    c=a=None
    try:
        data=read(path)
        if data.get('state')!='RESTORED':raise Incomplete('journal is not terminal RESTORED; no owner/ledger inspection')
        digest=sha(path);c,a=sources(data['lifecycle']);plan,pins=binding(data,journal,c,a)
        released,reap=owner_reap(data,journal,plan,c,a)
        # Unmodified raw helper retains its historical74384 exclusion. The
        # independently checked binding also rejects reuse of this cycle's PID.
        restored=a.restore_evidence(data,journal,released)
        need(restored['http_worker_returncode']==0,'successful monitored POST worker required for next cycle')
        for p,h in pins.items():need(sha(p)==h,'source changed during offline audit')
        need(sha(path)==digest,'journal changed during offline audit')
        result.update(status='PASS',qualified=True,journal_sha256=digest,owner_plan_sha256=data['owner_plan']['sha256'],
            service_restore=restored,owner_reap=reap,controller_sha256=data['controller_sha256'],lifecycle=data['lifecycle'],raw_auditor_sha256=RAW_AUDITOR_SHA,
            source_pins=pins,weight_shards_rehashed=False,
            adapter_scope='Unmodified frozen raw restoration audit plus current-cycle original/replacement inequality; historical74384 exclusion retained. No globals patch, source-path spoof or evidence rewriting.')
    except (Incomplete,FileNotFoundError) as error:result.update(status='INCOMPLETE',error=str(error))
    except Exception as error:
        incomplete=isinstance(error,frozen_v2.Incomplete) or (a is not None and isinstance(error,a.Incomplete)) or (c is not None and isinstance(error,c.NotReady))
        result.update(status='INCOMPLETE' if incomplete else 'FAIL',error_type=type(error).__name__,error=str(error))
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--journal',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);args=p.parse_args(argv)
    need(not args.out.exists(),'new output file required');result=audit(args.journal)
    with args.out.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    return {'PASS':0,'FAIL':1,'INCOMPLETE':2}[result['status']]


if __name__=='__main__':raise SystemExit(main())
