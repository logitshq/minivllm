"""Offline E024 v5 single-child fixed-service restoration audit; --journal DIR --out NEW_JSON.

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
CONTROLLER_SHA='6f61d28450d9c44401c2e5a4e6f8945ce82ec9430f56456476c7eda0cd4cd578'
OWNER_SHA='49e4ef7396a81c12c167b8a33075725c0d2af679d91d652b204fb408ddfda07b'
OWNER_TEST_SHA='962ec5981a1af5db34e56358f4eecbe3ea5dc0202f14082cd0e12eead479ea7c'
DIAGNOSTIC_OWNER_PLAN_SHA='4dff5f46d0b35d038ffbb0e32ebb6197b047f4da6e847b3160f3ae8685caf6a4'
DIAGNOSTIC_PLAN_SHA='5025d37768a5fa74d6e3e256a96ff8b8e4e92b31837183d5ffe031d9473e7123'
FROZEN_V4_AUDIT_SHA='af46699fcea1065c4fcf69a85225a916987fba484098d53ef2acb905eb4ae3a8'
RAW_AUDITOR_SHA='1b7b526dac98c43f5b77f915aec3173d755d1efb28ed49b989e7601293ddaf6d'
NUMERIC_PLAN_SHA='21ba52e43ce54c2b463ab46bfd212c53d06d6303530c77939e3e6b801df547fb'
SCOPE='Independent raw restoration evidence and fixed E024-cold-owner reap protocol only; no numerical or performance qualification, no current live observation.'


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


def sources():
    need(sha(HERE/'audit_restore_v4.py')==FROZEN_V4_AUDIT_SHA,'frozen v4 audit source changed')
    controller=load(HERE/'controller_v5.py',CONTROLLER_SHA,'_restore_v5_protocol')
    raw=load(HERE.parent/'pareto_realistic/audit_absent_qualification.py',RAW_AUDITOR_SHA,'_restore_v5_raw_audit')
    return controller,raw


def binding(data,journal,c,a):
    expected=dict(schema='omlx-cold-restart-journal-v5',state='RESTORED',plan_sha256=c.PLAN_SHA,
        approved_contract=c.CONTRACT,controller_sha256=CONTROLLER_SHA,support_sha256=c.BASE_SHA)
    need(all(data.get(k)==v for k,v in expected.items()),'fixed v5 restoration journal binding')
    need(not data.get('post_monitor_failed') and not any(e.get('event') in
        ('post_monitor_incomplete','post_worker_cleanup_failed') for e in data.get('events',[])),
        'incomplete monitored load/cleanup cannot become a predecessor')
    prior,_=c.predecessor(data['predecessor'])
    need(data['original']==prior['replacement'] and data['original_status']==prior['final_status']
         and data['global_settings_stat']==prior['global_settings_stat']
         and data['model_metadata']==prior['model_metadata'],'predecessor stable service binding')
    for name,ref in [('predecessor.json',data['predecessor']),('predecessor-audit.json',data['predecessor']['audit'])]:
        need(data['predecessor_copies'].get(name)==ref['sha256'] and sha(journal/name)==ref['sha256'],
             'frozen predecessor copy hash')
    c.fixed_identity(data['original']);c.fixed_identity(data['replacement'])
    need(data['replacement']['pid']!=data['original']['pid'],'replacement reused original PID')
    a.status_receipt(data['original_status'])
    plan_path=Path(data['owner_plan']['path']);need(plan_path.is_absolute() and sha(plan_path)==data['owner_plan']['sha256'],'owner plan SHA')
    plan=read(plan_path)
    need(set(plan)=={'schema','execution','diagnostic_owner_plan','cold_restart','trials','deadline_seconds'}
         and plan['schema']=='E024-cold-owner-plan-v1' and type(plan['deadline_seconds']) is int
         and plan['deadline_seconds']==1830,'fixed E024 cold owner plan')
    trials=c.trial_names(plan['trials']);need(data.get('trials')==trials,'journal/plan diagnostic phase')
    owner_path=HERE.parent/'long_decode_graph/cold_owner.py'
    expected_owner=dict(path=str(owner_path),sha256=OWNER_SHA)
    need(data['owner_source']==plan['execution']['owner']==expected_owner and sha(owner_path)==OWNER_SHA,
         'owner source identity')
    fixed=next(ast.literal_eval(n.value) for n in ast.parse(owner_path.read_text()).body
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='FIXED' for t in n.targets))
    pins={str((owner_path.parent/name).resolve()):digest for name,digest in fixed.items()}
    pins.update({str(owner_path):OWNER_SHA,str(owner_path.with_name('test_cold_owner_cpu.py')):OWNER_TEST_SHA})
    need(plan['execution']==dict(owner=expected_owner,pins=pins),'exact source dependency set')
    for path,digest in pins.items():need(sha(path)==digest,'owner dependency SHA changed')
    need(plan['cold_restart']==dict(journal_path=str(journal),controller=dict(path=str(HERE/'controller_v5.py'),sha256=CONTROLLER_SHA),
        approved_contract=c.CONTRACT,proposal_plan_sha256=c.PLAN_SHA,predecessor=data['predecessor']),'owner/controller/predecessor plan binding')
    def fixed_plan(item,digest):
        need(set(item)=={'path','sha256'} and item['sha256']==digest and Path(item['path']).is_absolute()
             and sha(item['path'])==digest,'unchanged E024 inherited plan identity')
        return read(item['path'])
    old=fixed_plan(plan['diagnostic_owner_plan'],DIAGNOSTIC_OWNER_PLAN_SHA)
    q=fixed_plan(old['qualification_plan'],DIAGNOSTIC_PLAN_SHA)
    fixed_plan(old['inherited_plan'],NUMERIC_PLAN_SHA)
    need(q['e023_plan']==old['inherited_plan'] and q['deadline_s']==1800
         and q['qualifier_sha256']==fixed['qualify.py'],'unchanged diagnostic qualifier/deadline binding')
    need(finite(data['removed_at']) and data['removed_at']>0 and type(data['id']) is str and data['id'],'journal lifecycle identity')
    return plan,pins


def no_child(data,owner,attempt,files,c):
    terminal=read(owner/'terminal.json');files.append(owner/'terminal.json')
    shared={k:attempt[k] for k in ('owner_pid','owner_identity','owner_source','owner_plan_sha256','journal_id','removed_at')}
    need(terminal.get('schema')=='cold-service-owner-terminal-v1' and terminal.get('status')=='NO_CHILD_STARTED'
         and all(terminal.get(k)==v for k,v in shared.items())
         and terminal.get('attempt_sha256')==sha(owner/'attempt.json')
         and terminal.get('spawn_intent_written') is False and terminal.get('child_started') is False
         and finite(terminal.get('terminal_at')) and terminal['terminal_at']>=attempt['created_at'],'no-child terminal proof')
    need(all(not (owner/name).exists() for name in ('spawn-intent.json','child.json','release.json')),'outer spawn evidence forbids no-child')
    if (owner/'children.json').exists():
        need(read(owner/'children.json')==dict(schema='cold-service-owner-children-v1',children=[]),'nonempty child index')
        files.append(owner/'children.json')
    found={str(p.relative_to(owner)) for p in (owner/'phases').rglob('*')
           if p.name in ('intent.json','child.json','release.json','spawn-intent.json','no-popen.json')}
    c.registered_trials(SimpleNamespace(data=data))
    need(not found and terminal.get('no_popen_receipt') is None,
         'single direct diagnostic cannot resolve nested spawn evidence')
    return terminal['terminal_at']


def owner_reap(data,journal,plan,c,a):
    owner=Path(data['owner_directory']);need(owner.is_absolute(),'absolute owner directory')
    if data['reap'].get('no_experiment') is True:
        need(data['reap']=={'no_experiment':True} and not owner.exists(),'no-experiment bypass')
        return data['removed_at'],dict(status='NO_EXPERIMENT',children=[])
    attempt=read(owner/'attempt.json');stopped=read(owner/'stopped-journal.json');owner_pid=attempt['owner_pid']
    a.identity(attempt['owner_identity'],pid=owner_pid)
    need(attempt['schema']=='cold-service-owner-attempt-v1' and attempt['phase']=='PRESPAWN'
         and attempt['journal_id']==data['id'] and attempt['journal_path']==str(journal)
         and attempt['controller_sha256']==CONTROLLER_SHA and attempt['owner_source']==data['owner_source']
         and attempt['owner_plan_sha256']==data['owner_plan']['sha256'] and attempt['removed_at']==data['removed_at']
         and attempt['stopped_journal_sha256']==sha(owner/'stopped-journal.json')
         and finite(attempt['created_at']) and attempt['created_at']>=data['removed_at'],'durable owner attempt binding')
    stable=('schema','id','removed_at','controller_sha256','support_sha256','plan_sha256','approved_contract',
            'original','original_status','global_settings_stat','model_metadata','log_start',
            'owner_directory','owner_source','owner_plan','predecessor','predecessor_copies','trials')
    need(stopped['state']=='STOPPED' and all(stopped.get(k)==data[k] for k in stable)
         and data['events'][:len(stopped['events'])]==stopped['events'],'STOPPED/final immutable prefix')
    need(sha(owner/'owner-plan.json')==data['owner_plan']['sha256'],'owner saved plan SHA')
    complete=read(owner/'complete.json')
    need(complete['schema']=='E024-cold-owner-result-v1' and complete['status'] in ('PASS','FAIL')
         and complete['owner_pid']==owner_pid and complete['owner_plan_sha256']==data['owner_plan']['sha256']
         and complete.get('planned_trials')==data['trials'],
         'owner terminal identity (numeric success is not required)')
    files=[owner/'attempt.json',owner/'stopped-journal.json'];children=[]
    terminal=owner/'terminal.json';nochild=terminal.exists() and read(terminal).get('status')=='NO_CHILD_STARTED'
    if nochild:
        released=no_child(data,owner,attempt,files,c)
    else:
        paths,pids=c.phase_reaps(SimpleNamespace(data=data),owner,attempt);files.extend(paths)
        children=read(owner/'children.json')['children'];released=read(owner/'release.json')['time']
        copies={**plan['execution']['pins'],str(HERE/'controller_v5.py'):CONTROLLER_SHA}
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
        frozen_v4_audit_source_sha256=FROZEN_V4_AUDIT_SHA)
    c=a=None
    try:
        data=read(path)
        if data.get('state')!='RESTORED':raise Incomplete('journal is not terminal RESTORED; no owner/ledger inspection')
        digest=sha(path);c,a=sources();plan,pins=binding(data,journal,c,a)
        released,reap=owner_reap(data,journal,plan,c,a)
        # A private identity adapter removes only the old hardcoded 74384
        # exclusion. The actual frozen physical/log/lifecycle checks are reused.
        original_identity=a.identity
        def replacement_identity(row,*,pid=None,service=False):
            original_identity(row,pid=pid,service=False)
            if service:
                need(row['ppid']==1 and row['comm']=='omlx-server'
                     and row['pid']!=data['original']['pid'],'current-cycle replacement identity')
        a.identity=replacement_identity
        restored=a.restore_evidence(data,journal,released)
        need(restored['http_worker_returncode']==0,'successful monitored POST worker required for next cycle')
        for p,h in pins.items():need(sha(p)==h,'source changed during offline audit')
        need(sha(path)==digest,'journal changed during offline audit')
        result.update(status='PASS',qualified=True,journal_sha256=digest,owner_plan_sha256=data['owner_plan']['sha256'],
            service_restore=restored,owner_reap=reap,controller_sha256=CONTROLLER_SHA,raw_auditor_sha256=RAW_AUDITOR_SHA,
            source_pins=pins,weight_shards_rehashed=False,
            adapter_scope='Frozen raw audit identity predicate uses current original PID; no source-path spoof or evidence rewriting.')
    except (Incomplete,FileNotFoundError) as error:result.update(status='INCOMPLETE',error=str(error))
    except Exception as error:
        incomplete=(a is not None and isinstance(error,a.Incomplete)) or (c is not None and isinstance(error,c.NotReady))
        result.update(status='INCOMPLETE' if incomplete else 'FAIL',error_type=type(error).__name__,error=str(error))
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--journal',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);args=p.parse_args(argv)
    need(not args.out.exists(),'new output file required');result=audit(args.journal)
    with args.out.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    return {'PASS':0,'FAIL':1,'INCOMPLETE':2}[result['status']]


if __name__=='__main__':raise SystemExit(main())
