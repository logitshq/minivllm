"""Offline Lithos two-phase cold restoration audit; no numeric/performance claim.

Reuse the frozen raw physical/log/MTP/resident-PLE audit without patching its
module globals. Validate real new controller/owner sources and direct server
reap receipts, including failed first/second-phase pre-Popen rejection. No
service/HTTP/process/GPU operation or model payload hashing is performed.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

HERE=Path(__file__).resolve().parent
CONTROLLER_SHA='becfd425ba67cd42acbd172cc24ff15e9ba6f38bc2ae7009e68da8936bc89b92'
OWNER_SHA='a6796b153d558954899a46a96cd8ead828928e8627aea1ba8a8b6453ab74bddd'
FORMAL_AUDIT_PATH=HERE.parent/'omlx_cold_restart/audit_restore_v4.py'
FORMAL_AUDIT_SHA='af46699fcea1065c4fcf69a85225a916987fba484098d53ef2acb905eb4ae3a8'
RAW_PATH=HERE.parent/'pareto_realistic/audit_absent_qualification.py'
RAW_AUDITOR_SHA='1b7b526dac98c43f5b77f915aec3173d755d1efb28ed49b989e7601293ddaf6d'
SCOPE='Raw restoration physical/log/worker evidence and retained Lithos direct-server child ps/port bytes; owner absence and final HTTP/PID/job/listener are controller receipts. No numerical/performance or live observation claim.'


def load(path,digest,name):
    if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:raise ValueError('frozen audit support changed: '+str(path))
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


formal=load(FORMAL_AUDIT_PATH,FORMAL_AUDIT_SHA,'lithos_frozen_restore_helpers')
Incomplete,need,sha,read,finite=formal.Incomplete,formal.need,formal.sha,formal.read,formal.finite
# Pure argument-based NO_CHILD validator already covers nested no-Popen proofs.
no_child=formal.no_child


def sources():
    need(sha(HERE/'owner.py')==OWNER_SHA,'frozen Lithos owner changed')
    return (load(HERE/'controller.py',CONTROLLER_SHA,'lithos_restore_protocol'),
            load(RAW_PATH,RAW_AUDITOR_SHA,'lithos_original_raw_restore_audit'))


def binding(data,journal,c,a):
    expected=dict(schema=c.JOURNAL_SCHEMA,state='RESTORED',plan_sha256=c.PLAN_SHA,
        approved_contract=c.CONTRACT,controller_sha256=CONTROLLER_SHA,support_sha256=c.BASE_SHA)
    need(all(data.get(k)==v for k,v in expected.items()),'real Lithos restoration journal identity')
    need(not data.get('post_monitor_failed') and not any(e.get('event') in
        ('post_monitor_incomplete','post_worker_cleanup_failed') for e in data.get('events',[])),
        'incomplete monitored load/cleanup cannot pass')
    prior,_=c.predecessor(data['predecessor'])
    need(data['original']==prior['replacement'] and data['original_status']==prior['final_status']
         and data['global_settings_stat']==prior['global_settings_stat'] and data['model_metadata']==prior['model_metadata'],
         'predecessor full service identity/settings/model binding')
    for name,ref in [('predecessor.json',data['predecessor']),('predecessor-audit.json',data['predecessor']['audit'])]:
        need(data['predecessor_copies'].get(name)==ref['sha256'] and sha(journal/name)==ref['sha256'],'retained predecessor bytes')
    c.fixed_identity(data['original']);c.fixed_identity(data['replacement'])
    need(data['replacement']['pid']!=data['original']['pid'],'replacement reused current original PID')
    a.status_receipt(data['original_status'])
    ref=data['owner_plan'];plan=c.owner_plan(ref['path'],ref['sha256'],journal)
    need(data['owner_source']==plan['execution']['owner'] and data['owner_source']==dict(path=str(HERE/'owner.py'),sha256=OWNER_SHA)
         and data['trials']==plan['trials'] and data['predecessor']==plan['cold_restart']['predecessor'],
         'owner/source/phase/predecessor binding')
    need(finite(data['removed_at']) and data['removed_at']>0 and type(data['id']) is str and data['id'],'runtime journal identity')
    pins=c.source_pins(plan)
    pins.update({str(FORMAL_AUDIT_PATH):FORMAL_AUDIT_SHA,str(RAW_PATH):RAW_AUDITOR_SHA,str(Path(__file__).resolve()):sha(__file__)})
    return plan,pins


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
    need(complete['schema']=='E023-lithos-D512-owner-result-v1' and complete['status'] in ('PASS','FAIL')
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
        for source,digest in plan['execution']['pins'].items():
            need(sha(owner/'preflight/frozen-sources'/source.lstrip('/'))==digest,'pre-spawn frozen source copy changed')
        for row,pid in zip(children,pids):
            release=read(owner/row['release']['path']);ref=release['absence']
            folder=owner/'phases'/row['phase'];raw_path=a.within(str(owner/ref['path']),folder)
            need(raw_path.name.startswith('child-absence-') and sha(raw_path)==ref['sha256'],'raw child absence binding')
            raw=read(raw_path);need(raw['command']==['/bin/ps','-p',str(pid),'-o','pid=']
                and type(raw['returncode']) is int and raw['returncode']==1 and type(raw['stdout']) is str
                and not raw['stdout'].strip(),'raw child strict absence')
            if row['phase'] in data['trials']:
                port_path=raw_path.with_name(raw_path.name.replace('child-absence-','port-absence-'))
                port=read(port_path);need(port['command']==['/usr/sbin/lsof','-nP','-a',
                    f"-iTCP:{plan['http']['port']}",'-sTCP:LISTEN','-Fpn']
                    and type(port['returncode']) is int and port['returncode']==1
                    and type(port['stdout']) is str and not port['stdout'].strip(),'raw formal port absence')
    expected=dict(no_experiment=False,no_child_started=nochild,owner_pid=owner_pid,
                  receipts=[dict(path=str(p),sha256=sha(p)) for p in files])
    need(data['reap']==expected,'controller bound reap receipts differ from actual terminal chain')
    need(all(p.stat().st_mtime>=data['removed_at'] for p in files),'stale owner terminal files')
    return released,dict(status='NO_CHILD_STARTED' if nochild else 'ALL_OBSERVED_CHILDREN_REAPED',
        owner_status=complete['status'],owner_pid=owner_pid,children=[dict(phase=r['phase'],pid=r['pid']) for r in children],
        scope='Phase intents/releases and retained child ps/port bytes checked; owner absence at restoration is a pinned-controller receipt.')

def audit(journal):
    journal=Path(journal).resolve();path=journal/'journal.json'
    result=dict(schema='cold-service-restoration-audit-v1',status='INCOMPLETE',qualified=False,
        performance_qualified=False,numerical_qualified=False,ledger_opened=False,scope=SCOPE,audit_source_sha256=sha(__file__),
        frozen_formal_audit_source_sha256=FORMAL_AUDIT_SHA)
    c=a=None
    try:
        data=read(path)
        if data.get('state')!='RESTORED':raise Incomplete('journal is not terminal RESTORED; no owner/ledger inspection')
        digest=sha(path);c,a=sources();plan,pins=binding(data,journal,c,a)
        released,reap=owner_reap(data,journal,plan,c,a)
        restored=a.restore_evidence(data,journal,released)
        need(restored['http_worker_returncode']==0,'successful monitored POST worker required for next cycle')
        for p,h in pins.items():need(sha(p)==h,'source changed during offline audit')
        need(sha(path)==digest,'journal changed during offline audit')
        result.update(status='PASS',qualified=True,journal_sha256=digest,owner_plan_sha256=data['owner_plan']['sha256'],
            service_restore=restored,owner_reap=reap,controller_sha256=CONTROLLER_SHA,raw_auditor_sha256=RAW_AUDITOR_SHA,
            source_pins=pins,weight_shards_rehashed=False,
            adapter_scope='Unmodified frozen raw audit plus current-cycle original/replacement inequality; historical74384 exclusion retained. No global patch, source-path spoof or evidence rewriting.')
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
