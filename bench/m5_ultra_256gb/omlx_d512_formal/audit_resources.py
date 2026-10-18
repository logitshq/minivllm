"""Offline oMLX six-trial raw resources/reap audit; no process or model action.

The unchanged, pinned Lithos formal raw-resource algorithm is reused under its
real source path. It already reads numeric indices correctly beyond 999. Only
this new owner/phase binding is added. No source-path or schema spoofing.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
OWNER_SHA='3d9a17b72ea748d002775b5467f1046076f283f646be0e24776d98005d6bdd90'
PHASES=['r1-c1','r1-c8','r2-c8','r2-c1','r3-c1','r3-c8']
DEPENDENCIES={
    '../lithos_d512_formal/audit_resources.py':'398ff7a52dba307418f4c2725f33988b1571d65b11e3f46c9e97759f3a167c9e',
    '../omlx_d512/audit_qualification.py':'e75a8a36b8fa53fcca0ff19c668092537e58c88c3257e4baa25f6235fba100f8',
}


def load(name):
    path=(HERE/name).resolve()
    if hashlib.sha256(path.read_bytes()).hexdigest()!=DEPENDENCIES[name]:
        raise ValueError('frozen audit dependency changed: '+name)
    spec=importlib.util.spec_from_file_location('_omlx_formal_resource_'+path.stem,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


legacy=load('../omlx_d512/audit_qualification.py')
raw=load('../lithos_d512_formal/audit_resources.py')
need,same,read,sha,relative,finite=legacy.need,legacy.same,legacy.read,legacy.sha,legacy.relative,legacy.finite
resources=raw.resources


def closed_bindings(root):
    """Actual canonical six-child evidence, before the raw resource sweep."""
    complete=read(root/'complete.json');attempt=read(root/'attempt.json');plan=read(root/'owner-plan.json')
    need(sha(HERE/'owner.py')==OWNER_SHA and len(OWNER_SHA)==64,'frozen formal owner changed/unsealed')
    need(complete['schema']=='E023-omlx-D512-formal-owner-result-v1' and complete['status']=='PASS'
         and complete['errors']==[] and complete['first_failure'] is None
         and complete['formal_trials_complete'] is True and complete['all_children_reaped'] is True
         and complete['planned_trials']==PHASES,'closed successful six-phase owner required')
    plan_sha=sha(root/'owner-plan.json');pid=complete['owner_pid']
    need(type(pid) is int and pid>1 and attempt['owner_pid']==pid and attempt['owner_identity']['pid']==pid
         and attempt['schema']=='cold-service-owner-attempt-v1'
         and complete['owner_plan_sha256']==attempt['owner_plan_sha256']==plan_sha,'actual owner/plan/attempt')
    need(plan['schema']=='E023-omlx-D512-formal-owner-plan-v1' and plan['trials']==PHASES
         and plan['execution']['owner']==attempt['owner_source']==dict(path=str(HERE/'owner.py'),sha256=OWNER_SHA),
         'formal owner source/plan')
    stopped=read(root/'stopped-journal.json');cold=plan['cold_restart']
    need(stopped['schema']=='E023-omlx-D512-formal-cold-journal-v1' and stopped['state']=='STOPPED'
         and stopped['owner_directory']==str(root) and stopped['owner_plan']['sha256']==plan_sha
         and stopped['owner_source']==attempt['owner_source'] and stopped['id']==attempt['journal_id']
         and stopped['removed_at']==attempt['removed_at'] and stopped['trials']==PHASES
         and stopped['predecessor']==cold['predecessor']
         and attempt['stopped_journal_sha256']==sha(root/'stopped-journal.json'),'real STOPPED binding')
    need(attempt['journal_path']==cold['journal_path'] and attempt['controller_sha256']==stopped['controller_sha256']==cold['controller']['sha256']
         and stopped['approved_contract']==cold['approved_contract']
         and stopped['plan_sha256']==cold['proposal_plan_sha256'],'real controller binding')
    index=read(root/'children.json');final=read(root/'release.json');spawn=read(root/'spawn-intent.json')
    need(index['schema']=='cold-service-owner-children-v1' and final['schema']=='cold-service-owner-release-v2'
         and final['children_sha256']==sha(root/'children.json') and final['all_children_reaped'] is True
         and final.get('no_popen_receipt') is None,'complete release/index')
    children=index['children'];same(final['children'],children,'final children differ')
    need([c['phase'] for c in children]==PHASES,'exact six child order')
    bound={k:attempt[k] for k in ('owner_pid','owner_identity','owner_source','owner_plan_sha256','journal_id','removed_at')}
    bound['attempt_sha256']=sha(root/'attempt.json')
    need(all(final.get(k)==v for k,v in bound.items()) and
         all(spawn.get(k)==bound[k] for k in ('owner_pid','owner_plan_sha256','journal_id','attempt_sha256')),'attempt release/intent binding')
    need(all(finite(t) for t in (attempt['removed_at'],attempt['created_at'],final['time']))
         and attempt['removed_at']<=attempt['created_at']<=final['time'],'owner terminal clock')
    previous=attempt['created_at'];seen=set();lives=[]
    for entry in children:
        phase=entry['phase'];folder=root/'phases'/phase
        intent=read(relative(root,entry['intent'],f'phases/{phase}/intent.json'))
        child=read(relative(root,entry['child'],f'phases/{phase}/child.json'))
        release=read(relative(root,entry['release'],f'phases/{phase}/release.json'))
        inner=read(folder/'trial/release.json');cpid=entry['pid'];identity=child['identity']
        need(all(intent.get(k)==bound[k] for k in ('owner_pid','owner_plan_sha256','journal_id','attempt_sha256'))
             and intent['phase']==phase,'phase intent binding')
        need(type(cpid) is int and cpid>1 and cpid!=pid and cpid not in seen,'unique direct server PID');seen.add(cpid)
        need(child['phase']==phase and child['pid']==cpid and child['owner_pid']==pid
             and type(identity) is dict and identity['pid']==cpid and identity['ppid']==pid
             and entry['identity']==identity and child['ownership']=='direct-Popen-returned','actual child identity')
        need(inner['pid']==release['pid']==cpid and release['phase']==phase
             and inner['waited_reaped'] is release['waited_reaped'] is True
             and inner['strict_absence'] is release['strict_absence'] is True and inner['port_absent'] is True,'actual child reap')
        legacy.explained_release(inner,release)
        need(all(finite(t) for t in (intent['time'],child['spawned_at'],inner['time'],release['time']))
             and previous<=intent['time']<=child['spawned_at']<=inner['time']<=release['time']<=final['time'],'sequential child clocks')
        previous=release['time']
        absent=read(relative(root,release['absence']))
        same(absent['command'],['/bin/ps','-p',str(cpid),'-o','pid='],'raw reaped PID query')
        need(type(absent['returncode']) is int and absent['returncode']==1 and absent['stdout']=='' and absent['stderr']=='','raw child strict absence')
        ports=list(folder.glob('port-absence-*.json'));need(bool(ports),'outer port absence')
        for path in ports:
            row=read(path);same(row['command'],['/usr/sbin/lsof','-nP','-a',f"-iTCP:{plan['http']['port']}",'-sTCP:LISTEN','-Fpn'],'actual port command')
            need(type(row['returncode']) is int and row['returncode']==1 and row['stdout']=='' and row['stderr']=='','actual port absence')
        lives.append(dict(phase=phase,pid=cpid,spawned_at=child['spawned_at'],released_at=release['time'],inner_released_at=inner['time']))
    expected={f'phases/{phase}/{name}' for phase in PHASES for name in
              ('intent.json','child.json','release.json','trial/spawn-intent.json','trial/child.json','trial/release.json')}
    found={str(p.relative_to(root)) for p in (root/'phases').rglob('*') if p.name in
           ('intent.json','child.json','release.json','spawn-intent.json','no-popen.json')}
    same(sorted(found),sorted(expected),'omitted or unresolved spawn evidence')
    return complete,plan,stopped,attempt,children,lives


def audit(owner):
    root=Path(owner).resolve()
    result=dict(schema='oMLX-D512-formal-resource-audit-v1',status='INCOMPLETE',
        independent_resource_verified=False,performance_qualified=False,restoration_verified=False,
        owner_directory=str(root),audit_source_sha256=sha(__file__))
    try:
        complete,plan,stopped,attempt,children,lives=closed_bindings(root)
        summary=resources(root,complete,lives,stopped['original']['pid'],attempt['created_at'],PHASES)
        result.update(status='PASS',independent_resource_verified=True,resource_summary=summary,
            owner_source_sha256=OWNER_SHA,owner_complete_sha256=sha(root/'complete.json'),
            owner_plan_sha256=complete['owner_plan_sha256'],resource_algorithm_sha256=DEPENDENCIES['../lithos_d512_formal/audit_resources.py'],
            scope='Unchanged raw pressure/AC/thermal/swap/inventory gates and six actual direct server lifetimes. No timing, quality or restoration qualification.')
    except FileNotFoundError as exc:result['error']=str(exc)
    except Exception as exc:result.update(status='FAIL',error_type=type(exc).__name__,error=str(exc))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--owner',required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();result=audit(args.owner)
    with args.out.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    raise SystemExit(0 if result['status']=='PASS' else 1)
