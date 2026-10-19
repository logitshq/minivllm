"""Retry-only ownership binding for the unchanged combined formal factory.

The frozen v1 auditor supplies the exact original 864-request wire/count/window,
observer-OFF, tokenizer, resource and pairing algorithms. This version changes
only the true retry lifecycle source paths and owner-plan schema. Original failed
v1 evidence is retained; this module neither starts services nor grants partial
performance eligibility. The factory remains at its original path and SHA.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

HERE=Path(__file__).resolve().parent
NATIVE=HERE.parent/'pareto_realistic'
DONOR=HERE.parent/'gdn_graph_combined_formal/audit_formal.py'
DONOR_SHA='b1956036eb7c6df3939e68e2ba5b69b965bfb80951c694107325adb6feed3e5f'
OWNER_SHA='f7f44001d11533aa6106d3e5dc5c2635dfb736433f10e7ea680a5a83f2863d9e'
CONTROLLER_SHA='ca441479a0d2049bcb4daa5a8009b26824b7557a31be8c5c1c2ca57e01ce33cd'
RESTORE_SHA='7ab477f0d449bf08f78259a820d067f902d2032314ff4ebd389157b2578ec57d'
FORMAL=HERE.parent/'gdn_graph_combined_formal/formal.py'
FORMAL_SHA='4b567f8ed6090175f3639aa109584ca26c1426165c03de3167c3ebde74891a89'

def frozen(path,digest):
    if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:raise ValueError('fixed audit support changed: '+str(path))
    spec=importlib.util.spec_from_file_location('_combined_formal_audit_'+path.stem,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

donor=frozen(DONOR,DONOR_SHA)
Incomplete,need,same,finite,read,sha,ref,load=(donor.Incomplete,donor.need,donor.same,donor.finite,donor.read,donor.sha,donor.ref,donor.load)
PHASES=donor.PHASES
SUPPORT,SUPPORT_SHA,FROZEN_V1,FROZEN_V1_SHA=donor.SUPPORT,donor.SUPPORT_SHA,donor.FROZEN_V1,donor.FROZEN_V1_SHA
RAW_SHA,RESOURCE_SHA,LEGACY_SHA=donor.RAW_SHA,donor.RESOURCE_SHA,donor.LEGACY_SHA
support,pinned,relative,outer_absence,inner_absence=(donor.support,donor.pinned,donor.relative,donor.outer_absence,donor.inner_absence)
adapter_copies,preparation_pins,tokenizer_reference=(donor.adapter_copies,donor.preparation_pins,donor.tokenizer_reference)
raw_metrics,resources,paired_summary,input_inventory,legacy=(donor.raw_metrics,donor.resources,donor.paired_summary,donor.input_inventory,donor.legacy)
# Direct functions preserve their original globals and actual factory path.
observer_identity,audit_trials=donor.observer_identity,donor.audit_trials

def owner_binding(root,plan,complete):
    """Closed real12 canonical intents/children/reaps, no live OS observation."""
    attempt=read(root/'attempt.json');index=read(root/'children.json');release=read(root/'release.json')
    stopped=read(root/'stopped-journal.json');spawn=read(root/'spawn-intent.json');digest=sha(root/'owner-plan.json')
    source=dict(path=str(HERE/'owner.py'),sha256=OWNER_SHA)
    need(attempt['schema']=='cold-service-owner-attempt-v1' and type(complete['owner_pid']) is int
         and complete['owner_pid']>1 and attempt['owner_pid']==complete['owner_pid']
         and attempt['owner_identity']['pid']==complete['owner_pid'],'owner identity/attempt')
    need(plan['schema']=='GDN-graph-combined-http-formal-owner-plan-v2' and plan['trials']==list(PHASES)
         and plan['execution']['owner']==attempt['owner_source']==source
         and complete['owner_plan_sha256']==attempt['owner_plan_sha256']==digest,'real plan/source')
    cold=plan['cold_restart']
    need(stopped['state']=='STOPPED' and stopped['owner_directory']==str(root)
         and stopped['owner_plan']['sha256']==digest and stopped['owner_source']==source
         and stopped['id']==attempt['journal_id'] and stopped['removed_at']==attempt['removed_at']
         and stopped['trials']==list(PHASES) and stopped['predecessor']==cold['predecessor']
         and attempt['stopped_journal_sha256']==sha(root/'stopped-journal.json')
         and attempt['journal_path']==cold['journal_path']
         and attempt['controller_sha256']==stopped['controller_sha256']==cold['controller']['sha256']==CONTROLLER_SHA,
         'actual stopped/controller binding')
    bound={k:attempt[k] for k in ('owner_pid','owner_identity','owner_source','owner_plan_sha256','journal_id','removed_at')}
    bound['attempt_sha256']=sha(root/'attempt.json')
    need(index['schema']=='cold-service-owner-children-v1' and release['schema']=='cold-service-owner-release-v2'
         and release['children_sha256']==sha(root/'children.json') and release['all_children_reaped'] is True
         and release.get('no_popen_receipt') is None,'complete final release')
    same(release['children'],index['children'],'final children');need([r['phase'] for r in index['children']]==list(PHASES),'all12 direct children')
    need(all(release.get(k)==v for k,v in bound.items()) and all(spawn.get(k)==bound[k] for k in
        ('owner_pid','owner_plan_sha256','journal_id','attempt_sha256')),'owner intent/release bindings')
    need(finite(attempt['created_at']) and finite(release['time']) and attempt['removed_at']<=attempt['created_at']<=release['time'],'owner clock')
    previous=attempt['created_at'];seen=set();lives=[];proofs={}
    for row in index['children']:
        phase=row['phase'];folder=root/'phases'/phase
        intent=read(relative(root,row['intent'],f'phases/{phase}/intent.json'))
        child=read(relative(root,row['child'],f'phases/{phase}/child.json'))
        end=read(relative(root,row['release'],f'phases/{phase}/release.json'));inner=read(folder/'trial/release.json')
        pid=row['pid'];identity=child['identity']
        need(type(pid) is int and pid>1 and pid!=complete['owner_pid'] and pid not in seen,'unique actual server PID');seen.add(pid)
        need(all(intent.get(k)==bound[k] for k in ('owner_pid','owner_plan_sha256','journal_id','attempt_sha256')) and intent['phase']==phase,'phase intent')
        need(child['phase']==phase and child['pid']==pid and child['owner_pid']==complete['owner_pid']
             and type(identity) is dict and identity['pid']==pid and identity['ppid']==complete['owner_pid']
             and row['identity']==identity and child['ownership']=='direct-Popen-returned','actual direct child identity')
        need(end['phase']==phase and end['pid']==inner['pid']==pid and end['returncode']==inner['returncode']
             and type(end['returncode']) is int and end['waited_reaped'] is inner['waited_reaped'] is True
             and end['strict_absence'] is inner['strict_absence'] is inner['port_absent'] is True,'inner/outer actual reap')
        need(all(finite(t) for t in (intent['time'],child['spawned_at'],inner['time'],end['time']))
             and previous<=intent['time']<=child['spawned_at']<=inner['time']<=end['time']<=release['time'],'sequential phase clock')
        previous=end['time'];outer_absence(read(relative(root,end['absence'])),['/bin/ps','-p',str(pid),'-o','pid='])
        ports=list(folder.glob('port-absence-*.json'));need(ports,'outer listener closure retained')
        port=plan['http']['transport']['port']
        for path in ports:outer_absence(read(path),['/usr/sbin/lsof','-nP','-a',f'-iTCP:{port}','-sTCP:LISTEN','-Fpn'])
        proofs[phase]=inner_absence(folder/'trial',pid,port)
        lives.append(dict(phase=phase,pid=pid,spawned_at=child['spawned_at'],released_at=end['time'],
                          returncode=end['returncode'],inner_released_at=inner['time']))
    expected={f'phases/{phase}/{name}' for phase in PHASES for name in
              ('intent.json','child.json','release.json','trial/spawn-intent.json','trial/child.json','trial/release.json')}
    found={str(p.relative_to(root)) for p in (root/'phases').rglob('*') if p.name in
           ('intent.json','child.json','release.json','spawn-intent.json','no-popen.json')}
    same(sorted(found),sorted(expected),'omitted/unknown spawn evidence')
    return attempt,stopped,lives,proofs

def audit(owner,journal,plan_sha256):
    owner,journal=Path(owner).resolve(),Path(journal).resolve()
    result=dict(schema='GDN-graph-combined-http-formal-independent-audit-v2',status='INCOMPLETE',qualified=False,
        performance_qualified=False,ledger_opened=False,http_raw_opened=False,actual_prework_calls_observed=False,actual_compiled_invocations_observed=False,route_always_on=True,
        owner_directory=str(owner),audit_source_sha256=sha(__file__),
        reused_audit_source_sha256=DONOR_SHA,unchanged_factory_source=dict(path=str(FORMAL),sha256=FORMAL_SHA),raw_algorithm_source_sha256=SUPPORT_SHA,
        frozen_route_v1_audit_source_sha256=FROZEN_V1_SHA,
        scope='Actual route-only and combined captured histories are prerequisites. Both arms keep graph ON; only GDN switches. Rechecks864 wire requests and ownership/resources/restoration; no unseen-schedule full-state proof, formal graph-hit observations or general quality claim.')
    try:
        need(sha(DONOR)==DONOR_SHA and sha(SUPPORT)==SUPPORT_SHA and sha(FROZEN_V1)==FROZEN_V1_SHA,'frozen audit algorithms changed')
        same(list(support().PHASES),list(PHASES),'same reused twelve-phase algorithms')
        data=read(journal/'journal.json');complete=read(owner/'complete.json')
        if data.get('state')!='RESTORED' or complete.get('status')=='INCOMPLETE':raise Incomplete('owner/restoration not closed')
        need(data['owner_directory']==str(owner) and data['owner_plan']['sha256']==plan_sha256==sha(owner/'owner-plan.json'),
             'actual owner plan binding')
        for name,digest in (('owner.py',OWNER_SHA),('controller.py',CONTROLLER_SHA),('audit_restore.py',RESTORE_SHA)):
            need(len(digest)==64 and sha(HERE/name)==digest,'lifecycle source must be independently sealed')
        need(data['owner_source']==dict(path=str(HERE/'owner.py'),sha256=OWNER_SHA),'actual owner source')
        restore=load(HERE/'audit_restore.py',RESTORE_SHA,'_gdn_formal_restore')
        restoration=restore.audit(journal);result['restoration']=restoration
        need(restoration['status']=='PASS' and restoration['qualified'] is True
             and restoration['performance_qualified'] is False and restoration['owner_plan_sha256']==plan_sha256
             and restoration['journal_sha256']==sha(journal/'journal.json'),'independent restoration/reap')
        if complete.get('status')!='PASS':
            result.update(status='FAIL',producer_errors=complete.get('errors'),partial_points_unqualified=True);return result
        initial=input_inventory(owner);journal_sha=sha(journal/'journal.json')
        owned=load(HERE/'owner.py',OWNER_SHA,'_gdn_formal_owner_recipe')
        _,wrapper,_,numeric,specs=owned.validate_plan(SimpleNamespace(plan=owner/'owner-plan.json',plan_sha256=plan_sha256))
        need(complete['schema']=='GDN-graph-combined-http-formal-owner-result-v1' and complete['errors']==[]
             and complete['first_failure'] is None and complete['formal_trials_complete'] is True
             and complete['all_children_reaped'] is True and complete['performance_qualified'] is False
             and complete['planned_trials']==list(PHASES)==wrapper['trials'],'complete registered12 cycle')
        attempt,stopped,lives,absence_proofs=owner_binding(owner,wrapper,complete)
        same(stopped['original'],data['original'],'same removed original service')
        result['pins']=preparation_pins(owner,wrapper,numeric)
        # The original count qualification pins the exact CPU tokenizer runtime.
        audits={C:read(pinned(next(s for s in specs if s['concurrency']==C)['baseline_qualification']['audit'])) for C in (1,8)}
        tokens=audits[1]['tokenizer_runtime_pins'];same(tokens,audits[8]['tokenizer_runtime_pins'],'C1/C8 tokenizer runtime')
        import tokenizers
        need(str(Path(tokenizers.__file__).resolve()) in tokens,'actual CPU decoder module')
        for path,digest in tokens.items():need(sha(path)==digest,'CPU decoder runtime changed')
        tokenizer_path,tokenizer_proof=tokenizer_reference(specs[0]['tokenizer'],numeric,owner)
        result['tokenizer_reference']=tokenizer_proof
        tokenizer=tokenizers.Tokenizer.from_file(str(tokenizer_path))
        need(all(s['tokenizer']==specs[0]['tokenizer'] for s in specs),'one exact source/model tokenizer')
        result['http_raw_opened']=True
        children,points=audit_trials(owner,wrapper,specs,complete,lambda ids:tokenizer.decode(ids,skip_special_tokens=False))
        same(children,lives,'independent phase lifecycle accounting')
        result['resources']=resources(owner,complete,lives,data['original']['pid'],attempt['created_at'],list(PHASES))
        comparisons=paired_summary(points)
        same(tokenizer_reference(specs[0]['tokenizer'],numeric,owner)[1],tokenizer_proof,'tokenizer target changed during audit')
        same(input_inventory(owner),initial,'actual closed evidence changed during audit')
        need(sha(journal/'journal.json')==journal_sha==restoration['journal_sha256'],'restoration changed during audit')
        need(sha(DONOR)==DONOR_SHA and sha(SUPPORT)==SUPPORT_SHA and sha(FROZEN_V1)==FROZEN_V1_SHA,'audit algorithm sources changed')
        result.update(status='PASS',qualified=True,performance_qualified=True,owner_plan_sha256=plan_sha256,
            owner_complete_sha256=sha(owner/'complete.json'),journal_sha256=journal_sha,
            requests=dict(total=864,warm=96,formal=768),points=points,comparisons=comparisons,children=lives,
            inner_absence_evidence=absence_proofs,raw_file_pins=initial,
            source_pins={str(Path(__file__).resolve()):result['audit_source_sha256'],
                str(DONOR):DONOR_SHA,str(donor.DONOR):donor.DONOR_SHA,str(SUPPORT):SUPPORT_SHA,str(FROZEN_V1):FROZEN_V1_SHA,str(HERE/'owner.py'):OWNER_SHA,str(HERE/'controller.py'):CONTROLLER_SHA,
                str(HERE/'audit_restore.py'):RESTORE_SHA,str(FORMAL):FORMAL_SHA,
                str(NATIVE/'audit_http_owner_v2.py'):LEGACY_SHA,str(NATIVE/'audit_native_formal.py'):RAW_SHA,
                str(HERE.parent/'lithos_d512_formal/audit_resources.py'):RESOURCE_SHA},
            raw_point_files_modified=False,model_payload_independently_rehashed=False,
            absence_scope='Outer native System.read retained3fields, no stderr claim. Inner4field empty stderr is a separate earlier observation, not fabricated outer stderr.')
    except (Incomplete,FileNotFoundError) as exc:result.update(status='INCOMPLETE',reason=str(exc))
    except Exception as exc:result.update(status='FAIL',first_failure=dict(type=type(exc).__name__,message=str(exc)))
    return result

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('owner','journal','plan-sha256','out'):parser.add_argument('--'+name,required=True)
    args=parser.parse_args(argv);out=Path(args.out);need(not out.exists() and out.parent.is_dir(),'exclusive output with existing parent')
    result=audit(args.owner,args.journal,args.plan_sha256)
    with out.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    return {'PASS':0,'FAIL':1,'INCOMPLETE':2}[result['status']]

if __name__=='__main__':raise SystemExit(main())
