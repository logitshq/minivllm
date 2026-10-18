"""Independent incremental GDN-on-graph HTTP timing; no service/model action.

Both arms use the same combined entry with graph ON and GDN OFF versus ON.
Frozen GDN/route auditors supply native157 provenance, tokenizer, original raw
SSE/count/window/p50/p95, resource and process-pair algorithms unchanged. Only
actual ownership/entry/source/qualification metadata differs. All864 requests
and12 fresh processes must close before performance eligibility. No graph-hit
observation, offline tensor recomputation or unseen-arrival proof is inferred.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

HERE=Path(__file__).resolve().parent
NATIVE=HERE.parent/'pareto_realistic'
DONOR=HERE.parent/'gdn_prework_formal/audit_formal.py'
DONOR_SHA='70705b8ad9e2a1948306d801662302a4cdfb38a7a30370d5961c27fbbeb3f976'
OWNER_SHA='511b48fbe722ea031a7b4586439eaa67f25c9c9514af12b614de74fb8c53c103'
CONTROLLER_SHA='3b8b3033269fcd326e0a52e20470ee27faf65d313be3e3bf0da8881f26f3ac4c'
RESTORE_SHA='157d6470a9cb87223c419a3f4edd82e8556b3714a7157ea1ad505270c8ed7493'
FORMAL_SHA='4b567f8ed6090175f3639aa109584ca26c1426165c03de3167c3ebde74891a89'
FACTORY_LIFECYCLE_SHA='ec9cdc924f7987ad46eac6820b6264495c95fb182c726545a6e3ce283df7e654'
SERVER=HERE.parent/'gdn_graph_combined/http_server.py'
SERVER_SHA='287f34561c83437823923cf45b67f0aa6811f7eba911ee375f62519ba508ef64'


def frozen(path,digest):
    if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:raise ValueError('fixed audit support changed: '+str(path))
    spec=importlib.util.spec_from_file_location('_combined_formal_audit_'+path.stem,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


donor=frozen(DONOR,DONOR_SHA)
Incomplete,need,same,finite,read,sha,ref,load=(donor.Incomplete,donor.need,donor.same,donor.finite,donor.read,donor.sha,donor.ref,donor.load)
PHASES=donor.PHASES
SUPPORT,SUPPORT_SHA,FROZEN_V1,FROZEN_V1_SHA=donor.SUPPORT,donor.SUPPORT_SHA,donor.FROZEN_V1,donor.FROZEN_V1_SHA
RAW_SHA,RESOURCE_SHA,LEGACY_SHA,CAPTURE_SHA,CLIENT_SHA=donor.RAW_SHA,donor.RESOURCE_SHA,donor.LEGACY_SHA,donor.CAPTURE_SHA,donor.CLIENT_SHA
support,pinned,relative,outer_absence,inner_absence=(donor.support,donor.pinned,donor.relative,donor.outer_absence,donor.inner_absence)
adapter_copies,preparation_pins,tokenizer_reference=(donor.adapter_copies,donor.preparation_pins,donor.tokenizer_reference)
raw_metrics,resources,paired_summary,input_inventory,legacy=(donor.raw_metrics,donor.resources,donor.paired_summary,donor.input_inventory,donor.legacy)


def owner_binding(root,plan,complete):
    """Closed real12 canonical intents/children/reaps, no live OS observation."""
    attempt=read(root/'attempt.json');index=read(root/'children.json');release=read(root/'release.json')
    stopped=read(root/'stopped-journal.json');spawn=read(root/'spawn-intent.json');digest=sha(root/'owner-plan.json')
    source=dict(path=str(HERE/'owner.py'),sha256=OWNER_SHA)
    need(attempt['schema']=='cold-service-owner-attempt-v1' and type(complete['owner_pid']) is int
         and complete['owner_pid']>1 and attempt['owner_pid']==complete['owner_pid']
         and attempt['owner_identity']['pid']==complete['owner_pid'],'owner identity/attempt')
    need(plan['schema']=='GDN-graph-combined-http-formal-owner-plan-v1' and plan['trials']==list(PHASES)
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

def observer_identity(spec,stage):
    repeat,C,arm=stage.split('-');need(arm in ('off','on') and spec['phase']==stage
        and spec['repeat']==int(repeat[1:]) and spec['concurrency']==int(C[1:])
        and spec['arm']==arm and spec['observer_enabled'] is False,'real incremental phase/arm')
    command=spec['command'];identity=spec['identity']
    need(not any(flag in command for flag in ('--qualification-trace','--route-events','--dispatch-events'))
         and identity['step_observer'] is identity['qualification_trace_enabled'] is identity['per_token_observer'] is False,
         'all formal observers OFF')
    need(command[3]==str(SERVER) and command[-6:]==['--phase','formal','--route-decode-graph','on','--gdn-prework',arm]
         and all(command.count(flag)==1 for flag in ('--phase','--route-decode-graph','--gdn-prework')),
         'same actual entry, graph always ON, only GDN arm differs')
    facts=identity['combined_candidates']
    need(identity['server_source_path']==str(SERVER) and identity['server_source_sha256']==SERVER_SHA
         and identity['benchmark_phase']=='formal' and facts['schema']=='GDN-graph-combined-http-adapter-v1'
         and facts['route_decode_graph_enabled'] is facts['installed'] is True
         and facts['gdn_prework_enabled'] is (arm=='on') and type(facts['gdn_layer_installs']) is int
         and facts['gdn_layer_installs']==(48 if arm=='on' else 0)
         and facts['route_observer'] is facts['actual_compiled_invocations_observed'] is facts['identity_refill_enabled'] is False,
         'combined installation metadata is not a formal graph-hit observation')
    same(spec['qualification'],spec['route_qualification' if arm=='off' else 'combined_qualification'],
         'arm-specific independently qualified numerical history')


def audit_trials(owner,wrapper,specs,complete,decode_text):
    formal=load(HERE/'formal.py',FORMAL_SHA,'_combined_audit_formal_pure')
    cap=load(NATIVE/'http_capture_v2.py',CAPTURE_SHA,'_audit_formal_identity')
    support=legacy();names=wrapper['trials'];same(names,list(PHASES),'complete fixed twelve trials')
    index=read(owner/'children.json');need(index['schema']=='cold-service-owner-children-v1','child index schema')
    rows=index['children'];need([r['phase'] for r in rows]==names,'complete registered direct children')
    same(read(owner/'formal-specs.json'),specs,'actual prepared formal specs')
    need([r['phase'] for r in complete['phases']]==names,'all trial stages completed')
    saved=read(owner/'points.json')
    need(saved['schema']=='GDN-graph-combined-http-formal-points-v1' and saved['performance_qualified'] is False,'original unapproved point collection')
    same(saved['trials'],complete['phases'],'incremental saved points differ from final stages')
    children=[];points=[];all_uuids=set()
    for i,(row,spec,item) in enumerate(zip(rows,specs,complete['phases'])):
        stage=names[i];folder=owner/'phases'/stage;trial=folder/'trial'
        child=read(folder/'child.json');release=read(folder/'release.json')
        need(child['phase']==release['phase']==stage and child['pid']==row['pid']==release['pid']
             and child['ownership']=='direct-Popen-returned' and child['owner_pid']==complete['owner_pid']
             and child['identity'] is not None and child['identity']['pid']==child['pid']
             and child['identity']['ppid']==complete['owner_pid'],'actual owned trial PID')
        need(finite(child['spawned_at']) and finite(release['time']) and child['spawned_at']<=release['time']
             and release['waited_reaped'] is True and release['strict_absence'] is True,'owned trial reap/time')
        need(not children or children[-1]['released_at']<=child['spawned_at'],'trial processes overlap')
        result=read(trial/'complete.json')
        need(result['schema']=='GDN-graph-combined-http-formal-result-v1' and result['status']=='PASS' and result['errors']==[]
             and result['concurrency']==spec['concurrency'] and result['repeat']==spec['repeat'] and result['arm']==spec['arm'] and result['phase']==stage
             and result['observer_enabled'] is False and result['full_state_qualified'] is False
             and result['performance_qualified'] is False and all(result[k] is True for k in
                ('count_eligible','cleanup_complete','server_reaped','cleanup_task_resolved','client_task_resolved','small_pins_rechecked')),
             'closed valid formal trial required')
        same(result['factory_source'],dict(path=str(HERE/'formal.py'),sha256=FORMAL_SHA),'actual formal factory source')
        same(result['timing_lifecycle_source'],dict(path=str(NATIVE/'native_formal.py'),sha256=FACTORY_LIFECYCLE_SHA),'unchanged native timing lifecycle source')
        need(finite(result['started_at']) and finite(result['ended_at'])
             and result['started_at']<=child['spawned_at']<=result['ended_at']<=release['time'],'inner/outer lifecycle ordering')
        need(item['status']=='PASS' and item['arm']==spec['arm'],'completed trial stage/arm status');observer_identity(spec,stage)
        same(item['result'],dict(path=str((trial/'complete.json').relative_to(owner)),sha256=sha(trial/'complete.json')),'trial result receipt')
        same(item['numerical_audit'],spec['qualification']['audit'],'trial numerical binding')
        same(read(trial/'spec.json'),spec,'executed formal spec')
        same(read(trial/'launch.json'),dict(command=spec['command'],environment=wrapper['http']['environment']),'actual formal CLI/environment')
        need(not any(flag in spec['command'] for flag in ('--qualification-trace','--route-events','--dispatch-events'))
             and not (trial/'trace.jsonl').exists() and not (trial/'dispatch-events.jsonl').exists() and not (trial/'route-events.jsonl').exists()
             and spec['identity']['step_observer'] is False and spec['identity']['qualification_trace_enabled'] is False
             and spec['identity']['per_token_observer'] is False,'formal observer-OFF route')
        support.capture_artifacts(trial,result['artifacts'])
        same(read(trial/'child.json')['pid'],child['pid'],'inner returned Popen PID')
        for name in ('ready.json','ready-after-warm.json','ready-after-formal.json'):
            need(cap.ready_identity(read(trial/name),spec['identity']),'actual observer-OFF model/config identity')
        need(cap.owned_listener(read(trial/'listener.json'),child['pid'],wrapper['http']['transport']['port']),'actual localhost PID/listener')
        inner_release=read(trial/'release.json')
        need(inner_release['pid']==child['pid'] and inner_release['returncode']==release['returncode']
             and inner_release['waited_reaped'] is True and inner_release['strict_absence'] is True
             and inner_release['port_absent'] is True and finite(inner_release['time'])
             and child['spawned_at']<=inner_release['time']<=result['ended_at'],'inner wait/reap proof')
        fact=dict(phase=stage,pid=child['pid'],spawned_at=child['spawned_at'],released_at=release['time'],
                  returncode=release['returncode'],inner_released_at=inner_release['time'])
        metrics={}
        for phase,count in (('warm',8),('formal',64)):
            group=read(trial/(phase+'-group.json'));same(group,read(trial/phase/'group.json'),'full raw group duplicate binding')
            need(group['base_url']==f"http://127.0.0.1:{spec['transport']['port']}"
                 and group['request_timeout_s']==spec['deadlines']['request']
                 and group['group_timeout_s']==spec['deadlines'][phase]
                 and group['cleanup']['budget_s']==spec['deadlines']['close']
                 and group['cleanup']['frozen_client_sha256']==CLIENT_SHA
                 and group['cleanup']['client_close_started'] is True
                 and group['qualification_errors']==[], 'fixed local endpoint/client deadlines and cleanup')
            same(group['connection_policy'],dict(max_connections=2*spec['concurrency'],max_keepalive_connections=0,
                 retries=0,follow_redirects=False),'fixed actual connection policy')
            checked=formal.check_group(group,spec,phase,decode_text)
            same(read(trial/(phase+'-raw-check.json')),checked,'runner raw-check record differs from reparse')
            actual=raw_metrics(group);need(actual['requests']==count,'fixed warm/formal N')
            point=dict(concurrency=spec['concurrency'],repeat=spec['repeat'],**{k:actual[k] for k in
                ('requests','committed_output_tokens','common_window_s','aggregate_output_tok_s','e2e_p50_s','e2e_p95_s')})
            same(point,checked['point'],'independent wire count/window/p50/p95 calculation')
            need(not(all_uuids & set(actual['response_uuids'])),'UUID reused across trial/group requests')
            all_uuids.update(actual['response_uuids']);metrics[phase]=actual
        need(metrics['warm']['client_close_end_s']<=metrics['formal']['first_post_s'],'formal starts before warm client is closed')
        point=dict(concurrency=spec['concurrency'],repeat=spec['repeat'],**{k:metrics['formal'][k] for k in
            ('requests','committed_output_tokens','common_window_s','aggregate_output_tok_s','e2e_p50_s','e2e_p95_s')})
        same(result['point'],point,'terminal point differs from independent wire metrics')
        same(read(trial/'point.json'),dict(**point,count_eligible=True,performance_qualified=False),'raw point must remain unapproved')
        same(item['point'],dict(path=str((trial/'point.json').relative_to(owner)),sha256=sha(trial/'point.json')),'raw point digest')
        points.append(dict(phase=stage,arm=spec['arm'],**point,n_fresh_processes=1,max_http_outstanding=metrics['formal']['max_http_outstanding'],
            warm_requests_checked=8,formal_raw_group=ref(trial/'formal-group.json'),point_receipt=ref(trial/'point.json'),
            numerical_audit=spec['qualification']['audit'],combined_cycle_audit=spec['combined_qualification']['cycle_audit'],
            route_cycle_audit=spec['route_qualification']['cycle_audit'],route_always_on=True,
            actual_prework_calls_observed=False,actual_compiled_invocations_observed=False))
        children.append(fact)
    need(len(all_uuids)==864 and len(points)==12,'all864 requests across twelve fresh processes')
    return children,points

def audit(owner,journal,plan_sha256):
    owner,journal=Path(owner).resolve(),Path(journal).resolve()
    result=dict(schema='GDN-graph-combined-http-formal-independent-audit-v1',status='INCOMPLETE',qualified=False,
        performance_qualified=False,ledger_opened=False,http_raw_opened=False,actual_prework_calls_observed=False,actual_compiled_invocations_observed=False,route_always_on=True,
        owner_directory=str(owner),audit_source_sha256=sha(__file__),
        reused_audit_source_sha256=DONOR_SHA,raw_algorithm_source_sha256=SUPPORT_SHA,
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
                str(DONOR):DONOR_SHA,str(SUPPORT):SUPPORT_SHA,str(FROZEN_V1):FROZEN_V1_SHA,str(HERE/'owner.py'):OWNER_SHA,str(HERE/'controller.py'):CONTROLLER_SHA,
                str(HERE/'audit_restore.py'):RESTORE_SHA,str(HERE/'formal.py'):FORMAL_SHA,
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
