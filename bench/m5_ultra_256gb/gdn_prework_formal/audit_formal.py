"""Independent GDN-only paired HTTP timing; no service/model execution.

Use the frozen route v2 tokenizer, native157 preparation, raw count/timing,
resource and process-pair algorithms directly. Only owner/trial schema and
observer identity checks are adapted to real GDN producers; no donor globals,
paths, or recorded evidence are rewritten. The original route v1 remains pinned.
A closed full cycle is mandatory; installation is not a formal dispatch witness.
This grants measured performance eligibility, not full-state proof for unseen
schedules, arithmetic reexecution, E024/E028 composition or general quality.
"""
import argparse
from functools import lru_cache
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

HERE=Path(__file__).resolve().parent
NATIVE=HERE.parent/'pareto_realistic'
SUPPORT=HERE.parent/'route_http_formal/audit_formal_v2.py'
SUPPORT_SHA='b2bc55b87043d66d79154fd32dc7579cc7c6fdb457844a1b1d3df932ff41224e'
FROZEN_V1=HERE.parent/'route_http_formal/audit_formal.py'
FROZEN_V1_SHA='ed8ad64c3eb5fed052effd05981a386006f4ea1f752463bb0ada0c2f0e4ea5e0'
OWNER_SHA='0148335cd334524fe367aab00f3a5d2026d1848a928ab469f1d1ce975334c461'
CONTROLLER_SHA='38a6e904d928813efa8ba1f6de724693dccce196322401c4f44b0442ed4dcd35'
RESTORE_SHA='8ece5748f8328a110651981f40d3c7ba158194a9479052c544083d45a5621fc0'
FORMAL_SHA='e3fd332884975426487853108098933283e4b98705671cfe8dbccd37f91bfb7a'
RAW_SHA='b57e4e9657f3dddb04069290bdfbab4c99b0d6cbfb1427e454d0217d5463c870'
RESOURCE_SHA='398ff7a52dba307418f4c2725f33988b1571d65b11e3f46c9e97759f3a167c9e'
LEGACY_SHA='8d8aee0372497fcf591d53e4bc02b3d1e7eac7eb1d2bf639168a72802e2395bd'
CAPTURE_SHA='536fe41d6c8e2bdffe83ba6f43fba87dd6265deb4c84f6204bc9c7220f8280e3'
CLIENT_SHA='ef64a114f4f7d90ec5044d6bdcf058ca05f98dea445d6639db706a4fc75217dc'
PHASES=('r1-c1-off','r1-c1-on','r1-c8-on','r1-c8-off',
        'r2-c8-off','r2-c8-on','r2-c1-on','r2-c1-off',
        'r3-c1-off','r3-c1-on','r3-c8-on','r3-c8-off')

class Incomplete(Exception):pass

def need(ok,message):
    if not ok:raise ValueError(message)

def same(a,b,message):
    need(json.dumps(a,sort_keys=True,allow_nan=False)==json.dumps(b,sort_keys=True,allow_nan=False),message)

def finite(x):return type(x) in (int,float) and math.isfinite(x)

def read(path):
    need(Path(path).suffix!='.jsonl','no numerical/route ledger is read for formal timing')
    def pairs(items):
        result={}
        for key,value in items:need(key not in result,'duplicate JSON field');result[key]=value
        return result
    return json.loads(Path(path).read_bytes(),object_pairs_hook=pairs,
        parse_constant=lambda v:(_ for _ in ()).throw(ValueError('nonfinite '+v)))

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        while chunk:=stream.read(1<<20):h.update(chunk)
    return h.hexdigest()

def ref(path):return dict(path=str(path),sha256=sha(path))

def load(path,digest,name):
    need(len(digest)==64 and sha(path)==digest,'unsealed/changed source '+str(path))
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module);return module


@lru_cache(maxsize=1)
def support():return load(SUPPORT,SUPPORT_SHA,'_gdn_formal_frozen_route_v2')
def pinned(item):return support().pinned(item)
def relative(*args):return support().relative(*args)
def outer_absence(*args):return support().outer_absence(*args)
def inner_absence(*args):return support().inner_absence(*args)
def adapter_copies(*args):return support().adapter_copies(*args)
def preparation_pins(*args):return support().preparation_pins(*args)
def tokenizer_reference(*args):return support().tokenizer_reference(*args)
def raw_metrics(group):return support().raw_metrics(group)
def resources(*args):return support().resources(*args)
def paired_summary(points):return support().paired_summary(points)
def input_inventory(owner):return support().input_inventory(owner)
def legacy():return support().legacy()


def owner_binding(root,plan,complete):
    """Closed real12 canonical intents/children/reaps, no live OS observation."""
    attempt=read(root/'attempt.json');index=read(root/'children.json');release=read(root/'release.json')
    stopped=read(root/'stopped-journal.json');spawn=read(root/'spawn-intent.json');digest=sha(root/'owner-plan.json')
    source=dict(path=str(HERE/'owner.py'),sha256=OWNER_SHA)
    need(attempt['schema']=='cold-service-owner-attempt-v1' and type(complete['owner_pid']) is int
         and complete['owner_pid']>1 and attempt['owner_pid']==complete['owner_pid']
         and attempt['owner_identity']['pid']==complete['owner_pid'],'owner identity/attempt')
    need(plan['schema']=='GDN-prework-http-formal-owner-plan-v1' and plan['trials']==list(PHASES)
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
    repeat,C,arm=stage.split('-');need(spec['phase']==stage and spec['repeat']==int(repeat[1:])
        and spec['concurrency']==int(C[1:]) and spec['arm']==arm and spec['observer_enabled'] is False,'real phase/arm')
    command=spec['command'];identity=spec['identity']
    need(not any(flag in command for flag in ('--qualification-trace','--dispatch-events'))
         and identity['step_observer'] is identity['qualification_trace_enabled'] is identity['per_token_observer'] is False,'both observers OFF')
    if arm=='off':
        need(command[3]==str(NATIVE/'native_server.py') and '--gdn-prework' not in command
             and '--phase' not in command and 'gdn_prework' not in identity,'unchanged OFF actual entry')
    else:
        need(command[3]==str(HERE.parent/'gdn_prework_http/http_server.py')
             and command[-4:]==['--phase','formal','--gdn-prework','on'],'actual ON entry/flags')
        gdn=identity['gdn_prework']
        need(gdn['enabled'] is gdn['installed'] is True and type(gdn['installed_layers']) is int
             and gdn['installed_layers']==48 and gdn['dispatch_observer'] is False
             and gdn['actual_calls_observed'] is False and gdn['route_graph_enabled'] is gdn['identity_refill_enabled'] is False,
             'GDN installation identity is not runtime dispatch proof')

def audit_trials(owner,wrapper,specs,complete,decode_text):
    formal=load(HERE/'formal.py',FORMAL_SHA,'_gdn_audit_formal_pure')
    cap=load(NATIVE/'http_capture_v2.py',CAPTURE_SHA,'_audit_formal_identity')
    support=legacy();names=wrapper['trials'];same(names,list(PHASES),'complete fixed twelve trials')
    index=read(owner/'children.json');need(index['schema']=='cold-service-owner-children-v1','child index schema')
    rows=index['children'];need([r['phase'] for r in rows]==names,'complete registered direct children')
    same(read(owner/'formal-specs.json'),specs,'actual prepared formal specs')
    need([r['phase'] for r in complete['phases']]==names,'all trial stages completed')
    saved=read(owner/'points.json')
    need(saved['schema']=='GDN-prework-http-formal-points-v1' and saved['performance_qualified'] is False,'original unapproved point collection')
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
        need(result['schema']=='GDN-prework-http-formal-result-v1' and result['status']=='PASS' and result['errors']==[]
             and result['concurrency']==spec['concurrency'] and result['repeat']==spec['repeat'] and result['arm']==spec['arm'] and result['phase']==stage
             and result['observer_enabled'] is False and result['full_state_qualified'] is False
             and result['performance_qualified'] is False and all(result[k] is True for k in
                ('count_eligible','cleanup_complete','server_reaped','cleanup_task_resolved','client_task_resolved','small_pins_rechecked')),
             'closed valid formal trial required')
        need(finite(result['started_at']) and finite(result['ended_at'])
             and result['started_at']<=child['spawned_at']<=result['ended_at']<=release['time'],'inner/outer lifecycle ordering')
        need(item['status']=='PASS' and item['arm']==spec['arm'],'completed trial stage/arm status');observer_identity(spec,stage)
        same(item['result'],dict(path=str((trial/'complete.json').relative_to(owner)),sha256=sha(trial/'complete.json')),'trial result receipt')
        same(item['numerical_audit'],spec['qualification']['audit'],'trial numerical binding')
        same(read(trial/'spec.json'),spec,'executed formal spec')
        same(read(trial/'launch.json'),dict(command=spec['command'],environment=wrapper['http']['environment']),'actual formal CLI/environment')
        need(not any(flag in spec['command'] for flag in ('--qualification-trace','--dispatch-events'))
             and not (trial/'trace.jsonl').exists() and not (trial/'dispatch-events.jsonl').exists()
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
            numerical_audit=spec['qualification']['audit'],gdn_cycle_audit=spec['gdn_qualification']['cycle_audit'],
            actual_prework_calls_observed=False))
        children.append(fact)
    need(len(all_uuids)==864 and len(points)==12,'all864 requests across twelve fresh processes')
    return children,points

def audit(owner,journal,plan_sha256):
    owner,journal=Path(owner).resolve(),Path(journal).resolve()
    result=dict(schema='GDN-prework-http-formal-independent-audit-v1',status='INCOMPLETE',qualified=False,
        performance_qualified=False,ledger_opened=False,http_raw_opened=False,actual_prework_calls_observed=False,
        owner_directory=str(owner),audit_source_sha256=sha(__file__),
        reused_audit_source_sha256=SUPPORT_SHA,
        frozen_route_v1_audit_source_sha256=FROZEN_V1_SHA,
        scope='Representative baseline/GDN captured histories are prerequisites. Rechecks864 cycle wire requests and ownership/resources/restoration; no unseen-schedule full-state proof, formal dispatch observations, combined E024/E028 or general quality claim.')
    try:
        need(sha(SUPPORT)==SUPPORT_SHA and sha(FROZEN_V1)==FROZEN_V1_SHA,'frozen audit algorithms changed')
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
        need(complete['schema']=='GDN-prework-http-formal-owner-result-v1' and complete['errors']==[]
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
        need(sha(SUPPORT)==SUPPORT_SHA and sha(FROZEN_V1)==FROZEN_V1_SHA,'audit algorithm sources changed')
        result.update(status='PASS',qualified=True,performance_qualified=True,owner_plan_sha256=plan_sha256,
            owner_complete_sha256=sha(owner/'complete.json'),journal_sha256=journal_sha,
            requests=dict(total=864,warm=96,formal=768),points=points,comparisons=comparisons,children=lives,
            inner_absence_evidence=absence_proofs,raw_file_pins=initial,
            source_pins={str(Path(__file__).resolve()):result['audit_source_sha256'],
                str(SUPPORT):SUPPORT_SHA,str(FROZEN_V1):FROZEN_V1_SHA,str(HERE/'owner.py'):OWNER_SHA,str(HERE/'controller.py'):CONTROLLER_SHA,
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
