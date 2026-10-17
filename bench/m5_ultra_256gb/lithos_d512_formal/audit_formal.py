"""Offline six-process Lithos D512 committed-throughput/latency audit.

Timing is eligible only after raw NDJSON, observer-OFF launch, quiet zero-cache
log windows, actual direct children, resource isolation and service restoration
all pass. Normal-EOS semantics come from the exact observed v2 qualification;
this is not full-state equality or a new output-quality test.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
from types import SimpleNamespace

HERE=Path(__file__).resolve().parent
PHASES=('r1-c1','r1-c8','r2-c8','r2-c1','r3-c1','r3-c8')
FIXED={
    'owner.py':'49dfa344b0abca24ba3ec84a14b44018751cb57353b82fd16197ca4bd49c9b70',
    'formal.py':'fca180b4296e352e7b75db517ac897fe56d25ceea313d31b6a6c091a515c73ff',
    'contract.py':'2c5b221608e8c53b39116b2e13b247b9fb78d83c74cbee069179efbea9210b32',
    'audit_restore.py':'41d260e63ce13c5c5912217ae74169e5c3cd118af32f3667aefffc3637a2d66f',
    'audit_resources.py':'398ff7a52dba307418f4c2725f33988b1571d65b11e3f46c9e97759f3a167c9e',
    '../lithos_d512_retry/audit_qualification.py':'a932583f5a5234b0aff434d9b967a004fcdea05ff9b1c03a1d5d34fe86f6df08',
}

def load(name):
    p=(HERE/name).resolve()
    if hashlib.sha256(p.read_bytes()).hexdigest()!=FIXED[name]:raise ValueError('frozen audit dependency changed: '+name)
    spec=importlib.util.spec_from_file_location('_lithos_formal_audit_'+p.stem,p)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

legacy=load('../lithos_d512_retry/audit_qualification.py')
need,same,sha,read,ref,relative,finite=legacy.need,legacy.same,legacy.sha,legacy.read,legacy.ref,legacy.relative,legacy.finite
Incomplete=legacy.Incomplete

def closed_bindings(root,complete,plan):
    attempt=read(root/'attempt.json');stopped=read(root/'stopped-journal.json')
    pid=complete['owner_pid'];plan_sha=sha(root/'owner-plan.json')
    need(type(pid) is int and pid>1 and attempt['owner_pid']==pid,'actual owner PID')
    need(attempt['schema']=='cold-service-owner-attempt-v1' and
         attempt['owner_plan_sha256']==complete['owner_plan_sha256']==plan_sha and
         attempt['owner_source']==plan['execution']['owner'],'owner attempt/plan/source')
    need(attempt['owner_identity']['pid']==pid and attempt['stopped_journal_sha256']==sha(root/'stopped-journal.json'),'owner identity/stopped snapshot')
    need(stopped['schema']=='E023-lithos-D512-formal-cold-journal-v1' and stopped['state']=='STOPPED'
         and stopped['owner_directory']==str(root) and stopped['owner_plan']['sha256']==plan_sha
         and stopped['owner_source']==attempt['owner_source'] and stopped['id']==attempt['journal_id']
         and stopped['removed_at']==attempt['removed_at'] and stopped['trials']==list(PHASES)
         and stopped['predecessor']==plan['cold_restart']['predecessor'],'real STOPPED binding')
    index=read(root/'children.json');final=read(root/'release.json');spawn=read(root/'spawn-intent.json')
    need(index['schema']=='cold-service-owner-children-v1' and final['schema']=='cold-service-owner-release-v2','two-phase release schema')
    children=index['children'];need([c['phase'] for c in children]==list(PHASES),'exact complete six-phase coverage')
    same(final['children'],children,'final release/index differs')
    need(final['children_sha256']==sha(root/'children.json') and final['all_children_reaped'] is True
         and final.get('no_popen_receipt') is None,'all actual children closed')
    bound={k:attempt[k] for k in ('owner_pid','owner_identity','owner_source','owner_plan_sha256','journal_id','removed_at')}
    bound['attempt_sha256']=sha(root/'attempt.json')
    need(all(final.get(k)==v for k,v in bound.items()),'final attempt binding')
    need(all(spawn.get(k)==bound[k] for k in ('owner_pid','owner_plan_sha256','journal_id','attempt_sha256')),'global intent binding')
    need(all(finite(t) for t in (attempt['removed_at'],attempt['created_at'],final['time']))
         and attempt['removed_at']<=attempt['created_at']<=final['time'],'terminal time order')
    previous=attempt['created_at'];seen=set()
    for child in children:
        phase=child['phase'];folder=root/'phases'/phase
        intent=read(relative(root,child['intent'],f'phases/{phase}/intent.json'))
        own=read(relative(root,child['child'],f'phases/{phase}/child.json'))
        release=read(relative(root,child['release'],f'phases/{phase}/release.json'))
        need(all(intent.get(k)==bound[k] for k in ('owner_pid','owner_plan_sha256','journal_id','attempt_sha256'))
             and intent['phase']==phase,'phase intent binding')
        cpid=child['pid'];identity=own['identity']
        need(type(cpid) is int and cpid>1 and cpid!=pid and cpid not in seen,'unique actual server PID');seen.add(cpid)
        need(own['phase']==phase and own['pid']==cpid and own['owner_pid']==pid and
             type(identity) is dict and identity['pid']==cpid and identity['ppid']==pid and
             child['identity']==identity and own['ownership']=='direct-Popen-returned','actual direct child identity')
        need(release['phase']==phase and release['pid']==cpid and type(release['returncode']) is int
             and release['waited_reaped'] is True and release['strict_absence'] is True,'actual child reap')
        need(all(finite(t) for t in (intent['time'],own['spawned_at'],release['time'])) and
             previous<=intent['time']<=own['spawned_at']<=release['time']<=final['time'],'sequential phase times')
        previous=release['time']
        absent=read(relative(root,release['absence']))
        same(absent['command'],['/bin/ps','-p',str(cpid),'-o','pid='],'actual PID absence command')
        need(absent['returncode']==1 and absent['stdout']=='','actual reaped PID absence')
        ports=sorted(folder.glob('port-absence-*.json'));need(bool(ports),'outer port absence missing')
        for path in ports:
            raw=read(path);same(raw['command'],['/usr/sbin/lsof','-nP','-a',f"-iTCP:{plan['http']['port']}",'-sTCP:LISTEN','-Fpn'],'actual port absence command')
            need(raw['returncode']==1 and raw['stdout']=='','actual port absence')
    expected={f'phases/{phase}/{name}' for phase in PHASES for name in
        ('intent.json','child.json','release.json','trial/spawn-intent.json','trial/child.json','trial/release.json')}
    found={str(p.relative_to(root)) for p in (root/'phases').rglob('*')
        if p.name in ('intent.json','child.json','release.json','spawn-intent.json','no-popen.json')}
    same(sorted(found),sorted(expected),'unresolved or omitted phase spawn evidence')
    return children

def log_windows(folder,spec,groups,join):
    raw=(folder/'server.stderr').read_bytes();need(len(raw)<=spec['log_contract']['max_bytes'],'bounded complete stderr')
    labels=('warm-start','warm-end','formal-start','formal-end','after-reap')
    boundaries={label:read(folder/(label+'-log-boundary.json')) for label in labels}
    previous=None
    for label,row in boundaries.items():
        need(set(row)=={'label','path','device','inode','offset','prefix_sha256','wall_s','monotonic_s','perf_counter_s'}
             and row['label']==label and row['path']=='server.stderr','exact log boundary')
        need(all(type(row[k]) is int and row[k]>0 for k in ('device','inode'))
             and type(row['offset']) is int and 0<=row['offset']<=len(raw),'log identity/offset')
        prefix=raw[:row['offset']]
        need((not prefix or prefix.endswith(b'\n')) and hashlib.sha256(prefix).hexdigest()==row['prefix_sha256'],'unchanged full line prefix')
        need(all(finite(row[k]) and row[k]>0 for k in ('wall_s','monotonic_s','perf_counter_s')),'boundary clocks')
        if previous:
            need(all(row[k]==previous[k] for k in ('device','inode'))
                 and all(row[k]>=previous[k] for k in ('offset','wall_s','monotonic_s','perf_counter_s')),'one ordered stderr history')
        previous=row
    need(boundaries['after-reap']['offset']==len(raw),'whole final stderr retained')
    checks={}
    for phase,count in (('warm',8),('formal',64)):
        start,end=boundaries[phase+'-start'],boundaries[phase+'-end'];group=groups[phase]
        need(start['perf_counter_s']<=group['barrier_release_perf_counter_s']
             and end['perf_counter_s']>=group['cleanup']['end_perf_counter_s']
             and all(start['perf_counter_s']<=r['start_perf_counter_s']<r['done_perf_counter_s']<=r['eof_perf_counter_s']<=end['perf_counter_s'] for r in group['requests']),
             'log window brackets complete POST/DONE/drain')
        name=phase+'-generation.log';window=raw[start['offset']:end['offset']]
        need((folder/name).read_bytes()==window,'original byte slice')
        checks[phase]=join.generation_log_window(window,count,require_setup_zero=phase=='formal',expected_logger='__main__')
        expected=dict(schema='E023-lithos-formal-log-window-v1',phase=phase,start=start,end=end,
            artifact=dict(path=name,sha256=hashlib.sha256(window).hexdigest(),bytes=len(window)),check=checks[phase])
        same(read(folder/(phase+'-log-check.json')),expected,'recomputed log window')
    all_generations=join.generation_log_window(raw,72,require_setup_zero=False,expected_logger='__main__')
    same(read(folder/'log-final-check.json'),dict(status='PASS',complete_windows=['warm','formal'],
        final=boundaries['after-reap'],all_prefixes_unchanged=True,all_generations=all_generations),'complete log final receipt')
    return dict(warm=checks['warm'],formal=checks['formal'],all_generations=all_generations,
        per_wire_request_id_bound=False,scope='Exact quiet byte windows/counts; official Generation lines have no HTTP request IDs.')


def metrics(group,wire):
    rows=group['requests'];counts=[]
    for row in rows:
        terminal=wire(row)['terminal'];counts.append(terminal['eval_count']-int(terminal['done_reason']=='stop'))
    latency=[r['done_perf_counter_s']-r['start_perf_counter_s'] for r in rows]
    window=max(r['done_perf_counter_s'] for r in rows)-min(r['start_perf_counter_s'] for r in rows)
    return dict(requests=len(rows),committed_output_tokens=sum(counts),common_window_s=window,
        aggregate_output_tok_s=sum(counts)/window,e2e_p50_s=statistics.median(latency),
        e2e_p95_s=sorted(latency)[math.ceil(.95*len(rows))-1])


def trial_evidence(folder,spec,child,owner_pid,formal,contract):
    complete=read(folder/'complete.json');cpid=child['pid'];phase=child['phase']
    need(complete['schema']=='E023-lithos-D512-formal-result-v1' and complete['status']=='PASS'
         and complete['concurrency']==spec['concurrency'] and complete['repeat']==spec['repeat']
         and complete['errors']==[],'closed formal success')
    for name in ('count_eligible','server_reaped','cleanup_complete','cleanup_task_resolved','client_task_resolved','small_pins_rechecked','logs_rechecked','spawn_attempted','popen_returned'):
        need(complete[name] is True,'trial field: '+name)
    need(complete['performance_qualified'] is False and complete['full_state_qualified'] is False,'producer scope')
    actual={str(p.relative_to(folder)) for p in folder.rglob('*') if p.is_file() and p!=folder/'complete.json'}
    same(sorted(complete['artifacts']),sorted(actual),'complete retained artifact inventory')
    for name,row in complete['artifacts'].items():
        path=relative(folder,dict(path=name,sha256=row['sha256']))
        need(type(row['bytes']) is int and row['bytes']>=0 and path.stat().st_size==row['bytes'],'artifact size')
    same(read(folder/'spec.json'),spec,'registered whole trial spec')
    same(read(folder/'launch.json'),{k:spec[k] for k in ('command','cwd','environment')},'actual launch')
    need(spec['observer_enabled'] is False and spec['command'][1:4]==['-B','-m','monolith.serve']
         and not (folder/'probe').exists(),'official observer-OFF CLI')
    intent=read(folder/'spawn-intent.json');same(intent['command'],spec['command'],'inner command')
    need(intent['cwd']==spec['cwd'] and read(folder/'child.json')['pid']==cpid,'inner direct child')
    raw=read(folder/'child-identity.json')
    same(raw['command'],['/bin/ps','-p',str(cpid),'-o','pid=,ppid=,lstart=,comm='],'raw identity command')
    need(raw['returncode']==0 and not raw['stderr'] and raw['stdout'].split()[:2]==[str(cpid),str(owner_pid)],'raw direct PPID')
    cap=formal.load('../pareto_realistic/http_capture_v2.py')
    trial=formal.load('../lithos_d512_retry/trial.py');join=formal.load('../lithos_d512_retry/join.py')
    for name,command in [('child-absence.json',['/bin/ps','-p',str(cpid),'-o','pid=']),
         ('port-before.json',['/usr/sbin/lsof','-nP','-a',f"-iTCP:{spec['port']}",'-sTCP:LISTEN','-Fpn']),
         ('port-after.json',['/usr/sbin/lsof','-nP','-a',f"-iTCP:{spec['port']}",'-sTCP:LISTEN','-Fpn'])]:
        raw=read(folder/name);same(raw['command'],command,'raw absence command');need(cap.no_process(raw),'raw PID/port absence')
    release=read(folder/'release.json');outer=read(folder.parent/'release.json');own=read(folder.parent/'child.json')
    need(release['pid']==cpid and type(release['returncode']) is int and release['returncode']==outer['returncode']
         and all(release[k] is True for k in ('waited_reaped','strict_absence','port_absent')),'inner actual release')
    need(all(finite(t) for t in (complete['started_at'],complete['ended_at'],release['time']))
         and complete['started_at']<=own['spawned_at']<=release['time']<=complete['ended_at']<=outer['time'],'inner/outer trial time order')
    listener=read(folder/'listener.json');need(cap.owned_listener(listener,cpid,spec['port']),'raw owned listener')
    same(listener['command'],['/usr/sbin/lsof','-nP','-a','-p',str(cpid),f"-iTCP:{spec['port']}",'-sTCP:LISTEN','-Fpn'],'actual listener command')
    for name in ('ready.json','ready-after-warm.json','ready-after-formal.json'):
        need(trial.ready(read(folder/name),spec),'actual served target/capacity')
    groups={};checked={}
    for name in ('warm','formal'):
        group=read(folder/(name+'-group.json'));same(group,read(folder/name/'group.json'),'full group duplicate')
        need(group['frozen_scheduler_sha256']==formal.FIXED['../pareto_realistic/client.py'],'actual cleanup client')
        checked[name]=contract.check_group(group,spec,name)
        same(read(folder/(name+'-raw-check.json')),checked[name],'independently recomputed wire check')
        calculated=metrics(group,join.wire)
        same({k:checked[name]['point'][k] for k in calculated},calculated,'independent committed timing calculation')
        groups[name]=group
        same(complete['groups'][name],dict(path=name+'-group.json',sha256=sha(folder/(name+'-group.json'))),'group receipt')
        same(complete['log_windows'][name],dict(path=name+'-log-check.json',sha256=sha(folder/(name+'-log-check.json'))),'log receipt')
    need(groups['warm']['cleanup']['end_perf_counter_s']<=min(r['start_perf_counter_s'] for r in groups['formal']['requests']),'warm fully closed before formal')
    logs=log_windows(folder,spec,groups,join);point=checked['formal']['point']
    same(complete['point'],point,'terminal raw point')
    same(read(folder/'point.json'),dict(**point,count_eligible=True,performance_qualified=False),'unapproved retained point')
    return dict(phase=phase,**point,n_fresh_processes=1,warm_requests_checked=8,
        point_receipt=ref(folder/'point.json'),formal_raw_group=ref(folder/'formal-group.json'),
        qualification=spec['qualification'],log_windows=logs,pid=cpid)


def audit(owner,journal,plan_sha256):
    root,journal=Path(owner).resolve(),Path(journal).resolve()
    result=dict(schema='E023-lithos-D512-formal-independent-audit-v1',status='INCOMPLETE',qualified=False,
        performance_qualified=False,full_state_qualified=False,output_quality_qualified=False,
        raw_requests_opened=False,audit_source_sha256=sha(__file__),owner_directory=str(root))
    try:
        if not all((root/n).is_file() for n in ('complete.json','release.json','children.json','attempt.json')):
            raise Incomplete('terminal owner evidence absent; no request evidence opened')
        data=read(journal/'journal.json');complete=read(root/'complete.json')
        if data.get('state')!='RESTORED' or complete.get('status')=='INCOMPLETE':raise Incomplete('owner/restoration not terminal')
        if complete.get('status')!='PASS':
            result.update(status='FAIL',producer_errors=complete.get('errors'));return result
        start={str(p):sha(p) for p in (journal/'journal.json',root/'complete.json',root/'owner-plan.json')}
        need(data['owner_directory']==str(root) and data['owner_plan']['sha256']==plan_sha256==sha(root/'owner-plan.json'),'actual formal plan')
        need(data['owner_source']==dict(path=str(HERE/'owner.py'),sha256=FIXED['owner.py']),'frozen owner identity')
        restoration=load('audit_restore.py').audit(journal);result['restoration']=restoration
        need(restoration['status']=='PASS' and restoration['qualified'] is True and restoration['performance_qualified'] is False,'independent restoration/reap failed')
        owned=load('owner.py');formal=load('formal.py');contract=load('contract.py')
        _,plan,_,_,specs=owned.validate_plan(SimpleNamespace(plan=root/'owner-plan.json',plan_sha256=plan_sha256,out=root))
        need(complete['schema']=='E023-lithos-D512-formal-owner-result-v1' and complete['errors']==[]
             and complete['first_failure'] is None and complete['formal_trials_complete'] is True
             and complete['all_children_reaped'] is True and complete['performance_qualified'] is False
             and complete['planned_trials']==list(PHASES),'six closed formal trials')
        children=closed_bindings(root,complete,plan);result['preparation']=legacy.preparations(root,plan)
        same(read(root/'trial-specs.json'),specs,'exact prepared specs')
        published=read(root/'points.json');need(published['schema']=='E023-lithos-D512-formal-cycle-v1'
            and published['performance_qualified'] is False,'producer cycle remains unapproved')
        same(published['trials'],complete['phases'],'published cycle binding')
        need([row['phase'] for row in complete['phases']]==list(PHASES),'all phase entries')
        points=[];result['raw_requests_opened']=True
        for phase,entry,child,spec in zip(PHASES,complete['phases'],children,specs):
            need(entry['status']=='PASS' and entry['concurrency']==spec['concurrency'] and entry['repeat']==spec['repeat'],'phase result identity')
            for key,name in (('result','complete.json'),('point','point.json')):
                relative(root,entry[key],f'phases/{phase}/trial/{name}')
            points.append(trial_evidence(root/'phases'/phase/'trial',spec,child,complete['owner_pid'],formal,contract))
        resource=load('audit_resources.py').audit(root);result['resources']=resource
        need(resource['status']=='PASS' and resource['independent_resource_verified'] is True,'independent raw resources')
        need(all(sha(p)==h for p,h in start.items()) and restoration['journal_sha256']==sha(journal/'journal.json'),'terminal bindings changed')
        support={str((HERE/n).resolve()):h for n,h in FIXED.items()}
        for p,h in support.items():need(sha(p)==h,'audit source changed')
        medians=[]
        for c in (1,8):
            rows=[r for r in points if r['concurrency']==c];need(len(rows)==3,'three fresh processes per C')
            medians.append(dict(concurrency=c,n_fresh_processes=3,**{k:statistics.median(r[k] for r in rows)
                for k in ('aggregate_output_tok_s','e2e_p50_s','e2e_p95_s')},
                scope='Median of three process-level metrics; no pooled request percentile or confidence interval.'))
        result.update(status='PASS',qualified=True,performance_qualified=True,points=points,medians=medians,
            owner_plan_sha256=plan_sha256,owner_complete_sha256=sha(root/'complete.json'),journal_sha256=sha(journal/'journal.json'),
            support_source_pins=support,formal_requests=384,warm_requests=48,raw_point_files_modified=False,
            model_payload_independently_rehashed=False,
            scope='Fixed Qwen3.8-27B NVFP4 D8 normal-EOS cap512, observer OFF. Committed token counts from previously audited official route; not full-state or output-quality equality. HTTP C is not GPU batch size. Throughput uses first formal POST through final DONE; latency includes server queue, excludes waiting for a client slot. Native comparator has one process per C, not three paired replicates.')
    except (Incomplete,FileNotFoundError) as exc:result.update(status='INCOMPLETE',error=dict(type=type(exc).__name__,message=str(exc)))
    except Exception as exc:result.update(status='FAIL',error=dict(type=type(exc).__name__,message=str(exc)))
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('owner','journal','plan-sha256','out'):p.add_argument('--'+name,required=True)
    args=p.parse_args(argv);out=Path(args.out);need(not out.exists() and out.parent.is_dir(),'fresh audit result')
    result=audit(args.owner,args.journal,args.plan_sha256)
    with out.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({k:result[k] for k in ('status','performance_qualified','formal_requests') if k in result}))
    return {'PASS':0,'FAIL':1,'INCOMPLETE':2}[result['status']]

if __name__=='__main__':raise SystemExit(main())
