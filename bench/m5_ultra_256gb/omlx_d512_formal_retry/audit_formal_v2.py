"""Post-run oMLX D512 formal audit with separately versioned resource parser.

Derived from frozen retry/audit_formal.py 31935470b6cf6c0edad1e84b880c05284df1099586a6d335a479b252f452294d.
Only resource parser selection and reported auditor provenance change.

Original SSE usage already counts committed tokens: never subtract terminal EOS.
No raw-ID, full-state or output-quality claim. Retained tail bytes, all client
closers and direct server reap are checked; generator EOF is not inferred.
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
    'owner.py':'fd9b0131cda5edb553b360ed2a124e878d12435d9540630454cbfa9b5f35f1f0',
    'formal.py':'53c289348bd538276096111205296a362b09f53bb718be99035bc51daa492f97',
    'audit_restore.py':'03476fd4f63087209e242ca4e8b072296c1983bca281bf3671e9e5134520c15e',
    'audit_resources_v2.py':'d488e30a1cef33663657a45ab74269753e821b21f1539b078c097be35ee206c5',
    '../omlx_d512/audit_qualification.py':'e75a8a36b8fa53fcca0ff19c668092537e58c88c3257e4baa25f6235fba100f8',
    '../omlx_d512/join_v2.py':'9a09959a840503240f387dc1a6e447311e08b1187b54ba3516e09a60c18cd8d1',
    '../pareto_realistic/http_capture_v2.py':'536fe41d6c8e2bdffe83ba6f43fba87dd6265deb4c84f6204bc9c7220f8280e3',
}
CLIENT_SHA='ef64a114f4f7d90ec5044d6bdcf058ca05f98dea445d6639db706a4fc75217dc'


def load(name):
    path=(HERE/name).resolve()
    if len(FIXED[name])!=64 or hashlib.sha256(path.read_bytes()).hexdigest()!=FIXED[name]:
        raise ValueError('unsealed/changed actual audit dependency: '+name)
    spec=importlib.util.spec_from_file_location('_omlx_formal_audit_'+path.stem,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


legacy=load('../omlx_d512/audit_qualification.py')
wire_module=load('../omlx_d512/join_v2.py')
need,same,sha,read,ref,relative,finite=legacy.need,legacy.same,legacy.sha,legacy.read,legacy.ref,legacy.relative,legacy.finite
Incomplete=legacy.Incomplete


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def entry_binding(plan,spec):
    """The real original CLI script and its qualified interpreter, not a probe."""
    env=HERE.parents[2]/'.artifacts/omlx-baseline/.venv';entry=env/'bin/omlx';python=env/'bin/python'
    need(spec['command']==[str(entry),*spec['official_args']] and spec['observer_enabled'] is False,'actual official observer-OFF entry')
    binding=spec['qualification'];qualified=read(binding['owner_plan']['path'])
    need(sha(binding['owner_plan']['path'])==binding['owner_plan']['sha256'],'actual prior qualification plan')
    same(plan['verification'],qualified['verification'],'same qualified runtime/source/model inventory')
    pins=plan['verification']['groups']['omlx_runtime'];payload=entry.read_bytes()
    need(str(entry) in pins and hashlib.sha256(payload).hexdigest()==pins[str(entry)]['sha256']
         and len(payload)==pins[str(entry)]['bytes']
         and payload.splitlines()[0]==b'#!'+str(python).encode(),'official entry SHA/bytes/exact interpreter shebang')
    runtime=spec['qualification_identity']['runtime']
    need(str(python.resolve()) in pins and runtime['python_executable']==str(python)
         and runtime['python_resolved']==str(python.resolve()),'same qualified actual interpreter identity')
    return dict(entry=ref(entry),python=str(python),resolved_python=str(python.resolve()),
                interpreter_payload_independently_rehashed=False)


def raw_group(group,spec,phase,qualification):
    """Recompute original SSE/body/count/timing; never use claimed point metrics."""
    C=spec['concurrency'];jobs=spec['jobs'][phase];fixtures={f['fixture_id']:f for f in spec['fixtures']}
    count=8 if phase=='warm' else 64
    need(phase in ('warm','formal') and type(C) is int and C in (1,8)
         and len(jobs)==count and len(fixtures)==8,'fixed formal workload')
    need(group['schema']=='E023-http-group-v1' and group['phase']==phase and type(group['concurrency']) is int and group['concurrency']==C
         and group['group_error'] is None and group['qualification_errors']==[],'complete phase group')
    same(group['qualification'],qualification,'actual qualified count contract')
    need(qualification['status']=='qualified' and qualification['concurrency']==C
         and qualification['usage_count_rule']=='identity' and qualification['raw_ids_count_rule']=='none'
         and qualification['normal_eos'] is True
         and qualification['usage_semantics']=='committed_target_output_excluding_terminal_including_reasoning',
         'oMLX committed usage contract')
    same(qualification['request_fixture_map'],{j['request_id']:j['fixture_id'] for j in jobs},'qualified request bindings')
    need(group['base_url']==f"http://127.0.0.1:{spec['port']}"
         and group['request_timeout_s']==spec['deadlines']['request']
         and group['group_timeout_s']==spec['deadlines'][phase],'actual origin/deadlines')
    same(group['connection_policy'],dict(max_connections=2*C,max_keepalive_connections=0,retries=0,follow_redirects=False),'connection policy')
    clean=group['cleanup'];barrier=group['barrier_release_perf_counter_s'];deadline=group['group_deadline_perf_counter_s']
    need(clean['complete'] is True and clean['deadline_exceeded'] is False and clean['errors']==[]
         and clean['pending_tasks']==[] and clean['cancelled_again'] is False and clean['client_close_started'] is True
         and clean['adapter']=='native-close-cancellation-v1' and clean['frozen_client_sha256']==CLIENT_SHA
         and clean['budget_s']==spec['deadlines']['close'],'all bounded client closers complete')
    need(all(finite(t) for t in (barrier,deadline,clean['start_perf_counter_s'],clean['end_perf_counter_s']))
         and deadline==barrier+group['group_timeout_s']
         and clean['start_perf_counter_s']<=clean['end_perf_counter_s'],'group/cleanup clocks')
    need(len(group['requests'])==count and len({j['request_id'] for j in jobs})==count,'complete unique jobs')
    rows=[];response_ids=set();edges=[]
    for job,row in zip(jobs,group['requests']):
        need(set(job)=={'request_id','fixture_id','path','body'},'exact job fields')
        same({k:row[k] for k in job},job,'raw row/job binding')
        f=fixtures[job['fixture_id']]
        expected=dict(model=spec['served_model'],prompt=f['prompt_text'],temperature=0.0,top_p=1.0,repetition_penalty=1.0,
                      max_tokens=512,stream=True,stream_options={'include_usage':True})
        same(job['body'],expected,'frozen full prompt and normal EOS body')
        need(job['path']=='/v1/completions' and f['normal_eos'] is True and f['max_output_tokens']==512,'actual endpoint/cap')
        same(qualification['fixture_contracts'][job['fixture_id']],dict(path=job['path'],body_sha256=canonical_sha(expected)),'canonical full request body SHA')
        need(row['status']=='done' and type(row['http_status']) is int and row['http_status']==200
             and row['done'] is True and row['error'] is None and row['validation_errors']==[]
             and row['count_qualification_errors']==[] and row['transport_valid'] is True,'complete HTTP row')
        w=wire_module.wire(row);rid=w['response_id'];need(rid not in response_ids,'duplicate actual response ID');response_ids.add(rid)
        u=w['usage'];n=u['completion_tokens'];prompt=len(f['prompt_ids'])
        need(all(type(u[k]) is int and u[k]>=0 for k in ('prompt_tokens','completion_tokens','total_tokens'))
             and type(u['prompt_tokens_details']['cached_tokens']) is int and u['prompt_tokens_details']['cached_tokens']==0
             and u['prompt_tokens']==prompt and u['total_tokens']==prompt+n,'raw usage/prompt/cache0')
        counts=dict(prompt_tokens=prompt,completion_tokens=n,total_tokens=prompt+n,
                    prompt_tokens_details={'cached_tokens':0})
        same({k:u[k] for k in counts},counts,'exact official usage count/cache fields')
        # BaseUsage.model_post_init supplies these OpenAI/Anthropic aliases.
        # Optional wire presence never changes the committed-count convention.
        aliases={'input_tokens':prompt,'output_tokens':n}
        need(all(type(u[k]) is int and u[k]==expected for k,expected in aliases.items() if k in u),
             'optional official token aliases must be exact integers matching primary counts')
        timing={'model_load_duration','total_time','time_to_first_token','prompt_eval_duration',
                'generation_duration','prompt_tokens_per_second','generation_tokens_per_second'}
        # Actual installed Usage.model_dump(exclude_none=True) retains these
        # official extensions. Our performance point still uses client clocks.
        extras=set(u)-set(counts)-set(aliases)
        need(extras<=timing and all(finite(u[k]) and u[k]>=0 for k in extras),
             'finite nonnegative official usage timing extensions')
        need((w['finish_reason']=='length' and n==512) or (w['finish_reason']=='stop' and 0<=n<512),
             'normal EOS/length committed count')
        need(row['answer']==w['text'] and row['reasoning']=='' and (n!=0 or w['text']=='')
             and row['finish_reason']==w['finish_reason'] and row['response_ids']==[rid]
             and row['native_fields']=={},'original wire text/ID/no fabricated raw IDs')
        same(row['usage'],u,'retained usage equals original SSE')
        need(type(row['qualified_output_tokens']) is int and row['qualified_output_tokens']==n
             and row['reported_prompt_tokens']==prompt and row['reported_completion_tokens']==n
             and row['reported_cached_prompt_tokens']==0,'identity committed count; no EOS subtraction')
        start,done=row['start_perf_counter_s'],row['done_perf_counter_s']
        need(all(finite(t) for t in (start,done,row['absolute_deadline_perf_counter_s'],row['terminal_perf_counter_s'],row['e2e_s']))
             and barrier<=start<done<=deadline and row['absolute_deadline_perf_counter_s']==start+group['request_timeout_s']
             and done<=row['absolute_deadline_perf_counter_s'] and done<=clean['start_perf_counter_s']
             and row['terminal_perf_counter_s']==done and row['e2e_s']==done-start,'POST/DONE/absolute request clocks')
        edges.extend(((start,1),(done,-1)))
        rows.append(dict(request_id=job['request_id'],fixture_id=job['fixture_id'],response_id=rid,
            committed_output_tokens=n,finish_reason=w['finish_reason'],start_perf_counter_s=start,done_perf_counter_s=done,
            e2e_s=done-start,wire_sha256=w['sha256'],body_sha256=canonical_sha(expected)))
    active=peak=0
    for _,delta in sorted(edges):
        active+=delta;need(0<=active<=C,'actual HTTP outstanding outside registered C');peak=max(peak,active)
    need(active==0 and peak==C,'declared concurrency did not occur')
    latency=[r['e2e_s'] for r in rows];total=sum(r['committed_output_tokens'] for r in rows)
    window=max(r['done_perf_counter_s'] for r in rows)-min(r['start_perf_counter_s'] for r in rows)
    point=dict(concurrency=C,repeat=spec['repeat'],requests=count,committed_output_tokens=total,
        common_window_s=window,aggregate_output_tok_s=total/window,e2e_p50_s=statistics.median(latency),
        e2e_p95_s=sorted(latency)[math.ceil(.95*count)-1],max_http_outstanding=peak,
        stop_requests=sum(r['finish_reason']=='stop' for r in rows),cap_requests=sum(r['finish_reason']=='length' for r in rows),
        request_latencies_s=latency)
    summary=group['summary']
    need(summary['transport_complete'] is True and summary['eligible_under_supplied_qualification'] is True
         and summary['formal_point_eligible'] is (phase=='formal'),'original client complete/count eligibility')
    same([summary[k] for k in ('planned_requests','submitted_requests','done_requests','successful_requests','count_qualified_requests')],[count]*5,'all requests counted')
    same([summary['group_window_s'],summary['aggregate_output_tok_s'],summary['e2e_s']['p50'],summary['e2e_s']['p95'],summary['max_http_outstanding']],
         [window,total/window,point['e2e_p50_s'],point['e2e_p95_s'],peak],'independent raw metrics differ from client')
    return dict(point=point,requests=rows,response_ids=sorted(response_ids),generator_eof_proven=False,
                usage_count_rule='identity',raw_ids_count_rule='none')


def trial_evidence(folder,spec,child,owner_pid,formal):
    complete=read(folder/'complete.json');cpid=child['pid'];phase=child['phase']
    need(complete['schema']=='E023-omlx-D512-formal-result-v1' and complete['status']=='PASS'
         and complete['concurrency']==spec['concurrency'] and complete['repeat']==spec['repeat'] and complete['errors']==[],
         'closed formal success')
    for name in ('count_eligible','server_reaped','cleanup_complete','cleanup_task_resolved','client_task_resolved','small_pins_rechecked','spawn_attempted','popen_returned'):
        need(complete[name] is True,'formal terminal field: '+name)
    need(complete['performance_qualified'] is False and complete['full_state_qualified'] is False,'producer scope')
    actual={str(p.relative_to(folder)) for p in folder.rglob('*') if p.is_file() and p!=folder/'complete.json'}
    same(sorted(complete['artifacts']),sorted(actual),'complete retained artifact inventory')
    for name,item in complete['artifacts'].items():
        path=relative(folder,dict(path=name,sha256=item['sha256']))
        need(type(item['bytes']) is int and item['bytes']>=0 and path.stat().st_size==item['bytes'],'actual retained size')
    same(read(folder/'spec.json'),spec,'registered full trial spec')
    same(read(folder/'launch.json'),{k:spec[k] for k in ('command','cwd','environment')},'actual launch')
    need(spec['observer_enabled'] is False and spec['command']==[formal.OFFICIAL,*spec['official_args']]
         and not (folder/'probe').exists(),'original oMLX CLI observer OFF')
    intent=read(folder/'spawn-intent.json');same(intent['command'],spec['command'],'inner command')
    need(intent['cwd']==spec['cwd'] and read(folder/'child.json')['pid']==cpid,'inner direct server PID')
    cap=load('../pareto_realistic/http_capture_v2.py')
    identity=read(folder/'child-identity.json')
    same(identity['command'],['/bin/ps','-p',str(cpid),'-o','pid=,ppid=,lstart=,comm='],'actual direct identity command')
    need(type(identity['returncode']) is int and identity['returncode']==0 and identity['stderr']==''
         and identity['stdout'].split()[:2]==[str(cpid),str(owner_pid)],'actual direct PPID')
    for name,command in [('child-absence.json',['/bin/ps','-p',str(cpid),'-o','pid=']),
            ('port-before.json',['/usr/sbin/lsof','-nP','-a',f"-iTCP:{spec['port']}",'-sTCP:LISTEN','-Fpn']),
            ('port-after.json',['/usr/sbin/lsof','-nP','-a',f"-iTCP:{spec['port']}",'-sTCP:LISTEN','-Fpn'])]:
        row=read(folder/name);same(row['command'],command,'actual absence command');need(cap.no_process(row),'strict absence')
    release=read(folder/'release.json');outer=read(folder.parent/'release.json');own=read(folder.parent/'child.json')
    need(release['pid']==outer['pid']==cpid and all(release[k] is True for k in ('waited_reaped','strict_absence','port_absent')),'inner/outer actual reap')
    legacy.explained_release(release,outer);exit_evidence=legacy.events_and_exit(folder,complete,release,cpid)
    need(all(finite(t) for t in (complete['started_at'],complete['ended_at'],release['time'],outer['time']))
         and complete['started_at']<=own['spawned_at']<=release['time']<=complete['ended_at']<=outer['time'],'actual trial lifetime')
    listener=read(folder/'listener.json');need(cap.owned_listener(listener,cpid,spec['port']),'owned listener')
    same(listener['command'],['/usr/sbin/lsof','-nP','-a','-p',str(cpid),f"-iTCP:{spec['port']}",'-sTCP:LISTEN','-Fpn'],'owned listener query')
    trial=formal.load('../omlx_d512/trial.py')
    for name in ('ready.json','ready-after-warm.json','ready-after-formal.json'):
        need(trial.ready(read(folder/name),spec),'actual served model before/after groups')
    groups={};checks={};ids=[]
    for label in ('warm','formal'):
        group=read(folder/(label+'-group.json'));same(group,read(folder/label/'group.json'),'full client group duplicate')
        producer_check=formal.check_group(group,spec,label)
        same(read(folder/(label+'-raw-check.json')),producer_check,'recomputed producer raw check')
        checked=raw_group(group,spec,label,formal.count_contract(spec,label))
        same({k:producer_check['point'][k] for k in checked['point']},checked['point'],'independent committed timing check')
        same(complete['groups'][label],dict(path=label+'-group.json',sha256=sha(folder/(label+'-group.json'))),'actual group receipt')
        same(complete['raw_checks'][label],dict(path=label+'-raw-check.json',sha256=sha(folder/(label+'-raw-check.json'))),'actual raw check receipt')
        groups[label]=group;checks[label]=checked;ids.extend(checked['response_ids'])
    need(len(set(ids))==72,'response UUID reused across warm/formal')
    need(groups['warm']['cleanup']['end_perf_counter_s']<=min(r['start_perf_counter_s'] for r in groups['formal']['requests']),'warm fully closed before formal')
    boundaries=read(folder/'phase-boundaries.json');order=('warm-start','warm-closed','formal-start','formal-closed')
    need(set(boundaries)==set(order),'exact phase clock boundaries')
    for i,name in enumerate(order):
        b=boundaries[name];need(set(b)=={'time','monotonic','perf_counter_s'} and all(finite(v) for v in b.values()),'phase clocks')
        need(own['spawned_at']<=b['time']<=release['time'],'phase clock inside server lifetime')
        if i:need(all(b[k]>=boundaries[order[i-1]][k] for k in b),'ordered phase clocks')
    for label in ('warm','formal'):
        need(boundaries[label+'-start']['perf_counter_s']<=groups[label]['barrier_release_perf_counter_s']
             and boundaries[label+'-closed']['perf_counter_s']>=groups[label]['cleanup']['end_perf_counter_s'],
             'phase boundaries bracket complete client close')
    point=checks['formal']['point']
    same({k:complete['point'][k] for k in point},point,'terminal point exact')
    same(read(folder/'point.json'),dict(**complete['point'],count_eligible=True,performance_qualified=False),'unapproved retained point')
    logs={name:ref(folder/name) for name in ('server.stdout','server.stderr')}
    return dict(phase=phase,**point,n_fresh_processes=1,warm_requests_checked=8,
        point_receipt=ref(folder/'point.json'),formal_raw_group=ref(folder/'formal-group.json'),
        qualification=spec['qualification'],pid=cpid,response_ids=ids,exit_evidence=exit_evidence,
        retained_logs=logs,closed_logs_semantically_verified=False,generator_eof_proven=False)


def audit(owner,journal,plan_sha256):
    root,journal=Path(owner).resolve(),Path(journal).resolve()
    result=dict(schema='E023-omlx-D512-formal-independent-audit-v2',status='INCOMPLETE',qualified=False,
        performance_qualified=False,full_state_qualified=False,output_quality_qualified=False,
        raw_requests_opened=False,audit_source_sha256=sha(__file__),frozen_v1_source_sha256='31935470b6cf6c0edad1e84b880c05284df1099586a6d335a479b252f452294d',owner_directory=str(root))
    try:
        if not all((root/n).is_file() for n in ('complete.json','release.json','children.json','attempt.json')):
            raise Incomplete('terminal owner evidence absent; raw requests unopened')
        data=read(journal/'journal.json');complete=read(root/'complete.json')
        if data.get('state')!='RESTORED' or complete.get('status')=='INCOMPLETE':raise Incomplete('owner/restoration not terminal')
        if complete.get('status')!='PASS':
            result.update(status='FAIL',producer_errors=complete.get('errors'));return result
        start={str(p):sha(p) for p in (journal/'journal.json',root/'complete.json',root/'owner-plan.json')}
        need(data['owner_directory']==str(root) and data['owner_plan']['sha256']==plan_sha256==sha(root/'owner-plan.json'),'actual formal plan binding')
        need(data['owner_source']==dict(path=str(HERE/'owner.py'),sha256=FIXED['owner.py']),'frozen formal owner')
        restoration=load('audit_restore.py').audit(journal);result['restoration']=restoration
        need(restoration['status']=='PASS' and restoration['qualified'] is True,'independent restoration/reap')
        owned=load('owner.py');formal=load('formal.py');resource_module=load('audit_resources_v2.py')
        _,plan,_,_,specs=owned.validate_plan(SimpleNamespace(plan=root/'owner-plan.json',plan_sha256=plan_sha256,out=root))
        resource=resource_module.audit(root);result['resources']=resource
        need(resource['status']=='PASS' and resource['independent_resource_verified'] is True,'independent raw resource/lifetime gate')
        _,_,_,_,children,_=resource_module.closed_bindings(root)
        result['preparation']=legacy.preparations(root,plan)
        same(read(root/'trial-specs.json'),specs,'actual prepared specs')
        published=read(root/'points.json')
        need(published['schema']=='E023-omlx-D512-formal-cycle-v1' and published['performance_qualified'] is False
             and complete['performance_qualified'] is False,'producer remains unapproved')
        same(published['trials'],complete['phases'],'published cycle binding')
        need([r['phase'] for r in complete['phases']]==list(PHASES),'all six completed phase entries')
        points=[];all_ids=[];entries=[];result['raw_requests_opened']=True
        for phase,entry,child,spec in zip(PHASES,complete['phases'],children,specs):
            need(entry['status']=='PASS' and entry['concurrency']==spec['concurrency'] and entry['repeat']==spec['repeat'],'phase identity')
            for key,name in (('result','complete.json'),('point','point.json')):relative(root,entry[key],f'phases/{phase}/trial/{name}')
            entries.append(entry_binding(plan,spec))
            point=trial_evidence(root/'phases'/phase/'trial',spec,child,complete['owner_pid'],formal)
            all_ids.extend(point['response_ids']);points.append(point)
        need(len(all_ids)==432 and len(set(all_ids))==432,'actual response ID reuse across six fresh servers')
        need(all(sha(p)==h for p,h in start.items()) and restoration['journal_sha256']==sha(journal/'journal.json'),'terminal receipts changed')
        for name,h in FIXED.items():need(sha((HERE/name).resolve())==h,'audit dependency changed')
        medians=[]
        for c in (1,8):
            rows=[p for p in points if p['concurrency']==c];need(len(rows)==3 and {p['repeat'] for p in rows}=={1,2,3},'three independent process repeats')
            medians.append(dict(concurrency=c,n_fresh_processes=3,
                **{k:statistics.median(p[k] for p in rows) for k in ('aggregate_output_tok_s','e2e_p50_s','e2e_p95_s')},
                scope='Median of process metrics, not a pooled request percentile or CI.'))
        result.update(status='PASS',qualified=True,performance_qualified=True,points=points,process_metric_medians=medians,
            formal_requests=384,warm_requests=48,owner_plan_sha256=plan_sha256,source_pins=FIXED,
            actual_official_entries=entries,
            model_payload_independently_rehashed=False,recursive_runtime_coverage_verified=False,
            scope='Observer-OFF official oMLX D8 normal-EOS cap512; raw SSE usage is committed, no EOS subtraction. '
                  'POST-to-DONE includes server queue, excludes client-slot wait. Retained logs/closers/reap checked, '
                  'not generator EOF or all background-task health. No raw-ID/full-state/output-quality parity claim.')
    except (Incomplete,FileNotFoundError) as exc:result.update(status='INCOMPLETE',error=dict(type=type(exc).__name__,message=str(exc)))
    except Exception as exc:result.update(status='FAIL',error=dict(type=type(exc).__name__,message=str(exc)))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('owner','journal','plan-sha256','out'):p.add_argument('--'+name,required=True)
    args=p.parse_args();result=audit(args.owner,args.journal,args.plan_sha256)
    with Path(args.out).open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    raise SystemExit({'PASS':0,'FAIL':1,'INCOMPLETE':2}[result['status']])
