"""Independent E053 full numerical replay of a new candidate HTTP capture.

The original full-state/logit/ahead checker is unchanged. Actual raw two-wave
SSE/trace inputs determine all counts; the batched cursor checks Python tracing
and reuse per bound/B without interpreting them as GPU kernel counts. No tensor
payload recomputation, service ownership, arbitrary-arrival or speed grant.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

HERE=Path(__file__).resolve().parent;B=HERE.parent
ROUTES=HERE/'routes.py'
ROUTES_SHA='6c214e1a101481deccdc9ceed9a60f7d083055d26b5e7dd569022b873aeba2e6'
QUALIFIER=B/'c8_batched_http_replay_v1/qualify.py'
QUALIFIER_SHA='c0a73e649718586365b42e6fcb67e4d0fbdb309f2238526b798952a9bfd18ad5'
SERVER=B/'c8_batched_http_v1/server.py'
SERVER_SHA='a3737f1b364e190e3fe45a1779e1be228c4f434701bf8875dbe18f1d519afc7b'
TRIAL=B/'c8_batched_http_v2/trial.py'
TRIAL_SHA='a45bd7db4a30c6d6e16b1b2b2ef0ce4c139f9cc8b6fc344ee92219c42e1c7a72'
INTEGRATION=B/'gdn_batched_graph/integration.py'
INTEGRATION_SHA='6e2e4d4350c36476292f2d0ff2bc7e095561f76a2c233b1a99575027378d50b7'
GDN=B/'gdn_prework_batched/candidate.py'
GDN_SHA='acd2a23802296b9e420108203cc215c3170cd1eccd9a859e1a787432f98a5353'


def support():
    if hashlib.sha256(ROUTES.read_bytes()).hexdigest()!=ROUTES_SHA:raise ValueError('new cursor source changed')
    s=importlib.util.spec_from_file_location('_e053_actual_routes_audit',ROUTES)
    module=importlib.util.module_from_spec(s);s.loader.exec_module(module)
    return module,module.original


def terminal(a,top,claimed):
    a.need(top['schema']=='c8-batched-http-replay-result-v1' and top['status']=='PASS'
           and top['replay_numerical_checks_passed'] is top['source_input_pins_rechecked'] is top['independent_audit_required'] is True,
           'actual completed new HTTP replay required')
    a.need(all(top[k] is False for k in ('numerical_qualified','http_qualified','performance_qualified','owner_resources_restore_qualified')),
           'producer cannot grant independent qualification')
    a.need(all(type(top[k]) is int and top[k]>1 for k in ('pid','ppid')),'actual replay process receipt')
    a.need(type(top['steps']) is int and top['steps']>0 and top['total_requests']==16,'actual positive steps/full16')
    a.need(claimed['status']=='PASS' and claimed['plan_sha256']==a.BASE_PLAN_SHA
           and claimed['qualifier_sha256']=='26964fcbed9a2e5dcefaee4888244987897f3138d4a2b7add59bf63a7461b7bf'
           and claimed['numeric_only_not_http'] is claimed['source_model_pins_rechecked'] is True
           and claimed['total_requests']==16 and len(claimed['cases'])==1,'complete original base checks')
    case=claimed['cases'][0];a.same(top['case'],case,'original full case retained')
    a.need(case['status']=='PASS' and case['name']=='HTTP-C8' and case['steps']==top['steps']
           and len(case['completed'])==len({r['rid'] for r in case['completed']})==16,'complete unique16 drain')
    return case


def dynamic_counts(recording):
    return dict(expected_steps=len(recording['steps']),expected_requests=len(recording['jobs']),
        expected_raw_outputs=sum(len(j['raw_ids']) for j in recording['jobs'].values()),
        expected_committed_outputs=sum(len(j['committed_ids']) for j in recording['jobs'].values()))


def capture_binding(a,child,plan,http):
    capture=Path(plan['capture_directory']);spec=a.read(a.pinned(plan['capture_spec']))
    a.need(Path(plan['capture_spec']['path'])==capture/'spec.json' and spec['schema']=='c8-batched-http-spec-v2' and 'diagnostic_only' not in spec['identity']
           and spec['concurrency']==8 and spec['wave_names']==['warm','screen'] and len(spec['jobs'])==16,'actual new two-wave specification')
    files={p.name:p for p in capture.iterdir() if p.is_file()}
    expected={str(p) for p in files.values()}|{str(capture/n) for n in
        ('client/warm/group.json','client/screen/group.json','joined-input/receipt.json','joined-input/recording.json')}
    a.same(sorted(plan['capture_inputs']),sorted(expected),'exact captured input inventory')
    for path,digest in plan['capture_inputs'].items():a.pinned(dict(path=path,sha256=digest))
    complete=a.read(capture/'complete.json');process=a.read(capture/'child.json');release=a.read(capture/'release.json')
    a.need(complete['schema']=='c8-batched-http-result-v2' and complete['status']=='PASS'
           and all(complete[k] is True for k in ('capture_complete','server_reaped','cleanup_complete','cleanup_task_resolved',
               'client_task_resolved','small_pins_rechecked','route_events_retained')) and complete['errors']==[],'closed successful actual capture')
    a.need(all(complete[k] is False for k in ('numerical_qualified','full_state_qualified','performance_qualified','replay_input_checked')),
           'unqualified original capture preserved')
    a.need(type(process['pid']) is int and process['pid']>1 and process['pid']==release['pid']
           and all(release[k] is True for k in ('waited_reaped','strict_absence','port_absent')),'closed captured process receipt')
    a.same(sorted(complete['artifacts']),sorted(set(files)-{'complete.json'}),'complete original top-level inventory')
    for name,path in files.items():
        if name!='complete.json':a.same(complete['artifacts'][name],dict(sha256=a.sha(path),bytes=path.stat().st_size),'captured bytes: '+name)
    a.same(a.read(capture/'launch.json'),dict(command=spec['command']+['--qualification-trace',str(capture/'trace.jsonl'),
        '--route-events',str(capture/'route-events.jsonl')],environment=spec['environment']),'actual capture command/environment')
    a.same(spec['base_plan'],plan['qualification'],'same numeric model/workload')
    a.same({k:v for k,v in spec['environment'].items() if k.startswith('MLX_')},plan['environment'],'same MLX environment')
    identity=spec['identity'];combined=identity['combined_candidates']
    a.need(identity['server_source_path']==str(SERVER) and identity['server_source_sha256']==SERVER_SHA
           and combined['schema']=='GDN-batched-graph-http-adapter-v1'
           and combined['gdn_candidate_domain']=='B1..8' and combined['gdn_prework_enabled'] is True
           and combined['route_decode_graph_enabled'] is True and combined['installed'] is True
           and combined['gdn_layer_installs']==48 and combined['route_observer'] is True
           and combined['integration_path']==str(INTEGRATION) and combined['integration_sha256']==INTEGRATION_SHA
           and combined['trace_policy_source_path']==str(SERVER) and combined['trace_policy_source_sha256']==SERVER_SHA,
           'actual candidate and private batched TraceGraph source identity')
    a.need(plan['qualification']['sha256']==a.BASE_PLAN_SHA==a.sha(child/'base/plan.json'),'unchanged original numeric plan')
    numeric=a.read(child/'base/plan.json');fixtures,config,tokenizer,tokens=a.tokenizer_and_manifests(numeric,http)
    a.need(numeric['configuration']['cache_limit_gib']==2 and plan['allocator_cache_policy']=='unchanged-2gib','same allocator cache')
    a.same(spec['fixtures'],fixtures,'full fixed fixture manifest')
    group=a.read(capture/'group.json');rows=[];refs=[]
    for phase in ('warm','screen'):
        path=capture/'client'/phase/'group.json';wave=a.read(path);clean=wave['cleanup']
        a.need(len(wave['requests'])==8 and wave['group_error'] is None and clean['complete'] is True
               and clean['errors']==clean['pending_tasks']==[] and clean['deadline_exceeded'] is clean['cancelled_again'] is False
               and wave['summary']['transport_complete'] is True and wave['summary']['formal_point_eligible'] is False,'actual completed8 wave')
        rows.extend(wave['requests']);refs.append(dict(phase=phase,path=str(path),sha256=a.sha(path)))
    a.same(group['requests'],rows,'original raw body/SSE rows');a.same(group['source_groups'],refs,'raw group refs')
    join=a.load(B/'pareto_realistic/http_join.py',plan['sources'][str(B/'pareto_realistic/http_join.py')])
    with (capture/'trace.jsonl').open('rb') as stream:
        recording=join.join(group,stream,spec['jobs'],{f['fixture_id']:f for f in fixtures},identity,
            lambda ids:tokenizer.decode(ids,skip_special_tokens=False))
    a.need(len(recording['jobs'])==16 and recording['steps'] and all(j['cap']==512 for j in recording['jobs'].values()),'actual16 originalcaps complete history')
    a.same(a.read(plan['recording']['path']),recording,'new raw rejoin versus registered recording')
    a.same(a.read(child/'recording.json'),recording,'actual replay received unchanged full new recording')
    receipt=a.read(capture/'joined-input/receipt.json')
    a.need(receipt['schema']=='c8-batched-http-joined-input-v2' and receipt['inherited_join_schema']=='E047-joined-input-v1'
           and receipt['replay_numerical_checked'] is receipt['performance_qualified'] is False,'original unqualified joined receipt')
    for key,value in dict(spec_sha256=a.sha(capture/'spec.json'),source_groups=refs,trace_sha256=a.sha(capture/'trace.jsonl'),
        recording_sha256=a.sha(capture/'joined-input/recording.json'),sources=spec['sources']).items():a.same(receipt[key],value,'joined binding: '+key)
    for key,value in dynamic_counts(recording).items():a.same(plan[key],value,'dynamic count from actual raw inputs: '+key)
    header=dict(event='route-startup',schema='GDN-graph-combined-http-events-v1',pid=process['pid'],phase='HTTP-C8',identity=identity)
    a.same(plan['route_header'],header,'exact captured startup/PID/source')
    a.need(Path(plan['route_events']['path'])==capture/'route-events.jsonl','actual captured route path')
    return numeric,recording,fixtures,config,tokens,header


def audit(child,expected_sha):
    routes_module,a=support();child=Path(child).resolve()
    result=dict(schema='E053-independent-http-replay-audit-v1',status='INCOMPLETE',qualified=False,
        recorded_history_verified=False,http_history_qualified=False,candidate_http_capture_qualified=False,
        performance_qualified=False,owner_resources_restore_qualified=False,offline_tensor_bytes_rechecked=False,
        weight_shards_rehashed=False,child=str(child),plan_sha256=expected_sha,audit_source_sha256=a.sha(__file__),
        scope='Only the complete actual candidate two-wave HTTP history. No arbitrary arrivals, unobserved workloads or performance grant.',
        evidence_limit='Complete source-bound device full-array bit/finite receipts, independently reconstructed IDs/schedule/route/tracing. Successful raw tensor payloads not retained; no offline arithmetic or tensor-byte reread.',
        private_source_binding='Actual qualifier clones original combined adapted/Handle constructor globals to the pinned server TraceGraph, whose delta policy is B1..8 candidate/zero fallback. Original replay arithmetic and restore are unchanged.')
    try:
        a.need(len(QUALIFIER_SHA)==64,'new qualifier source not sealed')
        top=a.read(child/'complete.json');claimed=a.read(child/'base/complete.json');plan=a.read(child/'plan.json');case=terminal(a,top,claimed)
        a.need(a.sha(child/'plan.json')==expected_sha==top['plan_sha256'],'actual new replay plan SHA')
        a.need(plan['schema']=='c8-batched-http-replay-plan-v1' and all(plan[k] is False for k in
            ('numerical_qualified','http_qualified','performance_qualified')),'registered new replay scope')
        a.same(plan['switches'],dict(gdn_prework=True,route_graph=True),'combined actual candidate ON')
        source_pins={str(QUALIFIER):QUALIFIER_SHA,str(SERVER):SERVER_SHA,str(TRIAL):TRIAL_SHA,str(INTEGRATION):INTEGRATION_SHA,str(GDN):GDN_SHA,
            str(B/'gdn_graph_combined/http_qualification.py'):'33b86cb57bf3c4616762c1bd1cfea0abb8f7f063325e9dfe815642e8f591bc7d'}
        for path,digest in source_pins.items():a.need(plan['sources'][path]==digest,'actual candidate source chain: '+path)
        for path,digest in plan['sources'].items():a.pinned(dict(path=path,sha256=digest))
        for key in ('capture_spec','predecessor','qualification','recording','route_events'):a.pinned(plan[key])
        a.same(top['base_result'],dict(path=str(child/'base/complete.json'),sha256=a.sha(child/'base/complete.json'),returncode=0,status='PASS'),'base actual terminal')
        http=a.load(a.HTTP_AUDIT,a.HTTP_AUDIT_SHA);base=a.load(a.BASE,a.BASE_SHA);serving=a.load(a.SERVING,a.SERVING_SHA)
        numeric,recording,fixtures,config,tokens,header=capture_binding(a,child,plan,http)
        a.same(claimed['candidate'],numeric['candidate'],'actual native graph/gates settings')
        a.need(top['steps']==plan['expected_steps'] and top['total_requests']==plan['expected_requests'],'actual new full counts')
        a.need(top['recording_sha256']==a.sha(child/'recording.json'),'actual copied recording SHA')
        a.same(top['peak_mlx_bytes'],claimed['peak_bytes'],'MLX allocator receipt');a.same(top['elapsed_base_seconds'],claimed['elapsed_wall_s'],'elapsed receipt')
        captured=a.pinned(plan['route_events']);replay=a.pinned(top['replay_route_events']);a.need(replay==child/'replay-route-events.jsonl','local emitted route stream')
        stable={str(p):a.sha(p) for p in (child/'plan.json',child/'complete.json',child/'base/plan.json',child/'base/complete.json',child/'recording.json')}
        identities={p:a.file_identity(p) for p in (captured,replay,child/'base/steps.jsonl')};counts={}
        with captured.open('rb') as left,replay.open('rb') as right:
            routes=routes_module.DualRoutes(serving,left,right,header,plan['route_events'])
            facade=SimpleNamespace(**{**vars(base),'CaseAudit':routes_module.route_case_type(base,routes,counts)})
            result['ledger']=http.stream_ledger(child/'base/steps.jsonl',recording,fixtures,config,case,facade)
            result['routes']=routes.finish()
        routed=result['routes'];stats=routed['captured'];bounds=stats['route_hits']['HTTP-C8'];hits=sum(bounds.values())
        a.need(routes.replay_pid==top['pid'],'route sidecar actual replay PID')
        a.same(top['coverage'],dict(installs=1,restores=2,route_hits=stats['route_hits'],long_fallback_calls=stats['long_fallback_calls'],
            identity_refill_enabled=False,combined=routed['combined']),'independent combined route coverage')
        a.same(top['route_comparison'],dict(calls=routes.calls,actual_compiled_invocations=hits,fallbacks=stats['long_fallback_calls'],
            bounds=bounds,captured_sha256=stats['sha256'],bytes=stats['bytes'],lines=stats['lines'],
            scope='Ordered actual calls per recorded step, including actual ahead behavior; SUBMITTED alone is not numeric PASS.',
            combined_kinds={key:routed['combined'][key] for key in ('compiled-tracing','compiled-reuse','eager-fallback')},
            tracing_scope='Python tracing/fallback counts; compiled reuse does not execute48 Python hooks'),'complete independently counted routes')
        for key,value in dict(expected_route_calls=routes.calls,expected_compiled_calls=hits,expected_route_fallbacks=stats['long_fallback_calls']).items():
            a.same(plan[key],value,'actual route count: '+key)
        a.need(stats['sha256']==plan['route_events']['sha256'] and routed['replay']['sha256']==top['replay_route_events']['sha256'],'complete both stream SHA/EOF')
        ledger=result['ledger'];a.need(ledger['requests']==16 and ledger['steps']==plan['expected_steps']
            and ledger['raw_output_count']==plan['expected_raw_outputs'] and ledger['committed_output_count']==plan['expected_committed_outputs'],'full new drain')
        for path,identity in identities.items():a.need(a.file_identity(path)==identity,'closed stream changed')
        for path,digest in {**stable,**plan['sources'],**plan['capture_inputs'],**tokens}.items():a.need(a.sha(path)==digest,'source/input changed during audit')
        result.update(status='PASS',qualified=True,recorded_history_verified=True,http_history_qualified=True,
            candidate_http_capture_qualified=True,terminal_sha256=stable,raw_inputs=plan['capture_inputs'],
            helper_sources={str(ROUTES):ROUTES_SHA,str(routes_module.ORIGINAL):routes_module.ORIGINAL_SHA},
            actual_candidate_sources=source_pins,tokenizer_runtime_pins=tokens,ahead_retirement=counts,
            peak_mlx_bytes_receipt=top['peak_mlx_bytes'],mlx_peak_is_physical_footprint=False,
            tracing_scope='Observed per-layer Python tracing/eager deltas and same-key compiled reuse; no GPU kernel/completion count.')
    except FileNotFoundError as exc:result.update(status='INCOMPLETE',reason=str(exc))
    except Exception as exc:result.update(status='FAIL',first_failure=dict(type=type(exc).__name__,message=str(exc)))
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--child',required=True,type=Path)
    p.add_argument('--plan-sha256',required=True);p.add_argument('--out',required=True,type=Path);args=p.parse_args(argv)
    if args.out.exists() or not args.out.parent.is_dir():raise ValueError('new output in existing directory required')
    result=audit(args.child,args.plan_sha256)
    with args.out.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({k:result[k] for k in ('status','recorded_history_verified','candidate_http_capture_qualified')}))
    return {'PASS':0,'FAIL':1,'INCOMPLETE':2}[result['status']]


if __name__=='__main__':raise SystemExit(main())
