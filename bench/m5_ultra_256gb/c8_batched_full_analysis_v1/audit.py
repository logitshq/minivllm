"""Offline E051 full recorded-history audit; frozen per-step/full-drain checks.

The graph-only E047 wire history is a numerical reference, not a measurement of
candidate HTTP performance. A future candidate HTTP capture needs its own full
same-schedule qualification. Successful tensor bytes were not retained: this
checks all original per-array device receipts, not offline GPU arithmetic.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

B=Path(__file__).resolve().parents[1]
PREVIOUS=B/'c8_batched_prefix_analysis_v1/audit.py'
PREVIOUS_SHA='34bb62b7554a085577d328d23d75dd6d9d41a82e4d6f02e225d91166a764eb21'
QUALIFIER=B/'c8_batched_full_v1/qualify.py'
QUALIFIER_SHA='7c2b5d6596ae23e8d7eb550a8121ab3c5fece370e803614cfaaa97f96b26eeb3'
BASE_PLAN_SHA='21ba52e43ce54c2b463ab46bfd212c53d06d6303530c77939e3e6b801df547fb'


def support():
    if hashlib.sha256(PREVIOUS.read_bytes()).hexdigest()!=PREVIOUS_SHA:raise ValueError('frozen composition auditor changed')
    s=importlib.util.spec_from_file_location('_e051_composition_audit',PREVIOUS)
    module=importlib.util.module_from_spec(s);s.loader.exec_module(module)
    _,a=module.support()
    return module,a


def full_terminal(a,top,claimed):
    a.need(top['schema']=='c8-batched-gdn-graph-full-result-v1' and top['status']=='PASS'
           and top['source_input_pins_rechecked'] is True and top['full_history_qualified'] is True,
           'successful source-bound full producer required')
    a.need(top['http_qualified'] is top['performance_qualified'] is False
           and top['qualification_scope']=='Only the exact registered E047 recording; no new candidate HTTP capture.',
           'recorded-history scope only')
    a.need(all(type(top[k]) is int and top[k]>1 for k in ('pid','ppid')),'actual numerical process identity')
    a.need(top['steps']==1030 and top['total_requests']==16,'full1030/16, never49/0')
    a.need(claimed['status']=='PASS' and claimed['plan_sha256']==BASE_PLAN_SHA
           and claimed['qualifier_sha256']=='26964fcbed9a2e5dcefaee4888244987897f3138d4a2b7add59bf63a7461b7bf'
           and claimed['numeric_only_not_http'] is claimed['source_model_pins_rechecked'] is True
           and claimed['total_requests']==16 and len(claimed['cases'])==1,'original full base completion')
    case=claimed['cases'][0];a.same(top['case'],case,'actual full case passed through unchanged')
    a.need(case['status']=='PASS' and case['name']=='HTTP-C8' and case['steps']==1030
           and len(case['completed'])==len({r['rid'] for r in case['completed']})==16
           and case['raw_output_count']==case['committed_output_count']==8192,'all unique16/8192 completions')
    return case


def input_binding(a,child,plan,http):
    old=a.read(a.pinned(plan['prefix_plan']))
    original=a.read(a.pinned(old['base_prefix_plan']))
    for path,digest in original['inputs'].items():a.pinned(dict(path=path,sha256=digest))
    for key in ('qualification','recording','environment'):
        a.same(plan[key],original[key],'original full input contract: '+key)
    a.same(plan['capture_directory'],original['capture_directory'],'original E047 capture path')
    a.need(plan['qualification']['sha256']==BASE_PLAN_SHA==a.sha(child/'base/plan.json'),'unchanged original numeric plan')
    numeric=a.read(child/'base/plan.json');fixtures,config,tokenizer,tokens=a.tokenizer_and_manifests(numeric,http)
    a.need(numeric['configuration']['cache_limit_gib']==2,'original allocator cache policy')
    capture=Path(plan['capture_directory']);spec=a.read(capture/'spec.json')
    a.same({k:v for k,v in spec['environment'].items() if k.startswith('MLX_')},plan['environment'],'original MLX environment')
    a.same(spec['fixtures'],fixtures,'full original fixture manifest')
    a.need(spec['concurrency']==8 and spec['wave_names']==['warm','screen'] and len(spec['jobs'])==16,'original sequential C8 waves')
    group=a.read(capture/'group.json');rows=[];refs=[]
    for phase in ('warm','screen'):
        path=capture/'client'/phase/'group.json';wave=a.read(path);clean=wave['cleanup']
        a.need(len(wave['requests'])==8 and wave['group_error'] is None and clean['complete'] is True
               and clean['errors']==clean['pending_tasks']==[] and clean['deadline_exceeded'] is clean['cancelled_again'] is False,
               'original complete eight-request wave')
        rows.extend(wave['requests']);refs.append(dict(phase=phase,path=str(path),sha256=a.sha(path)))
    a.same(group['requests'],rows,'original raw body/SSE rows retained');a.same(group['source_groups'],refs,'derived raw sources')
    join=a.load(B/'pareto_realistic/http_join.py',plan['sources'][str(B/'pareto_realistic/http_join.py')])
    with (capture/'trace.jsonl').open('rb') as stream:
        recording=join.join(group,stream,spec['jobs'],{f['fixture_id']:f for f in fixtures},spec['identity'],
            lambda ids:tokenizer.decode(ids,skip_special_tokens=False))
    a.same(a.read(plan['recording']['path']),recording,'independent complete SSE/trace rejoin')
    a.need(len(recording['steps'])==1030 and len(recording['jobs'])==16
           and all(j['cap']==len(j['raw_ids'])==len(j['committed_ids'])==512 for j in recording['jobs'].values()),
           'unchanged full16 jobs/caps/8192 IDs')
    route_path=a.pinned(plan['route_events'])
    a.need(route_path==capture/'route-events.jsonl' and plan['route_events']['sha256']==original['inputs'][str(route_path)],'full original route input')
    header=dict(event='route-startup',schema='E024-route-http-events-v1',pid=a.read(capture/'release.json')['pid'],phase='HTTP-C8',identity=spec['identity'])
    return numeric,recording,fixtures,config,tokens,header,original['inputs']


def audit(child,expected_sha):
    previous,a=support();child=Path(child).resolve()
    result=dict(schema='E051-independent-full-audit-v1',status='INCOMPLETE',full_history_qualified=False,
        recorded_history_verified=False,candidate_http_capture_qualified=False,http_qualified=False,
        performance_qualified=False,owner_resources_restore_qualified=False,
        offline_tensor_bytes_rechecked=False,weight_shards_rehashed=False,
        audit_source_sha256=a.sha(__file__),child=str(child),plan_sha256=expected_sha,
        scope='Only the complete original graph-only E047 1030-step/16-request arrival history, replayed with actual batched GDN+graph. No future candidate HTTP capture, arbitrary arrivals or performance grant.',
        evidence_limit='Full per-array finite/bit comparison receipts from the pinned device producer; successful tensor payloads were not saved. No offline GPU arithmetic/tensor-byte reread.')
    try:
        top=a.read(child/'complete.json');claimed=a.read(child/'base/complete.json');plan=a.read(child/'plan.json')
        case=full_terminal(a,top,claimed)
        a.need(a.sha(child/'plan.json')==expected_sha==top['plan_sha256'],'explicit actual full plan SHA')
        a.need(plan['schema']=='c8-batched-gdn-graph-full-plan-v1'
               and plan['expected_steps']==1030 and plan['expected_requests']==16 and plan['expected_route_calls']==1028
               and plan['allocator_cache_policy']=='unchanged-2gib'
               and plan['full_history_qualified'] is plan['http_qualified'] is plan['performance_qualified'] is False,'registered full-run plan')
        a.same(plan['switches'],dict(gdn_prework=True,route_graph=True),'actual combined candidate ON')
        a.need(plan['sources'][str(QUALIFIER)]==QUALIFIER_SHA,'actual full qualifier source')
        for name,digest in previous.NEW_SOURCES.items():a.need(plan['sources'][str(B/name)]==digest,'frozen composition source chain')
        for path,digest in {**plan['sources'],**plan['predecessor_inputs']}.items():a.pinned(dict(path=path,sha256=digest))
        for name in ('prefix_plan','predecessor','qualification','recording','route_events'):a.pinned(plan[name])
        a.same(top['base_result'],dict(path=str(child/'base/complete.json'),sha256=a.sha(child/'base/complete.json'),returncode=0,status='PASS'),'actual base terminal binding')
        http=a.load(a.HTTP_AUDIT,a.HTTP_AUDIT_SHA);base=a.load(a.BASE,a.BASE_SHA);serving=a.load(a.SERVING,a.SERVING_SHA)
        numeric,recording,fixtures,config,tokens,header,inputs=input_binding(a,child,plan,http)
        a.same(claimed['candidate'],numeric['candidate'],'actual graph/gates settings')
        a.same(top['peak_mlx_bytes'],claimed['peak_bytes'],'MLX allocator receipt');a.same(top['elapsed_base_seconds'],claimed['elapsed_wall_s'],'base elapsed receipt')
        events=a.pinned(top['composition_events']);replay=a.pinned(top['replay_route_events']);captured=a.pinned(plan['route_events'])
        a.need(events==child/'composition.jsonl' and replay==child/'replay-route-events.jsonl','local emitted sidecars')
        stable={str(p):a.sha(p) for p in (child/'plan.json',child/'complete.json',child/'base/plan.json',child/'base/complete.json',events)}
        with events.open('rb') as stream:
            rows=[]
            for raw in stream:
                a.need(raw.endswith(b'\n'),'truncated composition row');rows.append(a.parse(raw))
        result['composition']=previous.composition(a,rows,top['composition'])
        identities={p:a.file_identity(p) for p in (captured,replay,child/'base/steps.jsonl')};counts={}
        with captured.open('rb') as capture_stream,replay.open('rb') as replay_stream:
            routes=a.DualRoutes(serving,capture_stream,replay_stream,header,plan['route_events'])
            base.CaseAudit=a.route_case_type(base,routes,counts)
            result['ledger']=http.stream_ledger(child/'base/steps.jsonl',recording,fixtures,config,case,base)
            result['routes']=routes.finish()  # Both complete streams must end here.
        stats=result['routes']['captured'];bounds=stats['route_hits']['HTTP-C8'];hits=sum(bounds.values())
        a.need(routes.replay_pid==top['pid'],'actual numerical child PID in route sidecar')
        a.same(top['coverage'],dict(installs=1,restores=2,route_hits=stats['route_hits'],long_fallback_calls=stats['long_fallback_calls'],identity_refill_enabled=False),'full actual route coverage')
        a.same(top['route_comparison'],dict(calls=routes.calls,actual_compiled_invocations=hits,
            fallbacks=stats['long_fallback_calls'],bounds=bounds,captured_sha256=stats['sha256'],bytes=stats['bytes'],lines=stats['lines'],
            scope='Ordered actual calls per recorded step, including actual ahead behavior; SUBMITTED alone is not numeric PASS.'),'independent complete route accounting')
        a.need(routes.calls==hits==1028 and stats['long_fallback_calls']==0
               and stats['sha256']==plan['route_events']['sha256']
               and result['routes']['replay']['sha256']==top['replay_route_events']['sha256'],'complete1028 calls and actual route SHAs')
        a.need(result['ledger']['requests']==16 and result['ledger']['steps']==1030
               and result['ledger']['raw_output_count']==result['ledger']['committed_output_count']==8192,'independent full drain')
        for path,identity in identities.items():a.need(a.file_identity(path)==identity,'closed stream changed during audit')
        for path,digest in {**stable,**plan['sources'],**plan['predecessor_inputs'],**inputs,**tokens}.items():a.need(a.sha(path)==digest,'closed source/input changed during audit')
        result.update(status='PASS',full_history_qualified=True,recorded_history_verified=True,
            terminal_sha256=stable,raw_inputs=inputs,tokenizer_runtime_pins=tokens,ahead_retirement=counts,
            helper_sources={str(PREVIOUS):PREVIOUS_SHA,str(a.HTTP_AUDIT):a.HTTP_AUDIT_SHA,str(a.BASE):a.BASE_SHA,str(a.SERVING):a.SERVING_SHA},
            actual_qualifier=dict(path=str(QUALIFIER),sha256=QUALIFIER_SHA),
            peak_mlx_bytes_receipt=top['peak_mlx_bytes'],mlx_peak_is_physical_footprint=False)
    except FileNotFoundError as exc:result.update(status='INCOMPLETE',reason=str(exc))
    except Exception as exc:result.update(status='FAIL',first_failure=dict(type=type(exc).__name__,message=str(exc)))
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--child',required=True,type=Path)
    p.add_argument('--plan-sha256',required=True);p.add_argument('--out',required=True,type=Path);args=p.parse_args(argv)
    if args.out.exists() or not args.out.parent.is_dir():raise ValueError('new output in existing directory required')
    result=audit(args.child,args.plan_sha256)
    with args.out.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({k:result[k] for k in ('status','recorded_history_verified','full_history_qualified')}))
    return {'PASS':0,'FAIL':1,'INCOMPLETE':2}[result['status']]


if __name__=='__main__':raise SystemExit(main())
