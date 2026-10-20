"""Offline E049 warm49 audit; frozen full-array/schedule checkers, prefix finish.

Successful tensor bytes were not retained: finite/bit checks below audit original
producer per-array receipts and complete coverage, not offline GPU arithmetic.
Raw SSE IDs, recorded schedule and route events are independently reconstructed.
No model/device import, owner/resources/restoration or performance qualification.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

B = Path(__file__).resolve().parents[1]
ROUTE = B / 'route_decode_graph/http_audit_v2.py'
ROUTE_SHA = '88f32e3b3f44206b48a1f581cbfe0bde0788060cbd2a0c5030a3ec472a0139fe'
PREFIX_SHA = 'a71b2e2fc3ff2b063efe234c5a75f9f85588f6b285e4daf3c82cee2f35d3f0b7'
BASE_PLAN_SHA = '21ba52e43ce54c2b463ab46bfd212c53d06d6303530c77939e3e6b801df547fb'
D06 = '93f79410dd74462c98a477df47f9132b'
SCOPE = 'Warm49 prefix only, incomplete by design; no full-history, performance or Pareto qualification.'


def support():
    if hashlib.sha256(ROUTE.read_bytes()).hexdigest() != ROUTE_SHA:
        raise ValueError('frozen independent route auditor changed')
    spec = importlib.util.spec_from_file_location('_e049_route_audit', ROUTE)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def prefix_finish(a, case, claimed, recording):
    """Only departure from frozen HTTP checker: registered incomplete boundary."""
    a.need(case.step == 49 and len(case.requests) == 8 and len(case.seen) == 8
           and not case.completed and len(recording['jobs']) == 16, 'actual49/8active/8future boundary')
    delivered = {rid: req['ids'] for rid, req in case.requests.items()}
    for rid, ids in delivered.items():
        a.same(ids, recording['jobs'][rid]['raw_ids'][:len(ids)], 'full recorded prefix IDs: ' + rid)
    a.need(len(delivered[D06]) == 49, 'D06 actual49 IDs')
    expected = dict(name='HTTP-C8-prefix', status='DIAGNOSTIC_PREFIX_COMPLETE',
        completion_state='INCOMPLETE_BY_DESIGN', steps=49, completed=[], delivered_prefix_ids=delivered,
        active_request_ids=sorted(case.requests), future_unadmitted_request_ids=sorted(set(recording['jobs'])-case.seen),
        full_history_qualified=False, performance_qualified=False, scope=SCOPE)
    a.same(claimed, expected, 'prefix receipt versus independently reconstructed state/IDs')
    return dict(steps=49, active_requests=8, future_unadmitted_requests=8, completed_requests=0,
        delivered_ids=sum(map(len, delivered.values())), d06_ids=delivered[D06],
        by_fixture={recording['jobs'][rid]['fixture_id']: dict(rid=rid, count=len(ids), ids=ids)
                    for rid, ids in delivered.items()}, decode_batch_histogram=dict(case.hist),
        array_counts=dict(case.array_counts), ahead_counts=dict(case.ahead_counts))


def stream_prefix(a, http, base, path, recording, fixtures, config, claimed):
    case = http.case_type(base)(recording, fixtures, config)
    events = []; digest = hashlib.sha256(); size = lines = 0
    wanted = ['load-pair-start', 'load-pair-complete', 'release-pair-start', 'release-pair-complete']
    before = a.file_identity(path)
    with path.open('rb') as stream:
        for raw in stream:
            size += len(raw); lines += 1; digest.update(raw)
            a.need(raw.endswith(b'\n'), 'truncated ledger line'); row = a.parse(raw)
            if 'event' in row:
                a.need(len(events) < 4 and row['event'] == wanted[len(events)], 'one-pair lifecycle order')
                t = row['elapsed_wall_s']; a.need(a.finite(t) and (not events or t >= events[-1]['elapsed_wall_s']), 'lifecycle clock')
                if row['event'].startswith('load-pair'):
                    a.need(case.step == 0 and row['max_seqs'] == 8, 'load precedes49 steps')
                    a.need(row.get('model_instances') == 2 if len(events) == 0 else row.get('independent_weight_arrays') == 1349,
                           'independent two-model weight handles')
                else:
                    a.need(case.step == 49, 'release only after registered49 steps')
                    a.need(row.get('model_instances') == 2 if len(events) == 2 else row.get('qualification_only') is True,
                           'pair release receipt')
                events.append(row)
            else:
                a.need(len(events) == 2 and case.step < 49, 'step outside registered loaded prefix')
                try: case.accept(row)
                except Exception as exc: raise ValueError(f'ledger line {lines}, step {row.get("step")}: {exc}') from exc
    a.need(a.file_identity(path) == before and size == before[2] and len(events) == 4, 'closed complete prefix ledger')
    return dict(sha256=digest.hexdigest(), bytes=size, lines=lines, events=events,
        bit_equal_receipts=case.array_counts['all_arrays'], finite_receipts=2*case.array_counts['all_arrays'],
        **prefix_finish(a, case, claimed, recording))


def routes_finish(a, routes, top, stop):
    captured, replay = routes.cursors
    emitted = replay.finish()  # Replay must end exactly at the checked boundary.
    for key in ('hits', 'fallbacks', 'lines'):
        a.same(getattr(captured, key), getattr(replay, key), 'captured/replayed route accounting: '+key)
    stats = dict(calls=routes.calls, hits=sum(sum(b.values()) for b in captured.hits.values()),
        fallbacks=captured.fallbacks, consumed_bytes=captured.bytes, consumed_lines=captured.lines,
        consumed_sha256=captured.digest.hexdigest(), full_route_stream_consumed=False)
    a.same(top['route_prefix'], stats, 'exact captured route prefix byte boundary')
    a.same(top['coverage'], dict(installs=1, restores=2, route_hits=captured.hits,
        long_fallback_calls=captured.fallbacks, identity_refill_enabled=False), 'independent graph coverage')
    tail = captured.stream.readline(); a.need(tail.endswith(b'\n'), 'original capture must retain unconsumed tail')
    next_row = a.parse(tail)
    a.need(next_row['event'] == 'route-call' and next_row['step'] > stop, 'unconsumed next route after49')
    a.need(routes.replay_pid == top['pid'] and routes.calls > 0, 'actual replay PID and graph calls')
    return dict(captured_prefix=stats, replay=emitted, next_unconsumed_route=next_row,
        actual_domains=[dict(B=b, used=u, physical_capacity=L) for b,u,L in sorted(routes.domain)])


def binding(a, child, expected_sha, http):
    plan = a.read(child/'plan.json'); top = a.read(child/'complete.json'); claimed = a.read(child/'base/complete.json')
    a.need(a.sha(child/'plan.json') == expected_sha == top['plan_sha256'], 'explicit actual child plan SHA')
    a.need(plan['schema'] == 'c8-native-graph-prefix-plan-v1' and plan['stop_after_step'] == 49
           and plan['completion_state'] == 'INCOMPLETE_BY_DESIGN'
           and plan['performance_qualified'] is plan['full_history_qualified'] is False, 'fixed prefix plan scope')
    a.need(top['schema'] == 'c8-native-graph-prefix-result-v1' and top['status'] == 'DIAGNOSTIC_PREFIX_COMPLETE'
           and top['completion_state'] == 'INCOMPLETE_BY_DESIGN' and top['source_input_pins_rechecked'] is True
           and top['performance_qualified'] is top['full_history_qualified'] is False, 'actual prefix terminal scope')
    a.need(all(type(top[k]) is int and top[k] > 1 for k in ('pid','ppid')), 'actual pid/ppid receipt')
    a.need(plan['sources'][str(B/'c8_prefix_replay_v1/prefix.py')] == PREFIX_SHA, 'exact frozen prefix producer')
    for path, digest in {**plan['sources'], **plan['inputs']}.items():
        a.pinned(dict(path=path, sha256=digest))
    for name in ('qualification','recording','analysis'): a.pinned(plan[name])
    a.need(plan['qualification']['sha256'] == BASE_PLAN_SHA == a.sha(child/'base/plan.json'), 'original numeric plan')
    numeric = a.read(child/'base/plan.json')
    a.same(top['base_result'], dict(returncode=0,status='PASS',sha256=a.sha(child/'base/complete.json')), 'base terminal binding')
    a.need(claimed['status'] == 'PASS' and claimed['plan_sha256'] == BASE_PLAN_SHA
           and claimed['qualifier_sha256'] == plan['sources'][str(B/'pareto_realistic/qualify_native.py')]
           and claimed['source_model_pins_rechecked'] is claimed['numeric_only_not_http'] is True
           and claimed['total_requests'] == 0 and len(claimed['cases']) == 1, 'base prefix/pin scope')
    a.same(claimed['candidate'], numeric['candidate'], 'actual graph/gates candidate settings')
    a.same(claimed['cases'][0], top['prefix'], 'top versus original case receipt')
    a.same(top['peak_mlx_bytes'], claimed['peak_bytes'], 'MLX allocator peak receipt only')
    a.same(top['elapsed_base_seconds'], claimed['elapsed_wall_s'], 'base elapsed receipt')
    fixtures, config, tokenizer, token_pins = a.tokenizer_and_manifests(numeric,http)
    capture = Path(plan['capture_directory']); spec = a.read(capture/'spec.json')
    a.same({k:v for k,v in spec['environment'].items() if k.startswith('MLX_')},
           plan['environment'], 'captured MLX environment versus replay plan')
    a.same(spec['fixtures'], fixtures, 'captured full fixture manifest')
    a.need(spec['concurrency'] == 8 and spec['wave_names'] == ['warm','screen'] and len(spec['jobs']) == 16, 'original two C8 waves')
    group = a.read(capture/'group.json'); rows = []; refs = []
    for phase in ('warm','screen'):
        path = capture/'client'/phase/'group.json'; wave = a.read(path); clean = wave['cleanup']
        a.need(len(wave['requests']) == 8 and wave['group_error'] is None and clean['complete'] is True
               and clean['errors'] == clean['pending_tasks'] == [] and clean['deadline_exceeded'] is clean['cancelled_again'] is False,
               'original complete eight-request wave')
        rows.extend(wave['requests']); refs.append(dict(phase=phase,path=str(path),sha256=a.sha(path)))
    a.same(group['requests'], rows, 'derived view preserves original raw SSE/body rows')
    a.same(group['source_groups'], refs, 'derived view raw sources')
    join = a.load(B/'pareto_realistic/http_join.py', plan['sources'][str(B/'pareto_realistic/http_join.py')])
    with (capture/'trace.jsonl').open('rb') as stream:
        recording = join.join(group,stream,spec['jobs'],{f['fixture_id']:f for f in fixtures},spec['identity'],
                              lambda ids: tokenizer.decode(ids,skip_special_tokens=False))
    a.same(a.read(plan['recording']['path']), recording, 'independent raw SSE/trace rejoin versus full unchanged recording')
    a.need(len(recording['steps']) == 1030 and len(recording['jobs']) == 16
           and all(j['cap'] == 512 for j in recording['jobs'].values()), 'original full jobs/caps/history unchanged')
    header = dict(event='route-startup',schema='E024-route-http-events-v1',
        pid=a.read(capture/'release.json')['pid'], phase='HTTP-C8', identity=spec['identity'])
    return plan, top, claimed, recording, fixtures, config, token_pins, header


def audit(child, expected_sha):
    a = support(); child = Path(child).resolve()
    result = dict(schema='E049-independent-prefix-audit-v1', status='INCOMPLETE', prefix_receipts_verified=False,
        completion_state='INCOMPLETE_BY_DESIGN', full_history_qualified=False, performance_qualified=False,
        owner_resources_restore_qualified=False, offline_tensor_bytes_rechecked=False, weight_shards_rehashed=False,
        audit_source_sha256=a.sha(__file__), child=str(child), plan_sha256=expected_sha,
        evidence_limit='Finite/bit equality is the source-pinned GPU producer per-array receipt. Raw full tensor bytes were not saved; no offline arithmetic or tensor-byte reread.',
        scope='Independently reconstructed 49-step warm history, IDs, state coverage and route prefix; not complete HTTP requests or a performance point.')
    try:
        top = a.read(child/'complete.json')
        a.need(top.get('status') == 'DIAGNOSTIC_PREFIX_COMPLETE', 'producer not successful terminal prefix')
        http, base, serving = [a.load(p,h) for p,h in ((a.HTTP_AUDIT,a.HTTP_AUDIT_SHA),(a.BASE,a.BASE_SHA),(a.SERVING,a.SERVING_SHA))]
        stable = {name:a.sha(child/name) for name in ('plan.json','complete.json','base/plan.json','base/complete.json')}
        plan,top,claimed,recording,fixtures,config,tokens,header = binding(a,child,expected_sha,http)
        route_path = Path(plan['capture_directory'])/'route-events.jsonl'; replay_path = child/'replay-route-events.jsonl'
        identities = {p:a.file_identity(p) for p in (route_path,replay_path,child/'base/steps.jsonl')}; counts = {}
        with route_path.open('rb') as captured, replay_path.open('rb') as replay:
            routes = a.DualRoutes(serving,captured,replay,header,dict(path=str(route_path),sha256=plan['inputs'][str(route_path)]))
            base.CaseAudit = a.route_case_type(base,routes,counts)
            result['ledger'] = stream_prefix(a,http,base,child/'base/steps.jsonl',recording,fixtures,config,top['prefix'])
            result['routes'] = routes_finish(a,routes,top,49)
        for path, identity in identities.items(): a.need(a.file_identity(path) == identity, 'stream changed during audit')
        for name,digest in stable.items(): a.need(a.sha(child/name) == digest, 'terminal changed during audit')
        for path,digest in {**plan['sources'],**plan['inputs'],**tokens}.items(): a.need(a.sha(path) == digest, 'input/source changed during audit')
        result.update(status='PASS',prefix_receipts_verified=True,terminal_sha256=stable,
            helper_sources={str(ROUTE):ROUTE_SHA,str(a.HTTP_AUDIT):a.HTTP_AUDIT_SHA,str(a.BASE):a.BASE_SHA,str(a.SERVING):a.SERVING_SHA},
            tokenizer_runtime_pins=tokens,raw_inputs=plan['inputs'],ahead_retirement=counts,
            peak_mlx_bytes_receipt=top['peak_mlx_bytes'],mlx_peak_is_physical_footprint=False)
    except FileNotFoundError as exc: result.update(status='INCOMPLETE',reason=str(exc))
    except Exception as exc: result.update(status='FAIL',first_failure=dict(type=type(exc).__name__,message=str(exc)))
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--child',required=True,type=Path)
    p.add_argument('--plan-sha256',required=True);p.add_argument('--out',required=True,type=Path);args=p.parse_args(argv)
    if args.out.exists() or not args.out.parent.is_dir(): raise ValueError('new output in existing directory required')
    result=audit(args.child,args.plan_sha256)
    with args.out.open('x') as stream: json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({k:result[k] for k in ('status','prefix_receipts_verified','completion_state')}))
    return {'PASS':0,'FAIL':1,'INCOMPLETE':2}[result['status']]


if __name__ == '__main__': raise SystemExit(main())
