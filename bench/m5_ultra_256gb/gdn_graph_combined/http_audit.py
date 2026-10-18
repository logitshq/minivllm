"""Independent combined HTTP audit: original full-state math plus actual traces.

Two separately streamed sidecars must each match independently reconstructed
semantic calls, not just each other. Python tracing counts distinguish compiled
reuse; no inference of48 hook calls per GPU submission. PASS applies only to the
captured8/16 arrival history. Resource/recovery/performance remain separate.
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
ROUTE=HERE.parent/'route_decode_graph'
HELPER=ROUTE/'http_audit_v2.py'
HELPER_SHA='88f32e3b3f44206b48a1f581cbfe0bde0788060cbd2a0c5030a3ec472a0139fe'
PRODUCER=HERE/'http_qualification.py'
PRODUCER_SHA='33b86cb57bf3c4616762c1bd1cfea0abb8f7f063325e9dfe815642e8f591bc7d'
HTTP_AUDIT=NATIVE/'audit_http_replay_v2.py'
HTTP_AUDIT_SHA='7cdcf5a2f02768ef75686b0ff0b0da9d21f5f9c82ee3866e24c1f869c709f031'
BASE=NATIVE/'audit_qualification.py'
BASE_SHA='b6d90413911f2b17e0951a2fcfdb5fe00666aa06d4382b603e29e5dc4659d4f1'
SERVING=ROUTE/'audit_serving.py'
SERVING_SHA='064634acae9510b5744b7c848f09ee0854ee8bf26fc480af78dbed7f159573ba'
BASE_PLAN_SHA='21ba52e43ce54c2b463ab46bfd212c53d06d6303530c77939e3e6b801df547fb'
SUPPORT={PRODUCER:PRODUCER_SHA,HTTP_AUDIT:HTTP_AUDIT_SHA,BASE:BASE_SHA,SERVING:SERVING_SHA,
    HELPER:HELPER_SHA,HERE/'audit.py':'7e7718ddf1e1b6d744ec374e19b5513248f0374306503aeb8908391edb403270'}


def load(path,expected):
    if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=expected:raise ValueError('fixed source changed: '+str(path))
    spec=importlib.util.spec_from_file_location('_combined_http_audit_'+Path(path).stem,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


helper=load(HELPER,HELPER_SHA)
need,same,parse,read,sha,pinned,finite,file_identity=(helper.need,helper.same,helper.parse,helper.read,
    helper.sha,helper.pinned,helper.finite,helper.file_identity)
Incomplete=helper.Incomplete
http_intervals,tokenizer_and_manifests=helper.http_intervals,helper.tokenizer_and_manifests


def cursor_type(serving):
    original=helper.route_cursor_type(serving)
    class Cursor(original):
        def __init__(self,stream):
            super().__init__(stream);self.seen=set()
            self.counters={0:[dict(candidate=0,fallback=0) for _ in range(48)]}
            self.combined=dict(installs=1,restores=1,layer_installs=48,layer_restores=48,
                **{'compiled-tracing':0,'compiled-reuse':0,'eager-fallback':0},traced_batches=[])
        def external_decode(self,phase,B):
            need(phase in ('HTTP-C1','HTTP-C8'),'selected actual HTTP phase')
            for row in self.counters[0]:row['candidate' if B==1 else 'fallback']+=1
        def route(self, phase, step, label, B, used, L):
            need(phase in ('HTTP-C1','HTTP-C8'), 'actual combined HTTP phase')
            slot = 0
            graph = 1 <= B <= 8 and L >= 1536
            bound = 1023 if used < 1024 else 1536; key = (slot, bound, B)
            row = self.take('combined-graph-call')
            need(set(row) == {'event', 'status', 'installation', 'phase', 'step', 'direct_label',
                'B', 'used', 'before', 'after', 'actual_invocations', 'kind', 'bound',
                'per_layer_candidate_delta', 'per_layer_fallback_delta'}, 'exact combined event fields')
            same(row['before'], self.counters[slot], 'continuous independently tracked48 counters')
            after = row['after']; need(type(after) is list and len(after) == 48, 'all48 after counters')
            deltas = []
            for left, right in zip(self.counters[slot], after):
                need(type(right) is dict and set(right) == {'candidate', 'fallback'}
                     and all(type(x) is int and x >= 0 for x in right.values()), 'strict integer after counters')
                deltas.append((right['candidate'] - left['candidate'], right['fallback'] - left['fallback']))
            need(all(x == deltas[0] for x in deltas) and min(deltas[0]) >= 0, 'uniform nonnegative layer trace')
            candidate, fallback = deltas[0]
            need(fallback == 0 if B == 1 else candidate == 0, 'correct original B1/multirow branch')
            if graph:
                if candidate + fallback:
                    kind = 'compiled-tracing'; self.seen.add(key)
                    self.combined['traced_batches'].append(dict(installation=slot, bound=bound, B=B,
                        python_passes=candidate + fallback))
                else:
                    need(key in self.seen, 'reuse needs prior actual same-instance/bound/B trace')
                    kind = 'compiled-reuse'
            else:
                need(candidate + fallback == 1, 'eager GDN fallback executes once per layer')
                kind = 'eager-fallback'
            expected = dict(event='combined-graph-call', status='SUBMITTED', installation=slot,
                phase=phase, step=step, direct_label=label, B=B, used=used, before=self.counters[slot],
                after=after, actual_invocations=int(graph), kind=kind, bound=bound,
                per_layer_candidate_delta=candidate, per_layer_fallback_delta=fallback)
            same(row, expected, 'combined event independent call/trace/branch binding')
            self.counters[slot] = [dict(x) for x in after]; self.combined[kind] += 1
            # Original callable count/domain/capacity and exact route row gate.
            super().route(phase, step, label, B, used, L)
    return Cursor


class DualRoutes:
    def __init__(self,serving,captured,replay,capture_header,capture_ref):
        cls=cursor_type(serving);self.cursors=[cls(captured),cls(replay)]
        same(self.cursors[0].take('route-startup'),capture_header,'capture startup')
        row=self.cursors[1].take('replay-route-startup')
        need(type(row.get('pid')) is int and row['pid']>1 and row['pid']!=capture_header['pid'],'distinct replay PID')
        self.replay_pid=row['pid']
        same(row,dict(event='replay-route-startup',schema='GDN-graph-combined-http-replay-events-v1',pid=self.replay_pid,
            phase=capture_header['phase'],captured_route_events=capture_ref),'replay startup binding')
        self.calls=0;self.domain=set()
    def route(self,phase,step,label,B,used,L):
        for cursor in self.cursors:cursor.route(phase,step,label,B,used,L)
        self.calls+=1;self.domain.add((B,used,L))
    def external_decode(self,phase,B):
        for cursor in self.cursors:cursor.external_decode(phase,B)
    def finish(self):
        rows=[cursor.finish() for cursor in self.cursors]
        for key in ('route_hits','long_fallback_calls','parameter_direct_calls','parameter_array_comparisons'):
            same(rows[0][key],rows[1][key],'independently derived route accounting')
        same(self.cursors[0].combined,self.cursors[1].combined,'independently checked combined tracing')
        need(self.calls>0 and sum(sum(x.values()) for x in rows[0]['route_hits'].values())>0,'actual compiled call required')
        need(self.cursors[0].combined['compiled-tracing']>0,'actual compiled Python trace required')
        return dict(captured=rows[0],replay=rows[1],replay_pid=self.replay_pid,calls=self.calls,
            combined=self.cursors[0].combined,
            actual_domains=[dict(B=b,used=u,physical_capacity=L) for b,u,L in sorted(self.domain)],
            call_binding='Independent semantic calls, consumed ahead skipped; no one-call-per-step or48-Python-calls-per-GPU assumption')


def route_case_type(base,routes,counts):
    original=helper.route_case_type(base,routes,counts)
    class Case(original):
        def check_arrays(self,row,calls,pre_staging,has_future,B):
            super().check_arrays(row,calls,pre_staging,has_future,B)
            consumed=row['ahead']['candidate']['consumed_prior_future']
            for key in calls:
                if key[0]=='decode' and key!=consumed and max(key[2])+1>1536:
                    routes.external_decode(self.name,len(key[1]))
    return Case


def capture_binding(producer, plan):
    spec, header = producer.capture_gate(plan)  # Real factory + fixed original receipt schema.
    directory = Path(plan['capture']['spec']['path']).parent
    complete = read(directory/'complete.json'); release = read(directory/'release.json'); pid = header['pid']
    # The actual frozen capture inventories top-level files, not a recursive
    # archive. Its group.json holds every original raw SSE chunk used below.
    expected_files = {p.name for p in directory.iterdir() if p.is_file() and p.name != 'complete.json'}
    need(set(complete['artifacts']) == expected_files, 'complete closed top-level capture inventory')
    for rel, row in complete['artifacts'].items():
        path = directory/rel
        need(not Path(rel).is_absolute() and '..' not in Path(rel).parts and not path.is_symlink()
             and path.resolve().is_relative_to(directory.resolve()), 'capture artifact stays inside directory')
        same(row, dict(sha256=sha(path), bytes=path.stat().st_size), 'capture artifact bytes/SHA: ' + rel)
    same(read(directory/'launch.json'), dict(command=spec['command']+['--qualification-trace',str(directory/'trace.jsonl'),
         '--route-events',str(directory/'route-events.jsonl')], environment=spec['environment']), 'actual capture CLI/environment')
    port = spec['transport']['port']
    listener_command = ['/usr/sbin/lsof','-nP','-a']
    port_arguments = [f'-iTCP:{port}','-sTCP:LISTEN','-Fpn']
    commands = {
        'child-identity.json': ['/bin/ps','-p',str(pid),'-o','pid=,ppid=,lstart=,comm='],
        'child-absence.json': ['/bin/ps','-p',str(pid),'-o','pid='],
        'listener.json': listener_command+['-p',str(pid)]+port_arguments,
        'port-before.json': listener_command+port_arguments,
        'port-after.json': listener_command+port_arguments,
    }
    for name, command in commands.items():
        same(read(directory/name)['command'], command, 'actual PID/port probe command: '+name)
    identity = read(directory/'child-identity.json'); fields = identity['stdout'].split()
    need(type(identity['returncode']) is int and identity['returncode'] == 0 and len(fields) >= 2 and fields[0] == str(pid)
         and fields[1].isdigit() and int(fields[1]) > 1, 'actual capture process identity receipt')
    capture = producer.load(HERE/'http_capture.py', '_e024_independent_capture_receipts')
    need(capture.no_process(read(directory/'port-before.json')), 'registered capture port initially absent')
    for name in ('ready.json', 'ready-after-client.json'):
        need(capture.donor.ready_identity(read(directory/name), spec['identity']), 'actual route server readiness identity')
    need(capture.owned_listener(read(directory/'listener.json'), pid, spec['transport']['port']), 'actual owned listener')
    events = []
    with (directory/'events.jsonl').open('rb') as stream:
        for raw in stream:
            need(raw.endswith(b'\n'), 'truncated capture event'); events.append(parse(raw))
    ownterm = any(r.get('label') == 'server-term' and r.get('pid') == pid and finite(r.get('time'))
                  and r['time'] <= release['time'] for r in events)
    need(release['returncode'] == 0 or (release['returncode'] == -15 and ownterm), 'normal or own explained TERM only')
    need(not any(r.get('label') == 'server-kill' for r in events), 'KILL cannot qualify captured HTTP')
    return spec, header


def binding(child, expected_sha, top, claimed, producer, http):
    need(sha(child/'plan.json') == expected_sha == top['plan_sha256'], 'explicit actual replay plan SHA')
    plan, numeric, _, _ = producer.validate(child/'plan.json', expected_sha)
    need(plan['qualifier_sha256'] == PRODUCER_SHA and plan['qualification']['sha256'] == BASE_PLAN_SHA
         and sha(child/'base/plan.json') == BASE_PLAN_SHA, 'actual qualifier/original numeric plan')
    producer.serving_gate(plan['serving'])
    C = plan['concurrency']; count = 8 if C == 1 else 16
    need(top['schema'] == 'GDN-graph-combined-recorded-http-result-v1' and top['prepare_only'] is False
         and top['replay_numerical_checks_passed'] is top['input_source_pins_rechecked'] is top['independent_audit_required'] is True
         and top['numerical_qualified'] is top['http_qualified'] is top['performance_qualified'] is top['owner_resources_restore_qualified'] is False
         and top['concurrency'] == C and type(top['total_requests']) is int and top['total_requests'] == count, 'new producer terminal scope')
    for name in ('sources', 'inputs', 'capture', 'serving'): same(top[name], plan[name], 'terminal binding: '+name)
    same(top['base_result'], dict(path='base/complete.json',sha256=sha(child/'base/complete.json'),returncode=0,status='PASS'), 'original numerical terminal bytes')
    need(claimed['plan_sha256'] == BASE_PLAN_SHA and claimed['qualifier_sha256'] == producer.FIXED[producer.BASE]
         and claimed['numeric_only_not_http'] is claimed['source_model_pins_rechecked'] is True
         and type(claimed['total_requests']) is int and claimed['total_requests'] == count
         and len(claimed['cases']) == 1 and claimed['cases'][0]['name'] == f'HTTP-C{C}', 'one-C base completed domain')
    same(claimed['candidate'], numeric['candidate'], 'actual candidate settings')
    spec, header = capture_binding(producer, plan)
    fixtures, config, tokenizer, token_pins = tokenizer_and_manifests(numeric, http)
    jobs = producer.load(NATIVE/'http_qualification.py', '_route_audit_jobs').jobs_for(fixtures, C)
    same(spec['jobs'], jobs, 'exact HTTP8/16 original body jobs')
    group = read(pinned(plan['inputs']['group'])); timing = http_intervals(group, spec)
    join = producer.load(NATIVE/'http_join.py', '_route_audit_join')
    with pinned(plan['inputs']['trace']).open('rb') as trace:
        recording = join.join(group, trace, jobs, {f['fixture_id']:f for f in fixtures}, spec['identity'],
                              lambda ids:tokenizer.decode(ids, skip_special_tokens=False))
    need(sha(child/'recording.json') == top['recording_sha256'], 'recording bytes SHA')
    same(read(child/'recording.json'), recording, 'independently joined raw SSE/trace versus recording')
    same(top['replay_route_events'], dict(path='replay-route-events.jsonl', sha256=sha(child/'replay-route-events.jsonl')), 'actual emitted route sidecar')
    return plan, recording, fixtures, config, token_pins, timing, header


def audit(child_dir, expected_plan_sha256):
    child = Path(child_dir).resolve()
    result = dict(schema='GDN-graph-combined-recorded-http-independent-v1', status='INCOMPLETE', qualified=False,
        child_directory=str(child), ledger_opened=False, http_history_qualified=False, performance_qualified=False,
        owner_resources_restore_qualified=False, offline_tensor_bytes_rechecked=False, weight_shards_rehashed=False,
        audit_source_sha256=sha(__file__), support_sources={str(p):h for p,h in SUPPORT.items()},
        scope='This recorded HTTP arrival history only; original64-layer device bit/finite receipts plus independent schedule/combined tracing/reuse/route/wire reconstruction. No offline GPU arithmetic, future requests or performance grant.')
    try:
        top = read(child/'complete.json')
        if top.get('status') in ('INCOMPLETE', 'PREPARED'): raise Incomplete('replay not terminal; ledger unopened')
        claimed = read(child/'base/complete.json') if (child/'base/complete.json').is_file() else {}
        if top.get('status') == 'PASS' and (not claimed or claimed.get('status') == 'INCOMPLETE'): raise Incomplete('base not terminal; ledger unopened')
        if top.get('status') != 'PASS' or claimed.get('status') != 'PASS':
            result.update(status='FAIL', producer_failure=dict(top_status=top.get('status'), base_status=claimed.get('status'),
                top_error=top.get('error'), base_error=claimed.get('error'))); return result
        for name in ('plan.json','base/plan.json','recording.json','base/steps.jsonl','replay-route-events.jsonl'):
            if not (child/name).is_file(): raise Incomplete('missing '+name)
        for p,h in SUPPORT.items():need(sha(p)==h,'fixed audited source')
        producer,http,base,serving=[load(p,h) for p,h in ((PRODUCER,PRODUCER_SHA),(HTTP_AUDIT,HTTP_AUDIT_SHA),(BASE,BASE_SHA),(SERVING,SERVING_SHA))]
        stable = {name:sha(child/name) for name in ('plan.json','complete.json','base/plan.json','base/complete.json','recording.json')}
        plan, recording, fixtures, config, tokens, timing, header = binding(child, expected_plan_sha256, top, claimed, producer, http)
        capture_path = Path(plan['inputs']['route_events']['path']); replay_path = child/'replay-route-events.jsonl'
        identities = {p:file_identity(p) for p in (capture_path,replay_path,child/'base/steps.jsonl')}; counts = {}
        with capture_path.open('rb') as captured, replay_path.open('rb') as replay:
            routes = DualRoutes(serving, captured, replay, header, plan['inputs']['route_events'])
            facade=SimpleNamespace(**{**vars(base),'CaseAudit':route_case_type(base,routes,counts)})
            result['ledger_opened'] = True
            ledger = http.stream_ledger(child/'base/steps.jsonl',recording,fixtures,config,claimed['cases'][0],facade)
            routed = routes.finish()
        C = plan['concurrency']; phase = f'HTTP-C{C}'; capture_stats = routed['captured']
        bounds = capture_stats['route_hits'].get(phase, {'1023':0,'1536':0}); hits = sum(bounds.values())
        same(top['coverage'], dict(installs=1,restores=2,route_hits=capture_stats['route_hits'],
             long_fallback_calls=capture_stats['long_fallback_calls'],identity_refill_enabled=False,combined=routed['combined']), 'independent actual route coverage')
        same(top['route_comparison'], dict(calls=routed['calls'],actual_compiled_invocations=hits,
             fallbacks=capture_stats['long_fallback_calls'],bounds=bounds,captured_sha256=capture_stats['sha256'],
             bytes=capture_stats['bytes'],lines=capture_stats['lines'],
             scope='Ordered actual calls per recorded step, including actual ahead behavior; SUBMITTED alone is not numeric PASS.',
             combined_kinds={key:routed['combined'][key] for key in ('compiled-tracing','compiled-reuse','eager-fallback')},
             tracing_scope='Python tracing/fallback counts; compiled reuse does not execute48 Python hooks'), 'producer route comparison versus independently derived calls')
        need(capture_stats['sha256'] == plan['inputs']['route_events']['sha256']
             and routed['replay']['sha256'] == top['replay_route_events']['sha256'], 'streamed actual route bytes')
        for p, ident in identities.items(): need(file_identity(p) == ident, 'closed stream changed during audit')
        for name, value in stable.items(): need(sha(child/name) == value, 'terminal evidence changed during audit')
        for ref in (*plan['inputs'].values(), *plan['capture'].values(), *plan['serving'].values()): pinned(ref)
        for p,h in {**plan['sources'],**tokens,**{str(p):h for p,h in SUPPORT.items()}}.items(): need(sha(p) == h, 'source changed during audit')
        need(ledger['requests'] == (8 if C == 1 else 16), 'complete fixed HTTP history')
        result.update(status='PASS',qualified=True,http_history_qualified=True,concurrency=C,
            plan_sha256=expected_plan_sha256,base_plan_sha256=BASE_PLAN_SHA,terminal_sha256=stable,
            raw_inputs=plan['inputs'],capture=plan['capture'],serving_prerequisite=plan['serving'],sources=plan['sources'],
            tokenizer_runtime_pins=tokens,ledger=ledger,routes=routed,http=timing,ahead_retirement=counts,
            peak_bytes_diagnostic=claimed['peak_bytes'],original_37_request_gate_replaced=False,
            unobserved='No new changed-parameter diagnostic or cancellation/key-invalidation coverage is inferred; the separately pinned37/42 serving prerequisite is not rerun here.')
    except (Incomplete,FileNotFoundError) as exc: result.update(status='INCOMPLETE',reason=str(exc))
    except Exception as exc: result.update(status='FAIL',first_failure=dict(type=type(exc).__name__,message=str(exc)))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--child',required=True,type=Path); parser.add_argument('--plan-sha256',required=True)
    parser.add_argument('--out',required=True,type=Path); args = parser.parse_args(argv)
    need(args.out.parent.is_dir() and not args.out.exists(), 'new audit output required')
    result = audit(args.child,args.plan_sha256)
    with args.out.open('x') as stream: json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    return {'PASS':0,'FAIL':1,'INCOMPLETE':2}[result['status']]


if __name__ == '__main__': raise SystemExit(main())
