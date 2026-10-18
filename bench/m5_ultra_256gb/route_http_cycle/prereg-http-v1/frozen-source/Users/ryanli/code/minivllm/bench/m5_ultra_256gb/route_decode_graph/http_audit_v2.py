"""Independent offline E024 route-ON recorded HTTP audit, never a device run.

Terminal capture/replay evidence is required before opening the large ledger.
The frozen numerical HTTP checker still owns all full-logit/state/ahead gates.
Two route cursors are checked against that independently reconstructed schedule;
mutual equality alone cannot qualify route calls. PASS is one captured history,
not future requests, owner/resources/restoration, or HTTP performance. Successful
GPU tensor payloads are unavailable for an independent arithmetic recomputation.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
NATIVE = HERE.parent / 'pareto_realistic'
ROOT = HERE.parents[2]
PRODUCER = HERE / 'http_qualification_v2.py'
PRODUCER_SHA = 'd3e2a4f76b7c384f6c6c986d0838b68227618502b0552e5cdf74f149029e39e7'
HTTP_AUDIT = NATIVE / 'audit_http_replay_v2.py'
HTTP_AUDIT_SHA = '7cdcf5a2f02768ef75686b0ff0b0da9d21f5f9c82ee3866e24c1f869c709f031'
BASE = NATIVE / 'audit_qualification.py'
BASE_SHA = 'b6d90413911f2b17e0951a2fcfdb5fe00666aa06d4382b603e29e5dc4659d4f1'
SERVING = HERE / 'audit_serving.py'
SERVING_SHA = '064634acae9510b5744b7c848f09ee0854ee8bf26fc480af78dbed7f159573ba'
BASE_PLAN_SHA = '21ba52e43ce54c2b463ab46bfd212c53d06d6303530c77939e3e6b801df547fb'
SERVING_RECEIPT = ROOT / 'docs/experiments/m5-ultra-256gb/E024-serving-v1/independent-numerical.json'
SERVING_RECEIPT_SHA = 'd07ad9e71573adf5737718066a089705d53a03592330e97fa508e30764958856'
SUPPORT = {PRODUCER: PRODUCER_SHA, HTTP_AUDIT: HTTP_AUDIT_SHA, BASE: BASE_SHA, SERVING: SERVING_SHA}


class Incomplete(Exception): pass

def need(ok, message):
    if not ok: raise ValueError(message)

def same(a, b, message):
    need(json.dumps(a, sort_keys=True, allow_nan=False) == json.dumps(b, sort_keys=True, allow_nan=False), message)

def parse(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            need(key not in result, 'duplicate JSON key: ' + key); result[key] = value
        return result
    def bad(value): raise ValueError('nonfinite JSON constant: ' + value)
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=bad)

def read(path):
    need(Path(path).suffix != '.jsonl', 'stream JSONL only')
    return parse(Path(path).read_bytes())

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while part := stream.read(1 << 20): digest.update(part)
    return digest.hexdigest()

def pinned(ref):
    need(type(ref) is dict and set(ref) == {'path', 'sha256'}, 'exact path/SHA reference')
    path = Path(ref['path'])
    need(path.is_absolute() and path.resolve() == path and not path.is_symlink() and path.is_file()
         and sha(path) == ref['sha256'], 'canonical pinned file: ' + str(path))
    return path

def load(path, expected):
    need(sha(path) == expected, 'frozen helper changed: ' + str(path))
    spec = importlib.util.spec_from_file_location('_route_http_audit_' + path.stem, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module

def finite(x): return type(x) in (int, float) and math.isfinite(x) and x >= 0

def file_identity(path):
    s = Path(path).stat(); return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns)


def serving_prerequisite(producer, binding):
    same(binding['audit'], dict(path=str(SERVING_RECEIPT), sha256=SERVING_RECEIPT_SHA),
         'immutable real37/42 prerequisite')
    pinned(binding['audit'])
    return producer.serving_gate(binding)


def http_intervals(group, spec):
    """Independently sweep actual half-open POST->DONE intervals, not declared C."""
    C, d = spec['concurrency'], spec['deadlines']; clean = group['cleanup']
    need(group['base_url'] == f"http://127.0.0.1:{spec['transport']['port']}"
         and group['request_timeout_s'] == d['request'] and group['group_timeout_s'] == d['group'], 'capture origin/budgets')
    same(group['connection_policy'], dict(max_connections=2*C, max_keepalive_connections=0, retries=0, follow_redirects=False), 'capture connection policy')
    need(clean['adapter'] == 'native-close-cancellation-v1'
         and clean['frozen_client_sha256'] == 'ef64a114f4f7d90ec5044d6bdcf058ca05f98dea445d6639db706a4fc75217dc'
         and clean['budget_s'] == d['close'] and clean['client_close_started'] is True
         and clean['complete'] is True and clean['errors'] == clean['pending_tasks'] == []
         and clean['deadline_exceeded'] is clean['cancelled_again'] is False, 'complete bounded client cleanup')
    barrier, end = group['barrier_release_perf_counter_s'], group['group_deadline_perf_counter_s']
    need(all(finite(x) for x in (barrier, end, clean['start_perf_counter_s'], clean['end_perf_counter_s']))
         and end == barrier+d['group'] and clean['start_perf_counter_s'] <= clean['end_perf_counter_s'], 'group/close clocks')
    edges = []
    for row in group['requests']:
        start, done = row['start_perf_counter_s'], row['done_perf_counter_s']
        need(all(finite(x) for x in (start, done, row['absolute_deadline_perf_counter_s'], row['terminal_perf_counter_s'], row['e2e_s']))
             and barrier <= start < done <= end and row['absolute_deadline_perf_counter_s'] == start+d['request']
             and done <= row['absolute_deadline_perf_counter_s'] and row['terminal_perf_counter_s'] == done
             and done <= clean['start_perf_counter_s'] and row['e2e_s'] == done-start, 'POST/DONE/deadline/close clocks')
        edges.extend(((start, 1), (done, -1)))
    active = peak = 0
    for _, delta in sorted(edges):
        active += delta; need(active >= 0, 'negative HTTP outstanding'); peak = max(peak, active)
    need(active == 0 and peak == C, 'actual HTTP concurrency differs from declared C')
    need(len(group['requests']) == (8 if C == 1 else 16), 'complete8/16 HTTP queue')
    return dict(max_http_outstanding=peak, requests=len(group['requests']), fully_drained=True,
                scope='Actual POST->DONE caller concurrency; not GPU batch size or a performance measurement.')


def route_cursor_type(serving):
    class Cursor(serving.RouteEvents):
        def take(self, event):
            raw = self.stream.readline()
            need(raw and raw.endswith(b'\n'), 'missing/truncated route event: ' + event)
            self.digest.update(raw); self.bytes += len(raw); self.lines += 1
            row = parse(raw); need(row.get('event') == event, 'route event order')
            return row
    return Cursor


class DualRoutes:
    def __init__(self, serving, captured, replay, capture_header, capture_ref):
        cls = route_cursor_type(serving); self.cursors = [cls(captured), cls(replay)]
        same(self.cursors[0].take('route-startup'), capture_header, 'capture route startup identity')
        row = self.cursors[1].take('replay-route-startup')
        need(type(row.get('pid')) is int and row['pid'] > 1 and row['pid'] != capture_header['pid'], 'distinct actual replay PID')
        self.replay_pid = row['pid']
        same(row, dict(event='replay-route-startup', schema='E024-route-http-replay-events-v1', pid=self.replay_pid,
            phase=capture_header['phase'], captured_route_events=capture_ref), 'replay header/source binding')
        self.calls = 0; self.domain = set()

    def route(self, phase, step, label, B, used, L):
        for cursor in self.cursors: cursor.route(phase, step, label, B, used, L)
        self.calls += 1; self.domain.add((B, used, L))

    def finish(self):
        rows = [cursor.finish() for cursor in self.cursors]
        for key in ('route_hits', 'long_fallback_calls', 'parameter_direct_calls', 'parameter_array_comparisons'):
            same(rows[0][key], rows[1][key], 'independently derived sidecar accounting')
        need(self.calls > 0 and sum(sum(x.values()) for x in rows[0]['route_hits'].values()) > 0,
             'no actual graph invocation in this captured C history')
        return dict(captured=rows[0], replay=rows[1], replay_pid=self.replay_pid, calls=self.calls,
            actual_domains=[dict(B=b, used=u, physical_capacity=L) for b, u, L in sorted(self.domain)],
            call_binding='Validated semantic call order per step; consumed ahead is not rerun. No explicit token key in sidecar.')


def route_case_type(base, routes, counts):
    original = base.CaseAudit
    class RouteCase(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs); self.old_ids = []; self.capacity = 0

        def check_arrays(self, row, calls, pre_staging, has_future, B):
            super().check_arrays(row, calls, pre_staging, has_future, B)
            decode = row['schedule']['decode']; requests = {x['rid']: x for x in row['schedule']['requests']}
            new = [r for r in decode if r not in self.old_ids]; keep = [r for r in decode if r in self.old_ids]
            L = self.capacity
            if decode and decode != self.old_ids:
                required = max([len(requests[r]['prompt_ids'])+requests[r]['sampling']['max_tokens'] for r in new]
                               + [self.capacity if keep else 0]); L = ((required+255)//256)*256
            for item in row['arrays']:
                if item['name'].rsplit('/', 1)[-1] not in ('K', 'V'): continue
                label = item['check']; wanted = None
                if label == 'before-step-state' and item['name'].startswith(('root/committed/', 'root/ahead/')): wanted = self.capacity
                elif label.startswith("call-state ('decode',") or label.endswith('/committed-not-rewritten-by-ahead'): wanted = L
                elif label in ('before-retirement-state', 'after-retirement-state') and item['name'].startswith(('root/committed/', 'root/ahead/')): wanted = L
                if wanted is not None: need(item['shape'][2] == wanted, 'reconstructed padded KV capacity')
            consumed = row['ahead']['candidate']['consumed_prior_future']
            for key in calls:
                if key[0] != 'decode' or key == consumed: continue
                used = max(key[2])+1
                if 257 <= used <= 1536: routes.route(self.name, self.step, None, len(key[1]), used, L)
            bucket = counts.setdefault(self.name, dict(consumed=0, discarded_prior=0, partial_retirement=0, refills=0))
            bucket['consumed'] += int(consumed is not None)
            bucket['discarded_prior'] += int(row['ahead']['candidate']['discarded_prior_future'] is not None)
            after = row['after']['engine']['batch_rids']
            bucket['partial_retirement'] += int(bool(after) and bool(set(decode)-set(after)))
            bucket['refills'] += int(bool(self.old_ids) and all(r in decode for r in self.old_ids) and bool(new))
            self.old_ids = list(after); self.capacity = L if after else 0
    return RouteCase


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
        need(capture.ready_identity(read(directory/name), spec['identity']), 'actual route server readiness identity')
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


def tokenizer_and_manifests(numeric, http):
    manifests = {name: read(pinned(numeric[name])) for name in ('fixtures', 'model_manifest', 'source_manifest')}
    model = manifests['model_manifest']; root = Path(numeric['package_root'])/'tutorial_code/miniengine'
    need(model['schema'] == 'E023-model-v1' and model['model'] == numeric['model']
         and model['revision'] == Path(numeric['model']).name, 'model identity')
    core = manifests['source_manifest']['files']
    need(set(core) == {str(p.relative_to(root)) for p in root.rglob('*.py')}, 'exact frozen core source closure')
    for rel, value in core.items():
        need(not Path(rel).is_absolute() and '..' not in Path(rel).parts and sha(root/rel) == value, 'frozen core SHA')
    config_path = Path(numeric['model'])/'config.json'; need(sha(config_path) == model['files']['config.json'], 'actual config SHA')
    config = read(config_path); config = config.get('text_config', config)
    need(config['layer_types'] == ['full_attention' if i%4 == 3 else 'linear_attention' for i in range(64)]
         and config['vocab_size'] == 248320 and config['hidden_size'] == 5120, 'full64 model layout')
    fs = manifests['fixtures']; need(numeric['fixtures']['sha256'] == http.load('audit_qualification.py', BASE_SHA).FIXTURE_SHA
         and fs['model'] == numeric['model'] and len(fs['fixtures']) == 8, 'fixed eight D fixtures')
    token_pins = {p: h for p, h in numeric['execution']['pins'].items() if 'tokenizers' in Path(p).parts
                  or Path(p).parent.name == 'tokenizers-0.23.3.dist-info'}
    need(token_pins and any(p.endswith('.so') for p in token_pins), 'actual CPU tokenizer runtime pins')
    for p, h in token_pins.items(): need(sha(p) == h, 'CPU tokenizer runtime SHA')
    import tokenizers
    need(str(Path(tokenizers.__file__).resolve()) in token_pins, 'actual loaded CPU tokenizer path')
    path = Path(numeric['model'])/'tokenizer.json'; need(sha(path) == model['files']['tokenizer.json'], 'tokenizer model SHA')
    tokenizer = tokenizers.Tokenizer.from_file(str(path))
    for f in fs['fixtures']:
        same(tokenizer.encode(f['prompt_text'], add_special_tokens=False).ids, f['prompt_ids'], 'full D prompt encoding')
    return fs['fixtures'], config, tokenizer, token_pins


def binding(child, expected_sha, top, claimed, producer, http):
    need(sha(child/'plan.json') == expected_sha == top['plan_sha256'], 'explicit actual replay plan SHA')
    plan, numeric, _, _ = producer.validate(child/'plan.json', expected_sha)
    need(plan['qualifier_sha256'] == PRODUCER_SHA and plan['qualification']['sha256'] == BASE_PLAN_SHA
         and sha(child/'base/plan.json') == BASE_PLAN_SHA, 'actual qualifier/original numeric plan')
    serving_prerequisite(producer, plan['serving'])
    C = plan['concurrency']; count = 8 if C == 1 else 16
    need(top['schema'] == 'E024-recorded-route-http-result-v1' and top['prepare_only'] is False
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
    result = dict(schema='E024-recorded-route-http-independent-audit-v2', status='INCOMPLETE', qualified=False,
        child_directory=str(child), ledger_opened=False, http_history_qualified=False, performance_qualified=False,
        owner_resources_restore_qualified=False, offline_tensor_bytes_rechecked=False, weight_shards_rehashed=False,
        audit_source_sha256=sha(__file__), support_sources={str(p):h for p,h in SUPPORT.items()},
        scope='This recorded HTTP arrival history only; original64-layer device bit/finite receipts plus independent schedule/route/wire reconstruction. No offline GPU arithmetic, future requests or performance grant.')
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
        producer, http, base, serving = [load(p,h) for p,h in SUPPORT.items()]
        stable = {name:sha(child/name) for name in ('plan.json','complete.json','base/plan.json','base/complete.json','recording.json')}
        plan, recording, fixtures, config, tokens, timing, header = binding(child, expected_plan_sha256, top, claimed, producer, http)
        capture_path = Path(plan['inputs']['route_events']['path']); replay_path = child/'replay-route-events.jsonl'
        identities = {p:file_identity(p) for p in (capture_path,replay_path,child/'base/steps.jsonl')}; counts = {}
        with capture_path.open('rb') as captured, replay_path.open('rb') as replay:
            routes = DualRoutes(serving, captured, replay, header, plan['inputs']['route_events'])
            base.CaseAudit = route_case_type(base, routes, counts)  # Private loaded module only; original numerical checker stays intact.
            result['ledger_opened'] = True
            ledger = http.stream_ledger(child/'base/steps.jsonl',recording,fixtures,config,claimed['cases'][0],base)
            routed = routes.finish()
        C = plan['concurrency']; phase = f'HTTP-C{C}'; capture_stats = routed['captured']
        bounds = capture_stats['route_hits'].get(phase, {'1023':0,'1536':0}); hits = sum(bounds.values())
        same(top['coverage'], dict(installs=1,restores=2,route_hits=capture_stats['route_hits'],
             long_fallback_calls=capture_stats['long_fallback_calls'],identity_refill_enabled=False), 'independent actual route coverage')
        same(top['route_comparison'], dict(calls=routed['calls'],actual_compiled_invocations=hits,
             fallbacks=capture_stats['long_fallback_calls'],bounds=bounds,captured_sha256=capture_stats['sha256'],
             bytes=capture_stats['bytes'],lines=capture_stats['lines'],
             scope='Ordered actual calls per recorded step, including actual ahead behavior; SUBMITTED alone is not numeric PASS.'), 'producer route comparison versus independently derived calls')
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
