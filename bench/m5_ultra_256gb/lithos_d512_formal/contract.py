"""Lithos D512 formal count contract, bound to the actual closed v2 qualification.

CPU metadata only. No tokenizer, server or model import. check_group reparses
retained NDJSON; it cannot establish full-state equality, output quality, cache
hits, resource isolation or restoration. Those gates remain explicit outside.
"""
import base64
import copy
from functools import lru_cache
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
BASE_SHA = '21ba52e43ce54c2b463ab46bfd212c53d06d6303530c77939e3e6b801df547fb'
OWNER_PLAN_SHA = 'e72792b61039414d0c16d9f5f3216619aed62d2d3c23536fe491d409383ab354'
RECEIPTS = {
    'audit': 'df64cd94a1ec47dc4f5dd8f749e61066eba17d126abc0860ad929b1703f16189',
    'resource_audit': 'f171bcbb8f0f709ececc7b9b35a1f25f5e0e4b3194ed5d542799a8fd5e38930e',
    'restoration_audit': '269a9a85b50674077fee8dc464a0edc4a0531fb13e5386d95dbc8cc57ba66e43',
}
FIXED = {
    '../lithos_d512_retry/join.py': '2d4dfa12c9672a7a9deff7a962ea18dd671f53c4e7010eed3589fa4a0826cd0d',
    '../lithos_d512_retry/serve_probe.py': 'c6f4915a83d15cf5c527d71e73d4e54683c0de3881de4c2352b9e0d0abc14703',
    '../pareto_realistic/lithos_client.py': '13a629f096434bc6c422803bf7bfd8992e24396cb122880b493e893cc5ecd2e9',
    '../pareto_realistic/client.py': 'ef64a114f4f7d90ec5044d6bdcf058ca05f98dea445d6639db706a4fc75217dc',
}
AUDIT_SOURCES = {
    'audit': ('audit_qualification.py', 'a932583f5a5234b0aff434d9b967a004fcdea05ff9b1c03a1d5d34fe86f6df08'),
    'resource_audit': ('audit_resources.py', 'e6468ac9bcc90fc2d4f3fcdfc55c5292e042a479905e38de9a85b5d5a0ba093f'),
    'restoration_audit': ('audit_restore.py', '002cb1839b00a0d617404801dcb08cca873deb048da6c39bc726c67f1ef8aa33'),
}


def need(ok, message):
    if not ok: raise ValueError(message)
def canonical(value): return json.dumps(value, sort_keys=True, allow_nan=False, separators=(',', ':')).encode()
def digest(value): return hashlib.sha256(canonical(value)).hexdigest()
def same(a, b, label): need(canonical(a) == canonical(b), label)
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def finite(x): return type(x) in (int, float) and math.isfinite(x) and x >= 0


@lru_cache(None)
def load(name):
    path = (HERE / name).resolve(); need(sha(path) == FIXED[name], 'frozen count dependency changed')
    spec = importlib.util.spec_from_file_location('_lithos_formal_' + path.stem, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def read(path): return load('../lithos_d512_retry/join.py').parse(Path(path).read_bytes())
def pinned(ref):
    need(type(ref) is dict and set(ref) == {'path', 'sha256'}, 'exact artifact reference')
    p = Path(ref['path']); need(p.is_absolute() and p.is_file() and not p.is_symlink() and sha(p) == ref['sha256'], 'artifact SHA/path')
    return p


def bind(base_plan, binding, concurrency, model):
    """No synthetic qualification: consume exact retained independent audits."""
    need(set(binding) == {'owner_directory', 'owner_plan', *RECEIPTS}, 'actual qualification binding')
    need(base_plan['sha256'] == BASE_SHA and binding['owner_plan']['sha256'] == OWNER_PLAN_SHA, 'fixed base/qualified owner plan')
    base = read(pinned(base_plan)); plan = read(pinned(binding['owner_plan']))
    root = Path(binding['owner_directory']); need(root.is_absolute() and root.is_dir(), 'retained qualification owner')
    need(sha(root/'owner-plan.json') == OWNER_PLAN_SHA, 'actual owner plan copy')
    same(plan['qualification'], base_plan, 'same qualified workload')
    audits = {}
    for name, expected in RECEIPTS.items():
        need(binding[name]['sha256'] == expected, 'unrecognized independent qualification receipt')
        audits[name] = read(pinned(binding[name]))
        filename, source_sha = AUDIT_SOURCES[name]
        need(sha(HERE.parent/'lithos_d512_retry'/filename) == source_sha == audits[name]['audit_source_sha256'], 'independent auditor source')
        need(audits[name]['status'] == 'PASS' and audits[name]['owner_plan_sha256'] == OWNER_PLAN_SHA, 'closed audit/plan')
    audit, resource, restore = (audits[k] for k in RECEIPTS)
    need(audit['qualified'] is True and audit['route_count_text_qualified'] is True
         and resource['independent_resource_verified'] is True and restore['qualified'] is True, 'three independent gates')
    need(audit['owner_directory'] == resource['owner_directory'] == str(root)
         and audit['owner_complete_sha256'] == resource['owner_complete_sha256'] == sha(root/'complete.json'), 'audited actual owner')
    need(audit['full_state_qualified'] is False and audit['performance_qualified'] is False
         and restore['numerical_qualified'] is False, 'bounded qualification scope')
    need(restore['service_restore']['status'] == 'PERSISTED_VISIBLE_RECEIPTS_CONSISTENT'
         and restore['owner_reap']['status'] == 'ALL_OBSERVED_CHILDREN_REAPED', 'closed restoration/reap')
    need([p['concurrency'] for p in audit['phases']] == [1, 8] and audit['requests'] == 24, 'both audited histories')
    phase = audit['phases'][0 if concurrency == 1 else 1]
    folder = root/'phases'/f'lithos-c{concurrency}'/'trial'
    for key, name in (('raw_join', 'raw-join.json'), ('trial_complete', 'complete.json'),
                      ('observer', 'probe/observer.jsonl'), ('group', 'group.json'), ('log_check', 'generation-log-check.json')):
        need(pinned(phase[key]) == folder/name, 'actual audited phase artifact path')
    done = read(folder/'complete.json'); spec_path = folder/'spec.json'
    need(sha(spec_path) == done['artifacts']['spec.json']['sha256'], 'audited phase spec')
    spec = read(spec_path); joined = read(folder/'raw-join.json')
    need(done['status'] == joined['status'] == 'PASS' and joined['qualified'] is True
         and spec['observer_enabled'] is True and spec['concurrency'] == joined['concurrency'] == concurrency
         and joined['max_http_outstanding'] == concurrency, 'actual observed C history')
    same(spec['base_plan'], base_plan, 'same observed base plan')
    fixtures = read(pinned(base['fixtures']))['fixtures']; same(spec['fixtures'], fixtures, 'whole qualified fixtures')
    same(spec['fixtures_ref'], base['fixtures'], 'fixture reference')
    need(spec['model_directory'] == base['model'] and spec['served_model'] == model
         and model == Path(base['model']).name, 'same snapshot/served model')
    same(spec['official_args'], joined['official_args'], 'observed original launch')
    need(spec['official_args'][spec['official_args'].index('--pack')+1] == spec['pack_dir'], 'observed pack')
    probe = load('../lithos_d512_retry/serve_probe.py'); client = load('../pareto_realistic/lithos_client.py')
    same(spec['jobs'], probe.jobs_for(fixtures, concurrency, model), 'original-message qualified jobs')
    need(joined['probe_sha256'] == FIXED['../lithos_d512_retry/serve_probe.py'], 'observed request builder pin')
    same(joined['source_pins'], probe.PINS, 'observed official source identity')
    need(len(joined['requests']) == phase['requests'] == (8 if concurrency == 1 else 16), 'whole C qualification')
    for job, row in zip(spec['jobs'], joined['requests']):
        f = next(f for f in fixtures if f['fixture_id'] == job['fixture_id'])
        same(row['prompt_ids'], f['prompt_ids'], 'qualified full prompt IDs')
        same(joined['fixture_contracts'][f['fixture_id']], dict(path=job['path'], body_sha256=client.canonical_body_sha256(job['body'])), 'qualified body/path')
    identity = dict(owner_plan_sha256=OWNER_PLAN_SHA, verification_sha256=digest(plan['verification']),
        verification_groups={k: digest(v) for k, v in plan['verification']['groups'].items()},
        model_manifest=base['model_manifest'], model_directory=base['model'], pack_dir=spec['pack_dir'],
        served_model=model, official_args=spec['official_args'], official_sources=joined['source_pins'],
        source_spec=dict(path=str(spec_path), sha256=sha(spec_path)), raw_join=phase['raw_join'],
        scope='Exact prior five-group runtime/source/model/pack identities; current full rehash belongs to outer owner.')
    return base, fixtures, joined, identity


def _contract(jobs, fixtures, binding, concurrency):
    client = load('../pareto_realistic/lithos_client.py')
    return dict(status='qualified', receipt_sha256=binding['audit']['sha256'], receipt_path=binding['audit']['path'],
        resource_audit=binding['resource_audit'], restoration_audit=binding['restoration_audit'], owner_plan=binding['owner_plan'],
        concurrency=concurrency, normal_eos=True, protocol=client.PROTOCOL, cache_contract=client.CACHE_CONTRACT,
        usage_count_rule=client.COUNT_RULE, usage_semantics=client.COUNT_SEMANTICS, raw_ids_count_rule='none',
        terminal_token_ids=[248046, 248044], no_post_terminal_tokens=True,
        fixture_contracts={j['fixture_id']: dict(path=j['path'], body_sha256=client.canonical_body_sha256(j['body'])) for j in jobs},
        request_fixture_map={j['request_id']: j['fixture_id'] for j in jobs},
        expected_prompt_tokens_by_fixture={f['fixture_id']: len(f['prompt_ids']) for f in fixtures})


def prepare(base_plan, qualification, concurrency, repeat, model):
    need(type(concurrency) is int and concurrency in (1, 8) and type(repeat) is int and 1 <= repeat <= 3, 'fixed C1/C8 repeat1..3')
    base, fixtures, joined, identity = bind(base_plan, qualification, concurrency, model)
    probe = load('../lithos_d512_retry/serve_probe.py'); originals = probe.jobs_for(fixtures, 1, model)
    jobs = {phase: [{**copy.deepcopy(originals[i % 8]), 'request_id': f'lithos-c{concurrency}-r{repeat}-{phase}-{i:02d}'}
            for i in range(count)] for phase, count in (('warm', 8), ('formal', 64))}
    sources = {str((HERE/name).resolve()): value for name, value in FIXED.items()}
    sources.update({str(probe.ROOT/name): value for name, value in probe.PINS.items()})
    sources.update({str(HERE.parent/'lithos_d512_retry'/name): value for name, value in AUDIT_SOURCES.values()})
    sources[str(Path(__file__).resolve())] = sha(__file__)
    for path, value in sources.items(): need(sha(path) == value, 'formal count source changed')
    contracts = {phase: _contract(q, fixtures, qualification, concurrency) for phase, q in jobs.items()}
    for phase, q in jobs.items(): need(load('../pareto_realistic/lithos_client.py')._qualification_errors(contracts[phase], q, concurrency) == [], 'client contract incompatible')
    return dict(base_plan=copy.deepcopy(base_plan), qualification=copy.deepcopy(qualification), qualification_identity=identity,
        concurrency=concurrency, repeat=repeat, fixtures=fixtures, fixtures_ref=base['fixtures'], jobs=jobs, contracts=contracts,
        model_directory=base['model'], served_model=model, pack_dir=identity['pack_dir'], sources=sources)


def count_contract(spec, phase):
    need(phase in ('warm', 'formal'), 'only warm/formal')
    _, fixtures, _, identity = bind(spec['base_plan'], spec['qualification'], spec['concurrency'], spec['served_model'])
    same(spec['fixtures'], fixtures, 'qualified full fixtures changed')
    same(spec['qualification_identity'], identity, 'qualified runtime/model/pack identity changed')
    need(type(spec['repeat']) is int and 1 <= spec['repeat'] <= 3, 'registered repeat')
    originals = load('../lithos_d512_retry/serve_probe.py').jobs_for(fixtures, 1, spec['served_model'])
    expected_jobs = [{**copy.deepcopy(originals[i % 8]), 'request_id': f"lithos-c{spec['concurrency']}-r{spec['repeat']}-{phase}-{i:02d}"}
                     for i in range(8 if phase == 'warm' else 64)]
    same(spec['jobs'][phase], expected_jobs, 'fixed original-message fixture cycle')
    expected = _contract(spec['jobs'][phase], spec['fixtures'], spec['qualification'], spec['concurrency'])
    same(spec['contracts'][phase], expected, 'bound phase count contract changed')
    return copy.deepcopy(expected)


def check_group(group, spec, phase):
    """Independent wire counts/timing only; no output-token oracle or retokenization."""
    client = load('../pareto_realistic/lithos_client.py'); raw = load('../lithos_d512_retry/join.py')
    jobs = spec['jobs'][phase]; C = spec['concurrency']; q = count_contract(spec, phase)
    need(group['schema'] == client.SCHEMA and group['protocol'] == client.PROTOCOL and group['phase'] == phase
         and type(group['concurrency']) is int and group['concurrency'] == C
         and group['group_error'] is None and group['qualification_errors'] == [], 'complete group contract')
    same(group['qualification'], q, 'actual supplied qualification')
    need(group['base_url'] == 'http://127.0.0.1:'+str(spec['port'])
         and group['request_timeout_s'] == spec['deadlines']['request']
         and group['group_timeout_s'] == spec['deadlines'][phase], 'registered endpoint/deadlines')
    need(group['frozen_scheduler_sha256'] == client.BASE_SHA256, 'actual frozen scheduler')
    same(group['connection_policy'], dict(max_connections=2*C, max_keepalive_connections=0, retries=0, follow_redirects=False), 'actual connection policy')
    cleanup = group['cleanup']
    need(cleanup['complete'] is True and cleanup['deadline_exceeded'] is False and cleanup['errors'] == []
         and cleanup['pending_tasks'] == [] and cleanup['cancelled_again'] is False and cleanup['client_close_started'] is True, 'all drain/close required')
    need(cleanup['budget_s'] == spec['deadlines']['close'], 'registered cleanup budget')
    need(all(finite(cleanup[k]) for k in ('start_perf_counter_s', 'end_perf_counter_s', 'budget_s'))
         and cleanup['budget_s'] > 0 and cleanup['end_perf_counter_s'] >= cleanup['start_perf_counter_s'], 'cleanup clock')
    need(len(jobs) == len(group['requests']) == (8 if phase == 'warm' else 64)
         and len({j['request_id'] for j in jobs}) == len(jobs), 'fixed complete unique queue')
    fixtures = {f['fixture_id']: f for f in spec['fixtures']}; need(len(fixtures) == 8, 'eight fixtures')
    barrier, deadline = group['barrier_release_perf_counter_s'], group['group_deadline_perf_counter_s']
    need(all(finite(x) and x > 0 for x in (barrier, deadline, group['request_timeout_s'], group['group_timeout_s']))
         and deadline == barrier + group['group_timeout_s'], 'absolute group clock')
    rows = []; events = []
    for job, row in zip(jobs, group['requests']):
        same({k: row.get(k) for k in job}, job, 'original request/job/path')
        f = fixtures[job['fixture_id']]
        expected_body = dict(model=spec['served_model'], messages=f['messages'], stream=True,
            options=dict(num_predict=512, temperature=0, top_p=1, top_k=0, seed=0))
        same(job['body'], expected_body, 'original messages, never rendered prompt')
        need(job['path'] == '/api/chat' and f['normal_eos'] is True and f['max_output_tokens'] == 512
             and f['eos_token_ids'] == [248046, 248044], 'qualified endpoint/EOS/cap')
        payload = base64.b64decode(row['request_body_bytes_base64'], validate=True)
        need(hashlib.sha256(payload).hexdigest() == row['request_body_bytes_sha256'], 'sent body bytes SHA')
        same(raw.parse(payload), expected_body, 'raw sent original messages')
        need(row['body_sha256'] == client.canonical_body_sha256(expected_body), 'canonical body SHA')
        need(row['done'] is True and type(row['terminal_count']) is int and row['terminal_count'] == 1
             and row['validation_errors'] == row['count_qualification_errors'] == [] and row['transport_valid'] is True, 'valid uniquely finished row')
        w = raw.wire(row); final = w['terminal']
        need(set(final) == {'model','created_at','message','done','done_reason','total_duration','load_duration',
             'prompt_eval_count','prompt_eval_duration','eval_count','eval_duration'}, 'official terminal fields')
        need(all(type(v) is int and v >= 0 for k, v in final.items() if k.endswith('_duration')), 'official duration units')
        n, finish = final['eval_count'], final['done_reason']
        need(type(n) is int and 1 <= n <= 512 and finish in ('stop', 'length')
             and (finish != 'length' or n == 512), 'raw normal-EOS/cap count')
        need(type(final['prompt_eval_count']) is int and final['prompt_eval_count'] == len(f['prompt_ids']), 'qualified prompt count')
        committed = n - int(finish == 'stop')
        same(row['terminal_record'], final, 'retained terminal fields')
        need(row['answer'] == w['text'] and row['reasoning'] == '' and (committed != 0 or w['text'] == ''), 'wire visible text/zero-token EOS')
        need(row['finish_reason'] == finish and type(row['qualified_output_tokens']) is int
             and row['qualified_output_tokens'] == committed and row['reported_eval_count'] == n
             and row['reported_cached_prompt_tokens'] is None, 'normalized count/cache unknown')
        same(row['count_normalization'], dict(rule=client.COUNT_RULE, reported_eval_count=n,
             terminal_eos_subtracted=int(finish == 'stop'), committed_output_tokens=committed,
             receipt_sha256=q['receipt_sha256']), 'EOS subtraction exactly once')
        start, end, eof = row['start_perf_counter_s'], row['done_perf_counter_s'], row['eof_perf_counter_s']
        need(all(finite(x) for x in (start,end,eof,row['absolute_deadline_perf_counter_s'])) and barrier <= start < end <= deadline
             and row['absolute_deadline_perf_counter_s'] == start+group['request_timeout_s']
             and end <= row['absolute_deadline_perf_counter_s'] and end <= eof <= cleanup['end_perf_counter_s']
             and end <= cleanup['start_perf_counter_s'] and row['e2e_s'] == end-start
             and row['terminal_perf_counter_s'] == end, 'POST/DONE/deadline/drain clocks')
        need(all(ch['client_perf_counter_s'] <= eof for ch in row['raw_ndjson_chunks']), 'raw tail clock after EOF')
        rows.append(dict(request_id=job['request_id'], fixture_id=f['fixture_id'], raw_output_tokens=n,
            committed_output_tokens=committed, finish_reason=finish, start_perf_counter_s=start,
            done_perf_counter_s=end, e2e_s=end-start, terminal_sha256=hashlib.sha256(w['terminal_line'].encode()).hexdigest()))
        events.extend(((start,1),(end,-1)))
    active = peak = 0
    for _, delta in sorted(events):
        active += delta; need(0 <= active <= C, 'HTTP outstanding outside C'); peak = max(peak, active)
    need(active == 0 and peak == C, 'actual HTTP concurrency differs from declaration')
    latency = [r['e2e_s'] for r in rows]; total = sum(r['committed_output_tokens'] for r in rows)
    window = max(r['done_perf_counter_s'] for r in rows)-min(r['start_perf_counter_s'] for r in rows)
    point = dict(concurrency=C, repeat=spec['repeat'], requests=len(rows), raw_output_tokens=sum(r['raw_output_tokens'] for r in rows),
        committed_output_tokens=total, common_window_s=window, aggregate_output_tok_s=total/window,
        e2e_p50_s=statistics.median(latency), e2e_p95_s=sorted(latency)[math.ceil(.95*len(rows))-1],
        eos_requests=sum(r['finish_reason']=='stop' for r in rows), cap_requests=sum(r['finish_reason']=='length' for r in rows),
        max_http_outstanding=peak, request_latencies_s=latency)
    summary = group['summary']
    need(summary['transport_complete'] is True and summary['eligible_under_supplied_qualification'] is True
         and summary['formal_point_eligible'] is (phase == 'formal'), 'client count eligibility')
    same([summary[k] for k in ('planned_requests','submitted_requests','done_requests','successful_requests','count_qualified_requests')], [len(rows)]*5, 'all requests closed/counted')
    same([summary['group_window_s'],summary['aggregate_output_tok_s'],summary['e2e_s']['p50'],summary['e2e_s']['p95'],summary['max_http_outstanding']],
         [window,total/window,point['e2e_p50_s'],point['e2e_p95_s'],peak], 'raw recomputed summary')
    return dict(schema='E023-lithos-formal-count-check-v1', status='PASS', count_eligible=True, point=point, requests=rows,
        full_state_qualified=False, performance_qualified=False, output_quality_qualified=False,
        scope='Conditional on audited normal-EOS route; wire has counts, no raw IDs. Cache/log/resource/reap/restoration gates remain external. HTTP C is not GPU batch size.')
