"""One fresh observer-OFF official oMLX D512 process: warm8, then formal64.

prepare/validate are CPU metadata operations; only explicit run owns a direct
server. A caller must provide settled resource/service checkpoints. The frozen
OpenAI SSE client counts already-committed usage without subtracting EOS. No
observer/seal, retokenized output oracle, GPU-batch or full-state claim is made.
"""
import argparse
import asyncio
import copy
import hashlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
QUAL_DIR = ROOT / '.artifacts/m5-ultra-256gb/omlx-d512-qualification-v1-owner'
QUAL_PLAN = HERE.parent / 'omlx_d512/prereg-qualification-v1/owner-plan.json'
ARCHIVE = ROOT / 'docs/experiments/m5-ultra-256gb/E023-omlx-d512-qualification-v1'
PLAN_SHA = 'e5294522149dfccccab4e26189280d77ddb6f3eb0b54fe78034646c59a71d4ef'
RECEIPTS = {
    'audit': ('independent-qualification.json', '8e9d77123e44d9485474976362dc70dcdbb3a14f6de290951d3ca989c33aa208'),
    'resource_audit': ('resource-summary.json', 'bc923557b7b59a230d10d5a163f0e5cfaf07c92074c06f04bfc0f7c2e4172531'),
    'restoration_audit': ('restoration-audit.json', '408bf10518e403c508d100abef613dd31b9a1db5bed76ceea34ab36fa20f6c7f'),
}
FIXED = {
    '../omlx_d512/trial.py': '18ceeadb0c47874308d58f87f93e61cc160d24bb551dd3e50c62f1084e4c0ec6',
    '../omlx_d512/serve_probe_v2.py': 'd0498a5fd825adcca97b4eee420dd89dbec9b6ee986b9a604986e286d2ad8b9e',
    '../omlx_d512/join_v2.py': '9a09959a840503240f387dc1a6e447311e08b1187b54ba3516e09a60c18cd8d1',
    '../omlx_d512/audit_qualification.py': 'e75a8a36b8fa53fcca0ff19c668092537e58c88c3257e4baa25f6235fba100f8',
    '../omlx_d512/audit_resources.py': '3db5d4a3bf6b7263e75381a23477ab407ee4476aa5c0e2a5a8ba774fdc2ba55a',
    '../omlx_d512/audit_restore.py': 'c5cc0afffd9494208c85f745c0cefb43ca1938fedd3f3c1b424b6ca094468127',
    '../pareto_realistic/native_cleanup.py': '6c292a9d3cb28c047d1dc92ac0fec4abe19fa8df84c172ef61d2f6ce41157ca6',
    '../pareto_realistic/client.py': 'ef64a114f4f7d90ec5044d6bdcf058ca05f98dea445d6639db706a4fc75217dc',
    '../pareto_realistic/http_capture_v2.py': '536fe41d6c8e2bdffe83ba6f43fba87dd6265deb4c84f6204bc9c7220f8280e3',
}
OFFICIAL = str(ROOT / '.artifacts/omlx-baseline/.venv/bin/omlx')


def need(ok, message):
    if not ok: raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while chunk := stream.read(1 << 20): digest.update(chunk)
    return digest.hexdigest()


def canonical(value): return json.dumps(value, sort_keys=True, allow_nan=False, separators=(',', ':')).encode()
def same(left, right, label): need(canonical(left) == canonical(right), label)
def digest(value): return hashlib.sha256(canonical(value)).hexdigest()
def finite(value): return type(value) in (int, float) and math.isfinite(value) and value >= 0


def load(name):
    path = (HERE / name).resolve(); need(name in FIXED and sha(path) == FIXED[name], 'frozen formal source changed')
    spec = importlib.util.spec_from_file_location('_omlx_formal_' + path.stem, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def read(path): return load('../omlx_d512/join_v2.py').parse(Path(path).read_bytes())


def write(path, value):
    path = Path(path); temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as stream:
        stream.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
        stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)
    fd = os.open(path.parent, os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)


def pinned(ref):
    need(type(ref) is dict and set(ref) == {'path', 'sha256'}, 'exact artifact reference')
    path = Path(ref['path'])
    need(path.is_absolute() and not path.is_symlink() and path.is_file() and sha(path) == ref['sha256'], 'artifact path/SHA')
    return path


def source_pins():
    return {**{str((HERE / name).resolve()): value for name, value in FIXED.items()}, str(Path(__file__).resolve()): sha(__file__)}


def qualification_binding():
    return dict(owner_directory=str(QUAL_DIR), owner_plan=dict(path=str(QUAL_PLAN), sha256=PLAN_SHA),
        **{key: dict(path=str(ARCHIVE / name), sha256=value) for key, (name, value) in RECEIPTS.items()})


def bind_qualification(binding, concurrency):
    """Exact completed observed histories, not an invented formal-output oracle."""
    need(type(concurrency) is int and concurrency in (1, 8), 'fixed C1/C8')
    same(binding, qualification_binding(), 'only actual fixed archived qualification')
    plan = read(pinned(binding['owner_plan']))
    audits = {name: read(pinned(binding[name])) for name in RECEIPTS}
    for name, filename in (('audit', 'audit_qualification.py'), ('resource_audit', 'audit_resources.py'), ('restoration_audit', 'audit_restore.py')):
        row = audits[name]; source = HERE.parent / 'omlx_d512' / filename
        need(row['status'] == 'PASS' and row['owner_plan_sha256'] == PLAN_SHA
             and row['audit_source_sha256'] == sha(source) == FIXED['../omlx_d512/' + filename], 'actual audit source/status/plan')
    audit, resource, restore = (audits[k] for k in RECEIPTS)
    need(audit['qualified'] is audit['route_count_text_qualified'] is True
         and resource['independent_resource_verified'] is restore['qualified'] is True
         and audit['full_state_qualified'] is audit['performance_qualified'] is False
         and restore['numerical_qualified'] is False, 'independent bounded qualification gates')
    need(audit['owner_directory'] == resource['owner_directory'] == str(QUAL_DIR)
         and audit['owner_complete_sha256'] == resource['owner_complete_sha256'] == sha(QUAL_DIR / 'complete.json')
         and sha(QUAL_DIR / 'owner-plan.json') == PLAN_SHA, 'actual closed owner binding')
    need(restore['owner_reap']['status'] == 'ALL_OBSERVED_CHILDREN_REAPED'
         and restore['service_restore']['status'] == 'PERSISTED_VISIBLE_RECEIPTS_CONSISTENT'
         and restore['journal_sha256'] == sha(ARCHIVE / 'restored-journal.json'), 'actual recovery closure')
    need(audit['requests'] == 24 and [r['concurrency'] for r in audit['phases']] == [1, 8]
         and [r['requests'] for r in audit['phases']] == [8, 16], 'both audited request histories')
    row = audit['phases'][0 if concurrency == 1 else 1]; folder = QUAL_DIR / 'phases' / f'omlx-c{concurrency}' / 'trial'
    # The large observer/SSE are already bound by the immutable independent audit.
    for key, filename in (('raw_join', 'raw-join.json'), ('trial_complete', 'complete.json'), ('request_seal', 'request-seal.json')):
        need(pinned(row[key]) == folder / filename, 'exact audited small phase artifact')
    done = read(folder / 'complete.json'); observed = read(folder / 'spec.json'); joined = read(folder / 'raw-join.json')
    need(sha(folder / 'spec.json') == done['artifacts']['spec.json']['sha256']
         and done['status'] == joined['status'] == 'PASS' and joined['protocol_route_count_qualified'] is True
         and observed['observer_enabled'] is True and observed['concurrency'] == concurrency
         and joined['max_http_outstanding'] == concurrency and joined['request_sealed'] is True,
         'actual observed C history')
    if concurrency == 8:
        need(any(int(k) >= 2 and type(v) is int and v > 0 for k, v in joined['batch_histogram'].items()), 'actual multirow history required')
    same(observed['base_plan'], plan['qualification'], 'same qualified base workload')
    base = read(pinned(plan['qualification'])); fixtures = read(pinned(base['fixtures']))['fixtures']
    same(observed['fixtures'], fixtures, 'full frozen fixtures'); same(observed['fixtures_ref'], base['fixtures'], 'fixture reference')
    probe = load('../omlx_d512/serve_probe_v2.py')
    same(observed['jobs'], probe.jobs_for(fixtures, concurrency), 'qualified original prompts and typed bodies')
    need(observed['model_directory'] == base['model'] == joined['loaded_identity']['model_path']
         and observed['served_model'] == 'Qwen3.8-27B-nvfp4'
         and joined['count_rule'] == 'identity: official completion_tokens excludes terminal EOS', 'same model/count semantics')
    probe.validate_runtime(joined['runtime']); probe.validate_loaded(joined['loaded_identity'], base['model'])
    for job, output in zip(observed['jobs'], joined['requests']):
        fixture = next(f for f in fixtures if f['fixture_id'] == job['fixture_id'])
        same(output['prompt_ids'], fixture['prompt_ids'], 'audited full prompt IDs')
    need(len(joined['requests']) == row['requests'], 'whole qualified history')
    identity = dict(owner_plan_sha256=PLAN_SHA, verification_sha256=digest(plan['verification']),
        verification_groups={k: digest(v) for k, v in plan['verification']['groups'].items()},
        source_spec=dict(path=str(folder / 'spec.json'), sha256=sha(folder / 'spec.json')), raw_join=row['raw_join'],
        base_plan=plan['qualification'], model_manifest=base['model_manifest'], model_directory=base['model'],
        served_model=observed['served_model'], official_args=observed['official_args'], official_sources=probe.PINS,
        loaded_identity=joined['loaded_identity'], runtime=joined['runtime'], observed_eos_requests=row['observed_terminal_eos'],
        scope='Prior observed route/count identity only; current runtime/model rehash and all resource/recovery gates belong to outer owner.')
    return dict(base_plan=plan['qualification'], fixtures_ref=base['fixtures'], fixtures=fixtures,
        model_directory=base['model'], served_model=observed['served_model'], qualification_identity=identity)


def jobs_for(fixtures, concurrency, repeat):
    originals = load('../omlx_d512/serve_probe_v2.py').jobs_for(fixtures, 1)
    return {phase: [{**copy.deepcopy(originals[i % 8]), 'request_id': f'omlx-c{concurrency}-r{repeat}-{phase}-{i:02d}'}
                   for i in range(count)] for phase, count in (('warm', 8), ('formal', 64))}


def _contract(jobs, fixtures, binding, concurrency):
    client = load('../pareto_realistic/native_cleanup.py')
    return dict(status='qualified', receipt_sha256=binding['audit']['sha256'], concurrency=concurrency, normal_eos=True,
        usage_semantics=client.COUNT_SEMANTICS, usage_count_rule='identity', raw_ids_count_rule='none',
        fixture_contracts={j['fixture_id']: dict(path=j['path'], body_sha256=client.canonical_body_sha256(j['body'])) for j in jobs},
        request_fixture_map={j['request_id']: j['fixture_id'] for j in jobs},
        expected_prompt_tokens_by_fixture={f['fixture_id']: len(f['prompt_ids']) for f in fixtures})


def prepare(out, concurrency, repeat, port, key, *, qualification, deadlines, environment):
    need(type(concurrency) is int and concurrency in (1, 8) and type(repeat) is int and 1 <= repeat <= 3, 'fixed C1/C8 repeat1..3')
    need(set(deadlines) == {'ready', 'request', 'warm', 'formal', 'close', 'term', 'kill', 'overall', 'resource_interval'}
         and all(finite(v) and v > 0 for v in deadlines.values()), 'explicit finite formal budgets')
    need(deadlines['overall'] >= sum(deadlines[k] for k in ('ready', 'warm', 'formal'))
         and deadlines['overall'] <= 3600 and deadlines['term'] == deadlines['kill'] == 15
         and deadlines['close'] <= 15 and deadlines['resource_interval'] <= 5, 'bounded original cleanup/resource contract')
    inherited = {k: v for k, v in deadlines.items() if k not in ('warm', 'formal')}; inherited['group'] = deadlines['formal']
    seed = load('../omlx_d512/trial.py').prepare(out, concurrency, port, key, deadlines=inherited, environment=environment)
    probe = load('../omlx_d512/serve_probe_v2.py'); launch = probe.specification(out, concurrency, port, key, observed=False)
    need(launch['command'] == [OFFICIAL, *seed['official_args']] and launch['observer_enabled'] is False, 'original observer-OFF CLI')
    data = bind_qualification(qualification, concurrency)
    for field in ('base_plan', 'fixtures_ref', 'fixtures', 'model_directory', 'served_model'):
        same(seed[field], data[field], 'same qualified ' + field)
    jobs = jobs_for(data['fixtures'], concurrency, repeat)
    contracts = {phase: _contract(rows, data['fixtures'], qualification, concurrency) for phase, rows in jobs.items()}
    sources = {**seed['sources'], **source_pins()}
    for path, value in sources.items(): need(sha(path) == value, 'small formal source changed')
    return dict(**data, schema='E023-omlx-D512-formal-spec-v1', output_directory=str(Path(out).resolve()),
        concurrency=concurrency, repeat=repeat, port=port, api_key=key, command=launch['command'], cwd=launch['cwd'],
        environment=dict(environment), deadlines=dict(deadlines), files=launch['files'], symlinks=launch['symlinks'],
        qualification=copy.deepcopy(qualification), qualification_binding=copy.deepcopy(qualification),
        observer_enabled=False, official_args=seed['official_args'], original_settings_plan_sha256=seed['original_settings_plan_sha256'],
        sources=sources, tokenizer_pins=seed['tokenizer_pins'], jobs=jobs, contracts=contracts,
        scope='Observer-OFF count/timing under the audited official route. No live raw IDs or GPU-batch/full-state observation; external source/runtime/resource/recovery audit still required.')


def validate(spec):
    need(spec.get('schema') == 'E023-omlx-D512-formal-spec-v1', 'formal spec schema')
    actual = prepare(spec['output_directory'], spec['concurrency'], spec['repeat'], spec['port'], spec['api_key'],
        qualification=spec['qualification_binding'], deadlines=spec['deadlines'], environment=spec['environment'])
    same(spec, actual, 'formal spec changed')


def count_contract(spec, phase):
    need(phase in ('warm', 'formal'), 'warm/formal only')
    bound = bind_qualification(spec['qualification'], spec['concurrency'])
    same(spec['qualification_binding'], spec['qualification'], 'qualification alias changed')
    for name in ('fixtures', 'qualification_identity', 'base_plan', 'model_directory', 'served_model'):
        same(spec[name], bound[name], 'bound ' + name + ' changed')
    need(type(spec['repeat']) is int and 1 <= spec['repeat'] <= 3, 'fixed repeat')
    same(spec['jobs'], jobs_for(bound['fixtures'], spec['concurrency'], spec['repeat']), 'fixed complete original-prompt queues')
    expected = _contract(spec['jobs'][phase], bound['fixtures'], spec['qualification'], spec['concurrency'])
    same(spec['contracts'][phase], expected, 'phase count contract changed')
    return expected


def check_group(group, spec, phase):
    """Reparse original wire; usage is committed already, never EOS-adjusted."""
    q = count_contract(spec, phase); jobs = spec['jobs'][phase]; C = spec['concurrency']
    wire = load('../omlx_d512/join_v2.py'); client = load('../pareto_realistic/native_cleanup.py')
    need(group['schema'] == 'E023-http-group-v1' and group['phase'] == phase and type(group['concurrency']) is int
         and group['concurrency'] == C and group['group_error'] is None and group['qualification_errors'] == [], 'complete group')
    same(group['qualification'], q, 'actual supplied count contract')
    need(group['base_url'] == f"http://127.0.0.1:{spec['port']}" and group['request_timeout_s'] == spec['deadlines']['request']
         and group['group_timeout_s'] == spec['deadlines'][phase], 'actual origin/deadlines')
    same(group['connection_policy'], dict(max_connections=2*C, max_keepalive_connections=0, retries=0, follow_redirects=False), 'connection policy')
    clean = group['cleanup']
    need(clean['complete'] is True and clean['errors'] == clean['pending_tasks'] == [] and clean['deadline_exceeded'] is False
         and clean['cancelled_again'] is False and clean['client_close_started'] is True
         and clean['adapter'] == 'native-close-cancellation-v1' and clean['frozen_client_sha256'] == client.BASE_SHA256
         and clean['budget_s'] == spec['deadlines']['close'], 'bounded frozen client cleanup')
    need(all(finite(clean[k]) for k in ('start_perf_counter_s', 'end_perf_counter_s'))
         and clean['end_perf_counter_s'] >= clean['start_perf_counter_s'], 'client close clocks')
    need(len(group['requests']) == len(jobs) == (8 if phase == 'warm' else 64), 'whole fixed queue')
    barrier, deadline = group['barrier_release_perf_counter_s'], group['group_deadline_perf_counter_s']
    need(finite(barrier) and finite(deadline) and deadline == barrier + spec['deadlines'][phase], 'absolute group clock')
    fixtures = {f['fixture_id']: f for f in spec['fixtures']}; rows = []; seen = set()
    for job, row in zip(jobs, group['requests']):
        same({k: row.get(k) for k in job}, job, 'actual original prompt/body/path/job')
        need(row['status'] == 'done' and row['done'] is True and type(row['http_status']) is int and row['http_status'] == 200
             and row['error'] is None and row['validation_errors'] == row['count_qualification_errors'] == []
             and row['transport_valid'] is True, 'complete valid request')
        value = wire.wire(row); rid = value['response_id']; need(rid not in seen, 'response UUID reused'); seen.add(rid)
        same(row['response_ids'], [rid], 'client/wire response UUID')
        usage = value['usage']; n = usage['completion_tokens']; prompt = len(fixtures[job['fixture_id']]['prompt_ids'])
        need(type(n) is int and 0 <= n <= 512 and (value['finish_reason'] != 'length' or n == 512)
             and (value['finish_reason'] != 'stop' or n < 512), 'normal EOS/cap committed usage')
        counts = dict(prompt_tokens=prompt, completion_tokens=n, total_tokens=prompt+n,
                      prompt_tokens_details={'cached_tokens': 0})
        same({k: usage.get(k) for k in counts}, counts, 'explicit uncached official usage')
        # Installed stream_completion includes these official timing extensions.
        # Preserve them verbatim, but derive our point only from client clocks.
        timing = {'model_load_duration', 'total_time', 'time_to_first_token', 'prompt_eval_duration',
                  'generation_duration', 'prompt_tokens_per_second', 'generation_tokens_per_second'}
        aliases = {'input_tokens': prompt, 'output_tokens': n}
        for key, expected in aliases.items():
            if key in usage:
                need(type(usage[key]) is int and usage[key] == expected, 'official usage alias count/type: ' + key)
        extra = set(usage)-set(counts)-set(aliases)
        need(extra <= timing and all(finite(usage[k]) for k in extra),
             'only finite official usage timing extensions')
        same(row['usage'], usage, 'parsed usage differs from retained wire')
        need(row['answer'] == value['text'] and row['reasoning'] == '' and (n != 0 or value['text'] == '')
             and row['finish_reason'] == value['finish_reason'] and type(row['qualified_output_tokens']) is int
             and row['qualified_output_tokens'] == row['reported_completion_tokens'] == n
             and row['reported_prompt_tokens'] == prompt and row['reported_cached_prompt_tokens'] == 0, 'unchanged identity count/text/finish')
        start, end = row['start_perf_counter_s'], row['done_perf_counter_s']
        need(all(finite(x) for x in (start, end, row['absolute_deadline_perf_counter_s'], row['terminal_perf_counter_s']))
             and barrier <= start < end <= deadline and row['absolute_deadline_perf_counter_s'] == start+spec['deadlines']['request']
             and end <= row['absolute_deadline_perf_counter_s'] and row['terminal_perf_counter_s'] == end
             and end <= clean['start_perf_counter_s'] and row['e2e_s'] == end-start, 'actual POST/DONE/close clocks')
        rows.append(dict(request_id=job['request_id'], fixture_id=job['fixture_id'], response_id=rid,
            committed_output_tokens=n, finish_reason=value['finish_reason'], start_perf_counter_s=start,
            done_perf_counter_s=end, e2e_s=end-start, wire_sha256=value['sha256'], wire_bytes=value['bytes']))
    peak = wire.http_outstanding(rows, C); latency = [r['e2e_s'] for r in rows]
    total = sum(r['committed_output_tokens'] for r in rows)
    window = max(r['done_perf_counter_s'] for r in rows)-min(r['start_perf_counter_s'] for r in rows)
    point = dict(concurrency=C, repeat=spec['repeat'], requests=len(rows), committed_output_tokens=total,
        common_window_s=window, aggregate_output_tok_s=total/window, e2e_p50_s=statistics.median(latency),
        e2e_p95_s=sorted(latency)[math.ceil(.95*len(rows))-1], stop_requests=sum(r['finish_reason']=='stop' for r in rows),
        cap_requests=sum(r['finish_reason']=='length' for r in rows), max_http_outstanding=peak, request_latencies_s=latency)
    summary = group['summary']
    need(summary['transport_complete'] is summary['eligible_under_supplied_qualification'] is True
         and summary['formal_point_eligible'] is (phase == 'formal'), 'client count eligibility')
    same([summary[k] for k in ('planned_requests', 'submitted_requests', 'done_requests', 'successful_requests', 'count_qualified_requests')],
         [len(rows)]*5, 'all planned requests counted')
    same([summary['group_window_s'], summary['aggregate_output_tok_s'], summary['e2e_s']['p50'], summary['e2e_s']['p95'], summary['max_http_outstanding']],
         [window, total/window, point['e2e_p50_s'], point['e2e_p95_s'], peak], 'wire/count/window versus client summary')
    return dict(schema='E023-omlx-D512-formal-count-check-v1', status='PASS', count_eligible=True, point=point, requests=rows,
        raw_output_ids_observed=False, full_state_qualified=False, performance_qualified=False,
        scope='Wire usage under separately qualified normal-EOS route; identity count, no EOS subtraction or tokenization oracle. HTTP overlap is not GPU batch size.')


class System:
    def __init__(self): self.base = load('../omlx_d512/trial.py').System()
    def __getattr__(self, name): return getattr(self.base, name)
    async def group(self, spec, phase, output):
        client = load('../pareto_realistic/native_cleanup.py'); d = spec['deadlines']
        return await client.run_group(f"http://127.0.0.1:{spec['port']}", spec['jobs'][phase], spec['concurrency'], phase=phase,
            qualification=count_contract(spec, phase), headers={'Authorization': 'Bearer '+spec['api_key']},
            request_timeout_s=d['request'], group_timeout_s=d[phase], cleanup_timeout_s=d['close'], output_dir=output)


async def run(spec, out, checkpoint, backend=None):
    need(callable(checkpoint), 'external settled resource/service checkpoint required'); validate(spec)
    out = Path(out).resolve(); need(str(out) == spec['output_directory'], 'registered output directory')
    spec = copy.deepcopy(spec); out.mkdir(parents=True, exist_ok=False)
    cap = load('../pareto_realistic/http_capture_v2.py'); trial = load('../omlx_d512/trial.py')
    system = backend or System(); d = spec['deadlines']; port = spec['port']
    result = dict(schema='E023-omlx-D512-formal-result-v1', status='INCOMPLETE', concurrency=spec['concurrency'], repeat=spec['repeat'],
        count_eligible=False, performance_qualified=False, full_state_qualified=False, server_reaped=False, cleanup_complete=False,
        spawn_attempted=False, popen_returned=False, errors=[], groups={}, raw_checks={}, started_at=time.time())
    write(out/'spec.json', spec); write(out/'launch.json', dict(command=spec['command'], cwd=spec['cwd'], environment=spec['environment']))
    write(out/'complete.json', result); child = task = None; term_sent = False; stdout = stderr = None; phase = 'preflight'; active_group = None; checked = {}
    events = (out/'events.jsonl').open('x'); boundaries = {}
    def event(label, **fields):
        events.write(json.dumps(dict(label=label, time=time.time(), monotonic=time.monotonic(), **fields))+'\n'); events.flush()
    def error(exc, where): result['errors'].append(dict(phase=where, type=type(exc).__name__, message=str(exc)))
    def ref(path): return dict(path=str(path.relative_to(out)), sha256=sha(path))
    def boundary(label):
        boundaries[label] = dict(time=time.time(), monotonic=time.monotonic(), perf_counter_s=time.perf_counter())
        write(out/'phase-boundaries.json', boundaries)
    async def check(label, pid):
        value = checkpoint(label, pid)
        if inspect.isawaitable(value): await value
        event('checkpoint', phase=label, pid=pid)
    async def group(name):
        try:
            value = await system.group(spec, name, out/name); write(out/(name+'-group.json'), value); return value
        except BaseException as exc:
            if hasattr(exc, 'result'): write(out/(name+'-group.json'), exc.result)
            raise
    try:
        async with asyncio.timeout(d['overall']):
            await check('pre-spawn', None); trial.materialize(spec, out)
            raw = await system.port(port); write(out/'port-before.json', raw); need(cap.no_process(raw), 'formal port occupied/unknown')
            need(spec['observer_enabled'] is False and spec['command'] == [OFFICIAL, *spec['official_args']], 'original observer OFF CLI')
            stdout = (out/'server.stdout').open('xb'); stderr = (out/'server.stderr').open('xb'); phase = 'spawn'
            write(out/'spawn-intent.json', dict(command=spec['command'], cwd=spec['cwd'], time=time.time())); result['spawn_attempted'] = True
            child = system.spawn(spec['command'], spec['environment'], stdout, stderr, cwd=spec['cwd']); result['popen_returned'] = True
            write(out/'child.json', dict(pid=child.pid, time=time.time())); await check('spawned', child.pid)
            identity = await system.identity(child.pid); write(out/'child-identity.json', identity)
            need(identity['returncode'] == 0 and identity['stdout'].split()[:2] == [str(child.pid), str(os.getpid())], 'direct formal server identity')
            phase = 'readiness'
            async with asyncio.timeout(d['ready']):
                sampled = time.monotonic()
                while True:
                    need(child.poll() is None, 'server exited before ready')
                    raw = await system.readiness(spec); event('readiness', raw=raw)
                    if trial.ready(raw, spec):
                        listener = await system.port(port, child.pid); write(out/'listener.json', listener)
                        need(cap.owned_listener(listener, child.pid, port), 'actual owned listener'); write(out/'ready.json', raw); break
                    if time.monotonic()-sampled >= d['resource_interval']:
                        await check('starting', child.pid); sampled = time.monotonic()
                    await asyncio.sleep(.1)
            await check('ready', child.pid)
            for phase in ('warm', 'formal'):
                active_group = phase; boundary(phase+'-start'); task = asyncio.create_task(group(phase), name='omlx-'+phase)
                while not task.done():
                    await asyncio.wait([task], timeout=d['resource_interval'])
                    need(child.poll() is None, 'server exited during requests'); await check(phase, child.pid)
                value = task.result(); boundary(phase+'-closed'); checked[phase] = check_group(value, spec, phase)
                need(checked[phase]['status'] == 'PASS' and checked[phase]['count_eligible'] is True, 'raw count/timing check failed')
                if phase == 'formal':
                    warm = read(out/'warm-group.json')
                    need(warm['cleanup']['end_perf_counter_s'] <= min(r['start_perf_counter_s'] for r in value['requests']), 'formal began before warm close')
                write(out/(phase+'-raw-check.json'), checked[phase]); result['groups'][phase] = ref(out/(phase+'-group.json'))
                result['raw_checks'][phase] = ref(out/(phase+'-raw-check.json')); await check('after-'+phase, child.pid)
                raw = await system.readiness(spec); write(out/('ready-after-'+phase+'.json'), raw); need(trial.ready(raw, spec), 'actual model identity changed')
    except BaseException as exc: error(exc, phase)
    finally:
        async def finish_owned():
            nonlocal term_sent
            try:
                if task is not None and not task.done(): task.cancel(); await asyncio.wait([task], timeout=d['close']+1)
                if task is not None and not task.done(): task.cancel(); error(TimeoutError('client unresolved'), 'client-cleanup')
            except BaseException as exc: error(exc, 'client-cleanup')
            finally:
                if child is not None:
                    try:
                        if child.poll() is None:
                            event('server-term', pid=child.pid)
                            try: child.terminate(); term_sent = True; await system.wait(child, d['term'])
                            except subprocess.TimeoutExpired: event('server-term-timeout', pid=child.pid)
                            except BaseException as exc: error(exc, 'server-term')
                            if child.poll() is None: event('server-kill', pid=child.pid); child.kill()
                        await system.wait(child, d['kill'])
                        absent = await system.absent(child.pid); write(out/'child-absence.json', absent)
                        after = await system.port(port); write(out/'port-after.json', after)
                        need(type(child.returncode) is int and cap.no_process(absent) and cap.no_process(after), 'strict server/port absence')
                        write(out/'release.json', dict(pid=child.pid, returncode=child.returncode, waited_reaped=True,
                            strict_absence=True, port_absent=True, term_sent=term_sent, time=time.time())); result['server_reaped'] = True
                        path = out/(str(active_group)+'-group.json')
                        result['cleanup_complete'] = task is None or (task.done() and path.is_file() and read(path)['cleanup']['complete'] is True)
                    except BaseException as exc: error(exc, 'server-cleanup')
        cleaner = asyncio.create_task(finish_owned(), name='omlx-formal-owned-cleanup')
        end = asyncio.get_running_loop().time()+d['close']+d['term']+d['kill']+12
        while not cleaner.done() and asyncio.get_running_loop().time() < end:
            try: await asyncio.wait([cleaner], timeout=max(0, end-asyncio.get_running_loop().time()))
            except asyncio.CancelledError as exc: error(exc, 'cancel-during-cleanup')
        if not cleaner.done(): cleaner.cancel(); result['cleanup_complete'] = False; error(TimeoutError('owned cleanup deadline'), 'cleanup')
        elif cleaner.cancelled(): result['cleanup_complete'] = False; error(RuntimeError('cleanup task cancelled'), 'cleanup')
        elif cleaner.exception() is not None: result['cleanup_complete'] = False; error(cleaner.exception(), 'cleanup')
        result['cleanup_task_resolved'] = cleaner.done(); result['client_task_resolved'] = task is None or task.done()
        try:
            await check('after-reap' if result['server_reaped'] else 'cleanup-incomplete', None if result['server_reaped'] or child is None else child.pid)
            validate(spec); result['small_pins_rechecked'] = True
            if result['server_reaped'] and result['cleanup_complete'] and not result['errors'] and list(checked) == ['warm', 'formal']:
                need(child.returncode == 0 or (child.returncode == -15 and term_sent), 'only normal or own explained SIGTERM exit')
                result.update(count_eligible=True, point=checked['formal']['point'])
                write(out/'point.json', dict(**result['point'], count_eligible=True, performance_qualified=False))
        except BaseException as exc: error(exc, 'postflight')
        if stdout is not None: stdout.close()
        if stderr is not None: stderr.close()
        events.close(); result['ended_at'] = time.time()
        result['status'] = 'PASS' if result['count_eligible'] and not result['errors'] else 'FAIL'
        result['artifacts'] = {str(p.relative_to(out)): dict(sha256=sha(p), bytes=p.stat().st_size)
            for p in sorted(out.rglob('*')) if p.is_file() and p != out/'complete.json'}
        write(out/'complete.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--spec', type=Path)
    parser.add_argument('--spec-sha256'); parser.add_argument('--out', type=Path); args = parser.parse_args(argv)
    if args.spec is None and args.spec_sha256 is None and args.out is None:
        print('DESCRIPTION ONLY. Explicit --spec/--spec-sha256/--out verifies CPU metadata; actual run requires external checkpoint.'); return 0
    need(args.spec is not None and args.spec_sha256 is not None and args.out is not None, 'complete explicit preparation arguments required')
    spec = read(pinned(dict(path=str(args.spec.resolve()), sha256=args.spec_sha256))); validate(spec)
    args.out.mkdir(parents=True, exist_ok=False)
    write(args.out/'prepared.json', dict(status='PREPARED', spec_sha256=args.spec_sha256, performance_qualified=False,
        scope='CPU metadata only; no server, resource or service operations.'))
    return 0


if __name__ == '__main__': raise SystemExit(main())
