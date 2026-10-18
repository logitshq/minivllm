"""One fresh native C1/C8 timing trial: eight warm then 64 formal requests.

The fixed server runs with every qualification observer OFF. This module owns
only its direct server; an outer owner supplies resource/absent-service guards,
runtime/weight pins and restoration. It never controls launchd. Count eligibility
and raw timing are retained separately from final resource/performance approval.
Independent numerical qualification covers representative recorded histories,
not every future arrival schedule. No old schedule's token oracle is reused.
The CLI validates a prepared spec only. Actual execution requires run() with an
outer checkpoint that settles in-flight resource probes before propagating cancel.
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
FIXED = {
    'http_capture_v2.py': '536fe41d6c8e2bdffe83ba6f43fba87dd6265deb4c84f6204bc9c7220f8280e3',
    'http_qualification.py': '471bebd8a114d3a7f5a190c20909f6896f17fd39f282656546c1366c3fa2c6e2',
    'http_join.py': 'a362c321c2329d7ab53ab6a1a5a96981bb95b44733739fdbb56dd3d38bef7b4e',
    'native_cleanup.py': '6c292a9d3cb28c047d1dc92ac0fec4abe19fa8df84c172ef61d2f6ce41157ca6',
    'client.py': 'ef64a114f4f7d90ec5044d6bdcf058ca05f98dea445d6639db706a4fc75217dc',
    'native_server.py': '9bcb37eff8bdecd79800afdedc82deacf9f7a0a9a7a1e0d62dc8ba49e70fb68f',
    'audit_http_replay.py': '1e26e35bdde5cf7b81da37101326a4ef6f87ef7db2775c6324b56d053645c995',
    'audit_http_replay_v2.py': '7cdcf5a2f02768ef75686b0ff0b0da9d21f5f9c82ee3866e24c1f869c709f031',
}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def load(name):
    path = HERE / (name + '.py')
    need(sha(path) == FIXED[path.name], 'frozen formal dependency changed: ' + str(path))
    spec = importlib.util.spec_from_file_location('_native_formal_' + name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def read(path):
    return json.loads(Path(path).read_bytes())


def pinned(item):
    need(set(item) == {'path', 'sha256'} and Path(item['path']).is_absolute()
         and sha(item['path']) == item['sha256'], 'exact absolute artifact pin required')
    return Path(item['path'])


def write(path, value):
    path = Path(path); temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def bind_qualification(binding, concurrency, base_plan):
    """Consume real audited artifacts; this function never manufactures an audit."""
    need(set(binding) == {'audit', 'replay_plan', 'child_directory'}, 'explicit replay audit/plan/child binding')
    audit = read(pinned(binding['audit']))
    replay_path = pinned(binding['replay_plan'])
    child = Path(binding['child_directory'])
    need(child.is_absolute() and child.is_dir(), 'explicit replay child directory')
    q = load('http_qualification')
    replay, numeric = q.validate(replay_path, binding['replay_plan']['sha256'])
    need(replay['qualification'] == base_plan and replay['concurrency'] == concurrency, 'same base plan/C required')
    need(audit.get('schema') == 'E023-recorded-http-independent-audit-v2'
         and audit.get('status') == 'PASS' and audit.get('qualified') is True
         and audit.get('concurrency') == concurrency
         and audit.get('plan_sha256') == binding['replay_plan']['sha256']
         and audit.get('base_plan_sha256') == base_plan['sha256']
         and audit.get('sources') == replay['sources'] and audit.get('raw_inputs') == replay['inputs']
         and audit.get('child_directory') == str(child.resolve())
         and audit.get('frozen_v1_source_sha256') == FIXED['audit_http_replay.py']
         and audit.get('base_auditor_sha256') == 'b6d90413911f2b17e0951a2fcfdb5fe00666aa06d4382b603e29e5dc4659d4f1'
         and audit.get('performance_qualified') is False
         and audit.get('owner_resources_restore_qualified') is False,
         'independent captured-history audit does not qualify this source/config/C')
    auditor = HERE / 'audit_http_replay_v2.py'
    need(auditor.is_file() and audit.get('audit_source_sha256') == sha(auditor) == FIXED[auditor.name],
         'actual numerical auditor source pin')
    for filename, field in (('complete.json', 'child_complete_sha256'),
                            ('base/complete.json', 'base_complete_sha256'), ('recording.json', 'recording_sha256')):
        need(sha(child / filename) == audit.get(field), 'audited replay artifact changed: ' + filename)
    need(sha(child / 'plan.json') == binding['replay_plan']['sha256'], 'child executed a different replay plan')
    result = read(child / 'complete.json')
    need(result['status'] == 'PASS' and result['numerical_qualified'] is True
         and result['input_source_pins_rechecked'] is True and result['sources'] == replay['sources']
         and result['inputs'] == replay['inputs'] and result['concurrency'] == concurrency,
         'replay producer/audit mismatch')
    count = 8 if concurrency == 1 else 16
    need(result['total_requests'] == count and audit['ledger']['requests'] == count, 'complete representative workload required')
    if concurrency == 8:
        need(any(int(b) >= 2 and n > 0 for b, n in audit['ledger']['decode_batch_histogram'].items()),
             'actual multirow qualification required for C8')
    return replay, numeric, audit


def prepare(base_plan, concurrency, repeat, *, qualification, environment, transport, deadlines):
    """CPU preparation; outer owner must separately verify full runtime and weights."""
    need(type(concurrency) is int and concurrency in (1, 8), 'only C1/C8')
    need(type(repeat) is int and 1 <= repeat <= 3, 'registered repeat index 1..3')
    replay, numeric, audit = bind_qualification(qualification, concurrency, base_plan)
    capture = load('http_capture_v2'); q = load('http_qualification')
    need(set(deadlines) == {'ready', 'request', 'warm', 'formal', 'close', 'term', 'kill', 'overall', 'resource_interval'}
         and all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in deadlines.values()),
         'finite explicit single-trial budgets')
    need(deadlines['overall'] >= deadlines['ready'] + deadlines['warm'] + deadlines['formal']
         and deadlines['resource_interval'] <= 5 and deadlines['term'] == deadlines['kill'] == 15,
         'trial budget covers readiness/warm/formal; TERM/KILL15 and resource interval<=5')
    cap_deadlines = {k: v for k, v in deadlines.items() if k not in ('warm', 'formal')}
    cap_deadlines['group'] = deadlines['formal']
    cap = capture.prepare(base_plan['path'], base_plan['sha256'], concurrency,
        python=numeric['execution']['python'], environment=environment, transport=transport, deadlines=cap_deadlines)
    need(replay['transport'] == {k: transport[k] for k in ('tcp_nodelay', 'io_timeout_seconds')},
         'same qualified socket contract')
    identity = copy.deepcopy(cap['identity'])
    identity.update(step_observer=False, qualification_trace_enabled=False)
    fixtures = q.read(q.pinned(numeric['fixtures']))['fixtures']
    jobs = {}
    for phase, count in (('warm', 8), ('formal', 64)):
        jobs[phase] = []
        for i in range(count):
            original = cap['jobs'][i % 8]
            jobs[phase].append({**copy.deepcopy(original),
                'request_id': f'native-c{concurrency}-r{repeat}-{phase}-{i:02d}'})
    model = q.read(q.pinned(numeric['model_manifest']))
    tokenizer = {'path': str(Path(numeric['model']) / 'tokenizer.json'), 'sha256': model['files']['tokenizer.json']}
    pinned(tokenizer)
    sources = {str(HERE / name): digest for name, digest in FIXED.items()}
    sources[str(Path(__file__).resolve())] = sha(__file__)
    return dict(schema='E023-native-formal-spec-v1', base_plan=base_plan, qualification=qualification,
        concurrency=concurrency, repeat=repeat, command=cap['command'], environment=environment,
        transport=transport, deadlines=deadlines, identity=identity, sources=sources,
        fixtures=fixtures, jobs=jobs, tokenizer=tokenizer,
        scope='Representative captured histories qualified; no claim all future formal schedules were numerically replayed. Runtime/weight/resource/restoration checks belong to outer owner.')


def validate(spec):
    need(spec.get('schema') == 'E023-native-formal-spec-v1', 'formal spec required')
    for path, digest in spec['sources'].items():
        need(sha(path) == digest, 'formal source changed: ' + path)
    actual = prepare(spec['base_plan'], spec['concurrency'], spec['repeat'], qualification=spec['qualification'],
                     environment=spec['environment'], transport=spec['transport'], deadlines=spec['deadlines'])
    need(json.dumps(actual, sort_keys=True) == json.dumps(spec, sort_keys=True), 'formal spec changed')


def count_contract(spec, phase):
    client = load('native_cleanup'); jobs = spec['jobs'][phase]
    return dict(status='qualified', receipt_sha256=spec['qualification']['audit']['sha256'],
        concurrency=spec['concurrency'], normal_eos=True,
        fixture_contracts={job['fixture_id']: dict(path=job['path'], body_sha256=client.canonical_body_sha256(job['body'])) for job in jobs},
        request_fixture_map={job['request_id']: job['fixture_id'] for job in jobs},
        usage_semantics=client.COUNT_SEMANTICS, usage_count_rule='identity', raw_ids_count_rule='native_committed_ids',
        terminal_token_ids=spec['identity']['eos_ids'],
        expected_prompt_tokens_by_fixture={f['fixture_id']: len(f['prompt_ids']) for f in spec['fixtures']})


def check_group(group, spec, phase, decode_text):
    """Reparse retained raw bytes. Original wire/group objects remain untouched."""
    join = load('http_join'); jobs = spec['jobs'][phase]
    need(group['schema'] == 'E023-http-group-v1' and group['phase'] == phase
         and group['concurrency'] == spec['concurrency'] and group['qualification'] == count_contract(spec, phase)
         and group['group_error'] is None, 'formal/warm group contract')
    cleanup = group['cleanup']
    need(cleanup['complete'] is True and cleanup['deadline_exceeded'] is False and cleanup['errors'] == []
         and cleanup['pending_tasks'] == [] and cleanup['cancelled_again'] is False
         and cleanup['adapter'] == 'native-close-cancellation-v1', 'client cleanup incomplete')
    rows = group['requests']; need(len(rows) == len(jobs) == (8 if phase == 'warm' else 64), 'complete fixed queue')
    fixtures = {f['fixture_id']: f for f in spec['fixtures']}; eos = set(spec['identity']['eos_ids'])
    seen = set(); total = 0; latency = []; starts = []; ends = []
    for job, row in zip(jobs, rows):
        join.same({k: row.get(k) for k in job}, job, 'row changed original body/path/job')
        need(row['status'] == 'done' and row['http_status'] == 200 and row['error'] is None
             and row['validation_errors'] == [] and row['count_qualification_errors'] == [], 'failed formal request')
        rid, terminal, text = join.raw_completion(row)
        need(rid not in seen, 'duplicate response UUID'); seen.add(rid)
        fixture = fixtures[job['fixture_id']]; prompt = fixture['prompt_ids']; cap = fixture['max_output_tokens']
        join.same(terminal['benchmark_prompt_ids'], prompt, 'full prompt IDs differ from fixture')
        raw = terminal['benchmark_output_ids']; committed = terminal['benchmark_committed_output_ids']
        need(join.ids(raw) and 1 <= len(raw) <= cap and join.ids(committed)
             and not any(token in eos for token in raw[:-1]), 'raw IDs/cap/nonterminal EOS')
        stopped = raw[-1] in eos
        need(stopped or len(raw) == cap, 'early non-EOS termination')
        join.same(committed, raw[:-1] if stopped else raw, 'committed IDs')
        join.same(terminal['benchmark_terminal_eos_id'], raw[-1] if stopped else None, 'terminal EOS')
        need(terminal['choices'][0]['finish_reason'] == ('stop' if stopped else 'length'), 'finish/EOS/cap')
        join.same(terminal['usage'], dict(prompt_tokens=len(prompt), completion_tokens=len(committed),
            total_tokens=len(prompt) + len(committed), prompt_tokens_details={'cached_tokens': 0}), 'actual usage')
        need(decode_text(committed) == text == row['answer'] and row['reasoning'] == '', 'wire text/committed decode')
        need(type(row['qualified_output_tokens']) is int and row['qualified_output_tokens'] == len(committed), 'count eligibility')
        start, end = row['start_perf_counter_s'], row['done_perf_counter_s']
        need(end > start and row['e2e_s'] == end - start, 'POST to DONE latency')
        starts.append(start); ends.append(end); latency.append(end - start); total += len(committed)
    window = max(ends) - min(starts)
    point = dict(concurrency=spec['concurrency'], repeat=spec['repeat'], requests=len(rows),
        committed_output_tokens=total, common_window_s=window,
        aggregate_output_tok_s=total / window, e2e_p50_s=statistics.median(latency),
        e2e_p95_s=sorted(latency)[math.ceil(.95 * len(latency)) - 1])
    summary = group['summary']
    need(summary['transport_complete'] is True and summary['eligible_under_supplied_qualification'] is True
         and summary['formal_point_eligible'] is (phase == 'formal')
         and summary['group_window_s'] == window and summary['aggregate_output_tok_s'] == point['aggregate_output_tok_s']
         and summary['e2e_s']['p50'] == point['e2e_p50_s'] and summary['e2e_s']['p95'] == point['e2e_p95_s'],
         'raw count/window/latency differs from client summary')
    return dict(status='PASS', count_eligible=True, raw_requests_checked=len(rows), point=point,
                scope='Raw fixture/UUID/text/EOS/count binding only; no new full-state schedule proof.')


class System:
    def __init__(self):
        self.base = load('http_capture_v2').System()
    def __getattr__(self, name):
        return getattr(self.base, name)
    def decoder(self, spec):
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(str(pinned(spec['tokenizer'])))
        return lambda ids: tokenizer.decode(ids, skip_special_tokens=False)
    async def group(self, spec, phase, output):
        client = load('native_cleanup'); d = spec['deadlines']
        return await client.run_group(f"http://127.0.0.1:{spec['transport']['port']}", spec['jobs'][phase],
            spec['concurrency'], phase=phase, qualification=count_contract(spec, phase),
            request_timeout_s=d['request'], group_timeout_s=d[phase], cleanup_timeout_s=d['close'], output_dir=output)


async def run(spec, out, *, checkpoint, backend=None):
    """Single fresh trial. Mandatory outer checkpoint owns resources and service absence."""
    need(callable(checkpoint), 'external resource checkpoint required'); validate(spec)
    cap = load('http_capture_v2'); system = backend or System(); spec = copy.deepcopy(spec)
    out = Path(out).resolve(); out.mkdir(parents=True, exist_ok=False)
    d = spec['deadlines']; port = spec['transport']['port']; child = task = None
    stdout = stderr = None; phase = 'preflight'; last_group = None
    result = dict(schema='E023-native-formal-result-v1', status='INCOMPLETE', concurrency=spec['concurrency'],
        repeat=spec['repeat'], count_eligible=False, performance_qualified=False, server_reaped=False,
        cleanup_complete=False, started_at=time.time(), errors=[],
        scope='One observer-OFF trial; measured point still requires outer runtime/resource/reap/restoration audit.')
    write(out / 'spec.json', spec); write(out / 'complete.json', result)
    write(out / 'launch.json', dict(command=spec['command'], environment=spec['environment']))
    events = (out / 'events.jsonl').open('x')
    def event(label, **fields):
        events.write(json.dumps(dict(label=label, time=time.time(), monotonic=time.monotonic(), **fields)) + '\n'); events.flush()
    def error(exc, where):
        result['errors'].append(dict(phase=where, type=type(exc).__name__, message=str(exc)))
    async def check(label, pid):
        value = checkpoint(label, pid)
        if inspect.isawaitable(value): await value
        event('checkpoint', phase=label, pid=pid)
    async def run_group(name):
        try:
            group = await system.group(spec, name, out / name)
            write(out / (name + '-group.json'), group); return group
        except BaseException as exc:
            if hasattr(exc, 'result'): write(out / (name + '-group.json'), exc.result)
            raise
    try:
        decode_text = system.decoder(spec)
        async with asyncio.timeout(d['overall']):
            await check('pre-spawn', None)
            before = await system.port(port); write(out / 'port-before.json', before)
            need(cap.no_process(before), 'formal port occupied or unknown')
            need('--qualification-trace' not in spec['command'], 'formal trace must be OFF')
            stdout = (out / 'server.stdout').open('xb'); stderr = (out / 'server.stderr').open('xb')
            phase = 'spawn'; write(out / 'spawn-intent.json', dict(command=spec['command'], time=time.time()))
            child = system.spawn(spec['command'], spec['environment'], stdout, stderr)
            write(out / 'child.json', dict(pid=child.pid, time=time.time())); await check('spawned', child.pid)
            identity = await system.identity(child.pid); write(out / 'child-identity.json', identity)
            need(identity['returncode'] == 0 and identity['stdout'].split()[:2] == [str(child.pid), str(os.getpid())], 'direct formal server identity')
            phase = 'readiness'
            async with asyncio.timeout(d['ready']):
                sampled = time.monotonic()
                while True:
                    need(child.poll() is None, 'formal server exited before ready')
                    raw = await system.readiness(port); event('readiness', raw=raw)
                    if cap.ready_identity(raw, spec['identity']):
                        listener = await system.port(port, child.pid); write(out / 'listener.json', listener)
                        need(cap.owned_listener(listener, child.pid, port), 'owned localhost listener')
                        write(out / 'ready.json', raw); break
                    if time.monotonic() - sampled >= d['resource_interval']:
                        await check('starting', child.pid); sampled = time.monotonic()
                    await asyncio.sleep(.1)
            await check('ready', child.pid)
            for phase in ('warm', 'formal'):
                task = asyncio.create_task(run_group(phase), name='native-' + phase)
                while not task.done():
                    await asyncio.wait([task], timeout=d['resource_interval'])
                    need(child.poll() is None, 'formal server exited during group')
                    await check(phase, child.pid)
                last_group = task.result()
                checked = check_group(last_group, spec, phase, decode_text)
                write(out / (phase + '-raw-check.json'), checked)
                if phase == 'formal':
                    result.update(count_eligible=True, point=checked['point'])
                    write(out / 'point.json', dict(**checked['point'], count_eligible=True, performance_qualified=False))
                await check('after-' + phase, child.pid)
                raw = await system.readiness(port); write(out / ('ready-after-' + phase + '.json'), raw)
                need(cap.ready_identity(raw, spec['identity']), 'server identity failed after group')
    except BaseException as exc:
        error(exc, phase)
    finally:
        async def finish_owned():
            try:
                if task is not None and not task.done():
                    task.cancel(); await asyncio.wait([task], timeout=d['close'] + 1)
                if task is not None and not task.done():
                    task.cancel(); error(TimeoutError('client unresolved'), 'client-cleanup')
            except BaseException as exc:
                error(exc, 'client-cleanup')
            finally:
                if child is not None:
                    try:
                        if child.poll() is None:
                            event('server-term', pid=child.pid)
                            try: child.terminate(); await system.wait(child, d['term'])
                            except subprocess.TimeoutExpired: event('server-term-timeout', pid=child.pid)
                            except BaseException as exc: error(exc, 'server-term')
                            if child.poll() is None:
                                event('server-kill', pid=child.pid); child.kill()
                        await system.wait(child, d['kill'])
                        absent = await system.absent(child.pid); write(out / 'child-absence.json', absent)
                        port_after = await system.port(port); write(out / 'port-after.json', port_after)
                        need(type(child.returncode) is int and cap.no_process(absent) and cap.no_process(port_after), 'strict server/port absence')
                        write(out / 'release.json', dict(pid=child.pid, returncode=child.returncode,
                            waited_reaped=True, strict_absence=True, port_absent=True, time=time.time()))
                        result['server_reaped'] = True
                        group_path = out / (phase + '-group.json')
                        result['cleanup_complete'] = task is None or (task.done() and group_path.exists()
                            and read(group_path)['cleanup']['complete'] is True)
                    except BaseException as exc: error(exc, 'server-cleanup')
        cleaner = asyncio.create_task(finish_owned(), name='native-formal-owned-cleanup')
        end = asyncio.get_running_loop().time() + d['close'] + d['term'] + d['kill'] + 12
        while not cleaner.done() and asyncio.get_running_loop().time() < end:
            try: await asyncio.wait([cleaner], timeout=max(0, end - asyncio.get_running_loop().time()))
            except asyncio.CancelledError as exc: error(exc, 'cancel-during-cleanup')
        if not cleaner.done():
            cleaner.cancel(); result['cleanup_complete'] = False; error(TimeoutError('owned cleanup deadline'), 'cleanup')
        elif cleaner.cancelled():
            result['cleanup_complete'] = False; error(RuntimeError('cleanup task cancelled'), 'cleanup')
        elif cleaner.exception() is not None:
            result['cleanup_complete'] = False; error(cleaner.exception(), 'cleanup')
        result['cleanup_task_resolved'] = cleaner.done()
        result['client_task_resolved'] = task is None or task.done()
        try:
            await check('after-reap' if result['server_reaped'] else 'cleanup-incomplete',
                        None if result['server_reaped'] or child is None else child.pid)
            validate(spec); result['small_pins_rechecked'] = True
        except BaseException as exc: error(exc, 'postflight')
        if stdout is not None: stdout.close()
        if stderr is not None: stderr.close()
        events.close(); result['ended_at'] = time.time()
        result['status'] = 'PASS' if result['count_eligible'] and result['cleanup_complete'] and not result['errors'] else 'FAIL'
        result['artifacts'] = {p.name: {'sha256': sha(p), 'bytes': p.stat().st_size}
            for p in sorted(out.iterdir()) if p.is_file() and p.name != 'complete.json'}
        write(out / 'complete.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--spec-sha256', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    spec = read(pinned({'path': str(args.spec.resolve()), 'sha256': args.spec_sha256}))
    validate(spec); args.out.mkdir(parents=True, exist_ok=False)
    write(args.out / 'prepared.json', dict(status='PREPARED', spec_sha256=args.spec_sha256,
        scope='CPU-only preparation. An outer owner must call run with resource/lifecycle checkpoint.'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
