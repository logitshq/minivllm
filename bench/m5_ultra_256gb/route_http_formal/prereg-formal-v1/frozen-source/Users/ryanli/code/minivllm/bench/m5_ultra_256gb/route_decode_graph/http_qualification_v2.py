"""Prepared route-ON HTTP replay adapter; no action without an explicit mode.

--prepare-only joins source-pinned actual HTTP data with the CPU tokenizer.
--run additionally invokes the original two-model numerical qualifier, installed
and observed per instance, AFTER an external owner has reaped the capture server.
This module never starts/stops a service or a subprocess. Outer resource/runtime
admission, hard timeout/reap and independent numerical/route audit remain required.

Producer PASS means completed checks for this recorded schedule, not independent
HTTP qualification, future-arrival correctness or a performance point. The real
37-request/42-direct-call serving audit is a separate pinned prerequisite, not rerun
or relabelled by this one-C adapter. Successful tensor bytes are not retained by
the inherited Comparator; first-failure original-dtype arrays are preserved.
"""
import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import traceback

HERE = Path(__file__).resolve().parent
NATIVE = HERE.parent / 'pareto_realistic'
OLD = HERE.parent / 'long_decode_graph'
BASE = NATIVE / 'qualify_native.py'
BASE_PLAN_SHA = '21ba52e43ce54c2b463ab46bfd212c53d06d6303530c77939e3e6b801df547fb'
SERVING_PLAN_SHA = 'e85f92aa1402025a669de4c2426460c1eced29f0145f128480f1fa9e66349c56'
FIXED = {
    BASE: '26964fcbed9a2e5dcefaee4888244987897f3138d4a2b7add59bf63a7461b7bf',
    NATIVE / 'http_qualification.py': '471bebd8a114d3a7f5a190c20909f6896f17fd39f282656546c1366c3fa2c6e2',
    NATIVE / 'http_join.py': 'a362c321c2329d7ab53ab6a1a5a96981bb95b44733739fdbb56dd3d38bef7b4e',
    NATIVE / 'http_replay.py': '1304c58c66e56b3aeaa582ef9f7d61317280cf3107f8f600485426119970fb65',
    NATIVE / 'http_capture_v2.py': '536fe41d6c8e2bdffe83ba6f43fba87dd6265deb4c84f6204bc9c7220f8280e3',
    NATIVE / 'native_server.py': '9bcb37eff8bdecd79800afdedc82deacf9f7a0a9a7a1e0d62dc8ba49e70fb68f',
    NATIVE / 'client.py': 'ef64a114f4f7d90ec5044d6bdcf058ca05f98dea445d6639db706a4fc75217dc',
    NATIVE / 'native_cleanup.py': '6c292a9d3cb28c047d1dc92ac0fec4abe19fa8df84c172ef61d2f6ce41157ca6',
    HERE / 'http_capture.py': 'b331a02988413356e113b3a1283ab0eb8d40de17796b5623a17df312c214151f',
    HERE / 'http_server_v2.py': '8ed952b4870465718a33e0af7e4ecddbf3fbd6265571e7efd3467b904c5621b7',
    HERE / 'integration.py': '4a8dc5518f2ea0a913ad0a5d753a5032324ccb69bacede416abe8ae126bfa800',
    HERE / 'candidate.py': '6531be25acb53713d6ddcd930d2c778f6640a3b1f53304fd5375d20b044525c3',
    HERE / 'serving_qualification.py': '75de7205febdae2698e99353bfc2908a79044591fbf67abb30a07cb4e54a7264',
    HERE / 'audit_serving.py': '064634acae9510b5744b7c848f09ee0854ee8bf26fc480af78dbed7f159573ba',
    OLD / 'qualify.py': 'c24219f431b08502a26a4f7a83c002214bddd7ea3962d6b1869b313e4c13faf7',
    OLD / 'candidate.py': 'eb14b671a66764fe677219762090acb495b89d384cf078f3f63b918a40aa2b84',
    OLD / 'model-source-pins.json': '7143538a168bfcd8b01292c9687472c8647db393c6d2a510d628f71c670260c5',
}
CAPTURE_NAMES = dict(spec='spec.json', complete='complete.json', child='child.json',
    release='release.json', absence='child-absence.json', port='port-after.json')
INPUT_NAMES = dict(group='group.json', trace='trace.jsonl', route_events='route-events.jsonl')


def need(ok, message):
    if not ok:
        raise ValueError(message)


def same(left, right, message):
    need(json.dumps(left, sort_keys=True, allow_nan=False) ==
         json.dumps(right, sort_keys=True, allow_nan=False), message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(1 << 20):
            h.update(block)
    return h.hexdigest()


def parse(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            need(key not in result, 'duplicate JSON field: ' + key)
            result[key] = value
        return result
    def bad(value):
        raise ValueError('nonfinite JSON constant: ' + value)
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=bad)


def read(path):
    return parse(Path(path).read_bytes())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def pinned(item):
    need(type(item) is dict and set(item) == {'path', 'sha256'}, 'exact path/SHA reference')
    path = Path(item['path'])
    need(path.is_absolute() and path.resolve() == path and path.is_file() and not path.is_symlink(),
         'canonical absolute regular file reference')
    need(sha(path) == item['sha256'], 'pinned input changed: ' + str(path))
    return path


def source_pins():
    return {**{str(p): h for p, h in FIXED.items()}, str(Path(__file__).resolve()): sha(__file__)}


def load(path, name):
    need(path in FIXED and sha(path) == FIXED[path], 'fixed module changed: ' + str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def serving_gate(binding):
    need(set(binding) == {'plan', 'audit'}, 'serving plan and independent audit required')
    plan = read(pinned(binding['plan']))
    audit = read(pinned(binding['audit']))
    need(binding['plan']['sha256'] == SERVING_PLAN_SHA and
         plan['schema'] == 'E024-route-serving-plan-v1' and
         plan['qualification']['sha256'] == BASE_PLAN_SHA and
         plan['qualifier_sha256'] == FIXED[HERE / 'serving_qualification.py'], 'fixed serving plan')
    need(audit['schema'] == 'E024-serving-independent-audit-v1' and audit['status'] == 'PASS'
         and audit['qualified'] is True and audit['plan_sha256'] == SERVING_PLAN_SHA
         and audit['base_plan_sha256'] == BASE_PLAN_SHA
         and audit['audit_source_sha256'] == FIXED[HERE / 'audit_serving.py']
         and audit['http_qualified'] is False and audit['performance_qualified'] is False,
         'actual independent serving PASS required; no transferred HTTP claim')
    coverage = audit['coverage']
    need(coverage['parameter_checks_complete'] is True and coverage['parameter_direct_calls'] == 42
         and coverage['identity_refill_enabled'] is False
         and [len(case['completed']) for case in audit['ledger']['cases']] == [8, 16, 13],
         'complete original37 plus42 serving prerequisite')
    for name in ('D-C1', 'D-C8'):
        need(all(coverage['route_hits'][name][bound] > 0 for bound in ('1023', '1536')),
             'serving prerequisite must cover both graph families')
    return dict(plan_sha256=binding['plan']['sha256'], audit_sha256=binding['audit']['sha256'],
        scope='Separate original37/42 same-schedule prerequisite; not the new HTTP schedule.')


def capture_gate(plan):
    need(set(plan['capture']) == set(CAPTURE_NAMES) and set(plan['inputs']) == set(INPUT_NAMES),
         'exact captured specification, termination and raw inputs required')
    paths = {k: pinned(v) for k, v in plan['capture'].items()}
    inputs = {k: pinned(v) for k, v in plan['inputs'].items()}
    directory = paths['spec'].parent
    for table, names in ((paths, CAPTURE_NAMES), (inputs, INPUT_NAMES)):
        for key, path in table.items():
            need(path == directory / names[key], 'mixed or renamed capture inputs: ' + key)
    spec, complete, child, release = [read(paths[k]) for k in ('spec', 'complete', 'child', 'release')]
    capture = load(HERE / 'http_capture.py', '_e024_replay_capture_contract')
    capture.validate(spec)
    need(spec['concurrency'] == plan['concurrency'] and spec['base_plan'] == plan['qualification'],
         'capture concurrency/base plan')
    need(complete['schema'] == 'E024-route-http-capture-result-v1' and complete['status'] == 'PASS'
         and all(complete.get(k) is True for k in ('capture_complete', 'server_reaped', 'cleanup_complete',
             'cleanup_task_resolved', 'client_task_resolved', 'small_pins_rechecked', 'route_events_retained'))
         and complete['errors'] == [] and complete['numerical_qualified'] is False
         and complete['performance_qualified'] is False, 'complete observed capture with closed cleanup required')
    pid = child['pid']
    need(type(pid) is int and pid > 1 and pid != os.getpid() and release['pid'] == pid
         and type(release['returncode']) is int
         and all(release[k] is True for k in ('waited_reaped', 'strict_absence', 'port_absent')),
         'separate captured server must have waited/reaped release')
    need(capture.no_process(read(paths['absence'])) and capture.no_process(read(paths['port'])),
         'captured server and port absence receipts required')
    need(type(release['time']) in (int, float) and math.isfinite(release['time'])
         and child['time'] <= release['time'] <= complete['ended_at'], 'capture/release time order')
    # Retained receipts are not a replacement for the outer owner's current
    # process/resource check. Bind them to the complete producer's actual bytes.
    for key, path in {**paths, **inputs}.items():
        if key == 'complete':
            continue
        same(complete['artifacts'][path.name], dict(sha256=sha(path), bytes=path.stat().st_size),
             'capture raw artifact binding: ' + path.name)
    header = dict(event='route-startup', schema=capture.ROUTE_SCHEMA, pid=pid,
        phase=f"HTTP-C{plan['concurrency']}", identity=spec['identity'])
    with inputs['route_events'].open('rb') as stream:
        raw = stream.readline()
        need(raw.endswith(b'\n'), 'complete route startup line required')
        same(parse(raw), header, 'route startup PID/source/phase binding')
    return spec, header


def validate(path, expected_sha):
    need(sha(path) == expected_sha, 'new HTTP replay plan SHA')
    plan = read(path)
    need(set(plan) == {'schema', 'qualifier_sha256', 'qualification', 'serving', 'concurrency',
        'capture', 'inputs', 'sources'} and plan['schema'] == 'E024-recorded-route-http-plan-v1',
        'exact route replay plan required')
    need(plan['qualifier_sha256'] == sha(__file__), 'actual adapter source identity')
    need(type(plan['concurrency']) is int and plan['concurrency'] in (1, 8), 'fixed C1/C8 domain')
    same(plan['sources'], source_pins(), 'complete fixed adapter/dependency source map')
    for source, digest in plan['sources'].items():
        need(sha(source) == digest, 'source changed: ' + source)
    need(plan['qualification']['sha256'] == BASE_PLAN_SHA, 'unchanged original model/workload plan')
    numeric = read(pinned(plan['qualification']))
    need(numeric['candidate'] == dict(compile_decode=True, fuse_decode=True, gates_compile=True, norm_reuse=False)
         and numeric['configuration']['max_len'] == 2048 and numeric['deadline_s'] == 1800,
         'original flags/context/deadline')
    serving_gate(plan['serving'])
    spec, header = capture_gate(plan)
    return plan, numeric, spec, header


def prepare(plan, numeric, spec):
    """Join unchanged real wire/trace; no candidate/runtime import here."""
    q = load(NATIVE / 'http_qualification.py', '_e024_replay_original_jobs')
    join = load(NATIVE / 'http_join.py', '_e024_replay_original_join')
    fixtures = read(pinned(numeric['fixtures']))['fixtures']
    model = read(pinned(numeric['model_manifest']))
    tokenizer_path = Path(numeric['model']) / 'tokenizer.json'
    need(sha(tokenizer_path) == model['files']['tokenizer.json'], 'original tokenizer SHA')
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    for fixture in fixtures:
        same(tokenizer.encode(fixture['prompt_text'], add_special_tokens=False).ids, fixture['prompt_ids'],
             'full actual HTTP prompt IDs: ' + fixture['fixture_id'])
    jobs = q.jobs_for(fixtures, plan['concurrency'])
    same(jobs, spec['jobs'], 'same actual captured request bodies/order')
    group = read(pinned(plan['inputs']['group']))
    with pinned(plan['inputs']['trace']).open('rb') as trace:
        recording = join.join(group, trace, jobs, {f['fixture_id']: f for f in fixtures},
            spec['identity'], lambda ids: tokenizer.decode(ids, skip_special_tokens=False))
    return recording, fixtures


class CapturedRoutes:
    """Streaming strict call-order comparison; no assumed calls-per-step count."""
    def __init__(self, stream, header, replay_file, captured_ref, phase):
        self.stream, self.output = stream, replay_file
        self.calls = self.hits = self.fallbacks = 0
        self.bounds = {'1023': 0, '1536': 0}
        self.phase = phase
        self.digest = hashlib.sha256()
        self.bytes = self.lines = 0
        same(self.take(), header, 'captured route startup changed')
        self.output.write(json.dumps(dict(event='replay-route-startup',
            schema='E024-route-http-replay-events-v1', pid=os.getpid(), phase=phase,
            captured_route_events=captured_ref), allow_nan=False) + '\n')
        self.output.flush()

    def take(self):
        raw = self.stream.readline()
        need(raw and raw.endswith(b'\n'), 'missing/truncated captured route event')
        self.digest.update(raw)
        self.bytes += len(raw)
        self.lines += 1
        return parse(raw)

    def emit(self, row):
        # Preserve the actual failed row before comparing to the captured event.
        self.output.write(json.dumps(row, allow_nan=False) + '\n')
        self.output.flush()
        expected = self.take()
        same(row, expected, 'actual captured/replayed route call differs')
        need(row['event'] == 'route-call' and row['phase'] == self.phase
             and type(row['step']) is int and row['step'] > 0 and row['direct_label'] is None
             and row['status'] == 'SUBMITTED' and type(row['actual_invocations']) is int
             and row['actual_invocations'] in (0, 1), 'completed original route witness row')
        self.calls += 1
        self.hits += row['actual_invocations']
        self.fallbacks += 1 - row['actual_invocations']
        if row['actual_invocations']:
            self.bounds['1023' if row['used'] < 1024 else '1536'] += 1

    def finish(self, captured_sha):
        need(self.stream.read(1) == b'', 'unconsumed/extra captured route calls')
        need(self.calls > 0 and self.hits > 0, 'no actual compiled HTTP invocation in this C scope')
        need(self.digest.hexdigest() == captured_sha, 'consumed route stream changed')
        return dict(calls=self.calls, actual_compiled_invocations=self.hits, fallbacks=self.fallbacks,
            bounds=self.bounds, captured_sha256=self.digest.hexdigest(), bytes=self.bytes, lines=self.lines,
            scope='Ordered actual calls per recorded step, including actual ahead behavior; SUBMITTED alone is not numeric PASS.')


@contextmanager
def adapted(base, replay, integration, witness_module, counter, recording, fixtures, routes):
    """Private base-module hooks only; retain the frozen replay math unchanged."""
    saved = base.Observer, base.workloads, base.run_case
    coverage = dict(installs=0, restores=0, route_hits={}, long_fallback_calls=0,
                    identity_refill_enabled=False)
    phase = f"HTTP-C{recording['concurrency']}"
    case = dict(name=f"D-C{recording['concurrency']}", concurrency=recording['concurrency'])

    class RouteObserver(saved[0]):
        def __init__(self, engine, comparator, label):
            self.route_install = self.route_witness = self.graph = self.saved_functions = None
            self.route_step = 0
            self.closed = False
            runner, scheduler = engine.runner, engine.scheduler
            original_model, original_decode = runner.model, runner.decode_model
            original_run = ('run' in vars(runner), vars(runner).get('run'))
            original_schedule = ('schedule' in vars(scheduler), vars(scheduler).get('schedule'))
            try:
                if label == 'candidate':
                    self.route_install = integration.install(runner, enabled=True)
                    self.graph = self.route_install.decoder.long
                    self.saved_functions = self.graph._functions
                    self.route_witness = witness_module.RouteWitness(self.graph, counter, routes.emit, coverage)
                    self.route_witness.phase = phase
                    self.route_install.decoder.long = self.route_witness
                    coverage['installs'] += 1
                super().__init__(engine, comparator, label)
            except BaseException:
                runner.model = original_model
                runner.decode_model = self.route_install.decoder if self.route_install else original_decode
                for target, name, original in ((runner, 'run', original_run), (scheduler, 'schedule', original_schedule)):
                    if original[0]: setattr(target, name, original[1])
                    else: vars(target).pop(name, None)
                if self.graph is not None:
                    self.graph._functions = self.saved_functions
                    self.route_install.decoder.long = self.graph
                if self.route_install is not None:
                    self.route_install.restore()
                raise

        def begin(self, expected_schedule=None):
            super().begin(expected_schedule)
            self.route_step += 1
            if self.route_witness is not None:
                self.route_witness.step = self.route_step
                self.route_witness.direct_label = None

        def restore(self):
            if self.closed:
                return
            super().restore()
            if self.route_install is not None:
                need(self.route_install.decoder.long is self.route_witness
                     and self.graph._functions is self.route_witness.functions, 'foreign route observer during release')
                self.route_install.decoder.long = self.graph
                self.graph._functions = self.saved_functions
                self.route_install.restore()
            self.calls.clear()
            self.previous_future = self.postrun = None
            self.engine = self.original_model = self.original_decode = self.original_run = self.original_schedule = None
            self.route_install = self.route_witness = self.graph = self.saved_functions = None
            self.closed = True
            coverage['restores'] += 1

    def workloads(actual):
        same(actual, fixtures, 'base loaded different fixtures')
        return [case]

    def run_case(actual, engines, observers, sampling, runner_module, cmp, ledger, deadline):
        same(actual, case, 'one selected actual HTTP C only')
        return replay.run_recorded_case(recording, engines, observers, sampling, runner_module,
            cmp, ledger, deadline, base=base)

    base.Observer, base.workloads, base.run_case = RouteObserver, workloads, run_case
    try:
        yield coverage
    finally:
        base.Observer, base.workloads, base.run_case = saved


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--plan-sha256')
    parser.add_argument('--out', type=Path)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--prepare-only', action='store_true')
    modes.add_argument('--run', action='store_true')
    args = parser.parse_args(argv)
    if not (args.prepare_only or args.run):
        print('DESCRIPTION_ONLY: explicit --prepare-only or --run is required; no files or runtime loaded.')
        return 0
    need(args.plan is not None and args.plan_sha256 is not None and args.out is not None,
         'explicit mode requires plan, plan SHA and fresh output')
    args.out.mkdir(parents=True, exist_ok=False)
    result = dict(schema='E024-recorded-route-http-result-v1', status='INCOMPLETE',
        plan_sha256=args.plan_sha256, prepare_only=args.prepare_only,
        numerical_qualified=False, http_qualified=False, performance_qualified=False,
        replay_numerical_checks_passed=False, independent_audit_required=True,
        owner_resources_restore_qualified=False,
        scope='Only this captured HTTP history; original arithmetic comparisons are producer receipts. Independent audit and current outer owner admission remain required.')
    write(args.out / 'complete.json', result)
    try:
        plan, numeric, spec, header = validate(args.plan, args.plan_sha256)
        write(args.out / 'plan.json', plan)
        result.update(sources=plan['sources'], inputs=plan['inputs'], capture=plan['capture'],
            serving=plan['serving'], concurrency=plan['concurrency'])
        recording, fixtures = prepare(plan, numeric, spec)
        write(args.out / 'recording.json', recording)
        result['recording_sha256'] = sha(args.out / 'recording.json')
        if args.prepare_only:
            result['status'] = 'PREPARED'
        else:
            base = load(BASE, '_e024_route_http_private_base')
            replay = load(NATIVE / 'http_replay.py', '_e024_route_http_private_replay')
            integration = load(HERE / 'integration.py', '_e024_route_http_private_install')
            witness = load(HERE / 'serving_qualification.py', '_e024_route_http_private_witness')
            counter = load(OLD / 'qualify.py', '_e024_route_http_private_counter')
            with pinned(plan['inputs']['route_events']).open('rb') as captured, \
                 (args.out / 'replay-route-events.jsonl').open('x') as emitted:
                routes = CapturedRoutes(captured, header, emitted, plan['inputs']['route_events'],
                    f"HTTP-C{plan['concurrency']}")
                with adapted(base, replay, integration, witness, counter, recording, fixtures, routes) as coverage:
                    code = base.main(['--plan', plan['qualification']['path'],
                        '--plan-sha256', plan['qualification']['sha256'], '--out', str(args.out / 'base')])
                result['coverage'] = coverage
                inner = read(args.out / 'base/complete.json')
                result['base_result'] = dict(path='base/complete.json', sha256=sha(args.out / 'base/complete.json'),
                    returncode=code, status=inner['status'])
                need(code == 0 and inner['status'] == 'PASS' and inner['source_model_pins_rechecked'] is True,
                     'original full-logit/state/ahead replay did not pass')
                count = 8 if plan['concurrency'] == 1 else 16
                need(inner['total_requests'] == count and len(inner['cases']) == 1
                     and inner['cases'][0]['name'] == f"HTTP-C{plan['concurrency']}"
                     and len(inner['cases'][0]['completed']) == count, 'complete one-C actual HTTP request set')
                need(coverage['installs'] == 1 and coverage['restores'] == 2, 'fresh pair installation/release')
                result['route_comparison'] = routes.finish(plan['inputs']['route_events']['sha256'])
                result.update(status='PASS', replay_numerical_checks_passed=True, total_requests=count)
            result['replay_route_events'] = dict(path='replay-route-events.jsonl',
                sha256=sha(args.out / 'replay-route-events.jsonl'))
        same(validate(args.plan, args.plan_sha256), (plan, numeric, spec, header), 'post-work source/input binding')
        result['input_source_pins_rechecked'] = True
    except BaseException as error:
        result.update(status='FAIL', replay_numerical_checks_passed=False,
            error=dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc()))
    finally:
        write(args.out / 'complete.json', result)
    print(json.dumps(dict(status=result['status'], out=str(args.out), independent_audit_required=True)), flush=True)
    return int(result['status'] not in ('PASS', 'PREPARED'))


if __name__ == '__main__':
    raise SystemExit(main())
