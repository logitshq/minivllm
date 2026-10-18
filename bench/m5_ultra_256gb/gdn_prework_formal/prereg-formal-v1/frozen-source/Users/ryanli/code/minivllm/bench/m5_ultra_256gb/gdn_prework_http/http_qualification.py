"""Prepared GDN-only HTTP replay adapter; no action without an explicit mode.

--prepare-only joins source-pinned actual HTTP data with the CPU tokenizer.
--run additionally invokes the original two-model numerical qualifier, installed
and observed per instance, AFTER an external owner has reaped the capture server.
This module never starts/stops a service or a subprocess. Outer resource/runtime
admission, hard timeout/reap and independent numerical/dispatch audit remain required.

Producer PASS means completed checks for this recorded schedule, not independent
HTTP qualification, future-arrival correctness or a performance point. The real
37-request serving audit is a separate pinned prerequisite, not rerun
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
GDN = HERE.parent / 'gdn_prework_unbound'
BASE = NATIVE / 'qualify_native.py'
BASE_PLAN_SHA = '21ba52e43ce54c2b463ab46bfd212c53d06d6303530c77939e3e6b801df547fb'
SERVING_PLAN_SHA = 'dfb6d3d306e6247e34dfdfe54746e12205b297e166e1f175e85024786d73163b'
SERVING_AUDIT_SHA = 'fee50c90c30c3224934489c66e5ecb4a5d7f0c4fb57ae0baa0576f6efe9a769b'
FIXED = {
    NATIVE / 'qualify_native.py': '26964fcbed9a2e5dcefaee4888244987897f3138d4a2b7add59bf63a7461b7bf',
    NATIVE / 'http_qualification.py': '471bebd8a114d3a7f5a190c20909f6896f17fd39f282656546c1366c3fa2c6e2',
    NATIVE / 'http_join.py': 'a362c321c2329d7ab53ab6a1a5a96981bb95b44733739fdbb56dd3d38bef7b4e',
    NATIVE / 'http_replay.py': '1304c58c66e56b3aeaa582ef9f7d61317280cf3107f8f600485426119970fb65',
    NATIVE / 'http_capture_v2.py': '536fe41d6c8e2bdffe83ba6f43fba87dd6265deb4c84f6204bc9c7220f8280e3',
    NATIVE / 'native_server.py': '9bcb37eff8bdecd79800afdedc82deacf9f7a0a9a7a1e0d62dc8ba49e70fb68f',
    NATIVE / 'client.py': 'ef64a114f4f7d90ec5044d6bdcf058ca05f98dea445d6639db706a4fc75217dc',
    NATIVE / 'native_cleanup.py': '6c292a9d3cb28c047d1dc92ac0fec4abe19fa8df84c172ef61d2f6ce41157ca6',
    GDN / 'candidate.py': '1bf5a837c6515ca505bc5d1beaa1718b88e1ccd52609f6814bb995afbe6c1260',
    GDN / 'serving.py': '3f493752d95996bd08cc1ea7389fc3159ab615e1ac162b36332eecc47dab862c',
    GDN / 'qualify.py': 'f3088a98d2aca6b4f267e7d4c258efbbf9f83e1beccb6745c88ae42343f88e89',
    GDN / 'audit.py': 'a3e97dc368da47735d7c8e9fd2bcb40d7decaccb9c5926084d53cffb10495d3d',
    GDN / 'audit_serving.py': '268cc336417a3d31dffc30c749701d132da610fbb41d9b0f4c85ad6adaa2f2bc',
    HERE / 'http_server.py': 'e4408d9df0c26a2f7fc5db93312902e4797b72a8958f67f1912b4b9bd628e3d4',
    HERE / 'http_capture.py': '0ecedc988d2b77a55f010bdd09e9f9581e4aa9618d7070a6eb4b81960e5849c8',
}
CAPTURE_NAMES = dict(spec='spec.json', complete='complete.json', child='child.json',
    release='release.json', absence='child-absence.json', port='port-after.json')
INPUT_NAMES = dict(group='group.json', trace='trace.jsonl', dispatch_events='dispatch-events.jsonl')


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
    need(set(binding) == {'plan','audit','complete'}, 'actual37 plan/audit/complete required')
    plan,audit,complete = [read(pinned(binding[k])) for k in ('plan','audit','complete')]
    need(binding['plan']['sha256'] == SERVING_PLAN_SHA and binding['audit']['sha256'] == SERVING_AUDIT_SHA,
         'immutable actual GDN37 prerequisite')
    child = Path(audit['child_directory'])
    need(Path(binding['plan']['path']) == child/'plan.json' and Path(binding['complete']['path']) == child/'complete.json',
         'actual same GDN37 child plan/terminal')
    need(plan['schema'] == 'GDN-prework-serving-plan-v1' and plan['qualification']['sha256'] == BASE_PLAN_SHA
         and plan['qualifier_sha256'] == FIXED[GDN/'serving.py'] and plan['deadline_s'] == 1800,'actual37 source/model plan')
    need(audit['schema'] == 'GDN-prework-serving-independent-v1' and audit['status'] == 'PASS'
         and audit['qualified'] is audit['same_schedule_qualified'] is True
         and audit['plan_sha256'] == SERVING_PLAN_SHA and audit['base_plan_sha256'] == BASE_PLAN_SHA
         and audit['audit_source_sha256'] == FIXED[GDN/'audit_serving.py'],'actual independent GDN37 PASS')
    need(complete['schema'] == 'GDN-prework-serving-result-v1' and complete['status'] == 'PASS'
         and complete['same_schedule_checks_passed'] is complete['source_pins_rechecked'] is True
         and complete['plan_sha256'] == SERVING_PLAN_SHA and complete['total_requests'] == 37,'actual37 producer terminal')
    same(audit['terminal_sha256']['plan.json'],binding['plan']['sha256'],'audited37 plan bytes')
    same(audit['terminal_sha256']['complete.json'],binding['complete']['sha256'],'audited37 terminal bytes')
    same(plan['sources'],audit['source_pins'],'audited37 source map')
    same(complete['sources'],plan['sources'],'37 producer sources')
    for p,h in plan['sources'].items():
        need(Path(p) in FIXED and FIXED[Path(p)] == h and sha(p) == h,'unchanged prerequisite source')
    same(complete['coverage'],audit['coverage'],'actual37 dispatch coverage')
    need([len(c['completed']) for c in audit['ledger']['cases']] == [8,16,13]
         and audit['coverage']['layer_installs'] == audit['coverage']['layer_restores'] == 96
         and audit['coverage']['route_graph_enabled'] is audit['coverage']['identity_refill_enabled'] is False,
         'original37/two-pair/only-GDN scope')
    for key in ('http_qualified','performance_qualified','combined_candidates_qualified','parameter_replacement_qualified'):
        need(audit[key] is False,'no transferred HTTP/performance/composition claim')
    return dict(plan_sha256=SERVING_PLAN_SHA,audit_sha256=SERVING_AUDIT_SHA,
                scope='Separate actual original37 GDN-only same-schedule prerequisite; not this HTTP history.')


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
    capture = load(HERE / 'http_capture.py', '_gdn_replay_capture_contract')
    capture.validate(spec)
    need(spec['concurrency'] == plan['concurrency'] and spec['base_plan'] == plan['qualification'],
         'capture concurrency/base plan')
    need(complete['schema'] == 'GDN-prework-http-capture-result-v1' and complete['status'] == 'PASS'
         and all(complete.get(k) is True for k in ('capture_complete', 'server_reaped', 'cleanup_complete',
             'cleanup_task_resolved', 'client_task_resolved', 'small_pins_rechecked', 'dispatch_events_retained'))
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
    header = dict(event='dispatch-startup', schema=capture.DISPATCH_SCHEMA, pid=pid,
        phase=f"HTTP-C{plan['concurrency']}", identity=spec['identity'])
    with inputs['dispatch_events'].open('rb') as stream:
        raw = stream.readline()
        need(raw.endswith(b'\n'), 'complete dispatch startup line required')
        same(parse(raw), header, 'dispatch startup PID/source/phase binding')
    return spec, header


def validate(path, expected_sha):
    need(sha(path) == expected_sha, 'new HTTP replay plan SHA')
    plan = read(path)
    need(set(plan) == {'schema', 'qualifier_sha256', 'qualification', 'serving', 'concurrency',
        'capture', 'inputs', 'sources'} and plan['schema'] == 'GDN-recorded-http-plan-v1',
        'exact GDN replay plan required')
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
    q = load(NATIVE / 'http_qualification.py', '_gdn_replay_original_jobs')
    join = load(NATIVE / 'http_join.py', '_gdn_replay_original_join')
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


class CapturedDispatch:
    """Match actual call-order/context/counters, without evaluating HTTP tensors."""
    def __init__(self,stream,header,output,captured_ref,phase):
        self.stream,self.output,self.phase = stream,output,phase
        self.digest=hashlib.sha256();self.bytes=self.lines=0
        self.counts=dict(candidate=0,fallback=0,b1_calls=0,multirow_calls=0,prefill_calls=0,short_bypass_calls=0)
        same(self.take(),header,'captured dispatch startup changed')
        self.output.write(json.dumps(dict(event='replay-dispatch-startup',schema='GDN-prework-http-replay-events-v1',
            pid=os.getpid(),phase=phase,captured_dispatch_events=captured_ref),allow_nan=False)+'\n');self.output.flush()
    def take(self):
        raw=self.stream.readline();need(raw and raw.endswith(b'\n'),'missing/truncated captured dispatch')
        self.digest.update(raw);self.bytes+=len(raw);self.lines+=1;return parse(raw)
    def emit(self,actual):
        # Original semantic key remains in the untouched base ledger. No copied
        # recorded token participates in either engine's model inputs.
        need(actual['case'] == self.phase.replace('HTTP-','D-'),'selected actual observer case')
        row=dict(event='dispatch-call',status=actual['status'],phase=self.phase,
                 **{k:actual[k] for k in ('step','mode','B','used')})
        for key in ('path','candidate','fallback'):
            if key in actual:row[key]=actual[key]
        self.output.write(json.dumps(row,allow_nan=False)+'\n');self.output.flush()
        same(row,self.take(),'actual captured/replayed GDN dispatch differs')
        need(row['status']=='SUBMITTED' and type(row['step']) is int and row['step']>0,'completed dispatch only')
        for key in ('candidate','fallback'):self.counts[key]+=row[key]
        field={'B1-candidate':'b1_calls','B>1-fallback':'multirow_calls',
               'prefill-untouched':'prefill_calls','short-packed-bypass':'short_bypass_calls'}[row['path']]
        self.counts[field]+=1
    def finish(self,captured_sha):
        need(self.stream.read(1)==b'' and self.digest.hexdigest()==captured_sha,'extra/changed captured dispatch')
        need(sum(self.counts[k] for k in ('b1_calls','multirow_calls','prefill_calls','short_bypass_calls'))>0,'no actual calls')
        if self.phase=='HTTP-C1':need(self.counts['b1_calls']>0 and self.counts['multirow_calls']==0,'actual C1 candidate coverage')
        else:need(self.counts['multirow_calls']>0,'actual C8 multirow fallback coverage')
        return dict(counts=self.counts,captured_sha256=self.digest.hexdigest(),bytes=self.bytes,lines=self.lines,
            scope='Ordered actual submissions, including ahead; full numerical gate separate, no performance claim.')


@contextmanager
def adapted(base,replay,serving,adapter,recording,fixtures,dispatch):
    # Crucial order: HTTP selection first, then serving.adapt's case/step wrapper.
    # Do NOT invoke frozen run_adapter inside serving.adapt: it overwrites that wrapper.
    saved=base.workloads,base.run_case
    case=dict(name=f"D-C{recording['concurrency']}",concurrency=recording['concurrency'])
    def workloads(actual):
        same(actual,fixtures,'same original fixtures');return [case]
    def run_case(actual,engines,observers,sampling,runner_module,cmp,ledger,deadline):
        same(actual,case,'one actual HTTP C only')
        return replay.run_recorded_case(recording,engines,observers,sampling,runner_module,cmp,ledger,deadline,base=base)
    base.workloads,base.run_case=workloads,run_case
    try:
        with serving.adapt(base,adapter,dispatch.emit,enabled=True) as coverage:yield coverage
    finally:base.workloads,base.run_case=saved


def require_coverage(coverage,C,counts):
    need(coverage['enabled'] is True and coverage['layer_installs']==coverage['layer_restores']==48
         and coverage['observer_restores']==2 and len(coverage['final_counters'])==1
         and coverage['identity_refill_enabled'] is coverage['route_graph_enabled'] is False,
         'one fresh independent pair;48 GDN installs/restores;both observers restored')
    same(coverage['per_case'],{f'D-C{C}':counts},'all actual one-C dispatch accounted')
    need(coverage['calls']==sum(counts[k] for k in ('b1_calls','multirow_calls','prefill_calls','short_bypass_calls')),
         'all actual calls accounted')
    need(counts['candidate']==48*counts['b1_calls'] and counts['fallback']==48*counts['multirow_calls'],'all48 layers')
    same(coverage['final_counters'],[[dict(candidate=counts['b1_calls'],fallback=counts['multirow_calls']) for _ in range(48)]],
         'no unobserved layer invocations')


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
    result = dict(schema='GDN-recorded-http-result-v1', status='INCOMPLETE',
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
            base = load(BASE, '_gdn_http_private_base')
            replay = load(NATIVE / 'http_replay.py', '_gdn_http_private_replay')
            adapter = load(GDN/'candidate.py', '_gdn_http_private_candidate')
            serving = load(GDN/'serving.py', '_gdn_http_private_serving')
            with pinned(plan['inputs']['dispatch_events']).open('rb') as captured, \
                 (args.out / 'replay-dispatch-events.jsonl').open('x') as emitted:
                dispatch = CapturedDispatch(captured, header, emitted, plan['inputs']['dispatch_events'],
                    f"HTTP-C{plan['concurrency']}")
                with adapted(base, replay, serving, adapter, recording, fixtures, dispatch) as coverage:
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
                result['dispatch_comparison'] = dispatch.finish(plan['inputs']['dispatch_events']['sha256'])
                require_coverage(coverage,plan['concurrency'],result['dispatch_comparison']['counts'])
                result.update(status='PASS', replay_numerical_checks_passed=True, total_requests=count)
            result['replay_dispatch_events'] = dict(path='replay-dispatch-events.jsonl',
                sha256=sha(args.out / 'replay-dispatch-events.jsonl'))
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
