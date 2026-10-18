"""Combined actual-HTTP replay; default inert, external cold owner required.

Use unchanged full-logit/state/ahead replay mathematics and captured admissions.
Recorded candidate tokens are comparison evidence, never model inputs. The
separate combined42/37 PASS is mandatory; its direct prelude is not rerun here.
TraceGraph records Python tracing versus real compiled callable reuse, not GPU
kernel counts or 48 Python calls per submission. Producer checks never grant
independent HTTP/performance or resource/restoration qualification.
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
ROUTE = HERE.parent / 'route_decode_graph'
OLD = HERE.parent / 'long_decode_graph'
BASE = NATIVE / 'qualify_native.py'
HELPER = ROUTE / 'http_qualification_v2.py'
HELPER_SHA = 'd3e2a4f76b7c384f6c6c986d0838b68227618502b0552e5cdf74f149029e39e7'
BASE_PLAN_SHA = '21ba52e43ce54c2b463ab46bfd212c53d06d6303530c77939e3e6b801df547fb'


def module(path, digest, name):
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
        raise ValueError('fixed HTTP helper changed: ' + str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec); spec.loader.exec_module(loaded); return loaded


donor = module(HELPER, HELPER_SHA, '_combined_frozen_http_replay')
need, same, sha, read, parse, write, pinned = donor.need, donor.same, donor.sha, donor.read, donor.parse, donor.write, donor.pinned
CAPTURE_NAMES, INPUT_NAMES = donor.CAPTURE_NAMES, donor.INPUT_NAMES
prepare = donor.prepare  # CPU tokenizer + original full-prompt/SSE/schedule join.
FIXED = {**donor.FIXED, HELPER: HELPER_SHA,
    HERE/'qualify.py': 'f131f3da91e847cf4f00e0644120919c49050227f5a35679b7a446aee8eaa6ee',
    HERE/'integration.py': '24f89e06f7e6d01041e6daaead40d1bacd927032a68218f8aa5a495bc092b851',
    HERE/'audit.py': '7e7718ddf1e1b6d744ec374e19b5513248f0374306503aeb8908391edb403270',
    HERE/'http_server.py': '287f34561c83437823923cf45b67f0aa6811f7eba911ee375f62519ba508ef64',
    HERE/'http_capture.py': '9b98abc3d8f383dba9bd25eafade73cea62e198abde0d734a1e0d2b8ed3a58f2',
    HERE.parent/'gdn_prework_unbound/candidate.py': '1bf5a837c6515ca505bc5d1beaa1718b88e1ccd52609f6814bb995afbe6c1260',
}


def source_pins():
    need(all(sha(p) == h for p,h in FIXED.items()), 'complete adapter/dependency pins')
    return {**{str(p): h for p,h in FIXED.items()}, str(Path(__file__).resolve()): sha(__file__)}

def load(path, name):
    need(path in FIXED, 'unregistered helper'); return module(path, FIXED[path], name)


def serving_gate(binding):
    need(set(binding) == {'plan','audit','complete'}, 'combined actual42/37 three references required')
    plan, audit, complete = [read(pinned(binding[k])) for k in ('plan','audit','complete')]
    child = Path(audit['child_directory'])
    need(Path(binding['plan']['path']) == child/'plan.json'
         and Path(binding['complete']['path']) == child/'complete.json', 'same actual combined prerequisite child')
    need(plan['schema'] == 'GDN-graph-combined-serving-plan-v1'
         and plan['qualifier_sha256'] == FIXED[HERE/'qualify.py']
         and plan['qualification']['sha256'] == BASE_PLAN_SHA and plan['deadline_s'] == 1800,
         'combined42/37 source/model plan')
    need(audit['schema'] == 'GDN-graph-combined-independent-v1' and audit['status'] == 'PASS'
         and audit['qualified'] is audit['same_schedule_qualified'] is True
         and audit['audit_source_sha256'] == FIXED[HERE/'audit.py']
         and audit['plan_sha256'] == binding['plan']['sha256']
         and audit['base_plan_sha256'] == BASE_PLAN_SHA, 'independent combined42/37 PASS')
    need(complete['schema'] == 'GDN-graph-combined-serving-result-v1' and complete['status'] == 'PASS'
         and complete['same_schedule_checks_passed'] is complete['source_pins_rechecked'] is True
         and complete['plan_sha256'] == binding['plan']['sha256']
         and complete['total_requests'] == 37 and complete['parameter_direct_calls'] == 42,
         'complete same combined producer terminal')
    for name,key in (('plan.json','plan'),('complete.json','complete')):
        same(audit['terminal_sha256'][name],binding[key]['sha256'],'audited prerequisite bytes')
    same(plan['sources'],audit['source_pins'],'audited combined source map')
    same(plan['sources'],complete['sources'],'producer source map')
    need(all(Path(p) in FIXED and FIXED[Path(p)] == h and sha(p) == h for p,h in plan['sources'].items()),
         'unchanged combined prerequisite implementation')
    same(complete['coverage'],audit['coverage'],'audited combined coverage')
    need([len(c['completed']) for c in audit['ledger']['cases']] == [8,16,13]
         and audit['coverage']['route']['parameter_direct_calls'] == 42
         and audit['coverage']['route']['parameter_checks_complete'] is True
         and audit['coverage']['combined']['layer_installs'] == audit['coverage']['combined']['layer_restores'] == 96,
         'full42/37 prerequisite, not a successful prefix')
    for result in (audit, complete):
        need(result['http_qualified'] is result['performance_qualified'] is False, 'separate numerical prerequisite only')
    return dict(plan_sha256=binding['plan']['sha256'],audit_sha256=binding['audit']['sha256'],
        scope='Actual combined42/37 prerequisite only; this HTTP schedule needs independent audit.')


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
    capture = load(HERE / 'http_capture.py', '_combined_replay_capture_contract')
    capture.validate(spec)
    need(spec['concurrency'] == plan['concurrency'] and spec['base_plan'] == plan['qualification'],
         'capture concurrency/base plan')
    need(complete['schema'] == 'GDN-graph-combined-http-capture-result-v1' and complete['status'] == 'PASS'
         and all(complete.get(k) is True for k in ('capture_complete', 'server_reaped', 'cleanup_complete',
             'cleanup_task_resolved', 'client_task_resolved', 'small_pins_rechecked', 'route_events_retained'))
         and complete['errors'] == [] and complete['numerical_qualified'] is False
         and complete['performance_qualified'] is False, 'complete observed capture with closed cleanup required')
    same(complete['lifecycle_source'],dict(path=str(capture.DONOR),sha256=capture.DONOR_SHA),'frozen lifecycle source')
    same(complete['capture_adapter_source'],dict(path=str(HERE/'http_capture.py'),sha256=FIXED[HERE/'http_capture.py']),'actual capture source')
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
        'capture', 'inputs', 'sources'} and plan['schema'] == 'GDN-graph-combined-recorded-http-plan-v1',
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


class CapturedRoutes(donor.CapturedRoutes):
    def __init__(self, stream, header, replay_file, captured_ref, phase):
        self.stream,self.output,self.phase = stream,replay_file,phase
        self.calls=self.hits=self.fallbacks=0; self.bounds={'1023':0,'1536':0}
        self.digest=hashlib.sha256();self.bytes=self.lines=0
        self.pending=None;self.trace_counts={'compiled-tracing':0,'compiled-reuse':0,'eager-fallback':0}
        same(self.take(),header,'captured combined startup changed')
        self.output.write(json.dumps(dict(event='replay-route-startup',
            schema='GDN-graph-combined-http-replay-events-v1',pid=os.getpid(),phase=phase,
            captured_route_events=captured_ref),allow_nan=False)+'\n');self.output.flush()

    def emit(self,row):
        if row['event']=='combined-graph-call':
            self.output.write(json.dumps(row,allow_nan=False)+'\n');self.output.flush()
            same(row,self.take(),'actual captured/replayed combined trace differs')
            need(self.pending is None and row['status']=='SUBMITTED' and row['phase']==self.phase
                 and row['installation']==0 and row['direct_label'] is None
                 and type(row['step']) is int and row['step']>0,'one combined submission before its route row')
            need(row['kind'] in self.trace_counts,'known tracing/reuse route')
            self.trace_counts[row['kind']]+=1;self.pending=row
        else:
            need(self.pending is not None,'route without preceding actual combined submission')
            for key in ('phase','step','direct_label','used','actual_invocations'):
                same(row[key],self.pending[key],'route/combined context consistency')
            same(row['B'],self.pending['B'],'same batch')
            super().emit(row);self.pending=None

    def finish(self,captured_sha):
        need(self.pending is None,'missing trailing original route witness')
        result=super().finish(captured_sha)
        need(self.trace_counts['compiled-tracing']>0,'actual tracing before compiled reuse')
        return dict(**result,combined_kinds=self.trace_counts,
            tracing_scope='Python tracing/fallback counts; compiled reuse does not execute48 Python hooks')


@contextmanager
def adapted(base,replay,integration,witness_module,counter,recording,fixtures,routes):
    combined=load(HERE/'qualify.py','_combined_http_private_trace')
    stats=dict(installs=0,restores=0,layer_installs=0,layer_restores=0,
        **{'compiled-tracing':0,'compiled-reuse':0,'eager-fallback':0},traced_batches=[])
    class Bridge:
        engine=None
        def install(self,runner,*,enabled):
            need(enabled is True and self.engine.runner is runner,'actual fresh candidate engine')
            scope=integration.install(self.engine,gdn_prework=True,route_graph=True)
            try:
                need(scope is not None and len(scope.gdn_handles)==48,'both installations before Observer')
                handle=combined.Handle(scope,witness_module,routes.emit,stats)
            except BaseException:
                if scope is not None:scope.restore()
                raise
            stats['installs']+=1;stats['layer_installs']+=48
            return handle
    bridge=Bridge()
    # Use HTTP adapted directly: unlike serving.adapt it never injects a42-call
    # prelude or replaces the selected captured-history run_case later.
    with donor.adapted(base,replay,bridge,witness_module,counter,recording,fixtures,routes) as coverage:
        inherited=base.Observer
        class Observer(inherited):
            def __init__(self,engine,cmp,label):
                need(bridge.engine is None,'serial observer construction')
                bridge.engine=engine
                try:
                    super().__init__(engine,cmp,label)
                    if self.route_install is not None:self.route_install.proxy.witness=self.route_witness
                finally:bridge.engine=None
        base.Observer=Observer
        coverage['combined']=stats
        try:yield coverage
        finally:base.Observer=inherited


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
    result = dict(schema='GDN-graph-combined-recorded-http-result-v1', status='INCOMPLETE',
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
            base = load(BASE, '_combined_route_http_private_base')
            replay = load(NATIVE / 'http_replay.py', '_combined_route_http_private_replay')
            integration = load(HERE / 'integration.py', '_combined_route_http_private_install')
            witness = load(ROUTE / 'serving_qualification.py', '_combined_route_http_private_witness')
            counter = load(OLD / 'qualify.py', '_combined_route_http_private_counter')
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
                stats=coverage['combined']
                need(stats['installs']==stats['restores']==1 and stats['layer_installs']==stats['layer_restores']==48,
                     'combined scope and48 layers restored')
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
