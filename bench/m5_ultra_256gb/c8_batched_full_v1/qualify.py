"""Default-OFF full original E047 history with the frozen E050 composition bridge."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import traceback

HERE = Path(__file__).resolve().parent; B = HERE.parent
PREFIX = B/'c8_batched_prefix_v1/prefix.py'
PREFIX_SHA = 'd16318dbad1181caac964ec5b331b4b208ca92c0ccd66fe87bbb01a54d4cd12e'
BRIDGE = B/'c8_batched_prefix_v1/bridge.py'
BRIDGE_SHA = 'ee212bee9b0bf9e82d62c762b21dc630294e588538342b70df0f19fbf940bd07'
SCOPE = ('Complete original E047 1030-step/16-request recorded-history numerical replay only; '
         'no new HTTP capture, arbitrary-arrival guarantee, performance or Pareto qualification.')
SCHEMA = 'c8-batched-gdn-graph-full-result-v1'


def need(ok, message):
    if not ok: raise ValueError(message)
def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(1 << 20): h.update(block)
    return h.hexdigest()
def read(path): return json.loads(Path(path).read_bytes())
def write(path, value): Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
def ref(path): return dict(path=str(Path(path).resolve()), sha256=sha(path))
def pinned(item):
    path = Path(item['path'])
    need(set(item) == {'path', 'sha256'} and path.is_absolute() and path.resolve() == path
         and path.is_file() and not path.is_symlink() and sha(path) == item['sha256'], 'pinned input changed')
    return path

def load(path, digest, name):
    need(sha(path) == digest, 'frozen source changed: '+str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def composition_ok(value):
    need(value['installs'] == value['restores'] == 1 and value['handle_count'] == 48
         and len(value['records']) == 1, 'one actual 48-layer composition lifetime')
    record = value['records'][0]; layers = record['layers']
    need(record['status'] == 'RESTORED' and record['handles_closed'] == 48
         and record['native_decode_restored'] is True and record['caches_cleared'] is True
         and len(layers) == len({r['layer'] for r in layers}) == 48, 'composition cleanup proof')
    need(all(type(r['final_counts']['candidate']) is int and r['final_counts']['candidate'] > 0
             and type(r['final_counts']['fallback']) is int and r['final_counts']['fallback'] == 0
             for r in layers), 'actual per-layer aggregate candidate branches; fallback forbidden')


def predecessor_ok(top, plan_sha):
    need(top['schema'] == 'c8-batched-gdn-graph-prefix-result-v1'
         and top['status'] == 'DIAGNOSTIC_PREFIX_COMPLETE' and top['plan_sha256'] == plan_sha
         and top['source_input_pins_rechecked'] is True, 'actual successful E050 prefix required')
    need(top['completion_state'] == 'INCOMPLETE_BY_DESIGN' and top['performance_qualified'] is False
         and top['full_history_qualified'] is False and top['base_result']['status'] == 'PASS'
         and top['base_result']['returncode'] == 0 and top['inherited_result']['returncode'] == 0
         and top['inherited_result']['status'] == 'DIAGNOSTIC_PREFIX_COMPLETE', 'E050 original numeric checks and scope')
    need(top['prefix']['steps'] == 49 and top['prefix']['completed'] == []
         and top['coverage']['installs'] == 1 and top['coverage']['restores'] == 2, 'actual E050 warm49/release')
    composition_ok(top['composition'])


def recording_ok(recording):
    need(recording['schema'] == 'E023-recorded-http-v1' and recording['concurrency'] == 8
         and type(recording['steps']) is list and len(recording['steps']) == 1030
         and len(recording['jobs']) == 16, 'full original 1030-step/16-job recording required')
    need(all(s['step'] == i for i, s in enumerate(recording['steps'], 1))
         and all(job['cap'] == 512 and len(job['raw_ids']) == len(job['committed_ids']) == 512
                 for job in recording['jobs'].values()), 'original full progress and caps/output targets')


def prepare(prefix_plan, predecessor):
    prefix_plan, predecessor = Path(prefix_plan).resolve(), Path(predecessor).resolve()
    prior = load(PREFIX, PREFIX_SHA, '_full_prior_prefix_contract')
    old = read(prefix_plan); prior.validate(old)
    top = read(predecessor); predecessor_ok(top, sha(prefix_plan))
    directory = predecessor.parent
    need(sha(directory/'plan.json') == sha(prefix_plan), 'actual E050 plan binding')
    inherited = pinned({k: top['inherited_result'][k] for k in ('path', 'sha256')})
    need(inherited == directory/'inherited/complete.json', 'original inherited result path')
    inner = read(inherited)
    need(inner['status'] == 'DIAGNOSTIC_PREFIX_COMPLETE' and inner['prefix'] == top['prefix']
         and top['base_result'] == inner['base_result']
         and inner['base_result']['sha256'] == sha(directory/'inherited/base/complete.json'),
         'E050 inherited/base result binding')
    events = pinned(top['composition_events'])
    need(events == directory/'composition.jsonl', 'actual composition events path')
    original = read(pinned(old['base_prefix_plan']))
    recording = read(pinned(original['recording'])); recording_ok(recording)
    numeric = read(pinned(original['qualification']))
    need(numeric['configuration']['cache_limit_gib'] == 2, 'unchanged original 2 GiB allocator cache')
    sources = {**old['sources'], str(PREFIX): PREFIX_SHA, str(BRIDGE): BRIDGE_SHA,
               str(Path(__file__).resolve()): sha(__file__)}
    need(all(sha(path) == digest for path, digest in sources.items()), 'fixed full source closure')
    inputs = [directory/n for n in ('plan.json', 'inherited/complete.json', 'inherited/base/complete.json',
        'inherited/base/steps.jsonl', 'inherited/replay-route-events.jsonl', 'composition.jsonl')]
    capture = Path(original['capture_directory'])
    return dict(schema='c8-batched-gdn-graph-full-plan-v1', sources=sources,
        prefix_plan=ref(prefix_plan), predecessor=ref(predecessor),
        predecessor_inputs={str(p): sha(p) for p in inputs},
        qualification=original['qualification'], recording=original['recording'],
        capture_directory=str(capture), route_events=ref(capture/'route-events.jsonl'),
        switches=dict(gdn_prework=True, route_graph=True), environment=original['environment'],
        allocator_cache_policy='unchanged-2gib', expected_steps=1030, expected_requests=16,
        expected_route_calls=1028, full_history_qualified=False, http_qualified=False,
        performance_qualified=False, scope=SCOPE)


def validate(plan):
    expected = prepare(pinned(plan['prefix_plan']), pinned(plan['predecessor']))
    need(json.dumps(plan, sort_keys=True) == json.dumps(expected, sort_keys=True), 'changed full replay plan')


def run_replay(q, base, replay, bridge, witness, counter, recording, fixtures, routes, qualification, out):
    """Use the original adapted.run_case verbatim: no prefix iterator or final-drain substitution."""
    bridge.integration_module(q)
    try:
        with q.adapted(base, replay, bridge, witness, counter, recording, fixtures, routes) as coverage:
            code = base.main(['--plan', qualification['path'], '--plan-sha256', qualification['sha256'],
                              '--out', str(out)])
        return code, coverage
    finally:
        bridge.close_private_hook()


def full_result_ok(inner, recording, coverage):
    need(inner['status'] == 'PASS' and inner['source_model_pins_rechecked'] is True
         and inner['total_requests'] == 16 and len(inner['cases']) == 1, 'all original numeric checks and 16 completions')
    case = inner['cases'][0]
    need(case['status'] == 'PASS' and case['name'] == 'HTTP-C8' and case['steps'] == 1030
         and len(case['completed']) == 16 and len({r['rid'] for r in case['completed']}) == 16,
         'complete recorded history, never a prefix')
    expected = {rid: dict(rid=rid, raw_ids=j['raw_ids'], committed_ids=j['committed_ids'])
                for rid, j in recording['jobs'].items()}
    need({r['rid']: r for r in case['completed']} == expected
         and case['raw_output_count'] == case['committed_output_count'] == 8192
         and case['decode_batch_histogram'] == recording['histogram'], 'complete unchanged output and schedule targets')
    need(coverage['installs'] == 1 and coverage['restores'] == 2, 'one fresh pair fully released')
    return case


def run(plan, out):
    validate(plan)
    need({k: v for k, v in os.environ.items() if k.startswith('MLX_')} == plan['environment'], 'exact registered MLX environment')
    out = Path(out).resolve(); out.mkdir(parents=True, exist_ok=False); write(out/'plan.json', plan)
    result = dict(schema=SCHEMA, status='INCOMPLETE', pid=os.getpid(), ppid=os.getppid(),
        plan_sha256=sha(out/'plan.json'), full_history_qualified=False, http_qualified=False,
        performance_qualified=False, scope=SCOPE)
    write(out/'complete.json', result); bridge = None
    try:
        prior = load(PREFIX, PREFIX_SHA, '_full_prior_prefix_execution')
        old = prior.load(prior.BASE, prior.BASE_SHA, '_full_original_prefix_helpers')
        q = old.load(old.ROUTE, old.ROUTE_SHA, '_full_private_route')
        base = q.load(q.BASE, '_full_private_base'); replay = q.load(q.NATIVE/'http_replay.py', '_full_private_replay')
        combined = prior.load(prior.COMBINED, prior.COMBINED_SHA, '_full_combined')
        module = load(BRIDGE, BRIDGE_SHA, '_full_frozen_bridge')
        witness = q.load(q.HERE/'serving_qualification.py', '_full_route_witness')
        counter = q.load(q.OLD/'qualify.py', '_full_route_counter')
        recording = read(pinned(plan['recording'])); numeric = read(pinned(plan['qualification']))
        fixtures = read(pinned(numeric['fixtures']))['fixtures']
        capture = Path(plan['capture_directory']); spec = read(capture/'spec.json')
        header = dict(event='route-startup', schema='E024-route-http-events-v1',
            pid=read(capture/'release.json')['pid'], phase='HTTP-C8', identity=spec['identity'])
        with pinned(plan['route_events']).open('rb') as captured, \
             (out/'replay-route-events.jsonl').open('x') as emitted, (out/'composition.jsonl').open('x') as events:
            routes = q.CapturedRoutes(captured, header, emitted, plan['route_events'], 'HTTP-C8')
            def emit(event):
                events.write(json.dumps(event, sort_keys=True, allow_nan=False)+'\n'); events.flush()
            bridge = module.Bridge(combined, emit)
            code, coverage = run_replay(q, base, replay, bridge, witness, counter, recording, fixtures,
                                       routes, plan['qualification'], out/'base')
            inner_path = out/'base/complete.json'; inner = read(inner_path)
            result.update(base_result=dict(path=str(inner_path), sha256=sha(inner_path), returncode=code,
                status=inner['status']), coverage=coverage, peak_mlx_bytes=inner.get('peak_bytes'),
                elapsed_base_seconds=inner.get('elapsed_wall_s'))
            need(code == 0, 'original full replay failed; inspect base failure/ledger')
            result['case'] = full_result_ok(inner, recording, coverage)
            result['composition'] = bridge.checked()
            comparison = routes.finish(plan['route_events']['sha256'])
            need(comparison['calls'] == comparison['actual_compiled_invocations'] == 1028
                 and comparison['fallbacks'] == 0, 'complete actual graph stream and zero route fallback')
            result.update(route_comparison=comparison, steps=1030, total_requests=16)
        validate(plan)
        result.update(status='PASS', source_input_pins_rechecked=True, full_history_qualified=True,
                      qualification_scope='Only the exact registered E047 recording; no new candidate HTTP capture.')
    except BaseException as exc:
        result.update(status='FAIL', error=dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc()))
    finally:
        if bridge is not None: result['composition'] = bridge.snapshot()
        for name, key in (('composition.jsonl', 'composition_events'), ('replay-route-events.jsonl', 'replay_route_events')):
            if (out/name).is_file(): result[key] = ref(out/name)
        write(out/'complete.json', result)
    print(json.dumps(dict(status=result['status'], out=str(out))))
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); modes = p.add_mutually_exclusive_group()
    modes.add_argument('--prepare', action='store_true'); modes.add_argument('--run', action='store_true')
    p.add_argument('--prefix-plan', type=Path); p.add_argument('--predecessor', type=Path)
    p.add_argument('--plan', type=Path); p.add_argument('--plan-sha256'); p.add_argument('--out', type=Path); a = p.parse_args()
    if a.prepare:
        need(a.prefix_plan and a.predecessor and a.out, 'actual E050 plan, successful prefix, fresh output required')
        plan = prepare(a.prefix_plan, a.predecessor); a.out.mkdir(parents=True, exist_ok=False); write(a.out/'plan.json', plan)
        print(json.dumps(dict(status='PREPARED_NOT_RUN', plan=str(a.out/'plan.json'), sha256=sha(a.out/'plan.json'))))
    elif a.run:
        need(a.plan and a.plan_sha256 and a.out and sha(a.plan) == a.plan_sha256, 'explicit full plan SHA/new output required')
        raise SystemExit(run(read(a.plan), a.out))
    else: print('DESCRIPTION ONLY: full original 1030-step history; original 2 GiB cache; explicit prepare/run required.')
