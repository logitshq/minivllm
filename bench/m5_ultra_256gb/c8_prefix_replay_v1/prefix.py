"""Default-OFF E047 warm49 same-history numerical diagnostic, not qualification.

The complete recording/caps stay unchanged. An explicit iterator boundary exits
only after the frozen replay loop finishes its last checked step. No rewritten
recording, fabricated retirement, sampled-ID injection, or source-code rewriting.
The existing two-engine loader, comparator, observers and release are reused.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import traceback

HERE = Path(__file__).resolve().parent
B = HERE.parent; R = B.parent.parent; A = R / '.artifacts/m5-ultra-256gb'
ROUTE = B / 'route_decode_graph/http_qualification_v2.py'
ROUTE_SHA = 'd3e2a4f76b7c384f6c6c986d0838b68227618502b0552e5cdf74f149029e39e7'
CAPTURE = B / 'c8_schedule_diagnostic_v1/trial.py'
CAPTURE_SHA = 'a140fa0b2cf4d6eb1bd58286696d4cc16ad3747bf396497817c91dcb98f5bad5'
MLX_ENV = {'MLX_MAX_OPS_PER_BUFFER': '50', 'MLX_MAX_MB_PER_BUFFER': '50',
           'MLX_BFS_MAX_WIDTH': '20', 'MLX_METAL_FAST_SYNCH': '0'}
SCOPE = 'Warm49 prefix only, incomplete by design; no full-history, performance or Pareto qualification.'


def need(ok, message):
    if not ok: raise ValueError(message)
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_bytes())
def write(path, value): Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
def ref(path): return dict(path=str(Path(path).resolve()), sha256=sha(path))
def pinned(item):
    path = Path(item['path'])
    need(set(item) == {'path', 'sha256'} and path.is_absolute() and path.resolve() == path
         and path.is_file() and not path.is_symlink() and sha(path) == item['sha256'], 'pinned input changed')
    return path

def load(path, digest, name):
    need(sha(path) == digest, 'fixed support changed')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def prepare(child, analysis):
    child = Path(child).resolve(); analysis = Path(analysis).resolve()
    capture = load(CAPTURE, CAPTURE_SHA, '_prefix_capture_contract')
    spec = read(child / 'spec.json'); capture.validate(spec)
    top, joined = read(child / 'complete.json'), read(child / 'joined-input/receipt.json')
    need(top['status'] == 'PASS' and top['server_reaped'] and top['cleanup_complete'], 'closed E047 capture required')
    recording_path = child / 'joined-input/recording.json'; recording = read(recording_path)
    need(joined['spec_sha256'] == sha(child / 'spec.json')
         and joined['recording_sha256'] == sha(recording_path)
         and joined['trace_sha256'] == sha(child / 'trace.jsonl'), 'actual joined input bindings')
    analysis_data = read(analysis)
    need(analysis_data['status'] == 'PASS_RAW_ANALYSIS'
         and analysis_data['inputs'][str(recording_path)] == sha(recording_path), 'analysis binds exact recording')
    recipes = [r for r in analysis_data['d06_recipes'] if r['wave'] == 'warm']
    need(len(recipes) == 1, 'one exact warm recipe')
    recipe = recipes[0]; stop = recipe['stop_after_global_step']; rid = recipe['rid']
    need(stop == 49 and recipe['first_difference_output_index'] == 46
         and recipe['stop_after_output_index'] == 48 and recipe['history_start_global_step'] == 1,
         'registered actual warm49 stop only')
    ids = [token for step in recording['steps'][:stop] for owner, token, done in step['results'] if owner == rid]
    need(ids == recording['jobs'][rid]['raw_ids'][:49] and len(ids) == 49
         and all(j['cap'] == 512 for j in recording['jobs'].values()), 'unmodified cap/output prefix')
    q = load(ROUTE, ROUTE_SHA, '_prefix_route_contract')
    sources = q.source_pins(); sources.update({str(CAPTURE): CAPTURE_SHA, str(Path(__file__).resolve()): sha(__file__)})
    need(all(sha(p) == h for p, h in sources.items()), 'source closure changed')
    inputs = [child / n for n in ('spec.json', 'complete.json', 'release.json', 'child-absence.json',
        'port-after.json', 'group.json', 'trace.jsonl', 'route-events.jsonl', 'joined-input/receipt.json')]
    inputs += [Path(x['path']) for x in joined['source_groups']]
    for x in joined['source_groups']: need(sha(x['path']) == x['sha256'], 'original raw wave changed')
    return dict(schema='c8-native-graph-prefix-plan-v1', sources=sources,
        qualification=spec['base_plan'], recording=ref(recording_path), analysis=ref(analysis),
        capture_directory=str(child), inputs={str(p): sha(p) for p in inputs}, recipe=recipe,
        environment=MLX_ENV, stop_after_step=stop, completion_state='INCOMPLETE_BY_DESIGN',
        memory_policy='Unchanged outer 64 GiB child / 240 GiB system / no swap-growth; prefix peak unknown.',
        performance_qualified=False, full_history_qualified=False, scope=SCOPE)


def validate(plan):
    expected = prepare(plan['capture_directory'], pinned(plan['analysis']))
    need(json.dumps(plan, sort_keys=True) == json.dumps(expected, sort_keys=True), 'changed prefix specification')


class PrefixBoundary(RuntimeError):
    """An intentional boundary outside the original loop's per-step try block."""


class PrefixSteps:
    def __init__(self, steps, stop, boundary):
        need(type(stop) is int and 0 < stop < len(steps), 'strict nonempty incomplete prefix required')
        self.steps, self.stop, self.boundary = steps, stop, boundary
    def __len__(self): return len(self.steps)  # Preserve the original finite-history guard.
    def __iter__(self):
        for step in self.steps[:self.stop]: yield step
        raise self.boundary  # Previous step has checked, flushed and cleared snapshots.


class PrefixLedger:
    def __init__(self, original): self.original = original; self.rows = []
    def write(self, text):
        result = self.original.write(text); row = json.loads(text)
        if 'step' in row: self.rows.append({k: row[k] for k in ('step', 'status', 'candidate_delivered') if k in row})
        return result
    def flush(self): return self.original.flush()


def run_prefix(replay, recording, stop, engines, observers, sampling, runner_module, cmp, ledger, deadline, *, base):
    """All frozen per-step checks run. Only the original final-drain branch is skipped."""
    boundary = PrefixBoundary('registered prefix complete; whole history remains unfinished')
    view = {**recording, 'steps': PrefixSteps(recording['steps'], stop, boundary)}
    observed = PrefixLedger(ledger)
    try:
        replay.run_recorded_case(view, engines, observers, sampling, runner_module, cmp, observed, deadline, base=base)
    except PrefixBoundary as exc:
        if exc is not boundary: raise
    else:
        raise RuntimeError('original replay returned without the registered prefix boundary')
    need(len(observed.rows) == stop and all(r['status'] == 'PASS' and r['step'] == i
         for i, r in enumerate(observed.rows, 1)), 'every original prefix step must finish checks')
    delivered, completed = {}, []
    for row in observed.rows:
        for rid, token, done in row['candidate_delivered']:
            delivered.setdefault(rid, []).append(token)
            if done:
                ids = list(delivered[rid]); committed = ids[:-1] if ids[-1] in recording['eos_token_ids'] else ids
                completed.append(dict(rid=rid, raw_ids=ids, committed_ids=committed))
    active = sorted(set(delivered) - {j['rid'] for j in completed})
    need(active, 'registered incomplete prefix unexpectedly drained all admitted work')
    return dict(name='HTTP-C8-prefix', status='DIAGNOSTIC_PREFIX_COMPLETE',
        completion_state='INCOMPLETE_BY_DESIGN', steps=stop, completed=completed,
        delivered_prefix_ids=delivered, active_request_ids=active,
        future_unadmitted_request_ids=sorted(set(recording['jobs']) - set(delivered)),
        full_history_qualified=False, performance_qualified=False, scope=SCOPE)


def integration_module(q):
    """E049 fixed graph-only hook. Future candidates require a separately pinned adapter."""
    return q.load(q.HERE / 'integration.py', '_prefix_install')


def run(plan, out):
    validate(plan)
    need({k: v for k, v in os.environ.items() if k.startswith('MLX_')} == plan['environment'], 'exact registered MLX environment')
    out = Path(out).resolve(); out.mkdir(parents=True, exist_ok=False); write(out / 'plan.json', plan)
    result = dict(schema='c8-native-graph-prefix-result-v1', status='INCOMPLETE', scope=SCOPE,
        pid=os.getpid(), ppid=os.getppid(), plan_sha256=sha(out / 'plan.json'),
        completion_state='INCOMPLETE_BY_DESIGN', full_history_qualified=False, performance_qualified=False)
    write(out / 'complete.json', result)
    try:
        q = load(ROUTE, ROUTE_SHA, '_prefix_route_execution')
        base = q.load(q.BASE, '_prefix_base'); replay = q.load(q.NATIVE / 'http_replay.py', '_prefix_loop')
        integration = integration_module(q)
        witness = q.load(q.HERE / 'serving_qualification.py', '_prefix_witness')
        counter = q.load(q.OLD / 'qualify.py', '_prefix_counter')
        recording = read(pinned(plan['recording'])); numeric = read(pinned(plan['qualification']))
        fixtures = read(pinned(numeric['fixtures']))['fixtures']
        child = Path(plan['capture_directory']); spec = read(child / 'spec.json')
        route_path = child / 'route-events.jsonl'
        header = dict(event='route-startup', schema='E024-route-http-events-v1',
            pid=read(child / 'release.json')['pid'], phase='HTTP-C8', identity=spec['identity'])
        with route_path.open('rb') as captured, (out / 'replay-route-events.jsonl').open('x') as emitted:
            routes = q.CapturedRoutes(captured, header, emitted, ref(route_path), 'HTTP-C8')
            with q.adapted(base, replay, integration, witness, counter, recording, fixtures, routes) as coverage:
                def run_case(case, engines, observers, sampling, runner_module, cmp, ledger, deadline):
                    need(case == dict(name='D-C8', concurrency=8), 'single original C8 engine pair')
                    return run_prefix(replay, recording, plan['stop_after_step'], engines, observers,
                        sampling, runner_module, cmp, ledger, deadline, base=base)
                base.run_case = run_case
                code = base.main(['--plan', plan['qualification']['path'], '--plan-sha256',
                    plan['qualification']['sha256'], '--out', str(out / 'base')])
            inner = read(out / 'base/complete.json')
            result.update(base_result=dict(returncode=code, status=inner['status'], sha256=sha(out / 'base/complete.json')),
                coverage=coverage, route_prefix=dict(calls=routes.calls, hits=routes.hits, fallbacks=routes.fallbacks,
                    consumed_bytes=routes.bytes, consumed_lines=routes.lines, consumed_sha256=routes.digest.hexdigest(),
                    full_route_stream_consumed=False), peak_mlx_bytes=inner.get('peak_bytes'),
                elapsed_base_seconds=inner.get('elapsed_wall_s'))
            need(code == 0 and inner['status'] == 'PASS' and inner['source_model_pins_rechecked'] is True,
                 'prefix arithmetic or cleanup failed; inspect original base failure')
            need(len(inner['cases']) == 1 and inner['cases'][0]['status'] == 'DIAGNOSTIC_PREFIX_COMPLETE'
                 and inner['cases'][0]['steps'] == 49 and inner['cases'][0]['completed'] == []
                 and inner['total_requests'] == 0 and coverage['installs'] == 1 and coverage['restores'] == 2,
                 'actual prefix boundary, unfinished requests and pair release')
            result['prefix'] = inner['cases'][0]
        validate(plan)
        result.update(status='DIAGNOSTIC_PREFIX_COMPLETE', source_input_pins_rechecked=True)
    except BaseException as exc:
        result.update(status='FAIL', error=dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc()))
    finally:
        write(out / 'complete.json', result)
    print(json.dumps(dict(status=result['status'], out=str(out))))
    return 0 if result['status'] == 'DIAGNOSTIC_PREFIX_COMPLETE' else 1


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); modes = p.add_mutually_exclusive_group()
    modes.add_argument('--prepare', action='store_true'); modes.add_argument('--run', action='store_true')
    p.add_argument('--child', type=Path); p.add_argument('--analysis', type=Path)
    p.add_argument('--plan', type=Path); p.add_argument('--plan-sha256'); p.add_argument('--out', type=Path); a = p.parse_args()
    if a.prepare:
        need(a.child and a.analysis and a.out, 'prepare needs closed capture, analysis and new output')
        plan = prepare(a.child, a.analysis); a.out.mkdir(parents=True, exist_ok=False); write(a.out / 'plan.json', plan)
        print(json.dumps(dict(status='PREPARED_NOT_RUN', plan=str(a.out / 'plan.json'), sha256=sha(a.out / 'plan.json'))))
    elif a.run:
        need(a.plan and a.plan_sha256 and a.out and sha(a.plan) == a.plan_sha256, 'explicit plan SHA/new output required')
        raise SystemExit(run(read(a.plan), a.out))
    else: print('DESCRIPTION ONLY: explicit prepare/run; original recording and caps unchanged; prefix incomplete by design.')
