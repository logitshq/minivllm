"""Prepared combined42-direct-call +37-request numerical gate, not HTTP/perf.

No arguments describe only. --prepare-plan writes a small immutable child plan.
Complete --plan/--plan-sha256/--out invokes the frozen loader and needs an external
cold/resource/reap owner. No service actions or lifecycle implementation here.
"""
import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
from pathlib import Path
import traceback

HERE = Path(__file__).resolve().parent
DONOR = HERE.parent / 'route_decode_graph/serving_qualification.py'
INTEGRATION = HERE / 'integration.py'
BASE_PLAN_SHA = '21ba52e43ce54c2b463ab46bfd212c53d06d6303530c77939e3e6b801df547fb'
FIXED = {
    DONOR: '75de7205febdae2698e99353bfc2908a79044591fbf67abb30a07cb4e54a7264',
    INTEGRATION: '24f89e06f7e6d01041e6daaead40d1bacd927032a68218f8aa5a495bc092b851',
    HERE.parent / 'pareto_realistic/qualify_native.py': '26964fcbed9a2e5dcefaee4888244987897f3138d4a2b7add59bf63a7461b7bf',
    HERE.parent / 'long_decode_graph/qualify.py': 'c24219f431b08502a26a4f7a83c002214bddd7ea3962d6b1869b313e4c13faf7',
}


def need(ok, message):
    if not ok: raise ValueError(message)
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_bytes())
def write(path, value, *, exclusive=False):
    with Path(path).open('x' if exclusive else 'w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')
def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def source_pins():
    for path, digest in FIXED.items(): need(sha(path) == digest, 'fixed helper source changed: ' + str(path))
    integration = load(INTEGRATION, '_combined_plan_install')
    return {**integration.source_pins(), **{str(p): s for p, s in FIXED.items()},
            str(Path(__file__).resolve()): sha(__file__)}


def make_plan(qualification):
    qualification = Path(qualification).resolve()
    need(sha(qualification) == BASE_PLAN_SHA, 'unchanged original37 plan required')
    return dict(schema='GDN-graph-combined-serving-plan-v1', qualifier_sha256=sha(__file__),
        qualification=dict(path=str(qualification), sha256=BASE_PLAN_SHA),
        sources=source_pins(), deadline_s=1800)


def validate(path, digest):
    need(sha(path) == digest, 'child plan SHA'); plan = read(path)
    need(set(plan) == {'schema', 'qualifier_sha256', 'qualification', 'sources', 'deadline_s'}, 'exact child fields')
    need(type(plan['deadline_s']) is int and plan['deadline_s'] == 1800, 'fixed1800 deadline')
    need(set(plan['qualification']) == {'path', 'sha256'}
         and Path(plan['qualification']['path']).is_absolute(), 'exact inherited reference')
    need(plan == make_plan(plan['qualification']['path']), 'registered child identity/source/switch contract')
    inherited = read(plan['qualification']['path'])
    need(type(inherited['deadline_s']) is int and inherited['deadline_s'] == 1800
         and inherited['candidate'] == dict(compile_decode=True, fuse_decode=True, gates_compile=True, norm_reuse=False),
         'inherited candidate/deadline unchanged')
    return plan


def delta(before, after, B):
    need(len(before) == len(after) == 48, 'all48 installed GDN layers')
    rows = []
    for left, right in zip(before, after):
        need(set(left) == set(right) == {'candidate', 'fallback'}
             and all(type(x) is int and x >= 0 for row in (left, right) for x in row.values()), 'integer GDN counters')
        rows.append((right['candidate'] - left['candidate'], right['fallback'] - left['fallback']))
    need(all(row == rows[0] for row in rows) and min(rows[0]) >= 0, 'mixed/stale layer traces')
    candidate, fallback = rows[0]
    need((fallback == 0 if B == 1 else candidate == 0), 'wrong frozen GDN branch for traced batch')
    return candidate, fallback


class TraceGraph:
    """Wrap Python submission only; never evaluate, change args, or repeat a call."""
    def __init__(self, graph, handles, emit, stats, installation):
        self.graph, self.handles, self.emit, self.stats = graph, handles, emit, stats
        self.installation = installation; self.witness = None; self.seen = set()
    @property
    def _functions(self): return self.graph._functions
    @_functions.setter
    def _functions(self, value): self.graph._functions = value

    def __call__(self, tokens, ctx):
        before = [dict(handle.calls) for handle in self.handles]
        invocations = self._functions.invocations
        witness = self.witness
        row = dict(event='combined-graph-call', status='FAILED', installation=self.installation,
            phase=witness.phase, step=witness.step, direct_label=witness.direct_label,
            B=tokens.shape[0], used=ctx.max_len, before=before)
        try:
            output = self.graph(tokens, ctx)
            after = [dict(handle.calls) for handle in self.handles]
            count = self._functions.invocations - invocations
            row.update(after=after, actual_invocations=count)
            candidate, fallback = delta(before, after, tokens.shape[0])
            need(count in (0, 1), 'one actual frozen compiled callable or eager fallback')
            bound = 1023 if ctx.max_len < 1024 else 1536
            key = (bound, tokens.shape[0])
            if count:
                if candidate + fallback:
                    self.seen.add(key); kind = 'compiled-tracing'
                    self.stats['traced_batches'].append(dict(installation=self.installation, bound=bound,
                        B=tokens.shape[0], python_passes=candidate + fallback))
                else:
                    need(key in self.seen, 'compiled reuse without observed same-bound/B tracing')
                    kind = 'compiled-reuse'
            else:
                kind = 'eager-fallback'
            self.stats[kind] += 1
            row.update(status='SUBMITTED', actual_invocations=count, kind=kind, bound=bound,
                per_layer_candidate_delta=candidate, per_layer_fallback_delta=fallback, after=after)
            return output
        finally:
            row.setdefault('after', [dict(handle.calls) for handle in self.handles])
            row.setdefault('actual_invocations', self._functions.invocations - invocations)
            self.emit(row)


class Handle:
    def __init__(self, scope, donor, emit, stats):
        self.scope, self.donor, self.stats = scope, donor, stats
        self.decoder = scope.graph_handle.decoder
        self.proxy = TraceGraph(self.decoder.long, scope.gdn_handles, emit, stats, stats['installs'])
        self.decoder.long = self.proxy; self.closed = False
    def restore(self):
        if self.closed: return
        long = self.decoder.long
        need(long is self.proxy or (type(long) is self.donor.RouteWitness and long.graph is self.proxy),
             'foreign long graph observer')
        need(self.scope.runner.decode_model is self.decoder and self.scope.runner.model is self.scope.model
             and 'run' not in vars(self.scope.runner), 'restore numerical observer before combined handle')
        self.decoder.long = self.proxy.graph
        self.scope.restore()
        self.stats['restores'] += 1; self.stats['layer_restores'] += 48
        self.proxy.graph = self.proxy.witness = None; self.proxy.handles = []
        self.scope = self.decoder = None; self.closed = True


@contextmanager
def adapt(base, donor, integration, old, fixtures, emit):
    """Reuse frozen RouteObserver and run_case; modify only the private base hooks."""
    stats = dict(installs=0, restores=0, layer_installs=0, layer_restores=0,
        **{'compiled-tracing': 0, 'compiled-reuse': 0, 'eager-fallback': 0}, traced_batches=[])
    class Bridge:
        engine = None
        def install(self, runner, *, enabled):
            need(enabled is True and self.engine.runner is runner, 'actual constructing engine required')
            scope = integration.install(self.engine, gdn_prework=True, route_graph=True)
            try:
                need(scope is not None and len(scope.gdn_handles) == 48, 'combined actual48 installation')
                handle = Handle(scope, donor, emit, stats)
            except BaseException:
                if scope is not None: scope.restore()
                raise
            stats['installs'] += 1; stats['layer_installs'] += 48
            return handle
    bridge = Bridge()
    with donor.adapt(base, bridge, old, fixtures, emit) as route_coverage:
        inherited = base.Observer
        class Observer(inherited):
            def __init__(self, engine, cmp, label):
                need(bridge.engine is None, 'serial observer construction')
                bridge.engine = engine
                try:
                    super().__init__(engine, cmp, label)
                    if self.route_install is not None:
                        self.route_install.proxy.witness = self.route_witness
                finally: bridge.engine = None
        base.Observer = Observer
        try: yield dict(route=route_coverage, combined=stats)
        finally: base.Observer = inherited


def require_complete(donor, completed, coverage):
    observed = donor.require_complete(completed, coverage['route'])
    stats = coverage['combined']
    need(stats['installs'] == stats['restores'] == 2
         and stats['layer_installs'] == stats['layer_restores'] == 96, 'complete combined lifetimes')
    batches = stats['traced_batches']
    need(all(any(row['B'] == 1 and row['bound'] == bound for row in batches) for bound in (1023, 1536)),
         'actual B1 GDN tracing in both graph families')
    need(any(row['B'] > 1 for row in batches) and stats['compiled-reuse'] > 0,
         'actual multirow fallback tracing and compiled reuse required')
    return observed


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--plan', type=Path); ap.add_argument('--plan-sha256'); ap.add_argument('--out', type=Path)
    ap.add_argument('--prepare-plan', type=Path); ap.add_argument('--qualification-plan', type=Path)
    args = ap.parse_args(argv)
    run = (args.plan, args.plan_sha256, args.out)
    if args.prepare_plan is not None:
        if any(run) or args.qualification_plan is None: ap.error('prepare needs only qualification-plan')
        plan = make_plan(args.qualification_plan); write(args.prepare_plan, plan, exclusive=True)
        print(json.dumps(dict(plan=str(args.prepare_plan.resolve()), sha256=sha(args.prepare_plan)))); return 0
    if args.qualification_plan is not None or (any(run) and not all(run)): ap.error('complete three execution arguments required')
    if not any(run):
        print(json.dumps(dict(schema='GDN-graph-combined-qualification-description-v1', run=False,
            direct_calls=42, requests=37, http_qualified=False, performance_qualified=False))); return 0
    plan = validate(args.plan, args.plan_sha256)
    args.out.mkdir(parents=True, exist_ok=False); write(args.out / 'plan.json', plan, exclusive=True)
    result = dict(schema='GDN-graph-combined-serving-result-v1', status='INCOMPLETE',
        plan_sha256=args.plan_sha256, sources=plan['sources'], same_schedule_checks_passed=False,
        independent_audit_required=True, http_qualified=False, performance_qualified=False,
        identity_refill_enabled=False, source_pins_rechecked=False)
    write(args.out / 'complete.json', result)
    try:
        donor = load(DONOR, '_combined_frozen_route_serving')
        base = donor.load(donor.BASE, '_combined_original37')
        old = donor.load(donor.OLD, '_combined_parameter_helpers')
        integration = load(INTEGRATION, '_combined_install')
        fixtures = base.pinned_json(read(plan['qualification']['path'])['fixtures'])['fixtures']
        with (args.out / 'route-events.jsonl').open('x') as events:
            def emit(row): events.write(json.dumps(row, allow_nan=False) + '\n'); events.flush()
            with adapt(base, donor, integration, old, fixtures, emit) as coverage:
                result['coverage'] = coverage
                code = base.main(['--plan', plan['qualification']['path'], '--plan-sha256', BASE_PLAN_SHA,
                                  '--out', str(args.out / 'base')])
        path = args.out / 'base/complete.json'; completed = read(path)
        result['base_result'] = dict(path='base/complete.json', sha256=sha(path), status=completed['status'], returncode=code)
        need(code == 0, 'original full numerical gate failed')
        observed = require_complete(donor, completed, coverage)
        need(validate(args.plan, args.plan_sha256) == plan, 'source/plan changed')
        result.update(status='PASS', same_schedule_checks_passed=True, source_pins_rechecked=True,
            total_requests=37, parameter_direct_calls=42, **observed,
            scope='Device full-logit/state comparisons and exact original42 parameter diagnostic +37 requests. '
                  'GDN Python counters prove tracing branches, not repeated GPU executions. '
                  'Independent numerical/ownership/resource audit still required; no HTTP/performance qualification.')
    except BaseException as exc:
        result.update(status='FAIL', error=dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc()))
        path = args.out / 'base/complete.json'
        if path.is_file(): result['base_result'] = dict(path='base/complete.json', sha256=sha(path), status=read(path).get('status'))
    finally: write(args.out / 'complete.json', result)
    print(json.dumps(dict(status=result['status'], out=str(args.out))), flush=True)
    return int(result['status'] != 'PASS')


if __name__ == '__main__': raise SystemExit(main())
