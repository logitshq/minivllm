"""Independent combined42/37 audit; stdlib, no device or service operations.

Reuse the frozen E024 parameter/route/full-state schedule algorithms. Added
evidence proves Python tracing branches and actual compiled callable submission,
not GPU kernel counts. Successful tensors are source-pinned device receipts;
resource/reap/restoration and HTTP/performance remain outside this audit.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from types import FunctionType

HERE = Path(__file__).resolve().parent
DONOR = HERE.parent / 'route_decode_graph/audit_serving.py'
DONOR_SHA = '064634acae9510b5744b7c848f09ee0854ee8bf26fc480af78dbed7f159573ba'
BASE_SHA = 'b6d90413911f2b17e0951a2fcfdb5fe00666aa06d4382b603e29e5dc4659d4f1'
NUMERIC_SHA = '21ba52e43ce54c2b463ab46bfd212c53d06d6303530c77939e3e6b801df547fb'
QUALIFIER_SHA = 'f131f3da91e847cf4f00e0644120919c49050227f5a35679b7a446aee8eaa6ee'
FIXED = {
    HERE / 'qualify.py': QUALIFIER_SHA,
    HERE / 'integration.py': '24f89e06f7e6d01041e6daaead40d1bacd927032a68218f8aa5a495bc092b851',
    HERE.parent / 'route_decode_graph/integration.py': '4a8dc5518f2ea0a913ad0a5d753a5032324ccb69bacede416abe8ae126bfa800',
    HERE.parent / 'route_decode_graph/candidate.py': '6531be25acb53713d6ddcd930d2c778f6640a3b1f53304fd5375d20b044525c3',
    HERE.parent / 'gdn_prework_unbound/candidate.py': '1bf5a837c6515ca505bc5d1beaa1718b88e1ccd52609f6814bb995afbe6c1260',
    HERE.parent / 'long_decode_graph/candidate.py': 'eb14b671a66764fe677219762090acb495b89d384cf078f3f63b918a40aa2b84',
    HERE.parent / 'long_decode_graph/model-source-pins.json': '7143538a168bfcd8b01292c9687472c8647db393c6d2a510d628f71c670260c5',
    HERE.parent / 'route_decode_graph/serving_qualification.py': '75de7205febdae2698e99353bfc2908a79044591fbf67abb30a07cb4e54a7264',
    HERE.parent / 'pareto_realistic/qualify_native.py': '26964fcbed9a2e5dcefaee4888244987897f3138d4a2b7add59bf63a7461b7bf',
    HERE.parent / 'long_decode_graph/qualify.py': 'c24219f431b08502a26a4f7a83c002214bddd7ea3962d6b1869b313e4c13faf7',
}


class Incomplete(Exception): pass
def need(ok, message):
    if not ok: raise ValueError(message)
def same(a, b, message):
    need(json.dumps(a, sort_keys=True, allow_nan=False) == json.dumps(b, sort_keys=True, allow_nan=False), message)
def read(path): return json.loads(Path(path).read_bytes())
def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''): digest.update(block)
    return digest.hexdigest()
def donor_module():
    need(sha(DONOR) == DONOR_SHA, 'frozen E024 auditor source')
    spec = importlib.util.spec_from_file_location('_combined_original_audit', DONOR)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def route_type(donor):
    class Routes(donor.RouteEvents):
        def __init__(self, stream):
            super().__init__(stream)
            self.seen = set(); self.counters = {i: [dict(candidate=0, fallback=0) for _ in range(48)] for i in (0, 1)}
            self.combined = dict(installs=2, restores=2, layer_installs=96, layer_restores=96,
                **{'compiled-tracing': 0, 'compiled-reuse': 0, 'eager-fallback': 0}, traced_batches=[])

        def external_decode(self, phase, B):
            # Original short decoder falls eagerly beyond1536. There is no
            # long-route event, but its per-layer Python counter advances once.
            slot = 0 if phase == 'D-C1' else 1
            for row in self.counters[slot]: row['candidate' if B == 1 else 'fallback'] += 1

        def route(self, phase, step, label, B, used, L):
            need(phase in ('parameter-diagnostic', 'D-C1', 'D-C8', 'lifecycle'), 'actual route phase')
            slot = 0 if phase in ('parameter-diagnostic', 'D-C1') else 1
            graph = 1 <= B <= 8 and L >= 1536
            bound = 1023 if used < 1024 else 1536; key = (slot, bound, B)
            row = self.take('combined-graph-call')
            need(set(row) == {'event', 'status', 'installation', 'phase', 'step', 'direct_label',
                'B', 'used', 'before', 'after', 'actual_invocations', 'kind', 'bound',
                'per_layer_candidate_delta', 'per_layer_fallback_delta'}, 'exact combined event fields')
            same(row['before'], self.counters[slot], 'continuous independently tracked48 counters')
            after = row['after']; need(type(after) is list and len(after) == 48, 'all48 after counters')
            deltas = []
            for left, right in zip(self.counters[slot], after):
                need(type(right) is dict and set(right) == {'candidate', 'fallback'}
                     and all(type(x) is int and x >= 0 for x in right.values()), 'strict integer after counters')
                deltas.append((right['candidate'] - left['candidate'], right['fallback'] - left['fallback']))
            need(all(x == deltas[0] for x in deltas) and min(deltas[0]) >= 0, 'uniform nonnegative layer trace')
            candidate, fallback = deltas[0]
            need(fallback == 0 if B == 1 else candidate == 0, 'correct original B1/multirow branch')
            if graph:
                if candidate + fallback:
                    kind = 'compiled-tracing'; self.seen.add(key)
                    self.combined['traced_batches'].append(dict(installation=slot, bound=bound, B=B,
                        python_passes=candidate + fallback))
                else:
                    need(key in self.seen, 'reuse needs prior actual same-instance/bound/B trace')
                    kind = 'compiled-reuse'
            else:
                need(candidate + fallback == 1, 'eager GDN fallback executes once per layer')
                kind = 'eager-fallback'
            expected = dict(event='combined-graph-call', status='SUBMITTED', installation=slot,
                phase=phase, step=step, direct_label=label, B=B, used=used, before=self.counters[slot],
                after=after, actual_invocations=int(graph), kind=kind, bound=bound,
                per_layer_candidate_delta=candidate, per_layer_fallback_delta=fallback)
            same(row, expected, 'combined event independent call/trace/branch binding')
            self.counters[slot] = [dict(x) for x in after]; self.combined[kind] += 1
            # Original callable count/domain/capacity and exact route row gate.
            super().route(phase, step, label, B, used, L)
    return Routes


def case_type(base, donor, routes, counts):
    original = donor.extended_case(base, routes, counts)
    class Case(original):
        def check_arrays(self, row, calls, pre_staging, has_future, B):
            super().check_arrays(row, calls, pre_staging, has_future, B)
            consumed = row['ahead']['candidate']['consumed_prior_future']
            for key in calls:
                if key[0] == 'decode' and key != consumed and max(key[2]) + 1 > 1536:
                    routes.external_decode(self.name, len(key[1]))
    return Case


def stream_with(base, case, *args, **kwargs):
    fn = base.stream_ledger
    need(type(fn) is FunctionType and fn.__closure__ is None and fn.__globals__ is vars(base), 'original stream identity')
    namespace = dict(fn.__globals__); namespace['CaseAudit'] = case
    bound = FunctionType(fn.__code__, namespace, fn.__name__, fn.__defaults__)
    bound.__kwdefaults__ = fn.__kwdefaults__
    return bound(*args, **kwargs)


def binding(child, digest, base):
    plan, top, claimed = [read(child / name) for name in ('plan.json', 'complete.json', 'base/complete.json')]
    if any(x.get('status') == 'INCOMPLETE' for x in (top, claimed)): raise Incomplete('producer not terminal')
    need(top['status'] == claimed['status'] == 'PASS', 'producer numerical failure')
    need(sha(child / 'plan.json') == digest == top['plan_sha256'], 'registered child plan')
    need(set(plan) == {'schema', 'qualifier_sha256', 'qualification', 'sources', 'deadline_s'}
         and plan['schema'] == 'GDN-graph-combined-serving-plan-v1'
         and type(plan['deadline_s']) is int and plan['deadline_s'] == 1800, 'exact combined plan')
    fixed = {str(p): h for p, h in FIXED.items()}
    same(plan['sources'], fixed, 'exact registered producer source closure')
    need(plan['qualifier_sha256'] == QUALIFIER_SHA, 'actual qualifier source')
    for path, digest in fixed.items(): need(sha(path) == digest, 'producer source changed: ' + path)
    same(top['sources'], fixed, 'terminal source pins')
    need(top['schema'] == 'GDN-graph-combined-serving-result-v1' and top['same_schedule_checks_passed'] is True
         and top['source_pins_rechecked'] is True and top['independent_audit_required'] is True
         and top['http_qualified'] is False and top['performance_qualified'] is False
         and top['identity_refill_enabled'] is False, 'producer terminal scope')
    same(top['total_requests'], 37, 'all37'); same(top['parameter_direct_calls'], 42, 'all42 direct calls')
    same(top['base_result'], dict(path='base/complete.json', sha256=sha(child/'base/complete.json'),
        status='PASS', returncode=0), 'original numerical terminal/exit')
    ref = plan['qualification']
    need(set(ref) == {'path', 'sha256'} and Path(ref['path']).is_absolute()
         and ref['sha256'] == sha(ref['path']) == sha(child/'base/plan.json') == NUMERIC_SHA, 'original numeric plan')
    numeric = read(child/'base/plan.json')
    need(claimed['plan_sha256'] == NUMERIC_SHA
         and claimed['qualifier_sha256'] == fixed[str(HERE.parent/'pareto_realistic/qualify_native.py')]
         and claimed['numeric_only_not_http'] is True and claimed['source_model_pins_rechecked'] is True, 'original terminal/source gates')
    same(claimed['total_requests'], 37, 'original37'); same(claimed['candidate'], numeric['candidate'], 'candidate flags')
    same(numeric['candidate'], dict(compile_decode=True, fuse_decode=True, gates_compile=True, norm_reuse=False), 'fixed flags')
    same(numeric['deadline_s'], 1800, 'original deadline')
    manifests = {}
    for field in ('fixtures', 'source_manifest', 'model_manifest'):
        item = numeric[field]; need(sha(item['path']) == item['sha256'], 'manifest SHA: ' + field); manifests[field] = read(item['path'])
    root = Path(numeric['package_root'])/'tutorial_code/miniengine'; core = manifests['source_manifest']['files']
    need(set(core) == {str(p.relative_to(root)) for p in root.rglob('*.py')}, 'complete frozen core set')
    for rel, digest in core.items():
        need(not Path(rel).is_absolute() and '..' not in Path(rel).parts and sha(root/rel) == digest, 'frozen core source')
    model = manifests['model_manifest']; config_path = Path(numeric['model'])/'config.json'
    need(model['schema'] == 'E023-model-v1' and model['model'] == numeric['model']
         and model['revision'] == Path(numeric['model']).name and sha(config_path) == model['files']['config.json'], 'model identity/config')
    config = read(config_path); config = config.get('text_config', config)
    need(config['layer_types'] == ['full_attention' if i % 4 == 3 else 'linear_attention' for i in range(64)]
         and config['vocab_size'] == 248320 and config['hidden_size'] == 5120, '64-layer model dimensions')
    fixtures = manifests['fixtures']; need(numeric['fixtures']['sha256'] == base.FIXTURE_SHA and fixtures['model'] == numeric['model'], 'fixture identity')
    fixtures = fixtures['fixtures']; need(len(fixtures) == 8 and len({f['fixture_id'] for f in fixtures}) == 8, 'eight D fixtures')
    eos = fixtures[0]['eos_token_ids']
    need(all(f['eos_token_ids'] == eos and f['normal_eos'] is True and f['max_output_tokens'] == 512 for f in fixtures), 'normal EOS/cap512')
    return plan, top, claimed, fixtures, config, eos


def audit(child_dir, expected_plan_sha256):
    child = Path(child_dir).resolve()
    result = dict(schema='GDN-graph-combined-independent-v1', status='INCOMPLETE', qualified=False,
        same_schedule_qualified=False, ledger_opened=False, http_qualified=False, performance_qualified=False,
        owner_resources_restore_qualified=False, offline_tensor_bytes_rechecked=False,
        audit_source_sha256=sha(__file__), donor_auditor_sha256=DONOR_SHA, base_auditor_sha256=BASE_SHA,
        child_directory=str(child), scope='Independent original42/37 schedule/state/parameter receipts plus GDN tracing and actual compiled-call evidence; not offline tensor/GPU recomputation.')
    try:
        for name in ('complete.json', 'plan.json', 'base/complete.json', 'base/plan.json', 'base/steps.jsonl', 'route-events.jsonl'):
            if not (child/name).is_file(): raise Incomplete('missing ' + name)
        donor = donor_module(); base = donor.base_auditor()
        need(donor.BASE_SHA == BASE_SHA, 'original numerical auditor identity')
        plan, top, claimed, fixtures, config, eos = binding(child, expected_plan_sha256, base)
        stable = {name: sha(child/name) for name in ('plan.json', 'complete.json', 'base/complete.json', 'base/plan.json')}
        paths = [child/'base/steps.jsonl', child/'route-events.jsonl']; before = [p.stat() for p in paths]; counts = {}
        with paths[1].open('rb') as stream:
            routes = route_type(donor)(stream); routes.parameters(fixtures, config['layer_types'])
            case = case_type(base, donor, routes, counts); result['ledger_opened'] = True
            ledger = stream_with(base, case, paths[0], fixtures, config, eos, claimed['cases'], partial=False)
            result['ledger'] = ledger; result['route_events'] = routes.finish()
        need([len(c['completed']) for c in ledger['cases']] == [8, 16, 13] and not ledger['partial_cases']
             and ledger['truncated_final_line'] is None, 'complete untruncated37')
        route_coverage = dict(installs=2, restores=4, parameter_checks_complete=True, parameter_direct_calls=42,
            route_hits=routes.hits, long_fallback_calls=routes.fallbacks, cases=counts, identity_refill_enabled=False)
        same(top['coverage'], dict(route=route_coverage, combined=routes.combined), 'independently reconstructed complete coverage')
        for case_name in ('D-C1', 'D-C8'):
            need(all(routes.hits.get(case_name, {}).get(b, 0) > 0 for b in ('1023', '1536'))
                 and counts[case_name]['consumed'] > 0, 'both graph families/actual ahead: ' + case_name)
        need(sum(c['partial_retirement'] for c in counts.values()) > 0 and counts['D-C8']['refills'] > 0, 'partial retirement/refill')
        traces = routes.combined['traced_batches']
        need(all(any(r['B'] == 1 and r['bound'] == bound for r in traces) for bound in (1023, 1536))
             and any(r['B'] > 1 for r in traces) and routes.combined['compiled-reuse'] > 0, 'combined trace/reuse coverage')
        discarded = sum(c['discarded_prior'] for c in counts.values())
        same(top['discarded_ahead_observed'], discarded > 0, 'observed discarded-ahead flag')
        same(top['discarded_prior_ahead_count'], discarded, 'actual discarded-ahead count')
        need(all(sha(child/name) == digest for name, digest in stable.items()), 'terminal changed during audit')
        for path, old, size in zip(paths, before, (ledger['bytes'], routes.bytes)):
            new = path.stat(); need((old.st_dev, old.st_ino, old.st_size, old.st_mtime_ns) ==
                (new.st_dev, new.st_ino, new.st_size, new.st_mtime_ns) and size == old.st_size, 'closed evidence changed/truncated')
        need(all(sha(path) == digest for path, digest in FIXED.items()) and sha(DONOR) == DONOR_SHA
             and sha(donor.BASE) == BASE_SHA, 'audited source changed')
        result.update(status='PASS', qualified=True, same_schedule_qualified=True,
            plan_sha256=expected_plan_sha256, base_plan_sha256=NUMERIC_SHA, terminal_sha256=stable,
            coverage=dict(route=route_coverage, combined=routes.combined), source_pins={str(p): h for p, h in FIXED.items()},
            parameter_scope=[x[0] for x in donor.PARAMETERS], weight_shards_rehashed=False,
            observed_eos_requests=sum(bool(r['raw_ids']) and r['raw_ids'][-1] in eos for c in ledger['cases'] for r in c['completed']),
            uncovered='No all-parameter/hot-reload, unobserved ahead invalidation, HTTP/performance, E028 or owner/resource/restoration qualification.')
    except (Incomplete, FileNotFoundError) as exc: result.update(status='INCOMPLETE', reason=str(exc))
    except Exception as exc: result.update(status='FAIL', first_failure=dict(type=type(exc).__name__, message=str(exc)))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--child', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True); parser.add_argument('--out', type=Path, required=True); args = parser.parse_args(argv)
    need(args.out.parent.is_dir() and not args.out.exists(), 'new output with existing parent required')
    result = audit(args.child, args.plan_sha256)
    with args.out.open('x') as stream: json.dump(result, stream, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps(dict(status=result['status'], qualified=result['qualified'])))
    return int(not result['qualified'])


if __name__ == '__main__': raise SystemExit(main())
