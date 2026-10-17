"""E028 preparation adapter for the frozen service-absent E023 owner.

The unchanged outer service protocol binds this file, a NEW cold journal and an
E028 three-field child plan. The original E023 numerical plan is really verified
in e023-input-preflight/; no fabricated plan or alternate module HERE is used.
Only private validate_plan/load_prepare/receipt-plan bindings are configured.
The frozen run owns pressure-1 admission, absence checks, 1830s work deadline,
durable spawn/no-child evidence and bounded reap. This file never controls the
service. Cold removal/recovery still require their explicit approved contract.
The old flat E023 audit is not compatible with E028's nested child evidence;
successful preparation is not full-model qualification or a performance point.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace


HERE = Path(__file__).resolve().parent
BASE = HERE.parent / 'pareto_realistic/qualify_absent_owner.py'
BASE_SHA = '5beb6c9bcfa4e22041a386cdbea3f5ecee210cdcb9b345512c50b6c6f873c62d'
ORIGINAL_PLAN_SHA = '21ba52e43ce54c2b463ab46bfd212c53d06d6303530c77939e3e6b801df547fb'
CONTROLLER = HERE.parent / 'omlx_cold_restart/controller_v2.py'
CONTROLLER_SHA = 'f06d0ba60e0966fe83741f9c5e60b74404a84154242760d005170a47327496be'
FIXED = {
    HERE.parent / 'pareto_realistic/qualify_native.py': '26964fcbed9a2e5dcefaee4888244987897f3138d4a2b7add59bf63a7461b7bf',
    HERE / 'integration.py': 'ea0d6cb6af7c80d7580b4ab1c8319a4b76110fd1d73cf660c048c8dd45cf0c52',
    HERE / 'candidate.py': 'bb9f8a461625e97c80e75140c9d9b7b8718a633595ab4f3a422267e752ded8f5',
}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_owner():
    need(sha(BASE) == BASE_SHA, 'frozen absent owner source changed')
    spec = importlib.util.spec_from_file_location('_e028_private_absent_owner', BASE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_child(item, base):
    need(set(item) == {'path', 'sha256'}, 'E028 child path/SHA pair required')
    path = base.absolute(item['path'])
    raw = path.read_bytes()
    need(base.sha_bytes(raw) == item['sha256'], 'E028 child plan SHA mismatch')
    plan = json.loads(raw)
    need(set(plan) == {'schema', 'qualification', 'sources'}
         and plan['schema'] == 'E028-model-qualification-plan-v1', 'E028 child schema')
    q = plan['qualification']
    need(set(q) == {'path', 'sha256'} and q['sha256'] == ORIGINAL_PLAN_SHA,
         'unchanged original E023 plan required')
    original = base.absolute(q['path']).read_bytes()
    need(base.sha_bytes(original) == ORIGINAL_PLAN_SHA, 'original numerical plan SHA')
    numeric = json.loads(original)
    expected = {str(p): digest for p, digest in FIXED.items()}
    expected[str(HERE / 'model_qualification.py')] = sha(HERE / 'model_qualification.py')
    need(plan['sources'] == expected, 'exact four E028 source pins required')
    for source, digest in expected.items():
        need(sha(source) == digest, 'E028 source changed: ' + source)
    need(numeric['deadline_s'] == 1800 and numeric['execution']['qualifier'] == {
        'path': str(HERE.parent / 'pareto_realistic/qualify_native.py'),
        'sha256': FIXED[HERE.parent / 'pareto_realistic/qualify_native.py']}, 'original qualifier/deadline binding')
    return plan, raw, numeric


def validate_plan(args, base):
    raw = base.absolute(args.plan).read_bytes()
    need(base.sha_bytes(raw) == args.plan_sha256, 'owner plan SHA mismatch')
    plan = json.loads(raw)
    need(set(plan) == {'schema', 'execution', 'qualification', 'cold_restart', 'deadline_seconds'}
         and plan['schema'] == 'E023-service-absent-owner-plan-v1', 'service protocol wrapper schema')
    need(type(plan['deadline_seconds']) is int and plan['deadline_seconds'] == 1830, 'fixed 1830s work deadline')
    need(plan['execution'] == {'owner': {'path': str(Path(__file__).resolve()), 'sha256': sha(__file__)}},
         'actual E028 owner source pin')
    _, _, numeric = read_child(plan['qualification'], base)
    cold = plan['cold_restart']
    need(set(cold) == {'journal_path', 'controller', 'approved_contract', 'proposal_plan_sha256'}, 'cold contract fields')
    base.absolute(cold['journal_path'])
    need(cold['controller'] == {'path': str(CONTROLLER), 'sha256': CONTROLLER_SHA}
         and sha(CONTROLLER) == CONTROLLER_SHA, 'fixed v2 controller source')
    need(cold['approved_contract'] == base.CONTRACT and cold['proposal_plan_sha256'] == base.PROPOSAL_SHA,
         'approved contract/proposal binding')
    need(sha(BASE) == BASE_SHA, 'frozen absent owner source changed')
    return plan, raw, numeric  # load_resources receives the REAL original runtime pins.


def prepare(args, check, base, original_prepare):
    item = {'path': str(args.plan), 'sha256': args.plan_sha256}
    plan, raw, numeric = read_child(item, base)
    nested = args.out / 'e023-input-preflight'
    nested.mkdir()
    q = plan['qualification']
    _, env = original_prepare(SimpleNamespace(out=nested, plan=Path(q['path']),
        plan_sha256=q['sha256'], deadline_seconds=args.deadline_seconds), check)
    check()
    need(read_child(item, base)[0] == plan, 'E028 child plan changed during preparation')
    verified = {}
    for name, digest in {**plan['sources'], str(BASE): BASE_SHA}.items():
        check()
        path = Path(name)
        content = path.read_bytes()
        need(base.sha_bytes(content) == digest, 'adapter source changed before freezing: ' + name)
        target = args.out / 'frozen-source' / name.lstrip('/')
        target.parent.mkdir(parents=True, exist_ok=True)
        base.durable(target, content, exclusive=True, raw=True)
        verified[name] = {'sha256': digest, 'bytes': len(content), 'source_copied': True}
    base.durable(args.out / 'adapter-pins.json', verified, exclusive=True)
    base.durable(args.out / 'qualification-plan.json', raw, exclusive=True, raw=True)
    base.durable(args.out / 'adapter.json', {
        'schema': 'E028-model-owner-adapter-v1', 'status': 'PREPARED',
        'full_model_qualified': False, 'performance_qualified': False,
        'independent_e028_audit_required': True, 'child_plan_sha256': args.plan_sha256,
        'inherited_plan_sha256': ORIGINAL_PLAN_SHA, 'inherited_evidence': 'e023-input-preflight',
        'frozen_absent_owner_sha256': BASE_SHA, 'pressure_required': 1,
        'work_deadline_seconds': 1830, 'child_deadline_seconds': 1800,
        'owner_result_schema': 'E023-service-absent-owner-result-v1',
        'qualification_scope': 'E028 full-model numerical only; no HTTP/performance qualification'}, exclusive=True)
    # original_prepare supplies the pinned Python and sanitized four-variable MLX environment.
    return [numeric['execution']['python'], '-I', '-B', str(HERE / 'model_qualification.py'),
            '--plan', str(args.plan), '--plan-sha256', args.plan_sha256,
            '--out', str(args.out / 'qualification')], env


def configured(args):
    base = load_owner()
    wrapper, _, _ = validate_plan(args, base)
    original_prepare = base.load_prepare()
    base.validate_plan = lambda parsed: validate_plan(parsed, base)
    base.load_prepare = lambda: lambda parsed, check: prepare(parsed, check, base, original_prepare)
    base.NUMERICAL_SHA = wrapper['qualification']['sha256']  # receipt only; old plan remains unchanged.
    return base


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    args.plan, args.out = args.plan.absolute(), args.out.absolute()
    need(args.out.parent.is_dir() and not args.out.exists(), 'new output in existing parent required')
    base = configured(args)
    args.plan, args.out = base.absolute(args.plan), base.absolute(args.out)
    return base.run(args)


if __name__ == '__main__':
    raise SystemExit(main())
