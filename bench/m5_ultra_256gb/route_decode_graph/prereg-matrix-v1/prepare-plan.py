"""Freeze and CPU-prepare the fixed 52-point route matrix; no service/device calls.

Exclusive output files make this a one-shot registration, not an overwrite or
retry tool. The root separately creates the real PREPARED service journal and
receipt.json before it may invoke execute.py.
"""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import importlib.util
import json
import re
import time
import traceback

R = Path('/Users/ryanli/code/minivllm')
B = R / 'bench/m5_ultra_256gb'
A = R / '.artifacts/m5-ultra-256gb'
P = B / 'route_decode_graph/prereg-matrix-v1'
NAME = 'route-decode-matrix-v1'
FIXED = {
    'route_decode_graph/qualifier_matrix.py': '6130d948450c825302d9857450204f8503c0dab449050a1bb59eaa51b77920e3',
    'route_decode_graph/candidate.py': '6531be25acb53713d6ddcd930d2c778f6640a3b1f53304fd5375d20b044525c3',
    'route_decode_graph/qualify.py': 'fe8e1c044e97885ec7c97fddce04a6e34f5b9067be4d982c5f6ad1071c59c801',
    'long_decode_isolation/qualify.py': '9d8c17a4bfc7f6ea9d61f79f466cad45059b71c9606d7b8d29c05c40213cad9e',
    'long_decode_graph/qualify.py': 'c24219f431b08502a26a4f7a83c002214bddd7ea3962d6b1869b313e4c13faf7',
    'long_decode_graph/candidate.py': 'eb14b671a66764fe677219762090acb495b89d384cf078f3f63b918a40aa2b84',
    'long_decode_graph/model-source-pins.json': '7143538a168bfcd8b01292c9687472c8647db393c6d2a510d628f71c670260c5',
    'pareto_realistic/qualify_native.py': '26964fcbed9a2e5dcefaee4888244987897f3138d4a2b7add59bf63a7461b7bf',
    'cold_cycle/lifecycle.json': 'd9417e51a781275f18c296e20ac98874e61b8ce17b8bf0902ba63e27b5b2b0a8',
    'cold_cycle/controller.py': 'b5237c458afa8324ffa6d5e6a5c49167578daec7425ff036112ec8e108d61213',
    'cold_cycle/owner.py': '4c2cc6f305de3ecf0c6495e45838baae885efbf7a8f479ed2043dc0ffb67f31c',
    'cold_cycle/audit_restore.py': '08f5a24dc9d100ca5e5b4ed469ec6587f7ba5b36b3910e6e580e7027f32b5d21',
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ref(path):
    return dict(path=str(path), sha256=sha(path))


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    for name, expected in FIXED.items():
        if sha(B / name) != expected:
            raise ValueError('frozen registration source changed: ' + name)
    for suffix in ('owner', 'cold-journal', 'execution', 'cpu-prepare'):
        if (A / (NAME + '-' + suffix)).exists():
            raise ValueError('fresh artifact destination required: ' + suffix)
    for name in ('route-plan.json', 'owner-plan.json', 'execute.py', 'preparation.json'):
        if (P / name).exists():
            raise ValueError('registration already exists: ' + name)
    owner = load(B / 'cold_cycle/owner.py', 'registered_matrix_owner')
    qualifier = load(B / 'route_decode_graph/qualifier_matrix.py', 'registered_matrix_qualifier')
    controller = owner.controller()
    old_path = B / 'long_decode_graph/prereg-v1/owner-plan.json'
    if sha(old_path) != owner.OWNER_PLAN_SHA:
        raise ValueError('original preparation plan changed')
    old = json.loads(old_path.read_bytes())
    q = dict(schema='E024-route-matrix-plan-v1', qualifier_sha256=FIXED['route_decode_graph/qualifier_matrix.py'],
             candidate=ref(B / 'route_decode_graph/candidate.py'), e024_plan=old['qualification_plan'], deadline_s=1800)
    qualifier.validate(q)
    write(P / 'route-plan.json', q)
    prior = R / 'docs/experiments/m5-ultra-256gb/E024-route-canary-v1'
    dependency_pins = {str(B / name): digest for name, digest in FIXED.items()
                       if not name.startswith('cold_cycle/')}
    # Exact inherited helpers remain explicit as well as independently checked
    # by the unchanged original preparation and the lifecycle support loader.
    dependency_pins.update({str((B / 'cold_cycle' / name).resolve()): digest
                            for name, digest in controller.SUPPORT_PINS.items()})
    dependency_pins[str(controller.BASE_PATH)] = controller.BASE_SHA
    plan = dict(schema=controller.RUN_SCHEMA, lifecycle=ref(B / 'cold_cycle/lifecycle.json'),
        predecessor=dict(**ref(prior / 'restored-journal.json'), audit=ref(prior / 'restoration-audit.json')),
        experiment=dict(phase='e024-route-matrix', qualifier=ref(B / 'route_decode_graph/qualifier_matrix.py'),
                        plan=ref(P / 'route-plan.json'), source_pins=dependency_pins),
        inherited_owner_plan=ref(old_path),
        outputs=dict(journal=str(A / (NAME + '-cold-journal')), out=str(A / (NAME + '-owner'))),
        budgets=dict(work_seconds=1830, child_seconds=1800, term_seconds=15, kill_seconds=15))
    write(P / 'owner-plan.json', plan)
    plan_sha = sha(P / 'owner-plan.json')
    owner.validate_plan(SimpleNamespace(plan=P / 'owner-plan.json', plan_sha256=plan_sha,
                                        out=Path(plan['outputs']['out'])))
    # Preserve the reviewed canary wrapper's finally-recovery algorithm. Only
    # registration paths, entrypoints, pinned plan and descriptive scope change.
    template = B / 'route_decode_graph/prereg-canary-v1/execute.py'
    source = template.read_text()
    source = source.replace('prereg-canary-v1', 'prereg-matrix-v1')
    source = source.replace('route-decode-canary-v1', NAME)
    source = source.replace('omlx_cold_restart/controller_v7.py', 'cold_cycle/controller.py')
    source = source.replace('route_decode_graph/cold_owner.py', 'cold_cycle/owner.py')
    source = re.sub(r"PLAN_SHA = '[0-9a-f]+'", "PLAN_SHA = '" + plan_sha + "'", source)
    source = source.replace('diagnostic-owner', 'matrix-owner')
    source = source.replace(
        'new E024 route-preserving canary at B1/used257: original eager257 vs compiled1023, independent identical input state; full original-bit logits and 128-state gate; every numerical/invariant failure stops; no performance run',
        'new E024 route-preserving 52-point matrix: 40 B1..8/used257,1023,1024,1280,1536 points plus uniform/mixed boundary chains and batch reuse; all original-bit logits and 128-state comparisons, ten sparse full snapshots, all core gates OFF; first failure stops; no HTTP/performance qualification')
    with (P / 'execute.py').open('x') as stream:
        stream.write(source)
    cpu = A / (NAME + '-cpu-prepare')
    cpu.mkdir(exist_ok=False)
    start = time.monotonic()
    receipt = dict(schema='E024-route-matrix-cpu-prepare-v1', status='INCOMPLETE', plan_sha256=plan_sha,
                   gpu_executed=False, service_changed=False, service_prepared=False)
    def check():
        if time.monotonic() - start > 120:
            raise TimeoutError('CPU preparation exceeded 120 seconds')
    try:
        command, env = owner.prepare(plan, cpu / 'prepared', check)
        check()
        receipt.update(status='PASS', command=command,
            mlx_environment={key: value for key, value in env.items() if key.startswith('MLX_')})
    except BaseException as error:
        receipt.update(status='FAIL', error=dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc()))
        raise
    finally:
        receipt['elapsed_seconds'] = time.monotonic() - start
        write(cpu / 'prepared.json', receipt)
    pins = owner.source_pins(plan)
    for path in (P / 'prepare-plan.py', P / 'execute.py', P / 'README.md', template,
                 B / 'cold_cycle/test_cpu.py', B / 'route_decode_graph/test_matrix_cpu.py'):
        pins[str(path)] = sha(path)
    write(P / 'preparation.json', dict(schema='E024-route-matrix-prereg-preparation-v1', status='CPU_PREPARED',
        owner_plan_sha256=plan_sha, route_plan_sha256=sha(P / 'route-plan.json'), lifecycle=plan['lifecycle'],
        cpu_prepare=ref(cpu / 'prepared.json'), source_pins=pins, contract=qualifier.fixed_plan(),
        service_prepared=False, numerical_qualified=False, performance_qualified=False))
    print(json.dumps(dict(plan_sha256=plan_sha, route_plan_sha256=sha(P / 'route-plan.json'),
        lifecycle_sha256=plan['lifecycle']['sha256'], cpu_prepare='PASS',
        journal=plan['outputs']['journal'], owner_directory=plan['outputs']['out'])))


if __name__ == '__main__':
    main()
