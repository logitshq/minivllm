"""Default OFF. Thin E046 adapter to the existing bounded E045 pause owner."""
import argparse
import json
from pathlib import Path
import shutil
from types import FunctionType
import importlib.util

R = Path('/Users/ryanli/code/minivllm'); B = R/'bench/m5_ultra_256gb'; A = R/'.artifacts/m5-ultra-256gb'
HERE = Path(__file__).resolve().parent; Q = B/'gdn_prework_batched'


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


outer = load(B/'concurrency_config_v2/execute.py', '_e046_existing_pause_owner')
runner = load(HERE/'runner.py', '_e046_single_numerical_child')
sha, write = runner.sha, runner.write


def preserve_sources(plan, out):
    """Check exact runtime and retain small sources; qualifier hashes all weights twice."""
    runner.verify_pins(plan['runtime_pins'])
    rows = []
    for i, (name, digest) in enumerate(sorted(plan['pins'].items())):
        source = Path(name); assert sha(source) == digest and source.stat().st_size < 16*(1<<20)
        target = out/'sources'/f'{i:03d}-{source.name}'; target.parent.mkdir(exist_ok=True)
        shutil.copyfile(source, target); assert sha(target) == digest
        rows.append(dict(original=name, copy=str(target), sha256=digest, bytes=target.stat().st_size))
    write(out/'source-copies.json', rows)
    write(out/'runtime-pins-verified.json', plan['runtime_pins'])


def prepare():
    # Metadata/source-only, after the preceding screen has returned and restored.
    previous_plan = B/'concurrency_config_v2/prereg-v1/plan.json'
    assert sha(previous_plan) == '7b14c40ddf20381dd6fc130addfe18706eb0935d6c54fb105bea39b1667a057b', 'exact E045 v2 predecessor plan'
    previous = json.loads((A/'concurrency-config-v2-lease/lease.json').read_text())
    assert previous['closed'] is True and (A/'concurrency-config-v2-execution/final.json').is_file(), 'E045 not closed'
    resume = json.loads((A/'concurrency-config-v2-lease/resume.json').read_text())
    final = json.loads((A/'concurrency-config-v2-execution/final.json').read_text())
    waited = json.loads((A/'concurrency-config-v2-lease/job-0.json').read_text())
    assert previous['service_pid'] == outer.helper.PID and ' '.join(previous['service_identity'].split()) == outer.helper.IDENTITY
    assert resume['identity_match'] is True and resume['state_after'] and 'T' not in resume['state_after']
    assert final['service_identity_after'] == outer.helper.IDENTITY and waited['waited'] is True
    assert waited['pid'] == previous['child_pid'] and waited['command'] == previous['jobs'][0]
    assert runner.process(waited['pid']) is None and runner.process(previous['owner_pid']) is None, 'E045 owned processes remain'
    q = load(Q/'qualify.py', '_e046_original_qualifier_contract')
    old_plan = A/'long-decode-graph-v1-inputs/qualification-plan.json'
    child = dict(schema='GDN-batched-prework-real-layer-plan-v1', qualifier_sha256=sha(Q/'qualify.py'),
        candidate=dict(path=str(Q/'candidate.py'), sha256=q.CANDIDATE_SHA),
        e024_plan=dict(path=str(old_plan), sha256=q.OLD_PLAN_SHA), deadline_s=1500)
    h, old, support, candidate, original, inherited = q.validate(child)
    assert inherited['execution']['python'] == str(R/'.venv/bin/python')
    pre = HERE/'prereg-v1'; pre.mkdir(); child_path = pre/'child-plan.json'; write(child_path, child)
    refs = [Path(__file__), HERE/'runner.py', HERE/'terminal_hold.py', previous_plan,
        A/'concurrency-config-v2-lease/lease.json', A/'concurrency-config-v2-lease/resume.json',
        A/'concurrency-config-v2-lease/job-0.json',
        A/'concurrency-config-v2-execution/final.json', Q/'qualify.py', Q/'candidate.py', Q/'audit.py',
        Q.parent/'gdn_prework_unbound/audit.py', q.BASE, q.HELPER, q.OLD, q.CORE_PINS,
        Path(original['e023_qualifier']['path']), Path(original['candidate']['path']), old_plan,
        Path(original['e023_plan']['path']), child_path,
        B/'concurrency_config_v2/execute.py', B/'concurrency_config_v2/runner.py',
        B/'nvfp4_m8_register/real-prereg-v2/execute.py', B/'suspended_service.py',
        B/'resources_v2.py', B/'resources.py', B/'resource_base.py', B/'glm_flash/physical_guard.py']
    for key in ('source_manifest','model_manifest','fixtures'): refs.append(Path(inherited[key]['path']))
    core = support.pinned_json(inherited['source_manifest'])
    core_root = Path(inherited['package_root'])/'tutorial_code/miniengine'
    refs.extend(core_root/name for name in core['files'])
    pins = {str(path.resolve()):sha(path) for path in refs}
    for name,digest in core['files'].items(): assert pins[str((core_root/name).resolve())] == digest
    service = dict(pid=outer.helper.PID, identity=outer.helper.IDENTITY, process=runner.process(outer.helper.PID))
    assert outer.helper.identity() == outer.helper.IDENTITY and service['process']
    outer.helper.health()
    outputs = dict(execution=str(A/'gdn-batched-v1-execution'), screen=str(A/'gdn-batched-v1-owner'),
                   lease=str(A/'gdn-batched-v1-lease'), numerical=str(A/'gdn-batched-v1-numerical'))
    command = [str(R/'.venv/bin/python'),'-I','-B',str(Q/'qualify.py'),'--plan',str(child_path),
               '--plan-sha256',sha(child_path),'--out',outputs['numerical']]
    plan = dict(schema='E046-gdn-batched-device-plan-v1', pins=pins, runtime_pins=inherited['execution']['pins'],
        numerical_command=command, child_environment=inherited['execution']['environment'], service=service,
        screen_deadline_seconds=1650, driver_deadline_seconds=1800, max_observation_gap_seconds=12,
        physical=dict(child_limit_gib=64, system_limit_gib=240, swap_growth_bytes=0, swapouts_growth=0,
                      probe_seconds=5, interval_seconds=3, hard_allocation_cap=False), outputs=outputs,
        scope='Original B2/B8 36-call real-activation local GDN byte qualification; paused resident service; no HTTP/performance or graph composition.')
    path = pre/'plan.json'; write(path, plan)
    commands = dict(run=[str(R/'.venv/bin/python'),'-I','-B',str(HERE/'execute.py'),'--run','--plan-sha256',sha(path)],
        audit=[str(R/'.venv/bin/python'),'-I','-B',str(Q/'audit.py'),'--child',outputs['numerical'],
               '--plan-sha256',sha(child_path),'--out',str(A/'gdn-batched-v1-independent-numerical.json')])
    write(pre/'commands.json', commands)
    print(json.dumps(dict(status='PREPARED_NOT_RUN', plan_sha256=sha(path), child_plan_sha256=sha(child_path), commands=commands)))


def run(expected):
    # Reuse the exact repaired bounded cleanup functions; no shared module mutation.
    # The inherited final.json deliberately retains E045's historical generic schema.
    namespace = dict(vars(outer), HERE=HERE, runner=runner, verify_payloads=preserve_sources)
    for name in ('bind_runner','bounded_cleanup','run'):
        fn = getattr(outer, name); namespace[name] = FunctionType(fn.__code__, namespace, name, fn.__defaults__, fn.__closure__)
    code = namespace['run'](expected)
    plan = json.loads((HERE/'prereg-v1/plan.json').read_text()); out = Path(plan['outputs']['execution'])
    original = json.loads((out/'final.json').read_text())
    write(out/'e046-summary.json', dict(schema='E046-device-cycle-summary-v1', returncode=code,
        outer_status=original['status'], raw_outer_final_sha256=sha(out/'final.json'),
        independent_numerical_audit_pending=True, full_model_qualified=False, http_qualified=False,
        performance_qualified=False, scope=plan['scope']))
    return code


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); g = p.add_mutually_exclusive_group()
    g.add_argument('--prepare', action='store_true'); g.add_argument('--run', action='store_true'); p.add_argument('--plan-sha256'); a=p.parse_args()
    if a.prepare: prepare()
    elif a.run: raise SystemExit(run(a.plan_sha256))
    else: print('DESCRIPTION ONLY: after E045 closure, prepare, CPU-review, commit, then explicit run with plan SHA.')
