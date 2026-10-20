"""Default-inert E052 adapter to the repaired E045 pause/restore owner."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from types import FunctionType

HERE = Path(__file__).resolve().parent
B = HERE.parent; R = B.parent.parent; A = R / '.artifacts/m5-ultra-256gb'
BASE = B / 'concurrency_config_v2/execute.py'
BASE_SHA = '34a604c62f6de4eddcff2be53abd2d72716384ee67d13d1bf963869d35695a20'


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


assert hashlib.sha256(BASE.read_bytes()).hexdigest() == BASE_SHA
outer = load(BASE, '_e052_pause_owner')
runner = load(HERE / 'runner.py', '_e052_resource_owner')
sha, write = runner.sha, runner.write


def prepare():
    # Metadata only: no pause, model import, model hashing, or new live listener.
    predecessor = B / 'c8_batched_full_v1/prereg-v1/plan.json'
    assert sha(predecessor) == 'c895c6e85c9c6e7ef8928f749086fd8012777a67c769b20b3cd13d7eaae4f618'
    previous = json.loads(predecessor.read_text())['outputs']
    lp = Path(previous['lease']); ep = Path(previous['execution'])
    lease, job, resume, final = [json.loads(p.read_text()) for p in
        (lp / 'lease.json', lp / 'job-0.json', lp / 'resume.json', ep / 'final.json')]
    assert lease['closed'] is True and job['waited'] is True and job['returncode'] == 0
    assert job['pid'] == lease['child_pid'] and job['command'] == lease['jobs'][0]
    assert resume['identity_match'] is True and resume['state_after'] and 'T' not in resume['state_after']
    assert final['status'] == 'PASS_CAPTURE_AND_RESTORE' and final['service_identity_after'] == outer.helper.IDENTITY
    assert lease['service_pid'] == outer.helper.PID and ' '.join(lease['service_identity'].split()) == outer.helper.IDENTITY
    assert runner.process(job['pid']) is None and runner.process(lease['owner_pid']) is None
    trial = load(HERE / 'trial.py', '_e052_prepare_trial')
    base = A / 'gdn-graph-combined-formal-prepare-v1/specs/r1-c8-off.json'
    actual=Path(previous['numerical'])/'complete.json'
    audit_path=A/'c8-batched-full-v1-independent-audit-attempt1/audit.json'
    audit=json.loads(audit_path.read_text())
    assert audit['status']=='PASS' and audit['full_history_qualified'] is True
    assert audit['performance_qualified'] is False
    assert audit['plan_sha256']==json.loads(predecessor.read_text())['numerical_plan_sha256']
    resource_path=R/'docs/experiments/m5-ultra-256gb/E051-c8-batched-gdn-graph-full-v1/resource-review.json'
    resource=json.loads(resource_path.read_text());assert resource['status']=='PASS_SAVED_OBSERVATIONS'
    spec = trial.prepare(dict(path=str(base), sha256=sha(base)), dict(path=str(actual),sha256=sha(actual)))
    pre = HERE / 'prereg-v1'; pre.mkdir(); (pre / 'specs').mkdir()
    path = pre / 'specs/c8-batched-mb50-two-waves.json'; trial.write(path, spec)
    refs = [dict(path=str(path), sha256=sha(path))]
    pins = dict(spec['sources'])
    for p in [*HERE.glob('*.py'), BASE, B / 'concurrency_config_v2/runner.py',
              B / 'nvfp4_m8_register/real-prereg-v2/execute.py', B / 'suspended_service.py',
              B / 'resources_v2.py', B / 'resources.py', B / 'resource_base.py',
              B / 'glm_flash/physical_guard.py', predecessor, base, path, actual, audit_path, resource_path,
              lp / 'lease.json', lp / 'job-0.json', lp / 'resume.json', ep / 'final.json']:
        pins[str(p)] = sha(p)
    numeric = Path(spec['base_plan']['path']); pins[str(numeric)] = spec['base_plan']['sha256']
    q = json.loads(numeric.read_text())
    for ref in (q['source_manifest'], q['model_manifest'], q['fixtures']): pins[ref['path']] = ref['sha256']
    service = dict(pid=outer.helper.PID, identity=outer.helper.IDENTITY, process=runner.process(outer.helper.PID))
    assert outer.helper.identity() == outer.helper.IDENTITY and service['process']
    plan = dict(schema='E052-c8-batched-http-plan-v1', pins=pins, specs=refs, service=service,
        numeric_plan=str(numeric), screen_deadline_seconds=1740, driver_deadline_seconds=1890,
        max_observation_gap_seconds=12,
        physical=dict(child_limit_gib=64, system_limit_gib=240, swap_growth_bytes=0,
            swapouts_growth=0, probe_seconds=5, interval_seconds=3, hard_allocation_cap=False),
        outputs=dict(execution=str(A / 'c8-batched-http-v1-execution'),
                     screen=str(A / 'c8-batched-http-v1-screen'), lease=str(A / 'c8-batched-http-v1-lease')),
        scope='Instrumented two-wave C8 batched GDN+graph MB50 capture; observer may alter arrival/schedule. Independent replay of this new history required; no performance point.')
    write(pre / 'plan.json', plan)
    child = Path(plan['outputs']['screen']) / 'trials' / spec['label']
    commands = dict(run=[str(R / '.venv/bin/python'), '-I', '-B', str(HERE / 'execute.py'),
        '--run', '--plan-sha256', sha(pre / 'plan.json')], join=[str(R / '.venv/bin/python'), '-I', '-B',
        str(HERE / 'trial.py'), '--join-child', str(child), '--spec-sha256', sha(path)])
    write(pre / 'commands.json', commands)
    print(json.dumps(dict(status='PREPARED_NOT_RUN', plan_sha256=sha(pre / 'plan.json'), commands=commands)))


def run(expected):
    scope = dict(vars(outer), HERE=HERE, runner=runner)
    for name in ('bind_runner', 'bounded_cleanup', 'run'):
        fn = getattr(outer, name)
        scope[name] = FunctionType(fn.__code__, scope, name, fn.__defaults__, fn.__closure__)
    return scope['run'](expected)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); g = p.add_mutually_exclusive_group()
    g.add_argument('--prepare', action='store_true'); g.add_argument('--run', action='store_true')
    p.add_argument('--plan-sha256'); a = p.parse_args()
    if a.prepare: prepare()
    elif a.run: raise SystemExit(run(a.plan_sha256))
    else: print('DESCRIPTION ONLY: prepare, review/CPU/commit, explicit run with plan SHA.')
