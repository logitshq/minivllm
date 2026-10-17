"""E027: bounded local numerical gate, never a performance benchmark.

Reuses the independently exercised E026 resource/physical/lifecycle checks.
The outer authorized suspended_service lease alone controls the other server.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
spec = importlib.util.spec_from_file_location('e026_owner_checks', BASE / 'conv_tail_owned_copy/layer_owner.py')
checks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checks)


def prepare(args, check):
    checks.need(checks.sha(args.plan, check) == args.plan_sha256, 'plan SHA mismatch')
    plan = json.loads(args.plan.read_text())
    checks.need(set(plan) == {'schema', 'execution', 'bundle', 'resource_policy', 'expected_cases', 'expected_comparisons', 'scope'}, 'plan fields')
    checks.need(plan['schema'] == 'E027-shared-W-numeric-v1' and plan['expected_cases'] == 33 and plan['expected_comparisons'] == 66, 'numeric scope')
    checks.need(checks.equal(plan['resource_policy'], checks.POLICY), 'fixed numerical resource policy')
    ex = plan['execution']
    checks.need(set(ex) == {'python', 'screen', 'pins', 'environment'}, 'execution fields')
    checks.need(checks.equal(ex['environment'], checks.ENV), 'fixed environment')
    checks.need(Path(ex['screen']) == HERE / 'screen.py' and Path(ex['python']).is_absolute() and os.access(ex['python'], os.X_OK), 'screen/python')
    bundle = Path(plan['bundle']['path'])
    checks.need(bundle.is_absolute() and checks.sha(bundle, check) == plan['bundle']['sha256'], 'bundle manifest')
    pins = ex['pins']
    required = {HERE / p for p in ('owner.py', 'screen.py', 'candidate.py')}
    required |= {BASE / p for p in ('resources_v2.py', 'resources.py', 'resource_base.py', 'suspended_service.py', 'conv_tail_owned_copy/layer_owner.py', 'conv_tail_owned_copy/layer_screen.py')}
    required |= {Path(ex['python']).resolve(), bundle, bundle.parent / 'weights.bin'}
    checks.need(required <= {Path(p) for p in pins}, 'required pins missing')
    for fragment in ('/mlx/core.', '/mlx/lib/libmlx.dylib', '/mlx/lib/mlx.metallib', '/numpy/__init__.py', '/numpy/_core/_multiarray_umath.'):
        checks.need(any(fragment in p for p in pins), 'missing runtime pin ' + fragment)
    checks.verify_pins(pins, args.out, 'pins-before.json', check, freeze=True)
    checks.write(args.out, 'plan.json', plan)
    env = {k: v for k, v in os.environ.items() if not k.startswith(('MLX_', 'METAL_', 'DYLD_', 'OMLX_', 'MONOLITH_', 'LITHOS_')) and k not in ('PYTHONPATH', 'PYTHONHOME')}
    env.update(ex['environment'])
    return plan, env


def run(args):
    args.out.mkdir(parents=True, exist_ok=False)
    budget, samples, physical, failures = checks.Budget(), [], [], []
    child = plan = resource = None
    child_dir = args.out / 'child'
    previous = {s: signal.getsignal(s) for s in (signal.SIGALRM, signal.SIGINT, signal.SIGTERM)}
    def stop(signum, frame):
        if signum == signal.SIGALRM: raise TimeoutError('E027 owner deadline')
        raise InterruptedError(signal.Signals(signum).name)
    def failure(stage):
        failures.append({'stage': stage, 'traceback': traceback.format_exc()})
        checks.write(args.out, 'failures.json', failures)
    def probe(label, process=None):
        row = resource.snapshot(label, process.pid if process else None, suspended_pids=(74384,))
        samples.append(row)
        checks.write(args.out, 'resources.json', samples)
        terminal = process is not None and process.poll() == 0
        row['terminal_exit0_observed'] = terminal
        row['owner_monotonic_s'] = time.monotonic()
        checks.guard(samples, process.pid if process else None, terminal_exit0=terminal)
        checks.write(args.out, 'resources.json', samples)
        if process is not None:
            index = len(physical)
            physical.append(checks.footprint(process, args.out, index))
            row['physical_receipt'] = f'footprint-{index:03d}.json'
            checks.write(args.out, 'resources.json', samples)
    try:
        for sig in previous: signal.signal(sig, stop)
        budget.alarm()
        plan, env = prepare(args, budget.check)
        sys.path.insert(0, str(BASE))
        import resources_v2 as resource_module
        resource = resource_module
        probe('preflight')
        child_dir.mkdir()
        command = [plan['execution']['python'], '-I', '-B', plan['execution']['screen'], '--run', '--out', str(child_dir / 'screen'), '--bundle-manifest', plan['bundle']['path'], '--bundle-sha256', plan['bundle']['sha256']]
        budget.arm()
        with (child_dir / 'stdout.txt').open('w') as stdout, (child_dir / 'stderr.txt').open('w') as stderr:
            child = subprocess.Popen(command, env=env, stdout=stdout, stderr=stderr)
            checks.write(child_dir, 'launch.json', {'pid': child.pid, 'ppid': os.getpid(), 'command': command, 'environment': plan['execution']['environment'], 'arm_seconds': 120, 'spawned_monotonic': time.monotonic(), 'arm_deadline_monotonic': budget.arm_end})
            while child.poll() is None:
                budget.check(); probe('running', child); budget.check()
                try: child.wait(timeout=1)
                except subprocess.TimeoutExpired: pass
            child.wait(timeout=0)
        checks.need(child.returncode == 0, 'numerical child failed')
        complete = json.loads((child_dir / 'screen/complete.json').read_text())
        checks.need(complete['status'] == 'PASS' and complete.get('device_run') is True
                    and complete.get('pid') == child.pid and complete.get('ppid') == os.getpid(), 'incomplete/wrong numeric child')
        screen_spec = importlib.util.spec_from_file_location('e027_cpu_contract', HERE / 'screen.py')
        screen = importlib.util.module_from_spec(screen_spec); screen_spec.loader.exec_module(screen)
        checks.need(checks.equal(complete['contract'], screen.contract(True)), 'child contract mismatch')
        cases = complete['cases']
        checks.need([r['case'] for r in cases] == [r['case'] for r in screen.contract(True)['cases']]
                    and all(r['status'] == 'PASS' and r['inputs_unchanged'] is True for r in cases), 'case coverage/failure')
        checks.need(len(complete['checks']) == 66 and all(r['exact'] is True for r in complete['checks'])
                    and len(complete['physical_receipts']) == 68 and len(complete['bank_receipts']) == 11, 'comparison/phase/bank coverage')
        checks.need(complete.get('source_pins_rechecked') is True and complete.get('bundle_rechecked') is True
                    and complete.get('performance_qualified') is False and complete.get('full_model_qualified') is False, 'wrong qualification scope')
        checks.need(all(plan['execution']['pins'].get(p) == h for p, h in complete['sources'].items()), 'screen sources differ from plan')
        checks.need(complete['bundle_manifest_sha256'] == plan['bundle']['sha256'], 'child bundle differs')
        checks.need(0 < complete['elapsed_wall_s'] <= 120, 'child deadline exceeded')
        checks.write(args.out, 'child-receipt.json', {'complete_sha256': checks.sha(child_dir / 'screen/complete.json'), 'scope': 'child local check; independent raw audit still required'})
        checks.reap(child, child_dir); child = None
        budget.arm_end = None; budget.alarm()
        probe('after-child')
        checks.verify_pins(plan['execution']['pins'], args.out, 'pins-after.json', budget.check)
        checks.need(checks.sha(args.plan, budget.check) == args.plan_sha256, 'plan changed')
    except BaseException:
        failure('execution')
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGINT, signal.SIG_IGN); signal.signal(signal.SIGTERM, signal.SIG_IGN)
        try:
            budget.cleanup_alarm()
            if child is not None:
                try: checks.reap(child, child_dir)
                except BaseException: failure('reap')
            if resource is not None:
                try: probe('postflight')
                except BaseException: failure('postflight')
            if plan is not None and not (args.out / 'pins-after.json').exists():
                try: checks.verify_pins(plan['execution']['pins'], args.out, 'pins-after.json', budget.cleanup_check)
                except BaseException: failure('pins-after')
        except BaseException:
            failure('cleanup')
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            for sig, handler in previous.items(): signal.signal(sig, handler)
        passed = not failures and (args.out / 'child-receipt.json').exists()
        checks.write(args.out, 'complete.json', {'schema': 'E027-shared-W-owner-v1', 'status': 'PASS' if passed else 'FAIL', 'owner_pid': os.getpid(), 'failures': failures, 'samples': len(samples), 'physical_samples': len(physical), 'elapsed_wall_s': time.monotonic() - budget.start, 'performance_qualified': False, 'raw_audit': 'PENDING', 'service_restoration': 'outer lease'})
    return int(not passed)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--plan', type=Path, required=True)
    ap.add_argument('--plan-sha256', required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    args.plan, args.out = args.plan.absolute(), args.out.absolute()
    return run(args)


if __name__ == '__main__':
    raise SystemExit(main())
