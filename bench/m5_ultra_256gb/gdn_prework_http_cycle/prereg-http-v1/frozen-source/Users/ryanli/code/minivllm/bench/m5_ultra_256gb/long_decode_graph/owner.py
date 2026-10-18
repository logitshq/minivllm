"""E024 fixed preparation adapter for the frozen E023 bounded child owner.

Separate schema E024-diagnostic-owner-v1: qualification_plan {path,sha256},
inherited_plan {path,sha256}, execution {python,qualifier,pins,environment}.
The strict six-field E024 qualifier plan remains unchanged. Evidence from
the real inherited E023 prepare is retained in e023-input-preflight/; source
and runtime hashes are not replaced with synthetic plans. No MLX imports.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
LEGACY = HERE.parent / 'pareto_realistic/qualify_owner.py'
LEGACY_SHA = '1efaf440e991fed64dedf15455be2d37223e709fec41bb07e8c0295a2975c1c2'
QUALIFIER_SHA = 'c24219f431b08502a26a4f7a83c002214bddd7ea3962d6b1869b313e4c13faf7'
CANDIDATE_SHA = 'eb14b671a66764fe677219762090acb495b89d384cf078f3f63b918a40aa2b84'
MODEL_SOURCE_SHA = '7143538a168bfcd8b01292c9687472c8647db393c6d2a510d628f71c670260c5'
SUPPORT_SHA = '26964fcbed9a2e5dcefaee4888244987897f3138d4a2b7add59bf63a7461b7bf'


def write(out, name, value):
    (out / name).write_text(json.dumps(value, indent=2) + '\n')


def load_legacy():
    if hashlib.sha256(LEGACY.read_bytes()).hexdigest() != LEGACY_SHA:
        raise ValueError('frozen E023 owner SHA mismatch')
    spec = importlib.util.spec_from_file_location('e024_fixed_e023_owner', LEGACY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare(args, check, legacy_prepare):
    verified = {}
    def pin(path, expected):
        path = Path(path)
        if not path.is_absolute() or '..' in path.parts:
            raise ValueError('absolute pin path without traversal required')
        copied = path.suffix in ('.py', '.json', '.md')
        target = args.out / 'frozen-source' / str(path).lstrip('/')
        if copied:
            target.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        with path.open('rb') as stream, (target.open('wb') if copied else open(os.devnull, 'wb')) as dest:
            while chunk := stream.read(8 << 20):
                check()
                digest.update(chunk)
                if copied:
                    dest.write(chunk)
        if digest.hexdigest() != expected:
            raise ValueError('E024 pin mismatch: ' + str(path))
        verified[str(path)] = {'sha256': expected, 'bytes': path.stat().st_size, 'source_copied': copied}
        write(args.out, 'verified-pins.json', verified)
    def pinned_json(item):
        if set(item) != {'path', 'sha256'}:
            raise ValueError('path/SHA pair required')
        pin(item['path'], item['sha256'])
        return json.loads(Path(item['path']).read_text())
    plan = pinned_json({'path': str(args.plan), 'sha256': args.plan_sha256})
    if (set(plan) != {'schema', 'qualification_plan', 'inherited_plan', 'execution'}
            or plan['schema'] != 'E024-diagnostic-owner-v1'):
        raise ValueError('exact E024 owner plan fields required')
    qualification = pinned_json(plan['qualification_plan'])
    inherited = pinned_json(plan['inherited_plan'])
    execution = plan['execution']
    if (set(qualification) != {'schema', 'qualifier_sha256', 'e023_plan', 'e023_qualifier', 'candidate', 'deadline_s'}
            or qualification['schema'] != 'E024-diagnostic-qualification-v1'
            or qualification['qualifier_sha256'] != QUALIFIER_SHA
            or qualification['e023_plan'] != plan['inherited_plan']
            or qualification['e023_qualifier'] != inherited['execution']['qualifier']
            or execution['python'] != inherited['execution']['python']
            or execution['environment'] != inherited['execution']['environment']):
        raise ValueError('E024/E023 plan, executable or environment binding differs')
    budget = qualification['deadline_s']
    if type(budget) not in (int, float) or not math.isfinite(budget) or not 0 < budget <= min(1800, args.deadline_seconds):
        raise ValueError('invalid E024 child deadline')
    expected_sources = {str(LEGACY): LEGACY_SHA,
        str(HERE / 'qualify.py'): QUALIFIER_SHA, str(HERE / 'candidate.py'): CANDIDATE_SHA,
        str(HERE / 'model-source-pins.json'): MODEL_SOURCE_SHA,
        str(LEGACY.with_name('qualify_native.py')): SUPPORT_SHA}
    if (execution['qualifier'] != {'path': str(HERE / 'qualify.py'), 'sha256': QUALIFIER_SHA}
            or qualification['candidate'] != {'path': str(HERE / 'candidate.py'), 'sha256': CANDIDATE_SHA}):
        raise ValueError('only fixed adjacent E024 qualifier/candidate may execute')
    required = set(expected_sources) | {str(HERE / n) for n in ('owner.py', 'test_owner_cpu.py', 'test_cpu.py', 'test_qualify_cpu.py')}
    if (not required <= set(execution['pins'])
            or any(execution['pins'].get(p) != sha for p, sha in expected_sources.items())
            or any(execution['pins'].get(p) != sha for p, sha in inherited['execution']['pins'].items())):
        raise ValueError('missing/different E024 source or inherited runtime pins')
    # This is the original plan, not an invented plan modified for the adapter.
    nested = args.out / 'e023-input-preflight'
    nested.mkdir()
    proxy = SimpleNamespace(plan=Path(plan['inherited_plan']['path']), plan_sha256=plan['inherited_plan']['sha256'],
                            out=nested, deadline_seconds=args.deadline_seconds)
    _, env = legacy_prepare(proxy, check)
    previous = json.loads((nested / 'verified-pins.json').read_text())
    for path, expected in execution['pins'].items():
        if path in previous and previous[path]['sha256'] == expected:
            verified[path] = {**previous[path], 'inherited_evidence': 'e023-input-preflight/verified-pins.json'}
        else:
            pin(path, expected)
    write(args.out, 'verified-pins.json', verified)
    write(args.out, 'plan.json', plan)
    write(args.out, 'qualification-plan.json', qualification)
    write(args.out, 'preparation.json', {'legacy_owner_sha256': LEGACY_SHA,
          'inherited_evidence': 'e023-input-preflight', 'same_python_and_clean_environment': True})
    return [execution['python'], '-I', '-B', execution['qualifier']['path'], '--plan', plan['qualification_plan']['path'],
            '--plan-sha256', plan['qualification_plan']['sha256'], '--out', str(args.out / 'qualification')], env


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--suspended-pid', type=int, required=True)
    parser.add_argument('--deadline-seconds', type=float, required=True)
    args = parser.parse_args(argv)
    if not math.isfinite(args.deadline_seconds) or not 0 < args.deadline_seconds <= 1830 or args.suspended_pid <= 0:
        parser.error('positive PID and finite deadline <=1830 required')
    args.plan, args.out = args.plan.absolute(), args.out.absolute()
    legacy = load_legacy()
    original_prepare = legacy.prepare
    legacy.prepare = lambda parsed, check: prepare(parsed, check, original_prepare)
    return legacy.run(args)


if __name__ == '__main__':
    raise SystemExit(main())
