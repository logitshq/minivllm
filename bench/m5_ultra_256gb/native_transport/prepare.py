"""Prepare E022 from frozen E020 sources/pins; no model import or large hash."""
import argparse
import os
from pathlib import Path
import shutil
import sys
import sysconfig
import traceback

from common import (BASE, CONFIGS, DEFAULT_E020, GATE, HERE, ROOT, digest,
                    frozen_e020, launch_spec, read, write)


def prepare(out, e020_path, e020_sha):
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    write(out / 'attempt.json', {'pid': os.getpid(), 'stage': 'CPU preparation; no model/GPU/large-payload hashing'})
    e020_path = e020_path.resolve()
    assert digest(e020_path) == e020_sha, 'reviewed E020 plan mismatch'
    e020 = read(e020_path)
    assert Path(sys.executable).resolve() == Path(e020['runtime_versions']['controller-native']['python']).resolve(), 'prepare with the same interpreter as the scheduled controller'
    assert e020['schema'] == 'E020-v1' and e020['concurrency'] == 1 and e020['normal_eos']
    assert e020['warm_requests'] == e020['formal_requests'] == 3
    assert len(e020['workload']['prompt_ids']) == 19 and len(e020['oracle_ids']['native']) == 32
    complete = BASE / 'external-spec-http-v1/complete.json'
    finished = read(complete)
    assert finished['status'] == 'PASS_CONTRACT' and finished['pins_full_sha_match']
    assert all(r['valid'] for r in finished['rows']) and len(finished['rows']) == 15
    pins, reused = {}, {}
    source_dir = ROOT / 'bench/m5_ultra_256gb/external_spec_http'
    for name in ('run', 'client', 'contract', 'launch', 'native_server'):
        original = source_dir / (name + '.py')
        frozen = frozen_e020(e020_path, original)
        expected = e020['pins'][str(original)]
        assert digest(frozen) == expected
        pins[str(frozen)] = expected
        reused[name] = str(frozen)
    for name in ('resources_v2', 'resources', 'resource_base', 'suspended_service'):
        original = source_dir.parent / (name + '.py')
        frozen = frozen_e020(e020_path, original)
        expected = e020['pins'][str(original)]
        assert digest(frozen) == expected
        pins[str(frozen)] = expected
        reused[name] = str(frozen)
    for path in [e020_path, complete, *sorted(HERE.glob('*.py')), HERE / 'README.md']:
        pins[str(path)] = digest(path)
    # Actual standard library implementing the only treatment.
    stdlib = Path(sysconfig.get_path('stdlib'))
    for name in ('socketserver.py', 'socket.py', 'http/server.py', 'threading.py'):
        path = stdlib / name
        pins[str(path)] = digest(path)
    snapshots = out / 'frozen-sources'
    for filename, sha in pins.items():
        path = Path(filename)
        destination = snapshots / str(path).lstrip('/')
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        assert digest(destination) == sha
    plan = {'schema': 'E022-v1', 'status': 'PREPARED_NOT_RUN', 'configs': CONFIGS, 'gate': GATE,
        'e020_plan': str(e020_path), 'e020_plan_sha256': e020_sha,
        'inherited_pin_policy': 'Reuse all E020 model/runtime/core pins and stats by reference; full SHA before/after scheduled run, no large hash during prepare.',
        'inherited_pin_count': len(e020['pins']), 'reused_sources': reused,
        'native_server_sha256': pins[reused['native_server']],
        'pins': pins, 'pin_stats': {p: {'bytes': Path(p).stat().st_size, 'mtime_ns': Path(p).stat().st_mtime_ns} for p in pins},
        'expected_counts': {'servers': 6, 'warm_requests': 18, 'formal_requests': 18, 'formal_tokens_if_valid': 576},
        'environment': e020['environment'], 'normal_eos': True, 'concurrency': 1,
        'warm_requests': 3, 'formal_requests': 3, 'cooldown_seconds_after_each_request': 2,
        'socket_contract': 'Every accepted socket readback must match OFF=0 / ON=1. Any setup/readback/receipt failure writes permanent fatal evidence and exits whole server code 70; never readiness retry.',
        'lifecycle': 'Unmodified frozen E020 run_one/client/resource/lease helpers; direct owner PID, absolute request/probe bounds, cancel reap and outer restore.',
        'scope': 'Only TCP_NODELAY differs; E020 raw measurements remain. No payload padding, model changes, step observer or end-to-end GPU-speed claim.'}
    write(out / 'plan.json', plan)
    write(out / 'launch-preview.json', [{**c, **launch_spec(plan, e020, c['arm'],
        Path('<new-output>') / c['arm'] / f'pair{c["pair"]}-native-none', '<ephemeral-port>')} for c in CONFIGS])
    write(out / 'preparation.json', {'status': 'PASS_CPU_PREPARED_NOT_RUN', 'plan_sha256': digest(out / 'plan.json'),
        'new_small_pins': len(pins), 'inherited_pins': len(e020['pins']), 'large_hashes_deferred': True})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--e020-plan', type=Path, default=DEFAULT_E020)
    parser.add_argument('--e020-plan-sha256', required=True)
    args = parser.parse_args()
    existed = args.out.exists()
    try:
        prepare(args.out, args.e020_plan, args.e020_plan_sha256)
    except BaseException:
        if not existed and args.out.is_dir():
            write(args.out / 'failure.json', {'traceback': traceback.format_exc(), 'no_GPU_started': True})
        raise


if __name__ == '__main__':
    main()
