"""E022 six fresh native servers, frozen E020 lifecycle/client, explicit OFF/ON.

Run only in the sole GPU slot under suspended_service.py after preregistration
review/commit. Do not wrap with guarded_v2. --help imports no model runtime.
"""
import argparse
import os
from pathlib import Path
import signal
import traceback

from common import (CONFIGS, GATE, digest, launch_spec, load_lifecycle, read,
                    summarize, write)


def verify_small(plan):
    for filename, expected in plan['pins'].items():
        path = Path(filename)
        saved, stat = plan['pin_stats'][filename], path.stat()
        assert stat.st_size == saved['bytes'] and stat.st_mtime_ns == saved['mtime_ns'], filename
        assert digest(path) == expected, filename


def transport_result(result, plan, config, directory):
    """An old success receipt never overrides a later terminal socket failure."""
    failures = []
    failure = directory / 'socket-failure.json'
    receipt_path = directory / 'socket.json'
    receipt = read(receipt_path) if receipt_path.exists() else None
    if failure.exists():
        failures.append('permanent accepted-socket failure receipt exists')
    release = result.get('release') or {}
    if not receipt:
        failures.append('missing actual accepted-socket receipt')
    else:
        expected = config['arm'] == 'on'
        if not (receipt['schema'] == 'E022-socket-v2' and receipt['pid'] == release.get('pid')
                and receipt['tcp_nodelay_requested'] is expected
                and type(receipt['tcp_nodelay_actual']) is int
                and bool(receipt['tcp_nodelay_actual']) is expected
                and receipt['tcp_nodelay_actual_enabled'] is expected
                and receipt['wbufsize'] == 0 and receipt['rbufsize'] == -1
                and receipt['server_source'] == plan['reused_sources']['native_server']
                and receipt['server_source_sha256'] == plan['native_server_sha256']):
            failures.append('accepted-socket identity/option/source mismatch')
    valid = bool(result['valid'] and not failures)
    return {**config, 'valid': valid, 'native_result': str(directory / 'result.json'),
        'socket_receipt': receipt, 'transport_failures': failures,
        'native_contract_failures': result['contract_failures'],
        'diagnostic_medians_including_invalid_samples': result['diagnostic_medians_including_invalid_samples'],
        **{key: result[key] if valid else None for key in ('median_wall_s', 'median_TTFT_s', 'median_output_tok_s')}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--lease', type=Path, required=True)
    args = parser.parse_args()
    def stop(signum, frame):
        raise KeyboardInterrupt(f'E022 controller received signal {signum}')
    signal.signal(signal.SIGTERM, stop)
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    rows = []
    try:
        assert digest(args.plan) == args.plan_sha256, 'reviewed E022 plan mismatch'
        plan = read(args.plan)
        assert plan['schema'] == 'E022-v2' and plan['configs'] == CONFIGS and plan['gate'] == GATE
        verify_small(plan)
        assert digest(plan['e020_plan']) == plan['e020_plan_sha256']
        e020 = read(plan['e020_plan'])
        selected = {k: v for k, v in os.environ.items()
            if k.startswith(('MLX_', 'METAL_', 'OMLX_', 'MONOLITH_', 'LITHOS_', 'DYLD_'))
            or k in ('PYTHONPATH', 'PYTHONHOME', 'PYTHONDONTWRITEBYTECODE')}
        assert selected == plan['environment'] == e020['environment'], selected
        lifecycle = load_lifecycle(plan)
        lifecycle.lease_check(args.lease, os.getpid(), 1020)
        write(args.out / 'attempt.json', {'pid': os.getpid(), 'parent_pid': os.getppid(),
            'plan': str(args.plan.resolve()), 'plan_sha256': args.plan_sha256,
            'lease': str(args.lease.resolve()), 'configs': CONFIGS})
        write(args.out / 'plan.json', plan)
        lifecycle.verify_pins(e020, full=True)
        write(args.out / 'pins-before.json', {'inherited_all_full_sha_match': True, 'new_pins_match': True})
        for arm in ('off', 'on'):
            (args.out / arm).mkdir()
        for config in CONFIGS:
            verify_small(plan)
            lifecycle.verify_pins(e020, full=False)
            def spec_adapter(parent_plan, parent_arm, out, port, key):
                assert parent_plan is e020 and parent_arm == 'native-none'
                return launch_spec(plan, e020, config['arm'], out, port)
            lifecycle.specification = spec_adapter
            print('START', config, flush=True)
            result = lifecycle.run_one(e020, {'pair': config['pair'], 'arm': 'native-none'},
                                       args.out / config['arm'], args.lease)
            directory = args.out / config['arm'] / f'pair{config["pair"]}-native-none'
            row = transport_result(result, plan, config, directory)
            write(directory / 'transport-result.json', row)
            rows.append(row)
            write(args.out / 'progress.json', summarize(rows))
            if row['transport_failures']:
                raise RuntimeError('accepted-socket contract failure; no replacement or continued matrix')
            print({k: row[k] for k in ('pair', 'arm', 'valid', 'median_wall_s', 'median_output_tok_s')}, flush=True)
        verify_small(plan)
        lifecycle.verify_pins(e020, full=True)
        write(args.out / 'pins-after.json', {'inherited_all_full_sha_match': True, 'new_pins_match': True})
        result = summarize(rows)
        result.update(status='PASS_ADVANCE' if result['advance'] else 'FAIL_GATE_RESULTS_RETAINED',
            inherited_pins_full_sha_match=True, new_pins_sha_match=True,
            service_restore='Audit outer lease resume.json after controller exit; not asserted here.')
        write(args.out / 'complete.json', result)
        print(result['status'], flush=True)
    except BaseException as error:
        write(args.out / 'failure.json', {'type': type(error).__name__, 'text': str(error),
            'traceback': traceback.format_exc(), 'completed': summarize(rows)})
        raise


if __name__ == '__main__':
    main()
