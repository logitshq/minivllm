"""E025 v2 bounded same-split numerical screen; default/help is CPU-only.

Explicit --run --out NEW runs three T19 projections and six fallback lengths
per projection: 33 exact comparisons, 68 resource phases on PASS. Every comparison
stores both original-dtype arrays, including successful prefixes and fallbacks.
--fixtures MANIFEST --fixtures-sha256 SHA optionally supplies the old fixed input
contract; otherwise the original seed and synthetic construction are unchanged.
No model loading, performance measurement or HTTP qualification is performed.
The 120s Python alarm needs an external hard owner for stalled native/GPU calls.
Memory observations (MLX 1GiB, physical 2GiB, cache0) are not OS hard caps.
"""
import argparse
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
LEDGER_PATH = HERE.parent / 'shared_w_m1/screen.py'
PHYSICAL_PATH = HERE.parent / 'conv_tail_owned_copy/layer_screen.py'
FIXED_PINS = {
    HERE / 'candidate.py': 'c67a3478e0d12f81f982c83a68ccc1fc4f2a336e87f6f5af0b28699b41f08b3e',
    HERE / 'installed-sources.json': '9e72dd0a1a0091cd5c239a853c574bbf23583a6d7651f886b41b5f7cee2e8364',
    HERE / 'LICENSE.MLX': 'ccfab7ccb2ea306f71531c8ca77bb55507606cd90768b1e32b8b52ab5b48cf01',
    LEDGER_PATH: 'd01f28e36b07176e73830b424f3ab65113f1263cc1f810678f2c88b50c11dacf',
    PHYSICAL_PATH: 'a0423d0648b19236f972598ce5acb5e8e9cdd617fadb2646f1c7fef01d9ec302',
}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


need(sha(HERE / 'candidate.py') == FIXED_PINS[HERE / 'candidate.py'], 'fixed candidate SHA')
c = load('e025_v2_candidate', HERE / 'candidate.py')


def source_pins():
    for path, expected in FIXED_PINS.items():
        need(sha(path) == expected, 'fixed source SHA: ' + str(path))
    return {**{str(p): h for p, h in FIXED_PINS.items()}, str(Path(__file__).resolve()): sha(__file__)}


def fixed_plan():
    return dict(schema='E025-same-split-screen-v2', normal_M=list(c.NORMAL_M),
        fallback_M=list(c.FALLBACK_M), shapes=[dict(name=n, K=k, N=o) for n, k, o in c.SHAPES],
        geometries=[c.geometry(m, k, n) for _, k, n in c.SHAPES for m in c.NORMAL_M],
        dtype='bfloat16', quantization=dict(mode='nvfp4', bits=4, group_size=16),
        seed=20261017, cache_limit_bytes=0, mlx_observation_budget_bytes=1 << 30,
        physical_observation_budget_bytes=2 << 30, child_alarm_seconds=120,
        physical_reading_semantics='separate-int-bounds;current=max(top,aux-current);peak=max(top,aux-peak);aux-peak>=aux-current',
        expected_checks=33, expected_phases=68, expected_kernels=6,
        normal_cases=3, fallback_cases=18, only_numerical_diagnostic=True,
        default_OFF=True, performance_qualified=False, full_model_qualified=False,
        observed_limits_not_OS_hard_caps=True,
        phase_order='initial; each check before_check,after_check; final_release only on PASS',
        check_order=['OFF-vs-original', 'reference-vs-original-QMM', 'every-partial',
                     'final', 'ON-callable', 'six unregistered-fallback lengths'],
        rawpair_keys=['reference', 'candidate'],
        gate='original native QMM first; finite and exact original BF16 bytes, shape and dtype; no tolerance')


def fixture_manifest(path, expected_sha):
    need(expected_sha and sha(path) == expected_sha, 'fixture manifest SHA mismatch')
    data = json.loads(Path(path).read_text())
    need(set(data) == {'schema', 'cases'} and data['schema'] == 'E025-projection-inputs-v1'
         and set(data['cases']) == {n for n, _, _ in c.SHAPES}, 'exact three-case fixture manifest required')
    for item in data['cases'].values():
        need(set(item) == {'path', 'sha256'} and sha(item['path']) == item['sha256'], 'fixture tensor SHA mismatch')
    return data


def stop_signal(signum, frame=None):
    if signum == signal.SIGALRM:
        raise TimeoutError('120s E025-v2 screen SIGALRM deadline')
    raise InterruptedError('E025-v2 screen cancellation ' + signal.Signals(signum).name)


def record_pair(mx, out, index, label, left, right):
    """Save both arrays before judging the gate; returned row never hides a mismatch."""
    mx.eval(left, right)
    raw = out / 'pairs' / f'{index:03d}.safetensors'
    raw.parent.mkdir(exist_ok=True)
    mx.save_safetensors(str(raw), {'reference': left, 'candidate': right})
    same_meta = left.shape == right.shape and left.dtype == right.dtype
    left_finite = bool(mx.all(mx.isfinite(left)).item())
    right_finite = bool(mx.all(mx.isfinite(right)).item())
    bits = bool(mx.array_equal(left.view(mx.uint8), right.view(mx.uint8)).item()) if same_meta else False
    row = dict(index=index, label=label, left_shape=list(left.shape), right_shape=list(right.shape),
        left_dtype=str(left.dtype), right_dtype=str(right.dtype), left_finite=left_finite,
        right_finite=right_finite, finite=left_finite and right_finite, bytes_equal=bits,
        metadata_equal=same_meta, rawpair=dict(path=str(raw.relative_to(out)), sha256=sha(raw)),
        status='PASS' if same_meta and left_finite and right_finite and bits else 'FAIL')
    if same_meta and not bits:
        bad = (left.view(mx.uint8) != right.view(mx.uint8)).reshape(-1)
        row['first_differing_byte_index'] = int(mx.argmax(bad.astype(mx.int32)).item())
    if row['status'] != 'PASS' and not (out / 'first-difference.json').exists():
        write(out / 'first-difference.json', row)
    return row


def input_snapshot(mx, out, name, arrays, suffix):
    path = out / f'{name}-input-{suffix}.safetensors'
    mx.save_safetensors(str(path), arrays)
    return dict(path=path.name, sha256=sha(path))


def execute(mx, out, result, captures, ledger, emit):
    enabled, disabled = c.SameSplitPartial(mx, enabled=True), c.SameSplitPartial(mx)
    need(enabled._hardware(), 'requires MLX0.32.3 and Apple M5 Ultra applegpu_g17d')
    result['device'] = dict(name=mx.device_info().get('device_name'),
                            architecture=mx.metal.device_info().get('architecture'))
    mx.random.seed(result['contract']['seed'])
    active_input = None

    def equal(label, left, right):
        ledger.phase('before_check', label)
        row = record_pair(mx, out, len(result['checks']), label, left, right)
        result['checks'].append(row)
        emit(dict(event='array-comparison', **row))
        write(out / 'checks.json', result['checks'])
        need(row['status'] == 'PASS', 'strict numerical gate rejected: ' + label)
        ledger.phase('after_check', label)

    def close_input():
        name, arrays, row = active_input
        row['after'] = input_snapshot(mx, out, name, arrays, 'after')
        row['bytes_equal'] = row['before']['sha256'] == row['after']['sha256']
        write(out / 'inputs.json', result['inputs'])
        emit(dict(event='input-unchanged', **row))
        need(row['bytes_equal'], 'input tensor bytes or serialization changed')

    try:
        for name, K, N in c.SHAPES:
            emit(dict(event='begin-projection', name=name, K=K, N=N))
            if captures:
                arrays = mx.load(captures['cases'][name]['path'])
            else:
                # Preserve v1 RNG calls, construction order, full128-row input, and seed.
                lo = mx.random.randint(0, 65536, (N, K // 8), dtype=mx.uint32)
                hi = mx.random.randint(0, 65536, (N, K // 8), dtype=mx.uint32)
                arrays = {'activation': (mx.random.normal((128, K)) / 16).astype(mx.bfloat16),
                          'weight': lo | (hi << 16),
                          'scales': mx.random.randint(0x30, 0x41, (N, K // 16), dtype=mx.uint8)}
                del lo, hi
            need(set(arrays) == {'activation', 'weight', 'scales'}, 'exact input tensor names required')
            x, w, s = (arrays[k] for k in ('activation', 'weight', 'scales'))
            need(x.shape == (128, K) and w.shape == (N, K // 8) and s.shape == (N, K // 16)
                 and x.dtype == mx.bfloat16 and w.dtype == mx.uint32 and s.dtype == mx.uint8, 'input shape/dtype mismatch')
            mx.eval(x, w, s)
            row = dict(name=name, before=input_snapshot(mx, out, name, arrays, 'before'),
                       x_shape=list(x.shape), x_dtype=str(x.dtype), synthetic_only=captures is None)
            result['inputs'].append(row)
            active_input = (name, arrays, row)
            row['x_finite'] = bool(mx.all(mx.isfinite(x)).item())
            write(out / 'inputs.json', result['inputs'])
            need(row['x_finite'], 'nonfinite activation input')
            for M in c.NORMAL_M:
                label = f'{name}-M{M}'
                values = {'original': enabled.original(x[:M], w, s), 'default_OFF': disabled(x[:M], w, s)}
                equal(label + '/OFF-vs-original', values['original'], values['default_OFF'])
                values['reference_partial'] = enabled.partials(x[:M], w, s, 'reference')
                values['reference_final'] = mx.sum(values['reference_partial'], axis=0)
                equal(label + '/reference-vs-original-QMM', values['original'], values['reference_final'])
                values['candidate_partial'] = enabled.partials(x[:M], w, s, 'candidate')
                equal(label + '/every-partial', values['reference_partial'], values['candidate_partial'])
                values['candidate_final'] = mx.sum(values['candidate_partial'], axis=0)
                equal(label + '/final', values['original'], values['candidate_final'])
                values['enabled_callable'] = enabled(x[:M], w, s)
                equal(label + '/ON-callable', values['original'], values['enabled_callable'])
                result['normal_cases_passed'] += 1
                del values
            before = len(enabled.receipts)
            for M in c.FALLBACK_M:
                equal(f'{name}-M{M}/unregistered-fallback', enabled.original(x[:M], w, s), enabled(x[:M], w, s))
                result['fallback_cases_passed'] += 1
            need(len(enabled.receipts) == before and not disabled.receipts, 'OFF/fallback compiled custom kernel')
            close_input()
            active_input = None
            del arrays, x, w, s
            mx.synchronize()
            mx.clear_cache()
    finally:
        primary = sys.exc_info()[0] is not None
        try:
            if active_input is not None:
                close_input()
        except BaseException as error:
            evidence = dict(type=type(error).__name__, message=str(error), primary_failure_preserved=primary)
            write(out / 'input-exit-error.json', evidence)
            if not primary:
                raise
        finally:
            write(out / 'kernel-source-receipts.json', enabled.receipts)
            result['kernel_count'] = len(enabled.receipts)
            enabled.kernels.clear()
            disabled.kernels.clear()


def recheck_sources(before, captures, args):
    need(source_pins() == before, 'screen/support source changed')
    for geometry in fixed_plan()['geometries']:
        for kind in ('reference', 'candidate'):
            c.build_source(kind, geometry, c.installed_root())
    if captures:
        need(fixture_manifest(args.fixtures, args.fixtures_sha256) == captures, 'fixture manifest changed')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--describe', action='store_true')
    mode.add_argument('--run', action='store_true')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--fixtures', type=Path)
    parser.add_argument('--fixtures-sha256')
    args = parser.parse_args(argv)
    plan = fixed_plan()
    if not args.run:
        print(json.dumps(plan, indent=2))
        return 0
    if args.out is None or bool(args.fixtures) != bool(args.fixtures_sha256):
        parser.error('--run requires --out; --fixtures and --fixtures-sha256 must be paired')
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    result = dict(schema='E025-same-split-result-v2', status='INCOMPLETE', device_run=False,
        pid=os.getpid(), ppid=os.getppid(), contract=plan, checks=[], inputs=[], physical_receipts=[],
        normal_cases_passed=0, fallback_cases_passed=0, source_pins_rechecked=False,
        performance_qualified=False, full_model_qualified=False, elapsed_wall_s=0)
    write(args.out / 'complete.json', result)
    write(args.out / 'plan.json', plan)
    old_handlers = {s: signal.getsignal(s) for s in (signal.SIGALRM, signal.SIGINT, signal.SIGTERM)}
    mx = ledger = None
    with (args.out / 'events.jsonl').open('x') as events:
        def emit(row):
            events.write(json.dumps(dict(**row, elapsed_wall_s=time.monotonic() - started), allow_nan=False) + '\n')
            events.flush()
        try:
            for sig in old_handlers:
                signal.signal(sig, stop_signal)
            signal.setitimer(signal.ITIMER_REAL, max(0.001, 120 - (time.monotonic() - started)))
            result['sources'] = source_pins()
            captures = fixture_manifest(args.fixtures, args.fixtures_sha256) if args.fixtures else None
            result['synthetic_only'] = captures is None
            result['fixtures'] = captures
            recheck_sources(result['sources'], captures, args)
            support = load('e025_v2_physical', PHYSICAL_PATH)
            shared = load('e025_v2_ledger', LEDGER_PATH)
            import mlx.core as mx
            result['device_run'] = True
            mx.set_cache_limit(0)
            mx.synchronize()
            mx.clear_cache()
            mx.reset_peak_memory()
            ledger = shared.Ledger(args.out, result, mx, support, started)
            ledger.phase('initial')
            execute(mx, args.out, result, captures, ledger, emit)
            need(len(result['checks']) == 33 and result['normal_cases_passed'] == 3
                 and result['fallback_cases_passed'] == 18 and result['kernel_count'] == 6, 'coverage mismatch')
            recheck_sources(result['sources'], captures, args)
            result['source_pins_rechecked'] = True
            gc.collect()
            mx.synchronize()
            mx.clear_cache()
            ledger.phase('final_release')
            need(ledger.count == 68, 'phase coverage mismatch')
            result['status'] = 'PASS'
        except BaseException as error:
            result.update(status='FAIL', error=dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc()))
            # Persist the primary failure before any native cleanup that could stall.
            result['elapsed_wall_s'] = time.monotonic() - started
            write(args.out / 'complete.json', result)
        finally:
            try:
                if mx is not None and result['status'] != 'PASS':
                    gc.collect()
                    mx.synchronize()
                    mx.clear_cache()
            except BaseException as error:
                result['cleanup_error'] = dict(type=type(error).__name__, message=str(error))
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
                for sig, handler in old_handlers.items():
                    signal.signal(sig, handler)
            result['elapsed_wall_s'] = time.monotonic() - started
            write(args.out / 'complete.json', result)
    print(json.dumps(dict(status=result['status'], out=str(args.out))))
    return int(result['status'] != 'PASS')


if __name__ == '__main__':
    raise SystemExit(main())
