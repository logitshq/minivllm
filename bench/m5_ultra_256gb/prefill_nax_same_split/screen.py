"""E025 three-projection numerical screen, never HTTP or performance.

CPU-only: --describe or --help. Device execution requires --run --out NEW.
Default inputs are fixed-seed synthetic operator tensors (NOT model qualification).
For actual captures use --fixtures MANIFEST --fixtures-sha256 SHA. Exact schema:
{schema:'E025-projection-inputs-v1', cases:{gdn_z|mlp_down|out:{path,sha256}}}.
Each safetensors file contains activation BF16[128,K], weight UINT32[N,K/8],
scales UINT8[N,K/16], with no other tensors. No model loader is involved.
The root owner must enforce a hard timeout; Python cannot preempt a stuck GPU.
"""
import argparse
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import traceback


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('e025_same_split_candidate', HERE / 'candidate.py')
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def fixed_plan():
    return {'schema': 'E025-same-split-screen-v1', 'normal_M': list(c.NORMAL_M),
        'fallback_M': list(c.FALLBACK_M), 'shapes': [dict(name=n, K=k, N=o) for n, k, o in c.SHAPES],
        'geometries': [c.geometry(m, k, n) for _, k, n in c.SHAPES for m in c.NORMAL_M],
        'dtype': 'bfloat16', 'quantization': {'mode': 'nvfp4', 'bits': 4, 'group_size': 16},
        'seed': 20261017, 'cache_limit_bytes': 2 << 30, 'only_numerical_diagnostic': True,
        'default_OFF': True, 'source_sha256': {p.name: sha(p) for p in
            (HERE / 'candidate.py', HERE / 'screen.py', c.PINS)}}


def fixture_manifest(path, expected_sha):
    if not expected_sha or sha(path) != expected_sha:
        raise ValueError('fixture manifest SHA mismatch')
    data = json.loads(Path(path).read_text())
    if (set(data) != {'schema', 'cases'} or data['schema'] != 'E025-projection-inputs-v1'
            or set(data['cases']) != {n for n, _, _ in c.SHAPES}):
        raise ValueError('exact three-case fixture manifest required')
    for item in data['cases'].values():
        if set(item) != {'path', 'sha256'} or sha(item['path']) != item['sha256']:
            raise ValueError('fixture tensor SHA mismatch')
    return data


def execute(out, plan, captures, emit):
    import mlx.core as mx
    mx.set_cache_limit(2 << 30)
    enabled, disabled = c.SameSplitPartial(mx, enabled=True), c.SameSplitPartial(mx)
    if not enabled._hardware():
        raise ValueError('requires pinned MLX0.32.3 and Apple M5 Ultra applegpu_g17d')
    emit({'event': 'device', 'name': mx.device_info().get('device_name'),
          'architecture': mx.metal.device_info().get('architecture')})
    mx.random.seed(plan['seed'])
    results, active_input = [], None
    def equal(label, left, right):
        mx.eval(left, right)
        same_meta = left.shape == right.shape and left.dtype == right.dtype
        finite = bool(mx.all(mx.isfinite(left)).item()) and bool(mx.all(mx.isfinite(right)).item())
        bits = bool(mx.array_equal(left.view(mx.uint8), right.view(mx.uint8)).item()) if same_meta else False
        row = {'event': 'array-comparison', 'label': label, 'left_shape': list(left.shape),
               'right_shape': list(right.shape), 'left_dtype': str(left.dtype), 'right_dtype': str(right.dtype),
               'finite': finite, 'bytes_equal': bits, 'status': 'PASS' if same_meta and finite and bits else 'FAIL'}
        if row['status'] == 'FAIL':
            failure = out / 'first-difference.safetensors'
            mx.save_safetensors(str(failure), {'reference': left, 'candidate': right})
            row['original_dtype_artifact'] = {'path': failure.name, 'sha256': sha(failure)}
            if same_meta and not bits:
                bad = (left.view(mx.uint8) != right.view(mx.uint8)).reshape(-1)
                row['first_differing_byte_index'] = int(mx.argmax(bad.astype(mx.int32)).item())
            write(out / 'first-difference.json', row)
        emit(row)
        if row['status'] != 'PASS':
            raise RuntimeError('strict numerical gate rejected: ' + label)
    try:
        for name, K, N in c.SHAPES:
            emit({'event': 'begin-projection', 'name': name, 'K': K, 'N': N})
            if captures:
                arrays = mx.load(captures['cases'][name]['path'])
            else:
                # Independent high/low uint16 words cover every valid FP4 nibble.
                lo = mx.random.randint(0, 65536, (N, K // 8), dtype=mx.uint32)
                hi = mx.random.randint(0, 65536, (N, K // 8), dtype=mx.uint32)
                arrays = {'activation': (mx.random.normal((128, K)) / 16).astype(mx.bfloat16),
                          'weight': lo | (hi << 16),
                          'scales': mx.random.randint(0x30, 0x41, (N, K // 16), dtype=mx.uint8)}
                del lo, hi
            if set(arrays) != {'activation', 'weight', 'scales'}:
                raise ValueError('exact input tensor names required')
            x, w, s = (arrays[k] for k in ('activation', 'weight', 'scales'))
            if (x.shape != (128, K) or w.shape != (N, K // 8) or s.shape != (N, K // 16)
                    or x.dtype != mx.bfloat16 or w.dtype != mx.uint32 or s.dtype != mx.uint8):
                raise ValueError('input shape/dtype mismatch')
            mx.eval(x, w, s)
            initial = out / f'{name}-input-before.safetensors'
            mx.save_safetensors(str(initial), arrays)
            input_sha = sha(initial)
            active_input = (name, arrays, input_sha)
            emit({'event': 'input-frozen', 'name': name, 'path': initial.name, 'sha256': input_sha,
                  'synthetic_only': captures is None})
            for M in c.NORMAL_M:
                label = f'{name}-M{M}'
                values = {'original': enabled.original(x[:M], w, s),
                          'default_OFF': disabled(x[:M], w, s)}
                equal(label + '/OFF-vs-original', values['original'], values['default_OFF'])
                values['reference_partial'] = enabled.partials(x[:M], w, s, 'reference')
                values['reference_final'] = mx.sum(values['reference_partial'], axis=0)
                # An independently reconstructed reference is invalid unless this passes.
                equal(label + '/reference-vs-original-QMM', values['original'], values['reference_final'])
                values['candidate_partial'] = enabled.partials(x[:M], w, s, 'candidate')
                equal(label + '/every-partial', values['reference_partial'], values['candidate_partial'])
                values['candidate_final'] = mx.sum(values['candidate_partial'], axis=0)
                equal(label + '/final', values['original'], values['candidate_final'])
                # Also exercise the actual modular ON callable.
                values['enabled_callable'] = enabled(x[:M], w, s)
                equal(label + '/ON-callable', values['original'], values['enabled_callable'])
                raw = out / f'{label}-outputs.safetensors'
                mx.save_safetensors(str(raw), values)
                results.append({'name': label, 'status': 'PASS', 'raw': raw.name, 'sha256': sha(raw)})
                del values
            before = len(enabled.receipts)
            for M in c.FALLBACK_M:
                equal(f'{name}-M{M}/unregistered-fallback', enabled.original(x[:M], w, s), enabled(x[:M], w, s))
            if len(enabled.receipts) != before or disabled.receipts:
                raise RuntimeError('OFF or fallback unexpectedly compiled custom kernel')
            after = out / f'{name}-input-after.safetensors'
            mx.save_safetensors(str(after), arrays)
            after_sha = sha(after)
            emit({'event': 'input-unchanged', 'name': name, 'before_sha256': input_sha,
                  'after_sha256': after_sha, 'bytes_equal': input_sha == after_sha})
            if after_sha != input_sha:
                raise RuntimeError('input tensor bytes or serialization changed')
            active_input = None
            del arrays, x, w, s
            mx.synchronize()
            mx.clear_cache()
        return {'normal_cases': results, 'normal_cases_passed': len(results),
                'fallback_cases_passed': len(c.SHAPES) * len(c.FALLBACK_M)}
    finally:
        primary_error = sys.exc_info()[0] is not None
        try:
            if active_input is not None:
                name, arrays, input_sha = active_input
                # Preserve input-integrity evidence even when the first partial fails.
                after = out / f'{name}-input-after.safetensors'
                mx.save_safetensors(str(after), arrays)
                after_sha = sha(after)
                emit({'event': 'input-unchanged-on-exit', 'name': name,
                      'before_sha256': input_sha, 'after_sha256': after_sha,
                      'bytes_equal': input_sha == after_sha})
        except BaseException as exit_error:
            evidence = {'event': 'input-exit-snapshot-failed', 'type': type(exit_error).__name__,
                        'message': str(exit_error), 'primary_failure_preserved': primary_error}
            write(out / 'input-exit-error.json', evidence)
            emit(evidence)
            if not primary_error:
                raise
        finally:
            write(out / 'kernel-source-receipts.json', enabled.receipts)
            enabled.kernels.clear()
            disabled.kernels.clear()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--describe', action='store_true')
    mode.add_argument('--run', action='store_true')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--fixtures', type=Path)
    parser.add_argument('--fixtures-sha256')
    args = parser.parse_args(argv)
    plan = fixed_plan()
    if args.describe:
        print(json.dumps(plan, indent=2))
        return 0
    if args.out is None or bool(args.fixtures) != bool(args.fixtures_sha256):
        parser.error('--run requires --out; --fixtures and --fixtures-sha256 must be paired')
    captures = fixture_manifest(args.fixtures, args.fixtures_sha256) if args.fixtures else None
    # Check every installed small source pin before importing the GPU runtime.
    root = c.installed_root()
    for g in plan['geometries']:
        for arm in ('reference', 'candidate'):
            c.build_source(arm, g, root)
    args.out.mkdir(parents=True, exist_ok=False)
    plan.update(input_kind='captured' if captures else 'synthetic', fixtures=captures,
                fixture_manifest_sha256=args.fixtures_sha256)
    write(args.out / 'plan.json', plan)
    result = {'status': 'INCOMPLETE', 'scope': 'isolated projection numerical diagnostic',
              'synthetic_only': captures is None, 'timing_performed': False}
    started = time.monotonic()
    with (args.out / 'events.jsonl').open('x') as ledger:
        def emit(row):
            ledger.write(json.dumps({**row, 'elapsed_wall_s': time.monotonic() - started}) + '\n')
            ledger.flush()
        try:
            result.update(execute(args.out, plan, captures, emit))
            for name, value in plan['source_sha256'].items():
                if sha(HERE / name) != value:
                    raise RuntimeError('source changed during diagnostic: ' + name)
            for g in plan['geometries']:
                for arm in ('reference', 'candidate'):
                    c.build_source(arm, g, root)
            if args.fixtures:
                fixture_manifest(args.fixtures, args.fixtures_sha256)
            result['status'] = 'PASS'
        except BaseException as exc:
            result.update(status='FAIL', error={'type': type(exc).__name__, 'message': str(exc),
                          'traceback': traceback.format_exc()})
        finally:
            try:
                gc.collect()
                mx = sys.modules.get('mlx.core')
                if mx is not None:
                    mx.synchronize()
                    mx.clear_cache()
                    result['active_bytes_after_release'] = mx.get_active_memory()
            except BaseException as exc:
                result.update(status='FAIL', cleanup_error=type(exc).__name__)
            result['elapsed_wall_s'] = time.monotonic() - started
            write(args.out / 'complete.json', result)
    print(json.dumps({'status': result['status'], 'out': str(args.out)}))
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
