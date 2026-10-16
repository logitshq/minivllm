"""CPU-safe preparation entry; GPU execution requires --execute-gpu and explicit Y.
No MLX/NumPy imports until main has validated evidence and the execution flag.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import importlib.metadata
import json
import sys
import time

D = Path(__file__).resolve().parent
P = D.parent
PROTO = P / 'v23-gdn-conv-silu-prototype'
ACCEPTED = P / 'candidate-v17-gdn-gates-build1-root/tutorial_code/miniengine'
DEFAULT_STATE = P / 'v18-recurrent-numeric-build1/gpu-recurrent-numeric-AC-run1/real-L0-D1'
DEFAULT_EVIDENCE = P / 'v23-root-control-build1/native-pipeline.json'


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def source_check():
    pins = json.loads((D / 'source-pins.json').read_text())
    for path, expected in pins.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise ValueError('source changed: ' + path)
    return pins


def evidence_check(path, sum_y):
    if type(sum_y) is not int or sum_y not in (1, 2, 3, 4):
        raise ValueError('explicit sum_y must be integer 1..4')
    value = json.loads(Path(path).read_text())
    if value.get('schema') != 'v23-native-reduction-pipeline-v1':
        raise ValueError('requires independent native pipeline evidence schema')
    if value.get('kernel') != 'col_reduce_small_1_reduce_sumbfloat16' or value.get('mlx_version') != '0.32.3':
        raise ValueError('wrong reference kernel/runtime')
    if value.get('native_library_compile_options_overridden') is not False:
        raise ValueError('native pipeline options differ')
    m, width = value['max_total_threads_per_threadgroup'], value['thread_execution_width']
    if type(m) is not int or m < 32 or width != 32 or min(4, m // 32) != sum_y:
        raise ValueError('provided Y differs from independently measured pipeline M')
    assumptions = value['selector_assumptions']
    if assumptions != {'dtype': 'bfloat16', 'shape': [1, 4, 10240], 'strides': [40960, 10240, 1], 'row_contiguous': True, 'axis': 1, 'ndim_template': 1, 'index_type': 'int'}:
        raise ValueError('reference selector assumptions differ')
    return {'path': str(Path(path).resolve()), 'sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            'explicit_sum_y': sum_y, 'independent_native_pipeline': value,
            'choice_from_numerical_outputs': False}


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def arguments(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--execute-gpu', action='store_true')
    ap.add_argument('--sum-y', required=True, type=int)
    ap.add_argument('--sum-evidence', type=Path, default=DEFAULT_EVIDENCE)
    ap.add_argument('--checkpoint-dir', type=Path)
    ap.add_argument('--weight-layer', type=int, default=0)
    ap.add_argument('--state-bank', type=Path, default=DEFAULT_STATE)
    ap.add_argument('--real-bank', type=Path, action='append', default=[])
    ap.add_argument('--chain-steps', type=int, default=8)
    ap.add_argument('--output-dir', type=Path)
    return ap.parse_args(argv)


def main(argv=None):
    args = arguments(argv)
    pins = source_check()
    independent = evidence_check(args.sum_evidence, args.sum_y)
    if not 2 <= args.chain_steps <= 32:
        raise ValueError('chain_steps must be 2..32')
    if not args.execute_gpu:
        print(json.dumps({'status': 'CPU_PREFLIGHT_ONLY', 'MLX_imported': False, 'GPU_run': False,
                          'source_files_checked': len(pins), 'explicit_sum_y': args.sum_y,
                          'independent_pipeline_evidence': independent,
                          'real_preconv_banks_found_by_preparation': 0,
                          'required_before_execution': ['--execute-gpu', '--checkpoint-dir', '--output-dir (new)']}))
        return 0
    if args.checkpoint_dir is None or args.output_dir is None:
        raise ValueError('GPU execution requires checkpoint-dir and new output-dir')
    versions = {n: importlib.metadata.version(n) for n in ('mlx', 'mlx-metal')}
    if versions != {'mlx': '0.32.3', 'mlx-metal': '0.32.3'}:
        raise ValueError('requires native pipeline evidence runtime MLX/mlx-metal 0.32.3')
    args.output_dir.mkdir(exist_ok=False, parents=False)
    receipt = {'status': 'STARTED', 'scope': 'synthetic conv + checkpoint small weights + saved recurrent state',
               'sum_y_evidence': independent, 'source_pins': pins, 'runtime_versions': versions, 'cases': [], 'chain': [],
               'real_preconv_cases': [], 'logits_cache_lifecycle_qualified': False,
               'runtime_sources': {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in D.iterdir() if f.suffix in ('.py', '.metal')},
               'actual_product_internal_flags_observed': False,
               'pending': ['native internal product flag bits (GPU descriptor strides are observed)',
                           'genuine real preconv capture unless complete --real-bank supplied', 'logits/IDs/cache/lifecycle']}
    start = time.perf_counter()
    write(args.output_dir / 'receipt.json', receipt)
    try:
        import mlx.core as mx
        import mlx.nn as nn
        import numpy as np
        if mx.default_device() != mx.gpu:
            raise ValueError('requires actual GPU default device')
        receipt['device_info'] = mx.device_info()
        if receipt['device_info'].get('device_name') != independent['independent_native_pipeline']['device'] or receipt['device_info'].get('architecture') != independent['independent_native_pipeline']['architecture']:
            raise ValueError('actual GPU device/architecture differs from native PSO evidence')
        receipt['numpy_version'] = np.__version__
        # Isolated small-array qualification only; no serving/model objects exist.
        mx.set_cache_limit(64 << 20)
        mx.set_memory_limit(512 << 20)
        receipt['qualification_limits_bytes'] = {'cache': 64 << 20, 'memory': 512 << 20}
        banks = module('v23_numeric_banks', D / 'banks.py')
        reference = module('v23_accepted_conv_reference', D / 'reference.py')
        candidate = module('v23_archived_prototype', PROTO / 'conv_silu_raw_tail.py')
        recurrent = module('v23_original_recurrence', ACCEPTED / 'kernels/gdn_metal.py')
        weights, weight_provenance = banks.checkpoint_conv(args.checkpoint_dir, args.weight_layer, mx, np)
        g, beta, initial_state = (banks.saved_array(args.state_bank, k, mx, np) for k in ('g', 'beta', 'S'))
        receipt['weight_provenance'] = weight_provenance
        receipt['state_provenance'] = {'bank': str(args.state_bank.resolve()), 'bank_sha256': banks.sha(args.state_bank / 'bank.json'),
                                       'use': 'saved real g/beta/S; synthetic conv inputs are not real activations'}
        probe = mx.fast.metal_kernel(name='v23_conv_numeric_probe', input_names=['raw', 'old_tail', 'w'],
            output_names=['conv_out', 'probe_tail'], source=(D / 'conv_probe.metal').read_text(),
            header=candidate.HEADER, ensure_row_contiguous=True, compile_options=candidate.COMPILE_OPTIONS)
        layout_probe = mx.fast.metal_kernel(name='v23_original_product_layout', input_names=['product'],
            output_names=['layout'], source=(D / 'layout_probe.metal').read_text(), ensure_row_contiguous=False,
            compile_options={'math_mode': 'safe'})

        def raw_bytes(value):
            mx.eval(value)
            if value.dtype == mx.bfloat16:
                host = np.ascontiguousarray(np.asarray(value.view(mx.uint16)), dtype='<u2')
                if np.any((host & 0x7f80) == 0x7f80):
                    raise ValueError('nonfinite BF16')
            elif value.dtype == mx.float32:
                host = np.ascontiguousarray(np.asarray(value), dtype='<f4')
                if not np.isfinite(host).all():
                    raise ValueError('nonfinite FP32')
            else:
                raise ValueError('unexpected numerical dtype')
            return host.tobytes()

        def signatures(values):
            return {k: hashlib.sha256(raw_bytes(v)).hexdigest() for k, v in values.items()}

        def equal(label, left, right):
            if left.shape != right.shape or left.dtype != right.dtype:
                raise ValueError(f'{label}: shape/dtype differ')
            a, b = raw_bytes(left), raw_bytes(right)
            if a != b:
                offset = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
                raise ValueError(f'{label}: strict bytes differ at offset {offset}')
            return {'shape': list(left.shape), 'dtype': str(left.dtype), 'bytes': len(a), 'sha256': hashlib.sha256(a).hexdigest(), 'strict_equal': True}

        def layout(product):
            mx.eval(product)
            descriptor, = layout_probe(inputs=[product], grid=(1, 1, 1), threadgroup=(1, 1, 1),
                                      output_shapes=[(6,)], output_dtypes=[mx.uint64])
            mx.eval(descriptor)
            data = descriptor.tolist()
            shape, strides = data[:3], data[3:]
            derived = {'row_contiguous': strides == [40960, 10240, 1], 'column_contiguous': False}
            result = {'shape': shape, 'strides_elements': strides, 'dtype': str(product.dtype),
                      'source': 'original product GPU descriptor; ensure_row_contiguous=False',
                      'derived_flags_from_observed_strides': derived, 'internal_flags_observed': False}
            if shape != [1, 4, 10240] or strides != [40960, 10240, 1]:
                write(args.output_dir / 'unexpected-product-layout.json', result)
                raise ValueError('actual product layout does not satisfy native PSO selector assumptions; STOP')
            return result

        def split_heads(act):
            q, k, v = mx.split(act, [2048, 4096], axis=-1)
            q, k, v = q.reshape(1, 16, 128), k.reshape(1, 16, 128), v.reshape(1, 48, 128)
            return mx.repeat(q, 3, axis=1), mx.repeat(k, 3, axis=1), v

        def qualify(bank, state_ref, state_cand, tail_ref=None, tail_cand=None):
            receipt['active_case'] = bank['name']
            raw, weight = bank['raw'], bank['w']
            old_ref = bank['old_tail'] if tail_ref is None else tail_ref
            old_cand = bank['old_tail'] if tail_cand is None else tail_cand
            actual_g, actual_beta = bank.get('g', g), bank.get('beta', beta)
            inputs = {'raw': raw, 'old_ref': old_ref, 'old_cand': old_cand, 'w': weight,
                      'g': actual_g, 'beta': actual_beta, 'state_ref': state_ref, 'state_cand': state_cand}
            before = signatures(inputs)
            receipt['active_input_sha256_before'] = before
            write(args.output_dir / 'receipt.json', receipt)
            conv, act_ref, next_ref, product = reference.conv_reference(raw, old_ref, weight, mx, nn)
            product_layout = layout(product)
            act_cand, next_cand = candidate.conv_silu_raw_tail(raw, old_cand, weight, sum_y=args.sum_y, backend=mx)
            conv_cand, probe_tail = probe(inputs=[raw, old_cand, weight], template=[('SUM_Y', args.sum_y)],
                grid=candidate.GRID, threadgroup=candidate.THREADGROUP, output_shapes=list(candidate.OUTPUT_SHAPES),
                output_dtypes=[mx.bfloat16, mx.bfloat16])
            mx.eval(conv, act_ref, next_ref, conv_cand, act_cand, next_cand, probe_tail)
            checks = {name: equal(name, left, right) for name, left, right in (
                ('conv', conv, conv_cand), ('act', act_ref, act_cand), ('tail', next_ref, next_cand), ('probe_tail', next_ref, probe_tail))}
            q, k, v = split_heads(act_ref)
            o_ref, new_ref = recurrent.gated_delta_step_metal(q, k, v, actual_g, actual_beta, state_ref)
            q, k, v = split_heads(act_cand)
            o_cand, new_cand = recurrent.gated_delta_step_metal(q, k, v, actual_g, actual_beta, state_cand)
            mx.eval(o_ref, new_ref, o_cand, new_cand)
            checks['o'] = equal('o', o_ref, o_cand)
            checks['S_new'] = equal('S_new', new_ref, new_cand)
            after = signatures(inputs)
            if before != after:
                raise ValueError('candidate or probe mutated input bytes')
            result = {'name': bank['name'], 'kind': bank.get('kind', 'real_preconv_bank'),
                      'weight_kind': bank.get('weight_kind', 'saved'), 'checks': checks,
                      'input_sha256_before': before, 'input_sha256_after': after,
                      'product_layout': product_layout, 'input_readonly': True}
            return result, next_ref, next_cand, new_ref, new_cand

        for bank in banks.synthetic(np, mx, weights):
            result, *_ = qualify(bank, initial_state, initial_state)
            receipt['cases'].append(result)
            write(args.output_dir / 'receipt.json', receipt)
            mx.clear_cache()
        # Genuine preconv banks are optional and must actually contain raw/tail/w.
        for directory in args.real_bank:
            actual = {k: banks.saved_array(directory, k, mx, np) for k in banks.CONTRACTS}
            actual.update(name=directory.name, kind='real_preconv_bank', weight_kind='saved_real_bank')
            result, *_ = qualify(actual, actual['S'], actual['S'])
            receipt['real_preconv_cases'].append(result)
            write(args.output_dir / 'receipt.json', receipt)
        sr = sc = initial_state
        tr = tc = mx.zeros((1, 3, 10240), mx.bfloat16)
        for step in range(args.chain_steps):
            raw = mx.array((((np.arange(10240, dtype=np.float32) + step * 13) % 47) - 23).reshape(1, 10240) / np.float32(32)).astype(mx.bfloat16)
            bank = {'name': f'synthetic-chain-{step:02d}', 'kind': 'synthetic_preconv_with_saved_real_state',
                    'raw': raw, 'old_tail': tr, 'w': weights, 'weight_kind': 'real_checkpoint'}
            result, tr, tc, sr, sc = qualify(bank, sr, sc, tr, tc)
            receipt['chain'].append(result)
            write(args.output_dir / 'receipt.json', receipt)
        receipt['status'] = 'PASS_CONV_NUMERIC_SYNTHETIC_ONLY' if not args.real_bank else 'PASS_CONV_NUMERIC_WITH_REAL_BANKS_ONLY'
        receipt['GPU_numeric_run'] = True
        receipt['engine_enabled'] = False
        return 0
    except Exception as error:
        receipt['status'] = 'FAIL_STOP'
        receipt['error'] = {'type': type(error).__name__, 'message': str(error)}
        raise
    finally:
        receipt['wall_seconds'] = time.perf_counter() - start
        write(args.output_dir / 'receipt.json', receipt)


if __name__ == '__main__':
    raise SystemExit(main())
