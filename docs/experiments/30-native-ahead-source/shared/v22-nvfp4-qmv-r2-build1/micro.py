#!/usr/bin/env python3
"""Root-only V22 NVFP4 R4-control/R2 micro; no full model or adoption.

V21 CPU identity/range loader/timing/supervisor preserved with target constants narrowed.

CPU check reads only source and safetensor headers. GPU mode is root-only and
loads only selected tensor ranges via numpy.memmap, never a complete shard.
"""
import argparse
import copy
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import re
from pathlib import Path
import statistics
import struct
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone

sys.dont_write_bytecode = True
BASE = Path(__file__).resolve().parent
P = BASE.parent
ROOT = P / 'candidate-v16-gdn-projection-build1-root'
MANIFEST = P / 'source-manifest-v16-gdn-projection-build1.json'
CHECKPOINT = Path('/Users/ryanli/.cache/huggingface/hub/models--mlx-community--Qwen3.8-27B-nvfp4/snapshots/5ff8ef173ad0d7c3aae92f0be43031a6ab8067c6')
LAYERS, ROWS = (0,), (17408,)
NAMES, MODULE = ('gate_proj',), 'mlp'
SEGMENTS = len(LAYERS) * len(NAMES)
REAL_BANK = P / 'swiglu-actual-micro-build1/gpu-actual-micro-run1'
Q = {'group_size': 16, 'bits': 4, 'mode': 'nvfp4'}
K, SEED, PRIME, REPEATS, PAIRS = 5120, 20261016, 128, 512, 6
ORDERS = (('A', 'B'), ('B', 'A')) * 3
SCHEDULE = (('A', 'A'), ('B', 'B')) + ORDERS + (('A', 'A'), ('B', 'B'))
CONTROLS = {'MLX_MAX_OPS_PER_BUFFER': '50', 'MLX_MAX_MB_PER_BUFFER': '50',
            'MLX_BFS_MAX_WIDTH': '20', 'MLX_METAL_FAST_SYNCH': '0'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def utc():
    return datetime.now(timezone.utc).isoformat()


def header(path):
    with Path(path).open('rb') as f:
        raw_size = f.read(8)
        assert len(raw_size) == 8, 'Truncated safetensors prefix'
        size = struct.unpack('<Q', raw_size)[0]
        assert 0 < size < (64 << 20), 'Invalid header size'
        raw = f.read(size)
    assert len(raw) == size, 'Truncated safetensors header'
    return json.loads(raw), 8 + size, hashlib.sha256(raw).hexdigest()


def tensor_meta(path, key, dtype, shape, parsed=None):
    table, data_start, header_sha = parsed or header(path)
    item = table[key]
    assert item['dtype'] == dtype and item['shape'] == list(shape), (key, item)
    lo, hi = item['data_offsets']
    width = {'U32': 4, 'U8': 1}[dtype]
    assert type(lo) is type(hi) is int and 0 <= lo < hi
    assert hi - lo == math.prod(shape) * width and data_start + hi <= Path(path).stat().st_size
    return {'path': str(path), 'key': key, 'dtype': dtype, 'shape': list(shape),
            'offset': data_start + lo, 'nbytes': hi - lo, 'header_sha256': header_sha}


def real_metadata():
    receipt_path = REAL_BANK / 'receipt.json'
    receipt = json.loads(receipt_path.read_text())
    assert sha(receipt_path) == '94a9f5ab7ae3780f306e66d71e4f3b0f2fb559eef4d5a89ae179ef63ef3355c8'
    assert receipt['A_status'] == 'ACTUAL64_NUMERIC_PASS'
    inputs = {}
    for layer in LAYERS:
        row = next(r for r in receipt['rows'] if r['layer'] == layer)
        capture = receipt['request']['captures'][str(layer)]
        assert row['x']['shape'] == [1, K] and row['x']['dtype'] == 'mlx.core.bfloat16'
        assert capture['mode'] == 'decode' and capture['capture_empty'] is True and capture['first_normal_decode'] is True
        path = REAL_BANK / f'L{layer:02d}-x.raw.le.bin'
        assert path.stat().st_size == K * 2 and sha(path) == row['x']['raw_sha256']
        inputs[layer] = dict(row['x'], path=str(path), capture=capture, nbytes=path.stat().st_size)
    return {'receipt_path': str(receipt_path), 'receipt_sha256': sha(receipt_path),
            'capture_driver_path': str(REAL_BANK.parent / 'check_swiglu_actual_micro_v1.py'),
            'capture_driver_sha256': sha(REAL_BANK.parent / 'check_swiglu_actual_micro_v1.py'),
            'source': 'Historical first normal decode MLP input (after post_attention_layernorm), V17/gates ON; not newly captured',
            'old_experiment_status': receipt['status'], 'inputs': inputs}


def control_passes(control):
    return all(len(ratios) == 2 and .98 <= statistics.median(ratios) <= 1.02
               and all(.95 <= ratio <= 1.05 for ratio in ratios) for ratios in control.values())


def identity():
    frozen = json.loads((BASE / 'source-manifest.json').read_text())
    for name, expected in frozen['files'].items():
        assert sha(BASE / name) == expected, ('Prototype changed', name)
    for rec in frozen['upstream_sources'].values():
        assert sha(rec['path']) == rec['sha256'], rec['path']
    manifest = json.loads(MANIFEST.read_text())
    source = ROOT / 'tutorial_code/miniengine'
    assert Path(manifest['source_root']) == source
    for name, rec in manifest['files'].items():
        assert sha(source / name) == rec['after_sha256'], name
    index_path = CHECKPOINT / 'model.safetensors.index.json'
    index = json.loads(index_path.read_text())['weight_map']
    parsed, tensors = {}, {}
    for layer in LAYERS:
        tensors[layer] = []
        for name, n in zip(NAMES, ROWS):
            fields = []
            for suffix, dtype, shape in [('weight', 'U32', (n, K // 8)), ('scales', 'U8', (n, K // 16))]:
                tail = f'.layers.{layer}.{MODULE}.{name}.{suffix}'
                keys = [key for key in index if key.endswith(tail)]
                assert len(keys) == 1, tail
                key = keys[0]
                path = CHECKPOINT / index[key]
                if path not in parsed:
                    parsed[path] = header(path)
                fields.append(tensor_meta(path, key, dtype, shape, parsed[path]))
            tensors[layer].append(fields)
    config = json.loads((CHECKPOINT / 'config.json').read_text())
    assert config['quantization'] == Q and config['text_config']['hidden_size'] == K
    return {'driver_sha256': sha(__file__), 'protocol_sha256': sha(BASE / 'protocol.json'), 'prototype_manifest_sha256': sha(BASE / 'source-manifest.json'), 'source_manifest_sha256': sha(MANIFEST),
            'source_root': str(source), 'checkpoint': str(CHECKPOINT), 'tensors': tensors,
            'index_sha256': sha(index_path), 'config_sha256': sha(CHECKPOINT / 'config.json'),
            'resource_wrapper_sha256': sha(P / 'resource_snapshot_power_v1.py'),
            'helper_sha256': sha(source / 'model/projection_packing.py'),
            'target_tensor_payload_bytes': sum(t['nbytes'] for group in tensors.values() for pair in group for t in pair),
            'real_inputs': real_metadata(),
            'plan': {'layers': LAYERS, 'rows': ROWS, 'seed': SEED, 'prime': PRIME,
                     'repeats': REPEATS, 'orders': ORDERS, 'schedule': SCHEDULE,
                     'position_control': {'median_slot2_div_slot1_bounds': [.98, 1.02],
                                          'each_pair_bounds': [.95, 1.05]},
                     'controls': CONTROLS, 'input_kind': '8 synthetic timing + L0 historical real byte checks', 'variants': {'A': 'original mx.quantized_matmul', 'B': 'R2 custom QMV'}, 'qualification': ['all R4 bytes before R2', 'all R2 bytes before timing'], 'timeout_s': 170, 'adopted': False,
                     'input_pool': ['zero', 'alternating cancellation', 'random amp1/16',
                                    'random amp1', 'random amp16', 'random0.25a', 'random0.25b', 'random0.25c'],
                     'minimum_median_reduction': 0.05, 'minimum_wins': 5,
                     'power_domain': 'Battery exploration', 'full_model': False}}


def load_range(np, mx, meta):
    """Map exactly one tensor and detach it before the MLX constructor."""
    dtype = np.dtype('<u4' if meta['dtype'] == 'U32' else 'u1')
    mapping = np.memmap(meta['path'], mode='r', dtype=dtype,
                        offset=meta['offset'], shape=tuple(meta['shape']))
    try:
        host = np.array(mapping, copy=True)
        value = mx.array(host)
        mx.eval(value)
        return value
    finally:
        mapping._mmap.close()


def block(mx, invoke, variant):
    """The same prime and synchronizations precede either timed variant."""
    prime_utc, prime_begin = utc(), time.perf_counter_ns()
    for j in range(PRIME):
        result = invoke(variant, j)
    mx.synchronize()
    prime_end = time.perf_counter_ns()
    start_utc = utc()
    begin = time.perf_counter_ns()
    for j in range(REPEATS):
        result = invoke(variant, j)
    mx.synchronize()
    end = time.perf_counter_ns()
    return {'variant': variant, 'prime_start_utc': prime_utc,
            'prime_start_ns': prime_begin, 'prime_end_ns': prime_end,
            'prime_repeats': PRIME, 'start_utc': start_utc, 'end_utc': utc(),
            'start_ns': begin, 'end_ns': end, 'total_ns': end - begin,
            'ns_per_call': (end - begin) / REPEATS, 'repeats': REPEATS,
            'groups_per_call': len(LAYERS), 'segments_per_call': SEGMENTS}, result


def make_kernel(mx):
    return mx.fast.metal_kernel(name='v22_nvfp4_qmv_rows',
        input_names=['weight', 'scales', 'x'], output_names=['output'],
        source=(BASE / 'kernel_body.metal').read_text(),
        header=(BASE / 'kernel_header.metal').read_text())


def project(mx, kernel, weight, scales, x, rows_per_simd):
    assert rows_per_simd in (4, 2)
    assert x.dtype == mx.bfloat16 and x.shape == (1, K)
    assert weight.dtype == mx.uint32 and weight.shape == (17408, K // 8)
    assert scales.dtype == mx.uint8 and scales.shape == (17408, K // 16)
    return kernel(inputs=[weight, scales, x], template=[('T', mx.bfloat16), ('R', rows_per_simd)],
                  grid=(64, 17408 // (2 * rows_per_simd), 1), threadgroup=(64, 1, 1),
                  output_shapes=[(1, 17408)], output_dtypes=[mx.bfloat16])[0]


def gpu(out, ident):
    import numpy as np
    import mlx.core as mx
    sys.path.insert(0, str(P))
    from resource_snapshot_power_v1 import snapshot, deltas, guard
    assert importlib.metadata.version('mlx') == '0.32.3'
    info = mx.metal.device_info()
    assert mx.default_device() == mx.gpu
    arch = re.fullmatch(r'applegpu_g(\d+)[a-z]*', str(info.get('architecture', '')))
    assert arch and int(arch[1]) < 17 and 'M4 Max' in str(info), info
    assert all(os.environ.get(k) == v for k, v in CONTROLS.items()), CONTROLS
    write(out / 'runtime.json', {'mlx': mx.__file__, 'mlx_version': importlib.metadata.version('mlx'),
          'device_info': info, 'controls': {k: os.environ[k] for k in os.environ if k.startswith('MLX_')},
          'full_model_loaded': False, 'math_policy_override': False, 'kernels': [4, 2]})
    resources, checks, startups, pairs = [], [], [], []

    def probe(label):
        mx.synchronize()
        s = snapshot(label, os.getpid())
        s['mlx_memory'] = {'active': mx.get_active_memory(), 'cache': mx.get_cache_memory(), 'peak': mx.get_peak_memory()}
        resources.append(s); deltas(resources); write(out / 'resources.json', resources); guard(resources)
        assert s['power_metadata']['status'] == 'known' and s['power_metadata']['source'] == 'Battery Power'
        assert s['processinfo_thermal_metadata']['status'] == 'known' and s['processinfo_thermal_metadata']['code'] == 0
        assert s['pressure_level_value'] == 1
        assert s['swap_used_mb'] <= resources[0]['swap_used_mb'], 'Swap usage growth'
        assert s['vm_counters']['Swapouts'] == resources[0]['vm_counters']['Swapouts'], 'Swapouts growth'
        assert s['known_inference_process_count'] == 0 and s['exact_owned_pid_metadata']['status'] == 'known'
        return s

    def raw(value): return np.asarray(mx.contiguous(value).view(mx.uint16)).tobytes()

    def compare(left, right, label):
        mx.eval(left, right)
        assert left.shape == right.shape == (1, 17408) and left.dtype == right.dtype == mx.bfloat16
        lb, rb = raw(left), raw(right)
        item = {'label': label, 'shape': list(left.shape), 'dtype': str(left.dtype), 'nbytes': len(lb),
                'finite': bool(mx.all(mx.isfinite(left)).item() and mx.all(mx.isfinite(right)).item()),
                'left_sha256': hashlib.sha256(lb).hexdigest(), 'right_sha256': hashlib.sha256(rb).hexdigest(),
                'differing_bytes': sum(a != b for a, b in zip(lb, rb))}
        item['exact'] = item['finite'] and len(lb) == len(rb) and lb == rb
        checks.append(item); write(out / 'bytes.json', checks)
        assert item['exact'], ('Strict BF16 byte failure; no timing permitted', item)

    probe('fresh-no-model')
    rng = np.random.default_rng(SEED)
    host = [np.zeros((1, K), np.float32), np.tile(np.array([1., -1.], np.float32), K // 2)[None]]
    host += [(rng.standard_normal((1, K)) * amp).astype(np.float32) for amp in (.0625, 1, 16, .25, .25, .25)]
    pool = [mx.array(x, dtype=mx.bfloat16) for x in host]
    mx.eval(*pool)
    inputs = [{'id': j, 'shape': list(x.shape), 'dtype': str(x.dtype),
               'raw_sha256': hashlib.sha256(raw(x)).hexdigest(), 'synthetic': True} for j, x in enumerate(pool)]
    assert len({item['raw_sha256'] for item in inputs}) == 8
    real = ident['real_inputs']['inputs'][0]
    words = np.fromfile(real['path'], dtype='<u2', count=K)
    assert words.shape == (K,)
    historical = mx.array(words, dtype=mx.uint16).view(mx.bfloat16).reshape(1, K)
    mx.eval(historical)
    assert hashlib.sha256(raw(historical)).hexdigest() == real['raw_sha256']
    inputs.append({'id': 'historical-real-L0', 'shape': [1, K], 'dtype': str(historical.dtype),
                   'raw_sha256': real['raw_sha256'], 'synthetic': False})
    write(out / 'inputs.json', inputs)
    wm, sm = ident['tensors'][0][0]
    weight, scales = load_range(np, mx, wm), load_range(np, mx, sm)
    assert weight.nbytes + scales.nbytes == ident['target_tensor_payload_bytes']
    kernel = make_kernel(mx)
    def original(x): return mx.quantized_matmul(x, weight, scales, transpose=True, **Q)
    def custom(x, r): return project(mx, kernel, weight, scales, x, r)
    # Compile/first call outside measured blocks. R2 is never requested before
    # every R4 control input has matched the original primitive.
    t = time.perf_counter_ns(); z = original(pool[0]); mx.eval(z)
    startups.append({'variant': 'original', 'first_call_ns': time.perf_counter_ns() - t})
    t = time.perf_counter_ns(); z = custom(pool[0], 4); mx.eval(z)
    startups.append({'variant': 'R4', 'first_call_ns': time.perf_counter_ns() - t})
    write(out / 'startup.json', startups)
    qualify = [(f'synthetic{j}', x) for j, x in enumerate(pool)] + [('historical-real-L0', historical)]
    for label, x in qualify: compare(original(x), custom(x, 4), f'R4-QMM.{label}')
    probe('R4-all-byte-pass')
    t = time.perf_counter_ns(); z = custom(pool[0], 2); mx.eval(z)
    startups.append({'variant': 'R2', 'first_call_ns': time.perf_counter_ns() - t})
    write(out / 'startup.json', startups)
    for label, x in qualify:
        candidate = custom(x, 2)
        compare(original(x), candidate, f'R2-QMM.{label}')
        compare(custom(x, 4), candidate, f'R2-R4.{label}')
    assert len(checks) == 27 and all(c['exact'] for c in checks)
    write(out / 'qualification.json', {'status': 'STRICT-R4-R2-BYTES-PASS', 'checks': len(checks), 'timing_allowed': True})
    probe('all-byte-pass-timing-ready')

    def invoke(variant, j):
        value = original(pool[j % 8]) if variant == 'A' else custom(pool[j % 8], 2)
        mx.eval(value)  # Every output, including prime, is consumed.
        return [value]

    for index, order in enumerate(SCHEDULE):
        probe(f'pair{index}-before')
        a, ao = block(mx, invoke, order[0]); b, bo = block(mx, invoke, order[1])
        # CPU hashes only after both blocks finish.
        for rec, values in ((a, ao), (b, bo)):
            rec['hashes'] = [hashlib.sha256(raw(v)).hexdigest() for v in values]
        assert a['hashes'] == b['hashes'], 'Paired output hashes differ'
        item = {'pair': index, 'order': order, 'blocks': [a, b],
                'kind': 'position-control' if order[0] == order[1] else 'AB',
                'slot2_div_slot1': b['ns_per_call'] / a['ns_per_call']}
        if item['kind'] == 'AB':
            by = {a['variant']: a, b['variant']: b}
            item['reduction'] = 1 - by['B']['ns_per_call'] / by['A']['ns_per_call']
        pairs.append(item); write(out / 'pairs.json', pairs)
        del ao, bo
        probe(f'pair{index}-after')
    ab = [item for item in pairs if item['kind'] == 'AB']
    controls = {v: [item['slot2_div_slot1'] for item in pairs if item['kind'] == 'position-control' and item['order'][0] == v] for v in ('A', 'B')}
    control_ok = control_passes(controls)
    medians = {v: statistics.median(next(b['ns_per_call'] for b in item['blocks'] if b['variant'] == v) for item in ab) for v in ('A', 'B')}
    reduction, wins = 1 - medians['B'] / medians['A'], sum(item['reduction'] > 0 for item in ab)
    probe('timing-complete')
    status = 'CONTROL_DRIFT_STOP' if not control_ok else ('EXPLORATORY-PASS' if reduction >= .05 and wins >= 5 else 'STOP-MICRO-GATE')
    result = {'status': status, 'all_bytes_exact': True, 'checks': len(checks),
              'medians_ns': medians, 'median_reduction': reduction, 'wins': wins, 'pairs': 6,
              'control_passed': control_ok, 'control_slot2_div_slot1': controls,
              'minimum_reduction': .05, 'minimum_wins': 5, 'power_domain': 'Battery exploration',
              'scope': 'L0 gate K5120 N17408 only; original QMM vs R2; 8 synthetic timing + historical real bytes',
              'target_tensor_payload_bytes': ident['target_tensor_payload_bytes'],
              'adopted': False, 'full_model_qualified': False, 'E2E_run': False, 'AC_frontier_entry': False}
    write(out / 'result.json', result)
    return 0 if status == 'EXPLORATORY-PASS' else 2


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode', choices=('cpu', 'run', 'child'))
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--owner-pid', type=int, help='Internal child: exact supervisor PID')
    args = ap.parse_args()
    ident = identity()
    if args.mode == 'cpu':
        args.out.mkdir(exist_ok=False)
        assert not any(k == 'mlx' or k.startswith(('mlx.', 'tutorial_code')) for k in sys.modules)
        write(args.out / 'cpu.json', {'status': 'CPU-SOURCE-HEADER-PASS', 'identity': ident,
                                      'mlx_imported': False, 'GPU_used': False, 'checkpoint_tensor_payload_loaded': False, 'real_input_bytes_hashed': len(LAYERS) * K * 2})
        print('CPU source/header/real-input provenance checks passed; no MLX/GPU/model/checkpoint-payload/resource calls')
        return 0
    if args.mode == 'child':
        assert args.owner_pid == os.getppid() and args.owner_pid > 1, 'Child requires live exact supervisor'
        try:
            return gpu(args.out, ident)
        except BaseException as exc:
            write(args.out / 'failure.json', {'status': 'FAIL-STOP', 'error': repr(exc),
                  'traceback': traceback.format_exc(), 'pid': os.getpid(), 'adopted': False, 'timing_allowed': (args.out / 'qualification.json').exists()})
            write(args.out / 'result.json', {'status': 'FAIL-STOP', 'error': repr(exc),
                  'pid': os.getpid(), 'adopted': False,
                  'timing_allowed': (args.out / 'qualification.json').exists()})
            traceback.print_exc()
            return 1
        finally:
            write(args.out / 'child-finally.json', {'pid': os.getpid(), 'utc': utc(), 'exiting': True})
    args.out.mkdir(exist_ok=False)
    write(args.out / 'identity.json', ident)
    argv = [sys.executable, str(Path(__file__).resolve()), 'child', '--out', str(args.out), '--owner-pid', str(os.getpid())]
    with (args.out / 'child.log').open('w') as log:
        proc = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT)
        write(args.out / 'child.json', {'pid': proc.pid, 'argv': argv, 'timeout_s': 170})
        try:
            rc = proc.wait(timeout=170)
        except BaseException as exc:
            write(args.out / 'supervisor-failure.json', {'error': repr(exc), 'pid': proc.pid})
            # Only this Popen's direct child is signalled, never a name match.
            if proc.poll() is None:
                proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                if proc.poll() is None:
                    proc.kill()
                proc.wait(timeout=3)
            rc = 1
        finally:
            check = subprocess.run(['/bin/ps', '-p', str(proc.pid), '-o', 'pid='], text=True, capture_output=True)
            write(args.out / 'cleanup.json', {'pid': proc.pid, 'exit_code': proc.returncode,
                  'ps_returncode': check.returncode, 'ps_stdout': check.stdout, 'ps_stderr': check.stderr,
                  'owned_pid_absent': check.returncode == 1 and check.stdout == '' and check.stderr == ''})
    assert check.returncode == 1 and check.stdout == '' and check.stderr == ''
    print(f'child exit={rc}; reaped; evidence={args.out}')
    return rc


if __name__ == '__main__':
    sys.exit(main())
