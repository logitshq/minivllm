"""V23 local host-wall stationary micro. CPU import is stdlib-only.
No model/E2E conclusion follows from a local micro result. Root owns GPU process.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import statistics
import sys
import time

D = Path(__file__).resolve().parent
P = D.parent
NUMERIC = P / 'v23-conv-numeric-build1'
PROTO = P / 'v23-gdn-conv-silu-prototype'
NUMERIC_RECEIPT = P / 'v23-root-control-build1/numeric-attempt1/child-output/receipt.json'
NUMERIC_SHA = 'f4eb0d3a39af3358b1b5b1b5e2bdd887681fd0b1c414602ed89504ff65c45020'
CHECKPOINT = Path('/Users/ryanli/.cache/huggingface/hub/models--mlx-community--Qwen3.8-27B-nvfp4/snapshots/5ff8ef173ad0d7c3aae92f0be43031a6ab8067c6')
PRIME, REPEATS, SEED = 128, 4096, 20261016023
ORDERS = (('A', 'B'), ('B', 'A')) * 3
SCHEDULE = (('A', 'A'), ('B', 'B')) + ORDERS + (('A', 'A'), ('B', 'B'))
CONTROLS = {'MLX_MAX_OPS_PER_BUFFER': '50', 'MLX_MAX_MB_PER_BUFFER': '50',
            'MLX_BFS_MAX_WIDTH': '20', 'MLX_METAL_FAST_SYNCH': '0'}
BANK_NAMES = ('zero', 'alternating-tail', 'random-amp1/16', 'random-amp1',
              'random-amp16', 'head-boundaries', 'zero-tail-nonzero-raw', 'small-nonzero-tail')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def control_passes(control):
    return all(len(ratios) == 2 and .98 <= statistics.median(ratios) <= 1.02
               and all(.95 <= ratio <= 1.05 for ratio in ratios) for ratios in control.values())


def numeric_check(path):
    if sha(path) != NUMERIC_SHA:
        raise ValueError('only the independently completed numeric PASS receipt is eligible')
    value = json.loads(Path(path).read_text())
    if value['status'] != 'PASS_CONV_NUMERIC_SYNTHETIC_ONLY' or value['GPU_numeric_run'] is not True:
        raise ValueError('numeric PASS is required before micro execution')
    if len(value['cases']) != 6 or len(value['chain']) != 8 or value['sum_y_evidence']['explicit_sum_y'] != 4:
        raise ValueError('numeric scope or explicit branch differs')
    expected = json.loads((NUMERIC / 'source-pins.json').read_text())
    if value['source_pins'] != expected:
        raise ValueError('numeric source identity differs')
    for case in value['cases'] + value['chain']:
        if not case['input_readonly'] or case['input_sha256_before'] != case['input_sha256_after']:
            raise ValueError('numeric input immutability missing')
        if set(case['checks']) != {'conv', 'act', 'tail', 'probe_tail', 'o', 'S_new'}:
            raise ValueError('numeric checks missing')
        if not all(v['strict_equal'] is True for v in case['checks'].values()):
            raise ValueError('numeric strict byte check missing')
        layout = case['product_layout']
        if layout['shape'] != [1, 4, 10240] or layout['strides_elements'] != [40960, 10240, 1]:
            raise ValueError('actual numeric product layout differs')
    return {'path': str(Path(path).resolve()), 'sha256': NUMERIC_SHA, 'scope': value['status'],
            'cases': 6, 'chain_steps': 8, 'sum_y': 4, 'real_preconv_qualified': False}


def identity(numeric_receipt):
    pins = json.loads((D / 'source-pins.json').read_text())
    for filename, expected in pins.items():
        if sha(filename) != expected:
            raise ValueError('source changed: ' + filename)
    eligible = numeric_check(numeric_receipt)
    return {'source_pins': pins, 'numeric_eligibility': eligible, 'driver_sha256': sha(__file__),
            'timing_block_sha256': sha(D / 'timing_block.py'),
            'plan': {'seed': SEED, 'prime': PRIME, 'repeats': REPEATS, 'schedule': SCHEDULE,
                     'bank_names': BANK_NAMES, 'outputs_consumed_each_call': ['act', 'new_tail'],
                     'power_source': 'AC Power', 'controls': CONTROLS,
                     'minimum_median_reduction': .05, 'minimum_wins': 5,
                     'control_median_bounds': [.98, 1.02], 'control_each_bounds': [.95, 1.05],
                     'timing_scope': 'local Python/build/eval/sync host wall only; no GPU kernel or E2E attribution',
                     'prime_repeat_reason': 'one light C10240 group:4096 fixed repeats vs V21 512, predeclared before any timing'}}


def make_pool(np, mx):
    rng = np.random.Generator(np.random.PCG64(SEED))
    channels = np.arange(10240, dtype=np.float32)
    pool = []
    for index, name in enumerate(BANK_NAMES):
        scale = (1., 1., 1/16, 1., 16., 1., 1., 1/4)[index]
        raw = rng.uniform(-scale, scale, (1, 10240)).astype(np.float32)
        tail = rng.uniform(-scale, scale, (1, 3, 10240)).astype(np.float32)
        if index == 0:
            raw.fill(0); tail.fill(0)
        elif index == 1:
            raw[:] = np.where(channels % 2, -1., 1.)
            tail[:] = np.stack([raw[0], -raw[0], raw[0] / 2])[None]
        elif index == 5:
            boundary = (channels // 128 - 39) / np.float32(64) + (channels % 128) / np.float32(4096)
            raw[:] = boundary
            tail[:] = np.stack([-boundary, boundary / 2, -boundary / 4])[None]
        elif index == 6:
            tail.fill(0)
        pool.append((name, mx.array(raw).astype(mx.bfloat16), mx.array(tail).astype(mx.bfloat16)))
    mx.eval(*[value for _, raw, tail in pool for value in (raw, tail)])
    return pool


def gpu(args, ident):
    versions = {k: importlib.metadata.version(k) for k in ('mlx', 'mlx-metal')}
    if versions != {'mlx': '0.32.3', 'mlx-metal': '0.32.3'}:
        raise ValueError('requires the numeric PASS runtime MLX/mlx-metal0.32.3')
    if {key: os.environ.get(key) for key in CONTROLS} != CONTROLS:
        raise ValueError('MLX controls must match the fixed stationary plan')
    import mlx.core as mx
    import mlx.nn as nn
    import numpy as np
    banks = module('v23_micro_small_banks', NUMERIC / 'banks.py')
    reference = module('v23_micro_accepted_reference', NUMERIC / 'reference.py')
    candidate = module('v23_micro_archived_candidate', PROTO / 'conv_silu_raw_tail.py')
    timing = module('v23_micro_stationary_timing', D / 'timing_block.py')
    resource = module('v23_micro_resource_probe', P / 'resource_snapshot_power_v1.py')
    pipeline = json.loads((P / 'v23-root-control-build1/native-pipeline.json').read_text())
    device = mx.device_info()
    if mx.default_device() != mx.gpu or device.get('device_name') != pipeline['device'] or device.get('architecture') != pipeline['architecture']:
        raise ValueError('actual GPU differs from numeric/PSO device')
    mx.set_cache_limit(64 << 20)
    mx.set_memory_limit(512 << 20)
    args.out.mkdir(exist_ok=False)
    receipt = {'status': 'STARTED', 'identity': ident, 'runtime': {'versions': versions, 'device': device},
               'pairs': [], 'checks_before': [], 'checks_after': [], 'resources': [],
               'adopted': False, 'model_qualified': False, 'E2E_run': False}
    write(args.out / 'receipt.json', receipt)
    resources = receipt['resources']
    start = time.perf_counter()
    try:
        def probe(label):
            mx.synchronize()
            s = resource.snapshot(label, os.getpid())
            s['mlx_memory'] = {'active': mx.get_active_memory(), 'cache': mx.get_cache_memory(), 'peak': mx.get_peak_memory()}
            resources.append(s)
            resource.deltas(resources)
            write(args.out / 'resources.json', resources)
            resource.guard(resources)
            if s['power_metadata']['status'] != 'known' or s['power_metadata']['source'] != 'AC Power':
                raise ValueError('fixed AC power domain required')
            if s['processinfo_thermal_metadata']['status'] != 'known' or s['processinfo_thermal_metadata']['code'] != 0 or s['pressure_level_value'] != 1:
                raise ValueError('thermal/pressure guard')
            if s['swap_used_mb'] > resources[0]['swap_used_mb'] or s['vm_counters']['Swapouts'] != resources[0]['vm_counters']['Swapouts']:
                raise ValueError('swap growth')
            if s['known_inference_process_count'] != 0 or s['exact_owned_pid_metadata']['status'] != 'known':
                raise ValueError('process inventory guard')
            return s

        probe('before-small-weight-load')
        weight, provenance = banks.checkpoint_conv(CHECKPOINT, 0, mx, np)
        pool = make_pool(np, mx)
        receipt['weight'] = provenance
        receipt['allocator_limits_bytes'] = {'cache': 64 << 20, 'memory': 512 << 20}
        timing.PRIME, timing.REPEATS = PRIME, REPEATS

        def raw(value):
            mx.eval(value)
            if value.dtype != mx.bfloat16:
                raise ValueError('expected BF16 output/input')
            words = np.ascontiguousarray(np.asarray(value.view(mx.uint16)), dtype='<u2')
            if np.any((words & 0x7f80) == 0x7f80):
                raise ValueError('nonfinite BF16')
            return words.tobytes()

        def inputs_hash():
            return {'weight': hashlib.sha256(raw(weight)).hexdigest(), 'pool': [
                {'name': name, 'raw': hashlib.sha256(raw(x)).hexdigest(), 'tail': hashlib.sha256(raw(t)).hexdigest()}
                for name, x, t in pool]}

        def values(variant, j):
            _, x, tail = pool[j % 8]
            if variant == 'A':
                _, act, new_tail, _ = reference.conv_reference(x, tail, weight, mx, nn)
                return act, new_tail
            return candidate.conv_silu_raw_tail(x, tail, weight, sum_y=4, backend=mx)

        def invoke(variant, j):
            outputs = values(variant, j)
            mx.eval(*outputs)  # BOTH act and raw tail consumed on EVERY call.
            return outputs

        def qualify(label):
            before = inputs_hash()
            if len({(r['raw'], r['tail']) for r in before['pool']}) != 8:
                raise ValueError('timing pool does not contain8 distinct raw/tail banks')
            checks = []
            for j, (name, _, _) in enumerate(pool):
                left, right = invoke('A', j), invoke('B', j)
                row = {'bank': name, 'outputs': []}
                for output_name, a, b in zip(('act', 'new_tail'), left, right):
                    if a.shape != b.shape or a.dtype != b.dtype or raw(a) != raw(b):
                        raise ValueError('strict ' + label + ' bytes/shape/dtype: ' + name + '/' + output_name)
                    row['outputs'].append({'name': output_name, 'shape': list(a.shape), 'dtype': str(a.dtype),
                                           'bytes': len(raw(a)), 'sha256': hashlib.sha256(raw(a)).hexdigest()})
                checks.append(row)
            after = inputs_hash()
            if before != after:
                raise ValueError('input mutation during strict qualification')
            return {'checks': checks, 'input_readonly': True, 'input_sha256': before}

        receipt['checks_before'] = qualify('before')
        probe('timing-ready')
        for index, order in enumerate(SCHEDULE):
            probe(f'pair{index}-before')
            before = inputs_hash()
            # No hash, JSON write, resource probe or conditional work between blocks.
            first, first_outputs = timing.block(mx, invoke, order[0])
            second, second_outputs = timing.block(mx, invoke, order[1])
            # Only after BOTH complete: symmetric output hashes and input readonly.
            for rec, outputs in ((first, first_outputs), (second, second_outputs)):
                hash_begin = time.perf_counter_ns()
                rec['hashes'] = [hashlib.sha256(raw(value)).hexdigest() for value in outputs]
                rec['hash_wall_ns'] = time.perf_counter_ns() - hash_begin
            after = inputs_hash()
            if before != after or first['hashes'] != second['hashes']:
                raise ValueError('paired output or readonly hash mismatch')
            item = {'pair': index, 'order': order, 'blocks': [first, second], 'input_sha256_before': before,
                    'input_sha256_after': after, 'kind': 'position-control' if order[0] == order[1] else 'AB',
                    'slot2_div_slot1': second['ns_per_call'] / first['ns_per_call']}
            if item['kind'] == 'AB':
                by = {first['variant']: first, second['variant']: second}
                item['reduction'] = 1 - by['B']['ns_per_call'] / by['A']['ns_per_call']
            receipt['pairs'].append(item)
            write(args.out / 'receipt.json', receipt)
            probe(f'pair{index}-after')
        receipt['checks_after'] = qualify('after')
        if receipt['checks_before']['checks'] != receipt['checks_after']['checks']:
            raise ValueError('before/after exact output signatures differ')
        if receipt['checks_before']['input_sha256'] != receipt['checks_after']['input_sha256']:
            raise ValueError('timing changed frozen timing inputs')
        probe('after-strict-final')
        ab = [p for p in receipt['pairs'] if p['kind'] == 'AB']
        controls = {v: [p['slot2_div_slot1'] for p in receipt['pairs'] if p['kind'] == 'position-control' and p['order'][0] == v] for v in ('A', 'B')}
        medians = {v: statistics.median(next(b['ns_per_call'] for b in p['blocks'] if b['variant'] == v) for p in ab) for v in ('A', 'B')}
        reduction, wins = 1 - medians['B'] / medians['A'], sum(p['reduction'] > 0 for p in ab)
        control_ok = control_passes(controls)
        receipt.update(status='CONTROL_DRIFT_STOP' if not control_ok else ('EXPLORATORY-PASS' if reduction >= .05 and wins >= 5 else 'STOP-MICRO-GATE'),
                       control_passed=control_ok, control_slot2_div_slot1=controls, medians_ns=medians,
                       median_reduction=reduction, wins=wins, all_bytes_exact=True,
                       timing_scope='local host wall only; no E2E/model/Pareto claim')
        return 0 if receipt['status'] == 'EXPLORATORY-PASS' else 2
    except Exception as error:
        receipt.update(status='FAIL_STOP', error={'type': type(error).__name__, 'message': str(error)})
        raise
    finally:
        receipt['wall_seconds'] = time.perf_counter() - start
        write(args.out / 'receipt.json', receipt)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode', choices=('cpu', 'run'))
    ap.add_argument('--numeric-receipt', type=Path, default=NUMERIC_RECEIPT)
    ap.add_argument('--out', type=Path)
    args = ap.parse_args(argv)
    ident = identity(args.numeric_receipt)  # All gates precede any MLX import.
    if args.mode == 'cpu':
        print(json.dumps({'status': 'CPU_READY_NUMERIC_PASS_BOUND', 'MLX_imported': False, 'GPU_run': False, 'identity': ident}))
        return 0
    if args.out is None:
        raise ValueError('run requires a NEW --out directory; root owns external timeout/PID cleanup')
    return gpu(args, ident)


if __name__ == '__main__':
    raise SystemExit(main())
