"""E026 tiny allocation diagnostic. Default/help are stdlib CPU only.

Print the fixed diagnostic contract by default. --run --arm {off,on-lazy,
on-materialize} --out NEW_DIR explicitly imports MLX and uses the GPU. A future
root owner must launch each arm fresh, enforce its hard deadline, and own the
resource/service contract. This file neither starts a listener nor loads weights.
No full-model numeric qualification, E023 cause or speed result follows from it.
"""
import argparse
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import struct
import sys
import time
import traceback


HERE = Path(__file__).resolve().parent
CANDIDATE_SHA = '4d9658d34734e3c252375c4b76fcb2eccf83555121d7eb51d7f3e81eb2c48ccc'
SOURCE_MANIFEST_SHA = '3f749bb8608c6693903cc4a4579bbcd76cefcd1cf8069e097e61f7b1700dee32'
LENGTHS = (1, 19, 935, 978)
WIDTH, ROWS, WORDS = 10240, 3, 30720
TAIL_BYTES, MEMORY_GUARD_BYTES, DEADLINE_SECONDS = 61440, 64 << 20, 30
ARMS = ('off', 'on-lazy', 'on-materialize')
KINDS = ('bits-0', 'bits-30720', 'bits-61440', 'finite')


def need(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    return digest(Path(path).read_bytes())


def fixture_bytes(kind):
    """Little-endian BF16 payload words; no BF16 floating arithmetic/conversion."""
    need(kind in KINDS, 'unknown fixture kind')
    if kind == 'finite':
        # Includes positive/negative zero; every exponent is below 255.
        words = [((i // 2) % 0x7f80) | ((i % 2) << 15) for i in range(WORDS)]
    else:
        offset = int(kind.split('-')[1])
        words = [(offset + i) & 0xffff for i in range(WORDS)]
    return struct.pack('<' + 'H' * WORDS, *words)


def contract(arm=None):
    parent = (max(LENGTHS) + ROWS) * WIDTH * 2
    return {'schema': 'E026-tiny-backing-v1', 'arm': arm, 'lengths': list(LENGTHS),
        'fixture_kinds': list(KINDS), 'case_count': len(LENGTHS) * len(KINDS),
        'shape': [ROWS, WIDTH], 'dtype': 'bfloat16', 'payload_encoding': 'little-endian uint16 words',
        'all_word_payloads_covered_per_length': 65536, 'tail_logical_bytes': TAIL_BYTES,
        'parent_rule': 'host uint16 [T+3,10240], prefix 0x3f80, last 3 rows fixture; GPU equal-size BF16 view',
        'cache_limit_bytes': 0, 'memory_boundary_guard_bytes': MEMORY_GUARD_BYTES,
        'largest_parent_logical_bytes': parent,
        'conservative_named_buffer_envelope_bytes': 3 * parent + 4 * TAIL_BYTES,
        'envelope_scope': 'At most host parent + uploaded parent + optional host validation copy, plus small tail views/copy/bytes. Excludes runtime and hidden allocator workspace; not a hard process-memory cap.',
        'memory_guard_scope': 'active/peak checked after each recorded boundary; cannot prevent transient excess. No set_memory_limit hard-cap claim.',
        'child_best_effort_alarm_seconds': DEADLINE_SECONDS,
        'owner_requirement': 'Fresh process per arm; external hard deadline <=30s plus bounded reap, approved resource and service policy. No owner implemented here.',
        'candidate_sha256': CANDIDATE_SHA, 'source_manifest_sha256': SOURCE_MANIFEST_SHA,
        'memory_gate': 'ON distinct compact buffer and old-parent active-byte drop after sources/gc/sync; OFF retains original backing. All output bits and full parent digest unchanged.',
        'scope': 'Single contiguous synthetic parent/tail only; no strided/compact control, convolution, weights, observer, full-model correctness or throughput/latency measurement.'}


def verify_sources():
    paths = {HERE / 'candidate.py': CANDIDATE_SHA, HERE / 'source-pins.json': SOURCE_MANIFEST_SHA}
    need(all(file_sha(p) == expected for p, expected in paths.items()), 'candidate/source manifest changed')
    manifest = json.loads((HERE / 'source-pins.json').read_text())
    for item in manifest['sources'].values():
        path = (HERE / item['copy']).resolve()
        need(path.is_relative_to(HERE), 'source copy outside directory')
        paths[path] = item['sha256']
    need(all(file_sha(p) == expected for p, expected in paths.items()), 'copied MLX source changed')
    paths[Path(__file__).resolve()] = file_sha(__file__)
    return {str(p): expected for p, expected in paths.items()}


def load_candidate():
    need(file_sha(HERE / 'candidate.py') == CANDIDATE_SHA, 'candidate SHA mismatch before import')
    spec = importlib.util.spec_from_file_location('e026_owned_tail_candidate', HERE / 'candidate.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    need(Path(module.own_conv_tail.__code__.co_filename).resolve() == HERE / 'candidate.py', 'candidate actual callable path')
    return module


def snapshot(mx, groups, label, case_id, started):
    """Caller supplies evaluated roots only. Return numbers, never array refs."""
    api = getattr(mx, 'get_array_buffer_size', None)
    need(callable(api), 'required get_array_buffer_size API unavailable')
    return {'case': case_id, 'phase': label, 'pid': os.getpid(), 'ppid': os.getppid(),
        'monotonic_s': time.monotonic(), 'elapsed_s': time.monotonic() - started,
        'active_bytes': int(mx.get_active_memory()), 'peak_bytes': int(mx.get_peak_memory()),
        'cache_bytes': int(mx.get_cache_memory()),
        'unique_buffer_bytes': int(api(*groups.values())),
        'group_buffer_bytes': {name: int(api(tree)) for name, tree in groups.items()},
        'groups_are_evaluated': True}


def guard_snapshot(row):
    for key in ('active_bytes', 'peak_bytes', 'cache_bytes', 'unique_buffer_bytes'):
        need(type(row[key]) is int and row[key] >= 0, 'invalid memory measurement')
    need(row['active_bytes'] < MEMORY_GUARD_BYTES and row['peak_bytes'] < MEMORY_GUARD_BYTES,
         'post-boundary active/peak reached 64MiB; transient hard cap was not guaranteed')
    need(row['unique_buffer_bytes'] <= row['active_bytes'], 'backing union exceeds active accounting')


def ownership_checks(arm, original, evaluated, before_drop, after_sync, output_dropped):
    parent = original['group_buffer_bytes']['parent']
    need(original['group_buffer_bytes']['tail'] == parent and original['unique_buffer_bytes'] == parent,
         'original tail did not retain exactly the parent buffer')
    output = evaluated['group_buffer_bytes']['output']
    need(after_sync['unique_buffer_bytes'] == output, 'retained output backing changed unexpectedly')
    if arm == 'off':
        need(output == parent and evaluated['unique_buffer_bytes'] == parent, 'OFF failed to preserve original alias')
    else:
        need(TAIL_BYTES <= output <= TAIL_BYTES + 16384 and output < parent, 'ON output not compact')
        need(evaluated['unique_buffer_bytes'] == parent + output, 'ON did not establish distinct allocation')
        need(before_drop['active_bytes'] - after_sync['active_bytes'] >= parent,
             'ON compact output exists but old backing active-byte release not established')
    need(output_dropped['unique_buffer_bytes'] == 0, 'array root unexpectedly survived final drop')
    return {'original_backing_bytes': parent, 'retained_output_backing_bytes': output,
        'active_bytes_released_after_source_drop': before_drop['active_bytes'] - after_sync['active_bytes'],
        'retained_backing_reduction_bytes': parent - output, 'source_release_proven_for_on': arm != 'off'}


def raw_words(np, mx, value):
    # Local temporary view/host array does not escape and retain the parent.
    return np.asarray(value.view(mx.uint16)).astype('<u2', copy=False).tobytes()


def full_parent_sha(np, mx, parent):
    # Source is contiguous. No numerical cast and no permanent oracle alias.
    words = np.asarray(parent.view(mx.uint16))
    return digest(memoryview(words).cast('B'))


def one_case(mx, np, candidate, arm, length, kind, out, record):
    case_id = f'T{length}-{kind}'
    expected = fixture_bytes(kind)
    fixture_path = out / (case_id + '.expected-bf16-u16le.bin')
    fixture_path.write_bytes(expected)
    mx.clear_cache(); mx.reset_peak_memory()
    start = record('before_parent', {}, case_id)
    host = np.full((length + ROWS, WIDTH), 0x3f80, dtype=np.uint16)
    host[-ROWS:] = np.frombuffer(expected, dtype='<u2').reshape(ROWS, WIDTH)
    expected_parent_sha = digest(memoryview(host).cast('B'))
    parent = mx.array(host).view(mx.bfloat16)
    mx.eval(parent)
    del host
    tail = parent[length:]
    mx.eval(tail)
    original = record('after_parent_and_tail_eval', {'parent': parent, 'tail': tail}, case_id)
    output = candidate.own_conv_tail(mx, tail, enabled=arm != 'off', materialize=arm == 'on-materialize')
    need((output is tail) == (arm == 'off'), 'helper route silently fell back or OFF lost identity')
    # ON-lazy output is deliberately excluded: API requires evaluated arrays.
    graph_groups = {'parent': parent, 'tail': tail}
    if arm != 'on-lazy':
        graph_groups['output'] = output
    record('after_helper_graph_build', graph_groups, case_id)
    del graph_groups
    mx.eval(output)
    evaluated = record('after_output_eval', {'parent': parent, 'tail': tail, 'output': output}, case_id)
    need(output.dtype == mx.bfloat16 and output.shape == (ROWS, WIDTH), 'original dtype/shape changed')
    actual = raw_words(np, mx, output)
    actual_path = out / (case_id + '.actual-bf16-u16le.bin')
    actual_path.write_bytes(actual)
    need(actual == expected and raw_words(np, mx, tail) == expected, 'tail raw uint16 bits differ')
    actual_parent_sha = full_parent_sha(np, mx, parent)
    need(actual_parent_sha == expected_parent_sha, 'original parent mutated')
    finite = None
    if kind == 'finite':
        finite = bool(mx.all(mx.isfinite(output)).item())
        need(finite, 'finite fixture became nonfinite')
    before_drop = record('after_validation_before_drop', {'parent': parent, 'tail': tail, 'output': output}, case_id)
    del parent, tail
    record('after_drop_sources', {'output': output}, case_id)
    gc.collect()
    record('after_gc', {'output': output}, case_id)
    mx.synchronize()
    after_sync = record('after_sync', {'output': output}, case_id)
    mx.clear_cache()
    record('after_clear_cache', {'output': output}, case_id)
    del output
    record('after_drop_output', {}, case_id)
    gc.collect(); mx.synchronize(); mx.clear_cache()
    dropped = record('after_final_gc_sync_clear_cache', {}, case_id)
    memory = ownership_checks(arm, original, evaluated, before_drop, after_sync, dropped)
    need(dropped['active_bytes'] <= start['active_bytes'], 'active arrays remain after all experiment roots released')
    return {'case': case_id, 'status': 'PASS', 'length': length, 'kind': kind,
        'shape': [ROWS, WIDTH], 'dtype': 'bfloat16', 'bit_equal': True,
        'finite': finite, 'nan_payloads_are_data_not_finite_qualification': kind != 'finite',
        'expected_file': fixture_path.name, 'actual_file': actual_path.name,
        'expected_sha256': digest(expected), 'actual_sha256': digest(actual),
        'parent_sha256_before': expected_parent_sha, 'parent_sha256_after': actual_parent_sha,
        'parent_logical_bytes': (length + ROWS) * WIDTH * 2, 'memory': memory}


def execute(args):
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    result = {'status': 'INCOMPLETE', 'arm': args.arm, 'pid': os.getpid(), 'ppid': os.getppid(),
              'contract': contract(args.arm), 'cases': [], 'device_run': True}
    def write_result():
        (args.out / 'complete.json').write_text(json.dumps(result, indent=2) + '\n')
    def stop(*unused):
        raise TimeoutError('30s child best-effort deadline; external owner must enforce hard timeout')
    need(signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0), 'existing alarm unsupported')
    previous = {sig: signal.signal(sig, stop) for sig in (signal.SIGALRM, signal.SIGTERM, signal.SIGINT)}
    signal.setitimer(signal.ITIMER_REAL, DEADLINE_SECONDS)
    write_result()
    try:
        result['source_pins'] = verify_sources()
        # Explicit --run is the only route to these imports.
        import mlx.core as mx
        import numpy as np
        candidate = load_candidate()
        need(candidate.supported_backend(mx), 'pinned MLX/M5 Ultra/g17d domain required')
        need(callable(getattr(mx, 'get_array_buffer_size', None)), 'buffer API unavailable')
        need(sys.byteorder == 'little', 'fixed payload fixture requires little-endian host')
        result['runtime'] = {'mlx_file': mx.__file__, 'numpy_file': np.__file__,
                             'device': mx.device_info(), 'metal': mx.metal.device_info()}
        mx.set_cache_limit(0)
        gc.collect(); mx.synchronize(); mx.clear_cache(); mx.reset_peak_memory()
        with (args.out / 'phases.jsonl').open('x') as ledger:
            def record(label, groups, case_id):
                row = snapshot(mx, groups, label, case_id, started)
                ledger.write(json.dumps(row) + '\n'); ledger.flush()
                guard_snapshot(row)
                if time.monotonic() - started >= DEADLINE_SECONDS:
                    stop()
                return row
            for length in LENGTHS:
                for kind in KINDS:
                    result['cases'].append(one_case(mx, np, candidate, args.arm, length, kind, args.out, record))
                    write_result()
        need(verify_sources() == result['source_pins'], 'source changed during diagnostic')
        result.update(status='PASS', source_pins_rechecked=True,
                      scope='Tiny backing/bit diagnostic only; no full model, integration, E023 cause or performance claim')
    except BaseException as error:
        result.update(status='FAIL', error=str(error), traceback=traceback.format_exc())
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        # Do not add an unbounded GPU cleanup after deadline/failure. Process exit
        # and the external owner's direct reap release this isolated child.
        result['elapsed_wall_s'] = time.monotonic() - started
        write_result()
    print(json.dumps({'status': result['status'], 'out': str(args.out)}))
    return int(result['status'] != 'PASS')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--arm', choices=ARMS)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args(argv)
    if not args.run:
        if args.out:
            parser.error('--out requires --run; default creates no files and imports no device runtime')
        print(json.dumps(contract(args.arm), indent=2)); return 0
    if args.arm is None or args.out is None:
        parser.error('--run requires --arm and --out')
    return execute(args)


if __name__ == '__main__':
    raise SystemExit(main())
