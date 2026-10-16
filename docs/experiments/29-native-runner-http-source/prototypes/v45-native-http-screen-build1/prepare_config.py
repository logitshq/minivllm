"""CPU-only binding of a completed runner qualification; never launches anything."""
import argparse
import hashlib
import json
from pathlib import Path

D = Path(__file__).resolve().parent
P = D.parent


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    with Path(path).open('x') as f: json.dump(value, f, indent=2, allow_nan=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--receipt', type=Path, required=True)
    ap.add_argument('--receipt-sha256', required=True)
    ap.add_argument('--qual-dir', type=Path, required=True)
    ap.add_argument('--candidate', type=Path, required=True)
    ap.add_argument('--library', type=Path, required=True)
    ap.add_argument('--library-sha256', required=True)
    ap.add_argument('--mode', choices=('conv', 'normreuse_cache', 'conv_normreuse_cache'), required=True)
    args = ap.parse_args()
    assert not (D / 'attempt.consumed.json').exists()
    assert sha(args.receipt) == args.receipt_sha256
    assert sha(args.library) == args.library_sha256
    q = json.loads(args.receipt.read_text())
    assert q['status'] == 'PASS_NATIVE_RUNNER_C1_QUEUED_RESET_FALLBACK_STRICT'
    assert q['model_loads'] == 1 and q['terminal_cleanup'] and q['captured_originals_readonly']
    assert q['strict_output_checks'] == 12408 and q['native_steps'] == 124 and q['request_count'] == 4
    f = q['native']['feedback']
    assert f['icb']['execution_count'] == 124 and f['icb']['bank_executions'] == [64, 60]
    assert f['pingpong']['eligible_count'] == 48
    source_pins = json.loads((args.qual_dir / 'source-pins.json').read_text())
    for path, digest in source_pins.items(): assert sha(path) == digest, path
    source_pins[str(args.receipt)] = args.receipt_sha256
    source_pins[str(args.qual_dir / 'source-pins.json')] = sha(args.qual_dir / 'source-pins.json')
    package = args.candidate / 'tutorial_code/miniengine'
    files = {str(p.relative_to(package)): sha(p) for p in package.rglob('*.py')}
    assert len(files) in (26, 27)
    for name, digest in files.items():
        assert source_pins[str(package / name)] == digest
    adapter = P / 'v41-native-runner-build1/native_runner.py'
    assert sha(adapter) == 'bf7bea08b09dfc2905966b95e77d244ec25947a0c3a0d9506cddd05ca81cfea0'
    assert source_pins[str(adapter)] == sha(adapter)
    assert source_pins[str(args.library)] == args.library_sha256
    prior = P / 'm4-final-native15-official-attempt2'
    old_pins = json.loads((prior / 'pins.json').read_text())
    for path, digest in old_pins['native_binaries'].items():
        assert sha(path) == digest
        source_pins[path] = digest
    config = {'mode': args.mode, 'model_dir': json.loads((P / 'workload.json').read_text())['model_dir'],
              'package_root': str(args.candidate), 'package_sources': files,
              'native_library': str(args.library), 'native_library_sha256': args.library_sha256,
              'adapter_source': str(adapter), 'scheduler_max_seqs': 1, 'fixed_cap': 256,
              'fullqual_receipt': str(args.receipt), 'fullqual_receipt_sha256': args.receipt_sha256,
              'fullqual_status': q['status'], 'source_pins': source_pins}
    write_new(D / 'runtime-config.json', config)
    controls = {str(p): sha(p) for p in D.glob('*.py')}
    controls[str(D / 'runtime-config.json')] = sha(D / 'runtime-config.json')
    for name in ('common.py', 'client.py', 'official_entry.py', 'registry.json', 'pins.json'):
        controls[str(prior / name)] = sha(prior / name)
    for name in ('metal_buffer_bootstrap_v1.py', 'power_paired_support_v1.py',
                 'resource_snapshot_power_v1.py', 'resource_snapshot_precise_v1.py', 'workload.json'):
        controls[str(P / name)] = sha(P / name)
    write_new(D / 'pins.json', {'files': controls, 'runtime_config_sha256': sha(D / 'runtime-config.json'),
        'scope': 'C1 NOL/OLN/LNO; same official trees/client definitions; no Pareto victory asserted'})
    print(json.dumps({'status': 'PASS_CPU_CONFIG_BOUND', 'pins_sha256': sha(D / 'pins.json'),
                      'source_pins': len(source_pins), 'controller_pins': len(controls), 'GPU_used': False}))


if __name__ == '__main__': main()
