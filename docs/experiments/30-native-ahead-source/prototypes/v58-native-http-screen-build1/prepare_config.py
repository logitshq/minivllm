"""CPU-only binding of a completed runner qualification; never launches anything."""
import argparse
import hashlib
import json
from pathlib import Path

D = Path(__file__).resolve().parent
P = D.parent
QUAL_SHA = '3405239d749a876a09f896a3ef06fc1f6603a317f24ba87c7711b53782435089'
QUAL_RELEASE = P / 'v23-root-control-build1/fullmodel-v56native-attempt1/external-release.json'
QUAL_RELEASE_SHA = '778ffbc60f73ce56d9c0e990ed0c7cd16f4db81a74ad57f34754821fb0587c39'


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
    assert QUAL_SHA is not None and args.receipt_sha256 == QUAL_SHA
    assert args.receipt == P / 'v23-root-control-build1/fullmodel-v56native-attempt1/child-output/receipt.json'
    assert args.qual_dir == P / 'v56-native-async-runner-build1'
    assert args.candidate == P / 'candidate-v44-conv-normreuse-build1-root'
    assert args.library == P / 'v55-native-async-build2/libv55replay.dylib'
    assert args.library_sha256 == 'bb3c6ebb035eb204a09e4e99828aa7438a11ff991a3a883cdb310a4128fa1e3e'
    assert args.mode == 'conv_normreuse_cache'
    assert sha(QUAL_RELEASE) == QUAL_RELEASE_SHA
    released = json.loads(QUAL_RELEASE.read_text())
    assert released['status'] == 'PASS_EXTERNAL_RELEASE' and released['GPU_released'] and released['tool_exit'] == 0
    assert sha(args.receipt) == args.receipt_sha256
    assert sha(args.library) == args.library_sha256
    q = json.loads(args.receipt.read_text())
    assert q['status'] == 'PASS_ASYNC_NATIVE_RUNNER_C1_QUEUED_RESET_EARLYSTOP_FALLBACK_STRICT'
    assert q['model_loads'] == 1 and q['terminal_cleanup'] and q['captured_originals_readonly']
    assert q['strict_output_checks'] == 12540 and q['native_steps'] == 161 and q['request_count'] == 6
    assert q['native_completed_steps'] == q['total_submitted_steps'] == 162 and q['total_discarded_steps'] == 1
    assert q['normal_strict_output_checks'] == 12408 and q['early_stop']['all132_output_checks'] == 0
    assert q['early_stop']['native_delivered_delta'] == 6 and q['early_stop']['native_completed_delta'] == 7 and q['early_stop']['discarded_delta'] == 1
    f = q['native']['feedback']
    assert f['icb']['execution_count'] == 162 and f['icb']['bank_executions'] == [84, 78]
    p = q['native']['pipeline']
    assert p['enabled'] and p['depth'] == 2 and p['maximum_outstanding'] == 2
    assert p['total_submitted'] == p['total_completed'] == 162 and p['total_delivered'] == 161
    assert p['total_discarded'] == 1 and p['total_failed'] == 0 and p['inflight'] == 0 and p['pending'] == []
    assert f['pingpong']['eligible_count'] == 48
    source_pins = json.loads((args.qual_dir / 'source-pins.json').read_text())
    for path, digest in source_pins.items(): assert sha(path) == digest, path
    source_pins[str(args.receipt)] = args.receipt_sha256
    source_pins[str(QUAL_RELEASE)] = QUAL_RELEASE_SHA
    source_pins[str(args.qual_dir / 'source-pins.json')] = sha(args.qual_dir / 'source-pins.json')
    package = args.candidate / 'tutorial_code/miniengine'
    files = {str(p.relative_to(package)): sha(p) for p in package.rglob('*.py')}
    assert len(files) in (26, 27)
    for name, digest in files.items():
        assert source_pins[str(package / name)] == digest
    adapter = P / 'v56-native-async-runner-build1/native_runner.py'
    assert sha(adapter) == 'af252476a377e9624173a8dbb028650c283b56d5beb0248639b0f00c271dbd59'
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
