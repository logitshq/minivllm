"""Archive one closed CPU toy attempt; no tests, runtime or device operations."""
from pathlib import Path, PurePosixPath
import hashlib
import json
import sys
import tarfile

R = Path('/Users/ryanli/code/minivllm')
A = R / '.artifacts/m5-ultra-256gb/state-log-exact-cpu-v1'
O = R / 'docs/experiments/m5-ultra-256gb/state-log-exact-cpu-v1'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write(p, x):
    with p.open('x') as stream:
        json.dump(x, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def listing(root):
    result = {}
    for p in sorted(root.rglob('*')):
        if p.is_symlink():
            raise ValueError('symlink in CPU evidence: '+str(p))
        if p.is_file():
            result[str(p.relative_to(root))] = p
        elif not p.is_dir():
            raise ValueError('nonregular CPU evidence: '+str(p))
    return result


def main():
    if sys.argv[1:] != ['--run']:
        print('DESCRIPTION_ONLY: --run archives the closed first CPU attempt')
        return 0
    result = json.loads((A/'result.json').read_bytes())
    toy = json.loads((A/'toy-result.json').read_bytes())
    summary = json.loads((A/'summary.json').read_bytes())
    pins = json.loads((A/'source-pins.json').read_bytes())
    assert result['status'] == 'PASS' and result['expected_tests'] == 8
    assert result['runs'] == [{'name':'tests','returncode':0},{'name':'toy','returncode':0}]
    assert result['source_unchanged'] and result['toy_result_sha256'] == sha(A/'toy-result.json')
    assert result['runner_sha256'] == sha(A/'runner.py')
    assert toy['status'] == 'PASS' and toy['toy_bit_exact'] and len(toy['cases']) == 24
    assert toy['algebraic_counterexample']['sequential'] == 0 and toy['algebraic_counterexample']['combined_update_sum'] == 1
    assert summary['toy_result_sha256'] == sha(A/'toy-result.json') and summary['result_sha256'] == sha(A/'result.json')
    assert 'Ran 8 tests' in (A/'tests.stderr').read_text() and (A/'tests.stderr').read_text().rstrip().endswith('OK')
    for name, pin in pins.items():
        copy = Path(pin['copy'])
        assert copy.parent == A/'sources' and copy.is_file() and not copy.is_symlink()
        assert sha(copy) == sha(R/name) == pin['sha256']
        assert copy.stat().st_size == (R/name).stat().st_size == pin['bytes']
    originals = listing(A)
    O.mkdir(exist_ok=False)
    with (O/'archive.py').open('xb') as stream:
        stream.write(Path(__file__).read_bytes())
    (O/'README.md').write_text('''# Ordered state-log: first CPU evidence\n\nThe first stdlib-only attempt passed all 8 tests and 24 toy trajectories. It compared full FP32 states and outputs after 2,304 updates, plus clone/reset/retire behavior. The deliberately algebraic delta merge produces 1 where ordered replay produces 0; this expected counterexample is retained, not tolerated away.\n\nThe archive contains every original attempt file: runner, commands, stdout/stderr, exact source snapshots, results and summary. The original source snapshot includes the read-only native kernel used for contract comparison. Source paths and every member SHA/size are in manifest.json. archive-verification.json records decompressed-member and original-tree rechecks. No test was rerun during archiving.\n\nThis is a CPU arithmetic toy, not native Metal equivalence, model/HTTP qualification or performance evidence. Persistent-write counts are analytical only.\n''')
    inputs = {'cpu/'+name:p for name,p in originals.items()}
    inputs.update({'archive-metadata/archive.py':O/'archive.py',
                   'archive-metadata/README.md':O/'README.md',
                   'source-notes/RESULTS.md':R/'bench/m5_ultra_256gb/state_log_exact_reference/RESULTS.md'})
    records = {name:{'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size} for name,p in inputs.items()}
    write(O/'manifest.json', records)
    target = O/'evidence.tar.xz'
    with target.open('xb') as stream:
        with tarfile.open(fileobj=stream, mode='w:xz', preset=1, dereference=True) as tar:
            for name,p in sorted(inputs.items()):
                assert not p.is_symlink()
                tar.add(p, arcname=name, recursive=False)
    seen = set()
    with tarfile.open(target, 'r:xz') as tar:
        for member in tar:
            path = PurePosixPath(member.name)
            assert member.isfile() and not path.is_absolute() and '..' not in path.parts
            assert member.name in records and member.name not in seen
            data = tar.extractfile(member).read()
            record = records[member.name]
            assert member.size == len(data) == record['bytes']
            assert hashlib.sha256(data).hexdigest() == record['sha256']
            seen.add(member.name)
    assert seen == set(records)
    assert listing(A) == originals
    for name,p in inputs.items():
        assert not p.is_symlink() and sha(p) == records[name]['sha256'] and p.stat().st_size == records[name]['bytes']
    for name,pin in pins.items():
        assert sha(R/name) == pin['sha256'] and (R/name).stat().st_size == pin['bytes']
    verification = {'schema':'state-log-exact-cpu-archive-v1','status':'PASS',
                    'tests':8,'toy_cases':24,'members':len(records),
                    'logical_bytes':sum(v['bytes'] for v in records.values()),
                    'archive_bytes':target.stat().st_size,'archive_sha256':sha(target),
                    'manifest_sha256':sha(O/'manifest.json'),
                    'all_members_decompressed_and_verified':True,
                    'original_tree_membership_and_bytes_rechecked':True,
                    'original_sources_rechecked':True,'tests_rerun':False,
                    'metal_qualified':False,'model_qualified':False,'performance_qualified':False}
    write(O/'archive-verification.json', verification)
    print(json.dumps(verification,sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
