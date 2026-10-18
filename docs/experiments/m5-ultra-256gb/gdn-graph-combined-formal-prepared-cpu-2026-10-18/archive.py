"""Retain closed combined-formal CPU attempts; never execute tests or devices."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile
import time

ROOT = Path('/Users/ryanli/code/minivllm')
OUT = ROOT / 'docs/experiments/m5-ultra-256gb/gdn-graph-combined-formal-prepared-cpu-2026-10-18'
NAMES = ('gdn-graph-combined-formal-cpu-v1', 'gdn-graph-combined-formal-source-seal-v1',
         'combined-formal-lifecycle-cpu-v1', 'combined-formal-lifecycle-cpu-v2')
TREES = {name: ROOT / '.artifacts/m5-ultra-256gb' / name for name in NAMES}
TREES['combined-formal-archive-preflight-v1'] = ROOT / '.artifacts/m5-ultra-256gb/combined-formal-archive-preflight-v1'
HERE = ROOT / 'bench/m5_ultra_256gb/gdn_graph_combined_formal'
FILES = ('formal.py', 'test_formal_cpu.py', 'owner.py', 'controller.py', 'audit_restore.py',
         'audit_formal.py', 'test_owner_cpu.py', 'test_controller_cpu.py',
         'test_audit_restore_cpu.py', 'test_audit_formal_cpu.py', 'README.md', 'execute.py')
README = '''# Combined GDN + graph: formal CPU preparation

The factory's 12 checks passed. The lifecycle and independent-auditor suite's second full attempt passed all 58 checks (19 owner, 15 controller, 11 restoration and 13 formal-audit). These are 70 distinct CPU checks, not numerical, actual HTTP-history or performance qualification.

The first 58-check attempt remains intact: 53 passed and five failed because its runner rejected asyncio's SIGINT-handler installation before reaching the mocked HTTP body. The v2 runner suppresses only asyncio runner SIGINT handler changes without touching the OS; it retains the other process, network, signal and backend-import guards. All production and test sources stayed unchanged between these two attempts. Both runners, source snapshots, commands, results and full stdout/stderr are retained.

The initial factory12 ran with an UNSEALED cycle-auditor pin. A separate recorded seal replaced that one literal with the actual frozen auditor SHA, then verified the 42-source closure. It did not change factory arithmetic, protocol or lifecycle logic. The final factory source was also included in the subsequent 58-check source-before/source-after snapshots; the original 12-check source and outcome have not been relabeled as a rerun.

Both formal arms use the same combined server with route graph ON; GDN prework alone changes OFF/ON. Each fresh process has eight warm and 64 formal requests, three paired repetitions at C1 and C8 (12 processes; 864 total requests, 768 formal). The factory requires the independently qualified HTTP histories; these CPU receipts do not satisfy that gate. No formal timing or observed compiled-call claim is made here.

`execute.py` is retained as an additional source snapshot only. It was not in these suites' tested source registry, and this archive does not claim its execution was tested. Actual owners, journals, model weights and runtime payloads are excluded.

`raw.tar.xz` retains the complete four CPU/seal evidence trees and one packer-preflight failure tree and the explicit small source closure. `manifest.json` maps every regular member to its original path, size and SHA-256. Archive verification streams and checks all members without extracting, then checks the original tree memberships and bytes again. Symlinks, special members, duplicates, absolute paths and traversal are rejected. No original evidence was modified or tests rerun.

```sh
python3 archive.py --verify .
```

Optional `--originals` additionally rechecks the source/CPU files at their original absolute paths. Later intentional changes can invalidate that optional comparison without invalidating the archived bytes. The manifest, verification, summary, README and packer copy remain alongside the tar.
'''

def need(ok, message):
    if not ok: raise ValueError(message)
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path): return json.loads(path.read_bytes())
def save(path, data):
    with path.open('x') as f: json.dump(data, f, indent=2, allow_nan=False); f.write('\n')
def valid_file(path):
    need(path.is_file() and not path.is_symlink(), 'Expected regular file: ' + str(path))
def inventory():
    found = {}
    for name, folder in TREES.items():
        need(folder.is_dir() and not folder.is_symlink(), 'Missing closed CPU tree: ' + str(folder))
        for path in sorted(folder.rglob('*')):
            need(not path.is_symlink(), 'Symlink in evidence: ' + str(path))
            if path.is_file(): found['raw/' + name + '/' + path.relative_to(folder).as_posix()] = path
            else: need(path.is_dir(), 'Special evidence entry: ' + str(path))
    closure = read(TREES[NAMES[1]] / 'source-closure.json')['source_pins']
    sources = {Path(p) for p in closure}
    sources.update(HERE / n for n in FILES)
    sources.add(ROOT / 'bench/m5_ultra_256gb/gdn_prework_formal/audit_formal.py')
    for path in sorted(sources):
        valid_file(path)
        found['source/' + path.relative_to(ROOT).as_posix()] = path
    return found

def entries(files):
    return {name: {'original_path': str(path), 'bytes': path.stat().st_size, 'sha256': sha(path)}
            for name, path in files.items()}

def validate_attempts():
    result = {n: read(p / 'result.json') for n, p in TREES.items()}
    first, seal, failed, passed = (result[n] for n in NAMES)
    for item, count in ((first, 12), (passed, 58)):
        need(item['status'] == 'PASS' and item['tests_run'] == count and
             item['failures'] == item['errors'] == item['skipped'] == 0 and
             item['source_unchanged'] is True and item['denied'] == [] and item['error'] is None,
             'Unexpected successful CPU attempt')
    need(failed['status'] == 'FAIL' and failed['tests_run'] == 58 and failed['failures'] == 5 and
         failed['errors'] == failed['skipped'] == 0 and failed['denied'] == ['signal'] * 5 and
         failed['source_unchanged'] is True and failed['error'] is None, 'Original failure changed')
    # Source-copy verification uses the actual retained registries, including the original runner.
    for name in (NAMES[0], NAMES[2], NAMES[3]):
        folder = TREES[name]
        registry = folder / ('source-pins.json' if name == NAMES[0] else 'source-before.json')
        pins = read(registry)
        if name != NAMES[0]: need(pins == read(folder / 'source-after.json'), 'Changed tested sources')
        for index, (original, data) in enumerate(pins.items()):
            if name == NAMES[0]:
                copy = folder / data['copy']; digest = data['sha256']
                need(copy.stat().st_size == data['bytes'], 'Source copy size')
            else:
                copy = folder / 'sources' / (f'{index:02}-' + Path(original).name); digest = data
            valid_file(copy); need(sha(copy) == digest, 'Tested source snapshot mismatch')
    a = read(TREES[NAMES[2]] / 'source-before.json')
    b = read(TREES[NAMES[3]] / 'source-before.json')
    need({p: s for p, s in a.items() if p.startswith(str(ROOT))} ==
         {p: s for p, s in b.items() if p.startswith(str(ROOT))}, 'Production/test changes between58 attempts')
    for p, digest in b.items():
        if p.startswith(str(ROOT)): need(sha(Path(p)) == digest, 'Current frozen source changed')
    folder = TREES[NAMES[1]]
    before, after = (folder / 'formal-before.py').read_bytes(), (folder / 'formal.py').read_bytes()
    need(seal['status'] == 'PASS' and hashlib.sha256(before).hexdigest() == seal['before'] and
         hashlib.sha256(after).hexdigest() == seal['after'] and
         seal['prior_cpu_result_sha256'] == sha(TREES[NAMES[0]] / 'result.json'), 'Seal binding')
    needle = b"CYCLE_AUDITOR:'UNSEALED'"
    replacement = b"CYCLE_AUDITOR:'45c98bf8d8eacbb07c060b05310c686320fd9cce3cd9bbaece0cd6fc887c492f'"
    need(before.count(needle) == 1 and before.replace(needle, replacement) == after,
         'Seal is not the one registered literal change')
    closure = read(folder / 'source-closure.json')
    need(closure['status'] == 'PASS' and closure['source_sha256'] == seal['after'] and
         len(closure['source_pins']) == 42 and closure['denied'] == [], 'Source closure receipt')
    for p, digest in closure['source_pins'].items(): need(sha(Path(p)) == digest, 'Closure source changed')
    return result

def verify(folder, originals=False):
    manifest = read(folder / 'manifest.json'); expected = manifest['members']
    archive = folder / manifest['archive']['filename']
    need(sha(archive) == manifest['archive']['sha256'] and archive.stat().st_size == manifest['archive']['bytes'], 'Archive hash/size')
    found = set()
    with tarfile.open(archive, 'r:xz') as tar:
        for member in tar:
            p = PurePosixPath(member.name)
            need(member.name in expected and member.name not in found and member.isfile() and
                 not p.is_absolute() and '..' not in p.parts and p.parts[0] in {'source', 'raw'},
                 'Unsafe/unexpected archive member: ' + member.name)
            digest = hashlib.sha256(); count = 0
            with tar.extractfile(member) as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b''): digest.update(chunk); count += len(chunk)
            need(count == expected[member.name]['bytes'] and digest.hexdigest() == expected[member.name]['sha256'], 'Member hash/size')
            found.add(member.name)
    need(found == set(expected), 'Archive membership differs')
    if originals: need(entries(inventory()) == expected, 'Original tree membership or bytes changed')
    return {'status': 'PASS', 'members': len(found), 'regular_bytes': sum(v['bytes'] for v in expected.values()),
            'archive_sha256': sha(archive), 'archive_bytes': archive.stat().st_size, 'originals_rechecked': originals}

def run():
    results = validate_attempts(); files = inventory(); before = entries(files)
    OUT.mkdir(exist_ok=False)
    archive = OUT / 'raw.tar.xz'
    with tarfile.open(archive, 'w:xz', preset=1, dereference=True) as tar:
        for name, path in files.items(): tar.add(path, arcname=name, recursive=False)
    save(OUT / 'manifest.json', {'schema': 'combined-formal-prepared-cpu-archive-v1', 'created_at': time.time(),
         'archive': {'filename': archive.name, 'sha256': sha(archive), 'bytes': archive.stat().st_size}, 'members': before})
    verification = verify(OUT, True); save(OUT / 'archive-verification.json', verification)
    save(OUT / 'summary.json', {'schema': 'combined-formal-prepared-cpu-summary-v1',
         'status': 'PASS_WITH_RETAINED_PRIOR_RUNNER_FAILURE', 'distinct_successful_checks': 70,
         'factory_checks': 12, 'lifecycle_and_audit_checks': 58, 'attempts': results,
         'execute_source_tested_by_these_suites': False, 'archive': verification,
         'numerical_qualified': False, 'http_qualified': False, 'performance_qualified': False})
    with (OUT / 'archive.py').open('xb') as f: f.write(Path(__file__).read_bytes())
    with (OUT / 'README.md').open('x') as f: f.write(README)
    print(json.dumps(verification, sort_keys=True))

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', action='store_true'); p.add_argument('--verify', type=Path); p.add_argument('--originals', action='store_true')
    a = p.parse_args()
    if a.verify: print(json.dumps(verify(a.verify.resolve(), a.originals), sort_keys=True))
    elif a.run: run()
    else: print('DESCRIPTION_ONLY: closed CPU evidence archive; no tests, model or service actions.')
if __name__ == '__main__': main()
