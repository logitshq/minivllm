"""Archive E007–E009 decisions and a content-addressed inventory of local evidence."""
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / '.artifacts/m5-ultra-256gb'
OUT = ROOT / 'docs/experiments/m5-ultra-256gb/systematic-v1'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda: f.read(1 << 20), b''):
            h.update(data)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    inventory = {}
    for experiment, stem in [('E007', 'gdn-standalone'), ('E009', 'gdn-state-reuse')]:
        source = BASE / (stem + '-screen-v1')
        owner = BASE / (stem + '-owner-v1')
        passed = (source / 'complete.json').exists()
        if passed:
            assert read(source / 'complete.json')['status'] == 'PASS'
            assert read(owner / 'complete.json')['status'] == 'PASS'
        else:
            assert (owner / 'failure.json').exists(), 'Incomplete experiment without a failure receipt'
        assert read(owner / 'release.json')['strict_absence']
        plan = read(source / 'plan.json')
        for name, sha in plan['source_sha256'].items():
            assert digest(ROOT / name) == sha, name
        target = OUT / experiment
        target.mkdir(exist_ok=True)
        for path in source.glob('*.json'):
            shutil.copy2(path, target / path.name)
        for name in ('release.json', 'resources.json'):
            shutil.copy2(owner / name, target / ('owner-' + name))
        if not passed:
            shutil.copy2(owner / 'failure.json', target / 'owner-failure.json')
            shutil.copy2(owner / 'child.log', target / 'child.log')
        for directory in (source, owner):
            for path in sorted(directory.rglob('*')):
                if path.is_file():
                    inventory[str(path.relative_to(ROOT))] = {'sha256': digest(path), 'bytes': path.stat().st_size}
    target = OUT / 'E008'
    target.mkdir(exist_ok=True)
    directory = BASE / 'allocator-http-v1'
    assert read(directory / 'complete.json')['status'] == 'PASS'
    assert read(directory / 'summary.json')['status'] == 'PASS'
    plan = read(directory / 'plan.json')
    for path, sha in plan['pins'].items():
        assert digest(Path(path)) == sha, path
    for path, sha in plan['source_files'].items():
        assert digest(Path(plan['source']) / 'tutorial_code/miniengine' / path) == sha, path
    for name in ('plan.json', 'summary.json', 'source.json'):
        shutil.copy2(directory / name, target / name)
    for path in sorted(directory.rglob('*')):
        if path.is_file():
            inventory[str(path.relative_to(ROOT))] = {'sha256': digest(path), 'bytes': path.stat().st_size}
    archived = {str(p.relative_to(OUT)): {'sha256': digest(p), 'bytes': p.stat().st_size}
                for p in OUT.rglob('*.json') if p.name != 'manifest.json'}
    manifest = {'raw_files': inventory, 'archived_files': archived,
                'raw_bytes': sum(v['bytes'] for v in inventory.values()),
                'scope': 'Raw SSE events remain local under .artifacts; paths are evidence locations, not portable dependencies.',
                'E007_timing_erratum': 'cold_ms means first timed call after correctness evaluation, not cold compilation.'}
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'raw_files': len(inventory), 'raw_bytes': manifest['raw_bytes'],
                      'archive': str(OUT)}, indent=2))


if __name__ == '__main__':
    main()
