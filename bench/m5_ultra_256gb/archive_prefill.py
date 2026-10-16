"""Archive E010 including rejected timing, final checks, source and restoration."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    a = p.parse_args()
    root = Path(__file__).resolve().parents[2]
    a.out.mkdir(parents=True)
    folders = ['prefill-nax-screen-v1', 'prefill-nax-owner-v1', 'prefill-boundary-v1', 'prefill-boundary-owner-v1',
               'prefill-lifecycle-v1', 'prefill-lifecycle-owner-v1', 'prefill-http-v1',
               'prefill-integrated-boundaries-v1', 'prefill-integrated-boundaries-owner-v1',
               'prefill-integrated-lifecycle-v1', 'prefill-integrated-lifecycle-owner-v1',
               'prefill-http-v2', 'prefill-service-lease-v1', 'prefill-integrated-source-v1',
               'prefill-frontier-extension-v1', 'prefill-service-lease-v2', 'retire-batch-source-v1']
    assert all((a.base / folder).is_dir() for folder in folders)
    files = [(p, Path('raw') / p.relative_to(a.base)) for folder in folders for p in sorted((a.base / folder).rglob('*')) if p.is_file()]
    for name in ('prefill-integrated-boundaries-v1.adapter.json', 'prefill-integrated-lifecycle-v1.adapter.json',
                 'prefill-isolated-jobs-v1.json', 'prefill-isolated-v1.log', 'prefill-http-v1.log',
                 'prefill-extension-jobs-v1.json', 'prefill-extension-v1.log', 'fullqual-build2/workload.json'):
        files.append((a.base / name, Path('raw') / name))
    bench = root / 'bench/m5_ultra_256gb'
    for folder in ('prefill_nax', 'prefill_http', 'prefill_http_v2', 'prefill_integration'):
        files += [(p, Path('source') / p.relative_to(root)) for p in sorted((bench / folder).glob('*.py'))]
    for name in ('resources_v2.py', 'resources.py', 'resource_base.py', 'guarded_v2.py', 'suspended_service.py',
                 'audit_prefill.py', 'prefill_report.py', 'archive_prefill.py', 'prefill_frontier_extension.py',
                 'summarize_prefill_frontier.py', 'plot_prefill_frontier.py'):
        files.append((bench / name, Path('source') / (bench / name).relative_to(root)))
    files += [(p, Path('source') / p.relative_to(root)) for p in sorted((bench / 'decode_batch').glob('*.py'))]
    files += [(bench / 'lithos/client.py', Path('source/bench/m5_ultra_256gb/lithos/client.py'))]
    files += [(p, Path('raw/build2-harness') / p.name) for p in sorted((a.base / 'build2-harness').glob('*.py'))]
    manifest = {str(arc): {'sha256': sha(p), 'bytes': p.stat().st_size} for p, arc in files}
    assert len(manifest) == len(files)
    with tarfile.open(a.out / 'evidence.tar.gz', 'w:gz') as archive:
        for p, arc in files:
            archive.add(p, arcname=str(arc), recursive=False)
    (a.out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    for name in ('summary.json', 'audit.json'):
        shutil.copy(a.base / 'prefill-http-v2' / name, a.out / name)
    shutil.copy(a.base / 'prefill-frontier-extension-v1/frontier.json', a.out / 'frontier.json')
    (a.out / 'archive.json').write_text(json.dumps({'files': len(files), 'uncompressed_bytes': sum(v['bytes'] for v in manifest.values()),
          'archive_sha256': sha(a.out / 'evidence.tar.gz'), 'scope': 'Text evidence and exact source only; no models, environments or weight buffers'}, indent=2) + '\n')
    with tarfile.open(a.out / 'evidence.tar.gz') as archive:
        for member in archive.getmembers():
            data = archive.extractfile(member).read()
            assert len(data) == manifest[member.name]['bytes']
            assert hashlib.sha256(data).hexdigest() == manifest[member.name]['sha256']
    print('PASS archive', len(files), 'files', (a.out / 'evidence.tar.gz').stat().st_size, 'bytes compressed')


if __name__ == '__main__':
    main()
