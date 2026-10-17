"""Archive E025's actual early failure; re-read every tar member including links.

Run after both audits exist. Default output is this directory; --out NEW can
rebuild elsewhere. Refuses to overwrite an existing archive or manifest.
No device/runtime import or service action. Original evidence remains unchanged.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / '.artifacts/m5-ultra-256gb'
NAMES = ('prefill-nax-v2-owner', 'prefill-nax-v2-service-lease', 'prefill-nax-v2-inputs',
         'prefill-nax-v2-cpu-prepare', 'prefill-nax-v2-independent-audit',
         'prefill-nax-v2-root-repeated-audit')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''): h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    archive_path = args.out / 'raw-evidence.tar.xz'
    if archive_path.exists() or (args.out / 'manifest.json').exists():
        raise FileExistsError('archive/manifest already exists; use a new --out directory')
    independent = SOURCE / 'prefill-nax-v2-independent-audit/audit.json'
    repeated = SOURCE / 'prefill-nax-v2-root-repeated-audit/audit.json'
    assert independent.read_bytes() == repeated.read_bytes(), 'root/independent audit differ'
    audit = json.loads(independent.read_text())
    assert audit['status'] == 'PASS' and audit['experiment_status'] == 'REJECTED_NUMERICAL_PARTIAL'
    files = []
    for name in NAMES:
        root = SOURCE / name
        assert root.is_dir(), 'missing evidence root: ' + name
        files.extend((p, p.relative_to(SOURCE).as_posix()) for p in sorted(root.rglob('*')) if p.is_file())
    review = SOURCE / 'prefill-nax-v2-root-raw-review.json'
    files.append((review, review.name))
    for name in ('audit_failure_v2.py', 'test_audit_failure_v2_cpu.py'):
        files.append((ROOT / 'bench/m5_ultra_256gb/prefill_nax_same_split' / name, 'audit-source/' + name))
    files.append((Path(__file__).resolve(), 'archive-source/archive-evidence.py'))
    assert len({name for _, name in files}) == len(files)
    records, prior, unique = {}, {}, 0
    with tarfile.open(archive_path, 'w:xz', preset=3) as archive:
        for path, name in files:
            assert not path.is_symlink(), 'unexpected symlink: ' + str(path)
            size, sha = path.stat().st_size, digest(path)
            info = archive.gettarinfo(str(path), arcname=name)
            records[name] = dict(bytes=size, sha256=sha)
            key = size, sha
            if key in prior:
                info.type, info.linkname, info.size = tarfile.LNKTYPE, prior[key], 0
                archive.addfile(info)
            else:
                prior[key] = name; unique += size
                with path.open('rb') as stream: archive.addfile(info, stream)
            assert path.stat().st_size == size and digest(path) == sha, 'source changed while archiving'
    with tarfile.open(archive_path, 'r:xz') as archive:
        members = archive.getmembers()
        assert len(members) == len(records) and {m.name for m in members} == set(records)
        for member in members:
            assert member.isfile() or member.islnk(), 'unexpected member type'
            stream = archive.extractfile(member)
            h, size = hashlib.sha256(), 0
            while block := stream.read(4 << 20): h.update(block); size += len(block)
            assert size == records[member.name]['bytes'] and h.hexdigest() == records[member.name]['sha256'], member.name
    manifest = dict(archive=archive_path.name, sha256=digest(archive_path),
        archive_bytes=archive_path.stat().st_size, members=len(records),
        raw_bytes=sum(r['bytes'] for r in records.values()), unique_payload_bytes=unique,
        every_member_reread_and_sha256_verified=True, identical_payloads_stored_as_tar_hardlinks=True,
        preregistration_commit='4d97a8d', plan_sha256=audit['plan_sha256'], files=records)
    (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    for source, name in (
        ('prefill-nax-v2-owner/complete.json', 'owner-summary.json'),
        ('prefill-nax-v2-owner/child/screen/complete.json', 'child-summary.json'),
        ('prefill-nax-v2-root-raw-review.json', 'root-raw-review.json'),
        ('prefill-nax-v2-independent-audit/audit.json', 'independent-audit.json'),
        ('prefill-nax-v2-independent-audit/cpu-challenges.json', 'cpu-challenges.json'),
        ('prefill-nax-v2-root-repeated-audit/audit.json', 'root-repeated-audit.json')):
        shutil.copyfile(SOURCE / source, args.out / name)
    print(json.dumps({k:v for k,v in manifest.items() if k != 'files'}, indent=2))


if __name__ == '__main__': main()
