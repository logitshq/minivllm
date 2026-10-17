from pathlib import Path
import hashlib, json, tarfile, shutil

ROOT = Path('/Users/ryanli/code/minivllm')
SOURCE = ROOT / '.artifacts/m5-ultra-256gb'
DEST = ROOT / 'docs/experiments/m5-ultra-256gb/E027-native-bridge-v1'
DEST.mkdir()
sha = lambda raw: hashlib.sha256(raw).hexdigest()
roots = [SOURCE / name for name in (
    'shared-w-native-bridge-v1-owner',
    'shared-w-native-bridge-v1-service-lease',
    'shared-w-native-bridge-v1-inputs')]
files = [p for root in roots for p in sorted(root.rglob('*')) if p.is_file()]
files += [SOURCE / name for name in (
    'shared-w-native-bridge-v1-root-difference.json',
    'shared-w-native-bridge-v1-independent-failure-audit-final.json',
    'shared-w-native-bridge-v1-root-repeated-audit.json')]
assert files[-1].read_bytes() == files[-2].read_bytes(), 'root/independent audit differ'
rows, prior, unique = {}, {}, 0
with tarfile.open(DEST / 'raw-evidence.tar.xz', 'w:xz', preset=3) as archive:
    for path in files:
        raw = path.read_bytes()
        name, digest = path.relative_to(SOURCE).as_posix(), sha(raw)
        info = archive.gettarinfo(str(path), arcname=name)
        rows[name] = {'bytes': len(raw), 'sha256': digest}
        key = len(raw), digest
        if key in prior:
            info.type, info.linkname, info.size = tarfile.LNKTYPE, prior[key], 0
            archive.addfile(info)
        else:
            prior[key] = name
            unique += len(raw)
            with path.open('rb') as stream:
                archive.addfile(info, stream)
with tarfile.open(DEST / 'raw-evidence.tar.xz', 'r:xz') as archive:
    assert len(archive.getmembers()) == len(rows)
    for member in archive:
        raw = archive.extractfile(member).read()
        assert len(raw) == rows[member.name]['bytes']
        assert sha(raw) == rows[member.name]['sha256']
manifest = dict(archive='raw-evidence.tar.xz',
    sha256=sha((DEST / 'raw-evidence.tar.xz').read_bytes()),
    archive_bytes=(DEST / 'raw-evidence.tar.xz').stat().st_size,
    members=len(rows), raw_bytes=sum(r['bytes'] for r in rows.values()),
    unique_payload_bytes=unique,
    every_member_reread_and_sha256_verified=True,
    identical_payloads_stored_as_tar_hardlinks=True,
    files=rows)
(DEST / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
for src, name in (
    ('shared-w-native-bridge-v1-owner/complete.json', 'owner-summary.json'),
    ('shared-w-native-bridge-v1-root-difference.json', 'root-difference.json'),
    ('shared-w-native-bridge-v1-independent-failure-audit-final.json', 'independent-audit.json'),
    ('shared-w-native-bridge-v1-root-repeated-audit.json', 'root-repeated-audit.json')):
    shutil.copyfile(SOURCE / src, DEST / name)
shutil.copyfile(__file__, DEST / 'archive-evidence.py')
print(json.dumps({k: v for k, v in manifest.items() if k != 'files'}, indent=2))
