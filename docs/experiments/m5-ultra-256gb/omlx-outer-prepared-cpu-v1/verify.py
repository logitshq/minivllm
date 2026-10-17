"""Verify this source/evidence archive, without tests or backend imports."""
from pathlib import Path
import hashlib,json,tarfile
D=Path(__file__).resolve().parent
manifest=json.loads((D/'manifest.json').read_bytes())['files']
receipt=json.loads((D/'archive-verification.json').read_bytes())
archive=D/receipt['archive']['path']
sha=lambda b:hashlib.sha256(b).hexdigest()
assert archive.stat().st_size==receipt['archive']['bytes']
assert sha(archive.read_bytes())==receipt['archive']['sha256']
seen=set()
with tarfile.open(archive,'r:gz') as stream:
    for member in stream:
        assert member.isfile() and member.name not in seen and member.name in manifest
        raw=stream.extractfile(member).read();expected=manifest[member.name]
        assert len(raw)==member.size==expected['bytes'] and sha(raw)==expected['sha256']
        seen.add(member.name)
assert seen==set(manifest)
print(json.dumps({'status':'PASS','members':len(seen),'archive_sha256':receipt['archive']['sha256']}))
