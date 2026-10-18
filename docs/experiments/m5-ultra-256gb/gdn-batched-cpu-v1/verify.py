"""Independently verify every regular archive member and sidecar (stdlib only)."""
import hashlib,json
from pathlib import Path,PurePosixPath
import tarfile
import tempfile
import shutil
root=Path(__file__).resolve().parent
manifest=json.loads((root/'manifest.json').read_bytes())
receipt=json.loads((root/'archive-verification.json').read_bytes())
def info(path):
    raw=path.read_bytes()
    return dict(sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))
assert info(root/'manifest.json')==receipt['manifest']
assert info(root/'evidence.tar.xz')==receipt['archive']
files=manifest['files'];seen=set()
with tempfile.TemporaryDirectory(prefix='gdn-http-cpu-check-') as folder:
    base=Path(folder)
    with tarfile.open(root/'evidence.tar.xz','r:xz') as archive:
        for member in archive:
            p=PurePosixPath(member.name)
            assert member.isfile() and not p.is_absolute() and '..' not in p.parts
            assert member.name not in seen and member.name in files
            row=files[member.name];assert member.size==row['bytes']
            seen.add(member.name);dest=base/member.name;dest.parent.mkdir(parents=True,exist_ok=True)
            stream=archive.extractfile(member);assert stream is not None
            with stream,dest.open('xb') as target:shutil.copyfileobj(stream,target)
            assert info(dest)=={k:row[k] for k in ('sha256','bytes')}
assert seen==set(files) and len(seen)==receipt['members']
for name,expected in receipt['sidecars'].items():assert info(root/name)==expected
print(json.dumps(dict(status='PASS',members=len(seen),archive=receipt['archive'])))
