"""Independent stdlib member verification, manual safe extraction, no backends."""
import hashlib,json,tarfile,tempfile
from pathlib import Path,PurePosixPath

def verify(folder):
 folder=Path(folder);manifest=json.loads((folder/'manifest.json').read_bytes());expected=manifest['members'];seen=set()
 archive=folder/'evidence.tar.xz';sha=hashlib.sha256(archive.read_bytes()).hexdigest()
 with tempfile.TemporaryDirectory() as temporary,tarfile.open(archive,'r:xz') as tar:
  for member in tar:
   path=PurePosixPath(member.name)
   if not member.isfile() or path.is_absolute() or '..' in path.parts or str(path)!=member.name or member.name in seen or member.name not in expected:raise ValueError('unsafe/unexpected archive member')
   row=expected[member.name]
   if member.size!=row['bytes']:raise ValueError('member size')
   raw=tar.extractfile(member).read(member.size+1)
   if len(raw)!=member.size or hashlib.sha256(raw).hexdigest()!=row['sha256']:raise ValueError('member SHA')
   target=Path(temporary)/member.name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
   if target.stat().st_size!=row['bytes'] or hashlib.sha256(target.read_bytes()).hexdigest()!=row['sha256']:raise ValueError('safe extraction mismatch')
   seen.add(member.name)
 if seen!=set(expected):raise ValueError('missing archive member')
 result=dict(members=len(seen),raw_bytes=sum(r['bytes'] for r in expected.values()),archive_bytes=archive.stat().st_size,archive_sha256=sha)
 summary=json.loads((folder/'summary.json').read_bytes())
 for key in ('raw_bytes','archive_bytes','archive_sha256'):
  if summary[key]!=result[key]:raise ValueError('summary mismatch')
 if summary['files']!=len(seen):raise ValueError('summary count')
 return result
if __name__=='__main__':print(json.dumps(verify(Path(__file__).resolve().parent)))
