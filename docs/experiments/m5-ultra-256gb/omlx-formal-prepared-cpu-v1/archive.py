"""Archive the two closed CPU attempts only; never import benchmark code."""
from pathlib import Path
import hashlib,io,json,os,tarfile
R=Path('/Users/ryanli/code/minivllm');A=R/'.artifacts/m5-ultra-256gb'
D=R/'docs/experiments/m5-ultra-256gb/omlx-formal-prepared-cpu-v1'
NAMES=('omlx-formal-prepared-cpu-v1','omlx-formal-outer-python-cpu-v1')
def sha(b):return hashlib.sha256(b).hexdigest()
def write(p,v):
 raw=(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode() if not isinstance(v,bytes) else v
 with p.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
results=[];files={};payloads={}
for name in NAMES:
 source=A/name;assert source.is_dir() and not source.is_symlink()
 result=json.loads((source/'result.json').read_bytes());results.append(result)
 assert result['source_pins_unchanged'] and result['source_pins_before']==result['source_pins_after']
 assert result['forbidden_modules_loaded']==[] and result['denied_operations']==[]
 assert result['actual_help_process_count']==0 and result['subprocess_audit_events']==[]
 for p,item in result['source_pins_before'].items():
  data=(source/'frozen-source'/p.lstrip('/')).read_bytes()
  assert len(data)==item['bytes'] and sha(data)==item['sha256']
 original=json.loads((source/'artifact-manifest.json').read_bytes())
 for rel,item in original['files'].items():
  data=(source/rel).read_bytes();assert len(data)==item['bytes'] and sha(data)==item['sha256']
 for p in sorted(source.rglob('*')):
  assert not p.is_symlink()
  if not p.is_file():continue
  rel=name+'/'+p.relative_to(source).as_posix();data=p.read_bytes()
  files[rel]=dict(source=str(p),sha256=sha(data),bytes=len(data));payloads[rel]=data
first,second=results
assert (first['status'],first['tests'],first['failures'],first['errors'])==('FAIL',79,0,1)
assert (second['status'],second['tests'],second['failures'],second['errors'])==('PASS',1,0,0)
assert first['prefix']==str(R/'.artifacts/omlx-baseline/.venv') and second['prefix']==str(R/'.venv')
assert 'RuntimeError: outer runtime package missing: httpx' in (A/NAMES[0]/'stderr.txt').read_text()
assert 'test_actual_prepared_observer_off_cli_boundary_and_interpreter_inventory' in (A/NAMES[1]/'stderr.txt').read_text()
# Only the two explicitly different runner files differ between source inventories.
changed=[p for p in sorted(set(first['source_pins_before'])|set(second['source_pins_before']))
         if first['source_pins_before'].get(p)!=second['source_pins_before'].get(p)]
assert changed==['/private/tmp/run-omlx-formal-cpu-v1.py','/private/tmp/run-omlx-formal-outer-python-cpu-v1.py']
D.mkdir(exist_ok=False)
write(D/'archive.py',Path(__file__).read_bytes())
write(D/'manifest.json',dict(schema='omlx-formal-cpu-archive-manifest-v1',files=files))
archive=D/'evidence.tar.xz'
with archive.open('xb') as destination:
 with tarfile.open(fileobj=destination,mode='w:xz',preset=3) as t:
  for name,data in payloads.items():
   info=tarfile.TarInfo(name);info.size=len(data);info.mode=0o644;info.mtime=0;t.addfile(info,io.BytesIO(data))
 destination.flush();os.fsync(destination.fileno())
seen=set()
with tarfile.open(archive,'r:xz') as t:
 for member in t:
  assert member.isfile() and member.name in files and member.name not in seen
  seen.add(member.name);data=t.extractfile(member).read();ref=files[member.name]
  assert len(data)==ref['bytes']==member.size and sha(data)==ref['sha256']
  assert data==Path(ref['source']).read_bytes()
assert seen==set(files)
for label,name in zip(('initial','outer-python'),NAMES):
 for filename in ('result.json','stderr.txt','stdout.txt'):
  write(D/(label+'-'+filename),(A/name/filename).read_bytes())
verification=dict(schema='omlx-formal-prepared-cpu-archive-v1',status='PASS',members=len(files),
 raw_bytes=sum(x['bytes'] for x in files.values()),archive=dict(path=archive.name,bytes=archive.stat().st_size,sha256=sha(archive.read_bytes())),
 manifest_sha256=sha((D/'manifest.json').read_bytes()),every_member_reread_sha_verified=True,every_original_file_byte_rechecked=True,
 source_snapshots_verified=True,production_and_test_sources_identical_across_attempts=True,
 initial=dict(status='FAIL',tests=79,passed=78,failures=0,errors=1,prefix=first['prefix']),
 followup=dict(status='PASS',tests=1,failures=0,errors=0,prefix=second['prefix']),total_test_executions=80,
 complete_combined_rerun=False,backend_subprocesses=0,forbidden_modules_loaded=[],
 numerical_qualified=False,http_qualified=False,performance_qualified=False)
write(D/'archive-verification.json',verification)
write(D/'README.md',b'''# oMLX formal CPU preparation\n\nThe original combined attempt ran 79 tests under the pinned oMLX vendor Python: 78 passed and one raised an error. The real prepared-CLI/inventory test correctly rejected that interpreter's `sys.prefix`, because the owner inventory belongs to the repository's outer `.venv`; the error was `outer runtime package missing: httpx`. This was an interpreter-selection mismatch, not a production-code repair.\n\nThe exact affected test then passed under the registered outer `.venv`. No production or test source changed. Both original runner files, commands, full stdout/stderr, result metadata and exact source snapshots are retained. This is 80 test executions across two attempts; the full 79-test suite was not rerun. The original FAIL remains FAIL.\n\nThe tests used CPU fixtures and guarded fake backends. Both receipts record zero real subprocesses, signals/network attempts and forbidden runtime modules. No model, GPU or real server ran. These results do not grant actual HTTP, numerical or performance qualification.\n\n`evidence.tar.xz` contains every file from both original attempt directories. `manifest.json` maps every member to its source path, byte count and SHA-256. All archived members were read back and compared both by hash and byte for byte with their original files. The six sidecars are exact copies of the two result/stdout/stderr records; `archive-verification.json` records coverage and archive hashes.\n''')
fd=os.open(D,os.O_RDONLY)
try:os.fsync(fd)
finally:os.close(fd)
print(json.dumps(verification,indent=2));print('VERIFICATION_SHA',sha((D/'archive-verification.json').read_bytes()))
