"""Archive closed CPU evidence only; never import benchmark/runtime modules."""
from pathlib import Path
import hashlib, io, json, os, tarfile

R=Path('/Users/ryanli/code/minivllm'); A=R/'.artifacts/m5-ultra-256gb'
D=R/'docs/experiments/m5-ultra-256gb/omlx-formal-retry-prepared-cpu-v1'
NAMES=('omlx-formal-retry-prepared-cpu-v1','omlx-formal-retry-outer-python-cpu-v1')
def sha(data): return hashlib.sha256(data).hexdigest()
def write(path,value):
    data=value if isinstance(value,bytes) else (json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode()
    with path.open('xb') as stream: stream.write(data); stream.flush(); os.fsync(stream.fileno())

results=[]; files={}; payloads={}
def include(path,name):
    assert path.is_file() and not path.is_symlink() and name not in files
    data=path.read_bytes(); files[name]=dict(source=str(path),sha256=sha(data),bytes=len(data)); payloads[name]=data

for name in NAMES:
    source=A/name; assert source.is_dir() and not source.is_symlink()
    result=json.loads((source/'result.json').read_bytes()); results.append(result)
    assert result['status']=='PASS' and all(result[k]==0 for k in ('failures','errors','skipped','actual_help_process_count'))
    assert result['source_pins_unchanged'] and result['source_pins_before']==result['source_pins_after']
    assert result['forbidden_modules_loaded']==result['denied_operations']==result['subprocess_audit_events']==[]
    for path,item in result['source_pins_before'].items():
        data=(source/'frozen-source'/path.lstrip('/')).read_bytes()
        assert len(data)==item['bytes'] and sha(data)==item['sha256']
    original=json.loads((source/'artifact-manifest.json').read_bytes())
    for rel,item in original['files'].items():
        data=(source/rel).read_bytes(); assert len(data)==item['bytes'] and sha(data)==item['sha256']
    for path in sorted(source.rglob('*')):
        assert not path.is_symlink()
        if path.is_file(): include(path,name+'/'+path.relative_to(source).as_posix())

first,second=results
assert first['tests']==82 and second['tests']==1
assert first['prefix']==str(R/'.artifacts/omlx-baseline/.venv') and second['prefix']==str(R/'.venv')
ids=[{row['id'] for row in value['test_outcomes']} for value in results]
assert len(ids[0])==82 and len(ids[1])==1 and not ids[0]&ids[1]
assert all(row['status']=='PASS' for value in results for row in value['test_outcomes'])
changed=[p for p in sorted(set(first['source_pins_before'])|set(second['source_pins_before']))
         if first['source_pins_before'].get(p)!=second['source_pins_before'].get(p)]
assert changed==['/private/tmp/run-omlx-formal-retry-cpu-v1.py','/private/tmp/run-omlx-formal-retry-outer-python-cpu-v1.py']
regression_root=A/'omlx-d512-formal-v1-owner/phases/r1-c1/trial'
regression_hashes={'warm-group.json':'0d1f42f0b276db1d05c42824591a060ee9cf336f76b71751e2a470e4c9ee1d4c',
                   'spec.json':'dde6fbc9bd9ee3cf5fee138891bc2e200d9daad9eab3a0aa1c8062b2feb59b7b'}
for name,digest in regression_hashes.items():
    assert sha((regression_root/name).read_bytes())==digest
    include(regression_root/name,'actual-v1-warm-regression/'+name)
group=json.loads((regression_root/'warm-group.json').read_bytes())
assert len(group['requests'])==8 and group['summary']['formal_point_eligible'] is False
assert any(x.endswith('test_actual_failed_v1_warm_original_bytes_now_pass_count_gate_only') for x in ids[0])
assert any(x.endswith('test_all_eight_actual_retained_usage_schemas_reparse_without_eos_subtraction') for x in ids[0])
D.mkdir(exist_ok=False)
write(D/'archive.py',Path(__file__).read_bytes())
write(D/'manifest.json',dict(schema='omlx-formal-retry-cpu-archive-manifest-v1',files=files))
archive=D/'evidence.tar.xz'
with archive.open('xb') as destination:
    with tarfile.open(fileobj=destination,mode='w:xz',preset=3) as stream:
        for name,data in payloads.items():
            info=tarfile.TarInfo(name); info.size=len(data); info.mode=0o644; info.mtime=0
            stream.addfile(info,io.BytesIO(data))
    destination.flush(); os.fsync(destination.fileno())
seen=set()
with tarfile.open(archive,'r:xz') as stream:
    for member in stream:
        assert member.isfile() and member.name in files and member.name not in seen
        seen.add(member.name); data=stream.extractfile(member).read(); ref=files[member.name]
        assert len(data)==ref['bytes']==member.size and sha(data)==ref['sha256']
        assert data==Path(ref['source']).read_bytes()
assert seen==set(files)
for label,name in zip(('vendor','outer-python'),NAMES):
    for filename in ('result.json','stderr.txt','stdout.txt'):
        write(D/(label+'-'+filename),(A/name/filename).read_bytes())
verification=dict(schema='omlx-formal-retry-prepared-cpu-archive-v1',status='PASS',members=len(files),
    raw_bytes=sum(x['bytes'] for x in files.values()),archive=dict(path=archive.name,bytes=archive.stat().st_size,sha256=sha(archive.read_bytes())),
    manifest_sha256=sha((D/'manifest.json').read_bytes()),every_member_reread_sha_verified=True,
    every_original_file_byte_rechecked=True,source_snapshots_verified=True,
    production_and_test_sources_identical_across_attempts=True,
    vendor=dict(status='PASS',tests=82,failures=0,errors=0,skipped=0,prefix=first['prefix']),
    outer=dict(status='PASS',tests=1,failures=0,errors=0,skipped=0,prefix=second['prefix']),
    total_test_executions=83,unique_tests=83,backend_subprocesses=0,forbidden_modules_loaded=[],
    actual_warm_regression=dict(requests=8,files=regression_hashes,scope='Retained original v1 wire usage reparse only; no new HTTP or inference.'),
    numerical_qualified=False,http_qualified=False,performance_qualified=False)
write(D/'archive-verification.json',verification)
write(D/'README.md',b'''# oMLX formal retry CPU preparation

The first retry CPU attempts passed all 83 distinct tests: 82 under the pinned oMLX vendor Python, plus the actual outer-runtime inventory test under the repository `.venv`. There were no failures, errors or skips. Both guarded runners record zero real subprocesses, forbidden runtime imports or denied backend operations. Production and test source snapshots are identical between these attempts; only the two explicit runner files differ.

The regression tests reread the original failed formal-v1 C1 warm group's complete retained SSE bytes: eight requests and their unchanged spec. They verify the optional `input_tokens`/`output_tokens` aliases as strict integer counts, reject mismatches and invalid types, and retain the original committed-token semantics. This is offline wire-schema regression, not new HTTP inference, mathematical qualification or a performance point. The original failed experiment and its earlier CPU failure remain preserved in their separate archives; neither was rewritten as PASS.

`evidence.tar.xz` contains every file from both CPU attempt directories, including commands, first-attempt results, full stdout/stderr, exact source snapshots and manifests. It also contains the two unchanged real warm-regression input files. `manifest.json` maps every member to its original absolute source, size and SHA-256. Every tar member was read back and checked by SHA and byte-for-byte against its original. The six result/stdout/stderr sidecars are exact copies. `archive-verification.json` records the counts and archive hashes. No tests were rerun while creating this archive.
''')
fd=os.open(D,os.O_RDONLY)
try: os.fsync(fd)
finally: os.close(fd)
print(json.dumps(verification,indent=2)); print('VERIFICATION_SHA',sha((D/'archive-verification.json').read_bytes()))
