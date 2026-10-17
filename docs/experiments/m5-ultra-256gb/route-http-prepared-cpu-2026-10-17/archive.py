from pathlib import Path
import hashlib,json,shutil,difflib
R=Path('/Users/ryanli/code/minivllm')
A=R/'.artifacts/m5-ultra-256gb'
O=R/'docs/experiments/m5-ultra-256gb/route-http-prepared-cpu-2026-10-17'
assert not O.exists()
O.mkdir()
manifest={}
for version in ('v1','v2'):
    source=A/('route-http-prepared-cpu-'+version)
    target=O/version
    assert source.is_dir() and not source.is_symlink()
    shutil.copytree(source,target,symlinks=False)
    for p in sorted(source.rglob('*')):
        assert not p.is_symlink()
        if not p.is_file():continue
        relative=p.relative_to(source)
        data=p.read_bytes();copied=(target/relative).read_bytes()
        assert data==copied
        manifest[version+'/'+relative.as_posix()]={'source':str(p),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
p='source-before/bench/m5_ultra_256gb/route_decode_graph/test_http_capture_cpu.py'
old=(O/'v1'/p).read_bytes();new=(O/'v2'/p).read_bytes()
assert hashlib.sha256(old).hexdigest()=='052afe7b1c02e2b5ceaf7f58428f114f70e5fdf02329a9bcca6d0ac53027ea21'
assert hashlib.sha256(new).hexdigest()=='b3a75e8c632cbec784105a2525582c4c97db615c61ccbe44f7c4ae1364311dca'
assert new==old.replace(b"Path(self.temp.name) / 'capture'",b"Path(self.temp.name).resolve() / 'capture'",1)
v1=json.loads((O/'v1/result.json').read_bytes());v2=json.loads((O/'v2/result.json').read_bytes())
assert (v1['status'],v1['tests'],v1['failures'],v1['errors'])==('FAIL',23,1,0)
assert (v2['status'],v2['tests'],v2['failures'],v2['errors'])==('PASS',11,0,0)
for result in (v1,v2):assert result['source_pins_unchanged'] and result['forbidden_modules_loaded']==[]
changed=[p for p,h in v1['source_pins'].items() if v2['source_pins'][p]!=h]
assert changed==[str(R/'bench/m5_ultra_256gb/route_decode_graph/test_http_capture_cpu.py')]
with (O/'fixture-fix.diff').open('x') as f:
    f.writelines(difflib.unified_diff(old.decode().splitlines(True),new.decode().splitlines(True),fromfile='v1/test_http_capture_cpu.py',tofile='v2/test_http_capture_cpu.py'))
with (O/'manifest.json').open('x') as f:json.dump({'schema':'route-http-cpu-evidence-manifest-v1','files':manifest},f,indent=2);f.write('\n')
summary={'schema':'route-http-cpu-archive-v1','status':'PASS','v1':{'tests':23,'failures':1,'errors':0,'scope':'server12 PASS; capture10 PASS/1 fixture-path assertion FAIL'},'v2':{'tests':11,'failures':0,'errors':0,'scope':'affected capture suite only'},'source_files_changed':changed,'copied_files':len(manifest),'copied_bytes':sum(x['bytes'] for x in manifest.values()),'every_copy_byte_verified':True,'all_source_pins_stable_during_tests':True,'forbidden_runtime_modules_loaded':[],'numerical_qualified':False,'http_qualified':False,'performance_qualified':False}
with (O/'archive-verification.json').open('x') as f:json.dump(summary,f,indent=2);f.write('\n')
(O/'README.md').write_text('''# Prepared E024 route HTTP CPU evidence — 2026-10-17

The first guarded run executed 23 tests: all 12 server tests passed; capture had 10 passes and one failed fixture assertion. The capture completed successfully, but its actual argv used the correctly resolved macOS `/private/var/...` path while the test expected `/var/...`.

The only repair was `Path(self.temp.name).resolve()` in the capture test fixture. Production sources and gates were unchanged. The focused second run executed all 11 capture tests successfully. The 12 server tests were not repeated. This is 34 total test executions across two attempts, not a fresh combined 23-test rerun.

Both attempts preserve their original command, runner, full stdout/stderr, source/metadata snapshots and before/after hashes. `fixture-fix.diff` is the exact one-line repair. `manifest.json` maps every copied file to its original source, size and SHA; copies were compared byte for byte.

The runner rejected model/MLX imports and real subprocess, socket and process-signal backends. Capture used fake direct children; the server tests used the actual donor worker and witness bookkeeping with fake callables. Neither attempt loaded a model, launched a server, used the GPU or granted numerical, real HTTP or performance qualification. Actual capture/replay and independent audits remain required.
''')
(O/'archive.py').write_bytes(Path(__file__).read_bytes())
print(json.dumps(summary))
