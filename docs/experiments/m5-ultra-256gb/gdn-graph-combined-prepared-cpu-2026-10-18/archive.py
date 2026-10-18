"""Archive only closed CPU evidence and explicit small source files."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile
import time

ROOT=Path('/Users/ryanli/code/minivllm')
OUT=ROOT/'docs/experiments/m5-ultra-256gb/gdn-graph-combined-prepared-cpu-2026-10-18'
TREES={name:ROOT/'.artifacts/m5-ultra-256gb'/name for name in (
    'gdn-graph-combined-cpu-v1','gdn-graph-combined-qualifier-cpu-v1',
    'gdn-graph-combined-prepare-v1','combined-driver-v2-cpu-v1')}
SOURCES=[ROOT/'bench/m5_ultra_256gb/gdn_graph_combined'/name for name in
         ('integration.py','test_integration_cpu.py','qualify.py','test_qualify_cpu.py','README.md','QUALIFICATION.md')]
SOURCES += [ROOT/'bench/m5_ultra_256gb/cold_observed_cycle'/name for name in
            ('_reuse.py','controller.py','owner.py','audit_restore.py','lifecycle.json','README.md',
             'execute_combined_v1.py','execute_combined_v2.py','test_execute_combined_v2_cpu.py')]
SOURCES += [ROOT/'docs/optimization-guide/research/2026-10-18-ready-and-shared-bandwidth.md']

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,data):
    with path.open('x') as f: json.dump(data,f,indent=2);f.write('\n')
def inventory():
    result={}
    for label,folder in TREES.items():
        for path in sorted(folder.rglob('*')):
            if path.is_symlink(): raise ValueError('Unexpected symlink '+str(path))
            if path.is_file(): result['raw/'+label+'/'+path.relative_to(folder).as_posix()]=path
    for path in SOURCES: result['source/'+path.relative_to(ROOT).as_posix()]=path
    return result
def entries(files):
    return {name:{'original_path':str(path),'bytes':path.stat().st_size,'sha256':sha(path)} for name,path in files.items()}
def verify(folder,originals=False):
    manifest=json.loads((folder/'manifest.json').read_bytes());expected=manifest['members']
    archive=folder/manifest['archive']['filename']
    if sha(archive)!=manifest['archive']['sha256']: raise ValueError('Archive SHA mismatch')
    found={}
    with tarfile.open(archive,'r:xz') as tar:
        for member in tar:
            parts=PurePosixPath(member.name).parts
            if member.name not in expected or member.name in found or not member.isfile() or parts[0] not in {'raw','source'} or '..' in parts or member.name.startswith('/'):
                raise ValueError('Unexpected/unsafe member '+member.name)
            digest=hashlib.sha256();count=0
            with tar.extractfile(member) as f:
                for chunk in iter(lambda:f.read(1024*1024),b''):digest.update(chunk);count+=len(chunk)
            row=expected[member.name]
            if count!=row['bytes'] or digest.hexdigest()!=row['sha256']: raise ValueError('Member mismatch '+member.name)
            found[member.name]=True
    if set(found)!=set(expected): raise ValueError('Missing members')
    if originals:
        actual=entries(inventory())
        if actual!=expected: raise ValueError('Original files/membership changed')
    return {'status':'PASS','members':len(found),'regular_bytes':sum(r['bytes'] for r in expected.values()),
            'archive_sha256':sha(archive),'archive_bytes':archive.stat().st_size,'originals_rechecked':originals}
def run():
    first=json.loads((TREES['gdn-graph-combined-cpu-v1']/'result.json').read_bytes())
    second=json.loads((TREES['gdn-graph-combined-qualifier-cpu-v1']/'result.json').read_bytes())
    bridge=json.loads((TREES['gdn-graph-combined-prepare-v1']/'cpu-result.json').read_bytes())
    driver=json.loads((TREES['combined-driver-v2-cpu-v1']/'result.json').read_bytes())
    for result,count in ((first,10),(second,10),(driver,3)):
        assert result['status']=='PASS' and result['tests_run']==count and result['failures']==result['errors']==result['skipped']==0
        assert result['source_unchanged'] is True and result['denied']==[]
    assert bridge['success'] is True and bridge['tests']==6 and bridge['failures']==bridge['errors']==bridge['skips']==0
    files=inventory();before=entries(files)
    OUT.mkdir(exist_ok=False)
    archive=OUT/'raw.tar.xz'
    with tarfile.open(archive,'w:xz',preset=3) as tar:
        for name,path in files.items():tar.add(path,arcname=name,recursive=False)
    manifest={'schema':'combined-prepared-cpu-archive-v1','scope':'Closed CPU evidence and explicit small sources only',
              'created_at':time.time(),'archive':{'filename':archive.name,'bytes':archive.stat().st_size,'sha256':sha(archive)},'members':before}
    save(OUT/'manifest.json',manifest)
    verification=verify(OUT,True);save(OUT/'archive-verification.json',verification)
    summary={'schema':'combined-prepared-cpu-summary-v1','status':'PASS','tests':{'integration':10,'qualifier':10,'observed_baseline_bridge':6,'driver_v2':3,'total':29},
             'driver_first_attempt':driver,'archive':verification,'numerical_qualified':False,'http_qualified':False,'performance_qualified':False,
             'scope':'CPU mechanisms only. Live combined device run and developing numerical auditor/GLM monitor excluded.'}
    save(OUT/'summary.json',summary)
    (OUT/'archive.py').write_bytes(Path(__file__).read_bytes())
    (OUT/'README.md').write_text('''# Combined GDN + route graph: CPU preparation evidence

The retained first attempts passed **20 combination tests + 6 observed-baseline bridge checks + 3 driver control-flow tests = 29 CPU checks**. These results grant no model numerical, HTTP, or performance qualification. The concurrent device experiment is outside this archive.

The two combination suites retain their guarded runner, original stdout/stderr, exact source copies and before/after source checks. The bridge retains its actual six-check preparation script, stdout, source identities and generated plans. The new driver suite retains its original three-test attempt; real process, socket, signal and backend operations were blocked and no denied operation occurred.

`execute_combined_v1.py` and its live execution remain unchanged. The separate v2 distinguishes a synchronous owner invocation that returned from an unknown launch outcome before selecting `--no-experiment`; it also requires the final journal to be `RESTORED`. Fake tests cover those branches. V2 still requires a real matching PREPARED journal and cannot resume the already-started v1 cycle.

`raw.tar.xz` contains the complete four closed CPU/preparation trees and explicitly listed small source snapshots. It excludes live owner/journal/execution data, developing combined `audit.py`, GLM monitor, weights and runtime binaries. The existing preparation plan is retained evidence, not permission to rerun it. Every regular tar member was compared by size and SHA-256 with its original, then the original file inventory was rechecked.

Verify the archived bytes without extracting or reading live originals:

```sh
python3 archive.py --verify .
```

The verifier requires exact manifest membership and only regular files under `raw/` or `source/`; it never calls `extractall`. `--originals` additionally rechecks the original closed inputs and current source files, so a later deliberate source change can make that optional check fail without invalidating the retained archive. `summary.json`, `manifest.json` and `archive-verification.json` are small sidecars. The included research note remains a proposal, with no ReaDy implementation or speed claim on this hardware.
''')
    print(json.dumps(verification,sort_keys=True))
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',action='store_true');p.add_argument('--verify',type=Path);p.add_argument('--originals',action='store_true');a=p.parse_args()
    if a.verify:print(json.dumps(verify(a.verify.resolve(),a.originals),sort_keys=True))
    elif a.run:run()
    else:print('DESCRIPTION_ONLY: new closed CPU archive; no service/model actions.')
if __name__=='__main__':main()
