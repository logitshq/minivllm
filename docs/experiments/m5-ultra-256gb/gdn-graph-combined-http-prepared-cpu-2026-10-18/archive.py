"""Archive only closed CPU evidence and explicit small source files."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile
import time

ROOT=Path('/Users/ryanli/code/minivllm')
OUT=ROOT/'docs/experiments/m5-ultra-256gb/gdn-graph-combined-http-prepared-cpu-2026-10-18'
TREES={name:ROOT/'.artifacts/m5-ultra-256gb'/name for name in (
    'gdn-graph-combined-http-cpu-v1','gdn-graph-combined-http-cpu-v2',
    'gdn-graph-combined-http-audit-cpu-v1','combined-http-cycle-cpu-v1',
    'combined-http-cycle-audit-cpu-v1')}
SOURCES=[ROOT/'bench/m5_ultra_256gb/gdn_graph_combined'/name for name in
         ('http_server.py','http_capture.py','http_qualification.py','http_audit.py','test_http_cpu.py','test_http_audit_cpu.py')]
SOURCES += [ROOT/'bench/m5_ultra_256gb/gdn_graph_combined_http_cycle'/name for name in
            ('owner.py','controller.py','audit_restore.py','audit_cycle.py','baseline.json','execute.py',
             'test_owner_cpu.py','test_contract_cpu.py','test_audit_cycle_cpu.py')]

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
    results={name:json.loads((folder/'result.json').read_bytes()) for name,folder in TREES.items()}
    failed=results['gdn-graph-combined-http-cpu-v1']
    assert failed['status']=='FAIL' and failed['tests_run']==14 and failed['failures']==4 and failed['errors']==failed['skipped']==0
    assert failed['source_unchanged'] is True and failed['denied']==['socket.__new__']*4
    counts={'gdn-graph-combined-http-cpu-v2':14,'gdn-graph-combined-http-audit-cpu-v1':9,
            'combined-http-cycle-cpu-v1':14,'combined-http-cycle-audit-cpu-v1':13}
    for name,count in counts.items():
        row=results[name]
        assert row['status']=='PASS' and row['tests_run']==count and row['failures']==row['errors']==row['skipped']==0
        assert row['source_unchanged'] is True and row['denied']==[]
    files=inventory();before=entries(files)
    OUT.mkdir(exist_ok=False)
    archive=OUT/'raw.tar.xz'
    with tarfile.open(archive,'w:xz',preset=1) as tar:
        for name,path in files.items():tar.add(path,arcname=name,recursive=False)
    manifest={'schema':'combined-http-prepared-cpu-archive-v1','scope':'Closed CPU evidence and explicit small sources only',
              'created_at':time.time(),'archive':{'filename':archive.name,'bytes':archive.stat().st_size,'sha256':sha(archive)},'members':before}
    save(OUT/'manifest.json',manifest)
    verification=verify(OUT,True);save(OUT/'archive-verification.json',verification)
    summary={'schema':'combined-http-prepared-cpu-summary-v1','status':'PASS_WITH_RETAINED_PRIOR_RUNNER_FAILURE',
             'tests':counts,'successful_current_tests':sum(counts.values()),'attempts':results,
             'archive':verification,'numerical_qualified':False,'http_qualified':False,'performance_qualified':False,
             'scope':'CPU mechanisms only. Concurrent actual HTTP capture/replay and future formal cycle excluded.'}
    save(OUT/'summary.json',summary)
    (OUT/'archive.py').write_bytes(Path(__file__).read_bytes())
    (OUT/'README.md').write_text(README)
    print(json.dumps(verification,sort_keys=True))
README="""# Combined GDN + route: HTTP CPU preparation

The current suites passed **14 HTTP entry checks + 9 independent HTTP audit checks + 14 lifecycle checks + 13 cycle audit checks = 50 CPU checks**. These are fake control-flow and source-binding checks, with no model numerical, HTTP-history or performance qualification.

The first HTTP-entry attempt is preserved in full: **10 passed and 4 failed** because its guard rejected asyncio's local socketpair creation (`socket.__new__`). The separate v2 runner permitted only the internal AF_UNIX socketpairs while still forbidding actual network connections, process launches, signals and backend imports. Its full14 checks passed; production sources did not change between those attempts. This archive retains both runners, stdout/stderr, exact source snapshots and original result bytes. The other three suites passed their first attempts.

The frozen producer uses real C1×8 and C8×16 HTTP histories, explicit route sidecars, then separately owned numerical replay processes. The cycle auditor consumes two explicitly SHA-bound independent numerical audits, independently rechecks their terminal/input/source/PID bindings, and directly reuses the frozen resource and preparation algorithms. Its PASS requires the real four-phase lifecycle and independent restoration. No one-call-per-step assumption or offline tensor-arithmetic claim is introduced.

`raw.tar.xz` contains the complete five closed CPU trees and the15 explicit new HTTP/lifecycle source, baseline and driver files listed by `manifest.json`. It excludes the active owner/journal/execution, model weights, runtime binaries and the still-developing formal implementation. The baseline is retained provenance, not permission to rerun an already-started cycle. Each regular member was compared by size/SHA-256 against its original and the original inventory was rechecked.

Verify without extracting or reading current originals:

```sh
python3 archive.py --verify .
```

Only exact manifest members under `raw/` and `source/` are accepted; symlinks, absolute paths, traversal and duplicate members are rejected. The verifier never extracts the tar. Optional `--originals` rechecks the original closed trees and current source files; a later deliberate source change can invalidate that optional check without changing archived bytes. Small summary/manifest/verification files remain outside the tar.
"""
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',action='store_true');p.add_argument('--verify',type=Path);p.add_argument('--originals',action='store_true');a=p.parse_args()
    if a.verify:print(json.dumps(verify(a.verify.resolve(),a.originals),sort_keys=True))
    elif a.run:run()
    else:print('DESCRIPTION_ONLY: new closed CPU archive; no service/model actions.')
if __name__=='__main__':main()
