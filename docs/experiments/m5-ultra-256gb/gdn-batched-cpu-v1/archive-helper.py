"""Archive the completed GDN HTTP16 CPU/queue evidence, never rerun tests."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import tarfile
import tempfile

ROOT=Path('/Users/ryanli/code/minivllm')
ART=ROOT/'.artifacts/m5-ultra-256gb'
CPU=ART/'gdn-http-prepared-cpu-v1'
QUEUE=ART/'gdn-http-prepared-cpu-v1-queue'
OUT=ROOT/'docs/experiments/m5-ultra-256gb/gdn-http-cpu-v1'
RUNNER_SHA='22e9ef7a46e1a2ea76940abc8e5152088e124d5f060f02c2ee63c1eeb51b35cc'
VERIFY='''"""Independently verify every regular archive member and sidecar (stdlib only)."""
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
'''

def need(ok,message):
    if not ok:raise ValueError(message)
def read(p):return json.loads(p.read_bytes())
def info(p):
    need(p.is_file() and not p.is_symlink(),'regular file '+str(p))
    a=p.stat();need(a.st_size<=32*(1<<20),'bounded evidence '+str(p));raw=p.read_bytes();b=p.stat()
    attrs=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)
    need(attrs(a)==attrs(b) and len(raw)==a.st_size,'changed source '+str(p))
    return dict(sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))
def write(p,data):
    with p.open('x') as f:json.dump(data,f,indent=2,allow_nan=False);f.write('\n')
def listing(root):
    need(root.is_dir() and not root.is_symlink(),'real evidence tree')
    files=[]
    for folder,dirs,names in os.walk(root,followlinks=False):
        for name in sorted(dirs+names):
            p=Path(folder)/name;need(not p.is_symlink(),'no symlink '+str(p))
            if p.is_file():
                need(p.suffix not in ('.safetensors','.bin','.so','.dylib','.metallib','.pack','.xz'), 'no device/runtime/archive payload')
                files.append(p)
            else:need(p.is_dir(),'special file')
    return sorted(files)

def run():
    result=read(CPU/'result.json');queue=read(QUEUE/'status.json')
    need(result['schema']=='GDN-http-prepared-cpu-v1' and result['status']=='PASS'
         and result['expected_tests']==result['discovered_tests']==result['tests']==16,'whole HTTP16 PASS')
    need(all(result[k]==0 for k in ('failures','errors','skipped','expected_failures','unexpected_successes'))
         and result['denied_operations']==result['forbidden_modules_loaded']==result['loader_errors']==[], 'no ignored failures/operations')
    need(result['source_unchanged'] is True and result['closure_unchanged'] is True
         and result['source_before']==result['source_after'] and len(result['test_outcomes'])==16
         and all(x['status']=='PASS' for x in result['test_outcomes']), 'all tests and source invariants')
    need(queue['status']==queue['test_status']=='PASS' and queue['tests']==16 and type(queue['returncode']) is int and queue['returncode']==0
         and queue['cpu_tests_started'] is True and queue['cpu_tests_completed'] is True and queue['service_actions'] is False
         and queue['source_sha256']==RUNNER_SHA and queue['result_sha256']==info(CPU/'result.json')['sha256'], 'closed actual queue/result')
    need(queue['started_at']<=queue['child_started_at']<=queue['child_ended_at']<=queue['ended_at'], 'closed queue chronology')
    for p in (CPU/'runner.py',QUEUE/'cpu-runner.py'):need(info(p)['sha256']==RUNNER_SHA,'actual runner bytes')
    for p,row in result['source_before'].items():need(info(CPU/'frozen-source'/p.lstrip('/'))==row,'full source snapshot')
    for name,row in result['closure_inputs'].items():need(info(CPU/name)=={k:row[k] for k in ('sha256','bytes')},'closure snapshot')
    need(queue['closure_sha256']==info(CPU/'closure-status.json')['sha256'],'actual queue closure binding')
    need(result['closure_gate']['status']=='EXPERIMENT_FAILED_EVIDENCE_RETAINED'
         and result['closure_gate']['experiment_audit_status']=='FAIL'
         and result['closure_gate']['restoration_qualified'] is True
         and result['closure_gate']['archive_verified'] is True,'retain actual failed experiment/restoration distinction')
    inventory=read(CPU/'artifact-manifest.json')
    cpu_files=listing(CPU)
    need(inventory['schema']=='cpu-artifact-inventory-v1'
         and set(inventory['files'])=={str(p.relative_to(CPU)) for p in cpu_files if p.name!='artifact-manifest.json'},'whole producer inventory')
    for name,row in inventory['files'].items():need(info(CPU/name)==row,'producer inventory exact bytes')
    before={label:[str(p.relative_to(root)) for p in listing(root)] for label,root in (('cpu',CPU),('queue',QUEUE))}
    OUT.mkdir(exist_ok=False)
    (OUT/'verify.py').write_text(VERIFY)
    shutil.copyfile(__file__,OUT/'archive.py')
    write(OUT/'archive-command.json',dict(argv=[sys.executable,str(Path(__file__).resolve()),'--run'],source=info(Path(__file__)),device_operations=False))
    mapping={}
    for label,root in (('cpu',CPU),('queue',QUEUE)):
        for p in listing(root):mapping[label+'/'+str(p.relative_to(root))]=p
    for name in ('archive.py','verify.py','archive-command.json'):mapping[name]=OUT/name
    files={name:dict(source=str(p),**info(p)) for name,p in sorted(mapping.items())}
    write(OUT/'manifest.json',dict(schema='GDN-http-cpu-archive-manifest-v1',files=files))
    with tarfile.open(OUT/'evidence.tar.xz','w:xz',preset=3,dereference=True) as archive:
        for name,p in sorted(mapping.items()):archive.add(p,arcname=name,recursive=False)
    with tempfile.TemporaryDirectory(prefix='gdn-http-cpu-archive-') as folder:
        seen=set();base=Path(folder)
        with tarfile.open(OUT/'evidence.tar.xz','r:xz') as archive:
            for member in archive:
                name=member.name;p=PurePosixPath(name)
                need(member.isfile() and not p.is_absolute() and '..' not in p.parts and name not in seen
                     and name in files and member.size==files[name]['bytes'],'safe exact archive member')
                seen.add(name);dest=base/name;dest.parent.mkdir(parents=True,exist_ok=True)
                stream=archive.extractfile(member);need(stream is not None,'regular payload')
                with stream,dest.open('xb') as target:shutil.copyfileobj(stream,target)
                need(info(dest)=={k:files[name][k] for k in ('sha256','bytes')},'safe extracted bytes')
        need(seen==set(files),'all members verified')
    for name,p in mapping.items():need(info(p)=={k:files[name][k] for k in ('sha256','bytes')},'original bytes unchanged')
    for label,root in (('cpu',CPU),('queue',QUEUE)):
        need(before[label]==[str(p.relative_to(root)) for p in listing(root)],'tree membership unchanged')
    sidecars={'cpu-result.json':CPU/'result.json','queue-status.json':QUEUE/'status.json',
              'closure-status.json':CPU/'closure-status.json','restoration-audit.json':CPU/'restoration-audit.json'}
    for name,p in sidecars.items():shutil.copyfile(p,OUT/name)
    summary=dict(schema='GDN-http-cpu-archive-summary-v1',status='PASS',tests=16,failures=0,errors=0,skipped=0,
        source_files=len(result['source_before']),source_unchanged=True,denied_operations=[],
        runner_sha256=RUNNER_SHA,result=info(CPU/'result.json'),queue=info(QUEUE/'status.json'),
        closed_prior_experiment_status='EXPERIMENT_FAILED_EVIDENCE_RETAINED',prior_restoration_qualified=True,
        members=len(files),raw_bytes=sum(x['bytes'] for x in files.values()),
        device_qualified=False,numerical_qualified=False,http_qualified=False,performance_qualified=False,
        scope='Complete fake CPU tests and queue evidence only; historical failed formal audit retained unchanged.')
    write(OUT/'summary.json',summary)
    compressed=info(OUT/'evidence.tar.xz')
    (OUT/'README.md').write_text(f'''# GDN HTTP: actual 16-test CPU evidence

The complete frozen `gdn_prework_http.test_http_cpu` module passed **16/16** once,
with zero failures, errors, skips, expected failures, unexpected successes or
forbidden operations. All **{len(result['source_before'])} source/metadata snapshots**
matched before and after execution. The runner and queue terminal receipts,
full stdout/stderr, per-test/subtest outcomes and complete source copies are retained.
No tests were rerun by this archiver.

The queue waited for the prior E024 route formal cycle to close, independently
restore and archive. That cycle's recorded status was
`EXPERIMENT_FAILED_EVIDENCE_RETAINED`, with its formal audit `FAIL` and restoration
`PASS`; this CPU success neither changes nor repairs that historical result.
The guard blocked device/backend imports, model payload access, real subprocess,
network and signal delivery. Local asyncio socketpair/handler registration was
allowed, as recorded by the runner. These are fake CPU contracts, **not a model,
real HTTP request, numerical, full-state or performance qualification**.

- Archive: {len(files)} members; {summary['raw_bytes']:,} uncompressed bytes;
  {compressed['bytes']:,} compressed bytes.
- `evidence.tar.xz` SHA256: `{compressed['sha256']}`.
- Frozen runner SHA256: `{RUNNER_SHA}`.
- Actual result SHA256: `{summary['result']['sha256']}`.

[Manifest](manifest.json) binds every member to its original source path, size and
SHA256. [Verification](archive-verification.json) records safe extraction,
all-member checking and unchanged original trees. Run `python -B verify.py` from
any directory to independently recheck the compressed archive and sidecars;
it uses only the standard library and performs no service/device actions.
''')
    write(OUT/'archive-verification.json',dict(schema='GDN-http-cpu-archive-verification-v1',status='PASS',
        archive=compressed,manifest=info(OUT/'manifest.json'),members=len(files),raw_bytes=summary['raw_bytes'],
        all_safe_extracted_members_verified=True,all_original_files_rechecked=True,all_tree_members_rechecked=True,
        sidecars={name:info(OUT/name) for name in (*sidecars,'summary.json','README.md','verify.py')},
        tests_rerun=False,device_executed=False,qualification_granted=False))
    print(json.dumps(dict(status='PASS',directory=str(OUT),members=len(files),raw_bytes=summary['raw_bytes'],archive=compressed)))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',action='store_true');a=p.parse_args()
    if a.run:run()
    else:print('DESCRIPTION_ONLY: completed GDN HTTP16 CPU/queue archival; --run required.')
if __name__=='__main__':main()
