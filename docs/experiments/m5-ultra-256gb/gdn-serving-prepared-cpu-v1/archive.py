"""Small closed CPU/preparation evidence only; --run explicitly archives it.
Uses the already captured PREPARED snapshot, never rereads the active journal.
No tests, runtime imports, service operations or canary tensor payload reads.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import tarfile
import tempfile

R=Path('/Users/ryanli/code/minivllm');A=R/'.artifacts/m5-ultra-256gb'
P=R/'bench/m5_ultra_256gb/gdn_prework_unbound/prereg-serving-v1'
OUT=R/'docs/experiments/m5-ultra-256gb/gdn-serving-prepared-cpu-v1'
PRIOR=R/'docs/experiments/m5-ultra-256gb/E029-gdn-prework-canary-v1'
PLAN='c908eed8087cd72a86a4994a7fc37ffb41bfd719726ad04ff6d8b6f0768c14d2'
CHILD='dfb6d3d306e6247e34dfdfe54746e12205b297e166e1f175e85024786d73163b'
PREPARED='70a4269e63af463c721855a255c1c577b1fe8b124abba577cafcafecd08e1fcd'
TREES={'guard-first':A/'gdn-serving-prepared-cpu-v1','guard-v2':A/'gdn-serving-prepared-cpu-v2',
       'actual-registration':A/'gdn-prework-serving-v1-registration',
       'inherited-cpu-prepare':A/'gdn-prework-serving-v1-cpu-prepare',
       'preregistration':P,'prepared-service':OUT/'prepared-service-snapshot'}
DENY={'.safetensors','.bin','.so','.dylib','.metallib','.pack','.xz','.gz','.zip'}
def need(ok,message):
    if not ok:raise ValueError(message)
def read(path):return json.loads(path.read_bytes())
def info(path):
    need(path.is_file() and not path.is_symlink(),'regular input '+str(path))
    raw=path.read_bytes();need(len(raw)<8*(1<<20),'small evidence only '+str(path))
    return dict(sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))
def sha(path):return info(path)['sha256']
def write(path,value):
    with path.open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')
def listing(root):
    need(root.is_dir() and not root.is_symlink(),'real source tree')
    result=[]
    for folder,dirs,names in os.walk(root,followlinks=False):
        for name in sorted(dirs+names):
            path=Path(folder)/name;need(not path.is_symlink(),'unexpected symlink')
            if path.is_file():
                need(path.suffix not in DENY,'no model/runtime/archive payload');result.append(path)
            else:need(path.is_dir(),'special source file')
    return sorted(result)
def run():
    need(OUT.is_dir() and not (OUT/'evidence.tar.xz').exists() and not (OUT/'manifest.json').exists(),'fresh prepared archive')
    first,v2,registration,prepared=[read(TREES[k]/'result.json') for k in ('guard-first','guard-v2','actual-registration','inherited-cpu-prepare')]
    receipt=read(P/'receipt.json');journal=read(OUT/'prepared-journal.json')
    need(first['schema']=='GDN-prework-serving-prepared-cpu-v1' and first['status']=='FAIL' and first['tests']==0
         and first['denied_operations']==[dict(kind='backend import',detail='tutorial_code')],'retain actual first guard rejection')
    need(v2['schema']=='GDN-prework-serving-prepared-cpu-v2' and v2['status']=='PASS' and v2['tests']==40
         and v2['failures']==v2['errors']==v2['skipped']==0 and v2['source_unchanged'] is True
         and v2['denied_operations']==v2['forbidden_modules_loaded']==[]
         and v2['allowed_cpu_imports_unchanged'] is True,'complete actual 40 CPU PASS')
    need(v2['previous_attempt']['sha256']==sha(TREES['guard-first']/'result.json')
         ==sha(TREES['guard-v2']/'first-attempt-result.json'),'actual retained failed attempt')
    need(registration['status']==prepared['status']=='PASS' and registration['source_unchanged'] is True
         and registration['denied_operations']==registration['forbidden_modules_loaded']==[],'actual CPU preparation PASS')
    need(receipt['schema']=='GDN-prework-serving-prereg-receipt-v1' and receipt['status']=='PREPARED'
         and receipt['owner_plan_sha256']==sha(P/'owner-plan.json')==PLAN
         and receipt['child_plan_sha256']==sha(P/'child-plan.json')==CHILD
         and receipt['prepared_journal_sha256']==sha(OUT/'prepared-journal.json')==PREPARED
         and journal['state']=='PREPARED' and journal['owner_plan']['sha256']==PLAN,'exact registered historical PREPARED bytes')
    for path,digest in receipt['source_pins'].items():need(sha(Path(path))==digest,'registered small source changed')
    for row in (first,v2):
        need(row['source_before']==row['source_after'] and row['source_unchanged'] is True,'whole attempt immutable source contract')
    for row in (registration,prepared):need(row['owner_plan_sha256']==PLAN,'actual prepared plan')
    mapping={}
    for label,root in TREES.items():
        for path in listing(root):mapping[label+'/'+str(path.relative_to(root))]=path
    tmpnames=('run-gdn-serving-prepared-cpu-v1.py','run-gdn-serving-prepared-cpu-v2.py',
              'prepare-gdn-serving-v1.py','prepare-gdn-serving-v2.py')
    for name in tmpnames:mapping['runners/'+name]=Path('/tmp')/name
    for name in ('archive-verification.json','summary.json','restored-journal.json','restoration-audit.json',
                 'independent-numerical.json','resource-summary.json'):
        mapping['canary-reference/'+name]=PRIOR/name
    mapping['prepared-service-snapshot-manifest.json']=OUT/'prepared-service-snapshot-manifest.json'
    shutil.copyfile(__file__,OUT/'archive.py');mapping['archive.py']=OUT/'archive.py'
    write(OUT/'archive-command.json',dict(argv=[sys.executable,str(Path(__file__).resolve()),'--run'],
        source_sha256=sha(Path(__file__)),scope='CPU evidence archival only; actual PREPARED snapshot preserved before launch.'))
    mapping['archive-command.json']=OUT/'archive-command.json'
    files={name:dict(source=str(path),**info(path)) for name,path in sorted(mapping.items())}
    write(OUT/'manifest.json',dict(schema='GDN-serving-prepared-cpu-archive-manifest-v1',files=files))
    with tarfile.open(OUT/'evidence.tar.xz','w:xz',preset=3,dereference=True) as archive:
        for name,path in sorted(mapping.items()):archive.add(path,arcname=name,recursive=False)
    # Every entry is regular, relative and unique. Safely extract and compare
    # actual bytes against every recorded source hash, including zero-byte files.
    with tempfile.TemporaryDirectory(prefix='gdn-serving-cpu-verify-') as folder:
        target=Path(folder);seen=set()
        with tarfile.open(OUT/'evidence.tar.xz','r:xz') as archive:
            for member in archive:
                name=member.name;p=PurePosixPath(name)
                need(member.isfile() and not p.is_absolute() and '..' not in p.parts and name not in seen
                     and name in files and member.size==files[name]['bytes'],'safe exact regular member')
                seen.add(name);dest=target/name;dest.parent.mkdir(parents=True,exist_ok=True)
                stream=archive.extractfile(member);need(stream is not None,'regular payload')
                with stream,dest.open('xb') as out:shutil.copyfileobj(stream,out)
                need(info(dest)=={k:files[name][k] for k in ('sha256','bytes')},'extracted SHA/size mismatch')
        need(seen==set(files),'complete archive membership')
    for name,path in mapping.items():need(info(path)=={k:files[name][k] for k in ('sha256','bytes')},'input changed while packing')
    sidecars={'guard-first-result.json':TREES['guard-first']/'result.json','guard-v2-result.json':TREES['guard-v2']/'result.json',
              'registration-result.json':TREES['actual-registration']/'result.json',
              'cpu-prepare-result.json':TREES['inherited-cpu-prepare']/'result.json','registration-receipt.json':P/'receipt.json'}
    for name,path in sidecars.items():shutil.copyfile(path,OUT/name)
    summary=dict(schema='GDN-serving-prepared-cpu-archive-summary-v1',
        first_attempt=dict(status='FAIL',tests=0,reason='Guard rejected pure stdlib tutorial_code scheduler during discovery',result=info(TREES['guard-first']/'result.json')),
        retry=dict(status='PASS',tests=40,failures=0,errors=0,skipped=0,denied_operations=[],result=info(TREES['guard-v2']/'result.json')),
        actual_cpu_prepare='PASS',actual_readonly_service_prepare='PREPARED',owner_plan_sha256=PLAN,
        child_plan_sha256=CHILD,prepared_journal_sha256=PREPARED,registered_source_count=len(receipt['source_pins']),
        members=len(files),raw_bytes=sum(x['bytes'] for x in files.values()),
        canary_archive_reference=info(PRIOR/'archive-verification.json'),canary_payload_duplicated=False,
        numerical_qualified=False,http_qualified=False,performance_qualified=False,
        scope='CPU and historical read-only preparation only. No serving experiment result or current service-state claim.')
    write(OUT/'summary.json',summary)
    archive_info=dict(sha256=hashlib.sha256((OUT/'evidence.tar.xz').read_bytes()).hexdigest(),bytes=(OUT/'evidence.tar.xz').stat().st_size)
    write(OUT/'archive-verification.json',dict(schema='GDN-serving-prepared-cpu-archive-verification-v1',status='PASS',
        archive=archive_info,manifest=info(OUT/'manifest.json'),members=len(files),raw_bytes=summary['raw_bytes'],
        all_members_regular=True,all_safe_extracted_members_verified=True,all_small_inputs_rechecked=True,
        source_pins_unchanged=True,prepared_journal_sha256=PREPARED,owner_plan_sha256=PLAN,child_plan_sha256=CHILD,
        sidecars={name:info(OUT/name) for name in (*sidecars,'prepared-journal.json','prepared-service-snapshot-manifest.json','summary.json')},
        gpu_executed_by_archiver=False,qualification_granted=False))
    (OUT/'README.md').write_text(f'''# GDN serving: CPU and preparation evidence

The original guard attempt stopped during discovery (0 tests) because it denied `tutorial_code`, including the pure standard-library scheduler. Its source, failure, stdout/stderr and snapshots are retained unchanged. The v2 guard admits only the exact namespace and five SHA-pinned CPU modules; all other device/backend, subprocess, network and signal operations remain denied. The complete three-module suite then passed **40/40**, with no errors, skips or denied operations and unchanged sources.

Actual inherited CPU preparation passed. The separate read-only controller preparation and sealed receipt reached **PREPARED**. The historical journal snapshot was saved before launch; this archive makes no claim about the current service or subsequent experiment.

- Owner plan: `{PLAN}`.
- Child plan: `{CHILD}`.
- PREPARED journal: `{PREPARED}`.
- Registered source/evidence pins: {len(receipt['source_pins'])}.
- Archive: **{len(files)} members**, **{summary['raw_bytes']:,} uncompressed bytes**, **{archive_info['bytes']:,} compressed bytes**.
- `evidence.tar.xz` SHA256: `{archive_info['sha256']}`.

Every member was restricted to a unique regular relative path, safely extracted, then checked for exact SHA256 and size. All packed small inputs were rechecked. Both actual CPU attempts, original inherited preflight, registration, preregistered sources, retained runner versions, command receipts, and historical service preparation are included. No tests were rerun during archiving.

The [existing canary archive](../E029-gdn-prework-canary-v1/README.md) retains its 60 raw tensor files; this archive stores only its small evidence references. It does not duplicate the 176,734,788-byte tensor payload. Preparation does not qualify the planned 37-request serving workload, live HTTP, parameter replacement, E024/E028 composition, or performance. Those require subsequent closed evidence and independent audits.
''')
    print(json.dumps(dict(status='PASS',directory=str(OUT),members=len(files),raw_bytes=summary['raw_bytes'],archive=archive_info)))
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',action='store_true');args=p.parse_args()
    if args.run:run()
    else:print('DESCRIPTION_ONLY: small CPU/preparation evidence; --run required.')
if __name__=='__main__':main()
