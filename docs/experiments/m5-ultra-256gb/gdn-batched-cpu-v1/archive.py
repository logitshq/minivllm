"""Closed GDN batched12 CPU evidence only; --run archives, never reruns tests."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import shutil
import sys
import tarfile
import tempfile

ROOT=Path('/Users/ryanli/code/minivllm')
CPU=ROOT/'.artifacts/m5-ultra-256gb/gdn-batched-prepared-cpu-v1'
OUT=ROOT/'docs/experiments/m5-ultra-256gb/gdn-batched-cpu-v1'
HELPER=Path('/tmp/archive-gdn-http-cpu-v1.py')
# Helper contains only stdlib archive primitives; its actual frozen copy is retained.
HELPER_SHA='48d97439ca2cafc408bfe9c7fb1da0709c33e13b466d071767c569f864c512ef'

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',action='store_true');args=parser.parse_args()
    if not args.run:
        print('DESCRIPTION_ONLY: closed batched12 CPU evidence; --run required.');return
    raw=HELPER.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=HELPER_SHA:raise ValueError('archive helper changed')
    spec=importlib.util.spec_from_file_location('_gdn_cpu_archive_primitives',HELPER)
    h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
    result=h.read(CPU/'result.json');inventory=h.read(CPU/'artifact-manifest.json')
    h.need(result['schema']=='GDN-batched-prepared-cpu-v1' and result['status']=='PASS'
           and result['expected_tests']==result['discovered_tests']==result['tests']==12,'whole batched12 PASS')
    h.need(all(result[k]==0 for k in ('failures','errors','skipped','expected_failures','unexpected_successes'))
           and result['denied_operations']==result['forbidden_modules_loaded']==result['loader_errors']==[],'no ignored failure/operation')
    h.need(result['source_before']==result['source_after'] and result['source_unchanged'] is True
           and result['closure_unchanged'] is True and len(result['source_before'])==14
           and len(result['test_outcomes'])==12 and all(x['status']=='PASS' for x in result['test_outcomes']), 'whole test/source invariants')
    before=h.listing(CPU)
    h.need(inventory['schema']=='cpu-artifact-inventory-v1' and set(inventory['files'])==
           {str(p.relative_to(CPU)) for p in before if p.name!='artifact-manifest.json'}, 'complete producer inventory')
    for name,row in inventory['files'].items():h.need(h.info(CPU/name)==row,'actual inventory bytes')
    for name,row in result['source_before'].items():h.need(h.info(CPU/'frozen-source'/name.lstrip('/'))==row,'actual source snapshot')
    for name,row in result['closure_inputs'].items():h.need(h.info(CPU/name)=={k:row[k] for k in ('sha256','bytes')},'closure snapshots')
    OUT.mkdir(exist_ok=False)
    (OUT/'verify.py').write_text(h.VERIFY);shutil.copyfile(__file__,OUT/'archive.py');shutil.copyfile(HELPER,OUT/'archive-helper.py')
    h.write(OUT/'archive-command.json',dict(argv=[sys.executable,str(Path(__file__).resolve()),'--run'],source=h.info(Path(__file__)),helper=h.info(HELPER),device_operations=False))
    mapping={'cpu/'+str(p.relative_to(CPU)):p for p in before}
    for name in ('verify.py','archive.py','archive-helper.py','archive-command.json'):mapping[name]=OUT/name
    files={name:dict(source=str(p),**h.info(p)) for name,p in sorted(mapping.items())}
    h.write(OUT/'manifest.json',dict(schema='GDN-batched-cpu-archive-manifest-v1',files=files))
    with tarfile.open(OUT/'evidence.tar.xz','w:xz',preset=3,dereference=True) as archive:
        for name,p in sorted(mapping.items()):archive.add(p,arcname=name,recursive=False)
    with tempfile.TemporaryDirectory(prefix='gdn-batched-cpu-verify-') as folder:
        base=Path(folder);seen=set()
        with tarfile.open(OUT/'evidence.tar.xz','r:xz') as archive:
            for member in archive:
                name=member.name;p=PurePosixPath(name)
                h.need(member.isfile() and not p.is_absolute() and '..' not in p.parts and name not in seen
                       and name in files and member.size==files[name]['bytes'],'safe exact archive member')
                seen.add(name);dest=base/name;dest.parent.mkdir(parents=True,exist_ok=True)
                stream=archive.extractfile(member);h.need(stream is not None,'regular payload')
                with stream,dest.open('xb') as target:shutil.copyfileobj(stream,target)
                h.need(h.info(dest)=={k:files[name][k] for k in ('sha256','bytes')},'safe extracted bytes')
        h.need(seen==set(files),'all members verified')
    for name,p in mapping.items():h.need(h.info(p)=={k:files[name][k] for k in ('sha256','bytes')},'original bytes unchanged')
    h.need(before==h.listing(CPU),'closed tree membership unchanged')
    shutil.copyfile(CPU/'result.json',OUT/'cpu-result.json')
    summary=dict(schema='GDN-batched-cpu-archive-summary-v1',status='PASS',tests=12,failures=0,errors=0,skipped=0,
        source_files=14,source_unchanged=True,denied_operations=[],result=h.info(CPU/'result.json'),
        runner=h.info(CPU/'runner.py'),members=len(files),raw_bytes=sum(x['bytes'] for x in files.values()),
        device_qualified=False,numerical_qualified=False,http_qualified=False,performance_qualified=False,
        scope='One complete fake CPU module; no device or service operation and no arithmetic/performance qualification.')
    h.write(OUT/'summary.json',summary);compressed=h.info(OUT/'evidence.tar.xz')
    (OUT/'README.md').write_text(f'''# GDN batched prework: actual 12-test CPU evidence

The first full `gdn_prework_batched.test_candidate_cpu` run passed **12/12**:
zero failures/errors/skips/expected failures/unexpected successes, no forbidden
operations or backend modules, and **14 unchanged source snapshots**. No source
fix or retry was needed. Full stdout/stderr, per-test/subtest results, the actual
guard runner, source copies, prerequisite snapshots and original inventory are
retained. The suite took 0.006 s (guard procedure 0.01524 s); neither is a device
performance result.

The guard denied all socket creation, real subprocesses, signal delivery/timers,
backend imports and model payload opens. It admitted execution only after the
prior formal cycle was closed, independently restored and archived. That prior
formal audit's FAIL remains unchanged. **This archive grants no numerical,
full-model, HTTP or performance qualification.** The candidate/README were not
changed after their source-prepared freeze; the later CPU result is recorded
separately in [RESULTS.md](../../../../bench/m5_ultra_256gb/gdn_prework_batched/RESULTS.md).

- {len(files)} regular members; {summary['raw_bytes']:,} raw bytes;
  {compressed['bytes']:,} compressed bytes.
- Archive SHA256: `{compressed['sha256']}`.
- CPU result SHA256: `{summary['result']['sha256']}`.
- Guard runner SHA256: `{summary['runner']['sha256']}`.

Every tar member was safely extracted and checked for exact size/SHA256;
all original bytes and tree membership were rechecked. No tests were rerun by
archiving. [Manifest](manifest.json) and [verification](archive-verification.json)
retain these bindings. `python -B verify.py` rechecks the complete archive and
sidecars independently using the standard library, with no service/device calls.
''')
    h.write(OUT/'archive-verification.json',dict(schema='GDN-batched-cpu-archive-verification-v1',status='PASS',
        archive=compressed,manifest=h.info(OUT/'manifest.json'),members=len(files),raw_bytes=summary['raw_bytes'],
        all_safe_extracted_members_verified=True,all_original_files_rechecked=True,all_tree_members_rechecked=True,
        sidecars={name:h.info(OUT/name) for name in ('cpu-result.json','summary.json','README.md','verify.py')},
        tests_rerun=False,device_executed=False,qualification_granted=False))
    print(json.dumps(dict(status='PASS',directory=str(OUT),members=len(files),raw_bytes=summary['raw_bytes'],archive=compressed)))
if __name__=='__main__':main()
