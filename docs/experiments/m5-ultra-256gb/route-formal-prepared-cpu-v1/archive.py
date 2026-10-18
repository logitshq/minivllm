"""Archive the closed route-formal fake CPU attempt, including a genuine FAIL.

This verifies/copies small recorded evidence only. No tests, services, model
payloads or runtime probes run. Default is inert; the destination must be new.
"""
from pathlib import Path, PurePosixPath
import hashlib
import json
import sys
import tarfile

ROOT=Path('/Users/ryanli/code/minivllm')
SOURCE=ROOT/'.artifacts/m5-ultra-256gb/route-formal-prepared-cpu-v1'
OUT=ROOT/'docs/experiments/m5-ultra-256gb/route-formal-prepared-cpu-v1'
RUNNER=Path('/tmp/run-route-formal-prepared-cpu-v1.py').resolve()
RUNNER_SHA='9d2da0e733c801f44915ba6947d9adbf17f409b98b551209eb1e232d9f16e74d'
PYTHON=ROOT/'.artifacts/omlx-baseline/.venv/bin/python'
SUITES={
 'bench.m5_ultra_256gb.route_http_formal.test_owner_cpu':14,
 'bench.m5_ultra_256gb.route_http_formal.test_controller_cpu':12,
 'bench.m5_ultra_256gb.route_http_formal.test_audit_restore_cpu':11,
 'bench.m5_ultra_256gb.route_http_formal.test_audit_formal_cpu':13,
 'bench.m5_ultra_256gb.route_http_formal.test_execute_cpu':12,
}

def need(ok,message):
    if not ok:raise ValueError(message)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_bytes())
def write(path,value):
    with Path(path).open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')
def files(root):
    need(root.is_dir() and root.resolve()==root and not root.is_symlink(),'canonical evidence directory')
    rows={}
    for path in root.rglob('*'):
        need(not path.is_symlink(),'symlink evidence refused')
        if path.is_file():rows[str(path.relative_to(root))]=path
        else:need(path.is_dir(),'nonregular evidence refused')
    return rows
def record(path):return dict(sha256=sha(path),bytes=path.stat().st_size)

def main():
    if sys.argv[1:]!=['--run']:print('DESCRIPTION_ONLY: archive one closed fake CPU attempt; no tests or backend actions.');return 0
    result=read(SOURCE/'result.json')
    need(result['schema']=='E024-route-formal-prepared-cpu-v1' and result['status'] in ('PASS','FAIL')
         and isinstance(result.get('ended_at'),str) and result['ended_at'],'closed actual attempt required')
    need(result['expected_tests']==62 and [(r['module'],r['expected_tests']) for r in result['suites']]==list(SUITES.items()),
         'all five actual whole-module registrations')
    command=[str(PYTHON),'-I','-B',str(RUNNER)]
    for module,count in SUITES.items():command+=['--suite',f'{module}:{count}']
    command+=['--out',str(SOURCE)]
    need(result['argv']==command and sha(SOURCE/'runner.py')==RUNNER_SHA,'actual guarded command/source')
    need(sha(RUNNER)==RUNNER_SHA,'runner changed since closed attempt')
    if result['status']=='PASS':
        need(result['tests']==62 and result['failures']==result['errors']==result['skipped']==0
             and result['source_unchanged'] is True and not result['denied_operations']
             and not result['forbidden_modules_loaded'] and not result.get('loader_errors')
             and not result.get('expected_failures') and not result.get('unexpected_successes')
             and all(r['discovered_tests']==r['expected_tests'] for r in result['suites'])
             and len(result['test_outcomes'])==62 and all(r['status']=='PASS' for r in result['test_outcomes']),
             'recorded PASS is not a complete clean 62-test run')
    original=files(SOURCE);manifest=read(SOURCE/'artifact-manifest.json')['files']
    need(set(original)-{'artifact-manifest.json'}==set(manifest),'complete actual CPU artifact inventory')
    for name,meta in manifest.items():need(record(original[name])==meta,'CPU artifact changed: '+name)
    for path,meta in result['source_before'].items():
        copied=SOURCE/'frozen-source'/str(Path(path)).lstrip('/')
        need(copied.is_file() and record(copied)==meta,'original source snapshot mismatch: '+path)
    OUT.mkdir(exist_ok=False)
    (OUT/'archive.py').write_bytes(Path(__file__).read_bytes())
    for name in ('runner.py','result.json','stdout.txt','stderr.txt','artifact-manifest.json'):
        (OUT/name).write_bytes((SOURCE/name).read_bytes())
    counts='; '.join(f"{r['module'].rsplit('.',1)[-1]}: {r.get('discovered_tests','not loaded')}/{r['expected_tests']} discovered" for r in result['suites'])
    (OUT/'README.md').write_text(
        '# Route HTTP formal: prepared CPU evidence\n\n'
        f"Recorded result: **{result['status']}**, {result['tests']} tests run, {result['failures']} failures, "
        f"{result['errors']} errors and {result['skipped']} skips. Expected total: 62.\n\n{counts}.\n\n"
        'These are fake CPU ownership, restoration-protocol, raw HTTP parser/count/metric and driver-routing tests. '
        'The recorded guard denies real subprocesses, network connections, process signals and model/backend imports. '
        'The archived result retains every actual test outcome, guard event, source snapshot and stdout/stderr byte; '
        'a failed attempt is retained as failed, never promoted by successful packing.\n\n'
        'The historical GDN canary RESTORED receipts are only CPU preparation context. This run did not stop, '
        'load, restore or benchmark a real service. It grants no GPU, model, numerical or performance qualification. '
        'In particular, fake 864-request fixtures and OFF/ON installation identities do not establish real graph hits.\n\n'
        'evidence.tar.xz contains the complete original CPU tree and archive metadata. manifest.json maps every '
        'member to its source path, SHA-256 and size. archive-verification.json records decompressed member and '
        'unchanged-original verification. No weight or runtime payload is copied.\n')
    inputs={'cpu/'+name:path for name,path in original.items()}
    for name in ('archive.py','README.md'):inputs['archive-metadata/'+name]=OUT/name
    records={name:dict(path=str(path),**record(path)) for name,path in inputs.items()}
    write(OUT/'manifest.json',records);target=OUT/'evidence.tar.xz'
    with target.open('xb') as stream,tarfile.open(fileobj=stream,mode='w:xz',preset=1,dereference=True) as tar:
        for name,path in sorted(inputs.items()):tar.add(path,arcname=name,recursive=False)
    seen=set()
    with tarfile.open(target,'r:xz') as tar:
        for member in tar:
            name=PurePosixPath(member.name)
            need(member.isfile() and not name.is_absolute() and '..' not in name.parts
                 and member.name in records and member.name not in seen,'unsafe/duplicate archive member')
            raw=tar.extractfile(member).read();meta=records[member.name]
            need(len(raw)==meta['bytes'] and hashlib.sha256(raw).hexdigest()==meta['sha256'],'member bytes mismatch')
            seen.add(member.name)
    need(seen==set(records),'missing archive members')
    need(files(SOURCE)==original,'source tree changed while archiving')
    for name,path in inputs.items():need(record(path)=={k:records[name][k] for k in ('sha256','bytes')},'original bytes changed')
    verification=dict(status='PASS',cpu_status=result['status'],tests=result['tests'],expected_tests=62,
        members=len(records),logical_bytes=sum(r['bytes'] for r in records.values()),archive_bytes=target.stat().st_size,
        archive_sha256=sha(target),runner_sha256=RUNNER_SHA,result_sha256=sha(SOURCE/'result.json'),
        all_members_rechecked=True,originals_rechecked=True,source_tree_rechecked=True,
        model_payloads_included=False,tests_rerun=False,performance_qualified=False)
    write(OUT/'archive-verification.json',verification);print(json.dumps(verification));return 0

if __name__=='__main__':raise SystemExit(main())
