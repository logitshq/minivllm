"""Preserve the exact failed E024 formal-v1 CPU registration and partial plan.

Default is description-only. --run writes an exclusive small archive, reopens
and safely extracts every regular member, then rechecks all original bytes.
No tests, service, process probes, runtime imports or model payload traversals.
Archive verification PASS never changes the original registration FAIL.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import sys
import tarfile
import tempfile
import traceback

ROOT=Path('/Users/ryanli/code/minivllm')
ART=ROOT/'.artifacts/m5-ultra-256gb'
REG=ART/'route-http-formal-v1-registration'
PREREG=ROOT/'bench/m5_ultra_256gb/route_http_formal/prereg-formal-v1'
OUT=ROOT/'docs/experiments/m5-ultra-256gb/route-http-formal-registration-failure-v1'
RESULT_SHA='fa9e4b860f246d09808fe01bfb8a64a41d4346098406659dc57e97abf38b6c14'
WRAPPER_SHA='4280736526e01198aa1f94c705c55458ee9bfe27966871432d7205ffe777039a'
REGISTRAR_SHA='348594a483e2a2e8dad1e0f5bed71667d6c529f965788d30769f8af111b224c3'
PLAN_SHA='d2f66bc9b744e5325476fb9d6c3734c4639d7f80d5d4848b86178c87016baeae'
WRONG_SHA='a594ab8885b6ba24dd24e644a037b06c6af3a47846d02ae22796a797bc4e96f8'
PREPARATION_SHA='57c9f1131ba6a927f8d50badfd92e7f01b93940d746e22d7f4a0ead4b293ead3'
COMMANDS_SHA='08e895d12a31b7724fa587793ce3e6e18174ddd6181a746c7d5617ad6b926e46'


def need(ok,message):
    if not ok:raise ValueError(message)
def read(path):return json.loads(path.read_bytes())
def info(path):
    h=hashlib.sha256();size=0
    with path.open('rb') as stream:
        while block:=stream.read(1<<20):h.update(block);size+=len(block)
    return dict(sha256=h.hexdigest(),bytes=size)
def write(path,value):
    with path.open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')
def listing(root):
    need(root.is_dir() and not root.is_symlink(),'regular original tree required')
    paths=sorted(root.rglob('*'))
    need(all(not p.is_symlink() and (p.is_file() or p.is_dir()) for p in paths),'symlink/nonregular original entry')
    return [p for p in paths if p.is_file()]
def command_value(command,flag):
    need(command.count(flag)==1,'one command flag: '+flag);return command[command.index(flag)+1]
def normalized_header(member):
    member.uid=member.gid=0;member.uname=member.gname='';member.mtime=0
    return member


def run():
    need(not OUT.exists(),'exclusive new archive directory')
    trees={'registration':REG,'partial-preregistration':PREREG}
    original={label:listing(root) for label,root in trees.items()}
    need(info(REG/'result.json')['sha256']==RESULT_SHA,'exact original FAIL receipt')
    result=read(REG/'result.json')
    need(result['schema']=='E024-route-formal-guarded-registration-v1' and result['status']=='FAIL'
         and result['error']['type']=='RuntimeError' and result['error']['message']=='route formal plan SHA','exact failure scope')
    need(all(result[k] is False for k in ('gpu_executed','service_changed','service_prepared','numerical_qualified','http_qualified','performance_qualified')),
         'registration must not claim execution/qualification')
    need(result['source_unchanged'] is True and result['source_before']==result['source_after']
         and result['denied_operations']==result['forbidden_modules_loaded']==[],'actual unchanged guarded execution')
    need(info(REG/'runner.py')['sha256']==WRAPPER_SHA,'actual wrapper source snapshot')
    inventory=read(REG/'artifact-manifest.json')
    need(inventory['schema']=='cpu-registration-artifact-manifest-v1'
         and set(inventory['files'])=={str(p.relative_to(REG)) for p in original['registration']}-{'artifact-manifest.json'},'original complete artifact inventory')
    for name,expected in inventory['files'].items():need(info(REG/name)==expected,'original inventory hash: '+name)
    for path,expected in result['source_before'].items():need(info(REG/'frozen-source'/path.lstrip('/'))==expected,'actual source snapshot: '+path)
    prep=read(PREREG/'preparation.json');commands=read(PREREG/'commands.json')
    need(info(PREREG/'preparation.json')['sha256']==PREPARATION_SHA and info(PREREG/'commands.json')['sha256']==COMMANDS_SHA,'original malformed build receipts unchanged')
    need(info(PREREG/'owner-plan.json')['sha256']==PLAN_SHA and prep['owner_plan_sha256']==WRONG_SHA!=PLAN_SHA,'retain incorrect claimed owner SHA')
    need(prep['schema']=='E024-route-formal-prereg-build-v1' and prep['status']=='PLAN_BUILT'
         and prep['service_prepared'] is prep['runtime_weights_verified'] is prep['performance_qualified'] is False,'partial build scope')
    for name,flag in (('cpu_prepare','--plan-sha256'),('readonly_service_prepare','--owner-plan-sha256'),('owner','--plan-sha256'),('independent_formal_audit','--plan-sha256')):
        need(command_value(commands[name],flag)==WRONG_SHA,'original bad command preserved: '+name)
    test_path=str(ROOT/'bench/m5_ultra_256gb/route_http_formal/test_execute_cpu.py')
    registrar=str(ROOT/'bench/m5_ultra_256gb/route_http_formal/preregister.py')
    need(prep['source_pins'][test_path]==WRONG_SHA and prep['source_pins'][registrar]==REGISTRAR_SHA,'wrong SHA came from registered test source')
    need(next(reversed(prep['source_pins']))==test_path,'actual final pin-loop item')
    for path,digest in prep['source_pins'].items():
        need(info(PREREG/'frozen-source'/path.lstrip('/'))['sha256']==digest,'all original preregistered source copies')
    expected_copies={str(Path(path).relative_to('/')) for path in prep['source_pins']}
    need({str(p.relative_to(PREREG/'frozen-source')) for p in listing(PREREG/'frozen-source')}==expected_copies,'exact frozen-source closure')
    need(info(REG/'cpu-result.json')['sha256']==result['cpu_receipt']['sha256'],'original CPU receipt copy')
    need(read(REG/'cpu-result.json')['status']=='PASS','prior CPU tests distinct from failed registration')
    for path in (ART/'route-http-formal-v1-cold-journal',ART/'route-http-formal-v1-owner',ART/'route-http-formal-v1-execution',ART/'route-http-formal-v1-cpu-prepare',PREREG/'receipt.json'):
        need(not path.exists(),'unexpected v1 execution/preparation artifact: '+str(path))
    # These filesystem checks plus guarded-wrapper receipts establish this local
    # attempted registration's scope, not an independent observation of the OS.
    OUT.mkdir()
    (OUT/'archive.py').write_bytes(Path(__file__).read_bytes())
    write(OUT/'pack-command.json',dict(argv=[sys.executable,*sys.argv],created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        scope='Read-only original small evidence, exclusive archive writes; no backend, tests, service or device.'))
    for name,source in (('failed-result.json',REG/'result.json'),('original-preparation.json',PREREG/'preparation.json'),
                        ('original-commands.json',PREREG/'commands.json')):
        (OUT/name).write_bytes(source.read_bytes())
    summary=dict(schema='E024-route-formal-registration-failure-summary-v1',registration_status='FAIL',
        result_sha256=RESULT_SHA,actual_owner_plan_sha256=PLAN_SHA,incorrect_recorded_sha256=WRONG_SHA,
        wrong_sha_source=test_path,registrar_sha256=REGISTRAR_SHA,wrapper_sha256=WRAPPER_SHA,
        error=result['error'],source_pin_count=len(prep['source_pins']),
        original_tree_file_counts={k:len(v) for k,v in original.items()},
        service_prepared=False,gpu_executed=False,performance_qualified=False,execution_registered=False,
        cause='The registrar reused digest as the source-pin loop variable; after the loop it wrote the final test source SHA into owner-plan command/preparation fields. Actual owner CPU preparation rejected that SHA.',
        scope='Original failure and complete partial files preserved without repair. Archival verification is separate from registration success.')
    write(OUT/'summary.json',summary)
    (OUT/'README.md').write_text('# E024 route formal: failed CPU registration v1\n\n'
        'This archive preserves the actual failed registration, its guarded wrapper, stdout/stderr, all source snapshots, the original inventory, and the complete partial preregistration directory. It does not repair or replace any original evidence.\n\n'
        f'The real owner-plan SHA is `{PLAN_SHA}`. The registrar reused its `digest` variable in the pin loop, then saved the final `test_execute_cpu.py` SHA `{WRONG_SHA}` into commands and preparation. The actual owner CPU preparation rejected `route formal plan SHA`. The original guard receipt remains **FAIL**.\n\n'
        'The wrapper reports no GPU execution, service change or service PREPARED action; its backend/process/network/signal guards stayed intact. No v1 cold journal, owner, execution, CPU-prepare output or sealed receipt exists. These are registration-scope receipt/filesystem checks, not a fresh OS probe. Prior CPU tests passing does not make this registration valid.\n\n'
        'A separate v2 registrar may fix this issue; it is outside this immutable failure archive. The compressed archive contains both entire raw trees, including all frozen-source bytes and empty stdout/stderr. `manifest.json` covers every regular tar member; `archive-verification.json` records reopened and safe-extracted per-member SHA/size plus original input rechecks. No model/runtime payload or inference ledger is included.\n')
    entries={label+'/'+str(p.relative_to(trees[label])):dict(source=str(p),**info(p)) for label,paths in original.items() for p in paths}
    for name in ('archive.py','pack-command.json','README.md','summary.json','failed-result.json','original-preparation.json','original-commands.json'):
        entries[name]=dict(source=str(OUT/name),**info(OUT/name))
    write(OUT/'manifest.json',dict(schema='E024-route-formal-registration-failure-manifest-v1',files=entries))
    archive=OUT/'evidence.tar.xz'
    with tarfile.open(archive,'w:xz',dereference=True,preset=1) as tar:
        for name,row in entries.items():tar.add(row['source'],arcname=name,recursive=False,filter=normalized_header)
    with tempfile.TemporaryDirectory(prefix='route-formal-registration-verify-') as temporary:
        extracted=Path(temporary)
        with tarfile.open(archive,'r:xz') as tar:
            members=tar.getmembers()
            need(len(members)==len(entries) and len({m.name for m in members})==len(members) and {m.name for m in members}==set(entries),'exact unique archive members')
            for member in members:
                name=PurePosixPath(member.name)
                need(member.isfile() and not name.is_absolute() and '..' not in name.parts and name.parts,'safe regular tar member only')
                target=extracted.joinpath(*name.parts);target.parent.mkdir(parents=True,exist_ok=True)
                with tar.extractfile(member) as source,target.open('xb') as sink:shutil.copyfileobj(source,sink,1<<20)
                expected={k:entries[member.name][k] for k in ('sha256','bytes')}
                need(info(target)==expected,'safe-extracted member SHA/bytes')
    for name,row in entries.items():need(info(Path(row['source']))=={k:row[k] for k in ('sha256','bytes')},'original bytes changed: '+name)
    for label,root in trees.items():need(listing(root)==original[label],'original membership changed: '+label)
    verification=dict(schema='E024-route-formal-registration-failure-archive-verification-v1',status='PASS',
        original_registration_status='FAIL',archive=info(archive),members=len(entries),raw_bytes=sum(r['bytes'] for r in entries.values()),
        manifest_sha256=info(OUT/'manifest.json')['sha256'],failed_result_sha256=RESULT_SHA,
        all_safe_extracted_member_hashes_verified=True,all_original_files_rechecked=True,all_original_memberships_rechecked=True,
        original_partial_prereg_preserved=True,model_payloads_rehashed=False,runtime_payloads_rehashed=False,
        service_actions=False,gpu_executed=False,performance_qualified=False)
    write(OUT/'archive-verification.json',verification);print(json.dumps(verification))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',action='store_true');args=parser.parse_args()
    if args.run:
        need(not OUT.exists(),'exclusive new archive directory')
        try:run()
        except BaseException as exc:
            if OUT.exists() and not (OUT/'archive-error.json').exists():
                write(OUT/'archive-error.json',dict(status='FAIL',type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()))
            raise
    else:print('DESCRIPTION_ONLY: archives the exact failed registration and original partial preregistration; --run required, no service/device/test actions.')
