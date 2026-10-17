"""Fixed Lithos D512 v2 evidence packer. No work without --run.

After the closed owner and independent audits exist, archive every evidence file,
then extract and hash every member and re-hash every source/tree. No audit,
process, service, runtime import, model payload or weights.pack is executed/read.
A failure leaves the new partial destination for inspection; it is never reused.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile

ROOT=Path('/Users/ryanli/code/minivllm')
ART=ROOT/'.artifacts/m5-ultra-256gb'
NAME='lithos-d512-qualification-v2'
PREREG=ROOT/'bench/m5_ultra_256gb/lithos_d512_retry/prereg-qualification-v2'
PLAN_SHA='e72792b61039414d0c16d9f5f3216619aed62d2d3c23536fe491d409383ab354'
DEFAULT_OUT=ROOT/'docs/experiments/m5-ultra-256gb/E023-lithos-d512-qualification-v2'
TREE_KEYS=('owner','cold-journal','execution','analysis')
PHASES=('lithos-c1','lithos-c8')
FORBIDDEN_SUFFIXES=('.safetensors','.pack','.dylib','.so','.bin')


def need(ok,message):
    if not ok:raise ValueError(message)

def read(path):return json.loads(Path(path).read_bytes())

def stream_hash(f):
    h=hashlib.sha256();size=0
    while chunk:=f.read(1<<20):h.update(chunk);size+=len(chunk)
    return h.hexdigest(),size

def info(path):
    path=Path(path);need(path.is_file() and not path.is_symlink(),'regular input required: '+str(path))
    sig=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)
    before=path.stat()
    with path.open('rb') as f:
        need(sig(os.fstat(f.fileno()))==sig(before),'input replaced before open: '+str(path))
        digest,size=stream_hash(f);need(sig(os.fstat(f.fileno()))==sig(before),'open input changed: '+str(path))
    after=path.stat();need(sig(before)==sig(after) and size==after.st_size,'input changed: '+str(path))
    return dict(sha256=digest,bytes=size,device=after.st_dev,inode=after.st_ino,mtime_ns=after.st_mtime_ns)

def write(path,value):
    with Path(path).open('xb') as f:
        f.write((json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode());f.flush();os.fsync(f.fileno())

def files(root):
    need(root.is_dir() and not root.is_symlink(),'evidence directory absent/symlink: '+str(root))
    result=[]
    for path in sorted(root.rglob('*')):
        need(not path.is_symlink() and (path.is_dir() or path.is_file()),'link/special evidence file: '+str(path))
        if path.is_file():
            need(not path.name.endswith(FORBIDDEN_SUFFIXES),'unexpected payload inside evidence tree; inspect instead of silently skipping: '+str(path))
            result.append(path)
    return result

def fixed_terminal(trees,args):
    owner=trees['raw/owner'];cold=trees['raw/cold-journal'];execution=trees['raw/execution']
    plan=read(PREREG/'owner-plan.json');receipt=read(PREREG/'receipt.json')
    need(info(PREREG/'owner-plan.json')['sha256']==info(owner/'owner-plan.json')['sha256']==PLAN_SHA
         ==receipt['owner_plan_sha256'],'fixed registered plan')
    complete=read(owner/'complete.json');attempt=read(owner/'attempt.json')
    need(complete['schema']=='E023-lithos-D512-owner-result-v1' and complete['status'] in ('PASS','FAIL')
         and complete['owner_plan_sha256']==attempt['owner_plan_sha256']==PLAN_SHA
         and complete['owner_pid']==attempt['owner_pid'] and complete['planned_trials']==list(PHASES)
         and complete['all_children_reaped'] is True and complete['performance_qualified'] is False,
         'closed actual owner required; no partial/live owner archive')
    journal=read(cold/'journal.json');final=read(execution/'final.json')
    need(journal['state']=='RESTORED' and final['journal_state']=='RESTORED' and final['recovery_returncode']==0
         and journal['owner_plan']['sha256']==PLAN_SHA,'completed restoration required')
    audits={name:read(path) for name,path in args.audit_paths.items()}
    restore=audits['restoration-audit.json']
    need(restore['schema']=='cold-service-restoration-audit-v1' and restore['status']=='PASS'
         and restore['qualified'] is True and restore['numerical_qualified'] is False
         and restore['performance_qualified'] is False and restore['journal_sha256']==info(cold/'journal.json')['sha256']
         and restore['owner_plan_sha256']==PLAN_SHA and restore['controller_sha256']==plan['cold_restart']['controller']['sha256']
         and restore['service_restore']['replacement']==journal['replacement'],'independent restoration receipt binding')
    need(audits['independent-qualification.json']['schema']=='E023-lithos-D512-qualification-audit-v1'
         and audits['resource-summary.json']['schema']=='Lithos-D512-raw-resource-audit-v1','separate independent audit types')
    for name,audit in audits.items():
        need(audit['status'] in ('PASS','FAIL','INCOMPLETE') and audit['performance_qualified'] is False,'audit terminal scope')
        if name!='restoration-audit.json' and audit['status']=='PASS':
            need(audit['owner_plan_sha256']==PLAN_SHA and audit['owner_complete_sha256']==info(owner/'complete.json')['sha256']
                 and Path(audit['owner_directory']).resolve()==owner,'closed audit source binding')
    qualification=audits['independent-qualification.json']
    if qualification['status']=='PASS':
        need(qualification['qualified'] is True and qualification['route_count_text_qualified'] is True
             and qualification['full_state_qualified'] is False and qualification['requests']==24
             and [(r['concurrency'],r['requests']) for r in qualification['phases']]==[(1,8),(8,16)],
             'complete observed route/count/text qualification receipt fields')
    if complete['status']=='PASS':
        for phase in PHASES:
            trial=owner/'phases'/phase/'trial'
            for relative in ('group.json','probe/observer.jsonl','server.stdout','server.stderr',
                             'raw-join.json','generation-log-check.json','complete.json','release.json','launch.json','spec.json'):
                need((trial/relative).is_file(),'successful trial raw evidence missing: '+str(trial/relative))
    pins={**plan['execution']['pins'],**receipt['source_pins']}
    audit_sources={'independent-qualification.json':'audit_qualification.py','resource-summary.json':'audit_resources.py','restoration-audit.json':'audit_restore.py'}
    for name,source in audit_sources.items():
        path=PREREG.parent/source
        need(info(path)['sha256']==pins[str(path)],'registered independent auditor source pin')
        # Some frozen FAIL envelopes omit audit_source_sha256. Preserve that
        # omission; archived registered source is not a fabricated receipt claim.
        if 'audit_source_sha256' in audits[name]:
            need(audits[name]['audit_source_sha256']==pins[str(path)],'reported auditor source pin')
        else:need(audits[name]['status']!='PASS','passing audit must identify its source')
    summary=dict(schema='Lithos-D512-qualification-archive-summary-v2',owner_status=complete['status'],
        plan_sha256=PLAN_SHA,owner_pid=complete['owner_pid'],restored_identity=journal['replacement'],
        audit_statuses={name:a['status'] for name,a in audits.items()},
        audit_source_claim_present={name:'audit_source_sha256' in a for name,a in audits.items()},
        observed_phases=[row['phase'] for row in read(owner/'children.json')['children']],
        route_count_text_qualified=audits['independent-qualification.json'].get('route_count_text_qualified') is True,
        full_state_qualified=False,performance_qualified=False,pareto_eligible=False,
        scope='Observed qualification-cycle archive only; C8 means concurrent HTTP callers with official FIFO GPU execution. No formal timing point or new numerical/full-state claim.')
    return plan,pins,summary

def archive(args):
    out=args.out.resolve();need(not out.exists() and out.parent.is_dir(),'fresh destination in existing parent required')
    trees={'raw/'+key:ART/(NAME+'-'+key) for key in TREE_KEYS};trees['preregistration']=PREREG
    need(all(path.resolve().parent==trees['raw/analysis'] for path in args.audit_paths.values()),'audits must be direct files in fixed analysis directory')
    plan,pins,summary=fixed_terminal(trees,args)
    inputs={};snapshots={}
    for prefix,root in trees.items():
        paths=files(root);snapshots[prefix]=[p.relative_to(root).as_posix() for p in paths]
        inputs.update({prefix+'/'+p.relative_to(root).as_posix():p for p in paths})
    for path,digest in pins.items():
        p=Path(path);need(p.is_relative_to(ROOT) and info(p)['sha256']==digest,'registered source changed: '+path)
        inputs['registered-source/'+p.relative_to(ROOT).as_posix()]=p
    # Full official Lithos source is small and explicitly pinned; runtime binaries,
    # target weights and weights.pack remain external, described by verification.json.
    for path,row in plan['verification']['groups']['lithos_source'].items():
        p=Path(path);need(info(p)['sha256']==row['sha256'],'official source changed: '+path)
        inputs['official-source/'+p.relative_to('/Users/ryanli/code/lithos-metal').as_posix()]=p
    pack=[(Path(p),r) for p,r in plan['verification']['groups']['pack'].items() if Path(p).name=='manifest.json']
    need(len(pack)==1 and info(pack[0][0])['sha256']==pack[0][1]['sha256'],'pack metadata pin')
    inputs['external-metadata/target-pack-manifest.json']=pack[0][0]
    inputs['archive-source/archive.py']=Path(__file__).resolve()
    out.mkdir()
    sidecars={'restored-journal.json':trees['raw/cold-journal']/'journal.json',**args.audit_paths}
    for name,path in sidecars.items():
        with (out/name).open('xb') as f:f.write(path.read_bytes())
        need((out/name).read_bytes()==path.read_bytes(),'exact sidecar copy')
        inputs[name]=out/name
    write(out/'summary.json',summary);inputs['summary.json']=out/'summary.json'
    (out/'ARCHIVE.md').write_text('''# Lithos D512 C1/C8 observed qualification evidence

This archive retains the fixed official target-only Lithos C1/C8 qualification
cycle, all original HTTP NDJSON/chunk timestamps, request bodies, observer logs,
Generation tokens, server logs, resource receipts and recovery evidence.
Read summary.json and the independent audits for the actual outcome, including
any rejection; packaging PASS means byte preservation only.

Observers are ON. C8 uses concurrent HTTP callers and official FIFO GPU execution.
There is no formal timing/Pareto point or full-state numerical qualification.
The restored service has its persisted configuration and visible MTP/resident PLE
contract; exact hidden runtime state and warm KV restoration are not claimed.

All regular files in four fixed artifact trees and preregistration are included.
CPU preparation records are retained in preregistration; there is no separate
cpu-prepare artifact tree for this cycle.
Registered harness/auditor sources and all 221 pinned official Lithos source files
are retained. Target model/pack payloads, Python/runtime binaries are not copied or
rehashed by this packer: complete supplied identities and producer pre/post checks
remain in verification/owner receipts. The target pack manifest is copied exactly.

manifest.json maps each tar member to original path, SHA256, size and stat.
archive-verification.json proves full extraction/member SHA, original rehash,
unchanged tree membership and exact restoration sidecars. These two receipts sit
outside the tar to avoid recursive hashing. No evidence file is overwritten.
''')
    inputs['ARCHIVE.md']=out/'ARCHIVE.md'
    records={name:dict(source=str(p),**info(p)) for name,p in sorted(inputs.items())}
    manifest=dict(schema='Lithos-D512-evidence-manifest-v2',members=records,included_members=len(records),
        all_original_bytes=sum(r['bytes'] for r in records.values()),tree_members=snapshots,
        excluded_external_payloads='No model/pack weights or runtime binaries copied; explicit original path/hash/size inventories remain in complete verification.json and owner pre/post receipts.')
    write(out/'manifest.json',manifest)
    package=out/'evidence.tar.xz'
    # Parent was created exclusively above; xb prevents accidental archive replacement.
    with package.open('xb') as f,tarfile.open(fileobj=f,mode='w:xz',dereference=True,preset=6) as tar:
        for name,p in sorted(inputs.items()):tar.add(p,arcname=name,recursive=False)
    with tempfile.TemporaryDirectory(prefix='verify-',dir=out) as temp:
        seen=set()
        with tarfile.open(package,'r:xz') as tar:
            for member in tar:
                name=member.name;rel=PurePosixPath(name)
                need(member.isfile() and not rel.is_absolute() and '..' not in rel.parts and name not in seen and name in records,'unsafe/duplicate/unexpected tar member')
                expected=records[name];need(member.size==expected['bytes'],'tar member size')
                dest=Path(temp).joinpath(*rel.parts);dest.parent.mkdir(parents=True,exist_ok=True)
                with tar.extractfile(member) as source,dest.open('xb') as output:shutil.copyfileobj(source,output,1<<20)
                got=info(dest);need((got['sha256'],got['bytes'])==(expected['sha256'],expected['bytes']),'extracted member SHA: '+name);seen.add(name)
        need(seen==set(records),'omitted tar member')
    for name,p in inputs.items():need(info(p)=={k:v for k,v in records[name].items() if k!='source'},'original changed: '+str(p))
    for prefix,root in trees.items():need([p.relative_to(root).as_posix() for p in files(root)]==snapshots[prefix],'evidence tree changed')
    for name,path in sidecars.items():need((out/name).read_bytes()==path.read_bytes(),'sidecar changed')
    package_info=info(package)
    result=dict(schema='Lithos-D512-archive-verification-v2',status='PASS',archive_sha256=package_info['sha256'],
        archive_bytes=package_info['bytes'],manifest_sha256=info(out/'manifest.json')['sha256'],included_members=len(records),
        all_archived_member_hashes_verified=True,all_original_files_rehashed=True,all_tree_members_rechecked=True,
        exact_restoration_sidecars=True,model_pack_payloads_rehashed=False,performance_qualified=False,
        compressed_file_under_100_MiB=package_info['bytes']<(100<<20))
    write(out/'archive-verification.json',result);return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',action='store_true');p.add_argument('--out',type=Path,default=DEFAULT_OUT)
    analysis=ART/(NAME+'-analysis')
    for flag,name in [('qualification-audit','independent-qualification.json'),('resource-audit','resource-summary.json'),('restoration-audit','restoration-audit.json')]:
        p.add_argument('--'+flag,type=Path,default=analysis/name)
    args=p.parse_args();args.audit_paths={'independent-qualification.json':args.qualification_audit.resolve(),
        'resource-summary.json':args.resource_audit.resolve(),'restoration-audit.json':args.restoration_audit.resolve()}
    if not args.run:
        print(json.dumps(dict(action='PROPOSAL_ONLY_NO_INPUT_SCAN',trees={k:str(ART/(NAME+'-'+k)) for k in TREE_KEYS},
            preregistration=str(PREREG),out=str(args.out),audits={k:str(v) for k,v in args.audit_paths.items()},
            prerequisite='Closed owner, RESTORED journal, completed audits; run only after root ends all writers.'),indent=2));return 0
    print(json.dumps(archive(args),indent=2));return 0

if __name__=='__main__':raise SystemExit(main())
