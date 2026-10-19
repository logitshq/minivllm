"""Retention only, after driver exit and saved audits; default describes, never starts services/audits."""
import argparse, hashlib, importlib.util, json, os, tarfile
from pathlib import Path
R=Path('/Users/ryanli/code/minivllm'); A=R/'.artifacts/m5-ultra-256gb'; N=R/'bench/m5_ultra_256gb/c1_profile_cycle_v1'
PLAN_SHA='f70b5cf3015f3c382696ce24e4f2891e66e56916c68b0617ec81f3ab31911343'
OUT=R/'docs/experiments/m5-ultra-256gb/E035-c1-pipeline-profile'
HELPER=R/'docs/experiments/m5-ultra-256gb/E034-glm-flash-load-v3/archive.py'

def run(execution):
    spec=importlib.util.spec_from_file_location('e034_retention_helpers',HELPER); h=importlib.util.module_from_spec(spec); spec.loader.exec_module(h)
    need,read,pin,write=h.need,h.read,h.pin,h.write
    reg=N/'prereg-v1'; plan=read(reg/'owner-plan.json'); receipt=read(reg/'receipt.json'); commands=read(reg/'commands.json')
    owner=Path(plan['outputs']['owner']); journal=Path(plan['cold_restart']['journal_path'])
    analysis=Path(commands['closed_audit'][-1]); restore=Path(commands['restoration_audit'][-1])
    need(analysis==N/'analysis-v1/independent-diagnostic.json' and restore==analysis.with_name('restoration-audit.json'),'actual registered audit destinations')
    final=read(execution/'final.json'); data=read(journal/'journal.json'); closed=read(analysis); restoration=read(restore)
    need(pin(reg/'owner-plan.json')['sha256']==PLAN_SHA==receipt['owner_plan_sha256']==final['owner_plan_sha256'],'actual plan binding')
    need(pin(reg/'receipt.json')==pin(execution/'execution-receipt.json'),'actual driver sealed receipt copy')
    need(data['schema']=='C1-profile-cold-journal-v1' and final['journal_state']==data['state'] and data['trials']==['off','runner','full'] and data['owner_plan']==dict(path=str(reg/'owner-plan.json'),sha256=PLAN_SHA),'actual three-mode journal')
    need(closed['schema']=='C1-profile-closed-audit-v1' and restoration['schema']=='cold-service-restoration-audit-v1' and all(x['status'] in ('PASS','FAIL','INCOMPLETE') for x in (closed,restoration)),'saved actual audits required')
    need(not OUT.exists(),'fresh E035 required; preserve earlier attempts')
    roots=dict(owner=owner,journal=journal,execution=execution,registration=reg,analysis=analysis.parent,
        cpu_v1=A/'c1-profile-cycle-cpu-v1',cpu_v2=A/'c1-profile-cycle-cpu-v2')
    files={}; counts={}; checks=[]; excluded=[]
    for label,root in roots.items():
        need(root.is_absolute() and root==root.resolve() and root.is_dir() and not root.is_symlink(),'actual regular evidence root: '+str(root))
        pending=[root]; count=0
        while pending:
            with os.scandir(pending.pop()) as entries: entries=sorted(entries,key=lambda e:e.name)
            for entry in entries:
                p=Path(entry.path); need(not entry.is_symlink(),'evidence link is not followed: '+str(p))
                if entry.is_dir(follow_symlinks=False): pending.append(p); continue
                need(entry.is_file(follow_symlinks=False) and p.suffix not in h.BINARY_SUFFIXES,'unexpected nonregular/model/binary evidence: '+str(p))
                files[label+'/'+p.relative_to(root).as_posix()]=p; count+=1
        counts[label]=count
    for source,digest in sorted(receipt['source_pins'].items()):
        p=Path(source)
        if p.suffix not in ('.py','.json'):
            excluded.append(dict(path=source,registered_sha256=digest,reason='Runtime/binary reference only; no payload hash/copy.')); continue
        if p.is_symlink() or not p.is_file(): checks.append(dict(path=source,status='MISSING_OR_SYMLINK')); continue
        actual=pin(p); checks.append(dict(path=source,expected_sha256=digest,actual=actual,status='PASS' if actual['sha256']==digest else 'FAIL'))
        files['registered-sources/'+p.as_posix().lstrip('/')]=p
    complete=read(owner/'complete.json') if (owner/'complete.json').is_file() else {}
    execution_ok=all(final.get(k)==0 for k in ('stop_returncode','owner_returncode','recovery_returncode')) and data['state']=='RESTORED'
    statuses=[closed['status'],restoration['status'],complete.get('status','INCOMPLETE')]
    if not execution_ok or any(c['status']!='PASS' for c in checks): statuses.append('FAIL')
    scope='Three C1 diagnostic modes off/runner/full, eight warm plus eight diagnostic requests each if complete. Shared admission/DONE/drain overhead. Python spans overlap and are not exclusive GPU time. No Pareto point, speedup qualification or new numerical qualification; actual failed/incomplete attempts remain unchanged. No model weights copied or rehashed.'
    summary=dict(schema='C1-profile-archive-v1',status=h.result_status(statuses),owner_plan_sha256=PLAN_SHA,
        registration_receipt_sha256=pin(reg/'receipt.json')['sha256'],journal_state=data['state'],owner_status=complete.get('status','MISSING'),
        audit_statuses=dict(restoration=restoration['status'],closed_diagnostic=closed['status']),
        execution_returncodes={k:final.get(k) for k in ('stop_returncode','owner_returncode','recovery_returncode')},
        diagnostic_complete=closed.get('diagnostic_complete',False),modes=closed.get('modes',[]),resources=closed.get('resources'),
        restoration_qualified=restoration.get('qualified',False),numerical_qualified=False,performance_qualified=False,
        errors=dict(owner=complete.get('errors'),restoration=restoration.get('error'),closed_diagnostic=closed.get('error')),evidence_roots=counts,scope=scope)
    OUT.mkdir(); h.copy_exact(__file__,OUT/'archive.py'); h.copy_exact(HELPER,OUT/'e034-retention-helpers.py')
    for source,name in ((analysis,'independent-diagnostic.json'),(restore,'restoration-audit.json'),(journal/'journal.json','journal.json'),(execution/'final.json','execution-final.json'),(reg/'owner-plan.json','owner-plan.json'),(reg/'receipt.json','registration-receipt.json')): h.copy_exact(source,OUT/name)
    if data['state']=='RESTORED': h.copy_exact(journal/'journal.json',OUT/'restored-journal.json')
    write(OUT/'summary.json',summary); write(OUT/'registered-source-checks.json',dict(checks=checks,excluded=excluded))
    (OUT/'README.md').write_text('# E035: C1 pipeline diagnostic\n\nStatus: **'+summary['status']+'**. Restoration: **'+restoration['status']+'**; independent diagnostic: **'+closed['status']+'**.\n\n'+scope+'\n\nThe manifest retains owner, cold journal, execution, registration, analysis, both actual CPU attempts and registered text sources. Every archive member is checked against both its original source bytes and the saved manifest. No symlinks are followed.\n')
    for p in OUT.iterdir(): files['archive-metadata/'+p.name]=p
    manifest={name:dict(source=str(p),**pin(p)) for name,p in sorted(files.items())}; write(OUT/'manifest.json',manifest); files['manifest.json']=OUT/'manifest.json'
    archive=OUT/'evidence.tar.xz'
    with tarfile.open(archive,'x:xz',preset=1) as tar:
        for name,p in sorted(files.items()): tar.add(p,arcname=name,recursive=False)
    seen=set(); total=0
    with tarfile.open(archive,'r|xz') as tar:
        for member in tar:
            need(member.isfile() and member.name in files and member.name not in seen,'unexpected archive member'); seen.add(member.name)
            stream=tar.extractfile(member); digest=hashlib.sha256(); size=0
            for block in iter(lambda:stream.read(8<<20),b''): digest.update(block); size+=len(block)
            actual=dict(sha256=digest.hexdigest(),bytes=size); need(actual==pin(files[member.name]),'archive/source mismatch')
            if member.name!='manifest.json': need(all(actual[k]==manifest[member.name][k] for k in actual),'source changed since manifest')
            total+=size
    need(seen==set(files),'missing archive members')
    verification=dict(status='PASS',members=len(seen),raw_bytes=total,archive=pin(archive),scope='Byte retention verification only; no qualification promotion.')
    write(OUT/'archive-verification.json',verification)
    print(json.dumps(dict(directory=str(OUT),summary_status=summary['status'],verification=verification),indent=2)); return 0

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--run',action='store_true'); p.add_argument('--execution',type=Path); a=p.parse_args()
    if not a.run:
        if a.execution: p.error('--execution requires --run')
        print('DESCRIPTION ONLY: after driver exit, restoration and saved audits, retain E035 evidence without starting audits/services.'); return 0
    if not a.execution or not a.execution.is_absolute() or a.execution!=a.execution.resolve(): p.error('explicit canonical absolute --execution required')
    return run(a.execution)
if __name__=='__main__': raise SystemExit(main())
