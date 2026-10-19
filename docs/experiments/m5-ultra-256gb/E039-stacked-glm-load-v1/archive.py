"""Explicit post-execution retention only; never starts an audit, model or service.

Default describes. --run requires the actual driver final receipt and both saved
audits. FAIL/INCOMPLETE statuses remain FAIL/INCOMPLETE. This is not mini-engine,
full-state, HTTP, quality, performance or physical-safety qualification.
"""
import argparse
import hashlib
import json
import os
import stat
from pathlib import Path
import tarfile

R=Path('/Users/ryanli/code/minivllm')
A=R/'.artifacts/m5-ultra-256gb'
B=R/'bench/m5_ultra_256gb'
REG=A/'glm-stacked-registration-v1'
OWNER=A/'glm-stacked-owner-v1'
JOURNAL=A/'glm-stacked-cold-journal-v1'
EXEC=A/'glm-stacked-execution-v1'
ANALYSIS=A/'glm-stacked-analysis-v1'
OUT=R/'docs/experiments/m5-ultra-256gb/E039-stacked-glm-load-v1'
FAILED_ARCHIVE=A/'glm-stacked-archive-attempt-v1'
KNOWN_MODEL_ALIAS=OWNER/'phases/glm-official-load/probe/models/GLM-5.3-Flash-4bit-stacked-v1'
PLAN_SHA='ca2a2e280c29aea239c640d41a5b1e9e6af71138cabfac6f02e22d887967affa'
RECEIPT_SHA='9d3cd2212212d6000d7f87e5a07947108bf0053dc26274e6387aacd11a1450aa'
BINARY_SUFFIXES={'.safetensors','.gguf','.npz','.npy','.dylib','.so','.bin','.metallib','.a','.o'}

def need(ok,message):
    if not ok:raise ValueError(message)
def read(p):return json.loads(Path(p).read_bytes())
def write(p,value):
    with Path(p).open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
def pin(p):
    p=Path(p);need(p.is_file() and not p.is_symlink(),'actual regular evidence file: '+str(p))
    before=p.stat();h=hashlib.sha256()
    with p.open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):h.update(block)
    after=p.stat()
    need((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)==
         (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns),'file changed: '+str(p))
    return dict(sha256=h.hexdigest(),bytes=before.st_size)
def copy_exact(source,target):
    expected=pin(source)
    with Path(source).open('rb') as src,Path(target).open('xb') as dst:
        for chunk in iter(lambda:src.read(8<<20),b''):dst.write(chunk)
    need(pin(target)==expected==pin(source),'copy changed: '+str(source))
def result_status(values):
    if any(v=='FAIL' for v in values):return 'FAIL'
    if any(v!='PASS' for v in values):return 'INCOMPLETE'
    return 'PASS'

def alias_metadata(path,expected_target):
    need(path==KNOWN_MODEL_ALIAS,'unexpected evidence symlink: '+str(path))
    value=path.lstat();need(stat.S_ISLNK(value.st_mode),'registered model alias is not a symlink')
    target=os.readlink(path)
    need(target==expected_target,'registered model alias literal target changed')
    return dict(path=str(path),literal_target=target,target_utf8_sha256=hashlib.sha256(target.encode('utf-8')).hexdigest(),
        lstat={key:getattr(value,key) for key in ('st_dev','st_ino','st_mode','st_uid','st_gid','st_nlink','st_size','st_mtime_ns','st_ctime_ns')},
        target_followed=False,payload_copied=False,
        scope='The only accepted symlink is the original load_probe model-directory alias; target text and lstat only, no target stat/read/walk.')

def run():
    # File-only readiness: root waits for the driver process, then runs both audits.
    # Even an unsuccessful restoration is retained; no receipt is promoted here.
    final=read(EXEC/'final.json');journal=read(JOURNAL/'journal.json')
    restoration=read(ANALYSIS/'restoration-audit.json');closed=read(ANALYSIS/'closed-audit.json')
    need(final['owner_plan_sha256']==PLAN_SHA and final['journal_state']==journal['state'],
         'actual matching driver final/journal/plan required')
    need(pin(REG/'owner-plan.json')['sha256']==PLAN_SHA and pin(REG/'receipt.json')['sha256']==RECEIPT_SHA,
         'original actual registration bytes changed')
    need(journal['schema']=='GLM-cold-load-journal-v3' and
         journal['owner_plan']==dict(path=str(REG/'owner-plan.json'),sha256=PLAN_SHA),
         'actual v3 journal owner binding')
    need(restoration['schema']=='cold-service-restoration-audit-v1' and
         closed['schema']=='GLM-closed-load-audit-v3' and
         all(x['status'] in ('PASS','FAIL','INCOMPLETE') for x in (restoration,closed)),
         'actual saved audit schemas/statuses')
    need(not OUT.exists(),'fresh E039 archive required; preserve any earlier attempt')
    plan=read(REG/'owner-plan.json');receipt=read(REG/'receipt.json')
    commands=read(REG/'commands.json')
    need(commands['restoration_audit'][-1]==str(ANALYSIS/'restoration-audit.json') and
         commands['closed_audit'][-1]==str(ANALYSIS/'closed-audit.json'),
         'saved audit destinations match actual inputs')
    owner=read(OWNER/'complete.json') if (OWNER/'complete.json').is_file() else {}
    source_checks=[];source_files={};excluded=[]
    # Explicit registered paths only. No model-cache walk or payload hash.
    for source,expected in sorted(receipt['source_pins'].items()):
        path=Path(source)
        if path.suffix not in ('.py','.json'):
            excluded.append(dict(path=source,registered_sha256=expected,
                reason='Binary/interpreter reference retained in original manifests; payload not copied or rehashed.'))
            continue
        if not path.is_file() or path.is_symlink():
            source_checks.append(dict(path=source,expected_sha256=expected,status='MISSING_OR_SYMLINK'));continue
        actual=pin(path);source_checks.append(dict(path=source,expected_sha256=expected,
            actual=actual,status='PASS' if actual['sha256']==expected else 'FAIL'))
        source_files['registered-sources/'+path.as_posix().lstrip('/')]=path
    roots=dict(owner=OWNER,**{'cold-journal':JOURNAL},execution=EXEC,registration=REG,analysis=ANALYSIS,inputs=A/'glm-stacked-inputs-v1')
    probe_ref=plan['probe_plan'];probe_plan_path=Path(probe_ref['path'])
    need(pin(probe_plan_path)['sha256']==probe_ref['sha256'],'registered probe plan changed')
    expected_target=read(probe_plan_path)['snapshot']
    need(type(expected_target) is str and Path(expected_target).is_absolute(),'explicit absolute snapshot target')
    files={};root_inventory={};aliases=[]
    for prefix,root in roots.items():
        if not root.exists():root_inventory[prefix]=dict(exists=False,files=0);continue
        need(not root.is_symlink() and root.is_dir(),'real artifact root required')
        count=0;pending=[root]
        # Check symlink before file/directory classification. No target stat/walk.
        while pending:
            directory=pending.pop()
            with os.scandir(directory) as entries:entries=sorted(entries,key=lambda entry:entry.name)
            for entry in entries:
                p=Path(entry.path)
                if entry.is_symlink():
                    aliases.append(alias_metadata(p,expected_target));continue
                if entry.is_dir(follow_symlinks=False):pending.append(p);continue
                need(entry.is_file(follow_symlinks=False),'unexpected nonregular evidence entry: '+str(p))
                need(p.suffix not in BINARY_SUFFIXES,'unexpected binary/model payload in evidence root: '+str(p))
                files[prefix+'/'+p.relative_to(root).as_posix()]=p;count+=1
        root_inventory[prefix]=dict(exists=True,files=count)
    need(len(aliases)<=1,'duplicate model alias evidence')
    files.update(source_files)
    files['audit-runner/run-glm-stacked-audits-v1.py']=Path('/tmp/run-glm-stacked-audits-v1.py')
    prior=R/'docs/experiments/m5-ultra-256gb/E036-glm-expert-repack-v1'
    for name in ('producer-complete.json','header-verification.json','archive-verification.json'):
        files['e036-linked-metadata/'+name]=prior/name
    OUT.mkdir();copy_exact(__file__,OUT/'archive.py')
    write(OUT/'model-alias-evidence.json',dict(aliases=aliases,scope='Only registered model-directory alias retained as literal target/lstat; no link is followed.'))
    for source,name in ((ANALYSIS/'restoration-audit.json','restoration-audit.json'),
                        (ANALYSIS/'closed-audit.json','closed-audit.json'),
                        (JOURNAL/'journal.json','journal.json'),(EXEC/'final.json','execution-final.json'),
                        (REG/'owner-plan.json','owner-plan.json'),(REG/'receipt.json','registration-receipt.json')):
        copy_exact(source,OUT/name)
    if journal['state']=='RESTORED':copy_exact(JOURNAL/'journal.json',OUT/'restored-journal.json')
    probe_folder=OWNER/'phases/glm-official-load/probe'
    partial=dict(scope='Unqualified producer/raw observations, retained even when diagnostic fails.')
    if (OWNER/'cold-physical.json').is_file():partial['cold_admission']=read(OWNER/'cold-physical.json').get('admission')
    physical=sorted(OWNER.glob('physical-*/sample.json'),key=lambda p:int(p.parent.name.split('-')[-1]))
    if physical:
        last=read(physical[-1]);partial['last_physical_sample']=dict(path=str(physical[-1]),parsed=last.get('parsed'),started_at=last.get('started_at'),ended_at=last.get('ended_at'))
        baseline=partial.get('cold_admission',{});values=last.get('parsed',{})
        if all(k in baseline for k in ('baseline_swap_bytes','baseline_swapouts')) and all(k in values for k in ('swap_bytes','swapouts')):
            partial['swap_deltas']=dict(bytes=values['swap_bytes']-baseline['baseline_swap_bytes'],swapouts=values['swapouts']-baseline['baseline_swapouts'],scope='System-wide counters; swap bytes derived by original parser from reported sysctl units. No process attribution.')
    if (OWNER/'monitor/monitor.json').is_file():
        monitor=read(OWNER/'monitor/monitor.json');partial['monitor']={k:monitor.get(k) for k in ('status','errors','read_commands_settled','unjoined_workers','unresolved_read_commands')}
    if (probe_folder/'complete.json').is_file():
        terminal=read(probe_folder/'complete.json');partial['probe']={k:terminal.get(k) for k in ('status','completed_requests','error','shutdown')}
    if (probe_folder/'events.jsonl').is_file():
        partial['observed_memory_boundaries']=[]
        for line in (probe_folder/'events.jsonl').read_text().splitlines():
            event=json.loads(line)
            if event.get('kind')=='memory':partial['observed_memory_boundaries'].append(event)
    partial['loaded_engine_receipt_present']=(probe_folder/'loaded-engine.json').is_file()
    partial['request_receipts_present']=sum((probe_folder/f'request-{i}.json').is_file() for i in range(4))
    events=[json.loads(line) for line in (probe_folder/'events.jsonl').read_text().splitlines()] if (probe_folder/'events.jsonl').is_file() else []
    partial['actual_request_count']=partial['request_receipts_present']
    partial['batch_before_events']=sum(e.get('kind')=='batch_before' for e in events)
    partial['official_load_before_events']=sum(e.get('kind')=='memory' and e.get('phase')=='official_vlm_load:before' for e in events)
    partial['official_load_after_events']=sum(e.get('kind')=='memory' and e.get('phase')=='official_vlm_load:after' for e in events)
    partial['pool_load_returned_events']=sum(e.get('kind')=='memory' and e.get('phase')=='after_pool_load' for e in events)
    partial['probe_terminal_present']=(probe_folder/'complete.json').is_file()
    partial['cooperative_shutdown_receipt_present']=(probe_folder/'shutdown.json').is_file()
    partial['child_returncode']=owner.get('child_returncode')
    partial['scope_note']='No loader-return, await-success or request completion is inferred from restoration or shutdown-start events.'
    write(OUT/'partial-observations.json',partial)
    statuses=[restoration['status'],closed['status'],owner.get('status','INCOMPLETE')]
    execution_ok=(final.get('stop_returncode')==0 and final.get('owner_returncode')==0 and
                  final.get('recovery_returncode')==0 and journal['state']=='RESTORED')
    if not execution_ok:statuses.append('FAIL')
    if any(x['status']!='PASS' for x in source_checks):statuses.append('FAIL')
    scope=('Original official oMLX EnginePool ordinary GLM glm5_next load and four sequential C1 plain '
           'completions capped 8/32/32/32, normal EOS, repeated greedy token receipt checks and drained '
           'queues; actual outcome is given by the unchanged audits. Raw observed physical resources '
           'and two-lane monitor receipts retained. No mini-engine, full-state/reference, independent '
           'text decode, HTTP, chat-quality, performance or physical-safety qualification. No weight '
           'payload was copied or rehashed by this archive.')
    summary=dict(schema='GLM-stacked-closed-load-archive-v1',status=result_status(statuses),
        owner_plan_sha256=PLAN_SHA,registration_receipt_sha256=RECEIPT_SHA,
        journal_state=journal['state'],owner_status=owner.get('status','MISSING'),
        audit_statuses=dict(restoration=restoration['status'],closed_diagnostic=closed['status']),
        execution_returncodes={k:final.get(k) for k in ('stop_returncode','owner_returncode','recovery_returncode')},
        observed_diagnostic_complete=closed.get('observed_diagnostic_complete',False),
        requests=closed.get('requests'),resources=closed.get('resources'),
        restoration_qualified=restoration.get('qualified',False),
        full_model_correctness_qualified=False,chat_quality_qualified=False,
        performance_qualified=False,physical_resource_safety_qualified=False,
        errors=dict(owner=owner.get('errors'),restoration=restoration.get('error'),closed_diagnostic=closed.get('error')),
        evidence_roots=root_inventory,scope=scope)
    write(OUT/'summary.json',summary)
    write(OUT/'registered-source-checks.json',dict(checks=source_checks,excluded=excluded,
        scope='Read-only current text-source equality to original registration; originals remain in retained owner and registration evidence.'))
    (OUT/'README.md').write_text('''# E039: stacked GLM official load — failed diagnostic, restored service

Actual driver exit was 1: stop 0, owner 1, recovery 0. The journal is RESTORED.
Independent restoration audit PASS (exit 0); closed diagnostic FAIL (exit 1,
load diagnostic did not complete). The replacement is PID 52818, PPID 1,
Mon Oct 19 09:22:08 2026, omlx-server. Restoration does not qualify the GLM load.
The original execution-final.json is copied unchanged, including its original
observed_diagnostic_complete field.

The actual E036 58-shard stacked checkpoint entered the official oMLX loader.
At physical sample 0018 the unchanged zero-swap-growth gate failed: reported
swap bytes increased from 1,978,925,056 to 1,980,172,861 (+1,247,805 bytes,
about 1.19 MiB), and cumulative Swapouts increased 782,442 to 782,518 (+76).
These are whole-system counters, not process attribution. Child footprint was
148,261,492,112 bytes and system footprint 171,148,001,280 bytes, below the
220/240 GiB ceilings; pressure was 1. The observed failure is the swap gate,
not evidence of an allocation OOM. No memory gate was relaxed.

The original plan had four plain requests capped 8/32/32/32. Actual completed
requests: **0**; token batch events: **0**. There is one official_vlm_load:before,
no official_vlm_load:after or after_pool_load, no loaded-engine receipt, no
probe complete.json and no cooperative shutdown receipt. The shutdown-start
event after cancellation does not prove loader return or successful owner await.
The owned child exited -9 and the complete direct-child/read-command reap chain
was retained. Monitor read commands settled; the diagnostic remains FAIL.

The E036 raw-byte proof and independent header proof remain valid provenance;
this attempt establishes neither loaded-model memory fit nor inference
correctness. It contains no mini-engine GLM run, speed, HTTP result or Pareto
point. The original official loader, remaining MTP sanitize and gate/up fusion
were not replaced by another engine or offload fallback.

The archive retains owner, journal, exact execution and registration, both
saved audit commands/outputs/exit codes, raw resource observations, input
inventory, explicit registered text sources, and small linked E036 proofs.
The registered model-directory alias is retained only as literal target/lstat,
never followed. No weight payload is copied or rehashed. Every tar member and
source is reread and SHA-verified; manifest.json maps each member to its source.
There was no failed archive attempt before this archive.

See summary.json for unchanged FAIL/PASS statuses, partial-observations.json for
actual request/load boundaries and raw counter deltas, and archive-verification.json
for retention integrity. File-backed cache versus MLX allocation overlap remains
a next-experiment hypothesis; these global samples do not prove its cause.
''')
    for p in sorted(OUT.iterdir()):files['archive-metadata/'+p.name]=p
    manifest={name:dict(source=str(p),**pin(p)) for name,p in sorted(files.items())}
    write(OUT/'manifest.json',manifest);files['manifest.json']=OUT/'manifest.json'
    archive=OUT/'evidence.tar.xz'
    with tarfile.open(archive,'x:xz',preset=1) as tar:
        for name,p in sorted(files.items()):tar.add(p,arcname=name,recursive=False)
    seen=set();raw_bytes=0
    with tarfile.open(archive,'r|xz') as tar:
        for member in tar:
            need(member.isfile() and member.name in files and member.name not in seen,'unexpected/duplicate archive member')
            seen.add(member.name);h=hashlib.sha256();size=0;stream=tar.extractfile(member)
            for block in iter(lambda:stream.read(8<<20),b''):h.update(block);size+=len(block)
            actual=dict(sha256=h.hexdigest(),bytes=size);expected=pin(files[member.name])
            need(actual==expected,'archive/original mismatch: '+member.name)
            if member.name!='manifest.json':need(all(actual[k]==manifest[member.name][k] for k in actual),'prearchive source mismatch')
            raw_bytes+=size
    need(seen==set(files),'missing archive members')
    for alias in aliases:need(alias_metadata(Path(alias['path']),expected_target)==alias,'model alias metadata changed during archive')
    verification=dict(status='PASS',members=len(seen),raw_bytes=raw_bytes,archive=pin(archive),
        scope='Retention only: every archived member and source byte stream verified. No qualification promotion.')
    write(OUT/'archive-verification.json',verification)
    print(json.dumps(dict(archive_directory=str(OUT),status=summary['status'],journal_state=journal['state'],
        audit_statuses=summary['audit_statuses'],members=verification['members'],archive=verification['archive'],scope=scope),indent=2))
    return 0

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',action='store_true');args=p.parse_args()
    if not args.run:
        print('DESCRIPTION ONLY: after driver exit and both saved audits, explicitly archive actual E039 evidence; no audit/service/model action.');return 0
    return run()
if __name__=='__main__':raise SystemExit(main())
