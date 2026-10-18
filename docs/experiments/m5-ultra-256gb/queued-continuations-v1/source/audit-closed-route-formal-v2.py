"""Offline closed E024 OFF/ON formal-cycle audits. Default: describe only.

The actual driver must already have returned from recovery with RESTORED.
Run the frozen restoration auditor first, independently of experiment success,
then the frozen integrated raw HTTP/count/timing/resource auditor. No new
resource or numerical algorithm, live probe, service action or model import.
Each attempt has an exclusive directory; failures and partial results remain.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback

R=Path('/Users/ryanli/code/minivllm'); B=R/'bench/m5_ultra_256gb'; A=R/'.artifacts/m5-ultra-256gb'
NAME='route-http-formal-v2'; P=B/'route_http_formal/prereg-formal-v2'
OWNER=A/(NAME+'-owner'); JOURNAL=A/(NAME+'-cold-journal'); EXECUTION=A/(NAME+'-execution')
PY=str(R/'.venv/bin/python')
PLAN_SHA='0920aceae4e8bb6ef5c68577d3ae905a2b5976287790a7bb7a4a6e4ea36470ce'
RESTORE=B/'route_http_formal/audit_restore.py'; FORMAL=B/'route_http_formal/audit_formal.py'
FIXED={str(RESTORE):'90a980d0dbe6a61fc1bab2ac6b2edaa9bf012bb60726cd0742801d11410eb0b1',
       str(FORMAL):'ed8ad64c3eb5fed052effd05981a386006f4ea1f752463bb0ada0c2f0e4ea5e0'}
PHASES=['r1-c1-off','r1-c1-on','r1-c8-on','r1-c8-off','r2-c8-off','r2-c8-on',
        'r2-c1-on','r2-c1-off','r3-c1-off','r3-c1-on','r3-c8-on','r3-c8-off']

def need(ok,message):
    if not ok:raise ValueError(message)
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_bytes())
def ref(path):return dict(path=str(path),sha256=sha(path),bytes=Path(path).stat().st_size)
def save(out,name,value):
    (out/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def invoke(out,name,source,argv,filename):
    output=out/filename
    row=dict(status='STARTED',argv=[PY,'-I','-B',str(source),*argv,'--out',str(output)],
             cwd=str(R),source=ref(source),started_at=now(),timeout_seconds=600)
    save(out,name+'-execution.json',row);started=time.monotonic()
    try:
        with (out/(name+'.stdout')).open('xb') as stdout,(out/(name+'.stderr')).open('xb') as stderr:
            child=subprocess.run(row['argv'],cwd=R,stdout=stdout,stderr=stderr,timeout=600,check=False)
        row.update(status='RETURNED',returncode=child.returncode)
    except BaseException as error:
        row.update(status='ERROR',error=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc()))
    finally:
        row.update(ended_at=now(),elapsed_seconds=time.monotonic()-started)
        if output.exists():
            row['output']=ref(output)
            try:row['audit_status']=read(output).get('status')
            except BaseException as error:row['output_error']=dict(type=type(error).__name__,message=str(error))
        save(out,name+'-execution.json',row)
    return row

def run(out):
    out=Path(out).absolute()
    need(out.parent==A and re.fullmatch(NAME+r'-analysis(?:-repeat-[1-9][0-9]*)?',out.name),'fixed fresh analysis path')
    need(not out.exists(),'never overwrite any audit attempt');out.mkdir()
    (out/'runner.py').write_bytes(Path(__file__).read_bytes())
    save(out,'command.json',dict(argv=sys.argv,cwd=str(Path.cwd()),scope='offline audits only'))
    result=dict(schema='E024-route-formal-closed-audits-v1',status='INCOMPLETE',started_at=now(),
        owner_plan_sha256=PLAN_SHA,inputs={},source_pins={},jobs={},execution_pass=False,
        restoration_qualified=False,performance_qualified=False,resources_verified_with_formal=False,
        scope='Twelve fresh OFF/ON servers,96 warm+768 formal requests only if complete independent audit PASS. Restoration is independent. No proof of every future schedule or observed formal graph hits.')
    save(out,'execution-summary.json',result)
    try:
        final=read(EXECUTION/'final.json');journal=read(JOURNAL/'journal.json')
        need(final['journal_state']==journal['state']=='RESTORED' and type(final['recovery_returncode']) is int
             and final['recovery_returncode']==0,'actual finally-recovery must already be closed')
        receipt=read(P/'receipt.json');plan=read(P/'owner-plan.json')
        need(receipt['schema']=='E024-route-formal-prereg-receipt-v1' and receipt['status']=='PREPARED'
             and receipt['owner_plan_sha256']==sha(P/'owner-plan.json')==PLAN_SHA,'actual registered fixed plan')
        need(plan['schema']=='E024-route-http-formal-owner-plan-v1' and plan['trials']==PHASES
             and plan['cold_restart']['journal_path']==str(JOURNAL),'actual formal plan/phase binding')
        need(journal['schema']=='E024-route-http-formal-cold-journal-v1' and journal['trials']==PHASES
             and journal['owner_directory']==str(OWNER)
             and journal['owner_plan']==dict(path=str(P/'owner-plan.json'),sha256=PLAN_SHA)
             and journal['owner_source']==plan['execution']['owner'],'actual restored journal binding')
        for source,digest in FIXED.items():
            need(receipt['source_pins'].get(source)==digest==sha(source),'registered frozen auditor source')
        for name,path in (('driver-final.json',EXECUTION/'final.json'),('restored-journal.json',JOURNAL/'journal.json'),
                          ('registration-receipt.json',P/'receipt.json')):
            result['inputs'][name]=ref(path);(out/name).write_bytes(path.read_bytes())
        # Restoration remains a separate persisted result even when later raw
        # audit, source copying, or performance qualification fails.
        restore_job=invoke(out,'restoration',RESTORE,['--journal',str(JOURNAL)],'restoration-audit.json')
        result['jobs']['restoration']=restore_job;save(out,'execution-summary.json',result)
        need(restore_job.get('returncode')==0 and restore_job.get('audit_status')=='PASS','independent restoration failed')
        restore=read(out/'restoration-audit.json')
        need(restore['schema']=='cold-service-restoration-audit-v1' and restore['qualified'] is True
             and restore['performance_qualified'] is False and restore['numerical_qualified'] is False
             and restore['journal_sha256']==sha(JOURNAL/'journal.json')
             and restore['owner_plan_sha256']==PLAN_SHA and restore['audit_source_sha256']==FIXED[str(RESTORE)],
             'independent restoration source/identity/scope')
        result['restoration_qualified']=True;save(out,'execution-summary.json',result)
        pins=dict(receipt['source_pins']);pins[str(Path(__file__).resolve())]=sha(__file__)
        for source,digest in pins.items():
            path=Path(source)
            need(path.is_absolute() and not path.is_symlink() and path.suffix in ('.py','.json','.jsonl','.md','.metal','.h')
                 and path.stat().st_size<32*(1<<20),'registered small metadata/source only')
            raw=path.read_bytes();need(hashlib.sha256(raw).hexdigest()==digest,'registered source changed: '+source)
            target=out/'frozen-source'/source.lstrip('/');target.parent.mkdir(parents=True,exist_ok=True)
            with target.open('xb') as stream:stream.write(raw)
            result['source_pins'][source]=digest
        # Always call the original formal auditor, including a failed/unstarted
        # owner: its FAIL/INCOMPLETE is kept, never replaced by synthetic PASS.
        job=invoke(out,'formal',FORMAL,['--owner',str(OWNER),'--journal',str(JOURNAL),'--plan-sha256',PLAN_SHA],
                   'independent-formal.json')
        result['jobs']['formal']=job
        terminal=read(OWNER/'complete.json') if (OWNER/'complete.json').exists() else None
        if terminal is not None:
            need(terminal['owner_plan_sha256']==PLAN_SHA and terminal['planned_trials']==PHASES,'closed owner terminal')
            result['inputs']['owner-complete.json']=ref(OWNER/'complete.json')
        result['execution_pass']=bool(terminal and final['stop_returncode']==0 and final['owner_returncode']==0
            and terminal['status']=='PASS' and terminal['formal_trials_complete'] is True and terminal['all_children_reaped'] is True)
        if job.get('returncode')==0 and job.get('audit_status')=='PASS':
            value=read(out/'independent-formal.json')
            need(value['schema']=='E024-route-http-formal-independent-audit-v1'
                 and value['audit_source_sha256']==FIXED[str(FORMAL)] and value['owner_directory']==str(OWNER)
                 and value['owner_plan_sha256']==PLAN_SHA and value['qualified'] is value['performance_qualified'] is True
                 and value['requests']==dict(total=864,warm=96,formal=768)
                 and value['restoration']==restore and value['journal_sha256']==sha(JOURNAL/'journal.json')
                 and value['resources'] and value['actual_compiled_invocations_observed'] is False,
                 'complete independent formal/count/resource/source binding')
            result['performance_qualified']=result['resources_verified_with_formal']=True
        result['status']='PASS' if result['execution_pass'] and result['performance_qualified'] else 'FAIL'
    except BaseException as error:
        result.update(status='FAIL',error=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc()))
    finally:
        result['sources_unchanged']=bool(result['source_pins'])
        for path,digest in result['source_pins'].items():
            try:unchanged=sha(path)==digest
            except BaseException:unchanged=False
            if not unchanged:result['sources_unchanged']=False
        for name,item in result['inputs'].items():
            try:unchanged=ref(Path(item['path']))==item
            except BaseException:unchanged=False
            if not unchanged:result.setdefault('changed_inputs',[]).append(name)
        if not result['sources_unchanged'] or result.get('changed_inputs'):
            result['status']='FAIL'
        if result['status']!='PASS':
            result['performance_qualified']=False;result['resources_verified_with_formal']=False
        result['ended_at']=now();save(out,'execution-summary.json',result)
        save(out,'manifest.json',{str(p.relative_to(out)):dict(sha256=sha(p),bytes=p.stat().st_size)
                                 for p in out.rglob('*') if p.is_file()})
    print(json.dumps({k:result[k] for k in ('status','restoration_qualified','performance_qualified')}))
    return int(result['status']!='PASS')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',action='store_true')
    parser.add_argument('--out',type=Path,default=A/(NAME+'-analysis'));args=parser.parse_args()
    if args.run:raise SystemExit(run(args.out))
    print('DESCRIPTION_ONLY: no audit launched; actual final RESTORED required.')
