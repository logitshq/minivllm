"""Explicit prepared GLM stop -> direct owner -> finally recover driver.

Empty argv describes; --run needs an actual sealed registration and fresh
execution directory. No PREPARED journal or success receipt is fabricated.
The frozen owner owns the official GLM probe directly; this driver waits the
owner. Unknown subprocess outcome never receives --no-experiment recovery.
"""
import argparse
import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import time

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
REGISTRAR_SHA='7f937f709b0c6806a0dfca7606122e1178b2c495b4574665e7df1eea2862e842'


def need(ok,message):
    if not ok:raise ValueError(message)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_bytes())
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(folder,name,value):
    (folder/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def registrar():
    path=HERE/'preregister.py';need(sha(path)==REGISTRAR_SHA,'reviewed registrar source changed')
    sp=importlib.util.spec_from_file_location('glm_execute_registered_contract',path);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m


def preflight(registration,execution):
    registration=Path(registration).resolve();execution=Path(execution).resolve()
    receipt_raw=(registration/'receipt.json').read_bytes();receipt=json.loads(receipt_raw)
    need(receipt['schema']=='GLM-cold-load-prepared-receipt-v1' and receipt['experiment_started'] is False,'actual sealed GLM receipt')
    pins=receipt['source_pins'];plan_path=registration/'owner-plan.json'
    need(type(pins) is dict and all(str(p) in pins for p in
        (Path(__file__).resolve(),HERE/'preregister.py',HERE/'controller.py',HERE/'owner.py',plan_path,registration/'commands.json')),
        'execution/plan/controller/source closure required')
    for path,digest in pins.items():need(Path(path).is_absolute() and sha(path)==digest,'registered source changed: '+path)
    need(sha(plan_path)==receipt['owner_plan_sha256'],'registered owner plan changed')
    plan=read(plan_path);journal=Path(plan['cold_restart']['journal_path']);owner=Path(plan['outputs']['owner'])
    need(plan['execution']['owner']['path']==str(HERE/'owner.py') and plan['cold_restart']['controller']['path']==str(HERE/'controller.py'),
         'actual adjacent owner/controller entry')
    python=receipt['outer_python'];need(python in pins and sha(python)==pins[python],'registered outer interpreter entry')
    reg=registrar();analysis=Path(read(registration/'plan-build.json')['outputs']['analysis'])
    reg.validate_outputs([registration,owner,journal,analysis,execution])
    need(execution.parent.is_dir() and not execution.exists() and not owner.exists(),'fresh execution/owner outputs')
    raw=(journal/'journal.json').read_bytes();data=json.loads(raw)
    need(hashlib.sha256(raw).hexdigest()==receipt['prepared_journal_sha256'] and data['state']=='PREPARED'
        and data['schema']=='GLM-cold-load-journal-v1' and data['owner_directory']==str(owner)
        and data['owner_plan']==dict(path=str(plan_path),sha256=receipt['owner_plan_sha256'])
        and data['owner_source']==plan['execution']['owner'] and data['trials']==['glm-official-load']
        and data['controller_sha256']==pins[str(HERE/'controller.py')],'actual unchanged PREPARED journal')
    commands=read(registration/'commands.json')
    need(commands==reg.build_commands(plan_path,receipt['owner_plan_sha256'],plan,analysis,python),'saved exact GLM commands')
    return dict(registration=registration,execution=execution,owner=owner,journal=journal,commands=commands,
                receipt_raw=receipt_raw,prepared_raw=raw,owner_plan_sha256=receipt['owner_plan_sha256'])


def run_command(folder,name,command):
    record=dict(phase=name,argv=command,start=now(),monotonic_start=time.monotonic())
    save(folder,name+'-command.json',record)
    try:
        with (folder/(name+'.stdout')).open('xb') as out,(folder/(name+'.stderr')).open('xb') as err:
            value=subprocess.run(command,cwd=REPO,stdout=out,stderr=err,check=False)
        need(type(value.returncode) is int,'actual waited subprocess return code')
        record['returncode']=value.returncode;return value.returncode
    except BaseException as error:
        record.update(outcome='UNKNOWN',error_type=type(error).__name__,error=str(error));raise
    finally:
        record.update(end=now(),elapsed_seconds=time.monotonic()-record['monotonic_start']);save(folder,name+'-command.json',record)


def execute(context,run=run_command):
    out=context['execution'];owner=context['owner'];journal=context['journal'];commands=context['commands']
    out.mkdir();(out/'prepared-journal.json').write_bytes(context['prepared_raw'])
    (out/'execution-receipt.json').write_bytes(context['receipt_raw']);(out/'execute.py').write_bytes(Path(__file__).read_bytes())
    save(out,'authorization.json',dict(recorded_at=now(),basis='Prior explicit human cold-stop/experiment/restore authorization',
        contract='cold-restart-persisted-visible-v1',scope='One direct official GLM load probe, original caps8/32/32/32 and explicit physical policy; no performance or full-model qualification.'))
    stop_rc=owner_rc=recover_rc=None;attempted=returned=False;primary=None
    try:
        stop_rc=run(out,'stop',commands['controller_stop'])
        if stop_rc==0 and read(journal/'journal.json')['state']=='STOPPED':
            attempted=True;save(out,'owner-invocation.json',dict(attempted=True,returned=False,time=now()))
            owner_rc=run(out,'glm-owner',commands['owner']);returned=True
            save(out,'owner-invocation.json',dict(attempted=True,returned=True,returncode=owner_rc,time=now()))
    except BaseException as error:
        primary=error;save(out,'primary-error.json',dict(type=type(error).__name__,message=str(error)));raise
    finally:
        recovery_error=None
        try:
            state=read(journal/'journal.json')['state']
            if state!='PREPARED':
                command=list(commands['controller_recover'])
                if not owner.exists() and (not attempted or returned):command.append('--no-experiment')
                recover_rc=run(out,'recover',command)
        except BaseException as error:
            recovery_error=error;save(out,'recovery-error.json',dict(type=type(error).__name__,message=str(error)))
        finally:
            try:state=read(journal/'journal.json')['state']
            except Exception:state='UNREADABLE'
            save(out,'final.json',dict(time=now(),journal_state=state,owner_plan_sha256=context['owner_plan_sha256'],
                stop_returncode=stop_rc,owner_returncode=owner_rc,recovery_returncode=recover_rc,
                owner_invocation_attempted=attempted,owner_invocation_returned=returned,
                recovery_error=None if recovery_error is None else type(recovery_error).__name__,
                observed_diagnostic_complete=False,independent_audit_required=True))
        if recovery_error is not None and primary is None:raise recovery_error
    return int(stop_rc!=0 or owner_rc!=0 or recover_rc!=0 or state!='RESTORED')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',action='store_true')
    p.add_argument('--registration',type=Path);p.add_argument('--out',type=Path);a=p.parse_args(argv)
    if not a.run:
        need(not any((a.registration,a.out)),'explicit --run required')
        print('DESCRIPTION_ONLY: actual sealed PREPARED registration and fresh execution output required.');return 0
    need(a.registration and a.out,'complete explicit driver paths required')
    return execute(preflight(a.registration,a.out))


if __name__=='__main__':raise SystemExit(main())
