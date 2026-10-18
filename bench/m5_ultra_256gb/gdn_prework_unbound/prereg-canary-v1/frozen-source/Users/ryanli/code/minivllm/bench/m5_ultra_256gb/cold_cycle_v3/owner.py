"""Stable single-direct-child owner; no service mutations or numerical gate.

Fixed pressure1/AC/thermal/swap, work1830 and bounded TERM/KILL/wait are retained
from v7. The immutable run plan supplies only a phase and pinned Python qualifier.
Original4dff prepare preserves runtime/source/model checks and actual old CLI.
Both child elapsed and total work are enforced. Exit0 means execution success,
never numerical/fullmodel/performance qualification.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

HERE=Path(__file__).resolve().parent
OWNER_PLAN_SHA='4dff5f46d0b35d038ffbb0e32ebb6197b047f4da6e847b3160f3ae8685caf6a4'
QUALIFICATION_SHA='5025d37768a5fa74d6e3e256a96ff8b8e4e92b31837183d5ffe031d9473e7123'
NUMERIC_SHA='21ba52e43ce54c2b463ab46bfd212c53d06d6303530c77939e3e6b801df547fb'
CONTROLLER=HERE/'controller.py'


def controller():
    # Fixed adjacent module; manifest is verified before any backend operation.
    spec=importlib.util.spec_from_file_location('_stable_cold_controller',CONTROLLER)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module



def need(ok,message):
    if not ok:raise ValueError(message)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(relative):
    path=(HERE/relative).resolve();need(sha(path)==controller().SUPPORT_PINS[relative],'frozen E024 support changed')
    spec=importlib.util.spec_from_file_location('_e024_cold_'+path.stem,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def source_pins(plan):
    c=controller();return c.source_pins(plan,c.lifecycle(plan['lifecycle']))


def _predecessor(plan):
    return controller().predecessor(plan['predecessor'],plan['lifecycle'])[0]


def validate_plan(args):
    c=controller();plan,manifest=c.run_plan(args.plan,args.plan_sha256)
    need(manifest['sources']['owner']==dict(path=str(Path(__file__).resolve()),sha256=sha(__file__)),'actual stable owner source')
    need(plan['outputs']['out']==str(args.out),'actual owner output path')
    base=load('../pareto_realistic/qualify_absent_owner.py')
    old=json.loads(Path(plan['inherited_owner_plan']['path']).read_bytes())
    for key,digest in (('qualification_plan',QUALIFICATION_SHA),('inherited_plan',NUMERIC_SHA)):
        item=old[key];need(item['sha256']==digest==sha(item['path']),'unchanged inherited numerical plan')
    numeric=json.loads(Path(old['inherited_plan']['path']).read_bytes())
    return base,plan,Path(args.plan).read_bytes(),numeric,_predecessor(plan)


def stopped(base,plan,args,prior,binding=None):
    c=controller();manifest=c.lifecycle(plan['lifecycle'])
    directory=Path(plan['outputs']['journal']);raw=(directory/'journal.json').read_bytes();j=json.loads(raw)
    expected=dict(schema=c.JOURNAL_SCHEMA,state='STOPPED',plan_sha256=base.PROPOSAL_SHA,
        controller_sha256=manifest['sources']['controller']['sha256'],approved_contract=base.CONTRACT,
        original=prior['replacement'],owner_directory=str(args.out),
        owner_plan=dict(path=str(args.plan),sha256=args.plan_sha256),owner_source=manifest['sources']['owner'],
        predecessor=plan['predecessor'],trials=[plan['experiment']['phase']],lifecycle=plan['lifecycle'])
    need(all(j.get(k)==v for k,v in expected.items()),'bound stable service is not STOPPED')
    need(type(j.get('id')) is str and j['id'] and type(j.get('removed_at')) in (int,float)
         and math.isfinite(j['removed_at']) and 0<j['removed_at']<=time.time()
         and type(j.get('predecessor_copies')) is dict,'STOPPED runtime identity')
    stable={k:j[k] for k in (*expected,'id','removed_at','predecessor_copies')}
    need(binding is None or stable==binding,'STOPPED identity changed');return raw,stable


def prepare(plan,out,check):
    """Original fixed4dff preparation; retain its actual CLI before adaptation."""
    out.mkdir(exist_ok=False);adapter=load('../long_decode_graph/owner.py');original=adapter.load_legacy().prepare
    item=plan['inherited_owner_plan']
    command,env=adapter.prepare(SimpleNamespace(out=out,plan=Path(item['path']),plan_sha256=item['sha256'],
                                               deadline_seconds=1830),check,original)
    base=load('../pareto_realistic/qualify_absent_owner.py');ref=plan['experiment']['plan'];check()
    raw=Path(ref['path']).read_bytes();need(base.sha_bytes(raw)==ref['sha256'],'experiment plan changed during prepare')
    q=plan['experiment']['qualifier'];need(sha(q['path'])==q['sha256'],'experiment qualifier changed during prepare')
    base.durable(out/'inherited-launch.json',dict(command=command,original_qualification_plan_sha256=QUALIFICATION_SHA),exclusive=True)
    base.durable(out/'experiment-plan.json',raw,exclusive=True,raw=True)
    return [command[0],'-I','-B',q['path'],'--plan',ref['path'],'--plan-sha256',ref['sha256'],
            '--out',str(out/'qualification')],env


def run(args,*,system=None,resources=None):
    base,plan,plan_raw,numeric,prior=validate_plan(args)
    system=system or base.System();directory=Path(plan['outputs']['journal']);PHASE=plan['experiment']['phase']
    started=time.monotonic();child=None;child_started=None;intent=False;deadline_active=False
    samples=[];errors=[];children=[];attempt=binding=None;phase='attempt';absence_count=0
    folder=args.out/'phases'/PHASE
    def failure(exc,where):
        row=dict(phase=where,type=type(exc).__name__,error=str(exc),time=time.time());errors.append(row)
        base.durable(args.out/'errors.json',errors)
        if len(errors)==1:base.durable(args.out/'first-error.json',row,exclusive=True)
    def stop(*unused):raise TimeoutError('E024 owner absolute deadline or cancellation')
    def check(*,cleanup=False):
        if not cleanup and time.monotonic()-started>=1830:stop()
        return stopped(base,plan,args,prior,binding)
    def probe(label,pid=None,*,cleanup=False):
        nonlocal absence_count
        check(cleanup=cleanup);proof=base.absence(system,binding['original']['pid']);n=len(samples)
        base.durable(args.out/f'absence-{absence_count:03d}.json',dict(label=label,time=time.time(),resource_index_if_sampled=n,**proof),exclusive=True)
        absence_count+=1
        need(proof['status']=='PASS','removed service PID/job/8001 reappeared or unknown')
        row=resources.snapshot(label,pid,suspended_pids=());samples.append(row)
        base.durable(args.out/f'resource-{n:03d}.json',row,exclusive=True);resources.guard(samples)
        need(row['authorized_suspended_pids']==[] and all(p['owned'] and p['pid']==pid and not p['authorized_suspended']
             for p in row['native_inference_processes']),'foreign inference process or suspended exemption')
        need(row['pressure_level_value']==1 and row['power_metadata']['source']=='AC Power'
             and row['processinfo_thermal_metadata']['code'] in (0,1),'pressure/power/thermal guard')
        check(cleanup=cleanup)
    def ref(path):return dict(path=str(path.relative_to(args.out)),sha256=sha(path))
    def index():base.durable(args.out/'children.json',dict(schema='cold-service-owner-children-v1',children=children))
    def bound():return {k:attempt[k] for k in ('owner_pid','owner_identity','owner_source','owner_plan_sha256','journal_id','removed_at')}
    def freeze_sources():
        target=args.out/'owner-source';target.mkdir()
        pins=source_pins(plan)
        for name,digest in pins.items():
            check();path=Path(name);raw=path.read_bytes();need(base.sha_bytes(raw)==digest,'owner source changed')
            dest=target/name.lstrip('/');dest.parent.mkdir(parents=True,exist_ok=True)
            base.durable(dest,raw,exclusive=True,raw=True)
        base.durable(target/'manifest.json',pins,exclusive=True)
    with base.journal_lock(directory):
        snapshot,binding=stopped(base,plan,args,prior);identity=system.identity(os.getpid())
        need(identity['pid']==os.getpid(),'owner identity')
        args.out.mkdir(exist_ok=False);folder.parent.mkdir()
        base.durable(args.out/'stopped-journal.json',snapshot,exclusive=True,raw=True)
        attempt=dict(schema='cold-service-owner-attempt-v1',owner_pid=os.getpid(),owner_identity=identity,
            owner_source=controller().lifecycle(plan['lifecycle'])['sources']['owner'],owner_plan_sha256=args.plan_sha256,journal_id=binding['id'],
            journal_path=str(directory),controller_sha256=binding['controller_sha256'],removed_at=binding['removed_at'],
            created_at=time.time(),phase='PRESPAWN',stopped_journal_sha256=base.sha_bytes(snapshot))
        base.durable(args.out/'attempt.json',attempt,exclusive=True);index()
    try:
        remaining=1830-(time.monotonic()-started);need(remaining>0,'work deadline before prepare')
        system.start_deadline(remaining,stop);deadline_active=True
        base.durable(args.out/'owner-plan.json',plan_raw,exclusive=True,raw=True)
        phase='prepare';freeze_sources();resources=resources or base.load_resources(numeric)
        probe('before-prepare');command,env=prepare(plan,folder,check)
        base.durable(args.out/'launch.json',dict(command=command,environment={k:v for k,v in env.items() if k.startswith('MLX_')},
            experiment_plan_sha256=plan['experiment']['plan']['sha256'],original_qualification_plan_sha256=QUALIFICATION_SHA,original_owner_plan_sha256=OWNER_PLAN_SHA),exclusive=True)
        probe('after-prepare-before-spawn');validate_plan(args)
        with (folder/'child.log').open('x') as log:
            with base.journal_lock(directory):
                check();absent=base.absence(system,binding['original']['pid'])
                base.durable(folder/'pre-spawn-absence.json',absent,exclusive=True)
                need(absent['status']=='PASS','service absence changed before intent')
                phase='spawn';intent=True
                evidence=dict(phase=PHASE,owner_pid=os.getpid(),journal_id=binding['id'],owner_plan_sha256=args.plan_sha256,
                              attempt_sha256=sha(args.out/'attempt.json'),time=time.time())
                base.durable(args.out/'spawn-intent.json',evidence,exclusive=True)
                base.durable(folder/'intent.json',evidence,exclusive=True)
                child=system.spawn(command,log,env);child_started=time.monotonic()
                need(type(child.pid) is int and child.pid>1 and child.pid!=os.getpid(),'direct returned child PID')
                payload=dict(phase=PHASE,pid=child.pid,owner_pid=os.getpid(),identity=None,
                             ownership='direct-Popen-returned',spawned_at=time.time())
                base.durable(folder/'child.json',payload,exclusive=True)
                children.append(dict(phase=PHASE,pid=child.pid,identity=None,intent=ref(folder/'intent.json'),child=ref(folder/'child.json'),release=None));index()
                identity=system.identity(child.pid);need(identity['pid']==child.pid and identity['ppid']==os.getpid(),'direct child identity')
                payload['identity']=children[0]['identity']=identity;base.durable(folder/'child.json',payload)
                children[0]['child']=ref(folder/'child.json');index()
            phase=PHASE
            while child.poll() is None:
                need(time.monotonic()-child_started<=plan['budgets']['child_seconds'],'child elapsed deadline exceeded')
                probe(PHASE+':running',child.pid);check()
                need(time.monotonic()-child_started<=plan['budgets']['child_seconds'],'child elapsed deadline exceeded')
                try:child.wait(timeout=min(5,1830-(time.monotonic()-started),plan['budgets']['child_seconds']-(time.monotonic()-child_started)))
                except subprocess.TimeoutExpired:pass
            child.wait(timeout=0)
        need(time.monotonic()-child_started<=plan['budgets']['child_seconds'],'child elapsed deadline exceeded')
        need(child.returncode==0,f'experiment child exit {child.returncode}')
        # Do not interpret experiment-specific numeric schemas or trust a child PASS.
    except BaseException as exc:failure(exc,phase)
    finally:
        if deadline_active:system.cleanup_mode()
        if child is not None:
            try:
                if child.poll() is None:
                    try:child.terminate();child.wait(timeout=15)
                    except BaseException as exc:failure(exc,'terminate/wait')
                    if child.poll() is None:child.kill()
                child.wait(timeout=15)
                raw=system.read(['/bin/ps','-p',str(child.pid),'-o','pid='])
                base.durable(folder/'child-absence.json',raw,exclusive=True)
                need(type(child.returncode) is int and raw['returncode']==1 and not raw['stdout'].strip(),'strict child wait/reap/absence')
                need(len(children)==1 and children[0]['pid']==child.pid,'known child receipt required')
                base.durable(folder/'release.json',dict(phase=PHASE,pid=child.pid,returncode=child.returncode,
                    waited_reaped=True,strict_absence=True,absence=ref(folder/'child-absence.json'),time=time.time()),exclusive=True)
                children[0]['release']=ref(folder/'release.json');index()
            except BaseException as exc:failure(exc,'cleanup')
        try:
            if deadline_active:system.postflight_deadline()
            if resources is not None:
                probe('postflight',cleanup=True);base.durable(args.out/'resources.json',resources.deltas(samples))
            validate_plan(args);prepare(plan,args.out/'postflight',lambda:check(cleanup=True))
        except BaseException as exc:failure(exc,'postflight')
        finally:
            if deadline_active:system.restore_deadline()
        if not intent:
            need(child is None and not children and errors,'NO_CHILD requires pre-intent failure')
            base.durable(args.out/'terminal.json',dict(schema='cold-service-owner-terminal-v1',status='NO_CHILD_STARTED',
                **bound(),attempt_sha256=sha(args.out/'attempt.json'),spawn_intent_written=False,child_started=False,
                terminal_at=time.time(),phase=errors[0]['phase'],error=errors[0]),exclusive=True)
        elif len(children)==1 and children[0]['release'] is not None:
            base.durable(args.out/'release.json',dict(schema='cold-service-owner-release-v2',**bound(),
                attempt_sha256=sha(args.out/'attempt.json'),children=children,children_sha256=sha(args.out/'children.json'),
                all_children_reaped=True,time=time.time()),exclusive=True)
        base.durable(args.out/'complete.json',dict(schema='cold-single-child-owner-result-v1',status='FAIL' if errors else 'PASS',
            owner_pid=os.getpid(),owner_plan_sha256=args.plan_sha256,planned_trials=[PHASE],errors=errors,
            first_failure=errors[0] if errors else None,spawn_intent_written=intent,child_started=True if child else False if not intent else None,
            all_children_reaped=(args.out/'release.json').exists(),samples=len(samples),elapsed_wall_s=time.monotonic()-started,
            lifecycle=plan['lifecycle'],child_returncode=child.returncode if child else None,independent_numeric_audit_required=True,full_model_workload_qualified=False,performance_qualified=False),exclusive=True)
    return int(bool(errors))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--plan-sha256',required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args(argv);args.plan,args.out=args.plan.resolve(),args.out.resolve()
    need(args.out.parent.is_dir() and not args.out.exists(),'fresh output in existing parent required')
    return run(args)


if __name__=='__main__':raise SystemExit(main())
