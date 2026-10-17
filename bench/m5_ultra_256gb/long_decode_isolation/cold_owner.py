"""One fixed E024 dispatch-isolation child while the restored oMLX service is absent.

The old E024 qualifier, candidate and two plans are unchanged. Their original
prepare validates the real inherited E023 runtime/source manifest. The child
itself checks all model shards before/after its gates-OFF four-arm dispatch diagnostic.
This owner neither stops/restores services nor launches an HTTP server. Its
one real phase is e024-isolation; no formal-trial labels or source paths are
impersonated. Pressure1, AC/thermal/swap, 1830s work and bounded cleanup remain.
Import/help are stdlib-only. CPU preparation is not numerical qualification.
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
PHASE='e024-isolation'
OWNER_PLAN_SHA='4dff5f46d0b35d038ffbb0e32ebb6197b047f4da6e847b3160f3ae8685caf6a4'
QUALIFICATION_SHA='5025d37768a5fa74d6e3e256a96ff8b8e4e92b31837183d5ffe031d9473e7123'
NUMERIC_SHA='21ba52e43ce54c2b463ab46bfd212c53d06d6303530c77939e3e6b801df547fb'
FIXED={
    'qualify.py':'9d8c17a4bfc7f6ea9d61f79f466cad45059b71c9606d7b8d29c05c40213cad9e',
    '../long_decode_graph/owner.py':'d9469fe561ee5697068de1582d1e167d2a0e0da222cc03d4449028435c232134',
    '../long_decode_graph/qualify.py':'c24219f431b08502a26a4f7a83c002214bddd7ea3962d6b1869b313e4c13faf7',
    '../long_decode_graph/candidate.py':'eb14b671a66764fe677219762090acb495b89d384cf078f3f63b918a40aa2b84',
    '../long_decode_graph/model-source-pins.json':'7143538a168bfcd8b01292c9687472c8647db393c6d2a510d628f71c670260c5',
    '../pareto_realistic/qualify_absent_owner.py':'5beb6c9bcfa4e22041a386cdbea3f5ecee210cdcb9b345512c50b6c6f873c62d',
    '../pareto_realistic/qualify_owner.py':'1efaf440e991fed64dedf15455be2d37223e709fec41bb07e8c0295a2975c1c2',
}


def need(ok,message):
    if not ok:raise ValueError(message)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(relative):
    path=(HERE/relative).resolve();need(sha(path)==FIXED[relative],'frozen E024 support changed')
    spec=importlib.util.spec_from_file_location('_e024_cold_'+path.stem,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def source_pins():
    return {**{str((HERE/name).resolve()):digest for name,digest in FIXED.items()},
            str(Path(__file__).resolve()):sha(__file__),
            str(HERE/'test_cold_owner_cpu.py'):sha(HERE/'test_cold_owner_cpu.py')}


def _predecessor(cold):
    path=Path(cold['controller']['path'])
    need(sha(path)==cold['controller']['sha256'],'actual controller changed')
    spec=importlib.util.spec_from_file_location('_e024_actual_controller_v6',path)
    controller=importlib.util.module_from_spec(spec);spec.loader.exec_module(controller)
    prior,_=controller.predecessor(cold['predecessor']);return prior


def validate_plan(args):
    base=load('../pareto_realistic/qualify_absent_owner.py')
    def pinned(item,digest=None):
        need(set(item)=={'path','sha256'} and (digest is None or item['sha256']==digest),'fixed plan reference')
        raw=base.absolute(item['path']).read_bytes()
        need(base.sha_bytes(raw)==item['sha256'],'plan reference SHA');return json.loads(raw)
    raw=base.absolute(args.plan).read_bytes();need(base.sha_bytes(raw)==args.plan_sha256,'cold wrapper plan SHA')
    plan=json.loads(raw)
    need(set(plan)=={'schema','execution','diagnostic_owner_plan','isolation_plan','cold_restart','trials','deadline_seconds'}
         and plan['schema']=='E024-isolation-cold-owner-plan-v1' and plan['trials']==[PHASE]
         and type(plan['deadline_seconds']) is int and plan['deadline_seconds']==1830,'fixed single diagnostic owner plan')
    need(plan['execution']==dict(owner=dict(path=str(Path(__file__).resolve()),sha256=sha(__file__)),
                                pins=source_pins()),'exact actual cold owner/source pins')
    for path,digest in plan['execution']['pins'].items():need(sha(path)==digest,'source changed: '+path)
    old=pinned(plan['diagnostic_owner_plan'],OWNER_PLAN_SHA)
    qualification=pinned(old['qualification_plan'],QUALIFICATION_SHA)
    numeric=pinned(old['inherited_plan'],NUMERIC_SHA)
    need(qualification['e023_plan']==old['inherited_plan'] and qualification['deadline_s']==1800
         and qualification['qualifier_sha256']==FIXED['../long_decode_graph/qualify.py'],'unchanged E024 diagnostic binding')
    isolation=pinned(plan['isolation_plan'])
    need(set(isolation)=={'schema','qualifier_sha256','e024_plan','deadline_s'}
         and isolation['schema']=='E024-dispatch-isolation-plan-v1'
         and isolation['qualifier_sha256']==FIXED['qualify.py']
         and isolation['e024_plan']==old['qualification_plan']
         and type(isolation['deadline_s']) in (int,float) and math.isfinite(isolation['deadline_s'])
         and 0<isolation['deadline_s']<=1800,'fixed four-arm isolation plan/deadline')
    cold=plan['cold_restart'];controller_path=HERE.parent/'omlx_cold_restart/controller_v6.py'
    need(set(cold)=={'journal_path','controller','approved_contract','proposal_plan_sha256','predecessor'}
         and cold['controller']['path']==str(controller_path)
         and sha(controller_path)==cold['controller']['sha256']
         and cold['approved_contract']==base.CONTRACT and cold['proposal_plan_sha256']==base.PROPOSAL_SHA,
         'actual v6 cold contract binding')
    base.absolute(cold['journal_path'])
    prior=_predecessor(cold)
    return base,plan,raw,numeric,prior


def stopped(base,plan,args,prior,binding=None):
    directory=Path(plan['cold_restart']['journal_path']);raw=(directory/'journal.json').read_bytes();j=json.loads(raw)
    expected=dict(schema='omlx-cold-restart-journal-v6',state='STOPPED',plan_sha256=base.PROPOSAL_SHA,
        controller_sha256=plan['cold_restart']['controller']['sha256'],approved_contract=base.CONTRACT,
        original=prior['replacement'],owner_directory=str(args.out),
        owner_plan=dict(path=str(args.plan),sha256=args.plan_sha256),owner_source=plan['execution']['owner'],
        predecessor=plan['cold_restart']['predecessor'],trials=[PHASE])
    need(all(j.get(k)==v for k,v in expected.items()),'bound v6 service is not STOPPED')
    need(type(j.get('id')) is str and j['id'] and type(j.get('removed_at')) in (int,float)
         and math.isfinite(j['removed_at']) and 0<j['removed_at']<=time.time()
         and type(j.get('predecessor_copies')) is dict,'STOPPED runtime identity')
    stable={k:j[k] for k in (*expected,'id','removed_at','predecessor_copies')}
    need(binding is None or stable==binding,'STOPPED identity changed');return raw,stable


def prepare(plan,out,check):
    """Run unchanged old E024 preparation, retaining its actual CLI before adaptation."""
    out.mkdir(exist_ok=False);adapter=load('../long_decode_graph/owner.py');original=adapter.load_legacy().prepare
    item=plan['diagnostic_owner_plan']
    command,env=adapter.prepare(SimpleNamespace(out=out,plan=Path(item['path']),plan_sha256=item['sha256'],
                                               deadline_seconds=1830),check,original)
    base=load('../pareto_realistic/qualify_absent_owner.py');ref=plan['isolation_plan'];check()
    raw=Path(ref['path']).read_bytes();need(base.sha_bytes(raw)==ref['sha256'],'isolation plan changed during prepare')
    need(sha(HERE/'qualify.py')==FIXED['qualify.py'],'isolation qualifier changed during prepare')
    base.durable(out/'inherited-launch.json',dict(command=command,original_qualification_plan_sha256=QUALIFICATION_SHA),exclusive=True)
    base.durable(out/'isolation-plan.json',raw,exclusive=True,raw=True)
    return [command[0],'-I','-B',str(HERE/'qualify.py'),'--plan',ref['path'],'--plan-sha256',ref['sha256'],
            '--out',str(out/'qualification')],env


def run(args,*,system=None,resources=None):
    base,plan,plan_raw,numeric,prior=validate_plan(args)
    system=system or base.System();directory=Path(plan['cold_restart']['journal_path'])
    started=time.monotonic();child=None;intent=False;deadline_active=False
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
        pins={**plan['execution']['pins'],plan['cold_restart']['controller']['path']:plan['cold_restart']['controller']['sha256']}
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
            owner_source=plan['execution']['owner'],owner_plan_sha256=args.plan_sha256,journal_id=binding['id'],
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
            isolation_plan_sha256=plan['isolation_plan']['sha256'],original_qualification_plan_sha256=QUALIFICATION_SHA,original_owner_plan_sha256=OWNER_PLAN_SHA),exclusive=True)
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
                child=system.spawn(command,log,env)
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
                probe(PHASE+':running',child.pid);check()
                try:child.wait(timeout=min(5,1830-(time.monotonic()-started)))
                except subprocess.TimeoutExpired:pass
            child.wait(timeout=0)
        need(child.returncode==0,f'diagnostic exit {child.returncode}')
        result=json.loads((folder/'qualification/complete.json').read_bytes())
        need(result['schema']=='E024-dispatch-isolation-result-v1' and result['status']=='PASS'
             and result['diagnostic_complete'] is True
             and result['full_model_qualified'] is False
             and result['performance_qualified'] is False,'diagnostic producer incomplete')
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
        base.durable(args.out/'complete.json',dict(schema='E024-isolation-cold-owner-result-v1',status='FAIL' if errors else 'PASS',
            owner_pid=os.getpid(),owner_plan_sha256=args.plan_sha256,planned_trials=[PHASE],errors=errors,
            first_failure=errors[0] if errors else None,spawn_intent_written=intent,child_started=True if child else False if not intent else None,
            all_children_reaped=(args.out/'release.json').exists(),samples=len(samples),elapsed_wall_s=time.monotonic()-started,
            independent_numeric_audit_required=True,full_model_workload_qualified=False,performance_qualified=False),exclusive=True)
    return int(bool(errors))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--plan-sha256',required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args(argv);args.plan,args.out=args.plan.resolve(),args.out.resolve()
    need(args.out.parent.is_dir() and not args.out.exists(),'fresh output in existing parent required')
    return run(args)


if __name__=='__main__':raise SystemExit(main())
