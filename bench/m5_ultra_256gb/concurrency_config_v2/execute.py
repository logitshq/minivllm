"""Prepare/run one bounded workload-specific configuration screen and restore oMLX."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

R=Path('/Users/ryanli/code/minivllm');B=R/'bench/m5_ultra_256gb';A=R/'.artifacts/m5-ultra-256gb';HERE=Path(__file__).resolve().parent

def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
runner=load(HERE/'runner.py','config_screen_runner')
helper=load(B/'nvfp4_m8_register/real-prereg-v2/execute.py','existing_service_identity')
sha,write=runner.sha,runner.write

def prepare():
    trial=load(HERE/'trial.py','config_screen_trial_prepare')
    p=HERE/'prereg-v1';p.mkdir();(p/'specs').mkdir()
    refs=[];pins={}
    for c,arm,values in ((1,'on',[50,200,16,800,50]),(8,'off',[50,800,16,200,50])):
        base=A/f'gdn-graph-combined-formal-prepare-v1/specs/r1-c{c}-{arm}.json'
        ref=dict(path=str(base),sha256=sha(base));pins[str(base)]=ref['sha256']
        for i,mb in enumerate(values):
            label=f'c{c}-'+('start50' if i==0 else 'end50' if i==4 else f'mb{mb}')
            spec=trial.prepare(ref,mb,label);path=p/'specs'/(label+'.json');write(path,spec)
            refs.append(dict(path=str(path),sha256=sha(path)));pins[str(path)]=sha(path)
            pins.update(spec['sources'])
    for path in [*HERE.glob('*.py'),B/'suspended_service.py',B/'resources_v2.py',B/'resources.py',B/'resource_base.py',B/'glm_flash/physical_guard.py',B/'nvfp4_m8_register/real-prereg-v2/execute.py']:
        pins[str(path)]=sha(path)
    numeric=A/'realistic-qualification-v1-inputs/plan.json';q=json.loads(numeric.read_text());pins[str(numeric)]=sha(numeric)
    for ref in (q['source_manifest'],q['model_manifest']):pins[ref['path']]=ref['sha256']
    service=dict(pid=helper.PID,identity=helper.IDENTITY,process=runner.process(helper.PID));assert helper.identity()==helper.IDENTITY
    plan=dict(schema='E045-concurrency-config-screen-plan-v1',pins=pins,specs=refs,service=service,numeric_plan=str(numeric),
        screen_deadline_seconds=1800,driver_deadline_seconds=1950,max_observation_gap_seconds=12,
        physical=dict(child_limit_gib=64,system_limit_gib=240,swap_growth_bytes=0,swapouts_growth=0,probe_seconds=5,interval_seconds=3,hard_allocation_cap=False),
        outputs=dict(execution=str(A/'concurrency-config-v2-execution'),screen=str(A/'concurrency-config-v2-screen'),lease=str(A/'concurrency-config-v2-lease')),
        scope='One nominal MB threshold axis16/50/200/800 at fixedops50,BFS20,sync0. C1 GDN+graph; C8 samecombinedentry graphonly. Screening only, no full-state or frontier qualification.')
    write(p/'plan.json',plan);print(json.dumps(dict(status='PREPARED',plan_sha256=sha(p/'plan.json'),trials=len(refs))))

def verify_payloads(plan,out):
    q=json.loads(Path(plan['numeric_plan']).read_text());pins=dict(q['execution']['pins'])
    manifest=json.loads(Path(q['model_manifest']['path']).read_text())
    pins.update({str(Path(q['model'])/name):digest for name,digest in manifest['files'].items()})
    sources=json.loads(Path(q['source_manifest']['path']).read_text())
    pins.update({str(Path(q['package_root'])/'tutorial_code/miniengine'/name):digest for name,digest in sources['files'].items()})
    observed={}
    for path,digest in pins.items():
        f=Path(path);before=f.stat();actual=sha(f);after=f.stat()
        assert actual==digest and (before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns),'payload changed: '+path
        observed[path]=dict(sha256=actual,bytes=after.st_size,inode=after.st_ino,mtime_ns=after.st_mtime_ns)
    write(out/'verified-runtime-model-sources.json',observed)

def bind_runner(lease, plan, previous=None):
    path=Path(plan['outputs']['lease'])/'lease.json'
    if not path.exists():return previous
    row=json.loads(path.read_text());pid=row.get('child_pid')
    assert row['owner_pid']==lease.pid and row['jobs']==[plan['screen_command']],'lease provenance'
    if not pid:return previous
    current=runner.process(pid)
    if current is None:return previous  # Natural exit: allow lease finally to restore.
    assert current['ppid']==lease.pid and (previous is None or current==previous),'runner identity changed'
    return current

def bounded_cleanup(lease, plan, bound):
    """Unwind the runner before its lease; never interrupt lease restoration."""
    actions=[]
    bound=bind_runner(lease,plan,bound)
    if bound and runner.process(bound['pid']) is not None:
        assert runner.process(bound['pid'])==bound,'cleanup runner identity changed'
        os.kill(bound['pid'],signal.SIGTERM);actions.append(dict(pid=bound['pid'],signal='TERM'))
    try:lease.wait(timeout=90);return actions
    except subprocess.TimeoutExpired:pass
    assert bound,'unresolved runner without bound identity'
    # Only identities saved by the owned runner's post-spawn checkpoint qualify.
    for path in sorted((Path(plan['outputs']['screen'])/'resources').glob('identity-*.json')):
        expected=json.loads(path.read_text());current=runner.process(expected['pid'])
        if current is None:continue
        assert expected['ppid']==bound['pid'] and current==expected,'cleanup server identity changed'
        for sig,budget in ((signal.SIGTERM,15),(signal.SIGKILL,15)):
            current=runner.process(expected['pid'])
            if current is None:break
            assert current==expected,'cleanup server identity changed'
            os.kill(expected['pid'],sig);actions.append(dict(pid=expected['pid'],signal=sig.name))
            deadline=time.monotonic()+budget
            while runner.process(expected['pid'])==expected and time.monotonic()<deadline:time.sleep(.2)
        assert runner.process(expected['pid']) is None,'server remains; retain runner ownership'
    current=runner.process(bound['pid'])
    if current is not None:
        assert current==bound,'cleanup runner identity changed'
        children=subprocess.run(['/bin/ps','-axo','pid=,ppid='],capture_output=True,text=True,check=True,timeout=5)
        assert not any(int(row.split()[1])==bound['pid'] for row in children.stdout.splitlines() if row.strip()),'unresolved runner child'
        os.kill(bound['pid'],signal.SIGKILL);actions.append(dict(pid=bound['pid'],signal='KILL'))
    lease.wait(timeout=20)
    return actions

def run(expected):
    path=HERE/'prereg-v1/plan.json';assert sha(path)==expected;plan=json.loads(path.read_text());runner.verify_pins(plan['pins'])
    out=Path(plan['outputs']['execution']);out.mkdir();write(out/'plan.json',plan)
    assert helper.identity()==plan['service']['identity'] and runner.process(helper.PID)==plan['service']['process']
    write(out/'before-health.json',helper.health());verify_payloads(plan,out)
    runner.verify_pins(plan['pins'])
    limits=runner.physical.policy(child_limit_bytes=64*(1<<30),work_seconds=1800,interval_seconds=3)
    admission=runner.physical.measure(out/'physical-before',os.getpid());baseline=runner.physical.checked(admission,limits)
    actual={**plan,'physical_baseline':baseline};write(out/'execution-plan.json',actual);digest=sha(out/'execution-plan.json')
    command=[str(R/'.venv/bin/python'),'-I','-B',str(HERE/'runner.py'),'--plan',str(out/'execution-plan.json'),'--plan-sha256',digest,'--out',plan['outputs']['screen']]
    write(out/'jobs.json',[command]);argv=[str(R/'.venv/bin/python'),'-I','-B',str(B/'suspended_service.py'),'--pid',str(helper.PID),'--jobs',str(out/'jobs.json'),'--out',plan['outputs']['lease']]
    write(out/'invocation.json',dict(argv=argv,started_at=time.time(),plan_sha256=expected))
    lease=None;bound=None;errors=[];started=time.monotonic();cleanup_plan={**plan,'screen_command':command}
    def stop(*unused):raise InterruptedError('configuration owner interrupted')
    signal.signal(signal.SIGTERM,stop)
    try:
        with (out/'stdout.txt').open('x') as stdout,(out/'stderr.txt').open('x') as stderr:
            lease=subprocess.Popen(argv,cwd=R,stdout=stdout,stderr=stderr)
            while lease.poll() is None:
                assert time.monotonic()-started<plan['driver_deadline_seconds'],'driver absolute deadline'
                current=bind_runner(lease,cleanup_plan,bound)
                if bound is None and current is not None:write(out/'runner-identity.json',current)
                bound=current
                time.sleep(.5)
            assert lease.returncode==0,'screen/lease failed'
    except BaseException as exc:errors.append(dict(type=type(exc).__name__,message=str(exc)))
    finally:
        if lease is not None and lease.poll() is None:
            try:write(out/'forced-cleanup.json',dict(actions=bounded_cleanup(lease,cleanup_plan,bound)))
            except BaseException as exc:errors.append(dict(type=type(exc).__name__,message='UNRESOLVED cleanup: '+str(exc)))
        try:
            lp=Path(plan['outputs']['lease'])/'lease.json';l=json.loads(lp.read_text());job=json.loads(lp.with_name('job-0.json').read_text())
            assert l['closed'] is True and job['waited'] is True and job['command']==command
            assert bound and job['pid']==bound['pid'] and bound['ppid']==lease.pid,'bound runner wait receipt'
            assert runner.process(job['pid']) is None and runner.process(lease.pid) is None,'owned processes remain'
            resume=json.loads(lp.with_name('resume.json').read_text());assert resume['identity_match'] and resume['state_after'] and 'T' not in resume['state_after']
            assert helper.identity()==plan['service']['identity']
            write(out/'after-health.json',helper.health())
            assert job['returncode']==0 and json.loads((Path(plan['outputs']['screen'])/'result.json').read_text())['status']=='PASS_CAPTURE_ONLY'
            runner.verify_pins(plan['pins'])
        except BaseException as exc:errors.append(dict(type=type(exc).__name__,message='closure: '+str(exc)))
        result=dict(schema='E045-config-driver-result-v1',status='PASS_CAPTURE_AND_RESTORE' if not errors else 'FAIL',errors=errors,
            lease_returncode=None if lease is None else lease.returncode,service_identity_after=helper.identity(),elapsed_seconds=time.monotonic()-started,
            full_state_qualified=False,performance_qualified=False,engine_pareto_point=False)
        write(out/'final.json',result);print(json.dumps(result),flush=True)
    return 0 if not errors else 1
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group();g.add_argument('--prepare',action='store_true');g.add_argument('--run',action='store_true');p.add_argument('--plan-sha256');a=p.parse_args()
    if a.prepare:prepare()
    elif a.run:raise SystemExit(run(a.plan_sha256))
    else:print('DESCRIPTION ONLY: explicit prepare then run requires frozen plan SHA.')
