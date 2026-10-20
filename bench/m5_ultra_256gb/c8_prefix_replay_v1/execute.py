"""Default-OFF E049 warm49 replay owner; original history is incomplete by design."""
import argparse
import importlib.util
import json
from pathlib import Path
from types import FunctionType

R=Path('/Users/ryanli/code/minivllm');B=R/'bench/m5_ultra_256gb';A=R/'.artifacts/m5-ultra-256gb'
HERE=Path(__file__).resolve().parent

def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

previous=load(B/'gdn_batched_device_v1/execute.py','_e049_existing_adapter')
outer=previous.outer
runner=load(HERE/'runner.py','_e049_prefix_owner')
sha,write=runner.sha,runner.write
preserve_sources=previous.preserve_sources


def prepare():
    predecessor=B/'c8_schedule_diagnostic_v1/prereg-v1/plan.json'
    assert sha(predecessor)=='6d31b9f9efbdb453b02675c3d0fcba666f49b48b3e1b8fc7aca8189f42fae9b7'
    outputs=json.loads(predecessor.read_text())['outputs'];lp=Path(outputs['lease']);ep=Path(outputs['execution'])
    lease,job,resume,final=[json.loads(p.read_text()) for p in (lp/'lease.json',lp/'job-0.json',lp/'resume.json',ep/'final.json')]
    assert lease['closed'] is True and job['waited'] is True and job['returncode']==0
    assert job['pid']==lease['child_pid'] and job['command']==lease['jobs'][0]
    assert resume['identity_match'] is True and resume['state_after'] and 'T' not in resume['state_after']
    assert final['status']=='PASS_CAPTURE_AND_RESTORE' and final['service_identity_after']==outer.helper.IDENTITY
    assert runner.process(job['pid']) is None and runner.process(lease['owner_pid']) is None
    q=load(HERE/'prefix.py','_e049_prefix_contract')
    child=q.prepare(Path(outputs['screen'])/'trials/c8-mb50-two-waves',A/'c8-schedule-v1-analysis/analysis.json')
    pre=HERE/'prereg-v1';pre.mkdir();child_path=pre/'child-plan.json';q.write(child_path,child)
    prior_path=B/'gdn_batched_device_v1/prereg-v1/plan.json'
    assert sha(prior_path)=='08325135360d35faa5bc0b9294aca21ee4dc8fa7c974c49b752682184bc13e3b'
    prior=json.loads(prior_path.read_text())
    pins={**prior['pins'],**child['sources'],**child['inputs']}
    for ref in (child['qualification'],child['recording'],child['analysis']):pins[ref['path']]=ref['sha256']
    for path in [*HERE.glob('*.py'),predecessor,prior_path,B/'gdn_batched_device_v1/execute.py',
                 lp/'lease.json',lp/'job-0.json',lp/'resume.json',ep/'final.json',child_path]:pins[str(path)]=sha(path)
    runner.verify_pins(pins)
    service=dict(pid=outer.helper.PID,identity=outer.helper.IDENTITY,process=runner.process(outer.helper.PID))
    assert outer.helper.identity()==outer.helper.IDENTITY and service['process'];outer.helper.health()
    outputs=dict(execution=str(A/'c8-prefix-v1-execution'),screen=str(A/'c8-prefix-v1-owner'),
                 lease=str(A/'c8-prefix-v1-lease'),numerical=str(A/'c8-prefix-v1-numerical'))
    command=[str(R/'.venv/bin/python'),'-I','-B',str(HERE/'prefix.py'),'--run','--plan',str(child_path),
             '--plan-sha256',sha(child_path),'--out',outputs['numerical']]
    plan=dict(schema='E049-c8-prefix-owner-plan-v1',pins=pins,runtime_pins=prior['runtime_pins'],
        numerical_command=command,child_environment=child['environment'],service=service,
        prefix_plan_sha256=sha(child_path),prefix_rid=child['recipe']['rid'],
        screen_deadline_seconds=1650,driver_deadline_seconds=1800,max_observation_gap_seconds=12,
        physical=prior['physical'],outputs=outputs,scope=child['scope'])
    write(pre/'plan.json',plan)
    command=[str(R/'.venv/bin/python'),'-I','-B',str(HERE/'execute.py'),'--run','--plan-sha256',sha(pre/'plan.json')]
    write(pre/'commands.json',dict(run=command))
    print(json.dumps(dict(status='PREPARED_NOT_RUN',plan_sha256=sha(pre/'plan.json'),child_plan_sha256=sha(child_path),command=command)))


def run(expected):
    scope=dict(vars(outer),HERE=HERE,runner=runner,verify_payloads=preserve_sources)
    for name in ('bind_runner','bounded_cleanup','run'):
        fn=getattr(outer,name);scope[name]=FunctionType(fn.__code__,scope,name,fn.__defaults__,fn.__closure__)
    return scope['run'](expected)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group()
    g.add_argument('--prepare',action='store_true');g.add_argument('--run',action='store_true');p.add_argument('--plan-sha256');a=p.parse_args()
    if a.prepare:prepare()
    elif a.run:raise SystemExit(run(a.plan_sha256))
    else:print('DESCRIPTION ONLY: reviewed warm49 numerical prefix with bounded restore; no full-history claim.')
