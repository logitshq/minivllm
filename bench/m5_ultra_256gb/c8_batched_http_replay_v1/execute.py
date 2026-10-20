"""Default-OFF E053 actual candidate HTTP replay owner."""
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

previous=load(B/'gdn_batched_device_v1/execute.py','_e053_existing_adapter')
outer=previous.outer
runner=load(HERE/'runner.py','_e053_prefix_owner')
sha,write=runner.sha,runner.write
preserve_sources=previous.preserve_sources


CAPTURE_PLAN_SHA = '5290ddc18239d422e846f76ff4fa222b48188c6fb57bae68ea5119ec3bd9066c'

def prepare():
    predecessor=B/'c8_batched_http_v2/prereg-v1/plan.json'
    assert sha(predecessor)==CAPTURE_PLAN_SHA
    capture_plan=json.loads(predecessor.read_text());outputs=capture_plan['outputs']
    lp=Path(outputs['lease']);ep=Path(outputs['execution'])
    lease,job,resume,final=[json.loads(p.read_text()) for p in (lp/'lease.json',lp/'job-0.json',lp/'resume.json',ep/'final.json')]
    assert lease['closed'] is True and job['waited'] is True and job['returncode']==0
    assert job['pid']==lease['child_pid'] and job['command']==lease['jobs'][0]
    assert resume['identity_match'] is True and resume['state_after'] and 'T' not in resume['state_after']
    assert final['status']=='PASS_CAPTURE_AND_RESTORE' and final['service_identity_after']==outer.helper.IDENTITY
    assert runner.process(job['pid']) is None and runner.process(lease['owner_pid']) is None
    resource_path=R/'docs/experiments/m5-ultra-256gb/E052-c8-batched-http-capture-v2/resource-review.json'
    assert json.loads(resource_path.read_text())['status']=='PASS_SAVED_OBSERVATIONS'
    spec_ref=capture_plan['specs'][0];assert sha(spec_ref['path'])==spec_ref['sha256']
    spec=json.loads(Path(spec_ref['path']).read_text());capture=Path(outputs['screen'])/'trials'/spec['label']
    q=load(HERE/'qualify.py','_e053_replay_contract');child=q.prepare(capture,spec_ref['sha256'])
    pre=HERE/'prereg-v1';pre.mkdir();child_path=pre/'child-plan.json';q.write(child_path,child)
    prior_path=B/'c8_batched_full_v1/prereg-v1/plan.json';prior=json.loads(prior_path.read_text())
    assert sha(prior_path)=='c895c6e85c9c6e7ef8928f749086fd8012777a67c769b20b3cd13d7eaae4f618'
    pins={**prior['pins'],**child['sources']}
    for item in (child['qualification'],child['recording'],child['capture_spec'],child['route_events']):pins[item['path']]=item['sha256']
    for p in [*HERE.glob('*.py'),predecessor,prior_path,child_path,resource_path,
              lp/'lease.json',lp/'job-0.json',lp/'resume.json',ep/'final.json']:pins[str(p)]=sha(p)
    runner.verify_pins(pins)
    service=dict(pid=outer.helper.PID,identity=outer.helper.IDENTITY,process=runner.process(outer.helper.PID))
    assert outer.helper.identity()==outer.helper.IDENTITY and service['process'];outer.helper.health()
    outputs=dict(execution=str(A/'c8-batched-http-replay-v1-execution'),screen=str(A/'c8-batched-http-replay-v1-owner'),
                 lease=str(A/'c8-batched-http-replay-v1-lease'),numerical=str(A/'c8-batched-http-replay-v1-numerical'))
    command=[str(R/'.venv/bin/python'),'-I','-B',str(HERE/'qualify.py'),'--run','--plan',str(child_path),
             '--plan-sha256',sha(child_path),'--out',outputs['numerical']]
    plan=dict(schema='E053-c8-batched-http-replay-owner-plan-v1',pins=pins,runtime_pins=prior['runtime_pins'],
        numerical_command=command,child_environment=child['environment'],service=service,
        numerical_plan_sha256=sha(child_path),expected={k:v for k,v in child.items() if k.startswith('expected_')},
        screen_deadline_seconds=1650,driver_deadline_seconds=1800,max_observation_gap_seconds=12,
        physical=prior['physical'],outputs=outputs,scope=child['scope'],memory_rationale=prior['memory_rationale'])
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
    else:print('DESCRIPTION ONLY: reviewed actual candidate HTTP numerical history with bounded restore; no performance claim.')
