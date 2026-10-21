"""Default-OFF E058: real Logits-agent feature validation and local GPU cost."""
import argparse
import importlib.util
import json
from pathlib import Path
from types import FunctionType

R=Path('/Users/ryanli/code/minivllm');B=R/'bench/m5_ultra_256gb';A=R/'.artifacts/m5-ultra-256gb'
HERE=Path(__file__).resolve().parent
CAPTURE_PLAN_SHA='9ef4dda5a8cc95b7ab53df20a0e88273ff7a2d5ba03d3781b0ed1d66f0577ac4'
SCREEN=B/'rowwise_mlp_shared_cost_v1/screen.py'
SCREEN_SHA='2922675fca185ca0fd04f3da6d22c96268fe4b41d66bdd305a2ec914f778be9c'

def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

previous=load(B/'gdn_batched_device_v1/execute.py','_e058_existing_adapter')
outer=previous.outer
runner=load(HERE/'runner.py','_e058_feature_owner')
sha,write=runner.sha,runner.write
preserve_sources=previous.preserve_sources

def capture_predecessor():
    assert isinstance(CAPTURE_PLAN_SHA,str) and len(CAPTURE_PLAN_SHA)==64 and all(c in '0123456789abcdef' for c in CAPTURE_PLAN_SHA), 'E057 plan remains unsealed'
    path=B/'rowwise_mlp_shared_cost_device_v1/prereg-v1/plan.json'
    assert sha(path)==CAPTURE_PLAN_SHA, 'actual E057 predecessor plan'
    return path

def prepare():
    predecessor=capture_predecessor();prior=json.loads(predecessor.read_text())
    outputs=prior['outputs'];lp=Path(outputs['lease']);ep=Path(outputs['execution'])
    lease,job,resume,final=[json.loads(p.read_text()) for p in (lp/'lease.json',lp/'job-0.json',lp/'resume.json',ep/'final.json')]
    assert lease['closed'] is True and job['waited'] is True and job['returncode']==0
    assert job['pid']==lease['child_pid'] and job['command']==lease['jobs'][0]
    assert lease['service_pid']==outer.helper.PID and ' '.join(lease['service_identity'].split())==outer.helper.IDENTITY
    assert resume['identity_match'] is True and resume['state_after'] and 'T' not in resume['state_after']
    assert final['status']=='PASS_CAPTURE_AND_RESTORE' and final['service_identity_after']==outer.helper.IDENTITY
    assert runner.process(job['pid']) is None and runner.process(lease['owner_pid']) is None
    assert sha(SCREEN)==SCREEN_SHA, 'frozen shared MLP cost adapter'
    predecessor_result=Path(outputs['numerical'])/'complete.json'
    predecessor_plan=Path(outputs['numerical'])/'plan.json'
    audit=A/'c1-mlp-shared-cost-v1-independent.json'
    resource_review=R/'docs/experiments/m5-ultra-256gb/E057-c1-mlp-shared-full-cost-v1/resource-review.json'
    observed=json.loads(resource_review.read_text())
    assert observed['status']=='PASS_SAVED_OBSERVATIONS' and observed['recorded_service_restored'] is True
    assert json.loads(audit.read_text())['parent_plan_sha256']==CAPTURE_PLAN_SHA
    def ref(p):return dict(path=str(p),sha256=sha(p))
    sources={str(p):sha(p) for p in HERE.glob('*.py')}
    for folder in ('rowwise_mlp_shared_features','nvfp4_multirow_shared'):
        sources.update({str(p):sha(p) for p in (B/folder).glob('*.py')})
    screen=load(HERE/'screen.py','_e058_feature_screen')
    child=dict(schema='E058-feature-plan-v1',deadline_s=1500,sources=sources,
        e057_plan=ref(predecessor_plan),e057_result=ref(predecessor_result),e057_audit=ref(audit))
    cost,candidate,old_plan,q,deps,seed=screen.validate(child)
    inherited=deps[-1]
    pre=HERE/'prereg-v1';pre.mkdir();child_path=pre/'child-plan.json';write(child_path,child)
    pins=dict(prior['pins']);pins.update(sources)
    for p in (predecessor,predecessor_plan,predecessor_result,audit,resource_review,child_path,
              lp/'lease.json',lp/'job-0.json',lp/'resume.json',ep/'final.json'):
        pins[str(p)]=sha(p)
    runner.verify_pins(pins)
    service=dict(pid=outer.helper.PID,identity=outer.helper.IDENTITY,process=runner.process(outer.helper.PID))
    assert outer.helper.identity()==outer.helper.IDENTITY and service['process'];outer.helper.health()
    outputs={k:str(A/f'e058-agent-features-v1-{v}') for k,v in
        (('execution','execution'),('screen','owner'),('lease','lease'),('numerical','numerical'))}
    command=[str(R/'.venv/bin/python'),'-I','-B',str(HERE/'screen.py'),'--run','--plan',str(child_path),
             '--plan-sha256',sha(child_path),'--out',outputs['numerical']]
    plan=dict(schema='E058-agent-feature-owner-plan-v1',pins=pins,runtime_pins=inherited['execution']['pins'],
        numerical_command=command,child_environment=inherited['execution']['environment'],service=service,
        numerical_plan_sha256=sha(child_path),screen_deadline_seconds=1650,driver_deadline_seconds=1800,
        max_observation_gap_seconds=12,physical=prior['physical'],outputs=outputs,
        scope='Real T4/T8, all rejection prefixes and continuations, five-layer features and complete raw state. Teacher-forced verifier/capture local cost only; no draft, actual acceptance, HTTP or Pareto claim.')
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
    else:print('DESCRIPTION ONLY: E058 agent features, bounded owner and restore; no engine performance claim.')
