"""Default-OFF E057 C1 shared MLP+head full-verifier local cost owner."""
import argparse
import importlib.util
import json
from pathlib import Path
from types import FunctionType

R=Path('/Users/ryanli/code/minivllm');B=R/'bench/m5_ultra_256gb';A=R/'.artifacts/m5-ultra-256gb'
HERE=Path(__file__).resolve().parent
CAPTURE_PLAN_SHA='a64cbd784d913454981137895b3ad77db38648b95cc22faa8f494469f9e2d98c'
SCREEN=B/'rowwise_mlp_shared_cost_v1/screen.py'
SCREEN_SHA='2922675fca185ca0fd04f3da6d22c96268fe4b41d66bdd305a2ec914f778be9c'

def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

previous=load(B/'gdn_batched_device_v1/execute.py','_e057_existing_adapter')
outer=previous.outer
runner=load(HERE/'runner.py','_e057_prefix_owner')
sha,write=runner.sha,runner.write
preserve_sources=previous.preserve_sources


def capture_predecessor():
    assert isinstance(CAPTURE_PLAN_SHA,str) and len(CAPTURE_PLAN_SHA)==64 and all(c in '0123456789abcdef' for c in CAPTURE_PLAN_SHA), 'E056 plan remains unsealed'
    path=B/'rowwise_mlp_shared_device_v1/prereg-v1/plan.json'
    assert sha(path)==CAPTURE_PLAN_SHA, 'actual E056 predecessor plan'
    return path


def prepare():
    predecessor=capture_predecessor()
    outputs=json.loads(predecessor.read_text())['outputs'];lp=Path(outputs['lease']);ep=Path(outputs['execution'])
    lease,job,resume,final=[json.loads(p.read_text()) for p in (lp/'lease.json',lp/'job-0.json',lp/'resume.json',ep/'final.json')]
    assert lease['closed'] is True and job['waited'] is True and job['returncode']==0
    assert job['pid']==lease['child_pid'] and job['command']==lease['jobs'][0]
    assert lease['service_pid']==outer.helper.PID and ' '.join(lease['service_identity'].split())==outer.helper.IDENTITY
    assert resume['identity_match'] is True and resume['state_after'] and 'T' not in resume['state_after']
    assert final['status']=='PASS_CAPTURE_AND_RESTORE' and final['service_identity_after']==outer.helper.IDENTITY
    assert runner.process(job['pid']) is None and runner.process(lease['owner_pid']) is None
    assert sha(SCREEN)==SCREEN_SHA, 'frozen shared MLP cost adapter'
    screen=load(SCREEN,'_e057_shared_mlp_cost_contract')
    old_plan=A/'c1-verifier-cost-v1-numerical/plan.json'
    e056=A/'c1-mlp-shared-canary-v1-numerical';canary_plan=e056/'plan.json';canary_result=e056/'complete.json'
    canary_audit=A/'c1-mlp-shared-canary-v1-independent.json'
    audit=json.loads(canary_audit.read_text())
    assert outputs['numerical']==str(e056) and audit['parent_plan_sha256']==CAPTURE_PLAN_SHA, 'same E056 lifecycle and numerical audit'
    canary_resources=R/'docs/experiments/m5-ultra-256gb/E056-c1-mlp-shared-canary-v1/resource-review.json'
    resources=json.loads(canary_resources.read_text())
    assert resources['status']=='PASS_SAVED_OBSERVATIONS' and resources['recorded_service_restored'] is True, 'E056 saved physical and restore review'
    def ref(path):return dict(path=str(path),sha256=sha(path))
    child=screen.prepare(ref(old_plan),ref(canary_plan),ref(canary_result),ref(canary_audit),deadline_s=1500)
    base,candidate,inherited_cost,q,deps,seed=screen.validate(child)
    bridge,frozen,h,old,support,original,inherited=deps
    pre=HERE/'prereg-v1';pre.mkdir();child_path=pre/'child-plan.json';screen.write(child_path,child)
    previous_plan=B/'gdn_batched_device_v1/prereg-v1/plan.json'
    assert sha(previous_plan)=='08325135360d35faa5bc0b9294aca21ee4dc8fa7c974c49b752682184bc13e3b'
    prior=json.loads(previous_plan.read_text());pins=dict(prior['pins'])
    extra=[*HERE.glob('*.py'),predecessor,previous_plan,old_plan,canary_plan,canary_result,canary_audit,canary_resources,child_path,SCREEN,screen.BASE,screen.CANDIDATE,screen.CANARY,screen.AUDITOR,screen.SHARED,q.HEAD,base.QUALIFIER,
           lp/'lease.json',lp/'job-0.json',lp/'resume.json',ep/'final.json']
    extra.extend(Path(m.__file__).resolve() for m in (base,q,bridge,frozen,h,old,support,candidate))
    for p in extra:pins[str(p)]=sha(p)
    for key in ('source_manifest','model_manifest','fixtures'):
        item=inherited[key];pins[item['path']]=item['sha256']
    runner.verify_pins(pins)
    service=dict(pid=outer.helper.PID,identity=outer.helper.IDENTITY,process=runner.process(outer.helper.PID))
    assert outer.helper.identity()==outer.helper.IDENTITY and service['process'];outer.helper.health()
    outputs=dict(execution=str(A/'c1-mlp-shared-cost-v1-execution'),screen=str(A/'c1-mlp-shared-cost-v1-owner'),
                 lease=str(A/'c1-mlp-shared-cost-v1-lease'),numerical=str(A/'c1-mlp-shared-cost-v1-numerical'))
    command=[str(R/'.venv/bin/python'),'-I','-B',str(SCREEN),'--run','--plan',str(child_path),
             '--plan-sha256',sha(child_path),'--out',outputs['numerical']]
    plan=dict(schema='E057-c1-mlp-shared-cost-owner-plan-v1',pins=pins,runtime_pins=inherited['execution']['pins'],
        numerical_command=command,child_environment=inherited['execution']['environment'],service=service,
        numerical_plan_sha256=sha(child_path),screen_deadline_seconds=1650,driver_deadline_seconds=1800,
        max_observation_gap_seconds=12,physical=prior['physical'],outputs=outputs,
        scope='C1 used1023 T4/T8 full shared MLP+head verifier, selected-prefix and continuation qualification then complete local commit cost versus native causal AR; no draft, acceptance, HTTP or Pareto claim.')
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
    else:print('DESCRIPTION ONLY: reviewed C1 full-verifier cost with bounded restore; no engine performance claim.')
