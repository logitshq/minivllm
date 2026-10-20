"""Default-OFF E056 C1 shared MLP activation canary owner."""
import argparse
import importlib.util
import json
from pathlib import Path
from types import FunctionType

R=Path('/Users/ryanli/code/minivllm');B=R/'bench/m5_ultra_256gb';A=R/'.artifacts/m5-ultra-256gb'
HERE=Path(__file__).resolve().parent
CAPTURE_PLAN_SHA='fc24144f4c0676ca6de3aa26f93bb6eabb2976d8addbc403ec35e228b484ed4b'
CANARY=B/'rowwise_mlp_shared/canary.py'
CANARY_SHA='c1aeae682a5822a0483926fb588f3e184d307a1aeb7fc3a2380a0d16bf85ffd8'

def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

previous=load(B/'gdn_batched_device_v1/execute.py','_e056_existing_adapter')
outer=previous.outer
runner=load(HERE/'runner.py','_e056_prefix_owner')
sha,write=runner.sha,runner.write
preserve_sources=previous.preserve_sources


def capture_predecessor():
    assert isinstance(CAPTURE_PLAN_SHA,str) and len(CAPTURE_PLAN_SHA)==64 and all(c in '0123456789abcdef' for c in CAPTURE_PLAN_SHA), 'E055 plan remains unsealed'
    path=B/'c8_batched_formal_v1/prereg-v1/plan.json'
    assert sha(path)==CAPTURE_PLAN_SHA, 'actual E055 predecessor plan'
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
    assert sha(CANARY)==CANARY_SHA, 'frozen MLP activation canary'
    canary=load(CANARY,'_e056_canary_contract')
    e054=A/'c1-verifier-cost-v1-numerical';old_plan=e054/'plan.json';old_result=e054/'complete.json'
    old_audit=A/'c1-verifier-cost-v1-independent-v2.json'
    child=dict(schema='C1-MLP-shared-activation-plan-v1',canary_sha256=CANARY_SHA,
        e054_plan=dict(path=str(old_plan),sha256=sha(old_plan)),e054_result=dict(path=str(old_result),sha256=sha(old_result)),
        e054_audit=dict(path=str(old_audit),sha256=sha(old_audit)),deadline_s=1500)
    screen,q,deps,seed,candidate,shared,inherited_cost=canary.validate(child)
    bridge,frozen,h,old,support,original,inherited=deps
    pre=HERE/'prereg-v1';pre.mkdir();child_path=pre/'child-plan.json';canary.write(child_path,child)
    previous_plan=B/'gdn_batched_device_v1/prereg-v1/plan.json'
    assert sha(previous_plan)=='08325135360d35faa5bc0b9294aca21ee4dc8fa7c974c49b752682184bc13e3b'
    prior=json.loads(previous_plan.read_text());pins=dict(prior['pins'])
    extra=[*HERE.glob('*.py'),predecessor,previous_plan,old_plan,old_result,old_audit,child_path,CANARY,canary.SCREEN,canary.CANDIDATE,canary.SHARED,q.HEAD,screen.QUALIFIER,
           lp/'lease.json',lp/'job-0.json',lp/'resume.json',ep/'final.json']
    extra.extend(Path(m.__file__).resolve() for m in (screen,q,bridge,frozen,h,old,support,candidate,shared))
    for p in extra:pins[str(p)]=sha(p)
    for key in ('source_manifest','model_manifest','fixtures'):
        item=inherited[key];pins[item['path']]=item['sha256']
    runner.verify_pins(pins)
    service=dict(pid=outer.helper.PID,identity=outer.helper.IDENTITY,process=runner.process(outer.helper.PID))
    assert outer.helper.identity()==outer.helper.IDENTITY and service['process'];outer.helper.health()
    outputs=dict(execution=str(A/'c1-mlp-shared-canary-v1-execution'),screen=str(A/'c1-mlp-shared-canary-v1-owner'),
                 lease=str(A/'c1-mlp-shared-canary-v1-lease'),numerical=str(A/'c1-mlp-shared-canary-v1-numerical'))
    command=[str(R/'.venv/bin/python'),'-I','-B',str(CANARY),'--run','--plan',str(child_path),
             '--plan-sha256',sha(child_path),'--out',outputs['numerical']]
    plan=dict(schema='E056-c1-mlp-shared-canary-owner-plan-v1',pins=pins,runtime_pins=inherited['execution']['pins'],
        numerical_command=command,child_environment=inherited['execution']['environment'],service=service,
        numerical_plan_sha256=sha(child_path),screen_deadline_seconds=1650,driver_deadline_seconds=1800,
        max_observation_gap_seconds=12,physical=prior['physical'],outputs=outputs,
        scope='C1 used1023 T4/T8 real normalized MLP activation shadow canary: 12 projection pairs and native-returning traversal provenance only; shared shadow outputs do not feed the model; no timing or engine performance claim.')
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
    else:print('DESCRIPTION ONLY: reviewed C1 MLP activation shadow canary with bounded restore; no engine performance claim.')
