"""Default-OFF E050 warm49 replay owner; original history is incomplete by design."""
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

previous=load(B/'gdn_batched_device_v1/execute.py','_e050_existing_adapter')
outer=previous.outer
runner=load(HERE/'runner.py','_e050_prefix_owner')
sha,write=runner.sha,runner.write
preserve_sources=previous.preserve_sources


def prepare():
    predecessor=B/'c8_prefix_replay_v1/prereg-v1/plan.json'
    assert sha(predecessor)=='eb89c92e329f7bc1e5629ae05ac911457ed86bd656a92cc1fed9ac63be6099a5'
    prior=json.loads(predecessor.read_text());outputs=prior['outputs']
    lp=Path(outputs['lease']);ep=Path(outputs['execution'])
    lease,job,resume,final=[json.loads(p.read_text()) for p in (lp/'lease.json',lp/'job-0.json',lp/'resume.json',ep/'final.json')]
    assert lease['closed'] is True and job['waited'] is True and job['returncode']==0
    assert job['pid']==lease['child_pid'] and job['command']==lease['jobs'][0]
    assert resume['identity_match'] is True and resume['state_after'] and 'T' not in resume['state_after']
    assert final['status']=='PASS_CAPTURE_AND_RESTORE' and final['service_identity_after']==outer.helper.IDENTITY
    assert runner.process(job['pid']) is None and runner.process(lease['owner_pid']) is None
    audit_path=A/'c8-prefix-v1-independent-audit-attempt2/audit.json'
    audit=json.loads(audit_path.read_text())
    assert audit['schema']=='E049-independent-prefix-audit-v1' and audit['status']=='PASS'
    assert audit['prefix_receipts_verified'] is True and audit['completion_state']=='INCOMPLETE_BY_DESIGN'
    assert audit['full_history_qualified'] is False and audit['offline_tensor_bytes_rechecked'] is False
    assert audit['plan_sha256']==prior['prefix_plan_sha256']
    resource_path=R/'docs/experiments/m5-ultra-256gb/E049-c8-warm-prefix-diagnostic-v1/resource-review.json'
    resource=json.loads(resource_path.read_text());assert resource['status']=='PASS_SAVED_OBSERVATIONS'
    q=load(HERE/'prefix.py','_e050_prefix_contract')
    base_plan=predecessor.parent/'child-plan.json';actual=Path(outputs['numerical'])/'complete.json'
    child=q.prepare(base_plan,actual)
    pre=HERE/'prereg-v1';pre.mkdir();child_path=pre/'child-plan.json';q.write(child_path,child)
    pins={**prior['pins'],**child['sources']}
    for item in (child['qualification'],child['recording'],child['base_prefix_plan'],child['predecessor']):
        pins[item['path']]=item['sha256']
    # The large predecessor ledger is pinned and rehashed by the child's specification;
    # the inherited source copier retains small sources only, not large ledger payloads.
    for path in [*HERE.glob('*.py'),predecessor,base_plan,actual,audit_path,resource_path,
                 lp/'lease.json',lp/'job-0.json',lp/'resume.json',ep/'final.json',child_path]:pins[str(path)]=sha(path)
    runner.verify_pins(pins)
    service=dict(pid=outer.helper.PID,identity=outer.helper.IDENTITY,process=runner.process(outer.helper.PID))
    assert outer.helper.identity()==outer.helper.IDENTITY and service['process'];outer.helper.health()
    outputs=dict(execution=str(A/'c8-batched-prefix-v1-execution'),screen=str(A/'c8-batched-prefix-v1-owner'),
                 lease=str(A/'c8-batched-prefix-v1-lease'),numerical=str(A/'c8-batched-prefix-v1-numerical'))
    command=[str(R/'.venv/bin/python'),'-I','-B',str(HERE/'prefix.py'),'--run','--plan',str(child_path),
             '--plan-sha256',sha(child_path),'--out',outputs['numerical']]
    plan=dict(schema='E050-c8-batched-prefix-owner-plan-v1',pins=pins,runtime_pins=prior['runtime_pins'],
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
