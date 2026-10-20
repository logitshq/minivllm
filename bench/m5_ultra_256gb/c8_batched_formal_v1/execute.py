"""Default-OFF E055 paired C8 HTTP throughput/latency owner."""
import argparse
import importlib.util
import json
from pathlib import Path
from types import FunctionType

HERE=Path(__file__).resolve().parent;B=HERE.parent;R=B.parent.parent;A=R/'.artifacts/m5-ultra-256gb'
def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

outer=load(B/'concurrency_config_v2/execute.py','_e055_pause_owner')
runner=load(HERE/'runner.py','_e055_resource_owner')
sha,write=runner.sha,runner.write
BASE_SHA='34a604c62f6de4eddcff2be53abd2d72716384ee67d13d1bf963869d35695a20'
assert sha(B/'concurrency_config_v2/execute.py')==BASE_SHA


def prepare():
    predecessor=B/'c8_batched_http_replay_v1/prereg-v1/plan.json'
    assert sha(predecessor)=='4a2d6cc768545654a7931ea4dbc3afc8837b0bdc16a159021c9ade2f0104de47'
    prior=json.loads(predecessor.read_text());outputs=prior['outputs']
    lp=Path(outputs['lease']);ep=Path(outputs['execution'])
    lease,job,resume,final=[json.loads(p.read_text()) for p in
        (lp/'lease.json',lp/'job-0.json',lp/'resume.json',ep/'final.json')]
    assert lease['closed'] is True and job['waited'] is True and job['returncode']==0
    assert job['pid']==lease['child_pid'] and job['command']==lease['jobs'][0]
    assert resume['identity_match'] is True and resume['state_after'] and 'T' not in resume['state_after']
    assert final['status']=='PASS_CAPTURE_AND_RESTORE' and final['service_identity_after']==outer.helper.IDENTITY
    assert runner.process(job['pid']) is None and runner.process(lease['owner_pid']) is None
    resource=R/'docs/experiments/m5-ultra-256gb/E053-c8-batched-http-replay-v1/resource-review.json'
    assert json.loads(resource.read_text())['status']=='PASS_SAVED_OBSERVATIONS'
    audit=A/'c8-batched-http-replay-v1-independent-audit-attempt1'
    child=Path(outputs['numerical']);childplan=child/'plan.json'
    assert sha(childplan)==prior['numerical_plan_sha256']
    binding=dict(audit=dict(path=str(audit),sha256=sha(audit)),
                 replay_plan=dict(path=str(childplan),sha256=sha(childplan)),child_directory=str(child))
    trial=load(HERE/'trial.py','_e055_prepare_trial')
    pre=HERE/'prereg-v1';pre.mkdir();(pre/'specs').mkdir();refs=[];pins={}
    for repeat,arms in ((1,('on','off')),(2,('off','on')),(3,('on','off'))):
        base=B/f'gdn_graph_combined_formal/prereg-formal-v1/specs/r{repeat}-c8-off.json'
        base_ref=dict(path=str(base),sha256=sha(base));pins[str(base)]=sha(base)
        for arm in arms:
            label=f'r{repeat}-c8-{arm}'
            spec=trial.prepare(base_ref,arm,label,binding)
            path=pre/'specs'/(label+'.json');write(path,spec)
            refs.append(dict(path=str(path),sha256=sha(path)));pins[str(path)]=sha(path);pins.update(spec['sources'])
    for p in [*HERE.glob('*.py'),B/'concurrency_config_v2/execute.py',B/'concurrency_config_v2/runner.py',
              B/'nvfp4_m8_register/real-prereg-v2/execute.py',B/'suspended_service.py',
              B/'resources_v2.py',B/'resources.py',B/'resource_base.py',B/'glm_flash/physical_guard.py',
              predecessor,resource,audit,childplan,lp/'lease.json',lp/'job-0.json',lp/'resume.json',ep/'final.json']:
        pins[str(p)]=sha(p)
    numeric=Path(spec['base_plan']['path']);pins[str(numeric)]=spec['base_plan']['sha256']
    q=json.loads(numeric.read_text())
    for ref in (q['source_manifest'],q['model_manifest'],q['fixtures']):pins[ref['path']]=ref['sha256']
    runner.verify_pins(pins)
    service=dict(pid=outer.helper.PID,identity=outer.helper.IDENTITY,process=runner.process(outer.helper.PID))
    assert outer.helper.identity()==outer.helper.IDENTITY and service['process'];outer.helper.health()
    plan=dict(schema='E055-c8-batched-formal-plan-v1',pins=pins,specs=refs,service=service,numeric_plan=str(numeric),
        screen_deadline_seconds=1740,driver_deadline_seconds=1890,max_observation_gap_seconds=12,
        physical=dict(child_limit_gib=64,system_limit_gib=240,swap_growth_bytes=0,swapouts_growth=0,
                      probe_seconds=5,interval_seconds=3,hard_allocation_cap=False),
        outputs=dict(execution=str(A/'c8-batched-formal-v1-execution'),screen=str(A/'c8-batched-formal-v1-screen'),
                     lease=str(A/'c8-batched-formal-v1-lease')),
        decision=dict(min_median_throughput_ratio=1.03,max_median_p50_ratio=.97,max_median_p95_ratio=1.02,
                      all_pair_throughput_above_one=True,all_pair_p50_below_one=True,max_off_rate_spread_fraction=.02),
        scope='Six fresh C8 processes,8 warm+64 formal each; observed OFF/ON candidate A/B only. Independent metrics/output/resource audit required; no automatic Pareto qualification.')
    write(pre/'plan.json',plan)
    command=[str(R/'.venv/bin/python'),'-I','-B',str(HERE/'execute.py'),'--run','--plan-sha256',sha(pre/'plan.json')]
    write(pre/'commands.json',dict(run=command))
    print(json.dumps(dict(status='PREPARED_NOT_RUN',plan_sha256=sha(pre/'plan.json'),trials=len(refs),command=command)))


def run(expected):
    scope=dict(vars(outer),HERE=HERE,runner=runner)
    for name in ('bind_runner','bounded_cleanup','run'):
        fn=getattr(outer,name);scope[name]=FunctionType(fn.__code__,scope,name,fn.__defaults__,fn.__closure__)
    return scope['run'](expected)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group()
    g.add_argument('--prepare',action='store_true');g.add_argument('--run',action='store_true');p.add_argument('--plan-sha256');args=p.parse_args()
    if args.prepare:prepare()
    elif args.run:raise SystemExit(run(args.plan_sha256))
    else:print('DESCRIPTION ONLY: independent E053 audit/resource closure required before prepare or formal device run.')
