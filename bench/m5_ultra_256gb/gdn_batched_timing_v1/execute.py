"""Default OFF E048 real-activation local timing, using the E046 resource owner."""
import argparse
import json
from pathlib import Path
from types import FunctionType
import importlib.util

R = Path('/Users/ryanli/code/minivllm'); B = R/'bench/m5_ultra_256gb'; A = R/'.artifacts/m5-ultra-256gb'
HERE = Path(__file__).resolve().parent


def load(path, name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


previous = load(B/'gdn_batched_device_v1/execute.py', '_e048_e046_adapter')
outer = previous.outer
runner = load(HERE/'runner.py', '_e048_timing_owner')
sha, write = runner.sha, runner.write
preserve_sources = previous.preserve_sources


def prepare():
    predecessor=B/'gdn_batched_device_v1/prereg-v1/plan.json'
    assert sha(predecessor)=='08325135360d35faa5bc0b9294aca21ee4dc8fa7c974c49b752682184bc13e3b'
    original=json.loads(predecessor.read_text()); paths=original['outputs']
    lease_dir=Path(paths['lease']); execution=Path(paths['execution'])
    lease,job,resume,final=[json.loads(p.read_text()) for p in
        (lease_dir/'lease.json',lease_dir/'job-0.json',lease_dir/'resume.json',execution/'final.json')]
    assert lease['closed'] is True and job['waited'] is True and job['returncode']==0
    assert job['pid']==lease['child_pid'] and job['command']==lease['jobs'][0]
    assert resume['identity_match'] is True and resume['state_after'] and 'T' not in resume['state_after']
    assert final['status']=='PASS_CAPTURE_AND_RESTORE' and final['service_identity_after']==outer.helper.IDENTITY
    assert runner.process(job['pid']) is None and runner.process(lease['owner_pid']) is None
    audit_path=A/'gdn-batched-v1-independent-numerical.json'
    audit=json.loads(audit_path.read_text())
    assert audit['status']=='PASS' and audit['canary_qualified'] is True and audit['raw_files']==140
    numerical=Path(paths['numerical'])
    for name,key in [('captures.json','captures_sha256'),('complete.json','complete_sha256'),('events.jsonl','ledger_sha256')]:
        assert sha(numerical/name)==audit[key]
    child=B/'gdn_batched_device_v1/prereg-v1/child-plan.json'
    assert sha(child)=='7df6cff99425698a1fb134df4f25dc8de4e717482f73039e56ec5a0a00930003'
    pins=dict(original['pins'])
    for path in [*HERE.glob('*.py'), predecessor, B/'gdn_batched_device_v1/execute.py',
                 lease_dir/'lease.json',lease_dir/'job-0.json',lease_dir/'resume.json',execution/'final.json',
                 audit_path,numerical/'captures.json',numerical/'complete.json',numerical/'events.jsonl']:
        pins[str(path)]=sha(path)
    runner.verify_pins(pins)
    service=dict(pid=outer.helper.PID,identity=outer.helper.IDENTITY,process=runner.process(outer.helper.PID))
    assert outer.helper.identity()==outer.helper.IDENTITY and service['process']; outer.helper.health()
    outputs=dict(execution=str(A/'gdn-batched-timing-v1-execution'),screen=str(A/'gdn-batched-timing-v1-owner'),
        lease=str(A/'gdn-batched-timing-v1-lease'),numerical=str(A/'gdn-batched-timing-v1-numerical'))
    command=[str(R/'.venv/bin/python'),'-I','-B',str(HERE/'screen.py'),'--run',
             '--plan',str(child),'--plan-sha256',sha(child),'--out',outputs['numerical']]
    plan=dict(schema='E048-gdn-batched-local-timing-plan-v1',pins=pins,runtime_pins=original['runtime_pins'],
        numerical_command=command,child_environment=original['child_environment'],service=service,
        screen_deadline_seconds=1650,driver_deadline_seconds=1800,max_observation_gap_seconds=12,
        physical=original['physical'],outputs=outputs,
        scope='12 real-activation B2/B8 local GDN cases, native batched baseline, 1200 paired timings; no engine/HTTP/compiled performance qualification.')
    pre=HERE/'prereg-v1';pre.mkdir();write(pre/'plan.json',plan)
    commands=dict(run=[str(R/'.venv/bin/python'),'-I','-B',str(HERE/'execute.py'),'--run','--plan-sha256',sha(pre/'plan.json')])
    write(pre/'commands.json',commands)
    print(json.dumps(dict(status='PREPARED_NOT_RUN',plan_sha256=sha(pre/'plan.json'),commands=commands)))


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
    else:print('DESCRIPTION ONLY: prepare/review/commit followed by explicit plan-bound run.')
