"""Default-OFF E059 bounded generation experiment; independent outputs, same service lease."""
import argparse, importlib.util, json
from pathlib import Path
from types import FunctionType
R=Path('/Users/ryanli/code/minivllm');B=R/'bench/m5_ultra_256gb';A=R/'.artifacts/m5-ultra-256gb';HERE=Path(__file__).resolve().parent

def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
previous=load(B/'e058_agent_run_v1/execute.py','_e059_previous_owner')
outer=previous.outer
runner=load(HERE/'runner.py','_e059_owner')
sha,write=runner.sha,runner.write
preserve_sources=previous.preserve_sources

def prepare():
    predecessor=B/'e058_agent_run_v1/prereg-v1/plan.json'
    assert sha(predecessor)=='de5b72f580b9dcdfe09dcf07afa7703d3080d28d6e0d3471a94c9a1469dc4ebf'
    prior=json.loads(predecessor.read_text());outputs=prior['outputs'];lp=Path(outputs['lease']);ep=Path(outputs['execution'])
    lease,job,resume,final=[json.loads(p.read_text()) for p in (lp/'lease.json',lp/'job-0.json',lp/'resume.json',ep/'final.json')]
    assert lease['closed'] and job['waited'] and job['returncode']==0
    assert runner.process(job['pid']) is None and runner.process(lease['owner_pid']) is None
    assert resume['identity_match'] is True and resume['state_after'] and 'T' not in resume['state_after']
    assert final['status']=='PASS_CAPTURE_AND_RESTORE' and final['service_identity_after']==outer.helper.IDENTITY
    def ref(p):return dict(path=str(p),sha256=sha(p))
    sources={str(p):sha(p) for p in HERE.glob('*.py')}
    for folder in ('rowwise_mlp_shared_features','nvfp4_multirow_shared'):
        sources.update({str(p):sha(p) for p in (B/folder).glob('*.py')})
    draft_plan=json.loads((A/'draft-selector-integrated-v1-r3/plan.json').read_text());draft=Path(draft_plan['drafter'])
    draft_pins={str(draft/'config.json'):draft_plan['drafter_config_sha256'],str(draft/'model.safetensors'):draft_plan['drafter_weights_sha256']}
    child=dict(schema='E059-generation-plan-v1',deadline_s=1500,sources=sources,
        e058_plan=ref(B/'e058_agent_run_v1/prereg-v1/child-plan.json'),e058_result=ref(Path(outputs['numerical'])/'complete.json'),
        e058_audit=ref(R/'docs/experiments/m5-ultra-256gb/E058-logits-agent-feature-run-v1/independent-audit.json'),
        drafter=str(draft),drafter_pins=draft_pins,
        prompts=[dict(label='calibration',context='A software engineer studies memory bandwidth, arithmetic intensity, and latency. ',question='\nWrite a short Python function to compute Fibonacci numbers iteratively.\n'),
                 dict(label='math',context='A mathematics lesson explains integers, multiplication, addition, and careful step by step reasoning. ',question='\nCompute 17 * 23 and explain the calculation briefly.\n'),
                 dict(label='chinese',context='科学教育关注观察、证据和清楚的解释。人们通过记录自然现象理解世界。',question='\n请解释月亮为什么会有阴晴圆缺。\n')],
        output_cap=32,groups=3,warmups=1,
        decision='Screening only. Qualification all IDs/full state/features first; record all24 timings. Adopt no production configuration from only two synthetic contexts. No new HTTP/Pareto claim.',
        scope='Original Qwen NVFP4 native vs real DFlash8 + E058 T4/T8, one calibration and two test1024-token contexts. Native no-capture strong runner; greedy EOS/cap. Prefill excluded from decode timings.')
    screen=load(HERE/'screen.py','_e059_screen_prepare');_,deps=screen.validate(child);inherited=deps[4][-1]
    pre=HERE/'prereg-v1';pre.mkdir();child_path=pre/'child-plan.json';write(child_path,child)
    pins=dict(prior['pins']);pins.update(sources)
    for p in (predecessor,child_path,Path(child['e058_result']['path']),Path(child['e058_audit']['path']),lp/'lease.json',lp/'job-0.json',lp/'resume.json',ep/'final.json'):
        pins[str(p)]=sha(p)
    runner.verify_pins(pins)
    service=dict(pid=outer.helper.PID,identity=outer.helper.IDENTITY,process=runner.process(outer.helper.PID));assert service['process'] and outer.helper.identity()==outer.helper.IDENTITY;outer.helper.health()
    outputs={k:str(A/f'e059-deepinfer-v1-{v}') for k,v in (('execution','execution'),('screen','owner'),('lease','lease'),('numerical','numerical'))}
    command=[str(R/'.venv/bin/python'),'-I','-B',str(HERE/'screen.py'),'--run','--plan',str(child_path),'--plan-sha256',sha(child_path),'--out',outputs['numerical']]
    plan=dict(schema='E059-generation-owner-plan-v1',pins=pins,runtime_pins=inherited['execution']['pins'],numerical_command=command,
        child_environment=inherited['execution']['environment'],service=service,numerical_plan_sha256=sha(child_path),screen_deadline_seconds=1650,driver_deadline_seconds=1800,
        max_observation_gap_seconds=12,physical=prior['physical'],outputs=outputs,scope=child['scope'])
    write(pre/'plan.json',plan);command=[str(R/'.venv/bin/python'),'-I','-B',str(HERE/'execute.py'),'--run','--plan-sha256',sha(pre/'plan.json')]
    write(pre/'commands.json',dict(run=command));print(json.dumps(dict(status='PREPARED_NOT_RUN',plan_sha256=sha(pre/'plan.json'),child_plan_sha256=sha(child_path),command=command)))

def run(expected):
    scope=dict(vars(outer),HERE=HERE,runner=runner,verify_payloads=preserve_sources)
    for name in ('bind_runner','bounded_cleanup','run'):
        fn=getattr(outer,name);scope[name]=FunctionType(fn.__code__,scope,name,fn.__defaults__,fn.__closure__)
    return scope['run'](expected)
if __name__=='__main__':
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group();g.add_argument('--prepare',action='store_true');g.add_argument('--run',action='store_true');p.add_argument('--plan-sha256');a=p.parse_args()
    if a.prepare:prepare()
    elif a.run:raise SystemExit(run(a.plan_sha256))
    else:print('Default OFF: E059 generation cost, no HTTP/frontier claim.')
