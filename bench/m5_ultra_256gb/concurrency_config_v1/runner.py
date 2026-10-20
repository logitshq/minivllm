"""Bounded diagnostic HTTP sweep; this owner runs inside one authorized pause lease."""
import argparse
import asyncio
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

R=Path('/Users/ryanli/code/minivllm');B=R/'bench/m5_ultra_256gb';HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(B));import resources_v2 as resources
sys.path.insert(0,str(B/'glm_flash'));import physical_guard as physical

def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def write(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2);f.write('\n')

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while block:=f.read(8<<20):h.update(block)
    return h.hexdigest()

def process(pid):
    p=subprocess.run(['/bin/ps','-p',str(pid),'-o','lstart=,ppid=,comm='],capture_output=True,text=True,timeout=5)
    if p.returncode==1 and not p.stdout.strip() and not p.stderr.strip():return None
    assert p.returncode==0 and not p.stderr.strip(),'process identity unavailable'
    fields=p.stdout.split(None,6);assert len(fields)==7 and fields[5].isdigit()
    return dict(pid=pid,start=' '.join(fields[:5]),ppid=int(fields[5]),executable=fields[6].strip())

def verify_pins(pins):
    for path,digest in pins.items():assert sha(path)==digest,'source changed: '+path

async def run(plan,out):
    trial=load(HERE/'trial.py','config_screen_trial')
    verify_pins(plan['pins']);out.mkdir();(out/'trials').mkdir();(out/'resources').mkdir()
    write(out/'plan.json',plan)
    service=plan['service'];identity=process(service['pid']);assert identity==service['process']
    records=[];physical_records=[];bindings={};results=[];errors=[];started=time.monotonic()
    baseline=plan['physical_baseline'];limits=physical.policy(child_limit_bytes=64*(1<<30),work_seconds=1800,interval_seconds=3)
    def observe(label,pid):
        assert time.monotonic()-started<plan['screen_deadline_seconds'],'screen absolute deadline'
        assert process(service['pid'])==identity,'service identity changed'
        state=subprocess.run(['/bin/ps','-p',str(service['pid']),'-o','state='],capture_output=True,text=True,check=True,timeout=5).stdout.strip()
        assert state and 'T' in state,'original service resumed during screen'
        before=None
        if pid is not None:
            before=process(pid);assert before and before['ppid']==os.getpid(),'server must be a live direct child'
            if pid not in bindings:bindings[pid]=before;write(out/'resources'/f'identity-{pid}.json',before)
            assert before==bindings[pid],'server PID identity changed'
        n=len(records);path=out/'resources'/f'physical-{n:04d}'
        row=physical.measure(path,os.getpid() if pid is None else pid)
        if pid is not None:assert process(pid)==before,'server exited during resource observation'
        value=physical.checked(row,limits)
        assert value['swap_bytes']<=baseline['swap_bytes'] and value['swapouts']==baseline['swapouts'],'swap grew during screen'
        if pid is not None:assert max(value[k] for k in ('top_bytes','aux_current_bytes','aux_peak_bytes'))<=64*(1<<30),'server physical footprint exceeds64GiB'
        if physical_records:assert row['started_monotonic']-physical_records[-1]['ended_monotonic']<=plan['max_observation_gap_seconds'],'resource observation gap'
        physical_records.append(row)
        code="import sys,json;sys.path.insert(0,sys.argv[1]);import resources_v2 as r;print(json.dumps(r.snapshot(sys.argv[2],None if sys.argv[3]=='None' else int(sys.argv[3]),suspended_pids=(int(sys.argv[4]),))))"
        probe=subprocess.run([str(R/'.venv/bin/python'),'-I','-B','-c',code,str(B),label,str(pid),str(service['pid'])],
            capture_output=True,text=True,check=True,timeout=5)
        sample=json.loads(probe.stdout);records.append(sample)
        write(out/'resources'/f'resource-{n:04d}.json',sample)
        resources.guard(records)
        assert sample['power_metadata']['source']=='AC Power' and sample['processinfo_thermal_metadata']['code'] in (0,1),'power or thermal state'
    async def checkpoint(label,pid):
        # Settle the bounded read-only probe before native cleanup touches its PID.
        task=asyncio.create_task(asyncio.to_thread(observe,label,pid));cancelled=False
        while not task.done():
            try:await asyncio.shield(task)
            except asyncio.CancelledError:cancelled=True
        task.result()
        if cancelled:raise asyncio.CancelledError

    try:
        for ref in plan['specs']:
            assert sha(ref['path'])==ref['sha256'];spec=json.loads(Path(ref['path']).read_text())
            label=spec['label'];print(json.dumps(dict(event='TRIAL_START',label=label,time=time.time())),flush=True)
            result=await trial.run(spec,out/'trials'/label,checkpoint=checkpoint)
            results.append(dict(label=label,**result));write(out/f'trial-{label}.json',results[-1])
            print(json.dumps(dict(event='TRIAL_END',label=label,status=result['status'],point=result.get('point'),time=time.time())),flush=True)
            assert result['status']=='PASS' and result['server_reaped'] and result['cleanup_complete'],'trial failed: '+label
        verify_pins(plan['pins'])
    except BaseException as exc:
        errors.append(dict(type=type(exc).__name__,message=str(exc)))
    finally:
        final=dict(schema='E045-concurrency-config-screen-result-v1',status='PASS_CAPTURE_ONLY' if not errors and len(results)==len(plan['specs']) else 'FAIL',
            trials=[dict(label=r['label'],status=r['status'],point=r.get('point'),server_reaped=r.get('server_reaped')) for r in results],
            errors=errors,resource_samples=len(records),physical_samples=len(physical_records),elapsed_seconds=time.monotonic()-started,
            full_state_qualified=False,performance_qualified=False,engine_pareto_point=False,
            scope='8 warm plus8 measured requests per fresh process. New environment has no inherited full-state qualification; later independent raw-output audit required.')
        write(out/'result.json',final);print(json.dumps(final),flush=True)
    return 0 if final['status']=='PASS_CAPTURE_ONLY' else 1

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True);p.add_argument('--plan-sha256',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    assert sha(a.plan)==a.plan_sha256;plan=json.loads(a.plan.read_text())
    async def start():
        loop=asyncio.get_running_loop();task=asyncio.current_task()
        for sig in (signal.SIGTERM,signal.SIGINT):loop.add_signal_handler(sig,task.cancel)
        return await run(plan,a.out)
    return asyncio.run(start())
if __name__=='__main__':raise SystemExit(main())
