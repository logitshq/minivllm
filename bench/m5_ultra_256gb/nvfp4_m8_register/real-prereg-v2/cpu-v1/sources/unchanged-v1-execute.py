"""One bounded native-M8 register-tile real-weight projection screen under the existing authorized pause lease."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request

R=Path('/Users/ryanli/code/minivllm')
B=R/'bench/m5_ultra_256gb'
A=R/'.artifacts/m5-ultra-256gb'
HERE=Path(__file__).resolve().parent
Q=HERE.parent
PID=52818
IDENTITY='Mon Oct 19 09:22:08 2026 omlx-server'
ENV_KEYS=('MLX_MAX_OPS_PER_BUFFER','MLX_MAX_MB_PER_BUFFER','MLX_BFS_MAX_WIDTH','MLX_METAL_FAST_SYNCH')

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,obj):
    with p.open('x') as f: json.dump(obj,f,indent=2); f.write('\n')
def identity():
    p=subprocess.run(['/bin/ps','-p',str(PID),'-o','lstart=,comm='],capture_output=True,text=True,timeout=5)
    return ' '.join(p.stdout.split()) if p.returncode==0 else None
def process_record(pid):
    p=subprocess.run(['/bin/ps','-p',str(pid),'-o','lstart=,ppid=,comm='],capture_output=True,text=True,timeout=5)
    if p.returncode==1 and not p.stdout.strip() and not p.stderr.strip():return None
    assert p.returncode==0 and not p.stderr.strip(),'process identity unknown'
    parts=p.stdout.split(None,6);assert len(parts)==7 and parts[5].isdigit()
    return dict(pid=pid,start=' '.join(parts[:5]),ppid=int(parts[5]),executable=parts[6].strip())
def http(route):
    with urllib.request.urlopen('http://127.0.0.1:8001'+route,timeout=5) as r:
        assert r.status==200
        return json.load(r)
def health():
    h={p:http(p) for p in ('/health','/api/status','/v1/models/status')}
    s=h['/api/status']; assert s['active_requests']==s['waiting_requests']==s['models_loading']==0
    assert h['/health']['status']=='healthy' and s['version']=='0.7.0'
    target='Qwen3.8-Flash-Next-oQ6e-mtp'
    assert s['default_model']==target and s['loaded_models']==[target]
    models=h['/v1/models/status']['models']; loaded=[m for m in models if m['loaded']]
    assert len(loaded)==1 and not any(m['is_loading'] for m in models)
    m=loaded[0];assert m['id']==target and m['model_path']==str(R.parent/'models/mlx'/target)
    assert m['engine_type']=='vlm' and m['config_model_type']=='qwen4_exp' and m['pinned'] is False
    return h
def prepare():
    sources=[Path(__file__).resolve(),B/'suspended_service.py',B/'resources_v2.py',B/'resources.py',B/'resource_base.py',B/'glm_flash/physical_guard.py',Q/'real_projection_probe_v1.py',Q/'candidate.py']
    native=R/'.venv/lib/python3.13/site-packages/mlx/include/mlx/backend/metal/kernels'
    sources.extend(native/name for name in ('fp_quantized.h','fp4.h','fp8.h'))
    pins={str(p):sha(p) for p in sources}
    plan=dict(schema='native-m8-register-projection-pause-v1',pins=pins,service_pid=PID,service_identity=IDENTITY,
        environment={k:v for k,v in os.environ.items() if k.startswith('MLX_')},
        child=[str(R/'.venv/bin/python'),'-I','-B',str(Q/'real_projection_probe_v1.py'),'--run','--out',str(A/'m8-register-real-projection-v1')],owner_deadline_seconds=160,
        physical=dict(system_limit_bytes=240*(1<<30),child_limit_bytes=4*(1<<30),swap_growth_bytes=0,swapouts_growth=0,probe_seconds=5,max_sample_gap_seconds=5,hard_allocation_cap=False),
        scope='Native-M8 nv4 tiles versus native nv8 register tile, two real-weight projections with synthetic BF16 inputs; no full-model or HTTP speed.',
        outputs=dict(execution=str(A/'m8-register-projection-execution-v1'),lease=str(A/'m8-register-projection-lease-v1')))
    write(HERE/'plan.json',plan);print(json.dumps(dict(status='PREPARED',plan_sha256=sha(HERE/'plan.json'))))
def run(expected):
    p=HERE/'plan.json'; assert sha(p)==expected; plan=json.loads(p.read_text())
    assert all(sha(q)==h for q,h in plan['pins'].items()),'source changed'
    assert plan['environment']=={k:v for k,v in os.environ.items() if k.startswith('MLX_')},'exact MLX environment changed'
    assert identity()==IDENTITY,'original service identity changed'
    out=Path(plan['outputs']['execution']); out.mkdir()
    write(out/'plan.json',plan); write(out/'before-health.json',health())
    sys.path.insert(0,str(B)); import resources_v2 as resources
    sys.path.insert(0,str(B/'glm_flash')); import physical_guard as physical
    limits=physical.policy(child_limit_bytes=4*(1<<30),work_seconds=120,interval_seconds=5)
    base=physical.measure(out/'physical-before',os.getpid()); baseline=physical.checked(base,limits)
    observed=[]
    samples=[resources.snapshot('before-pause')]; write(out/'resource-000.json',samples[-1])
    assert samples[0]['pressure_level_value']==1 and samples[0]['swap_used_mb'] is not None
    write(out/'jobs.json',[plan['child']])
    command=[sys.executable,'-I','-B',str(B/'suspended_service.py'),'--pid',str(PID),'--jobs',str(out/'jobs.json'),'--out',plan['outputs']['lease']]
    write(out/'invocation.json',dict(argv=command,started_at=time.time(),plan_sha256=expected))
    child=None; numerical_identity=None; errors=[]; started=time.monotonic(); next_sample=0
    def stop(*args): raise InterruptedError('owner signal')
    signal.signal(signal.SIGTERM,stop)
    try:
        with (out/'stdout.txt').open('x') as stdout,(out/'stderr.txt').open('x') as stderr:
            child=subprocess.Popen(command,cwd=R,stdout=stdout,stderr=stderr,env=dict(os.environ))
            while child.poll() is None:
                assert time.monotonic()-started<160,'bounded owner deadline'
                lease_path=Path(plan['outputs']['lease'])/'lease.json'
                if lease_path.exists() and time.monotonic()>=next_sample:
                    lease=json.loads(lease_path.read_text())
                    if not lease.get('closed') and lease.get('child_pid'):
                        current=process_record(lease['child_pid'])
                        assert current and current['ppid']==child.pid,'owned numerical child identity'
                        if numerical_identity is None:
                            numerical_identity=current;write(out/'numerical-child-identity.json',current)
                        assert current==numerical_identity,'numerical child identity changed'
                        row=physical.measure(out/f'physical-{len(observed):04d}',lease['child_pid'])
                        assert process_record(lease['child_pid'])==numerical_identity,'child exited/changed during physical observation'
                        value=physical.checked(row,limits)
                        assert max(value[k] for k in ('top_bytes','aux_current_bytes','aux_peak_bytes'))<=limits['child_limit_bytes'],'child physical ceiling'
                        assert value['swap_bytes']<=baseline['swap_bytes'] and value['swapouts']==baseline['swapouts'],'runtime swap growth'
                        if observed:assert row['started_monotonic']-observed[-1]['ended_monotonic']<=5,'physical sample gap'
                        observed.append(row)
                        sample=resources.snapshot('prefix-running',lease['child_pid'],suspended_pids=(PID,))
                        samples.append(sample);write(out/f'resource-{len(samples)-1:03d}.json',sample)
                        # The lease can finish while a read-only probe is in flight.
                        if child.poll() is None and not json.loads(lease_path.read_text()).get('closed'):
                            resources.guard(samples)
                            assert sample['power_metadata']['source']=='AC Power'
                            assert sample['processinfo_thermal_metadata']['code'] in (0,1)
                    next_sample=time.monotonic()+1
                time.sleep(.2)
            assert child.returncode==0,'projection child or lease failed'
    except BaseException as e:
        errors.append(dict(type=type(e).__name__,message=str(e)))
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try:child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                errors.append(dict(type='LeaseTimeout',message='Lease cleanup exceeded 30s; reconcile exact owned numerical child before lease exit.'))
                try:
                    lease=json.loads((Path(plan['outputs']['lease'])/'lease.json').read_text())
                    current=process_record(lease['child_pid'])
                    if current:
                        assert current['ppid']==child.pid and (numerical_identity is None or current==numerical_identity),'cannot bind remaining child'
                        os.kill(current['pid'],signal.SIGKILL)
                        for _ in range(50):
                            if process_record(current['pid']) is None:break
                            time.sleep(.1)
                        assert process_record(current['pid']) is None,'remaining numerical child not absent'
                    try:child.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        child.kill();child.wait(timeout=5)
                except BaseException as e:errors.append(dict(type=type(e).__name__,message='UNRESOLVED child/lease cleanup: '+str(e)))
        try:
            # Existing lease normally resumes in its finally; wait for its watchdog only if needed.
            for _ in range(10):
                raw=subprocess.run(['/bin/ps','-p',str(PID),'-o','state='],capture_output=True,text=True,timeout=5).stdout.strip()
                if raw and 'T' not in raw:break
                time.sleep(1)
            assert identity()==IDENTITY and raw and 'T' not in raw,'service not restored'
            write(out/'after-health.json',health())
        except BaseException as e:errors.append(dict(type=type(e).__name__,message=str(e)))
        after=resources.snapshot('after-resume');write(out/'after-resource.json',after)
        lease_path=Path(plan['outputs']['lease'])/'lease.json'
        lease=json.loads(lease_path.read_text()) if lease_path.exists() else None
        try:
            assert lease and lease.get('closed'),'lease did not close'
            resume=json.loads(lease_path.with_name('resume.json').read_text())
            assert resume['identity_match'] and resume['state_after'] and 'T' not in resume['state_after'],'lease resume receipt'
            result_root=Path(plan['child'][-1]);raw=json.loads((result_root/'result.json').read_text())
            assert raw['status']=='PASS_LOCAL_PROJECTION_ONLY' and len(raw['cases'])==2
            count=0
            for label in ('head','up'):
                case=result_root/label/'T8';assert json.loads((case/'numerical.json').read_text())['status']=='PASS'
                rows=[json.loads(line) for line in (case/'timings.jsonl').read_text().splitlines()]
                assert len(rows)==200 and all(row['elapsed_ns']>0 for row in rows)
                count+=len(rows)
            assert count==400
            job=json.loads(lease_path.with_name('job-0.json').read_text())
            assert job['waited'] is True and job['returncode']==0 and job['command']==plan['child'],'actual successful child wait'
            assert numerical_identity and job['pid']==numerical_identity['pid'] and child.pid==numerical_identity['ppid'],'child provenance'
            assert process_record(job['pid']) is None,'numerical child not absent'
            assert observed and all(sha(q)==h for q,h in plan['pins'].items())
        except BaseException as e:errors.append(dict(type=type(e).__name__,message=str(e)))
        result=dict(status='PASS_LOCAL_PROJECTION_ONLY' if not errors else 'FAIL',errors=errors,
            lease_returncode=None if child is None else child.returncode,lease_closed=bool(lease and lease.get('closed')),
            service_identity_after=identity(),samples=len(samples),physical_samples=len(observed),ended_at=time.time(),performance_qualified=False,
            scope=plan['scope'])
        write(out/'final.json',result);print(json.dumps(result),flush=True)
    return 0 if not errors else 1
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prepare',action='store_true');parser.add_argument('--run',action='store_true');parser.add_argument('--plan-sha256');a=parser.parse_args()
    assert not(a.prepare and a.run)
    if a.prepare:prepare()
    elif a.run:raise SystemExit(run(a.plan_sha256))
    else:print('DESCRIPTION ONLY: --prepare freezes one probe; --run requires its SHA and existing pause authorization.')
