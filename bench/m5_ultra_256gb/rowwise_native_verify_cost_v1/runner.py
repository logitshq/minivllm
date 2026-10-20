"""One direct numerical child inside the existing pause lease; no GPU imports."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import time

R = Path('/Users/ryanli/code/minivllm'); B = R/'bench/m5_ultra_256gb'
HERE = Path(__file__).resolve().parent


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


base = load(B/'concurrency_config_v2/runner.py', '_e054_existing_resource_tools')
sha, write, process, verify_pins = base.sha, base.write, base.process, base.verify_pins
physical, resources = base.physical, base.resources
hold = load(HERE/'terminal_hold.py', '_e054_terminal_hold_protocol')


def child_environment(expected):
    return {**{k:v for k,v in os.environ.items() if not k.startswith('MLX_')}, **expected}


def close_child(child, identity, out):
    """Reuse the direct-Popen reap proof, including unknown post-spawn identity."""
    if child is None:
        receipt = dict(pid=None, not_spawned=True, waited=False, absent=False, errors=[])
    else:
        def absence(pid):
            command = ['/bin/ps','-p',str(pid),'-o','pid=']
            row = subprocess.run(command, capture_output=True, text=True, timeout=5)
            return dict(command=command, returncode=row.returncode, stdout=row.stdout, stderr=row.stderr)
        raw = physical.reap_owned(child, expected_pid=child.pid, owner_pid=os.getpid(),
                                  absence=absence, out=out/'child-reap-raw.json')
        receipt = dict(pid=child.pid, identity=identity, waited=raw['waited_reaped'],
                       absent=raw['strict_absence'], returncode=raw['returncode'], errors=raw['errors'],
                       raw_status=raw['status'])
    write(out/'child-reap.json', receipt)
    return receipt


def validate_complete(complete, pid, ppid, plan):
    assert complete['schema'] == 'C1-verifier-cost-result-v1'
    assert complete['status'] == 'PASS_LOCAL_COST_ONLY'
    assert complete['pid'] == pid and complete['ppid'] == ppid
    assert complete['plan_sha256'] == plan['numerical_plan_sha256']
    assert complete['numerical_screen_qualified'] is True
    assert complete['device_cleanup_completed'] is True and complete['source_model_pins_rechecked'] is True
    assert complete['http_qualified'] is complete['performance_qualified'] is complete['engine_pareto_point'] is False
    cases = complete['qualification']
    assert len(cases) == 6 and {(x['T'],x['k']) for x in cases} == {(4,1),(4,2),(4,4),(8,1),(8,4),(8,8)}
    assert all(x['status'] == 'PASS' for x in cases)
    raw=complete['raw_comparisons']
    labels={f'T{t}/k{k}/{kind}' for t in (4,8) for k in (1,t//2,t) for kind in ('selected','continuation')}
    labels|={f'T{t}/{arm}' for t in (4,8) for arm in ('host','device')}
    assert len(raw)==16 and {x['label'] for x in raw}==labels
    assert all(len(x['files'])==2 and all(f['finite'] is True for f in x['files']) for x in raw)
    timings = complete['timings']
    assert len(timings) == 4 and {(x['T'],x['baseline']) for x in timings} == {(4,'device'),(4,'host'),(8,'device'),(8,'host')}
    assert all(x['k'] == x['T'] and len(x['samples']) == 100 for x in timings)
    for timing in timings:
        expected=[]
        for group in range(5):
            for pair in range(10):
                order=['native','candidate'] if (group*10+pair)%2==0 else ['candidate','native']
                expected.extend((group,pair,ordinal,arm,order) for ordinal,arm in enumerate(order))
        actual=[(s['group'],s['pair'],s['ordinal'],s['arm'],s['order']) for s in timing['samples']]
        assert actual==expected, 'complete ordered alternating timing pairs'
        assert all(type(s['elapsed_ns']) is int and s['elapsed_ns']>0 for s in timing['samples'])


def run(plan, out):
    verify_pins(plan['pins']); out.mkdir(); (out/'resources').mkdir()
    write(out/'plan.json', plan)
    service = plan['service']; baseline = plan['physical_baseline']
    limits = physical.policy(child_limit_bytes=64*(1<<30), work_seconds=1800, interval_seconds=3)
    started = time.monotonic(); physical_rows = []; resource_rows = []; errors = []; child = identity = None

    def observe(label, pid=None):
        assert time.monotonic()-started < plan['screen_deadline_seconds'], 'numerical owner deadline'
        assert process(service['pid']) == service['process'], 'service identity changed'
        state = subprocess.run(['/bin/ps','-p',str(service['pid']),'-o','state='], capture_output=True,
                               text=True, check=True, timeout=5).stdout.strip()
        assert state and 'T' in state, 'original service is not paused'
        if pid is not None:
            assert identity and process(pid) == identity, 'numerical child changed or exited'
        n = len(physical_rows)
        row = physical.measure(out/'resources'/f'physical-{n:04d}', os.getpid() if pid is None else pid)
        if pid is not None: assert process(pid) == identity, 'child exited during physical observation'
        value = physical.checked(row, limits)
        assert max(value[k] for k in ('top_bytes','aux_current_bytes','aux_peak_bytes')) <= 64*(1<<30), '64 GiB child ceiling'
        assert value['swap_bytes'] <= baseline['swap_bytes'] and value['swapouts'] == baseline['swapouts'], 'runtime swap growth'
        prior = started if not physical_rows else physical_rows[-1]['ended_monotonic']
        assert row['started_monotonic']-prior <= plan['max_observation_gap_seconds'], 'physical observation gap'
        physical_rows.append(row)
        code = "import sys,json;sys.path.insert(0,sys.argv[1]);import resources_v2 as r;print(json.dumps(r.snapshot(sys.argv[2],None if sys.argv[3]=='None' else int(sys.argv[3]),suspended_pids=(int(sys.argv[4]),))))"
        command = [str(R/'.venv/bin/python'),'-I','-B','-c',code,str(B),label,str(pid),str(service['pid'])]
        probe = subprocess.run(command, capture_output=True, text=True, timeout=5)
        write(out/'resources'/f'probe-{n:04d}.json', dict(command=command, returncode=probe.returncode,
                                                       stdout=probe.stdout, stderr=probe.stderr))
        assert probe.returncode == 0 and not probe.stderr.strip(), 'bounded metadata probe failed'
        sample = json.loads(probe.stdout); resource_rows.append(sample)
        write(out/'resources'/f'resource-{n:04d}.json', sample); resources.guard(resource_rows)
        assert sample['power_metadata']['source'] == 'AC Power' and sample['processinfo_thermal_metadata']['code'] in (0,1), 'power/thermal'

    previous = {}
    def stop(*unused): raise InterruptedError('numerical owner cancellation')
    try:
        previous = {s:signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT)}
        for sig in previous: signal.signal(sig, stop)
        observe('paused-admission')
        original_command = plan['numerical_command']
        command_path = out/'original-qualifier-command.json'; write(command_path, original_command)
        command = [str(R/'.venv/bin/python'),'-I','-B',str(HERE/'terminal_hold.py'),'--run',
                   '--command-json',str(command_path),'--command-sha256',sha(command_path),'--control',str(out)]
        write(out/'child-command.json', command)
        with (out/'child.stdout').open('x') as stdout, (out/'child.stderr').open('x') as stderr:
            child = subprocess.Popen(command, cwd=R, env=child_environment(plan['child_environment']), stdout=stdout, stderr=stderr)
            identity = process(child.pid)
            assert identity and identity['ppid'] == os.getpid(), 'direct numerical child binding'
            write(out/'resources'/f'identity-{child.pid}.json', identity)
            next_sample = 0; terminal = None
            while child.poll() is None:
                assert time.monotonic()-started < plan['screen_deadline_seconds'], 'numerical owner deadline'
                done_path = out/'numerical-done.json'
                if done_path.exists():
                    terminal = json.loads(done_path.read_text())
                    assert terminal['schema']=='E054-terminal-hold-v1', 'actual terminal hold protocol'
                    assert terminal['pid'] == child.pid and terminal['ppid'] == os.getpid() and terminal['command'] == original_command
                    assert type(terminal['qualifier_returncode']) is int and time.monotonic() < terminal['ack_deadline_monotonic']
                    observe('numerical-done-before-exit', child.pid)
                    ref = terminal['qualifier_complete']; complete_path = Path(plan['outputs']['numerical'])/'complete.json'
                    assert ref is not None and ref['path'] == str(complete_path) and ref['sha256'] == sha(complete_path), 'completed numerical receipt missing/changed'
                    complete = json.loads(complete_path.read_text())
                    assert complete['schema'] == 'C1-verifier-cost-result-v1'
                    assert complete['pid'] == child.pid and complete['ppid'] == os.getpid(), 'completed numerical provenance'
                    assert complete['status'] in ('PASS_LOCAL_COST_ONLY','FAIL','INCOMPLETE')
                    assert terminal['qualifier_returncode'] == (0 if complete['status'] == 'PASS_LOCAL_COST_ONLY' else 1), 'original terminal status/exit disagree'
                    hold.atomic(out/'numerical-exit-ack.json', dict(pid=child.pid, owner_pid=os.getpid(),
                        done_sha256=sha(done_path), command=original_command, final_physical_and_metadata_complete=True,
                        physical_sha256=sha(out/'resources'/f'physical-{len(physical_rows)-1:04d}'/'sample.json'),
                        metadata_sha256=sha(out/'resources'/f'resource-{len(resource_rows)-1:04d}.json')))
                    child.wait(timeout=10)
                    break
                if time.monotonic() >= next_sample:
                    observe('c1-verifier-cost-running', child.pid); next_sample = time.monotonic()+3
                time.sleep(.2)
            child.wait(timeout=5)
            assert terminal is not None, 'child exited without terminal hold/final observation'
            assert child.returncode == terminal['qualifier_returncode'] == 0 and terminal['wrapper_errors'] == [], 'numerical qualifier/terminal hold failed or incomplete'
        assert len(physical_rows) >= 2, 'no running child resource sample'
        complete = json.loads((Path(plan['outputs']['numerical'])/'complete.json').read_text())
        validate_complete(complete, child.pid, os.getpid(), plan)
        verify_pins(plan['pins'])
    except BaseException as exc:
        errors.append(dict(type=type(exc).__name__, message=str(exc)))
    finally:
        # Ignore repeated cancellation only while settling this direct child; the
        # outer driver's bounded fallback can still KILL this owner if necessary.
        for sig in previous: signal.signal(sig, signal.SIG_IGN)
        reap = close_child(child, identity, out)
        if child is None or not reap['waited'] or not reap['absent'] or reap['errors']:
            errors.append(dict(type='ReapFailure', message='numerical child not cleanly waited/absent'))
        result = dict(schema='E054-c1-cost-owner-v1', status='PASS_CAPTURE_ONLY' if not errors else 'FAIL',
            errors=errors, child_reaped=reap['waited'] and reap['absent'], child_returncode=reap.get('returncode'),
            physical_samples=len(physical_rows), resource_samples=len(resource_rows), elapsed_seconds=time.monotonic()-started,
            independent_numerical_audit_pending=True, full_model_qualified=False, http_qualified=False, performance_qualified=False)
        write(out/'result.json', result); print(json.dumps(result), flush=True)
        if child is not None and not (reap['waited'] and reap['absent']):
            # Never return to the lease with an unresolved numerical descendant.
            # The already-bounded outer fallback resolves saved child identities
            # before killing this owner; unknown descendants require reconciliation.
            while True: time.sleep(1)
        for sig, old in previous.items(): signal.signal(sig, old)
    return 0 if not errors else 1


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--plan-sha256', required=True); p.add_argument('--out', type=Path, required=True); a = p.parse_args()
    assert sha(a.plan) == a.plan_sha256
    raise SystemExit(run(json.loads(a.plan.read_text()), a.out))
