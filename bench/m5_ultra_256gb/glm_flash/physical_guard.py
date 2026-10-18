"""Prepared physical-budget adapter for an explicitly owned GLM load probe.

No action on import and no CLI. The future GLM owner must pin this source, retain
its own cold-service/identity/runtime/model proof, call cold_admission before
Popen, and repeatedly call observe while owning the returned Popen handle.
This does not replace resources_v2 power/thermal/process checks or cold recovery.

Apple JSON semantics follow cold_cycle/controller.py:parse_resources and
conv_tail_owned_copy/layer_owner.py:parse_footprint. Unequal process top/current
readings are real observations; do not assume equality or add process/system
footprints. Only auxiliary process peak is a historical peak. No vmmap virtual
region size, RSS, logical tensor bytes or MLX limit becomes physical headroom.

System observations retain the existing 16 GiB reserve (240 GiB ceiling), but
sampling/TERM cannot enforce an allocation cap or bound inter-sample system
peaks. The caller chooses an explicit process ceiling <=240 GiB. A successful
report is a bounded observed diagnostic, never safe-fit/model/performance proof.
There is no exception for an unreadable already-exited process: this probe has
no independently retained final self-footprint history to close that gap.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time

GIB = 1 << 30
RAM = 256 * GIB
SYSTEM_LIMIT = 240 * GIB
COLD_LIMIT = 40 * GIB


def need(ok, message):
    if not ok: raise ValueError(message)


def number(value): return type(value) in (int, float) and math.isfinite(value)


def policy(*, child_limit_bytes, work_seconds, interval_seconds=1.0, probe_seconds=5.0):
    """Explicit plan fields, not a claim that a particular model will fit."""
    need(type(child_limit_bytes) is int and 0 < child_limit_bytes <= SYSTEM_LIMIT, 'explicit child physical ceiling')
    need(number(work_seconds) and 0 < work_seconds <= 1800, 'bounded child work deadline')
    need(number(interval_seconds) and 0 < interval_seconds <= 5, 'bounded sample cadence')
    need(number(probe_seconds) and 0 < probe_seconds <= 5, 'bounded aggregate physical probe time')
    return dict(schema='GLM-physical-observation-policy-v1', physical_ram=RAM,
        system_limit_bytes=SYSTEM_LIMIT, cold_limit_bytes=COLD_LIMIT,
        child_limit_bytes=child_limit_bytes, work_seconds=work_seconds,
        interval_seconds=interval_seconds, probe_seconds=probe_seconds,
        swap_growth_bytes=0, swapouts_growth=0, hard_allocation_cap=False)


def parse_raw(footprint, pid, pressure, swap, memory, vm_stat):
    need(type(pid) is int and pid > 1, 'exact PID')
    need(footprint.get('unit') == 'byte' and type(footprint.get('bytes per unit')) is int
         and footprint['bytes per unit'] == 1, 'physical byte units')
    need(footprint.get('errors') == [] and footprint.get('warnings') == [], 'physical errors/warnings')
    processes = footprint.get('processes')
    need(type(processes) is list and len(processes) == 1 and type(processes[0].get('pid')) is int
         and processes[0]['pid'] == pid, 'physical exact process PID')
    process = processes[0]; auxiliary = process['auxiliary']
    top, current, peak = process.get('footprint'), auxiliary.get('phys_footprint'), auxiliary.get('phys_footprint_peak')
    system = footprint.get('auxiliary', {}).get('sys_footprint')
    need(all(type(v) is int and v >= 0 for v in (top, current, peak, system))
         and system > 0 and peak >= current, 'known raw physical readings')
    need(re.fullmatch(r'\d+', pressure.strip()) and re.fullmatch(r'\d+', memory.strip()), 'pressure/RAM format')
    used = re.search(r'used = (\d+(?:\.\d+)?)M', swap)
    outs = re.search(r'^Swapouts:\s*(\d+)\.', vm_stat, re.M)
    need(used is not None and outs is not None and math.isfinite(float(used[1])), 'swap/Swapouts format')
    return dict(pid=pid, top_bytes=top, aux_current_bytes=current, aux_peak_bytes=peak,
        current_bytes=max(top, current), peak_bytes=max(top, peak), system_bytes=system,
        pressure=int(pressure), physical_ram=int(memory), swap_bytes=round(float(used[1])*(1 << 20)),
        swapouts=int(outs[1]))


def exclusive_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')
        stream.flush(); os.fsync(stream.fileno())


def measure(directory, pid, *, timeout_seconds=5.0, run=subprocess.run,
            clock=time.monotonic, wall=time.time):
    """One explicit read-only probe with a single total deadline; retain failure.

    Uses one process plus system footprint from the same Apple report. No service
    commands, model imports, signals or admission decisions happen here. The
    owner must check exact live Popen/identity immediately around this call.
    """
    need(type(pid) is int and pid > 1 and number(timeout_seconds) and 0 < timeout_seconds <= 5, 'probe PID/deadline')
    directory = Path(directory).resolve(); directory.mkdir(parents=False, exist_ok=False)
    raw_path = directory/'footprint.raw.json'
    commands = [
        ['/usr/bin/footprint','-p',str(pid),'--noCategories','--sysFootprint','-f','bytes','-j',str(raw_path)],
        ['/usr/sbin/sysctl','-n','kern.memorystatus_vm_pressure_level'],
        ['/usr/sbin/sysctl','vm.swapusage'], ['/usr/sbin/sysctl','-n','hw.memsize'], ['/usr/bin/vm_stat']]
    began = clock(); result = dict(schema='GLM-physical-sample-v1', pid=pid, owner_pid=os.getpid(),
        started_monotonic=began, started_at=wall(), commands=[], status='FAIL')
    try:
        for argv in commands:
            remaining = began + timeout_seconds - clock(); need(remaining > 0, 'physical probe deadline')
            row = dict(command=argv)
            try:
                value = run(argv, capture_output=True, text=True, timeout=remaining)
                row.update(returncode=value.returncode, stdout=value.stdout, stderr=value.stderr)
            except BaseException as error:
                row.update(returncode=None, error_type=type(error).__name__, error=str(error),
                    stdout=getattr(error,'stdout',None), stderr=getattr(error,'stderr',None))
                for key in ('stdout','stderr'):
                    if isinstance(row[key], bytes): row[key] = row[key].decode(errors='replace')
                raise
            finally:
                result['commands'].append(row)
            need(type(row['returncode']) is int and row['returncode'] == 0
                 and type(row['stdout']) is type(row['stderr']) is str and not row['stderr'].strip(), 'physical command failed/unknown')
        need(raw_path.is_file() and not raw_path.is_symlink() and raw_path.stat().st_size <= (8 << 20), 'bounded actual footprint JSON')
        raw = raw_path.read_bytes()
        result['raw'] = dict(path=str(raw_path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        result['parsed'] = parse_raw(json.loads(raw), pid, *(r['stdout'] for r in result['commands'][1:]))
        need(clock()-began <= timeout_seconds, 'physical probe exceeded total deadline')
        result['status'] = 'MEASURED'
        return result
    except BaseException as error:
        result['error'] = dict(type=type(error).__name__, message=str(error)); raise
    finally:
        result.update(ended_monotonic=clock(), ended_at=wall())
        exclusive_json(directory/'sample.json', result)


def checked(row, limits):
    need(limits == policy(**{k:limits[k] for k in ('child_limit_bytes','work_seconds','interval_seconds','probe_seconds')}),
         'physical policy differs from explicit fixed contract')
    need(row.get('status') == 'MEASURED' and row.get('error') is None, 'unknown/failed physical observation')
    begin, end = row['started_monotonic'], row['ended_monotonic']
    need(number(begin) and number(end) and 0 <= end-begin <= limits['probe_seconds'], 'physical sample duration')
    value = row['parsed']
    need(all(type(value.get(k)) is int and value[k] >= 0 for k in
         ('pid','top_bytes','aux_current_bytes','aux_peak_bytes','current_bytes','peak_bytes','system_bytes',
          'pressure','physical_ram','swap_bytes','swapouts'))
         and value['aux_peak_bytes'] >= value['aux_current_bytes']
         and value['current_bytes'] == max(value['top_bytes'],value['aux_current_bytes'])
         and value['peak_bytes'] == max(value['top_bytes'],value['aux_peak_bytes']), 'strict parsed physical values')
    need(value['pid'] == row['pid'] and value['physical_ram'] == RAM and value['pressure'] == 1
         and value['system_bytes'] > 0
         and value['system_bytes'] <= SYSTEM_LIMIT, 'physical system budget/pressure/RAM')
    return value


def cold_admission(samples, limits, *, owner_pid):
    """Additional physical prerequisite only; never creates probe admission.json."""
    need(type(owner_pid) is int and owner_pid > 1 and len(samples) == 2, 'two cold owner samples')
    values = [checked(row, limits) for row in samples]
    need(all(row['pid'] == row['owner_pid'] == owner_pid for row in samples), 'cold sample owner identity')
    first, last = values
    need(4 <= samples[1]['ended_monotonic']-samples[0]['ended_monotonic'] <= 60, 'cold stability interval')
    need(max(x['system_bytes'] for x in values) <= COLD_LIMIT
         and abs(last['system_bytes']-first['system_bytes']) <= GIB, 'cold physical stability')
    need(last['swap_bytes'] <= first['swap_bytes'] and last['swapouts'] == first['swapouts'], 'cold swap growth')
    return dict(owner_pid=owner_pid, baseline_swap_bytes=last['swap_bytes'], baseline_swapouts=last['swapouts'],
        last_sample_monotonic=samples[-1]['ended_monotonic'], cold_system_bytes=last['system_bytes'],
        scope='Observed cold physical prerequisite only; no model fit or service absence assertion.')


def observe(row, limits, cold, *, child_pid, child_started_monotonic, previous=None):
    """Caller retains raw sample and aborts owned work on any exception."""
    value = checked(row, limits)
    need(type(child_pid) is int and child_pid > 1 and child_pid != cold['owner_pid']
         and row['pid'] == child_pid and row['owner_pid'] == cold['owner_pid'], 'exact direct child sample binding')
    need(number(child_started_monotonic) and child_started_monotonic >= cold['last_sample_monotonic']
         and row['started_monotonic'] >= child_started_monotonic
         and row['ended_monotonic']-child_started_monotonic <= limits['work_seconds'], 'child observation/work deadline')
    prior_end = child_started_monotonic if previous is None else previous['ended_monotonic']
    need(0 <= row['started_monotonic']-prior_end <= limits['interval_seconds'], 'unsampled gap exceeded')
    need(all(value[k] <= limits['child_limit_bytes'] for k in ('top_bytes','aux_current_bytes','aux_peak_bytes')),
         'child physical current/history budget exceeded')
    need(value['swap_bytes'] <= cold['baseline_swap_bytes'] and value['swapouts'] == cold['baseline_swapouts'], 'runtime swap growth')
    if previous is not None:
        old = checked(previous, limits)
        need(old['pid'] == child_pid and value['aux_peak_bytes'] >= old['aux_peak_bytes'], 'auxiliary historical peak decreased')
    return dict(observed_within_budget=True, parsed=value,
        scope='One bounded observation; derived max may decrease and inter-sample system peak remains unknown.')


def reap_owned(child, *, expected_pid, owner_pid, absence, out):
    """Explicit returned Popen handle only; outer cleanup/cancellation owns policy.

    No handle means unknown spawn, never NO_CHILD. Failure preserves an explicit
    unresolved receipt. The outer controller must refuse restore until its own
    strict process proof succeeds; this helper never starts/restores a service.
    """
    need(child is not None and type(expected_pid) is int and expected_pid > 1
         and child.pid == expected_pid and owner_pid == os.getpid(), 'actual direct Popen handle required')
    out=Path(out); need(not out.exists(), 'new reap receipt required')
    row=dict(schema='GLM-owned-reap-v1', pid=expected_pid, owner_pid=owner_pid,
        ownership='direct-Popen-returned', waited_reaped=False, strict_absence=False,
        returncode=None, actions=[], errors=[])
    try:
        if child.poll() is None:
            row['actions'].append('TERM')
            try: child.terminate(); child.wait(timeout=15)
            except BaseException as error: row['errors'].append(dict(stage='TERM/wait',type=type(error).__name__,message=str(error)))
        if child.poll() is None:
            row['actions'].append('KILL')
            try: child.kill()
            except BaseException as error: row['errors'].append(dict(stage='KILL',type=type(error).__name__,message=str(error)))
        code=child.wait(timeout=15)
        need(type(code) is int and child.returncode == code, 'actual wait return code')
        row.update(returncode=code, waited_reaped=True)
        raw=absence(expected_pid); row['absence']=raw
        need(raw['command']==['/bin/ps','-p',str(expected_pid),'-o','pid=']
             and type(raw['returncode']) is int and raw['returncode']==1
             and raw['stdout'].strip()==raw['stderr'].strip()=='', 'strict exact PID absence')
        row['strict_absence']=True
    except BaseException as error: row['errors'].append(dict(stage='final-reap',type=type(error).__name__,message=str(error)))
    finally:
        row.update(time=time.time(), status='REAPED' if row['waited_reaped'] and row['strict_absence'] else 'UNRESOLVED',
            performance_qualified=False, physical_resource_safety_qualified=False)
        exclusive_json(out,row)
    return row


def diagnostic_report(cold_samples, samples, limits, release, probe_complete, *,
                      child_started_monotonic, observation_error=None):
    """Keep resource, cooperative shutdown and OS reap outcomes distinct."""
    reaped=release.get('status')=='REAPED' and release.get('waited_reaped') is True and release.get('strict_absence') is True
    release_error=None
    try:
        raw=release['absence']
        need(reaped and release.get('ownership')=='direct-Popen-returned'
             and type(release.get('returncode')) is int
             and raw['command']==['/bin/ps','-p',str(release['pid']),'-o','pid=']
             and type(raw['returncode']) is int and raw['returncode']==1
             and raw['stdout'].strip()==raw['stderr'].strip()=='','missing exact final wait/absence proof')
    except (ValueError,KeyError,TypeError,AttributeError) as error:
        reaped=False; release_error=dict(type=type(error).__name__,message=str(error))
    try:
        cold=cold_admission(cold_samples,limits,owner_pid=release['owner_pid'])
        previous=None
        need(bool(samples),'no live physical samples')
        for row in samples:
            observe(row,limits,cold,child_pid=release['pid'],child_started_monotonic=child_started_monotonic,previous=previous)
            previous=row
    except (ValueError,KeyError,TypeError,AttributeError) as error:
        observation_error=observation_error or dict(type=type(error).__name__,message=str(error))
    probe_ok=(probe_complete.get('schema')=='GLM-official-load-result-v1' and probe_complete.get('status')=='PASS'
        and probe_complete.get('pid')==release.get('pid') and probe_complete.get('owner_pid')==release.get('owner_pid')
        and probe_complete.get('short_lifecycle_diagnostic_complete') is True
        and probe_complete.get('shutdown',{}).get('complete') is True
        and [r.get('cap') for r in probe_complete.get('completed_requests',[])]==[8,32,32,32])
    # This rechecks parsed observations; outer audit must still reread their
    # retained original command/footprint bytes and all source/admission pins.
    passed=bool(samples) and observation_error is None and reaped and release.get('returncode')==0 and probe_ok
    return dict(schema='GLM-physical-diagnostic-summary-v1', status='PASS' if passed else 'FAIL',
        short_lifecycle_diagnostic_complete=passed, server_reaped=reaped,
        observation_error=observation_error, release_error=release_error, sample_count=len(samples), probe_complete=probe_ok,
        physical_resource_safety_qualified=False, full_model_correctness_qualified=False, performance_qualified=False,
        limits=['No hard allocation cap; samples and termination can lag allocations.',
                'No independently recorded child physical peak after the last live sample.',
                'External cold/model/runtime/exclusive-resource proof and recovery are still mandatory.'])
