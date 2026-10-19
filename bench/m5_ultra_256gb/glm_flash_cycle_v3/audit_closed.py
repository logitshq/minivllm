"""Offline GLM v3 diagnostic audit: original raw gates, separate live cadences.

Only three frozen audit entry deltas: result schema, metadata label, and the
old 1:1 sample/checkpoint relation. Raw physical parsing, request token rules,
runtime inventory and restoration checks retain their original code. New lane
receipts bind every command to its original raw validator result and deadline.
No device imports, process queries or resource-safety/performance qualification.
"""
import ast as _ast
import hashlib as _hashlib
import importlib.util as _importlib
import inspect as _inspect
import math as _math
from pathlib import Path as _Path

_reuse_path = _Path(__file__).resolve().parents[1]/'glm_flash_cycle_v2/_reuse.py'
if _hashlib.sha256(_reuse_path.read_bytes()).hexdigest() != 'fbf523c586629ce6e5e2030d629b00c9f1cd99dfec9edef6c35ac8e054468f97':
    raise ValueError('fixed namespace reuse source changed')
_spec = _importlib.spec_from_file_location('_glm_v3_closed_reuse', _reuse_path)
_reuse = _importlib.module_from_spec(_spec); _spec.loader.exec_module(_reuse)
_original = _reuse.install(globals(), 'audit_closed.py', dict(
    OWNER_SHA='9d69c5a64dc50b7d65330ad130b8af9f757b91e884aa197d1e828775b27c88a8',
    RESTORE_SHA='fca6b668df7240ac12ce47380f99af5e670eecd885af4d21d44462268f3c7bab'))

MONITOR_POLICY = dict(schema='GLM-dual-cadence-policy-v1', physical_max_gap_seconds=1.0,
    metadata_max_gap_seconds=5.0, callback_seconds=5.0, target_gap_fraction=.5,
    join_seconds=7.0, metadata_per_physical_sample=False)
READ_POLICY = {k:v for k,v in MONITOR_POLICY.items() if k not in ('schema','metadata_per_physical_sample')}
READ_POLICY.update(read_cleanup_communicate_seconds=.5, read_cleanup_wait_seconds=.5)


def number(value):
    return type(value) in (int,float) and _math.isfinite(value)


def raw_command(row):
    keys = ('command','returncode','stdout','stderr') if 'stdout' in row else (
        'command','returncode','stdout_bytes','stdout_sha256','stderr')
    return {key:row[key] for key in keys}


def identity_command(row, identity):
    need(row['command'] == ['/bin/ps','-p',str(identity['pid']),'-o','pid=,ppid=,lstart=,comm=']
         and row['returncode'] == 0 and type(row['stdout']) is type(row['stderr']) is str
         and not row['stderr'].strip(), 'actual physical-boundary identity command')
    match = re.fullmatch(r'\s*(\d+)\s+(\d+)\s+([A-Za-z]{3} [A-Za-z]{3}\s+\d+ \d\d:\d\d:\d\d \d{4})\s+(.+?)\s*',row['stdout'])
    need(match and dict(pid=int(match[1]),ppid=int(match[2]),start=' '.join(match[3].split()),comm=match[4]) == identity,
         'physical-boundary full child identity')


def monitor_resources(owner, plan, complete, child, release, physical_paths):
    """Independent file-only lane accounting; raw physical/metadata gates run too."""
    owner = Path(owner); folder = owner/'monitor'; path = folder/'monitor.json'; report = read(path)
    need(plan['monitor_policy'] == MONITOR_POLICY == complete['monitor_policy']
         and complete['schema'] == 'GLM-cold-load-owner-result-v3'
         and complete['legacy_checkpoint_per_sample_contract'] is False, 'explicit dual-lane policy')
    need(complete['monitor'] == dict(started=True,read_commands_settled=True,status='COMPLETE',
         receipt=dict(path=str(path),sha256=sha(path))), 'successful original monitor binding')
    need(report['schema'] == 'GLM-dual-cadence-monitor-v1' and report['status'] == 'COMPLETE'
         and report['pid'] == child['pid'] and report['owner_pid'] == complete['owner_pid']
         and type(report['child_returncode']) is int and report['child_returncode'] == 0
         and report['policy'] == READ_POLICY and report['errors'] == []
         and report['unjoined_workers'] == [] and report['unresolved_read_commands'] == []
         and report['read_commands_settled'] is True and report['model_child_release_proven'] is False
         and report['physical_resource_safety_qualified'] is False and report['performance_qualified'] is False
         and set(report['lanes']) == set(report['commands']) == {'physical','metadata'}, 'complete two-lane monitor')
    start, end, deadline = (report[k] for k in ('started_monotonic','ended_monotonic','work_deadline'))
    need(all(number(v) for v in (start,end,deadline)) and child['spawned_monotonic'] <= start <= end
         and start < deadline <= child['spawned_monotonic']+plan['physical_policy']['work_seconds'], 'monitor actual work window')
    counts = {'physical':len(physical_paths),'metadata':complete['metadata_live_samples']}
    need(all(type(n) is int and n > 0 for n in counts.values())
         and complete['live_physical_samples'] == counts['physical']
         and complete['checkpoints'] == counts['metadata']+6, 'independent physical and metadata counts')
    answer = {}
    for lane in ('physical','metadata'):
        state = report['lanes'][lane]; rows = state['samples']; commands = report['commands'][lane]
        need(state['active'] is None and type(rows) is list and len(rows) == counts[lane]
             and type(commands) is list and len(commands) == 7*counts[lane], 'complete lane sample/command coverage')
        files = [folder/lane/f'sample-{i:04d}.json' for i in range(len(rows))]
        need(set((folder/lane).glob('sample-*.json')) == set(files), 'exact lane sample inventory')
        previous = start; max_gap = max_callback = max_command = 0.
        for i,(sample,sample_path) in enumerate(zip(rows,files)):
            need(not sample_path.is_symlink() and read(sample_path) == sample, 'original lane sample payload')
            a,b,d = (sample[k] for k in ('started_monotonic','ended_monotonic','deadline_monotonic'))
            need(all(number(v) for v in (a,b,d,sample['started_at'],sample['ended_at'],sample['gap_seconds'],sample['elapsed_seconds']))
                 and sample['index'] == i and sample['status'] == 'COMPLETE' and 'error' not in sample
                 and sample['prior_end_monotonic'] == previous and sample['gap_seconds'] == a-previous
                 and 0 <= a-previous <= MONITOR_POLICY[lane+'_max_gap_seconds']
                 and a <= b <= d == min(a+5.,deadline) and b <= end
                 and sample['elapsed_seconds'] == b-a
                 and child['spawned_at'] <= sample['started_at'] <= sample['ended_at'] <= release['time'], 'lane cadence/deadline/lifetime')
            need(sample['command_range'] == dict(start=i*7,stop=(i+1)*7), 'nonoverlapping complete command range')
            batch = commands[i*7:(i+1)*7]; prior_command_end = a
            for command in batch:
                ca,cb,cd = (command[k] for k in ('started_monotonic','ended_monotonic','deadline_monotonic'))
                need(all(number(v) for v in (ca,cb,cd,command['elapsed_seconds'],command['started_at'],command['ended_at']))
                     and prior_command_end <= ca <= cb <= cd <= d and cb <= b
                     and command['elapsed_seconds'] == cb-ca and cb-ca <= 5
                     and sample['started_at'] <= command['started_at'] <= command['ended_at'] <= sample['ended_at']
                     and command['status'] == 'COMPLETE' and 'error' not in command and 'cleanup_errors' not in command
                     and command['actions'] == [] and command['waited_reaped'] is True
                     and command['unknown_launch'] is False and type(command['returncode']) is int,
                     'actual successful bounded command sequence')
                prior_command_end = cb; max_command = max(max_command,cb-ca)
            if lane == 'physical':
                raw_path = physical_paths[i]; raw = read(raw_path)
                need(sample['result'] == dict(sample=dict(path=str(raw_path),sha256=sha(raw_path)),observed_within_budget=True),
                     'physical callback original raw reference')
                identity_command(batch[0],child['identity']); identity_command(batch[-1],child['identity'])
                need([raw_command(r) for r in batch[1:-1]] == raw['commands']
                     and a <= raw['started_monotonic'] <= batch[1]['started_monotonic']
                     and batch[-2]['ended_monotonic'] <= raw['ended_monotonic'] <= b, 'physical raw command/time binding')
            else:
                raw_path = owner/f'checkpoint-{i+5:04d}.json'; raw = read(raw_path)
                need(sample['result'] == dict(checkpoint=dict(path=str(raw_path),sha256=sha(raw_path)))
                     and raw['lane_sample_index'] == i and raw['label'] == 'running-metadata'
                     and [raw_command(r) for r in batch] == raw['commands']
                     and a <= raw['monotonic'] <= batch[0]['started_monotonic']
                     and batch[-1]['ended_monotonic'] <= raw['ended_monotonic'] <= b, 'metadata raw command/time binding')
            max_gap = max(max_gap,a-previous); max_callback = max(max_callback,b-a); previous = b
        answer[lane] = dict(samples=len(rows),commands=len(commands),maximum_end_to_start_gap_seconds=max_gap,
            maximum_callback_seconds=max_callback,maximum_command_seconds=max_command)
    return dict(status='PASS',lanes=answer,checkpoint_per_physical_sample=False,
                scope='Independent observed gap/deadline checks; no scheduler guarantee or final child peak proof.')


def _variant(name, changes):
    source = _inspect.getsource(getattr(_original,name))
    for before,after in changes:
        need(source.count(before) == 1, 'fixed audit delta anchor changed: '+name)
        source = source.replace(before,after,1)
    exec(compile(_ast.parse(source),__file__,'exec'),globals())
    return source


CHECKPOINT_DELTAS = [("['running']*(count-6)","['running-metadata']*(count-6)"),
                     ("label=='running'","label=='running-metadata'")]
_checkpoint_source = _variant('checkpoints',CHECKPOINT_DELTAS)
AUDIT_DELTAS = [("schema='GLM-closed-load-audit-v1'","schema='GLM-closed-load-audit-v3'"),
    ("need(len(paths)==complete['live_physical_samples'] and bool(paths) and complete['checkpoints']==len(paths)+6,'complete nonempty live sample/checkpoint set')",
     "need(len(paths)==complete['live_physical_samples'] and bool(paths),'complete nonempty live physical set')\n"
     "        lanes=monitor_resources(owner,plan,complete,child,release,paths)"),
    ("checkpoints=checkpoints(owner,plan,attempt,child,release,guard),no_final_child_peak_history=True,",
     "checkpoints=checkpoints(owner,plan,attempt,child,release,guard),dual_cadence=lanes,no_final_child_peak_history=True,")]
_audit_source = _variant('audit',AUDIT_DELTAS)

if __name__ == '__main__': raise SystemExit(main())
