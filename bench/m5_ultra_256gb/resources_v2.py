"""Process-title-aware probe; v1 evidence and original matcher stay immutable."""
from pathlib import Path
import subprocess
import resources as original

deltas = original.deltas


def native_processes(text, owned_pid=None, suspended_pids=()):
    processes = []
    names = {'omlx-server': 'omlx', 'omlx': 'omlx', 'lithos-metal': 'lithos',
             'llama-server': 'llama.cpp', 'mlx_lm.server': 'mlx-lm'}
    for line in text.splitlines():
        fields = line.split(None, 5)
        if len(fields) != 6:
            continue
        pid, ppid, rss, cpu, state, executable = fields
        name = Path(executable).name
        if name not in names and int(pid) != owned_pid:
            continue
        processes.append({'pid': int(pid), 'ppid': int(ppid), 'rss_kib': int(rss),
                          'cpu_percent': float(cpu), 'state': state, 'executable': executable,
                          'engine': names.get(name, 'explicit-owned-pid'), 'owned': int(pid) == owned_pid,
                          'authorized_suspended': int(pid) in suspended_pids and 'T' in state})
    return processes


def snapshot(label, owned_pid=None, *, suspended_pids=()):
    result = original.snapshot(label, owned_pid)
    raw = subprocess.run(['/bin/ps', '-axo', 'pid=,ppid=,rss=,pcpu=,stat=,comm='],
                         capture_output=True, text=True, timeout=10)
    native = native_processes(raw.stdout, owned_pid, suspended_pids) if raw.returncode == 0 else []
    result['native_process_inventory_ok'] = raw.returncode == 0
    result['native_inference_processes'] = native
    result['authorized_suspended_pids'] = list(suspended_pids)
    result['native_process_scope'] = 'Executable titles plus exact owned PID; no command arguments; unrecognized runtimes/GUI GPU use remain unknown'
    result['no_active_foreign_native_processes'] = all(p['owned'] or p['authorized_suspended'] for p in native)
    return result


def guard(samples):
    last = samples[-1]
    if not last.get('native_process_inventory_ok'):
        raise RuntimeError('Native process inventory unavailable')
    if not last['no_active_foreign_native_processes']:
        raise RuntimeError('Foreign inference executable present, including renamed servers; stop owned benchmark')
    # Keep original facts unchanged. Only an explicitly listed PID observed in
    # stopped state may be exempted from v1's command-line inventory guard.
    allowed = {p['pid'] for p in last['native_inference_processes'] if p['authorized_suspended']}
    approved = all(p['owned'] or p['pid'] in allowed for p in last['known_inference_processes'])
    original.guard(samples[:-1] + [{**last, 'only_owned_known_inference_processes': approved}])
