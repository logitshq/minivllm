"""Read-only power/thermal metadata wrapper preserving the pinned original probe.

snapshot(label, owned_pid=None), deltas and guard keep the original interfaces.
No GPU import, model load, process/app control, heavy counter, or setting changes.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import re
import subprocess

P = Path(__file__).resolve().parent
IDENTITY = P / 'precise-v2-source-identity.json'
SOURCE_KEY = 'resource_snapshot_precise_v1.py'
PINNED = json.loads(IDENTITY.read_text())[SOURCE_KEY]
BASE_PATH = (IDENTITY.parent / SOURCE_KEY).resolve()
BASE_SHA256 = hashlib.sha256(BASE_PATH.read_bytes()).hexdigest()
if BASE_SHA256 != PINNED:
    raise RuntimeError('Original resource probe differs from precise-v2 source identity')
_spec = importlib.util.spec_from_file_location('_pinned_original_resource_probe', BASE_PATH)
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)
deltas = _base.deltas
guard = _base.guard
THERMAL_NAMES = {0: 'nominal', 1: 'fair', 2: 'serious', 3: 'critical'}


def _run(command):
    try:
        r = subprocess.run(command, text=True, capture_output=True, timeout=10)
        return {'command': command, 'returncode': r.returncode,
                'stdout': r.stdout.strip(), 'stderr': r.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {'command': command, 'returncode': None, 'stdout': '',
                'stderr': type(exc).__name__ + ': ' + str(exc)}


def _battery():
    raw = _run(['/usr/bin/pmset', '-g', 'batt'])
    source = re.search(r"Now drawing from '([^']+)'", raw['stdout'])
    percent = re.search(r'(\d+)%', raw['stdout'])
    state = re.search(r'\d+%;\s*([^;\n]+)', raw['stdout'])
    known = raw['returncode'] == 0 and source is not None and percent is not None
    value = int(percent[1]) if percent is not None else None
    known = known and 0 <= value <= 100
    return {'status': 'known' if known else 'unknown',
            'percent': value if known else None,
            'source': source[1] if known else None,
            'state_text': state[1].strip() if state is not None else None,
            'raw': raw}


def _thermal():
    raw = _run(['/usr/bin/osascript', '-l', 'JavaScript', '-e',
                'ObjC.import("Foundation"); String($.NSProcessInfo.processInfo.thermalState)'])
    text = raw['stdout']
    code = int(text) if raw['returncode'] == 0 and re.fullmatch(r'\d+', text) else None
    return {'status': 'known' if code in THERMAL_NAMES else 'unknown',
            'code': code, 'name': THERMAL_NAMES.get(code, 'unknown'), 'raw': raw,
            'scope': 'NSProcessInfo system thermal state at this sample; no causal attribution'}


def _owned_pid(pid):
    if pid is None:
        return {'status': 'unknown', 'pid': None, 'reason': 'owned_pid not supplied', 'raw': None}
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        raise ValueError('owned_pid must be a positive integer or None')
    raw = _run(['/bin/ps', '-p', str(pid), '-o', 'pid=,ppid=,rss=,comm='])
    fields = raw['stdout'].split(None, 3)
    known = raw['returncode'] == 0 and len(fields) == 4 and all(x.isdigit() for x in fields[:3]) and int(fields[0]) == pid
    result = {'status': 'known' if known else 'unknown', 'pid': pid, 'raw': raw,
              'scope': 'Exact supplied PID only; process existence/RSS does not prove model residency or sole GPU use'}
    result.update(ppid=int(fields[1]) if known else None,
                  rss_kib=int(fields[2]) if known else None,
                  executable=fields[3] if known else None)
    return result


def snapshot(label, owned_pid=None):
    # The pinned base result, raw command text and matcher scope remain intact.
    if owned_pid is not None and (not isinstance(owned_pid, int) or isinstance(owned_pid, bool) or owned_pid <= 0):
        raise ValueError('owned_pid must be a positive integer or None')
    result = _base.snapshot(label, owned_pid)
    result['power_metadata'] = _battery()
    result['processinfo_thermal_metadata'] = _thermal()
    result['exact_owned_pid_metadata'] = _owned_pid(owned_pid)
    result['power_wrapper_identity'] = {
        'source_identity_path': str(IDENTITY), 'base_probe_path': str(BASE_PATH),
        'base_probe_sha256': BASE_SHA256,
        'wrapper_path': str(Path(__file__).resolve()),
        'wrapper_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scope': 'Metadata-only wrapper; base matcher/inventory/guard semantics unchanged; unavailable values explicit unknown'}
    return result


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--label', default='metadata-only-snapshot')
    ap.add_argument('--owned-pid', type=int)
    args = ap.parse_args()
    print(json.dumps(snapshot(args.label, args.owned_pid), indent=2))
