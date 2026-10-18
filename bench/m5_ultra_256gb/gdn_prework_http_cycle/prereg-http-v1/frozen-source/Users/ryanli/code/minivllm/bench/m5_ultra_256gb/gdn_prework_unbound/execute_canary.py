"""Run one preregistered GDN real-activation canary; always attempt restoration.

Execution-only driver. The fixed receipt seals sources, owner plan and the
read-only PREPARED journal. No service or subprocess action occurs on import.
"""
from pathlib import Path
import datetime
import hashlib
import json
import subprocess
import sys
import time

R = Path('/Users/ryanli/code/minivllm')
P = R / 'bench/m5_ultra_256gb/gdn_prework_unbound/prereg-canary-v1'
D = R / '.artifacts/m5-ultra-256gb/gdn-prework-canary-v1-execution'
J = R / '.artifacts/m5-ultra-256gb/gdn-prework-canary-v1-cold-journal'
O = R / '.artifacts/m5-ultra-256gb/gdn-prework-canary-v1-owner'
PY = str(R / '.venv/bin/python')
C = R / 'bench/m5_ultra_256gb/cold_cycle_v3/controller.py'
OWNER = C.with_name('owner.py')
A = ['--journal', str(J), '--approved-contract', 'cold-restart-persisted-visible-v1',
     '--plan-sha256', 'e96a4a4d5be6f6a69d597387bbc566eadedae81065bfa879445352d986515cb8']


def need(ok, message):
    if not ok: raise ValueError(message)


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()


def save(name, value):
    (D / name).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def run(name, command):
    record = dict(phase=name, argv=command, start=now(), monotonic_start=time.monotonic())
    save(name + '-command.json', record)
    print(name + ' START ' + record['start'], flush=True)
    with (D / (name + '.stdout')).open('xb') as out, (D / (name + '.stderr')).open('xb') as err:
        result = subprocess.run(command, cwd=R, stdout=out, stderr=err, check=False)
    record.update(returncode=result.returncode, end=now(), elapsed_seconds=time.monotonic()-record['monotonic_start'])
    save(name + '-command.json', record)
    print(name + ' END rc=' + str(result.returncode) + ' ' + record['end'], flush=True)
    return result.returncode


def main():
    if sys.argv[1:] != ['--run']:
        print('Prepared driver. Explicit --run requires the frozen receipt and actual PREPARED journal. No operations performed.')
        return 0
    receipt_raw = (P / 'receipt.json').read_bytes()
    receipt = json.loads(receipt_raw)
    need(receipt['schema'] == 'GDN-prework-canary-prereg-receipt-v1' and receipt['status'] == 'PREPARED',
         'actual canary registration receipt required')
    pins = receipt['source_pins']
    need(type(pins) is dict and all(str(path) in pins for path in
         (Path(__file__).resolve(), C, OWNER, P / 'owner-plan.json')), 'execution/source/plan pins missing')
    for path, expected in pins.items():
        need(Path(path).is_absolute() and sha(path) == expected, 'registered source changed: ' + path)
    plan_sha = receipt['owner_plan_sha256']
    need(type(plan_sha) is str and len(plan_sha) == 64 and sha(P / 'owner-plan.json') == plan_sha,
         'registered owner plan changed')
    raw = (J / 'journal.json').read_bytes()
    need(hashlib.sha256(raw).hexdigest() == receipt['prepared_journal_sha256'], 'prepared journal changed')
    need(json.loads(raw)['state'] == 'PREPARED' and not O.exists(), 'fresh prepared cycle required')
    D.mkdir(exist_ok=False)
    (D / 'prepared-journal.json').write_bytes(raw)
    (D / 'execution-receipt.json').write_bytes(receipt_raw)
    (D / 'execute.py').write_bytes(Path(__file__).read_bytes())
    save('authorization.json', dict(recorded_at=now(), basis='Prior explicit human authorization in this thread',
        answer='允许停止、实验后恢复', contract='cold-restart-persisted-visible-v1',
        scope='Same authorized cold-stop/experiment/restore contract. One direct original-CLI GDN local-layer canary: actual shortest and longest D fixtures, normal EOS, gates ON/norm OFF; 18 selected layer calls, 54 original-dtype array pairs and 60 raw files. Candidate carries its own state between the three actual reference activations. Exact child plan and original 4dff source/runtime/model preparation; work1830 and child1800 limits plus bounded hard reap. Prior route HTTP may have failed numerically but must be independently restored. Execution success is not numerical, full-model, HTTP or performance qualification; separate closed raw/resource/restoration audits are required. Unknown launch never becomes invented absence.'))
    stop_rc = owner_rc = restore_rc = None
    owner_invocation_attempted = owner_invocation_returned = False
    try:
        stop_rc = run('stop', [PY, '-I', '-B', str(C), 'stop', *A])
        if stop_rc == 0 and json.loads((J / 'journal.json').read_bytes())['state'] == 'STOPPED':
            owner_invocation_attempted = True
            save('owner-invocation.json', dict(attempted=True, returned=False, time=now()))
            owner_rc = run('gdn-canary-owner', [PY, '-I', '-B', str(OWNER),
                '--plan', str(P / 'owner-plan.json'), '--plan-sha256', plan_sha, '--out', str(O)])
            owner_invocation_returned = True
            save('owner-invocation.json', dict(attempted=True, returned=True, returncode=owner_rc, time=now()))
    finally:
        state = json.loads((J / 'journal.json').read_bytes())['state']
        if state != 'PREPARED':
            command = [PY, '-I', '-B', str(C), 'recover', *A, '--watchdog-seconds', '3600']
            # An absent directory is insufficient after an unknown process launch.
            # A normal subprocess.run return proves the direct owner was waited.
            if not O.exists() and (not owner_invocation_attempted or owner_invocation_returned):
                command.append('--no-experiment')
            restore_rc = run('recover', command)
        state = json.loads((J / 'journal.json').read_bytes())['state']
        save('final.json', dict(time=now(), journal_state=state, stop_returncode=stop_rc,
                               owner_returncode=owner_rc, recovery_returncode=restore_rc,
                               owner_invocation_attempted=owner_invocation_attempted,
                               owner_invocation_returned=owner_invocation_returned))
    return int(stop_rc != 0 or owner_rc != 0 or restore_rc != 0 or state != 'RESTORED')


if __name__ == '__main__':
    sys.exit(main())
