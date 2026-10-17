"""Run the frozen authorized E024 diagnostic cycle, always attempting bounded restoration."""
from pathlib import Path
import datetime
import hashlib
import json
import subprocess
import sys
import time

R = Path('/Users/ryanli/code/minivllm')
P = R / 'bench/m5_ultra_256gb/long_decode_graph/prereg-cold-v1'
D = R / '.artifacts/m5-ultra-256gb/long-decode-cold-v1-execution'
J = R / '.artifacts/m5-ultra-256gb/long-decode-cold-v1-cold-journal'
O = R / '.artifacts/m5-ultra-256gb/long-decode-cold-v1-owner'
PY = str(R / '.venv/bin/python')
C = R / 'bench/m5_ultra_256gb/omlx_cold_restart/controller_v5.py'
PLAN_SHA = '8e508928672abddec6a2c06680635dd26e1c8fb1a8456a3b092628cd4e800f2d'
A = ['--journal', str(J), '--approved-contract', 'cold-restart-persisted-visible-v1',
     '--plan-sha256', 'e96a4a4d5be6f6a69d597387bbc566eadedae81065bfa879445352d986515cb8']


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def save(name, value):
    (D / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


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


receipt = json.loads((P / 'receipt.json').read_bytes())
for path, expected in receipt['source_pins'].items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, path
assert hashlib.sha256((P / 'owner-plan.json').read_bytes()).hexdigest() == PLAN_SHA
raw = (J / 'journal.json').read_bytes()
assert hashlib.sha256(raw).hexdigest() == receipt['prepared_journal_sha256']
assert json.loads(raw)['state'] == 'PREPARED' and not O.exists()
D.mkdir(exist_ok=False)
(D / 'prepared-journal.json').write_bytes(raw)
(D / 'execute.py').write_bytes(Path(__file__).read_bytes())
save('authorization.json', dict(recorded_at=now(), basis='Prior explicit human authorization in this thread',
    answer='允许停止、实验后恢复', contract='cold-restart-persisted-visible-v1',
    scope='Continue the same authorized cold-stop/experiment/restore cycle; unchanged E024 gates-OFF diagnostic: original 32-point matrix, parameter replacement checks and short lifecycle; first numerical difference stops; no performance run'))
stop_rc = owner_rc = restore_rc = None
try:
    stop_rc = run('stop', [PY, '-I', '-B', str(C), 'stop', *A])
    if stop_rc == 0 and json.loads((J / 'journal.json').read_bytes())['state'] == 'STOPPED':
        owner_rc = run('diagnostic-owner', [PY, '-I', '-B', str(R / 'bench/m5_ultra_256gb/long_decode_graph/cold_owner.py'),
            '--plan', str(P / 'owner-plan.json'), '--plan-sha256', PLAN_SHA, '--out', str(O)])
finally:
    state = json.loads((J / 'journal.json').read_bytes())['state']
    if state != 'PREPARED':
        command = [PY, '-I', '-B', str(C), 'recover', *A, '--watchdog-seconds', '3600']
        if not O.exists():
            command.append('--no-experiment')
        restore_rc = run('recover', command)
    state = json.loads((J / 'journal.json').read_bytes())['state']
    save('final.json', dict(time=now(), journal_state=state, stop_returncode=stop_rc,
                           owner_returncode=owner_rc, recovery_returncode=restore_rc))
sys.exit(int(stop_rc != 0 or owner_rc != 0 or restore_rc != 0 or state != 'RESTORED'))
