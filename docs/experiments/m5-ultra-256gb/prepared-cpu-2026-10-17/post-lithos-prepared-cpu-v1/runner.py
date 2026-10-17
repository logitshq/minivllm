"""Run prepared follow-up CPU tests only after the real Lithos cycle restores."""
import datetime
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time

ROOT = Path('/Users/ryanli/code/minivllm')
ART = ROOT / '.artifacts/m5-ultra-256gb'
final = json.loads((ART / 'lithos-d512-formal-v1-execution/final.json').read_bytes())
journal = json.loads((ART / 'lithos-d512-formal-v1-cold-journal/journal.json').read_bytes())
assert final['journal_state'] == journal['state'] == 'RESTORED'
assert final['recovery_returncode'] == 0
out = ART / 'post-lithos-prepared-cpu-v1'
out.mkdir(exist_ok=False)
modules = [
    'cold_cycle_v2.test_cpu',
    'cold_cycle_v2.test_predecessor_cpu',
    'route_decode_graph.test_integration_cpu',
    'route_decode_graph.test_serving_qualification_cpu',
    'route_decode_graph.test_audit_serving_cpu',
    'route_decode_graph.test_execute_serving_cpu',
    'omlx_d512.test_v2_cpu',
    'omlx_d512.test_owner_cpu',
    'glm_flash.test_load_probe_cpu',
]
sources = []
for directory in ('cold_cycle_v2', 'route_decode_graph', 'omlx_d512', 'glm_flash'):
    sources.extend(sorted((ROOT / 'bench/m5_ultra_256gb' / directory).glob('*.py')))
def pins():
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
before = pins()
command = [str(ROOT / '.venv/bin/python'), '-B', '-m', 'unittest', '-v',
           *['bench.m5_ultra_256gb.' + name for name in modules]]
record = dict(status='RUNNING', command=command, source_pins=before,
              started_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
              scope='CPU files/fakes only; no GPU, service, numerical or performance qualification.')
def save():
    (out / 'result.json').write_text(json.dumps(record, indent=2) + '\n')
save()
started = time.monotonic()
(out / 'runner.py').write_bytes(Path(__file__).read_bytes())
try:
    with (out / 'stdout.txt').open('xb') as stdout, (out / 'stderr.txt').open('xb') as stderr:
        result = subprocess.run(command, cwd=ROOT, stdout=stdout, stderr=stderr, check=False, timeout=180)
except BaseException as exc:
    record.update(status='FAIL', error_type=type(exc).__name__, error=str(exc), elapsed_seconds=time.monotonic()-started)
    save()
    raise
after = pins()
text = (out / 'stderr.txt').read_text()
match = re.search(r'Ran (\d+) tests? in ([\d.]+)s', text)
passed = result.returncode == 0 and before == after and match is not None and int(match[1]) == 91
record.update(status='PASS' if passed else 'FAIL', returncode=result.returncode,
              elapsed_seconds=time.monotonic()-started, source_pins_unchanged=before == after,
              tests=int(match[1]) if match else None)
save()
print(json.dumps({k:v for k,v in record.items() if k not in ('source_pins','command')}))
if not passed:
    print(text[-18000:])
raise SystemExit(0 if passed else 1)
