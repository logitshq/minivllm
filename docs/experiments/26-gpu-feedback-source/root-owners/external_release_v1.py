"""Root post-tool proof for an already reaped owner and its recorded children."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

p = Path('/private/tmp/minivllm-engine-optimization-20261014')
ap = argparse.ArgumentParser()
ap.add_argument('--directory', required=True)
ap.add_argument('--tool-exit', required=True, type=int)
ap.add_argument('--result', required=True)
a = ap.parse_args()
o = p / 'v23-root-control-build1' / a.directory
assert o.parent == p / 'v23-root-control-build1' and o.is_dir()
assert not (o / 'external-release.json').exists()
owner = json.loads((o / 'owner-receipt.json').read_text())
assert owner['waited_reaped'] and owner['returncode'] == a.tool_exit
pids = {owner['owner_pid']}
for key in ['driver_pid', 'child_pid']:
    if key in owner: pids.add(owner[key])
child = o / 'child-output/child.json'
if child.exists(): pids.add(json.loads(child.read_text())['pid'])
rows = []
for pid in sorted(pids):
    q = subprocess.run(['/bin/ps', '-p', str(pid), '-o', 'pid='], text=True,
                       capture_output=True, timeout=10)
    assert q.returncode == 1 and q.stdout == q.stderr == ''
    rows.append({'pid': pid, 'returncode': q.returncode, 'stdout': q.stdout, 'stderr': q.stderr})
sys.path.insert(0, str(p / 'v25-c1-screen-build1'))
from driver import probe
samples = []
probe(samples, o / 'external-resource.json', 'external-release-' + a.directory, 'AC')
receipt = {'status': 'PASS_EXTERNAL_RELEASE', 'strict_absence': rows,
           'GPU_released': True, 'tool_exit': a.tool_exit, 'result': a.result}
(o / 'external-release.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt))
