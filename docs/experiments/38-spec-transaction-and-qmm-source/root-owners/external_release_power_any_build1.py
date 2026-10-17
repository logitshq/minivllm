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
ap.add_argument('--expected-power',choices=('AC','Battery','any'),required=True)
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
probe(samples, o / 'external-resource-anypower.json', 'external-release-' + a.directory, a.expected_power)
receipt = {'status': 'PASS_EXTERNAL_RELEASE', 'strict_absence': rows,
           'GPU_released': True, 'tool_exit': a.tool_exit, 'result': a.result,
           'release_expected_power':a.expected_power, 'release_observed_power':samples[-1]['power_metadata']['source'],
           'scope':'post-tool PID absence / external resources only; allowing release power metadata does not qualify or pool performance domains'}
(o / 'external-release.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt))
