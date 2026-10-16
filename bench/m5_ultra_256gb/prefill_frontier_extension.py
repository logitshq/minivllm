"""Fill C2/C4/C16 on the final native frontier after the paired acceptance gate."""
import argparse
from functools import partial
import importlib.util
import json
from pathlib import Path
import os
import signal
import sys
from types import SimpleNamespace

D = Path(__file__).resolve().parent
sys.path.insert(0, str(D))
import resources_v2
spec = importlib.util.spec_from_file_location('paired_prefill', D / 'prefill_http_v2/run.py')
paired = importlib.util.module_from_spec(spec)
spec.loader.exec_module(paired)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--parent', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--suspended-pid', required=True, type=int)
    a = p.parse_args()
    a.out = a.out.resolve()
    a.out.mkdir()
    summary = json.loads((a.parent / 'summary.json').read_text())
    assert summary['adopt_grouped_prefill'], 'Paired acceptance must finish before extension'
    original = json.loads((a.parent / 'plan.json').read_text())
    pins = dict(original['pins'])
    pins[str(Path(__file__).resolve())] = paired.digest(__file__)
    source = Path(original['source'])
    configs = [(c, replicate, 'candidate') for c in (2, 4, 16) for replicate in (1, 2, 3)]
    plan = {**original, 'pins': pins, 'configs': configs, 'parent': str(a.parent.resolve()),
            'scope': 'Candidate-only frontier extension; three fresh processes per C2/C4/C16, three warm plus three measured groups. Legacy pair field denotes replicate; no new paired baseline claim.',
            'acceptance': 'All count, source, resource and cleanup checks; no outlier removal or configuration selection.',
            'change': 'No code change: fill unmeasured concurrency points on final accepted core',
            'authorized_suspended_service_pid': a.suspended_pid}
    paired.write(a.out / 'plan.json', plan)
    manifest = a.out / 'source.json'
    paired.write(manifest, {'files': original['source_files']})
    paired.http.resources = SimpleNamespace(
        snapshot=partial(resources_v2.snapshot, suspended_pids=(a.suspended_pid,)),
        deltas=resources_v2.deltas, guard=resources_v2.guard)
    def verify():
        assert {k: v for k, v in os.environ.items() if k.startswith(('MLX_', 'METAL_'))} == paired.http.ENV
        assert all(paired.digest(p) == sha for p, sha in pins.items())
        assert all(paired.digest(source / 'tutorial_code/miniengine' / p) == sha for p, sha in original['source_files'].items())
    verify()
    rows = []
    for c, replicate, variant in configs:
        print('START', c, replicate, flush=True)
        rows.append(paired.run_one(a.out / f'C{c}-pair{replicate}-{variant}', source, manifest,
                                   original['workload'], c, replicate, variant))
        paired.write(a.out / 'progress.json', rows)
        print(json.dumps(rows[-1]), flush=True)
        verify()
    paired.write(a.out / 'complete.json', {'status': 'PASS', 'rows': rows})


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, lambda signum, frame: (_ for _ in ()).throw(KeyboardInterrupt(str(signum))))
    main()
