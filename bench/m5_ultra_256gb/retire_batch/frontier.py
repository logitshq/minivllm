"""Fresh-process HTTP measurements of the qualified batched-retirement source."""
import argparse, hashlib, importlib.util, json, os, signal, subprocess, traceback
from pathlib import Path

D = Path(__file__).resolve().parent
ROOT = D.parents[2]
BASE = ROOT / '.artifacts/m5-ultra-256gb'
driver = D.parent / 'decode_batch/frontier_modes.py'
spec = importlib.util.spec_from_file_location('retirement_http', driver)
http = importlib.util.module_from_spec(spec)
spec.loader.exec_module(http)

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while block := f.read(1 << 20):
            h.update(block)
    return h.hexdigest()

def write(p, obj):
    p.write_text(json.dumps(obj, indent=2) + '\n')

def main():
    def stop(signum, frame):
        raise KeyboardInterrupt(str(signum))
    signal.signal(signal.SIGTERM, stop)
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    out, source = a.out.resolve(), a.source.resolve()
    out.mkdir()
    inputs = out / 'inputs'
    inputs.mkdir()
    qualified = BASE / 'retire-batch-qual-v1'
    qual = json.loads((qualified / 'complete.json').read_text())
    assert qual['status'] == 'PASS'
    for p, h in qual['source_files'].items():
        assert digest(p) == h, p
    files = json.loads((BASE / 'pareto-mode-frontier-v1/inputs/frozen.json').read_text())['files']
    files.update({str(p): digest(p) for p in source.rglob('*.py')})
    files.update({str(p): digest(p) for p in D.glob('*.py')})
    files.update({str(p): digest(p) for p in qualified.glob('*.json')})
    files[str(BASE / 'retire-batch-plan-v1.json')] = digest(BASE / 'retire-batch-plan-v1.json')
    source_files = {str(p.relative_to(source / 'tutorial_code/miniengine')): digest(p)
                    for p in (source / 'tutorial_code/miniengine').rglob('*.py')}
    write(inputs / 'source.json', {'files': source_files})
    configs = [(c, pair, 'fused') for c in (8, 32, 64) for pair in (1, 2, 3)]
    write(inputs / 'frozen.json', {'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
          'implementation': 'batched retirement candidate; exact source hashes identify this uncommitted prototype',
          'files': files, 'environment': http.ENV, 'configs': configs})
    write(out / 'attempt.json', {'owner_pid': os.getpid(), 'configs': configs})
    http.OUT, http.SOURCE, http.INPUTS = out, source, inputs
    rows = []
    try:
        for c, pair, engine in configs:
            http.verify()
            print('START retirement', c, pair, flush=True)
            rows.append(http.run_one(c, pair, engine))
            write(out / 'progress.json', rows)
            print(json.dumps(rows[-1]), flush=True)
        http.verify()
        write(out / 'complete.json', {'rows': rows, 'owner_pid': os.getpid(),
              'scope': 'Three fresh processes each at C8/32/64; earlier baselines are unpaired.'})
    except BaseException:
        write(out / 'failure.json', {'traceback': traceback.format_exc(), 'completed': rows})
        raise

if __name__ == '__main__':
    main()
