"""E010 fresh-process paired HTTP: original versus exact grouped NAX prefill."""
import argparse
import hashlib
import importlib.util
import json
import os
import signal
import socket
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path

D = Path(__file__).resolve().parent
ROOT = D.parents[2]
BASE = ROOT / '.artifacts/m5-ultra-256gb'
spec = importlib.util.spec_from_file_location('previous_http', D.parent / 'decode_batch/frontier_modes.py')
http = importlib.util.module_from_spec(spec)
spec.loader.exec_module(http)
import httpx
from tokenizers import Tokenizer


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def run_one(out, source, manifest, workload, concurrency, pair, variant):
    out.mkdir()
    model = Path(workload['model_dir'])
    tok = Tokenizer.from_file(str(model / 'tokenizer.json'))
    prompt = tok.decode(workload['prompt_ids'], skip_special_tokens=False)
    assert tok.encode(prompt, add_special_tokens=False).ids == workload['prompt_ids']
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    url = f'http://127.0.0.1:{port}'
    command = [str(ROOT / '.venv/bin/python'), '-B', str(D / 'native_child.py'),
               '--model', str(model), '--port', str(port), '--batch', str(max(4, concurrency)),
               '--trace', str(out / 'engine-trace.json'), '--package-root', str(source),
               '--source-manifest', str(manifest), '--pool', '2', '--graph', '--fuse'] + (['--grouped'] if variant == 'candidate' else [])
    write(out / 'launch.json', {'command': command, 'environment': http.ENV,
          'C': concurrency, 'pair': pair, 'variant': variant})
    child = None
    samples, groups = [], {'warm': [], 'measure': []}
    try:
        http.snapshot(out, samples, 'preflight')
        with (out / 'server.log').open('w') as log:
            child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            write(out / 'owned-pid.json', {'pid': child.pid})
            start = last_probe = time.monotonic()
            with httpx.Client(trust_env=False, timeout=180) as client:
                while True:
                    assert child.poll() is None, 'server exited before ready'
                    try:
                        response = client.get(url + '/v1/models', timeout=1)
                        if response.status_code == 200:
                            identity = response.json()['benchmark_identity']
                            assert identity['allocator_pool_gib'] == 2 and identity['grouped_prefill'] == (variant == 'candidate') and identity['manifest_all_sources_match']
                            break
                    except httpx.HTTPError:
                        pass
                    assert time.monotonic() - start < 900, 'readiness timeout'
                    if time.monotonic() - last_probe > 10:
                        http.snapshot(out, samples, 'loading', child.pid)
                        last_probe = time.monotonic()
                    time.sleep(.2)
            group = http.make_group('tutorial', url, {}, prompt, tok, 'tutorial-core')
            for phase in groups:
                for i in range(3):
                    label = f'C{concurrency}-{phase}-{i}'
                    http.snapshot(out, samples, 'before-' + label, child.pid)
                    row = group(concurrency, label)
                    write(out / (label + '.json'), row)
                    groups[phase].append(row)
                    http.snapshot(out, samples, 'after-' + label, child.pid)
            # Trace is written when the engine becomes idle after each group.
            trace = json.loads((out / 'engine-trace.json').read_text())
            requests = [r for rows in groups.values() for g in rows for r in g['requests']]
            assert len(requests) == 6 * concurrency
            assert all(r['first_chunk_prefix_verified'] and r['full_visible_roundtrip_count_matches_usage'] for r in requests)
            grouped_calls = [r for r in trace['prefill_calls'] if r['mode'] == 'prefill_batch']
            assert bool(grouped_calls) == (variant == 'candidate' and concurrency > 1)
            result = {'C': concurrency, 'pair': pair, 'variant': variant, 'groups': groups,
                      'runtime_identity': identity, 'prefill_calls': trace['prefill_calls'],
                      'x': statistics.median(g['x_client_estimated_interactivity_tok_s_user'] for g in groups['measure']),
                      'y': statistics.median(g['y_total_output_throughput_tok_s_mac'] for g in groups['measure']),
                      'TTFT_s': statistics.median(statistics.median(r['client_ttft_text_s'] for r in g['requests']) for g in groups['measure']),
                      'peak_active_gib': trace['mlx_peak_bytes'] / 1024 ** 3,
                      'max_recorded_cache_gib': max(t['mlx_cache_bytes'] for t in trace['steps']) / 1024 ** 3,
                      'max_recorded_active_gib': max(t['mlx_active_bytes'] for t in trace['steps']) / 1024 ** 3}
            write(out / 'result.json', result)
    except BaseException:
        write(out / 'failure.json', {'traceback': traceback.format_exc()})
        raise
    finally:
        if child is not None:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=15)
            child.wait()
            write(out / 'release.json', {'pid': child.pid, 'returncode': child.returncode,
                                        'waited_reaped': True, 'absence': http.absence(child.pid)})
        http.snapshot(out, samples, 'postflight')
    return {k: v for k, v in result.items() if k not in ('groups', 'runtime_identity', 'prefill_calls')}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--suspended-pid', type=int)
    ap.add_argument('--workload', type=Path, required=True)
    args = ap.parse_args()
    sys.path.insert(0,str(D.parent))
    import resources_v2
    from functools import partial
    from types import SimpleNamespace
    suspended = (args.suspended_pid,) if args.suspended_pid is not None else ()
    http.resources=SimpleNamespace(snapshot=partial(resources_v2.snapshot,suspended_pids=suspended),guard=resources_v2.guard,deltas=resources_v2.deltas)
    args.out = args.out.resolve()
    args.out.mkdir()
    source = args.source.resolve()
    files = {str(p.relative_to(source / 'tutorial_code/miniengine')): digest(p)
             for p in (source / 'tutorial_code/miniengine').rglob('*.py')}
    manifest = args.out / 'source.json'
    write(manifest, {'files': files})
    pins = {str(p): digest(p) for p in D.glob('*.py')}
    pins.update({str(p): digest(p) for p in (D.parent / 'decode_batch').glob('*.py')})
    pins.update({str(p): digest(p) for p in (D.parent / 'prefill_nax').glob('*.py')})
    for name in ('resources_v2.py','resources.py','resource_base.py','suspended_service.py'):pins[str(D.parent/name)]=digest(D.parent/name)
    pins.update({str(p): digest(p) for p in (BASE / 'build2-harness').glob('*.py')})
    pins[str(D.parent / 'lithos/client.py')] = digest(D.parent / 'lithos/client.py')
    # Existing measured source is frozen; no dependency on mutable working code.
    frozen = json.loads((BASE / 'pareto-mode-frontier-v1/inputs/frozen.json').read_text())['files']
    # Pin runtime/model artifacts, not historical implementation scripts.
    for p, sha in frozen.items():
        if '/site-packages/' in p or '/snapshots/' in p:
            pins[p] = sha
    configs = [(c, pair, variant) for c in (1, 8, 32, 64) for pair in (1, 2, 3)
               for variant in (('reference', 'candidate') if pair % 2 else ('candidate', 'reference'))]
    write(args.out / 'plan.json', {'source': str(source), 'source_files': files,
          'pins': pins, 'workload': json.loads(args.workload.read_text()),
          'configs': configs, 'environment': http.ENV, 'authorized_suspended_service_pid':args.suspended_pid,
          'change': 'Integrated grouped NAX prefill versus disabled grouping on same frozen core; 2 GiB pool; native process-title probe v2; service paused by explicit user authorization; original contaminated run retained',
          'acceptance': 'C64 paired throughput geometric mean >=1.05 and all three pairs >1; paired TTFT ratio <=0.9; C1 median x/y >=0.97. C8/C32 each paired throughput >=0.97; exact qualification/count/resource/cleanup pass',
          'scope': '19 input / 32 output tokens, normal EOS; three warm and three measured groups per fresh process; no outlier removal'})
    def verify():
        for p, sha in pins.items():
            assert http.digest(p) == sha, p
        assert all(digest(source / 'tutorial_code/miniengine' / p) == sha for p, sha in files.items())
        assert {k: v for k, v in os.environ.items() if k.startswith(('MLX_', 'METAL_'))} == http.ENV
    verify()
    rows = []
    for c, pair, variant in configs:
        print('START', c, pair, variant, flush=True)
        rows.append(run_one(args.out / f'C{c}-pair{pair}-{variant}', source,
                            manifest, json.loads(args.workload.read_text()), c, pair, variant))
        write(args.out / 'progress.json', rows)
        print(json.dumps(rows[-1]), flush=True)
        verify()
    write(args.out / 'complete.json', {'status': 'PASS', 'rows': rows})


if __name__ == '__main__':
    def stop(signum, frame):
        raise KeyboardInterrupt(str(signum))
    signal.signal(signal.SIGTERM, stop)
    main()
