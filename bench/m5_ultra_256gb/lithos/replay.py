"""Stdlib-only reconstruction of every C1 request and process statistic."""
import hashlib
import json
import math
import statistics
from pathlib import Path

ROOT = Path('/Users/ryanli/code/minivllm/.artifacts/m5-ultra-256gb')
OUT = ROOT / 'lithos-frontier-v1'
complete = json.loads((OUT / 'complete.json').read_text())
freeze = json.loads((ROOT / 'lithos-inputs-v1/frozen.json').read_text())
for p, h in freeze['files'].items():
    assert hashlib.sha256(Path(p).read_bytes()).hexdigest() == h, p
def eq(a, b):
    assert math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12), (a, b)
assert len(complete['rows']) == 9
counts = {'requests': 0, 'measured_requests': 0, 'tokens': 0, 'resources': 0}
processes = []
for row in complete['rows']:
    path = Path(row['path'])
    assert hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256']
    data = json.loads(path.read_text())
    assert data['concurrency'] == 1 and data['normal_EOS'] and data['draft'] is None
    assert data['process_exit']['waited_reaped']
    assert data['process_exit']['absence']['returncode'] == 1
    assert not data['process_exit']['absence']['stdout']
    for phase in ['warm', 'measure']:
        assert len(data['groups'][phase]) == 3
        for g in data['groups'][phase]:
            assert len(g['requests']) == 1
            r = g['requests'][0]
            assert r['usage']['prompt_tokens'] == 19 and r['usage']['completion_tokens'] == 32
            assert r['finish_reason'] == 'length'
            assert r['first_chunk_prefix_verified'] and r['full_visible_roundtrip_count_matches_usage']
            assert len(r['visible_roundtrip_ids']) == 32
            if data['engine'] == 'tutorial':
                assert r['output_ids_when_available'] == r['visible_roundtrip_ids']
            eq(r['client_total_s'], r['done_perf_counter_s'] - r['start_perf_counter_s'])
            eq(r['client_ttft_text_s'], r['first_text_perf_counter_s'] - r['start_perf_counter_s'])
            eq(r['x_client_estimated_interactivity_tok_s_user'],
               (32 - r['first_chunk_reconstructed_tokens']) / (r['done_perf_counter_s'] - r['first_text_perf_counter_s']))
            eq(g['y_total_output_throughput_tok_s_mac'], 32 / r['client_total_s'])
            eq(g['x_client_estimated_interactivity_tok_s_user'], r['x_client_estimated_interactivity_tok_s_user'])
            counts['requests'] += 1
            counts['measured_requests'] += phase == 'measure'
            counts['tokens'] += 32
    for key, source in [('x', 'x_client_estimated_interactivity_tok_s_user'), ('y', 'y_total_output_throughput_tok_s_mac')]:
        eq(data[key], statistics.median(g[source] for g in data['groups']['measure']))
        eq(row[key], data[key])
    if data['engine'] == 'lithos':
        assert data['lithos_profile'] == 'apple-m5-ultra-80c'
        assert len(data['lithos_native_metrics']) == 6
        assert all(m['cached_prompt_tokens'] == 0 and m['verify_tokens'] == 1 for m in data['lithos_native_metrics'])
        assert all(m['setup_ms'] == 0 for m in data['lithos_native_metrics'][3:])
    for r in data['resources']:
        assert r['pressure_level_value'] == 1
        assert r['power_metadata']['source'] == 'AC Power'
        assert r['processinfo_thermal_metadata']['code'] in (0, 1)
        counts['resources'] += 1
    processes.append({k: row[k] for k in ['engine', 'pair', 'x', 'y', 'TTFT_s']})
assert counts['requests'] == 54 and counts['measured_requests'] == 27 and counts['tokens'] == 1728
summary = []
for engine in ['tutorial', 'omlx', 'lithos']:
    rows = [r for r in processes if r['engine'] == engine]
    assert sorted(r['pair'] for r in rows) == [1, 2, 3]
    summary.append(dict(engine=engine, **{k: statistics.median(r[k] for r in rows) for k in ['x', 'y', 'TTFT_s']}))
result = dict(status='PASS_CPU_RAW_REPLAY', counts=counts, processes=processes, summary=summary,
              scope='M5 Ultra C1, local Lithos native adaptation with accelerator on. No cross-engine exact-ID or full-model numerical-equivalence claim.')
(OUT / 'replay.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
