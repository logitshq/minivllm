"""Recompute added groups and compare current native medians with archived donors."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


def read(p):
    return json.loads(p.read_text())


def equal(a, b):
    assert math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-10), (a, b)


def dominates(a, b):
    return a['x'] >= b['x'] and a['y'] >= b['y'] and (a['x'] > b['x'] or a['y'] > b['y'])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--paired', required=True, type=Path)
    p.add_argument('--extension', required=True, type=Path)
    p.add_argument('--historical', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    a = p.parse_args()
    paired, plan = read(a.paired / 'summary.json'), read(a.extension / 'plan.json')
    assert paired['adopt_grouped_prefill'] and read(a.extension / 'complete.json')['status'] == 'PASS'
    parent = read(a.paired / 'plan.json')
    assert parent['source_files'] == plan['source_files']
    rows = [r for r in paired['rows'] if r['variant'] == 'candidate']
    measured_requests, resources = 0, 0
    for c, replicate, variant in plan['configs']:
        directory = a.extension / f'C{c}-pair{replicate}-{variant}'
        reported = read(directory / 'result.json')
        identity = reported['runtime_identity']
        assert identity['actual_source_files_sha256'] == plan['source_files'] and identity['grouped_prefill']
        assert identity['allocator_pool_gib'] == 2 and identity['compile_decode'] and identity['fuse_decode']
        values = []
        for phase in ('warm', 'measure'):
            for i in range(3):
                group = read(directory / f'C{c}-{phase}-{i}.json')
                requests = group['requests']
                assert len(requests) == c == group['actual_request_count']
                xs, ttfts = [], []
                for r in requests:
                    ids = r['output_ids_when_available']
                    assert len(ids) == 32 and ids == r['visible_roundtrip_ids']
                    assert r['finish_reason'] == 'length' and r['usage']['prompt_tokens'] == 19
                    assert r['usage']['completion_tokens'] == 32 and r['usage']['prompt_tokens_details']['cached_tokens'] == 0
                    first, start, done = (r[k] for k in ('first_text_perf_counter_s', 'start_perf_counter_s', 'done_perf_counter_s'))
                    assert start <= first <= done <= r['cleanup_complete_perf_counter_s']
                    assert r['text'] == ''.join(e['text'] for e in r['events'])
                    assert first == r['events'][0]['client_perf_counter_s']
                    assert [e['client_perf_counter_s'] for e in r['events']] == sorted(e['client_perf_counter_s'] for e in r['events'])
                    x = (32 - r['first_chunk_reconstructed_tokens']) / (done - first)
                    equal(x, r['x_client_estimated_interactivity_tok_s_user'])
                    xs.append(x)
                    ttfts.append(first - start)
                window = max(r['done_perf_counter_s'] for r in requests) - min(r['start_perf_counter_s'] for r in requests)
                x, y = statistics.median(xs), c * 32 / window
                equal(window, group['group_window_s'])
                equal(x, group['x_client_estimated_interactivity_tok_s_user'])
                equal(y, group['y_total_output_throughput_tok_s_mac'])
                assert group['total_completion_tokens'] == c * 32
                if phase == 'measure':
                    measured_requests += c
                    values.append((x, y, statistics.median(ttfts)))
        row = {'C': c, 'pair': replicate, 'variant': variant}
        for i, key in enumerate(('x', 'y', 'TTFT_s')):
            row[key] = statistics.median(v[i] for v in values)
            equal(row[key], reported[key])
        release = read(directory / 'release.json')
        assert release['waited_reaped'] and release['absence']['returncode'] == 1 and not release['absence']['stdout']
        for file in directory.glob('resource-*-raw.json'):
            r = read(file)
            assert r['native_process_inventory_ok'] and r['no_active_foreign_native_processes']
            assert r['pressure_level_value'] == 1 and r['processinfo_thermal_metadata']['code'] in (0, 1)
            resources += 1
        rows.append(row)
    assert len(rows) == 21 and measured_requests == 198
    lease_dir = a.extension.parent / 'prefill-service-lease-v2'
    restored = read(lease_dir / 'resume.json')
    assert read(lease_dir / 'lease.json')['closed'] and restored['identity_match']
    assert restored['state_after'] and 'T' not in restored['state_after']
    assert read(lease_dir / 'job-0.json')['returncode'] == 0
    assert not (lease_dir / 'watchdog-resume.json').exists()
    native = []
    for c in (1, 2, 4, 8, 16, 32, 64):
        samples = [r for r in rows if r['C'] == c]
        assert len(samples) == 3
        native.append({'C': c, 'engine': 'native', 'family': 'native',
                       **{k: statistics.median(r[k] for r in samples) for k in ('x', 'y', 'TTFT_s')},
                       'range': {k: [min(r[k] for r in samples), max(r[k] for r in samples)] for k in ('x', 'y', 'TTFT_s')}})
    historical = read(a.historical)
    external = [dict(r) for r in historical['points'] if r['family'] != 'fused']
    assert len(external) == 9
    points = native + external
    for point in points:
        point['own_frontier'] = not any(dominates(q, point) for q in points if q['family'] == point['family'])
        point['joint_frontier'] = not any(dominates(q, point) for q in points)
    coverage = [{'engine': r['engine'], 'C': r['C'], 'covered_by_native_C': [n['C'] for n in native if dominates(n, r)]} for r in external]
    result = {'status': 'PASS', 'native_processes': 21, 'paired_plus_extension_processes': 33,
              'additional_measured_requests': measured_requests, 'additional_resource_samples': resources,
              'service_restored_after_extension': restored,
              'paired_plus_extension_measured_requests': paired['measured_requests'] + measured_requests,
              'points': points, 'current_native_process_rows': rows,
              'historical_external_process_rows': [r for r in historical['all_processes'] if r['engine'] in ('omlx', 'omlx_aggressive', 'lithos')],
              'coverage': coverage, 'all_external_configuration_medians_covered': all(r['covered_by_native_C'] for r in coverage),
              'historical_source': str(a.historical), 'historical_sha256': hashlib.sha256(a.historical.read_bytes()).hexdigest(),
              'source_files': plan['source_files'],
              'scope': 'All native points are this final core. oMLX/Lithos points are archived earlier measurements, not rerun here. 19->32 only, target-only, normal EOS, prefix cache off; no confidence or universal dominance claim.'}
    a.out.write_text(json.dumps(result, indent=2) + '\n')
    print('PASS current native frontier;', sum(bool(r['covered_by_native_C']) for r in coverage), '/9 archived external medians covered')


if __name__ == '__main__':
    main()
