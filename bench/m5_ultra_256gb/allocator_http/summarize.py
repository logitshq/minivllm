"""Independently recompute every HTTP group and evaluate the frozen decision rule."""
import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def close(a, b):
    assert math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-10), (a, b)


def summarize(root):
    plan = read(root / 'plan.json')
    complete = read(root / 'complete.json')
    assert complete['status'] == 'PASS'
    rows, all_requests, measured_requests, resources = [], 0, 0, []
    for c, pair, pool in plan['configs']:
        directory = root / f'C{c}-pair{pair}-pool{pool}'
        reported = read(directory / 'result.json')
        identity = reported['runtime_identity']
        assert identity['allocator_pool_gib'] == pool
        assert identity['actual_source_files_sha256'] == plan['source_files']
        assert identity['compile_decode'] and identity['fuse_decode']
        measured = []
        for phase in ('warm', 'measure'):
            for i in range(3):
                group = read(directory / f'C{c}-{phase}-{i}.json')
                requests = group['requests']
                assert len(requests) == c == group['actual_request_count']
                all_requests += c
                measured_requests += c if phase == 'measure' else 0
                xs, ttfts = [], []
                for r in requests:
                    ids = r['output_ids_when_available']
                    assert len(ids) == 32 and ids == r['visible_roundtrip_ids']
                    assert r['usage']['prompt_tokens'] == 19 and r['usage']['completion_tokens'] == len(ids)
                    assert r['usage']['prompt_tokens_details']['cached_tokens'] == 0
                    assert r['finish_reason'] == 'length'
                    first, done, start = (r[k] for k in ('first_text_perf_counter_s', 'done_perf_counter_s', 'start_perf_counter_s'))
                    assert start <= first <= done <= r['cleanup_complete_perf_counter_s']
                    times = [e['client_perf_counter_s'] for e in r['events']]
                    assert times == sorted(times) and first == times[0] and times[-1] <= done
                    assert r['text'] == ''.join(e['text'] for e in r['events'])
                    assert r['first_chunk_prefix_verified'] and r['full_visible_roundtrip_count_matches_usage']
                    x = (len(ids) - r['first_chunk_reconstructed_tokens']) / (done - first)
                    close(x, r['x_client_estimated_interactivity_tok_s_user'])
                    xs.append(x)
                    ttfts.append(first - start)
                window = max(r['done_perf_counter_s'] for r in requests) - min(r['start_perf_counter_s'] for r in requests)
                tokens = sum(len(r['output_ids_when_available']) for r in requests)
                x, y = statistics.median(xs), tokens / window
                close(window, group['group_window_s'])
                close(y, group['y_total_output_throughput_tok_s_mac'])
                close(x, group['x_client_estimated_interactivity_tok_s_user'])
                assert tokens == group['total_completion_tokens']
                if phase == 'measure':
                    measured.append((x, y, statistics.median(ttfts)))
        result = {'C': c, 'pair': pair, 'pool_gib': pool}
        for index, key in enumerate(('x', 'y', 'TTFT_s')):
            result[key] = statistics.median(m[index] for m in measured)
            close(result[key], reported[key])
        trace = read(directory / 'engine-trace.json')
        result['peak_active_gib'] = trace['mlx_peak_bytes'] / 1024 ** 3
        result['max_recorded_cache_gib'] = max(s['mlx_cache_bytes'] for s in trace['steps']) / 1024 ** 3
        release = read(directory / 'release.json')
        assert release['waited_reaped'] and release['absence']['returncode'] == 1 and not release['absence']['stdout']
        for p in sorted(directory.glob('resource-*-raw.json')):
            resource = read(p)
            assert resource['pressure_level_value'] == 1
            assert resource['processinfo_thermal_metadata']['code'] in (0, 1)
            assert resource['power_metadata']['source'] == 'AC Power'
            resources.append(resource)
        rows.append(result)
    assert len(rows) == len(complete['rows']) == 12
    by_key = {(r['C'], r['pair'], r['pool_gib']): r for r in rows}
    ratios = [by_key[64, p, 16]['y'] / by_key[64, p, 2]['y'] for p in (1, 2, 3)]
    medians = {str(c): {str(pool): {key: statistics.median(r[key] for r in rows if r['C'] == c and r['pool_gib'] == pool)
                                   for key in ('x', 'y', 'TTFT_s')}
                        for pool in (2, 16)} for c in (1, 64)}
    control = {key: medians['1']['16'][key] / medians['1']['2'][key] for key in ('x', 'y')}
    gain = statistics.geometric_mean(ratios)
    adopt = gain >= 1.05 and min(ratios) > 1 and min(control.values()) >= .97
    return {'status': 'PASS', 'processes': len(rows), 'all_requests': all_requests,
            'measured_requests': measured_requests, 'measured_output_tokens': measured_requests * 32,
            'resource_samples': len(resources), 'max_thermal_code': max(r['processinfo_thermal_metadata']['code'] for r in resources),
            'rows': rows, 'medians': medians, 'C64_paired_throughput_ratios': ratios,
            'C64_geomean_throughput_ratio': gain, 'C1_control_ratios': control,
            'adopt_16_gib_on_m5': adopt,
            'decision': 'M5 workload-specific configuration eligible' if adopt else 'Retain 2 GiB default: preregistered performance threshold not met',
            'scope': 'Instrumented native-only paired HTTP; unchanged arithmetic/source. No new official-runtime comparison or universal Pareto claim.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.root)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('rows', 'medians')}, indent=2))
