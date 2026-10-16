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
    for c, pair, variant in plan['configs']:
        directory = root / f'C{c}-pair{pair}-{variant}'
        reported = read(directory / 'result.json')
        identity = reported['runtime_identity']
        assert identity['allocator_pool_gib'] == 2
        assert identity['grouped_prefill'] == (variant == 'candidate')
        adapter_path = str(Path(__file__).resolve().parent/'native_child.py')
        assert identity['integration_adapter_sha256'] == plan['pins'][adapter_path]
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
        result = {'C': c, 'pair': pair, 'variant': variant}
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
            assert resource['native_process_inventory_ok'] and resource['no_active_foreign_native_processes']
            assert resource['pressure_level_value'] == 1
            assert resource['processinfo_thermal_metadata']['code'] in (0, 1)
            assert resource['power_metadata']['source'] == 'AC Power'
            resources.append(resource)
        rows.append(result)
    assert len(rows) == len(complete['rows']) == 24
    by_key = {(r['C'], r['pair'], r['variant']): r for r in rows}
    ratios = {str(c): {key: [by_key[c, p, 'candidate'][key] / by_key[c, p, 'reference'][key]
                           for p in (1, 2, 3)] for key in ('x', 'y', 'TTFT_s')}
              for c in (1, 8, 32, 64)}
    geomeans = {c: {k: statistics.geometric_mean(v) for k, v in d.items()} for c, d in ratios.items()}
    medians = {str(c): {v: {key: statistics.median(r[key] for r in rows if r['C'] == c and r['variant'] == v)
                            for key in ('x', 'y', 'TTFT_s')}
                       for v in ('reference', 'candidate')} for c in (1, 8, 32, 64)}
    control = {key: medians['1']['candidate'][key] / medians['1']['reference'][key] for key in ('x', 'y')}
    adopt = (geomeans['64']['y'] >= 1.05 and min(ratios['64']['y']) > 1
             and geomeans['64']['TTFT_s'] <= .9 and min(control.values()) >= .97
             and all(geomeans[c]['y'] >= .97 for c in ('8', '32')))
    return {'status': 'PASS', 'processes': len(rows), 'all_requests': all_requests,
            'measured_requests': measured_requests, 'measured_output_tokens': measured_requests * 32,
            'resource_samples': len(resources), 'max_thermal_code': max(r['processinfo_thermal_metadata']['code'] for r in resources),
            'rows': rows, 'medians': medians, 'paired_ratios': ratios, 'paired_geomeans': geomeans,
            'C1_control_ratios': control, 'adopt_grouped_prefill': adopt,
            'decision': 'Integrated M5 grouped prefill passes preregistered HTTP gates' if adopt else 'Do not integrate: preregistered performance gate failed',
            'scope': 'Native-only paired HTTP, 19->32, C1/8/32/64; no new official-runtime comparison or universal Pareto claim.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.root)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('rows', 'medians')}, indent=2))
