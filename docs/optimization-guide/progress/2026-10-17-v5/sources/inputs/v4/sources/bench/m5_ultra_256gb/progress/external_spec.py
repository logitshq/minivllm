"""Offline E020 HTTP charts; never starts a model, server, or GPU runtime.

Run only after the GPU matrix ends. Reads the frozen contract results and
recomputes timestamps/medians/paired ratios; does not change an acceptance gate.
Matplotlib is imported only by render(). Failed processes remain in chartdata
and appear as diagnostics, never as a valid configuration median or speedup.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics as st

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULTS = ROOT / '.artifacts/m5-ultra-256gb/external-spec-http-v1'
ARMS = ('native-none', 'omlx-none', 'omlx-mtp', 'lithos-none', 'lithos-dspark')
LABELS = {'native-none': 'minivllm target-only', 'omlx-none': 'oMLX target-only',
          'omlx-mtp': 'oMLX + MTP', 'lithos-none': 'Lithos target-only',
          'lithos-dspark': 'Lithos + DSpark'}
COLORS = {'native-none': '#127E6B', 'omlx-none': '#B8672D', 'omlx-mtp': '#DF922F',
          'lithos-none': '#526BB1', 'lithos-dspark': '#9B63A4'}
MARKERS = {'native-none': 'o', 'omlx-none': 's', 'omlx-mtp': '^',
           'lithos-none': 'D', 'lithos-dspark': 'P'}
METRICS = ('latency_s', 'throughput', 'ttft_s')


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def same(actual, reported):
    assert math.isclose(actual, reported, rel_tol=1e-10, abs_tol=1e-10), (actual, reported)


class Sources:
    def __init__(self, output, snapshot=None):
        self.output, self.snapshot, self.manifest = output, snapshot, {}

    def read(self, original):
        original = str(original)
        relative = original.replace('.artifacts/', 'raw-artifacts/', 1)
        source = self.snapshot / relative if self.snapshot else ROOT / original
        target = self.output / 'sources' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        payload = target.read_bytes()
        self.manifest[original] = {'snapshot_path': relative,
            'sha256': hashlib.sha256(payload).hexdigest(), 'bytes': len(payload)}
        return json.loads(payload)


def request_metrics(row):
    """Partial/failed requests retain their available diagnostics, without repair."""
    start, first, done = (row.get(k) for k in
                         ('start_perf_counter_s', 'first_text_perf_counter_s', 'done_perf_counter_s'))
    latency = done - start if done is not None and start is not None else None
    ttft = first - start if first is not None and start is not None else None
    count = (row.get('usage') or {}).get('completion_tokens')
    throughput = count / latency if isinstance(count, int) and latency is not None and latency > 0 else None
    for actual, key in ((latency, 'client_total_s'), (ttft, 'client_ttft_text_s'), (throughput, 'output_tok_s')):
        if actual is not None and row.get(key) is not None:
            same(actual, row[key])
    return {'latency_s': latency, 'throughput': throughput, 'ttft_s': ttft,
            'completion_tokens': count, 'request_id': row['request_id'],
            'contract_failures': row.get('contract_failures'), 'error': row.get('error')}


def collect(sources, raw):
    plan = sources.read(f'{raw}/plan.json')
    complete = sources.read(f'{raw}/complete.json')
    assert plan['schema'] == 'E020-v1' and plan['concurrency'] == 1 and plan['normal_eos']
    assert plan['warm_requests'] == plan['formal_requests'] == 3
    assert complete['status'] in ('PASS_CONTRACT', 'FAIL_CONTRACT_RESULTS_RETAINED')
    assert complete['pins_full_sha_match'] is True
    expected = {(arm, pair) for arm in ARMS for pair in (1, 2, 3)}
    assert len(plan['configs']) == len(complete['rows']) == 15
    assert {(r['arm'], r['pair']) for r in plan['configs']} == expected
    reported = {(r['arm'], r['pair']): r for r in complete['rows']}
    assert set(reported) == expected
    processes = []
    for config in plan['configs']:
        arm, pair = config['arm'], config['pair']
        base = f'{raw}/pair{pair}-{arm}'
        result = sources.read(f'{base}/result.json')
        assert result == reported[arm, pair]
        # Retain lifecycle/route evidence without replacing its independent audit.
        route = sources.read(f'{base}/route.json')
        release = sources.read(f'{base}/release.json')
        cleanup = sources.read(f'{base}/cleanup.json')
        records = {'warm': [], 'formal': []}
        for phase in records:
            for index in range(3):
                original = f'{base}/{phase}-{index}.json'
                row = sources.read(original)
                assert (row['arm'], row['pair'], row['phase'], row['index']) == (arm, pair, phase, index)
                point = request_metrics(row)
                point['source'] = original
                records[phase].append(point)
                if result['valid']:
                    # Reconcile the frozen contract, not an alternative gate.
                    assert not row['contract_failures'] and row.get('error') is None
                    assert row['http_status'] == 200 and row['done'] and row['finish_reason'] == 'length'
                    assert row['usage']['prompt_tokens'] == 19 and point['completion_tokens'] == 32
                    assert row['visible_roundtrip_ids'] == plan['oracle_ids'][arm.split('-')[0]]
                    assert row['first_chunk_prefix_verified']
                    assert row.get('cached_prompt_tokens_when_available') in (None, 0)
                    if arm == 'native-none':
                        assert row['output_ids_when_available'] == plan['oracle_ids']['native']
                        assert row['cached_prompt_tokens_when_available'] == 0
                    assert all(math.isfinite(point[k]) and point[k] > 0 for k in METRICS)
                    assert row['start_perf_counter_s'] <= row['first_text_perf_counter_s'] <= row['done_perf_counter_s']
                    assert row['done_perf_counter_s'] <= row['cleanup_complete_perf_counter_s']
        medians = {k: st.median(r[k] for r in records['formal'])
                   if all(r[k] is not None for r in records['formal']) else None for k in METRICS}
        keys = {'latency_s': 'median_wall_s', 'throughput': 'median_output_tok_s', 'ttft_s': 'median_TTFT_s'}
        for key, field in keys.items():
            if medians[key] is not None:
                same(medians[key], result['diagnostic_medians_including_invalid_samples'][field])
            if result['valid']:
                same(medians[key], result[field])
            else:
                assert result[field] is None
        if result['valid']:
            assert route['valid'] and release['valid'] and not cleanup['cleanup_errors']
        processes.append({'arm': arm, 'pair': pair, 'valid': result['valid'],
            'medians': medians, 'requests': records, 'source': f'{base}/result.json',
            'contract_failures': result['contract_failures'], 'fatal_failure': result['fatal_failure']})
    points = []
    for arm in ARMS:
        rows = [p for p in processes if p['arm'] == arm]
        all_valid = all(p['valid'] for p in rows)
        points.append({'arm': arm, 'valid': all_valid,
            'valid_processes': sum(p['valid'] for p in rows),
            'medians': {k: st.median(p['medians'][k] for p in rows) if all_valid else None for k in METRICS},
            'range': {k: [min(p['medians'][k] for p in rows), max(p['medians'][k] for p in rows)]
                      if all_valid else None for k in METRICS}})
    pairs = []
    for engine, on in (('omlx', 'mtp'), ('lithos', 'dspark')):
        for pair in (1, 2, 3):
            off = next(p for p in processes if p['arm'] == engine + '-none' and p['pair'] == pair)
            candidate = next(p for p in processes if p['arm'] == engine + '-' + on and p['pair'] == pair)
            valid = off['valid'] and candidate['valid']
            ratios = {'throughput_speedup': candidate['medians']['throughput'] / off['medians']['throughput'] if valid else None,
                      'TTFT_speedup': off['medians']['ttft_s'] / candidate['medians']['ttft_s'] if valid else None,
                      'wall_speedup': off['medians']['latency_s'] / candidate['medians']['latency_s'] if valid else None}
            old = next(p for p in complete['pairs'] if p['engine'] == engine and p['pair'] == pair)
            assert old['valid'] == valid
            for key, value in ratios.items():
                if valid:
                    same(value, old[key])
                else:
                    assert old[key] is None
            pairs.append({'engine': engine, 'pair': pair, 'valid': valid, **ratios})
    aggregates = {}
    for engine in ('omlx', 'lithos'):
        values = [p['throughput_speedup'] for p in pairs if p['engine'] == engine and p['valid']]
        gm = st.geometric_mean(values) if len(values) == 3 else None
        old = complete['descriptive_aggregates'][engine]
        assert old['three_pairs_complete'] == (len(values) == 3)
        if gm is not None:
            same(gm, old['paired_throughput_geomean'])
        else:
            assert old['paired_throughput_geomean'] is None
        aggregates[engine] = {'paired_throughput_geomean': gm, 'valid_pairs': len(values)}
    return {'experiment': 'E020', 'status': complete['status'], 'processes': processes,
        'points': points, 'pairs': pairs, 'aggregates': aggregates,
        'formal_requests': 45, 'warm_requests': 45,
        'latency_definition': 'HTTP POST start to SSE DONE; cleanup after DONE excluded.',
        'throughput_definition': 'Actual completion usage / POST-to-DONE seconds; C1 only.',
        'aggregation': 'Process median of three formal requests; configuration median of all three fresh process medians. No invalid process dropped to manufacture a configuration median.',
        'scope': plan['scope'], 'decision': plan['decision'],
        'source_audit_scope': 'Timestamp/metric reconciliation only; resource, source and final service restoration require the independent E020 audit.'}


def render(data, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
        'axes.spines.top': False, 'axes.spines.right': False, 'figure.facecolor': 'white',
        'svg.fonttype': 'none', 'svg.hashsalt': 'minivllm-e020-http'})
    fig, ax = plt.subplots(figsize=(13, 9))
    fig.subplots_adjust(left=.10, right=.97, top=.72, bottom=.25)
    fig.text(.10, .95, 'E020 · full-request latency and output throughput', fontsize=21, weight='bold')
    fig.text(.10, .905, 'M5 Ultra 256 GiB · Qwen3.8-27B NVFP4 · C1 · 19 input / 32 output', fontsize=12)
    fig.text(.10, .866, 'Better = upper left. Three fresh servers per configuration; same-run target-only anchors.', fontsize=11)
    handles = []
    for arm in ARMS:
        rows = [p for p in data['processes'] if p['arm'] == arm]
        for row in rows:
            p = row['medians']
            if p['latency_s'] is not None and p['throughput'] is not None:
                ax.scatter(p['latency_s'] * 1000, p['throughput'], s=38,
                    marker=MARKERS[arm] if row['valid'] else 'x',
                    color=COLORS[arm] if row['valid'] else '#B03D48', alpha=.35, zorder=2)
        point = next(p for p in data['points'] if p['arm'] == arm)
        if point['valid']:
            p = point['medians']
            ax.scatter(p['latency_s'] * 1000, p['throughput'], s=150, marker=MARKERS[arm],
                       color=COLORS[arm], edgecolors='white', linewidth=1, zorder=4)
        handles.append(Line2D([], [], color=COLORS[arm], marker=MARKERS[arm], lw=0,
                             markersize=9, label=f"{LABELS[arm]} ({point['valid_processes']}/3 valid)"))
    fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(.095, .827), ncol=2,
               frameon=False, fontsize=10.5)
    ax.set_xlabel('Complete-request latency (ms; lower is better)', labelpad=10)
    ax.set_ylabel('Total output throughput (tokens/s; higher is better)', labelpad=10)
    ax.margins(x=.16, y=.18)
    ax.grid(color='#E5E8EA'); ax.set_axisbelow(True)
    invalid = sum(not p['valid'] for p in data['processes'])
    fig.text(.10, .055,
        'Large marks: medians of three fresh process medians. Faint marks: all process medians; no confidence intervals.\n'
        f'Failed process diagnostics: red × ({invalid}); no valid configuration point unless all three processes pass.\n'
        'C1 and fixed 32 output tokens make throughput = 32 / E2E latency. This is not a concurrency frontier sweep.\n'
        'Correctness is a prerequisite under each engine’s qualified output contract; cross-engine token identity is not claimed.\n'
        'Normal EOS; cache hits rejected. External live raw IDs unavailable: visible IDs checked against prior engine qualification.',
        fontsize=10, color='#50575E', linespacing=1.55)
    for extension in ('png', 'svg'):
        fig.savefig(output / f'e020-throughput-latency.{extension}', dpi=180,
                    metadata={'Date': None} if extension == 'svg' else None)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 7))
    fig.subplots_adjust(left=.065, right=.985, top=.75, bottom=.26, wspace=.25)
    fig.text(.065, .95, 'E020 · within-engine none / speculation pairs', fontsize=21, weight='bold')
    fig.text(.065, .902, 'All three fresh pairs retained. Above 1× is faster; no adoption threshold or selective reruns.', fontsize=12)
    for ax, (key, title) in zip(axes, [('throughput_speedup', 'Output throughput: spec / none'),
                                     ('wall_speedup', 'E2E latency: none / spec'),
                                     ('TTFT_speedup', 'First-text latency: none / spec')]):
        for engine, arm in (('omlx', 'omlx-mtp'), ('lithos', 'lithos-dspark')):
            points = [p for p in data['pairs'] if p['engine'] == engine]
            values = [p[key] if p['valid'] else float('nan') for p in points]
            ax.plot([1, 2, 3], values, marker=MARKERS[arm], color=COLORS[arm],
                    ms=7, lw=1.4, label=LABELS[arm])
            for p in points:
                if not p['valid']:
                    ax.annotate(f'{engine}: invalid', (p['pair'], .03),
                                xycoords=('data', 'axes fraction'), rotation=90,
                                fontsize=9, color='#B03D48', ha='center')
        ax.axhline(1, color='#596169', lw=1)
        ax.set_title(title, fontsize=11)
        ax.set_xticks([1, 2, 3]); ax.set_xlim(.7, 3.3)
        ax.set_xlabel('Fresh pair'); ax.set_ylabel('Speedup (×)')
        ax.grid(axis='y', color='#E5E8EA'); ax.set_axisbelow(True)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(.06, .855),
               ncol=2, frameon=False, fontsize=10)
    gm_text = '; '.join(f"{engine} throughput GM: " +
        (f"{a['paired_throughput_geomean']:.3f}×" if a['paired_throughput_geomean'] is not None
         else f"unavailable ({a['valid_pairs']}/3 valid pairs)") for engine, a in data['aggregates'].items())
    fig.text(.065, .07, gm_text + '\n'
        'Each ratio uses the corresponding two fresh process medians of three formal requests. All warm/formal raw rows are saved.\n'
        'Lines only connect pair observations. No pooling of request samples, confidence intervals, or global fastest claim.\n'
        'External visible outputs must match each framework’s prior raw-ID qualification; native is shown separately in the first figure.',
        fontsize=10.5, color='#50575E', linespacing=1.6)
    for extension in ('png', 'svg'):
        fig.savefig(output / f'e020-paired-ratios.{extension}', dpi=180,
                    metadata={'Date': None} if extension == 'svg' else None)
    plt.close(fig)
    return matplotlib.__version__


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=DEFAULT_RESULTS)
    parser.add_argument('--output', type=Path, required=True, help='New output directory; never overwrite an old chart.')
    parser.add_argument('--snapshot', type=Path, help='Prior output/sources for offline redraw.')
    parser.add_argument('--collect-only', action='store_true', help='Reconcile and save JSON without importing Matplotlib.')
    args = parser.parse_args()
    raw = args.results.resolve().relative_to(ROOT).as_posix()
    args.output.mkdir(parents=True, exist_ok=False)
    if args.snapshot:
        manifest = json.loads((args.snapshot.parent / 'source-manifest.json').read_text())
        for meta in manifest.values():
            assert hashlib.sha256((args.snapshot / meta['snapshot_path']).read_bytes()).hexdigest() == meta['sha256']
    sources = Sources(args.output, args.snapshot)
    data = collect(sources, raw)
    write(args.output / 'chartdata.json', data)
    relative = 'bench/m5_ultra_256gb/progress/external_spec.py'
    target = args.output / 'sources' / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(__file__, target)
    payload = target.read_bytes()
    sources.manifest[relative] = {'snapshot_path': relative,
        'sha256': hashlib.sha256(payload).hexdigest(), 'bytes': len(payload)}
    write(args.output / 'source-manifest.json', sources.manifest)
    version = None if args.collect_only else render(data, args.output)
    write(args.output / 'summary.json', {'status': 'HTTP_METRICS_RECONCILED',
        'matrix_status': data['status'], 'processes': len(data['processes']),
        'valid_processes': sum(p['valid'] for p in data['processes']),
        'formal_requests': data['formal_requests'], 'warm_requests': data['warm_requests'],
        'matplotlib': version, 'visual_inspection': 'Not performed by this script',
        'scope': data['scope'], 'source_audit_scope': data['source_audit_scope']})
    print(json.dumps({'output': str(args.output), 'matrix_status': data['status'],
                      'valid_processes': sum(p['valid'] for p in data['processes'])}))


if __name__ == '__main__':
    main()
