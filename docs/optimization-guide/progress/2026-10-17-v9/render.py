"""Offline v9 snapshot: unchanged eight v8 performance points; separate qualification panel."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PINS = {
    'v8-chartdata.json': '3939ad59d8b08f725bb7acea66438cc7b0d2be150fd405142febb06401b555b6',
    'v8-render.py': '25bea85c9dd03263394e8262542880bcee1f6c1cd116728e65b5e5817ab72c7b',
    'native-independent-audit.json': 'b669350e95ef07f4da5cac3a5cf6fb6fdaf2aea57e886d3ba60b9afd15e950bd',
    'lithos-independent-audit.json': '562787c59f3a3b96a71b279538f61b38fd042dae9f2bc9c6117e022db7a6fb77',
    'e024-numerical.json': 'd07ad9e71573adf5737718066a089705d53a03592330e97fa508e30764958856',
    'e024-resources.json': 'f9bdfcf5bc64f2a7c8add12b03d9934c34e57b1f7644d02c61ab0192863ebd4b',
    'e024-restoration.json': '1aa9176b3eb043c77902f9fdaacc75fc0d7e66f1b6fd0d08cdd21b6a54652ec8',
    'omlx-qualification.json': '8e9d77123e44d9485474976362dc70dcdbb3a14f6de290951d3ca989c33aa208',
    'omlx-resources.json': 'bc923557b7b59a230d10d5a163f0e5cfaf07c92074c06f04bfc0f7c2e4172531',
    'omlx-restoration.json': '408bf10518e403c508d100abef613dd31b9a1db5bed76ceea34ab36fa20f6c7f',
}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def nondominated(point, points, x):
    return not any(q[x] <= point[x] and q['aggregate_output_tok_s'] >= point['aggregate_output_tok_s']
                   and (q[x] < point[x] or q['aggregate_output_tok_s'] > point['aggregate_output_tok_s'])
                   for q in points)


def load_evidence():
    manifest = json.loads((HERE / 'sources/manifest.json').read_bytes())
    data = {}
    need(set(manifest) == set(PINS), 'exact source inventory')
    for name, sha in PINS.items():
        raw = (HERE / 'sources' / name).read_bytes()
        need(digest(raw) == sha == manifest[name]['sha256'] and len(raw) == manifest[name]['bytes'], name)
        if name.endswith('.json'):
            data[name] = json.loads(raw)
    base = data['v8-chartdata.json']; rows = base['points']
    need(len(rows) == 8, 'unchanged eight performance points')
    for engine, filename in [('mini-vLLM', 'native-independent-audit.json'), ('Lithos', 'lithos-independent-audit.json')]:
        audit = data[filename]
        need(audit['status'] == 'PASS' and audit['qualified'] is True
             and audit['performance_qualified'] is True and audit['restoration']['status'] == 'PASS', filename)
        selected = [r for r in rows if r['engine'] == engine]
        need(len(selected) == len(audit['points']), 'performance point count')
        for row in selected:
            point = next(p for p in audit['points'] if p['concurrency'] == row['concurrency'] and p['repeat'] == row['repeat'])
            need(all(row[k] == v for k, v in point.items()), 'original audit point unchanged')
            for x, flag in [('e2e_p50_s', 'nondominated_p50_among_displayed_observations'),
                            ('e2e_p95_s', 'nondominated_p95_among_displayed_observations')]:
                need(row[flag] == nondominated(row, rows, x), 'unchanged discrete frontier')
    for name in ('e024-numerical.json', 'e024-resources.json', 'e024-restoration.json',
                 'omlx-qualification.json', 'omlx-resources.json', 'omlx-restoration.json'):
        need(data[name]['status'] == 'PASS', name + ' PASS')
    e = data['e024-numerical.json']; o = data['omlx-qualification.json']
    need(e['qualified'] is True and e['http_qualified'] is False and e['performance_qualified'] is False
         and e['offline_tensor_bytes_rechecked'] is False, 'E024 qualification scope')
    requests = sum(len(c['completed']) for c in e['ledger']['cases'])
    need(requests == 37 and e['ledger']['array_counts']['all_arrays'] == 6622533
         and e['route_events']['parameter_direct_calls'] == 42 and e['discarded_ahead_observed'] is False,
         'E024 observed coverage')
    need(o['qualified'] is True and o['route_count_text_qualified'] is True
         and o['performance_qualified'] is False and o['full_state_qualified'] is False
         and o['requests'] == 24, 'oMLX qualification scope')
    phases = {p['concurrency']: p for p in o['phases']}
    need(set(phases) == {1, 8} and phases[1]['requests'] == 8 and phases[8]['requests'] == 16
         and phases[8]['max_http_outstanding'] == 8
         and max(map(int, phases[8]['actual_batch_histogram'])) == 8, 'actual oMLX C8 coverage')
    need(data['omlx-resources.json']['independent_resource_verified'] is True, 'oMLX resource audit')
    for name in ('e024-restoration.json', 'omlx-restoration.json'):
        need(data[name]['qualified'] is True, name + ' independent restoration')
    status = [
        dict(name='E024 route graph', status='PASS', requests=requests, array_comparisons=6622533,
             parameter_direct_calls=42, discarded_ahead_observed=False, http_qualified=False,
             performance_qualified=False, offline_tensor_bytes_rechecked=False,
             source='sources/e024-numerical.json', resource='sources/e024-resources.json',
             restoration='sources/e024-restoration.json'),
        dict(name='oMLX D512', status='PASS', requests=24, C1_requests=8, C8_requests=16,
             max_http_outstanding=8, max_observed_decode_batch=8, full_state_qualified=False,
             performance_qualified=False, source='sources/omlx-qualification.json',
             resource='sources/omlx-resources.json', restoration='sources/omlx-restoration.json'),
    ]
    chart = dict(schema='D512-progress-qualified-status-v9', hardware=base['hardware'], model=base['model'],
                 revision=base['revision'], workload=base['workload'], performance_scope=base['scope'],
                 points=rows, process_metric_medians=base['process_metric_medians'],
                 v8_performance_points_unchanged=True, qualification_status=status,
                 performance_point_count=8, sources=manifest)
    (HERE / 'chartdata.json').write_text(json.dumps(chart, indent=2, allow_nan=False) + '\n')
    return chart


def render(chart):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.spines.top': False,
                         'axes.spines.right': False, 'svg.fonttype': 'none', 'svg.hashsalt': 'D512-v9',
                         'figure.facecolor': 'white', 'savefig.facecolor': 'white'})
    rows = chart['points']; summaries = chart['process_metric_medians']
    colors = {'mini-vLLM': '#1267b1', 'Lithos': '#c2671b'}
    markers = {'mini-vLLM': 'o', 'Lithos': '^'}
    fig, axes = plt.subplots(1, 2, figsize=(14, 11), sharey=True)
    fig.subplots_adjust(left=.085, right=.97, bottom=.43, top=.80, wspace=.19)
    fig.text(.085, .955, 'Qwen3.8-27B · D512 throughput–latency', size=22, weight='bold', color='#142238')
    fig.text(.085, .917, 'M5 Ultra 256 GiB · same NVFP4 weights and prompts · normal EOS · formal observer OFF', size=12, color='#475569')
    fig.text(.085, .883, 'v9: eight measured performance points unchanged from v8; new qualifications shown separately below.', size=11, color='#475569')
    ymax = max(q['aggregate_output_tok_s'] for q in rows) * 1.23
    for ax, x, label in zip(axes, ('e2e_p50_s', 'e2e_p95_s'), ('p50', 'p95')):
        ax.grid(color='#e3e8ef', linewidth=.8, zorder=0)
        ax.set_xlim(0, max(q[x] for q in rows) * 1.17); ax.set_ylim(0, ymax)
        for q in rows:
            ax.scatter(q[x], q['aggregate_output_tok_s'], s=83, color=colors[q['engine']], marker=markers[q['engine']],
                       edgecolor='#142238' if nondominated(q, rows, x) else 'white', linewidth=1.2, zorder=4, alpha=.9)
        for q in summaries:
            selected = [r for r in rows if r['engine'] == q['engine'] and r['concurrency'] == q['concurrency']]
            anchor = min(selected, key=lambda r: abs(r[x] - q[x]))
            offset = (8, -23 if q['engine'] == 'mini-vLLM' and q['concurrency'] == 1 else 13)
            ax.annotate(f"{q['engine']} C{q['concurrency']}", (anchor[x], anchor['aggregate_output_tok_s']),
                        xytext=offset, textcoords='offset points', size=10, color=colors[q['engine']])
        ax.set_xlabel('Full POST-to-DONE ' + label + ' latency (s)  ← lower', labelpad=12)
        ax.xaxis.set_major_locator(MaxNLocator(6)); ax.yaxis.set_major_locator(MaxNLocator(6))
        ax.text(.025, .96, 'Better toward upper left', transform=ax.transAxes, va='top', size=10, color='#64748b')
    axes[0].set_ylabel('Aggregate committed output (tokens/s)  ↑ higher', labelpad=10)
    handles = [Line2D([], [], marker=markers[e], color='none', markerfacecolor=colors[e], markeredgecolor='white',
                       markersize=9, label=e + (' · 1 fresh process / C' if e == 'mini-vLLM' else ' · 3 fresh processes / C')) for e in colors]
    fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(.08, .851), frameon=False, ncol=2)
    for i, q in enumerate(summaries):
        x = .085 + i * .225
        fig.text(x, .35, f"{q['engine']} C{q['concurrency']} · n={q['fresh_processes']}", weight='bold', color=colors[q['engine']], size=11)
        fig.text(x, .32, f"{q['aggregate_output_tok_s']:.2f} tok/s", size=14, color='#142238')
        fig.text(x, .294, f"p50 {q['e2e_p50_s']:.2f}s · p95 {q['e2e_p95_s']:.2f}s", size=10, color='#475569')
    fig.text(.085, .253, 'Individual-process dots; values above are process-metric medians. Black edges: nondominated observed dots only.', size=10, color='#475569')
    fig.text(.085, .229, '8 warm + 64 formal requests / process. Queueing included. Historical native vs newer Lithos; no paired CI or interpolation.', size=10, color='#475569')
    fig.add_artist(Line2D([.085, .97], [.205, .205], transform=fig.transFigure, color='#ccd4df', linewidth=1))
    fig.text(.085, .177, 'QUALIFICATION STATUS — separate evidence, no new performance points', weight='bold', size=12, color='#142238')
    fig.text(.085, .143, 'E024 route graph · PASS', weight='bold', size=12, color='#1267b1')
    fig.text(.085, .115, '37 requests · 6,622,533 device array comparisons · 42 direct calls', size=10, color='#334155')
    fig.text(.085, .09, 'Independent receipt/schedule audit; raw tensor bytes not retained.', size=10, color='#475569')
    fig.text(.085, .065, 'Discarded ahead: 0 observed. HTTP / formal performance not measured.', size=10, color='#475569')
    fig.text(.56, .143, 'oMLX D512 route / count / text · PASS', weight='bold', size=12, color='#1267b1')
    fig.text(.56, .115, '24 requests (C1: 8, C8: 16) · C8 HTTP / decode peaks: 8 / 8', size=10, color='#334155')
    fig.text(.56, .09, 'Resource + restoration audits PASS; no cross-engine full-state claim.', size=10, color='#475569')
    fig.text(.56, .065, 'Formal performance not measured; no oMLX dot on these axes.', size=10, color='#475569')
    for suffix in ('png', 'svg'):
        target = HERE / ('D512-performance-and-qualification.' + suffix)
        fig.savefig(target, dpi=180, metadata={'Date': None} if suffix == 'svg' else None)
        if suffix == 'svg':
            target.write_text('\n'.join(line.rstrip() for line in target.read_text().splitlines()) + '\n')
    plt.close(fig)
    receipt = dict(status='PASS', performance_points=8, same_v8_points=True,
                   performance_points_sha256=digest(canonical(rows)), source_sha256=PINS,
                   renderer_sha256=digest(Path(__file__).read_bytes()),
                   chartdata_sha256=digest((HERE / 'chartdata.json').read_bytes()),
                   scope='Pinned audited summaries and unchanged v8 point values; no rerun of original measurement/audit.')
    (HERE / 'verification.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    render(load_evidence())
