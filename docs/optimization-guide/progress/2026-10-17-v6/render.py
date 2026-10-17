"""Render completed, independently audited D512 native operating points."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    raw = args.audit.read_bytes()
    audit = json.loads(raw)
    assert audit['schema'] == 'E023-native-formal-independent-audit-v1'
    assert audit['status'] == 'PASS' and audit['qualified'] is True
    assert audit['performance_qualified'] is True
    assert audit['restoration']['status'] == 'PASS'
    points = audit['points']
    assert len(points) == 2 and {p['concurrency'] for p in points} == {1, 8}
    assert all(p['requests'] == 64 and p['warm_requests_checked'] == 8
               and p['n_fresh_processes'] == p['repeat'] == 1 for p in points)
    for p in points:
        assert p['aggregate_output_tok_s'] == p['committed_output_tokens'] / p['common_window_s']
        assert 0 < p['e2e_p50_s'] <= p['e2e_p95_s']
    args.out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(args.audit, args.out / 'independent-audit.json')
    shutil.copy2(__file__, args.out / 'render.py')
    chart = {
        'schema': 'native-D512-throughput-latency-chart-v1',
        'source_audit_sha256': hashlib.sha256(raw).hexdigest(),
        'source_audit_path': str(args.audit.resolve()),
        'owner_plan_sha256': audit['owner_plan_sha256'],
        'model': 'mlx-community/Qwen3.8-27B-nvfp4',
        'revision': '5ff8ef173ad0d7c3aae92f0be43031a6ab8067c6',
        'hardware': 'M5 Ultra 256 GiB',
        'workload': 'Eight D coding fixtures, 886-978 prompt tokens, greedy normal EOS, output cap 512',
        'x': 'Per-request full POST-to-DONE p50 latency (seconds); smaller is better',
        'y': 'Actual committed output tokens / (last DONE - first POST); larger is better',
        'fresh_processes_per_concurrency': 1,
        'warm_requests_per_process': 8,
        'measured_requests_per_process': 64,
        'comparison_status': {'native': 'qualified', 'omlx_D512': 'pending', 'lithos_D512': 'pending'},
        'scope': 'One fresh process per point, no confidence interval, no competitor dominance claim. Numerical qualification covers recorded representative histories, not every new formal schedule.',
        'points': points,
    }
    for p in chart['points']:
        p['nondominated_among_these_native_points'] = not any(
            q['e2e_p50_s'] <= p['e2e_p50_s'] and q['aggregate_output_tok_s'] >= p['aggregate_output_tok_s']
            and (q['e2e_p50_s'] < p['e2e_p50_s'] or q['aggregate_output_tok_s'] > p['aggregate_output_tok_s'])
            for q in points)
    (args.out / 'chartdata.json').write_text(json.dumps(chart, indent=2, allow_nan=False) + '\n')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.fonttype': 'none', 'figure.facecolor': '#f7f9fc',
                         'savefig.facecolor': '#f7f9fc'})
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.subplots_adjust(left=.10, right=.72, bottom=.21, top=.78)
    fig.text(.10, .93, 'Qwen3.8-27B · D512 throughput–latency', size=21, weight='bold', color='#142238')
    fig.text(.10, .875, 'Native baseline on M5 Ultra 256 GiB · normal EOS · observer OFF', size=12, color='#475569')
    color = '#1267b1'
    ax.set_facecolor('white')
    ax.grid(color='#e3e8ef', linewidth=.8, zorder=0)
    ordered = sorted(points, key=lambda p: p['e2e_p50_s'])
    ax.plot([p['e2e_p50_s'] for p in ordered], [p['aggregate_output_tok_s'] for p in ordered],
            color=color, linestyle=(0, (3, 4)), alpha=.4, zorder=2)
    for p in ordered:
        ax.scatter(p['e2e_p50_s'], p['aggregate_output_tok_s'], s=125,
                   color=color, edgecolor='white', linewidth=1.5, zorder=4)
        ax.annotate(f"C{p['concurrency']}", (p['e2e_p50_s'], p['aggregate_output_tok_s']),
                    xytext=(10, 10), textcoords='offset points', weight='bold', color=color)
    ax.set_xlim(0, max(p['e2e_p50_s'] for p in points) * 1.22)
    ax.set_ylim(0, max(p['aggregate_output_tok_s'] for p in points) * 1.23)
    ax.set_xlabel('Full request p50 latency (s)  ← lower', labelpad=13)
    ax.set_ylabel('Aggregate committed output (tokens/s)  ↑ higher', labelpad=12)
    ax.xaxis.set_major_locator(MaxNLocator(6))
    ax.yaxis.set_major_locator(MaxNLocator(6))
    ax.text(.03, .96, 'Better toward upper left', transform=ax.transAxes,
            va='top', color='#64748b', size=10)
    y = .75
    for p in sorted(points, key=lambda p: p['concurrency']):
        fig.text(.77, y, f"C{p['concurrency']} · one fresh process", weight='bold', color=color, size=12)
        fig.text(.77, y-.050, f"{p['aggregate_output_tok_s']:.2f} tokens/s", size=15, color='#142238')
        fig.text(.77, y-.095, f"p50 {p['e2e_p50_s']:.2f} s · p95 {p['e2e_p95_s']:.2f} s", color='#475569', size=10)
        y -= .20
    fig.text(.77, .31, 'D512 comparison pending', weight='bold', size=10, color='#475569')
    fig.text(.77, .265, 'oMLX · Lithos', size=11, color='#64748b')
    fig.text(.10, .105, '8 warm + 64 measured requests per point · 886–978 prompt tokens · output cap 512', size=10, color='#475569')
    fig.text(.10, .067, 'Single-run observations; no confidence intervals. Dashed line is a visual guide, not interpolation.', size=10, color='#475569')
    for suffix in ('png', 'svg'):
        fig.savefig(args.out / ('native-D512-throughput-latency.' + suffix), dpi=180)
        if suffix == 'svg':
            svg = args.out / 'native-D512-throughput-latency.svg'
            svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines()) + '\n')
    plt.close(fig)
    lines = ['# Qwen3.8-27B D512: native measured operating points', '',
             'M5 Ultra 256 GiB; identical fixed NVFP4 weights and eight D coding fixtures. '
             'Greedy normal EOS, output cap 512; all qualification observers OFF during timing.', '',
             '| Concurrency | Fresh processes | Measured requests | Committed tokens | Window (s) | Output tok/s | p50 (s) | p95 (s) |',
             '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for p in sorted(points, key=lambda p: p['concurrency']):
        lines.append(f"| C{p['concurrency']} | 1 | 64 | {p['committed_output_tokens']} | {p['common_window_s']:.6f} | {p['aggregate_output_tok_s']:.6f} | {p['e2e_p50_s']:.6f} | {p['e2e_p95_s']:.6f} |")
    lines += ['', 'X is complete per-request POST-to-DONE p50 latency. Y is the sum of actual committed output tokens '
              'divided by the common first-POST to last-DONE window. The client uses a closed loop at the stated concurrency; '
              'latency includes server queueing, not waiting before a client slot becomes available.', '',
              'Each point has eight warm requests and 64 measured requests in one fresh server process. '
              'These are single-process observations, not confidence intervals. The dashed connector is a visual guide. '
              'The older output-cap-32 experiments belong to another workload and are not plotted here. '
              'oMLX and Lithos D512 results are pending; this chart establishes no competitor dominance.', '',
              'Correctness scope: independent full-logit/state replay of representative C1/C8 recorded HTTP histories, '
              'plus raw fixture/UUID/text/EOS/count checks on every new measured request. It is not a replay of every formal schedule. '
              'The independent cycle audit also checks source/runtime/model receipts, observed resources, child reaping and service restoration.', '',
              f"Source audit SHA256: `{chart['source_audit_sha256']}`. See [audit](independent-audit.json), "
              '[data](chartdata.json), [render source](render.py), [PNG](native-D512-throughput-latency.png), '
              '[SVG](native-D512-throughput-latency.svg).', '']
    (args.out / 'README.md').write_text('\n'.join(lines))
    print(json.dumps({'status': 'rendered', 'out': str(args.out.resolve()), 'points': len(points)}))


if __name__ == '__main__':
    main()
