"""v13 retains the predeclared v12 design; explicit --run after corrected audit PASS.

Reuse v11's two scatter panels, C1 insets and receipt/status tables. Every marker
is one actual process. Historical native stays gray/hollow and separate from
paired OFF. No fitted frontier, pooled percentiles, CI or current-link update.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
COLORS = {'native-historical': '#7b8490', 'native-off': '#1267b1', 'native-on': '#15806e',
          'omlx': '#a13d73', 'lithos': '#ba6e20'}
MARKERS = {'native-historical': 'o', 'native-off': 'o', 'native-on': 'D', 'omlx': 's', 'lithos': '^'}
LABELS = {'native-historical': 'Native historical', 'native-off': 'Native OFF',
          'native-on': 'Native ON', 'omlx': 'oMLX', 'lithos': 'Lithos'}


def need(ok, message):
    if not ok: raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load():
    receipt = json.loads((HERE/'preparation.json').read_bytes())
    need(receipt['schema'] == 'D512-progress-v13-preparation-v1' and receipt['status'] == 'PREPARED'
         and receipt['prepare_sha256'] == sha(HERE/'prepare.py')
         and receipt['renderer_sha256'] == sha(Path(__file__))
         and receipt['chartdata_sha256'] == sha(HERE/'chartdata.json')
         and receipt['manifest_sha256'] == sha(HERE/'sources/manifest.json'), 'sealed current sources/data')
    spec = importlib.util.spec_from_file_location('_v13_closed_receipt_builder', HERE/'prepare.py')
    builder = importlib.util.module_from_spec(spec); spec.loader.exec_module(builder)
    manifest = builder.parse((HERE/'sources/manifest.json').read_bytes()); data = {}
    need(set(manifest) == set(builder.PATHS), 'exact closed source set')
    for name, row in manifest.items():
        path = HERE/'sources'/name; raw = path.read_bytes()
        need(hashlib.sha256(raw).hexdigest() == row['sha256'] and len(raw) == row['bytes'], 'unchanged copied receipt: ' + name)
        if name.endswith('.json'): data[name] = builder.parse(raw)
    chart = builder.build(data, manifest)
    builder.same(chart, builder.parse((HERE/'chartdata.json').read_bytes()), 'display data must rederive from pinned receipts')
    return chart


def draw(chart):
    targets = [HERE/('D512-throughput-latency.'+suffix) for suffix in ('png', 'svg')]
    need(not any(p.exists() for p in targets) and not (HERE/'verification.json').exists(), 'new render only; retain failed attempts')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
        'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'none',
        'svg.hashsalt': 'D512-v13', 'figure.facecolor': 'white', 'savefig.facecolor': 'white'})
    points = chart['points']; display = chart['display']; summaries = chart['process_metric_medians']
    fig, axes = plt.subplots(1, 2, figsize=(17, 17), sharey=True)
    fig.subplots_adjust(left=.07, right=.97, bottom=.585, top=.865, wspace=.19)
    fig.text(.07, .971, 'Qwen3.8-27B · throughput vs full HTTP latency', size=23, weight='bold', color='#172536')
    fig.text(.07, .947, 'M5 Ultra 256 GiB · NVFP4 / fixed D prompts · normal EOS, cap512 · formal observers OFF', size=12, color='#475569')
    fig.text(.07, .925, '26 actual process observations · historical native n=1/C; OFF, ON, oMLX and Lithos n=3/configuration/C', size=11, color='#475569')
    handles = [Line2D([], [], marker=MARKERS[k], color='none', markersize=8,
        markerfacecolor='none' if k == 'native-historical' else COLORS[k],
        markeredgecolor=COLORS[k], label=LABELS[k]) for k in COLORS]
    fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(.065, .911), frameon=False, ncol=5)

    def scatter(ax, selection, x, size):
        for row in selection:
            p = points[row['point_index']]; k = row['configuration']
            ax.scatter(p[x], p['aggregate_output_tok_s'], s=size, marker=MARKERS[k],
                facecolors='none' if k == 'native-historical' else COLORS[k],
                edgecolors=COLORS[k] if k == 'native-historical' else 'white',
                linewidths=1.3 if k == 'native-historical' else .6, alpha=.9, zorder=4)

    for ax, x, label in zip(axes, ('e2e_p50_s', 'e2e_p95_s'), ('p50', 'p95')):
        ax.grid(color='#e1e7ef', linewidth=.8, zorder=0)
        ax.set_xlim(0, max(p[x] for p in points)*1.12)
        ax.set_ylim(0, max(p['aggregate_output_tok_s'] for p in points)*1.18)
        ax.set_xlabel('POST-to-DONE '+label+' latency (s)  ← lower', labelpad=12)
        ax.xaxis.set_major_locator(MaxNLocator(6)); ax.yaxis.set_major_locator(MaxNLocator(6))
        ax.text(.02, .96, 'Better toward upper left', transform=ax.transAxes, va='top', color='#64748b')
        scatter(ax, display, x, 73)
        small = [r for r in display if points[r['point_index']]['concurrency'] == 1]
        inset = ax.inset_axes([.53, .47, .43, .40]); inset.set_facecolor('#f6f8fb')
        xx = [points[r['point_index']][x] for r in small]
        yy = [points[r['point_index']]['aggregate_output_tok_s'] for r in small]
        dx, dy = max(max(xx)-min(xx), .1), max(max(yy)-min(yy), 1.)
        inset.set_xlim(min(xx)-.2*dx, max(xx)+.2*dx); inset.set_ylim(min(yy)-.2*dy, max(yy)+.3*dy)
        inset.grid(color='#dfe5ec', linewidth=.6); inset.tick_params(labelsize=8)
        inset.set_title('C1 zoom · all13 observations', fontsize=10, loc='left')
        inset.xaxis.set_major_locator(MaxNLocator(4)); inset.yaxis.set_major_locator(MaxNLocator(4))
        scatter(inset, small, x, 43)
    axes[0].set_ylabel('Aggregate committed output (tokens/s)  ↑ higher', labelpad=10)

    def table_at(rows, box, widths, color_column=False, font=9.5):
        ax = fig.add_axes(box); ax.axis('off')
        table = ax.table(cellText=rows, cellLoc='left', colWidths=widths, bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False); table.set_fontsize(font)
        for (r, c), cell in table.get_celld().items():
            cell.set_edgecolor('white'); cell.set_linewidth(2)
            cell.set_facecolor('#e7edf5' if r == 0 else '#f4f6fa' if r % 2 else 'white')
            cell.set_text_props(color='#172536', weight='bold' if r == 0 or c == 0 else 'normal')
            if color_column and r > 0 and c == 0:
                key = next(k for k, value in LABELS.items() if value == rows[r][0])
                cell.set_text_props(color=COLORS[key])

    fig.text(.07, .551, 'Process metrics · median [observed min, max] within each configuration / C', size=13, weight='bold', color='#172536')
    rows = [['Configuration', 'HTTP C', 'n', 'Committed tok/s', 'p50 end-to-end (s)', 'p95 end-to-end (s)']]
    for row in summaries:
        def cell(key):
            m = row['metrics'][key]
            return f"{m['median']:.2f}" if row['fresh_processes'] == 1 else f"{m['median']:.2f} [{m['minimum']:.2f}, {m['maximum']:.2f}]"
        rows.append([row['label'], str(row['concurrency']), str(row['fresh_processes']),
                     cell('aggregate_output_tok_s'), cell('e2e_p50_s'), cell('e2e_p95_s')])
    table_at(rows, [.07, .355, .9, .183], [.19, .07, .04, .23, .235, .235], True)
    fig.text(.07, .331, 'Audited adjacent OFF / ON pairs · ON÷OFF; throughput ↑, latency ↓', size=13, weight='bold', color='#172536')
    pairs = [['Repeat / C', 'Actual process order', 'Throughput ratio', 'p50 ratio', 'p95 ratio']]
    for row in chart['paired_comparisons']['pairs']:
        pairs.append([f"r{row['repeat']} / C{row['concurrency']}", ' → '.join(row['order']).upper(),
            *[f"{row[k]:.4f}×" for k in ('throughput_on_over_off', 'e2e_p50_on_over_off', 'e2e_p95_on_over_off')]])
    table_at(pairs, [.07, .202, .9, .117], [.16, .24, .20, .20, .20])
    fig.text(.07, .179, 'Qualification status · GDN37 remains separate from speed', size=13, weight='bold', color='#172536')
    qrows = [['Candidate', 'Verified status', 'Actual coverage', 'Scope / next gate']]
    for q in chart['qualification_status']:
        qrows.append([q['candidate'], q['status'], q['evidence'], q['next_gate']])
    table_at(qrows, [.07, .100, .9, .067], [.14, .26, .32, .28], font=9)
    notes = (
        '8 warm + 64 formal requests/process. Full POST→DONE includes server queueing; concurrency-slot waiting is excluded.',
        'Gray hollow native history is separate from the paired OFF baseline. Ranges are observed process min–max, not CIs.',
        'Pairing is by adjacent registered processes, not identical realized arrivals. No global-fastest or fitted-frontier claim.',
        'Readiness does not prove formal graph hits. No all-future-arrival, cross-engine full-state/quality or E024+GDN claim.',
        'Lithos C8 uses official FIFO execution. The original oMLX parser FAIL remains in v11 provenance.',
        'E024 tokenizer-path parser corrected offline; original FAIL and withheld v12 chart preserved. No timed rerun.')
    for y, text in zip((.084, .070, .056, .042, .028, .014), notes):
        fig.text(.07, y, text, size=9.5, color='#475569')
    for target in targets:
        fig.savefig(target, dpi=180, metadata={'Date': None} if target.suffix == '.svg' else None)
        if target.suffix == '.svg':
            target.write_text('\n'.join(line.rstrip() for line in target.read_text().splitlines())+'\n')
    plt.close(fig)
    report = dict(status='RENDERED', performance_points=26, historical_points=14, new_points=12,
        historical_metrics_unchanged=True, parser_correction=chart['parser_correction'],
        chartdata_sha256=sha(HERE/'chartdata.json'),
        renderer_sha256=sha(Path(__file__)), figures={p.name: dict(sha256=sha(p), bytes=p.stat().st_size) for p in targets},
        visual_review_completed=False, current_link_updated=False,
        scope='Rendering/data checks only; human visual QA and any current-link update remain separate.')
    with (HERE/'verification.json').open('x') as stream:
        json.dump(report, stream, indent=2); stream.write('\n')
    print(json.dumps(report))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--run', action='store_true')
    if parser.parse_args().run: draw(load())
    else: print('DESCRIPTION_ONLY: requires complete closed v13 preparation; rendering and current-link changes are separate.')
