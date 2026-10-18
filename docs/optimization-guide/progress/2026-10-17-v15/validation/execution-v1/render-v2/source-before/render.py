"""Explicit v15 frontiers; adapted from v14 without its fixed winner/axis assumptions.

Default inert. Only --run imports matplotlib or writes NEW figure output. This is
a visualization of closed evidence, not another experiment or qualification.
"""
import argparse
import importlib.util
import json
from pathlib import Path


def helpers():
    path = Path(__file__).with_name('prepare.py')
    spec = importlib.util.spec_from_file_location('_v15_preparation', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true'); parser.add_argument('--repo', type=Path)
    parser.add_argument('--data', type=Path); parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    if not args.run:
        print('DESCRIPTION_ONLY: no results read, plotting package imported, or figure written.'); return
    h = helpers(); h.need(all((args.repo, args.data, args.out)), '--run needs --repo --data --out')
    h.need(not args.out.exists(), 'exclusive render output')
    data, blob = h.read(args.data / 'chartdata.json'); receipt, _ = h.read(args.data / 'preparation.json')
    h.need(receipt['status'] == 'PREPARED' and receipt['chartdata_sha256'] == h.sha(blob)
           and receipt['source_sha256'] == h.sha(Path(__file__).with_name('prepare.py').read_bytes()), 'frozen prepared data')
    audit, originals = h.closed_inputs(args.repo)
    old, old_raw = h.read(args.repo / h.OLD); h.need(h.sha(old_raw) == h.OLD_SHA, 'old 26 unchanged')
    originals['source-v14-data.json'] = old_raw
    h.need(set(receipt['input_pins']) == {'inputs/' + n for n in originals}, 'complete small input copies')
    for name, raw in originals.items():
        h.need((args.data / 'inputs' / name).read_bytes() == raw, 'closed input changed: ' + name)
        h.need(receipt['input_pins']['inputs/' + name] == dict(sha256=h.sha(raw), bytes=len(raw)), 'copy hash')
    expected = h.build(old, audit); expected['input_pins'] = receipt['input_pins']
    h.need(data == expected, 'recompute all 38 observations,14 summaries and frontiers')
    rows = data['process_metric_medians']; fronts = data['per_engine_frontiers']; joint = data['joint_frontiers']
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    colors = {'native': '#15806e', 'omlx': '#a13d73', 'lithos': '#ba6e20'}
    markers = {'native-historical': 'x', 'native-off': 'o', 'native-on': 'D',
               'native-gdn-off': 'v', 'native-gdn-on': 'P', 'omlx': 's', 'lithos': '^'}
    short = {'native-historical': 'Native historical N1', 'native-off': 'E024 OFF', 'native-on': 'E024 ON',
             'native-gdn-off': 'GDN OFF', 'native-gdn-on': 'GDN ON', 'omlx': 'oMLX', 'lithos': 'Lithos'}
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
        'axes.spines.right': False, 'figure.facecolor': 'white', 'savefig.facecolor': 'white',
        'svg.fonttype': 'none', 'svg.hashsalt': 'D512-frontier-v15'})
    handles = [Line2D([], [], color=colors[h.ENGINE[c]], marker=markers[c], ls='none',
                     label=short[c], markerfacecolor='none' if c.endswith('-off') else colors[h.ENGINE[c]]) for c in h.LABELS]
    handles += [Line2D([], [], color='#475569', lw=2, label='Within-engine measured frontier'),
                Line2D([], [], color='#172536', ls=(0, (5, 4)), lw=2, label='Joint measured frontier')]

    def draw(ax, x, normalized=False):
        fx = lambda value: 512 / value if normalized else value
        ax.set_axisbelow(True); ax.grid(color='#e2e8f0', lw=.75)
        for d in data['display']:
            p = data['points'][d['point_index']]; c = d['configuration']
            ax.scatter(fx(p[x]), p[h.Y], s=18, marker=markers[c],
                       color=colors[h.ENGINE[c]], alpha=.25, zorder=2)
        for engine, selected in fronts[x].items():
            ax.plot([fx(r['metrics'][x]['median']) for r in selected],
                    [r['metrics'][h.Y]['median'] for r in selected], color=colors[engine], lw=2.2, alpha=.8, zorder=3)
        for r in rows:
            c = r['configuration']; m = r['metrics']; color = colors[r['engine']]
            xx, yy = fx(m[x]['median']), m[h.Y]['median']
            lo, hi = sorted((fx(m[x]['minimum']), fx(m[x]['maximum'])))
            ax.errorbar(xx, yy, xerr=[[xx - lo], [hi - xx]],
                yerr=[[yy - m[h.Y]['minimum']], [m[h.Y]['maximum'] - yy]],
                fmt=markers[c], ms=7, color=color, mfc='white' if c.endswith('-off') else color,
                elinewidth=.8, capsize=2, zorder=5)
        f = joint[x]
        ax.plot([fx(r['metrics'][x]['median']) for r in f], [r['metrics'][h.Y]['median'] for r in f],
                color='#172536', ls=(0, (5, 4)), lw=2.2, zorder=4)
        for i, r in enumerate(f):
            xx, yy = fx(r['metrics'][x]['median']), r['metrics'][h.Y]['median']
            ax.scatter(xx, yy, s=170, facecolors='none', edgecolors='#172536', lw=1.3, zorder=6)
            ax.annotate(f"{short[r['configuration']]} C{r['concurrency']}", (xx, yy), xytext=(6, 13 + 12 * (i % 2)),
                        textcoords='offset points', fontsize=9, color='#172536')
        # Dynamic limits include every observed process/range; no winner or speed forecast.
        ax.set_xlim(0, max(fx(p[x]) for p in data['points']) * 1.15)
        ax.set_ylim(0, max(p[h.Y] for p in data['points']) * 1.2)
        ax.set_ylabel('Aggregate committed output throughput (tokens/s)')
        ax.set_xlabel('512 / median_process(p50 POST-to-DONE) (tokens/s)  → higher' if normalized
                      else f"Full HTTP {('p50' if x == h.XS[0] else 'p95')} POST-to-DONE latency (s)  ← lower")
        ax.text(.03, .95, 'BETTER ↗' if normalized else 'BETTER ↖', transform=ax.transAxes,
                va='top', color='#475569', weight='bold')
        # Preserve the full C8/Lithos range above while making C1 differences
        # readable. The inset contains the same observations, with no jitter.
        inset = ax.inset_axes([.06, .28, .36, .36] if normalized else [.48, .43, .48, .46])
        inset.set_facecolor('white'); inset.grid(color='#e2e8f0', lw=.6)
        c1 = [r for r in rows if r['concurrency'] == 1]
        raw_c1 = [p for p in data['points'] if p['concurrency'] == 1]
        for d in data['display']:
            p = data['points'][d['point_index']]; c = d['configuration']
            if p['concurrency'] == 1:
                inset.scatter(fx(p[x]), p[h.Y], s=12, marker=markers[c], color=colors[h.ENGINE[c]], alpha=.25)
        for r in c1:
            c = r['configuration']; m = r['metrics']; color = colors[r['engine']]
            xx, yy = fx(m[x]['median']), m[h.Y]['median']
            lo, hi = sorted((fx(m[x]['minimum']), fx(m[x]['maximum'])))
            inset.errorbar(xx, yy, xerr=[[xx-lo], [hi-xx]],
                yerr=[[yy-m[h.Y]['minimum']], [m[h.Y]['maximum']-yy]], fmt=markers[c], ms=5.5,
                color=color, mfc='white' if c.endswith('-off') else color, elinewidth=.6, capsize=2)
            offsets = ({'native-on': (8, -7), 'native-gdn-on': (7, 4), 'omlx': (-46, 7), 'lithos': (5, -13)}
                       if normalized else {'native-on': (-58, 8), 'native-gdn-on': (6, 6), 'omlx': (6, -1), 'lithos': (-35, -12)})
            if c in offsets:
                inset.annotate(short[c], (xx, yy), xytext=offsets[c], textcoords='offset points',
                               fontsize=8, color=color)
        xx = [fx(p[x]) for p in raw_c1]; yy = [p[h.Y] for p in raw_c1]
        xm = max(.1, (max(xx)-min(xx))*.14); ym = max(.2, (max(yy)-min(yy))*.22)
        inset.set_xlim(min(xx)-xm, max(xx)+xm); inset.set_ylim(min(yy)-ym, max(yy)+ym)
        inset.tick_params(labelsize=8); inset.set_title('C1 zoom · same process medians/ranges', fontsize=9, pad=6)
        if normalized:
            r = next(r for r in rows if r['configuration'] == 'native-gdn-on' and r['concurrency'] == 1)
            inset_point = (fx(r['metrics'][x]['median']), r['metrics'][h.Y]['median'])
            ax.annotate('Native GDN ON C1', inset_point, xytext=(-116, -31), textcoords='offset points',
                        fontsize=10, color=colors['native'], arrowprops=dict(arrowstyle='-', color=colors['native'], lw=.8))

    def furniture(fig, title):
        fig.suptitle(title, x=.065, ha='left', y=.975, fontsize=20, weight='bold', color='#172536')
        fig.text(.065, .925, 'M5 Ultra · Qwen3.8-27B NVFP4 · D512 · 38 actual process observations; 14 configuration/C groups', color='#475569')
        fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(.06, .90), ncol=5, frameon=False, fontsize=9)
        lines = [
            'Markers = process-metric medians; whiskers = observed process min–max, not confidence intervals. Historical Native N=1; other groups N=3.',
            'Native E024 ON and GDN ON are separate configurations. New GDN OFF and older E024 OFF remain separate cycles; no combined-switch result.',
            'Lines only join measured endpoints. C2–7 are UNKNOWN; neither a connecting segment nor an engine switch is an executable/interpolated setting.',
            'Full POST→DONE includes first-output latency and delivery. 512/p50 is not ITL/TPOT. Formal observers OFF; no formal GDN dispatch-count proof.'
        ]
        for y, line in zip((.143, .112, .081, .050), lines):
            fig.text(.065, y, line, fontsize=8.5, color='#475569')

    args.out.mkdir(parents=True); files = []
    for name, normalized in (('interactivity', True), ('latency', False)):
        if normalized:
            fig, ax = plt.subplots(figsize=(15, 8.7)); axes = [ax]; metrics = [h.XS[0]]
        else:
            fig, axes = plt.subplots(1, 2, figsize=(17, 8.7)); metrics = h.XS
        fig.subplots_adjust(left=.065, right=.965, bottom=.25, top=.77, wspace=.23)
        furniture(fig, 'Measured throughput–interactivity frontiers' if normalized else 'Measured throughput–latency frontiers')
        for ax, x in zip(axes, metrics):
            draw(ax, x, normalized)
        for ext in ('png', 'svg'):
            path = args.out / f'D512-frontier-{name}.{ext}'
            fig.savefig(path, dpi=170, metadata={'Date': None} if ext == 'svg' else None); files.append(path)
        plt.close(fig)
    report = dict(schema='D512-v15-render-verification-v1', status='RENDERED', visual_review='PENDING',
        renderer_sha256=h.sha(Path(__file__).read_bytes()), preparation_source_sha256=receipt['source_sha256'],
        chartdata_sha256=h.sha(blob), original_26_points_unchanged=True, all_38_medians_and_frontiers_recomputed=True,
        artifacts={p.name: dict(sha256=h.sha(p.read_bytes()), bytes=p.stat().st_size) for p in files})
    (args.out / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
