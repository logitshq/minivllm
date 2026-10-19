"""Explicit v16 frontiers; narrow extension of the reviewed v15 renderer.

Default inert. Only --run imports matplotlib or writes NEW figure output. This is
a visualization of closed evidence, not another experiment or qualification.
"""
import argparse
import importlib.util
import json
from pathlib import Path


def helpers():
    path = Path(__file__).with_name('prepare.py')
    spec = importlib.util.spec_from_file_location('_v16_preparation', path)
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
    h.need(receipt['schema'] == 'D512-v16-preparation-v1' and receipt['status'] == 'PREPARED'
           and receipt['chartdata_sha256'] == h.sha(blob)
           and receipt['source_sha256'] == h.sha(Path(__file__).with_name('prepare.py').read_bytes()), 'frozen prepared data')
    old, audit, originals = h.inputs(args.repo)
    h.need(set(receipt['input_pins']) == {'inputs/' + n for n in originals}, 'complete small input copies')
    for name, raw in originals.items():
        h.need((args.data / 'inputs' / name).read_bytes() == raw, 'closed input changed: ' + name)
        h.need(receipt['input_pins']['inputs/' + name] == dict(sha256=h.sha(raw), bytes=len(raw)), 'copy hash')
    expected = h.build(old, audit); expected['input_pins'] = receipt['input_pins']
    h.need(data == expected, 'recompute all 50 observations,18 summaries and frontiers')
    rows = data['process_metric_medians']; fronts = data['per_engine_frontiers']; joint = data['joint_frontiers']
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    colors = {'native': '#15806e', 'omlx': '#a13d73', 'lithos': '#ba6e20'}
    markers = {'native-historical': 'x', 'native-off': 'o', 'native-on': 'D',
               'native-gdn-off': 'v', 'native-gdn-on': 'P', 'omlx': 's', 'lithos': '^',
               'native-combined-off': '<', 'native-combined-on': '*'}
    short = {'native-historical': 'minivllm historical N1', 'native-off': 'E024 OFF', 'native-on': 'E024 ON',
             'native-gdn-off': 'GDN-only cycle OFF', 'native-gdn-on': 'GDN only', 'omlx': 'oMLX', 'lithos': 'Lithos',
             'native-combined-off': 'Combined cycle: graph only', 'native-combined-on': 'Combined: graph + GDN'}
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
        'axes.spines.right': False, 'figure.facecolor': 'white', 'savefig.facecolor': 'white',
        'svg.fonttype': 'none', 'svg.hashsalt': 'D512-frontier-v16'})
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
                fmt=markers[c], ms=10 if c == 'native-combined-on' else 7,
                color=color, mfc='white' if c.endswith('-off') else color,
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
        ax.set_xlabel('512 / median_process(p50 POST-to-DONE) (tokens/s; not ITL)  → higher' if normalized
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
            offsets = ({'native-on': (8, -9), 'native-gdn-on': (-16, -17), 'omlx': (-46, 7), 'lithos': (5, -13),
                        'native-combined-on': (8, 18)} if normalized else
                       {'native-on': (-58, 8), 'native-gdn-on': (6, 6), 'omlx': (6, -1), 'lithos': (-35, -12),
                        'native-combined-on': (8, 20)})
            if c in offsets:
                inset.annotate(short[c], (xx, yy), xytext=offsets[c], textcoords='offset points',
                               fontsize=8, color=color)
        xx = [fx(p[x]) for p in raw_c1]; yy = [p[h.Y] for p in raw_c1]
        xm = max(.1, (max(xx)-min(xx))*.14); ym = max(.2, (max(yy)-min(yy))*.22)
        inset.set_xlim(min(xx)-xm, max(xx)+xm); inset.set_ylim(min(yy)-ym, max(yy)+ym)
        inset.tick_params(labelsize=8); inset.set_title('C1 zoom · same process medians/ranges', fontsize=9, pad=6)
        # Do not preselect a winning configuration or attach a claimed gain.
        # Actual image review must resolve overlapping labels after evidence exists.

    def furniture(fig, title):
        fig.suptitle(title, x=.065, ha='left', y=.975, fontsize=20, weight='bold', color='#172536')
        fig.text(.065, .925, 'M5 Ultra · Qwen3.8-27B NVFP4 · D512 · 50 process observations; 18 configuration/C groups · Green = minivllm', color='#475569')
        fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(.06, .90), ncol=4, frameon=False, fontsize=9)
        lines = [
            'Markers = process-metric medians; whiskers = observed process min–max, not confidence intervals. minivllm historical N=1; other groups N=3.',
            'New paired cycle keeps graph ON, compares GDN OFF/ON. Older cycles stay separate; combined ON is measured, never summed percentage gains.',
            'Lines join measured endpoints only. HTTP concurrency C is not GPU batch size. C2–7 are UNKNOWN; segments are not measured settings.',
            'Full POST→DONE includes first-output latency and delivery. 512/p50 is not ITL/TPOT. Formal observers OFF; no formal graph/GDN hit-count proof.'
        ]
        for y, line in zip((.143, .112, .081, .050), lines):
            fig.text(.065, y, line, fontsize=8.5, color='#475569')

    args.out.mkdir(parents=True); files = []
    for name, normalized in (('interactivity', True), ('latency', False)):
        if normalized:
            fig, ax = plt.subplots(figsize=(16, 9.2)); axes = [ax]; metrics = [h.XS[0]]
        else:
            fig, axes = plt.subplots(1, 2, figsize=(18, 9.2)); metrics = h.XS
        fig.subplots_adjust(left=.065, right=.965, bottom=.25, top=.74, wspace=.23)
        furniture(fig, 'minivllm and peers: throughput–full-request speed' if normalized else 'minivllm and peers: throughput–latency frontiers')
        for ax, x in zip(axes, metrics):
            draw(ax, x, normalized)
        for ext in ('png', 'svg'):
            path = args.out / f'D512-frontier-{name}.{ext}'
            fig.savefig(path, dpi=170, metadata={'Date': None} if ext == 'svg' else None); files.append(path)
        plt.close(fig)
    # A concise, endpoint-only view for the user's engine-frontier comparison.
    # Detailed historical observations remain in the two figures above.
    fig = plt.figure(figsize=(16, 10))
    main_ax = fig.add_axes([.07, .27, .56, .51])
    c8_ax = fig.add_axes([.72, .53, .24, .25])
    c1_ax = fig.add_axes([.72, .18, .24, .25])
    engine_names = {'native': 'minivllm', 'omlx': 'oMLX', 'lithos': 'Lithos'}
    engine_markers = {'native': 'o', 'omlx': 's', 'lithos': '^'}
    selected = fronts[h.XS[0]]
    by_c = {c: {engine: next(r for r in selected[engine] if r['concurrency'] == c)
                for engine in selected} for c in (1, 8)}
    def pair(row):
        return 512 / row['metrics'][h.XS[0]]['median'], row['metrics'][h.Y]['median']
    for ax in (main_ax, c1_ax, c8_ax):
        ax.set_axisbelow(True); ax.grid(color='#e2e8f0', lw=.75)
    for engine, series in selected.items():
        coords = sorted(pair(r) for r in series)
        main_ax.plot([x for x, y in coords], [y for x, y in coords],
            color=colors[engine], marker=engine_markers[engine], lw=2.5, ms=8,
            label=engine_names[engine], zorder=3)
    for c, ax in ((1, c1_ax), (8, c8_ax)):
        for engine, row in by_c[c].items():
            x, y = pair(row); m = row['metrics']; color = colors[engine]
            limits = sorted(512 / m[h.XS[0]][bound] for bound in ('minimum', 'maximum'))
            ax.errorbar(x, y, xerr=[[x-limits[0]], [limits[1]-x]],
                yerr=[[y-m[h.Y]['minimum']], [m[h.Y]['maximum']-y]],
                fmt=engine_markers[engine], ms=7, color=color, elinewidth=1, capsize=3, zorder=3)
        # Lithos C8 is fully retained in the main chart; this zoom separates the
        # nearly coincident minivllm/oMLX endpoints without changing their values.
        zoom_engines = ('native', 'omlx') if c == 8 else tuple(by_c[c])
        xr = [512 / by_c[c][e]['metrics'][h.XS[0]][bound]
              for e in zoom_engines for bound in ('minimum', 'maximum')]
        yr = [by_c[c][e]['metrics'][h.Y][bound]
              for e in zoom_engines for bound in ('minimum', 'maximum')]
        xm=max(.1, (max(xr)-min(xr))*.35); ym=max(.2, (max(yr)-min(yr))*.45)
        ax.set_xlim(min(xr)-xm, max(xr)+xm); ax.set_ylim(min(yr)-ym, max(yr)+ym)
        for engine in zoom_engines:
            x,y=pair(by_c[c][engine]); label=engine_names[engine]
            offsets = {'native':(-6,12), 'omlx':(5,-18)} if c==8 else {'native':(-35,12),'omlx':(-35,10),'lithos':(5,-18)}
            ax.annotate(label,(x,y),xytext=offsets[engine],textcoords='offset points',color=colors[engine],fontsize=10)
        ratio=by_c[c]['native']['metrics'][h.Y]['median']/by_c[c]['omlx']['metrics'][h.Y]['median']-1
        ax.set_title(f'C{c} zoom · minivllm throughput {ratio:+.2%} vs oMLX',loc='left',fontsize=11,pad=12)
        ax.tick_params(labelsize=9)
        ax.set_xlabel('Full-request speed (tokens/s)', fontsize=10)
        ax.set_ylabel('Total output (tokens/s)', fontsize=10)
    # Explicit endpoint labels; C means measured HTTP concurrency, not batch.
    positions = {(1,'native'):(35,70),(1,'omlx'):(48,88),(1,'lithos'):(34,26),
                 (8,'native'):(27,185),(8,'omlx'):(7,145),(8,'lithos'):(2,68)}
    for c, group in by_c.items():
        for engine,row in group.items():
            x,y=pair(row)
            main_ax.annotate(f'{engine_names[engine]} C{c}',(x,y),xytext=positions[c,engine],
                textcoords='data',fontsize=11,color=colors[engine],
                arrowprops=dict(arrowstyle='-',color=colors[engine],lw=.8))
    main_ax.set_xlim(0,56); main_ax.set_ylim(0,200)
    main_ax.set_xlabel('Full-request speed: 512 / median-process p50 POST→DONE (tokens/s) →',labelpad=10,fontsize=11)
    main_ax.set_ylabel('Aggregate committed output throughput (tokens/s) →',fontsize=12)
    main_ax.text(.03,.95,'BETTER ↗',transform=main_ax.transAxes,color='#475569',weight='bold')
    fig.suptitle('minivllm vs oMLX vs Lithos: measured engine frontiers',x=.07,ha='left',y=.965,fontsize=22,weight='bold',color='#172536')
    fig.text(.07,.918,'M5 Ultra · Qwen3.8-27B NVFP4 · 512 output tokens/request · HTTP C1 and C8 · 3 fresh processes/group',fontsize=12,color='#475569')
    main_ax.legend(loc='upper left',bbox_to_anchor=(0,1.17),ncol=3,frameon=False,fontsize=12)
    fig.text(.07,.185,'minivllm frontier: C1 = graph + GDN; C8 = E024 graph. Each endpoint is a measured executable configuration.',fontsize=10,color='#475569')
    fig.text(.07,.145,'Lines connect measured endpoints only. C2–7 are unmeasured. HTTP concurrency C does not mean GPU batch size.',fontsize=10,color='#475569')
    fig.text(.07,.105,'Horizontal axis is full-request speed, not P90 token interactivity or ITL. Vertical axis is throughput, not tokens per dollar.',fontsize=10,color='#475569')
    fig.text(.07,.065,'Zoom whiskers show process min–max, not confidence intervals. Lithos C8 uses its measured FIFO service; full value is on the main chart.',fontsize=10,color='#475569')
    for ext in ('png','svg'):
        path=args.out/f'D512-engine-frontier.{ext}'
        fig.savefig(path,dpi=170,metadata={'Date':None} if ext=='svg' else None); files.append(path)
    plt.close(fig)
    report = dict(schema='D512-v16-render-verification-v1', status='RENDERED', visual_review='PENDING',
        renderer_sha256=h.sha(Path(__file__).read_bytes()), preparation_source_sha256=receipt['source_sha256'],
        chartdata_sha256=h.sha(blob), original_38_points_unchanged=True, all_50_medians_and_frontiers_recomputed=True,
        artifacts={p.name: dict(sha256=h.sha(p.read_bytes()), bytes=p.stat().st_size) for p in files})
    (args.out / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
