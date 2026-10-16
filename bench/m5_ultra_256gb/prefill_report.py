"""Plot only the fresh paired native measurements; do not mix historical donors."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--summary', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    a = p.parse_args()
    a.out.mkdir(exist_ok=True, parents=True)
    d = json.loads(a.summary.read_text())
    assert d['status'] == 'PASS'
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'figure.facecolor': '#FCFCFB', 'axes.facecolor': '#FCFCFB'})
    fig, (ax, tx) = plt.subplots(1, 2, figsize=(12.8, 6.7), gridspec_kw={'width_ratios': [1.4, 1]})
    colors = {'reference': '#82909C', 'candidate': '#197A67'}
    labels = {'reference': 'Original per-request prefill', 'candidate': 'Grouped NAX prefill'}
    for v in colors:
        points = [dict(C=c, **d['medians'][str(c)][v]) for c in (1, 8, 32, 64)]
        rows = [r for r in d['rows'] if r['variant'] == v]
        ax.scatter([r['x'] for r in rows], [r['y'] for r in rows], color=colors[v], s=19, alpha=.3)
        for row in points:
            dominated = any(q['x'] >= row['x'] and q['y'] >= row['y'] and
                            (q['x'] > row['x'] or q['y'] > row['y']) for q in points)
            row['frontier'] = not dominated
            ax.scatter(row['x'], row['y'], s=64, edgecolor=colors[v], linewidth=1.5,
                       facecolor='#FCFCFB' if dominated else colors[v], zorder=4)
            ax.annotate(f"C{row['C']}", (row['x'], row['y']), xytext=(5, 9 if v == 'candidate' else -15),
                        textcoords='offset points', color=colors[v], fontsize=9)
        frontier = sorted((r for r in points if r['frontier']), key=lambda r: r['x'])
        ax.plot([r['x'] for r in frontier], [r['y'] for r in frontier], color=colors[v], lw=1.4, label=labels[v])
    ax.set(xlabel='Client text-delivery proxy (tokens/s/user)', ylabel='Complete-request throughput (tokens/s/Mac)',
           xlim=(0, max(r['x'] for r in d['rows']) * 1.12), ylim=(0, max(r['y'] for r in d['rows']) * 1.17))
    ax.grid(alpha=.2)
    ax.legend(frameon=False, fontsize=9, loc='upper right')
    cs = [1, 8, 32, 64]
    for offset, v in [(-.18, 'reference'), (.18, 'candidate')]:
        tx.bar([i + offset for i in range(4)], [d['medians'][str(c)][v]['TTFT_s'] for c in cs],
               width=.34, color=colors[v], label=labels[v])
    tx.set(xticks=range(4), xticklabels=[f'C{c}' for c in cs], ylabel='Median time to first text (seconds)')
    tx.grid(axis='y', alpha=.2)
    tx.set_axisbelow(True)
    fig.suptitle('M5 Ultra · exact grouped prefill', x=.075, ha='left', fontsize=21, weight='bold')
    fig.text(.075, .91, 'Qwen3.8-27B NVFP4 · 19 input / 32 output · no prefix cache or drafter', fontsize=11, color='#53616B')
    fig.text(.075, .045, '24 fresh processes; three AB/BA/AB pairs per concurrency; three measured groups per process.\n'
             'Large marks: medians. Small marks: all processes. Hollow marks: dominated within that native variant.\n'
             'Lines connect measured frontier points only. Process spread is not a confidence interval.\n'
             'This figure compares native variants; oMLX and Lithos were not rerun in this experiment.',
             fontsize=9, color='#53616B', linespacing=1.5)
    fig.subplots_adjust(left=.075, right=.96, top=.84, bottom=.25, wspace=.3)
    fig.savefig(a.out / 'paired-prefill.png', dpi=180)
    fig.savefig(a.out / 'paired-prefill.pdf')


if __name__ == '__main__':
    main()
