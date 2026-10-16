"""Current native frontier, with explicitly historical external baselines."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

p = argparse.ArgumentParser()
p.add_argument('--results', required=True, type=Path)
p.add_argument('--out', required=True, type=Path)
a = p.parse_args()
d = json.loads(a.results.read_text())
assert d['status'] == 'PASS'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.spines.top': False,
                    'axes.spines.right': False, 'figure.facecolor': '#FCFCFB', 'axes.facecolor': '#FCFCFB'})
colors = {'native': '#197A67', 'omlx': '#CD6B2D', 'omlx_aggressive': '#CD6B2D', 'lithos': '#927A25'}
markers = {'native': 'o', 'omlx': 's', 'omlx_aggressive': '^', 'lithos': 'D'}
fig = plt.figure(figsize=(12.8, 9.1))
ax = fig.add_axes((.09, .29, .86, .49))
raw = [{**r, 'engine': 'native'} for r in d['current_native_process_rows']] + d['historical_external_process_rows']
for r in raw:
    ax.scatter(r['x'], r['y'], color=colors[r['engine']], marker=markers[r['engine']], s=19, alpha=.26)
for r in d['points']:
    e = r['engine']
    ax.scatter(r['x'], r['y'], s=67, marker=markers[e], edgecolor=colors[e], linewidth=1.5,
               facecolor=colors[e] if r['own_frontier'] else '#FCFCFB', zorder=4)
    if e in ('native', 'omlx'):
        ax.annotate(f"C{r['C']}", (r['x'], r['y']), xytext=(6, 9 if e == 'native' else -17),
                    textcoords='offset points', color=colors[e], fontsize=9)
for family in ('native', 'omlx', 'lithos'):
    points = sorted([r for r in d['points'] if r['family'] == family and r['own_frontier']], key=lambda r: r['x'])
    ax.plot([r['x'] for r in points], [r['y'] for r in points], color=colors[family], lw=1.5)
ax.set(xlim=(0, max(r['x'] for r in raw) * 1.09), ylim=(0, max(r['y'] for r in raw) * 1.15),
       xlabel='Client text-delivery proxy (tokens/s/user)', ylabel='Complete-request throughput (tokens/s/Mac)')
ax.grid(alpha=.22)
ix = ax.inset_axes((.65, .66, .31, .29))
for r in raw:
    if r['C'] == 1:
        ix.scatter(r['x'], r['y'], color=colors[r['engine']], marker=markers[r['engine']], alpha=.25, s=16)
for r in d['points']:
    if r['C'] == 1:
        ix.scatter(r['x'], r['y'], color=colors[r['engine']], marker=markers[r['engine']], s=42)
ix.set_title('C1 detail · same units', fontsize=10)
ix.tick_params(labelsize=8)
ix.grid(alpha=.15)
labels = {'native': 'Current native · grouped prefill', 'omlx': 'Historical oMLX · balanced',
          'omlx_aggressive': 'Historical oMLX · aggressive C1', 'lithos': 'Historical Lithos · C1 only'}
fig.legend(handles=[Line2D([0], [0], color=colors[e], marker=markers[e], label=label, lw=1.3) for e, label in labels.items()],
           loc='upper left', bbox_to_anchor=(.084, .85), frameon=False, ncol=2, fontsize=10)
fig.text(.09, .95, 'M5 Ultra · updated native Pareto measurements', fontsize=21, weight='bold')
fig.text(.09, .906, '256 GiB · Qwen3.8-27B NVFP4 · 19 input / 32 output · target-only decoding', fontsize=11.5, color='#53616B')
covered = sum(bool(r['covered_by_native_C']) for r in d['coverage'])
fig.text(.09, .873, f'{covered}/9 archived external configuration medians covered by a measured current native point', fontsize=11, color='#53616B')
fig.text(.09, .06, 'Native: final core, C1/2/4/8/16/32/64, three fresh processes per point. External: earlier archived sweep.\n'
         'Large marks: median of process medians; faint marks retain all processes. Spread is not a confidence interval.\n'
         'Filled marks: frontier within each family. Lines are guides between measured configurations.\n'
         'Three warm + three measured groups per process; greedy, normal EOS, no prefix cache or drafter.\n'
         'Lithos uses the local M5 profile and supports C1 on its official endpoint. External engines were not rerun here.\n'
         'Scope: this short workload only. Separate paired native results include TTFT and every raw sample.',
         fontsize=9.5, color='#53616B', linespacing=1.6)
a.out.mkdir(exist_ok=True, parents=True)
fig.savefig(a.out / 'current-native-frontier.png', dpi=180)
fig.savefig(a.out / 'current-native-frontier.pdf')
