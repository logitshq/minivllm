"""Render source-bound forward diagnostics, never HTTP/Pareto measurements."""
import hashlib
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

OUT = Path(__file__).resolve().parent
SOURCE = OUT.parents[2] / 'experiments/m5-ultra-256gb/E024-route-timing-v1/independent-numerical.json'
EXPECTED = '86564bf21aae71c733f4349fc87cefd67d7accc36eaf8783ac7ade13b51ef821'
raw = SOURCE.read_bytes()
assert hashlib.sha256(raw).hexdigest() == EXPECTED
source = json.loads(raw)
assert source['status'] == 'PASS' and source['pareto_eligible'] is False
assert len(source['configs']) == 8
rows = []
for c in source['configs']:
    arms = {}
    for arm in ('A', 'C'):
        samples = [r['forward_ms'] for r in c['raw_samples'] if not r['warm'] and r['arm'] == arm]
        assert len(samples) == 6 and all(x > 0 for x in samples)
        median = statistics.median(samples)
        assert abs(median - c['medians'][arm]['forward_ms']) < 1e-10
        arms[arm] = dict(samples_ms=samples, median_ms=median, min_ms=min(samples), max_ms=max(samples))
    ratio = arms['A']['median_ms'] / arms['C']['median_ms']
    assert abs(ratio - c['median_forward_speedup']) < 1e-12
    rows.append(dict(B=c['B'], used=c['used'], gates=c['gates'], arms=arms, forward_rate_gain=ratio-1))
assert {(r['B'], r['used'], r['gates']) for r in rows} == {(b, u, g) for b in (1,8) for u in (1023,1024) for g in (False,True)}
(OUT/'chartdata.json').write_text(json.dumps(dict(source=str(SOURCE.relative_to(OUT.parents[2])), source_sha256=EXPECTED,
    scope='forward-only; not HTTP, committed-token throughput, or Pareto', process_repetitions=1,
    samples_per_arm=6, interval='observed min-max; not confidence interval', rows=rows), indent=2)+'\n')
plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':11, 'axes.labelcolor':'#26313a', 'text.color':'#26313a',
    'axes.edgecolor':'#b7c0c8', 'xtick.color':'#596674', 'ytick.color':'#26313a', 'svg.fonttype':'none', 'svg.hashsalt':'E024-route-timing-v1'})
fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharex=True, sharey=True)
fig.subplots_adjust(left=.12, right=.98, bottom=.24, top=.72, wspace=.11)
colors = {'A':'#bac2ca', 'C':'#316db4'}
for ax, batch in zip(axes, (1,8)):
    group = [r for r in rows if r['B'] == batch]
    assert len(group)==4
    for y, row in enumerate(group):
        for arm, offset in [('A',-.19),('C',.19)]:
            v = row['arms'][arm]; med=v['median_ms']; yy=y+offset
            ax.barh(yy, med, height=.31, color=colors[arm], edgecolor='#5a6673' if arm=='A' else colors[arm], linewidth=.6)
            ax.errorbar(med, yy, xerr=[[med-v['min_ms']],[v['max_ms']-med]], fmt='none', ecolor='#26313a', capsize=3, linewidth=1)
            ax.text(v['max_ms']+.65, yy, f'{med:.2f}', va='center', fontsize=10)
    ax.set_title(f'Batch {batch}', fontsize=15, loc='left', pad=18, fontweight='bold')
    ax.set_xlim(0,48); ax.set_xticks([0,10,20,30,40]); ax.set_ylim(3.65,-.65)
    ax.set_yticks(range(4)); ax.set_yticklabels(['Used 1023\nGates OFF','Used 1023\nGates ON','Used 1024\nGates OFF','Used 1024\nGates ON'])
    ax.set_xlabel('Forward time (ms) · lower is better', labelpad=12)
    ax.set_axisbelow(True); ax.xaxis.grid(True,color='#e5e9ed',linewidth=.8)
    ax.tick_params(axis='y',length=0,pad=9)
    ax.spines[['top','right','left']].set_visible(False)
fig.text(.12,.945,'Route-preserving graph: isolated forward timing', fontsize=21,fontweight='bold')
fig.text(.12,.895,'M5 Ultra 256 GiB  ·  Qwen3.8-27B NVFP4  ·  17 Oct 2026', fontsize=12,color='#596674')
fig.legend(handles=[Patch(facecolor=colors['A'],edgecolor='#5a6673',label='Eager baseline (A)'),Patch(facecolor=colors['C'],label='Compiled graph (C)')],
    loc='upper left',bbox_to_anchor=(.114,.853),frameon=False,ncol=2,fontsize=12)
fig.text(.12,.135,'Medians of 6 samples per arm; whiskers show observed min–max, not confidence intervals.',fontsize=11)
fig.text(.12,.097,'One process, 3 ACCA blocks/configuration. Includes forward + all 129 outputs + sync; excludes teardown.',fontsize=10,color='#596674')
fig.text(.12,.06,'Not HTTP latency or committed-token throughput. Full sample bit checks PASS; no new Pareto points.',fontsize=10,color='#596674')
fig.savefig(OUT/'route-graph-forward.png',dpi=160,facecolor='white')
fig.savefig(OUT/'route-graph-forward.svg',facecolor='white',metadata={'Date':'2026-10-17'})
svg = OUT/'route-graph-forward.svg'
svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
print('Verified eight configurations, 96 formal samples and audit SHA; saved PNG/SVG/data.')
