from pathlib import Path
import json,statistics,hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('/Users/ryanli/code/minivllm');D=R/'docs/experiments/m5-ultra-256gb/E058-logits-agent-feature-run-v1'
complete=json.loads((D/'complete.json').read_text());assert complete['status']=='PASS_FEATURES_LOCAL_COST'
audit=json.loads((D/'independent-audit.json').read_text());assert audit['status']=='PASS' and audit['pairs_checked']==24
rows=complete['timings'];assert len(rows)==60
arms=('native','shared_without_features','shared_with_features');colors={'native':'#747980','shared_without_features':'#b38a36','shared_with_features':'#2b68b1'}
labels={'native':'Native + features','shared_without_features':'Shared, no features','shared_with_features':'Shared + features'}
plt.rcParams.update({'font.size':11,'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#b1b5ba','text.color':'#24292f','axes.labelcolor':'#24292f'})
fig,(ax,ratio)=plt.subplots(1,2,figsize=(12,6),gridspec_kw={'width_ratios':[1.45,1]})
summary={}
for wi,w in enumerate((4,8)):
 summary[w]={}
 for ai,arm in enumerate(arms):
  values=[r['elapsed_ns']/1e6 for r in rows if r['width']==w and r['arm']==arm]
  assert len(values)==10
  med=statistics.median(values);summary[w][arm]=med;x=wi*4+ai
  ax.bar(x,med,width=.65,color=colors[arm],alpha=.72,label=labels[arm] if wi==0 else None)
  ax.scatter([x+(i-4.5)*.034 for i in range(10)],values,color=colors[arm],edgecolors='white',linewidths=.5,s=20,zorder=3)
  ax.text(x, max(values)+4,f'{med:.2f}',ha='center',fontsize=10)
 calculated=[]
 for group in range(5):
  byarm={arm:statistics.median([r['elapsed_ns'] for r in rows if r['width']==w and r['group']==group and r['arm']==arm]) for arm in arms}
  calculated.append(byarm['native']/byarm['shared_with_features'])
 assert calculated==complete['timing_summary'][str(w)]['group_ratios']
 ratio.scatter([wi+(g-2)*.05 for g in range(5)],calculated,color=colors['shared_with_features'],s=42,edgecolors='white',linewidths=.6)
 ratio.hlines(statistics.median(calculated),wi-.25,wi+.25,color='#24292f',linewidth=2)
 ratio.text(wi,max(calculated)+.03,f'{statistics.median(calculated):.3f}× median',ha='center',fontsize=11)
ax.set_xticks([1,5],['T4','T8']);ax.set_ylabel('Elapsed ms per materialized forward (lower is better)');ax.set_ylim(0,max(r['elapsed_ns']/1e6 for r in rows)*1.2)
ax.set_title('Local forward latency',loc='left');ax.legend(frameon=False,loc='upper left',fontsize=9);ax.grid(axis='y',alpha=.17);ax.set_axisbelow(True)
ratio.axhline(1,color='#747980',linestyle='--',linewidth=1);ratio.set_xticks([0,1],['T4','T8']);ratio.set_xlim(-.5,1.5);ratio.set_ylim(.9,max(v for s in complete['timing_summary'].values() for v in s['group_ratios'])+.12);ratio.set_ylabel('Native / shared with features (higher is better)');ratio.set_title('Five within-group ratios',loc='left');ratio.grid(axis='y',alpha=.17)
fig.suptitle('E058 · Logits-agent modules on M5 Ultra',fontsize=17,x=.08,ha='left',y=.98)
fig.text(.08,.918,'Qwen3.8-27B NVFP4 · teacher-forced verification · no serving/Pareto claim',fontsize=11)
fig.text(.08,.055,'T is verifier input width, not concurrency. Seed uses 1023 positions; five target layers captured.\nLeft: 10 samples/arm/width, bars = medians. Right: 5 paired-group ratios. Dots show observations, not CIs.',fontsize=9,color='#505761')
fig.subplots_adjust(left=.08,right=.97,top=.83,bottom=.19,wspace=.36)
fig.savefig(D/'local-forward-cost.png',dpi=180);fig.savefig(D/'local-forward-cost.svg')
svg=D/'local-forward-cost.svg';svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
(D/'plot-values.json').write_text(json.dumps({'median_ms':summary,'source_sha256':hashlib.sha256((D/'complete.json').read_bytes()).hexdigest()},indent=2)+'\n')
print(json.dumps(summary,indent=2))
