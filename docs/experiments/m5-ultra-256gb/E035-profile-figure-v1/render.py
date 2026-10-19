"""Static diagnostic figure; source-backed, not a new frontier measurement."""
import hashlib,json,shutil
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

R=Path('/Users/ryanli/code/minivllm')
I=R/'bench/m5_ultra_256gb/c1_profile_cycle_v1/analysis-v1/readout-v1'
O=R/'docs/experiments/m5-ultra-256gb/E035-profile-figure-v1'
O.mkdir()
readout=json.loads((I/'readout.json').read_text())
calls=json.loads((I/'async-calls.json').read_text())
assert readout['status']=='PASS' and readout['performance_qualified'] is False
assert readout['mode_order']==['off','runner','full']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.labelcolor':'#273343','text.color':'#273343','xtick.color':'#455468','ytick.color':'#455468','axes.edgecolor':'#9AA5B3','svg.fonttype':'none'})
fig,(a,b)=plt.subplots(1,2,figsize=(13.8,6.0),gridspec_kw={'width_ratios':[1,1.45]})
fig.subplots_adjust(left=.075,right=.96,bottom=.24,top=.77,wspace=.59)
fig.suptitle('C1 pipeline diagnostic: locate the large cost first',x=.075,y=.96,ha='left',fontsize=20,fontweight='bold')
fig.text(.075,.885,'Qwen3.8-27B NVFP4 · M5 Ultra · 3 fresh servers · 8 warm + 8 diagnostic requests per mode',fontsize=11)
modes=readout['mode_order']
ids=[r['fixture_id'] for r in readout['modes']['off']['requests']]
values=[]
for mode in modes:
    rows=readout['modes'][mode]['requests'];assert [r['fixture_id'] for r in rows]==ids
    values.append([r['e2e_s'] for r in rows])
for j in range(8):a.plot(range(3),[v[j] for v in values],'-o',color='#A7ADB6',lw=.85,ms=4,alpha=.65)
medians=[readout['modes'][m]['e2e_s']['p50'] for m in modes]
a.scatter(range(3),medians,marker='D',s=65,color='#2368A2',zorder=4,label='Median')
for x,y in enumerate(medians):a.annotate(f'{y:.4f}',(x,y),xytext=(0,-22),textcoords='offset points',ha='center',color='#2368A2',fontsize=10)
a.set_xticks(range(3),['OFF','Runner','Full']);a.set_xlim(-.35,2.35);a.set_ylim(10.49,10.82)
a.set_ylabel('POST → DONE (seconds; focused axis)');a.set_title('Same eight fixtures; fixed mode order',loc='left',fontsize=12,pad=14)
a.grid(axis='y',color='#E5E8EC');a.legend(frameon=False,loc='upper right')
full=readout['modes']['full']['profile'];s=calls['modes']['full']['call_ms']
rows=[('Current roots, already submitted',s['ahead_consumed_ordinal_1'],4072),('Compiled callable (nested)',full['per_kind']['compiled_callable']['clipped_duration_ms'],4088),('Decode submission (includes callable)',full['per_kind']['decode_graph_submission']['clipped_duration_ms'],4088),('Successor async submission',s['ahead_consumed_ordinal_2'],4072)]
for y,(name,d,n) in enumerate(rows):
    assert d['count']==n
    color='#2368A2' if y==3 else '#738194'
    b.scatter(d['mean'],y,s=75,color=color,zorder=3)
    b.text(d['mean']*1.18,y,f"{d['mean']:.5f} ms" if y==0 else f"{d['mean']:.3f} ms",va='center',fontsize=10,color=color)
b.set_xscale('log');b.set_xlim(.002,90);b.set_ylim(-.6,3.6);b.invert_yaxis()
b.set_yticks(range(4),[r[0] for r in rows]);b.set_xticks([.01,.1,1,10]);b.xaxis.set_major_formatter(FuncFormatter(lambda x,p:f'{x:g}'))
b.set_xlabel('Mean recorded wall interval per call (ms; log scale)');b.set_title('Full mode: steady calls dominate the profile',loc='left',fontsize=12,pad=14)
b.grid(axis='x',color='#E5E8EC')
fig.text(.075,.14,'The repeated current-root call totals 18.5 ms across 4,072 steps: 0.022% of local service time.',fontsize=12,fontweight='bold')
fig.text(.075,.091,'Intervals overlap; compiled calls are nested. Successor submission can include backend pacing / waiting, not only GPU work.',fontsize=10,color='#5C6778')
fig.text(.075,.051,'One process per mode, fixed order; observer differences are descriptive. No new numerical qualification or Pareto point.',fontsize=10,color='#5C6778')
for ext in ('png','svg'):fig.savefig(O/f'C1-profile-diagnostic.{ext}',dpi=180,facecolor='white')
plt.close(fig)
shutil.copyfile(__file__,O/'render.py')
def pin(p):return {'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}
inputs={str(I/n):pin(I/n) for n in ['readout.json','async-calls.json']}
(O/'provenance.json').write_text(json.dumps({'inputs':inputs,'renderer':pin(O/'render.py'),'outputs':{n:pin(O/n) for n in ['C1-profile-diagnostic.png','C1-profile-diagnostic.svg']},'performance_qualified':False},indent=2)+'\n')
print(json.dumps({'status':'RENDERED_REQUIRES_VISUAL_REVIEW','directory':str(O)}))
