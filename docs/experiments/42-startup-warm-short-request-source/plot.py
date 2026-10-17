"""Actual V123 short-request throughput/latency plot; CPU Matplotlib only."""
from pathlib import Path
import json,statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
D=Path(__file__).resolve().parent;r=json.loads((D/'plot-data.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','svg.hashsalt':'v123-short-request'})
fig,ax=plt.subplots(figsize=(8.4,5.2),layout='constrained')
for mode,color,marker in [('ordinary','#326A9D','o'),('speculative','#B77815','s')]:
 a=[x for x in r if x['mode']==mode];xx=[x['latency_ms'] for x in a];yy=[x['throughput_tok_s'] for x in a];x=statistics.median(xx);y=statistics.median(yy)
 ax.scatter(xx,yy,s=65,c=color,marker=marker,alpha=.6,edgecolors='white',linewidths=.7,label=f'{mode}: 3 measured requests')
 ax.scatter([x],[y],s=170,c=color,marker=marker,edgecolors='#222222',linewidths=1.2,zorder=4)
 ax.annotate(f'{mode} median\n{x:.3f} ms / {y:.3f} tok/s',(x,y),xytext=(15,-42) if mode=='ordinary' else (-15,22),textcoords='offset points',ha='left' if mode=='ordinary' else 'right',fontsize=10,color=color,arrowprops={'arrowstyle':'-','color':color,'lw':.8})
ax.set(xlabel='HTTP POST → [DONE] latency (ms; lower is better)',ylabel='Completion throughput (tokens/s; higher is better)',xlim=(0,3000),ylim=(0,30));ax.grid(color='#E6E6E6',linewidth=.7);ax.set_axisbelow(True);ax.legend(loc='lower left',frameon=False,fontsize=9)
fig.suptitle('V123 short-request screen: speculative throughput was lower',fontsize=13,weight='bold');ax.set_title('C1 · 19 input / 32 output tokens · one fresh speculative → ordinary pair',fontsize=9,color='#555555',pad=12)
fig.text(.02,-.045,'Large outlined marks = medians; small marks = 3 measured requests per arm (may overlap). Warmups excluded.\nFixed 32 tokens/request: throughput = 32 / latency(s). One fresh pair; not a complete frontier or adoption result.',fontsize=8,color='#555555')
fig.savefig(D/'short-request-throughput-latency.png',dpi=180,bbox_inches='tight',metadata={'Software':'Matplotlib; saved V123 data only'});fig.savefig(D/'short-request-throughput-latency.svg',bbox_inches='tight',metadata={'Date':None,'Creator':'Matplotlib; saved V123 data only'})
