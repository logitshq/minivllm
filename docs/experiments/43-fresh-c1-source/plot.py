"""Fresh V125 C1 slice: actual full-request latency versus completion throughput."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
D=Path(__file__).resolve().parent;r=json.loads((D/'plot-data.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','svg.hashsalt':'v125-fresh-c1'})
fig,ax=plt.subplots(figsize=(9,5.5),layout='constrained')
styles={'native':('#245A81','o','native CAP64/direct'),'omlx':('#AF702B','s','oMLX'),'lithos':('#7A597B','D','Lithos')}
for e,(color,marker,label) in styles.items():
 a=[x for x in r['arms'] if x['engine']==e];c=next(x for x in r['configurations'] if x['engine']==e)
 ax.scatter([x['latency_ms'] for x in a],[x['throughput_tok_s'] for x in a],s=55,marker=marker,facecolors='none',edgecolors=color,alpha=.7,linewidths=1.2,label=f'{label}: 3 fresh arms')
 x,y=c['latency_ms'],c['throughput_tok_s'];ax.scatter([x],[y],s=180 if c['joint_non_dominated'] else 145,marker=marker,facecolors=color if c['joint_non_dominated'] else 'white',edgecolors='#263238' if c['joint_non_dominated'] else color,linewidths=1.8,zorder=4)
 offset={'native':(15,22),'omlx':(20,-42),'lithos':(-20,26)}[e]
 ax.annotate(f'{label}\n{x:.3f} ms / {y:.3f} tok/s'+('\nC1 non-dominated' if e=='native' else ''),(x,y),xytext=offset,textcoords='offset points',ha='right' if e=='lithos' else 'left',fontsize=10,color=color,arrowprops={'arrowstyle':'-','color':color,'lw':.8})
ax.set(xlabel='Full HTTP request latency p50 (ms; POST → [DONE]; lower is better)',ylabel='Total output throughput (tokens/s; higher is better)',xlim=(1170,1520),ylim=(20.8,27.5));ax.grid(color='#E7EBEF',linewidth=.7);ax.set_axisbelow(True);ax.legend(loc='lower left',frameon=False,fontsize=9)
fig.suptitle('V125 fresh C1 Pareto slice: native leads the measured configurations',fontsize=13,weight='bold');ax.set_title('Qwen3.8-27B NVFP4 · M4 Max 48 GiB · 19 input / 32 output · 9 fresh loads',fontsize=9,color='#555555',pad=12)
fig.text(.02,-.045,'Large marks = median of 3 fresh arms; small hollow marks = all 9 fresh-arm medians (may overlap). Focused axes.\nC1 only; no old C2/C4 points pooled. Each request has 32 tokens, so C1 throughput and latency are reciprocal.',fontsize=8,color='#555555')
fig.savefig(D/'fresh-c1-throughput-latency.png',dpi=180,bbox_inches='tight',metadata={'Software':'Matplotlib; saved V125 C1 data only'});fig.savefig(D/'fresh-c1-throughput-latency.svg',bbox_inches='tight',metadata={'Date':None,'Creator':'Matplotlib; saved V125 C1 data only'})
