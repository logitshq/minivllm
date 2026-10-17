"""Static actual V121 pilot chart. CPU only; requires Matplotlib."""
from pathlib import Path
import json,statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
D=Path(__file__).resolve().parent
rows=json.loads((D/'plot-data.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','svg.hashsalt':'v121-pilot'})
fig,ax=plt.subplots(figsize=(8.4,5.2),layout='constrained');fig.patch.set_facecolor('white')
styles={'off':('#326A9D','o'),'on':('#B77815','s')}
for arm,(color,marker) in styles.items():
 r=[x for x in rows if x['arm']==arm]
 ax.scatter([x['latency_ms'] for x in r],[x['throughput_tok_s'] for x in r],s=65,c=color,marker=marker,alpha=.65,edgecolors='white',linewidths=.7,label=f'TCP_NODELAY {arm.upper()}: 3 requests')
 x=statistics.median(x['latency_ms'] for x in r);y=statistics.median(x['throughput_tok_s'] for x in r)
 ax.scatter([x],[y],s=180,c=color,marker=marker,edgecolors='#222222',linewidths=1.2,zorder=4)
 ax.annotate(f'{arm.upper()} median\n{x:.3f} ms / {y:.3f} tok/s',(x,y),xytext=(-15,-50) if arm=='off' else (15,25),textcoords='offset points',ha='center' if arm=='off' else 'left',color=color,fontsize=10,arrowprops={'arrowstyle':'-','color':color,'lw':.8})
ax.set(xlabel='HTTP POST → [DONE] latency (ms; lower is better)',ylabel='Completion throughput (tokens/s; higher is better)',xlim=(1224.5,1234.8),ylim=(25.91,26.15))
ax.grid(color='#E6E6E6',linewidth=.7);ax.set_axisbelow(True)
ax.legend(loc='upper right',frameon=False,fontsize=9)
fig.suptitle('V121 transport pilot: no benefit in the first OFF/ON pair',fontsize=13,weight='bold')
ax.set_title('Native C1 · fixed 32 output tokens · 1 fresh server per setting · 3 measured requests',fontsize=9,color='#555555',pad=12)
fig.text(.02,-.045,'Large outlined marks = medians; small marks = individual measured requests. No warmups plotted.\nFocused axes. With 32 tokens/request, throughput = 32 / latency(s); these axes are not independent.',fontsize=8,color='#555555')
fig.savefig(D/'transport-pilot.png',dpi=180,bbox_inches='tight',metadata={'Software':'Matplotlib; saved V121 data only'})
fig.savefig(D/'transport-pilot.svg',bbox_inches='tight',metadata={'Date':None,'Creator':'Matplotlib; saved V121 data only'})
