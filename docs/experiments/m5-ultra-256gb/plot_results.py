import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
out=Path(__file__).resolve().parent;data=json.loads((out/'results.json').read_text());front=data['frontier'];points=front['points_median3_process_medians'];processes=front['processes']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold','figure.facecolor':'#f7f8fa','axes.facecolor':'white'})
fig,(ax,bx)=plt.subplots(1,2,figsize=(12.4,6.5),gridspec_kw={'width_ratios':[1.5,1]});fig.subplots_adjust(left=.075,right=.97,bottom=.30,top=.79,wspace=.27)
colors={'tutorial':'#176b90','omlx':'#cf6b31'};labels={'tutorial':'minivllm V19-on (experimental)','omlx':'oMLX 0.7.1.dev1 balanced'}
for engine in ['tutorial','omlx']:
 raw=[r for r in processes if r['engine']==engine];rows=sorted([r for r in points if r['engine']==engine],key=lambda r:r['C'])
 ax.scatter([r['x'] for r in raw],[r['y'] for r in raw],color=colors[engine],s=25,alpha=.25,zorder=2)
 ax.plot([r['x'] for r in rows],[r['y'] for r in rows],color=colors[engine],marker='o' if engine=='tutorial' else 's',markersize=7,lw=1.6,label=labels[engine],zorder=3)
 for r in rows:ax.annotate('C'+str(r['C']),(r['x'],r['y']),xytext=(5,8),textcoords='offset points',color=colors[engine],weight='bold',fontsize=10)
pareto=sorted([r for r in points if r['pareto_nondominated']],key=lambda r:r['x']);ax.plot([r['x'] for r in pareto],[r['y'] for r in pareto],ls='--',color='#505862',lw=1,alpha=.6,zorder=1)
ax.set(xlim=(28,57),ylim=(28,116),xlabel='Client text-delivery proxy x (tokens/s/user)',ylabel='End-to-end throughput y (tokens/s/Mac)',title='C1 / C2 / C4: three fresh pairs per level')
ax.grid(alpha=.15);ax.legend(loc='upper right',frameon=False,fontsize=9)
changes=data['e2e']['pair_y_percent'];xs=range(3);bx.axhline(0,color='#666',lw=.8);bx.bar(xs,changes,color=['#21836a' if x>0 else '#bd4545' for x in changes],width=.62)
for i,v in enumerate(changes):bx.text(i,v+(.16 if v>=0 else -.17),f'{v:+.2f}%',ha='center',va='bottom' if v>=0 else 'top',fontsize=12,weight='bold')
bx.set(xticks=list(xs),xticklabels=['Pair 1\nOFF → ON','Pair 2\nON → OFF','Pair 3\nOFF → ON'],ylim=(-4.3,4.4),ylabel='V19 ON vs OFF throughput change (%)',title='Internal C1 V19 effect: DO_NOT_ADOPT');bx.yaxis.set_major_locator(MultipleLocator(2));bx.grid(axis='y',alpha=.15);bx.set_axisbelow(True)
bx.text(.5,.94,'One negative pair fails the adoption rule',ha='center',va='top',transform=bx.transAxes,fontsize=9,color='#9c3232')
fig.text(.075,.935,'M5 Ultra · 80 GPU cores · 256 GiB',fontsize=21,weight='bold',color='#142739')
fig.text(.075,.875,'Qwen3.8-27B NVFP4  ·  MLX 0.32.3  ·  AC power  ·  19-token prompt / 32-token output',fontsize=11,color='#485664')
fig.text(.075,.055,'Left: median of three process medians; faint dots retain every process. Dashed line: observed Pareto frontier.\nRight: a separate six-process OFF/ON experiment; aggregate y +2.46%, but pair 3 is negative.\nx is a client delivery proxy, not GPU token rate. No M4 data pooled. Lithos has no supported M5 Ultra profile.',fontsize=9,color='#485664',linespacing=1.55)
fig.savefig(out/'m5-ultra-256gb-pareto.png',dpi=180);fig.savefig(out/'m5-ultra-256gb-pareto.pdf');print(out/'m5-ultra-256gb-pareto.png')
