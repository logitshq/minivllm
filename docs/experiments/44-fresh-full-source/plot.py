"""Standalone saved-evidence Pareto chart. CPU only; no model/runtime imports."""
from pathlib import Path
import json,sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
D=Path(__file__).resolve().parent
r=json.loads((D/'plot-data.json').read_text())
colors={'native':'#245A81','omlx':'#AF702B','lithos':'#7A597B'}
markers={1:'o',2:'s',4:'^'}
names={'native':'native','omlx':'oMLX','lithos':'Lithos'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#9CA5AC','axes.labelcolor':'#283640','xtick.color':'#40505B','ytick.color':'#40505B','svg.fonttype':'none','svg.hashsalt':'v128-full-pareto'})
fig=plt.figure(figsize=(11.2,7.1),facecolor='white')
ax=fig.add_axes([.09,.30,.86,.52])
ax.set_axisbelow(True);ax.grid(color='#E8ECEF',linewidth=.7)
# These are guides between observed configurations, not an interpolated feasible region.
for e in colors:
 pts=sorted([c for c in r['configurations'] if c['engine']==e and c['engine_non_dominated']],key=lambda q:q['latency_ms'])
 if len(pts)>1:ax.plot([c['latency_ms'] for c in pts],[c['throughput_tok_s'] for c in pts],color=colors[e],lw=1.1,ls=(0,(4,3)),alpha=.7,zorder=1)
front=sorted([c for c in r['configurations'] if c['joint_non_dominated']],key=lambda q:q['latency_ms'])
ax.plot([c['latency_ms'] for c in front],[c['throughput_tok_s'] for c in front],color='#253139',lw=1.8,zorder=2)
for q in r['arms']:
 ax.scatter(q['latency_ms'],q['throughput_tok_s'],s=43,marker=markers[q['concurrency']],facecolors='none',edgecolors=colors[q['engine']],linewidths=1.1,alpha=.7,zorder=3)
offsets={'native:C1':(20,32),'omlx:C1':(-70,-30),'lithos:C1':(12,22),'native:C2':(-16,27),'omlx:C2':(14,-26),'native:C4':(-15,-34),'omlx:C4':(-16,-31)}
for c in r['configurations']:
 x,y=c['latency_ms'],c['throughput_tok_s'];color=colors[c['engine']];joint=c['joint_non_dominated'];marker=markers[c['concurrency']]
 ax.scatter(x,y,s=150 if marker!='^' else 180,marker=marker,facecolors=color if joint else 'white',edgecolors='#253139' if joint else color,linewidths=1.7,zorder=5)
 dx,dy=offsets[c['config']];label=f"{names[c['engine']]} C{c['concurrency']}\n{x:,.1f} ms · {y:.3f} tok/s"
 ax.annotate(label,(x,y),xytext=(dx,dy),textcoords='offset points',ha='right' if c['config'] in ('native:C4','omlx:C4') else 'left',va='center',fontsize=9.3,color=color,zorder=6,bbox={'facecolor':'white','edgecolor':'none','pad':1.8,'alpha':.94},arrowprops={'arrowstyle':'-','color':color,'lw':.75,'shrinkA':2,'shrinkB':6})
ax.set(xlim=(1160,2420),ylim=(17,68),xticks=[1200,1400,1600,1800,2000,2200,2400],yticks=[20,30,40,50,60],xlabel='Full HTTP request latency p50 (ms)  ← lower is better',ylabel='Total output throughput (tokens/s)  → higher is better')
ax.xaxis.labelpad=11;ax.yaxis.labelpad=11
fig.text(.09,.956,'M4 Max · Qwen3.8-27B',fontsize=17,weight='bold',color='#23333E')
fig.text(.09,.914,'Throughput vs full-request latency',fontsize=14,color='#23333E')
fig.text(.09,.873,'NVFP4 · 48 GiB · 19 input / 32 greedy output · 7 configurations · 21 fresh loads',fontsize=10,color='#5F6E78')
handles=[Line2D([],[],color=colors[e],marker='o',mfc=colors[e] if e=='native' else 'white',mec=colors[e],ls='none',markersize=7,label=names[e]) for e in colors]
handles += [Line2D([],[],color='#253139',lw=1.8,label='Joint non-dominated frontier')]
fig.legend(handles=handles,loc='upper right',bbox_to_anchor=(.955,.958),frameon=False,ncol=2,fontsize=9,columnspacing=1.5,handlelength=1.7)
shape=[Line2D([],[],color='#596670',marker=markers[c],mfc='white',ls='none',markersize=7,label=f'C{c}') for c in (1,2,4)]
shape += [Line2D([],[],color='#596670',marker='o',mfc='white',ls='none',markersize=4,label='Small = fresh arm'),Line2D([],[],color='#596670',marker='o',mfc='#596670',ls='none',markersize=8,label='Large = configuration median')]
fig.legend(handles=shape,loc='lower left',bbox_to_anchor=(.08,.149),frameon=False,ncol=5,fontsize=9,handletextpad=.6,columnspacing=1.4)
fig.text(.09,.125,'Filled large marks are jointly non-dominated; hollow large marks are dominated. All 21 fresh-arm points remain at their true coordinates (some overlap).',fontsize=8.3,color='#5F6E78')
fig.text(.09,.096,'Latency: request p50 within each group → median of 3 measured groups → median of 3 fresh loads. Throughput uses the same group/arm hierarchy.',fontsize=8.3,color='#5F6E78')
fig.text(.09,.067,'Native C1: CAP64/direct asynchronous ICB. Native C2/C4: accepted25 real batching/CAP256. Static startup profiles; no same-process migration.',fontsize=8.3,color='#5F6E78')
fig.text(.09,.038,'Lines only guide between observed configurations; they do not establish interpolated operating points. Lithos C2/C4: N/A. This workload/configuration set only.',fontsize=8.3,color='#5F6E78')
fig.savefig(D/'fresh-full-throughput-latency.png',dpi=190,metadata={'Software':'Matplotlib; saved V128 same-campaign rows only'})
fig.savefig(D/'fresh-full-throughput-latency.svg',metadata={'Date':None,'Creator':'Matplotlib; saved V128 same-campaign rows only'})
(D/'render-metadata.json').write_text(json.dumps({'renderer':'Matplotlib','version':matplotlib.__version__,'backend':matplotlib.get_backend(),'python':sys.executable,'font':'DejaVu Sans','small_arm_marks':len(r['arms']),'configuration_marks':len(r['configurations']),'joint_frontier':r['joint_frontier'],'GPU_used':False},indent=2)+'\n')
print(json.dumps({'PNG':str(D/'fresh-full-throughput-latency.png'),'SVG':str(D/'fresh-full-throughput-latency.svg'),'arms':len(r['arms']),'configuration_points':len(r['configurations']),'joint_frontier':r['joint_frontier']}))
