"""Nine observed configurations from two saved AC campaigns. CPU/Agg only."""
from pathlib import Path
import hashlib,json,sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
D=Path(__file__).resolve().parent
r=json.loads((D/'plot-data.json').read_text())
colors={'native':'#245A81','omlx':'#AF702B','lithos':'#7A597B'}
markers={1:'o',2:'s',4:'^'};names={'native':'native','omlx':'oMLX','lithos':'Lithos'}
assert len(r['configurations'])==9 and len(r['arms'])==27
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#9CA5AC','axes.labelcolor':'#283640','xtick.color':'#40505B','ytick.color':'#40505B','svg.fonttype':'none','svg.hashsalt':'v130-queued-two-campaigns'})
fig=plt.figure(figsize=(13.8,8.8),facecolor='white')
ax=fig.add_axes([.082,.292,.88,.51]);ax.set_axisbelow(True);ax.grid(color='#E8ECEF',linewidth=.7)
for e in colors:
 q=sorted([a for a in r['configurations']if a['engine']==e],key=lambda a:a['concurrency'])
 ax.plot([a['latency_ms']for a in q],[a['throughput_tok_s']for a in q],color=colors[e],lw=1.2,ls=(0,(4,3)),alpha=.7,zorder=1)
f=sorted([a for a in r['configurations']if a['joint_non_dominated']],key=lambda a:a['latency_ms'])
ax.plot([a['latency_ms']for a in f],[a['throughput_tok_s']for a in f],color='#253139',lw=1.8,zorder=2)
def marks(a,axis,small=False):
 c=colors[a['engine']];m=markers[a['concurrency']]
 axis.scatter(a['latency_ms'],a['throughput_tok_s'],s=38 if small else(145 if m!='^'else 174),marker=m,facecolors='none'if small or not a['joint_non_dominated']else c,edgecolors=c if small or not a['joint_non_dominated']else'#253139',linewidths=1.0 if small else 1.7,alpha=.66 if small else 1,zorder=3 if small else 5)
# Arm observations at their actual coordinates; never jittered or converted to CIs.
for a in r['arms']:marks({**a,'joint_non_dominated':False},ax,True)
for a in r['configurations']:marks(a,ax)
offsets={'native:C2':(-3,25,'left'),'omlx:C2':(30,-30,'left'),'native:C4':(-15,24,'right'),'omlx:C4':(-15,-29,'right'),'lithos:C2':(16,27,'left'),'lithos:C4':(-8,28,'right')}
for a in r['configurations']:
 if a['concurrency']==1:continue
 dx,dy,ha=offsets[a['config']];label=f"{names[a['engine']]} C{a['concurrency']} · {a['campaign']}\n{a['latency_ms']:,.1f} ms · {a['throughput_tok_s']:.3f} tok/s"
 ax.annotate(label,(a['latency_ms'],a['throughput_tok_s']),xytext=(dx,dy),textcoords='offset points',ha=ha,va='center',fontsize=9.6,color=colors[a['engine']],zorder=6,bbox={'facecolor':'white','edgecolor':'none','pad':1.8,'alpha':.95},arrowprops={'arrowstyle':'-','color':colors[a['engine']],'lw':.75,'shrinkA':2,'shrinkB':6})
ax.add_patch(Rectangle((1175,19.2),420,9.7,facecolor='none',edgecolor='#B6BFC6',lw=.8,ls=(0,(3,3)),zorder=1))
ax.text(1212,18.2,'C1: enlarged at upper right',fontsize=8.7,color='#687780')
ax.set(xlim=(1130,4080),ylim=(16,69),xticks=[1200,1600,2000,2400,2800,3200,3600,4000],yticks=[20,30,40,50,60],xlabel='Full HTTP request latency p50 (ms)  ← lower is better',ylabel='Total output throughput (tokens/s)  → higher is better')
ax.xaxis.labelpad=12;ax.yaxis.labelpad=12
z=fig.add_axes([.568,.493,.362,.274],facecolor='white');z.set_axisbelow(True);z.grid(color='#EDF0F3',lw=.6)
for sp in z.spines.values():sp.set_visible(True);sp.set_color('#BAC4CC');sp.set_linewidth(.8)
for a in r['arms']:
 if a['concurrency']==1:marks({**a,'joint_non_dominated':False},z,True)
for a in r['configurations']:
 if a['concurrency']!=1:continue
 marks(a,z);dx,dy,ha={'native':(8,21,'left'),'omlx':(-12,-29,'left'),'lithos':(-7,35,'right')}[a['engine']]
 z.annotate(f"{names[a['engine']]} C1 · {a['campaign']}\n{a['latency_ms']:,.1f} ms · {a['throughput_tok_s']:.3f}",(a['latency_ms'],a['throughput_tok_s']),xytext=(dx,dy),textcoords='offset points',ha=ha,va='center',fontsize=8.8,color=colors[a['engine']],bbox={'facecolor':'white','edgecolor':'none','pad':1.4,'alpha':.95},arrowprops={'arrowstyle':'-','color':colors[a['engine']],'lw':.7,'shrinkA':2,'shrinkB':6},zorder=6)
z.set(xlim=(1180,1600),ylim=(19.0,29.3),xticks=[1200,1400,1600],yticks=[20,24,28]);z.tick_params(labelsize=8.5);z.set_title('C1 detail · different scale (ms; tokens/s)',loc='left',fontsize=10,color='#40505B',pad=7)
fig.text(.082,.957,'M4 Max · throughput vs full-request latency',fontsize=20,weight='bold',color='#23333E')
fig.text(.082,.911,'Qwen3.8-27B · NVFP4 · 48 GiB · 19 input / 32 greedy output · C1 / C2 / C4 · 9 configurations',fontsize=11.4,color='#5F6E78')
fig.text(.082,.875,'27 fresh loads across TWO separate AC campaigns: V128 native/oMLX + V130 official queued Lithos',fontsize=10.4,color='#5F6E78')
handles=[Line2D([],[],color=colors[e],marker='o',mfc=colors[e]if e=='native'else'white',mec=colors[e],ls='none',markersize=7,label=names[e])for e in colors]
handles+=[Line2D([],[],color='#253139',lw=1.8,label='Descriptive joint non-dominated points')]
fig.legend(handles=handles,loc='upper left',bbox_to_anchor=(.076,.855),frameon=False,ncol=4,fontsize=9.4,columnspacing=1.8,handlelength=1.8)
shapes=[Line2D([],[],color='#596670',marker=markers[c],mfc='white',ls='none',markersize=7,label=f'C{c}')for c in (1,2,4)]
shapes+=[Line2D([],[],color='#596670',marker='o',mfc='white',ls='none',markersize=4,label='Small: fresh arm'),Line2D([],[],color='#596670',marker='o',mfc='#596670',ls='none',markersize=8,label='Large: median of 3 fresh arms')]
fig.legend(handles=shapes,loc='lower left',bbox_to_anchor=(.076,.182),frameon=False,ncol=5,fontsize=9.3,handletextpad=.6,columnspacing=1.9)
notes=[
'V128 native/oMLX used HTTP/SSE; V130 Lithos used official Ollama NDJSON. Independent campaigns: descriptive comparison, not paired.',
'Native C1 = CAP64/direct ICB; native C2/C4 = accepted25 batching/CAP256. Lithos = one generation at a time with FIFO queuing.',
'Latency includes server queuing. Per-group request p50 → median of 3 formal groups → median of 3 fresh arms; throughput uses the same hierarchy.',
'Filled large marks are non-dominated within these 9 points. Guides connect observed C levels only; no interpolated feasibility or statistical significance claim.'
]
for y,t in zip([.161,.130,.099,.068],notes):fig.text(.082,y,t,fontsize=8.8,color='#5F6E78')
fig.savefig(D/'lithos-queued-throughput-latency.png',dpi=190,metadata={'Software':'Matplotlib; saved V128 + V130 independent campaign rows only'})
fig.savefig(D/'lithos-queued-throughput-latency.svg',metadata={'Date':None,'Creator':'Matplotlib; saved V128 + V130 independent campaign rows only'})
metadata={'renderer':'Matplotlib','version':matplotlib.__version__,'backend':matplotlib.get_backend(),'python_version':sys.version.split()[0],'font':'DejaVu Sans','unique_configuration_marks':9,'unique_small_fresh_arm_marks':27,'C1_detail_duplicates':3,'independent_campaigns':['V128','V130'],'descriptive_frontier':r['joint_frontier_descriptive'],'GPU_used':False,'data_sha256':hashlib.sha256((D/'plot-data.json').read_bytes()).hexdigest()}
(D/'render-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n');print(json.dumps(metadata))
