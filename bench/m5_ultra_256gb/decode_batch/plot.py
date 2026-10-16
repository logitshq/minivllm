"""Static scientific figure from independently replayed measurements."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
d=json.loads((a.out/'m5-ultra-pareto-results.json').read_text());points=d['points'];raw=d['all_processes']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,
                    'axes.spines.right':False,'figure.facecolor':'#FCFCFB','axes.facecolor':'#FCFCFB',
                    'text.color':'#20272D','axes.labelcolor':'#303B43','xtick.color':'#52606A','ytick.color':'#52606A'})
colors={'fused':'#2368A0','fused_capped':'#2368A0','omlx':'#D66B28','omlx_aggressive':'#D66B28','lithos':'#97812C'}
markers={'fused':'o','fused_capped':'o','omlx':'s','omlx_aggressive':'^','lithos':'D'}
labels={'fused':'minivllm · batch fusion','omlx':'oMLX · balanced','omlx_aggressive':'oMLX · aggressive C1','lithos':'Lithos · C1 only'}
if any(p['engine']=='fused_retired' for p in points):
    colors['fused_retired']='#2368A0';markers['fused_retired']='*'
    labels['fused_retired']='minivllm · + batch retirement'
fig=plt.figure(figsize=(12.8,9.2));ax=fig.add_axes((.09,.29,.86,.49))
for engine in colors:
    ps=[p for p in points if p['engine']==engine];rs=[r for r in raw if r['engine']==engine]
    ax.scatter([r['x'] for r in rs],[r['y'] for r in rs],s=23,marker=markers[engine],color=colors[engine],alpha=.24,zorder=2)
    for p in ps:
        ax.scatter(p['x'],p['y'],s=73,marker=markers[engine],facecolor=colors[engine] if p['own_frontier'] else '#FCFCFB',edgecolor=colors[engine],linewidth=1.5,zorder=4)
        if engine in ('lithos','omlx_aggressive'):continue
        offset=(7,10) if engine.startswith('fused') else (5,-17)
        if engine=='omlx' and p['C'] in (16,32,64):offset={16:(7,10),32:(7,-17),64:(-18,-24)}[p['C']]
        if engine=='fused' and p['C']==16:offset=(7,-17)
        if engine=='fused' and p['C']==32:offset=(8,-13)
        if engine=='fused_capped':offset=(-48,-20)
        if engine=='fused_retired':offset=(7,-16) if p['C']==8 else (7,12)
        label='C64/B32' if engine=='fused_capped' else 'C'+str(p['C'])+(' R' if engine=='fused_retired' else '')
        ax.annotate(label,(p['x'],p['y']),xytext=offset,textcoords='offset points',color=colors[engine],fontsize=10)
for family in ('fused','omlx','lithos'):
    frontier=sorted([p for p in points if p['family']==family and p['own_frontier']],key=lambda p:p['x'])
    ax.plot([p['x'] for p in frontier],[p['y'] for p in frontier],color=colors[family],lw=1.6,zorder=2)
ax.set(xlim=(0,max(p['x'] for p in points)*1.08),ylim=(0,max(220,max(p['y'] for p in points)*1.14)),
       xlabel='Client text-delivery proxy (tokens/s/user)',ylabel='Complete-request throughput (tokens/s/Mac)')
ax.grid(color='#D9DFE2',lw=.6,alpha=.75);ax.set_axisbelow(True)
ax.text(.54,.96,'Higher is better ↗',ha='right',va='top',transform=ax.transAxes,fontsize=11,color='#52606A')
handles=[Line2D([0],[0],color=colors[e],marker=markers[e],lw=1.5,markersize=7,label=labels[e]) for e in labels]
fig.legend(handles=handles,loc='upper left',bbox_to_anchor=(.084,.848),frameon=False,ncol=len(labels),fontsize=9)
fig.text(.09,.945,'M5 Ultra · measured Pareto frontier',fontsize=22,weight='bold')
fig.text(.09,.902,'256 GiB · Qwen3.8-27B NVFP4 · 19 input / 32 output tokens · target-only decoding',fontsize=12,color='#52606A')
covered=sum(bool(c['covered_by_native_C']) for c in d['coverage'])
fig.text(.09,.871,f'{covered}/{len(d["coverage"])} external configuration medians covered by an actual minivllm point · C1–C64 sweep',fontsize=11.5,color='#303B43')
ix=ax.inset_axes((.64,.72,.32,.23));c1=[p for p in points if p['C']==1]
for e in labels:
    if e=='fused_retired':continue
    p=next(p for p in c1 if p['engine']==e);rs=[r for r in raw if r['C']==1 and r['engine']==e]
    ix.scatter([r['x'] for r in rs],[r['y'] for r in rs],s=13,color=colors[e],marker=markers[e],alpha=.3)
    ix.scatter(p['x'],p['y'],s=39,color=colors[e],marker=markers[e])
ix.set_xlim(min(p['process_range']['x'][0] for p in c1)-1,max(p['process_range']['x'][1] for p in c1)+1)
ix.set_ylim(min(p['process_range']['y'][0] for p in c1)-1,max(p['process_range']['y'][1] for p in c1)+1)
ix.set_title('C1 detail · same units',fontsize=10,pad=6);ix.tick_params(labelsize=8);ix.grid(alpha=.2)
notes=('Large marks: median of three process medians (three measured groups per process). Faint marks retain all processes.\n'
       'Filled marks: frontier within each engine (native versions and oMLX modes combined). Hollow marks: dominated configurations.\n'
       'Connecting lines are visual guides; intermediate configurations were not measured. Process spread is not a confidence interval.\n'
       'Greedy, normal EOS, no prefix cache or drafter. Three warm groups precede measurement; all measured time retained.\n'
       'Lithos uses a local M5 Ultra profile; its official HTTP endpoint supports C1 only. x is a delivery proxy, not raw GPU speed.\n'
       'C64/B32 caps active sequences at32; R retires finished rows once per step. TTFT and all results are retained in the report.')
fig.text(.09,.057,notes,fontsize=9.4,color='#52606A',linespacing=1.6)
fig.savefig(a.out/'m5-ultra-pareto.png',dpi=180)
fig.savefig(a.out/'m5-ultra-pareto.pdf')
print(a.out/'m5-ultra-pareto.png')
