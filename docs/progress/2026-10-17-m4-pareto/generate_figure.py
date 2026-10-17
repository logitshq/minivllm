"""Standalone saved-row scatter. CPU Matplotlib Agg; no MLX/GPU imports."""
from pathlib import Path
import json
import sys
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt,font_manager
from matplotlib.lines import Line2D
D=Path(__file__).resolve().parent
r=json.loads((D/'pareto.json').read_text())
def dominates(a,b):
    return a['latency_ms']<=b['latency_ms'] and a['throughput_tok_s']>=b['throughput_tok_s'] and (a['latency_ms']<b['latency_ms'] or a['throughput_tok_s']>b['throughput_tok_s'])
def frontier(rows):
    return [v['point_id']for v in sorted(rows,key=lambda v:v['latency_ms']) if not any(dominates(w,v)for w in rows)]
assert frontier(r['configurations'])==r['joint_frontier']==['omlx-C1','native-C4']
for engine,ids in r['per_engine_frontiers'].items():
    assert frontier([v for v in r['configurations']if v['engine']==engine])==ids
for v in r['configurations']:
    assert v['joint_dominated_by']==[w['point_id']for w in r['configurations']if dominates(w,v)]

font=Path('/System/Library/Fonts/STHeiti Medium.ttc')
assert font.exists()
font_manager.fontManager.addfont(str(font));family=font_manager.FontProperties(fname=str(font)).get_name()
plt.rcParams.update({'font.family':family,'font.size':12,'axes.unicode_minus':False,'svg.fonttype':'path','svg.hashsalt':'native15-pareto-20261017','axes.edgecolor':'#71808A','axes.labelcolor':'#263238','text.color':'#263238','xtick.color':'#53616B','ytick.color':'#53616B','figure.facecolor':'white','savefig.facecolor':'white'})
COLORS={'native':'#245A81','omlx':'#AF702B','lithos':'#7A597B'}
NAMES={'native':'本方 native15（旧）','omlx':'oMLX','lithos':'Lithos'}
SHAPES={1:'o',2:'s',4:'D'}
fig=plt.figure(figsize=(13.4,8.0));ax=fig.add_axes([.09,.29,.59,.48]);tab=fig.add_axes([.735,.29,.24,.48]);tab.axis('off')
fig.text(.045,.955,'M4 Throughput–Latency Pareto（旧 native15 轮次）',fontsize=21,fontweight='bold')
fig.text(.045,.899,'Qwen3.8-27B · NVFP4 g16/b4 · M4 Max 48 GiB · 19 输入 → 32 输出',fontsize=13)
colors=[Line2D([0],[0],marker='o',linestyle='none',color=c,markersize=8,label=NAMES[e]) for e,c in COLORS.items()]
fig.legend(handles=colors,loc='upper left',bbox_to_anchor=(.04,.875),ncol=3,frameon=False,columnspacing=2.1,handletextpad=.6,fontsize=11)
shapes=[Line2D([0],[0],marker=m,linestyle='none',markerfacecolor='white',markeredgecolor='#52606A',markersize=7,label=f'C{b}') for b,m in SHAPES.items()]
fig.legend(handles=shapes,loc='upper left',bbox_to_anchor=(.525,.875),ncol=3,frameon=False,columnspacing=1.1,handletextpad=.4,fontsize=11)
ax.spines[['top','right']].set_visible(False);ax.grid(color='#E7EBEF',linewidth=.8);ax.set_axisbelow(True)
ax.set_xlim(0,4300);ax.set_ylim(0,45);ax.set_xticks([0,1000,2000,3000,4000]);ax.set_yticks([0,10,20,30,40])
ax.set_xlabel('完整请求延迟 p50（ms，POST start → SSE DONE；越低越好）',labelpad=12,fontsize=11.2)
ax.set_ylabel('总输出吞吐（token/s；越高越好）',labelpad=12,fontsize=11.5)
by_id={row['point_id']:row for row in r['configurations']}
for ids in r['per_engine_frontiers'].values():
    if len(ids)>1:
        rs=[by_id[i]for i in ids]
        ax.plot([v['latency_ms']for v in rs],[v['throughput_tok_s']for v in rs],color=COLORS[rs[0]['engine']],lw=1.1,alpha=.65,zorder=1)
rs=[by_id[i]for i in r['joint_frontier']]
ax.plot([v['latency_ms']for v in rs],[v['throughput_tok_s']for v in rs],color='#263238',lw=2.3,linestyle=(0,(5,3)),zorder=2.5)
ax.text(1980,32.6,'联合非支配前沿',color='#263238',fontsize=11.5,rotation=25,ha='left')
frontier_legend=[Line2D([0],[0],color='#263238',lw=2.3,ls='--',label='联合 Pareto 前沿'),Line2D([0],[0],color='#7D8B94',lw=1.1,label='各家已测前沿（连线非插值）')]
fig.legend(handles=frontier_legend,loc='upper left',bbox_to_anchor=(.04,.822),ncol=2,frameon=False,fontsize=10.5,columnspacing=2)
for row in r['arms']:
    ax.scatter(row['latency_ms'],row['throughput_tok_s'],s=34,marker=SHAPES[row['concurrency']],facecolors='none',edgecolors=COLORS[row['engine']],alpha=.55,linewidths=1.1,zorder=2)
offsets={('native',1):(-79,14),('omlx',1):(25,-4),('lithos',1):(-13,-28),('native',2):(14,14),('omlx',2):(18,-20),('native',4):(-95,15),('omlx',4):(18,-17)}
for row in r['configurations']:
    e,b=row['engine'],row['concurrency'];x,y=row['latency_ms'],row['throughput_tok_s']
    ax.scatter(x,y,s=155 if row['joint_non_dominated'] else 126,marker=SHAPES[b],facecolors=COLORS[e] if row['joint_non_dominated'] else 'white',edgecolors='#263238' if row['joint_non_dominated'] else COLORS[e],linewidths=1.6,zorder=4)
    label=('native15' if e=='native' else NAMES[e])+f' C{b}'
    if row['joint_non_dominated']:label+='（联合）'
    if e=='native' and b==2:label+='\n被自家 C1 支配'
    ax.annotate(label,(x,y),xytext=offsets[(e,b)],textcoords='offset points',fontsize=10.5,color=COLORS[e],fontweight='bold',ha='left',va='center',arrowprops={'arrowstyle':'-','color':COLORS[e],'lw':.7},zorder=5)
ax.text(.025,.965,'更低延迟 / 更高吞吐 ↖',transform=ax.transAxes,fontsize=10,color='#64717B',va='top')
tab.text(0,.98,'7 配置中位数 / Pareto',fontsize=14,fontweight='bold',va='top')
tab.text(0,.875,'配置',fontsize=10,color='#67737C');tab.text(.61,.875,'延迟 ms',ha='right',fontsize=10,color='#67737C');tab.text(1,.875,'总 token/s',ha='right',fontsize=10,color='#67737C')
tab.plot([0,1],[.842,.842],color='#E0E6EA',lw=1)
for i,row in enumerate(r['configurations']):
    y=.78-i*.082;e,b=row['engine'],row['concurrency'];label=('native15' if e=='native' else NAMES[e])+f' C{b}'
    tab.text(0,y,label,color=COLORS[e],fontsize=10.8,va='center');tab.text(.61,y,f"{row['latency_ms']:,.0f}",ha='right',fontsize=10.8,va='center');tab.text(1,y,f"{row['throughput_tok_s']:.2f}",ha='right',fontsize=10.8,va='center')
tab.text(0,.145,'Lithos C2 / C4：N/A',fontsize=11,fontweight='bold',color=COLORS['lithos'])
tab.text(0,.077,'HTTP 429 边界；没有零值点。',fontsize=10,color='#64717B')
tab.set_xlim(0,1);tab.set_ylim(0,1)
fig.text(.045,.18,'实心大点：联合非支配；空心大点：联合被支配；小点：全部 21 个 fresh arm。',fontsize=11.5)
fig.text(.045,.124,'粗虚线连接联合前沿的两个实测配置；细线为各家前沿。连接不代表插值、可行的新配置或统计显著性。',fontsize=10.5,color='#596771')
fig.text(.045,.075,'正确性为准入条件；最新 spec 短请求未获性能收益，新旧轮次不混合；旧轮次有 thermal 0/1 混合与重复波动。',fontsize=10.5,color='#596771')
fig.text(.045,.032,'证据：m4-final-native15-official-attempt2 · 270 HTTP / 135 measured / 4,320 measured tokens · 快照 2026-10-17',fontsize=10,color='#6C7780')
for ext in ('png','svg'):
    meta={'Creator':'CPU Matplotlib; replayed saved start-to-DONE HTTP timing'} if ext=='svg' else {'Software':'CPU Matplotlib'}
    if ext=='svg':meta['Date']=None
    fig.savefig(D/('m4-throughput-latency-pareto.'+ext),dpi=180,metadata=meta)
plt.close(fig)
assert not any(n=='mlx' or n.startswith('mlx.') for n in sys.modules)
(D/'render-metadata.json').write_text(json.dumps({'renderer':'Matplotlib','version':matplotlib.__version__,'backend':matplotlib.get_backend(),'python':sys.executable,'font':str(font),'font_family':family,'SVG_text':'glyph paths','small_arm_marks':len(r['arms']),'configuration_marks':len(r['configurations']),'joint_frontier':r['joint_frontier'],'per_engine_frontiers':r['per_engine_frontiers'],'GPU_used':False},ensure_ascii=False,indent=2)+'\n')
print('RENDERED: measured Pareto scatter PNG+SVG, 21arms,7configurations,2joint-frontier endpoints.')
