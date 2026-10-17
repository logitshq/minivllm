"""Standalone saved-row scatter. CPU Matplotlib Agg; no MLX/GPU imports."""
from pathlib import Path
import json
import sys
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt,font_manager
from matplotlib.lines import Line2D
D=Path(__file__).resolve().parent
r=json.loads((D/'reviewed_rows.json').read_text())
font=Path('/System/Library/Fonts/STHeiti Medium.ttc')
assert font.exists()
font_manager.fontManager.addfont(str(font));family=font_manager.FontProperties(fname=str(font)).get_name()
plt.rcParams.update({'font.family':family,'font.size':12,'axes.unicode_minus':False,'svg.fonttype':'path','svg.hashsalt':'native15-latency-20261017','axes.edgecolor':'#71808A','axes.labelcolor':'#263238','text.color':'#263238','xtick.color':'#53616B','ytick.color':'#53616B','figure.facecolor':'white','savefig.facecolor':'white'})
COLORS={'native':'#245A81','omlx':'#AF702B','lithos':'#7A597B'}
NAMES={'native':'本方 native15（旧）','omlx':'oMLX','lithos':'Lithos'}
SHAPES={1:'o',2:'s',4:'D'}
fig=plt.figure(figsize=(13.4,8.0));ax=fig.add_axes([.09,.29,.59,.48]);tab=fig.add_axes([.735,.29,.24,.48]);tab.axis('off')
fig.text(.045,.955,'旧 M4 完整对照：请求延迟与总吞吐',fontsize=21,fontweight='bold')
fig.text(.045,.899,'Qwen3.8-27B · NVFP4 g16/b4 · M4 Max 48 GiB · 19 输入 → 32 输出',fontsize=13)
colors=[Line2D([0],[0],marker='o',linestyle='none',color=c,markersize=8,label=NAMES[e]) for e,c in COLORS.items()]
fig.legend(handles=colors,loc='upper left',bbox_to_anchor=(.04,.875),ncol=3,frameon=False,columnspacing=2.1,handletextpad=.6,fontsize=11)
shapes=[Line2D([0],[0],marker=m,linestyle='none',markerfacecolor='white',markeredgecolor='#52606A',markersize=7,label=f'C{b}') for b,m in SHAPES.items()]
fig.legend(handles=shapes,loc='upper left',bbox_to_anchor=(.525,.875),ncol=3,frameon=False,columnspacing=1.1,handletextpad=.4,fontsize=11)
ax.spines[['top','right']].set_visible(False);ax.grid(color='#E7EBEF',linewidth=.8);ax.set_axisbelow(True)
ax.set_xlim(0,4300);ax.set_ylim(0,45);ax.set_xticks([0,1000,2000,3000,4000]);ax.set_yticks([0,10,20,30,40])
ax.set_xlabel('完整请求延迟 p50（ms，POST start → SSE DONE；越低越好）',labelpad=12,fontsize=11.2)
ax.set_ylabel('总输出吞吐（token/s；越高越好）',labelpad=12,fontsize=11.5)
for row in r['arms']:
    ax.scatter(row['latency_ms'],row['throughput_tok_s'],s=34,marker=SHAPES[row['concurrency']],facecolors='none',edgecolors=COLORS[row['engine']],alpha=.55,linewidths=1.1,zorder=2)
offsets={('native',1):(-79,14),('omlx',1):(17,20),('lithos',1):(-13,-28),('native',2):(14,14),('omlx',2):(18,-20),('native',4):(-95,15),('omlx',4):(18,-17)}
for row in r['configurations']:
    e,b=row['engine'],row['concurrency'];x,y=row['latency_ms'],row['throughput_tok_s']
    ax.scatter(x,y,s=133,marker=SHAPES[b],color=COLORS[e],edgecolors='white',linewidths=1.5,zorder=4)
    label=('native15' if e=='native' else NAMES[e])+f' C{b}'
    ax.annotate(label,(x,y),xytext=offsets[(e,b)],textcoords='offset points',fontsize=10.5,color=COLORS[e],fontweight='bold',ha='left',va='center',arrowprops={'arrowstyle':'-','color':COLORS[e],'lw':.7},zorder=5)
ax.text(.025,.965,'更低延迟 / 更高吞吐 ↖',transform=ax.transAxes,fontsize=10,color='#64717B',va='top')
tab.text(0,.98,'7 配置中位数',fontsize=14,fontweight='bold',va='top')
tab.text(0,.875,'配置',fontsize=10,color='#67737C');tab.text(.61,.875,'延迟 ms',ha='right',fontsize=10,color='#67737C');tab.text(1,.875,'总 token/s',ha='right',fontsize=10,color='#67737C')
tab.plot([0,1],[.842,.842],color='#E0E6EA',lw=1)
for i,row in enumerate(r['configurations']):
    y=.78-i*.082;e,b=row['engine'],row['concurrency'];label=('native15' if e=='native' else NAMES[e])+f' C{b}'
    tab.text(0,y,label,color=COLORS[e],fontsize=10.8,va='center');tab.text(.61,y,f"{row['latency_ms']:,.0f}",ha='right',fontsize=10.8,va='center');tab.text(1,y,f"{row['throughput_tok_s']:.2f}",ha='right',fontsize=10.8,va='center')
tab.text(0,.145,'Lithos C2 / C4：N/A',fontsize=11,fontweight='bold',color=COLORS['lithos'])
tab.text(0,.077,'HTTP 429 边界；没有零值点。',fontsize=10,color='#64717B')
tab.set_xlim(0,1);tab.set_ylim(0,1)
fig.text(.045,.18,'大点：7 配置中位数；空心小点：全部 21 个 fresh arm。只显示实测离散点，不连成可行前沿。',fontsize=11.5)
fig.text(.045,.124,'每 arm 3 warm + 3 measure groups；请求 p50 → 组中位 → 3 fresh-arm 中位。每组吞吐含排队、prefill 与 HTTP。',fontsize=10.5,color='#596771')
fig.text(.045,.075,'历史 native15 轮次含 thermal 0/1 混合与重复波动；无置信区间或显著性结论，不代表最新候选或完整前沿胜出。',fontsize=10.5,color='#596771')
fig.text(.045,.032,'证据：m4-final-native15-official-attempt2 · 270 HTTP / 135 measured / 4,320 measured tokens · 快照 2026-10-17',fontsize=10,color='#6C7780')
for ext in ('png','svg'):
    meta={'Creator':'CPU Matplotlib; replayed saved start-to-DONE HTTP timing'} if ext=='svg' else {'Software':'CPU Matplotlib'}
    if ext=='svg':meta['Date']=None
    fig.savefig(D/('native15-throughput-latency.'+ext),dpi=180,metadata=meta)
plt.close(fig)
assert not any(n=='mlx' or n.startswith('mlx.') for n in sys.modules)
(D/'render-metadata.json').write_text(json.dumps({'renderer':'Matplotlib','version':matplotlib.__version__,'backend':matplotlib.get_backend(),'python':sys.executable,'font':str(font),'font_family':family,'SVG_text':'glyph paths','small_arm_marks':len(r['arms']),'configuration_marks':len(r['configurations']),'GPU_used':False},ensure_ascii=False,indent=2)+'\n')
print('RENDERED: one saved-evidence scatter, PNG+SVG, 21 small arm marks and7 configuration medians.')
