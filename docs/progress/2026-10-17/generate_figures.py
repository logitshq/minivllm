"""Render reviewed_rows.json to standalone PNG+SVG; CPU Matplotlib Agg only."""
from pathlib import Path
import argparse, json
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt, font_manager
from matplotlib.lines import Line2D

HERE=Path(__file__).resolve().parent
ap=argparse.ArgumentParser()
ap.add_argument('--rows',type=Path,default=HERE/'reviewed_rows.json')
ap.add_argument('--out',type=Path,default=HERE)
ap.add_argument('--font',type=Path)
a=ap.parse_args();d=json.loads(a.rows.read_text());a.out.mkdir(parents=True,exist_ok=True)
candidates=[a.font,Path('/System/Library/Fonts/STHeiti Medium.ttc'),Path('/System/Library/Fonts/PingFang.ttc'),Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')]
font=next((x for x in candidates if x and x.exists()),None)
if font is None:raise SystemExit('A Chinese font is required; supply --font PATH (no silent missing-glyph fallback).')
font_manager.fontManager.addfont(str(font));family=font_manager.FontProperties(fname=str(font)).get_name()
plt.rcParams.update({'font.family':family,'font.size':12,'axes.unicode_minus':False,'svg.fonttype':'path','svg.hashsalt':'progress-20261017','axes.edgecolor':'#747B84','axes.labelcolor':'#252B32','text.color':'#252B32','xtick.color':'#505964','ytick.color':'#505964','figure.facecolor':'white','savefig.facecolor':'white'})
BLUE='#245A81';GOLD='#AF702B';LIGHT='#D9DFE5';CHAR='#252B32'
def export(fig,name):
    for ext in ('png','svg'):
        metadata={'Creator':'CPU Matplotlib; reviewed saved benchmark rows'} if ext=='svg' else {'Software':'CPU Matplotlib'}
        if ext=='svg':metadata['Date']=None
        fig.savefig(a.out/(name+'.'+ext),dpi=180,metadata=metadata)
    plt.close(fig)
def clean(ax):
    ax.spines[['top','right']].set_visible(False);ax.grid(axis='x',color='#E7EBEF',linewidth=.8);ax.set_axisbelow(True)

fig,axs=plt.subplots(1,2,figsize=(12.2,6.1),sharex=True,sharey=True)
fig.subplots_adjust(left=.15,right=.97,top=.73,bottom=.25,wspace=.16)
fig.suptitle('C1 HTTP 配对测量：普通 none 与外部 MTP',x=.06,y=.96,ha='left',fontsize=20,fontweight='bold')
fig.text(.06,.888,'M4 Max 48 GiB · 19输入 / 32输出 · 3对fresh服务 · 每点为3次测量中位数',fontsize=12)
legend=[Line2D([0],[0],marker='o',color='none',markerfacecolor=BLUE,markeredgecolor=BLUE,markersize=8,label='none（普通官方）'),Line2D([0],[0],marker='s',color='none',markerfacecolor='white',markeredgecolor=GOLD,markeredgewidth=1.8,markersize=8,label='外部 MTP（block=3）')]
fig.legend(handles=legend,loc='upper left',bbox_to_anchor=(.055,.868),ncol=2,frameon=False,columnspacing=2)
ys=[2,1,0]
for ax,key,title in zip(axs,['x','y'],['x：首文本块后交互速率（客户端估计）','y：整请求输出吞吐']):
    clean(ax);ax.set_xlim(0,33);ax.set_xticks([0,10,20,30]);ax.set_ylim(-.5,2.5);ax.set_title(title,fontsize=13,pad=20,loc='left');ax.set_xlabel('token/s（越高越好）',labelpad=10)
    ax.set_yticks(ys,['P1  none→MTP','P2  MTP→none','P3  none→MTP'])
    for y,p in zip(ys,d['http']['pairs']):
        n,m=p['none'][key],p['mtp'][key];ax.plot([m,n],[y,y],color=LIGHT,linewidth=2,zorder=1)
        ax.scatter([n],[y],s=65,color=BLUE,zorder=3);ax.scatter([m],[y],s=65,marker='s',facecolors='white',edgecolors=GOLD,linewidths=1.8,zorder=3)
        ax.annotate(f'{n:.2f}',(n,y),xytext=(0,12),textcoords='offset points',ha='center',color=BLUE,fontsize=11)
        ax.annotate(f'{m:.2f}',(m,y),xytext=(0,-22),textcoords='offset points',ha='center',color=GOLD,fontsize=11)
        ax.text((n+m)/2,y+.04,f"{p[key+'_signed_MTP_pct']:.1f}%",ha='center',va='bottom',fontsize=10,color='#626C77')
fig.text(.06,.13,'3对均为 MTP 较慢。连线仅表示配对；标注为 100×(MTP/none−1)，未构造置信区间。',fontsize=11)
fig.text(.06,.076,'仅此C1配置筛查；不混入手搓引擎，不代表完整前沿。原始计时API无raw IDs，保存visible32；资格另见V109。',fontsize=10,color='#626C77')
fig.text(.06,.028,'V111 · 36HTTP / 18计时 / 576token · AC/thermal0/pressure1（有分页） · 快照2026-10-17',fontsize=10,color='#626C77')
export(fig,'c1-none-vs-mtp-paired')

fig,ax=plt.subplots(figsize=(12.2,5.8));fig.subplots_adjust(left=.31,right=.96,top=.72,bottom=.25)
fig.suptitle('同输入 Linear 局部数值：497项输出逐字节对照',x=.06,y=.96,ha='left',fontsize=20,fontweight='bold')
fig.text(.06,.885,'V112 与 V113 · 原 first-verify block 的同497个输入 · 不是整模运行次数',fontsize=12)
for y,row in zip([1,0],d['local_linear']['rows']):
    exact,different=row['exact'],row['different'];ax.barh(y,exact,height=.43,color=BLUE)
    if different:ax.barh(y,different,left=exact,height=.43,color=GOLD,hatch='///',edgecolor='white',linewidth=0)
    if exact:ax.text(exact/2,y,f'{exact} exact',va='center',ha='center',color='white',fontweight='bold',fontsize=13)
    if different:ax.text(exact+different/2,y,f'{different} 不同',va='center',ha='center',color='white',fontweight='bold',fontsize=13,bbox={'facecolor':GOLD,'edgecolor':'none','pad':3})
    else:ax.text(497,y+.34,'0项不同',ha='right',fontsize=11,color=BLUE)
clean(ax);ax.grid(axis='x',visible=False);ax.set_xlim(0,497);ax.set_xticks([0,100,200,300,400,497]);ax.set_ylim(-.6,1.6);ax.set_xlabel('Linear 输出对照项数（总计497）',labelpad=12)
ax.set_yticks([1,0],['V112  batch M=T\n对照 serial M=1','V113  broadcast batch=T, M=1\n对照同一 serial M=1'])
fig.text(.06,.13,'局部497项已全部exact；完整序列、缓存及整模资格仍需独立验证。',fontsize=12,fontweight='bold')
fig.text(.06,.075,'此图不表示速度提升。V114整合因资源压力停止：A96旧IDs通过，B未到达；尚无完整修复结论。',fontsize=10.5,color='#626C77')
fig.text(.06,.028,'保存证据 V112 / V113 · 严格字节比较，不设数值容差 · 快照2026-10-17',fontsize=10,color='#626C77')
export(fig,'local-linear-exactness')
(a.out/'render-metadata.json').write_text(json.dumps({'backend':matplotlib.get_backend(),'matplotlib':matplotlib.__version__,'font_family':family,'font_path':str(font),'SVG_text':'glyph paths for portable Chinese','GPU_used':False},ensure_ascii=False,indent=2)+'\n')
print('RENDERED: two figures, PNG + SVG')
