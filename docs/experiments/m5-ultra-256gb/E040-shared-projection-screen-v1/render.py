"""Standalone static figure of independently recomputed E040 timing ratios."""
import hashlib,json,os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/minivllm-plot-cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
D=Path(__file__).resolve().parent;raw=(D/'independent-audit.json').read_bytes();a=json.loads(raw)
assert a['status']=='PASS_OFFLINE_AUDIT' and a['raw_timing_samples']==2400
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','svg.hashsalt':'E040-v1'})
fig,axs=plt.subplots(1,2,figsize=(12.5,7),sharey=True)
fig.subplots_adjust(left=.095,right=.97,top=.72,bottom=.27,wspace=.16)
styles={'native_b1':('#286a9c','o',-.16,'Native MLX · separate M1'),'custom_b1':('#b26c20','s',.16,'Custom kernel · separate B1')}
for ax,projection,title in zip(axs,('head','up'),('Vocabulary head · N=248,320','Layer 0 up projection · N=17,408')):
 ax.set_title(title,loc='left',fontsize=13,pad=12,color='#253449')
 ax.axhline(1,color='#475569',ls='--',lw=1.3)
 ax.text(2.45,1.008,'parity',ha='right',fontsize=9,color='#475569')
 for baseline,(color,marker,dodge,label) in styles.items():
  for index,T in enumerate((2,4,8)):
   row=next(r for r in a['rows'] if r['projection']==projection and r['T']==T and r['baseline']==baseline)
   ys=[g['ratio_baseline_over_candidate'] for g in row['groups']]
   xs=[index+dodge+(i-2)*.027 for i in range(5)]
   ax.scatter(xs,ys,s=34,marker=marker,facecolors='none',edgecolors=color,linewidths=1,alpha=.85,zorder=3)
   median=row['median_group_ratio'];ax.scatter([index+dodge],[median],s=80,marker=marker,c=color,edgecolors='white',linewidths=.9,zorder=4)
   ax.text(index+dodge,min(ys)-.027,f'{median:.3f}×',ha='center',fontsize=10,color=color)
 ax.set_xticks([0,1,2],['T2','T4','T8']);ax.set_xlabel('Rows per projection call (T)',labelpad=10)
 ax.set_xlim(-.45,2.45);ax.set_ylim(.80,1.45);ax.set_yticks([.80,.90,1.,1.10,1.20,1.30,1.40],['0.80×','0.90×','1.00×','1.10×','1.20×','1.30×','1.40×'])
 ax.grid(axis='y',color='#e2e8f0',lw=.7);ax.set_axisbelow(True)
axs[0].set_ylabel('Speed ratio = baseline latency / candidate latency',labelpad=12)
legend=[Line2D([0],[0],marker=m,color=c,linestyle='none',markersize=7,label=l) for c,m,_,l in styles.values()]
fig.legend(handles=legend,loc='upper left',bbox_to_anchor=(.087,.83),frameon=False,ncol=2,fontsize=11)
fig.text(.095,.95,'E040 · Shared-weight projection — not engine performance',fontsize=18,weight='bold',color='#172536')
fig.text(.095,.902,'M5 Ultra · real Qwen3.8-27B NVFP4 weights · K=5,120 · synthetic BF16 inputs · 2026-10-19',fontsize=11,color='#475569')
fig.text(.095,.132,'Small marks: all five sequential groups. Large marks: median of the five group ratios. Focused ratio scale.',fontsize=10,color='#475569')
fig.text(.095,.093,'Each group: 20 paired AB/BA samples per baseline; fresh operations + eval + synchronization + concat/staging.',fontsize=10,color='#475569')
fig.text(.095,.054,'Above 1 favors the candidate. Head T4/T8 beat both local baselines; other cases are mixed.',fontsize=11,color='#253449')
artifacts={}
for ext in ('png','svg'):
 p=D/f'local-projection-speed.{ext}';assert not p.exists();fig.savefig(p,dpi=170,metadata={'Date':None} if ext=='svg' else None)
 artifacts[p.name]=dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
plt.close(fig)
r=dict(status='RENDERED',visual_review='PENDING',independent_audit_sha256=hashlib.sha256(raw).hexdigest(),render_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),artifacts=artifacts)
(D/'render-receipt.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
