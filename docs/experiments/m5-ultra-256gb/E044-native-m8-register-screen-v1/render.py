"""Two separate native-M8 local screens; preserve E044 parent FAIL on the figure."""
import hashlib,json,os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/minivllm-plot-cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
D=Path(__file__).resolve().parent;refs=json.loads((D/'chart-sources.json').read_text());data=[]
for ref in refs:
 p=D/ref['copy'];raw=p.read_bytes();assert hashlib.sha256(raw).hexdigest()==ref['sha256'];a=json.loads(raw)
 assert a['status']=='PASS_OFFLINE_AUDIT' and a['raw_timing_samples']==400 and len(a['rows'])==2
 if ref['experiment']=='E044':assert a['original_parent_status']=='FAIL' and a['original_lease_returncode']==-2 and a['original_lease_closed'] is False
 for row in a['rows']:
  assert row['T']==8 and row['K']==5120 and row['baseline']=='native_m8' and len(row['groups'])==5
  data.append(dict(experiment=ref['experiment'],**row))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','svg.hashsalt':'E043-E044-v1'})
fig,axs=plt.subplots(1,2,figsize=(12.8,7),sharey=True);fig.subplots_adjust(left=.095,right=.97,top=.75,bottom=.29,wspace=.18)
styles=[('E043','#286a9c','o'),('E044','#b26c20','D')]
for ax,projection,title in zip(axs,('head','up'),('Vocabulary head · N=248,320','Layer 0 up projection · N=17,408')):
 ax.set_title(title,loc='left',fontsize=13,pad=14,color='#253449');ax.axhline(1,color='#475569',ls='--',lw=1.4);ax.text(1.48,1.012,'native M8 parity',ha='right',fontsize=10,color='#475569')
 for x,(experiment,color,marker) in enumerate(styles):
  row=next(r for r in data if r['projection']==projection and r['experiment']==experiment);ys=[g['ratio_baseline_over_candidate'] for g in row['groups']]
  ax.scatter([x+(i-2)*.045 for i in range(5)],ys,s=45,marker=marker,facecolors='none',edgecolors=color,linewidths=1.2,zorder=3)
  median=row['median_group_ratio'];ax.scatter([x],[median],s=95,marker=marker,color=color,edgecolors='white',linewidths=1,zorder=4)
  ax.text(x,min(ys)-.04,f'{median:.3f}×',ha='center',color=color,fontsize=12)
 ax.set_xticks([0,1],['E043 · shared tile','E044 · register tile']);ax.set_xlim(-.48,1.48);ax.set_ylim(.45,1.075);ax.set_yticks([.5,.6,.7,.8,.9,1.],['0.50×','0.60×','0.70×','0.80×','0.90×','1.00×']);ax.grid(axis='y',color='#e2e8f0',lw=.7);ax.set_axisbelow(True)
axs[0].set_ylabel('Speed ratio = native M8 latency / candidate latency',labelpad=12)
fig.text(.095,.945,'Native-M8 local screens · both candidates slower',fontsize=20,weight='bold',color='#172536')
fig.text(.095,.893,'M5 Ultra · real Qwen3.8-27B NVFP4 weights · T8/K5,120 · synthetic BF16 inputs',fontsize=12,color='#475569')
fig.text(.095,.845,'Separate experiment rounds; not paired against each other and not engine Pareto points.',fontsize=11,color='#475569')
fig.text(.095,.18,'Small marks: all five sequential groups. Large marks: median group ratio. Below 1 favors native M8.',fontsize=11,color='#253449')
fig.text(.095,.137,'Each group has 20 paired AB/BA samples; 400 raw timings retained per experiment. Focused ratio scale.',fontsize=10,color='#475569')
fig.text(.095,.094,'E043 parent PASS. E044 parent FAIL (lease −2, unclosed); watchdog + later reconciliation restored service.',fontsize=10,color='#8a3636')
fig.text(.095,.051,'Saved numerical outputs match native M8. These warm local results reject adoption; no whole-model speed claim.',fontsize=10,color='#475569')
artifacts={}
for ext in ('png','svg'):
 p=D/f'native-m8-screen-comparison.{ext}';assert not p.exists();fig.savefig(p,dpi=170,metadata={'Date':None} if ext=='svg' else None);artifacts[p.name]=dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
plt.close(fig)
(D/'chartdata.json').write_text(json.dumps(dict(rows=data,source_refs=refs,scope='Separate local rounds; no paired inter-experiment or engine conclusion.'),indent=2)+'\n')
r=dict(status='RENDERED',visual_review='PENDING',render_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),artifacts=artifacts,sources=refs)
(D/'render-receipt.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
