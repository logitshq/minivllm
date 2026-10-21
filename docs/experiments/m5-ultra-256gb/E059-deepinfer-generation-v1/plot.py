"""Render the closed, independently audited generation screen without pooling runs."""
from pathlib import Path
import hashlib,json,statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
complete=json.loads((HERE/'complete.json').read_text());audit=json.loads((HERE/'independent-audit.json').read_text())
assert audit['status']=='PASS' and audit['complete_sha256']==hashlib.sha256((HERE/'complete.json').read_bytes()).hexdigest()
rows=complete['timings'];arms=list(dict.fromkeys(r['arm'] for r in rows));assert arms[0]=='native'
names={'native':'Native AR','fixed2':'Verify T2','fixed4':'Verify T4','fixed8':'Verify T8','adaptive':'Frozen policy'}
colors={'native':'#59677c','fixed2':'#7755c9','fixed4':'#1e9e92','fixed8':'#dc8851','adaptive':'#4b8cc9'}
fig,axes=plt.subplots(1,2,figsize=(12.6,5.2),sharex=True)
fig.patch.set_facecolor('#fcfcfa');values=[]
limit=max(r['elapsed_ns']/1e6 for r in rows)*1.20
for case,ax in enumerate(axes):
 ax.set_facecolor('#fcfcfa')
 for i,arm in enumerate(arms):
  samples=[r['elapsed_ns']/1e6 for r in rows if r['case']==case and r['arm']==arm];m=statistics.median(samples)
  ax.barh(i,m,color=colors[arm],height=.53,alpha=.87)
  ax.scatter(samples,[i-.075,i,i+.075],s=30,c='#172735',edgecolor='white',linewidth=.8,zorder=4)
  ax.text(m+18,i,f'{m:.1f} ms',va='center',fontsize=10,fontweight='bold',color='#253546')
  values.append(dict(case=['math','chinese'][case],arm=arm,median_ms=m,all_samples_ms=samples))
 ax.set_yticks(range(4),[names[a] for a in arms]);ax.invert_yaxis();ax.set_xlim(0,limit);ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
 ax.set_xlabel('Generation latency (ms) — lower is better',fontsize=10);ax.set_title(['Math prompt','Chinese prompt'][case],loc='left',fontweight='bold',fontsize=13,pad=15)
 for spine in ('top','right','left'):ax.spines[spine].set_visible(False)
 ax.spines['bottom'].set_color('#bbc2c8');ax.tick_params(axis='y',length=0)
fig.suptitle(('E060 • Shorter verification reduces rejected work' if 'fixed2' in arms else 'E059 • Real speculation exposes workload dependence'),x=.07,ha='left',fontsize=18,fontweight='bold',color='#132738')
fig.text(.07,.88,'M5 Ultra 256 GiB · Qwen3.8-27B NVFP4 · medians + every individual observation',fontsize=11,color='#4d5a68')
fig.text(.07,.075,'1024-token synthetic context; 31 decoded tokens after the prefill token; three samples per arm and prompt.',fontsize=9,color='#4d5a68')
fig.text(.07,.037,'Host-driven generation loop; prefill and HTTP excluded. All qualified final logits/features/KV/GDN bytes match native.',fontsize=9,color='#4d5a68')
fig.subplots_adjust(left=.10,right=.98,top=.78,bottom=.23,wspace=.32)
fig.savefig(HERE/'generation-cost.png',dpi=170);fig.savefig(HERE/'generation-cost.svg');plt.close(fig)
p=HERE/'generation-cost.svg';p.write_text('\n'.join(line.rstrip() for line in p.read_text().splitlines())+'\n')
(HERE/'plot-values.json').write_text(json.dumps({'complete_sha256':audit['complete_sha256'],'audit_sha256':hashlib.sha256((HERE/'independent-audit.json').read_bytes()).hexdigest(),'values':values},indent=2)+'\n')
