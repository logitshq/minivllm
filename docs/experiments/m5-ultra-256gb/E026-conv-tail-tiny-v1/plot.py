"""Render audited backing bytes only. No inference or performance data."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
D=Path(__file__).resolve().parent
rows=json.loads((D/'chartdata.json').read_text())['rows']
assert len(rows)==48
lengths=[1,19,935,978]
values={}
for arm in ('off','on-lazy','on-materialize'):
 for n in lengths:
  picked=[r for r in rows if r['arm']==arm and r['length']==n]
  assert len(picked)==4 and len({r['retained_output_backing_bytes'] for r in picked})==1
  values[arm,n]=picked[0]['retained_output_backing_bytes']
for n in lengths:assert values['on-lazy',n]==values['on-materialize',n]==65536
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(figsize=(12,6.4))
fig.subplots_adjust(left=.12,right=.94,bottom=.24,top=.72)
fig.text(.12,.94,'A small GDN state can release a large prefill buffer',fontsize=21,weight='bold')
fig.text(.12,.88,'M5 Ultra · E026 synthetic state diagnostic · all 48 cases preserve every uint16 bit',fontsize=12)
fig.text(.12,.82,'Retained backing after dropping the original parent and tail; lower is better',color='#475569')
for i,n in enumerate(lengths):
 off=values['off',n]/1024; on=values['on-lazy',n]/1024
 ax.plot([on,off],[i,i],color='#cbd5e1',lw=2,zorder=1)
 ax.scatter(off,i,s=105,marker='s',color='#64748b',label='OFF: original view' if i==0 else None,zorder=3)
 ax.scatter(on,i,s=105,color='#0f766e',label='ON: lazy and immediate (same bytes)' if i==0 else None,zorder=4)
 ax.annotate(f'{off/1024:.2f} MiB' if off>=1024 else f'{off:g} KiB',(off,i),xytext=(8,0),textcoords='offset points',va='center',fontsize=11)
ax.set_xscale('log',base=2)
ax.set_xlim(48,45000)
ax.set_xticks([64,128,512,2048,8192,32768],['64 KiB','128 KiB','512 KiB','2 MiB','8 MiB','32 MiB'])
ax.set_yticks(range(4),[f'T{n}' for n in lengths]);ax.invert_yaxis();ax.set_ylim(3.5,-.5)
ax.set_ylabel('Synthetic prefill length')
ax.set_xlabel('Unique full allocator backing held by the output (logarithmic scale)',labelpad=12)
ax.grid(axis='x',alpha=.18);ax.set_axisbelow(True)
ax.legend(loc='upper left',bbox_to_anchor=(0,1.17),ncol=2,frameon=False,fontsize=11)
fig.text(.12,.10,'Three fresh processes; four payload fixtures per length. Values agree across all four fixtures, not repeat timing samples.',fontsize=10.5,color='#475569')
fig.text(.12,.055,'Memory-only result. No model loaded; no throughput/latency claim. Full-model qualification and performance tests remain pending.',fontsize=10.5,color='#475569')
fig.savefig(D/'retained-backing.png',dpi=180)
fig.savefig(D/'retained-backing.svg')
