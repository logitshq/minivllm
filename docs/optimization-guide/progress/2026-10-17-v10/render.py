"""Fourteen audited observations; discrete throughput / full HTTP latency only."""
from pathlib import Path
import hashlib,json,statistics,math
HERE=Path(__file__).resolve().parent
PINS={'omlx-audit-supplement-verification.json': '948eebfb40ed94d60e4327336deaa3e8fd34ba9b2985eefbd1fc05bfe619b33c', 'omlx-first-formal-failure.json': '4b6f610dda24d0c4d52ae7ec22452709491a84bcd460f36814ae9ca1e416530f', 'v9-chartdata.json': 'c0e5ac892a2d675b1d2e227cc1e8b214cb88630de7258fc018299f76d94a8534', 'native-independent-audit.json': 'b669350e95ef07f4da5cac3a5cf6fb6fdaf2aea57e886d3ba60b9afd15e950bd', 'lithos-independent-audit.json': '562787c59f3a3b96a71b279538f61b38fd042dae9f2bc9c6117e022db7a6fb77', 'omlx-independent-audit.json': 'b8582928c375524f8a0286360d606adbac834494a73910a6d85dcd9ee212c591', 'omlx-resources.json': 'a8a492863b4a3ea6b5467919bf2fefa8b6256e0803a4705488a69252095278df', 'omlx-restoration.json': '9aaf59b6482f22c24268b36e3ee8b661510d924b2a2ae3b4b43e204b5a934aca', 'omlx-archive-verification.json': 'e3e45750ba7edfeffa7347cfeb40ff6a395de6001dacf116cad92b6efee398b1'}
EXPECTED={'native-independent-audit.json': ('E023-native-formal-independent-audit-v1', 'b57e4e9657f3dddb04069290bdfbab4c99b0d6cbfb1427e454d0217d5463c870'), 'lithos-independent-audit.json': ('E023-lithos-D512-formal-independent-audit-v1', '768dc970dca201a700c94e07e17c5fdd399c903f444272ea150eb6336575c9c2'), 'omlx-independent-audit.json': ('E023-omlx-D512-formal-independent-audit-v2', 'a34a01882225ab1166fc8c4742c419c9ce2e697ed2e4931d49f7af65ae24232a'), 'omlx-resources.json': ('oMLX-D512-formal-resource-audit-v2', 'd488e30a1cef33663657a45ab74269753e821b21f1539b078c097be35ee206c5'), 'omlx-restoration.json': ('cold-service-restoration-audit-v1', '03476fd4f63087209e242ca4e8b072296c1983bca281bf3671e9e5134520c15e')}
METRICS=('aggregate_output_tok_s','e2e_p50_s','e2e_p95_s')
ENGINES=('mini-vLLM','oMLX','Lithos')
COLORS={'mini-vLLM':'#1267b1','oMLX':'#a13d73','Lithos':'#ba6e20'}
MARKERS={'mini-vLLM':'o','oMLX':'s','Lithos':'^'}
def need(ok,msg):
 if not ok:raise ValueError(msg)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def nondominated(p,rows,x):
 return not any(q[x]<=p[x] and q[METRICS[0]]>=p[METRICS[0]] and (q[x]<p[x] or q[METRICS[0]]>p[METRICS[0]]) for q in rows)
def load():
 manifest=json.loads((HERE/'sources/manifest.json').read_bytes());need(set(manifest)==set(PINS),'exact source set');data={}
 for name,pin in PINS.items():
  raw=(HERE/'sources'/name).read_bytes();need(sha(raw)==pin==manifest[name]['sha256'] and len(raw)==manifest[name]['bytes'],name)
  data[name]=json.loads(raw)
 for name,(schema,auditor) in EXPECTED.items():
  need(data[name]['schema']==schema and data[name]['audit_source_sha256']==auditor,name+' actual audit identity')
 need(data['omlx-archive-verification.json']['schema']=='E023-omlx-D512-formal-archive-verification-v1' and data['omlx-archive-verification.json']['status']=='PASS','closed archive verification')
 supplement=data['omlx-audit-supplement-verification.json'];need(supplement['schema']=='E023-omlx-formal-audit-supplement-verification-v1' and supplement['status']=='PASS','post-run audit supplement archive verified')
 need(supplement['original']['archive']==data['omlx-archive-verification.json']['archive'],'same original archive')
 for n,target in [('independent-formal.json','omlx-independent-audit.json'),('resource-summary.json','omlx-resources.json')]:
  need(supplement['audit_receipts'][n]==manifest[target],'new qualified receipt bound to supplement')
 need(supplement['original']['sidecar_sha256']['independent-formal.json']==PINS['omlx-first-formal-failure.json'],'original failure lineage')
 first=data['omlx-first-formal-failure.json'];need(first['status']=='FAIL' and first['raw_requests_opened'] is False,'preserved original failure before raw request reparse')
 base=data['v9-chartdata.json'];rows=[];summaries=[]
 for engine,name,expected in [('mini-vLLM','native-independent-audit.json',2),('oMLX','omlx-independent-audit.json',6),('Lithos','lithos-independent-audit.json',6)]:
  audit=data[name];need(audit['status']=='PASS' and audit['qualified'] is True and audit['performance_qualified'] is True and audit['restoration']['status']=='PASS',name+' formal/restoration')
  need(len(audit['points'])==expected,'point count')
  for p in audit['points']:
   need(p['concurrency'] in (1,8) and p['requests']==64 and p['warm_requests_checked']==8 and p['max_http_outstanding']==p['concurrency'],'actual request coverage')
   need(all(type(p[k]) in (int,float) and math.isfinite(p[k]) and p[k]>0 for k in METRICS),'finite metrics')
   rows.append(dict(p,engine=engine))
  for C in (1,8):
   rr=[p for p in rows if p['engine']==engine and p['concurrency']==C];need(len(rr)==expected//2,'per C process count')
   summaries.append(dict(engine=engine,concurrency=C,fresh_processes=len(rr),
    **{k:statistics.median(p[k] for p in rr) for k in METRICS},
    observed_process_ranges={k:[min(p[k] for p in rr),max(p[k] for p in rr)] for k in METRICS},
    scope='Median of process metrics, not pooled request percentiles or confidence intervals.'))
 old=[p for p in rows if p['engine']!='oMLX'];need(len(old)==8,'eight preserved historical points')
 for p in old:
  prior=next(r for r in base['points'] if all(r[k]==p[k] for k in ('engine','concurrency','repeat')))
  need(all(prior[k]==v for k,v in p.items()),'historical point unchanged')
 o=data['omlx-independent-audit.json'];r=data['omlx-resources.json'];restore=data['omlx-restoration.json']
 need(o['formal_requests']==384 and o['warm_requests']==48 and r['status']=='PASS' and r['independent_resource_verified'] is True and restore['qualified'] is True,'new full-cycle gates')
 need(o['owner_plan_sha256']==r['owner_plan_sha256']==restore['owner_plan_sha256'],'same actual owner')
 need(o['restoration']['journal_sha256']==restore['journal_sha256'] and o['restoration']['service_restore']['replacement']==restore['service_restore']['replacement'],'same actual independently restored journal and process')
 for C in (1,8):
  s=next(p for p in summaries if p['engine']=='oMLX' and p['concurrency']==C);a=next(p for p in o['process_metric_medians'] if p['concurrency']==C)
  need(all(s[k]==a[k] for k in METRICS) and a['n_fresh_processes']==3,'independent process medians')
 for p in rows:
  for x,flag in [('e2e_p50_s','nondominated_p50_among_displayed_observations'),('e2e_p95_s','nondominated_p95_among_displayed_observations')]:p[flag]=nondominated(p,rows,x)
 chart=dict(schema='D512-throughput-latency-progress-v10',hardware=base['hardware'],model=base['model'],revision=base['revision'],workload=base['workload'],points=rows,process_metric_medians=summaries,performance_point_count=len(rows),historical_point_metrics_unchanged=True,sources=manifest,
  scope='Fourteen discrete audited observer-OFF observations; oMLX uses a separately recorded post-run parser correction, with original FAIL preserved; native n=1/C historical, external engines n=3/C in newer cycles. Full POST-to-DONE includes server queueing but excludes waiting for a client concurrency slot. No paired confidence interval, interpolation, maximum-concurrency or global-frontier claim. Committed output counts under each qualified contract; no cross-framework full-state or output-quality equality. Lithos C8 HTTP uses official FIFO GPU execution.')
 (HERE/'chartdata.json').write_text(json.dumps(chart,indent=2,allow_nan=False)+'\n');return chart

def draw(c):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from matplotlib.lines import Line2D
 from matplotlib.ticker import MaxNLocator
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','svg.hashsalt':'D512-v10','figure.facecolor':'white','savefig.facecolor':'white'})
 rows=c['points'];ss=c['process_metric_medians'];fig,axs=plt.subplots(1,2,figsize=(16,11),sharey=True)
 fig.subplots_adjust(left=.075,right=.97,bottom=.43,top=.81,wspace=.19)
 fig.text(.075,.958,'Qwen3.8-27B · throughput vs full HTTP latency',size=23,weight='bold',color='#172536')
 fig.text(.075,.92,'M5 Ultra 256 GiB · identical NVFP4 weights / D prompts · normal EOS, cap 512 · formal observers OFF',size=12,color='#475569')
 fig.text(.075,.888,'14 measured points: native 1 process per C; oMLX and Lithos 3 processes per C. C1 is enlarged in each inset.',size=11,color='#475569')
 handles=[Line2D([],[],marker=MARKERS[e],color='none',markerfacecolor=COLORS[e],markeredgecolor='white',markersize=9,label=e+(' · historical' if e=='mini-vLLM' else '')) for e in ENGINES]
 fig.legend(handles=handles,loc='upper left',bbox_to_anchor=(.07,.86),frameon=False,ncol=3)
 for ax,x,lbl in zip(axs,('e2e_p50_s','e2e_p95_s'),('p50','p95')):
  ax.grid(color='#e1e7ef',linewidth=.8,zorder=0);ax.set_xlim(0,max(p[x] for p in rows)*1.16);ax.set_ylim(0,max(p[METRICS[0]] for p in rows)*1.21)
  ax.set_xlabel('POST-to-DONE '+lbl+' latency (s)  ← lower',labelpad=12);ax.xaxis.set_major_locator(MaxNLocator(6));ax.yaxis.set_major_locator(MaxNLocator(6))
  ax.text(.025,.965,'Better toward upper left',transform=ax.transAxes,va='top',size=10,color='#64748b')
  for p in rows:ax.scatter(p[x],p[METRICS[0]],s=78,color=COLORS[p['engine']],marker=MARKERS[p['engine']],edgecolor='#142238' if nondominated(p,rows,x) else 'white',linewidth=1.2,zorder=4,alpha=.9)
  for p in ss:
   if p['concurrency']==8:
    rr=[r for r in rows if r['engine']==p['engine'] and r['concurrency']==8];anchor=min(rr,key=lambda r:abs(r[x]-p[x]))
    offset={'mini-vLLM':(8,25),'oMLX':(8,-24),'Lithos':(8,11)}[p['engine']]
    ax.annotate(p['engine']+' C8',(anchor[x],anchor[METRICS[0]]),xytext=offset,textcoords='offset points',size=10,color=COLORS[p['engine']],arrowprops=dict(arrowstyle='-',color=COLORS[p['engine']],lw=.7))
  small=[p for p in rows if p['concurrency']==1];ins=ax.inset_axes([.52,.48,.43,.40]);ins.set_facecolor('#f6f8fb')
  xx=[p[x] for p in small];yy=[p[METRICS[0]] for p in small];dx=max(max(xx)-min(xx),.1);dy=max(max(yy)-min(yy),1.)
  ins.set_xlim(min(xx)-.3*dx,max(xx)+.3*dx);ins.set_ylim(min(yy)-.3*dy,max(yy)+.35*dy);ins.tick_params(labelsize=8);ins.grid(color='#dfe5ec',linewidth=.6)
  ins.set_title('C1 zoom',fontsize=10,loc='left');ins.xaxis.set_major_locator(MaxNLocator(4));ins.yaxis.set_major_locator(MaxNLocator(4))
  for p in small:ins.scatter(p[x],p[METRICS[0]],s=45,color=COLORS[p['engine']],marker=MARKERS[p['engine']],edgecolor='#142238' if nondominated(p,rows,x) else 'white',linewidth=.8,zorder=4)
 axs[0].set_ylabel('Aggregate committed output (tokens/s)  ↑ higher',labelpad=10)
 fig.text(.075,.366,'Process-metric medians',size=13,weight='bold',color='#172536')
 columns=[['Engine','HTTP C','Processes','Committed tok/s','p50 end-to-end','p95 end-to-end']]
 for p in ss:columns.append([p['engine'],str(p['concurrency']),str(p['fresh_processes']),f"{p[METRICS[0]]:.2f}",f"{p['e2e_p50_s']:.2f} s",f"{p['e2e_p95_s']:.2f} s"])
 tax=fig.add_axes([.075,.17,.895,.175]);tax.axis('off');table=tax.table(cellText=columns,cellLoc='left',colWidths=[.21,.12,.13,.19,.175,.175],bbox=[0,0,1,1]);table.auto_set_font_size(False);table.set_fontsize(10)
 for (rr,cc),cell in table.get_celld().items():
  cell.set_edgecolor('white');cell.set_linewidth(2);cell.set_facecolor('#e7edf5' if rr==0 else ('#f4f6fa' if rr%2 else 'white'))
  if rr==0:cell.set_text_props(weight='bold',color='#172536')
  elif cc==0:cell.set_text_props(weight='bold',color=COLORS[columns[rr][0]])
 fig.text(.075,.13,'Black outlines: nondominated observed points only. 8 warm + 64 formal requests per process; queueing included.',size=10,color='#475569')
 fig.text(.075,.106,'Separate measurement cycles; no paired CI or interpolation. Lithos C8 is HTTP concurrency with FIFO GPU execution.',size=10,color='#475569')
 fig.text(.075,.075,'E024 route graph: serving-state checks passed; actual HTTP qualification and formal speed remain pending.',size=10,color='#475569')
 fig.text(.075,.051,'Protocol/count qualification does not establish cross-engine full-state or answer-quality equality.',size=10,color='#475569')
 fig.text(.075,.027,'oMLX: separate post-run audit parser correction; original failed audit receipts are preserved.',size=10,color='#475569')
 for suffix in ('png','svg'):
  target=HERE/('D512-throughput-latency.'+suffix);fig.savefig(target,dpi=180,metadata={'Date':None} if suffix=='svg' else None)
  if suffix=='svg':target.write_text('\n'.join(line.rstrip() for line in target.read_text().splitlines())+'\n')
 plt.close(fig)
 report=dict(status='PASS',performance_points=len(rows),source_pins=PINS,renderer_sha256=sha(Path(__file__).read_bytes()),chartdata_sha256=sha((HERE/'chartdata.json').read_bytes()),historical_metrics_unchanged=True,scope='Offline audit-receipt/source checks and plotting; original measurements not rerun.')
 (HERE/'verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':draw(load())
