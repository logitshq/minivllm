"""v11: unchanged fourteen formal observations, separate closed qualification evidence."""
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
# Filled once by prepare.py after the registered GDN37 cycle is CLOSED and audited.
QUALIFICATION_PINS = {'v10-chartdata.json': '8f5a5edba7d5171ee3c0ad7caf6a1dfece80fbda00a78b07f932c7c9291acbe8', 'e024-cycle.json': '8cbde8d54b246d25deeed2e7c25460d954087a2daf2083efdc9779402d2e313f', 'e024-c1.json': 'b9443261ca47dbea5782e469dca1987737e51df34a12c01539fbc46fe2a99ff4', 'e024-c8.json': '4aac471df2778dd39ff74e10a7f965f2ad4690546c6dcaee9ce144e30c28e4d7', 'e024-restoration.json': '6bb4be00801ce4519a0ab399d171463e8c4a3e4c1802031295c965497316914e', 'e024-journal.json': 'dff835b4518710efe2fce1c6b3f12784432b0f3295a63aaa5503840e916ddd61', 'gdn-canary.json': '8765a6089c362e37a89f66f8584c0bf39a45cdf929757a2f7bb87d764cd11df4', 'gdn-canary-resources.json': '8631f0dc594ba2ad639a0b2b386c8f2bacca6c268ed278f6beda4efa5d5c708a', 'gdn-canary-restoration.json': '7ee69cc3babbe404438f526ddc8e97b068e24913386dd9b01c8298dc48fb24c4', 'gdn-canary-journal.json': '2091fa547a94a8538b44f64a3c8fdf964764113c5a8c5a7acf653861b7c8cf14', 'gdn-serving.json': 'fee50c90c30c3224934489c66e5ecb4a5d7f0c4fb57ae0baa0576f6efe9a769b', 'gdn-serving-resources.json': '874595f189d86c84b3e8af1900809225da8f83013d87526d4466e843f0e5f601', 'gdn-serving-restoration.json': '673d1ba136c784f445567015f15c4fb7c0b22c6635a33201b14026e2ba3edd29', 'gdn-serving-journal.json': 'abddf49c3c43407e7903dfabb06f3227ee993b464f2f66055e22928d8f6dcbb5', 'gdn-serving-owner.json': '3f39cce96fad5ee2b45d1b469d798815da42ad848c0184745a052d6a6ddfebad', 'gdn-serving-child.json': '8072b3bb91448abdc74b3312604b3491a97522d71e540be716a15f8a563e6ec5'}
V10_CHART_SHA = '8f5a5edba7d5171ee3c0ad7caf6a1dfece80fbda00a78b07f932c7c9291acbe8'
SERVING_PLAN_SHA = 'dfb6d3d306e6247e34dfdfe54746e12205b297e166e1f175e85024786d73163b'
SERVING_OWNER_SHA = 'c908eed8087cd72a86a4994a7fc37ffb41bfd719726ad04ff6d8b6f0768c14d2'

def qualification_status(q, manifest):
 """Consume small CLOSED independent receipts; never open a device or steps ledger."""
 def passed(name, schema, source=None):
  d=q[name];need(d['schema']==schema and d['status']=='PASS',name+' independent terminal PASS')
  if source is not None:need(d.get('audit_source_sha256',d.get('source_sha256'))==source,name+' auditor identity')
  return d
 def restored(prefix):
  a=passed(prefix+'-restoration.json','cold-service-restoration-audit-v1')
  j=q[prefix+'-journal.json'];need(j['state']=='RESTORED' and a['qualified'] is True,prefix+' restoration')
  need(a['journal_sha256']==manifest[prefix+'-journal.json']['sha256'],prefix+' exact journal')
  need(a['owner_plan_sha256']==j['owner_plan']['sha256'],prefix+' same owner plan')
  need(a['service_restore']['replacement']==j['replacement'],prefix+' same replacement')
  need(a['service_restore']['warm_kv_restored'] is False,prefix+' no warm KV claim')
  return a,j
 prior=q['v10-chartdata.json'];need(manifest['v10-chartdata.json']['sha256']==V10_CHART_SHA,'unchanged v10 data')
 cycle=passed('e024-cycle.json','E024-route-http-cycle-independent-audit-v1','bdcd32a06133bae37786eb148318a0d38847df2f2a20c94c984104394570cca4')
 need(all(cycle[k] is True for k in ('qualified','http_history_qualified','independent_resource_verified','restoration_verified')),'E024 complete cycle gate')
 need(cycle['performance_qualified'] is False,'E024 no measured speed')
 route_restore,route_journal=restored('e024')
 need(cycle['owner_plan_sha256']==route_restore['owner_plan_sha256'] and cycle['journal_sha256']==route_restore['journal_sha256'],'E024 cycle/restore binding')
 histories=[]
 for C,N in ((1,8),(8,16)):
  name=f'e024-c{C}.json';a=passed(name,'E024-recorded-route-http-independent-audit-v2','88f32e3b3f44206b48a1f581cbfe0bde0788060cbd2a0c5030a3ec472a0139fe')
  need(a['qualified'] is True and a['http_history_qualified'] is True and a['performance_qualified'] is False,'captured history scope')
  need(a['http']['requests']==N and a['http']['max_http_outstanding']==C and a['http']['fully_drained'] is True,'actual HTTP coverage')
  need(cycle['numerical_audits'][str(C)]['sha256']==manifest[name]['sha256'],'E024 exact numerical audit')
  need(cycle['numerical_audits'][str(C)]['ledger_sha256']==a['ledger']['sha256'],'E024 same recorded history')
  histories.append(dict(concurrency=C,requests=N,steps=a['ledger']['steps'],observed_eos=a['ledger']['observed_terminal_eos']))
 canary=passed('gdn-canary.json','GDN-prework-real-layer-independent-v1','a3e97dc368da47735d7c8e9fd2bcb40d7decaccb9c5926084d53cffb10495d3d')
 need(canary['canary_qualified'] is True,'GDN canary qualified')
 for k,v in dict(raw_files=60,full_output_state_pairs=54,causal_state_pairs=24,full_vocabulary_greedy_checks=6,device_comparison_arrays=162).items():need(canary[k]==v,'actual canary '+k)
 need(canary['full_model_qualified'] is False and canary['performance_qualified'] is False,'limited canary')
 passed('gdn-canary-resources.json','cold-single-child-resource-reparse-v1','5948ccbf7baeb21d1e71d9e8f7169b1bd9f84e65eb492746aba86fe377806c23')
 restored('gdn-canary')
 # This gate intentionally remains unsatisfied while the 37-request run/audit is live.
 serving=passed('gdn-serving.json','GDN-prework-serving-independent-v1','268cc336417a3d31dffc30c749701d132da610fbb41d9b0f4c85ad6adaa2f2bc')
 need(serving['qualified'] is True and serving['same_schedule_qualified'] is True,'actual GDN37 independent gate')
 need(serving['plan_sha256']==SERVING_PLAN_SHA,'registered GDN37 plan')
 for k in ('http_qualified','performance_qualified','combined_candidates_qualified','parameter_replacement_qualified','offline_tensor_bytes_rechecked'):
  need(serving[k] is False,'limited GDN37 scope: '+k)
 child=q['gdn-serving-child.json'];owner=q['gdn-serving-owner.json']
 need(child['schema']=='GDN-prework-serving-result-v1' and child['status']=='PASS' and child['plan_sha256']==SERVING_PLAN_SHA,'actual child terminal')
 need(serving['terminal_sha256']['complete.json']==manifest['gdn-serving-child.json']['sha256'],'audited child complete')
 need(owner['schema']=='cold-single-child-owner-result-v1' and owner['status']=='PASS' and owner['owner_plan_sha256']==SERVING_OWNER_SHA,'registered closed owner')
 need(owner['all_children_reaped'] is True and owner['child_returncode']==0 and owner['planned_trials']==['gdn-prework-serving'],'actual child reaped')
 serving_restore,journal=restored('gdn-serving');need(serving_restore['owner_plan_sha256']==SERVING_OWNER_SHA,'GDN37 restore owner')
 resources=passed('gdn-serving-resources.json','cold-single-child-resource-reparse-v1','5948ccbf7baeb21d1e71d9e8f7169b1bd9f84e65eb492746aba86fe377806c23')
 need(resources['phase']=='gdn-prework-serving' and resources['owner_pid']==owner['owner_pid'] and resources['owner_status']=='PASS' and resources['guard_violations']==[],'GDN37 actual resource scope')
 cases=serving['ledger']['cases'];counts=[len(c['completed']) for c in cases]
 need(counts==[8,16,13] and not serving['ledger']['partial_cases'] and serving['ledger']['truncated_final_line'] is None,'original37 complete cases')
 need(serving['coverage']==child['coverage'] and serving['coverage']['layer_installs']==serving['coverage']['layer_restores']==96,'same all-layer coverage')
 return prior,[
  dict(candidate='E024 route graph',status='HTTP history PASS',evidence='C1: 8 requests; C8: 16 requests',next_gate='Paired OFF/ON formal pending',histories=histories,performance_qualified=False),
  dict(candidate='GDN prework',status='Canary + 37-request PASS',evidence='18 canary calls; 8 / 16 / 13 serving requests',next_gate='HTTP + formal pending',canary_raw_pairs=54,serving_requests=sum(counts),observed_eos=serving['observed_eos_requests'],performance_qualified=False),
  dict(candidate='GLM 190 GiB',status='Download / load pending',evidence='No qualified measurement',next_gate='Load / numerical gates pending',basis='Maintainer status at this snapshot; no qualification receipt or performance point.',performance_qualified=False)]

def load_qualification_sources():
 need(type(QUALIFICATION_PINS) is dict,'v11 UNSEALED: wait for CLOSED independent GDN37 audit; run prepare.py only afterward')
 manifest=json.loads((HERE/'qualification-sources/manifest.json').read_bytes())
 need(set(manifest)==set(QUALIFICATION_PINS),'exact qualification receipt set');q={}
 for name,digest in QUALIFICATION_PINS.items():
  raw=(HERE/'qualification-sources'/name).read_bytes()
  need(sha(raw)==digest==manifest[name]['sha256'] and len(raw)==manifest[name]['bytes'],name+' immutable copy')
  q[name]=json.loads(raw)
 prior,status=qualification_status(q,manifest)
 return prior,status,manifest

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
 prior,qualification,qualification_sources=load_qualification_sources()
 need(rows==prior['points'] and summaries==prior['process_metric_medians'],'all fourteen v10 points / process medians EXACT unchanged')
 chart=dict(schema='D512-throughput-latency-progress-v11',qualification_status=qualification,qualification_sources=qualification_sources,previous_chartdata_sha256=V10_CHART_SHA,hardware=base['hardware'],model=base['model'],revision=base['revision'],workload=base['workload'],points=rows,process_metric_medians=summaries,performance_point_count=len(rows),historical_point_metrics_unchanged=True,sources=manifest,
  scope='Fourteen discrete audited observer-OFF observations; oMLX uses a separately recorded post-run parser correction, with original FAIL preserved; native n=1/C historical, external engines n=3/C in newer cycles. Full POST-to-DONE includes server queueing but excludes waiting for a client concurrency slot. No paired confidence interval, interpolation, maximum-concurrency or global-frontier claim. Committed output counts under each qualified contract; no cross-framework full-state or output-quality equality. Lithos C8 HTTP uses official FIFO GPU execution.')
 (HERE/'chartdata.json').write_text(json.dumps(chart,indent=2,allow_nan=False)+'\n');return chart

def draw(c):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from matplotlib.lines import Line2D
 from matplotlib.ticker import MaxNLocator
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','svg.hashsalt':'D512-v11','figure.facecolor':'white','savefig.facecolor':'white'})
 rows=c['points'];ss=c['process_metric_medians'];fig,axs=plt.subplots(1,2,figsize=(16,13.8),sharey=True)
 fig.subplots_adjust(left=.075,right=.97,bottom=.49,top=.83,wspace=.19)
 fig.text(.075,.967,'Qwen3.8-27B · throughput vs full HTTP latency',size=23,weight='bold',color='#172536')
 fig.text(.075,.936,'M5 Ultra 256 GiB · identical NVFP4 weights / D prompts · normal EOS, cap 512 · formal observers OFF',size=12,color='#475569')
 fig.text(.075,.91,'14 measured points: native 1 process per C; oMLX and Lithos 3 processes per C. C1 is enlarged in each inset.',size=11,color='#475569')
 handles=[Line2D([],[],marker=MARKERS[e],color='none',markerfacecolor=COLORS[e],markeredgecolor='white',markersize=9,label=e+(' · historical' if e=='mini-vLLM' else '')) for e in ENGINES]
 fig.legend(handles=handles,loc='upper left',bbox_to_anchor=(.07,.89),frameon=False,ncol=3)
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
 fig.text(.075,.438,'Process-metric medians · same 14 observations as v10',size=13,weight='bold',color='#172536')
 columns=[['Engine','HTTP C','Processes','Committed tok/s','p50 end-to-end','p95 end-to-end']]
 for p in ss:columns.append([p['engine'],str(p['concurrency']),str(p['fresh_processes']),f"{p[METRICS[0]]:.2f}",f"{p['e2e_p50_s']:.2f} s",f"{p['e2e_p95_s']:.2f} s"])
 tax=fig.add_axes([.075,.275,.895,.145]);tax.axis('off');table=tax.table(cellText=columns,cellLoc='left',colWidths=[.21,.12,.13,.19,.175,.175],bbox=[0,0,1,1]);table.auto_set_font_size(False);table.set_fontsize(10)
 for (rr,cc),cell in table.get_celld().items():
  cell.set_edgecolor('white');cell.set_linewidth(2);cell.set_facecolor('#e7edf5' if rr==0 else ('#f4f6fa' if rr%2 else 'white'))
  if rr==0:cell.set_text_props(weight='bold',color='#172536')
  elif cc==0:cell.set_text_props(weight='bold',color=COLORS[columns[rr][0]])
 fig.text(.075,.248,'Qualification progress · no new performance points',size=13,weight='bold',color='#172536')
 qcols=[['Candidate','Verified status','Actual coverage','Next gate']]
 for q in c['qualification_status']:qcols.append([q['candidate'],q['status'],q['evidence'],q['next_gate']])
 qax=fig.add_axes([.075,.135,.895,.10]);qax.axis('off')
 qt=qax.table(cellText=qcols,cellLoc='left',colWidths=[.16,.24,.34,.26],bbox=[0,0,1,1]);qt.auto_set_font_size(False);qt.set_fontsize(9.5)
 for (rr,cc),cell in qt.get_celld().items():
  cell.set_edgecolor('white');cell.set_linewidth(2);cell.set_facecolor('#e7edf5' if rr==0 else '#f4f6fa')
  cell.set_text_props(color='#172536',weight='bold' if rr==0 or cc==0 else 'normal')
 fig.text(.075,.109,'Black outlines: nondominated observations only. 8 warm + 64 formal requests / process; server queueing included.',size=10,color='#475569')
 fig.text(.075,.087,'Separate cycles; native n=1/C, external engines n=3/C. No paired CI or interpolation. Lithos C8 uses FIFO GPU execution.',size=10,color='#475569')
 fig.text(.075,.065,'Qualification is separate from speed. GDN serving uses device bit/finite receipts; no offline recomputation of every success tensor.',size=10,color='#475569')
 fig.text(.075,.043,'Protocol/count gates do not establish cross-engine full-state or answer-quality equality. No E024 + GDN composition claim.',size=10,color='#475569')
 fig.text(.075,.021,'oMLX: documented post-run audit parser correction; original FAIL preserved. GLM has no qualified performance point.',size=10,color='#475569')
 for suffix in ('png','svg'):
  target=HERE/('D512-throughput-latency.'+suffix);fig.savefig(target,dpi=180,metadata={'Date':None} if suffix=='svg' else None)
  if suffix=='svg':target.write_text('\n'.join(line.rstrip() for line in target.read_text().splitlines())+'\n')
 plt.close(fig)
 report=dict(status='PASS',performance_points=len(rows),source_pins=PINS,qualification_source_pins=QUALIFICATION_PINS,previous_chartdata_sha256=V10_CHART_SHA,qualification_status=c['qualification_status'],renderer_sha256=sha(Path(__file__).read_bytes()),chartdata_sha256=sha((HERE/'chartdata.json').read_bytes()),historical_metrics_unchanged=True,scope='Offline audit-receipt/source checks and plotting; original measurements not rerun.')
 (HERE/'verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':draw(load())
