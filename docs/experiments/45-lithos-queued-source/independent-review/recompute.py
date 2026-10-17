#!/usr/bin/env python3
"""CPU-only saved-evidence replay; no server, network, MLX, or live process probe."""
import ast, errno, hashlib, json, math, statistics as st
from pathlib import Path
P=Path('/private/tmp/minivllm-engine-optimization-20261014'); R=P/'v23-root-control-build1'
S=P/'v130-lithos-load-probe-prework-build1'; O=Path(__file__).resolve().parent
D=R/'lithos-ollama-v130perf-attempt1/driver-output'; Q=R/'lithos-ollama-v130qual-attempt1/driver-output'
def load(p): return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  while b:=f.read(1<<20): h.update(b)
 return h.hexdigest()
def put(name,v): (O/name).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def close(a,b): assert math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-8),(a,b)
def absent(x): assert x['absent'] and x['returncode']==1 and x['stdout']==x['stderr']==''
suite=load(D/'suite-complete.json'); pins=load(S/'pins.json'); ph=sha(S/'pins.json')
assert ph=='ee894ddd3a116089f4372cd1c8d2c2b2c23d02cad78a5812c6b350252e026679'
assert sha(D/'suite-complete.json')=='31ee70f078a70b13db44dfd9de7433b0b764d43a97d1e366e4bf9e459da4874d'
for name in ['client.py','qualify.py','driver.py']:
 assert sha(S/name)==pins['files'][str(S/name)]
assert sha(pins['visible_gold_source'])==pins['visible_gold_source_sha256']
gold=load(pins['visible_gold_source'])['requests'][0]['visible_roundtrip_ids']; assert len(gold)==32
# Re-encode saved text independently using tokenizer CPU only (no model runtime).
from tokenizers import Tokenizer
tok=Tokenizer.from_file(str(Path(load(P/'workload.json')['model_dir'])/'tokenizer.json'))
requests=[]; groups=[]; windows=[]; resource_rows=[]; cleanup_pids=[]; ports=[]; logs=0; log_prefix_proofs=[]

def group(path,kind,arm):
 g=load(path); rs=g['requests']; C=g['requested_concurrency']; assert len(rs)==C
 assert g['actual_request_count']==g['successful_requests']==C and g['rejected_requests']==g['retries']==0
 assert g['all_correctness_checks_passed'] and g['queue_wait_in_complete_client_latency']
 for r in rs:
  assert r['HTTP_status']==200 and r['error'] is None and r['retry_count']==0 and r['all_correctness_checks_passed']
  assert r['usage']['prompt_tokens']==19 and r['usage']['completion_tokens']==32 and r['finish_reason']=='length'
  assert r['internal_raw_ids'] is None and r['server_queue_wait_s_observed'] is None
  assert r['visible_roundtrip_ids']==gold==tok.encode(r['text'],add_special_tokens=False).ids
  a,h,f,t,c=[r[k+'_perf_counter_s'] for k in ('start','headers','first_text','terminal','cleanup')]
  assert a<=h<=f<=t<=c; close(t-a,r['complete_latency_s'])
  es=r['events']; ts=[e['received_perf_counter_s'] for e in es]; assert ts==sorted(ts) and a<=ts[0]<=ts[-1]<=t
  if r['protocol']=='ollama':
   assert r['terminal']=='NDJSON_DONE_TRUE' and r['endpoint']=='/api/chat'
   final=es[-1]['value']; assert final['done'] is True and final['done_reason']=='length'
   assert final['prompt_eval_count']==19 and final['eval_count']==32
   assert sum(e['value'].get('done') is True for e in es)==1
   text=''.join(e['value'].get('message',{}).get('content','') for e in es)
  else:
   assert kind=='qualification' and r['protocol']=='openai' and r['terminal']=='SSE_DONE'
   text=''.join(ch.get('delta',{}).get('content','') for e in es for ch in e['value'].get('choices',[]))
  assert text==r['text']
  requests.append({'kind':kind,'arm':arm,'group_id':g['group_id'],'request_id':r['request_id'],'C':C,'protocol':r['protocol'],'start_s':a,'terminal_s':t,'latency_ms':(t-a)*1000,'output_tokens':32})
 starts=[r['start_perf_counter_s'] for r in rs]; ends=[r['terminal_perf_counter_s'] for r in rs]
 a,b=min(starts),max(ends); x=st.median([(r['terminal_perf_counter_s']-r['start_perf_counter_s'])*1000 for r in rs]);y=32*C/(b-a)
 assert max(starts)<min(ends) and g['all_requested_POSTs_before_first_terminal']
 close(g['all_attempts_window_s'],b-a); close(g['arrival_spread_s'],max(starts)-min(starts));close(g['latency_p50_ms'],x);close(g['total_throughput_tok_s'],y)
 assert g['completed_token_count']==32*C
 windows.append((a,b));row={'kind':kind,'arm':arm,'group_id':g['group_id'],'C':C,'formal':'measure' in g['group_id'],'start_s':a,'terminal_s':b,'latency_p50_ms':x,'total_throughput_tok_s':y};groups.append(row);return row

def resource(path,kind):
 rows=load(path)
 if isinstance(rows,dict): rows=[rows]
 for r in rows:
  assert r['power_metadata']['source']=='AC Power' and r['processinfo_thermal_metadata']['code']==0 and r['pressure_level_value']==1
  assert r['process_inventory_ok']
  resource_rows.append({'kind':kind,'monotonic_s':r['monotonic_s'],'swap_used_mb':r['swap_used_mb'],'Swapouts':r['vm_counters']['Swapouts'],'free_percent':r['free_percent']})
 return rows

def armcheck(A,count):
 global logs
 rec=load(A/'receipt.json'); assert rec['source_pins_sha256']==ph and rec['HTTP_requests']==count and rec['completion_tokens']==32*count
 part=load(A/'participation.json'); text=(A/'server.log').read_text(); gen=[ast.literal_eval(x.split('Generation: ',1)[1]) for x in text.splitlines() if 'Generation: ' in x]
 assert len(gen)==count and gen==part['Generation_records']
 data=(A/'server.log').read_bytes(); h=hashlib.sha256(); matched=None; offset=0
 for line in data.splitlines(keepends=True):
  h.update(line); offset+=len(line)
  if h.hexdigest()==part['original_log_sha256']:matched=offset
 assert matched is not None
 assert b'Generation: ' not in data[matched:]
 log_prefix_proofs.append({'arm':A.name,'pre_cleanup_log_prefix_bytes':matched,'final_log_bytes':len(data),'appended_bytes':len(data)-matched,'pre_cleanup_sha256':part['original_log_sha256'],'final_sha256':sha(A/'server.log'),'no_appended_Generation_records':True})
 assert all(x['cached_prompt_tokens']==0 and x['prefix_source'] is None for x in gen)
 assert part['backend']=='m4_max' and part['draft'] is None and 'Serving backend=m4_max,' in text
 logs+=count
 pid=load(A/'owned-pid.json'); pre=load(A/'official-preimport.json')
 assert pre['pid']==pid['pid'] and pre['ppid']==pid['ppid'] and pre['files_checked']==196 and pre['mlx_not_imported']
 assert pre['source_native_tree_sha256']=='db3039dcbdc282c15795a6a9ce5e43c5afcdbab6d755a8fea6050d40a44f414e'
 assert pre['controls']=={'MLX_MAX_OPS_PER_BUFFER':'50','MLX_MAX_MB_PER_BUFFER':'50','MLX_BFS_MAX_WIDTH':'20','MLX_METAL_FAST_SYNCH':'0'}
 clean=load(A/'cleanup.json'); assert clean['failure'] is None and clean['cleanup_errors']==[]
 ex=clean['owned_exit']; assert ex==rec['owned_exit'] and ex['pid']==pid['pid'] and ex['wait_completed'] and ex['returncode']==-15;absent(ex['exact_pid_absence'])
 birth=load(A/'owned-birth.json')['identity']
 for name in ['owned-ready-identity.json','owned-pre-cleanup-identity.json']:
  ident=load(A/name);assert ident['same_start'] and ident['direct_Popen_child_live'] and ident['initial_identity_words']==birth==ident['current_identity_words']
 assert all(x['decision']=='SAME_IDENTITY_SIGNAL' and x['current_identity_sha256']==ex['recorded_birth_identity_sha256'] for x in ex['signal_identity_decisions'])
 pa=load(A/'port-absence.json');assert pa['absent'] and pa['localhost_only'] and all(x['connect_ex']==errno.ECONNREFUSED for x in pa['checks'])
 cleanup_pids.append(pid['pid']); ports.append(pa['port']);resource(A/'resources.json','performance_arm' if A!=Q else 'qualification_arm')

arms=[]
for x in suite['arms']:
 A=D/x['label']; C=x['concurrency']; gr=[group(A/f'C{C}-{phase}-{i}.json','performance',x['label']) for phase in ('warm','measure') for i in range(3)]
 formal=[r for r in gr if r['formal']]; xx=st.median(r['latency_p50_ms'] for r in formal);yy=st.median(r['total_throughput_tok_s'] for r in formal)
 close(xx,x['latency_p50_ms']);close(yy,x['total_throughput_tok_s']);armcheck(A,6*C)
 arms.append({'label':x['label'],'C':C,'replica':x['replica'],'latency_p50_ms':xx,'total_throughput_tok_s':yy})
assert len(arms)==9 and len({x['label'] for x in arms})==9
configs={str(c):{k:st.median(a[k] for a in arms if a['C']==c) for k in ('latency_p50_ms','total_throughput_tok_s')} for c in (1,2,4)}
for c in configs:
 for k in configs[c]:close(configs[c][k],suite['medians'][c][k])
resource(D/'campaign-resources.json','performance_campaign')
for f in sorted(Q.glob('*C*.json')):group(f,'qualification','qualification')
armcheck(Q,8)
assert len(requests)==134 and len(groups)==58 and logs==134
assert sum(x['kind']=='performance' for x in requests)==126 and sum(x['formal'] for x in groups)==27
assert sum(x['C'] for x in groups if x['kind']=='performance' and x['formal'])==63
for i in range(1,9):
 gap=load(D/f'cool-gap-{i:02}.json');a,b=gap['start_monotonic_s'],gap['end_monotonic_s'];assert gap['requested_seconds']==60 and b-a>=60 and gap['outside_HTTP_clocks'] and gap['prior_owned_cleanup_passed'] and gap['adaptive'] is False
 prev=[g for g in groups if g['arm']==arms[i-1]['label']];nxt=[g for g in groups if g['arm']==arms[i]['label']]
 assert max(g['terminal_s'] for g in prev)<a<b<min(g['start_s'] for g in nxt)
 assert not any(a<end and b>start for start,end in windows)

releases=[]
for root,receipt,expected_sha in [(D.parent,D/'suite-complete.json','e889f12091588f30abd8b3d729883375208722e3febc56d7746fc92452f02a47'),(Q.parent,Q/'receipt.json','edb43e2a7b87e73d8c9beb199cb8986c7161baa7e3e5eaf8ff0fb0943455a391')]:
 rel=load(root/'external-release.json'); own=load(root/'owner-receipt.json')
 assert sha(root/'external-release.json')==expected_sha and Path(rel['receipt'])==receipt and rel['receipt_sha256']==sha(receipt)
 assert rel['owner_receipt_sha256']==sha(root/'owner-receipt.json') and rel['resource_sha256']==sha(root/'external-release-resource-raw.json')
 assert rel['actual_tool_exit_code']==0 and own['stage']=='reaped' and own['returncode']==0 and own['waited_reaped'];absent(own['driver_absence'])
 expected_pids={own['owner_pid'],own['driver_pid']}|({load(D/a['label']/'owned-pid.json')['pid'] for a in arms} if root==D.parent else {load(Q/'owned-pid.json')['pid']})
 assert {x['pid'] for x in rel['strict_exact_pid_absences']}==expected_pids
 for a in rel['strict_exact_pid_absences']:absent(a)
 expected_ports={load(D/a['label']/'port-absence.json')['port'] for a in arms} if root==D.parent else {load(Q/'port-absence.json')['port']}
 assert {x['port'] for x in rel['ports']}==expected_ports
 for p in rel['ports']:assert p['absent'] and all(x['connect_ex']==errno.ECONNREFUSED for x in p['checks'])
 resource(root/'external-release-resource-raw.json','performance_release' if root==D.parent else 'qualification_release')
 releases.append({'kind':'performance' if root==D.parent else 'qualification','sha256':expected_sha,'pids':sorted(expected_pids),'ports':sorted(expected_ports),'reuse_saved_root_live_proof':True})
for k in ['receipt','external_release']:
 assert Path(suite['qualification'][k])==Q/('receipt.json' if k=='receipt' else '../external-release.json') or Path(suite['qualification'][k]).resolve()==(Q/('receipt.json' if k=='receipt' else '../external-release.json')).resolve()
 assert suite['qualification'][k+'_sha256']==sha(suite['qualification'][k])
assert all(not any(a<=r['monotonic_s']<=b for a,b in windows) for r in resource_rows)
counts={kind:sum(r['kind']==kind for r in resource_rows) for kind in sorted({r['kind'] for r in resource_rows})}
assert counts=={'performance_arm':135,'performance_campaign':27,'performance_release':1,'qualification_arm':11,'qualification_release':1}
assert len({r['Swapouts'] for r in resource_rows})==1 and len({r['swap_used_mb'] for r in resource_rows})==1
# A read-only byte seal; no raw logs, process argv, environment, or auth exported.
seal=[{'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)} for base in (D.parent,Q.parent) for p in sorted(base.rglob('*')) if p.is_file()]
put('log-prefix-proofs.json',log_prefix_proofs)
put('raw-source-seal.json',{'files':seal,'source_files':[{'path':str(S/n),'sha256':sha(S/n)} for n in ('client.py','qualify.py','driver.py','pins.json')]})
put('recomputed-rows.json',{'arms':arms,'config_medians':configs,'groups':groups,'requests':requests})
verification={'status':'PASS_SAVED_EVIDENCE_REPLAY_WITH_SCOPE_LIMITS','performance':{'fresh_arms':9,'groups':54,'HTTP_requests':126,'output_tokens':4032,'formal_groups':27,'formal_requests':63,'formal_tokens':2016},'qualification':{'groups':4,'HTTP_requests':8,'output_tokens':256,'OpenAI_requests':1,'Ollama_requests':7},'resources':{'counts':counts,'total':len(resource_rows),'power':'AC Power','thermal':0,'pressure':1,'Swapouts':resource_rows[0]['Swapouts'],'swap_used_mb':resource_rows[0]['swap_used_mb'],'all_outside_HTTP_windows':True,'continuous_monitoring_claim':False},'cooling':{'fixed_gaps':8,'minimum_seconds':60,'outside_HTTP_windows':True},'participation':{'Generation_records':logs,'performance_cached0_prefixNone':126,'qualification_cached0_prefixNone':8,'preimport_files_checked_each':196,'source_pins_count':len(pins['files']),'source_pins_sha256':ph,'source_full_tree_reaudit':False},'config_medians':configs,'releases':releases,'limits':['Visible re-tokenized IDs and usage are checked; no internal raw token IDs.','Pre-POST t0 is not wire arrival; queue wait included, exact queue duration unknown; no GPU batching claim.','AC snapshots do not prove no transient between samples.','Separate V128/V130 campaigns are descriptive comparison, not paired gain.','Per-arm no_new_performance_point=true is inherited qualification metadata; actual formal group clocks are verified.']}
put('verification.json',verification)
print(json.dumps({'status':verification['status'],'counts':verification['performance'],'resources':counts,'config_medians':configs},ensure_ascii=False))
