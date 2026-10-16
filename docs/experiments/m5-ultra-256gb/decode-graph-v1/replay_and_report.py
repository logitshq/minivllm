"""Independent stdlib replay; no model/runtime imports."""
import hashlib,json,math,statistics,subprocess,shutil
from pathlib import Path
R=Path('/Users/ryanli/code/minivllm'); B=R/'.artifacts/m5-ultra-256gb'; F=B/'graph-frontier-v1'
O=Path('/Users/ryanli/Documents/Codex/2026-10-15/chec/outputs')
def read(p):return json.loads(p.read_text())
def write(p,d):p.write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n')
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  while b:=f.read(1<<20):h.update(b)
 return h.hexdigest()
def close(a,b):assert math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-10),(a,b)
complete=read(F/'complete.json');frozen=read(B/'graph-inputs-v1/frozen.json')
assert len(complete['rows'])==len(frozen['configs'])==24
for p,h in frozen['files'].items():assert digest(Path(p))==h,p
qual=read(B/'graph-qual-v1/complete.json');assert qual['status']=='PASS' and qual['paired_forwards']==103
for q in read(B/'graph-qual-v1/checks.json'):assert q['exact'] and q['reference_sha256']==q['candidate_sha256'] and len(q['reference_sha256'])==129
sched=read(B/'graph-qual-v1/scheduler.json');assert sched['reference']==sched['candidate']
work=read(B/'fullqual-build2/workload.json');freegen=read(B/'graph-free-generation-v1/complete.json');assert freegen['status']=='PASS';expected_ids={r['B']:r['reference_ids'] for r in freegen['results']};assert all(r['reference_ids']==r['candidate_ids'] for r in freegen['results']);rows=[];samples=[];reqs=0;measured=0;native=0;absences=[]
for recorded in complete['rows']:
 p=Path(recorded['path']);assert digest(p)==recorded['sha256'];d=read(p)
 assert [d['concurrency'],d['pair'],d['engine']] in frozen['configs']
 for phase,groups in d['groups'].items():
  assert len(groups)==3
  for g in groups:
   requests=g['requests'];assert len(requests)==d['concurrency']
   reqs+=len(requests);measured+=len(requests) if phase=='measure' else 0
   for req in requests:
    assert req['usage']['prompt_tokens']==19 and req['usage']['completion_tokens']==32 and req['finish_reason']=='length'
    assert req['usage'].get('prompt_tokens_details',{}).get('cached_tokens',0)==0
    assert len(req['visible_roundtrip_ids'])==32
    close(req['client_total_s'],req['done_perf_counter_s']-req['start_perf_counter_s'])
    close(req['client_ttft_text_s'],req['first_text_perf_counter_s']-req['start_perf_counter_s'])
    close(req['x_client_estimated_interactivity_tok_s_user'],(32-req['first_chunk_reconstructed_tokens'])/(req['done_perf_counter_s']-req['first_text_perf_counter_s']))
    if d['engine'] in ('base','graph'):
     assert req['output_ids_when_available'] in expected_ids[d['concurrency']];assert req['output_ids_when_available']==req['visible_roundtrip_ids'];native+=1
   close(g['y_total_output_throughput_tok_s_mac'],len(requests)*32/(max(r['done_perf_counter_s'] for r in requests)-min(r['start_perf_counter_s'] for r in requests)))
   close(g['x_client_estimated_interactivity_tok_s_user'],statistics.median(r['x_client_estimated_interactivity_tok_s_user'] for r in requests))
 close(d['x'],statistics.median(g['x_client_estimated_interactivity_tok_s_user'] for g in d['groups']['measure']))
 close(d['y'],statistics.median(g['y_total_output_throughput_tok_s_mac'] for g in d['groups']['measure']))
 close(d['TTFT_s'],statistics.median(statistics.median(r['client_ttft_text_s'] for r in g['requests']) for g in d['groups']['measure']))
 if d['engine'] in ('base','graph'):
  ident=d['runtime_identity'];assert ident['compile_decode']==(d['engine']=='graph')
  assert ident['allocator_pool_gib']==(2 if d['engine']=='graph' else 0)
  assert ident['norm_reuse_enabled'] is False and ident['gates_compile_enabled'] is True
 samples+=d['resources'];rows.append(recorded)
 assert d['process_exit']['waited_reaped'];absences.append(d['process_exit']['pid'])
absences.append(complete['owner_pid'])
for pid in absences:
 p=subprocess.run(['/bin/ps','-p',str(pid),'-o','pid='],capture_output=True,text=True);assert p.returncode==1 and not p.stdout
write(F/'posttool-release.json',{'pids':absences,'all_strictly_absent':True})
assert all(s['pressure_level_value']==1 and s['processinfo_thermal_metadata']['code'] in (0,1) and s['swap_used_mb']==0 for s in samples)
summary=[]
for C in (1,2,4):
 for engine in ('base','graph','omlx','lithos'):
  selected=[r for r in rows if r['C']==C and r['engine']==engine]
  if selected:
   assert len(selected)==3
   summary.append({'C':C,'engine':engine,**{k:statistics.median(r[k] for r in selected) for k in ('x','y','TTFT_s')}})
def value(C,engine):return next(s for s in summary if s['C']==C and s['engine']==engine)
pairs=[]
for pair in (1,2,3):
 old=next(r for r in rows if r['C']==1 and r['pair']==pair and r['engine']=='base');new=next(r for r in rows if r['C']==1 and r['pair']==pair and r['engine']=='graph')
 pairs.append({'pair':pair,'x_pct':100*(new['x']/old['x']-1),'y_pct':100*(new['y']/old['y']-1)})
x_gain=100*(value(1,'graph')['x']/value(1,'base')['x']-1);y_gain=100*(value(1,'graph')['y']/value(1,'base')['y']-1)
adopt=all(p['y_pct']>0 for p in pairs) and x_gain>=0 and y_gain>=1
resources={'snapshots':len(samples),'pressure_levels':sorted(set(s['pressure_level_value'] for s in samples)),'thermal_codes':sorted(set(s['processinfo_thermal_metadata']['code'] for s in samples)),'swap_used_mb_max':max(s['swap_used_mb'] for s in samples),'counter_ranges':{k:[min(s['vm_counters'][k] for s in samples),max(s['vm_counters'][k] for s in samples)] for k in ('Pageouts','Swapouts')}}
replay={'status':'PASS','processes':24,'requests':reqs,'measured_requests':measured,'tokens_including_warm':reqs*32,'measured_tokens':measured*32,'native_requests_with_exact_raw_ids':native,'pins_verified':len(frozen['files']),'resources':resources,'owner_servers_absent':True}
write(F/'replay.json',replay)
result={'change':'Whole-step graph reuse + 2 GiB allocator pool; gates on, V19 off','source_commit':'2612e68','qualification':qual,'free_generation_validation':freegen,'summary':summary,'all_processes':rows,'paired_improvements':pairs,'aggregate_x_improvement_pct':x_gain,'aggregate_y_improvement_pct':y_gain,'bounded_adoption_pass':adopt,'replay':replay,'scope':'M5 Ultra 256GiB, fixed 27B NVFP4 revision, 19 input/32 output, greedy normal EOS, no drafter/prefix cache, C1/C2/C4; optimized used length<=256; warm steady-state HTTP boundaries','limits':['One prompt and short contexts; no broad workload or long-context win claim','C means HTTP concurrency; actual GPU batch differs','Lithos is locally adapted profile, C1 only','Eager baseline has allocator pool0; factorial screen separates graph and allocation pool effects','Other chat downloaded large models; resource counters retained','compile_decode remains default off; LLMEngine allocator default was already 2GiB, while prior HTTP harnesses forced 0']}
write(O/'m5-ultra-graph-results.json',result)
lines=['M5 Ultra 256GB：解码图复用实验','', '代码：2612e68（本地提交，未推送）。核心新增65行左右，默认关闭。','方案：复用整步解码图 + 2 GiB MLX内存分配池；不缓存prompt，不用draft。','支持范围：固定27B NVFP4/BF16、MLX0.32.3、B1/B2/B4、已用长度<=256；其他情况回原路径。','',f'正确性：103组完整前向，logits及128个状态数组逐位相同；动态批次、stop、边界回退、权重替换/恢复通过。',f'独立回放：{reqs}次请求（含预热）、{measured}次计时请求；{native}次本机请求的原始IDs与同batch原版逐token一致。','', '每格为3个新进程各自中位数的中位数：','C | 引擎 | 完整请求吞吐 tok/s | 首字延迟 ms | 交付速率估计 tok/s']
for s in summary:lines.append(f"{s['C']} | {s['engine']} | {s['y']:.2f} | {1000*s['TTFT_s']:.2f} | {s['x']:.2f}")
lines+=['',f'组合方案对eager/pool0：C1总吞吐 +{y_gain:.2f}%，交付速率 +{x_gain:.2f}%。','三对总吞吐变化：'+', '.join(format(p['y_pct'],'+.2f')+'%' for p in pairs),f'有界采用检查：{adopt}（不代表全面胜过外部引擎）。','', '探索记录：分层编译有小幅收益；整步编译保留。扩大发送buffer只作探索，正式比较仍用50/50/20/0。','矩阵padding切换路径没有稳定收益且出现数值差异；自制窄QMV词表投影出现1项差异，均未采用。','内存池收益和图复用收益来自独立因子screen；不可把全部加速归因于编译。','预热请求完整保留；稳定态结果不包含首次建图成本。','本次仍是单个19-token prompt、32-token输出的短请求，不推广到长上下文或全部任务。','不同batch的MLX数值路径可能不同；已独立验证B1/B2/B4自由生成与同batch原版一致，并覆盖全部HTTP原始IDs。','Lithos是本地M5 Ultra适配配置；其OpenAI接口仅C1。','后台另一个chat持续下载大模型；没有同时进行它的GPU benchmark。',f'资源：{json.dumps(resources,ensure_ascii=False)}','所有24个服务和owner均退出并核实PID不存在。','', '原始证据：'+str(F)]
(O/'m5-ultra-graph-report.txt').write_text('\n'.join(lines)+'\n')
archive=R/'docs/experiments/m5-ultra-256gb/decode-graph-v1';archive.mkdir(parents=True)
for name in ('m5-ultra-graph-results.json','m5-ultra-graph-report.txt'):shutil.copyfile(O/name,archive/name)
paths=[]
for name in ('graph-frontier-v1','graph-qual-v1','graph-qual-v1-owner','graph-factor-v1','graph-factor-v1-owner','graph-factor-wide-v1','graph-factor-wide-v1-owner','compile-probe-v1','whole-probe-v1','matrix-probe-v1','narrow-probe-v1','graph-exploration-source-v1','graph-inputs-v1','graph-source-v1','graph-free-generation-v1','graph-free-generation-v1-owner'):
 paths += [p for p in (B/name).rglob('*') if p.is_file() and p.suffix not in ('.pyc','.db','.sqlite') and not p.is_symlink()]
write(O/'m5-ultra-graph-evidence-manifest.json',{'files':{str(p):digest(p) for p in paths},'scope':'Result/source logs only; model symlinks excluded'})
print(json.dumps({'summary':summary,'pairs':pairs,'aggregate_gain':y_gain,'adoption':adopt,'replay':replay},indent=2))
