"""Independent stdlib replay; no model/runtime imports."""
import hashlib,json,math,statistics,subprocess,shutil
from pathlib import Path
R=Path('/Users/ryanli/code/minivllm'); B=R/'.artifacts/m5-ultra-256gb'; F=B/'qmv-frontier-v1'
O=Path('/Users/ryanli/Documents/Codex/2026-10-15/chec/outputs')
def read(p):return json.loads(p.read_text())
def write(p,d):p.write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n')
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  while b:=f.read(1<<20):h.update(b)
 return h.hexdigest()
def close(a,b):assert math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-10),(a,b)
complete=read(F/'complete.json');frozen=read(B/'qmv-inputs-v1/frozen.json')
assert len(complete['rows'])==len(frozen['configs'])==24
for p,h in frozen['files'].items():assert digest(Path(p))==h,p
qual=read(B/'qmv-qual-v1/complete.json');assert qual['status']=='PASS' and qual['paired_forwards']==214
for q in read(B/'qmv-qual-v1/checks.json'):assert q['exact'] and q['reference_sha256']==q['candidate_sha256'] and len(q['reference_sha256'])==129
sched=read(B/'qmv-qual-v1/scheduler.json');assert sched['reference']==sched['candidate']
work=read(B/'fullqual-build2/workload.json');freegen=read(B/'qmv-free-generation-v1/complete.json');assert freegen['status']=='PASS';expected_ids={r['B']:r['reference_ids'] for r in freegen['results']};assert all(r['reference_ids']==r['candidate_ids'] for r in freegen['results']);rows=[];samples=[];reqs=0;measured=0;native=0;absences=[]
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
    if d['engine'] in ('base','fused'):
     assert req['output_ids_when_available'] in expected_ids[d['concurrency']];assert req['output_ids_when_available']==req['visible_roundtrip_ids'];native+=1
   close(g['y_total_output_throughput_tok_s_mac'],len(requests)*32/(max(r['done_perf_counter_s'] for r in requests)-min(r['start_perf_counter_s'] for r in requests)))
   close(g['x_client_estimated_interactivity_tok_s_user'],statistics.median(r['x_client_estimated_interactivity_tok_s_user'] for r in requests))
 close(d['x'],statistics.median(g['x_client_estimated_interactivity_tok_s_user'] for g in d['groups']['measure']))
 close(d['y'],statistics.median(g['y_total_output_throughput_tok_s_mac'] for g in d['groups']['measure']))
 close(d['TTFT_s'],statistics.median(statistics.median(r['client_ttft_text_s'] for r in g['requests']) for g in d['groups']['measure']))
 if d['engine'] in ('base','fused'):
  ident=d['runtime_identity'];assert ident['compile_decode'] is True and ident['fuse_decode'] is True
  assert ident['allocator_pool_gib']==2
  assert ident['norm_reuse_enabled'] is False and ident['gates_compile_enabled'] is True
 samples+=d['resources'];rows.append(recorded)
 assert d['process_exit']['waited_reaped'];absences.append(d['process_exit']['pid'])
absences.append(complete['owner_pid'])
for pid in absences:
 p=subprocess.run(['/bin/ps','-p',str(pid),'-o','pid='],capture_output=True,text=True);assert p.returncode==1 and not p.stdout
write(F/'posttool-release.json',{'pids':absences,'all_strictly_absent':True})
assert all(s['pressure_level_value']==1 and s['processinfo_thermal_metadata']['code'] in (0,1) for s in samples)
summary=[]
for C in (1,2,4):
 for engine in ('base','fused','omlx','lithos'):
  selected=[r for r in rows if r['C']==C and r['engine']==engine]
  if selected:
   assert len(selected)==3
   summary.append({'C':C,'engine':engine,**{k:statistics.median(r[k] for r in selected) for k in ('x','y','TTFT_s')}})
def value(C,engine):return next(s for s in summary if s['C']==C and s['engine']==engine)
pairs=[]
for pair in (1,2,3):
 old=next(r for r in rows if r['C']==1 and r['pair']==pair and r['engine']=='base');new=next(r for r in rows if r['C']==1 and r['pair']==pair and r['engine']=='fused')
 pairs.append({'pair':pair,'x_pct':100*(new['x']/old['x']-1),'y_pct':100*(new['y']/old['y']-1)})
x_gain=100*(value(1,'fused')['x']/value(1,'base')['x']-1);y_gain=100*(value(1,'fused')['y']/value(1,'base')['y']-1)
adopt=all(p['y_pct']>0 for p in pairs) and x_gain>=0 and y_gain>=1
resources={'snapshots':len(samples),'pressure_levels':sorted(set(s['pressure_level_value'] for s in samples)),'thermal_codes':sorted(set(s['processinfo_thermal_metadata']['code'] for s in samples)),'swap_used_mb_max':max(s['swap_used_mb'] for s in samples),'counter_ranges':{k:[min(s['vm_counters'][k] for s in samples),max(s['vm_counters'][k] for s in samples)] for k in ('Pageouts','Swapouts')}}
replay={'status':'PASS','processes':24,'requests':reqs,'measured_requests':measured,'tokens_including_warm':reqs*32,'measured_tokens':measured*32,'native_requests_with_exact_raw_ids':native,'pins_verified':len(frozen['files']),'resources':resources,'owner_servers_absent':True}
write(F/'replay.json',replay)
result={'change':'Two-row NVFP4 QMV with FP contraction off; both native variants use projection/conv fusion and pool2','source_commit':frozen['source_commit'],'qualification':qual,'free_generation_validation':freegen,'summary':summary,'all_processes':rows,'paired_improvements':pairs,'aggregate_x_improvement_pct':x_gain,'aggregate_y_improvement_pct':y_gain,'bounded_adoption_pass':adopt,'replay':replay,'scope':'M5 Ultra 256GiB, fixed 27B NVFP4 revision, 19 input/32 output, greedy normal EOS, no drafter/prefix cache, C1/C2/C4; optimized used length<=256; warm steady-state HTTP boundaries','limits':['One prompt and short contexts; no broad workload or long-context win claim','C means HTTP concurrency; actual GPU batch differs','Lithos is locally adapted profile, C1 only','Baseline is frozen previous projection/conv fusion implementation; both native variants pool2','Separate large-model job completed before this series; residual nonzero swap is recorded','fuse_decode and compile_decode remain default off; fusion uses extra derived weights; cold pack/trace cost excluded from measured median']}


parameters=read(B/'qmv-qual-v1/parameters.json')
assert parameters['unchanged'] and read(B/'qmv-qual-v1/parameters-before.json')==read(B/'qmv-qual-v1/parameters-after.json')
assert read(B/'qmv-kernel-v1/complete.json')['status']=='PASS'
qmv_checks=read(B/'qmv-kernel-v1/qmv-checks.json')
assert len(qmv_checks)==771 and all(c['exact'] and c['reference']==c['candidate'] for c in qmv_checks)
result['parameter_validation']=parameters
result['kernel_validation']=read(B/'qmv-kernel-v1/complete.json')
result['against_external_C1']={engine:{'y_pct':100*(value(1,'fused')['y']/value(1,engine)['y']-1),'x_pct':100*(value(1,'fused')['x']/value(1,engine)['x']-1)} for engine in ('omlx','lithos')}
write(O/'m5-ultra-qmv-results.json',result)
lines=['M5 Ultra 256GB：窄 NVFP4 矩阵向量 kernel','',
       '代码：'+frozen['source_commit']+'（本地提交，未推送）。',
       '方案：在投影/卷积融合之上，NVFP4每个SIMD处理两行，关闭乘加收缩，保留原归约顺序。',
       '原始量化权重保持不变；追加派生权重 '+format(parameters['derived_bytes']/2**30,'.3f')+' GiB，参数替换时刷新。',
       '精确范围：MLX0.32.3、M5 Ultra、27B NVFP4/BF16、B1、used length<=256。B2/B4保留原图路径；其余回退。',
       '默认关闭；首次pack和编译成本保留在warm记录，不计入稳定态中位数。','',
       '正确性：214组完整前向，每组logits及128个状态数组逐位相同；16个GDN案例及771个QMV案例；参数内容哈希不变。',
       '覆盖4个提示词上下文、权重/scale替换恢复、边界、批次变化、stop和lookahead。',
       '自由生成：B1/B2/B4同batch原版逐token相同，另3组提示词的B1自由生成相同。',
       f'正式测量：{reqs}次请求（含预热），{measured}次计时请求；{native}次native原始IDs校验通过。','',
       '每格是3个新进程各自3次计时中位数的中位数：',
       'C | 引擎 | 完整请求吞吐 tok/s | 首字延迟 ms | 客户端交付速率估计 tok/s']
for s in summary:lines.append(f"{s['C']} | {s['engine']} | {s['y']:.2f} | {1000*s['TTFT_s']:.2f} | {s['x']:.2f}")
lines += ['',f'对上一版投影/卷积融合：C1完整请求吞吐 {y_gain:+.2f}%，交付速率 {x_gain:+.2f}%。',
          '三对完整请求吞吐变化：'+', '.join(f"{p['y_pct']:+.2f}%" for p in pairs),
          f'有界采用检查：{adopt}。']
for engine,values in result['against_external_C1'].items():
    lines.append(f"C1相对{engine}：完整请求吞吐 {values['y_pct']:+.2f}%，交付速率 {values['x_pct']:+.2f}%。")
lines += ['', '性能范围是单个19-token prompt、32-token输出，不是长上下文或广泛工作负载排名。',
          'C为HTTP并发数；官方实际GPU batch未知。Lithos使用本地M5 Ultra适配profile，其OpenAI接口仅支持C1。',
          'oMLX/Lithos官方没有暴露原始IDs；二者检查visible roundtrip、usage、缓存和正常EOS。',
          '所有配置相同MLX控制50/50/20/0；两个native版本均graph+2GiB分配池；无prompt cache、无draft。',
          'QMV探索：启用FP收缩时词表头有1项差异，关闭后单算子及完整模型对照一致；所有探索均保留。',
          '资源：'+json.dumps(resources,ensure_ascii=False),
          '24个服务及owner退出后均核实PID不存在。','原始证据：'+str(F)]
(O/'m5-ultra-qmv-report.txt').write_text('\n'.join(lines)+'\n')
archive=R/'docs/experiments/m5-ultra-256gb/decode-qmv-v1';archive.mkdir(parents=True,exist_ok=True)
for name in ('m5-ultra-qmv-results.json','m5-ultra-qmv-report.txt'):shutil.copyfile(O/name,archive/name)
print(json.dumps({'summary':summary,'pairs':pairs,'against_external_C1':result['against_external_C1'],'adoption':adopt,'replay':replay,'parameters':parameters},indent=2))
