"""Independent stdlib replay of client timestamps, counts, pins and dominance."""
import argparse,hashlib,json,math,statistics,subprocess
from pathlib import Path

R=Path(__file__).resolve().parents[3];B=R/'.artifacts/m5-ultra-256gb'
ap=argparse.ArgumentParser();ap.add_argument('--frontier',type=Path,default=B/'pareto-batch-frontier-v1')
ap.add_argument('--c1-frontier',type=Path,default=B/'pareto-mode-frontier-v1')
ap.add_argument('--capped-frontier',type=Path,default=B/'pareto-capped-frontier-v1')
ap.add_argument('--retirement-frontier',type=Path)
ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();F=a.frontier;O=a.out;O.mkdir(parents=True,exist_ok=True)
def read(p):return json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        while chunk:=f.read(1<<20):h.update(chunk)
    return h.hexdigest()
def close(x,y):assert math.isclose(x,y,rel_tol=1e-10,abs_tol=1e-10),(x,y)
def absent(pid):
    r=subprocess.run(['/bin/ps','-p',str(pid),'-o','pid='],capture_output=True,text=True)
    assert r.returncode==1 and not r.stdout and not r.stderr,pid
    return pid
def dominates(a,b):return a['x']>=b['x'] and a['y']>=b['y'] and (a['x']>b['x'] or a['y']>b['y'])

experiments=[];pinned={};entries=[];expected_keys=set()
roots=[('main',F),('modes',a.c1_frontier),('cap',a.capped_frontier)]
if a.retirement_frontier:roots.append(('retirement',a.retirement_frontier))
for label,root in roots:
    complete=read(root/'complete.json');frozen=read(root/'inputs/frozen.json')
    assert not (root/'failure.json').exists()
    for p,h in frozen['files'].items():
        assert p not in pinned or pinned[p]==h,p
        pinned[p]=h
    experiments.append((label,root,complete,frozen))
    entries.extend((label,root,frozen,r) for r in complete['rows'])
    expected_keys.update((label,*c) for c in frozen['configs'])
assert len({f['source_commit'] for _,_,_,f in experiments})==1
for p,h in pinned.items():assert digest(p)==h,p
qual=read(B/'pareto-batch-qual-v1/complete.json');kernel=read(B/'pareto-batch-kernel-v1/complete.json');free=read(B/'pareto-batch-free-v1/complete.json')
assert all(x['status']=='PASS' for x in (qual,kernel,free))
graph_only=read(B/'pareto-graph-only-v2/complete.json');assert graph_only['status']=='PASS'
assert all(all(r['bit_equal']) for r in read(B/'pareto-graph-only-v2/checks.json'))
assert all(r['exact'] and r['reference']==r['candidate'] for r in read(B/'pareto-batch-kernel-v1/checks.json'))
assert all(r['exact'] and all(r['bit_equal']) for r in read(B/'pareto-batch-qual-v1/checks.json'))
for name in ('scheduler','staggered'):
    s=read(B/f'pareto-batch-qual-v1/{name}.json');assert s['reference']==s['candidate']
assert read(B/'pareto-batch-qual-v1/parameters-before.json')==read(B/'pareto-batch-qual-v1/parameters-after.json')
assert all(r['reference_ids']==r['candidate_ids'] for r in free['results'])
known_ids={tuple(ids) for r in free['results'] for ids in r['reference_ids']}
rows=[];samples=[];pids=[];requests=measured=native=known_native=0;seen=set()
for experiment,root,configuration,recorded in entries:
    path=Path(recorded['path']);assert digest(path)==recorded['sha256'];d=read(path)
    key=(d['concurrency'],d['pair'],d['engine']);full_key=(experiment,*key)
    assert full_key not in seen and list(key) in configuration['configs'];seen.add(full_key)
    cleanup=read(path.parent/'cleanup.json');assert cleanup=={'first_failure':None,'cleanup_errors':[]}
    assert d['environment']==configuration['environment']
    if d['engine'].startswith('omlx'):
        settings=read(path.parent/'state/settings.json')
        assert settings['server']['burst_decode_mode']==('aggressive' if d['engine']=='omlx_aggressive' else 'balanced')
    for phase,groups in d['groups'].items():
        assert len(groups)==3
        for g in groups:
            rs=g['requests'];assert len(rs)==d['concurrency']
            for r in rs:
                requests+=1;measured+=phase=='measure'
                assert r['usage']['prompt_tokens']==19 and r['usage']['completion_tokens']==32 and r['finish_reason']=='length'
                assert r['usage'].get('prompt_tokens_details',{}).get('cached_tokens',0)==0
                assert len(r['visible_roundtrip_ids'])==32 and r['first_chunk_prefix_verified'] and r['full_visible_roundtrip_count_matches_usage']
                close(r['client_total_s'],r['done_perf_counter_s']-r['start_perf_counter_s'])
                close(r['client_ttft_text_s'],r['first_text_perf_counter_s']-r['start_perf_counter_s'])
                close(r['x_client_estimated_interactivity_tok_s_user'],(32-r['first_chunk_reconstructed_tokens'])/(r['done_perf_counter_s']-r['first_text_perf_counter_s']))
                if d['engine']=='fused':
                    assert r['output_ids_when_available']==r['visible_roundtrip_ids'];native+=1
                    known_native+=tuple(r['output_ids_when_available']) in known_ids
            close(g['y_total_output_throughput_tok_s_mac'],sum(r['usage']['completion_tokens'] for r in rs)/(max(r['done_perf_counter_s'] for r in rs)-min(r['start_perf_counter_s'] for r in rs)))
            close(g['x_client_estimated_interactivity_tok_s_user'],statistics.median(r['x_client_estimated_interactivity_tok_s_user'] for r in rs))
    close(d['x'],statistics.median(g['x_client_estimated_interactivity_tok_s_user'] for g in d['groups']['measure']))
    close(d['y'],statistics.median(g['y_total_output_throughput_tok_s_mac'] for g in d['groups']['measure']))
    close(d['TTFT_s'],statistics.median(statistics.median(r['client_ttft_text_s'] for r in g['requests']) for g in d['groups']['measure']))
    if d['engine']=='fused':
        identity=d['runtime_identity'];assert identity['compile_decode'] and identity['fuse_decode'] and identity['allocator_pool_gib']==2
        assert identity['manifest_all_sources_match'] and identity['gates_compile_enabled'] and not identity['norm_reuse_enabled']
        if experiment=='cap':assert d['concurrency']==64 and identity['server_max_seqs']==32
    if d['engine']=='lithos':
        assert all(x['cached_prompt_tokens']==0 and x['verify_tokens']==1 for x in d['lithos_native_metrics'])
    samples+=d['resources'];rows.append({**recorded,'experiment':experiment,
        'engine':{'cap':'fused_capped','retirement':'fused_retired'}.get(experiment,recorded['engine'])})
    assert d['process_exit']['waited_reaped'];pids.append(absent(d['process_exit']['pid']))
assert seen==expected_keys
for _,_,complete,_ in experiments:pids.append(absent(complete['owner_pid']))
for name in ('pareto-batch-qual-v1','pareto-batch-kernel-v1','pareto-batch-free-v1','pareto-graph-only-v2'):
    owner=B/(name+'-owner');pids.append(absent(read(owner/'attempt.json')['owner_pid']))
    release=read(owner/'release.json');assert release['waited_reaped'] and release['strict_absence'];pids.append(absent(release['pid']))
retirement_qual=None
if a.retirement_frontier:
    retirement_qual=read(B/'retire-batch-qual-v1/complete.json');assert retirement_qual['status']=='PASS'
    for case in read(B/'retire-batch-qual-v1/checks.json'):
        assert case['exact'] and case['reference']['ids']==case['candidate']['ids']
        for old,new in zip(case['reference']['rows'],case['candidate']['rows']):
            for key in ('outputs','batch','lens','requests','states'):assert old[key]==new[key]
    owner=B/'retire-batch-qual-v1-owner';pids.append(absent(read(owner/'attempt.json')['owner_pid']))
    release=read(owner/'release.json');assert release['waited_reaped'] and release['strict_absence'];pids.append(absent(release['pid']))
write(F/'posttool-release.json',{'all_strictly_absent':True,'pids':pids})
assert all(s['pressure_level_value']==1 and s['processinfo_thermal_metadata']['code'] in (0,1) for s in samples)
all_rows=rows
earlier_c1=[r for r in rows if r['experiment']=='main' and r['C']==1]
# This selection rule was recorded before any aggressive-mode measurements.
rows=[r for r in rows if (r['experiment']=='main' and r['C']>1) or (r['experiment']=='modes' and r['C']==1) or r['experiment'] in ('cap','retirement')]
points=[]
for C in sorted({r['C'] for r in rows}):
    for engine in ('fused','fused_capped','fused_retired','omlx','omlx_aggressive','lithos'):
        rs=[r for r in rows if r['C']==C and r['engine']==engine]
        if not rs:continue
        assert len(rs)==3
        points.append({'C':C,'engine':engine,'family':'omlx' if engine.startswith('omlx') else ('fused' if engine.startswith('fused') else engine),**{k:statistics.median(r[k] for r in rs) for k in ('x','y','TTFT_s')},
                       'process_range':{k:[min(r[k] for r in rs),max(r[k] for r in rs)] for k in ('x','y','TTFT_s')}})
for p in points:
    p['own_frontier']=not any(dominates(q,p) for q in points if q['family']==p['family'])
    p['joint_frontier']=not any(dominates(q,p) for q in points)
ours=[p for p in points if p['family']=='fused'];coverage=[]
for p in points:
    if p['family']=='fused':continue
    ds=[q for q in ours if dominates(q,p)]
    robust=[q['C'] for q in ds if all(q['process_range'][k][0]>p['process_range'][k][1] for k in ('x','y'))]
    same=next(q for q in ours if q['C']==p['C'])
    pairs=[]
    for pair in (1,2,3):
        x=next(r for r in rows if r['C']==p['C'] and r['engine']=='fused' and r['pair']==pair)
        y=next(r for r in rows if r['C']==p['C'] and r['engine']==p['engine'] and r['pair']==pair)
        pairs.append({'pair':pair,'dominates':dominates(x,y),'x_gain_pct':100*(x['x']/y['x']-1),'y_gain_pct':100*(x['y']/y['y']-1)})
    coverage.append({'engine':p['engine'],'C':p['C'],'covered_by_native_C':[q['C'] for q in ds],
                     'same_C_native_configuration':'fused (original batch-fusion version)',
                     'covering_native_configurations':[{'C':q['C'],'engine':q['engine']} for q in ds],
                     'covered_across_process_min_max_by_native_C':robust,
                     'same_C_x_gain_pct':100*(same['x']/p['x']-1),'same_C_y_gain_pct':100*(same['y']/p['y']-1),'pairs':pairs})
resources={'snapshots':len(samples),'pressure_levels':sorted({s['pressure_level_value'] for s in samples}),
           'thermal_codes':sorted({s['processinfo_thermal_metadata']['code'] for s in samples}),
           'swap_used_mib_max':max(s['swap_used_mb'] for s in samples),
           'counter_ranges':{k:[min(s['vm_counters'][k] for s in samples),max(s['vm_counters'][k] for s in samples)] for k in ('Pageouts','Swapouts')}}
replay={'status':'PASS','processes_validated':len(all_rows),'processes_in_final_comparison':len(rows),'requests_including_warm':requests,'measured_requests':measured,
        'tokens_including_warm':requests*32,'measured_tokens':measured*32,'native_requests_with_raw_ids':native,
        'native_sequences_matching_one_qualified_reference':known_native,'pins_verified':len(pinned),
        'resources':resources,'strict_cleanup':True}
write(F/'replay.json',replay)
result={'source_commit':frozen['source_commit'],'points':points,'all_processes':rows,'coverage':coverage,
        'runtime_versions':{'native_mlx':'0.32.3','omlx':'0.7.1.dev1','omlx_commit':None,'omlx_distribution':'https://github.com/jundot/omlx/releases/download/v0.7.1.dev1/omlx-0.7.1.dev1-cp313-cp313-macosx_15_0_universal2.whl','omlx_identity_note':'Official wheel; no proven VCS commit. Installed files are pinned. The separate source checkout is not the wheel identity.','omlx_mlx_lm':'0.31.4.dev132+g94cdcae13','lithos_upstream':'5485ad6','lithos_local_profile_commit':'223658090ae549e59220bdec459957cbdad7926a'},
        'earlier_C1_processes':earlier_c1,'all_validated_processes':all_rows,'C1_selection_plan':read(B/'pareto-mode-plan-v1.json'),
        'capped_C64_plan':read(B/'pareto-cap-plan-v1.json'),
        'all_comparator_medians_covered':all(c['covered_by_native_C'] for c in coverage),
        'all_paired_same_C_points_dominated':all(p['dominates'] for c in coverage for p in c['pairs']),
        'qualification':qual,'graph_only_qualification':graph_only,'kernel_qualification':kernel,'free_generation':free,'parameters':read(B/'pareto-batch-qual-v1/parameters.json'),
        'replay':replay,'scope':'M5 Ultra 256 GiB; fixed Qwen3.8-27B NVFP4; 19 input/32 output; greedy normal EOS; no prompt cache/draft; 3 fresh processes x 3 measured groups; warm steady-state; C is HTTP concurrency',
        'limits':['Only the sampled concurrency values and this short workload; not an entire continuous curve or universal workload claim',
                  'C1 uses the preplanned later four-configuration comparison, including oMLX aggressive; the earlier C1 experiment is fully retained separately',
                  'oMLX uses balanced at C2..64; aggressive changes the single-request burst budget only, while the concurrent budget remains 0.03 seconds',
                  'C64/B32 is a separately measured configuration: all64 HTTP requests complete, at most32 active sequences. Its three native processes are not paired with the earlier official C64 processes',
                  'Lithos official endpoint supports only C1; this is one measured point, with a local M5 Ultra profile',
                  'x is a client text-delivery estimate, not raw GPU generation speed; y includes prefill and HTTP delivery',
                  'Three process ranges are observations, not confidence intervals',
                  'B1..8 fusion is opt-in and used length <=256; larger batches/contexts stay eager',
                  'Three warm groups precede measurement; all measured time is retained, including any compilation for a previously unseen batch shape',
                  'Static batch output identities need not equal mixed-arrival HTTP output identities; both same-batch free generation and changing-row scheduler states are separately qualified']}
cap=next(p for p in points if p['engine']=='fused_capped')
result['capped_C64_comparison']={}
for engine in ('fused','omlx'):
    other=next(p for p in points if p['engine']==engine and p['C']==64)
    result['capped_C64_comparison'][engine]={'dominates_median':dominates(cap,other),
        'x_gain_pct':100*(cap['x']/other['x']-1),'y_gain_pct':100*(cap['y']/other['y']-1),
        'TTFT_change_pct':100*(cap['TTFT_s']/other['TTFT_s']-1),'paired':False}
if a.retirement_frontier:
    result['retirement_qualification']=retirement_qual
    result['retirement_plan']=read(B/'retire-batch-plan-v1.json')
    source=B/'retire-batch-source-v1'
    matches=all(digest(R/(R/Path(p)).resolve().relative_to(source.resolve()))==h for p,h in retirement_qual['source_files'].items())
    clean=subprocess.run(['git','diff','--quiet','HEAD','--','tutorial_code/miniengine'],cwd=R).returncode==0
    result['retirement_adopted_commit']=subprocess.check_output(['git','log','-1','--format=%H','--','tutorial_code/miniengine'],cwd=R,text=True).strip() if matches and clean else None
    result['retirement_step_diagnostic']={}
    for label,root in (('original',F),('batched_retirement',a.retirement_frontier)):
        times=[]
        for pair in (1,2,3):
            trace=read(root/f'C64-pair{pair}-fused/engine-trace.json')
            times.extend((x['end_s']-x['start_s'])*1000 for x in trace['steps']
                         if x['delivered_ids']==37 and x['active_decode_batch']==11 and x['pending_requests']==11)
        result['retirement_step_diagnostic'][label]={'times_ms':times,'median_ms':statistics.median(times),'n':len(times)}
    result['retirement_step_diagnostic']['scope']='Descriptive filter of all warm and measured C64 steps:37 tokens delivered,11 active/pending after step. Not a paired microbenchmark; all raw steps retained.'
    result['retirement_comparisons']=[]
    for point in points:
        if point['engine']!='fused_retired':continue
        for engine in ('fused','omlx'):
            other=next(p for p in points if p['engine']==engine and p['C']==point['C'])
            result['retirement_comparisons'].append({'C':point['C'],'against':engine,
                'x_gain_pct':100*(point['x']/other['x']-1),'y_gain_pct':100*(point['y']/other['y']-1),
                'TTFT_change_pct':100*(point['TTFT_s']/other['TTFT_s']-1),'paired':False})
    result['limits'].append('Batched-retirement measurements at C8/32/64 use a separately hashed implementation; original sweep baselines are unpaired and retained. No arithmetic changes.')
write(O/'m5-ultra-pareto-results.json',result)
lines=['M5 Ultra 256GB：目标模型单独解码的实测 Pareto 前沿','',
       '融合版本代码：'+frozen['source_commit'],
       '改动：投影和 GDN 融合扩到 B1–B8；保留 B1 内核；大批次保留 eager。测试服务器连接队列从 5 扩到 128。',
       '同一 27B NVFP4；19 输入/32 输出；greedy、正常 EOS、无草稿模型或前缀缓存。',
       '运行版本：MLX0.32.3；oMLX0.7.1.dev1官方wheel（VCS提交未知，安装文件哈希已固定；mlx-lm0.31.4.dev132+g94cdcae13）；Lithos5485ad6加本机M5配置2236580。',
       'C1 按事先写下的规则，使用包含 oMLX balanced/aggressive 的四配置复测；原 C1 数据完整保留在 earlier_C1_processes。C2–C64 使用原 sweep。',
       '另加 C64/B32 点：64 个 HTTP 请求全部完成，同时最多执行32条序列；三次新进程。官方 C64 数据来自紧邻的原 sweep，不宣称逐对配对。',
       '各点为三次独立进程中位数的中位数，每进程 3 次预热、3 次测量。',
       'x=首个文本块之后的客户端交付速率估计；y=完整并发请求窗口的总输出吞吐。单位 tok/s。','',
       'C | 引擎 | x 每用户交付 | y 整机吞吐 | TTFT ms | 自身 Pareto 点']
for p in points:lines.append(f"{p['C']} | {p['engine']} | {p['x']:.2f} | {p['y']:.2f} | {p['TTFT_s']*1000:.1f} | {p['own_frontier']}")
lines+=['','全部外部配置中位数被实测 minivllm 点覆盖：'+str(result['all_comparator_medians_covered']),
        '同并发三轮逐对也全部双指标领先：'+str(result['all_paired_same_C_points_dominated']),
        '覆盖定义：存在一个真实测量点，其 x、y 均不低于对方，至少一项严格更高；不依赖插值。','']
for c in coverage:lines.append(f"{c['engine']} C{c['C']}: 原融合版本同并发 x {c['same_C_x_gain_pct']:+.2f}%, y {c['same_C_y_gain_pct']:+.2f}%; 覆盖它的 minivllm C={c['covered_by_native_C']}")
lines+=['','C64/B32 与未限批和官方 C64 的比较：'+json.dumps(result['capped_C64_comparison'],ensure_ascii=False)]
if a.retirement_frontier:
    lines+=['','后续简化：每步批量清理结束请求，最多一次压缩状态；全结束时直接释放。只改两个核心文件，不改算术。',
            '采用提交：'+str(result['retirement_adopted_commit']),
            '完整状态检查：'+str(retirement_qual['cases'])+' 个场景逐步输出和全部剩余缓存签名一致；完整源码哈希保存在 results.json。',
            '新的 C8/C32/C64 测量，与之前基线未逐对配对：']
    for c in result['retirement_comparisons']:
        lines.append(f"  C{c['C']} 相对 {c['against']}: 交付速率 {c['x_gain_pct']:+.2f}%，吞吐 {c['y_gain_pct']:+.2f}%，TTFT {c['TTFT_change_pct']:+.2f}%。")
    diag=result['retirement_step_diagnostic']
    lines.append(f"步骤诊断：37个输出、剩11请求的步骤，中位耗时 {diag['original']['median_ms']:.1f} → {diag['batched_retirement']['median_ms']:.1f} ms（各14个样本，含预热，非配对微基准；全部原始步骤保留）。")
lines+=['',f"正确性：{qual['paired_forwards']} 组融合路径和 {graph_only['paired_forwards']} 组仅编译路径的完整 logits/128 个缓存数组逐位一致，128 组融合算子测试，1349 个原始参数哈希未变。",'B1–B8、B16/B32/B64 同批次自由生成一致；交错请求插入、退出、lookahead、stop 检查通过。',
        '额外派生权重 8.714 GiB，默认关闭。3 轮预热后测量；测量期间若出现新批次形状编译，耗时原样保留。',
        '高批次融合探索被拒绝：B12 起合并投影会改变归约路径；修复后 B32 性能回退。',
        '第一轮旧 HTTP 入口 C8 发生连接重置，失败记录完整保留，不纳入正式结果。',
        '仅编译路径验证 v1 遇到内存压力后由保护逻辑终止；进程已回收。资源恢复后 v2 的24组检查通过，失败记录保留。',
        '限制：只覆盖本图采样并发和短请求；没有证明长上下文、其他工作负载或带投机解码的优势。',
        'Lithos 仅官方接口支持的 C1 点；官方引擎没有提供原始 token IDs。',
        '复算：'+json.dumps(replay,ensure_ascii=False),'原始记录：'+str(F)]
(O/'m5-ultra-pareto-report.txt').write_text('\n'.join(lines)+'\n')
print(json.dumps({'points':points,'coverage':coverage,'replay':replay,'all_covered':result['all_comparator_medians_covered']},ensure_ascii=False,indent=2))
