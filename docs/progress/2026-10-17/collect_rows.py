"""CPU/stdlib: whitelist safe benchmark rows; never copy argv, settings or raw logs."""
from pathlib import Path
import argparse, collections, hashlib, json, statistics

ap = argparse.ArgumentParser()
ap.add_argument('--source-root', type=Path, required=True)
ap.add_argument('--out', type=Path, default=Path(__file__).resolve().parent)
args = ap.parse_args()
P = args.source_root
R = P / 'v23-root-control-build1'
sources = {}

def load(label, relative):
    p = P / relative
    b = p.read_bytes()
    sources[label] = {'path_relative_to_private_source_root': relative,
                      'sha256': hashlib.sha256(b).hexdigest(), 'bytes': len(b)}
    return json.loads(b)

tiny = load('108tiny', 'v23-root-control-build1/numeric-v108tiny-attempt1/child-output/receipt.json')
txn = load('108full', 'v23-root-control-build1/fullmodel-v108transaction-attempt1/child-output/receipt.json')
qual = load('109qual', 'v23-root-control-build1/omlx-mtp-v109-attempt1/child-output/receipt.json')
http = load('111http', 'v23-root-control-build1/omlx-mtp-v111-attempt1/child-output/suite-complete.json')
old = load('112local', 'v23-root-control-build1/diagnostic-v112-attempt1/child-output/receipt.json')
new = load('113local', 'v23-root-control-build1/diagnostic-v113-attempt2/child-output/receipt.json')
partial = load('114checkpoint', 'v23-root-control-build1/fullmodel-v114broadcast-attempt1/child-output/receipt.json')
owner = load('114owner', 'v23-root-control-build1/fullmodel-v114broadcast-attempt1/owner-receipt.json')
release = load('114release', 'v23-root-control-build1/fullmodel-v114broadcast-attempt1/external-release.json')
cpu = load('115cpu', 'v115-device-draft-selector-build1/CPU-READY.json')
clock_relative = 'v111-omlx-none-mtp-http-screen-build1/driver.py'
clock_bytes = (P / clock_relative).read_bytes()
sources['111clock_source'] = {'path_relative_to_private_source_root': clock_relative,
                            'sha256': hashlib.sha256(clock_bytes).hexdigest(), 'bytes': len(clock_bytes)}
assert tiny['strict_transactions'] == 21 and txn['status'] == 'FAIL_STOP'
assert txn['A_old96_IDs_passed'] and not txn['adopted']
assert qual['actual_requests'] == 6 and len(qual['MTP_per_request']) == 3
assert all(q['verify_calls'] == 17 and q['drafted'] == 34 and q['accepted'] == 15
           and q['raw_output_ids'] == qual['ordinary_control_raw32'] for q in qual['MTP_per_request'])
assert http['HTTP_requests'] == 36 and http['measured_requests'] == 18
assert http['HTTP_completion_tokens'] == 1152 and http['measured_tokens'] == 576
pairs = []
for p in http['pairs']:
    row = {'pair': p['pair'], 'order': p['order']}
    for mode in ('none', 'mtp'):
        row[mode] = {k: p[mode][k] for k in ('x', 'y', 'ttft_s', 'E2E_s')}
    for k in ('x', 'y'):
        signed = 100 * (row['mtp'][k] / row['none'][k] - 1)
        assert signed == p[k + '_signed_MTP_pct'] and signed < 0
        row[k + '_signed_MTP_pct'] = signed
    pairs.append(row)
assert len(pairs) == 3 and [p['order'] for p in pairs] == [['none','mtp'],['mtp','none'],['none','mtp']]
linear = [x for x in old['comparisons'] if x['kind'] == 'Linear']
assert len(linear) == 497
equal = sum(x['strict_equal'] for x in linear)
assert equal == 146
variant = new['summary']['by_variant']['broadcastM1_vs_rowM1']
assert variant['calls'] == variant['strict_equal'] == 497 and variant['strict_different'] == 0
assert new['original112_all497_input_output_hashes_reproduced']
assert new['keep0_restored_all128'] and new['target1349_identity_readonly'] and not new['timing_run']
assert partial['A_old96_IDs_passed'] and len(partial['requests']) == 3
assert all(q['arm'] == 'A' and q['strict_old_ID_equal'] and len(q['IDs']) == 32 for q in partial['requests'])
assert len(partial['transactions']) == 0 and partial['status'] == 'STARTED'
assert owner['stage'] == 'reaped' and owner['waited_reaped'] and owner['returncode'] == -15
assert release['result'] == 'V114_RESOURCE_PRESSURE_STOP_NO_FULLMODEL_QUALIFICATION'
assert release['GPU_released'] and cpu['status'] == 'FROZEN_CPU_READY_V115_DEVICE_SELECTOR_NUMERICAL_GPU_UNRUN'

rows = {
 'snapshot_date': '2026-10-17', 'scope': 'M4 Max 40 GPU cores / 48 GiB; Qwen3.8-27B NVFP4 (Qwen3.5 architecture); MLX 0.32.3',
 'http': {'experiment':'V111', 'pairs':pairs,
   'grain':'One point is the median of three measured C1 HTTP requests in one fresh service arm; all three signed pairs retained.',
   'warm_requests':18, 'measured_requests':18, 'measured_tokens':576,
   'x_definition':'Client-estimated tokens after first nonempty text chunk: (32 - reencoded first-chunk token count) / (done_time - first_text_time). Not ordinary per-step decode throughput.',
   'y_definition':'Total output tokens / HTTP group window; C1 has one request, hence 32 / (done_time - start_time).',
   'aggregation':'Three measured groups per arm; median per arm. Figure shows all six arm medians. Signed pair change = 100*(MTP/none-1).',
   'signed_pair_medians':{k:statistics.median(p[k+'_signed_MTP_pct'] for p in pairs) for k in ('x','y')},
   'limits':['Only C1, prompt19/output32, fresh official none vs external MTP default block3; not our native engine and not a complete frontier.',
             'No raw token IDs in timed API: visible text reencoded to32; separate V109 fresh-official-control raw32 qualification.',
             'No confidence intervals or statistical significance claim; three pairs are not pooled as independent requests.',
             'AC/thermal0/pressure1 samples; Swapouts stable, but transient swap_used and Pageouts growth occurred.']},
 'local_linear': {'denominator':497,'unit':'same-input Linear call outputs compared by exact bytes',
   'rows':[{'experiment':'V112','comparison':'batch M=T vs serial M=1','exact':equal,'different':497-equal},
           {'experiment':'V113','comparison':'broadcast batch=T, M=1 vs serial M=1','exact':497,'different':0}],
   'limits':['Same497 captured Linear inputs from one actual first-verify block; not497 independent model runs.',
             'A local exact-byte result is not full-model IDs/cache qualification, PSO identification, or a speed result.']},
 'sources':sources, 'safety':'Whitelisted aggregate/count rows only; no argv, API key, raw text stream, parameter arrays, or tensor payload copied.'}
args.out.mkdir(parents=True, exist_ok=True)
(args.out/'reviewed_rows.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2)+'\n')
stages = [
 {'stage':'V108','status':'FULLMODEL_OLD_ID_FAIL','title':'缓存前缀事务','observed':'21项tiny事务通过；普通A96旧IDs通过；投机B旧ID硬门失败。','limit':'整模FAIL保留，不能以官方文本一致改判。','sources':['108tiny','108full']},
 {'stage':'V109','status':'OFFICIAL_MTP_NUMERIC_PASS','title':'官方外部MTP资格','observed':'6条raw32与fresh普通官方控制一致；每MTP请求17 verify / 34提案 / 15接受 / 32交付。','limit':'观察器计时不排名；34提案不是34交付。','sources':['109qual']},
 {'stage':'V111','status':'C1_MTP_SLOWER','title':'官方none/MTP配对HTTP','observed':'3对均慢；有符号中位变化x −34.8496%、y −29.5934%。','limit':'仅两个C1配置，不是全局最好或完整Pareto前沿。','sources':['111http']},
 {'stage':'V112','status':'SAME_INPUT_DIAGNOSTIC','title':'逐Linear形状敏感性','observed':'497项同输入Linear：146 exact / 351不同。','limit':'局部逐算子诊断，无整模修复或速度结论。','sources':['112local']},
 {'stage':'V113','status':'LOCAL_BROADCAST_NUMERIC_PASS','title':'broadcast-M1局部数值','observed':'同497输入全部exact；原112输入/output hashes重现。','limit':'仅局部数值通过；无实际PSO、整模96 IDs修复或速度结论。','sources':['113local']},
 {'stage':'V114','status':'RESOURCE_STOP_B_UNREACHED','title':'target-verify整合','observed':'A三请求96旧IDs通过；资源压力触发SIGTERM，B未到达，事务0项。','limit':'checkpoint仍为STARTED；不是数值FAIL也不是整模PASS。进程已reap/外部释放。','sources':['114checkpoint','114owner','114release']},
 {'stage':'V115','status':'CPU_READY_GPU_UNRUN','title':'device draft selector','observed':'冻结CPU准备记录；GPU数值未运行。','limit':'CPU/source测试不替代GPU数值资格，无性能结论。','sources':['115cpu']}
]
ledger={'snapshot_date':'2026-10-17','stages':stages,'full_frontier_win':False,'default_promoted':False,'completion_percentage':None,'percentage_note':'不同资格层级不折算成完成百分比。','sources':sources}
(args.out/'progress.json').write_text(json.dumps(ledger, ensure_ascii=False, indent=2)+'\n')
print(json.dumps({'status':'SAFE_ROWS_COLLECTED','pairs':len(pairs),'local_calls':497,'stages':len(stages),'sources':len(sources)}))
