"""V85 AUTHOR SELF audit of saved evidence only; no MLX/GPU/probes/rerun."""
from pathlib import Path
import hashlib
import json
import math
import statistics

P = Path('/private/tmp/minivllm-engine-optimization-20261014')
D = Path(__file__).resolve().parent
R = P / 'v23-root-control-build1/kernel-micro-v85-attempt1'
O = R / 'child-output'
V85 = P / 'v85-gdn-two-kernel-native-micro-build1'
V81 = P / 'v81-gdn-two-kernel-native-range-build1'
refs = {}
checks = []


def h(path):
    path = Path(path)
    value = hashlib.sha256(path.read_bytes()).hexdigest()
    refs[str(path)] = {'sha256': value, 'bytes': path.stat().st_size}
    return value


def read(path):
    h(path)
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def check(condition, label):
    assert condition, label
    checks.append(label)


receipt = O / 'receipt.json'
check(h(receipt) == '83adff180b95e13e1aa9ba006399d7ab37e7a2d135c40abf8e4eba0cd7d6c083', 'exact_actual85_receipt_SHA')
r = read(receipt)
pins = read(V85 / 'source-pins.json')
old = read(V81 / 'source-pins.json')
check(r['pins'] == pins and len(pins) == 64 and len(old) == 56, 'receipt64_pins_and_original56')
check(all(pins.get(path) == value and h(path) == value for path, value in pins.items()), 'all64_current_file_bytes_match')
check(all(pins.get(path) == value for path, value in old.items()), 'original56_unchanged_subset')
q81 = read(P / 'v23-root-control-build1/fullmodel-v81num-attempt1/child-output/receipt.json')
check(q81['pins'] == old and r['qualified_v81_receipt_sha256'] == h(Path(r['qualified_v81_receipt'])), 'actual81_binding_exact')

references = {}
for category, count in [('public_AB_checks', 30), ('captured_checks', 60), ('native_checks', 480)]:
    rows = r[category]
    check(len(rows) == count and all(row['strict_equal'] and row['A_sha256'] == row['B_sha256'] for row in rows), category + '_strict_count_hash_equality')
    for row in rows:
        key = row['bank'], row['output']
        if category == 'public_AB_checks':
            references[key] = row['A_sha256']
            output = 0 if row['output'] == 'o' else 1
            path = O / 'numeric' / f'expected-b{row["bank"]:02d}-o{output}.raw'
            check(h(path) == row['A_sha256'] and path.stat().st_size == row['bytes'], f'public_raw_{key}')
        else:
            check(row['A_sha256'] == references[key], f'{category}_public_reference_{len(checks)}')
            if 'raw_file' in row:
                path = O / row['raw_file']
                check(path.resolve().is_relative_to(O.resolve()) and h(path) == row['B_sha256'] and path.stat().st_size == row['bytes'], f'native_final_raw_{len(checks)}')
check(len(references) == 30 and sum('raw_file' in row for row in r['native_checks']) == 60, '30_public_60_native_saved_raw_arrays')
check(r['completed_strict_counts'] == {'public_AB_checks': 30, 'captured_checks': 60, 'native_checks': 480}, 'truthful_numeric_completed_counts')
check(len(r['cases']) == 15 and sum(row['bank'].startswith('real-') for row in r['cases']) == 9 and sum(row['bank'].startswith('synthetic-') for row in r['cases']) == 6, '15_saved_9real_6synthetic_cases')
bankroot = P / 'v18-recurrent-numeric-build1/gpu-recurrent-numeric-AC-run1'
for case in r['cases']:
    directory = bankroot / case['bank']
    metadata = read(directory / 'bank.json')
    check(case['kind'] == metadata['kind'] and case['bank_json_sha256'] == h(directory / 'bank.json'), 'saved_case_metadata_' + case['bank'])
    for name, array in metadata['arrays'].items():
        path = directory / array['file']
        check(path.resolve().is_relative_to(directory.resolve()) and path.stat().st_size == array['bytes'] and h(path) == array['sha256'], 'saved_bank_raw_' + case['bank'] + '_' + name)

built = read(O / 'native-built.json')
check(built['capture_dispatches'] == built['ICB_commands'] == 60 and built['immutable_allocations'] == 60 and built['mutable_registered_outputs'] == 180, 'actual60dispatch_60RO_180mutable_build')
check(built['ranges'] == [[0, 30], [30, 60]] and built['internal_state_barriers_per_range'] == 15 and built['producer_consumer_edges_verified'] and built['original_and_arena_ranges_disjoint'], 'actual_ranges_dependency_barrier_proof')
check(len(built['PSOs']) == 4 and all(pso['supports_ICB'] for pso in built['PSOs']), 'four_actual_ICB_PSOs')
for i, record in enumerate(built['dispatch_records']):
    check((record['ordinal'], record['arm'], record['bank'], record['stage']) == (i, i // 30, (i % 30) // 2, i % 2), f'actual_dispatch_mapping_{i}')
    if record['stage'] == 1:
        args = {arg['name']: arg for arg in record['actual_active_arguments']}
        check(args['S']['index'] == 6 and args['S']['reflection_access'] == 0 and args['S_new']['index'] == 9 and args['S_new']['reflection_access'] in (1, 2) and record['ICB_barrier_before'] and record['captured_boundaries_before'] > 0, f'actual_S6_RO_Snew9_and_dependency_{i}')
allocations = {}
mutable = set()
packed = []
maps = built['bank_mappings']
check(len(maps) == 30, '30_actual_arm_bank_maps')
for i, bank in enumerate(maps):
    check((bank['arm'], bank['bank']) == (i // 15, i % 15), f'bank_map_order_{i}')
    for a, array in enumerate(bank['arrays']):
        check(array['array'] == a and array['logical_bytes'] <= array['allocation_bytes'], f'array_extent_{i}_{a}')
        identity = array['original_allocation']
        interval = (array['logical_address'], array['logical_address'] + array['allocation_bytes'])
        check(identity not in allocations or allocations[identity] == interval, f'identity_extent_consistent_{i}_{a}')
        allocations[identity] = interval
        if a < 4 and i >= 15:
            check(array == maps[i - 15]['arrays'][a], f'A_B_original_input_identity_{i}_{a}')
        if a >= 4:
            check(identity not in mutable, f'mutable_exclusive_{i}_{a}')
            mutable.add(identity)
            packed.append((array['arena_offset'], array['arena_offset'] + array['allocation_bytes']))
for ranges in (sorted(allocations.values()), sorted(packed)):
    check(all(a[1] <= b[0] for a, b in zip(ranges, ranges[1:])), 'recomputed_original_or_arena_disjoint')
check(len(allocations) == 240 and len(mutable) == 180, '60immutable_plus180exclusive_mutable_allocations')

protocol = read(V85 / 'protocol.json')
check([pair['order'] for pair in r['pairs']] == protocol['schedule'], 'exact_6ABBA_4AA_BB_schedule')
blocks = []
controls = {'A': [], 'B': []}
ab = []
for pair in r['pairs']:
    check(len(pair['blocks']) == 2, 'pair_has_two_blocks_' + str(pair['pair']))
    for block in pair['blocks']:
        index = len(blocks)
        blocks.append(block)
        n = block['native']
        check(block['prime_groups'] == 128 and block['timed_groups'] == 1024 and block['encoded_dispatches'] == 30720 and block['ns_per_dispatch'] == block['wall_ns'] / 30720 and block['ns_per_bank_two_dispatches'] == block['wall_ns'] / 15360, 'timed_block_divisors_' + str(index))
        check(n['last_CB_completed'] and n['last_groups'] == 1024 and n['observed_encoded_dispatches'] == 30720 and n['trailing_Buffers_barriers_last_loop'] == 1024 and n['last_arm'] == (0 if block['arm'] == 'A' else 1), 'actual_block_completion_' + str(index))
        check(n['total_encoded_dispatches'] == 2760 + (index + 1) * 34560 and n['loops'] == 16 + (index + 1) * 2, 'numeric_prime_timed_cumulative_counts_' + str(index))
        check(n['dispatch_records'] == built['dispatch_records'] and n['bank_mappings'] == built['bank_mappings'], 'bindings_stable_' + str(index))
        check(n['CBGPU_timestamps_valid'] and math.isfinite(n['CBGPUStartTime_seconds']) and n['CBGPUEndTime_seconds'] >= n['CBGPUStartTime_seconds'] > 0, 'official_seconds_CB_envelope_valid_' + str(index))
        check(block['terminal_check_indices'] == [index * 30, (index + 1) * 30] and block['terminal_strict_checks'] == 30, 'nonoverlapping_30_terminal_check_slice_' + str(index))
    first, second = pair['blocks']
    if pair['kind'] == 'control':
        ratio = second['wall_ns'] / first['wall_ns']
        check(pair['slot2_div_slot1'] == ratio, 'control_ratio_' + str(pair['pair']))
        controls[pair['order'][0]].append(ratio)
    else:
        values = {block['arm']: block['wall_ns'] for block in pair['blocks']}
        check(pair['reduction'] == 1 - values['B'] / values['A'], 'AB_reduction_' + str(pair['pair']))
        ab.append(pair)
check(len(blocks) == 20 and len(ab) == 6 and all(len(values) == 2 for values in controls.values()), '20blocks_6pairs_4controls')
medians = {arm: statistics.median(next(block['wall_ns'] / 30720 for block in pair['blocks'] if block['arm'] == arm) for pair in ab) for arm in ('A', 'B')}
gain = 1 - medians['B'] / medians['A']
wins = sum(pair['reduction'] > 0 for pair in ab)
control_passed = all(all(.95 <= value <= 1.05 for value in values) and .98 <= statistics.median(values) <= 1.02 for values in controls.values())
check(medians == r['medians_ns_per_dispatch'] and controls == r['controls'] and gain == r['median_reduction'] and wins == r['wins'] == 6 and control_passed == r['control_passed'] is True, 'statistics_recomputed_from_saved_wall_spans')
check(gain < .05 and r['status'] == 'STOP_MICRO_GATE' and r['adopted'] is False, 'strict5percent_gate_STOP_noadoption')
timed = r['timed_terminal_checks']
check(len(timed) == r['timed_terminal_checks_completed'] == r['timed_terminal_strict_checks'] == 600, '600_actual_terminal_checks_not_per_dispatch')
for index, row in enumerate(timed):
    block = blocks[index // 30]
    check(row['pair'] == index // 60 and row['block'] == (index // 30) % 2 and row['arm'] == block['arm'] and row['bank'] == (index % 30) // 2 and row['output'] == ('o', 'S_new')[index % 2], f'timed_check_unique_key_{index}')
    check(row['strict_equal'] and row['actual_sha256'] == row['expected_sha256'] == references[row['bank'], row['output']], f'timed_terminal_reference_hash_{index}')
readonly = r['readonly_checks'] + [pair['readonly'] for pair in r['pairs']]
check(len(readonly) == 27 and all(row['original_arrays'] == 255 and row['all_raw_sha256_unchanged'] and row['native_shape_constants_unchanged'] for row in readonly), '17numeric_plus10pair_readonly_checkpoints_255original_arrays')
check(len(read(O / 'inputs.json')) == 75 and len(read(O / 'captured-originals.json')) == 180, '75input_plus180captured_originals')
resources = read(O / 'resources.json')
check(len(resources) == 24, '24_raw_resource_samples')
for index, resource in enumerate(resources):
    raw = read(O / f'resource-{index:02d}-raw.json')
    check(all(resource.get(key) == value for key, value in raw.items()) and set(resource) - set(raw) == ({'swap_used_mb_delta', 'vm_counter_delta'} if index else set()), f'resource_raw_fields_match_derived_list_{index}')
    if index:
        previous = resources[index - 1]
        check(resource['swap_used_mb_delta'] == resource['swap_used_mb'] - previous['swap_used_mb'] and resource['vm_counter_delta'] == {key: value - previous['vm_counters'][key] for key, value in resource['vm_counters'].items() if key in previous['vm_counters']}, f'resource_deltas_recomputed_{index}')
    check(resource['power_metadata']['source'] == 'AC Power' and resource['processinfo_thermal_metadata']['code'] == 0 and resource['pressure_level_value'] == 1 and resource['swap_used_mb'] == 5006.88 and resource['vm_counters']['Swapouts'] == 17460761 and resource['process_inventory_ok'] and resource['only_owned_known_inference_processes'] and resource['owned_model_pid'] == 95239, f'saved_resource_guard_fields_{index}')
owner, release = read(R / 'owner-receipt.json'), read(R / 'external-release.json')
check(owner['owner_pid'] == 95238 and owner['driver_pid'] == 95239 and owner['returncode'] == 2 and owner['stage'] == 'reaped' and owner['waited_reaped'] and owner['absence']['absent'], 'owner_rc2_reaped')
check(release['status'] == 'PASS_EXTERNAL_RELEASE' and release['GPU_released'] and release['tool_exit'] == 2 and {row['pid'] for row in release['strict_absence']} == {95238, 95239} and all(row['returncode'] == 1 and not row['stdout'] and not row['stderr'] for row in release['strict_absence']), 'saved_external_release_and_both_PID_absence')
check(owner['driver_sha256'] == pins[str(V85 / 'micro.py')], 'owner_exact_driver_SHA')
external = read(R / 'external-resource.json')
check(len(external) == 1 and external[0]['known_inference_processes'] == [] and external[0]['pressure_level_value'] == 1 and external[0]['swap_used_mb'] == 5006.88 and external[0]['vm_counters']['Swapouts'] == 17460761, 'saved_external_resource_release')

gpu_medians = {arm: statistics.median((block['native']['CBGPUEndTime_seconds'] - block['native']['CBGPUStartTime_seconds']) * 1e9 / 30720 for pair in ab for block in pair['blocks'] if block['arm'] == arm) for arm in ('A', 'B')}
fullpath = P / 'v23-root-control-build1/native-decode-v71-attempt1/child-output/receipt.json'
full = read(fullpath)
measured = [request for request in full['requests'] if request['phase'].startswith('measure')]
check(len(measured) == 3, 'historical71_three_measured_engine_requests')
engine_ns = statistics.median((request['t2_ns'] - request['t1_ns']) / 31 for request in measured)
check(engine_ns == full['measured_summary']['median_post_first_token_decode_ns_per_step_including_seed'], 'historical71_engine_timestamps_recomputed')
config = read(Path('/Users/ryanli/.cache/huggingface/hub/models--mlx-community--Qwen3.8-27B-nvfp4/snapshots/5ff8ef173ad0d7c3aae92f0be43031a6ab8067c6/config.json'))
text = config['text_config']
layers = text['layer_types'].count('linear_attention')
check(text['num_hidden_layers'] == len(text['layer_types']) == 64 and layers == 48, 'config48_linear_layers_of64')
wall_scaled_ns = medians['A'] * 2 * layers
gpu_scaled_ns = gpu_medians['A'] * 2 * layers
conditional_saved_ns = (medians['A'] - medians['B']) * 2 * layers
selected = {'review_type': 'AUTHOR_SELF_AUDIT_NOT_INDEPENDENT', 'status': 'PASS_SAVED_EVIDENCE_AUDIT_STOP_UNCHANGED',
            'actual_receipt_SHA': h(receipt), 'checks': len(checks), 'pins': 64, 'old_pins_unchanged_subset': 56,
            'numeric_strict_counts': [30, 60, 480], 'timed_terminal_strict_checks': 600, 'timed_blocks': 20,
            'dispatch_counts': {'numeric': 2760, 'prime': 76800, 'timed': 614400, 'total': 693960},
            'wall_medians_ns_per_dispatch': medians, 'wall_medians_us_per_bank': {arm: value * 2 / 1000 for arm, value in medians.items()},
            'wall_median_reduction': gain, 'wins': wins, 'controls': controls,
            'control_medians': {arm: statistics.median(values) for arm, values in controls.items()}, 'control_passed': control_passed,
            'official_micro_CB_envelope_medians_ns_per_dispatch': gpu_medians, 'CB_timestamps_all20_valid': True,
            'GPU_envelope_is_not_exclusive_kernel_time_or_gate_statistic': True,
            'resource_samples': 24, 'swap_used_MB_unchanged': 5006.88, 'Swapouts_unchanged': 17460761,
            'owner': 95238, 'driver': 95239, 'returncode': 2, 'reaped_and_external_release': True,
            'full_reference': {'path': str(fullpath), 'SHA': h(fullpath), 'engine_ns_per_step': engine_ns, 'timing_definition': full['timing_definition'], 'same_workload_or_clock_as_micro': False},
            'conditional_scaling_NOT_a_fullmodel_bound': {'assumptions': '48 fullmodel GDN banks have identical amortized local cost and all other work unchanged; unproven', 'layers': layers, 'wall_A_scaled_ms': wall_scaled_ns / 1e6, 'micro_CB_A_scaled_ms': gpu_scaled_ns / 1e6, 'wall_scaled_fraction_of_reference': wall_scaled_ns / engine_ns, 'actual_local_AB_gain_scaled_saved_us': conditional_saved_ns / 1000, 'actual_local_AB_gain_scaled_fraction': conditional_saved_ns / engine_ns},
            'source_logical_S_bytes_not_DRAM': {'one_layer_read_plus_write': 2 * 48 * 128 * 128 * 4, '48_layer_read_plus_write': layers * 2 * 48 * 128 * 128 * 4, 'local_S_working_set': 15 * 48 * 128 * 128 * 4, '48_layer_S_working_set': layers * 48 * 128 * 128 * 4},
            'GPU_or_resource_probe_executed_by_reviewer': False, 'rerun_or_integration': False}
write(D / 'selected-fields.json', selected)
h(P / 'v44-gdn-activated-normreuse-build1/gdn_activated_norm_reuse_cached_metal.py')
h(Path(__file__))
review = f'''# V85 作者自审（非独立审计）

仅 CPU 复核保存的结果/源码/哈希；未运行 GPU、资源探测、micro 或集成，旧证据未改。

- receipt SHA `{h(receipt)}`，状态仍为 `STOP_MICRO_GATE`。
- 64 pins 当前字节一致，含原 56 pins 严格不变子集；原 V81 receipt 绑定正确。
- 30 public +60 captured +480 native strict；20 timed blocks ×30 最终 o/S 数组 =600 strict。不能宣称每个 dispatch 输出都验证。
- 17 numeric +10 pair 只读检查，各 255 原数组；240 原 allocation 与180 mutable arena extent 重新核验不重叠；实际 S6 ReadOnly/S_new9、四 PSO 与依赖 barrier 一致。
- numeric/prime/timed dispatch =2,760/76,800/614,400，总693,960；每 timed block 30,720。
- 重算 A/B = {medians['A']:.10f}/{medians['B']:.10f} ns/dispatch；下降 {gain*100:.8f}%，6/6 胜；controls PASS，仍低于5%门槛。
- 24 原始资源样本与汇总一致：AC、thermal0、pressure1；swap-used5006.88MB / Swapouts17460761 均未变；已知推理进程仅归属 driver95239。
- owner95238/driver95239，rc2、wait/reap、两 PID absence、external release PASS 均有保存证据；这里只核查这些快照，不声称持续观测所有 GPU 使用。
- 20 个 official CB seconds 时间戳有效；仅支持本微测整个 CB envelope。A 中位数为 {gpu_medians['A']*2/1000:.6f} µs/bank，wall 则为 {medians['A']*2/1000:.6f} µs/bank。
- `×48≈0.91ms` 只能是同成本假设下的外推情景，不能证明真实整模份额或严格上界。详见 SOURCE-ADVICE.md。

结论：保存计数、统计、hash、资源与 STOP 一致；不重跑、不采用。复核断言 {len(checks)} 项，脚本/源映射附后。
'''
(D / 'REVIEW.md').write_text(review)
advice = f'''# GDN 两 kernel：证据边界与下一步源码优先级（作者自审建议）

V85 保持 STOP：局部 wall A/B {medians['A']*2/1000:.6f}/{medians['B']*2/1000:.6f} µs/bank，下降 {gain*100:.6f}%、6/6 胜、controls PASS，仍未达5%。无需重跑或集成；以下是源码方向建议，不是新实现/性能计划。

| 可核实的量 | 实际范围 |
|---|---|
| A wall {medians['A']*2/1000:.6f} µs/bank | 15 fixed banks、每 bank 两 dispatch，128 prime/1024 timed；含 amortized native encode/commit/wait 和 barrier |
| A GPU CB envelope {gpu_medians['A']*2/1000:.6f} µs/bank | official `(GPUEndTime−GPUStartTime)` seconds，整条 micro CB 含调度/barrier；不是独占 shader 时长 |
| 历史 engine {engine_ns/1e6:.9f} ms/step | V71 三次 measure 的 `(t2−t1)/31` 中位数，包含 begin/reset、131 seed uploads 和 engine 调度；不同候选/进程/时钟口径 |
| 48 GDN layers | 固定 config 的64层中48个 linear_attention；这是源码/模型结构计数 |

因此 **不能从现有 timestamp 严格证明 GDN 只占整模约0.91ms，或其优化上限约2.6%**。能严格换算的是本微测每 bank 的 amortized wall/CB envelope。若额外假设48层成本与这里完全相同、其他工作不变，wall 外推为 {wall_scaled_ns/1e6:.6f}ms（参考 engine 的 {wall_scaled_ns/engine_ns*100:.4f}%），GPU-envelope 外推为 {gpu_scaled_ns/1e6:.6f}ms。把已见3.32462%局部收益继续同假设外推，才得到约 {conditional_saved_ns/1000:.3f}µs/step、{conditional_saved_ns/engine_ns*100:.4f}% latency；这些是条件情景，不是测得的整模收益，也不是上界。

跨 workload 的关键差异：micro 的15个固定 S 共45MiB，无反馈、独立 bank、同一 CB 长循环；48层 S 则为144MiB，整模还存在 token 状态反馈、投影→GDN→后继算子的 producer/consumer、缓存和调度变化。micro 外部已 materialize activation；它也没有投影/attention/MLP 工作交错。历史34.690ms还有 seed/setup 摊销。CB envelope 不能拆为各 shader 的 exclusive-ms，亦不能用这两个时钟直接相减。

源码上，V44 helper:63–74 已将16组 Q/K serial norm/rsqrt集中到 prework；:103–120 的 held32 state 对每个 S 元素保留一次读、一次 canonical S_new 写。每层逻辑 S read+write为6MiB，48层288MiB；这是源码逻辑字节数，不能当实测 DRAM 流量。V78只提前 materialize FP32 scaled Q/K，实际小幅局部改善与这个编辑边界一致，但还不足采用。

建议保留现有 defaultOFF 与旧 STOP，把接下来的 CPU 源码研究优先级放到整模关键路径的权重投影/搬运和既有融合边界：先核对原 pointer、量化解码/scale复用、读写布局及 producer/consumer 链，复用已有严格数学/负例约束。此处只给优先级判断；当前证据没有证明某个投影新方案会快，也没有授权新增 shader、perf 或整模测量。

证据：V85 actual receipt83adff18…d6c083；V71 db712cea…87d21c；V44 helper5f05383c…bca952；完整路径、SHA与复算值见 source-map.json / selected-fields.json。
'''
(D / 'SOURCE-ADVICE.md').write_text(advice)
(D / 'CHECKPOINT.md').write_text('COMPLETE: V85 AUTHOR SELF audit only; original STOP unchanged. No GPU/probe/rerun/integration. See REVIEW.md, SOURCE-ADVICE.md and verification.json.\n')
write(D / 'source-map.json', {'review_type': 'AUTHOR_SELF_NOT_INDEPENDENT', 'refs': dict(sorted(refs.items()))})
verification = {'status': 'PASS_AUTHOR_SELF_SAVED_EVIDENCE_AUDIT_STOP_UNCHANGED', 'assertions': len(checks), 'GPU_used': False, 'resource_probes_run': False, 'original_evidence_modified': False, 'checks': checks,
                'outputs': {name: hashlib.sha256((D / name).read_bytes()).hexdigest() for name in ('review.py', 'REVIEW.md', 'SOURCE-ADVICE.md', 'selected-fields.json', 'source-map.json')}}
write(D / 'verification.json', verification)
print(json.dumps({'status': verification['status'], 'assertions': len(checks), 'A_ns': medians['A'], 'B_ns': medians['B'], 'gain': gain, 'wins': wins, 'controls': control_passed, 'conditional_A48_ms_NOT_bound': wall_scaled_ns / 1e6, 'GPU_A48_ms_NOT_bound': gpu_scaled_ns / 1e6, 'reference_engine_ms_different_scope': engine_ns / 1e6, 'verification_SHA': hashlib.sha256((D / 'verification.json').read_bytes()).hexdigest()}))
