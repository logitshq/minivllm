"""Stdlib E018 initial/repeat evidence audit. Run only after sole-GPU timing ends.

Lives outside gdn_verify/ because screen pins every Python file in that folder.
"""
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import struct
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / '.artifacts/m5-ultra-256gb'
OUT = ROOT / 'docs/experiments/m5-ultra-256gb/E018-persistent-gdn-verify-v1'
RUNS = ['gdn-verify-v1'] + [f'gdn-verify-repeat-v1-run{i}' for i in (1, 2, 3)]
LENGTHS = [1, 3, 5, 8]
STATES = ['real', 'zero', 'independent_random']
PREREG_COMMIT = 'fe4ea8f'
TUTORIAL = 'docs/optimization-guide/tutorials/018-persistent-gdn-verify.md'


def read(path):
    return json.loads(path.read_text())


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def audit_inputs(path, metadata):
    raw = path.read_bytes()
    assert sha(raw) == metadata['sha256']
    header_size, = struct.unpack('<Q', raw[:8])
    header = json.loads(raw[8:8 + header_size])
    assert header.pop('__metadata__', None) in (None, {})
    body = raw[8 + header_size:]
    assert set(header) == set(metadata['arrays']) == {'q', 'k', 'v', 'g', 'beta', 'initial_state'}
    expected_shapes = {'q': [8, 1, 48, 128], 'k': [8, 1, 48, 128], 'v': [8, 1, 48, 128],
                       'g': [8, 1, 48], 'beta': [8, 1, 48], 'initial_state': [1, 48, 128, 128]}
    result, intervals = {}, []
    for name, item in header.items():
        dtype = 'BF16' if name in ('q', 'k', 'v') else 'F32'
        assert item['dtype'] == dtype and item['shape'] == expected_shapes[name] == metadata['arrays'][name]['shape']
        assert metadata['arrays'][name]['dtype'] == ('mlx.core.bfloat16' if dtype == 'BF16' else 'mlx.core.float32')
        lo, hi = item['data_offsets']
        assert 0 <= lo < hi <= len(body) and hi - lo == math.prod(item['shape']) * (2 if dtype == 'BF16' else 4)
        intervals.append((lo, hi))
        result[name] = {'shape': item['shape'], 'dtype': item['dtype'], 'sha256': sha(body[lo:hi]), 'bytes': hi - lo}
    intervals.sort()
    assert intervals[0][0] == 0 and intervals[-1][1] == len(body)
    assert all(a[1] == b[0] for a, b in zip(intervals, intervals[1:]))
    return result


def audit_run(run, workload):
    plan, summary = read(run / 'plan.json'), read(run / 'summary.json')
    assert plan['identity']['mlx'] == '0.32.3' and plan['identity']['device']['device_name'] == 'Apple M5 Ultra'
    assert plan['identity']['device']['memory_size'] == 256 * 1024 ** 3
    assert plan['lengths'] == LENGTHS and plan['states'] == STATES
    assert plan['warmups_per_arm'] == 2 and plan['abba_blocks'] == 3 and plan['pool_limit_bytes'] == 2 * 1024 ** 3
    assert plan['gate'] == {'all_output_and_history_bytes_exact': True, 'all_keep_prefixes_exact': True,
        'initial_state_unchanged': True, 'T3_T5_T8_geomean_speedup_min': 1.25, 'T1_speedup_min': .97, 'no_T_regression_below': .97}
    metadata = read(run / 'real-inputs.json')
    assert metadata['layer_index'] == 0 and metadata['prefix_ids'] == workload['prompt_ids']
    assert metadata['input_ids'] == workload['continuation_ids'][:8]
    inputs = audit_inputs(run / 'real-inputs.safetensors', metadata)
    checks = read(run / 'qualification.json')
    assert [(r['state'], r['T']) for r in checks] == [(s, t) for s in STATES for t in LENGTHS]
    prior, zero_sha = {}, sha(bytes(1 * 48 * 128 * 128 * 4))
    for row in checks:
        assert row == read(run / f"check-{row['state']}-T{row['T']}.json")
        assert row['passed'] is True and row['initial_state_unchanged'] is True
        length = row['T']
        labels = [label for t in range(length) for label in (f'output-{t}', f'history-{t}')]
        labels += [f'independent-prefix-keep-{keep}' for keep in range(length + 1)]
        assert [c['label'] for c in row['comparisons']] == labels and len(labels) == 3 * length + 1
        lookup = {c['label']: c for c in row['comparisons']}
        for c in row['comparisons']:
            assert c['exact'] is True and c['dtype'] == 'mlx.core.float32'
            assert c['shape'] == ([1, 48, 128] if c['label'].startswith('output-') else [1, 48, 128, 128])
            assert c['reference_sha256'] == c['candidate_sha256'] and len(c['reference_sha256']) == 64
            int(c['reference_sha256'], 16)
            key = row['state'], c['label']
            if key in prior:
                assert prior[key] == c['reference_sha256'], 'same prefix differed across T'
            prior[key] = c['reference_sha256']
        for keep in range(1, length + 1):
            assert lookup[f'independent-prefix-keep-{keep}']['reference_sha256'] == lookup[f'history-{keep - 1}']['reference_sha256']
        if row['state'] in ('real', 'zero'):
            expected = inputs['initial_state']['sha256'] if row['state'] == 'real' else zero_sha
            assert lookup['independent-prefix-keep-0']['reference_sha256'] == expected
    assert summary['qualification_passed'] and summary['qualification_cases'] == 12 and summary['exact_comparisons'] == 165
    assert len({prior[s, 'independent-prefix-keep-0'] for s in STATES}) == 3
    assert not summary['cold_compilation_claim'] and summary['no_whole_model_correctness_or_speed_claim']
    rows = []
    for length in LENGTHS:
        item = read(run / f'timing-T{length}.json')
        assert item['T'] == length
        assert set(item['first_post_qualification_ms']) == {'reference', 'candidate'}
        assert set(item['warmups_ms']) == {'reference', 'candidate'}
        assert all(len(values) == 2 for values in item['warmups_ms'].values())
        assert [(r['block'], r['arm']) for r in item['rounds']] == [(block, arm) for block in range(3) for arm in ('reference', 'candidate', 'candidate', 'reference')]
        samples = list(item['first_post_qualification_ms'].values())
        samples += [v for values in item['warmups_ms'].values() for v in values] + [r['ms'] for r in item['rounds']]
        assert len(samples) == 18 and all(math.isfinite(x) and x > 0 for x in samples)
        medians = {arm: statistics.median(r['ms'] for r in item['rounds'] if r['arm'] == arm) for arm in ('reference', 'candidate')}
        assert medians == item['medians_ms']
        speedup = medians['reference'] / medians['candidate']
        assert math.isclose(speedup, item['speedup'], rel_tol=1e-12)
        rows.append(item)
    assert rows == summary['timing']
    gm = statistics.geometric_mean(r['speedup'] for r in rows if r['T'] != 1)
    advance = gm >= 1.25 and all(r['speedup'] >= .97 for r in rows)
    assert math.isclose(gm, summary['T3_T5_T8_geomean_speedup'], rel_tol=1e-12)
    assert summary['advance'] is advance and summary['status'] == ('PASS_MODULE_SCREEN' if advance else 'REJECT_PERFORMANCE_GATE')
    assert summary['scope'] == plan['scope']
    memory = {key: summary[key] for key in ('micro_peak_memory_bytes', 'preparation_peak_memory_bytes', 'after_target_release_active_memory_bytes', 'before_timing_active_memory_bytes')}
    assert all(type(v) is int and v >= 0 for v in memory.values())
    owner = run.with_name(run.name + '-owner')
    child, release, completed = [read(owner / name) for name in ('child.json', 'release.json', 'complete.json')]
    assert child['pid'] == release['pid'] and release['returncode'] == 0 and release['waited_reaped'] and release['strict_absence']
    assert completed['status'] == 'PASS'
    resources = [read(p) for p in sorted(owner.glob('resource-*.json'))]
    assert len(resources) >= 3
    for resource in resources:
        assert resource['pressure_level_value'] == 1 and resource['processinfo_thermal_metadata']['code'] == 0
        assert resource['no_active_foreign_native_processes'] and resource['power_metadata']['source'] == 'AC Power'
        paused = [p for p in resource['native_inference_processes'] if p['pid'] == 74384]
        assert len(paused) == 1 and paused[0]['authorized_suspended'] and 'T' in paused[0]['state']
    assert len({r['swap_used_mb'] for r in resources}) == len({r['vm_counters']['Swapouts'] for r in resources}) == 1
    receipt = {'run': run.name, 'advance': advance, 'T3_T5_T8_geomean_speedup': gm,
               'timing': rows, 'qualification_cases': 12, 'exact_comparisons': 165,
               'formal_samples': 48, 'warm_samples': 16, 'first_post_qualification_samples': 8,
               'input_file_sha256': metadata['sha256'], 'input_arrays': inputs, 'memory': memory,
               'resource_samples': len(resources), 'swap_used_mb': resources[0]['swap_used_mb'],
               'swapouts': resources[0]['vm_counters']['Swapouts'], 'release': release, 'owner_pid': completed['owner_pid']}
    return plan, receipt, prior


def main():
    assert not OUT.exists(), 'Immutable evidence: use a new archive version'
    workload_path = BASE / 'fullqual-build2/workload.json'
    workload = read(workload_path)
    audited = [audit_run(BASE / name, workload) for name in RUNS]
    plans, rows, comparison_hashes = zip(*audited)
    assert all(plan == plans[0] for plan in plans), 'same-source, same-model, same-settings replication required'
    assert len({r['release']['pid'] for r in rows}) == 4
    assert all(r['input_arrays'] == rows[0]['input_arrays'] for r in rows), 'input tensor bytes differed across processes'
    assert all(checks == comparison_hashes[0] for checks in comparison_hashes), 'output/history receipts differed across processes'
    files = {}
    def collect(path, name=None):
        path = Path(path)
        name = name or str(path.relative_to(ROOT))
        data = path.read_bytes()
        if name in files:
            assert files[name] == data
        files[name] = data
        return data
    assert sha(collect(workload_path)) == plans[0]['workload_sha256']
    for source, expected in plans[0]['source_sha256'].items():
        assert sha(collect(Path(source))) == expected, source
    for name, expected in plans[0]['model']['files'].items():
        if not name.endswith('.safetensors'):
            assert sha(collect(Path(plans[0]['model']['path']) / name, f'model-metadata/{name}')) == expected, name
    for name in RUNS:
        for folder in (BASE / name, BASE / (name + '-owner')):
            for path in sorted(folder.rglob('*')):
                if path.is_file():
                    collect(path)
    resumes = []
    for prefix, selected in (('gdn-verify-v1', rows[:1]), ('gdn-verify-repeat-v1', rows[1:])):
        folder = BASE / (prefix + '-service-lease')
        lease, resume = read(folder / 'lease.json'), read(folder / 'resume.json')
        jobs_path = BASE / (prefix + '-jobs.json')
        jobs = read(jobs_path)
        assert lease['closed'] and lease['service_pid'] == 74384 and lease['jobs'] == jobs and len(jobs) == len(selected)
        assert resume['identity_match'] and resume['state_before'] == 'T' and resume['state_after'] == 'S'
        for index, row in enumerate(selected):
            job = read(folder / f'job-{index}.json')
            assert job['command'] == jobs[index] and job['pid'] == row['owner_pid'] and job['returncode'] == 0 and job['waited']
        resumes.append({'lease': prefix, **resume})
        collect(jobs_path)
        for path in sorted(folder.rglob('*')):
            if path.is_file():
                collect(path)
    collect(BASE / 'gdn-verify-v1-cpu.json')
    collect(Path(__file__))
    for name in ('guarded_v2.py', 'resources_v2.py', 'suspended_service.py'):
        collect(ROOT / 'bench/m5_ultra_256gb' / name)
    prereg = subprocess.check_output(['git', '-C', str(ROOT), 'show', PREREG_COMMIT + ':' + TUTORIAL])
    assert '三个' in prereg.decode() and '不与复测池化' in prereg.decode()
    files['preregistration/tutorial018-fe4ea8f.md'] = prereg
    repeated = all(row['advance'] for row in rows[1:])
    summary = {'status': 'PASS_INDEPENDENT_AUDIT', 'reproducibility_gate_passed': repeated,
               'decision': 'Advance only to multi-layer/transaction qualification' if repeated else 'Do not claim repeatability or integrate; at least one preregistered fresh process failed its own gate',
               'initial_screen': rows[0], 'fresh_repeats': list(rows[1:]), 'pooled_timing_statistic': None,
               'qualification_cases': 48, 'exact_comparisons': 660, 'formal_samples': 192, 'warm_samples': 64, 'first_post_qualification_samples': 32,
               'source_files_checked': len(plans[0]['source_sha256']), 'all_input_array_bytes_identical_across_processes': True,
               'all_comparison_hashes_identical_across_processes': True, 'service_resumes': resumes,
               'weight_hash_scope': 'screen checked model file SHA before and after each run; CPU audit checks metadata and retains payload hash receipts without rereading large weights',
               'correctness_scope': 'CPU audit validates recorded equality flags, shapes, hashes and prefix relationships; no GPU recomputation of output/state arrays',
               'scope': plans[0]['scope'], 'no_whole_model_or_HTTP_speed_claim': True}
    OUT.mkdir(parents=True)
    archive = OUT / 'evidence.tar.gz'
    with tarfile.open(archive, 'w:gz') as tar:
        for name, data in sorted(files.items()):
            item = tarfile.TarInfo(name)
            item.size = len(data)
            tar.addfile(item, io.BytesIO(data))
    manifest = {name: {'bytes': len(data), 'sha256': sha(data)} for name, data in sorted(files.items())}
    with tarfile.open(archive) as tar:
        assert len(tar.getnames()) == len(manifest) and set(tar.getnames()) == set(manifest)
        for name, item in manifest.items():
            data = tar.extractfile(name).read()
            assert len(data) == item['bytes'] and sha(data) == item['sha256']
    archive_sha = sha(archive.read_bytes())
    write(OUT / 'manifest.json', {'archive_sha256': archive_sha, 'files': manifest})
    write(OUT / 'tar.verify.json', {'status': 'PASS', 'archive_sha256': archive_sha, 'members': len(manifest), 'all_member_hashes_and_sizes_verified': True})
    write(OUT / 'summary.json', summary)
    table = '\n'.join('| ' + ('initial' if i == 0 else f'repeat {i}') + ' | ' + ' | '.join(f"{t['speedup']:.6f}×" for t in row['timing']) + f" | {row['T3_T5_T8_geomean_speedup']:.6f}× | {'PASS' if row['advance'] else 'REJECT'} |" for i, row in enumerate(rows))
    (OUT / 'README.md').write_text(f'''# E018：persistent GDN micro 初筛及三 fresh process 复测

独立审计通过；复测门结果 **{'PASS' if repeated else 'REJECT'}**。预注册要求三次 fresh process 各自 T3/5/8 几何平均至少 1.25×，任何 T 至少 0.97×。初筛单独呈现，没有池化，也没有补跑挑选。

| Run | T1 | T3 | T5 | T8 | GM T3/5/8 | Gate |
|---|---:|---:|---:|---:|---:|---|
{table}

每进程 12 cases、165 个输出/FP32 history/独立 keep-prefix 比较。四进程共 **660** 个记录逐项通过；shape、dtype、reference/candidate SHA 与前缀关系独立核对，initial state 保持不变。四份 safetensors 的每个输入 tensor 字节相同，BF16 q/k/v 与 FP32 gates/state dtype 保留。真实输入来自同一 19-token prefix 后首层的八个归档 continuation tokens，不声称为整模 greedy rollout。

每进程保留 48 formal、16 warm、8 first-post-qualification 样本，总计 192/64/32；ABBA 顺序及 median/speedup 独立重算。first 是资格后的调用，不能称冷编译。逐进程 memory、资源、release 均保存于 summary。所有资源快照 pressure=1、thermal=0，各 run swap/Swapouts 不增长；四个模型子进程 exit 0、wait/reap、strict absence 通过，两轮授权服务 lease 均恢复 T→S。

`evidence.tar.gz` 保存 initial 与三 repeats 的全部 raw、owner、lease/jobs、CPU 结果、34 个被测源、workload、模型 metadata 和 commit fe4ea8f 的复测预注册文本。{len(manifest)} 个成员逐 size/SHA 核验通过。权重本体不入 archive；其前后 SHA 校验由每次 screen 执行，CPU audit 不重复读取大 payload。

本实验只比较同一 literal GDN helper 的串行链与一个 kernel 内的串行链。CPU 审计核查设备比较记录，不重新执行 GPU 数值运算。不证明 chunk-WY 数学等价、整模 rollback/事务正确性、长生成或 HTTP 速度；核心默认未采用。
''')
    print(json.dumps({'status': summary['status'], 'reproducibility_gate_passed': repeated, 'members': len(manifest), 'archive_bytes': archive.stat().st_size}))


if __name__ == '__main__':
    main()
