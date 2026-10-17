"""CPU-only E016 receipt audit and immutable archive; no MLX import."""
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import tarfile

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / '.artifacts/m5-ultra-256gb'
OUT = ROOT / 'docs/experiments/m5-ultra-256gb/E016-draft-graph-v1'


def read(path):
    return json.loads(path.read_text())


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def main():
    assert not OUT.exists(), 'Evidence is immutable; use a new version'
    run = BASE / 'draft-graph-v1'
    owner = BASE / 'draft-graph-owner-v1'
    lease_dir = BASE / 'draft-graph-service-lease-v1'
    plan, done = read(run / 'plan.json'), read(run / 'complete.json')
    checks, cold, timing = [read(run / name) for name in ('checks.json', 'first-calls.json', 'timing.json')]
    assert done['status'] == 'PASS' and len(checks) == done['checks'] == 24
    expected = {f'{name}-B{block}{suffix}' for name in ('math', 'code', 'chinese')
                for block in (3, 5, 8) for suffix in ('', '-position-plus7')}
    expected |= {f'interleaved-{name}-step{step}' for name in ('math', 'code') for step in range(3)}
    assert len({row['case'] for row in checks}) == 24 and {row['case'] for row in checks} == expected
    for row in checks:
        assert len(row['bit_equal']) == len(row['array_order']) == 13 and all(value is True for value in row['bit_equal'])
        assert row['reference_ids'] == row['candidate_ids'] and len(row['reference_ids']) == row['block'] - 1
        assert row['array_order'] == ['backbone_hidden', 'normalized_hidden', 'logits'] + [
            f'layer_{layer}_{kind}' for layer in range(5) for kind in ('K', 'V')]
    seeds = read(run / 'interleaved-seeds.json')
    assert len(seeds['seed_kv_bit_equal']) == 10 and not any(seeds['seed_kv_bit_equal'])
    assert [row['case'] for row in cold] == [row['case'] for row in checks]
    previous, traces = 0, []
    for row in cold:
        assert math.isfinite(row['ms']) and row['ms'] > 0
        assert row['traces_before'] == previous
        assert row['traces_after'] - previous == len(row['new_shapes']) in (0, 1)
        if row['case'].endswith('-position-plus7') or row['case'].startswith('interleaved-code-'):
            assert not row['new_shapes'], 'Value-only change must reuse the shape graph'
        traces.extend(row['new_shapes'])
        previous = row['traces_after']
    assert previous == done['trace_count'] == 9
    assert len({json.dumps(trace, sort_keys=True) for trace in traces}) == 9
    assert len(timing) == done['timing_cases'] == 9
    assert {(row['prompt'], row['block']) for row in timing} == {
        (name, block) for name in ('math', 'code', 'chinese') for block in (3, 5, 8)}
    rows = []
    for row in timing:
        item = row['full_draft']
        assert set(item['first_post_qualification_ms']) == {'reference', 'candidate'}
        assert [(value['repeat'], value['arm']) for value in item['warm']] == [
            (repeat, arm) for repeat in range(2) for arm in ('reference', 'candidate')]
        assert [(value['repeat'], value['arm']) for value in item['samples']] == [
            (repeat, arm) for repeat in range(3) for arm in ('reference', 'candidate', 'candidate', 'reference')]
        assert all(math.isfinite(value) and value > 0 for value in item['first_post_qualification_ms'].values())
        assert all(math.isfinite(value['ms']) and value['ms'] > 0 for value in item['warm'] + item['samples'])
        medians = {arm: statistics.median(value['ms'] for value in item['samples'] if value['arm'] == arm)
                   for arm in ('reference', 'candidate')}
        assert medians == item['median_ms']
        speedup = medians['reference'] / medians['candidate']
        assert math.isclose(speedup, item['speedup'], rel_tol=1e-12)
        rows.append({'prompt': row['prompt'], 'block': row['block'], 'median_ms': medians, 'speedup': speedup})
    gain = statistics.geometric_mean(row['speedup'] for row in rows)
    minimum = min(row['speedup'] for row in rows)
    assert math.isclose(gain, done['full_draft_geomean_speedup'], rel_tol=1e-12)
    assert minimum == done['minimum_speedup']
    advance = gain >= 1.10 and minimum >= .97
    assert done['advance'] == advance is False
    release, child = read(owner / 'release.json'), read(owner / 'child.json')
    assert release['pid'] == child['pid'] and release['returncode'] == 0
    assert release['waited_reaped'] and release['strict_absence']
    assert read(owner / 'complete.json')['status'] == 'PASS'
    lease, resume, job = [read(lease_dir / name) for name in ('lease.json', 'resume.json', 'job-0.json')]
    assert lease['closed'] and lease['service_pid'] == 74384
    assert resume['identity_match'] and 'T' in resume['state_before'] and 'T' not in resume['state_after']
    assert job['pid'] == lease['child_pid'] == read(owner / 'complete.json')['owner_pid']
    assert job['returncode'] == 0 and job['waited']
    assert lease['jobs'] == read(BASE / 'draft-graph-jobs-v1.json') and lease['jobs'][0] == job['command']
    resources = [read(path) for path in sorted(owner.glob('resource-*.json'))]
    assert len(resources) == 5
    for resource in resources:
        assert resource['pressure_level_value'] == 1 and resource['processinfo_thermal_metadata']['code'] == 0
        assert resource['no_active_foreign_native_processes'] and resource['power_metadata']['source'] == 'AC Power'
        suspended = [proc for proc in resource['native_inference_processes'] if proc['pid'] == 74384]
        assert len(suspended) == 1 and suspended[0]['authorized_suspended'] and 'T' in suspended[0]['state']
    assert len({resource['swap_used_mb'] for resource in resources}) == 1
    assert len({resource['vm_counters']['Swapouts'] for resource in resources}) == 1
    files = {}
    def collect(path, name=None):
        files[name or str(path.relative_to(ROOT))] = path.read_bytes()
    for source, digest in plan['source_sha256'].items():
        path = ROOT / source
        assert sha(path.read_bytes()) == digest, source
        collect(path)
    for folder in (run, owner, lease_dir):
        for path in sorted(folder.rglob('*')):
            if path.is_file():
                collect(path)
    collect(BASE / 'draft-graph-jobs-v1.json')
    collect(Path(__file__))
    for name in ('guarded_v2.py', 'resources_v2.py', 'suspended_service.py'):
        collect(ROOT / 'bench/m5_ultra_256gb' / name)
    workload = BASE / 'fullqual-build2/workload.json'
    assert sha(workload.read_bytes()) == plan['workload_sha256']
    collect(workload)
    for name in ('target', 'drafter'):
        config = Path(plan[f'{name}_dir']) / 'config.json'
        assert sha(config.read_bytes()) == plan[f'{name}_config_sha256']
        collect(config, f'model-metadata/{name}-config.json')
    traced_calls = [row for row in cold if row['new_shapes']]
    summary = {'status': 'PASS_AUDIT_REJECT_OPTIMIZATION', 'decision': 'Do not integrate; fixed-shape full-draft gain fails preregistered 1.10x gate',
        'correctness_cases': 24, 'array_checks': 312, 'timing_cases': 9, 'timed_samples': 108,
        'full_draft_geomean_speedup': gain, 'minimum_speedup': minimum, 'advance': advance,
        'trace_count': len(traces), 'first_graph_call_ms': traced_calls[0]['ms'],
        'later_new_shape_first_call_ms_range': [min(row['ms'] for row in traced_calls[1:]), max(row['ms'] for row in traced_calls[1:])],
        'interleaved_seed_length': seeds['seed_length'], 'all_ten_seed_arrays_different': True,
        'value_only_position_and_request_changes_reused_graph': True,
        'peak_active_gib': done['peak_active_gib'], 'rows': rows,
        'scope': 'One process, fixed-shape draft module only. Rejected before long generation, buckets, target verification or HTTP.'}
    OUT.mkdir(parents=True)
    archive = OUT / 'evidence.tar.gz'
    with tarfile.open(archive, 'w:gz') as tar:
        for name, data in sorted(files.items()):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    manifest = {name: {'bytes': len(data), 'sha256': sha(data)} for name, data in sorted(files.items())}
    with tarfile.open(archive) as tar:
        assert set(tar.getnames()) == set(manifest) and len(tar.getnames()) == len(manifest)
        for name, metadata in manifest.items():
            data = tar.extractfile(name).read()
            assert len(data) == metadata['bytes'] and sha(data) == metadata['sha256']
    archive_sha = sha(archive.read_bytes())
    write(OUT / 'manifest.json', {'archive_sha256': archive_sha, 'files': manifest})
    write(OUT / 'tar.verify.json', {'status': 'PASS', 'archive_sha256': archive_sha, 'members': len(manifest), 'all_member_hashes_and_sizes_verified': True})
    write(OUT / 'summary.json', summary)
    write(OUT / 'audit.json', {'status': 'PASS', 'source_hashes_checked': len(plan['source_sha256']),
        'resource_samples': len(resources), 'pressure': 1, 'thermal': 0,
        'swap_used_mb': resources[0]['swap_used_mb'], 'swapouts': resources[0]['vm_counters']['Swapouts'],
        'release': release, 'resume': resume, 'archive_members': len(manifest),
        'model_weight_hashes': 'Recorded by screen before model load; retained in plan, not redundantly reread or archived by this CPU audit',
        'scope': 'Independently recomputed timing statistics, checked strict-check receipts, source identities, ownership, restoration and archive hashes; no GPU rerun'})
    table = '\n'.join(f"| {row['prompt']} | {row['block']} | {row['median_ms']['reference']:.3f} | {row['median_ms']['candidate']:.3f} | {row['speedup']:.4f}× |" for row in rows)
    (OUT / 'README.md').write_text(f'''# E016：draft backbone 图复用，拒绝采用

固定 shape 完整 draft 几何平均 **{gain:.4f}×**，最低 **{minimum:.4f}×**，未达到预注册 **1.10×** 门槛。单进程初筛结束，未合入生产，也未进入 bucket 或长生成资格。

| Prompt | Block | Reference ms | Graph ms | Speedup |
|---|---:|---:|---:|---:|
{table}

24 组检查的 312 个数组比较全部严格通过，草稿 ID 相同。检查包括 backbone hidden、normalized hidden、完整 logits 和每层 KV；math/code 两个 seed 的十个 KV 实际都不同，再交错推进各三步，未串请求。位置变化与同 shape 切换请求均复用已有图。

首次图调用 **{traced_calls[0]['ms']:.3f} ms**；其余新 shape 首调 **{summary['later_new_shape_first_call_ms_range'][0]:.3f}–{summary['later_new_shape_first_call_ms_range'][1]:.3f} ms**。共 9 次 trace；三个增长 KV 步骤每步新增 shape。所有首调和 warm 样本保留。这里的首调包含编译及实际运算，不能把差值称为纯编译时间。

峰值 MLX active **{done['peak_active_gib']:.3f} GiB**。5 次资源快照 pressure=1、thermal=0，swap 数值与 Swapouts 未增加；已知外部推理进程只有获授权暂停的服务。子进程 exit 0、wait/reap 与严格不存在检查通过，服务 PID 74384 身份匹配且从 T 恢复至 {resume['state_after']}。

`summary.json` 是重算结论，`audit.json` 是独立 CPU 审计，`evidence.tar.gz` 保存所有原始 run/owner/lease/jobs、逐 hash 核验的被测源、workload 与模型 config。`manifest.json` 和 `tar.verify.json` 核验全部 {len(manifest)} 个成员。模型权重 SHA 保存在原始 plan，权重本体未复制进 Git。

这只是该模型、短上下文、固定 shape 的 draft 模块初筛。没有 target verification、rollback、acceptance、生成或 HTTP 性能结论。
''')
    print(json.dumps({'status': summary['status'], 'gain': gain, 'array_checks': 312, 'archive_members': len(manifest)}))


if __name__ == '__main__':
    main()
