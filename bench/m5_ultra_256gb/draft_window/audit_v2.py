"""Independent CPU-only audit/archive of the predeclared E014-v2 2/16 GiB runs.

Reads frozen measured sources, never the subsequently edited working tree core.
Does not execute the measured implementation or import MLX.
"""
import datetime
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / '.artifacts/m5-ultra-256gb'
OUT = ROOT / 'docs/experiments/m5-ultra-256gb/E014-draft-window-v2'
LENGTHS = (0, 31, 2040, 2047, 2048, 2049, 4095, 4096, 4097, 32768)
TIMING_LENGTHS = (32, 2048, 8192, 32768, 131072)
PHASES = ('prepare_ms', 'draft_ms', 'teardown_ms', 'owned_lifecycle_ms')


def read(path):
    return json.loads(path.read_text())


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')


def expected_checks():
    rows = {}
    def append(label, block, before, physical, incoming):
        add = incoming or 0
        logical, resident = before + add, physical + add
        if resident > 4096:
            resident = 2048
        rows[(label, block)] = (logical, resident, incoming)
        return logical, resident
    for block in (3, 5, 8):
        append('cold-None-plus-real-context', block, 0, 0, 4)
    for length in LENGTHS:
        for block in (3, 5, 8):
            logical, physical = length, length if length <= 4096 else 2048
            for step, incoming in enumerate((4, None, 0, 1, 7)):
                logical, physical = append(f'replay-N{length}-step{step}', block, logical, physical, incoming)
            if length == 31:
                for n in (2049, 4097):
                    append(f'projection-input-{n}', block, 31, 31, n)
    for length in (4095, 4096, 4097):
        logical, physical = length, length if length <= 4096 else 2048
        for step, (block, incoming) in enumerate(((8, 1), (3, None), (5, 0), (8, 1), (3, 7))):
            logical, physical = append(f'variable-block-N{length}-step{step}', block, logical, physical, incoming)
    for length in TIMING_LENGTHS:
        for block in (3, 8):
            append(f'timing-replay-N{length}', block, length, length if length <= 4096 else 2048, 4)
    assert len(rows) == 184
    return rows


def validate_sample(s):
    assert all(type(s[k]) in (int, float) and math.isfinite(s[k]) and s[k] > 0 for k in PHASES)
    assert math.isclose(s['owned_lifecycle_ms'], s['draft_ms'] + s['teardown_ms'], rel_tol=1e-12)
    assert s['own_products_released_before_next_sample'] is True


def main():
    assert not OUT.exists(), 'Use a new archive version; never overwrite an existing evidence set'
    expected = expected_checks()
    files, table, resources, releases, pools = {}, [], [], [], []
    all_plans, all_checks = [], []
    jobs = read(BASE / 'draft-window-v2-jobs.json')
    lease_dir = BASE / 'draft-window-v2-service-lease'
    lease, resume = read(lease_dir / 'lease.json'), read(lease_dir / 'resume.json')
    assert lease['closed'] is True and lease['service_pid'] == 74384 and lease['jobs'] == jobs
    assert lease['service_identity'] == 'Fri Oct 16 08:01:58 2026     omlx-server'
    assert resume['identity_match'] is True and resume['state_before'] == 'T' and 'T' not in resume['state_after']

    def collect(path, name=None):
        key = name or str(path.relative_to(ROOT))
        data = path.read_bytes()
        if key in files:
            assert files[key] == data
        files[key] = data

    for job_index, pool in enumerate((2, 16)):
        run = BASE / f'draft-window-v2-cache{pool}'
        owner = BASE / f'draft-window-v2-cache{pool}-owner'
        plan, done = read(run / 'plan.json'), read(run / 'complete.json')
        checks, timings = read(run / 'checks.json'), read(run / 'timing.json')
        assert not list(run.glob('*failure*')) and not list(owner.glob('*failure*'))
        assert plan['experiment'] == 'E014-v2' and plan['cache_limit_gib'] == pool
        assert plan['predeclared_process_matrix_cache_gib'] == [2, 16]
        assert plan['check_lengths'] == list(LENGTHS) and plan['check_blocks'] == [3, 5, 8]
        assert plan['timing_lengths'] == list(TIMING_LENGTHS) and plan['timing_blocks'] == [3, 8]
        assert plan['model_hash_status'] == 'complete'
        assert done['status'] == 'PASS' and done['numeric_cases'] == len(checks) == 184
        assert done['timing_cases'] == len(timings) == 10 and done['cache_limit_gib'] == pool
        assert {(r['case'], r['block']) for r in checks} == set(expected)
        for row in checks:
            logical, physical, incoming = expected[(row['case'], row['block'])]
            assert row['start'] == row['logical_ctx_len'] == logical
            assert row['new_rows'] == incoming
            assert row['reference_physical_rows'] == [logical] * 5
            assert row['candidate_physical_rows'] == [physical] * 5
            assert row['reference_ids'] == row['candidate_ids'] and len(row['reference_ids']) == row['block'] - 1
            assert all(type(i) is int and 0 <= i < 248320 for i in row['reference_ids'])
            for name, count in (('selector_hidden_logits_bit_equal', 2), ('retained_suffix_bit_equal', 10),
                                ('attention_visible_suffix_bit_equal', 10), ('logical_length_equal', 5),
                                ('physical_at_most_2W', 5)):
                assert len(row[name]) == count and all(value is True for value in row[name])
            # Pinned checkpoint: 5 layers * K/V * 8 KV heads * 128 dims * BF16.
            assert row['tensor_logical_bytes'] == {'reference': logical * 20480, 'candidate': physical * 20480}
        frozen = read(run / 'frozen-source.json')
        assert frozen['status'] == 'PASS' and frozen['source_sha256'] == plan['source_sha256']
        assert len(plan['source_sha256']) == 33
        for source, digest in plan['source_sha256'].items():
            assert sha((run / 'frozen-source' / source).read_bytes()) == digest
        assert [(r['history_rows'], r['block']) for r in timings] == [(n, b) for n in TIMING_LENGTHS for b in (3, 8)]
        for row in timings:
            data, n, block = row['full_draft'], row['history_rows'], row['block']
            resident = n if n <= 4096 else 2048
            assert row['seed_tensor_logical_bytes'] == {'reference': n * 20480, 'candidate': resident * 20480}
            assert set(data['first_post_qualification']) == {'reference', 'candidate'}
            assert [(s['warm'], s['variant']) for s in data['warmup_samples']] == [(i, v) for i in range(2) for v in ('reference', 'candidate')]
            assert [(s['block'], s['arm'], s['variant']) for s in data['samples']] == [
                (i, a, v) for i in range(3) for a, v in enumerate(('reference', 'candidate', 'candidate', 'reference'))]
            for sample in [*data['first_post_qualification'].values(), *data['warmup_samples'], *data['samples']]:
                validate_sample(sample)
            medians = {p: {v: statistics.median(s[p] for s in data['samples'] if s['variant'] == v)
                           for v in ('reference', 'candidate')} for p in PHASES}
            assert medians == data['median_ms']
            gains = {p: medians[p]['reference'] / medians[p]['candidate'] for p in ('draft_ms', 'owned_lifecycle_ms')}
            assert all(math.isclose(gains[p], data['speedup'][p], rel_tol=1e-12) for p in gains)
            # Diagnose ABBA order without replacing the preregistered pooled median.
            order = {str(a): statistics.median(s['owned_lifecycle_ms'] for s in data['samples'] if s['arm'] == a) for a in range(4)}
            table.append({'cache_gib': pool, 'history_rows': n, 'block': block,
                          'median_ms': medians, 'speedup': gains, 'owned_lifecycle_ms_median_by_abba_arm': order})
        long = [r['speedup']['owned_lifecycle_ms'] for r in table if r['cache_gib'] == pool and r['history_rows'] in (32768, 131072)]
        short = [r['speedup']['owned_lifecycle_ms'] for r in table if r['cache_gib'] == pool and r['history_rows'] in (32, 2048)]
        gate = min(long) >= 1.10 and min(short) >= .97
        assert gate is done['advance'] and gate
        assert math.isclose(min(long), done['long_min_speedup'], rel_tol=1e-12)
        assert math.isclose(min(short), done['short_min_speedup'], rel_tol=1e-12)
        assert done['strong_reference_adoption_gate_passed'] is (True if pool == 16 else None)
        assert done['two_gib_compatibility_gate_passed'] is (True if pool == 2 else None)
        pools.append({'cache_gib': pool, 'gate': gate, 'long_min_speedup': min(long), 'short_min_speedup': min(short),
                      'formal_samples': 120, 'warmup_samples': 40, 'first_post_qualification_samples': 20})
        attempt, child, release = read(owner / 'attempt.json'), read(owner / 'child.json'), read(owner / 'release.json')
        owner_done, job_done = read(owner / 'complete.json'), read(lease_dir / f'job-{job_index}.json')
        assert owner_done['status'] == 'PASS' and attempt['owner_pid'] == owner_done['owner_pid'] == job_done['pid']
        assert job_done['command'] == jobs[job_index] and job_done['returncode'] == 0 and job_done['waited'] is True
        assert attempt['command'] == jobs[job_index][jobs[job_index].index('--cwd') + 2:]
        assert attempt['command'][attempt['command'].index('--cache-gib') + 1] == str(pool)
        assert child['pid'] == release['pid'] and release['returncode'] == 0 and release['waited_reaped'] is True and release['strict_absence'] is True
        current = subprocess.run(['/bin/ps', '-p', str(child['pid']), '-o', 'pid=,comm='], capture_output=True, text=True)
        assert current.returncode == 1 and not current.stdout and not current.stderr
        releases.append(dict(release, independently_absent_at_audit=True))
        samples = [read(p) for p in sorted(owner.glob('resource-*.json'))]
        assert len(samples) == 5 and [r['label'] for r in samples] == ['preflight', 'child-running', 'child-running', 'child-running', 'postflight']
        assert all(samples[i]['monotonic_s'] < samples[i + 1]['monotonic_s'] for i in range(len(samples) - 1))
        for r in samples:
            assert r['pressure_level_value'] == 1 and r['processinfo_thermal_metadata']['code'] == 0
            assert r['power_metadata']['source'] == 'AC Power' and r['process_inventory_ok'] and r['native_process_inventory_ok']
            assert r['no_active_foreign_native_processes'] is True
            stopped = [x for x in r['native_inference_processes'] if x['pid'] == 74384]
            assert len(stopped) == 1 and stopped[0]['authorized_suspended'] is True and 'T' in stopped[0]['state']
            if r['label'] == 'child-running':
                exact = r['exact_owned_pid_metadata']
                assert r['owned_model_pid'] == exact['pid'] == child['pid'] and exact['ppid'] == attempt['owner_pid'] and exact['status'] == 'known'
            timestamp = datetime.datetime.fromisoformat(r['recorded_at']).timestamp()
            assert lease['paused_at'] <= timestamp <= resume['time']
        resources.extend(samples)
        for folder in (run, owner):
            for path in sorted(folder.rglob('*')):
                if path.is_file():
                    collect(path)
        all_plans.append(plan)
        all_checks.append(checks)

    assert all_checks[0] == all_checks[1], 'All raw numeric checks and tokens must match between cache pools'
    for key in ('source_sha256', 'model_sha256', 'workload_sha256', 'target_revision', 'drafter_revision', 'baseline_commit', 'environment', 'mlx'):
        assert all_plans[0][key] == all_plans[1][key]
    assert len({r['swap_used_mb'] for r in resources}) == 1
    assert len({r['vm_counters']['Swapouts'] for r in resources}) == 1
    # The preregistered guard does not require constant global Swapins. Preserve
    # and report the 16 GiB postflight +3 rather than inventing a new gate here.
    counter_deltas = [{'cache_gib': pool, 'counter': name,
                       'first': resources[index * 5]['vm_counters'][name],
                       'last': resources[index * 5 + 4]['vm_counters'][name],
                       'delta': resources[index * 5 + 4]['vm_counters'][name] - resources[index * 5]['vm_counters'][name]}
                      for index, pool in enumerate((2, 16))
                      for name in ('Swapins', 'Swapouts', 'Pageouts', 'Compressions', 'Decompressions')]
    assert resources[4]['monotonic_s'] < resources[5]['monotonic_s'], 'Process windows must be sequential'
    for path in lease_dir.rglob('*'):
        if path.is_file():
            collect(path)
    collect(BASE / 'draft-window-v2-jobs.json')
    for path in sorted((BASE / 'draft-window-v2-audit').glob('*')):
        if path.is_file():
            collect(path)
    collect(Path(__file__))
    workload = BASE / 'fullqual-build2/workload.json'
    assert sha(workload.read_bytes()) == all_plans[0]['workload_sha256']
    collect(workload)
    for role, root in (('target', 'target_model'), ('drafter', 'drafter')):
        cfg = Path(all_plans[0][root]) / 'config.json'
        assert sha(cfg.read_bytes()) == all_plans[0]['model_sha256'][str(cfg)]
        collect(cfg, f'model-metadata/{role}-config.json')
    commit = all_plans[0]['baseline_commit']
    for source in ('bench/m5_ultra_256gb/guarded_v2.py', 'bench/m5_ultra_256gb/resources_v2.py',
                   'bench/m5_ultra_256gb/suspended_service.py', 'docs/optimization-guide/tutorials/014-bounded-draft-kv.md'):
        files[f'preregister/{source}'] = subprocess.check_output(['git', 'show', f'{commit}:{source}'], cwd=ROOT)
    OUT.mkdir()
    archive = OUT / 'evidence.tar.gz'
    with tarfile.open(archive, 'w:gz') as target:
        for name, data in sorted(files.items()):
            item = tarfile.TarInfo(name)
            item.size = len(data)
            target.addfile(item, io.BytesIO(data))
    manifest = {name: {'bytes': len(data), 'sha256': sha(data)} for name, data in sorted(files.items())}
    with tarfile.open(archive) as target:
        assert set(target.getnames()) == set(manifest)
        for name, expected in manifest.items():
            actual = target.extractfile(name).read()
            assert len(actual) == expected['bytes'] and sha(actual) == expected['sha256']
    write(OUT / 'manifest.json', {'files': manifest, 'archive_sha256': sha(archive.read_bytes())})
    write(OUT / 'summary.json', {'status': 'PASS', 'decision': 'Prototype passes predeclared strong-reference and compatibility gates; independent default-OFF core integration qualification next',
          'scope': 'Synthetic cache-state replay, not actual long-context generation; no target verification, rollback, HTTP or allocator-release claim',
          'main_metric': 'owned_lifecycle_ms', 'processes': 2, 'numeric_cases': 368,
          'formal_samples': 240, 'warmup_samples': 80, 'first_post_qualification_samples': 40,
          'pools': pools, 'rows': table})
    write(OUT / 'audit.json', {'status': 'PASS', 'archive_files': len(files), 'archive_members_verified': True,
          'frozen_sources_checked_per_process': 33, 'working_tree_core_used_as_measured_evidence': False,
          'model_weight_hashes': 'Recorded independently by both measured processes and equal; this CPU audit checks config/workload but does not rehash or archive large weights',
          'cross_pool_numeric_records_identical': True, 'expected_checks_reconstructed_independently': True,
          'all_timing_phases_recomputed': True, 'all_abba_and_warmup_orders_checked': True,
          'application_references': 'Measured script reports references dropped and seed restored; not a guarantee of allocator cache or driver release',
          'resources': len(resources), 'pressure': 1, 'thermal': 0, 'ac_power': True,
          'swap_used_mb': resources[0]['swap_used_mb'], 'swapouts': resources[0]['vm_counters']['Swapouts'],
          'vm_counter_deltas': counter_deltas,
          'paging_note': '16 GiB Swapins increases by 3 only in the last sampled interval ending at postflight; global counter cannot attribute cause or exact time. Swapouts, Pageouts and swap-used unchanged. First audit extra constant-Swapins assertion and its source are preserved.',
          'process_windows_nonoverlapping': True,
          'releases': releases, 'resume': resume})
    print(json.dumps({'status': 'PASS', 'archive_files': len(files), 'resources': len(resources), 'pools': pools}, indent=2))


if __name__ == '__main__':
    main()
