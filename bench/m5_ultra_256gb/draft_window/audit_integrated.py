"""CPU-only independent E014 public-switch matrix audit and immutable archive.

No MLX import and no GPU execution. Reconstructs numeric matrix independently;
uses frozen measured source rather than a later working tree.
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
OUT = ROOT / 'docs/experiments/m5-ultra-256gb/E014-draft-window-integrated-v1'
LENGTHS = (0, 31, 2040, 2047, 2048, 2049, 4095, 4096, 4097, 32768)
TIMING_LENGTHS = (32, 2048, 8192, 32768, 131072)
PHASES = ('prepare_ms', 'draft_ms', 'teardown_ms', 'owned_lifecycle_ms')


def read(path):
    return json.loads(path.read_text())


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')


def expected_matrix():
    base = {}
    def add(label, block, position, incoming):
        assert (label, block) not in base
        base[label, block] = (position, incoming)
    for block in (3, 5, 8):
        add('cold', block, 4, 4)
    for length in LENGTHS:
        for block in (3, 5, 8):
            position = length
            for step, incoming in enumerate((4, None, 0, 1, 7)):
                position += incoming or 0
                add(f'N{length}-step{step}', block, position, incoming)
            if length == 31:
                for incoming in (2049, 4097):
                    add(f'input{incoming}', block, length + incoming, incoming)
    for length in (4095, 4096, 4097):
        position = length
        for step, (block, incoming) in enumerate(((8, 1), (3, None), (5, 0), (8, 1), (3, 7))):
            position += incoming or 0
            add(f'variable-N{length}-step{step}', block, position, incoming)
    for length in TIMING_LENGTHS:
        for block in (3, 8):
            add(f'timing-N{length}', block, length + 4, 4)
    assert len(base) == 184
    expected = {(case, block, selector, bounded): expected
                for (case, block), expected in base.items()
                for selector in (False, True) for bounded in (False, True)}
    assert len(expected) == 736
    return expected


def validate_sample(s):
    assert all(type(s[k]) in (int, float) and math.isfinite(s[k]) and s[k] > 0 for k in PHASES)
    assert math.isclose(s['owned_lifecycle_ms'], s['draft_ms'] + s['teardown_ms'], rel_tol=1e-12)
    assert s['own_products_released_before_next_sample'] is True


def main():
    assert not OUT.exists(), 'Use a new archive version; never overwrite an existing evidence set'
    expected = expected_matrix()
    files, table, resources, releases, processes = {}, [], [], [], []
    plans, matrices, seeds, resource_groups = [], [], [], []
    jobs = read(BASE / 'draft-window-integrated-v1-jobs.json')
    lease_dir = BASE / 'draft-window-integrated-v1-service-lease'
    lease, resume = read(lease_dir / 'lease.json'), read(lease_dir / 'resume.json')
    assert lease['closed'] is True and lease['service_pid'] == 74384 and lease['jobs'] == jobs
    assert lease['service_identity'] == 'Fri Oct 16 08:01:58 2026     omlx-server'
    assert resume['identity_match'] is True and resume['state_before'] == 'T' and resume['state_after'] == 'S'

    def collect(path, name=None):
        key = name or str(path.relative_to(ROOT))
        data = path.read_bytes()
        if key in files:
            assert files[key] == data
        files[key] = data

    for run_id, pool in enumerate((16, 16, 16, 2), 1):
        run = BASE / f'draft-window-integrated-v1-run{run_id}'
        owner = BASE / f'draft-window-integrated-v1-run{run_id}-owner'
        plan, done = read(run / 'plan.json'), read(run / 'complete.json')
        checks, timings = read(run / 'checks.json'), read(run / 'timing.json')
        assert not list(run.glob('*failure*')) and not list(owner.glob('*failure*'))
        assert plan['experiment'] == 'E014-integrated' and plan['cache_gib'] == pool
        assert plan['predeclared_process_matrix'] == [16, 16, 16, 2]
        assert plan['numeric_variants'] == ['selectorOFF/kvOFF', 'selectorOFF/kvON', 'selectorON/kvOFF', 'selectorON/kvON']
        assert plan['model_hash_status'] == 'complete' and plan['device']['device_name'] == 'Apple M5 Ultra'
        assert plan['device']['memory_size'] == 256 * 1024**3
        assert done['status'] == 'PASS' and done['numeric_checks'] == len(checks) == 736
        assert done['timing_cases'] == len(timings) == 10 and done['cache_gib'] == pool
        assert len({(r['case'], r['block'], r['selector'], r['bounded']) for r in checks}) == len(checks)
        assert {(r['case'], r['block'], r['selector'], r['bounded']) for r in checks} == set(expected)
        observed_ids = {}
        for row in checks:
            position, incoming = expected[row['case'], row['block'], row['selector'], row['bounded']]
            assert row['position'] == position and row['new_rows'] == incoming
            assert type(row['selector']) is bool and type(row['bounded']) is bool
            assert row['oracle_ids'] == row['actual_ids'] and len(row['actual_ids']) == row['block'] - 1
            assert all(type(i) is int and 0 <= i < 248320 for i in row['actual_ids'])
            key = row['case'], row['block']
            if key in observed_ids:
                assert observed_ids[key] == row['actual_ids']
            observed_ids[key] = row['actual_ids']
            for field, count in (('hidden_logits_exact', 2), ('kv_suffix_exact', 10), ('visible_exact', 10),
                                 ('logical_exact', 5), ('physical_bound', 5)):
                assert len(row[field]) == count and all(value is True for value in row[field])
        frozen = read(run / 'frozen-source.json')
        assert frozen['status'] == 'PASS' and frozen['source_sha256'] == plan['source_sha256']
        assert len(plan['source_sha256']) == 36
        for source, digest in plan['source_sha256'].items():
            assert sha((run / 'frozen-source' / source).read_bytes()) == digest
        assert sha((run / 'frozen-reference/dflash2.py').read_bytes()) == plan['old_attention_sha256']
        prior = read(BASE / 'draft-window-v2-cache16/plan.json')
        assert prior['source_sha256']['tutorial_code/miniengine/spec/dflash2.py'] == plan['old_attention_sha256']
        collect(BASE / 'draft-window-v2-cache16/plan.json', 'prior-reference/plan.json')
        seed = read(run / 'seed.json')
        assert seed['four_loaded_drafters'] is True and seed['real_seed_rows'] == len(seed['prompt_ids']) - 4
        assert [(r['history_rows'], r['block']) for r in timings] == [(n, b) for n in TIMING_LENGTHS for b in (3, 8)]
        for row in timings:
            data, n, block = row['full_draft'], row['history_rows'], row['block']
            resident = n if n <= 4096 else 2048
            assert row['seed_logical_bytes'] == {'reference': n * 20480, 'candidate': resident * 20480}
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
            by_arm = {str(a): statistics.median(s['owned_lifecycle_ms'] for s in data['samples'] if s['arm'] == a) for a in range(4)}
            by_round = {str(i): {v: statistics.median(s['owned_lifecycle_ms'] for s in data['samples']
                                                  if s['block'] == i and s['variant'] == v)
                                for v in ('reference', 'candidate')} for i in range(3)}
            table.append({'run': run_id, 'cache_gib': pool, 'history_rows': n, 'block': block,
                          'median_ms': medians, 'speedup': gains,
                          'owned_lifecycle_ms_median_by_abba_arm': by_arm,
                          'owned_lifecycle_ms_median_by_abba_round': by_round})
        run_rows = [r for r in table if r['run'] == run_id]
        long = min(r['speedup']['owned_lifecycle_ms'] for r in run_rows if r['history_rows'] in (32768, 131072))
        short = min(r['speedup']['owned_lifecycle_ms'] for r in run_rows if r['history_rows'] in (32, 2048))
        gate = long >= 1.10 and short >= .97
        assert gate is done['gate']
        assert math.isclose(long, done['long_min'], rel_tol=1e-12) and math.isclose(short, done['short_min'], rel_tol=1e-12)
        failures = [{'history_rows': r['history_rows'], 'block': r['block'], 'speedup': r['speedup']['owned_lifecycle_ms']}
                    for r in run_rows if (r['history_rows'] in (32, 2048) and r['speedup']['owned_lifecycle_ms'] < .97)
                    or (r['history_rows'] in (32768, 131072) and r['speedup']['owned_lifecycle_ms'] < 1.10)]
        processes.append({'run': run_id, 'cache_gib': pool, 'gate': gate, 'long_min_speedup': long, 'short_min_speedup': short,
                          'failed_points': failures, 'numeric_cases': 736, 'formal_samples': 120,
                          'warmup_samples': 40, 'first_post_qualification_samples': 20})
        attempt, child, release = read(owner / 'attempt.json'), read(owner / 'child.json'), read(owner / 'release.json')
        owner_done, job_done = read(owner / 'complete.json'), read(lease_dir / f'job-{run_id - 1}.json')
        assert owner_done['status'] == 'PASS' and attempt['owner_pid'] == owner_done['owner_pid'] == job_done['pid']
        assert job_done['command'] == jobs[run_id - 1] and job_done['returncode'] == 0 and job_done['waited'] is True
        assert attempt['command'] == jobs[run_id - 1][jobs[run_id - 1].index('--cwd') + 2:]
        assert attempt['command'][attempt['command'].index('--cache-gib') + 1] == str(pool)
        assert plan['pid'] == child['pid'] == release['pid']
        assert release['returncode'] == 0 and release['waited_reaped'] is True and release['strict_absence'] is True
        current = subprocess.run(['/bin/ps', '-p', str(child['pid']), '-o', 'pid=,comm='], capture_output=True, text=True)
        assert current.returncode == 1 and not current.stdout and not current.stderr
        releases.append(dict(release, independently_absent_at_audit=True))
        samples = [read(p) for p in sorted(owner.glob('resource-*.json'))]
        assert len(samples) == 7 and [r['label'] for r in samples] == ['preflight'] + ['child-running'] * 5 + ['postflight']
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
        resource_groups.append(samples)
        for folder in (run, owner):
            for path in sorted(folder.rglob('*')):
                if path.is_file():
                    collect(path)
        plans.append(plan); matrices.append(checks); seeds.append(seed)

    assert all(x == matrices[0] for x in matrices)
    assert all(x == seeds[0] for x in seeds)
    for key in ('source_sha256', 'model_sha256', 'workload_sha256', 'target', 'drafter', 'commit', 'old_attention_sha256', 'device'):
        assert all(plan[key] == plans[0][key] for plan in plans)
    assert [p['gate'] for p in processes] == [True, False, True, True]
    adoption = all(p['gate'] for p in processes if p['cache_gib'] == 16)
    assert adoption is False
    assert len(processes[1]['failed_points']) == 1 and processes[1]['failed_points'][0]['history_rows'] == 32
    assert processes[1]['failed_points'][0]['block'] == 3
    assert len({r['swap_used_mb'] for r in resources}) == 1
    assert len({r['vm_counters']['Swapouts'] for r in resources}) == 1
    counter_deltas = [{'run': i + 1, 'cache_gib': processes[i]['cache_gib'], 'counter': name,
                       'first': samples[0]['vm_counters'][name], 'last': samples[-1]['vm_counters'][name],
                       'delta': samples[-1]['vm_counters'][name] - samples[0]['vm_counters'][name]}
                      for i, samples in enumerate(resource_groups)
                      for name in ('Swapins', 'Swapouts', 'Pageouts', 'Compressions', 'Decompressions')]
    assert all(resource_groups[i][-1]['monotonic_s'] < resource_groups[i+1][0]['monotonic_s'] for i in range(3))
    cpu = read(BASE / 'draft-window-integrated-cpu-v1.json')
    assert cpu['status'] == 'PASS_CORRECTED_TEST_SELECTION'
    assert [a['returncode'] for a in cpu['attempts']] == [5, 0, 0]
    assert 'Ran 6 tests' in cpu['attempts'][1]['stderr'] and 'Ran 2 tests' in cpu['attempts'][2]['stderr']
    collect(BASE / 'draft-window-integrated-cpu-v1.json')
    for path in lease_dir.rglob('*'):
        if path.is_file():
            collect(path)
    collect(BASE / 'draft-window-integrated-v1-jobs.json')
    collect(Path(__file__))
    workload = BASE / 'fullqual-build2/workload.json'
    assert sha(workload.read_bytes()) == plans[0]['workload_sha256']
    collect(workload)
    for role in ('target', 'drafter'):
        cfg = Path(plans[0][role]) / 'config.json'
        assert sha(cfg.read_bytes()) == plans[0]['model_sha256'][str(cfg)]
        collect(cfg, f'model-metadata/{role}-config.json')
    commit = plans[0]['commit']
    for source in ('bench/m5_ultra_256gb/guarded_v2.py', 'bench/m5_ultra_256gb/resources_v2.py',
                   'bench/m5_ultra_256gb/suspended_service.py', 'bench/m5_ultra_256gb/draft_selector/test_switch.py',
                   'docs/optimization-guide/tutorials/014-bounded-draft-kv.md'):
        files[f'preregister/{source}'] = subprocess.check_output(['git', 'show', f'{commit}:{source}'], cwd=ROOT)
    OUT.mkdir()
    archive = OUT / 'evidence.tar.gz'
    with tarfile.open(archive, 'w:gz') as target:
        for name, data in sorted(files.items()):
            item = tarfile.TarInfo(name);item.size = len(data)
            target.addfile(item, io.BytesIO(data))
    manifest = {name: {'bytes': len(data), 'sha256': sha(data)} for name, data in sorted(files.items())}
    with tarfile.open(archive) as target:
        assert set(target.getnames()) == set(manifest)
        for name, meta in manifest.items():
            actual = target.extractfile(name).read()
            assert len(actual) == meta['bytes'] and sha(actual) == meta['sha256']
    write(OUT / 'manifest.json', {'files': manifest, 'archive_sha256': sha(archive.read_bytes())})
    write(OUT / 'summary.json', {'status': 'NUMERICS_PASS_ADOPTION_GATE_FAIL', 'adoption': adoption,
          'decision': 'Keep default-OFF experimental module; one of three predeclared 16 GiB processes fails the short-history gate',
          'scope': 'Public loader four combinations; timed selector ON, KV OFF vs ON; synthetic cache-state replay, not long generation, target verify, rollback or HTTP',
          'main_metric': 'owned_lifecycle_ms', 'process_matrix_cache_gib': [16, 16, 16, 2],
          'numeric_cases': 2944, 'timing_cases': 40, 'formal_samples': 480, 'warmup_samples': 160,
          'first_post_qualification_samples': 80, 'processes': processes, 'rows': table})
    write(OUT / 'audit.json', {'status': 'PASS_AUDIT_ADOPTION_REJECTED', 'archive_files': len(files),
          'archive_members_verified': True, 'archive_bytes': archive.stat().st_size,
          'frozen_sources_checked_per_process': 36, 'prior_actual_attention_sha256': plans[0]['old_attention_sha256'],
          'working_tree_core_used_as_measured_evidence': False,
          'model_weight_hashes': 'Recorded independently by all four processes and equal. CPU audit rechecks config/workload; large weights not rehashed or archived',
          'all_numeric_records_and_seeds_identical_across_processes': True,
          'expected_736_case_matrix_reconstructed_independently': True,
          'all_timing_phases_and_abba_orders_recomputed': True,
          'application_references': 'Measured script checks own products dropped and seeds restored; not a proof of allocator cache or driver release',
          'resources': len(resources), 'pressure': 1, 'thermal': 0, 'ac_power': True,
          'swap_used_mb': resources[0]['swap_used_mb'], 'vm_counter_deltas': counter_deltas,
          'resource_scope': 'Known executables/exact owner sampled. Does not exclude every possible GUI or unidentified GPU activity; no causal attribution for timing drift',
          'process_windows_nonoverlapping': True, 'releases': releases, 'resume': resume,
          'cpu_checks': 'Initial zero-test wrong pattern preserved as failure; corrected 6 bounded-KV + 2 selector tests pass'})
    print(json.dumps({'status': 'PASS_AUDIT_ADOPTION_REJECTED', 'archive_files': len(files),
                      'archive_bytes': archive.stat().st_size, 'resources': len(resources), 'processes': processes}, indent=2))


if __name__ == '__main__':
    main()
