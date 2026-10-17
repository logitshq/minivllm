"""CPU-only independent audit and source-pinned archive for E012 integration."""
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import tarfile

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / '.artifacts/m5-ultra-256gb'
OUT = ROOT / 'docs/experiments/m5-ultra-256gb/E012-draft-selector-integrated-v1'


def read(path):
    return json.loads(path.read_text())


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def main():
    assert not (OUT / 'summary.json').exists(), 'Evidence is immutable; use a new version'
    files, rows, resources, releases = {}, [], [], []
    common_plan, first_checks = None, None

    def collect(path, name=None):
        files[name or str(path.relative_to(ROOT))] = path.read_bytes()

    for rep in (1, 2, 3):
        run = BASE / f'draft-selector-integrated-v1-r{rep}'
        owner = BASE / f'draft-selector-integrated-owner-v1-r{rep}'
        plan = read(run / 'plan.json')
        done = read(run / 'complete.json')
        selector = read(run / 'selector-checks.json')
        full = read(run / 'full-draft-checks.json')
        times = read(run / 'timing.json')
        assert done['status'] == 'PASS' and len(selector) == 33 and len(full) == 9
        assert len({r['case'] for r in selector}) == 33
        assert len({(r['prompt'], r['block']) for r in full}) == 9
        for c in selector:
            assert c['original_ids'] == c['instrumented_reference_ids'] == c['candidate_ids']
            assert len(c['scores_bit_equal']) == len(c['original_ids']) > 0
            assert all(c['scores_bit_equal'])
        for c in full:
            assert c['old_ids'] == c['new_ids'] and len(c['old_ids']) == c['block'] - 1
            assert len(c['draft_kv_bit_equal']) == 10 and all(c['draft_kv_bit_equal'])
        if first_checks is None:
            first_checks = (selector, full)
        else:
            assert first_checks == (selector, full), 'Fresh processes must preserve IDs/checks'
        assert len(times) == 9
        assert {(r['prompt'], r['block']) for r in times} == {(r['prompt'], r['block']) for r in full}
        for r in times:
            for label in ('selector', 'full_draft'):
                t = r[label]
                assert len(t['samples']) == 12
                assert [(s['block'], s['variant']) for s in t['samples']] == [
                    (b, v) for b in range(3) for v in ('reference', 'candidate', 'candidate', 'reference')]
                assert all(math.isfinite(s['ms']) and s['ms'] > 0 for s in t['samples'])
                med = {v: statistics.median(s['ms'] for s in t['samples'] if s['variant'] == v)
                       for v in ('reference', 'candidate')}
                assert med == t['median_ms']
                assert math.isclose(med['reference'] / med['candidate'], t['speedup'], rel_tol=1e-12)
            rows.append({'repetition': rep, 'prompt': r['prompt'], 'block': r['block'],
                         'selector_speedup': r['selector']['speedup'],
                         'full_draft_speedup': r['full_draft']['speedup'],
                         'full_draft_median_ms': r['full_draft']['median_ms']})
        gain = statistics.geometric_mean(r['full_draft']['speedup'] for r in times)
        assert math.isclose(gain, done['full_draft_geomean_speedup'], rel_tol=1e-12)
        assert done['advance'] and gain >= 1.05 and min(r['full_draft']['speedup'] for r in times) >= .97
        assert read(owner / 'complete.json')['status'] == 'PASS'
        release = read(owner / 'release.json')
        assert release['returncode'] == 0 and release['waited_reaped'] and release['strict_absence']
        releases.append(release)
        for path in sorted(owner.glob('resource-*.json')):
            r = read(path)
            assert r['pressure_level_value'] == 1 and r['processinfo_thermal_metadata']['code'] == 0
            assert r['no_active_foreign_native_processes']
            assert r['power_metadata']['source'] == 'AC Power'
            suspended = [p for p in r['native_inference_processes'] if p['pid'] == 74384]
            assert len(suspended) == 1 and suspended[0]['authorized_suspended'] and 'T' in suspended[0]['state']
            resources.append(r)
        for key in ('source_sha256', 'drafter_weights_sha256', 'drafter_config_sha256', 'workload_sha256'):
            if common_plan is not None:
                assert common_plan[key] == plan[key]
        common_plan = plan
        for source, digest in plan['source_sha256'].items():
            f = ROOT / source
            assert sha(f.read_bytes()) == digest
            collect(f)
        for folder in (run, owner):
            for f in folder.rglob('*'):
                if f.is_file():
                    collect(f)
        collect(run.with_suffix('.adapter.json'))
    assert len({r['swap_used_mb'] for r in resources}) == 1
    assert len({r['vm_counters']['Swapouts'] for r in resources}) == 1
    lease = BASE / 'draft-selector-integrated-service-lease-v1'
    resume = read(lease / 'resume.json')
    assert resume['identity_match'] and 'T' not in resume['state_after']
    for f in lease.rglob('*'):
        if f.is_file():
            collect(f)
    collect(BASE / 'draft-selector-integrated-jobs-v1.json')
    for f in (OUT / 'preregister.json', OUT / 'cpu-tests.txt', Path(__file__)):
        collect(f)
    for name in ('guarded_v2.py', 'resources_v2.py', 'suspended_service.py'):
        collect(ROOT / 'bench/m5_ultra_256gb' / name)
    workload = BASE / 'fullqual-build2/workload.json'
    assert sha(workload.read_bytes()) == common_plan['workload_sha256']
    collect(workload)
    cfg = Path(common_plan['drafter']) / 'config.json'
    assert sha(cfg.read_bytes()) == common_plan['drafter_config_sha256']
    collect(cfg, 'model-metadata/drafter-config.json')
    archive = OUT / 'evidence.tar.gz'
    with tarfile.open(archive, 'w:gz') as tf:
        for name, data in sorted(files.items()):
            item = tarfile.TarInfo(name)
            item.size = len(data)
            tf.addfile(item, io.BytesIO(data))
    manifest = {name: {'bytes': len(data), 'sha256': sha(data)} for name, data in sorted(files.items())}
    with tarfile.open(archive) as tf:
        assert set(tf.getnames()) == set(manifest)
        for name, metadata in manifest.items():
            assert sha(tf.extractfile(name).read()) == metadata['sha256']
    write(OUT / 'manifest.json', {'files': manifest, 'archive_sha256': sha(archive.read_bytes())})
    write(OUT / 'summary.json', {
        'status': 'PASS', 'decision': 'Qualified opt-in draft selector module; keep default OFF',
        'scope': 'M5 Ultra/MLX0.32.3/pinned Qwen3.8-27B and DFlash2; no target verify, rollback, generation or HTTP claim',
        'repetitions': 3, 'selector_checks': 99, 'full_draft_checks': 27,
        'geomean_by_process': [statistics.geometric_mean(r['full_draft_speedup'] for r in rows if r['repetition'] == rep) for rep in (1, 2, 3)],
        'all_27_cases_geomean': statistics.geometric_mean(r['full_draft_speedup'] for r in rows),
        'rows': rows,
        'raw_complete_note': 'The shared screen retains its generic next-integration decision text; this audit is the integration outcome.'})
    write(OUT / 'audit.json', {'status': 'PASS', 'resources': len(resources), 'pressure': 1, 'thermal': 0,
        'swap_used_mb': resources[0]['swap_used_mb'], 'swapouts': resources[0]['vm_counters']['Swapouts'],
        'all_source_hashes_checked': True, 'cross_process_ids_equal': True, 'releases': releases,
        'resume': resume, 'archive_files': len(files), 'archive_members_verified': True})
    print(json.dumps({'status': 'PASS', 'archive_files': len(files), 'resource_samples': len(resources),
        'geomean_by_process': read(OUT / 'summary.json')['geomean_by_process']}, indent=2))


if __name__ == '__main__':
    main()
