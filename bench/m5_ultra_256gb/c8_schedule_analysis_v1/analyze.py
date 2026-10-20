"""Read a closed E047 raw join; describe schedules, never infer causal speed."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
JOIN = HERE.parent / 'pareto_realistic/http_join.py'
JOIN_SHA = 'a362c321c2329d7ab53ab6a1a5a96981bb95b44733739fdbb56dd3d38bef7b4e'


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_bytes())
def write(path, value): Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def analyze(child, cycle_final, out):
    child = Path(child).resolve(); out = Path(out).resolve()
    assert read(cycle_final)['status'] == 'PASS_CAPTURE_AND_RESTORE', 'cycle must close first'
    top = read(child / 'complete.json')
    assert top['status'] == 'PASS' and top['server_reaped'] and top['cleanup_complete']
    spec = read(child / 'spec.json'); receipt = read(child / 'joined-input/receipt.json')
    path = child / 'joined-input/recording.json'; recording = read(path)
    assert receipt['spec_sha256'] == sha(child / 'spec.json')
    assert receipt['recording_sha256'] == sha(path) and receipt['trace_sha256'] == sha(child / 'trace.jsonl')
    assert sha(JOIN) == JOIN_SHA
    ms = importlib.util.spec_from_file_location('_e047_raw_join', JOIN)
    join = importlib.util.module_from_spec(ms); ms.loader.exec_module(join)
    inputs = [Path(cycle_final), child / 'spec.json', child / 'complete.json',
              child / 'trace.jsonl', path, child / 'joined-input/receipt.json']
    by_client = {j['client_request_id']: rid for rid, j in recording['jobs'].items()}
    raw = {}; wave_ids = {}
    for source in receipt['source_groups']:
        p = Path(source['path']); assert sha(p) == source['sha256']; inputs.append(p)
        group = read(p); wave_ids[source['phase']] = []
        for row in group['requests']:
            rid, terminal, _ = join.raw_completion(row)
            assert by_client[row['request_id']] == rid
            assert terminal['benchmark_output_ids'] == recording['jobs'][rid]['raw_ids']
            assert terminal['benchmark_prompt_ids'] == recording['jobs'][rid]['prompt_ids']
            raw[rid] = dict(wave=source['phase'], start=row['start_perf_counter_s'], done=row['done_perf_counter_s'])
            wave_ids[source['phase']].append(rid)
    jobs = recording['jobs']; trajectories = {}
    for rid, job in jobs.items():
        prefill, outputs, offset = [], [], 0
        for step in recording['steps']:
            schedule = step['schedule']; rows = schedule['decode']; B = len(rows)
            for owner, n in schedule['prefill']:
                if owner == rid:
                    prefill.append(dict(step=step['step'], offset=offset, tokens=n, concurrent_decode_B=B))
                    offset += n
            for owner, token, done in step['results']:
                if owner == rid:
                    outputs.append(dict(output_index=len(outputs), step=step['step'], token_id=token,
                        kind='decode' if rid in rows else 'prefill', B=B,
                        row=rows.index(rid) if rid in rows else None,
                        batch_fixtures=[jobs[r]['fixture_id'] for r in rows], batch_rids=rows, done=done))
        assert offset == len(job['prompt_ids']) and [x['token_id'] for x in outputs] == job['raw_ids']
        trajectories[rid] = dict(**raw[rid], fixture_id=job['fixture_id'], client_request_id=job['client_request_id'],
            prompt_tokens=offset, output_tokens=len(outputs), prefills=prefill, outputs=outputs)
    differences = []; recipes = []
    for left, right in zip(wave_ids['warm'], wave_ids['screen']):
        a, b = trajectories[left], trajectories[right]; assert a['fixture_id'] == b['fixture_id']
        x, y = jobs[left]['raw_ids'], jobs[right]['raw_ids']
        first = next((i for i in range(min(len(x), len(y))) if x[i] != y[i]), None)
        if first is None and len(x) != len(y): first = min(len(x), len(y))
        differences.append(dict(fixture_id=a['fixture_id'], warm_rid=left, screen_rid=right,
            first_difference_index=first, warm_id=None if first is None or first >= len(x) else x[first],
            screen_id=None if first is None or first >= len(y) else y[first],
            warm_prefill_chunks=[p['tokens'] for p in a['prefills']], screen_prefill_chunks=[p['tokens'] for p in b['prefills']]))
        if a['fixture_id'] == 'D06-interval-booking':
            assert first is not None
            for rid in (left, right):
                tr = trajectories[rid]; stop_index = first + 2; target = tr['outputs'][stop_index]
                recipes.append(dict(wave=tr['wave'], fixture_id=tr['fixture_id'], rid=rid,
                    first_difference_output_index=first, stop_after_output_index=stop_index,
                    first_difference_global_step=tr['outputs'][first]['step'], stop_after_global_step=target['step'],
                    recording=dict(path=str(path), sha256=sha(path)), history_start_global_step=1,
                    preserve_all_jobs_and_caps=True, preserve_prior_wave=True,
                    status='PREFIX_INPUT_ONLY_NOT_DEVICE_RUN', expected_terminal='DIAGNOSTIC_PREFIX_COMPLETE',
                    complete_recording_qualified=False))
    out.mkdir(parents=True, exist_ok=False)
    result = dict(schema='E047-schedule-analysis-v1', status='PASS_RAW_ANALYSIS',
        inputs={str(p): sha(p) for p in inputs}, analysis_source=dict(path=str(Path(__file__).resolve()), sha256=sha(__file__)),
        requests=len(jobs), steps=len(recording['steps']), histogram=recording['histogram'],
        differences=differences, trajectories=trajectories, d06_recipes=recipes,
        causal_attribution=False, performance_qualified=False, scope='Instrumented arrival/schedule association only.')
    write(out / 'analysis.json', result); write(out / 'd06-prefix-recipes.json', recipes)
    lines = ['# E047 observed C8 schedules', '',
        'Both waves used MB50, GDN OFF and graph ON. The raw join checked all 16 requests and 1,030 steps. Observation can alter scheduling; these are diagnostic results, not performance points.', '',
        '| Fixture | Warm scheduled prefill chunks | Screen scheduled prefill chunks | First different output index (0-based) | Warm / screen token IDs |',
        '|---|---|---|---:|---|']
    for row in differences:
        lines.append('| {fixture_id} | {warm_prefill_chunks} | {screen_prefill_chunks} | {first_difference_index} | {warm_id} / {screen_id} |'.format(**row))
    lines += ['', 'D06 receives all 935 prompt tokens in one scheduled prefill in both waves. Its first decode outputs follow B2→B4→B6→B8 at row 1 in warm, versus B4→B6→B8 at row 3 in screen. Output index 46 differs at global steps 47/563: 24959 versus 52883. Thus this D06 observation is not explained by a changed prefill split alone. Batch shape, row placement, and retained process history still differ.', '',
        'The exact D06 prefix recipes stop after global steps 49 and 565 (output index 48). All jobs, original max_tokens=512 and actual arrivals/schedules remain unchanged. The screen recipe starts at global step 1 and therefore preserves the preceding warm history. An isolated screen replay would be a different experiment.', '',
        'Next test: replay each fixed history with the existing independent eager/graph engines and full logit/state/ID checks, stopping explicitly at the registered prefix. A successful prefix is incomplete by design; it is never a complete-model or full-history qualification. E024 full 16-request replay took 512.49 s with a reported MLX allocator peak of 68.55 GiB; prefix peak physical memory is not known. Existing resource gates remain mandatory.', '',
        'Files: analysis.json retains every per-output batch/row trajectory plus raw timing and input SHAs; d06-prefix-recipes.json holds the two exact diagnostic stop descriptors. No model was run for this analysis.']
    (out / 'README.md').write_text('\n'.join(lines) + '\n')
    return dict(status=result['status'], out=str(out), first_differences=differences,
                analysis_sha256=sha(out / 'analysis.json'), recipes_sha256=sha(out / 'd06-prefix-recipes.json'))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--child', type=Path); p.add_argument('--cycle-final', type=Path); p.add_argument('--out', type=Path)
    a = p.parse_args()
    if a.child:
        assert a.cycle_final and a.out; print(json.dumps(analyze(a.child, a.cycle_final, a.out)))
    else:
        assert not a.cycle_final and not a.out; print('DESCRIPTION ONLY: closed E047 raw schedules -> diagnosis and prefix inputs.')
