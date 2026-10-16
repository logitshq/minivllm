"""Independent structural audit around the timestamp/token recomputation."""
import argparse
import hashlib
import json
from pathlib import Path


def read(p):
    return json.loads(p.read_text())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    a = p.parse_args()
    b = a.base
    layer_checks = read(b / 'prefill-nax-screen-v1/layer-checks.json')
    assert len(layer_checks) == 448 and all(r['output_exact'] and r['cache_exact'] for r in layer_checks)
    for name in ('prefill-boundary-v1', 'prefill-integrated-boundaries-v1'):
        assert read(b / name / 'complete.json')['status'] == 'PASS'
        checks = read(b / name / 'checks.json')
        assert len(checks) == 22 and all(r['logits_exact'] and r['cache_exact'] for r in checks)
    lifecycle_steps = {}
    for name in ('prefill-lifecycle-v1', 'prefill-integrated-lifecycle-v1'):
        assert read(b / name / 'complete.json')['status'] == 'PASS'
        count = 0
        for case in ('C64', 'mixed'):
            d = read(b / name / (case + '.json'))
            assert d['exact'] and d['reference'] == d['candidate']
            assert d['reference'][-1]['batch_ids'] == [] and d['reference'][-1]['staging'] == {}
            count += len(d['reference'])
        assert count == 16
        lifecycle_steps[name] = count
    root = b / 'prefill-http-v2'
    plan = read(root / 'plan.json')
    assert read(root / 'complete.json')['status'] == 'PASS'
    pids, records, ids_by_c = [], [], {}
    for c, pair, variant in plan['configs']:
        d = root / f'C{c}-pair{pair}-{variant}'
        result = read(d / 'result.json')
        trace = read(d / 'engine-trace.json')
        release = read(d / 'release.json')
        pids.append(release['pid'])
        assert release['waited_reaped'] and release['absence']['returncode'] == 1
        assert not release['absence']['stdout']
        grouped = [r['shape'] for r in trace['prefill_calls'] if r['mode'] == 'prefill_batch']
        assert bool(grouped) == (c > 1 and variant == 'candidate')
        assert all(2 <= B <= 32 and T == 19 for B, T in grouped)
        streams = ids_by_c.setdefault(c, {}).setdefault(variant, {})
        for phase in ('warm', 'measure'):
            for i in range(3):
                group = read(d / f'C{c}-{phase}-{i}.json')
                for r in group['requests']:
                    stream = tuple(r['output_ids_when_available'])
                    streams[stream] = streams.get(stream, 0) + 1
        records.append({'C': c, 'pair': pair, 'variant': variant, 'grouped_prefill_shapes': grouped,
                        'single_prefill_calls': sum(r['mode'] == 'prefill' for r in trace['prefill_calls'])})
    assert len(set(pids)) == len(pids) == 24
    # Free-arrival HTTP can form different decode batches even within the
    # reference arm. Preserve every observed stream instead of claiming that
    # asynchronous runs are a fixed arithmetic schedule. The fixed-schedule
    # numerical gate above remains byte-exact and is not relaxed here.
    stream_report = {str(c): {v: [{'ids': list(ids), 'requests_including_warm': count}
                                 for ids, count in sorted(streams.items())]
                             for v, streams in variants.items()}
                     for c, variants in ids_by_c.items()}
    lease = read(b / 'prefill-service-lease-v1/lease.json')
    resume = read(b / 'prefill-service-lease-v1/resume.json')
    assert lease['closed'] and resume['identity_match']
    assert resume['state_after'] and 'T' not in resume['state_after']
    assert not (b / 'prefill-service-lease-v1/watchdog-resume.json').exists()
    for i in range(3):
        job = read(b / 'prefill-service-lease-v1' / f'job-{i}.json')
        assert job['returncode'] == 0 and job['waited']
    # The two variants share precisely the final core; the control disables only grouping.
    source = Path(plan['source']) / 'tutorial_code/miniengine'
    assert all(hashlib.sha256((source / f).read_bytes()).hexdigest() == sha for f, sha in plan['source_files'].items())
    output = {'status': 'PASS', 'layer_checks': len(layer_checks), 'boundary_cases_per_implementation': 22,
              'lifecycle_steps': lifecycle_steps, 'http_processes': len(pids),
              'fixed_schedule_numerical_equality': True,
              'http_cross_process_id_equality_claimed': False,
              'http_stream_counts_by_C_and_variant': {str(c): {v: len(s) for v, s in variants.items()}
                                                     for c, variants in ids_by_c.items()},
              'output_streams': stream_report,
              'service_restored': resume, 'dispatch_records': records,
              'scope': 'Numerical/dispatch/ownership audit; performance comes from independent timestamp recomputation, not this file'}
    a.out.write_text(json.dumps(output, indent=2) + '\n')
    print('PASS', len(pids), 'processes; service restored; numerical and dispatch checks')


if __name__ == '__main__':
    main()
