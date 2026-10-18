"""Prepare v15 only after the registered GDN formal cycle is independently closed.

Stdlib only. Default describes; --run writes a NEW directory. No raw wire replay,
archive decompression, model/runtime import, service call, or performance run.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

OLD = 'docs/optimization-guide/progress/2026-10-17-v14/source-v13-chartdata.json'
OLD_SHA = 'ff7d06adbe1ecda482173a6747a7e673aa71a73a54713bfd2d8f2f6b52551ef0'
ARCHIVE = 'docs/experiments/m5-ultra-256gb/E029-gdn-http-formal-v1'
PLAN_SHA = '02337f0267a327ad7b025fd90ae18dec124cfe139811a767e9597c2b27af52f5'
AUDITOR_SHA = '70705b8ad9e2a1948306d801662302a4cdfb38a7a30370d5961c27fbbeb3f976'
RESTORE_SHA = '8ece5748f8328a110651981f40d3c7ba158194a9479052c544083d45a5621fc0'
PHASES = ('r1-c1-off', 'r1-c1-on', 'r1-c8-on', 'r1-c8-off', 'r2-c8-off', 'r2-c8-on',
          'r2-c1-on', 'r2-c1-off', 'r3-c1-off', 'r3-c1-on', 'r3-c8-on', 'r3-c8-off')
Y = 'aggregate_output_tok_s'
XS = ('e2e_p50_s', 'e2e_p95_s')
LABELS = {'native-historical': 'Native historical (N=1)', 'native-off': 'Native E024 cycle OFF',
          'native-on': 'Native E024 ON', 'native-gdn-off': 'Native GDN cycle OFF',
          'native-gdn-on': 'Native GDN ON', 'omlx': 'oMLX', 'lithos': 'Lithos'}
ENGINE = {c: ('native' if c.startswith('native-') else c) for c in LABELS}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read(path):
    path = Path(path)
    need(path.is_file() and not path.is_symlink() and path.stat().st_size < 32 * (1 << 20),
         'small regular metadata required: ' + str(path))
    def unique(items):
        out = {}
        for key, value in items:
            need(key not in out, 'duplicate JSON key'); out[key] = value
        return out
    raw = path.read_bytes()
    return json.loads(raw, object_pairs_hook=unique,
                      parse_constant=lambda x: need(False, 'nonfinite JSON ' + x)), raw


def positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def checked_points(audit):
    need(audit['schema'] == 'GDN-prework-http-formal-independent-audit-v1'
         and audit['status'] == 'PASS' and audit['qualified'] is True
         and audit['performance_qualified'] is True, 'complete independent formal PASS required')
    need(audit['audit_source_sha256'] == AUDITOR_SHA and audit['owner_plan_sha256'] == PLAN_SHA,
         'registered audit and plan identity')
    need(audit['requests'] == dict(total=864, warm=96, formal=768)
         and audit['http_raw_opened'] is True and audit['actual_prework_calls_observed'] is False
         and bool(audit['resources']), 'full request/resource scope; observers remain OFF')
    points = audit['points']
    need([p['phase'] for p in points] == list(PHASES), 'all twelve actual process points in registered order')
    for p, phase in zip(points, PHASES):
        repeat, concurrency, arm = phase.split('-')
        need(p['arm'] == arm and type(p['repeat']) is int and p['repeat'] == int(repeat[1:])
             and type(p['concurrency']) is int and p['concurrency'] == int(concurrency[1:]), 'phase identity')
        for key, expected in (('requests', 64), ('committed_output_tokens', 32768),
                              ('n_fresh_processes', 1), ('warm_requests_checked', 8),
                              ('max_http_outstanding', p['concurrency'])):
            need(type(p[key]) is int and p[key] == expected, 'point count: ' + key)
        need(all(positive(p[k]) for k in (Y, *XS, 'common_window_s'))
             and p[XS[0]] <= p[XS[1]], 'finite positive throughput/latency')
        need(math.isclose(p[Y], 32768 / p['common_window_s'], rel_tol=1e-12, abs_tol=1e-12),
             'committed-token common-window throughput')
        need(p['actual_prework_calls_observed'] is False, 'no formal call observer claim')
    return points


def closed_inputs(repo):
    """Consume small CLOSED sidecars, cross-binding their original archive entries.

    The existing archive verifier established full member integrity. This plotting
    step checks its receipt/manifest and selected small originals, not the stream.
    """
    repo = Path(repo).resolve(); archive = repo / ARCHIVE
    names = ('archive-verification.json', 'manifest.json', 'summary.json',
             'independent-formal.json', 'restoration-audit.json', 'restored-journal.json',
             'audit-execution-summary.json')
    values, raws = {}, {}
    for name in names:
        values[name], raws[name] = read(archive / name)
    v, m, summary = (values[n] for n in names[:3])
    audit, restore, journal, execution = (values[n] for n in names[3:])
    stream, raws['archive-stream.json'] = read(archive / 'archive-stream.json')
    need(v['schema'] == 'GDN-prework-formal-archive-verification-v1' and v['status'] == 'PASS'
         and all(v[k] is True for k in ('all_safe_extracted_members_verified',
                 'all_original_files_rehashed', 'all_tree_members_rechecked'))
         and v['packaging_grants_qualification'] is False, 'verified closed archive required')
    need(stream['archive'] == v['archive'] and stream['parts'] == v['parts'], 'verified archive stream identity')
    need(m['schema'] == 'GDN-prework-formal-archive-manifest-v1'
         and len(m['files']) == v['members']
         and sum(p['bytes'] for p in m['files'].values()) == v['raw_bytes'], 'archive manifest dimensions')
    need(m['files']['preregistration/owner-plan.json']['sha256'] == PLAN_SHA, 'registered plan archive member')
    def member(name, raw):
        record = m['files'][name]
        need(record['sha256'] == sha(raw) and record['bytes'] == len(raw), 'archive member binding: ' + name)
    for name in names[3:]:
        member('archive-sidecars/' + name, raws[name])
    member('archive-metadata/summary.json', raws['summary.json'])
    auditor = repo / 'bench/m5_ultra_256gb/gdn_prework_formal/audit_formal.py'
    need(m['source_pins'][str(auditor)] == AUDITOR_SHA
         and sha(auditor.read_bytes()) == AUDITOR_SHA, 'frozen independent auditor source')
    need(summary['schema'] == 'GDN-prework-formal-archive-summary-v1'
         and summary['owner_plan_sha256'] == PLAN_SHA
         and summary['owner_status'] == summary['audit_status'] == 'PASS'
         and summary['performance_qualified'] is summary['restoration_qualified'] is True
         and summary['planned_trials'] == summary['actual_closed_trial_names'] == list(PHASES),
         'all twelve owner trials and independent audit must pass')
    need(execution['schema'] == 'GDN-prework-formal-closed-audits-v1' and execution['status'] == 'PASS'
         and execution['owner_plan_sha256'] == PLAN_SHA and execution.get('ended_at')
         and all(execution[k] is True for k in ('execution_pass', 'performance_qualified',
                 'resources_verified_with_formal', 'restoration_qualified', 'sources_unchanged')),
         'closed audit execution, no FAIL/INCOMPLETE promotion')
    need(journal['schema'] == 'GDN-prework-http-formal-cold-journal-v1' and journal['state'] == 'RESTORED'
         and journal['owner_plan']['sha256'] == PLAN_SHA and journal['trials'] == list(PHASES), 'restored exact cycle')
    need(restore['schema'] == 'cold-service-restoration-audit-v1' and restore['status'] == 'PASS'
         and restore['qualified'] is True and restore['audit_source_sha256'] == RESTORE_SHA
         and restore['owner_plan_sha256'] == PLAN_SHA
         and restore['journal_sha256'] == sha(raws['restored-journal.json']) == audit['journal_sha256']
         and restore['service_restore']['replacement'] == journal['replacement']
         and audit['restoration'] == restore, 'independent restoration bindings')
    for label, filename, digest in (('formal', 'independent-formal.json', AUDITOR_SHA),
                                    ('restoration', 'restoration-audit.json', RESTORE_SHA)):
        job = execution['jobs'][label]
        need(job['status'] == 'RETURNED' and type(job['returncode']) is int and job['returncode'] == 0
             and job['audit_status'] == 'PASS' and job.get('ended_at') and job['source']['sha256'] == digest
             and job['output']['sha256'] == sha(raws[filename])
             and job['output']['bytes'] == len(raws[filename]), 'successful actual audit invocation')
    final_path = repo / '.artifacts/m5-ultra-256gb/gdn-http-formal-v1-execution/final.json'
    final, final_raw = read(final_path); member('execution/final.json', final_raw)
    need(final['journal_state'] == 'RESTORED'
         and all(type(final[k]) is int and final[k] == 0 for k in
                 ('stop_returncode', 'owner_returncode', 'recovery_returncode')), 'actual driver closed with recovery zero')
    need(execution['inputs']['driver-final.json']['sha256'] == sha(final_raw), 'audit driver input binding')
    member('owner/complete.json', (repo / '.artifacts/m5-ultra-256gb/gdn-http-formal-v1-owner/complete.json').read_bytes())
    need(m['files']['owner/complete.json']['sha256'] == audit['owner_complete_sha256'], 'owner terminal binding')
    need(bool(audit['raw_file_pins']), 'independent raw inventory')
    for name, pin in audit['raw_file_pins'].items():
        archived = m['files']['owner/' + name]
        need(pin == {k: archived[k] for k in ('sha256', 'bytes')}, 'audited raw archive membership')
    checked_points(audit)
    raws['driver-final.json'] = final_raw
    return audit, raws


def summaries(points, display):
    need([d['point_index'] for d in display] == list(range(len(points))), 'every observation once in original order')
    rows = []
    for configuration in LABELS:
        for concurrency in (1, 8):
            selected = [points[d['point_index']] for d in display
                        if d['configuration'] == configuration and points[d['point_index']]['concurrency'] == concurrency]
            need(len(selected) == (1 if configuration == 'native-historical' else 3), 'separate process group size')
            metrics = {}
            for key in (Y, *XS):
                values = [p[key] for p in selected]
                need(all(positive(v) for v in values), 'finite process observations')
                metrics[key] = dict(median=statistics.median(values), minimum=min(values), maximum=max(values))
            rows.append(dict(configuration=configuration, engine=ENGINE[configuration], label=LABELS[configuration],
                             concurrency=concurrency, fresh_processes=len(selected), metrics=metrics))
    return rows


def frontier(rows, x, maximize_x=False):
    def xv(r):
        value = r['metrics'][x]['median']
        return 512 / value if maximize_x else value
    def dominates(q, r):
        a, b = xv(q), xv(r); qy, ry = q['metrics'][Y]['median'], r['metrics'][Y]['median']
        return (a >= b if maximize_x else a <= b) and qy >= ry and (a != b or qy > ry)
    return sorted((r for r in rows if not any(dominates(q, r) for q in rows)), key=xv)


def build(old, audit):
    new = checked_points(audit)
    need(len(old['points']) == len(old['display']) == 26, 'immutable 26 observations')
    points = old['points'] + new
    display = old['display'] + [dict(point_index=26 + i, configuration='native-gdn-' + p['arm']) for i, p in enumerate(new)]
    rows = summaries(points, display)
    for previous in old['process_metric_medians']:
        row = next(r for r in rows if (r['configuration'], r['concurrency']) == (previous['configuration'], previous['concurrency']))
        need(row['metrics'] == previous['metrics'] and row['fresh_processes'] == previous['fresh_processes'], 'original group unchanged')
    fronts = {x: {engine: frontier([r for r in rows if r['engine'] == engine], x)
                   for engine in ('native', 'omlx', 'lithos')} for x in XS}
    joint = {x: frontier(rows, x) for x in XS}
    need(set((r['configuration'], r['concurrency']) for r in joint[XS[0]]) ==
         set((r['configuration'], r['concurrency']) for r in frontier(rows, XS[0], True)), 'reciprocal dominance invariance')
    return dict(schema='D512-explicit-frontier-v15',
        hardware=old['hardware'], model=old['model'], revision=old['revision'], workload=old['workload'],
        historical_sources=old['sources'], points=points, display=display, performance_point_count=38,
        previous_points_exactly_unchanged=True, previous_points_sha256=OLD_SHA,
        process_metric_medians=rows, per_engine_frontiers=fronts, joint_frontiers=joint,
        historical_paired_comparisons=old.get('paired_comparisons'), gdn_paired_comparisons=audit['comparisons'],
        historical_qualification_status=old.get('qualification_status'),
        gdn_performance_prerequisite=dict(audit_source_sha256=AUDITOR_SHA, owner_plan_sha256=PLAN_SHA,
            actual_prework_calls_observed=False),
        scope=['38 process observations,14 separate configuration/C groups; historical Native has N=1, all other groups N=3.',
            'New GDN OFF and earlier E024 OFF are separate cycles, never pooled. Pairing is within each registered cycle only.',
            'Native E024 ON and Native GDN ON are distinct configurations; no E024+GDN or E028 combination is measured.',
            'Each engine frontier may select any separately measured configuration. Joint segments do not describe an executable mixture.',
            'Only HTTP C1/C8 measured; C2–7 are unknown. HTTP concurrency is not GPU batch size.',
            'Medians/min–max are across process metrics, not pooled request percentiles or confidence intervals.',
            '512/median_process(p50_POST-to-DONE) is full-request normalized interactivity, not ITL or TPOT.',
            'Strict empirical nondominance is derived from all summaries; no significance threshold or predetermined winner.',
            'Formal observers OFF: prior actual histories qualified; no proof of unseen schedules, formal GDN hit counts, or general quality.',
            'Old per-point nondominance flags retain their original historical scope; use the new frontier fields.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true'); parser.add_argument('--repo', type=Path)
    parser.add_argument('--out', type=Path); args = parser.parse_args()
    if not args.run:
        print('DESCRIPTION_ONLY: no results read or files written; closed independent PASS + verified archive required.'); return
    need(args.repo is not None and args.out is not None, '--run requires --repo and fresh --out')
    need(not args.out.exists(), 'exclusive output; failed/old results are never overwritten')
    old, raw = read(args.repo / OLD); need(sha(raw) == OLD_SHA, 'frozen v14 source bytes')
    audit, evidence = closed_inputs(args.repo); result = build(old, audit)
    args.out.mkdir(parents=True); evidence['source-v14-data.json'] = raw
    pins = {}
    for name, blob in evidence.items():
        target = args.out / 'inputs' / name; target.parent.mkdir(exist_ok=True); target.write_bytes(blob)
        pins['inputs/' + name] = dict(sha256=sha(blob), bytes=len(blob))
    result['input_pins'] = pins
    (args.out / 'chartdata.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    receipt = dict(status='PREPARED', schema='D512-v15-preparation-v1', source_sha256=sha(Path(__file__).read_bytes()),
                   chartdata_sha256=sha((args.out / 'chartdata.json').read_bytes()), input_pins=pins,
                   archive_stream_rehashed=False, raw_http_reaudited=False, figures_created=False)
    (args.out / 'preparation.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
