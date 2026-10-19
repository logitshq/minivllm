"""Prepare v16 from CLOSED E033 evidence; default is inert.

Only --run reads small receipts and writes a new output directory. No model,
service, HTTP, raw-request audit, archive decompression or plotting is performed.
The 38 v15 observations remain unchanged JSON values; the input file is copied intact.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import statistics

OLD = 'docs/optimization-guide/progress/2026-10-17-v15/chartdata.json'
OLD_SHA = '7f44aa9223732127745684b5dc74646a7cb6e65720ef5615d78e4ac5264e61c7'
ARCHIVE = 'docs/experiments/m5-ultra-256gb/E033-gdn-graph-combined-formal-v2'
ARTIFACT_PREFIX = '.artifacts/m5-ultra-256gb/gdn-graph-combined-formal-v2'
SOURCE_DIR = 'bench/m5_ultra_256gb/gdn_graph_combined_formal_retry_v2'
PLAN_SHA = 'f4e19bd0a842b48ea26fb2d27f870fdf62409897c7f862067c7800de5949b6f1'
AUDITOR_SHA = '2391e9d4eab39df75064ae6ce07f6a0328e04ff21cbc977342d3f420d8e20b34'
RESTORE_SHA = '7ab477f0d449bf08f78259a820d067f902d2032314ff4ebd389157b2578ec57d'
PHASES = ('r1-c1-off', 'r1-c1-on', 'r1-c8-on', 'r1-c8-off', 'r2-c8-off', 'r2-c8-on',
          'r2-c1-on', 'r2-c1-off', 'r3-c1-off', 'r3-c1-on', 'r3-c8-on', 'r3-c8-off')
Y = 'aggregate_output_tok_s'
XS = ('e2e_p50_s', 'e2e_p95_s')
# Existing labels/order remain unchanged so every historical summary is identical.
LABELS = {'native-historical': 'Native historical (N=1)', 'native-off': 'Native E024 cycle OFF',
          'native-on': 'Native E024 ON', 'native-gdn-off': 'Native GDN cycle OFF',
          'native-gdn-on': 'Native GDN ON', 'omlx': 'oMLX', 'lithos': 'Lithos',
          'native-combined-off': 'minivllm combined cycle: graph ON / GDN OFF',
          'native-combined-on': 'minivllm combined cycle: graph ON / GDN ON'}
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
    need(audit['schema'] == 'GDN-graph-combined-http-formal-independent-audit-v2'
         and audit['status'] == 'PASS' and audit['qualified'] is True
         and audit['performance_qualified'] is True, 'complete independent formal PASS required')
    need(audit['audit_source_sha256'] == AUDITOR_SHA and audit['owner_plan_sha256'] == PLAN_SHA,
         'registered audit and plan identity')
    need(audit['requests'] == dict(total=864, warm=96, formal=768)
         and all(type(v) is int for v in audit['requests'].values())
         and audit['http_raw_opened'] is True and audit['route_always_on'] is True
         and audit['actual_prework_calls_observed'] is False
         and audit['actual_compiled_invocations_observed'] is False and bool(audit['resources']),
         'complete request/resource scope; graph ON, formal observers OFF')
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
        need(p['route_always_on'] is True and p['actual_prework_calls_observed'] is False
             and p['actual_compiled_invocations_observed'] is False, 'no formal dispatch observation claim')
    return points


def closed_inputs(repo):
    """Read the actual E033 packer's flat manifest/verification, not an E029 schema.

    The archive verifier established member integrity. This stage checks its
    receipt, manifest and small originals; it does not rehash the archive stream.
    """
    repo = Path(repo).resolve(); archive = repo / ARCHIVE
    names = ('archive-verification.json', 'manifest.json', 'summary.json',
             'independent-formal.json', 'restoration-audit.json', 'restored-journal.json',
             'execution-summary.json')
    values, raws = {}, {}
    for name in names:
        values[name], raws[name] = read(archive / name)
    v, m, summary = (values[n] for n in names[:3])
    audit, restore, journal, execution = (values[n] for n in names[3:])
    need(v['status'] == 'PASS' and type(v['members']) is int and v['members'] == len(m) + 1,
         'closed verified archive including embedded manifest required')
    need(type(v['raw_bytes']) is int and v['raw_bytes'] == sum(p['bytes'] for p in m.values()),
         'archive manifest dimensions')
    for name, pin in m.items():
        part = PurePosixPath(name)
        need(not part.is_absolute() and '..' not in part.parts and str(part) == name
             and type(pin['bytes']) is int and pin['bytes'] >= 0
             and isinstance(pin['sha256'], str) and len(pin['sha256']) == 64, 'safe exact archive inventory')
    stream = archive / 'evidence.tar.xz'
    need(stream.is_file() and not stream.is_symlink() and type(v['archive']['bytes']) is int
         and stream.stat().st_size == v['archive']['bytes'] > 0
         and len(v['archive']['sha256']) == 64, 'verified archive file retained (stat only here)')
    def member(name, raw):
        pin = m[name]
        need(pin['sha256'] == sha(raw) and pin['bytes'] == len(raw), 'archive member binding: ' + name)
    for name in names[2:]:
        member('archive-metadata/' + name, raws[name])
    member('analysis/independent-formal.json', raws['independent-formal.json'])
    member('analysis/restoration-audit.json', raws['restoration-audit.json'])
    member('analysis/execution-summary.json', raws['execution-summary.json'])
    member('cold-journal/journal.json', raws['restored-journal.json'])
    need(m['preparation/owner-plan.json']['sha256'] == PLAN_SHA, 'registered plan archive member')
    need(summary['schema'] == 'GDN-graph-combined-formal-archive-v2'
         and summary['status'] == summary['owner_status'] == 'PASS'
         and summary['owner_plan_sha256'] == PLAN_SHA and summary['performance_qualified'] is True
         and summary['audit_statuses'] == {'restoration-audit': 'PASS', 'independent-formal': 'PASS'},
         'all formal and restoration gates must pass, never promote retained failure')
    need(journal['schema'] == 'GDN-graph-combined-http-formal-cold-journal-v2'
         and journal['state'] == 'RESTORED' and journal['owner_plan']['sha256'] == PLAN_SHA
         and journal['trials'] == list(PHASES), 'restored exact twelve-process cycle')
    need(restore['schema'] == 'cold-service-restoration-audit-v1' and restore['status'] == 'PASS'
         and restore['qualified'] is True and restore['performance_qualified'] is False
         and restore['audit_source_sha256'] == RESTORE_SHA and restore['owner_plan_sha256'] == PLAN_SHA
         and restore['journal_sha256'] == sha(raws['restored-journal.json']) == audit['journal_sha256']
         and restore['service_restore']['replacement'] == journal['replacement']
         and audit['restoration'] == restore, 'independent restoration bindings')
    need(set(execution) == {'independent-formal', 'restoration-audit'}, 'both closed offline audit invocations')
    for label, digest, filename in (('independent-formal', AUDITOR_SHA, 'audit_formal.py'),
                                    ('restoration-audit', RESTORE_SHA, 'audit_restore.py')):
        job = execution[label]; raw = raws[label + '.json']; source = repo / SOURCE_DIR / filename
        need(job['status'] == 'PASS' and type(job['returncode']) is int and job['returncode'] == 0
             and positive(job['started_at']) and positive(job['ended_at']) and job['started_at'] <= job['ended_at']
             and job['source']['path'] == str(source) and job['source']['sha256'] == digest
             and job['output']['sha256'] == sha(raw) and job['output']['bytes'] == len(raw),
             'successful actual audit invocation and output')
        need(source.is_file() and not source.is_symlink() and source.stat().st_size < 1 << 20,
             'small original auditor source')
        source_raw = source.read_bytes(); need(sha(source_raw) == digest, 'registered auditor source unchanged')
        need(any(p['sha256'] == digest and p['bytes'] == len(source_raw) for p in m.values()),
             'auditor exact bytes retained in archive')
        raws['sources/' + filename] = source_raw
    member('sources/audit_formal.py', raws['sources/audit_formal.py'])
    final, final_raw = read(repo / (ARTIFACT_PREFIX + '-execution/final.json'))
    complete, complete_raw = read(repo / (ARTIFACT_PREFIX + '-owner/complete.json'))
    member('execution/final.json', final_raw); member('owner/complete.json', complete_raw)
    need(final['journal_state'] == 'RESTORED' and all(type(final[k]) is int and final[k] == 0
         for k in ('prepare_returncode', 'stop_returncode', 'owner_returncode', 'recovery_returncode')),
         'actual driver closed with all zero return codes')
    need(complete['status'] == 'PASS' and complete['formal_trials_complete'] is True
         and complete['all_children_reaped'] is True and complete['errors'] == []
         and complete['owner_plan_sha256'] == PLAN_SHA and complete['planned_trials'] == list(PHASES)
         and sha(complete_raw) == audit['owner_complete_sha256'], 'exact owner terminal binding')
    need(bool(audit['raw_file_pins']), 'independent raw inventory')
    for name, pin in audit['raw_file_pins'].items():
        archived = m['owner/' + name]
        need(pin == {k: archived[k] for k in ('sha256', 'bytes')}, 'audited raw archive membership')
    checked_points(audit)
    raws.update({'driver-final.json': final_raw, 'owner-complete.json': complete_raw})
    return audit, {'e033/' + name: raw for name, raw in raws.items()}


def summaries(points, display):
    need([d['point_index'] for d in display] == list(range(len(points))), 'every observation once in original order')
    need(set(d['configuration'] for d in display) == set(LABELS), 'exact independent configurations')
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
    need(old['schema'] == 'D512-explicit-frontier-v15'
         and len(old['points']) == len(old['display']) == 38
         and len(old['process_metric_medians']) == 14, 'immutable v15 38 observations /14 groups')
    points = old['points'] + new
    display = old['display'] + [dict(point_index=38 + i, configuration='native-combined-' + p['arm'])
                               for i, p in enumerate(new)]
    rows = summaries(points, display)
    for previous in old['process_metric_medians']:
        row = next(r for r in rows if (r['configuration'], r['concurrency']) ==
                   (previous['configuration'], previous['concurrency']))
        need(row == previous, 'original group unchanged')
    fronts = {x: {engine: frontier([r for r in rows if r['engine'] == engine], x)
                   for engine in ('native', 'omlx', 'lithos')} for x in XS}
    joint = {x: frontier(rows, x) for x in XS}
    need({(r['configuration'], r['concurrency']) for r in joint[XS[0]]} ==
         {(r['configuration'], r['concurrency']) for r in frontier(rows, XS[0], True)}, 'reciprocal dominance invariance')
    return dict(schema='D512-explicit-frontier-v16',
        hardware=old['hardware'], model=old['model'], revision=old['revision'], workload=old['workload'],
        historical_sources=old['historical_sources'], points=points, display=display, performance_point_count=50,
        previous_points_exactly_unchanged=True, previous_points_sha256=OLD_SHA,
        process_metric_medians=rows, per_engine_frontiers=fronts, joint_frontiers=joint,
        historical_paired_comparisons=old.get('historical_paired_comparisons'),
        gdn_paired_comparisons=old['gdn_paired_comparisons'],
        combined_cycle_paired_comparisons=audit['comparisons'],
        historical_qualification_status=old.get('historical_qualification_status'),
        gdn_performance_prerequisite=old['gdn_performance_prerequisite'],
        combined_performance_prerequisite=dict(audit_source_sha256=AUDITOR_SHA, owner_plan_sha256=PLAN_SHA,
            route_always_on=True, actual_prework_calls_observed=False, actual_compiled_invocations_observed=False),
        scope=['50 process observations,18 separate configuration/C groups; historical Native has N=1, other groups N=3.',
            'E033 compares the same entry with graph ON in both arms and GDN OFF/ON. It measures incremental GDN on graph, not combined versus all switches OFF.',
            'New graph-ON/GDN-OFF, earlier E024 ON, E024 OFF and GDN-only OFF are separate cycles, never pooled. Pairing is only within each registered cycle.',
            'Combined ON is a separately measured executable configuration; no sum or product of earlier percentage gains is used. E028 identity-refill remains OFF.',
            'Each engine frontier may select separately measured configurations. Joint segments do not describe an executable mixture.',
            'Only HTTP C1/C8 measured; C2–7 are unknown. HTTP concurrency is not GPU batch size.',
            'Medians/min–max are across process metrics, not pooled request percentiles or confidence intervals.',
            '512/median_process(p50_POST-to-DONE) is full-request normalized speed, not ITL or TPOT.',
            'Strict empirical nondominance is derived from all summaries; no significance threshold or predetermined winner.',
            'Formal observers OFF: prior captured histories qualified; no proof of unseen schedules, formal dispatch counts or general quality.',
            'Old per-point nondominance flags retain their original historical scope; use the new frontier fields.'])


def inputs(repo):
    repo = Path(repo).resolve()
    old, raw = read(repo / OLD); need(sha(raw) == OLD_SHA, 'frozen v15 chartdata bytes')
    audit, evidence = closed_inputs(repo); evidence['v15/chartdata.json'] = raw
    return old, audit, evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true'); parser.add_argument('--repo', type=Path)
    parser.add_argument('--out', type=Path); args = parser.parse_args()
    if not args.run:
        print('DESCRIPTION_ONLY: E033 full independent PASS + RESTORED + verified archive required; no files written.'); return
    need(args.repo is not None and args.out is not None, '--run requires --repo and fresh --out')
    need(not args.out.exists(), 'exclusive output; old or failed results are never overwritten')
    old, audit, evidence = inputs(args.repo); result = build(old, audit)
    args.out.mkdir(parents=True); pins = {}
    for name, blob in evidence.items():
        target = args.out / 'inputs' / name; target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(blob)
        pins['inputs/' + name] = dict(sha256=sha(blob), bytes=len(blob))
    result['input_pins'] = pins
    (args.out / 'chartdata.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    receipt = dict(status='PREPARED', schema='D512-v16-preparation-v1', source_sha256=sha(Path(__file__).read_bytes()),
        chartdata_sha256=sha((args.out / 'chartdata.json').read_bytes()), input_pins=pins,
        archive_stream_rehashed=False, raw_http_reaudited=False, figures_created=False)
    (args.out / 'preparation.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
