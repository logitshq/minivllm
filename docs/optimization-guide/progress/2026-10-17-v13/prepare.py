"""Prepare v13 from CLOSED audited receipts; default is description only.

This retains the v12 display policy fixed before E024 ON results were available.
Only the post-run tokenizer-path parser correction and archive bindings change. Keep all
fourteen v11 observations unchanged; append all twelve OFF/ON observations only
after complete formal/resource/restoration PASS and verified archival retention.
No model/runtime imports, raw requests, rendering, or current-link updates.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
V11 = HERE.parent / '2026-10-17-v11'
ARCHIVE = ROOT / 'docs/experiments/m5-ultra-256gb/E024-route-http-formal-v2-reparse-v1'
FIRST_ARCHIVE = ROOT / 'docs/experiments/m5-ultra-256gb/E024-route-http-formal-v2'
V12 = HERE.parent / '2026-10-17-v12'
BENCH = ROOT / 'bench/m5_ultra_256gb/route_http_formal'
ART = ROOT / '.artifacts/m5-ultra-256gb'
V11_SHA = 'df412e157da1e08436bdf81ef1cfaa126eaca42da7d12d6241b8ad6e46073123'
V11_RENDER_SHA = '6e34ab1253e708941cb4d40c7590341997d4e62b732e636668adf3814dc38f0c'
PLAN_SHA = '0920aceae4e8bb6ef5c68577d3ae905a2b5976287790a7bb7a4a6e4ea36470ce'
AUDIT_SHA = 'b2bc55b87043d66d79154fd32dc7579cc7c6fdb457844a1b1d3df932ff41224e'
FROZEN_V1_AUDIT_SHA = 'ed8ad64c3eb5fed052effd05981a386006f4ea1f752463bb0ada0c2f0e4ea5e0'
FIRST_FAILURE_SHA = 'a9ee8acd98c4b89eee3a96f709a6f4a63225a4b5b088aaae7093ac8d9b14f990'
FIRST_ARCHIVE_SHA = '9174c276c76405651cc351f8e248e98b68aba0c257831a8bc5706e4b2fb52b7b'
HISTORY_PINS = {
    'v12-prepare.py': '5bc8be7c8f6a5994eaff1895e254f9acfbaf483a309ccbf6af01a09c4ee0d966',
    'v12-render.py': 'a3b3333bd4e89db07a29277e37cd2411037e06c54a4faf25b05fcefc02bde8cc',
    'v12-README.md': 'c57fa7c2dd4c43d447536dcf219171026b0c34b602800ae2d9f6086ad5d32f77',
    'v12-render-queue.json': 'b285e00bc7de57ea129a70a7a84131029ff95a88da619c0ddf79fd34ec6bb5ae',
    'first-formal-failure.json': FIRST_FAILURE_SHA,
    'first-archive-verification.json': '74e420089ed0b23678c3b7295d5684b846c839080f2e656f9830986709092471',
    'tokenizer-cpu-result.json': '442d29d0f29039c25ac329419beecf31b0d92fba570c6fd3e2c28bb695192a29',
    'audit_formal_v2.py': AUDIT_SHA,
    'test_audit_formal_v2_cpu.py': '9bff5e7d2c1d52d18c0c39d4a286cd17e3be28c19128a82078dc30b8fc120bac',
}
RESTORE_SHA = '90a980d0dbe6a61fc1bab2ac6b2edaa9bf012bb60726cd0742801d11410eb0b1'
PACKER_SHA = '528101c543d6f85ec74828591a4c14bce3902733995830c3daf1d510606664fa'
PHASES = ('r1-c1-off', 'r1-c1-on', 'r1-c8-on', 'r1-c8-off',
          'r2-c8-off', 'r2-c8-on', 'r2-c1-on', 'r2-c1-off',
          'r3-c1-off', 'r3-c1-on', 'r3-c8-on', 'r3-c8-off')
METRICS = ('aggregate_output_tok_s', 'e2e_p50_s', 'e2e_p95_s')
CONFIGURATIONS = ('native-historical', 'native-off', 'native-on', 'omlx', 'lithos')
LABELS = dict(zip(CONFIGURATIONS, ('Native historical', 'Native OFF', 'Native ON', 'oMLX', 'Lithos')))
PATHS = {
    **{'v12-'+name: V12/name for name in ('prepare.py','render.py','README.md')},
    'v12-render-queue.json': ART/'progress-v12-offline-render-queue/status.json',
    'first-formal-failure.json': FIRST_ARCHIVE/'independent-formal.json',
    'first-archive-verification.json': FIRST_ARCHIVE/'archive-verification.json',
    'tokenizer-cpu-result.json': ART/'route-http-formal-tokenizer-cpu-v1/result.json',
    'audit_formal_v2.py': BENCH/'audit_formal_v2.py',
    'test_audit_formal_v2_cpu.py': BENCH/'test_audit_formal_v2_cpu.py',
    'v11-chartdata.json': V11 / 'chartdata.json',
    'v11-verification.json': V11 / 'verification.json',
    'v11-render.py': V11 / 'render.py',
    **{name: ARCHIVE / name for name in ('independent-formal.json', 'restoration-audit.json',
        'restored-journal.json', 'audit-execution-summary.json', 'summary.json',
        'archive-verification.json', 'archive-stream.json', 'archive.py')},
    'archive-manifest.json': ARCHIVE / 'manifest.json',
}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def same(a, b, message):
    need(json.dumps(a, sort_keys=True, allow_nan=False) ==
         json.dumps(b, sort_keys=True, allow_nan=False), message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def parse(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            need(key not in result, 'duplicate JSON key: ' + key)
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=pairs,
        parse_constant=lambda x: (_ for _ in ()).throw(ValueError('nonfinite JSON: ' + x)))


def write_new(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def build(data, manifest):
    """Pure receipt validation and display data; no reconstruction of raw timing."""
    for name,digest in HISTORY_PINS.items():
        need(manifest[name]['sha256']==digest,'unchanged predeclared policy/failure/correction source: '+name)
    first=data['first-formal-failure.json']; prior_archive=data['first-archive-verification.json']
    no_chart=data['v12-render-queue.json']; cpu=data['tokenizer-cpu-result.json']
    need(first['schema']=='E024-route-http-formal-independent-audit-v1' and first['status']=='FAIL'
         and first['audit_source_sha256']==FROZEN_V1_AUDIT_SHA and first['performance_qualified'] is False
         and first['http_raw_opened'] is False
         and first['first_failure']==dict(type='ValueError',message='canonical unchanged pinned ref'),
         'preserved actual first rejection, not a rewritten PASS')
    need(prior_archive['status']=='PASS' and prior_archive['archive']['sha256']==FIRST_ARCHIVE_SHA,
         'original failed-audit archive retained')
    need(no_chart['status']=='NO_NEW_SPEED_CHART' and no_chart['jobs']==[]
         and no_chart['rendered'] is no_chart['current_link_updated'] is False,
         'v12 correctly withheld a speed update')
    need(cpu['status']=='PASS' and cpu['tests']==8 and cpu['failures']==cpu['errors']==cpu['skipped']==0
         and cpu['sources_unchanged'] is True and cpu['denied']==[],
         'actual targeted tokenizer correction regression checks')
    old = data['v11-chartdata.json']; previous = data['v11-verification.json']
    need(manifest['v11-chartdata.json']['sha256'] == V11_SHA and
         manifest['v11-render.py']['sha256'] == V11_RENDER_SHA, 'fixed v11 chart/design')
    need(old['schema'] == 'D512-throughput-latency-progress-v11' and
         old['performance_point_count'] == len(old['points']) == 14, 'fourteen existing observations')
    need(previous['status'] == 'PASS' and previous['chartdata_sha256'] == V11_SHA and
         previous['renderer_sha256'] == V11_RENDER_SHA and previous['performance_points'] == 14,
         'actual v11 validation receipt')
    audit = data['independent-formal.json']; restore = data['restoration-audit.json']
    journal = data['restored-journal.json']; execution = data['audit-execution-summary.json']
    archive = data['summary.json']; verified = data['archive-verification.json']
    stream = data['archive-stream.json']; retained = data['archive-manifest.json']
    need(manifest['archive.py']['sha256'] == PACKER_SHA, 'reviewed immutable retention program')
    need(retained['schema'] == 'E024-route-formal-archive-manifest-v1' and
         verified['schema'] == 'E024-route-formal-archive-verification-v1' and verified['status'] == 'PASS',
         'complete archive verification required')
    for key in ('all_safe_extracted_members_verified', 'all_original_files_rehashed', 'all_tree_members_rechecked'):
        need(verified[key] is True, 'archive coverage: ' + key)
    same(verified['archive'], stream['archive'], 'same verified archive stream')
    same(verified['parts'], stream['parts'], 'same verified archive parts')
    need(verified['members'] == len(retained['files']) and verified['packaging_grants_qualification'] is False,
         'retention is not qualification')
    for name in ('independent-formal.json', 'restoration-audit.json', 'restored-journal.json',
                 'audit-execution-summary.json', 'summary.json', 'archive.py'):
        prefix = 'archive-metadata/' if name in ('summary.json', 'archive.py') else 'archive-sidecars/'
        entry = retained['files'][prefix + name]
        same({k: entry[k] for k in ('sha256', 'bytes')},
             {k: manifest[name][k] for k in ('sha256', 'bytes')}, 'exact retained sidecar: ' + name)
    need(journal['state'] == 'RESTORED' and journal['owner_plan']['sha256'] == PLAN_SHA,
         'actual registered cycle restored')
    need(restore['schema'] == 'cold-service-restoration-audit-v1' and restore['status'] == 'PASS'
         and restore['qualified'] is True and restore['performance_qualified'] is False
         and restore['audit_source_sha256'] == RESTORE_SHA and restore['owner_plan_sha256'] == PLAN_SHA
         and restore['journal_sha256'] == manifest['restored-journal.json']['sha256'],
         'independent restoration PASS')
    same(restore['service_restore']['replacement'], journal['replacement'], 'actual restored replacement')
    need(execution['schema'] == 'E024-route-formal-closed-audits-v1' and execution['status'] == 'PASS'
         and execution['owner_plan_sha256'] == PLAN_SHA and all(execution[k] is True for k in
         ('execution_pass', 'restoration_qualified', 'performance_qualified', 'resources_verified_with_formal', 'sources_unchanged')),
         'FAIL/partial cycle: no v13 speed update; leave v11 current')
    for label, name in (('formal', 'independent-formal.json'), ('restoration', 'restoration-audit.json')):
        job = execution['jobs'][label]
        need(job['status'] == 'RETURNED' and type(job['returncode']) is int and job['returncode'] == 0
             and job['audit_status'] == 'PASS', 'actual successful audit invocation')
        same({k: job['output'][k] for k in ('sha256', 'bytes')},
             {k: manifest[name][k] for k in ('sha256', 'bytes')}, 'actual audit output identity')
    need(audit['schema'] == 'E024-route-http-formal-independent-audit-v2' and audit['status'] == 'PASS'
         and audit['qualified'] is audit['performance_qualified'] is True
         and audit['owner_plan_sha256'] == PLAN_SHA and audit['audit_source_sha256'] == AUDIT_SHA
         and audit['http_raw_opened'] is True and audit['actual_compiled_invocations_observed'] is False,
         'complete observer-OFF independent formal audit')
    need(audit['frozen_v1_audit_source_sha256']==FROZEN_V1_AUDIT_SHA
         and audit['tokenizer_reference']['is_symlink'] is True,
         'explicit registered tokenizer-path parser correction provenance')
    correction=execution['parser_correction']
    need(correction['post_run'] is True
         and correction['original_audit']['sha256']==manifest['first-formal-failure.json']['sha256']
         and correction['cpu_checks']['sha256']==manifest['tokenizer-cpu-result.json']['sha256'],
         'same preserved first failure and actual correction CPU result')
    same(archive['parser_correction'],correction,'archived post-run correction identity')
    same(audit['restoration'], restore, 'same independently audited restoration')
    same(audit['requests'], dict(total=864, warm=96, formal=768), 'all twelve complete processes')
    need(audit['journal_sha256'] == restore['journal_sha256'] and
         audit['resources']['pressure'] == 1 and audit['resources']['samples'] > 0,
         'same actual integrated resource gate')
    need(archive['schema'] == 'E024-route-formal-archive-summary-v1' and archive['owner_status'] == 'PASS'
         and archive['audit_status'] == 'PASS' and archive['performance_qualified'] is True
         and archive['restoration_qualified'] is True and archive['owner_plan_sha256'] == PLAN_SHA,
         'closed archived complete cycle')
    same(archive['actual_closed_trial_names'], list(PHASES), 'all twelve archived trials')
    new = audit['points']; same([p['phase'] for p in new], list(PHASES), 'registered process order')
    for p in new:
        need(p['phase'] == f"r{p['repeat']}-c{p['concurrency']}-{p['arm']}" and
             p['arm'] in ('off', 'on') and p['concurrency'] in (1, 8) and p['repeat'] in (1, 2, 3)
             and p['requests'] == 64 and p['warm_requests_checked'] == 8
             and p['max_http_outstanding'] == p['concurrency'] and p['n_fresh_processes'] == 1
             and p['actual_compiled_invocations_observed'] is False
             and all(positive(p[k]) for k in METRICS), 'complete actual finite process observation')
    # Keep raw source point dictionaries intact, including historical annotations.
    points = old['points'] + new
    display = [dict(point_index=i, configuration={'mini-vLLM': 'native-historical',
        'oMLX': 'omlx', 'Lithos': 'lithos'}[p['engine']]) for i, p in enumerate(old['points'])]
    display += [dict(point_index=14+i, configuration='native-' + p['arm']) for i, p in enumerate(new)]
    summaries = []
    for configuration in CONFIGURATIONS:
        for C in (1, 8):
            group = [points[r['point_index']] for r in display if r['configuration'] == configuration
                     and points[r['point_index']]['concurrency'] == C]
            expected = 1 if configuration == 'native-historical' else 3
            need(len(group) == expected, 'no historical/paired configuration pooling')
            summaries.append(dict(configuration=configuration, label=LABELS[configuration], concurrency=C,
                fresh_processes=len(group), metrics={k: dict(median=statistics.median(p[k] for p in group),
                    minimum=min(p[k] for p in group), maximum=max(p[k] for p in group)) for k in METRICS}))
    for row in old['process_metric_medians']:
        configuration = {'mini-vLLM': 'native-historical', 'oMLX': 'omlx', 'Lithos': 'lithos'}[row['engine']]
        computed = next(s for s in summaries if s['configuration'] == configuration and s['concurrency'] == row['concurrency'])
        need(computed['fresh_processes'] == row['fresh_processes'], 'unchanged historical process count')
        for key in METRICS:
            same(computed['metrics'][key], dict(median=row[key], minimum=row['observed_process_ranges'][key][0],
                 maximum=row['observed_process_ranges'][key][1]), 'unchanged historical process summaries')
    comparisons = audit['comparisons']  # Display frozen auditor values, never newly inferred pairing.
    need(len(comparisons['pairs']) == 6 and len(comparisons['process_medians']) == 4, 'six audited pairs/four arm-C summaries')
    for i, pair in enumerate(comparisons['pairs']):
        first, second = new[2*i:2*i+2]; arms = {p['arm']: p for p in (first, second)}
        off, on = arms['off'], arms['on']
        same(pair, dict(repeat=off['repeat'], concurrency=off['concurrency'], order=[first['arm'], second['arm']],
            off_phase=off['phase'], on_phase=on['phase'],
            throughput_on_over_off=on[METRICS[0]]/off[METRICS[0]],
            e2e_p50_on_over_off=on[METRICS[1]]/off[METRICS[1]],
            e2e_p95_on_over_off=on[METRICS[2]]/off[METRICS[2]]), 'same frozen adjacent pair ratios')
    for row in comparisons['process_medians']:
        computed = next(s for s in summaries if s['configuration'] == 'native-' + row['arm']
                        and s['concurrency'] == row['concurrency'])
        need(row['n_fresh_processes'] == 3, 'three distinct fresh processes')
        same(row['metrics'], computed['metrics'], 'same frozen process metric medians/ranges')
    qualification = [dict(q) for q in old['qualification_status']]
    route = next(q for q in qualification if q['candidate'] == 'E024 route graph')
    route.update(status='HTTP history + paired formal PASS', evidence='12 fresh processes; 768 formal + 96 warm',
        next_gate='This configuration only; no future-arrival proof', performance_qualified=True)
    need(next(q for q in qualification if q['candidate'] == 'GDN prework')['performance_qualified'] is False,
         'GDN37 adds no speed point')
    return dict(schema='D512-throughput-latency-progress-v13',
        **{k: old[k] for k in ('hardware', 'model', 'revision', 'workload')},
        previous_chartdata_sha256=V11_SHA, points=points, display=display,
        display_policy_origin='v12 sources fixed before ON results; metrics/grouping/pairing policy unchanged',
        parser_correction=dict(post_run=True,original_audit_sha256=FIRST_FAILURE_SHA,
            original_archive_sha256=FIRST_ARCHIVE_SHA,corrected_audit_source_sha256=AUDIT_SHA,
            actual_corrected_audit_sha256=manifest['independent-formal.json']['sha256'],
            v12_speed_update_withheld=True,timed_experiment_rerun=False,
            scope='Registered HF tokenizer snapshot symlink handling only; original FAIL/raw evidence retained.'),
        process_metric_medians=summaries, historical_process_metric_medians=old['process_metric_medians'],
        paired_comparisons= comparisons, qualification_status=qualification,
        performance_point_count=26, historical_points_exactly_unchanged=True,
        native_historical_pooled_with_off=False, sources=manifest,
        legacy_point_annotation_scope='Any retained nondominated_* fields belong to the original fourteen-point v11 set only; v13 rendering does not use them.',
        scope='All26 audited process observations. Historical native n=1/C is separate from new paired OFF/ON n=3/configuration/C. '
              'Medians/min-max summarize process metrics, not pooled requests or confidence intervals. p50 and p95 are separate. '
              'No interpolation, global-fastest/frontier or observed-formal-graph-hit claim. Different cycles and realized arrival schedules; '
              'no proof of all future arrivals, cross-engine full-state/answer-quality equality or E024+GDN composition. '
              'Lithos C8 uses official FIFO GPU execution; original oMLX parser FAIL remains preserved in v11 provenance. '
              'E024 original tokenizer-path auditor FAIL and v12 withheld chart are retained; v13 uses the complete corrected offline audit, not a rerun.')


def main():
    need(not (HERE/'sources').exists() and not (HERE/'chartdata.json').exists()
         and not (HERE/'preparation.json').exists(), 'exclusive new preparation; never overwrite attempts')
    raws = {}
    for name, path in PATHS.items():
        need(path.is_file() and not path.is_symlink() and path.stat().st_size < 32*(1<<20), 'closed small receipt/source: ' + name)
        raws[name] = path.read_bytes()
    manifest = {name: dict(path=str(PATHS[name]), sha256=sha(raw), bytes=len(raw)) for name, raw in raws.items()}
    data = {name: parse(raw) for name, raw in raws.items() if name.endswith('.json')}
    chart = build(data, manifest)  # No output created unless every eligibility gate passes.
    need(all(path.read_bytes() == raws[name] for name, path in PATHS.items()), 'closed input changed while preparing')
    (HERE/'sources').mkdir()
    for name, raw in raws.items():
        with (HERE/'sources'/name).open('xb') as stream: stream.write(raw)
        need((HERE/'sources'/name).read_bytes() == raw, 'exact copied source')
    write_new(HERE/'sources/manifest.json', manifest)
    write_new(HERE/'chartdata.json', chart)
    write_new(HERE/'preparation.json', dict(schema='D512-progress-v13-preparation-v1', status='PREPARED',
        performance_points=26, historical_points=14, new_points=12, actual_requests=864,
        previous_chartdata_sha256=V11_SHA, owner_plan_sha256=PLAN_SHA,
        prepare_sha256=sha(Path(__file__).read_bytes()), renderer_sha256=sha((HERE/'render.py').read_bytes()),
        chartdata_sha256=sha((HERE/'chartdata.json').read_bytes()),
        manifest_sha256=sha((HERE/'sources/manifest.json').read_bytes()),
        current_link_updated=False, visual_review_required=True,
        scope='Closed small receipts only. No raw-request re-audit, tar payload rehash, model, render or current-link update.'))
    print('PREPARED:26 observations; render and visual QA remain separate; current link unchanged.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    if parser.parse_args().run: main()
    else: print('DESCRIPTION_ONLY: all formal/resource/restore/retention gates must PASS; otherwise retain v11.')
