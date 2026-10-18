"""Small invented arithmetic fixtures only. No actual timing/results/package import.

Prepared during a formal benchmark; do not execute until its owner authorizes it.
These tests do not validate a real audit/archive or visual output.
"""
import copy
import importlib.util
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location('_v15_fake_math', Path(__file__).with_name('prepare.py'))
m = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(m)


def fake_inputs():
    old = dict(points=[], display=[], process_metric_medians=[], paired_comparisons=['old-pairs'],
               hardware='FAKE', model='FAKE', revision='FAKE', workload='FAKE', sources=['FAKE'])
    for index, configuration in enumerate(list(m.LABELS)[:3] + ['omlx', 'lithos']):
        for c in (1, 8):
            for repeat in range(1, (2 if configuration == 'native-historical' else 4)):
                point = dict(concurrency=c, repeat=repeat, aggregate_output_tok_s=10 + index + repeat,
                             e2e_p50_s=20 + c + repeat, e2e_p95_s=30 + c + repeat,
                             historical_flag='unchanged')
                old['display'].append(dict(configuration=configuration, point_index=len(old['points'])))
                old['points'].append(point)
    audit = dict(schema='GDN-prework-http-formal-independent-audit-v1', status='PASS', qualified=True,
                 performance_qualified=True, audit_source_sha256=m.AUDITOR_SHA, owner_plan_sha256=m.PLAN_SHA,
                 requests=dict(total=864, warm=96, formal=768), http_raw_opened=True,
                 actual_prework_calls_observed=False, resources={'scope': 'FAKE'}, points=[], comparisons=['new-pairs'])
    for phase in m.PHASES:
        repeat, c, arm = phase.split('-'); repeat, c = int(repeat[1:]), int(c[1:])
        throughput = 30 + repeat if arm == 'off' else 40 + repeat
        audit['points'].append(dict(phase=phase, arm=arm, concurrency=c, repeat=repeat,
            requests=64, committed_output_tokens=32768, n_fresh_processes=1, warm_requests_checked=8,
            max_http_outstanding=c, common_window_s=32768 / throughput, aggregate_output_tok_s=throughput,
            e2e_p50_s=10 + c + repeat, e2e_p95_s=12 + c + repeat, actual_prework_calls_observed=False))
    display = old['display'] + [dict(configuration='native-gdn-' + p['arm'], point_index=26 + i)
                              for i, p in enumerate(audit['points'])]
    all_rows = m.summaries(old['points'] + audit['points'], display)
    old['process_metric_medians'] = [r for r in all_rows if not r['configuration'].startswith('native-gdn-')]
    return old, audit


def row(name, latency, throughput):
    return dict(configuration=name, metrics={m.XS[0]: dict(median=latency), m.Y: dict(median=throughput)})


class MathAndGates(unittest.TestCase):
    def test_old26_exact_and_off_cycles_separate(self):
        old, audit = fake_inputs(); original = copy.deepcopy(old)
        out = m.build(old, audit)
        self.assertEqual(out['points'][:26], original['points'])
        self.assertEqual(out['display'][:26], original['display'])
        self.assertEqual(old, original)
        self.assertEqual(len(out['points']), 38); self.assertEqual(len(out['process_metric_medians']), 14)
        rows = {r['configuration']: r for r in out['process_metric_medians'] if r['concurrency'] == 1}
        self.assertEqual(rows['native-off']['fresh_processes'], 3)
        self.assertEqual(rows['native-gdn-off']['fresh_processes'], 3)
        self.assertEqual(rows['native-off']['metrics'][m.Y]['median'], 13)
        self.assertEqual(rows['native-gdn-off']['metrics'][m.Y]['median'], 32)
        self.assertEqual(out['historical_paired_comparisons'], ['old-pairs'])
        self.assertEqual(out['gdn_paired_comparisons'], ['new-pairs'])

    def test_no_preselected_engine_winner(self):
        rows = [row('old', 10, 100), row('new', 9, 110), row('other', 30, 200)]
        self.assertEqual([r['configuration'] for r in m.frontier(rows, m.XS[0])], ['new', 'other'])
        rows[0]['metrics'][m.Y]['median'] = 300
        self.assertEqual([r['configuration'] for r in m.frontier(rows, m.XS[0])], ['new', 'old'])

    def test_equal_metrics_keep_distinct_configurations(self):
        rows = [row('E024', 10, 100), row('GDN', 10, 100)]
        self.assertEqual(len(m.frontier(rows, m.XS[0])), 2)

    def test_reciprocal_preserves_dominance_not_median_of_reciprocals(self):
        rows = [row('fast', 8, 60), row('dominated', 10, 50), row('batch', 20, 200)]
        direct = {r['configuration'] for r in m.frontier(rows, m.XS[0])}
        normalized = {r['configuration'] for r in m.frontier(rows, m.XS[0], True)}
        self.assertEqual(direct, normalized)
        self.assertEqual(direct, {'fast', 'batch'})

    def test_failed_or_incomplete_audit_never_adds_points(self):
        for status in ('FAIL', 'INCOMPLETE'):
            old, audit = fake_inputs(); audit['status'] = status
            with self.assertRaisesRegex(ValueError, 'formal PASS'):
                m.build(old, audit)

    def test_partial_or_reordered_process_prefix_rejected(self):
        for mode in ('missing', 'reordered'):
            old, audit = fake_inputs()
            if mode == 'missing': audit['points'].pop()
            else: audit['points'][0], audit['points'][1] = audit['points'][1], audit['points'][0]
            with self.assertRaisesRegex(ValueError, 'registered order'):
                m.build(old, audit)

    def test_wrong_scope_source_counts_and_nonfinite_rejected(self):
        for key, value in (('audit_source_sha256', 'wrong'), ('owner_plan_sha256', 'wrong'),
                           ('actual_prework_calls_observed', True), ('resources', {})):
            old, audit = fake_inputs(); audit[key] = value
            with self.assertRaises(ValueError): m.build(old, audit)
        for key, value in (('requests', True), ('committed_output_tokens', 32767),
                           ('aggregate_output_tok_s', float('nan')), ('common_window_s', 1)):
            old, audit = fake_inputs(); audit['points'][0][key] = value
            with self.assertRaises(ValueError): m.build(old, audit)

    def test_previous_summary_drift_refused(self):
        old, audit = fake_inputs()
        old['process_metric_medians'][0]['metrics'][m.Y]['median'] += 1
        with self.assertRaisesRegex(ValueError, 'original group unchanged'):
            m.build(old, audit)


if __name__ == '__main__':
    unittest.main()
