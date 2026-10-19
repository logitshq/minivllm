"""Eight fake-data gate/arithmetic tests, prepared but NOT RUN during timing.

No actual result, archive, plotting package, runtime or device is imported.
These mechanisms do not substitute for actual CLOSED metadata + image review.
"""
import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import statistics
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location('_v16_fake_math', Path(__file__).with_name('prepare.py'))
m = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(m)


def fake_inputs():
    old = dict(schema='D512-explicit-frontier-v15', points=[], display=[], process_metric_medians=[],
        historical_paired_comparisons=['FAKE-old-pairs'], gdn_paired_comparisons=['FAKE-GDN-pairs'],
        gdn_performance_prerequisite={'FAKE': True}, hardware='FAKE', model='FAKE', revision='FAKE',
        workload='FAKE', historical_sources=['FAKE'])
    for index, configuration in enumerate(list(m.LABELS)[:7]):
        for c in (1, 8):
            selected = []
            for repeat in range(1, 2 if configuration == 'native-historical' else 4):
                point = dict(concurrency=c, repeat=repeat, aggregate_output_tok_s=10 + index + repeat,
                    e2e_p50_s=20 + c + repeat, e2e_p95_s=30 + c + repeat, untouched='FAKE')
                old['display'].append(dict(configuration=configuration, point_index=len(old['points'])))
                old['points'].append(point); selected.append(point)
            old['process_metric_medians'].append(dict(configuration=configuration,
                engine=m.ENGINE[configuration], label=m.LABELS[configuration], concurrency=c,
                fresh_processes=len(selected), metrics={k: dict(median=statistics.median(p[k] for p in selected),
                    minimum=min(p[k] for p in selected), maximum=max(p[k] for p in selected)) for k in (m.Y, *m.XS)}))
    audit = dict(schema='GDN-graph-combined-http-formal-independent-audit-v2', status='PASS', qualified=True,
        performance_qualified=True, audit_source_sha256=m.AUDITOR_SHA, owner_plan_sha256=m.PLAN_SHA,
        requests=dict(total=864, warm=96, formal=768), http_raw_opened=True, route_always_on=True,
        actual_prework_calls_observed=False, actual_compiled_invocations_observed=False,
        resources={'scope': 'FAKE'}, points=[], comparisons=['FAKE-combined-pairs'])
    for phase in m.PHASES:
        repeat, c, arm = phase.split('-'); repeat, c = int(repeat[1:]), int(c[1:])
        throughput = (30 if arm == 'off' else 40) + repeat
        audit['points'].append(dict(phase=phase, arm=arm, concurrency=c, repeat=repeat,
            requests=64, committed_output_tokens=32768, n_fresh_processes=1, warm_requests_checked=8,
            max_http_outstanding=c, common_window_s=32768 / throughput, aggregate_output_tok_s=throughput,
            e2e_p50_s=10 + c + repeat, e2e_p95_s=12 + c + repeat, route_always_on=True,
            actual_prework_calls_observed=False, actual_compiled_invocations_observed=False))
    return old, audit


def row(name, latency, throughput):
    return dict(configuration=name, metrics={m.XS[0]: dict(median=latency), m.Y: dict(median=throughput)})


class MathAndGates(unittest.TestCase):
    def test_original38_and14groups_exact_and_cycles_not_pooled(self):
        old, audit = fake_inputs(); original = copy.deepcopy(old)
        result = m.build(old, audit)
        self.assertEqual(old, original)
        self.assertEqual(result['points'][:38], old['points'])
        self.assertEqual(result['display'][:38], old['display'])
        self.assertEqual(result['process_metric_medians'][:14], old['process_metric_medians'])
        self.assertEqual(len(result['points']), 50); self.assertEqual(len(result['process_metric_medians']), 18)
        rows = {r['configuration']: r for r in result['process_metric_medians'] if r['concurrency'] == 1}
        self.assertEqual(rows['native-on']['fresh_processes'], 3)
        self.assertEqual(rows['native-combined-off']['fresh_processes'], 3)
        self.assertNotEqual(rows['native-on']['metrics'], rows['native-combined-off']['metrics'])
        self.assertEqual(result['gdn_paired_comparisons'], ['FAKE-GDN-pairs'])
        self.assertEqual(result['combined_cycle_paired_comparisons'], ['FAKE-combined-pairs'])

    def test_winner_is_derived_and_equal_configurations_survive(self):
        rows = [row('old', 10, 100), row('new', 9, 110), row('other', 30, 200)]
        self.assertEqual([r['configuration'] for r in m.frontier(rows, m.XS[0])], ['new', 'other'])
        rows[0]['metrics'][m.Y]['median'] = 300
        self.assertEqual([r['configuration'] for r in m.frontier(rows, m.XS[0])], ['new', 'old'])
        self.assertEqual(len(m.frontier([row('A', 10, 100), row('B', 10, 100)], m.XS[0])), 2)

    def test_full_request_reciprocal_preserves_dominance(self):
        rows = [row('fast', 8, 60), row('dominated', 10, 50), row('batch', 20, 200)]
        self.assertEqual({r['configuration'] for r in m.frontier(rows, m.XS[0])}, {'fast', 'batch'})
        self.assertEqual({r['configuration'] for r in m.frontier(rows, m.XS[0], True)}, {'fast', 'batch'})

    def test_failed_incomplete_or_only_restoration_success_cannot_add_points(self):
        for status in ('FAIL', 'INCOMPLETE', 'EXPERIMENT_FAILED_EVIDENCE_RETAINED'):
            old, audit = fake_inputs(); audit['status'] = status; audit['restoration'] = {'status': 'PASS'}
            with self.assertRaisesRegex(ValueError, 'formal PASS'):
                m.build(old, audit)
        old, audit = fake_inputs(); audit['performance_qualified'] = False
        with self.assertRaises(ValueError): m.build(old, audit)

    def test_partial_reordered_or_changed_arm_rejected(self):
        for mode in ('missing', 'reordered', 'wrong-arm'):
            old, audit = fake_inputs()
            if mode == 'missing': audit['points'].pop()
            elif mode == 'reordered': audit['points'][0], audit['points'][1] = audit['points'][1], audit['points'][0]
            else: audit['points'][0]['arm'] = 'on'
            with self.assertRaises(ValueError): m.build(old, audit)

    def test_scope_counts_sources_nonfinite_and_graph_off_rejected(self):
        for key, value in (('audit_source_sha256', 'wrong'), ('owner_plan_sha256', 'wrong'),
                           ('route_always_on', False), ('actual_prework_calls_observed', True),
                           ('actual_compiled_invocations_observed', True), ('resources', {})):
            old, audit = fake_inputs(); audit[key] = value
            with self.assertRaises(ValueError): m.build(old, audit)
        for key, value in (('requests', True), ('committed_output_tokens', 32767), ('concurrency', True),
                           ('aggregate_output_tok_s', float('nan')), ('common_window_s', 1),
                           ('e2e_p95_s', 0.5), ('route_always_on', False)):
            old, audit = fake_inputs(); audit['points'][0][key] = value
            with self.assertRaises(ValueError): m.build(old, audit)

    def test_previous_summary_drift_refused(self):
        old, audit = fake_inputs(); old['process_metric_medians'][0]['metrics'][m.Y]['median'] += 1
        with self.assertRaisesRegex(ValueError, 'original group unchanged'): m.build(old, audit)

    def test_default_inert_and_missing_archive_leaves_no_result(self):
        with patch('sys.argv', ['prepare.py']), patch.object(m, 'inputs', side_effect=AssertionError('must not read')):
            with contextlib.redirect_stdout(io.StringIO()): m.main()
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp); path = repo / m.OLD; path.parent.mkdir(parents=True)
            raw = (json.dumps(fake_inputs()[0]) + '\n').encode(); path.write_bytes(raw)
            out = repo / 'future-output'
            with patch.object(m, 'OLD_SHA', m.sha(raw)), patch('sys.argv',
                    ['prepare.py', '--run', '--repo', str(repo), '--out', str(out)]):
                with self.assertRaisesRegex(ValueError, 'small regular metadata required'): m.main()
            self.assertFalse(out.exists())


if __name__ == '__main__':
    unittest.main()
