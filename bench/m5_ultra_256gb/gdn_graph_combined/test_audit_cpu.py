"""Independent evidence corruption tests; no MLX/model/device imports."""
import copy
import io
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
import tempfile
import unittest

from bench.m5_ultra_256gb.gdn_graph_combined import audit as a
from bench.m5_ultra_256gb.gdn_graph_combined import test_qualify_cpu as f


def encoded(rows): return io.BytesIO(b''.join((json.dumps(x) + '\n').encode() for x in rows))


def actual_fake_rows(sequence=((1, 1023), (1, 1023), (8, 1023), (8, 1023), (1, 1024))):
    graph, proxy, witness, stats, events = f.proxy_fixture()
    witness.phase = 'D-C1'
    for step, (B, used) in enumerate(sequence, 1):
        witness.step = step
        witness(SimpleNamespace(shape=(B,)), f.f.context(B, used))
    return events


class Tests(unittest.TestCase):
    def test_real_producer_witness_rows_independently_reparse_trace_reuse_and_batch(self):
        donor = a.donor_module(); rows = actual_fake_rows(); routes = a.route_type(donor)(encoded(rows))
        for step, (B, used) in enumerate(((1, 1023), (1, 1023), (8, 1023), (8, 1023), (1, 1024)), 1):
            routes.route('D-C1', step, None, B, used, 1536)
        proof = routes.finish()
        self.assertEqual((routes.combined['compiled-tracing'], routes.combined['compiled-reuse']), (3, 2))
        self.assertEqual(proof['lines'], 10); self.assertEqual(proof['route_hits']['D-C1'], {'1023': 4, '1536': 1})

    def test_actual_call_geometry_identity_kind_and_invocation_corruption_reject(self):
        original = actual_fake_rows(((1, 1023),))
        mutations = [('installation', 1), ('phase', 'D-C8'), ('step', 2), ('B', 8), ('used', 1024),
            ('kind', 'compiled-reuse'), ('actual_invocations', 0), ('actual_invocations', True), ('status', 'FAILED')]
        for field, value in mutations:
            rows = copy.deepcopy(original); rows[0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                a.route_type(a.donor_module())(encoded(rows)).route('D-C1', 1, None, 1, 1023, 1536)

    def test_counter_holes_mixed_layers_wrong_branch_and_bools_reject(self):
        original = actual_fake_rows(((1, 1023),))
        def hole(row): row['before'][0]['candidate'] = 3
        def mixed(row): row['after'][47]['candidate'] = 0
        def wrong(row):
            for x in row['after']: x.update(candidate=0, fallback=1)
            row.update(per_layer_candidate_delta=0, per_layer_fallback_delta=1)
        def boolean(row): row['after'][0]['candidate'] = True
        for change in (hole, mixed, wrong, boolean):
            rows = copy.deepcopy(original); change(rows[0])
            with self.assertRaises(ValueError): a.route_type(a.donor_module())(encoded(rows)).route('D-C1', 1, None, 1, 1023, 1536)

    def test_reuse_without_first_trace_or_with_other_batch_trace_rejects(self):
        rows = actual_fake_rows(((1, 1023), (1, 1023)))
        row = copy.deepcopy(rows[2]); route = copy.deepcopy(rows[3])
        # Rebase all counters to zero and preserve a superficially valid reuse.
        for field in ('before', 'after'): row[field] = [dict(candidate=0, fallback=0) for _ in range(48)]
        row['step'] = route['step'] = 1
        with self.assertRaisesRegex(ValueError, 'prior actual'):
            a.route_type(a.donor_module())(encoded([row, route])).route('D-C1', 1, None, 1, 1023, 1536)
        routes = a.route_type(a.donor_module())(encoded([row, route])); routes.seen.add((0, 1023, 8))
        with self.assertRaisesRegex(ValueError, 'prior actual'): routes.route('D-C1', 1, None, 1, 1023, 1536)

    def test_missing_extra_truncated_and_original_route_corruption_reject(self):
        rows = actual_fake_rows(((1, 1023),))
        for altered in (rows[:1], rows[1:], [rows[0], dict(rows[1], capacities=[[1535, 1535]] * 16)]):
            with self.assertRaises(ValueError): a.route_type(a.donor_module())(encoded(altered)).route('D-C1', 1, None, 1, 1023, 1536)
        routes = a.route_type(a.donor_module())(encoded(rows + rows))
        routes.route('D-C1', 1, None, 1, 1023, 1536)
        with self.assertRaises(ValueError): routes.finish()
        with self.assertRaises(ValueError):
            a.route_type(a.donor_module())(io.BytesIO(encoded(rows).getvalue()[:-1])).route('D-C1', 1, None, 1, 1023, 1536)

    def test_existing_parameter_array_bit_finite_and_dtype_gates_are_unchanged(self):
        donor = a.donor_module(); label = 'test-logits'
        expected = [donor.array_row(label, 'root', [1, 248320])]
        for field, value in (('bit_equal', False), ('candidate_finite', False), ('dtype', 'mlx.core.float32')):
            bad = copy.deepcopy(expected); bad[0][field] = value
            row = dict(event='parameter-comparison', label=label, arrays=bad)
            with self.assertRaises(ValueError): a.route_type(donor)(encoded([row])).compare(label, expected)
        self.assertIs(a.route_type(donor).parameters, donor.RouteEvents.parameters)

    def test_stream_injection_preserves_original_globals_and_honors_class(self):
        module = ModuleType('_cpu_original_stream')
        exec('class CaseAudit: pass\ndef stream_ledger(): return CaseAudit\n', vars(module))
        original = module.CaseAudit
        replacement = type('Replacement', (), {})
        self.assertIs(a.stream_with(module, replacement), replacement)
        self.assertIs(module.CaseAudit, original)

    def test_beyond_graph_domain_counter_updates_are_per_instance_and_batch(self):
        routes = a.route_type(a.donor_module())(encoded([]))
        routes.external_decode('D-C1', 1); routes.external_decode('D-C8', 8)
        routes.external_decode('lifecycle', 1)
        self.assertEqual(routes.counters[0], [dict(candidate=1, fallback=0)] * 48)
        self.assertEqual(routes.counters[1], [dict(candidate=1, fallback=1)] * 48)

    def test_missing_terminal_never_opens_ledger_or_qualifies(self):
        with tempfile.TemporaryDirectory() as name:
            result = a.audit(Path(name), '0' * 64)
        self.assertEqual(result['status'], 'INCOMPLETE')
        self.assertFalse(result['qualified']); self.assertFalse(result['ledger_opened'])
        self.assertFalse(result['http_qualified']); self.assertFalse(result['performance_qualified'])


if __name__ == '__main__': unittest.main()
