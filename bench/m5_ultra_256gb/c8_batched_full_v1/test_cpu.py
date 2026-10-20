"""Small stdlib contracts; no backend, model, service, or numerical qualification."""
import copy
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

from bench.m5_ultra_256gb.c8_batched_full_v1 import qualify as q


def recording():
    jobs = {f'r{i}': dict(raw_ids=list(range(512)), committed_ids=list(range(512)), cap=512) for i in range(16)}
    return dict(schema='E023-recorded-http-v1', concurrency=8, jobs=jobs,
                steps=[dict(step=i) for i in range(1, 1031)], histogram={'8': 1030})


def composition():
    return dict(installs=1, restores=1, handle_count=48, records=[dict(status='RESTORED', handles_closed=48,
        native_decode_restored=True, caches_cleared=True,
        layers=[dict(layer=i, final_counts=dict(candidate=5, fallback=0)) for i in range(48)])])


def predecessor():
    return dict(schema='c8-batched-gdn-graph-prefix-result-v1', status='DIAGNOSTIC_PREFIX_COMPLETE',
        plan_sha256='abc', source_input_pins_rechecked=True, completion_state='INCOMPLETE_BY_DESIGN',
        performance_qualified=False, full_history_qualified=False, base_result=dict(status='PASS', returncode=0),
        inherited_result=dict(status='DIAGNOSTIC_PREFIX_COMPLETE', returncode=0), prefix=dict(steps=49, completed=[]),
        coverage=dict(installs=1, restores=2), composition=composition())


def inner(data):
    complete = [dict(rid=rid, raw_ids=j['raw_ids'], committed_ids=j['committed_ids']) for rid, j in data['jobs'].items()]
    return dict(status='PASS', source_model_pins_rechecked=True, total_requests=16,
        cases=[dict(status='PASS', name='HTTP-C8', steps=1030, completed=complete,
                    raw_output_count=8192, committed_output_count=8192, decode_batch_histogram=data['histogram'])])


class Contract(unittest.TestCase):
    def test_recording_is_full_list_and_original_caps(self):
        data = recording(); q.recording_ok(data)
        for mutate in (lambda d: d['steps'].pop(), lambda d: d['jobs'].pop('r0'),
                       lambda d: d['jobs']['r0'].update(cap=49), lambda d: d['steps'][49].update(step=49)):
            bad = copy.deepcopy(data); mutate(bad)
            with self.assertRaises(ValueError): q.recording_ok(bad)
    def test_failed_or_mismatched_predecessor_never_qualifies(self):
        top = predecessor(); q.predecessor_ok(top, 'abc')
        for field, value in [('status','FAIL'), ('plan_sha256','other'), ('source_input_pins_rechecked',False),
                             ('full_history_qualified',True)]:
            bad = copy.deepcopy(top); bad[field] = value
            with self.assertRaises(ValueError): q.predecessor_ok(bad, 'abc')
    def test_missing_layer_fallback_and_incomplete_restore_reject(self):
        good = composition(); q.composition_ok(good)
        for mutate in (lambda d: d.update(restores=0), lambda d: d['records'][0]['layers'].pop(),
                       lambda d: d['records'][0]['layers'][0]['final_counts'].update(candidate=0),
                       lambda d: d['records'][0]['layers'][0]['final_counts'].update(fallback=1)):
            bad = copy.deepcopy(good); mutate(bad)
            with self.assertRaises(ValueError): q.composition_ok(bad)
    def test_full_output_counts_ids_and_histogram_must_match(self):
        data = recording(); actual = inner(data); coverage = dict(installs=1, restores=2)
        self.assertIs(q.full_result_ok(actual, data, coverage), actual['cases'][0])
        for mutate in (lambda d: d.update(total_requests=0), lambda d: d['cases'][0].update(steps=49),
                       lambda d: d['cases'][0]['completed'].pop(),
                       lambda d: d['cases'][0]['completed'][0]['raw_ids'].__setitem__(1, -1),
                       lambda d: d['cases'][0].update(decode_batch_histogram={'8': 49})):
            bad = copy.deepcopy(actual); mutate(bad)
            with self.assertRaises(ValueError): q.full_result_ok(bad, data, coverage)
    def test_actual_frozen_replay_seam_retains_recording_and_final_checks(self):
        route = q.load(q.B/'route_decode_graph/http_qualification_v2.py',
            'd3e2a4f76b7c384f6c6c986d0838b68227618502b0552e5cdf74f149029e39e7', '_full_test_route')
        data = recording(); before = copy.deepcopy(data); seen = []
        base = NS(Observer=object, workloads='original workloads', run_case='original case')
        def recorded(actual, *args, **kwargs):
            self.assertIs(actual, data); self.assertIs(type(actual['steps']), list)
            self.assertEqual(len(actual['steps']), 1030); self.assertIs(kwargs['base'], base)
            seen.append('original-full-replay'); return 'full-result'
        def main(argv):
            self.assertEqual(base.workloads([]), [dict(name='D-C8', concurrency=8)])
            self.assertEqual(base.run_case(dict(name='D-C8', concurrency=8), *([None]*7)), 'full-result')
            return 0
        base.main = main
        bridge = NS(integration_module=lambda module: seen.append(module), close_private_hook=lambda: seen.append('closed'))
        result = q.run_replay(route, base, NS(run_recorded_case=recorded), bridge, None, None,
            data, [], None, dict(path='/unused/plan.json', sha256='unused'), Path('/unused/output'))
        self.assertEqual(result[0], 0); self.assertEqual(seen, [route, 'original-full-replay', 'closed'])
        self.assertEqual(data, before); self.assertEqual(base.run_case, 'original case')
    def test_original_replay_exception_propagates_and_hook_closes(self):
        from contextlib import contextmanager
        seen = []
        @contextmanager
        def adapted(*args):
            try: yield {}
            finally: seen.append('adapted-exit')
        def fail(argv): raise RuntimeError('original replay failure')
        bridge = NS(integration_module=lambda module: None, close_private_hook=lambda: seen.append('hook-closed'))
        with self.assertRaisesRegex(RuntimeError, 'original replay failure'):
            q.run_replay(NS(adapted=adapted), NS(main=fail), None, bridge, None, None,
                         recording(), [], None, dict(path='p', sha256='s'), Path('unused'))
        self.assertEqual(seen, ['adapted-exit', 'hook-closed'])
    def test_original_route_finish_rejects_partial_or_extra_stream(self):
        import io,hashlib,json
        route = q.load(q.B/'route_decode_graph/http_qualification_v2.py',
            'd3e2a4f76b7c384f6c6c986d0838b68227618502b0552e5cdf74f149029e39e7', '_full_test_finish')
        header = dict(event='route-startup')
        row = dict(event='route-call', phase='HTTP-C8', step=1, direct_label=None,
                   status='SUBMITTED', actual_invocations=1, used=1024)
        raw = b''.join((json.dumps(x)+'\n').encode() for x in (header,row))
        digest = hashlib.sha256(raw).hexdigest()
        reader = route.CapturedRoutes(io.BytesIO(raw), header, io.StringIO(), {'sha256':digest}, 'HTTP-C8')
        with self.assertRaisesRegex(ValueError, 'unconsumed'): reader.finish(digest)
        reader = route.CapturedRoutes(io.BytesIO(raw), header, io.StringIO(), {'sha256':digest}, 'HTTP-C8')
        reader.emit(row); self.assertEqual(reader.finish(digest)['calls'], 1)
        with self.assertRaisesRegex(ValueError, 'changed'): reader.finish('wrong')
    def test_frozen_bridge_and_prefix_sources_match_explicit_pins(self):
        self.assertEqual(q.sha(q.PREFIX), q.PREFIX_SHA)
        self.assertEqual(q.sha(q.BRIDGE), q.BRIDGE_SHA)
        bridge = q.load(q.BRIDGE, q.BRIDGE_SHA, '_full_test_bridge')
        self.assertIsNone(bridge.Bridge(None, lambda row: None).install(None))


if __name__ == '__main__': unittest.main()
