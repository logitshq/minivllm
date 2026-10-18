"""Focused CPU-only route replay binding/ownership tests, never GPU evidence.

Capture receipts are made by the actual prepared capture function using its
existing fake Backend. Observer tests use the real frozen Observer/RouteWitness
with fake models; no tokenizer, tensor runtime, server or subprocess is used.
"""
import asyncio
import copy
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from bench.m5_ultra_256gb.route_decode_graph import http_qualification_v2 as m
from bench.m5_ultra_256gb.route_decode_graph import http_capture as capture
from bench.m5_ultra_256gb.route_decode_graph.test_http_capture_cpu import Backend
from bench.m5_ultra_256gb.route_decode_graph import test_serving_qualification_cpu as fake


def ref(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=m.sha(path))


class BindingTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        base = capture.donor.HERE / 'prereg-qualification-v1/plan.json'
        numeric = m.read(base)
        spec = capture.prepare(base, m.sha(base), 8,
            python=numeric['execution']['python'], environment=numeric['execution']['environment'],
            transport=dict(port=18181, tcp_nodelay=True, io_timeout_seconds=300.0),
            deadlines=dict(ready=1, request=1, group=1, close=.05, term=.05, kill=.05,
                           overall=3, resource_interval=.01))
        self.capture_dir = self.root / 'capture'
        async def check(*unused): pass
        result = await capture.capture(spec, self.capture_dir, checkpoint=check, backend=Backend(spec))
        self.assertEqual(result['status'], 'PASS')
        audit_path = self.root / 'synthetic-serving-audit.json'
        m.write(audit_path, dict(schema='E024-serving-independent-audit-v1', status='PASS', qualified=True,
            plan_sha256=m.SERVING_PLAN_SHA, base_plan_sha256=m.BASE_PLAN_SHA,
            audit_source_sha256=m.FIXED[m.HERE/'audit_serving.py'], http_qualified=False,
            performance_qualified=False, coverage=dict(parameter_checks_complete=True,
                parameter_direct_calls=42, identity_refill_enabled=False,
                route_hits={name: {'1023': 1, '1536': 1} for name in ('D-C1', 'D-C8')}),
            ledger=dict(cases=[dict(completed=[None] * n) for n in (8, 16, 13)])))
        self.plan = dict(schema='E024-recorded-route-http-plan-v1', qualifier_sha256=m.sha(m.__file__),
            qualification=spec['base_plan'], concurrency=8,
            serving=dict(plan=ref(m.HERE/'prereg-serving-v1/serving-plan.json'), audit=ref(audit_path)),
            capture={k: ref(self.capture_dir/name) for k, name in m.CAPTURE_NAMES.items()},
            inputs={k: ref(self.capture_dir/name) for k, name in m.INPUT_NAMES.items()},
            sources=m.source_pins())
        self.plan_path = self.root / 'replay-plan.json'

    def validate(self, plan=None):
        m.write(self.plan_path, self.plan if plan is None else plan)
        return m.validate(self.plan_path, m.sha(self.plan_path))

    def update_capture_receipt(self, key, change):
        path = Path(self.plan['capture'][key]['path'])
        value = m.read(path)
        change(value)
        m.write(path, value)
        self.plan['capture'][key] = ref(path)
        complete_path = self.capture_dir/'complete.json'
        complete = m.read(complete_path)
        complete['artifacts'][path.name] = dict(sha256=m.sha(path), bytes=path.stat().st_size)
        m.write(complete_path, complete)
        self.plan['capture']['complete'] = ref(complete_path)

    def test_actual_capture_producer_receipts_bind_exact_plan_and_sources(self):
        plan, numeric, spec, header = self.validate()
        self.assertEqual(plan, self.plan)
        self.assertEqual(numeric['candidate'], dict(compile_decode=True, fuse_decode=True,
                                                  gates_compile=True, norm_reuse=False))
        self.assertTrue(spec['identity']['route_decode_graph']['route_observer'])
        self.assertEqual(header['pid'], Backend(spec).child.pid)
        self.assertEqual(header['phase'], 'HTTP-C8')
        self.assertEqual(spec['jobs'][0]['body']['max_tokens'], 512)

    def test_reap_and_current_absence_receipts_cannot_be_absent_or_claim_false(self):
        self.update_capture_receipt('release', lambda row: row.update(waited_reaped=False))
        with self.assertRaisesRegex(ValueError, 'waited/reaped'):
            self.validate()
        self.update_capture_receipt('release', lambda row: row.update(waited_reaped=True))
        self.update_capture_receipt('absence', lambda row: row.update(returncode=0, stdout='23456'))
        with self.assertRaisesRegex(ValueError, 'absence receipts'):
            self.validate()
        Path(self.plan['capture']['absence']['path']).unlink()
        with self.assertRaisesRegex(ValueError, 'regular file'):
            self.validate()

    def test_wrong_route_file_or_raw_artifact_binding_is_rejected(self):
        original = Path(self.plan['inputs']['route_events']['path'])
        wrong = self.root / 'same-bytes-other-route-events.jsonl'
        wrong.write_bytes(original.read_bytes())
        changed = copy.deepcopy(self.plan)
        changed['inputs']['route_events'] = ref(wrong)
        with self.assertRaisesRegex(ValueError, 'mixed or renamed'):
            self.validate(changed)
        # Update the caller's SHA only: complete.artifacts still binds the
        # originally captured bytes, even if the attacker keeps valid JSON.
        original.write_bytes(original.read_bytes() + b'{}\n')
        self.plan['inputs']['route_events'] = ref(original)
        with self.assertRaisesRegex(ValueError, 'raw artifact binding'):
            self.validate()

    def test_unqualified_serving_changed_source_and_wrong_base_plan_are_rejected(self):
        bad = copy.deepcopy(self.plan)
        bad['sources'][str(m.HERE/'http_server_v2.py')] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'source map'):
            self.validate(bad)
        bad = copy.deepcopy(self.plan)
        bad['qualification']['sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'original model/workload'):
            self.validate(bad)
        audit_path = Path(self.plan['serving']['audit']['path'])
        audit = m.read(audit_path)
        audit['qualified'] = False
        m.write(audit_path, audit)
        self.plan['serving']['audit'] = ref(audit_path)
        with self.assertRaisesRegex(ValueError, 'independent serving PASS'):
            self.validate()

    def test_prepare_result_does_not_grant_any_qualification(self):
        self.validate()
        out = self.root/'prepared'
        with patch.object(m, 'prepare', return_value=({'schema': 'CPU-joined-recording'}, [])), \
             redirect_stdout(io.StringIO()):
            code = m.main(['--prepare-only', '--plan', str(self.plan_path),
                          '--plan-sha256', m.sha(self.plan_path), '--out', str(out)])
        self.assertEqual(code, 0)
        result = m.read(out/'complete.json')
        self.assertEqual(result['status'], 'PREPARED')
        self.assertTrue(result['independent_audit_required'])
        for key in ('numerical_qualified', 'http_qualified', 'performance_qualified',
                    'replay_numerical_checks_passed', 'owner_resources_restore_qualified'):
            self.assertIs(result[key], False, key)


def row(*, step=1, used=1023, hit=1):
    capacity = 1536 if hit else 1024
    return dict(event='route-call', phase='HTTP-C8', step=step, direct_label=None,
        status='SUBMITTED', B=8, used=used, capacities=[[capacity, capacity]] * 16,
        expected_graph=bool(hit), actual_invocations=hit, functions=1 if hit else 0)


def route_reader(rows):
    header = dict(event='route-startup', pid=23456, phase='HTTP-C8', identity={'cpu': True})
    raw = ''.join(json.dumps(x) + '\n' for x in [header, *rows]).encode()
    output = io.StringIO()
    sha = hashlib.sha256(raw).hexdigest()
    routes = m.CapturedRoutes(io.BytesIO(raw), header, output,
        {'path': '/cpu/route-events.jsonl', 'sha256': sha}, 'HTTP-C8')
    return routes, output, sha


class RouteTests(unittest.TestCase):
    def test_ordered_multiple_calls_same_step_and_fallback_are_not_collapsed(self):
        rows = [row(), row(used=1024), row(step=2, hit=0)]
        routes, output, digest = route_reader(rows)
        for value in rows: routes.emit(value)
        result = routes.finish(digest)
        self.assertEqual((result['calls'], result['actual_compiled_invocations'], result['fallbacks']), (3, 2, 1))
        self.assertEqual(result['bounds'], {'1023': 1, '1536': 1})
        self.assertEqual([json.loads(x) for x in output.getvalue().splitlines()][1:], rows)

    def test_first_mismatching_actual_row_is_saved_before_failure(self):
        routes, output, _ = route_reader([row()])
        actual = row(used=1024)
        with self.assertRaisesRegex(ValueError, 'route call differs'):
            routes.emit(actual)
        self.assertEqual(json.loads(output.getvalue().splitlines()[-1]), actual)

    def test_missing_and_extra_calls_both_fail(self):
        routes, output, _ = route_reader([])
        with self.assertRaisesRegex(ValueError, 'missing/truncated'):
            routes.emit(row())
        self.assertEqual(json.loads(output.getvalue().splitlines()[-1]), row())
        routes, _, digest = route_reader([row(), row(step=2)])
        routes.emit(row())
        with self.assertRaisesRegex(ValueError, 'unconsumed/extra'):
            routes.finish(digest)

    def test_matching_only_fallback_does_not_prove_actual_graph_use(self):
        value = row(hit=0)
        routes, _, digest = route_reader([value])
        routes.emit(value)
        with self.assertRaisesRegex(ValueError, 'no actual compiled'):
            routes.finish(digest)


class AdapterTests(unittest.TestCase):
    def inputs(self):
        base, counter = fake.base_module(), fake.old_module()
        installed = fake.FakeIntegration()
        recording = dict(concurrency=8)
        routes = SimpleNamespace(emit=lambda row: None)
        return base, counter, installed, recording, routes

    def test_install_before_original_proxy_and_reverse_cleanup(self):
        base, counter, installed, recording, routes = self.inputs()
        reference, candidate = fake.engine(), fake.engine()
        original_ref, original_candidate = reference.runner.decode_model, candidate.runner.decode_model
        with m.adapted(base, None, installed, fake.q, counter, recording, [], routes) as coverage:
            left, right = base.Observer(reference, object(), 'reference'), base.Observer(candidate, object(), 'candidate')
            handle = installed.handles[0]
            graph = installed.graphs[0]
            self.assertEqual(len(installed.handles), 1)
            self.assertIs(right.original_decode, handle.decoder)
            self.assertIs(candidate.runner.decode_model.target, handle.decoder)
            self.assertIs(handle.decoder.long, right.route_witness)
            right.begin()
            self.assertEqual((right.route_witness.phase, right.route_witness.step), ('HTTP-C8', 1))
            left.restore()
            right.restore()
            self.assertIs(reference.runner.decode_model, original_ref)
            self.assertIs(candidate.runner.decode_model, original_candidate)
            self.assertTrue(handle.closed)
            self.assertEqual(graph._functions, {})
            self.assertIsNone(right.engine)
            self.assertIsNone(right.route_witness)
            right.restore()
            self.assertEqual((coverage['installs'], coverage['restores']), (1, 2))

    def test_constructor_failure_after_proxy_installation_rolls_back(self):
        base, counter, installed, recording, routes = self.inputs()
        original = base.Observer
        class Broken(original):
            def __init__(self, *args):
                super().__init__(*args)
                raise RuntimeError('failure after proxy')
        base.Observer = Broken
        candidate = fake.engine()
        before = dict(vars(candidate.runner))
        with m.adapted(base, None, installed, fake.q, counter, recording, [], routes):
            with self.assertRaisesRegex(RuntimeError, 'failure after proxy'):
                base.Observer(candidate, object(), 'candidate')
        self.assertEqual(set(vars(candidate.runner)), set(before))
        for key, value in before.items(): self.assertIs(vars(candidate.runner)[key], value)
        self.assertNotIn('schedule', vars(candidate.scheduler))
        self.assertTrue(installed.handles[0].closed)
        self.assertEqual(installed.graphs[0]._functions, {})

    def test_original_replay_is_delegated_and_private_hooks_restore_on_error(self):
        base, counter, installed, recording, routes = self.inputs()
        before = base.Observer, base.workloads, base.run_case
        calls = []
        def replay(*args, **kwargs):
            calls.append((args, kwargs))
            return 'frozen-replay-result'
        with self.assertRaisesRegex(RuntimeError, 'scope failure'):
            with m.adapted(base, SimpleNamespace(run_recorded_case=replay), installed,
                           fake.q, counter, recording, [], routes):
                case = base.workloads([])[0]
                result = base.run_case(case, [], [], None, None, None, None, None)
                self.assertEqual(result, 'frozen-replay-result')
                self.assertIs(calls[0][0][0], recording)
                self.assertIs(calls[0][1]['base'], base)
                raise RuntimeError('scope failure')
        self.assertEqual((base.Observer, base.workloads, base.run_case), before)

    def test_default_main_performs_no_validation_loading_or_writes(self):
        with patch.object(m, 'validate', side_effect=AssertionError('validation happened')), \
             patch.object(m, 'load', side_effect=AssertionError('module loaded')), \
             patch.object(Path, 'mkdir', side_effect=AssertionError('directory written')), \
             redirect_stdout(io.StringIO()) as output:
            self.assertEqual(m.main([]), 0)
        self.assertIn('DESCRIPTION_ONLY', output.getvalue())


if __name__ == '__main__':
    unittest.main()
