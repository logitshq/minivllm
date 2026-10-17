"""CPU mechanisms only: fake owned child/wire; no listener, model or GPU.

Written during a device experiment. Do not interpret source existence as a test
run or any numerical/HTTP qualification receipt.
"""
import asyncio
import ast
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from bench.m5_ultra_256gb.route_decode_graph import http_capture as m
from bench.m5_ultra_256gb.pareto_realistic.test_http_capture_v2_cpu import Backend as DonorBackend
from bench.m5_ultra_256gb.pareto_realistic.native_cleanup import GroupCancelled


class Backend(DonorBackend):
    def __init__(self, spec, *, route='valid', **kwargs):
        super().__init__(spec, **kwargs)
        self.route = route

    def spawn(self, command, environment, stdout, stderr):
        self.started = True
        assert command[:-4] == self.spec['command']
        assert environment == self.spec['environment']
        assert command[-4] == '--qualification-trace'
        assert command[-2] == '--route-events'
        assert command[3] == str(m.SERVER)
        Path(command[-3]).write_text(json.dumps(dict(event='startup', identity=self.spec['identity'])) + '\n')
        if self.route != 'missing':
            row = dict(event='route-startup', schema=m.ROUTE_SCHEMA, pid=self.child.pid,
                       phase=f"HTTP-C{self.spec['concurrency']}", identity=self.spec['identity'])
            if self.route == 'wrong-pid': row['pid'] += 1
            if self.route == 'wrong-source':
                row = copy.deepcopy(row)
                row['identity']['server_source_sha256'] = '0' * 64
            Path(command[-1]).write_text(json.dumps(row) + '\n')
        stdout.write(b'CPU MOCK route server\n'); stderr.write(b'CPU MOCK diagnostics\n')
        return self.child


class CaptureTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.out = Path(self.temp.name).resolve() / 'capture'; self.checks = []
        self.plan = m.donor.HERE / 'prereg-qualification-v1/plan.json'
        numeric = json.loads(self.plan.read_text())
        self.inputs = dict(python=numeric['execution']['python'], environment=numeric['execution']['environment'],
            transport=dict(port=18181, tcp_nodelay=True, io_timeout_seconds=300),
            deadlines=dict(ready=1, request=1, group=1, close=.05, term=.05,
                           kill=.05, overall=3, resource_interval=.01))
        self.spec = m.prepare(self.plan, m.sha(self.plan), 8, **self.inputs)

    async def checkpoint(self, label, pid): self.checks.append((label, pid))

    def test_prepare_real_entry_sources_identity_and_original_jobs(self):
        for c, count in ((1, 8), (8, 16)):
            spec = m.prepare(self.plan, m.sha(self.plan), c, **self.inputs)
            m.validate(spec)
            old = m.donor.prepare(self.plan, m.sha(self.plan), c, **self.inputs)
            self.assertEqual(spec['jobs'], old['jobs'])
            self.assertEqual(len(spec['jobs']), count)
            self.assertEqual(spec['command'][3], str(m.SERVER))
            self.assertEqual(spec['command'][-4:], ['--phase', 'qualification', '--route-decode-graph', 'on'])
            self.assertEqual(spec['identity']['configuration'], old['identity']['configuration'])
            self.assertEqual(spec['identity']['server_source_sha256'], m.SERVER_SHA)
            route = spec['identity']['route_decode_graph']
            self.assertTrue(route['enabled'] and route['installed'] and route['route_observer'])
            self.assertFalse(route['actual_compiled_invocations_observed'])
            for source in (m.HERE/'serving_qualification.py', m.HERE.parent/'long_decode_graph/qualify.py',
                           m.HERE/'candidate.py', m.SERVER, m.DONOR):
                self.assertEqual(spec['sources'][str(source)], m.sha(source))

    def test_changed_argv_source_identity_or_body_cannot_reuse_spec(self):
        cases = []
        changed = copy.deepcopy(self.spec); changed['command'][3] = str(m.donor.HERE/'native_server.py'); cases.append(changed)
        changed = copy.deepcopy(self.spec); changed['command'][-1] = 'off'; cases.append(changed)
        changed = copy.deepcopy(self.spec); changed['sources'][str(m.SERVER)] = '0'*64; cases.append(changed)
        changed = copy.deepcopy(self.spec); changed['identity']['route_decode_graph']['route_observer'] = False; cases.append(changed)
        changed = copy.deepcopy(self.spec); changed['jobs'][0]['body']['prompt'] += 'extra'; cases.append(changed)
        for changed in cases:
            with self.subTest(changed=changed['command']):
                with self.assertRaisesRegex(ValueError, 'spec was changed'): m.validate(changed)

    async def test_capture_keeps_separate_sidecars_and_reaps_without_qualification(self):
        backend = Backend(self.spec)
        result = await m.capture(self.spec, self.out, checkpoint=self.checkpoint, backend=backend)
        self.assertEqual(result['status'], 'PASS')
        self.assertTrue(result['server_reaped'] and result['cleanup_complete'] and result['route_events_retained'])
        self.assertFalse(result['numerical_qualified'] or result['performance_qualified'])
        self.assertEqual(backend.child.actions, ['TERM'])
        self.assertEqual(self.checks[-1], ('after-reap', None))
        self.assertIn('trace.jsonl', result['artifacts'])
        self.assertIn('route-events.jsonl', result['artifacts'])
        self.assertEqual(json.loads((self.out/'launch.json').read_text())['command'][-4:],
                         ['--qualification-trace', str(self.out/'trace.jsonl'),
                          '--route-events', str(self.out/'route-events.jsonl')])

    async def test_wrong_readiness_stops_before_http_and_still_reaps(self):
        backend = Backend(self.spec, wrong_identity=True)
        result = await m.capture(self.spec, self.out, checkpoint=self.checkpoint, backend=backend)
        self.assertEqual(result['status'], 'FAIL')
        self.assertFalse(backend.group_started.is_set())
        self.assertEqual(backend.child.actions, ['TERM'])

    async def test_missing_or_misbound_route_sidecar_rejects_after_reap(self):
        for route in ('missing', 'wrong-pid', 'wrong-source'):
            backend = Backend(self.spec, route=route)
            result = await m.capture(self.spec, self.out.with_name(route), checkpoint=self.checkpoint, backend=backend)
            self.assertEqual(result['status'], 'FAIL')
            self.assertTrue(result['server_reaped'])
            self.assertFalse(result.get('route_events_retained', False))
            self.assertEqual(result['errors'][-1]['phase'], 'postflight')

    async def test_guard_failure_preserves_partial_group_and_reaps(self):
        backend = Backend(self.spec, block=True)
        async def guard(label, pid):
            if label == 'capturing': raise RuntimeError('mock resource rejection')
        result = await m.capture(self.spec, self.out, checkpoint=guard, backend=backend)
        self.assertEqual(result['status'], 'FAIL')
        self.assertEqual(json.loads((self.out/'group.json').read_text())['group_error']['type'], 'CancelledError')
        self.assertEqual(backend.child.actions, ['TERM'])

    async def test_repeated_cancel_cannot_skip_owned_kill_wait(self):
        backend = Backend(self.spec, block=True, ignore_term=True)
        closing, allow_close = asyncio.Event(), asyncio.Event()
        async def slow_group(spec, output):
            backend.group_started.set()
            try: await asyncio.Future()
            except asyncio.CancelledError:
                closing.set(); await allow_close.wait()
                partial = backend.group_result(); partial['group_error'] = {'type': 'CancelledError'}
                raise GroupCancelled(partial)
        backend.group = slow_group
        task = asyncio.create_task(m.capture(self.spec, self.out, checkpoint=self.checkpoint, backend=backend))
        await backend.group_started.wait(); task.cancel(); await closing.wait()
        self.assertIsNone(backend.child.poll())
        task.cancel(); await asyncio.sleep(0); task.cancel(); await asyncio.sleep(0)
        allow_close.set(); result = await task
        self.assertEqual(result['status'], 'FAIL')
        self.assertTrue(any(e['phase'] == 'cancel-during-cleanup' for e in result['errors']))
        self.assertEqual(backend.child.actions, ['TERM', 'KILL'])
        self.assertTrue(json.loads((self.out/'release.json').read_text())['waited_reaped'])

    async def test_failed_strict_absence_cannot_pass_or_emit_release(self):
        backend = Backend(self.spec, ignore_term=True, bad_reap=True)
        result = await m.capture(self.spec, self.out, checkpoint=self.checkpoint, backend=backend)
        self.assertEqual(result['status'], 'FAIL')
        self.assertFalse(result['cleanup_complete'])
        self.assertFalse((self.out/'release.json').exists())
        self.assertEqual(self.checks[-1], ('cleanup-incomplete', backend.child.pid))

    async def test_pre_spawn_checkpoint_failure_never_creates_a_child(self):
        backend = Backend(self.spec)
        async def guard(label, pid):
            if label == 'pre-spawn': raise RuntimeError('mock journal not STOPPED')
        result = await m.capture(self.spec, self.out, checkpoint=guard, backend=backend)
        self.assertEqual(result['status'], 'FAIL')
        self.assertFalse(backend.started)
        self.assertFalse((self.out/'child.json').exists())
        self.assertFalse((self.out/'release.json').exists())

    async def test_postflight_pin_failure_does_not_become_success(self):
        backend = Backend(self.spec)
        with mock.patch.object(m, 'validate', side_effect=[None, ValueError('mock source changed')]):
            result = await m.capture(self.spec, self.out, checkpoint=self.checkpoint, backend=backend)
        self.assertEqual(result['status'], 'FAIL')
        self.assertTrue(result['server_reaped'])
        self.assertIn('mock source changed', result['errors'][-1]['message'])


class SourceBoundaryTests(unittest.TestCase):
    def test_lifecycle_control_flow_is_frozen_capture_with_only_local_bindings(self):
        self.assertEqual(m.sha(m.DONOR), m.DONOR_SHA)
        def function(path):
            return next(n for n in ast.parse(path.read_text()).body
                        if isinstance(n, ast.AsyncFunctionDef) and n.name == 'capture')
        old, new = function(m.DONOR), function(Path(m.__file__))
        old_bindings = {n.targets[0].id: n for n in old.body if isinstance(n, ast.Assign)
                        and isinstance(n.targets[0], ast.Name) and n.targets[0].id in ('command', 'result')}
        class BindingOnly(ast.NodeTransformer):
            def visit_Assign(self, node):
                if isinstance(node.targets[0], ast.Name) and node.targets[0].id in old_bindings:
                    return copy.deepcopy(old_bindings[node.targets[0].id])
                if (isinstance(node.targets[0], ast.Subscript) and isinstance(node.targets[0].value, ast.Name)
                    and node.targets[0].value.id == 'result' and isinstance(node.targets[0].slice, ast.Constant)
                    and node.targets[0].slice.value == 'route_events_retained'):
                    return None
                return self.generic_visit(node)
            def visit_Expr(self, node):
                if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == 'route_startup':
                    return None
                return self.generic_visit(node)
        new = BindingOnly().visit(new)
        self.assertEqual(ast.dump(old), ast.dump(new))


if __name__ == '__main__': unittest.main()
