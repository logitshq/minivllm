"""E047 pure/fake integration checks: no model imports, processes or sockets."""
import asyncio
import base64
import copy
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from bench.m5_ultra_256gb.c8_schedule_diagnostic_v1 import trial as t, server as s, runner as r, execute as e
from bench.m5_ultra_256gb.pareto_realistic.test_http_capture_v2_cpu import Child
from bench.m5_ultra_256gb.pareto_realistic.native_cleanup import GroupCancelled
from bench.m5_ultra_256gb.pareto_realistic import test_http_join_cpu as j

BASE = t.HERE.parents[2] / '.artifacts/m5-ultra-256gb/gdn-graph-combined-formal-prepare-v1/specs/r1-c8-off.json'


def actual_spec(): return t.prepare(dict(path=str(BASE), sha256=t.sha(BASE)))


class Pure(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.spec = actual_spec()

    def test_prepare_pins_exact_bodies_configuration_and_actual_cli(self):
        spec = self.spec; t.validate(spec)
        self.assertEqual(spec['environment']['MLX_MAX_MB_PER_BUFFER'], '50')
        self.assertEqual(len(spec['jobs']), 16)
        self.assertEqual(len({j['request_id'] for j in spec['jobs']}), 16)
        for a, b in zip(spec['jobs'][:8], spec['jobs'][8:]):
            self.assertEqual(a['body'], b['body']); self.assertEqual(a['fixture_id'], b['fixture_id'])
        command = spec['command'][4:] + ['--qualification-trace', '/tmp/e047-cpu-trace', '--route-events', '/tmp/e047-cpu-route']
        args = s.arguments(command)
        self.assertEqual((args.gdn_prework, args.route_decode_graph, args.max_seqs), ('off', 'on', 8))
        self.assertEqual(spec['identity']['combined_candidates']['gdn_layer_installs'], 0)
        self.assertFalse(spec['performance_qualified'] or spec['numerical_qualified'])
        for p in (t.CAPTURE, t.JOIN, t.REPLAY, s.COMBINED, s.ROUTE): self.assertIn(str(p), spec['sources'])

    def test_changed_spec_and_incorrect_cli_are_rejected(self):
        for key, value in [('MLX_MAX_MB_PER_BUFFER', '200'), ('MLX_MAX_OPS_PER_BUFFER', '200')]:
            spec = copy.deepcopy(self.spec); spec['environment'][key] = value
            with self.assertRaises(ValueError): t.validate(spec)
        command = self.spec['command'][4:] + ['--qualification-trace', '/tmp/trace', '--route-events', '/tmp/route']
        command[command.index('--gdn-prework') + 1] = 'on'
        with self.assertRaisesRegex(ValueError, 'GDN OFF'): s.arguments(command)

    def test_observer_passes_actual_graph_handle_without_GDN(self):
        seen = []
        class Observer:
            def __init__(self, engine, handle, *args, **kwargs): seen.append((engine, handle, args))
        graph = object(); engine = object()
        cls = s.observer_type(SimpleNamespace(QualificationObserver=Observer), None)
        cls(engine, SimpleNamespace(graph_handle=graph, gdn_handles=[]), 'donor', 'trace')
        self.assertEqual(seen, [(engine, graph, ('donor', 'trace'))])
        with self.assertRaisesRegex(ValueError, 'graph-only'):
            cls(engine, SimpleNamespace(graph_handle=graph, gdn_handles=[object()]))

    def test_runner_private_namespace_reuses_original_cleanup_and_new_directory(self):
        ns = r.namespace()
        self.assertEqual(ns['HERE'], t.HERE)
        self.assertIs(ns['run'].__code__, r.base.run.__code__)
        self.assertIs(ns['main'].__globals__, ns)
        self.assertIs(ns['physical'], r.base.physical)
        with patch.object(e, 'outer') as outer:
            # Inspect the real inherited source pins without calling the owner.
            self.assertEqual(t.sha(e.BASE), e.BASE_SHA)

    def test_two_wave_join_uses_original_raw_fields_and_full_schedule(self):
        group, events, jobs, fixtures, identity, _ = j.fixture()
        group['summary']['transport_complete'] = True
        group['cleanup']['adapter'] = 'native-close-cancellation-v1'
        other, other_events, other_jobs, _, _, terminals = j.fixture()
        mapping = {j.A: '3' + j.A[1:], j.B: '4' + j.B[1:]}
        other_events = json.loads(json.dumps(other_events).replace(j.A, mapping[j.A]).replace(j.B, mapping[j.B]))
        for row, job, terminal in zip(other['requests'], other_jobs, terminals):
            job['request_id'] += '-wave2'; row['request_id'] = job['request_id']
            terminal['benchmark_request_id'] = mapping[terminal['benchmark_request_id']]
            terminal['id'] = 'cmpl-' + terminal['benchmark_request_id']; j.replace_wire(row, terminal)
            row['start_perf_counter_s'] += 10; row['done_perf_counter_s'] += 10
            for chunk in row['raw_sse_chunks']: chunk['client_perf_counter_s'] += 10
        other['summary']['transport_complete'] = True
        other['cleanup']['adapter'] = 'native-close-cancellation-v1'
        other_events[2]['finished_ids'] = [j.B]
        original = copy.deepcopy([group, other])
        view = t.join_view([group, other], [{'phase': 'warm'}, {'phase': 'screen'}])
        result = t.module(t.JOIN).join(view, j.lines(events + other_events[1:]), jobs + other_jobs,
                                      fixtures, identity, j.decode)
        self.assertEqual([group, other], original)
        self.assertEqual(len(result['jobs']), 4)
        self.assertEqual(result['jobs'][mapping[j.B]]['raw_ids'], [20, 21, 99])
        self.assertNotIn('barrier_release_perf_counter_s', view)
        self.assertFalse(result['performance_qualified'])


class Fake:
    def __init__(self, spec, *, bad_identity=False, fail_first=False, block_second=False):
        self.spec = spec; self.child = Child(); self.calls = []; self.waited = []
        self.bad_identity = bad_identity; self.fail_first = fail_first; self.block_second = block_second
        self.second_started = asyncio.Event()
    def spawn(self, command, environment, stdout, stderr):
        assert command[:-4] == self.spec['command'] and environment == self.spec['environment']
        Path(command[-3]).write_text(json.dumps(dict(event='startup', identity=self.spec['identity'])) + '\n')
        Path(command[-1]).write_text(json.dumps(dict(event='route-startup', schema='E024-route-http-events-v1',
            pid=self.child.pid, phase='HTTP-C8', identity=self.spec['identity'])) + '\n')
        return self.child
    async def port(self, port, pid=None):
        return dict(returncode=0 if pid else 1, stdout=f'p{pid}\nn127.0.0.1:{port}\n' if pid else '', stderr='')
    async def absent(self, pid): return dict(returncode=1, stdout='', stderr='')
    async def identity(self, pid): return dict(returncode=0, stdout=f'{pid} {os.getpid()} CPU', stderr='')
    async def readiness(self, port):
        identity = copy.deepcopy(self.spec['identity'])
        if self.bad_identity: identity['configuration']['max_seqs'] = 7
        return dict(status=200, body_base64=base64.b64encode(json.dumps(
            dict(data=[dict(id='native-target-only')], benchmark_identity=identity)).encode()).decode())
    async def wait(self, child, seconds): self.waited.append(child.pid); return child.returncode
    async def group(self, spec, output):
        self.calls.append(copy.deepcopy(spec['jobs']))
        group = dict(schema='E023-http-group-v1', phase='qualification', qualification=None, concurrency=8,
            requests=copy.deepcopy(spec['jobs']), group_error=None,
            cleanup=dict(complete=True, adapter='native-close-cancellation-v1', deadline_exceeded=False,
                pending_tasks=[], errors=[], cancelled_again=False),
            summary=dict(transport_complete=True, formal_point_eligible=False))
        if self.fail_first:
            group['group_error'] = dict(type='CPU_failure'); group['summary']['transport_complete'] = False
        if len(self.calls) == 2:
            self.second_started.set()
            if self.block_second:
                try: await asyncio.Future()
                except asyncio.CancelledError:
                    group['group_error'] = dict(type='CancelledError')
                    group['summary']['transport_complete'] = False
                    raise GroupCancelled(group)
        return group


class Capture(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls): cls.spec = actual_spec()
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.out = Path(self.temp.name) / 'child'; self.checks = []
    async def checkpoint(self, label, pid): self.checks.append((label, pid))
    async def test_one_child_two_drained_groups_and_exact_retain(self):
        fake = Fake(self.spec); result = await t.run(self.spec, self.out, checkpoint=self.checkpoint, backend=fake)
        self.assertEqual(result['status'], 'PASS', result['errors'])
        self.assertEqual(result['label'], self.spec['label'])
        self.assertEqual(fake.calls, [self.spec['jobs'][:8], self.spec['jobs'][8:]])
        self.assertEqual(fake.child.actions, ['TERM']); self.assertTrue(fake.waited)
        self.assertEqual(self.checks[-1], ('after-reap', None))
        self.assertTrue(result['cleanup_complete'] and result['server_reaped'])
        view = t.read(self.out / 'group.json')
        self.assertTrue(view['derived_view']); self.assertEqual(view['requests'], self.spec['jobs'])
        for ref in view['source_groups']: self.assertEqual(t.sha(ref['path']), ref['sha256'])
        self.assertFalse(result['numerical_qualified'] or result['performance_qualified'])
        planned = Path(self.temp.name) / 'spec.json'; t.write(planned, self.spec)
        self.assertEqual(t.sha(planned), t.sha(self.out / 'spec.json'))
    async def test_first_failed_group_does_not_launch_second_and_is_preserved(self):
        fake = Fake(self.spec, fail_first=True)
        result = await t.run(self.spec, self.out, checkpoint=self.checkpoint, backend=fake)
        self.assertEqual(result['status'], 'FAIL'); self.assertEqual(len(fake.calls), 1)
        self.assertTrue((self.out / 'client/warm/group.json').is_file())
        self.assertFalse((self.out / 'client/screen').exists())
        self.assertTrue(result['server_reaped'])
    async def test_second_wave_cancel_preserves_both_raw_groups_and_reaps(self):
        fake = Fake(self.spec, block_second=True)
        task = asyncio.create_task(t.run(self.spec, self.out, checkpoint=self.checkpoint, backend=fake))
        await fake.second_started.wait(); task.cancel(); result = await task
        self.assertEqual(result['status'], 'FAIL'); self.assertTrue(result['server_reaped'])
        self.assertTrue((self.out / 'client/screen/group.json').is_file())
        self.assertEqual(len(t.read(self.out / 'group.json')['requests']), 16)
    async def test_wrong_readiness_refuses_requests_and_reaps(self):
        fake = Fake(self.spec, bad_identity=True)
        result = await t.run(self.spec, self.out, checkpoint=self.checkpoint, backend=fake)
        self.assertEqual(result['status'], 'FAIL'); self.assertEqual(fake.calls, [])
        self.assertTrue(result['server_reaped'])


if __name__ == '__main__': unittest.main()
