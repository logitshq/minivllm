"""Capture lifecycle mocks only: no actual process, HTTP, socket or model run."""
import asyncio
import base64
import copy
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from bench.m5_ultra_256gb.pareto_realistic import http_capture as m
from bench.m5_ultra_256gb.pareto_realistic.native_cleanup import GroupCancelled


class Child:
    pid = 23456
    returncode = None
    def __init__(self, ignore_term=False):
        self.ignore_term = ignore_term; self.actions = []
    def poll(self): return self.returncode
    def terminate(self):
        self.actions.append('TERM')
        if not self.ignore_term: self.returncode = -15
    def kill(self):
        self.actions.append('KILL'); self.returncode = -9


class Backend:
    def __init__(self, spec, *, wrong_identity=False, ignore_term=False, block=False, bad_reap=False):
        self.spec, self.child = spec, Child(ignore_term)
        self.wrong_identity, self.block, self.bad_reap = wrong_identity, block, bad_reap
        self.started, self.waited = False, []
        self.group_started = asyncio.Event()
    def spawn(self, command, environment, stdout, stderr):
        self.started = True
        assert command[:-2] == self.spec['command']
        assert environment == self.spec['environment']
        Path(command[-1]).write_text(json.dumps({'event': 'startup', 'identity': self.spec['identity']}) + '\n')
        stdout.write(b'CPU MOCK server\n'); stderr.write(b'CPU MOCK stderr\n')
        return self.child
    async def port(self, port, pid=None):
        return dict(command=['CPU-MOCK-lsof'], returncode=0 if pid else 1,
                    stdout=f'p{pid}\nn127.0.0.1:{port}\n' if pid else '', stderr='')
    async def absent(self, pid):
        return dict(command=['CPU-MOCK-ps'], returncode=0 if self.bad_reap else 1,
                    stdout=str(pid) if self.bad_reap else '', stderr='')
    async def identity(self, pid):
        return dict(command=['CPU-MOCK-identity'], returncode=0,
                    stdout=f'{pid} {os.getpid()} Mon Oct 19 00:00:00 2026 python', stderr='')
    async def readiness(self, port):
        identity = copy.deepcopy(self.spec['identity'])
        if self.wrong_identity: identity['configuration']['max_seqs'] = 99
        return dict(status=200, headers={}, body_base64=base64.b64encode(json.dumps(
            {'data': [{'id': 'native-target-only'}], 'benchmark_identity': identity}).encode()).decode())
    async def wait(self, child, seconds):
        self.waited.append(seconds)
        if child.poll() is None: raise subprocess.TimeoutExpired('CPU-MOCK', seconds)
        return child.returncode
    def group_result(self):
        return dict(schema='E023-http-group-v1', phase='qualification', qualification=None,
                    concurrency=self.spec['concurrency'], requests=copy.deepcopy(self.spec['jobs']),
                    group_error=None, cleanup=dict(complete=True, adapter='native-close-cancellation-v1'),
                    summary=dict(transport_complete=True, formal_point_eligible=False))
    async def group(self, spec, output):
        self.group_started.set()
        if self.block:
            try: await asyncio.Future()
            except asyncio.CancelledError:
                partial = self.group_result()
                partial['group_error'] = {'type': 'CancelledError'}
                partial['summary']['transport_complete'] = False
                raise GroupCancelled(partial)
        return self.group_result()


class CaptureTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.out = Path(self.temp.name) / 'capture'
        plan = m.HERE / 'prereg-qualification-v1/plan.json'
        numeric = json.loads(plan.read_text())
        self.spec = m.prepare(plan, m.sha(plan), 8, python=numeric['execution']['python'],
            environment=numeric['execution']['environment'],
            transport=dict(port=18181, tcp_nodelay=True, io_timeout_seconds=300),
            deadlines=dict(ready=1, request=1, group=1, close=.05, term=.05,
                           kill=.05, overall=3, resource_interval=.01))
        self.checks = []
    async def checkpoint(self, label, pid):
        self.checks.append((label, pid))

    def test_prepare_exact_original_cli_environment_and_fixed_jobs(self):
        self.assertEqual(len(self.spec['jobs']), 16)
        cmd = self.spec['command']
        self.assertEqual(cmd[1:3], ['-I', '-B'])
        self.assertEqual(cmd[cmd.index('--max-seqs') + 1], '8')
        self.assertEqual(cmd[cmd.index('--max-batched-tokens') + 1], '2048')
        self.assertEqual(cmd[cmd.index('--tcp-nodelay') + 1], 'on')
        self.assertEqual(cmd[cmd.index('--io-timeout-seconds') + 1], '300.0')
        self.assertIs(type(self.spec['identity']['configuration']['io_timeout_seconds']), float)
        wrong = copy.deepcopy(self.spec); wrong['command'] += ['--ignore-eos']
        with self.assertRaisesRegex(ValueError, 'spec was changed'): m.validate(wrong)
        wrong = copy.deepcopy(self.spec); wrong['environment']['PYTHONPATH'] = '/unexpected'
        with self.assertRaisesRegex(ValueError, 'environment'): m.validate(wrong)

    async def test_capture_reaps_before_return_and_never_grants_qualification(self):
        backend = Backend(self.spec)
        result = await m.capture(self.spec, self.out, checkpoint=self.checkpoint, backend=backend)
        self.assertEqual(result['status'], 'PASS')
        self.assertTrue(result['cleanup_complete'])
        self.assertEqual(backend.child.actions, ['TERM'])
        self.assertTrue(backend.waited)
        self.assertEqual(self.checks[-1], ('after-reap', None))
        self.assertFalse(result['numerical_qualified'] or result['performance_qualified'])
        self.assertTrue(json.loads((self.out / 'release.json').read_text())['strict_absence'])
        for name in ('trace.jsonl', 'group.json', 'ready.json', 'server.stdout', 'server.stderr'):
            self.assertIn(name, result['artifacts'])

    async def test_wrong_actual_identity_stops_before_requests_and_reaps(self):
        backend = Backend(self.spec, wrong_identity=True)
        result = await m.capture(self.spec, self.out, checkpoint=self.checkpoint, backend=backend)
        self.assertEqual(result['status'], 'FAIL')
        self.assertFalse(backend.group_started.is_set())
        self.assertEqual(backend.child.actions, ['TERM'])
        self.assertIn('identity', result['errors'][0]['message'])

    async def test_resource_failure_cancels_and_preserves_group_partial_then_reaps(self):
        backend = Backend(self.spec, block=True)
        async def guard(label, pid):
            await self.checkpoint(label, pid)
            if label == 'capturing': raise RuntimeError('CPU mocked pressure failure')
        result = await m.capture(self.spec, self.out, checkpoint=guard, backend=backend)
        self.assertEqual(result['status'], 'FAIL')
        self.assertEqual(result['errors'][0]['message'], 'CPU mocked pressure failure')
        self.assertEqual(json.loads((self.out / 'group.json').read_text())['group_error']['type'], 'CancelledError')
        self.assertEqual(backend.child.actions, ['TERM'])

    async def test_external_cancellation_keeps_partial_and_reaps(self):
        backend = Backend(self.spec, block=True)
        task = asyncio.create_task(m.capture(self.spec, self.out, checkpoint=self.checkpoint, backend=backend))
        await backend.group_started.wait(); task.cancel()
        result = await task
        self.assertEqual(result['status'], 'FAIL')
        self.assertEqual(result['errors'][0]['type'], 'CancelledError')
        self.assertEqual(backend.child.actions, ['TERM'])
        self.assertTrue((self.out / 'group.json').exists())

    async def test_repeated_cancel_during_response_cleanup_cannot_skip_server_reap(self):
        backend = Backend(self.spec, block=True, ignore_term=True)
        closing, allow_close = asyncio.Event(), asyncio.Event()
        async def slow_group(spec, output):
            backend.group_started.set()
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                closing.set()
                await allow_close.wait()
                partial = backend.group_result()
                partial['group_error'] = {'type': 'CancelledError'}
                raise GroupCancelled(partial)
        backend.group = slow_group
        task = asyncio.create_task(m.capture(self.spec, self.out, checkpoint=self.checkpoint, backend=backend))
        await backend.group_started.wait(); task.cancel()
        await closing.wait()
        self.assertIsNone(backend.child.poll())  # Still owns a live server during close.
        task.cancel(); await asyncio.sleep(0)
        task.cancel(); await asyncio.sleep(0)
        allow_close.set()
        result = await task
        self.assertEqual(result['status'], 'FAIL')
        self.assertTrue(any(e['phase'] == 'cancel-during-cleanup' for e in result['errors']))
        self.assertEqual(backend.child.actions, ['TERM', 'KILL'])
        self.assertTrue(json.loads((self.out / 'release.json').read_text())['waited_reaped'])
        self.assertTrue((self.out / 'group.json').exists())

    async def test_term_timeout_kills_and_false_absence_never_passes(self):
        backend = Backend(self.spec, ignore_term=True, bad_reap=True)
        result = await m.capture(self.spec, self.out, checkpoint=self.checkpoint, backend=backend)
        self.assertEqual(result['status'], 'FAIL')
        self.assertEqual(backend.child.actions, ['TERM', 'KILL'])
        self.assertFalse(result['cleanup_complete'])
        self.assertFalse((self.out / 'release.json').exists())
        self.assertEqual(self.checks[-1], ('cleanup-incomplete', backend.child.pid))


if __name__ == '__main__': unittest.main()
