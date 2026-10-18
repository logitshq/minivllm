"""Protocol/ownership mocks, never a device or numerical qualification receipt.

The synthetic audit files exist only inside TemporaryDirectory. Real frozen
client queue/parser/cleanup and server completion serializer run without sockets.
"""
import asyncio
import base64
import copy
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import httpx

from bench.m5_ultra_256gb.pareto_realistic import native_formal as m
from bench.m5_ultra_256gb.pareto_realistic import native_server as server


def pin(path):
    return {'path': str(path.resolve()), 'sha256': m.sha(path)}


def mock_qualification(root, C):
    """Synthetic binding fixture only; no tensor/ledger/device result is created."""
    root.mkdir(); (root / 'base').mkdir()
    q = m.load('http_qualification')
    base = pin(m.HERE / 'prereg-qualification-v1/plan.json')
    for name in ('group.json', 'trace.jsonl', 'recording.json', 'base/complete.json'):
        (root / name).write_text('{}\n')
    replay = dict(schema='E023-recorded-http-plan-v1', qualification=base, concurrency=C,
        transport=dict(tcp_nodelay=True, io_timeout_seconds=300.0),
        inputs={name: pin(root / file) for name, file in (('group', 'group.json'), ('trace', 'trace.jsonl'))},
        sources={str(p): m.sha(p) for p in (*q.FIXED, *q.ADAPTERS)})
    m.write(root / 'plan.json', replay)
    complete = dict(status='PASS', numerical_qualified=True, input_source_pins_rechecked=True,
                    sources=replay['sources'], inputs=replay['inputs'], concurrency=C,
                    total_requests=8 if C == 1 else 16)
    m.write(root / 'complete.json', complete)
    audit = dict(schema='E023-recorded-http-independent-audit-v2', status='PASS', qualified=True,
        concurrency=C, plan_sha256=m.sha(root / 'plan.json'), base_plan_sha256=base['sha256'],
        sources=replay['sources'], raw_inputs=replay['inputs'], performance_qualified=False,
        owner_resources_restore_qualified=False, audit_source_sha256=m.FIXED['audit_http_replay_v2.py'],
        frozen_v1_source_sha256=m.FIXED['audit_http_replay.py'],
        base_auditor_sha256='b6d90413911f2b17e0951a2fcfdb5fe00666aa06d4382b603e29e5dc4659d4f1',
        child_directory=str(root.resolve()), child_complete_sha256=m.sha(root / 'complete.json'),
        base_complete_sha256=m.sha(root / 'base/complete.json'), recording_sha256=m.sha(root / 'recording.json'),
        ledger=dict(requests=8 if C == 1 else 16, decode_batch_histogram={'1' if C == 1 else '8': 1}))
    m.write(root / 'cpu-mock-audit.json', audit)
    binding = dict(audit=pin(root / 'cpu-mock-audit.json'), replay_plan=pin(root / 'plan.json'), child_directory=str(root))
    return base, binding


def prepare(root, C=8):
    base, binding = mock_qualification(root, C)
    numeric = m.read(base['path'])
    return m.prepare(base, C, 1, qualification=binding, environment=numeric['execution']['environment'],
        transport=dict(port=18181, tcp_nodelay=True, io_timeout_seconds=300.0),
        deadlines=dict(ready=.08, request=2, warm=2, formal=2, close=.1,
                       term=15, kill=15, overall=5, resource_interval=.01))


def decode(ids):
    return ''.join('答' + str(token) for token in ids)


def wire(data):
    return ('data: ' + (data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)) + '\n\n').encode()


class Stream(httpx.AsyncByteStream):
    def __init__(self, raw): self.raw = raw
    async def __aiter__(self):
        # Force actual outstanding HTTP requests in the real client task queue.
        await asyncio.sleep(0)
        yield self.raw[:7]
        await asyncio.sleep(0)
        yield self.raw[7:]


async def group(spec, phase, output=None):
    fixtures = {f['prompt_text']: f for f in spec['fixtures']}
    submitted = []
    async def handler(request):
        body = json.loads(request.content); f = fixtures[body['prompt']]
        i = len(submitted); submitted.append(body)
        eos = spec['identity']['eos_ids'][0]
        raw = [10, eos] if i % 3 == 0 else [eos] if i % 3 == 1 else [11] * 512
        rid = f'{i + (1000 if phase == "formal" else 1):032x}'
        terminal = server.completion(server.Job(rid, f['prompt_ids'], 512), raw, set(spec['identity']['eos_ids']))
        text = decode(terminal['benchmark_committed_output_ids'])
        delta = wire(dict(id='cmpl-' + rid, choices=[dict(index=0, text=text, finish_reason=None)])) if text else b''
        return httpx.Response(200, stream=Stream(delta + wire(terminal) + wire('[DONE]')),
                              headers={'content-type': 'text/event-stream'})
    client = m.load('native_cleanup')
    return await client.run_group('http://cpu-mock.invalid', spec['jobs'][phase], spec['concurrency'], phase=phase,
        qualification=m.count_contract(spec, phase), request_timeout_s=2, group_timeout_s=2,
        cleanup_timeout_s=.1, transport=httpx.MockTransport(handler), output_dir=output)


def rewrite_terminal(row, mutate):
    raw = b''.join(base64.b64decode(c['body_bytes_base64']) for c in row['raw_sse_chunks'])
    frames = raw.decode().split('\n\n')[:-1]
    for i, frame in enumerate(frames):
        if 'benchmark_request_id' in frame:
            terminal = json.loads(frame[6:]); mutate(terminal)
            frames[i] = 'data: ' + json.dumps(terminal, ensure_ascii=False)
    raw = ('\n\n'.join(frames) + '\n\n').encode()
    row['raw_sse_chunks'] = [dict(client_perf_counter_s=row['done_perf_counter_s'],
                                body_bytes_base64=base64.b64encode(raw).decode())]
    row['raw_body_bytes'] = len(raw)


class Child:
    pid = 23456
    def __init__(self, ignore_term=False):
        self.returncode = None; self.actions = []; self.ignore_term = ignore_term
    def poll(self): return self.returncode
    def terminate(self):
        self.actions.append('TERM')
        if not self.ignore_term: self.returncode = -15
    def kill(self): self.actions.append('KILL'); self.returncode = -9


class Backend:
    def __init__(self, spec, *, ignore_term=False, bad_reap=False, wrong_identity=False, block=False):
        self.spec = spec; self.child = Child(ignore_term)
        self.bad_reap = bad_reap; self.wrong_identity = wrong_identity; self.block = block
        self.phases = []; self.waited = []; self.group_started = asyncio.Event()
    def decoder(self, spec): return decode
    def spawn(self, command, environment, stdout, stderr):
        assert command == self.spec['command'] and '--qualification-trace' not in command
        assert environment == self.spec['environment']
        stdout.write(b'CPU mock server\n'); stderr.write(b'')
        return self.child
    async def port(self, port, pid=None):
        return dict(returncode=0 if pid else 1, stdout=f'p{pid}\nf4\nn127.0.0.1:{port}\n' if pid else '', stderr='')
    async def absent(self, pid):
        return dict(returncode=0 if self.bad_reap else 1, stdout=str(pid) if self.bad_reap else '', stderr='')
    async def identity(self, pid):
        return dict(returncode=0, stdout=f'{pid} {os.getpid()} CPU MOCK', stderr='')
    async def readiness(self, port):
        identity = copy.deepcopy(self.spec['identity'])
        if self.wrong_identity: identity['step_observer'] = True
        return dict(status=200, body_base64=base64.b64encode(json.dumps(dict(
            data=[{'id': 'native-target-only'}], benchmark_identity=identity)).encode()).decode())
    async def wait(self, child, seconds):
        self.waited.append(seconds)
        if child.poll() is None: raise subprocess.TimeoutExpired('CPU-MOCK', seconds)
        return child.returncode
    async def group(self, spec, phase, output):
        self.phases.append(phase); self.group_started.set()
        if self.block:
            try: await asyncio.Future()
            except asyncio.CancelledError:
                raise m.load('native_cleanup').GroupCancelled(dict(group_error={'type': 'CancelledError'},
                                                                  cleanup={'complete': False}))
        return await group(spec, phase, output)


class Tests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.spec = prepare(self.root / 'mock-audit')
        self.out = self.root / 'trial'; self.checkpoints = []
    async def checkpoint(self, label, pid): self.checkpoints.append((label, pid))

    def test_prepare_and_real_artifact_binding_rejects_changed_sources_or_receipts(self):
        m.validate(self.spec)
        self.assertEqual([len(self.spec['jobs'][p]) for p in ('warm', 'formal')], [8, 64])
        self.assertNotIn('--qualification-trace', self.spec['command'])
        self.assertFalse(self.spec['identity']['step_observer'])
        self.assertFalse(self.spec['identity']['qualification_trace_enabled'])
        self.assertNotIn('expected_raw_ids_by_request', m.count_contract(self.spec, 'formal'))
        for key in ('schema', 'concurrency', 'sources', 'child_directory', 'audit_source_sha256', 'qualified'):
            original = m.read(self.spec['qualification']['audit']['path']); bad = copy.deepcopy(original)
            bad[key] = 1 if key == 'concurrency' else {} if key == 'sources' else False if key == 'qualified' else 'wrong'
            m.write(self.spec['qualification']['audit']['path'], bad)
            changed = copy.deepcopy(self.spec['qualification'])
            changed['audit'] = pin(Path(changed['audit']['path']))
            with self.subTest(key=key), self.assertRaises(ValueError):
                m.bind_qualification(changed, 8, self.spec['base_plan'])
            m.write(self.spec['qualification']['audit']['path'], original)
        bad = copy.deepcopy(self.spec); bad['command'] += ['--qualification-trace', '/tmp/forbidden']
        with self.assertRaises(ValueError): m.validate(bad)
        bad = copy.deepcopy(self.spec); bad['identity']['step_observer'] = True
        with self.assertRaises(ValueError): m.validate(bad)

    async def test_real_client_variable_eos_queue_and_raw_metric_math_c1_c8(self):
        for C in (1, 8):
            spec = self.spec if C == 8 else prepare(self.root / 'mock-c1', C)
            actual = await group(spec, 'formal'); before = copy.deepcopy(actual)
            checked = m.check_group(actual, spec, 'formal', decode)
            self.assertEqual(actual, before)
            self.assertEqual(checked['point']['committed_output_tokens'], 22 + 21 * 512)
            self.assertEqual(checked['raw_requests_checked'], 64)
            self.assertEqual(actual['summary']['max_http_outstanding'], C)
            self.assertEqual(actual['summary']['actual_output_tokens']['min'], 0)
            self.assertEqual(actual['summary']['actual_output_tokens']['max'], 512)

    async def test_full_prompt_text_eos_uuid_clock_and_cleanup_corruption_rejected(self):
        original = await group(self.spec, 'formal')
        changes = {
            'same_length_wrong_prompt': lambda r: rewrite_terminal(r, lambda t: t['benchmark_prompt_ids'].__setitem__(0, 777)),
            'usage': lambda r: rewrite_terminal(r, lambda t: t['usage'].__setitem__('completion_tokens', 999)),
            'committed': lambda r: rewrite_terminal(r, lambda t: t.__setitem__('benchmark_committed_output_ids', [12])),
            'eos': lambda r: rewrite_terminal(r, lambda t: t.__setitem__('benchmark_terminal_eos_id', None)),
            'answer': lambda r: r.__setitem__('answer', 'wrong'),
            'clock': lambda r: r.__setitem__('done_perf_counter_s', r['done_perf_counter_s'] + .1),
            'body': lambda r: r['body'].__setitem__('temperature', 1),
        }
        for key, mutate in changes.items():
            bad = copy.deepcopy(original); mutate(bad['requests'][0])
            with self.subTest(key=key), self.assertRaises(ValueError): m.check_group(bad, self.spec, 'formal', decode)
        bad = copy.deepcopy(original); bad['requests'][1]['raw_sse_chunks'] = copy.deepcopy(bad['requests'][0]['raw_sse_chunks'])
        with self.assertRaises(ValueError): m.check_group(bad, self.spec, 'formal', decode)
        for key in ('complete', 'cancelled_again'):
            bad = copy.deepcopy(original); bad['cleanup'][key] = key != 'complete'
            with self.subTest(key=key), self.assertRaises(ValueError): m.check_group(bad, self.spec, 'formal', decode)

    async def test_success_owns_one_fresh_server_reaps_and_keeps_point_unapproved(self):
        backend = Backend(self.spec)
        result = await m.run(self.spec, self.out, checkpoint=self.checkpoint, backend=backend)
        self.assertEqual(result['status'], 'PASS', result['errors'])
        self.assertEqual(backend.phases, ['warm', 'formal'])
        self.assertEqual(backend.child.actions, ['TERM'])
        self.assertTrue(result['server_reaped'] and result['cleanup_complete'])
        self.assertFalse(result['performance_qualified'])
        self.assertFalse(m.read(self.out / 'point.json')['performance_qualified'])
        self.assertEqual(self.checkpoints[-1], ('after-reap', None))
        self.assertFalse((self.out / 'trace.jsonl').exists())
        self.assertTrue(m.read(self.out / 'release.json')['waited_reaped'])

    async def test_warm_invalid_stops_before_formal_and_preserves_raw(self):
        backend = Backend(self.spec); real = backend.group
        async def wrong(spec, phase, output):
            value = await real(spec, phase, output); value['requests'][0]['answer'] = 'wrong'; return value
        backend.group = wrong
        result = await m.run(self.spec, self.out, checkpoint=self.checkpoint, backend=backend)
        self.assertEqual(result['status'], 'FAIL'); self.assertEqual(backend.phases, ['warm'])
        self.assertTrue(result['server_reaped']); self.assertTrue((self.out / 'warm-group.json').is_file())
        self.assertFalse((self.out / 'point.json').exists())

    async def test_readiness_observer_mismatch_never_sends_request(self):
        backend = Backend(self.spec, wrong_identity=True)
        result = await m.run(self.spec, self.out, checkpoint=self.checkpoint, backend=backend)
        self.assertEqual(result['status'], 'FAIL'); self.assertEqual(backend.phases, [])
        self.assertTrue(result['server_reaped'])

    async def test_resource_failure_preserves_partial_and_strict_reap(self):
        backend = Backend(self.spec, block=True)
        async def guard(label, pid):
            await self.checkpoint(label, pid)
            if label == 'warm': raise RuntimeError('CPU mock pressure failed')
        result = await m.run(self.spec, self.out, checkpoint=guard, backend=backend)
        self.assertEqual(result['status'], 'FAIL'); self.assertTrue(result['server_reaped'])
        self.assertEqual(result['errors'][0]['message'], 'CPU mock pressure failed')
        self.assertEqual(m.read(self.out / 'warm-group.json')['group_error']['type'], 'CancelledError')
        self.assertEqual(backend.phases, ['warm'])

    async def test_repeated_cancellation_during_close_does_not_skip_term_kill_wait(self):
        backend = Backend(self.spec, ignore_term=True)
        closing, allow_close = asyncio.Event(), asyncio.Event()
        async def slow(spec, phase, output):
            backend.group_started.set()
            try: await asyncio.Future()
            except asyncio.CancelledError:
                closing.set(); await allow_close.wait()
                raise m.load('native_cleanup').GroupCancelled(dict(group_error={'type': 'CancelledError'}, cleanup={'complete': False}))
        backend.group = slow
        task = asyncio.create_task(m.run(self.spec, self.out, checkpoint=self.checkpoint, backend=backend))
        await backend.group_started.wait(); task.cancel(); await closing.wait()
        task.cancel(); await asyncio.sleep(0); task.cancel(); await asyncio.sleep(0); allow_close.set()
        result = await task
        self.assertEqual(result['status'], 'FAIL'); self.assertTrue(result['server_reaped'])
        self.assertEqual(backend.child.actions, ['TERM', 'KILL']); self.assertTrue(backend.waited)
        self.assertTrue((self.out / 'warm-group.json').exists())

    async def test_surviving_pid_never_grants_release_or_final_point_approval(self):
        backend = Backend(self.spec, bad_reap=True)
        result = await m.run(self.spec, self.out, checkpoint=self.checkpoint, backend=backend)
        self.assertEqual(result['status'], 'FAIL'); self.assertFalse(result['server_reaped'])
        self.assertFalse((self.out / 'release.json').exists())
        self.assertEqual(self.checkpoints[-1], ('cleanup-incomplete', backend.child.pid))
        self.assertFalse(result['performance_qualified'])

    async def test_repeated_external_cancel_during_server_wait_still_reaps(self):
        backend = Backend(self.spec, ignore_term=True)
        waiting, allow_wait = asyncio.Event(), asyncio.Event()
        real_wait = backend.wait
        async def slow_wait(child, seconds):
            if child.poll() is None:
                waiting.set(); await allow_wait.wait()
            return await real_wait(child, seconds)
        backend.wait = slow_wait
        task = asyncio.create_task(m.run(self.spec, self.out, checkpoint=self.checkpoint, backend=backend))
        await waiting.wait(); self.assertIsNone(backend.child.poll())
        task.cancel(); await asyncio.sleep(0); task.cancel(); await asyncio.sleep(0); allow_wait.set()
        result = await task
        self.assertEqual(result['status'], 'FAIL'); self.assertTrue(result['server_reaped'])
        self.assertEqual(backend.child.actions, ['TERM', 'KILL'])
        self.assertTrue(m.read(self.out / 'release.json')['strict_absence'])
        self.assertTrue(any(e['phase'] == 'cancel-during-cleanup' for e in result['errors']))

    async def test_cancel_during_resource_callback_retains_owned_server_reap(self):
        backend = Backend(self.spec)
        checking = asyncio.Event()
        async def guard(label, pid):
            await self.checkpoint(label, pid)
            if label == 'ready':
                checking.set(); await asyncio.Future()
        task = asyncio.create_task(m.run(self.spec, self.out, checkpoint=guard, backend=backend))
        await checking.wait(); task.cancel()
        result = await task
        self.assertEqual(result['status'], 'FAIL'); self.assertTrue(result['server_reaped'])
        self.assertEqual(backend.phases, [])
        self.assertEqual(backend.child.actions, ['TERM'])
        self.assertEqual(self.checkpoints[-1], ('after-reap', None))


if __name__ == '__main__': unittest.main()
