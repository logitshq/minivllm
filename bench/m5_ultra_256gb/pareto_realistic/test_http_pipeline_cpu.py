"""CPU wiring test: actual Scheduler/BatchWorker -> raw SSE -> join -> replay.

HTTPX MockTransport opens no socket. The existing one-layer NumPy Runner emits
its own tokens; neither schedules nor sampled outputs are supplied by this
fixture. Short synthetic prompts/caps are deliberately NOT the real D512 model
qualification, a device test, a transport performance measurement, or E028.
"""
import asyncio
import copy
import io
import json
import queue
import tempfile
from types import SimpleNamespace as NS
import unittest

import httpx

from bench.m5_ultra_256gb.pareto_realistic import client, http_join, http_replay
from bench.m5_ultra_256gb.pareto_realistic import native_server as native
from bench.m5_ultra_256gb.pareto_realistic import qualify_native as base
from bench.m5_ultra_256gb.pareto_realistic.test_qualify_cpu import CPU, Engine, SamplingParams
from tutorial_code.miniengine.engine.scheduler import Scheduler


def decode(ids):
    return ''.join('答' + str(token) for token in ids)


def wire(value):
    return ('data: ' + (value if isinstance(value, str) else
            json.dumps(value, ensure_ascii=False)) + '\n\n').encode()


def engine_for(concurrency):
    engine = Engine(concurrency)
    # A small genuine token budget exercises chunked prefill and mixed steps.
    engine.scheduler = Scheduler(concurrency, 2048, 16, engine.eos_ids)
    return engine


async def capture(concurrency):
    fixtures, jobs = {}, []
    for i, length in enumerate((2, 5, 3, 8, 4, 6, 7, 9)):
        f = dict(fixture_id=f'CPU-{i}', prompt_text=f'CPU-only prompt {i}',
                 prompt_ids=[i + 1] * length, max_output_tokens=4,
                 native_path='/v1/completions', normal_eos=True, eos_token_ids=[99])
        fixtures[f['fixture_id']] = f
    for i in range(8 if concurrency == 1 else 16):
        f = fixtures[f'CPU-{i % 8}']
        jobs.append(dict(request_id=f'cpu-c{concurrency}-r{i:02d}',
            fixture_id=f['fixture_id'], path='/v1/completions', body=dict(
                model='native-target-only', prompt=f['prompt_text'], max_tokens=4,
                temperature=0, top_p=1.0, repetition_penalty=1.0, stream=True,
                stream_options={'include_usage': True}, n=1)))
    identity = dict(configuration=dict(max_seqs=concurrency, max_len=2048,
        max_batched_tokens=16), eos_ids=[99], step_observer=True,
        qualification_trace_enabled=True, per_token_observer=False,
        drafter=None, normal_eos=True, prefix_cache=False, scope='one-layer CPU mock')
    trace = []
    def record(event):
        trace.append((json.dumps(event) + '\n').encode())
    record(dict(event='startup', identity=identity))
    engine = engine_for(concurrency)
    original_run = engine.runner.run
    native.observe_runner(engine, record)
    worker = native.BatchWorker(trace=record)
    worker.initialize(engine, SamplingParams, identity)
    by_prompt = {f['prompt_text']: f['prompt_ids'] for f in fixtures.values()}
    class Tokenizer:
        def encode(self, text, *, add_special_tokens):
            assert add_special_tokens is False
            return NS(ids=list(by_prompt[text]))

    submitted = {}
    class Stream(httpx.AsyncByteStream):
        def __init__(self, job):
            self.job, self.closed = job, False
        async def __aiter__(self):
            raw, text = [], ''
            while True:
                try:
                    item = self.job.events.get_nowait()
                except queue.Empty:
                    await asyncio.sleep(0)
                    continue
                if 'error' in item:
                    raise RuntimeError(item['error'])
                raw.append(item['token'])
                visible = raw[:-1] if item['done'] and raw[-1] in engine.eos_ids else raw
                current = decode(visible)
                if current != text:
                    payload = wire(dict(id='cmpl-' + self.job.rid, choices=[
                        dict(index=0, text=current[len(text):], finish_reason=None)]))
                    # Preserve actual client receive clocks across split UTF-8 bytes.
                    split = payload.index('答'.encode()) + 1
                    yield payload[:split]
                    yield payload[split:]
                    text = current
                if item['done']:
                    yield wire(native.completion(self.job, raw, engine.eos_ids))
                    yield wire('[DONE]')
                    return
        async def aclose(self):
            self.closed = True

    streams = []
    async def handle(request):
        assert request.url.path == '/v1/completions'
        ids, cap = native.parse_request(json.loads(request.content), Tokenizer(), 2048)
        job = worker.submit(ids, cap)  # The server, not this fixture, generates UUIDs.
        submitted[request.headers['x-request-id']] = job.rid
        stream = Stream(job); streams.append(stream)
        return httpx.Response(200, headers={'content-type': 'text/event-stream'}, stream=stream)

    stop = asyncio.Event()
    async def pump():
        while not stop.is_set():
            worker.tick()
            await asyncio.sleep(0)
    task = asyncio.create_task(pump())
    try:
        group = await client.run_group('http://cpu.invalid', jobs, concurrency,
            transport=httpx.MockTransport(handle), phase='qualification',
            request_timeout_s=5, group_timeout_s=10, cleanup_timeout_s=1)
    finally:
        stop.set()
        await task
        engine.runner.run = original_run
    assert not engine.has_work() and not worker.active and not worker.jobs
    assert all(s.closed for s in streams)
    return group, trace, jobs, fixtures, identity, submitted


class PipelineTests(unittest.IsolatedAsyncioTestCase):
    def replay(self, recording):
        with tempfile.TemporaryDirectory() as out:
            cmp = base.Comparator(CPU(), out)
            engines = [engine_for(recording['concurrency']) for _ in range(2)]
            observers = [base.Observer(e, cmp, arm)
                         for e, arm in zip(engines, ('reference', 'candidate'))]
            ledger = io.StringIO()
            try:
                result = http_replay.run_recorded_case(recording, engines, observers,
                    SamplingParams, NS(prefill_batch_backend_supported=lambda: True),
                    cmp, ledger, lambda: None, base=base)
                rows = [json.loads(line) for line in ledger.getvalue().splitlines()]
                self.assertTrue(all(row['status'] == 'PASS' for row in rows))
                self.assertTrue(any(a['check'].startswith('logits ')
                                    for row in rows for a in row['arrays']))
                self.assertTrue(all(not e.has_work() and not e.requests for e in engines))
                self.assertTrue(all(not o.calls and o.postrun is None for o in observers))
                return result
            finally:
                for observer in observers:
                    observer.restore()

    async def test_actual_client_worker_scheduler_join_and_numerical_replay(self):
        for C in (1, 8):
            with self.subTest(C=C):
                group, trace, jobs, fixtures, identity, submitted = await capture(C)
                self.assertIsNone(group['group_error'])
                self.assertTrue(group['cleanup']['complete'])
                self.assertFalse(group['summary']['formal_point_eligible'])
                recording = http_join.join(group, iter(trace), jobs, fixtures, identity, decode)
                self.assertEqual(len(recording['jobs']), 8 if C == 1 else 16)
                for rid, job in recording['jobs'].items():
                    self.assertRegex(rid, r'^[0-9a-f]{32}$')
                    self.assertEqual(rid, submitted[job['client_request_id']])
                    self.assertNotEqual(rid, job['client_request_id'])
                self.assertEqual(len(trace), 1 + 4 * len(recording['steps']))
                self.assertTrue(any(s['step'] > 1 and s['arrivals'] for s in recording['steps']))
                if C == 8:
                    self.assertGreaterEqual(max(map(int, recording['histogram'])), 2)
                    self.assertTrue(any(s['arrivals'] and s['before']['running']
                                        for s in recording['steps']))
                    self.assertTrue(any(s['schedule']['decode'] and s['schedule']['prefill']
                                        for s in recording['steps']))
                result = self.replay(recording)
                self.assertEqual(result['status'], 'PASS')
                self.assertEqual(result['raw_output_count'], 4 * len(jobs))
                self.assertEqual(result['decode_batch_histogram'], recording['histogram'])
                self.assertFalse(result['performance_qualified'])

    async def test_crosswired_sse_uuid_and_corrupt_actual_schedule_are_rejected(self):
        group, trace, jobs, fixtures, identity, _ = await capture(8)
        bad = copy.deepcopy(group)
        # Preserve original job/body IDs, but attach another request's wire receipt.
        for key in ('raw_sse_chunks', 'raw_body_bytes', 'done_perf_counter_s', 'start_perf_counter_s'):
            bad['requests'][0][key] = copy.deepcopy(group['requests'][1][key])
        with self.assertRaises(ValueError):
            http_join.join(bad, iter(trace), jobs, fixtures, identity, decode)
        events = [json.loads(line) for line in trace]
        actual = next(e for e in events if e['event'] == 'actual_scheduler_output' and len(e['decode']) >= 2)
        actual['decode'].reverse()
        with self.assertRaises(ValueError):
            http_join.join(group, ((json.dumps(e) + '\n').encode() for e in events),
                           jobs, fixtures, identity, decode)


if __name__ == '__main__':
    unittest.main()
