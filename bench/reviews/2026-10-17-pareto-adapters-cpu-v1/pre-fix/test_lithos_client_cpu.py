"""In-memory protocol/queue tests only: no socket, server, tokenizer or model."""
import asyncio
import base64
import hashlib
import json
from pathlib import Path
import tempfile
import time
import unittest

import httpx

from bench.m5_ultra_256gb.pareto_realistic import client as original
from bench.m5_ultra_256gb.pareto_realistic import lithos_client as client


def jobs(n=4, cap=8):
    return [{'request_id': f'r{i}', 'fixture_id': f'D{i % 2}', 'path': '/api/chat',
             'body': {'model': 'mock-only', 'messages': [{'role': 'user', 'content': f'task {i % 2}'}],
                      'stream': True, 'options': {'num_predict': cap, 'temperature': 0,
                                                 'top_p': 1, 'top_k': 0, 'seed': 0}}} for i in range(n)]


def qualified(c=2, n=4, cap=8):
    frozen = jobs(n, cap)
    return {'status': 'qualified', 'receipt_sha256': 'a' * 64, 'concurrency': c,
            'normal_eos': True, 'protocol': client.PROTOCOL, 'cache_contract': client.CACHE_CONTRACT,
            'usage_count_rule': client.COUNT_RULE, 'usage_semantics': client.COUNT_SEMANTICS,
            'raw_ids_count_rule': 'none', 'terminal_token_ids': [248046, 248044],
            'no_post_terminal_tokens': True,
            'expected_prompt_tokens_by_fixture': {'D0': 978, 'D1': 886},
            'request_fixture_map': {j['request_id']: j['fixture_id'] for j in frozen},
            'fixture_contracts': {j['fixture_id']: {'path': j['path'],
                'body_sha256': client.canonical_body_sha256(j['body'])} for j in frozen}}


def event(data):
    return (json.dumps(data, ensure_ascii=False) + '\n').encode()


def delta(text='answer'):
    return {'model': 'mock-only', 'done': False,
            'message': {'role': 'assistant', 'content': text}}


def terminal(rid='r0', count=3, finish='stop'):
    return {'model': 'mock-only', 'done': True, 'done_reason': finish,
            'message': {'role': 'assistant', 'content': ''},
            'prompt_eval_count': 978 if int(rid[1:]) % 2 == 0 else 886,
            'eval_count': count, 'total_duration': 1, 'eval_duration': 1}


class Stream(httpx.AsyncByteStream):
    def __init__(self, chunks, delays=None, eof_event=None, eof_delay=0):
        self.chunks = chunks
        self.delays = delays or [0] * len(chunks)
        self.eof_event, self.eof_delay = eof_event, eof_delay
        self.closed = False

    async def __aiter__(self):
        for delay, chunk in zip(self.delays, self.chunks):
            await asyncio.sleep(delay)
            yield chunk
        if self.eof_event is not None:
            await self.eof_event.wait()
        await asyncio.sleep(self.eof_delay)

    async def aclose(self):
        self.closed = True


def response(stream, status=200):
    return httpx.Response(status, headers={'content-type': 'application/x-ndjson'}, stream=stream)


class ClientTests(unittest.IsolatedAsyncioTestCase):
    async def run_mock(self, handler, *, c=2, n=4, qualification=True, **kwargs):
        options = dict(phase='warm', request_timeout_s=1, group_timeout_s=3, cleanup_timeout_s=.2)
        options.update(kwargs)
        return await client.run_group('http://mock.invalid', jobs(n), c,
            qualification=qualified(c, n) if qualification is True else qualification,
            transport=httpx.MockTransport(handler), **options)

    async def test_real_eight_outstanding_fifo_and_formal_actual_count_window(self):
        calls, bodies = [], []
        gpu_lock, first_eight = asyncio.Lock(), asyncio.Event()
        active = peak = 0

        class FifoStream(Stream):
            async def __aiter__(self):
                nonlocal active
                await first_eight.wait()
                yield event(delta())
                await asyncio.sleep(.001)
                # Match the official worker: release single-generator ownership
                # before its completed result reaches the network consumer.
                active -= 1
                gpu_lock.release()
                yield self.chunks[0]

        async def handler(request):
            nonlocal active, peak
            rid = request.headers['X-Request-ID']
            calls.append(rid)
            bodies.append(json.loads(request.content))
            if len(calls) == 8:
                first_eight.set()
            await gpu_lock.acquire()
            active += 1
            peak = max(peak, active)
            count, finish = (8, 'length') if int(rid[1:]) % 2 else (3, 'stop')
            return response(FifoStream([event(terminal(rid, count, finish))]))

        before = original._request, original._cleanup, original.run_group
        result = await self.run_mock(handler, c=8, n=64, phase='formal')
        rows, summary = result['requests'], result['summary']
        self.assertEqual(before, (original._request, original._cleanup, original.run_group))
        self.assertEqual(peak, 1)
        self.assertEqual(summary['max_http_outstanding'], 8)
        self.assertGreater(summary['time_at_peak_outstanding_s'], 0)
        self.assertEqual(calls, [f'r{i}' for i in range(64)])
        self.assertEqual(bodies, [job['body'] for job in jobs(64)])
        self.assertTrue(summary['formal_point_eligible'])
        self.assertEqual(summary['actual_output_tokens']['sum'], 32 * (2 + 8))
        self.assertEqual(summary['cap_ratio'], .5)
        self.assertEqual(summary['early_eos_ratio'], .5)
        self.assertEqual(summary['group_window_s'], max(r['done_perf_counter_s'] for r in rows)
                         - min(r['start_perf_counter_s'] for r in rows))
        self.assertEqual(summary['aggregate_output_tok_s'], 320 / summary['group_window_s'])
        for row, job in zip(rows, jobs(64)):
            self.assertEqual(row['body'], job['body'])
            self.assertEqual(row['body_sha256'], client.canonical_body_sha256(job['body']))
            self.assertEqual(json.loads(base64.b64decode(row['request_body_bytes_base64'])), job['body'])
            self.assertNotIn('raw_sse_chunks', row)
            self.assertNotIn('sse_events', row)
            self.assertTrue(row['stream_eof'])
        self.assertEqual(result['schema'], client.SCHEMA)

    async def test_fragmented_utf8_receive_clock_refill_before_eof_and_close_order(self):
        streams = []
        async def handler(request):
            rid = request.headers['X-Request-ID']
            text, end = event(delta('答')), event(terminal(rid))
            pos = text.index('答'.encode()) + 1
            stream = Stream([text[:pos], text[pos:], end[:-1], end[-1:]], eof_delay=.03)
            streams.append(stream)
            return response(stream)

        class CloseCheckedTransport(httpx.MockTransport):
            async def aclose(self):
                self.closed_after_responses = all(stream.closed for stream in streams)

        transport = CloseCheckedTransport(handler)
        result = await client.run_group('http://mock.invalid', jobs(2), 1,
            qualification=qualified(1, 2), phase='warm', transport=transport, cleanup_timeout_s=.2)
        rows = result['requests']
        self.assertLess(rows[1]['start_perf_counter_s'], rows[0]['eof_perf_counter_s'])
        self.assertEqual(rows[0]['answer'], '答')
        self.assertEqual(rows[0]['done_perf_counter_s'], rows[0]['raw_ndjson_chunks'][3]['client_perf_counter_s'])
        self.assertEqual(rows[0]['first_output_perf_counter_s'], rows[0]['raw_ndjson_chunks'][1]['client_perf_counter_s'])
        self.assertGreater(rows[0]['eof_perf_counter_s'] - rows[0]['done_perf_counter_s'], .02)
        self.assertTrue(transport.closed_after_responses)
        self.assertTrue(result['cleanup']['complete'])

    async def test_bad_terminal_and_tail_protocol_cases(self):
        good = event(terminal())
        corruptions = {
            'missing': [event(delta())], 'truncated': [good[:-1]],
            'duplicate_same_chunk': [good + good], 'duplicate_later_chunk': [good, good],
            'trailing_partial': [good, b'{'], 'extra_blank': [good, b'\n'],
            'invalid_utf8_tail': [good, b'\xff'], 'incomplete_utf8_tail': [good, b'\xe7'],
            'server_error': [event({'error': 'failed'})],
            'bad_done_bool': [event({**terminal(), 'done': 1})],
            'wrong_model': [event({**terminal(), 'model': 'other'})],
            'duplicate_keys': [good.replace(b'"done": true', b'"done": false,"done": true')],
            'hidden_reasoning': [event({**delta(), 'message': {'role': 'assistant', 'content': '', 'thinking': 'x'}})],
            'nonempty_final': [event({**terminal(), 'message': {'role': 'assistant', 'content': 'repeat'}})],
        }
        for name, chunks in corruptions.items():
            with self.subTest(name=name):
                async def handler(request):
                    return response(Stream(chunks))
                result = await self.run_mock(handler, c=1, n=1)
                self.assertIsNone(result['summary']['aggregate_output_tok_s'])
                self.assertFalse(result['summary']['eligible_under_supplied_qualification'])
                self.assertEqual(b''.join(base64.b64decode(x['body_bytes_base64'])
                                         for x in result['requests'][0]['raw_ndjson_chunks']), b''.join(chunks))

    async def test_count_normalization_requires_bound_qualification(self):
        async def handler(request):
            return response(Stream([event(delta()), event(terminal())]))
        result = await self.run_mock(handler, c=1, n=1, qualification=None)
        row = result['requests'][0]
        self.assertEqual(row['reported_eval_count'], 3)
        self.assertEqual(row['usage']['eval_count'], 3)
        self.assertIsNone(row['count_normalization'])
        self.assertIsNone(row['qualified_output_tokens'])
        self.assertIsNone(result['summary']['aggregate_output_tok_s'])
        self.assertTrue(result['summary']['transport_complete'])
        qualified_result = await self.run_mock(handler, c=1, n=1)
        self.assertEqual(qualified_result['requests'][0]['qualified_output_tokens'], 2)
        self.assertEqual(qualified_result['requests'][0]['reported_eval_count'], 3)

    async def test_corrupt_qualification_fields_never_grant_counts(self):
        changes = [('protocol', 'other'), ('cache_contract', 'cache-enabled'),
                   ('usage_count_rule', 'identity'), ('concurrency', 8), ('concurrency', True),
                   ('terminal_token_ids', [99]), ('no_post_terminal_tokens', False),
                   ('request_fixture_map', {}), ('raw_ids_count_rule', 'all_ids'),
                   ('fixture_contracts', {'D0': {'path': '/api/chat', 'body_sha256': '0' * 64}})]
        async def handler(request):
            return response(Stream([event(terminal())]))
        for field, value in changes:
            with self.subTest(field=field):
                q = qualified(1, 1)
                q[field] = value
                result = await self.run_mock(handler, c=1, n=1, qualification=q)
                self.assertTrue(result['qualification_errors'])
                self.assertIsNone(result['requests'][0]['qualified_output_tokens'])
                self.assertEqual(result['requests'][0]['reported_eval_count'], 3)

    async def test_count_finish_validation_and_empty_eos_output(self):
        bad = [(0, 'stop', 978), (True, 'stop', 978), (9, 'stop', 978),
               (7, 'length', 978), (3, 'other', 978), (3, 'stop', 999)]
        for count, finish, prompt in bad:
            with self.subTest(count=count, finish=finish, prompt=prompt):
                async def handler(request):
                    return response(Stream([event({**terminal(count=count, finish=finish), 'prompt_eval_count': prompt})]))
                result = await self.run_mock(handler, c=1, n=1)
                self.assertIsNone(result['summary']['aggregate_output_tok_s'])
        async def eos_only(request):
            return response(Stream([event(terminal(count=1))]))
        result = await self.run_mock(eos_only, c=1, n=1)
        self.assertEqual(result['requests'][0]['qualified_output_tokens'], 0)
        self.assertIsNone(result['requests'][0]['ttft_s'])
        self.assertEqual(result['summary']['aggregate_output_tok_s'], 0)

    async def test_unqualified_request_options_rejected_before_any_send(self):
        calls = []
        async def handler(request):
            calls.append(request)
            raise AssertionError('must not send')
        mutations = [lambda j: j.update(path='/v1/chat/completions'),
                     lambda j: j['body']['options'].update(stop=['x']),
                     lambda j: j['body']['options'].update(num_predict=True),
                     lambda j: j['body'].update(keep_alive=0),
                     lambda j: j['body']['messages'].append({'role': 'user', 'content': 'another'}),
                     lambda j: j['body']['options'].update(temperature=.1)]
        for mutate in mutations:
            frozen = jobs(1)
            mutate(frozen[0])
            with self.assertRaises(ValueError):
                await client.run_group('http://mock.invalid', frozen, 1, phase='warm', transport=httpx.MockTransport(handler))
        self.assertFalse(calls)

    async def test_http_error_no_retry_and_raw_retained(self):
        calls = []
        async def handler(request):
            rid = request.headers['X-Request-ID']
            calls.append(rid)
            return (response(Stream([b'{"error":"model_busy"}']), 429) if rid == 'r1'
                    else response(Stream([event(terminal(rid))])))
        result = await self.run_mock(handler)
        self.assertEqual(calls, ['r0', 'r1', 'r2', 'r3'])
        self.assertEqual(result['requests'][1]['http_status'], 429)
        self.assertEqual(base64.b64decode(result['requests'][1]['raw_ndjson_chunks'][0]['body_bytes_base64']), b'{"error":"model_busy"}')
        self.assertIsNone(result['summary']['aggregate_output_tok_s'])

    async def test_absolute_request_deadline_and_group_deadline(self):
        async def trickle(request):
            return response(Stream([event(delta())] * 100, [.004] * 100))
        started = time.perf_counter()
        result = await self.run_mock(trickle, c=1, n=1, request_timeout_s=.025)
        self.assertLess(time.perf_counter() - started, .25)
        self.assertEqual(result['requests'][0]['error']['type'], 'TimeoutError')
        self.assertGreater(len(result['requests'][0]['raw_ndjson_chunks']), 1)
        async def blocked(request):
            return response(Stream([event(terminal(request.headers['X-Request-ID']))], [.5]))
        result = await self.run_mock(blocked, c=2, n=8, group_timeout_s=.02)
        self.assertEqual(result['group_error']['type'], 'GroupDeadline')
        self.assertEqual(result['summary']['submitted_requests'], 2)
        self.assertEqual([r['status'] for r in result['requests']], ['cancelled'] * 2 + ['not_started'] * 6)
        self.assertTrue(result['cleanup']['complete'])

    async def test_external_cancel_persists_correct_ndjson_envelope(self):
        entered = asyncio.Event()
        async def handler(request):
            entered.set()
            return response(Stream([event(delta()), event(terminal(request.headers['X-Request-ID']))], [0, 1]))
        with tempfile.TemporaryDirectory() as temp:
            destination = Path(temp) / 'new-result'
            task = asyncio.create_task(self.run_mock(handler, n=8, output_dir=destination))
            await entered.wait()
            await asyncio.sleep(.005)
            task.cancel()
            with self.assertRaises(client.GroupCancelled) as caught:
                await task
            result = caught.exception.result
            saved = json.loads((destination / 'group.json').read_text())
            self.assertEqual(result, saved)
            self.assertEqual(saved['schema'], client.SCHEMA)
            self.assertIn('NDJSON', saved['raw_bytes_semantics'])
            self.assertNotIn('raw_sse_chunks', saved['requests'][0])
            self.assertFalse((destination / 'group.json.tmp').exists())
            self.assertTrue(result['cleanup']['complete'])

    async def test_stalled_tail_bounded_cleanup_fails_not_assumed_eof(self):
        never = asyncio.Event()
        streams = []
        async def handler(request):
            stream = Stream([event(terminal(request.headers['X-Request-ID']))], eof_event=never)
            streams.append(stream)
            return response(stream)
        started = time.perf_counter()
        result = await self.run_mock(handler, c=1, n=2, cleanup_timeout_s=.02)
        self.assertLess(time.perf_counter() - started, .2)
        self.assertEqual(result['summary']['done_requests'], 2)
        self.assertEqual(result['summary']['submitted_requests'], 2)
        self.assertFalse(result['cleanup']['complete'])
        self.assertTrue(result['cleanup']['deadline_exceeded'])
        self.assertIsNone(result['summary']['aggregate_output_tok_s'])
        self.assertTrue(all(stream.closed for stream in streams))
        self.assertTrue(all(not row['stream_eof'] for row in result['requests']))

    async def test_unresolved_response_close_blocks_client_close_and_reports_tasks(self):
        allow_close = asyncio.Event()
        class ResistantStream(Stream):
            async def aclose(self):
                try:
                    await allow_close.wait()
                except asyncio.CancelledError:
                    await allow_close.wait()
                self.closed = True
        class CloseCheckedTransport(httpx.MockTransport):
            client_closed = False
            async def aclose(self):
                self.client_closed = True
        async def handler(request):
            return response(ResistantStream([event(terminal())]))
        transport = CloseCheckedTransport(handler)
        try:
            result = await client.run_group('http://mock.invalid', jobs(1), 1,
                qualification=qualified(1, 1), phase='warm', transport=transport,
                request_timeout_s=1, group_timeout_s=1, cleanup_timeout_s=.02)
            self.assertFalse(result['cleanup']['complete'])
            self.assertFalse(result['cleanup']['client_close_started'])
            self.assertFalse(transport.client_closed)
            self.assertIn('drain-close:r0', result['cleanup']['pending_tasks'])
            self.assertIsNone(result['summary']['aggregate_output_tok_s'])
        finally:
            allow_close.set()
            pending = [task for task in asyncio.all_tasks()
                       if task is not asyncio.current_task() and task.get_name() == 'drain-close:r0']
            if pending:
                await asyncio.wait_for(asyncio.gather(*pending, return_exceptions=True), .2)

    async def test_raw_bound_retains_partial_and_frozen_source_unchanged(self):
        async def handler(request):
            return response(Stream([event(delta()), event(terminal())]))
        result = await self.run_mock(handler, c=1, n=1, max_raw_bytes_per_request=10)
        row = result['requests'][0]
        self.assertTrue(row['raw_body_truncated'])
        self.assertEqual(row['raw_body_bytes'], 10)
        self.assertEqual(base64.b64decode(row['raw_ndjson_chunks'][0]['body_bytes_base64']), event(delta())[:10])
        self.assertIsNone(result['summary']['aggregate_output_tok_s'])
        self.assertEqual(hashlib.sha256(client.BASE_PATH.read_bytes()).hexdigest(), client.BASE_SHA256)


if __name__ == '__main__':
    unittest.main()
