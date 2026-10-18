"""Pure in-memory HTTPX tests: no listener, network, tokenizer, MLX, or model."""
import asyncio
import base64
import json
from pathlib import Path
import tempfile
import time
import unittest

import httpx

from bench.m5_ultra_256gb.pareto_realistic.client import (
    COUNT_SEMANTICS, GroupCancelled, _distribution, canonical_body_sha256, run_group,
)


def jobs(n=4, cap=8):
    return [{'request_id': f'r{i}', 'fixture_id': f'f{i % 2}',
             'path': '/v1/completions', 'body': {
                 'model': 'mock-only', 'prompt': f'fixture {i % 2}',
                 'max_tokens': cap, 'temperature': 0, 'stream': True,
                 'stream_options': {'include_usage': True}}} for i in range(n)]


def qualified(c=2, rule='none', n=4, cap=8):
    frozen = jobs(n, cap)
    return {'status': 'qualified', 'receipt_sha256': 'a' * 64,
            'concurrency': c, 'normal_eos': True,
            'request_fixture_map': {j['request_id']: j['fixture_id'] for j in frozen},
            'fixture_contracts': {j['fixture_id']: {'path': j['path'],
                'body_sha256': canonical_body_sha256(j['body'])} for j in frozen},
            'usage_semantics': COUNT_SEMANTICS, 'usage_count_rule': 'identity',
            'raw_ids_count_rule': rule, 'terminal_token_ids': [99],
            'expected_prompt_tokens_by_fixture': {'f0': 10, 'f1': 10}}


def event(value):
    return ('data: ' + (value if isinstance(value, str) else json.dumps(value)) + '\n\n').encode()


def payload(rid, n=3, finish='stop', native=False, reasoning=False):
    first = {'id': 'cmpl-' + rid, 'choices': [{'index': 0, 'delta': {'content': 'answer'}}]}
    if reasoning:
        first['choices'][0]['delta']['reasoning_content'] = 'reason'
    final = {'id': 'cmpl-' + rid, 'choices': [{'index': 0, 'text': '', 'finish_reason': finish}],
             'usage': {'completion_tokens': n, 'prompt_tokens': 10, 'total_tokens': 10 + n,
                       'prompt_tokens_details': {'cached_tokens': 0}}}
    if native:
        committed = list(range(n))
        eos = 99 if finish == 'stop' else None
        final.update(benchmark_output_ids=committed + ([eos] if eos is not None else []),
                     benchmark_committed_output_ids=committed, benchmark_terminal_eos_id=eos)
    return [event(first), event(final), event('[DONE]')]


class Stream(httpx.AsyncByteStream):
    def __init__(self, chunks, delays=None, close_event=None):
        self.chunks = chunks
        self.delays = delays or [0] * len(chunks)
        self.close_event = close_event
        self.closed = False

    async def __aiter__(self):
        for delay, chunk in zip(self.delays, self.chunks):
            await asyncio.sleep(delay)
            yield chunk

    async def aclose(self):
        if self.close_event is not None:
            await self.close_event.wait()
        self.closed = True


def response(stream, status=200):
    return httpx.Response(status, headers={'content-type': 'text/event-stream'}, stream=stream)


class ClientTests(unittest.IsolatedAsyncioTestCase):
    async def run_mock(self, handler, *, c=2, n=4, qualification=True, **kwargs):
        q = qualified(c, n=n) if qualification is True else qualification
        return await run_group('http://mock.invalid', jobs(n), c,
            qualification=q, phase='warm', request_timeout_s=1, group_timeout_s=3,
            cleanup_timeout_s=.2, transport=httpx.MockTransport(handler), **kwargs)

    async def test_refill_initial_overlap_and_fixed_formal_count(self):
        calls = []
        async def handler(request):
            rid = request.headers['X-Request-ID']
            calls.append(rid)
            delay = .04 if rid == 'r0' else .001
            return response(Stream(payload(rid), [0, delay, 0]))
        result = await run_group('http://mock.invalid', jobs(64), 2,
            qualification=qualified(n=64), transport=httpx.MockTransport(handler),
            request_timeout_s=1, group_timeout_s=3, cleanup_timeout_s=.2)
        rows, summary = result['requests'], result['summary']
        self.assertEqual(calls, [f'r{i}' for i in range(64)])
        self.assertEqual(summary['max_http_outstanding'], 2)
        self.assertLess(rows[2]['start_perf_counter_s'], rows[0]['done_perf_counter_s'])
        self.assertTrue(summary['eligible_under_supplied_qualification'])
        self.assertEqual(summary['actual_output_tokens']['sum'], 192)
        self.assertEqual(summary['aggregate_output_tok_s'], 192 / summary['group_window_s'])
        self.assertEqual(summary['group_window_s'], max(r['done_perf_counter_s'] for r in rows)
                         - min(r['start_perf_counter_s'] for r in rows))

    async def test_early_eos_variable_lengths_reasoning_and_actual_total(self):
        lengths = [1, 8, 2, 3]
        async def handler(request):
            rid = request.headers['X-Request-ID']
            n = lengths[int(rid[1:])]
            return response(Stream(payload(rid, n, 'length' if n == 8 else 'stop', reasoning=True)))
        result = await self.run_mock(handler)
        s = result['summary']
        self.assertEqual(s['actual_output_tokens']['sum'], 14)
        self.assertEqual(s['early_eos_ratio'], .75)
        self.assertEqual(s['cap_ratio'], .25)
        self.assertEqual(s['aggregate_output_tok_s'], 14 / s['group_window_s'])
        self.assertEqual(result['requests'][0]['answer'], 'answer')
        self.assertEqual(result['requests'][0]['reasoning'], 'reason')
        self.assertNotIn('visible_roundtrip_ids', result['requests'][0])

    async def test_429_retained_once_no_replacement_no_valid_point(self):
        calls = []
        async def handler(request):
            rid = request.headers['X-Request-ID']
            calls.append(rid)
            return (response(Stream([b'{"error":"model_busy"}']), 429) if rid == 'r1'
                    else response(Stream(payload(rid))))
        result = await self.run_mock(handler)
        self.assertEqual(calls, ['r0', 'r1', 'r2', 'r3'])
        row = result['requests'][1]
        self.assertEqual(row['http_status'], 429)
        self.assertEqual(base64.b64decode(row['raw_sse_chunks'][0]['body_bytes_base64']), b'{"error":"model_busy"}')
        self.assertIsNone(result['summary']['aggregate_output_tok_s'])
        self.assertEqual(result['summary']['successful_requests'], 3)
        self.assertEqual(result['summary']['diagnostic_successful_output_tokens'], 9)

    async def test_absolute_request_deadline_defeats_trickle(self):
        async def handler(request):
            rid = request.headers['X-Request-ID']
            chunks = [payload(rid)[0]] + [b': heartbeat\n\n'] * 100
            return response(Stream(chunks, [.005] * len(chunks)))
        started = time.perf_counter()
        result = await run_group('http://mock.invalid', jobs(1), 1,
            qualification=qualified(1, n=1), phase='warm', request_timeout_s=.035,
            group_timeout_s=1, cleanup_timeout_s=.1, transport=httpx.MockTransport(handler))
        self.assertLess(time.perf_counter() - started, .3)
        row = result['requests'][0]
        self.assertEqual(row['error']['type'], 'TimeoutError')
        self.assertEqual(row['answer'], 'answer')
        self.assertGreater(len(row['raw_sse_chunks']), 1)
        self.assertFalse(row['done'])
        self.assertIsNone(result['summary']['aggregate_output_tok_s'])

    async def test_group_deadline_preserves_partial_and_unsent_tail(self):
        async def handler(request):
            rid = request.headers['X-Request-ID']
            return response(Stream(payload(rid), [0, .5, 0]))
        result = await run_group('http://mock.invalid', jobs(8), 2,
            qualification=qualified(n=8), phase='warm', request_timeout_s=1,
            group_timeout_s=.025, cleanup_timeout_s=.1, transport=httpx.MockTransport(handler))
        self.assertEqual(result['group_error']['type'], 'GroupDeadline')
        self.assertEqual(result['summary']['submitted_requests'], 2)
        self.assertEqual([r['status'] for r in result['requests']], ['cancelled'] * 2 + ['not_started'] * 6)
        self.assertTrue(result['cleanup']['complete'])
        self.assertEqual(result['requests'][0]['answer'], 'answer')

    async def test_external_cancel_writes_partial_then_raises_with_result(self):
        entered = asyncio.Event()
        async def handler(request):
            entered.set()
            return response(Stream(payload(request.headers['X-Request-ID']), [0, 1, 0]))
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / 'new-output'
            task = asyncio.create_task(run_group('http://mock.invalid', jobs(8), 2,
                phase='warm', qualification=qualified(n=8), request_timeout_s=2,
                group_timeout_s=3, cleanup_timeout_s=.1, output_dir=destination,
                transport=httpx.MockTransport(handler)))
            await entered.wait()
            await asyncio.sleep(.005)
            task.cancel()
            with self.assertRaises(GroupCancelled) as raised:
                await task
            result = raised.exception.result
            self.assertEqual(result['requests'][0]['answer'], 'answer')
            self.assertFalse(result['summary']['eligible_under_supplied_qualification'])
            saved = json.loads((destination / 'group.json').read_text())
            self.assertEqual(saved, result)
            self.assertEqual(result['summary']['submitted_requests'], 2)
            self.assertTrue(result['cleanup']['complete'])

    async def test_close_is_bounded_and_does_not_delay_refill(self):
        never = asyncio.Event()
        async def handler(request):
            return response(Stream(payload(request.headers['X-Request-ID']), close_event=never))
        started = time.perf_counter()
        result = await run_group('http://mock.invalid', jobs(4), 1,
            phase='warm', qualification=qualified(1), request_timeout_s=1,
            group_timeout_s=1, cleanup_timeout_s=.03, transport=httpx.MockTransport(handler))
        self.assertLess(time.perf_counter() - started, .3)
        self.assertEqual(result['summary']['done_requests'], 4)
        self.assertTrue(result['cleanup']['deadline_exceeded'])
        self.assertFalse(result['summary']['eligible_under_supplied_qualification'])

    async def test_unqualified_usage_stays_reported_not_throughput(self):
        async def handler(request):
            return response(Stream(payload(request.headers['X-Request-ID'])))
        result = await self.run_mock(handler, qualification=None)
        self.assertTrue(result['summary']['transport_complete'])
        self.assertIsNone(result['summary']['aggregate_output_tok_s'])
        self.assertEqual(result['requests'][0]['reported_completion_tokens'], 3)
        self.assertIsNone(result['requests'][0]['qualified_output_tokens'])
        self.assertIsNone(result['summary']['diagnostic_successful_tokens_per_full_attempt_s'])

    async def test_native_raw_committed_terminal_and_explicit_oracle(self):
        async def handler(request):
            return response(Stream(payload(request.headers['X-Request-ID'], native=True)))
        q = qualified(rule='native_committed_ids')
        q['expected_raw_ids_by_request'] = {'r0': [0, 1, 2, 99]}
        result = await self.run_mock(handler, qualification=q)
        self.assertTrue(result['summary']['eligible_under_supplied_qualification'])
        row = result['requests'][0]
        self.assertEqual(row['native_fields']['benchmark_output_ids'], [0, 1, 2, 99])
        self.assertEqual(row['qualified_output_tokens'], 3)
        q['expected_raw_ids_by_request']['r0'] = [4, 1, 2, 99]
        bad = await self.run_mock(handler, qualification=q)
        self.assertFalse(bad['summary']['eligible_under_supplied_qualification'])
        self.assertIn('actual raw IDs differ from explicitly supplied per-request oracle', bad['requests'][0]['validation_errors'])

    async def test_native_eos_only_zero_output_is_not_replaced(self):
        async def handler(request):
            rid = request.headers['X-Request-ID']
            chunks = payload(rid, n=0, native=True)
            return response(Stream(chunks[1:]))
        result = await self.run_mock(handler, qualification=qualified(rule='native_committed_ids'))
        self.assertTrue(result['summary']['eligible_under_supplied_qualification'])
        self.assertEqual(result['summary']['aggregate_output_tok_s'], 0)
        self.assertEqual(result['summary']['ttft_s']['n'], 0)
        self.assertEqual(result['summary']['early_eos_ratio'], 1)

    async def test_native_terminal_eos_at_cap_passes_cap_plus_one_fails(self):
        for count, eligible in ((2, True), (3, False)):
            with self.subTest(committed_count=count):
                async def handler(request):
                    return response(Stream(payload(request.headers['X-Request-ID'], n=count, native=True)))
                result = await run_group('http://mock.invalid', jobs(1, cap=3), 1,
                    qualification=qualified(1, 'native_committed_ids', n=1, cap=3),
                    phase='qualification', transport=httpx.MockTransport(handler))
                self.assertEqual(result['summary']['eligible_under_supplied_qualification'], eligible)
                self.assertEqual(len(result['requests'][0]['native_fields']['benchmark_output_ids']), count + 1)
                self.assertFalse(result['summary']['formal_point_eligible'])

    async def test_wrong_counts_duplicate_ids_and_missing_done_invalidate(self):
        async def handler(request):
            rid = request.headers['X-Request-ID']
            chunks = payload('same-response-id', native=True)
            if rid == 'r1':
                data = json.loads(chunks[1].decode()[6:])
                data['usage']['completion_tokens'] = 7
                chunks[1] = event(data)
            if rid == 'r3':
                chunks.pop()
            return response(Stream(chunks))
        result = await self.run_mock(handler, qualification=qualified(rule='native_committed_ids'))
        self.assertIsNone(result['summary']['aggregate_output_tok_s'])
        self.assertTrue(any('usage' in e for e in result['requests'][1]['validation_errors']))
        self.assertTrue(any('reused' in e for e in result['requests'][0]['validation_errors']))
        self.assertEqual(result['requests'][3]['error']['type'], 'StreamProtocolError')

    async def test_sse_chunk_boundaries_multiline_and_unicode_preserved(self):
        async def handler(request):
            rid = request.headers['X-Request-ID']
            first = ('data: {"id":"cmpl-' + rid + '",\r\n'
                     'data: "choices":[{"text":"你好"}]}\r\n\r\n').encode()
            rest = b''.join(payload(rid)[1:])
            data = b':comment\r\n\r\n' + first + rest
            return response(Stream([data[i:i + 1] for i in range(len(data))]))
        result = await self.run_mock(handler, n=1, c=1)
        self.assertEqual(result['requests'][0]['answer'], '你好')
        self.assertTrue(result['summary']['eligible_under_supplied_qualification'])
        raw = b''.join(base64.b64decode(x['body_bytes_base64']) for x in result['requests'][0]['raw_sse_chunks'])
        self.assertIn('你好'.encode(), raw)

    async def test_raw_limit_retains_prefix_and_invalidates(self):
        async def handler(request):
            return response(Stream([b'x' * 100]))
        result = await self.run_mock(handler, n=1, c=1, max_raw_bytes_per_request=20)
        row = result['requests'][0]
        self.assertEqual(row['raw_body_bytes'], 20)
        self.assertTrue(row['raw_body_truncated'])
        self.assertIsNone(result['summary']['aggregate_output_tok_s'])

    async def test_malformed_metadata_still_preserves_all_rows(self):
        async def handler(request):
            rid = request.headers['X-Request-ID']
            chunks = payload(rid)
            data = json.loads(chunks[1].decode()[6:])
            data['usage']['prompt_tokens_details'] = ['bad-object']
            data['choices'][0]['finish_reason'] = {'unexpected': 'object'}
            chunks[1] = event(data)
            return response(Stream(chunks))
        result = await self.run_mock(handler)
        self.assertEqual(len(result['requests']), 4)
        self.assertEqual(result['summary']['done_requests'], 4)
        self.assertIsNone(result['summary']['aggregate_output_tok_s'])
        self.assertTrue(all(r['raw_sse_chunks'] for r in result['requests']))
        self.assertTrue(all(r['validation_errors'] for r in result['requests']))

    async def test_second_cancel_during_cleanup_is_saved_and_bounded(self):
        never = asyncio.Event()
        calls = []
        async def handler(request):
            calls.append(request.headers['X-Request-ID'])
            return response(Stream(payload(calls[-1]), [0, 1, 0], close_event=never))
        task = asyncio.create_task(run_group('http://mock.invalid', jobs(8), 2,
            phase='warm', qualification=qualified(n=8), request_timeout_s=2,
            group_timeout_s=3, cleanup_timeout_s=.04, transport=httpx.MockTransport(handler)))
        await asyncio.sleep(.005)
        task.cancel()
        await asyncio.sleep(.005)
        task.cancel()
        with self.assertRaises(GroupCancelled) as caught:
            await task
        self.assertEqual(len(calls), 2)
        self.assertTrue(caught.exception.result['cleanup']['cancelled_again'])
        self.assertIsNone(caught.exception.result['summary']['aggregate_output_tok_s'])

    async def test_output_directory_cannot_be_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileExistsError):
                await run_group('http://mock.invalid', jobs(1), 1, phase='warm', output_dir=directory,
                                transport=httpx.MockTransport(lambda request: None))

    async def test_scope_and_formal_size_validation(self):
        with self.assertRaises(ValueError):
            await run_group('http://mock.invalid', jobs(2), 2)
        duplicate = jobs(2)
        duplicate[1]['request_id'] = duplicate[0]['request_id']
        with self.assertRaises(ValueError):
            await run_group('http://mock.invalid', duplicate, 2, phase='warm')
        async def handler(request):
            return response(Stream(payload(request.headers['X-Request-ID'])))
        result = await self.run_mock(handler, qualification=qualified(8))
        self.assertIsNone(result['summary']['aggregate_output_tok_s'])
        self.assertIn('invalid qualification field: concurrency', result['qualification_errors'])

    async def test_request_path_body_and_exact_queue_map_bind_qualification(self):
        async def handler(request):
            return response(Stream(payload(request.headers['X-Request-ID'])))
        for field, changed in (('prompt', 'different prompt'), ('model', 'different model'),
                               ('max_tokens', 9), ('temperature', .5),
                               ('chat_template_kwargs', {'enable_thinking': True}),
                               ('path', '/v1/chat/completions'), ('request_id', 'changed-rid')):
            with self.subTest(field=field):
                altered = jobs()
                if field in {'path', 'request_id'}:
                    altered[0][field] = changed
                else:
                    altered[0]['body'][field] = changed
                result = await run_group('http://mock.invalid', altered, 2, phase='warm',
                    qualification=qualified(), transport=httpx.MockTransport(handler))
                self.assertFalse(result['summary']['eligible_under_supplied_qualification'])
                self.assertIsNone(result['summary']['aggregate_output_tok_s'])
                expected = 'request_fixture_map' if field == 'request_id' else 'fixture_contracts'
                self.assertIn('invalid qualification field: ' + expected, result['qualification_errors'])

    async def test_custom_stop_minimum_output_and_ignore_eos_are_rejected(self):
        for field, value in (('stop', ['end']), ('stop_token_ids', [99]), ('min_tokens', 1),
                             ('min_new_tokens', 1), ('ignore_eos', False),
                             ('forced_eos_token_id', 99)):
            with self.subTest(field=field):
                altered = jobs()
                altered[0]['body'][field] = value
                with self.assertRaises(ValueError):
                    await run_group('http://mock.invalid', altered, 2, phase='warm',
                        qualification=qualified(), transport=httpx.MockTransport(lambda request: None))

    def test_canonical_body_digest_is_order_invariant_but_content_exact(self):
        self.assertEqual(canonical_body_sha256({'z': '你好', 'a': [1, False]}),
                         canonical_body_sha256({'a': [1, False], 'z': '你好'}))
        self.assertNotEqual(canonical_body_sha256({'prompt': 'hello'}),
                            canonical_body_sha256({'prompt': 'hello '}))
        with self.assertRaises(ValueError):
            canonical_body_sha256({'temperature': float('nan')})
        with self.assertRaises(ValueError):
            canonical_body_sha256({1: 'non-JSON-key'})

    def test_p95_nearest_rank_not_interpolated(self):
        values = list(range(64))
        self.assertEqual(_distribution(values)['p95'], 60)
        self.assertEqual(_distribution(values)['p50'], 31.5)


if __name__ == '__main__':
    unittest.main()
