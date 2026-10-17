"""In-memory cancellation regression tests; fake receipts confer no real gate."""
import asyncio
import base64
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import httpx

from bench.m5_ultra_256gb.pareto_realistic import client as original
from bench.m5_ultra_256gb.pareto_realistic import native_cleanup as adapter
from bench.m5_ultra_256gb.pareto_realistic.test_client_cpu import (
    jobs, payload, qualified, response, Stream,
)


class CleanupTests(unittest.IsolatedAsyncioTestCase):
    async def group(self, module, transport, *, n=1, **kwargs):
        return await module.run_group('http://mock.invalid', jobs(n), 1,
            qualification=qualified(1, n=n), phase='warm', transport=transport,
            request_timeout_s=1, group_timeout_s=1, cleanup_timeout_s=.03, **kwargs)

    def assert_invalid(self, result, task):
        self.assertTrue(all(row['done'] for row in result['requests']))
        self.assertFalse(result['cleanup']['complete'])
        self.assertTrue(any(e['task'] == task and e['type'] == 'CancelledError'
                            for e in result['cleanup']['errors']))
        self.assertFalse(result['summary']['eligible_under_supplied_qualification'])
        self.assertFalse(result['summary']['formal_point_eligible'])
        self.assertIsNone(result['summary']['aggregate_output_tok_s'])

    async def test_response_close_cancelled_after_done_old_false_pass_new_rejected(self):
        class CancelResponse(httpx.Response):
            async def aclose(self):
                raise asyncio.CancelledError('independent response close cancelled')
        async def handler(request):
            return CancelResponse(200, stream=Stream(payload(request.headers['X-Request-ID'])))
        old = await self.group(original, httpx.MockTransport(handler))
        self.assertTrue(old['cleanup']['complete'])  # Reproduce the frozen defect.
        self.assertTrue(old['summary']['eligible_under_supplied_qualification'])
        with tempfile.TemporaryDirectory() as out:
            result = await self.group(adapter, httpx.MockTransport(handler), output_dir=Path(out) / 'run')
            self.assert_invalid(result, 'close:r0')
            saved = json.loads((Path(out) / 'run/group.json').read_text())
            self.assertEqual(saved, result)
            self.assertEqual(b''.join(base64.b64decode(x['body_bytes_base64'])
                for x in saved['requests'][0]['raw_sse_chunks']), b''.join(payload('r0')))

    async def test_client_close_cancelled_after_done_old_false_pass_new_rejected(self):
        class CancelClient(httpx.MockTransport):
            async def aclose(self):
                raise asyncio.CancelledError('independent client close cancelled')
        async def handler(request):
            return response(Stream(payload(request.headers['X-Request-ID'])))
        old = await self.group(original, CancelClient(handler))
        self.assertTrue(old['cleanup']['complete'])
        self.assertTrue(old['summary']['eligible_under_supplied_qualification'])
        result = await self.group(adapter, CancelClient(handler))
        self.assert_invalid(result, 'close:client')

    async def test_done_refills_before_close_but_client_closes_after_responses(self):
        second_started = asyncio.Event()
        streams = []
        class FirstStream(Stream):
            async def aclose(self):
                await second_started.wait()
                await super().aclose()
        async def handler(request):
            rid = request.headers['X-Request-ID']
            if rid == 'r1':
                second_started.set()
            stream = (FirstStream if rid == 'r0' else Stream)(payload(rid))
            streams.append(stream)
            return response(stream)
        class CheckClose(httpx.MockTransport):
            async def aclose(self):
                self.after_responses = all(s.closed for s in streams)
        before = (original._cleanup, original._request, original.run_group)
        transport = CheckClose(handler)
        result = await self.group(adapter, transport, n=2)
        self.assertTrue(result['cleanup']['complete'])
        self.assertTrue(transport.after_responses)
        self.assertTrue(result['summary']['eligible_under_supplied_qualification'])
        self.assertEqual(result['cleanup']['adapter'], adapter.CLEANUP_REVISION)
        self.assertEqual(result['schema'], 'E023-http-group-v1')
        self.assertEqual(result['summary']['group_window_s'],
            max(r['done_perf_counter_s'] for r in result['requests'])
            - min(r['start_perf_counter_s'] for r in result['requests']))
        self.assertEqual(before, (original._cleanup, original._request, original.run_group))
        self.assertIs(adapter.run_group.__globals__, adapter._BASE.__dict__)
        self.assertEqual(adapter._BASE._request.__code__.co_code, original._request.__code__.co_code)

    async def test_resistant_response_close_is_bounded_blocks_client_and_retains_pending(self):
        release = asyncio.Event()
        class Resistant(Stream):
            async def aclose(self):
                try:
                    await release.wait()
                except asyncio.CancelledError:
                    await release.wait()
                self.closed = True
        class CheckClose(httpx.MockTransport):
            client_closed = False
            async def aclose(self):
                self.client_closed = True
        async def handler(request):
            return response(Resistant(payload(request.headers['X-Request-ID'])))
        transport = CheckClose(handler)
        try:
            result = await asyncio.wait_for(self.group(adapter, transport), .5)
            self.assertFalse(result['cleanup']['complete'])
            self.assertTrue(result['cleanup']['deadline_exceeded'])
            self.assertIn('close:r0', result['cleanup']['pending_tasks'])
            self.assertFalse(result['cleanup']['client_close_started'])
            self.assertFalse(transport.client_closed)
            self.assertIsNone(result['summary']['aggregate_output_tok_s'])
        finally:
            release.set()
            pending = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()
                       and t.get_name() == 'close:r0']
            if pending:
                await asyncio.wait_for(asyncio.gather(*pending, return_exceptions=True), .2)

    async def test_controller_cancel_retains_partial_rows_and_raises_adapter_exception(self):
        entered = asyncio.Event()
        class Waiting(Stream):
            async def __aiter__(self):
                entered.set()
                await asyncio.Event().wait()
                yield b''
        async def handler(request):
            return response(Waiting([]))
        task = asyncio.create_task(self.group(adapter, httpx.MockTransport(handler)))
        await entered.wait()
        task.cancel()
        with self.assertRaises(adapter.GroupCancelled) as caught:
            await task
        result = caught.exception.result
        self.assertEqual(result['requests'][0]['status'], 'cancelled')
        self.assertTrue(result['cleanup']['complete'])  # Owned worker cancel is expected.
        self.assertFalse(result['summary']['eligible_under_supplied_qualification'])
        self.assertIsNone(result['summary']['aggregate_output_tok_s'])

    async def test_changed_frozen_source_is_rejected_before_private_exec(self):
        with patch.object(Path, 'read_bytes', return_value=b'raise AssertionError("executed")'):
            with self.assertRaisesRegex(ValueError, 'frozen HTTP client source changed'):
                adapter._private_base()


if __name__ == '__main__':
    unittest.main()
