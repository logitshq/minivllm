"""E023 CPU contract tests. No listener, actual tokenizers, MLX, or models.

The engine is a deterministic test double with two finite slots. These tests
verify HTTP/worker routing and lifecycle, not GPU math or real scheduler state.
"""
import importlib.util
import io
import json
from pathlib import Path
import queue
import subprocess
import sys
import threading
import types
import unittest
from unittest.mock import patch


PATH = Path(__file__).with_name('native_server.py')
SPEC = importlib.util.spec_from_file_location('e023_native_cpu_target', PATH)
server = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = server
SPEC.loader.exec_module(server)


class Engine:
    """Mock engine: prompt[0] selects a token script, capped per request."""
    def __init__(self, scripts=None, slots=2):
        self.scripts = scripts or {1: [10, 11, 12, 13], 2: [20, 99], 3: [30, 31, 32]}
        self.slots = slots
        self.scheduler = types.SimpleNamespace(waiting=[], running=[])
        self.runner = types.SimpleNamespace(batch=types.SimpleNamespace(req_ids=[]))
        self.calls, self.batches = [], []
        self.on_step = None

    def add_request(self, ids, sampling, req_id):
        self.calls.append((threading.get_ident(), 'add', req_id, sampling))
        self.scheduler.waiting.append(types.SimpleNamespace(
            req_id=req_id, script=self.scripts[ids[0]], cap=sampling.max_tokens, at=0))
        return req_id

    def has_work(self):
        return bool(self.scheduler.running or self.scheduler.waiting)

    def step(self):
        self.calls.append((threading.get_ident(), 'step'))
        while self.scheduler.waiting and len(self.scheduler.running) < self.slots:
            self.scheduler.running.append(self.scheduler.waiting.pop(0))
        self.batches.append([r.req_id for r in self.scheduler.running])
        if self.on_step is not None:
            callback, self.on_step = self.on_step, None
            callback()
        result, surviving = [], []
        for r in self.scheduler.running:
            token = r.script[r.at % len(r.script)]
            r.at += 1
            done = token == 99 or r.at == r.cap
            result.append((r.req_id, token, done))
            if not done:
                surviving.append(r)
        self.scheduler.running = surviving
        self.runner.batch.req_ids = [r.req_id for r in surviving]
        return result


class Tokenizer:
    def encode(self, text, add_special_tokens):
        assert add_special_tokens is False
        return types.SimpleNamespace(ids=[] if not text else [int(text[0])])

    def decode(self, ids, skip_special_tokens):
        assert skip_special_tokens is False
        return ''.join({10: 'a', 11: 'b', 12: '<think>', 20: 'z', 99: '<EOS>'}.get(i, str(i)) for i in ids)


def worker(engine=None, trace=None):
    value = server.BatchWorker(trace)
    value.initialize(engine or Engine(), types.SimpleNamespace, {'eos_ids': [99]})
    return value


def events(job):
    result = []
    while True:
        try:
            result.append(job.events.get_nowait())
        except queue.Empty:
            return result


def fake_handler(worker_, payload=None, tokenizer=None, writer=None):
    cls = server.handler_class(worker_, tokenizer or Tokenizer(),
        max_len=64, tcp_nodelay=True, io_timeout=.02)
    obj = cls.__new__(cls)
    obj.path = '/v1/completions'
    body = json.dumps(payload or {
        'model': 'local', 'prompt': '1 prompt', 'temperature': 0,
        'max_tokens': 3, 'stream': True, 'stream_options': {'include_usage': True}}).encode()
    obj.headers = {'Content-Length': str(len(body))}
    obj.rfile, obj.wfile = io.BytesIO(body), writer or io.BytesIO()
    obj.responses = []
    obj.send_response = lambda status: obj.responses.append(status)
    obj.send_header = lambda *_: None
    obj.end_headers = lambda: None
    obj.send_error = lambda code, *args: obj.responses.append(code)
    return obj


class WorkerTests(unittest.TestCase):
    def test_admit_together_retire_independently_and_refill(self):
        engine = Engine()
        w = worker(engine)
        a, b, c = w.submit([1], 4), w.submit([2], 4), w.submit([3], 2)
        w.tick()
        self.assertEqual(engine.batches, [[a.rid, b.rid]])
        self.assertEqual(len(engine.scheduler.waiting), 1)
        w.tick()  # b ends at EOS while a continues.
        self.assertTrue(b.finished)
        self.assertFalse(a.finished)
        w.tick()  # c refills b's slot; a remains live.
        self.assertEqual(engine.batches[-1], [a.rid, c.rid])
        w.tick()
        self.assertFalse(w.active)
        self.assertFalse(w.jobs)
        self.assertEqual(events(a), [{'token': x, 'done': x == 13} for x in [10, 11, 12, 13]])
        self.assertEqual(events(b), [{'token': 20, 'done': False}, {'token': 99, 'done': True}])
        self.assertEqual(events(c), [{'token': 30, 'done': False}, {'token': 31, 'done': True}])
        self.assertTrue(all(call[3].ignore_eos is False and call[3].temperature == 0
                            for call in engine.calls if call[1] == 'add'))

    def test_arrival_during_step_enters_next_boundary(self):
        engine, delivered = Engine(), []
        w = worker(engine)
        a = w.submit([1], 3)
        engine.on_step = lambda: delivered.append(w.submit([3], 1))
        w.tick()
        b = delivered[0]
        self.assertEqual(engine.batches[-1], [a.rid])
        self.assertFalse(b.admitted)
        w.tick()
        self.assertEqual(engine.batches[-1], [a.rid, b.rid])
        self.assertTrue(b.finished)
        self.assertFalse(a.finished)

    def test_concurrent_http_submitters_never_touch_engine(self):
        w, jobs, gate = worker(), [], threading.Barrier(9)
        def submit(i):
            gate.wait(timeout=2)
            jobs.append(w.submit([i % 3 + 1], 1))
        threads = [threading.Thread(target=submit, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        gate.wait(timeout=2)
        for t in threads:
            t.join(2)
            self.assertFalse(t.is_alive())
        self.assertEqual(len(jobs), 8)
        self.assertFalse(w.engine.calls)
        while not all(j.finished for j in jobs):
            w.tick()
        self.assertEqual({c[0] for c in w.engine.calls}, {threading.get_ident()})
        self.assertEqual(len({j.rid for j in jobs}), 8)
        self.assertTrue(all(len(events(j)) == 1 for j in jobs))

    def test_pending_cancel_does_not_cancel_survivor(self):
        w = worker()
        a, b = w.submit([1], 2), w.submit([2], 1)
        w.cancel(a)
        w.tick()
        self.assertFalse(a.admitted)
        self.assertTrue(b.finished)
        self.assertIsNone(w.failure)
        self.assertIn('error', events(a)[0])
        self.assertEqual([c[2] for c in w.engine.calls if c[1] == 'add'], [b.rid])

    def test_active_cancel_permanently_wakes_all_and_idle_worker(self):
        w = worker()
        a, b = w.submit([1], 3), w.submit([2], 4)
        w.tick()
        c = w.submit([3], 3)
        w.cancel(a, 'disconnected')
        for j in (a, b, c):
            self.assertIn('error', events(j)[-1])
        self.assertIn('active request', w.failure)
        self.assertFalse(w.tick())
        with self.assertRaises(RuntimeError):
            w.submit([1], 1)
        idle = worker()
        t = threading.Thread(target=idle.run, daemon=True)
        t.start()
        idle.fail('test stop')
        t.join(2)
        self.assertFalse(t.is_alive())

    def test_finished_disconnect_is_already_released(self):
        w = worker()
        j = w.submit([1], 1)
        w.tick()
        w.cancel(j)
        self.assertIsNone(w.failure)
        self.assertFalse(w.active)

    def test_add_or_step_exception_fails_all_queued_requests(self):
        for name in ('add_request', 'step'):
            with self.subTest(name=name):
                w = worker()
                a, b = w.submit([1], 3), w.submit([2], 3)
                with patch.object(w.engine, name, side_effect=RuntimeError('injected ' + name)):
                    w.run()
                self.assertIn('injected ' + name, w.failure)
                self.assertTrue(all('error' in events(j)[-1] for j in (a, b)))

    def test_wrong_rid_malformed_outputs_and_missing_retirement_fail(self):
        for results in ([('unknown', 10, False)], None, [('CURRENT', True, False)], [('CURRENT', 10, 1)]):
            with self.subTest(results=results):
                w = worker()
                j = w.submit([1], 3)
                output = [(j.rid if rid == 'CURRENT' else rid, tok, done)
                          for rid, tok, done in results] if results else []
                def broken_step():
                    w.engine.scheduler.waiting.clear()
                    w.engine.scheduler.running.clear()
                    return output
                with patch.object(w.engine, 'step', side_effect=broken_step):
                    w.run()
                self.assertIsNotNone(w.failure)
                self.assertIn('error', events(j)[-1])

    def test_observers_are_opt_in(self):
        w = worker()
        j = w.submit([1], 1)
        w.tick()
        self.assertIsNone(w.trace)
        record = []
        observed = worker(trace=record.append)
        j = observed.submit([1], 1)
        observed.tick()
        self.assertEqual([r['event'] for r in record], ['before_step', 'after_step'])
        self.assertEqual(record[-1]['retired'], [j.rid])
        self.assertEqual(record[-1]['surviving_decode_rows'], [])
        engine = Engine()
        engine.runner.run = lambda _: {'mock': 3}
        records = []
        server.observe_runner(engine, records.append)
        out = types.SimpleNamespace(decode=[types.SimpleNamespace(req_id='d')],
            prefill=[(types.SimpleNamespace(req_id='p'), 127)], finished_ids=['f'])
        self.assertEqual(engine.runner.run(out), {'mock': 3})
        self.assertEqual(records[0], {'event': 'actual_scheduler_output',
            'decode': ['d'], 'prefill': [{'rid': 'p', 'tokens': 127}], 'finished_ids': ['f']})


class HTTPTests(unittest.TestCase):
    def test_eos_raw_and_committed_accounting(self):
        j = server.Job('r', [7, 8], 3)
        eos = server.completion(j, [10, 99], {99})
        self.assertEqual(eos['benchmark_output_ids'], [10, 99])
        self.assertEqual(eos['benchmark_committed_output_ids'], [10])
        self.assertEqual(eos['usage']['completion_tokens'], 1)
        self.assertEqual(eos['usage']['total_tokens'], 3)
        self.assertEqual(eos['choices'][0]['finish_reason'], 'stop')
        zero = server.completion(j, [99], {99})
        self.assertEqual(zero['usage']['completion_tokens'], 0)
        cap = server.completion(j, [10, 11, 12], {99})
        self.assertEqual(cap['choices'][0]['finish_reason'], 'length')
        self.assertEqual(cap['usage']['completion_tokens'], 3)
        self.assertEqual(server.completion(j, [10, 11, 99], {99})['choices'][0]['finish_reason'], 'stop')
        for ids in ([], [10], [99, 10], [10, 11, 12, 13]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                server.completion(j, ids, {99})

    def test_parse_variable_cap_and_reject_silent_contract_changes(self):
        body = {'prompt': '1 input', 'stream': True, 'max_tokens': 7}
        self.assertEqual(server.parse_request(body, Tokenizer(), 8), ([1], 7))
        for changes in ({'max_tokens': 8}, {'max_tokens': True}, {'max_tokens': 0},
                        {'prompt': ''}, {'stream': False}, {'temperature': .1},
                        {'top_p': .9}, {'repetition_penalty': 1.1}, {'n': 2},
                        {'ignore_eos': True}, {'stop': ['end']},
                        {'stream_options': {'include_usage': False}}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                server.parse_request(body | changes, Tokenizer(), 8)

    def test_stream_final_ids_eos_usage_and_special_tokens(self):
        w = worker()
        submit = w.submit
        def completed(ids, cap):
            j = submit(ids, cap)
            while not j.finished:
                w.tick()
            return j
        with patch.object(w, 'submit', side_effect=completed):
            h = fake_handler(w)
            h.do_POST()
        wire = h.wfile.getvalue().decode()
        self.assertEqual(h.responses, [200])
        rows = [json.loads(line[6:]) for line in wire.splitlines()
                if line.startswith('data: ') and line != 'data: [DONE]']
        self.assertEqual(''.join(r['choices'][0]['text'] for r in rows), 'ab<think>')
        self.assertEqual(rows[-1]['benchmark_output_ids'], [10, 11, 12])
        self.assertEqual(rows[-1]['usage']['completion_tokens'], 3)
        self.assertTrue(wire.endswith('data: [DONE]\n\n'))
        self.assertIsNone(w.failure)
        with patch.object(w, 'submit', side_effect=completed):
            h = fake_handler(w, {'prompt': '2 prompt', 'max_tokens': 4, 'stream': True})
            h.do_POST()
        wire = h.wfile.getvalue().decode()
        rows = [json.loads(line[6:]) for line in wire.splitlines()
                if line.startswith('data: ') and line != 'data: [DONE]']
        self.assertNotIn('<EOS>', wire)
        self.assertEqual(rows[-1]['benchmark_output_ids'], [20, 99])
        self.assertEqual(rows[-1]['benchmark_committed_output_ids'], [20])
        self.assertEqual(rows[-1]['choices'][0]['finish_reason'], 'stop')

    def test_partial_unicode_waits_for_complete_decoding(self):
        class PartialTokenizer(Tokenizer):
            def decode(self, ids, skip_special_tokens):
                return ['\ufffd', 'é', 'éx'][len(ids) - 1]
        w, submit = worker(), None
        submit = w.submit
        def completed(ids, cap):
            j = submit(ids, cap)
            while not j.finished:
                w.tick()
            return j
        with patch.object(w, 'submit', side_effect=completed):
            h = fake_handler(w, tokenizer=PartialTokenizer())
            h.do_POST()
        self.assertIsNone(w.failure)
        self.assertNotIn('\ufffd', h.wfile.getvalue().decode())
        self.assertIn('é', h.wfile.getvalue().decode())

    def test_broken_delivery_and_queue_timeout_release_waiters(self):
        class BrokenWriter(io.BytesIO):
            def write(self, data):
                raise BrokenPipeError('mock broken socket')
        for broken in (False, True):
            with self.subTest(broken=broken):
                w = worker()
                original, observed = w.submit, []
                def admitted(ids, cap):
                    j = original(ids, cap)
                    observed.append(j)
                    w.tick()  # One token then no worker; HTTP must not hang.
                    return j
                h = fake_handler(w, writer=BrokenWriter() if broken else None)
                with patch.object(w, 'submit', side_effect=admitted):
                    h.do_POST()
                self.assertIn('active request', w.failure)
                self.assertTrue(h.close_connection)
                self.assertFalse(w.jobs)
                self.assertIn('error', events(observed[0])[-1])

    def test_queued_error_has_no_done_and_bad_input_never_submits(self):
        w = worker()
        original = w.submit
        def failed(ids, cap):
            j = original(ids, cap)
            w.fail('mock engine failure')
            return j
        with patch.object(w, 'submit', side_effect=failed):
            h = fake_handler(w)
            h.do_POST()
        self.assertIn('mock engine failure', h.wfile.getvalue().decode())
        self.assertNotIn('[DONE]', h.wfile.getvalue().decode())
        w = worker()
        h = fake_handler(w, {'prompt': '1 prompt', 'stream': True, 'max_tokens': 0})
        h.do_POST()
        self.assertEqual(h.responses, [400])
        self.assertFalse(w.jobs)

    def test_models_identity_and_failed_readiness(self):
        w = worker()
        w.identity.update(configuration={'max_seqs': 64}, source_manifest_sha256='mock',
                          qualification_trace_enabled=False)
        h = fake_handler(w)
        h.path = '/v1/models'
        h.do_GET()
        self.assertEqual(json.loads(h.wfile.getvalue())['benchmark_identity'], w.identity)
        w.fail('injected')
        h = fake_handler(w)
        h.path = '/v1/models'
        h.do_GET()
        self.assertEqual(h.responses, [503])

    def test_boolean_socket_readback_accepts_nonzero_flag(self):
        for enabled, raw in ((False, 0), (True, 1), (True, 4)):
            with self.subTest(enabled=enabled, raw=raw):
                w = worker()
                cls = server.handler_class(w, Tokenizer(), max_len=64,
                    tcp_nodelay=enabled, io_timeout=1)
                h = cls.__new__(cls)
                h.connection = types.SimpleNamespace(getsockopt=lambda *args: raw)
                with patch.object(server.BaseHTTPRequestHandler, 'setup'):
                    h.setup()
                self.assertIsNone(w.failure)

    def test_socket_readback_mismatch_is_permanent_even_if_thread_swallows(self):
        for enabled, raw in ((True, 0), (False, 1), (False, 4)):
            with self.subTest(enabled=enabled, raw=raw):
                w = worker()
                cls = server.handler_class(w, Tokenizer(), max_len=64,
                    tcp_nodelay=enabled, io_timeout=1)
                h = cls.__new__(cls)
                h.connection = types.SimpleNamespace(getsockopt=lambda *args: raw)
                with patch.object(server.BaseHTTPRequestHandler, 'setup'), self.assertRaises(RuntimeError):
                    h.setup()
                self.assertIn('accepted socket setup failed', w.failure)
                with self.assertRaises(RuntimeError):
                    w.submit([1], 1)

    def test_help_and_unknown_flags_cannot_import_gpu_or_open_listener(self):
        code = '''
import builtins, http.server, runpy, socket, sys
original_import = builtins.__import__
def checked(name, *args, **kwargs):
    if name == 'mlx' or name.startswith('mlx.') or name == 'tokenizers':
        raise AssertionError('forbidden import: ' + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = checked
socket.socket = lambda *a, **k: (_ for _ in ()).throw(AssertionError('listener forbidden'))
sys.argv = [sys.argv[1], sys.argv[2]]
runpy.run_path(sys.argv[0], run_name='__main__')
'''
        for flag, status in (('--help', 0), ('--nonexistent-experiment', 2)):
            result = subprocess.run([sys.executable, '-B', '-c', code, str(PATH), flag],
                capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, status, result.stderr)
            self.assertNotIn('AssertionError', result.stderr)


if __name__ == '__main__':
    unittest.main()
