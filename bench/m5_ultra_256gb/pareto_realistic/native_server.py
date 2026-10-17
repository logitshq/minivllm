"""E023 target-only batching HTTP shim; model imports exist only in main().

One GPU worker drains arrivals before every engine step. Every rid has its own
delivery queue. Normal EOS is retained in raw IDs but excluded from completion
usage. An active disconnect invalidates this benchmark server: the core has no
cancel API, so this shim does not edit scheduler or KV state behind its back.
"""
import argparse
from dataclasses import dataclass, field
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import queue
import socket
import sys
import threading
import traceback
import uuid


@dataclass
class Job:
    rid: str
    prompt_ids: list[int]
    cap: int
    events: queue.Queue = field(default_factory=queue.Queue)
    cancelled: bool = False
    admitted: bool = False
    finished: bool = False


class BatchWorker:
    """Only the worker thread calls add_request/step; HTTP threads submit/cancel."""
    def __init__(self, trace=None):
        self.engine = self.sampling = None
        self.identity = {}
        self.ready = threading.Event()
        self.pending = queue.Queue()
        self.jobs, self.active = {}, {}
        self.lock = threading.Lock()
        self.failure = None
        self.trace = trace

    def initialize(self, engine, sampling, identity):
        with self.lock:
            if self.failure is not None:
                raise RuntimeError(self.failure)
            self.engine, self.sampling, self.identity = engine, sampling, identity
            self.ready.set()

    def submit(self, prompt_ids, cap):
        job = Job(uuid.uuid4().hex, list(prompt_ids), cap)
        with self.lock:
            if not self.ready.is_set() or self.failure is not None:
                raise RuntimeError(self.failure or 'model not ready')
            self.jobs[job.rid] = job
            self.pending.put(job)
        return job

    def fail(self, message):
        with self.lock:
            if self.failure is not None:
                return
            self.failure = str(message)
            for job in self.jobs.values():
                job.events.put({'error': self.failure})
            self.jobs.clear()
            self.active.clear()
            self.pending.put(None)  # Wake an idle worker; no stranded queue.get.
            self.ready.set()

    def cancel(self, job, reason='HTTP disconnected'):
        with self.lock:
            if job.finished or job.cancelled:
                return
            job.cancelled = True
            active = job.admitted
            if not active:
                self.jobs.pop(job.rid, None)
                job.events.put({'error': reason})
        if active:
            self.fail(f'active request {job.rid} cancelled: {reason}')

    def tick(self, *, block=False):
        """One deterministic drain/admit/step boundary; CPU mocks call it directly."""
        if self.failure is not None:
            return False
        incoming = []
        if not self.engine.has_work():
            if self.active:
                raise RuntimeError('engine lost unfinished requests')
            try:
                first = self.pending.get(block=block)
            except queue.Empty:
                return False
            if first is None:
                return False
            incoming.append(first)
        while True:
            try:
                job = self.pending.get_nowait()
            except queue.Empty:
                break
            if job is None:
                return False
            incoming.append(job)
        admitted = []
        for job in incoming:
            with self.lock:
                if self.failure is not None:
                    return False
                if job.cancelled:
                    continue
                job.admitted = True
                self.active[job.rid] = job
            rid = self.engine.add_request(job.prompt_ids, self.sampling(
                max_tokens=job.cap, temperature=0, ignore_eos=False), req_id=job.rid)
            if rid != job.rid:
                raise RuntimeError('engine changed the request ID')
            admitted.append(rid)
        if self.failure is not None:
            return False
        if not self.engine.has_work():
            if self.active:
                raise RuntimeError('engine failed to schedule submitted requests')
            return True
        if self.trace is not None:
            self.trace({'event': 'before_step', 'newly_submitted_to_scheduler': admitted,
                        'scheduler_running': [r.req_id for r in self.engine.scheduler.running],
                        'scheduler_waiting': [r.req_id for r in self.engine.scheduler.waiting]})
        results = self.engine.step()
        retired = []
        for rid, token, done in results:
            with self.lock:
                if self.failure is not None:
                    return False
                job = self.active.get(rid)
                if job is None:
                    raise RuntimeError(f'output for unknown or retired request {rid}')
                if type(token) is not int or type(done) is not bool:
                    raise TypeError('engine output must be (rid, int token, bool done)')
                job.events.put({'token': token, 'done': done})
                if done:
                    job.finished = True
                    self.active.pop(rid)
                    self.jobs.pop(rid)
                    retired.append(rid)
        if self.trace is not None:
            self.trace({'event': 'after_step', 'results': results, 'retired': retired,
                        'surviving_decode_rows': list(self.engine.runner.batch.req_ids)})
        return True

    def run(self):
        try:
            while self.failure is None:
                self.tick(block=True)
        except BaseException:
            self.fail(traceback.format_exc())


def observe_runner(engine, trace):
    """Qualification only: record the actual SchedulerOutput passed to ModelRunner."""
    original = engine.runner.run
    def observed(out):
        trace({'event': 'actual_scheduler_output',
               'decode': [r.req_id for r in out.decode],
               'prefill': [{'rid': r.req_id, 'tokens': n} for r, n in out.prefill],
               'finished_ids': list(out.finished_ids)})
        result = original(out)
        trace({'event': 'actual_model_runner_batch', 'decode_rows': list(engine.runner.batch.req_ids)})
        return result
    engine.runner.run = observed


def completion(job, raw_ids, eos_ids):
    if not raw_ids or len(raw_ids) > job.cap:
        raise ValueError('empty or over-cap engine completion')
    if any(token in eos_ids for token in raw_ids[:-1]):
        raise ValueError('tokens were emitted after EOS')
    eos = raw_ids[-1] in eos_ids
    if not eos and len(raw_ids) != job.cap:
        raise ValueError('engine finished before cap without EOS')
    committed = raw_ids[:-1] if eos else list(raw_ids)
    return {'id': 'cmpl-' + job.rid,
        'choices': [{'index': 0, 'text': '', 'finish_reason': 'stop' if eos else 'length'}],
        'usage': {'prompt_tokens': len(job.prompt_ids), 'completion_tokens': len(committed),
                  'total_tokens': len(job.prompt_ids) + len(committed),
                  'prompt_tokens_details': {'cached_tokens': 0}},
        'benchmark_request_id': job.rid, 'benchmark_prompt_ids': job.prompt_ids,
        'benchmark_output_ids': list(raw_ids), 'benchmark_committed_output_ids': committed,
        'benchmark_terminal_eos_id': raw_ids[-1] if eos else None}


def parse_request(body, tokenizer, max_len):
    if not isinstance(body, dict):
        raise ValueError('request must be an object')
    known = {'model', 'prompt', 'temperature', 'max_tokens', 'repetition_penalty',
             'top_p', 'stream', 'stream_options', 'n'}
    if set(body) - known:
        raise ValueError('unsupported request fields: ' + ', '.join(sorted(set(body) - known)))
    if (body.get('temperature', 0) != 0 or body.get('repetition_penalty', 1.0) != 1.0
            or body.get('top_p', 1.0) != 1.0 or body.get('n', 1) != 1):
        raise ValueError('benchmark accepts greedy, unmodified logits and one completion only')
    if body.get('stream') is not True or not isinstance(body.get('prompt'), str):
        raise ValueError('stream=true and a string prompt are required')
    if body.get('stream_options', {'include_usage': True}) != {'include_usage': True}:
        raise ValueError('stream_options must request include_usage=true only')
    cap = body.get('max_tokens')
    if type(cap) is not int or cap < 1:
        raise ValueError('max_tokens must be a positive integer')
    ids = tokenizer.encode(body['prompt'], add_special_tokens=False).ids
    if not ids or len(ids) + cap > max_len:
        raise ValueError('empty prompt or prompt + max_tokens exceeds configured max_len')
    return ids, cap


def handler_class(worker, tokenizer, *, max_len, tcp_nodelay, io_timeout):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'
        disable_nagle_algorithm = tcp_nodelay
        timeout = io_timeout

        def setup(self):
            try:
                super().setup()
                actual = self.connection.getsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY)
                if actual != int(tcp_nodelay):
                    raise RuntimeError(f'TCP_NODELAY readback {actual} != {int(tcp_nodelay)}')
            except BaseException:
                worker.fail('accepted socket setup failed: ' + traceback.format_exc())
                raise

        def log_message(self, *unused):
            pass

        def do_GET(self):
            if self.path != '/v1/models':
                self.send_error(404)
                return
            if not worker.ready.is_set() or worker.failure is not None:
                self.send_error(503)
                return
            payload = json.dumps({'data': [{'id': 'native-target-only'}],
                                  'benchmark_identity': worker.identity}).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_POST(self):
            if self.path != '/v1/completions':
                self.send_error(404)
                return
            if not worker.ready.is_set() or worker.failure is not None:
                self.send_error(503)
                return
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= 8 << 20:
                    raise ValueError('Content-Length must be between 1 byte and 8 MiB')
                body = json.loads(self.rfile.read(size))
                ids, cap = parse_request(body, tokenizer, max_len)
            except (ValueError, TypeError, KeyError) as error:
                self.send_error(400, str(error))
                return
            job, finished = None, False
            try:
                job = worker.submit(ids, cap)
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.send_header('Cache-Control', 'no-cache')
                self.send_header('Connection', 'close')
                self.end_headers()
                def send(value):
                    self.wfile.write(('data: ' + json.dumps(value, ensure_ascii=False) + '\n\n').encode())
                    self.wfile.flush()
                raw, previous = [], ''
                eos_ids = set(worker.identity['eos_ids'])
                while True:
                    item = job.events.get(timeout=io_timeout)
                    if 'error' in item:
                        send({'error': item['error']})
                        return
                    token, done = item['token'], item['done']
                    raw.append(token)
                    if token in eos_ids and not done:
                        raise ValueError('engine emitted a nonterminal EOS')
                    visible = raw[:-1] if token in eos_ids else raw
                    # Hide only terminal EOS. Preserve reasoning/control tokens
                    # in the wire text; usage counts committed IDs directly.
                    decoded = tokenizer.decode(visible, skip_special_tokens=False)
                    if not done:
                        decoded = decoded.rstrip('\ufffd')
                    if not decoded.startswith(previous):
                        raise ValueError('non-monotone cumulative decoding')
                    piece, previous = decoded[len(previous):], decoded
                    if piece:
                        send({'id': 'cmpl-' + job.rid,
                              'choices': [{'index': 0, 'text': piece, 'finish_reason': None}]})
                    if done:
                        send(completion(job, raw, eos_ids))
                        self.wfile.write(b'data: [DONE]\n\n')
                        self.wfile.flush()
                        finished = True
                        return
            except (BrokenPipeError, ConnectionResetError, TimeoutError, queue.Empty) as error:
                if job is not None:
                    worker.cancel(job, str(error))
            except BaseException:
                worker.fail('HTTP delivery failed: ' + traceback.format_exc())
            finally:
                self.close_connection = True
                if job is not None and not finished:
                    worker.cancel(job)
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--port', required=True, type=int)
    parser.add_argument('--package-root', required=True, type=Path)
    parser.add_argument('--source-manifest', required=True, type=Path)
    parser.add_argument('--source-manifest-sha256', required=True)
    for name in ('max-len', 'max-seqs', 'max-batched-tokens', 'cache-limit-gib'):
        parser.add_argument('--' + name, required=True, type=int)
    for name in ('compile-decode', 'fuse-decode', 'gates-compile', 'norm-reuse', 'tcp-nodelay'):
        parser.add_argument('--' + name, required=True, choices=('off', 'on'))
    parser.add_argument('--io-timeout-seconds', required=True, type=float)
    parser.add_argument('--qualification-trace', type=Path)
    args = parser.parse_args()
    if not (args.max_len > 0 and args.max_seqs > 0 and args.max_batched_tokens >= args.max_seqs
            and args.cache_limit_gib >= 0 and math.isfinite(args.io_timeout_seconds)
            and args.io_timeout_seconds > 0 and 0 < args.port < 65536):
        parser.error('invalid capacity, allocator, timeout or port')
    if args.fuse_decode == 'on' and args.compile_decode != 'on':
        parser.error('fuse-decode requires compile-decode')
    expected_sha = hashlib.sha256(args.source_manifest.read_bytes()).hexdigest()
    if expected_sha != args.source_manifest_sha256:
        raise RuntimeError('native source manifest SHA mismatch')
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(args.model / 'tokenizer.json'))
    trace_file = args.qualification_trace.open('x') if args.qualification_trace else None
    def trace(event):
        trace_file.write(json.dumps(event) + '\n')
        trace_file.flush()
    worker = BatchWorker(trace=trace if trace_file else None)

    def load_and_run():
        try:
            sys.path.insert(0, str(args.package_root.resolve()))
            import mlx.core as mx
            import tutorial_code.miniengine as package
            from tutorial_code.miniengine.engine.llm_engine import LLMEngine
            from tutorial_code.miniengine.sampling_params import SamplingParams
            root = Path(package.__file__).resolve().parent
            actual = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                      for path in root.rglob('*.py')}
            if actual != json.loads(args.source_manifest.read_text())['files']:
                raise RuntimeError('loaded native source snapshot differs from manifest')
            engine = LLMEngine(str(args.model), max_seqs=args.max_seqs, max_len=args.max_len,
                max_batched_tokens=args.max_batched_tokens, dtype=mx.bfloat16,
                cache_limit_gb=args.cache_limit_gib, drafter_dir=None,
                compile_decode=args.compile_decode == 'on', fuse_decode=args.fuse_decode == 'on')
            for layer in engine.model.layers:
                if not layer.is_attn:
                    layer.linear_attn.set_gates_compile(args.gates_compile == 'on')
                    layer.linear_attn.set_norm_reuse(args.norm_reuse == 'on')
            if engine.runner.spec is not None:
                raise RuntimeError('target-only server unexpectedly loaded a drafter')
            configuration = {name: getattr(args, name) for name in
                ('max_len', 'max_seqs', 'max_batched_tokens', 'cache_limit_gib', 'compile_decode',
                 'fuse_decode', 'gates_compile', 'norm_reuse', 'tcp_nodelay', 'io_timeout_seconds')}
            identity = {'configuration': configuration, 'source_files': actual,
                'source_manifest_sha256': expected_sha, 'manifest_all_sources_match': True,
                'package_root_loaded': str(root), 'model_snapshot_path': str(args.model.resolve()),
                'model_snapshot_name': args.model.resolve().name,
                'model_config_sha256': hashlib.sha256((args.model / 'config.json').read_bytes()).hexdigest(),
                'server_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'step_observer': trace_file is not None, 'per_token_observer': False,
                'qualification_trace_enabled': trace_file is not None, 'drafter': None,
                'normal_eos': True, 'eos_ids': sorted(engine.eos_ids), 'prefix_cache': False,
                'completion_usage': 'committed target IDs excluding terminal EOS; raw IDs include EOS',
                'text_decode': 'skip_special_tokens=False; terminal EOS excluded',
                'cancellation': 'pending skipped; active cancellation permanently fails this benchmark server',
                'weights_identity_scope': 'external controller must verify model weight pins; this server checks source/config identity'}
            if trace_file:
                trace({'event': 'startup', 'identity': identity})
                observe_runner(engine, trace)
            mx.synchronize()
            mx.clear_cache()  # Once at load; never between requests.
            worker.initialize(engine, SamplingParams, identity)
            worker.run()
        except BaseException:
            worker.fail(traceback.format_exc())

    threading.Thread(target=load_and_run, daemon=True).start()
    class Server(ThreadingHTTPServer):
        request_queue_size = 128
        daemon_threads = True
    server = Server(('127.0.0.1', args.port), handler_class(worker, tokenizer,
        max_len=args.max_len, tcp_nodelay=args.tcp_nodelay == 'on', io_timeout=args.io_timeout_seconds))
    try:
        server.serve_forever()
    finally:
        worker.fail('server shutdown')
        server.server_close()


if __name__ == '__main__':
    main()
