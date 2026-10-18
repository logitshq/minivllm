"""Default-OFF, C1-only diagnostic HTTP entry with explicit drained closure.

No-argument invocation describes the contract. Explicit original server arguments
plus --diagnostic-out and --drain-token start a server. This is never a formal
point. All modes share admission/DONE/drain bookkeeping; runner/full add taps.
Normal stop is an Event, not BatchWorker.failure. SIGTERM cannot flush a profile.
"""
import argparse
import hashlib
from http.server import ThreadingHTTPServer
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import threading
import time
import traceback

HERE = Path(__file__).resolve().parent
COMBINED = HERE.parent / 'gdn_graph_combined/http_server.py'
FIXED = {COMBINED: '287f34561c83437823923cf45b67f0aa6811f7eba911ee375f62519ba508ef64',
    HERE / 'instrumentation.py': '58873f2eb395a0ca7d8ab1bac35f75b67581332187ada9be3de74fbdf77ef1bf',
    HERE / 'delivery.py': '6d9e3efe46db08e1783ae851787bd79120a03b7501aa65cbf67b4be266b4e685'}
SCHEMA = 'C1-pipeline-diagnostic-http-v1'
DONE = b'data: [DONE]\n\n'
DRAIN_PATH = '/_diagnostic/drain'


def need(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path, name):
    need(sha(path) == FIXED[path], 'frozen diagnostic dependency changed: ' + str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def contract(mode='off', warm_requests=0):
    return dict(schema=SCHEMA, profile=mode, expected_requests=8, warm_requests=warm_requests,
        concurrency=1, drain_path=DRAIN_PATH, drain_header='X-Diagnostic-Drain-Token',
        normal_eos=True, performance_qualified=False, numerical_qualified=False,
        shared_control='OFF/runner/full share request, terminal-flush and drain bookkeeping',
        warm_scope='First warm_requests submissions are warm; remaining eight are diagnostic requests',
        limitations=['Python wall spans overlap; not exclusive GPU time',
                    'OFF control overhead remains; runner/full observer overhead must be measured separately',
                    'DONE flush is local successful write/flush, not remote receipt',
                    'Drained proves no further worker/handler submission, not completion of every async ahead GPU operation',
                    'External owner must enforce deadline, process reap and client raw/count correctness'])


class Drain:
    """Bounded scalar receipts; never retains Job, model, tensor or handler objects."""
    def __init__(self, token, *, warm_requests=0):
        need(type(token) is str and len(token) >= 32, 'at least 32-character drain token required')
        need(type(warm_requests) is int and warm_requests in (0, 8), 'warm requests must be 0 or 8')
        self.token, self.warm = token, warm_requests
        self.total = warm_requests + 8
        self.lock, self.local = threading.RLock(), threading.local()
        self.stop = threading.Event()
        self.phase, self.attempts = 'OPEN', 0
        self.handlers, self.active = set(), {}
        self.rows, self.errors = [], []
        self.accepted_at = self.stopped_at = None

    def handler_enter(self, key):
        with self.lock:
            need(key not in self.handlers, 'duplicate handler')
            self.handlers.add(key)

    def handler_exit(self, key):
        with self.lock:
            need(key in self.handlers and key not in self.active, 'unclosed completion handler')
            self.handlers.remove(key)

    def begin(self, key):
        with self.lock:
            need(self.phase == 'OPEN', 'diagnostic admission closed')
            # The client may see DONE and submit again before the old handler's
            # finally runs. Only unflushed completions overlap the POST→DONE
            # interval; all old handlers still block final drain until joined.
            need(key in self.handlers and key not in self.active
                 and all(row['done_flush_ns'] is not None for row in self.active.values()),
                 'C1 requires one completion before its normal DONE flush')
            need(self.attempts < self.total, 'fixed diagnostic request count exceeded')
            self.attempts += 1
            row = dict(ordinal=self.attempts, phase='warm' if self.attempts <= self.warm else 'diagnostic',
                       rid=None, entered_ns=time.perf_counter_ns(), done_flush_ns=None, handler_exit_ns=None)
            self.rows.append(row)
            self.active[key] = row
            self.local.row = row
            return row

    def submitted(self, job):
        with self.lock:
            row = getattr(self.local, 'row', None)
            need(row is not None and row['rid'] is None and self.phase == 'OPEN', 'submission outside active request')
            need(type(job.rid) is str and job.rid and all(r['rid'] != job.rid for r in self.rows), 'duplicate request ID')
            row['rid'] = job.rid

    def done(self):
        with self.lock:
            row = getattr(self.local, 'row', None)
            need(row is not None and row['rid'] is not None and row['done_flush_ns'] is None,
                 'DONE without a unique actual submission')
            row['done_flush_ns'] = time.perf_counter_ns()

    def end(self, key):
        with self.lock:
            row = self.local.row
            row['handler_exit_ns'] = time.perf_counter_ns()
            if row['done_flush_ns'] is None:
                self.errors.append('request did not complete its original DONE flush: ' + str(row['ordinal']))
            self.active.pop(key)
            del self.local.row

    def worker_empty(self, worker):
        with worker.lock:
            need(worker.ready.is_set() and worker.engine is not None and worker.failure is None, 'worker not healthy/ready')
            need(not worker.jobs and not worker.active and worker.pending.empty(), 'worker has submitted/active/pending jobs')
        need(not worker.engine.has_work(), 'engine still has work')

    def request(self, token, worker, handler_key):
        with self.lock:
            need(token == self.token, 'wrong drain token')
            need(self.phase == 'OPEN' and not self.errors, 'drain requires healthy OPEN state')
            need(self.attempts == self.total and len(self.rows) == self.total
                 and all(r['rid'] is not None and r['done_flush_ns'] is not None
                         and r['handler_exit_ns'] is not None for r in self.rows), 'not all fixed requests reached normal DONE')
            need(not self.active and self.handlers == {handler_key}, 'completion/readiness handlers still active')
            self.worker_empty(worker)
            self.phase, self.accepted_at = 'DRAIN_REQUESTED', time.perf_counter_ns()

    def stop_worker(self, worker):
        with self.lock:
            need(self.phase == 'DRAIN_REQUESTED' and not self.handlers and not self.active and not self.errors,
                 'handlers must actually stop before worker drain')
            self.worker_empty(worker)
            self.stop.set()
            worker.pending.put(None)  # Wake original tick(block=True); never call fail for normal stop.

    def verify_stopped(self, worker, thread):
        need(not thread.is_alive() and thread.ident is not None, 'worker thread has not actually stopped')
        with self.lock, worker.lock, worker.pending.mutex:
            need(self.stop.is_set() and not self.handlers and not self.active and not self.errors
                 and worker.failure is None and not worker.jobs and not worker.active
                 and len(worker.pending.queue) <= 1 and all(item is None for item in worker.pending.queue),
                 'not a clean diagnostic worker stop')
            need(not worker.engine.has_work(), 'engine work survived stop')
            self.phase, self.stopped_at = 'DRAINED', time.perf_counter_ns()

    def receipt(self):
        with self.lock:
            return dict(phase=self.phase, expected_requests=8, warm_requests=self.warm,
                attempts=self.attempts, normal_done_flushes=sum(r['done_flush_ns'] is not None for r in self.rows),
                active_handlers=len(self.handlers), active_completions=len(self.active),
                accepted_at_ns=self.accepted_at, stopped_at_ns=self.stopped_at,
                requests=[dict(row) for row in self.rows], errors=list(self.errors))


class DoneWriter:
    """Forward every original byte/flush once; count DONE only after successful flush."""
    def __init__(self, writer, drain):
        self.writer, self.drain, self.pending = writer, drain, False
    def __getattr__(self, name):
        return getattr(self.writer, name)
    def write(self, value):
        result = self.writer.write(value)
        if value == DONE:
            need(not self.pending and result == len(value), 'incomplete/duplicate DONE write')
            self.pending = True
        return result
    def flush(self):
        result = self.writer.flush()
        if self.pending:
            self.drain.done()
            self.pending = False
        return result


class SubmitTap:
    def __init__(self, worker, drain):
        self.worker, self.original = worker, worker.submit
        self.saved = ('submit' in vars(worker), vars(worker).get('submit'))
        def submit(*args, **kwargs):
            job = self.original(*args, **kwargs)
            drain.submitted(job)
            return job
        self.submit = submit
        worker.submit = submit
    def restore(self):
        need(self.worker.submit is self.submit, 'foreign submit hook')
        if self.saved[0]: self.worker.submit = self.saved[1]
        else: vars(self.worker).pop('submit', None)
        self.worker = self.original = self.saved = self.submit = None


def handler_class(base, worker, drain):
    class Handler(base):
        def setup(self):
            super().setup()
            self.wfile = DoneWriter(self.wfile, drain)
            drain.handler_enter(id(self))
            self._diagnostic_registered = True
        def finish(self):
            try: super().finish()
            finally:
                if getattr(self, '_diagnostic_registered', False):
                    drain.handler_exit(id(self))
        def do_GET(self):
            try: return super().do_GET()
            finally: self.close_connection = True  # No idle readiness connection can strand the drain.
        def do_POST(self):
            self.close_connection = True
            if self.path == DRAIN_PATH:
                try:
                    need(self.headers.get('Content-Length', '0') == '0', 'drain has no body')
                    drain.request(self.headers.get('X-Diagnostic-Drain-Token'), worker, id(self))
                except RuntimeError as exc:
                    self.send_error(409, str(exc)); return
                payload = b'{"status":"DRAIN_ACCEPTED","performance_qualified":false}\n'
                try:
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Content-Length', str(len(payload)))
                    self.send_header('Connection', 'close')
                    self.end_headers(); self.wfile.write(payload); self.wfile.flush()
                except BaseException:
                    with drain.lock: drain.errors.append('drain response failed: ' + traceback.format_exc())
                    raise
                return
            if self.path != '/v1/completions': return super().do_POST()
            try: drain.begin(id(self))
            except RuntimeError as exc:
                with drain.lock: drain.errors.append('rejected completion: ' + str(exc))
                self.send_error(409, str(exc)); return
            try: return super().do_POST()
            finally: drain.end(id(self))
    return Handler


def worker_loop(worker, stop):
    """Only change to native run is the explicit stop predicate; tick is original."""
    try:
        while worker.failure is None and not stop.is_set():
            worker.tick(block=True)
    except BaseException:
        worker.fail(traceback.format_exc())


def restore_scopes(scopes):
    errors = []
    for scope in scopes:  # SubmitTap, delivery, runner, combined, in that order.
        if scope is not None:
            try: scope.restore()
            except BaseException: errors.append(traceback.format_exc())
    return errors


def finalize(worker, thread, drain, scopes, recorder, pins, out, *, timeout,
             handlers_joined, prior_error=None):
    """Caller has closed/joined the HTTP server; no restore/write while threads live."""
    errors = [prior_error] if prior_error else []
    clean = False
    try:
        need(not errors and handlers_joined is True, 'entry failed or HTTP handlers not joined')
        drain.stop_worker(worker)
        thread.join(timeout)
        drain.verify_stopped(worker, thread)
        clean = True
    except BaseException:
        errors.append(traceback.format_exc())
        worker.fail('diagnostic aborted before clean closure')
        if thread.ident is not None: thread.join(timeout)
    stopped = thread.ident is None or not thread.is_alive()
    handlers_stopped = handlers_joined is True and not drain.handlers and not drain.active
    if stopped and handlers_stopped:
        errors.extend(restore_scopes(scopes))
    else:
        errors.append('active worker/handlers: hooks deliberately left installed; external owner must reap')
    profile = None
    if clean and not errors and recorder is not None:
        try:
            profile = recorder.write_once(out / 'profile.json', sources=pins, drained=True)
            if profile['status'] != 'RECORDED': errors.append('profile recorder is incomplete')
        except BaseException: errors.append(traceback.format_exc())
    result = dict(schema=SCHEMA, status='RECORDED' if clean and not errors else 'INCOMPLETE',
        pid=os.getpid(), drain=drain.receipt(), worker_stopped=stopped,
        handlers_stopped=handlers_stopped, worker_failure=worker.failure, errors=errors,
        profile_status=None if profile is None else profile['status'], sources=pins,
        numerical_qualified=False, performance_qualified=False)
    with (out / 'closure.json').open('x') as stream:
        json.dump(result, stream, allow_nan=False, indent=2); stream.write('\n')
    return result


def arguments(argv, combined):
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument('--profile', choices=('off', 'runner', 'full'), default='off')
    parser.add_argument('--diagnostic-out', type=Path, required=True)
    parser.add_argument('--drain-token', required=True)
    parser.add_argument('--warm-requests', type=int, choices=(0, 8), default=0)
    parser.add_argument('--worker-join-seconds', type=float, default=10.0)
    diagnostic, rest = parser.parse_known_args(argv)
    args = combined.arguments(rest)
    need(args.phase == 'formal' and args.qualification_trace is None and args.route_events is None,
         'diagnostic reuses unobserved entry only; no qualification observers')
    need(args.max_seqs == 1 and args.max_len == 2048 and args.compile_decode == args.fuse_decode
         == args.gates_compile == 'on' and args.norm_reuse == 'off', 'current C1 combined configuration required')
    need(math.isfinite(diagnostic.worker_join_seconds) and diagnostic.worker_join_seconds > 0,
         'positive bounded worker join required')
    need(len(diagnostic.drain_token) >= 32, 'drain token too short')
    return args, diagnostic


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print(json.dumps(contract(), indent=2)); return 0
    combined = load(COMBINED, '_c1_diagnostic_combined_entry')
    args, options = arguments(argv, combined)
    pins = {**combined.source_pins(), **{str(p): digest for p, digest in FIXED.items()}, str(HERE / 'http_server.py'): sha(__file__)}
    for path, digest in pins.items(): need(sha(path) == digest, 'diagnostic source pin mismatch')
    out = options.diagnostic_out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    drain = Drain(options.drain_token, warm_requests=options.warm_requests)
    donor = combined.load_fixed(combined.DONOR, '_c1_diagnostic_native_server')
    integration = combined.load_fixed(combined.INTEGRATION, '_c1_diagnostic_combined_integration')
    instrumentation = load(HERE / 'instrumentation.py', '_c1_diagnostic_instrumentation') if options.profile != 'off' else None
    delivery = load(HERE / 'delivery.py', '_c1_diagnostic_delivery') if options.profile == 'full' else None
    recorder = instrumentation.Recorder() if instrumentation else None
    need(sha(args.source_manifest) == args.source_manifest_sha256, 'native source manifest SHA mismatch')
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(args.model / 'tokenizer.json'))
    worker = donor.BatchWorker()
    delivery_scope = delivery.install(None, worker, donor, tokenizer, recorder, enabled=True) if delivery else None
    submit_scope = SubmitTap(worker, drain)
    scopes = dict(runner=None, combined=None)

    def load_and_run():
        try:
            # Same model/config construction as the pinned combined entry. The
            # scheduler, tick, sample and SSE implementation remain its donor's.
            sys.path.insert(0, str(args.package_root.resolve()))
            import mlx.core as mx
            import tutorial_code.miniengine as package
            from tutorial_code.miniengine.engine.llm_engine import LLMEngine
            from tutorial_code.miniengine.sampling_params import SamplingParams
            root = Path(package.__file__).resolve().parent
            actual = {str(p.relative_to(root)): sha(p) for p in root.rglob('*.py')}
            need(actual == json.loads(args.source_manifest.read_text())['files'], 'loaded native sources differ')
            engine = LLMEngine(str(args.model), max_seqs=args.max_seqs, max_len=args.max_len,
                max_batched_tokens=args.max_batched_tokens, dtype=mx.bfloat16,
                cache_limit_gb=args.cache_limit_gib, drafter_dir=None,
                compile_decode=args.compile_decode == 'on', fuse_decode=args.fuse_decode == 'on')
            for layer in engine.model.layers:
                if not layer.is_attn:
                    layer.linear_attn.set_gates_compile(args.gates_compile == 'on')
                    layer.linear_attn.set_norm_reuse(args.norm_reuse == 'on')
            need(engine.runner.spec is None, 'unexpected drafter')
            scopes['combined'] = integration.install(engine, gdn_prework=args.gdn_prework == 'on',
                                                       route_graph=args.route_decode_graph == 'on')
            if instrumentation:
                scopes['runner'] = instrumentation.install(engine.runner, enabled=True, recorder=recorder)
            if delivery_scope: delivery_scope.attach_engine(engine)
            identity = dict(configuration={name: getattr(args, name) for name in
                ('max_len','max_seqs','max_batched_tokens','cache_limit_gib','compile_decode',
                 'fuse_decode','gates_compile','norm_reuse','tcp_nodelay','io_timeout_seconds')},
                source_files=actual, source_manifest_sha256=args.source_manifest_sha256,
                manifest_all_sources_match=True, package_root_loaded=str(root),
                model_snapshot_path=str(args.model.resolve()), model_config_sha256=sha(args.model / 'config.json'),
                eos_ids=sorted(engine.eos_ids), drafter=None, normal_eos=True, prefix_cache=False,
                step_observer=False, qualification_trace_enabled=False,
                combined_entry=combined.identity_fields(args, combined.source_pins(),
                    installed=scopes['combined'] is not None, route_observer=False),
                benchmark_phase='diagnostic', diagnostic=contract(options.profile, options.warm_requests),
                drain_token_sha256=hashlib.sha256(options.drain_token.encode()).hexdigest(),
                server_source_path=str(HERE / 'http_server.py'), server_source_sha256=sha(__file__))
            for path, digest in pins.items(): need(sha(path) == digest, 'source changed during load')
            with (out / 'identity.json').open('x') as stream: json.dump(identity, stream, indent=2)
            mx.synchronize(); mx.clear_cache()  # Original once-at-load policy, not per request.
            worker.initialize(engine, SamplingParams, identity)
            worker_loop(worker, drain.stop)
        except BaseException:
            worker.fail(traceback.format_exc())

    base = delivery_scope.handler_class if delivery_scope else lambda **kw: donor.handler_class(worker, tokenizer, **kw)
    handler = handler_class(base(max_len=args.max_len, tcp_nodelay=args.tcp_nodelay == 'on',
                                io_timeout=args.io_timeout_seconds), worker, drain)
    class Server(ThreadingHTTPServer):
        request_queue_size = 128
        daemon_threads = False
        block_on_close = True
    thread = threading.Thread(target=load_and_run, name='diagnostic-model-worker', daemon=False)
    server = None; error = None
    try:
        server = Server(('127.0.0.1', args.port), handler)
        server.timeout = 0.1
        thread.start()
        while drain.phase == 'OPEN' and worker.failure is None:
            server.handle_request()
        need(drain.phase == 'DRAIN_REQUESTED' and worker.failure is None, worker.failure or 'no explicit drain')
    except BaseException:
        error = traceback.format_exc()
        worker.fail(error)
    finally:
        # server_close joins all non-daemon HTTP handlers, including control.
        # It can wait for the configured socket deadline; outer ownership is
        # still required for a hard process limit. Never promise SIGTERM flush.
        handlers_joined = server is None
        if server is not None:
            try:
                server.server_close()
                handlers_joined = True
            except BaseException:
                error = (error or '') + '\nserver close failed: ' + traceback.format_exc()
        result = finalize(worker, thread, drain,
            [submit_scope, delivery_scope, scopes['runner'], scopes['combined']], recorder, pins, out,
            timeout=options.worker_join_seconds, handlers_joined=handlers_joined, prior_error=error)
    if result['status'] != 'RECORDED': raise RuntimeError('diagnostic incomplete: ' + str(out / 'closure.json'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
