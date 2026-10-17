"""Prepared E024 HTTP entry v2; default OFF, no qualification implied.

Reuse the exact frozen E023 worker, handler, queue, SSE and completion semantics.
Only argument/load/identity glue is local. ON installs the pinned per-instance
decoder after existing layer flags and before the optional scheduling observer.
The native context limit stays 2048; the candidate retains its own length/cache
guards. E028 remains OFF. Installation is not proof of compiled invocation.

Formal mode forbids both observers. Qualification requires the original step
trace and, for route ON, a separate actual-call sidecar. Original four-event
scheduling traces remain unchanged. SUBMITTED is routing evidence, not proof
of completed arithmetic. Independent capture/replay qualification is required
before formal use; frozen factories cannot simply replace their command path.

No model imports or service operations occur on import. The external owner is
responsible for source/model/runtime pins, resource admission, deadlines and
TERM/KILL/wait/strict absence. Python cleanup is best effort, not a reap receipt.
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
import traceback


HERE = Path(__file__).resolve().parent
DONOR = HERE.parent / 'pareto_realistic/native_server.py'
INTEGRATION = HERE / 'integration.py'
WITNESS = HERE / 'serving_qualification.py'
COUNTER = HERE.parent / 'long_decode_graph/qualify.py'
ROUTE_SCHEMA = 'E024-route-http-events-v1'
FIXED = {
    DONOR: '9bcb37eff8bdecd79800afdedc82deacf9f7a0a9a7a1e0d62dc8ba49e70fb68f',
    INTEGRATION: '4a8dc5518f2ea0a913ad0a5d753a5032324ccb69bacede416abe8ae126bfa800',
    HERE / 'candidate.py': '6531be25acb53713d6ddcd930d2c778f6640a3b1f53304fd5375d20b044525c3',
    HERE.parent / 'long_decode_graph/candidate.py': 'eb14b671a66764fe677219762090acb495b89d384cf078f3f63b918a40aa2b84',
    HERE.parent / 'long_decode_graph/model-source-pins.json': '7143538a168bfcd8b01292c9687472c8647db393c6d2a510d628f71c670260c5',
    WITNESS: '75de7205febdae2698e99353bfc2908a79044591fbf67abb30a07cb4e54a7264',
    COUNTER: 'c24219f431b08502a26a4f7a83c002214bddd7ea3962d6b1869b313e4c13faf7',
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_pins():
    for path, digest in FIXED.items():
        if sha(path) != digest:
            raise RuntimeError('fixed HTTP adapter source changed: ' + str(path))
    return {**{str(path): digest for path, digest in FIXED.items()},
            str(Path(__file__).resolve()): sha(__file__)}


def load_fixed(path, name):
    if sha(path) != FIXED[path]:
        raise RuntimeError('fixed module source changed: ' + str(path))
    if name in sys.modules:
        raise RuntimeError('private HTTP adapter module already loaded: ' + name)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # The donor's dataclass uses its real module.
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def arguments(argv=None):
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
    parser.add_argument('--route-events', type=Path)
    parser.add_argument('--phase', choices=('formal', 'qualification'), default='formal')
    parser.add_argument('--route-decode-graph', choices=('off', 'on'), default='off')
    args = parser.parse_args(argv)
    if not (args.max_len > 0 and args.max_seqs > 0 and args.max_batched_tokens >= args.max_seqs
            and args.cache_limit_gib >= 0 and math.isfinite(args.io_timeout_seconds)
            and args.io_timeout_seconds > 0 and 0 < args.port < 65536):
        parser.error('invalid capacity, allocator, timeout or port')
    if args.fuse_decode == 'on' and args.compile_decode != 'on':
        parser.error('fuse-decode requires compile-decode')
    if (args.phase == 'qualification') != (args.qualification_trace is not None):
        parser.error('qualification requires its step trace; formal forbids observers')
    wants_route_events = args.phase == 'qualification' and args.route_decode_graph == 'on'
    if wants_route_events != (args.route_events is not None):
        parser.error('route-events required exactly for qualification with route ON')
    if args.route_events is not None and args.route_events.resolve() == args.qualification_trace.resolve():
        parser.error('route-events and qualification-trace must be separate files')
    if args.route_decode_graph == 'on' and not (
            args.max_len == 2048 and args.max_seqs in (1, 8)
            and args.compile_decode == args.fuse_decode == args.gates_compile == 'on'
            and args.norm_reuse == 'off'):
        parser.error('route ON requires current C1/C8 context2048, short+fuse+gates ON, norm OFF')
    return args


def identity_fields(args, pins, *, installed, route_observer):
    """Pure identity additions for the factory and actual loaded server.

    The server passes actual handle/observer booleans. A factory may construct
    the expected values; neither claims an invocation before requests begin.
    """
    enabled = args.route_decode_graph == 'on'
    if type(installed) is not bool or enabled != installed:
        raise RuntimeError('route installation does not match requested enablement')
    observed = args.phase == 'qualification' and enabled
    if type(route_observer) is not bool or route_observer != observed:
        raise RuntimeError('route observer does not match requested phase')
    route = dict(schema='E024-route-http-adapter-v2', enabled=enabled,
        installed=installed, sources=pins,
        bounds=[1023, 1536], eligible_batch_range=[1, 8], eligible_used_range=[257, 1536],
        minimum_physical_kv_capacity=1536, identity_refill_enabled=False,
        route_observer=route_observer, actual_compiled_invocations_observed=False,
        witness_source_sha256=pins[str(WITNESS)], witness_counter_source_sha256=pins[str(COUNTER)],
        route_events_schema=ROUTE_SCHEMA,
        qualification_scope='external independent route/model/HTTP qualification required')
    return dict(server_source_sha256=pins[str(Path(__file__).resolve())],
        server_source_path=str(Path(__file__).resolve()),
        frozen_server_source_sha256=FIXED[DONOR], frozen_server_source_path=str(DONOR),
        benchmark_phase=args.phase, route_decode_graph=route)


class QualificationObserver:
    """Single-worker observer scope; restore only after that worker has stopped.

    Reuses the pinned actual-call counter without a second eligibility check or
    model evaluation. A failed construction rolls back its own partial hooks.
    """
    def __init__(self, engine, installation, donor, trace_file, route_file, phase,
                 witness_module=None, counter_module=None):
        self.runner = engine.runner
        self.saved_run = ('run' in vars(self.runner), vars(self.runner).get('run'))
        self.trace_file, self.route_file, self.phase = trace_file, route_file, phase
        self.step = 0
        self.closed = False
        self.witness = self.graph = self.dispatch = self.saved_functions = self.observed_run = None
        self.coverage = dict(route_hits={}, long_fallback_calls=0)
        try:
            if route_file is not None:
                if installation is None or witness_module is None or counter_module is None:
                    raise RuntimeError('route observation requires installed graph and pinned witness modules')
                self.dispatch = installation.decoder
                self.graph = self.dispatch.long
                self.saved_functions = self.graph._functions
                self.witness = witness_module.RouteWitness(self.graph, counter_module,
                    self.emit_route, self.coverage)
                self.witness.phase = phase
                self.dispatch.long = self.witness
            donor.observe_runner(engine, self.trace)
            self.observed_run = self.runner.run
        except BaseException:
            self._restore_run()
            if self.graph is not None:
                self.graph._functions = self.saved_functions
                self.dispatch.long = self.graph
            raise

    def _restore_run(self):
        if self.saved_run[0]:
            self.runner.run = self.saved_run[1]
        else:
            vars(self.runner).pop('run', None)

    def emit_route(self, row):
        self.route_file.write(json.dumps(row, allow_nan=False) + '\n')
        self.route_file.flush()

    def trace(self, event):
        if event.get('event') == 'before_step':
            self.step += 1
            if self.witness is not None:
                self.witness.step = self.step
                self.witness.direct_label = None
        self.trace_file.write(json.dumps(event, allow_nan=False) + '\n')
        self.trace_file.flush()

    def startup(self, identity):
        if self.route_file is not None:
            self.emit_route(dict(event='route-startup', schema=ROUTE_SCHEMA,
                pid=os.getpid(), phase=self.phase, identity=identity))
        self.trace(dict(event='startup', identity=identity))

    def restore(self):
        if self.closed:
            return
        if self.runner.run is not self.observed_run:
            raise RuntimeError('foreign runner observer prevents cleanup')
        self._restore_run()  # Scheduling observer first, then route witness.
        if self.witness is not None:
            if self.dispatch.long is not self.witness or self.graph._functions is not self.witness.functions:
                raise RuntimeError('foreign graph observer prevents cleanup')
            self.dispatch.long = self.graph
            self.graph._functions = self.saved_functions
        self.runner = self.dispatch = self.graph = self.witness = self.saved_functions = None
        self.observed_run = self.saved_run = None
        self.closed = True


def main(argv=None):
    args = arguments(argv)
    pins = source_pins()
    donor = load_fixed(DONOR, '_e024_http_v2_frozen_native_server')
    integration = load_fixed(INTEGRATION, '_e024_http_v2_route_integration')
    witness_module = counter_module = None
    if args.route_events is not None:
        witness_module = load_fixed(WITNESS, '_e024_http_v2_frozen_route_witness')
        counter_module = load_fixed(COUNTER, '_e024_http_v2_frozen_call_counter')
    expected_sha = sha(args.source_manifest)
    if expected_sha != args.source_manifest_sha256:
        raise RuntimeError('native source manifest SHA mismatch')
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(args.model / 'tokenizer.json'))
    trace_file = args.qualification_trace.open('x') if args.qualification_trace else None
    try:
        route_file = args.route_events.open('x') if args.route_events else None
    except BaseException:
        if trace_file is not None:
            trace_file.close()
        raise
    observation = None

    def trace(event):
        if observation is None:
            raise RuntimeError('scheduling trace used before observer installation')
        observation.trace(event)

    worker = donor.BatchWorker(trace=trace if trace_file else None)

    def load_and_run():
        nonlocal observation
        engine = installation = None
        try:
            sys.path.insert(0, str(args.package_root.resolve()))
            import mlx.core as mx
            import tutorial_code.miniengine as package
            from tutorial_code.miniengine.engine.llm_engine import LLMEngine
            from tutorial_code.miniengine.sampling_params import SamplingParams
            root = Path(package.__file__).resolve().parent
            actual = {str(path.relative_to(root)): sha(path) for path in root.rglob('*.py')}
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
            # Default OFF does not inspect or mutate the runner. ON is fresh-only.
            installation = integration.install(engine.runner, enabled=args.route_decode_graph == 'on')
            if trace_file is not None:
                observation = QualificationObserver(engine, installation, donor, trace_file, route_file,
                    'HTTP-C' + str(args.max_seqs), witness_module, counter_module)
            configuration = {name: getattr(args, name) for name in
                ('max_len', 'max_seqs', 'max_batched_tokens', 'cache_limit_gib', 'compile_decode',
                 'fuse_decode', 'gates_compile', 'norm_reuse', 'tcp_nodelay', 'io_timeout_seconds')}
            identity = {'configuration': configuration, 'source_files': actual,
                'source_manifest_sha256': expected_sha, 'manifest_all_sources_match': True,
                'package_root_loaded': str(root), 'model_snapshot_path': str(args.model.resolve()),
                'model_snapshot_name': args.model.resolve().name,
                'model_config_sha256': sha(args.model / 'config.json'),
                'step_observer': trace_file is not None, 'per_token_observer': False,
                'qualification_trace_enabled': trace_file is not None, 'drafter': None,
                'normal_eos': True, 'eos_ids': sorted(engine.eos_ids), 'prefix_cache': False,
                'completion_usage': 'committed target IDs excluding terminal EOS; raw IDs include EOS',
                'text_decode': 'skip_special_tokens=False; terminal EOS excluded',
                'cancellation': 'pending skipped; active cancellation permanently fails this benchmark server',
                'weights_identity_scope': 'external controller must verify model weight pins; this server checks source/config identity'}
            identity.update(identity_fields(args, pins, installed=installation is not None,
                route_observer=observation is not None and observation.witness is not None))
            if source_pins() != pins:
                raise RuntimeError('HTTP adapter sources changed during model load')
            if observation is not None:
                observation.startup(identity)
            mx.synchronize()
            mx.clear_cache()  # Same once-at-load policy; never between requests.
            worker.initialize(engine, SamplingParams, identity)
            worker.run()
        except BaseException:
            worker.fail(traceback.format_exc())
        finally:
            # Only this worker can submit model work. Its run() has now returned
            # (or startup failed). Never restore the installation from main while
            # this thread may still be loading or stepping the engine.
            try:
                if observation is not None:
                    observation.restore()
                if installation is not None:
                    installation.restore()
            except BaseException:
                message = 'HTTP adapter worker cleanup failed: ' + traceback.format_exc()
                worker.fail(message)
                print(message, file=sys.stderr, flush=True)
            finally:
                if trace_file is not None:
                    trace_file.close()
                if route_file is not None:
                    route_file.close()

    class Server(ThreadingHTTPServer):
        request_queue_size = 128
        daemon_threads = True

    thread = threading.Thread(target=load_and_run, daemon=True)
    server = None
    try:
        server = Server(('127.0.0.1', args.port), donor.handler_class(worker, tokenizer,
            max_len=args.max_len, tcp_nodelay=args.tcp_nodelay == 'on', io_timeout=args.io_timeout_seconds))
        thread.start()
        server.serve_forever()
    finally:
        worker.fail('server shutdown')
        if server is not None:
            server.server_close()
        if thread.ident is None:
            if trace_file is not None:
                trace_file.close()
            if route_file is not None:
                route_file.close()
        # No main-thread model cleanup, no assumed join/reap, no signal handler:
        # the external direct-child owner retains the hard termination boundary.


if __name__ == '__main__':
    main()
