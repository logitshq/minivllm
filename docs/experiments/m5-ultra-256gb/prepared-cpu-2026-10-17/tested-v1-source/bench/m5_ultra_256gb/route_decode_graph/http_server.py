"""Prepared E024 HTTP entry; default OFF, no HTTP/model qualification implied.

Reuse the exact frozen E023 worker, handler, queue, SSE and completion semantics.
Only argument/load/identity glue is local. ON installs the pinned per-instance
decoder after existing layer flags and before the optional scheduling observer.
The native context limit stays 2048; the candidate retains its own length/cache
guards. E028 remains OFF. Installation is not proof of compiled invocation.

Formal mode forbids observers. Qualification mode requires the original step
trace; this first entry has no route-call observer. A future source-bound HTTP
capture/replay adapter must add actual route evidence and pass independent
qualification before formal use. Frozen capture/formal factories do not accept
this new entry merely by replacing their command path.

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
from pathlib import Path
import sys
import threading
import traceback


HERE = Path(__file__).resolve().parent
DONOR = HERE.parent / 'pareto_realistic/native_server.py'
INTEGRATION = HERE / 'integration.py'
FIXED = {
    DONOR: '9bcb37eff8bdecd79800afdedc82deacf9f7a0a9a7a1e0d62dc8ba49e70fb68f',
    INTEGRATION: '4a8dc5518f2ea0a913ad0a5d753a5032324ccb69bacede416abe8ae126bfa800',
    HERE / 'candidate.py': '6531be25acb53713d6ddcd930d2c778f6640a3b1f53304fd5375d20b044525c3',
    HERE.parent / 'long_decode_graph/candidate.py': 'eb14b671a66764fe677219762090acb495b89d384cf078f3f63b918a40aa2b84',
    HERE.parent / 'long_decode_graph/model-source-pins.json': '7143538a168bfcd8b01292c9687472c8647db393c6d2a510d628f71c670260c5',
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
    if args.route_decode_graph == 'on' and not (
            args.max_len == 2048 and args.max_seqs in (1, 8)
            and args.compile_decode == args.fuse_decode == args.gates_compile == 'on'
            and args.norm_reuse == 'off'):
        parser.error('route ON requires current C1/C8 context2048, short+fuse+gates ON, norm OFF')
    return args


def route_identity(args, installation, pins):
    enabled = args.route_decode_graph == 'on'
    if enabled != (installation is not None):
        raise RuntimeError('route installation does not match requested enablement')
    return dict(schema='E024-route-http-adapter-v1', enabled=enabled,
        installed=installation is not None, sources=pins,
        bounds=[1023, 1536], eligible_batch_range=[1, 8], eligible_used_range=[257, 1536],
        minimum_physical_kv_capacity=1536, identity_refill_enabled=False,
        route_observer=False, actual_compiled_invocations_observed=False,
        qualification_scope='external independent route/model/HTTP qualification required')


def main(argv=None):
    args = arguments(argv)
    pins = source_pins()
    donor = load_fixed(DONOR, '_e024_http_frozen_native_server')
    integration = load_fixed(INTEGRATION, '_e024_http_route_integration')
    expected_sha = sha(args.source_manifest)
    if expected_sha != args.source_manifest_sha256:
        raise RuntimeError('native source manifest SHA mismatch')
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(args.model / 'tokenizer.json'))
    trace_file = args.qualification_trace.open('x') if args.qualification_trace else None

    def trace(event):
        trace_file.write(json.dumps(event) + '\n')
        trace_file.flush()

    worker = donor.BatchWorker(trace=trace if trace_file else None)

    def load_and_run():
        engine = installation = observed_run = saved_run = None
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
            configuration = {name: getattr(args, name) for name in
                ('max_len', 'max_seqs', 'max_batched_tokens', 'cache_limit_gib', 'compile_decode',
                 'fuse_decode', 'gates_compile', 'norm_reuse', 'tcp_nodelay', 'io_timeout_seconds')}
            identity = {'configuration': configuration, 'source_files': actual,
                'source_manifest_sha256': expected_sha, 'manifest_all_sources_match': True,
                'package_root_loaded': str(root), 'model_snapshot_path': str(args.model.resolve()),
                'model_snapshot_name': args.model.resolve().name,
                'model_config_sha256': sha(args.model / 'config.json'),
                'server_source_sha256': pins[str(Path(__file__).resolve())],
                'server_source_path': str(Path(__file__).resolve()),
                'frozen_server_source_sha256': FIXED[DONOR], 'frozen_server_source_path': str(DONOR),
                'benchmark_phase': args.phase,
                'route_decode_graph': route_identity(args, installation, pins),
                'step_observer': trace_file is not None, 'per_token_observer': False,
                'qualification_trace_enabled': trace_file is not None, 'drafter': None,
                'normal_eos': True, 'eos_ids': sorted(engine.eos_ids), 'prefix_cache': False,
                'completion_usage': 'committed target IDs excluding terminal EOS; raw IDs include EOS',
                'text_decode': 'skip_special_tokens=False; terminal EOS excluded',
                'cancellation': 'pending skipped; active cancellation permanently fails this benchmark server',
                'weights_identity_scope': 'external controller must verify model weight pins; this server checks source/config identity'}
            if source_pins() != pins:
                raise RuntimeError('HTTP adapter sources changed during model load')
            if trace_file:
                trace({'event': 'startup', 'identity': identity})
                saved_run = ('run' in vars(engine.runner), vars(engine.runner).get('run'))
                donor.observe_runner(engine, trace)
                observed_run = engine.runner.run
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
                if observed_run is not None:
                    if engine.runner.run is not observed_run:
                        raise RuntimeError('foreign runner observer prevents cleanup')
                    if saved_run[0]:
                        engine.runner.run = saved_run[1]
                    else:
                        vars(engine.runner).pop('run', None)
                if installation is not None:
                    installation.restore()
            except BaseException:
                message = 'HTTP adapter worker cleanup failed: ' + traceback.format_exc()
                worker.fail(message)
                print(message, file=sys.stderr, flush=True)
            finally:
                if trace_file is not None:
                    trace_file.close()

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
        if thread.ident is None and trace_file is not None:
            trace_file.close()
        # No main-thread model cleanup, no assumed join/reap, no signal handler:
        # the external direct-child owner retains the hard termination boundary.


if __name__ == '__main__':
    main()
