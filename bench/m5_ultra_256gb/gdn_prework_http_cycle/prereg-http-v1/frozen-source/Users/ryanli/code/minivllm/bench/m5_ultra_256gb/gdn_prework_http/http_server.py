"""Default-OFF GDN-only HTTP entry, preserving frozen native transport/math.

Only the48 original GDN instances are installed; E024/E028 remain OFF. The
qualification dispatch observer reads Python counters/context metadata only:
no array evaluation, token copies, or full-state snapshot inside this server.
Formal mode forbids observers but is NOT qualified by this source. An external
direct-child owner must provide actual runtime/model/resource/deadline/reap gates.
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
CANDIDATE = HERE.parent / 'gdn_prework_unbound/candidate.py'
SERVING = HERE.parent / 'gdn_prework_unbound/serving.py'
DISPATCH_SCHEMA = 'GDN-prework-http-events-v1'
FIXED = {
    DONOR: '9bcb37eff8bdecd79800afdedc82deacf9f7a0a9a7a1e0d62dc8ba49e70fb68f',
    CANDIDATE: '1bf5a837c6515ca505bc5d1beaa1718b88e1ccd52609f6814bb995afbe6c1260',
    SERVING: '3f493752d95996bd08cc1ea7389fc3159ab615e1ac162b36332eecc47dab862c',
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
    parser.add_argument('--dispatch-events', type=Path)
    parser.add_argument('--phase', choices=('formal', 'qualification'), default='formal')
    parser.add_argument('--gdn-prework', choices=('off', 'on'), default='off')
    args = parser.parse_args(argv)
    if not (args.max_len > 0 and args.max_seqs > 0 and args.max_batched_tokens >= args.max_seqs
            and args.cache_limit_gib >= 0 and math.isfinite(args.io_timeout_seconds)
            and args.io_timeout_seconds > 0 and 0 < args.port < 65536):
        parser.error('invalid capacity, allocator, timeout or port')
    if args.fuse_decode == 'on' and args.compile_decode != 'on':
        parser.error('fuse-decode requires compile-decode')
    if (args.phase == 'qualification') != (args.qualification_trace is not None):
        parser.error('qualification requires its step trace; formal forbids observers')
    wants_dispatch_events = args.phase == 'qualification' and args.gdn_prework == 'on'
    if wants_dispatch_events != (args.dispatch_events is not None):
        parser.error('dispatch-events required exactly for qualification with GDN ON')
    if args.dispatch_events is not None and args.dispatch_events.resolve() == args.qualification_trace.resolve():
        parser.error('dispatch-events and qualification-trace must be separate files')
    if args.gdn_prework == 'on' and not (
            args.max_len == 2048 and args.max_seqs in (1, 8)
            and args.compile_decode == args.fuse_decode == args.gates_compile == 'on'
            and args.norm_reuse == 'off'):
        parser.error('GDN ON requires current C1/C8 context2048, short+fuse+gates ON, norm OFF')
    return args


def identity_fields(args, pins, *, installed, dispatch_observer):
    enabled = args.gdn_prework == 'on'
    if type(installed) is not bool or enabled != installed:
        raise RuntimeError('GDN install differs from explicit switch')
    if type(dispatch_observer) is not bool or dispatch_observer != (enabled and args.phase == 'qualification'):
        raise RuntimeError('GDN observer differs from explicit phase')
    return dict(server_source_sha256=pins[str(Path(__file__).resolve())],
        server_source_path=str(Path(__file__).resolve()),
        frozen_server_source_sha256=FIXED[DONOR], frozen_server_source_path=str(DONOR),
        benchmark_phase=args.phase, gdn_prework=dict(schema='GDN-prework-http-adapter-v1',
            enabled=enabled, installed=installed, installed_layers=48 if installed else 0,
            sources=pins, dispatch_observer=dispatch_observer, dispatch_events_schema=DISPATCH_SCHEMA,
            route_graph_enabled=False, identity_refill_enabled=False,
            candidate_domain='B1 ordinary decode; original short/prefill and B>1 fallback retained',
            actual_calls_observed=False, scope='Installation only; actual dispatch and numerical replay required'))


class Installation:
    def __init__(self, handles): self.handles, self.closed = handles, False
    def restore(self):
        if self.closed: return
        first = None
        for handle in reversed(self.handles):
            try: handle.restore()
            except BaseException as exc:
                if first is None: first = exc
        self.handles.clear(); self.closed = True
        if first is not None: raise first


def install(engine, adapter, *, enabled=False):
    if type(enabled) is not bool: raise TypeError('explicit boolean switch')
    if not enabled: return None  # No engine/adapter inspection while OFF.
    if engine.runner.spec is not None or engine.requests:
        raise RuntimeError('fresh target-only model required')
    layers = [layer.linear_attn for layer in engine.model.layers if not layer.is_attn]
    if len(engine.model.layers) != 64 or len(layers) != 48:
        raise RuntimeError('actual64/48 layer model required')
    scope = Installation([])
    try:
        for layer in layers:
            handle = adapter.install(layer, sys.modules[type(layer).__module__], enabled=True)
            if handle is None: raise RuntimeError('GDN installation unexpectedly OFF')
            scope.handles.append(handle)
    except BaseException:
        scope.restore(); raise
    return scope


class CallProxy:
    def __init__(self, target, observer): self.target, self.observer = target, observer
    def __getattr__(self, name): return getattr(self.target, name)
    def __call__(self, tokens, ctx, **kwargs):
        return self.observer.call(self.target, tokens, ctx, kwargs)


class QualificationObserver:
    """Pure Python dispatch observations around actual model/decode calls.

    No logits/caches/tokens are inspected or retained. Current/future calls are
    separate actual invocations; consuming an existing ahead creates no event.
    """
    def __init__(self, engine, installation, donor, trace_file, dispatch_file, phase, serving):
        self.runner = engine.runner
        self.saved_run = ('run' in vars(self.runner), vars(self.runner).get('run'))
        self.original_model, self.original_decode = self.runner.model, self.runner.decode_model
        self.trace_file, self.dispatch_file, self.phase = trace_file, dispatch_file, phase
        self.installation, self.serving = installation, serving
        self.step = 0; self.closed = False; self.proxies = None; self.observed_run = None
        try:
            if dispatch_file is not None:
                if installation is None or serving is None or len(installation.handles) != 48:
                    raise RuntimeError('dispatch requires actual48 installed layers')
                self.proxies = (CallProxy(self.original_model,self), CallProxy(self.original_decode,self))
                self.runner.model, self.runner.decode_model = self.proxies
            donor.observe_runner(engine,self.trace); self.observed_run = self.runner.run
        except BaseException:
            self._restore_run()
            self.runner.model, self.runner.decode_model = self.original_model, self.original_decode
            raise

    def _restore_run(self):
        if self.saved_run[0]: self.runner.run = self.saved_run[1]
        else: vars(self.runner).pop('run',None)

    def emit(self,row):
        self.dispatch_file.write(json.dumps(row,allow_nan=False)+'\n');self.dispatch_file.flush()

    def call(self,target,tokens,ctx,kwargs):
        before = [dict(h.calls) for h in self.installation.handles]
        row = dict(event='dispatch-call',status='FAILED',phase=self.phase,step=self.step,
                   mode=ctx.mode,B=tokens.shape[0],used=ctx.max_len)
        try:
            result = target(tokens,ctx,**kwargs)
            after = [dict(h.calls) for h in self.installation.handles]
            row.update(status='SUBMITTED',**self.serving.dispatch_delta(before,after,ctx.mode,tokens.shape[0],ctx.max_len))
            return result
        finally: self.emit(row)

    def trace(self,event):
        if event.get('event') == 'before_step': self.step += 1
        self.trace_file.write(json.dumps(event,allow_nan=False)+'\n');self.trace_file.flush()

    def startup(self,identity):
        if self.dispatch_file is not None:
            self.emit(dict(event='dispatch-startup',schema=DISPATCH_SCHEMA,pid=os.getpid(),phase=self.phase,identity=identity))
        self.trace(dict(event='startup',identity=identity))

    def restore(self):
        if self.closed: return
        if self.runner.run is not self.observed_run: raise RuntimeError('foreign scheduling observer')
        self._restore_run()
        if self.proxies is not None:
            if self.runner.model is not self.proxies[0] or self.runner.decode_model is not self.proxies[1]:
                raise RuntimeError('foreign model observer')
            self.runner.model,self.runner.decode_model = self.original_model,self.original_decode
        self.runner=self.original_model=self.original_decode=self.proxies=self.observed_run=None
        self.saved_run=self.installation=self.serving=None;self.closed=True


def restore_scopes(observation, installation):
    first = None
    for scope in (observation, installation):
        if scope is not None:
            try: scope.restore()
            except BaseException as exc:
                if first is None: first = exc
    if first is not None: raise first


def main(argv=None):
    args = arguments(argv)
    pins = source_pins()
    donor = load_fixed(DONOR, '_gdn_http_frozen_native_server')
    adapter = serving = None
    if args.gdn_prework == 'on':
        adapter = load_fixed(CANDIDATE, '_gdn_http_frozen_candidate')
    if args.dispatch_events is not None:
        serving = load_fixed(SERVING, '_gdn_http_frozen_dispatch_contract')
    expected_sha = sha(args.source_manifest)
    if expected_sha != args.source_manifest_sha256:
        raise RuntimeError('native source manifest SHA mismatch')
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(args.model / 'tokenizer.json'))
    trace_file = args.qualification_trace.open('x') if args.qualification_trace else None
    try:
        dispatch_file = args.dispatch_events.open('x') if args.dispatch_events else None
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
            installation = install(engine, adapter, enabled=args.gdn_prework == 'on')
            if trace_file is not None:
                observation = QualificationObserver(engine, installation, donor, trace_file, dispatch_file,
                    'HTTP-C' + str(args.max_seqs), serving)
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
                dispatch_observer=observation is not None and observation.proxies is not None))
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
                restore_scopes(observation, installation)
            except BaseException:
                message = 'HTTP adapter worker cleanup failed: ' + traceback.format_exc()
                worker.fail(message)
                print(message, file=sys.stderr, flush=True)
            finally:
                if trace_file is not None:
                    trace_file.close()
                if dispatch_file is not None:
                    dispatch_file.close()

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
            if dispatch_file is not None:
                dispatch_file.close()
        # No main-thread model cleanup, no assumed join/reap, no signal handler:
        # the external direct-child owner retains the hard termination boundary.


if __name__ == '__main__':
    main()
