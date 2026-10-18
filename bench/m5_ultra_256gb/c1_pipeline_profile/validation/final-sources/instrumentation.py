"""Default-OFF C1 Python pipeline diagnostics; no device imports or extra eval.

Install before requests, after optional decoder installation. Only this runner's
methods and its graph callable dictionaries are scoped. The frozen run code uses
a PRIVATE globals dictionary with a transparent mx proxy; shared mlx is untouched.
CPU submission, existing eval wait and delivery intervals overlap. They are not
exclusive kernel times. Profiled runs cannot supply formal Pareto points.
"""
import hashlib
import json
from pathlib import Path
import sys
import threading
from time import perf_counter_ns
from types import FunctionType, MethodType

HERE = Path(__file__).resolve().parent
RUNNER_SHA = '14fd964c57814846887585a9cbbdf1e11d2500ab80c23b10c945d2aa9316b7b1'
SHORT_SHA = '22111101109b9d9c69e8fd114aa114533fe708e462303d5e72bedd6d217da1bf'
ROUTE_SHA = '4a8dc5518f2ea0a913ad0a5d753a5032324ccb69bacede416abe8ae126bfa800'
CANDIDATE_SHA = '6531be25acb53713d6ddcd930d2c778f6640a3b1f53304fd5375d20b044525c3'
SCHEMA = 'C1-pipeline-profile-v1'


def need(ok, message):
    if not ok: raise ValueError(message)


class Recorder:
    """Bounded scalar-only memory log. No disk access until write_once()."""
    def __init__(self, *, clock=perf_counter_ns, max_events=200000):
        need(type(max_events) is int and max_events > 0, 'positive event bound required')
        self.clock, self.max_events = clock, max_events
        self.rows, self.lock, self.local = [], threading.Lock(), threading.local()
        self.overflow = False; self.closed = False; self.written = False

    def event(self, kind, start, **fields):
        end = self.clock()
        row = dict(kind=kind, start_ns=start, end_ns=end,
                   thread=threading.get_ident(), **fields)
        with self.lock:
            if self.closed or len(self.rows) >= self.max_events:
                self.overflow = True
                return  # Preserve the measured program; incomplete log is never PASS.
            self.rows.append(dict(sequence=len(self.rows), **row))

    def timed(self, kind, fn, *args, fields=None, **kwargs):
        start = self.clock(); error = None
        try: return fn(*args, **kwargs)
        except BaseException as exc:
            error = type(exc).__name__; raise
        finally: self.event(kind, start, error=error, **(fields or {}))

    def write_once(self, path, *, sources, drained):
        need(self.closed and drained is True, 'restore hooks and drain worker/handlers before writing')
        need(not self.written, 'profile already written')
        self.written = True
        failed = self.overflow or any(row.get('error') for row in self.rows)
        result = dict(schema=SCHEMA, status='INCOMPLETE' if failed else 'RECORDED',
            event_count=len(self.rows), overflow=self.overflow, sources=sources, events=self.rows,
            performance_qualified=False, numerical_qualified=False,
            scope='Python wall intervals; overlapping async submission/wait/delivery; not exclusive GPU time',
            limitations=['instrumentation overhead uncorrected; require separate OFF/ON overhead check',
                        'existing eval duration can include scheduler/driver work, not only GPU waiting',
                        'graph callable first invocation can include tracing/compilation',
                        'drained is a caller assertion; external owner must prove process cleanup'])
        with Path(path).open('x') as stream:
            json.dump(result, stream, allow_nan=False, separators=(',', ':')); stream.write('\n')
        return result


class _Mx:
    def __init__(self, backend, session): self.backend, self.session = backend, session
    def __getattr__(self, name): return getattr(self.backend, name)
    def eval(self, *args, **kwargs):
        return self.session.call('existing_eval', self.backend.eval, *args, **kwargs)
    def async_eval(self, *args, **kwargs):
        return self.session.call('existing_async_eval', self.backend.async_eval, *args, **kwargs)


class _Invocation:
    def __init__(self, fn, session, graph, key):
        self.fn, self.session, self.graph, self.key = fn, session, graph, key
        self.calls = 0
    def __call__(self, *args, **kwargs):
        self.calls += 1
        return self.session.call('compiled_callable', self.fn, *args,
            fields=dict(graph=self.graph, key=repr(self.key), invocation=self.calls), **kwargs)


class _Functions(dict):
    def __init__(self, session, graph): super().__init__(); self.session, self.graph = session, graph
    def __setitem__(self, key, fn):
        need(callable(fn), 'compiled entry must be callable')
        self.session.recorder.event('callable_family_created', self.session.recorder.clock(),
                                    step=self.session.step, graph=self.graph, key=repr(key))
        super().__setitem__(key, _Invocation(fn, self.session, self.graph, key))


class Session:
    """Internal injectable scope; use source-verifying install() on actual models."""
    def __init__(self, runner, fn, recorder, graphs, sources):
        self.runner, self.recorder, self.sources = runner, recorder, sources
        self.previous = runner.decode_model
        self.saved_run = ('run' in vars(runner), vars(runner).get('run'))
        self.step = 0; self.current = None; self.closed = False; self.graphs = []
        private = dict(fn.__globals__)
        private['mx'] = _Mx(private['mx'], self)
        sample = private['sample']
        private['sample'] = lambda *a, **kw: self.call('sample_graph', sample, *a, **kw)
        clone = FunctionType(fn.__code__, private, fn.__name__, fn.__defaults__, fn.__closure__)
        clone.__kwdefaults__ = fn.__kwdefaults__
        self.cloned = MethodType(clone, runner)
        self.observed_run, self.observed_decode = self.run, self.decode
        try:
            for name, graph in graphs:
                need(type(graph._functions) is dict and not graph._functions, 'attach before graph use')
                old = graph._functions; tap = _Functions(self, name)
                self.graphs.append((graph, old, tap)); graph._functions = tap
            runner.run, runner.decode_model = self.observed_run, self.observed_decode
        except BaseException:
            for graph, old, _ in self.graphs: graph._functions = old
            raise

    def call(self, kind, fn, *args, fields=None, **kwargs):
        info = dict(step=self.step, role=(self.current or {}).get('role'))
        info.update(fields or {})
        return self.recorder.timed(kind, fn, *args, fields=info, **kwargs)

    def decode(self, tokens, ctx):
        need(self.current is not None, 'decode outside runner scope')
        used = ctx.max_len; base = self.current['used']
        role = 'current' if used == base else 'future' if used == base + 1 else 'unexpected'
        self.current['calls'].append(role); self.current['role'] = role
        domain = 'short' if 0 < used <= 256 else 'long-low' if used < 1024 else 'long-high' if used <= 1536 else 'outside'
        try:
            return self.call('decode_graph_submission', self.previous, tokens, ctx,
                fields=dict(used=used, batch=tokens.shape[0], requested_domain=domain,
                            domain_is_not_eligibility_or_execution_proof=True))
        finally: self.current['role'] = None

    def run(self, out):
        need(self.current is None and not self.closed, 'nested or closed profile scope')
        need(len(out.decode) <= 1 and len(out.prefill) <= 1, 'C1 diagnostic only')
        self.step += 1
        positions = [r.num_computed_tokens for r in out.decode]
        rids = [r.req_id for r in out.decode]
        sampling = tuple((r.sampling.temperature, r.sampling.top_p, r.sampling.top_k,
                          r.sampling.max_tokens, r.sampling.ignore_eos,
                          tuple(r.sampling.stop_token_ids)) for r in out.decode)
        key = (rids, positions, sampling, tuple(r.output_token_ids[-1] for r in out.decode))
        ahead_present = self.runner._ahead is not None
        matched = bool(out.decode and self.runner.batch.req_ids == rids and ahead_present
                       and self.runner._ahead[0] == key)
        self.current = dict(used=max(positions)+1 if positions else 0, calls=[], role=None)
        start = self.recorder.clock(); error = None
        try: return self.cloned(out)
        except BaseException as exc: error = type(exc).__name__; raise
        finally:
            calls = self.current['calls']
            self.recorder.event('runner_step', start, step=self.step, rids=list(rids),
                prefill_rids=[r.req_id for r, _ in out.prefill], used=self.current['used'],
                prior_ahead_present=ahead_present, prior_ahead_key_matched=matched,
                ahead_consumed=bool(matched and 'current' not in calls and error is None),
                actual_decode_calls=list(calls), successor_retained=self.runner._ahead is not None,
                error=error)
            self.current = None

    def restore(self):
        if self.closed: return
        need(self.current is None, 'cannot restore while runner active')
        need(self.runner.run is self.observed_run and self.runner.decode_model is self.observed_decode,
             'restore foreign observers before profile')
        need(all(graph._functions is tap for graph, _, tap in self.graphs), 'foreign graph observer')
        for graph, old, tap in self.graphs:
            old.update({key: wrapper.fn for key, wrapper in tap.items()}); graph._functions = old
            tap.clear()
        self.runner.decode_model = self.previous
        if self.saved_run[0]: self.runner.run = self.saved_run[1]
        else: vars(self.runner).pop('run', None)
        self.closed = True; self.recorder.closed = True
        self.runner = self.previous = self.cloned = self.observed_run = self.observed_decode = None
        self.saved_run = None; self.graphs.clear()

    def __enter__(self): return self
    def __exit__(self, *unused): self.restore(); return False


def _verify_class(obj, filename, digest, sources):
    cls = type(obj); initializer = cls.__init__
    need(type(initializer) is FunctionType, 'Python native class required')
    namespace = initializer.__globals__
    # The frozen E024 factory deliberately uses a private module not registered
    # in sys.modules. Bind its actual function globals and file, not an alias.
    need(namespace.get(cls.__name__) is cls and namespace.get('__name__') == cls.__module__,
         'actual loaded class required')
    path = Path(namespace['__file__']).resolve()
    need(path.name == filename and hashlib.sha256(path.read_bytes()).hexdigest() == digest,
         'source differs: '+str(path))
    need(Path(initializer.__code__.co_filename).resolve() == path, 'foreign class initializer')
    call = vars(cls).get('__call__')
    if call is not None:
        need(type(call) is FunctionType and call.__globals__ is namespace
             and Path(call.__code__.co_filename).resolve() == path, 'foreign decoder callable')
    sources[str(path)] = digest
    return namespace


def install(runner, *, enabled=False, clock=perf_counter_ns, max_events=200000, recorder=None):
    """OFF touches nothing. ON supports pinned native eager/short/E024 decoders."""
    if type(enabled) is not bool: raise TypeError('enabled must be bool')
    if not enabled: return None
    sources = {str(Path(__file__).resolve()): hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    namespace = _verify_class(runner, 'model_runner.py', RUNNER_SHA, sources)
    fn = type(runner).run
    need(type(fn) is FunctionType and fn.__globals__ is namespace
         and Path(fn.__code__.co_filename).resolve() == Path(namespace['__file__']).resolve()
         and runner.run.__func__ is fn and 'run' not in vars(runner), 'unwrapped native runner required')
    need(namespace['mx'] is sys.modules.get('mlx.core'), 'actual shared MLX backend required')
    need(runner.spec is None and runner._ahead is None and not runner.batch.req_ids
         and not runner.staging and not runner.feats, 'fresh target-only runner required')
    decoder = runner.decode_model; graphs = []
    if decoder is not runner.model:
        if type(decoder).__name__ == '_Dispatch':
            _verify_class(decoder, 'integration.py', ROUTE_SHA, sources)
            _verify_class(decoder.long, 'candidate.py', CANDIDATE_SHA, sources)
            need(decoder.long.model is runner.model, 'foreign long model')
            graphs.append(('long', decoder.long)); decoder = decoder.original
        if decoder is not runner.model:
            _verify_class(decoder, 'compiled_decode.py', SHORT_SHA, sources)
            need(decoder.model is runner.model, 'foreign short model')
            graphs.append(('short', decoder))
    if recorder is None: recorder = Recorder(clock=clock, max_events=max_events)
    need(type(recorder) is Recorder and not recorder.closed and not recorder.written, 'active recorder required')
    return Session(runner, fn, recorder, graphs, sources)
