"""Default-OFF E026 tail staging at the existing eager runner eval boundary.

CPU-safe import; no model loader or global/core patch. Explicit use:

    with StagedPrefillTailScope(runner, enabled=True):
        runner.run(scheduled)

The pinned ModelRunner.run has one synchronous mx.eval, after all model calls
and before token delivery. Only that runner instance is temporarily rebound to
the ORIGINAL run code with private globals; its mx view forwards everything
except this one eval. Immediately before the original eval, ordinary prefill
staging caches receive lazy owned-tail copies, which join the SAME eval call.
There is no per-layer eval, GDN/conv hook, extra synchronization or extra state.
The actual grouped set excludes grouped prefill; decode-only and speculative
runs use the original bound method. Recorded/snapshotted caches are excluded.

Scope: exact pinned eager target-only runner/model, BF16 NVFP4 M5 Ultra. No
arbitrary outer mx.compile, simultaneous use, live model replacement, or model
wrapping during the scope. A retained compiled closure created in this scope
is unsupported. Numerical observers may wrap the runner AFTER scope entry and
delegate to its saved method, but must restore that wrapper before scope exit;
wrapping/replacing runner.model is deliberately unsupported.

Default OFF does not validate/import the backend or change any attribute.
Scope exit restores the exact prior run attribute and clears strong instance
references. Exceptions clear temporary eval arguments; like original run,
already-produced cache updates are not rolled back after a failed forward/eval.
Only caller-owned request caches retain their ordinary state/owned tail graphs.

This builds all prefill graphs before staging. It does not reduce graph-build
retention or promise a lower execution peak: copy work co-evaluated with outputs
may still overlap large parents, and original eval arguments can retain those
parents until that eval returns. CPU tests prove routing/lifetime mechanics
only, not actual MLX ownership, bit equality, full-model qualification or speed.
Real same-schedule full-logit/cache/ahead checks remain mandatory before HTTP
throughput/complete-latency measurements or any production adoption.
"""
import hashlib
import importlib.util
from pathlib import Path
import sys
import threading
from types import FunctionType, MethodType, SimpleNamespace

HERE = Path(__file__).resolve().parent
CANDIDATE_SHA = '4d9658d34734e3c252375c4b76fcb2eccf83555121d7eb51d7f3e81eb2c48ccc'
CORE_PINS = {
    'engine/model_runner.py': '14fd964c57814846887585a9cbbdf1e11d2500ab80c23b10c945d2aa9316b7b1',
    'cache.py': '056a3bf3bc42163e93f55eb914fa35d5c081f803da9b5faa740588a67057a791',
    'model/gdn.py': 'c88096bf3b51a1ad7a24dd4f4e2c71bbb6b220b288b83a9242ca190d6c0a5030',
    'model/qwen3_5.py': 'dae9694d717a0ebe80c9c54317c65f5d206e4d5695bc898fe0e93eb04a8590ce',
    'context.py': '134614bd19cedae748b0db415cf1feac70cebd9826a641a843936db8327d6ba7',
}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _verify_runner(runner):
    original = type(runner).run
    if type(original) is not FunctionType or original.__closure__ is not None:
        raise ValueError('exact unwrapped ModelRunner.run required')
    path = Path(original.__code__.co_filename).resolve()
    core = path.parent.parent
    if path != core / 'engine/model_runner.py' or runner.run.__func__ is not original:
        raise ValueError('untouched ModelRunner instance required')
    for relative, expected in CORE_PINS.items():
        if _sha(core / relative) != expected:
            raise ValueError('core source mismatch: ' + relative)
    module = sys.modules[type(runner).__module__]
    if Path(module.__file__).resolve() != path or original.__globals__ is not vars(module):
        raise ValueError('loaded runner source/globals mismatch')
    make_cache = original.__globals__['make_cache']
    cache_cls = make_cache.__globals__['GDNCache']
    model_call = type(runner.model).__call__
    for fn, expected in ((make_cache, core / 'cache.py'),
                         (cache_cls.__init__, core / 'cache.py'),
                         (model_call, core / 'model/qwen3_5.py')):
        if type(fn) is not FunctionType or fn.__closure__ is not None or Path(fn.__code__.co_filename).resolve() != expected:
            raise ValueError('unrecognized model/cache function')
    if _sha(HERE / 'candidate.py') != CANDIDATE_SHA:
        raise ValueError('frozen tail helper changed')
    spec = importlib.util.spec_from_file_location('_e026_staged_tail_helper', HERE / 'candidate.py')
    helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
    mx = original.__globals__['mx']
    if (not helper.supported_backend(mx)
            or runner.model.cfg.quantization != {'group_size': 16, 'bits': 4, 'mode': 'nvfp4'}
            or runner.model.embed_tokens._output_dtype != mx.bfloat16):
        raise ValueError('staging requires the pinned BF16 NVFP4 M5 Ultra backend')
    return SimpleNamespace(original=original, mx=mx, cache_cls=cache_cls, helper=helper)


class _RuntimeView:
    def __init__(self, runtime, evaluate):
        self._runtime, self.eval = runtime, evaluate

    def __getattr__(self, name):
        return getattr(self._runtime, name)


class StagedPrefillTailScope:
    def __init__(self, runner, *, enabled=False, on_boundary=None):
        if type(enabled) is not bool:
            raise TypeError('enabled must be bool')
        if on_boundary is not None and not callable(on_boundary):
            raise TypeError('on_boundary must be callable')
        self._runner, self.enabled, self.on_boundary = runner, enabled, on_boundary
        self._active = self._closed = self._running = self._installed = False
        self._out = self._model = self._saved = self._cloned = self._deps = None
        self._boundary_count = 0

    def __enter__(self):
        if self._active or self._closed:
            raise RuntimeError('scope already active or closed')
        self._thread = threading.get_ident()
        if not self.enabled:
            self._active = True
            return self
        try:
            self._deps = _verify_runner(self._runner)
            self._model = self._runner.model
            original = self._deps.original
            private = dict(original.__globals__)
            private['mx'] = _RuntimeView(self._deps.mx, self._evaluate)
            self._cloned = FunctionType(original.__code__, private, original.__name__, original.__defaults__)
            self._cloned.__kwdefaults__ = original.__kwdefaults__
            self._present, self._saved = 'run' in vars(self._runner), vars(self._runner).get('run')
            self._active = True
            object.__setattr__(self._runner, 'run', MethodType(self._run, self._runner))
            self._installed = True
        except BaseException:
            self.__exit__(*sys.exc_info())
            raise
        return self

    def _run(self, runner, out):
        if not self._active or threading.get_ident() != self._thread or runner is not self._runner:
            raise RuntimeError('inactive, foreign or cross-thread runner scope')
        if self._running:
            raise RuntimeError('reentrant eager runner unsupported')
        if runner.model is not self._model:
            raise RuntimeError('model replacement inside staging scope unsupported')
        # Capture/speculative code and all decode-only async/ahead work stay on
        # the original function/globals. There are no layer wrappers at all.
        if runner.spec is not None or not out.prefill:
            return self._deps.original(runner, out)
        self._running, self._out, self._boundary_count = True, out, 0
        try:
            result = self._cloned(runner, out)
            if self._boundary_count != 1:
                raise RuntimeError('expected one original runner eval boundary')
            return result
        finally:
            self._out, self._running = None, False

    def _evaluate(self, *args, **kwargs):
        frame = sys._getframe(1)
        try:
            if (not self._running or threading.get_ident() != self._thread
                    or frame.f_code is not self._deps.original.__code__
                    or frame.f_locals.get('self') is not self._runner
                    or frame.f_locals.get('out') is not self._out
                    or type(frame.f_locals.get('grouped')) is not set):
                raise RuntimeError('unrecognized eager eval call site')
            grouped = frame.f_locals['grouped']
        finally:
            del frame
        tails, seen = [], set()
        ordinary = skipped = 0
        try:
            for req, n in self._out.prefill:
                if req.req_id in grouped:
                    skipped += 1
                    continue
                ordinary += 1
                for cache in self._runner.staging[req.req_id]:
                    if (type(cache) is not self._deps.cache_cls
                            or set(vars(cache)) != {'conv_state', 'S', 'offset', 'record', 'tape', '_snap'}
                            or cache.record or cache.tape is not None or cache._snap is not None):
                        continue
                    if cache.offset != req.num_computed_tokens + n:
                        raise RuntimeError('prefill staging offset does not match this scheduler step')
                    if id(cache) in seen:
                        raise RuntimeError('aliased staging cache in a scheduler step')
                    seen.add(id(cache))
                    tail = cache.conv_state
                    if tail is None or tail.shape != (3, 10240) or tail.dtype != self._deps.mx.bfloat16:
                        continue
                    owned = self._deps.helper.own_conv_tail(self._deps.mx, tail, enabled=True, materialize=False)
                    if owned is tail:
                        raise RuntimeError('eligible staged tail did not take the owned-copy path')
                    cache.conv_state = owned
                    tails.append(owned)
            self._boundary_count += 1
            result = self._deps.mx.eval(*args, *tails, **kwargs)
            if self.on_boundary is not None:
                self.on_boundary(dict(ordinary_prefill_requests=ordinary, grouped_requests_skipped=skipped,
                    owned_tails=len(tails), existing_eval_calls=self._boundary_count, per_layer_eval_calls=0))
            return result
        finally:
            tails.clear()  # No staged tensor or caller frame survives this call.

    def __exit__(self, exc_type, exc, tb):
        try:
            if self._installed:
                if self._present:
                    object.__setattr__(self._runner, 'run', self._saved)
                else:
                    object.__delattr__(self._runner, 'run')
        finally:
            self._runner = self._model = self._saved = self._cloned = self._deps = self._out = None
            self.on_boundary = None
            self._active = self._running = self._installed = False
            self._closed = True
        return False
