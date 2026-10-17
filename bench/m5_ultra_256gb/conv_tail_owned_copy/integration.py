"""E026 scoped eager single-sequence prefill integration, default copy OFF.

CPU-safe import. No model loader, MLX import, device run, or production patch.
Use only the pinned eager ModelRunner / GatedDeltaNet route, outside mx.compile:

    layers = {i: b.linear_attn for i, b in enumerate(model.layers) if not b.is_attn}
    with PrefillTailScope(gdn_module, layers, enabled=True, materialize=False,
                         on_call=receipts.append):
        runner.run(scheduled)  # ordinary eager runner, never mx.compile(...)

For the first single-layer diagnostic, scope.single_layer(i, x, cache) is an
explicit eager entry. A caller must NOT compile either entry or retain compiled
closures made inside the scope. MLX 0.32.3 exposes no Python tracing predicate;
this is a source-pinned execution contract, not a universal tracing detector.
The frozen CompiledDecode calls only ctx.mode='decode', which never enters this
hook. It is permitted to remain configured on the ordinary runner.

Only selected instances' _prefill attributes are temporary. Each replacement
uses the ORIGINAL _prefill code object with a private globals dictionary whose
sole replacement is causal_conv1d. That function calls the ORIGINAL convolution
exactly once, returns its y object unchanged, and passes only its [3,10240] BF16
tail to the frozen helper. The module global, class methods, _prefill_batch,
_decode, _decode_fused, and all arithmetic stay unchanged. Finally restores exact
instance attributes. Same-thread, non-nested use only. Callback rows contain
scalars/shape lists only, never tensors or cache/model references. Observer work
is diagnostic and is not a performance measurement.

Next real-run qualification (NOT executed or qualified by this module):
1. Register fresh sequential off/on-lazy/on-materialize runs, the same pinned
   model, input, runtime, sources and resource owner. Single GDN layer first:
   real projection inputs at T=1/19/935/978, fresh and continuing cache. Compare
   convolution y, returned GDN output, conv_state, S, offset and unchanged
   parameters using original-dtype unsigned views, shape/dtype and finite gates.
   Archive full first-difference/partial evidence; no tolerances or retries.
2. Then one fixed real D fixture through all 64 layers, with every one of 48 GDN
   layers witnessed exactly once on ordinary non-batched prefill. Compare each
   layer output and KV/GDN state, final full logits and raw token IDs bitwise.
   Use identical observer snapshots in arms; evaluate the same products before
   comparisons. Explicitly evaluate ON-lazy tails at that common validation
   boundary; only ON-materialize evaluates at the hook. Do not infer release
   while an oracle/snapshot still owns the source. Separately record deduplicated
   get_array_buffer_size and active/cache before and after dropping those refs.
3. Verify callback absence on normal decode (B1/B8), grouped prefill, recording,
   capture/spec/verify and calls outside the scope; restore on injected failure.
   These finite diagnostics do not replace the original 37-request same-schedule
   E023 qualification. Full logits, committed/staging/ahead state and lifecycle
   gates must still pass before HTTP correctness/throughput/latency qualification.
   Tiny backing PASS alone proves neither real-model qualification nor E023
   pressure causation. No predicted throughput or memory-peak result is claimed.
"""
import hashlib
import importlib.util
from pathlib import Path
import sys
import threading
from types import FunctionType, MethodType

HERE = Path(__file__).resolve().parent
CANDIDATE_SHA = '4d9658d34734e3c252375c4b76fcb2eccf83555121d7eb51d7f3e81eb2c48ccc'
CORE_PINS = {
    'model/gdn.py': 'c88096bf3b51a1ad7a24dd4f4e2c71bbb6b220b288b83a9242ca190d6c0a5030',
    'model/qwen3_5.py': 'dae9694d717a0ebe80c9c54317c65f5d206e4d5695bc898fe0e93eb04a8590ce',
    'context.py': '134614bd19cedae748b0db415cf1feac70cebd9826a641a843936db8327d6ba7',
    'engine/model_runner.py': '14fd964c57814846887585a9cbbdf1e11d2500ab80c23b10c945d2aa9316b7b1',
    'engine/compiled_decode.py': '22111101109b9d9c69e8fd114aa114533fe708e462303d5e72bedd6d217da1bf',
}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _verify(module):
    path = Path(module.__file__).resolve()
    root = path.parent.parent
    if path != root / 'model/gdn.py':
        raise ValueError('expected pinned model/gdn.py')
    for relative, expected in CORE_PINS.items():
        if _sha(root / relative) != expected:
            raise ValueError('core source mismatch: ' + relative)
    if _sha(HERE / 'candidate.py') != CANDIDATE_SHA:
        raise ValueError('candidate source mismatch')
    for fn in (module.GatedDeltaNet._prefill, module.GatedDeltaNet.__call__, module.causal_conv1d):
        if (type(fn) is not FunctionType or fn.__closure__ is not None
                or Path(fn.__code__.co_filename).resolve() != path
                or fn.__globals__ is not vars(module)):
            raise ValueError('unrecognized loaded GDN function')
    spec = importlib.util.spec_from_file_location('_e026_owned_tail', HERE / 'candidate.py')
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    return helper


class PrefillTailScope:
    """Explicit eager diagnostic scope; required coverage is checked by caller."""

    def __init__(self, module, layers, *, enabled=False, materialize=False, on_call=None):
        if type(enabled) is not bool or type(materialize) is not bool:
            raise TypeError('enabled and materialize must be bool')
        if materialize and not enabled:
            raise ValueError('materialize requires enabled=True')
        if on_call is not None and not callable(on_call):
            raise TypeError('on_call must be callable')
        self.module, self.layers = module, dict(layers)
        if not self.layers or any(type(i) is not int or i < 0 for i in self.layers):
            raise ValueError('nonempty layer-index mapping required')
        if len({id(layer) for layer in self.layers.values()}) != len(self.layers):
            raise ValueError('duplicate layer object')
        self.enabled, self.materialize, self.on_call = enabled, materialize, on_call
        self._saved, self._active, self._single, self._count = [], False, None, 0
        self._closed = False

    def __enter__(self):
        if self._active or self._closed:
            raise RuntimeError('scope already active or closed')
        self._helper = _verify(self.module)
        self._thread = threading.get_ident()
        self._core = Path(self.module.__file__).resolve().parent.parent
        cls = self.module.GatedDeltaNet
        for layer in self.layers.values():
            if type(layer) is not cls or layer._prefill.__func__ is not cls._prefill:
                raise ValueError('only untouched exact GatedDeltaNet instances supported')
        self._active = True
        try:
            for index, layer in self.layers.items():
                present = '_prefill' in vars(layer)
                self._saved.append((layer, present, vars(layer).get('_prefill')))
                object.__setattr__(layer, '_prefill', MethodType(self._wrapper(index), layer))
        except BaseException:
            self.__exit__(*sys.exc_info())
            raise
        return self

    def _normal_route(self, frame):
        ctx = frame.f_locals.get('ctx')
        if not (frame.f_code is self.module.GatedDeltaNet.__call__.__code__
                and type(ctx) is self.module.ForwardContext and ctx.mode == 'prefill'
                and set(vars(ctx)) == {'mode', 'caches', 'positions', 'lens', 'max_len', 'prefill_shape'}
                and ctx.prefill_shape is None
                and not any(getattr(ctx, key, False) for key in ('record', 'capture', 'speculative', 'verify'))):
            return False
        # Qwen capture is a model argument, not ForwardContext metadata. Inspect
        # only pinned Python ancestors; do not retain their frames or tensors.
        while frame is not None:
            path = Path(frame.f_code.co_filename).resolve()
            if path == self._core / 'model/qwen3_5.py' and frame.f_code.co_name == '__call__':
                if frame.f_locals.get('capture'):
                    return False
            if path == self._core / 'engine/model_runner.py' and frame.f_code.co_name == 'run':
                if frame.f_locals['self'].spec is not None:
                    return False
            frame = frame.f_back
        return True

    def _wrapper(self, index):
        original = self.module.GatedDeltaNet._prefill
        globals_copy = dict(original.__globals__)
        original_conv = self.module.causal_conv1d
        call = {'active': False, 'route': None, 'offset': None, 'seen': 0}

        def convolution(x, w, state=None):
            if not call['active'] or threading.get_ident() != self._thread:
                raise RuntimeError('convolution outside eager prefill scope')
            y, tail = original_conv(x, w, state)
            mx = self.module.mx
            eligible = (x.ndim == 2 and x.shape[0] >= 1 and x.shape[1] == 10240
                        and w.shape == (10240, 4) and w.dtype == mx.bfloat16
                        and tail.shape == (3, 10240) and tail.dtype == mx.bfloat16
                        and x.dtype == mx.bfloat16 and self._helper.supported_backend(mx))
            result = (self._helper.own_conv_tail(mx, tail, enabled=self.enabled,
                                               materialize=self.materialize)
                      if eligible else tail)
            self._count += 1
            call['seen'] += 1
            row = dict(sequence=self._count, layer=index, route=call['route'],
                       input_shape=list(x.shape), tail_shape=list(tail.shape),
                       input_dtype=str(x.dtype), tail_dtype=str(tail.dtype),
                       offset=call['offset'], state_present=state is not None,
                       eligible=eligible, enabled=self.enabled,
                       materialize=self.materialize, replaced=result is not tail,
                       y_identity_preserved=True)
            if self.on_call is not None:
                self.on_call(row)
            return y, result

        globals_copy['causal_conv1d'] = convolution
        cloned = FunctionType(original.__code__, globals_copy, original.__name__, original.__defaults__)
        cloned.__kwdefaults__ = original.__kwdefaults__

        def wrapped(layer, x, cache):
            if not self._active or threading.get_ident() != self._thread:
                raise RuntimeError('inactive or cross-thread prefill scope')
            if call['active']:
                raise RuntimeError('reentrant prefill unsupported')
            frame = sys._getframe(1)
            regular = self._normal_route(frame)
            del frame
            safe_cache = (cache is None or not any(getattr(cache, key, False)
                          for key in ('record', 'capture', 'speculative', 'verify')))
            if not safe_cache or not (regular or self._single == index):
                return original(layer, x, cache)
            call.update(active=True, route='normal-prefill' if regular else 'single-layer-eager',
                        offset=None if cache is None else int(cache.offset), seen=0)
            try:
                output = cloned(layer, x, cache)
                if call['seen'] != 1:
                    raise RuntimeError('expected exactly one causal convolution per prefill')
                return output
            finally:
                call.update(active=False, route=None, offset=None)

        return wrapped

    def single_layer(self, index, x, cache):
        if not self._active or self._single is not None:
            raise RuntimeError('single_layer requires active non-reentrant scope')
        self._single = index
        try:
            return self.layers[index]._prefill(x, cache)
        finally:
            self._single = None

    def __exit__(self, exc_type, exc, tb):
        errors = []
        try:
            for layer, present, value in reversed(self._saved):
                try:
                    if present:
                        object.__setattr__(layer, '_prefill', value)
                    else:
                        object.__delattr__(layer, '_prefill')
                except BaseException as error:
                    errors.append(repr(error))
        finally:
            self._saved.clear()
            self.layers.clear()  # A retained receipt scope must not retain model layers.
            self._active, self._single = False, None
            self._closed = True
        if errors:
            message = 'scope restoration failed: ' + '; '.join(errors)
            if exc is None:
                raise RuntimeError(message)
            exc.add_note(message)
        return False
