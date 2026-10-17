"""E024 isolated graph-only prototype. OFF by default; no GPU qualification yet.

Only normal target decode with 256 < used length <= 1536 and B1..8 is eligible.
The graph captures model parameters as inputs; request cache is explicit input
and output. This file never installs itself in ModelRunner or modifies core.
"""
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace


ATTENTION_BOUND = 1536
MIN_USED_LENGTH = 257
PINS = Path(__file__).with_name('model-source-pins.json')


def verify_model_sources(root):
    """Small Python source hashes only; the owner separately pins model weights."""
    root = Path(root).resolve()
    raw = PINS.read_bytes()
    expected = json.loads(raw)['files']
    actual = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in root.rglob('*.py')}
    if actual != expected:
        raise RuntimeError('E024 loaded model source differs from model-source-pins.json')
    return {'root': str(root), 'manifest_sha256': hashlib.sha256(raw).hexdigest(),
            'files': actual}


def _load_runtime(model):
    # Resolve from the actual model class, including a controller's frozen tree.
    model_file = Path(inspect.getfile(type(model))).resolve()
    root = model_file.parent.parent
    if model_file != root / 'model/qwen3_5.py':
        raise RuntimeError('E024 requires the pinned Qwen model implementation')
    receipt = verify_model_sources(root)
    import mlx.core as mx
    from tutorial_code.miniengine import cache, context
    from tutorial_code.miniengine.engine import compiled_decode
    from tutorial_code.miniengine.model import gdn
    for module, relative in ((cache, 'cache.py'), (context, 'context.py'),
                             (compiled_decode, 'engine/compiled_decode.py'), (gdn, 'model/gdn.py')):
        if Path(module.__file__).resolve() != root / relative:
            raise RuntimeError('E024 mixed model/core import roots: ' + relative)
    return SimpleNamespace(mx=mx, BatchKV=cache.BatchKV, BatchState=cache.BatchState,
        ForwardContext=context.ForwardContext, states=compiled_decode.CompiledDecode._states,
        identity=gdn._m5_ultra_decode_identity, source_receipt=receipt)


class LongContextDecode:
    """Bench callable: LongContextDecode(engine.model, enabled=True)(tokens, ctx).

    Enablement and model ownership are fixed at construction. Replace array
    parameters on the original model normally; recreate this object after any
    model structure/non-array configuration change (except gates/norm flags).
    """
    def __init__(self, model, *, enabled=False):
        if type(enabled) is not bool:
            raise TypeError('enabled must be bool')
        self._model = model
        self._enabled = enabled
        self._runtime = _load_runtime(model) if enabled else None
        self._functions = {}
        self._bound = ATTENTION_BOUND
        self._minimum = MIN_USED_LENGTH
        self._layer_ids = tuple(id(layer) for layer in model.layers) if enabled else ()
        self.source_receipt = self._runtime.source_receipt if enabled else None

    @property
    def model(self):
        return self._model

    @property
    def enabled(self):
        return self._enabled

    @property
    def bound(self):
        return self._bound

    def _eligible(self, tokens, ctx):
        r, model = self._runtime, self._model
        mx = r.mx
        if not (type(ctx) is r.ForwardContext and ctx.mode == 'decode'
                and set(vars(ctx)) == {'mode', 'caches', 'positions', 'lens', 'max_len', 'prefill_shape'}
                and ctx.prefill_shape is None and tokens.ndim == 1
                and 1 <= tokens.shape[0] <= 8 and tokens.dtype in (mx.int32, mx.int64)
                and type(ctx.max_len) is int and self._minimum <= ctx.max_len <= self._bound
                and ctx.positions is not None and ctx.lens is not None
                and ctx.positions.shape == ctx.lens.shape == tokens.shape
                and ctx.positions.dtype in (mx.int32, mx.int64)
                and ctx.lens.dtype in (mx.int32, mx.int64)
                and model.cfg.hidden_size == 5120 and len(model.layers) == 64
                and tuple(id(layer) for layer in model.layers) == self._layer_ids
                and model.cfg.quantization == {'group_size': 16, 'bits': 4, 'mode': 'nvfp4'}
                and model.embed_tokens._output_dtype == mx.bfloat16
                and mx.default_device() == mx.gpu and r.identity()
                and len(ctx.caches) == len(model.layers)):
            return False
        B = tokens.shape[0]
        for layer, cache in zip(model.layers, ctx.caches):
            if layer.is_attn:
                if not (type(cache) is r.BatchKV and set(vars(cache)) == {'K', 'V'}
                        and cache.K.ndim == cache.V.ndim == 4
                        and cache.K.shape == cache.V.shape and cache.K.shape[0] == B
                        and cache.K.shape[2] >= self._bound
                        and cache.K.dtype == cache.V.dtype == mx.bfloat16):
                    return False
            elif not (type(cache) is r.BatchState and set(vars(cache)) == {'conv', 'S'}
                      and cache.conv.shape == (B, 3, 10240) and cache.conv.dtype == mx.bfloat16
                      and cache.S.shape == (B, 48, 128, 128) and cache.S.dtype == mx.float32):
                return False
        return True

    def __call__(self, tokens, ctx):
        model = self._model
        if not self._enabled or not self._eligible(tokens, ctx):
            return model(tokens, ctx)
        r, bound = self._runtime, self._bound
        key = (bound, tuple((layer.linear_attn._gates_compile_enabled,
                             layer.linear_attn._norm_reuse_enabled)
                            for layer in model.layers if not layer.is_attn))
        if key not in self._functions:
            layout = tuple(layer.is_attn for layer in model.layers)
            def forward(ids, positions, lens, states):
                # Independent KV array wrappers protect the caller's handles
                # from __setitem__ while sharing evaluated input storage.
                caches = [r.BatchKV(s[0][:], s[1][:]) if attn else r.BatchState(*s)
                          for attn, s in zip(layout, states)]
                logits = model(ids, r.ForwardContext('decode', caches,
                    positions=positions, lens=lens, max_len=bound))
                return logits, r.states(caches)
            self._functions[key] = r.mx.compile(forward, inputs=model)
        logits, states = self._functions[key](tokens, ctx.positions, ctx.lens, r.states(ctx.caches))
        if len(states) != len(ctx.caches) or any(len(pair) != 2 for pair in states):
            raise RuntimeError('E024 graph returned an incomplete state tree')
        for cache, state in zip(ctx.caches, states):
            if type(cache) is r.BatchKV:
                cache.K, cache.V = state
            else:
                cache.conv, cache.S = state
        return logits
