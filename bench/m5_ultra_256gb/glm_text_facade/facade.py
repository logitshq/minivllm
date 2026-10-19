"""Default-OFF, borrowed C1 GLM text reference; no loader or runtime imports.

Call the complete session inside ONE submission to the already loaded engine's
single MLX executor. The caller owns exclusive admission, immutable parameters,
source verification, evaluation/output evidence and eventual pool shutdown.
This facade neither implements the engine's sampler nor proves EnginePool parity.
"""
from pathlib import Path
import sys
from threading import current_thread


SITE = Path('/Users/ryanli/code/minivllm/.artifacts/omlx-baseline/.venv/lib/python3.13/site-packages')
VENDOR = SITE / 'omlx/patches/mlx_vlm_glm5_next_compat/vendor/mlx_vlm/models/glm5_next'
SOURCE_FILES = {
    'VLMBatchedEngine': SITE / 'omlx/engine/vlm.py',
    'Model': VENDOR / 'glm5_next.py',
    'LanguageModel': VENDOR / 'language.py',
    'VLMModelAdapter': SITE / 'omlx/models/vlm.py',
    **{name: SITE / 'mlx_vlm/models/cache.py' for name in ('ArraysCache', 'KVCache', 'CacheList')},
    'PoolingCache': SITE / 'omlx/patches/deepseek_v4/cache_extras.py',
}
FIELDS = {
    'ArraysCache': frozenset(('cache', '_left_padding', '_left_padding_advance',
        '_lengths', '_lengths_advance', '_speculation', '_speculation_generation', 'metadata_revision')),
    'KVCache': frozenset(('keys', 'values', 'offset')),
    'CacheList': frozenset(('caches',)),
    'PoolingCache': frozenset(('ratio', 'buf_kv', 'buf_gate', 'remainder', '_pool_buf',
        '_pool_len', '_undo', '_undo_chain', 'prev_win_kv', 'prev_win_gate', '_mtp_cross_boundary_rollback')),
}
EMPTY_SCHEDULER = ('waiting', 'running', 'prefilling', 'requests', '_prefill_states',
    'request_id_to_uid', 'uid_to_request_id', '_pending_async_removes',
    '_inflight_store_futures', '_vlm_mtp_active', '_pending_abort_ids')
VOCAB = 154880


def need(ok, message):
    if not ok:
        raise ValueError(message)


def class_record(obj, name):
    """Constructor code locates injected PoolingCache despite rewritten module."""
    cls = type(obj)
    code = getattr(cls.__init__, '__code__', None)
    need(cls.__name__ == name and code is not None, 'unsupported class: ' + name)
    path = Path(code.co_filename).resolve()
    need(path == SOURCE_FILES[name], 'different installed class source: ' + name)
    return dict(file=str(path), module=cls.__module__, name=cls.__qualname__)


def bind_loaded(engine, expected_identity=None, *, enabled=False, mx=None, assert_exclusive=None):
    """OFF does not inspect arguments. ON requires the loaded-engine identity.

    expected_identity is the existing load probe's loaded-engine.json mapping.
    Its SHA fields are NOT rehashed here; the outer caller must verify source and
    model receipts. assert_exclusive() must raise unless its admission/parameter
    lease is still held. It is not replaced by a guessed empty-queue permission.
    """
    if enabled is False:
        return None
    need(enabled is True and callable(assert_exclusive), 'explicit enable/exclusive lease required')
    return Reference(engine, expected_identity, mx, assert_exclusive)


class Session:
    def __init__(self, owner, prompt, cache):
        self._owner, self.prompt, self._cache = owner, prompt, cache
        self.consumed = self.steps = 0
        self.prefilled = self.poisoned = self.retired = False


class Reference:
    def __init__(self, engine, identity, mx, guard):
        need(mx is not None and mx is sys.modules.get('mlx.core'), 'pass the already loaded mlx.core module')
        self._engine, self.mx, self._guard = engine, mx, guard
        self._core = engine._engine.engine
        self._scheduler = self._core.scheduler
        self._executor, self._stream = self._core._mlx_executor, self._core._mlx_stream
        self._wrapper = engine._vlm_model
        self._language = self._wrapper.language_model
        self._adapter = self._scheduler.model
        self._session = None
        self._closed = False
        self._check()
        actual = {key: class_record(obj, name) for key, obj, name in (
            ('engine', engine, 'VLMBatchedEngine'), ('wrapper', self._wrapper, 'Model'),
            ('language', self._language, 'LanguageModel'), ('adapter', self._adapter, 'VLMModelAdapter'))}
        need(type(identity) is dict, 'loaded-engine identity required')
        for key, row in actual.items():
            need(all(identity[key].get(k) == v for k, v in row.items()), 'loaded identity differs: ' + key)
        need(identity['model_type'] == engine.model_type == self._language.model_type == 'glm5_next', 'model family')
        layers = self._language.layers
        need(len(layers) == identity['layers'] == 45 and sum(bool(x.is_linear) for x in layers) == identity['kda_layers'] == 34,
             'fixed 45-layer / 34-KDA model required')
        need(self._language.args.vocab_size == VOCAB, 'fixed vocabulary required')
        self.sources = actual
        self._linear = tuple(bool(x.is_linear) for x in layers)

    def _thread(self):
        need(not self._closed and self._executor is self._core._mlx_executor,
             'released reference or changed executor')
        need(self._executor._max_workers == 1 and current_thread() in self._executor._threads,
             'must run inside the loaded engine single executor')
        need(self._stream is self._core._mlx_stream is self._scheduler._stream, 'changed engine stream')

    def _check(self):
        self._thread()
        self._guard()
        need(self._engine._vlm_model is self._wrapper and self._wrapper.language_model is self._language and
             self._scheduler.model is self._adapter and self._adapter._vlm_model is self._wrapper and
             self._adapter._language_model is self._language, 'loaded object replaced')
        need(self._engine.prefix_cache_enabled is False, 'prefix cache enabled')
        for key in EMPTY_SCHEDULER:
            need(not getattr(self._scheduler, key), 'scheduler is not drained: ' + key)
        # The initial domain is fresh, unbatched caches. Do not silently accept
        # a surviving BatchGenerator or emulate its singleton merge operation.
        need(self._scheduler.batch_generator is None, 'existing batch generator unsupported')
        for key in ('_output_collectors', '_stream_states', '_finished_events'):
            need(not getattr(self._core, key), 'engine output ownership remains: ' + key)
        need(self._core._closing is False, 'engine closing')
        need(self._adapter._pending_embeds is None and not self._adapter._pending_kwargs and
             self._adapter._batch_rope_deltas is None and not self._adapter._uid_rope_deltas and
             not self._adapter._uid_text_positions and self._adapter._uses_mrope is False,
             'media / position state unsupported')
        need(self._scheduler._vlm_mtp_drafter is None and self._scheduler._specprefill_draft_model is None and
             self._scheduler._turboquant_kv_bits is None and self._scheduler.block_aware_cache is None,
             'non-ordinary scheduler route')

    def _tree(self, value):
        if isinstance(value, self.mx.array):
            return dict(tensor=value, shape=tuple(value.shape), dtype=str(value.dtype))
        if value is None or type(value) in (int, bool):
            return value
        if type(value) in (list, tuple):
            return tuple(self._tree(x) for x in value)
        raise ValueError('unsupported cache metadata value: ' + type(value).__name__)

    def _cache(self, cache, expected):
        source = class_record(cache, expected)
        need(set(vars(cache)) == FIELDS[expected], 'cache field drift: ' + expected)
        if expected == 'CacheList':
            need(type(cache.caches) is tuple and len(cache.caches) == 2, 'DSA cache layout')
            return dict(kind=expected, source=source, fields=dict(caches=(
                self._cache(cache.caches[0], 'KVCache'), self._cache(cache.caches[1], 'PoolingCache'))))
        if expected == 'ArraysCache':
            need(type(cache.cache) is list and len(cache.cache) == 2 and cache._speculation is None,
                 'KDA cache layout / active speculation')
            need(cache._left_padding is None and cache._lengths is None, 'batched KDA metadata unsupported')
            need(all(type(getattr(cache, key)) is int and getattr(cache, key) >= 0 for key in
                     ('_left_padding_advance', '_lengths_advance', '_speculation_generation', 'metadata_revision')),
                 'KDA integer metadata')
            for value in cache.cache:
                need(value is None or (isinstance(value, self.mx.array) and value.shape[0] == 1), 'KDA must be B1')
        elif expected == 'KVCache':
            need(type(cache.offset) is int and cache.offset >= 0 and cache.step == 256, 'KV offset/step')
            need((cache.keys is None) == (cache.values is None), 'partial KV cache')
            if cache.keys is not None:
                need(len(cache.keys.shape) == len(cache.values.shape) == 4 and
                     cache.keys.shape[:3] == cache.values.shape[:3] and cache.keys.shape[0] == 1 and
                     cache.values.shape[3] == 0 and cache.offset <= cache.keys.shape[2], 'latent KV backing geometry')
        else:
            need(type(cache.ratio) is int and cache.ratio == 4 and type(cache.remainder) is int and
                 0 <= cache.remainder < 4 and type(cache._pool_len) is int and cache._pool_len >= 0,
                 'pooling scalar metadata')
            need(type(cache._undo_chain) is bool and cache._mtp_cross_boundary_rollback is True, 'pooling undo policy')
            need(cache._undo is None or (type(cache._undo) is tuple and len(cache._undo) == 8), 'pooling undo layout')
            for key in ('buf_kv', 'buf_gate', '_pool_buf', 'prev_win_kv', 'prev_win_gate'):
                value = getattr(cache, key)
                need(value is None or (isinstance(value, self.mx.array) and value.shape[0] == 1), 'pooling must be B1')
            need((cache._pool_buf is None and cache._pool_len == 0) or
                 (cache._pool_buf is not None and cache._pool_len <= cache._pool_buf.shape[1]), 'pool backing length')
        fields = {key: self._tree(value) for key, value in vars(cache).items()}
        if expected == 'KVCache':
            fields['step'] = cache.step  # class-level capacity increment, absent from vars()
        return dict(kind=expected, source=source, fields=fields)

    def _caches(self, session):
        need(type(session._cache) is list and len(session._cache) == 45, '45 cache layers required')
        objects = [obj for cache in session._cache for obj in
                   ([cache, *cache.caches] if type(cache).__name__ == 'CacheList' else [cache])]
        need(len({id(obj) for obj in objects}) == len(objects), 'cache objects aliased across layers')
        return tuple(self._cache(cache, 'ArraysCache' if linear else 'CacheList')
                     for cache, linear in zip(session._cache, self._linear))

    def _own(self, session, *, poisoned_ok=False):
        self._check()
        need(session is self._session and session._owner is self and not session.retired, 'foreign/retired session')
        need(poisoned_ok or not session.poisoned, 'failed session cannot continue')

    @staticmethod
    def _ids(ids, minimum, maximum):
        need(type(ids) in (tuple, list) and minimum <= len(ids) <= maximum and
             all(type(x) is int and 0 <= x < VOCAB for x in ids), 'plain C1 integer token IDs required')
        return tuple(ids)

    def new_request(self, prompt_ids):
        self._check()
        need(self._session is None, 'only one privately owned C1 request')
        prompt = self._ids(prompt_ids, 1, 128)
        with self.mx.stream(self._stream):
            session = Session(self, prompt, self._adapter.make_cache())
            self._caches(session)
            for cache, linear in zip(session._cache, self._linear):
                if linear:
                    need(all(x is None for x in cache.cache), 'new KDA cache must be empty')
                else:
                    kv, pool = cache.caches
                    need(kv.keys is None and kv.offset == 0 and pool.remainder == pool._pool_len == 0 and
                         all(getattr(pool, key) is None for key in
                             ('buf_kv', 'buf_gate', '_pool_buf', '_undo', 'prev_win_kv', 'prev_win_gate')),
                         'new DSA cache must be empty')
        self._session = session
        return session

    def prefill_prefix(self, session, ids_without_last_prompt_token):
        self._own(session)
        ids = self._ids(ids_without_last_prompt_token, 0, 127)
        need(not session.prefilled and session.consumed == 0 and ids == session.prompt[:-1], 'exact fresh prompt prefix required')
        try:
            with self.mx.stream(self._stream):
                if ids:
                    # Same prefix boundary and state evaluation as scheduler.py
                    # 4001/4166–4181. GLM adapter does not skip its LM head.
                    self._adapter._omlx_prefill(self.mx.array([list(ids)], dtype=self.mx.int32), cache=session._cache)
                    self.mx.eval([c.state for c in session._cache])
                self._caches(session)
            session.consumed, session.prefilled = len(ids), True
        except BaseException:
            session.poisoned = True  # no rollback of partially advanced state
            raise

    def step(self, session, pending_ids):
        self._own(session)
        ids = self._ids(pending_ids, 1, 1)
        need(session.prefilled and session.steps < 32 and session.consumed < 2048, 'prefix/short teacher-forced domain')
        need(session.steps != 0 or ids[0] == session.prompt[-1], 'first pending ID must be last prompt token')
        try:
            with self.mx.stream(self._stream):
                logits = self._adapter(self.mx.array([list(ids)], dtype=self.mx.int32), cache=session._cache)
                need(isinstance(logits, self.mx.array) and tuple(logits.shape) == (1, 1, VOCAB), 'full C1 vocabulary logits required')
                self._caches(session)
            session.consumed += 1
            session.steps += 1
            return logits  # no sample, cast, eval, synchronize or cache.advance()
        except BaseException:
            session.poisoned = True
            raise

    def snapshot(self, session):
        self._own(session, poisoned_ok=True)
        return dict(schema='GLM-text-borrowed-snapshot-v1', consumed=session.consumed,
            steps=session.steps, poisoned=session.poisoned, layers=self._caches(session),
            semantics='current borrowed handles; inspect/evaluate/save before another call; not clone or restore',
            evaluated=False, numerical_qualified=False, performance_qualified=False)

    def retire(self, session):
        self._thread()  # local release remains possible after model/lease failure
        need(session is self._session and session._owner is self, 'foreign session')
        session._cache = session._owner = None
        session.retired = True
        self._session = None

    def close(self):
        self._thread()
        need(self._session is None, 'retire request before releasing reference')
        self._closed = True
        self._engine = self._core = self._scheduler = self._executor = self._stream = None
        self._wrapper = self._language = self._adapter = self.mx = self._guard = None
