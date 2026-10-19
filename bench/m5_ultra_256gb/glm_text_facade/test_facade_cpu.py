"""Source-prepared stdlib fakes: ownership/API only, no GLM numerical proof.

The class-source checker is mocked explicitly for fake objects except in the
dedicated rejection test. No executor is started and no MLX module is imported.
"""
from contextlib import contextmanager
from threading import current_thread
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from bench.m5_ultra_256gb.glm_text_facade import facade as f


class Array:
    def __init__(self, values=None, dtype='int32', *, shape=None):
        self.values, self.dtype = values, dtype
        self.shape = shape if shape is not None else (len(values), len(values[0]))


class MX:
    array = Array
    int32 = 'int32'

    def __init__(self):
        self.current, self.evaluated = None, []

    @contextmanager
    def stream(self, stream):
        prior, self.current = self.current, stream
        try:
            yield
        finally:
            self.current = prior

    def eval(self, state):
        self.evaluated.append((self.current, state))


class ArraysCache:
    def __init__(self):
        self.cache = [None, None]
        self._left_padding = self._lengths = self._speculation = None
        self._left_padding_advance = self._lengths_advance = 0
        self._speculation_generation = self.metadata_revision = 0

    @property
    def state(self):
        return self.cache


class KVCache:
    step = 256

    def __init__(self):
        self.keys = self.values = None
        self.offset = 0

    @property
    def state(self):
        return (self.keys, self.values)


class PoolingCache:
    def __init__(self):
        self.ratio = 4
        self.remainder = self._pool_len = 0
        self.buf_kv = self.buf_gate = self._pool_buf = self._undo = None
        self.prev_win_kv = self.prev_win_gate = None
        self._undo_chain = False
        self._mtp_cross_boundary_rollback = True

    @property
    def state(self):
        return self.buf_kv, self.buf_gate, self._pool_buf, self.prev_win_kv, self.prev_win_gate


class CacheList:
    def __init__(self):
        self.caches = (KVCache(), PoolingCache())

    @property
    def state(self):
        return [x.state for x in self.caches]


def fake_record(obj, name):
    if name in f.FIELDS:
        f.need(type(obj).__name__ == name, 'fake cache class differs')
    return dict(file='CPU-MOCK/' + name, name=name, module='CPU-MOCK')


class Adapter:
    def __init__(self, wrapper, mx, stream):
        self._vlm_model, self._language_model = wrapper, wrapper.language_model
        self.mx, self.stream, self.calls = mx, stream, []
        self._pending_embeds = self._batch_rope_deltas = None
        self._pending_kwargs, self._uid_rope_deltas = {}, {}
        self._uid_text_positions = set()
        self._uses_mrope = False
        self.fail = False

    def make_cache(self):
        return [ArraysCache() if x.is_linear else CacheList() for x in self._language_model.layers]

    def _call(self, kind, ids, cache):
        assert self.mx.current is self.stream
        self.calls.append((kind, ids.values, cache))
        if self.fail:
            raise RuntimeError('CPU fake partial forward failure')
        for layer in cache:
            if type(layer) is ArraysCache:
                layer.cache[:] = [Array(dtype='bfloat16', shape=(1, 3, 4)),
                                  Array(dtype='float32', shape=(1, 2, 3, 3))]
            else:
                kv, _ = layer.caches
                kv.offset += ids.shape[1]
                kv.keys = Array(dtype='bfloat16', shape=(1, 1, 256, 4))
                kv.values = Array(dtype='bfloat16', shape=(1, 1, 256, 0))
        return Array(dtype='bfloat16', shape=(1, ids.shape[1], f.VOCAB))

    def _omlx_prefill(self, ids, cache):
        return self._call('prefill', ids, cache)

    def __call__(self, ids, cache):
        return self._call('step', ids, cache)


def fixture():
    mx, stream = MX(), object()
    language = SimpleNamespace(model_type='glm5_next', args=SimpleNamespace(vocab_size=f.VOCAB),
        layers=[SimpleNamespace(is_linear=i < 34) for i in range(45)])
    wrapper = SimpleNamespace(language_model=language, vision_model=object())
    adapter = Adapter(wrapper, mx, stream)
    scheduler = SimpleNamespace(**{key: {} for key in f.EMPTY_SCHEDULER}, model=adapter,
        _stream=stream, batch_generator=None, _vlm_mtp_drafter=None,
        _specprefill_draft_model=None, _turboquant_kv_bits=None, block_aware_cache=None)
    executor = SimpleNamespace(_max_workers=1, _threads={current_thread()})
    core = SimpleNamespace(scheduler=scheduler, _mlx_executor=executor, _mlx_stream=stream,
        _output_collectors={}, _stream_states={}, _finished_events={}, _closing=False)
    engine = SimpleNamespace(_engine=SimpleNamespace(engine=core), _vlm_model=wrapper,
        prefix_cache_enabled=False, model_type='glm5_next')
    identity = {k: fake_record(None, n) for k, n in (
        ('engine', 'VLMBatchedEngine'), ('wrapper', 'Model'),
        ('language', 'LanguageModel'), ('adapter', 'VLMModelAdapter'))}
    identity.update(model_type='glm5_next', layers=45, kda_layers=34)
    return engine, identity, mx, adapter


class FacadeTests(unittest.TestCase):
    def setUp(self):
        self.source = patch.object(f, 'class_record', side_effect=fake_record)
        self.source.start()
        self.addCleanup(self.source.stop)
        self.engine, self.identity, self.mx, self.adapter = fixture()
        self.modules = patch.dict('sys.modules', {'mlx.core': self.mx})
        self.modules.start()  # explicit fake module; never import a device package
        self.addCleanup(self.modules.stop)
        self.ref = f.bind_loaded(self.engine, self.identity, enabled=True, mx=self.mx, assert_exclusive=lambda: None)

    def test_off_never_inspects_poison_arguments(self):
        class Poison:
            def __getattribute__(self, name):
                raise AssertionError(name)
        self.assertIsNone(f.bind_loaded(Poison(), Poison(), mx=Poison(), assert_exclusive=Poison()))

    def test_exact_prefix_pending_token_and_original_adapter(self):
        wrapper, vision = self.engine._vlm_model, self.engine._vlm_model.vision_model
        session = self.ref.new_request([7, 8, 9])
        self.ref.prefill_prefix(session, [7, 8])
        logits = self.ref.step(session, [9])
        self.ref.step(session, [12])  # caller teacher forcing, no sampled token injected
        self.assertEqual([(x[0], x[1]) for x in self.adapter.calls],
                         [('prefill', [[7, 8]]), ('step', [[9]]), ('step', [[12]])])
        self.assertTrue(all(x[2] is session._cache for x in self.adapter.calls))
        self.assertEqual((session.consumed, session.steps, logits.shape), (4, 2, (1, 1, f.VOCAB)))
        self.assertEqual(len(self.mx.evaluated), 1)  # official prefix state boundary only
        self.assertIs(self.mx.evaluated[0][0], self.ref._stream)
        self.assertIs(self.engine._vlm_model, wrapper)
        self.assertIs(wrapper.vision_model, vision)

    def test_wrong_prefix_first_pending_and_batch_rejected_before_call(self):
        s = self.ref.new_request([2, 3])
        with self.assertRaises(ValueError):
            self.ref.prefill_prefix(s, [3])
        self.assertEqual(self.adapter.calls, [])
        self.ref.prefill_prefix(s, [2])
        for ids in ([4], [3, 4], [True], [[3]]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                self.ref.step(s, ids)
        self.assertEqual(len(self.adapter.calls), 1)

    def test_wrong_executor_thread_refuses_before_tensor_creation(self):
        self.ref._executor._threads = set()
        with self.assertRaisesRegex(ValueError, 'single executor'):
            self.ref.new_request([1])
        self.assertEqual(self.adapter.calls, [])

    def test_live_scheduler_or_old_batch_cannot_be_borrowed(self):
        for key, value in (('running', {'rid': object()}), ('batch_generator', object())):
            with self.subTest(key=key):
                previous = getattr(self.ref._scheduler, key)
                setattr(self.ref._scheduler, key, value)
                with self.assertRaises(ValueError):
                    self.ref.new_request([1])
                setattr(self.ref._scheduler, key, previous)

    def test_sessions_private_and_retired_handles_not_reused(self):
        s = self.ref.new_request([1])
        old = s._cache
        with self.assertRaises(ValueError):
            self.ref.new_request([2])
        self.ref.retire(s)
        t = self.ref.new_request([2])
        self.assertTrue(all(a is not b for a, b in zip(old, t._cache)))
        with self.assertRaises(ValueError):
            self.ref.step(s, [1])

    def test_unknown_metadata_and_active_speculation_are_not_dropped(self):
        s = self.ref.new_request([1])
        s._cache[0].unknown_future_field = 1
        with self.assertRaisesRegex(ValueError, 'field drift'):
            self.ref.snapshot(s)
        del s._cache[0].unknown_future_field
        s._cache[0]._speculation = {'active': True}
        with self.assertRaisesRegex(ValueError, 'speculation'):
            self.ref.snapshot(s)

    def test_snapshot_keeps_full_backing_undo_and_original_dtype(self):
        s = self.ref.new_request([1, 2, 3])
        self.ref.prefill_prefix(s, [1, 2])
        kv, pool = s._cache[34].caches
        pool._pool_buf = Array(dtype='bfloat16', shape=(1, 128, 4))
        pool._pool_len = 1
        pool.prev_win_kv = Array(dtype='bfloat16', shape=(1, 1, 4, 4))
        pool._undo = (None, None, 0, None, pool._pool_buf, pool._pool_buf, pool.prev_win_kv, None)
        snap = self.ref.snapshot(s)
        k, p = snap['layers'][34]['fields']['caches']
        self.assertIs(k['fields']['keys']['tensor'], kv.keys)
        self.assertEqual(k['fields']['keys']['shape'][2], 256)  # not logical offset=2
        self.assertEqual(k['fields']['offset'], 2)
        self.assertEqual(p['fields']['_pool_buf']['shape'][1], 128)
        self.assertIs(p['fields']['_undo'][6]['tensor'], pool.prev_win_kv)
        self.assertEqual(snap['layers'][0]['fields']['cache'][1]['dtype'], 'float32')
        self.assertFalse(snap['evaluated'])
        self.assertEqual(len(self.mx.evaluated), 1)

    def test_failed_call_poisoned_but_retire_close_preserve_engine(self):
        s = self.ref.new_request([1])
        self.ref.prefill_prefix(s, [])
        self.adapter.fail = True
        with self.assertRaises(RuntimeError):
            self.ref.step(s, [1])
        self.assertTrue(s.poisoned)
        self.adapter.fail = False
        with self.assertRaisesRegex(ValueError, 'failed session'):
            self.ref.step(s, [1])
        self.ref.retire(s)
        adapter = self.ref._adapter
        self.ref.close()
        self.assertIs(self.engine._engine.engine.scheduler.model, adapter)
        self.assertIsNone(s._cache)

    def test_replacement_or_revoked_lease_refuses_and_local_retire_works(self):
        s = self.ref.new_request([1])
        def revoked():
            raise RuntimeError('CPU fake admission lease revoked')
        self.ref._guard = revoked
        with self.assertRaisesRegex(RuntimeError, 'lease revoked'):
            self.ref.snapshot(s)
        self.ref._guard = lambda: None
        self.engine._vlm_model.language_model = object()
        with self.assertRaisesRegex(ValueError, 'object replaced'):
            self.ref.snapshot(s)
        self.ref.retire(s)
        self.ref.close()

    def test_actual_class_checker_rejects_fake_source_without_import(self):
        self.source.stop()
        with self.assertRaisesRegex(ValueError, 'different installed class source'):
            f.class_record(KVCache(), 'KVCache')


if __name__ == '__main__':
    unittest.main()
