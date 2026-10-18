"""CPU ownership/cache mechanics only; no MLX import or numerical claim."""
import builtins
from contextlib import contextmanager
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from bench.m5_ultra_256gb.route_decode_graph import test_integration_cpu as route_fixture
from bench.m5_ultra_256gb.gdn_prework_unbound import test_candidate_cpu as gdn_fixture
from bench.m5_ultra_256gb.gdn_graph_combined import integration as m


class Long:
    def __init__(self, model, *, enabled):
        assert enabled is True
        self.model, self._functions = model, {}
        self.hooks_at_construction = tuple('_decode' in vars(x.linear_attn)
            for x in model.layers if not x.is_attn)

    def __call__(self, tokens, ctx):
        # Deliberately only a Python cache fake, NOT an MLX compiled graph.
        key = 1023 if ctx.max_len < 1024 else 1536
        self._functions.setdefault(key, object())
        return ('long', ctx.max_len)


@contextmanager
def fixture():
    graph, gdn = m._dependencies()
    with route_fixture.fixture(compiled=True) as (runner, native, cache, mx):
        module = ModuleType('_combined_fake_gdn')
        cls = type('GatedDeltaNet', (gdn_fixture.GDN,), {'__module__': module.__name__})
        module.GatedDeltaNet = cls
        layers = []
        for index in range(64):
            if index % 4 == 0:
                layers.append(SimpleNamespace(is_attn=True))
            else:
                layer = cls(); layer._compiled_decode_gates = None
                layers.append(SimpleNamespace(is_attn=False, linear_attn=layer))
        runner.model.layers = layers; runner.batch.layers = [None] * 64
        engine = SimpleNamespace(model=runner.model, runner=runner, requests={})
        with patch.dict(sys.modules, {module.__name__: module}), \
             patch.object(m, '_dependencies', return_value=(graph, gdn)), \
             patch.object(graph, '_candidate', return_value=Long), \
             patch.object(gdn, '_load_fused', return_value=object()):
            yield engine, graph, gdn, cls, native, cache


class Tests(unittest.TestCase):
    def test_off_and_import_never_inspect_engine_or_import_backend(self):
        class Poison:
            def __getattribute__(self, name): raise AssertionError(name)
        with patch.object(m, '_dependencies', side_effect=AssertionError('loaded dependency')):
            self.assertIsNone(m.install(Poison()))
            for value in (1, None, 'on'):
                with self.assertRaises(TypeError): m.install(Poison(), gdn_prework=value)
                with self.assertRaises(TypeError): m.install(Poison(), route_graph=value)
        original = builtins.__import__
        def guarded(name, *args, **kwargs):
            if name.split('.')[0] in ('mlx', 'numpy', 'torch', 'tutorial_code'):
                raise AssertionError('runtime import ' + name)
            return original(name, *args, **kwargs)
        with patch('builtins.__import__', guarded):
            spec = importlib.util.spec_from_file_location('_combined_inert', m.__file__)
            module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
            self.assertIsNone(module.install(Poison()))

    def test_all_four_switch_settings_leave_native_methods_and_parameter_handles(self):
        for prework, route in ((False, False), (True, False), (False, True), (True, True)):
            with self.subTest(prework=prework, route=route), fixture() as (e, graph, gdn, cls, native, cache):
                runner, model, decoder = e.runner, e.model, e.runner.decode_model
                layers = [x.linear_attn for x in model.layers if not x.is_attn]
                snapshots = [dict(vars(x)) for x in layers]
                method, run, rebuild = cls._decode, native.ModelRunner.run, cache.DecodeBatch.rebuild
                scope = m.install(e, gdn_prework=prework, route_graph=route)
                self.assertIs(runner.model, model); self.assertIs(runner.batch.model, model)
                self.assertIs(native.ModelRunner.run, run); self.assertIs(cache.DecodeBatch.rebuild, rebuild)
                self.assertIs(cls._decode, method)
                self.assertEqual(runner.decode_model is decoder, not route)
                for layer, before in zip(layers, snapshots):
                    self.assertEqual(set(vars(layer)) - set(before), {'_decode'} if prework else set())
                    for key, value in before.items(): self.assertIs(vars(layer)[key], value)
                if scope is not None: scope.restore()
                self.assertIs(runner.decode_model, decoder)
                for layer, before in zip(layers, snapshots):
                    self.assertEqual(set(vars(layer)), set(before))

    def test_all_gdn_hooks_exist_before_graph_construction_and_native_routes_stay(self):
        with fixture() as (e, *_):
            scope = m.install(e, gdn_prework=True, route_graph=True)
            long = scope.graph_handle.decoder.long
            self.assertEqual(long.hooks_at_construction, (True,) * 48)
            for used in (0, 1, 256, 257, 1023, 1024, 1536, 1537, 2048):
                expected = 'long' if 257 <= used <= 1536 else 'original'
                self.assertEqual(e.runner.decode_model('x', SimpleNamespace(max_len=used)), (expected, used))
            self.assertEqual(e.runner.model('prefill', SimpleNamespace(max_len=None)), ('original', None))
            scope.restore()

    def test_live_state_cached_decoder_or_flags_reject_before_any_hook(self):
        changes = [lambda e: e.requests.update(active=object()),
            lambda e: setattr(e.runner, '_ahead', object()),
            lambda e: e.runner.staging.update(active=[]),
            lambda e: setattr(e.runner, 'spec', object()),
            lambda e: e.runner.batch.req_ids.append('active'),
            lambda e: e.runner.decode_model._functions.update(old=object()),
            lambda e: setattr(e.runner.decode_model, '_packed', object()),
            lambda e: setattr(e.model.layers[1].linear_attn, '_gates_compile_enabled', False),
            lambda e: setattr(e.model.layers[1].linear_attn, '_norm_reuse_enabled', True),
            lambda e: setattr(e.model.layers[1].linear_attn, '_compiled_decode_gates', object())]
        for change in changes:
            with fixture() as (e, *_):
                original = e.runner.decode_model; change(e)
                with self.assertRaises(ValueError): m.install(e, gdn_prework=True, route_graph=True)
                self.assertIs(e.runner.decode_model, original)
                self.assertFalse(any('_decode' in vars(x.linear_attn) for x in e.model.layers if not x.is_attn))

    def test_partial_gdn_install_rolls_back_original_instances(self):
        with fixture() as (e, graph, gdn, *_):
            install = gdn.install; calls = []
            def partial(layer, module, **kwargs):
                if len(calls) == 7: raise RuntimeError('eighth install failed')
                calls.append(layer); return install(layer, module, **kwargs)
            with patch.object(gdn, 'install', side_effect=partial):
                with self.assertRaisesRegex(RuntimeError, 'eighth'): m.install(e, gdn_prework=True, route_graph=True)
            self.assertFalse(any('_decode' in vars(x.linear_attn) for x in e.model.layers if not x.is_attn))
            self.assertEqual(e.runner.decode_model._functions, {})

    def test_graph_failure_rolls_back_all_hooks_without_global_method_changes(self):
        with fixture() as (e, graph, gdn, cls, *_):
            method = cls._decode; decoder = e.runner.decode_model
            with patch.object(graph, '_candidate', side_effect=RuntimeError('graph factory')):
                with self.assertRaisesRegex(RuntimeError, 'graph factory'):
                    m.install(e, gdn_prework=True, route_graph=True)
            self.assertIs(cls._decode, method); self.assertIs(e.runner.decode_model, decoder)
            self.assertFalse(any('_decode' in vars(x.linear_attn) for x in e.model.layers if not x.is_attn))

    def test_restore_clears_graphs_before_reverse_gdn_and_drops_references(self):
        with fixture() as (e, *_):
            short = e.runner.decode_model
            scope = m.install(e, gdn_prework=True, route_graph=True)
            long = scope.graph_handle.decoder.long
            long._functions['old'] = object(); short._functions['old'] = object(); short._packed = object()
            order = []
            for index, handle in enumerate(scope.gdn_handles):
                restore = handle.restore
                def wrapped(index=index, restore=restore):
                    self.assertIs(e.runner.decode_model, short)
                    self.assertEqual(long._functions, {}); self.assertEqual(short._functions, {})
                    self.assertIsNone(short._packed); order.append(index); restore()
                handle.restore = wrapped
            scope.restore(); scope.restore()
            self.assertEqual(order, list(reversed(range(48))))
            self.assertTrue(scope.closed); self.assertEqual(scope.gdn_handles, [])
            self.assertIsNone(scope.runner); self.assertIsNone(scope.model); self.assertIsNone(scope.short)
            with self.assertRaisesRegex(ValueError, 'one installation'): m.install(e, route_graph=True)

    def test_foreign_observer_refusal_is_atomic_then_retry_restores(self):
        for target in ('run', 'decode', 'gdn'):
            with self.subTest(target=target), fixture() as (e, *_):
                scope = m.install(e, gdn_prework=True, route_graph=True)
                long = scope.graph_handle.decoder.long; long._functions['retain'] = object()
                decoder = e.runner.decode_model; layer = scope.gdn_handles[0].layer; hook = layer._decode
                if target == 'run': e.runner.run = object()
                elif target == 'decode': e.runner.decode_model = object()
                else: object.__setattr__(layer, '_decode', object())
                with self.assertRaisesRegex(ValueError, 'observer'): scope.restore()
                self.assertEqual(len(long._functions), 1); self.assertFalse(scope.closed)
                self.assertTrue(all(not handle.closed for handle in scope.gdn_handles))
                if target == 'run': del e.runner.run
                elif target == 'decode': e.runner.decode_model = decoder
                else: object.__setattr__(layer, '_decode', hook)
                scope.restore()

    def test_failed_layer_restore_still_attempts_others_and_can_retry(self):
        with fixture() as (e, *_):
            scope = m.install(e, gdn_prework=True, route_graph=True)
            handle = scope.gdn_handles[17]; restore = handle.restore
            with patch.object(handle, 'restore', side_effect=RuntimeError('restore failure')):
                with self.assertRaisesRegex(RuntimeError, 'restore failure'): scope.restore()
            self.assertEqual(scope.gdn_handles, [handle]); self.assertIsNone(scope.graph_handle)
            scope.restore(); self.assertTrue(handle.closed); self.assertTrue(scope.closed)

    def test_frozen_sources_are_checked_before_dependency_execution(self):
        pins = m.source_pins()
        self.assertEqual(len(pins), 6)
        with patch.dict(m.FIXED, {m.GDN: '0' * 64}):
            with self.assertRaisesRegex(ValueError, 'frozen dependency'): m._dependencies()


if __name__ == '__main__': unittest.main()
