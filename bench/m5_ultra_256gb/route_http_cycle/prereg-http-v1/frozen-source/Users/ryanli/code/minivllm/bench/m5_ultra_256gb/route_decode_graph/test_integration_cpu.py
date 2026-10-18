"""CPU routing/ownership tests only: fake backend, no numeric qualification.

These tests are prepared during performance work; run only after CPU permission.
Real pinned runner/cache class bodies are reused from the E028 CPU fixture.
"""
import __future__
import ast
import builtins
from contextlib import contextmanager
import gc
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

from bench.m5_ultra_256gb.identity_refill import test_integration_cpu as original
from bench.m5_ultra_256gb.route_decode_graph import integration as m


class Model:
    def __init__(self, layers): self.layers, self.calls = layers, []
    def __call__(self, tokens, ctx):
        self.calls.append((tokens, ctx)); return ('original', ctx.max_len)


class Long:
    def __init__(self, model, *, enabled):
        assert enabled is True
        self.model, self.calls = model, []
    def __call__(self, tokens, ctx):
        self.calls.append((tokens, ctx))
        if not getattr(ctx, 'supported', True): return self.model(tokens, ctx)
        return ('long', ctx.max_len)


@contextmanager
def fixture(compiled=False):
    with original.fixture() as (runner, module, cache, mx):
        model = Model(runner.model.layers); runner.model = runner.batch.model = runner.decode_model = model
        extra = {}
        if compiled:
            short = ModuleType('_e028_fake_native.engine.compiled_decode')
            short.__file__ = str(original.CORE/'engine/compiled_decode.py')
            tree = ast.parse(Path(short.__file__).read_text())
            cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'CompiledDecode')
            short.__dict__.update(mx=mx, ForwardContext=type('UnmatchedContext', (), {}))
            exec(compile(ast.Module(body=[cls], type_ignores=[]), short.__file__, 'exec',
                 flags=__future__.annotations.compiler_flag), vars(short))
            extra[short.__name__] = short
            runner.decode_model = short.CompiledDecode(model, fuse_decode=True)
        with patch.dict(sys.modules, extra), patch.object(m, '_candidate', return_value=Long):
            yield runner, module, cache, mx


class Tests(unittest.TestCase):
    def test_off_is_identity_without_inspection_import_or_factory(self):
        class Poison:
            def __getattribute__(self, name): raise AssertionError('OFF inspected runner')
        with patch.object(m, '_candidate', side_effect=AssertionError('candidate loaded')):
            self.assertIsNone(m.install(Poison())); self.assertIsNone(m.install(Poison(), enabled=False))
            with self.assertRaises(TypeError): m.install(Poison(), enabled=1)
        importer = builtins.__import__
        def guarded(name, *args, **kwargs):
            if name.split('.')[0] in ('mlx', 'numpy', 'torch', 'tutorial_code'): raise AssertionError(name)
            return importer(name, *args, **kwargs)
        with patch('builtins.__import__', guarded):
            spec = importlib.util.spec_from_file_location('_route_off_import', m.__file__)
            subject = importlib.util.module_from_spec(spec); spec.loader.exec_module(subject)
            self.assertIsNone(subject.install(Poison()))

    def test_only_decoder_changes_and_all_thresholds_retain_original_short(self):
        for compiled in (False, True):
            with self.subTest(compiled=compiled), fixture(compiled) as (r, module, cache, mx):
                before = dict(vars(r)); run = type(r).run; rebuild = cache.DecodeBatch.rebuild
                handle = m.install(r, enabled=True)
                self.assertIs(handle.decoder.original, before['decode_model'])
                for used in (0,1,256,257,1023,1024,1536,1537,2048,True,257.0):
                    ctx = SimpleNamespace(max_len=used)
                    expected = 'long' if type(used) is int and 257 <= used <= 1536 else 'original'
                    self.assertEqual(r.decode_model('tokens', ctx),(expected, used))
                self.assertEqual(set(vars(r)),set(before))
                for key,value in before.items():
                    if key != 'decode_model': self.assertIs(vars(r)[key],value,key)
                self.assertIs(type(r).run,run); self.assertIs(cache.DecodeBatch.rebuild,rebuild)
                handle.restore(); self.assertIs(r.decode_model,before['decode_model'])

    def test_unsupported_long_layout_calls_guard_once_and_eager_fallback(self):
        with fixture() as (r, *_):
            handle=m.install(r, enabled=True); ctx=SimpleNamespace(max_len=1024,supported=False)
            self.assertEqual(r.decode_model('x',ctx),('original',1024))
            self.assertEqual(handle.decoder.long.calls,[('x',ctx)])
            self.assertEqual(r.model.calls,[('x',ctx)]); handle.restore()

    def test_independent_refill_and_graph_switches_and_install_order(self):
        for refill, graph, graph_first in ((False,False,False),(True,False,False),
                (False,True,False),(True,True,False),(True,True,True)):
            with self.subTest(refill=refill,graph=graph,graph_first=graph_first), fixture() as (r, module, cache, mx):
                batch,decoder=r.batch,r.decode_model
                if graph_first: handle=m.install(r,enabled=graph)
                original.integration.install(r,enabled=refill)
                if not graph_first: handle=m.install(r,enabled=graph)
                self.assertIs(r.model,decoder)
                self.assertEqual(r.batch is batch,not refill)
                self.assertEqual(r.decode_model is decoder,not graph)
                if refill: self.assertIs(type(r.batch).__mro__[1],cache.DecodeBatch)
                if handle: handle.restore()

    def test_exception_scope_restores_and_releases_refs_without_reinstall(self):
        with fixture() as (r, *_):
            decoder=r.decode_model; handle=m.install(r,enabled=True); reference=weakref.ref(handle.decoder.long)
            with self.assertRaisesRegex(RuntimeError,'body failure'):
                with handle: raise RuntimeError('body failure')
            self.assertIs(r.decode_model,decoder); self.assertTrue(handle.closed)
            self.assertIsNone(handle.runner); self.assertIsNone(handle.previous); self.assertIsNone(handle.decoder)
            gc.collect(); self.assertIsNone(reference()); handle.restore()
            with self.assertRaisesRegex(ValueError,'already attempted'): m.install(r,enabled=True)

    def test_foreign_observer_is_not_overwritten_then_restore_can_retry(self):
        with fixture() as (r, *_):
            handle=m.install(r,enabled=True); installed=r.decode_model; observer=object(); r.decode_model=observer
            with self.assertRaisesRegex(ValueError,'restore observer'): handle.restore()
            self.assertIs(r.decode_model,observer); self.assertFalse(handle.closed)
            r.decode_model=installed; handle.restore(); self.assertIs(r.decode_model,r.model)

    def test_live_spec_or_wrapped_decoder_rejected_without_assignment(self):
        changes=[lambda r:r.batch.req_ids.append('live'),lambda r:r.batch.lens.append(1),
            lambda r:r.batch.layers.__setitem__(0,object()),lambda r:r.staging.update(live=[]),
            lambda r:r.feats.update(live=[]),lambda r:setattr(r,'_ahead',object()),
            lambda r:setattr(r,'spec',object()),lambda r:setattr(r,'decode_model',SimpleNamespace(model=r.model))]
        for change in changes:
            with fixture() as (r, *_):
                change(r);before=dict(vars(r))
                with self.assertRaises(ValueError): m.install(r,enabled=True)
                for key,value in before.items(): self.assertIs(vars(r)[key],value)

    def test_failed_construction_and_native_identity_rejected_without_assignment(self):
        with fixture() as (r, module, cache, mx):
            decoder=r.decode_model
            with patch.object(m,'_candidate',side_effect=RuntimeError('factory error')):
                with self.assertRaisesRegex(RuntimeError,'factory error'):m.install(r,enabled=True)
            self.assertIs(r.decode_model,decoder)
            handle=m.install(r,enabled=True);handle.restore()
        with fixture() as (r, module, cache, mx):
            module.mx=object()
            with self.assertRaisesRegex(ValueError,'MLX backend'):m.install(r,enabled=True)
            self.assertIs(r.decode_model,r.model)
        with fixture() as (r, module, cache, mx):
            cache.DecodeBatch.rebuild=lambda *args:None
            with self.assertRaisesRegex(ValueError,'loaded native method'):m.install(r,enabled=True)
            self.assertIs(r.decode_model,r.model)


if __name__=='__main__':unittest.main()
