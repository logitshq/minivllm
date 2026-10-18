"""E028 installation tests with real pinned class bodies and fake MLX only."""
import __future__
import ast
import builtins
from contextlib import contextmanager
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch


HERE = Path(__file__).resolve().parent
CORE = HERE.parents[2] / 'tutorial_code/miniengine'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


integration = load('e028_integration_test_subject', HERE / 'integration.py')


@contextmanager
def fixture():
    mx, package = ModuleType('mlx.core'), ModuleType('mlx')
    mx.array = type('UnusedArrayAnnotation', (), {})
    package.core = mx
    cache = ModuleType('_e028_fake_native.cache')
    cache.__file__ = str(CORE / 'cache.py')
    runner_module = ModuleType('_e028_fake_native.engine.model_runner')
    runner_module.__file__ = str(CORE / 'engine/model_runner.py')
    modules = {'mlx': package, 'mlx.core': mx, cache.__name__: cache,
               runner_module.__name__: runner_module}
    with patch.dict(sys.modules, modules):
        # Cache imports only the fake backend. Execute the real runner class
        # without importing unrelated model/sampling modules or a device runtime.
        exec(compile(Path(cache.__file__).read_text(), cache.__file__, 'exec'), vars(cache))
        parsed = ast.parse(Path(runner_module.__file__).read_text())
        cls = next(n for n in parsed.body if isinstance(n, ast.ClassDef) and n.name == 'ModelRunner')
        runner_module.__dict__.update(mx=mx, DecodeBatch=cache.DecodeBatch,
            Request=type('Request', (), {}), SchedulerOutput=type('SchedulerOutput', (), {}))
        # Defer only annotations so this fake test also works on macOS Python 3.9.
        code = compile(ast.Module(body=[cls], type_ignores=[]), runner_module.__file__,
                       'exec', flags=__future__.annotations.compiler_flag)
        exec(code, vars(runner_module))
        model = SimpleNamespace(layers=[SimpleNamespace(is_attn=True), SimpleNamespace(is_attn=False)])
        runner = runner_module.ModelRunner(model, 2048)
        yield runner, runner_module, cache, mx


class IntegrationTests(unittest.TestCase):
    def test_off_does_not_inspect_runner_or_load_candidate(self):
        class Poison:
            def __getattribute__(self, name):
                raise AssertionError('OFF touched runner')
        with patch.object(integration, '_candidate', side_effect=AssertionError('loaded candidate')):
            self.assertIsNone(integration.install(Poison()))
            self.assertIsNone(integration.install(Poison(), enabled=False))
            with self.assertRaises(TypeError): integration.install(Poison(), enabled=1)

    def test_import_and_off_do_not_import_device_libraries(self):
        original = builtins.__import__
        def guarded(name, *args, **kwargs):
            if name.split('.')[0] in ('mlx', 'numpy', 'torch', 'tutorial_code'):
                raise AssertionError('runtime import ' + name)
            return original(name, *args, **kwargs)
        with patch('builtins.__import__', guarded):
            module = load('e028_integration_import_check', HERE / 'integration.py')
            self.assertIsNone(module.install(object()))

    def test_on_replaces_only_fresh_instance_batch_with_real_factory(self):
        with fixture() as (runner, module, cache, mx):
            before = dict(vars(runner)); original_rebuild = cache.DecodeBatch.rebuild
            other = module.ModelRunner(runner.model, 2048)
            installed = integration.install(runner, enabled=True)
            self.assertIs(runner.batch, installed)
            self.assertIs(installed.model, runner.model)
            self.assertTrue(installed.enabled)
            self.assertEqual(installed.counters, {'identity_refill': 0, 'fallback': 0})
            self.assertEqual((installed.req_ids, installed.lens, installed.layers), ([], [], [None, None]))
            self.assertIs(type(installed).__mro__[1], cache.DecodeBatch)
            self.assertIs(cache.DecodeBatch.rebuild, original_rebuild)
            self.assertIs(type(other.batch), cache.DecodeBatch)
            for key, value in before.items():
                if key != 'batch': self.assertIs(vars(runner)[key], value, key)
            self.assertEqual(set(vars(runner)), set(before))
            with self.assertRaisesRegex(ValueError, 'already installed'):
                integration.install(runner, enabled=True)

    def test_live_or_speculative_runner_is_rejected_without_changes(self):
        changes = [
            lambda r: r.batch.req_ids.append('active'),
            lambda r: r.batch.lens.append(3),
            lambda r: r.batch.layers.__setitem__(0, object()),
            lambda r: r.batch.layers.pop(),
            lambda r: r.staging.update(active=[]),
            lambda r: r.feats.update(active=[]),
            lambda r: setattr(r, '_ahead', object()),
            lambda r: setattr(r, 'spec', object()),
            lambda r: setattr(r.batch, 'model', object()),
        ]
        for index, change in enumerate(changes):
            with self.subTest(index=index), fixture() as (runner, _, _, _):
                change(runner); before = dict(vars(runner))
                with self.assertRaises(ValueError): integration.install(runner, enabled=True)
                for key, value in before.items(): self.assertIs(vars(runner)[key], value)

    def test_mixed_backend_and_source_pins_are_rejected(self):
        with fixture() as (runner, module, cache, _):
            old = runner.batch; module.mx = ModuleType('wrong_backend')
            with self.assertRaisesRegex(ValueError, 'MLX backend'): integration.install(runner, enabled=True)
            self.assertIs(runner.batch, old)
        for field in ('RUNNER_SHA256', 'CACHE_SHA256', 'CANDIDATE_SHA256'):
            with self.subTest(field=field), fixture() as (runner, _, _, _):
                old = runner.batch
                with patch.object(integration, field, '0'*64), self.assertRaisesRegex(ValueError, 'source mismatch'):
                    integration.install(runner, enabled=True)
                self.assertIs(runner.batch, old)
        with fixture() as (runner, module, cache, _):
            module.__file__ = str(CORE / 'other/model_runner.py')
            with self.assertRaisesRegex(ValueError, 'source roots'): integration.install(runner, enabled=True)

    def test_patched_native_method_is_rejected(self):
        with fixture() as (runner, module, cache, _):
            old = runner.batch
            cache.DecodeBatch.rebuild = lambda *args: None
            with self.assertRaisesRegex(ValueError, 'loaded native method'): integration.install(runner, enabled=True)
            self.assertIs(runner.batch, old)

    def test_failed_factory_leaves_original_runner_intact(self):
        with fixture() as (runner, _, _, _):
            old = runner.batch
            def fail(*args): raise RuntimeError('injected factory failure')
            with patch.object(integration, '_candidate', return_value=SimpleNamespace(make_batch_class=fail)):
                with self.assertRaisesRegex(RuntimeError, 'factory failure'): integration.install(runner, enabled=True)
            self.assertIs(runner.batch, old)


if __name__ == '__main__': unittest.main()
