"""Prepared stdlib-only tests; no oMLX/MLX/model/service imports.

These tests exercise observer ownership/teardown bookkeeping, not model loading.
They were written during the formal GPU run and must not be reported as run until
the root owner explicitly ends that CPU-test exclusion interval.
"""
import concurrent.futures
import importlib.util
from pathlib import Path
import threading
import types
import unittest

SPEC = importlib.util.spec_from_file_location('glm_load_probe_cpu', Path(__file__).with_name('load_probe.py'))
P = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(P)


class Log:
    def __init__(self):
        self.events = []

    def emit(self, kind, **values):
        self.events.append(dict(kind=kind, **values))


def observer():
    return P.Observer(Log(), {'source_pins': {str(Path(__file__).resolve()): P.sha(__file__)}}, None)


def passthrough(original):
    def call(*args, **kwargs):
        return original(*args, **kwargs)
    return call


class ObserverCPU(unittest.TestCase):
    def test_instance_binding_return_identity_and_final_restore(self):
        class FakeScheduler:
            def process(self, responses, *, suffix):
                self.seen = (responses, suffix)
                return responses
        original = FakeScheduler.__dict__['process']
        log = observer()
        values = object()
        try:
            log.hook(FakeScheduler, 'process', passthrough)
            instance = FakeScheduler()
            self.assertIs(instance.process(values, suffix='exact'), values)
            self.assertEqual(instance.seen, (values, 'exact'))
            self.assertEqual(log.active_calls(), {})
        finally:
            receipt = log.close()
        self.assertTrue(receipt['complete'])
        self.assertIs(FakeScheduler.__dict__['process'], original)

    def test_exception_keeps_identity_and_restores_partial_install(self):
        failure = RuntimeError('loader failure')
        def load():
            raise failure
        module = types.SimpleNamespace(load=load)
        class BadDescriptor:
            @staticmethod
            def method():
                return 1
        log = observer()
        try:
            log.hook(module, 'load', passthrough)
            with self.assertRaises(RuntimeError) as caught:
                module.load()
            self.assertIs(caught.exception, failure)
            with self.assertRaisesRegex(ValueError, 'descriptor'):
                log.hook(BadDescriptor, 'method', passthrough)
            self.assertEqual(log.active_calls(), {})
        finally:
            receipt = log.close()
        self.assertIs(module.load, load)
        self.assertTrue(receipt['complete'])
        self.assertEqual(receipt['restored'], ['load'])

    def test_running_executor_call_is_not_erased_by_attribute_restore(self):
        entered, finish = threading.Event(), threading.Event()
        result = object()
        def load():
            entered.set()
            if not finish.wait(2):
                raise RuntimeError('test worker stuck')
            return result
        module = types.SimpleNamespace(load=load)
        log = observer()
        log.hook(module, 'load', passthrough)
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(module.load)
            try:
                self.assertTrue(entered.wait(1))
                receipt = log.close()
                self.assertIs(module.load, load)
                self.assertTrue(receipt['complete'])
                self.assertEqual(receipt['active_at_restore'], {'load': 1})
                self.assertFalse(P.shutdown_is_complete(returned=True, retained=[],
                    engine_cleared=True, load_attempted=True, load_returned=False,
                    active_calls=log.active_calls(), hooks_restored=True))
            finally:
                finish.set()
            self.assertIs(future.result(timeout=1), result)
        self.assertEqual(log.active_calls(), {})
        # The orphaned async await did not return an engine even though the
        # executor function later exited: empty entries still cannot grant PASS.
        self.assertFalse(P.shutdown_is_complete(returned=True, retained=[],
            engine_cleared=True, load_attempted=True, load_returned=False,
            active_calls={}, hooks_restored=True))

    def test_failed_restore_still_restores_other_seams(self):
        def first():
            return 1
        def second():
            return 2
        module = types.SimpleNamespace(first=first, second=second)
        log = observer()
        log.hook(module, 'first', passthrough)
        log.hook(module, 'second', passthrough)
        actual_stop = log.patches[-1][-1].stop
        def broken_stop():
            actual_stop()
            raise RuntimeError('injected restore failure')
        log.patches[-1][-1].stop = broken_stop
        receipt = log.close()
        self.assertFalse(receipt['complete'])
        self.assertIs(module.first, first)
        self.assertIs(module.second, second)
        self.assertEqual(receipt['restored'], ['first'])

    def test_shutdown_requires_every_boundary(self):
        valid = dict(returned=True, retained=[], engine_cleared=True,
            load_attempted=True, load_returned=True, active_calls={}, hooks_restored=True)
        self.assertTrue(P.shutdown_is_complete(**valid))
        for key, value in (('returned', False), ('retained', ['model']),
                           ('engine_cleared', False), ('load_returned', False),
                           ('active_calls', {'load': 1}), ('hooks_restored', False)):
            with self.subTest(key=key):
                self.assertFalse(P.shutdown_is_complete(**dict(valid, **{key: value})))


if __name__ == '__main__':
    unittest.main()
