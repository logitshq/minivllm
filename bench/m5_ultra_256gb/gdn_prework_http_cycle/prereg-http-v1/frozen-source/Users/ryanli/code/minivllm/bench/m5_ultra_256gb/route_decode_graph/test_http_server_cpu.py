"""Prepared, unrun CPU-only entry/observer tests. No model or socket operations.

Real pinned donor worker/observer and RouteWitness bookkeeping run against fake
callables only. These tests do not grant HTTP, arithmetic or device qualification.
"""
import ast
import builtins
from contextlib import redirect_stderr
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from bench.m5_ultra_256gb.route_decode_graph import http_server_v2 as m
from bench.m5_ultra_256gb.route_decode_graph import integration


def argv(*extra):
    return ['--model', '/cpu/model', '--port', '18023', '--package-root', '/cpu/source',
        '--source-manifest', '/cpu/manifest.json', '--source-manifest-sha256', '0' * 64,
        '--max-len', '2048', '--max-seqs', '1', '--max-batched-tokens', '2048',
        '--cache-limit-gib', '2', '--compile-decode', 'on', '--fuse-decode', 'on',
        '--gates-compile', 'on', '--norm-reuse', 'off', '--tcp-nodelay', 'on',
        '--io-timeout-seconds', '300.0', *extra]


def context(B=1, used=1023, capacity=1536):
    return SimpleNamespace(max_len=used, caches=[SimpleNamespace(
        K=SimpleNamespace(shape=(B, 4, capacity, 256)),
        V=SimpleNamespace(shape=(B, 4, capacity, 256))) for _ in range(16)])


class Graph:
    def __init__(self):
        self._functions = {}
        self.calls = self.compiled_calls = 0
        self.output = object()

    def _eligible(self, *args):
        raise AssertionError('observer duplicated eligibility')

    def __call__(self, tokens, ctx):
        self.calls += 1
        if min(c.K.shape[2] for c in ctx.caches) < 1536:
            return self.output
        key = 1023 if ctx.max_len < 1024 else 1536
        if key not in self._functions:
            def compiled(*args):
                self.compiled_calls += 1
                return self.output
            self._functions[key] = compiled
        return self._functions[key](tokens, ctx)


class Runner:
    def __init__(self):
        self.original = object()
        self.decode_model = self.original
        self.batch = SimpleNamespace(req_ids=[])

    def run(self, out):
        self.batch.req_ids = [r.req_id for r in out.decode]
        return self.decode_model(SimpleNamespace(shape=(len(out.decode),)), context(len(out.decode)))


class Engine:
    def __init__(self):
        self.runner = Runner()
        self.scheduler = SimpleNamespace(running=[], waiting=[])

    def add_request(self, ids, sampling, *, req_id):
        self.scheduler.waiting.append(SimpleNamespace(req_id=req_id))
        return req_id

    def has_work(self):
        return bool(self.scheduler.waiting or self.scheduler.running)

    def step(self):
        rows = list(self.scheduler.waiting)
        self.scheduler.waiting.clear()
        self.scheduler.running[:] = rows
        self.runner.run(SimpleNamespace(decode=rows, prefill=[], finished_ids=[]))
        self.scheduler.running.clear()
        self.runner.batch.req_ids.clear()
        # A lifecycle-only CPU marker, never a model/token oracle.
        return [(row.req_id, 42, True) for row in rows]


def install_fake(engine):
    graph = Graph()
    dispatch = integration._Dispatch(engine.runner.decode_model, graph)
    handle = integration.Installation(engine.runner, engine.runner.decode_model, dispatch)
    engine.runner.decode_model = dispatch
    return handle, graph


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.donor = m.load_fixed(m.DONOR, '_http_v2_cpu_donor')
        cls.witness = m.load_fixed(m.WITNESS, '_http_v2_cpu_witness')
        cls.counter = m.load_fixed(m.COUNTER, '_http_v2_cpu_counter')

    def make_observer(self, engine, handle, trace=None, route=None):
        trace = io.StringIO() if trace is None else trace
        route = io.StringIO() if route is None else route
        observer = m.QualificationObserver(engine, handle, self.donor, trace, route,
            'HTTP-C1', self.witness, self.counter)
        return observer, trace, route

    def test_import_and_default_off_do_not_import_model_or_load_sources(self):
        original = builtins.__import__
        def guarded(name, *args, **kwargs):
            if name.split('.')[0] in ('mlx', 'mlx_lm', 'mlx_vlm', 'torch', 'numpy', 'tutorial_code'):
                raise AssertionError('forbidden runtime import: ' + name)
            return original(name, *args, **kwargs)
        with patch('builtins.__import__', guarded):
            spec = importlib.util.spec_from_file_location('_http_v2_cpu_import', m.__file__)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            with patch.object(module, 'load_fixed', side_effect=AssertionError('source load')):
                args = module.arguments(argv())
        self.assertEqual((args.phase, args.route_decode_graph), ('formal', 'off'))
        self.assertIsNone(args.qualification_trace)
        self.assertIsNone(args.route_events)

    def test_exact_qualification_pair_and_formal_on_without_observers(self):
        args = m.arguments(argv('--phase', 'qualification', '--route-decode-graph', 'on',
            '--qualification-trace', '/cpu/trace.jsonl', '--route-events', '/cpu/route.jsonl'))
        self.assertEqual(args.max_len, 2048)
        self.assertEqual((args.gates_compile, args.norm_reuse), ('on', 'off'))
        self.assertEqual(args.route_events, Path('/cpu/route.jsonl'))
        formal = m.arguments(argv('--route-decode-graph', 'on'))
        self.assertIsNone(formal.qualification_trace)
        off = m.arguments(argv('--phase', 'qualification', '--qualification-trace', '/cpu/trace'))
        self.assertEqual(off.route_decode_graph, 'off')
        self.assertIsNone(off.route_events)

    def test_missing_mixed_observers_shared_path_and_changed_settings_rejected(self):
        on = ['--phase', 'qualification', '--route-decode-graph', 'on',
              '--qualification-trace', '/cpu/trace', '--route-events', '/cpu/route']
        cases = [
            ['--qualification-trace', '/cpu/trace'], ['--route-events', '/cpu/route'],
            ['--phase', 'qualification'],
            ['--phase', 'qualification', '--qualification-trace', '/cpu/trace', '--route-events', '/cpu/route'],
            on[:-2], [*on, '--route-events', '/cpu/trace'],
            [*on, '--max-len', '1536'], [*on, '--gates-compile', 'off'],
            [*on, '--norm-reuse', 'on'], [*on, '--max-seqs', '2']]
        for extra in cases:
            with self.subTest(extra=extra), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    m.arguments(argv(*extra))

    def test_identity_reports_real_sources_observation_not_future_hits(self):
        args = m.arguments(argv('--phase', 'qualification', '--route-decode-graph', 'on',
            '--qualification-trace', '/cpu/trace', '--route-events', '/cpu/route'))
        pins = {str(p): h for p, h in m.FIXED.items()}
        pins[str(Path(m.__file__).resolve())] = 'new-http-source'
        fields = m.identity_fields(args, pins, installed=True, route_observer=True)
        route = fields['route_decode_graph']
        self.assertEqual(fields['server_source_sha256'], 'new-http-source')
        self.assertEqual(fields['server_source_path'], str(Path(m.__file__).resolve()))
        self.assertEqual(fields['frozen_server_source_sha256'], m.FIXED[m.DONOR])
        self.assertTrue(route['route_observer'])
        self.assertFalse(route['actual_compiled_invocations_observed'])
        self.assertFalse(route['identity_refill_enabled'])
        self.assertEqual(route['witness_source_sha256'], pins[str(m.WITNESS)])
        self.assertEqual(route['witness_counter_source_sha256'], pins[str(m.COUNTER)])
        for installed, observed in ((False, True), (True, False), (1, True), (True, 1)):
            with self.assertRaises(RuntimeError):
                m.identity_fields(args, pins, installed=installed, route_observer=observed)

    def test_witness_counts_actual_calls_and_preserves_same_step_ahead_order(self):
        engine = Engine()
        handle, graph = install_fake(engine)
        observer, trace, route = self.make_observer(engine, handle)
        observer.startup({'source': 'test'})
        observer.trace({'event': 'before_step'})
        for used in (1023, 1024):
            result = handle.decoder.long(SimpleNamespace(shape=(1,)), context(used=used))
            self.assertIs(result, graph.output)
        handle.decoder.long(SimpleNamespace(shape=(1,)), context(capacity=1024))
        rows = [json.loads(line) for line in route.getvalue().splitlines()]
        self.assertEqual(rows[0]['event'], 'route-startup')
        self.assertEqual(rows[0]['schema'], m.ROUTE_SCHEMA)
        self.assertEqual([r['step'] for r in rows[1:]], [1, 1, 1])
        self.assertEqual([r['actual_invocations'] for r in rows[1:]], [1, 1, 0])
        self.assertTrue(all(r['phase'] == 'HTTP-C1' and r['direct_label'] is None for r in rows[1:]))
        self.assertEqual((graph.calls, graph.compiled_calls), (3, 2))
        self.assertEqual(observer.coverage, dict(route_hits={'HTTP-C1': {'1023': 1, '1536': 1}}, long_fallback_calls=1))
        self.assertNotIn('route-call', trace.getvalue())
        observer.restore()
        handle.restore()

    def test_real_donor_worker_trace_and_uuid_delivery_remain_unmodified(self):
        engine = Engine()
        handle, graph = install_fake(engine)
        observer, trace, route = self.make_observer(engine, handle)
        worker = self.donor.BatchWorker(trace=observer.trace)
        worker.initialize(engine, lambda **kwargs: SimpleNamespace(**kwargs), {'cpu': True})
        observer.startup({'cpu': True})
        ids = []
        for _ in range(2):
            job = worker.submit([11, 12], 1)
            ids.append(job.rid)
            self.assertTrue(worker.tick())
            self.assertEqual(job.events.get_nowait(), {'token': 42, 'done': True})
        events = [json.loads(line) for line in trace.getvalue().splitlines()]
        expected = ['before_step', 'actual_scheduler_output', 'actual_model_runner_batch', 'after_step']
        self.assertEqual([x['event'] for x in events], ['startup', *expected, *expected])
        self.assertEqual(events[1]['newly_submitted_to_scheduler'], [ids[0]])
        self.assertEqual(events[5]['newly_submitted_to_scheduler'], [ids[1]])
        self.assertNotEqual(ids[0], ids[1])
        routes = [json.loads(line) for line in route.getvalue().splitlines()][1:]
        self.assertEqual([x['step'] for x in routes], [1, 2])
        self.assertEqual((graph.calls, graph.compiled_calls), (2, 2))
        observer.restore()
        handle.restore()

    def test_restore_order_releases_observer_then_witness_then_installation(self):
        engine = Engine()
        original = engine.runner.decode_model
        handle, graph = install_fake(engine)
        functions = graph._functions
        observer, _, _ = self.make_observer(engine, handle)
        witness = observer.witness
        self.assertIs(handle.decoder.long, witness)
        observer.restore()
        self.assertNotIn('run', vars(engine.runner))
        self.assertIs(handle.decoder.long, graph)
        self.assertIs(graph._functions, functions)
        self.assertIsNone(observer.runner)
        self.assertIsNone(observer.witness)
        self.assertFalse(handle.closed)
        handle.restore()
        self.assertIs(engine.runner.decode_model, original)
        observer.restore()

    def test_qualification_off_only_uses_scheduling_observer(self):
        engine = Engine()
        before = engine.runner.decode_model
        trace = io.StringIO()
        observer = m.QualificationObserver(engine, None, self.donor, trace, None, 'HTTP-C1')
        self.assertIsNone(observer.witness)
        self.assertIs(engine.runner.decode_model, before)
        observer.startup({'route': False})
        self.assertEqual(len(trace.getvalue().splitlines()), 1)
        observer.restore()

    def test_failed_observer_constructor_restores_partial_methods_and_witness(self):
        engine = Engine()
        original = lambda out: None
        engine.runner.run = original
        handle, graph = install_fake(engine)
        functions = graph._functions
        def broken(engine, trace):
            engine.runner.run = lambda out: 'partial-observer'
            raise RuntimeError('observer constructor failed')
        with patch.object(self.donor, 'observe_runner', side_effect=broken):
            with self.assertRaisesRegex(RuntimeError, 'constructor failed'):
                self.make_observer(engine, handle)
        self.assertIs(engine.runner.run, original)
        self.assertIs(handle.decoder.long, graph)
        self.assertIs(graph._functions, functions)
        handle.restore()

    def test_failed_witness_constructor_restores_its_partial_function_dict(self):
        engine = Engine()
        handle, graph = install_fake(engine)
        functions = graph._functions
        def broken(graph, *args):
            graph._functions = {'partial': object()}
            raise RuntimeError('witness constructor failed')
        with patch.object(self.witness, 'RouteWitness', side_effect=broken):
            with self.assertRaisesRegex(RuntimeError, 'witness constructor failed'):
                self.make_observer(engine, handle)
        self.assertNotIn('run', vars(engine.runner))
        self.assertIs(handle.decoder.long, graph)
        self.assertIs(graph._functions, functions)
        handle.restore()

    def test_foreign_runner_or_graph_observer_is_not_overwritten(self):
        for field in ('run', 'long'):
            with self.subTest(field=field):
                engine = Engine()
                handle, _ = install_fake(engine)
                observer, _, _ = self.make_observer(engine, handle)
                target = engine.runner if field == 'run' else handle.decoder
                foreign = object()
                setattr(target, field, foreign)
                with self.assertRaisesRegex(RuntimeError, 'foreign'):
                    observer.restore()
                self.assertIs(getattr(target, field), foreign)
                self.assertFalse(observer.closed)
                self.assertFalse(handle.closed)

    def test_main_reuses_handler_and_worker_and_keeps_cleanup_on_worker(self):
        tree = ast.parse(Path(m.__file__).read_text())
        main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
        worker = next(n for n in main.body if isinstance(n, ast.FunctionDef) and n.name == 'load_and_run')
        calls = [(n.lineno, ast.unparse(n.func)) for n in ast.walk(worker) if isinstance(n, ast.Call)]
        line = lambda name: next(n for n, call in calls if call == name)
        self.assertLess(line('integration.install'), line('QualificationObserver'))
        self.assertLess(line('QualificationObserver'), line('identity_fields'))
        self.assertLess(line('identity_fields'), line('worker.initialize'))
        self.assertLess(line('worker.run'), line('observation.restore'))
        self.assertLess(line('observation.restore'), line('installation.restore'))
        text = ast.unparse(main)
        self.assertIn('donor.BatchWorker(', text)
        self.assertIn('donor.handler_class(', text)
        self.assertNotIn('parameter_checks(', text)
        self.assertNotIn('ThreadingHTTPServer(', text)
        # The worker alone performs teardown. Main may only close files when no
        # thread was started; no parallel model restore or forced tensor eval.
        main_finally = next(n for n in reversed(main.body) if isinstance(n, ast.Try)).finalbody
        final = ast.unparse(ast.Module(body=main_finally, type_ignores=[]))
        self.assertNotIn('.restore(', final)
        self.assertNotIn('mx.eval(', ast.unparse(tree))


if __name__ == '__main__':
    unittest.main()
