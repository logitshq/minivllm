"""CPU fake behavior only: no socket, tokenizer runtime, model, or GPU execution."""
import ast
import contextlib
import copy
import importlib.util
import inspect
import io
import json
from pathlib import Path
import queue
import sys
import tempfile
import textwrap
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from . import http_server as entry


class Worker:
    def __init__(self):
        self.ready = threading.Event(); self.ready.set()
        self.lock, self.pending = threading.Lock(), queue.Queue()
        self.jobs, self.active, self.failure = {}, {}, None
        self.engine = SimpleNamespace(has_work=lambda: False)
    def fail(self, message):
        if self.failure is None: self.failure = message
        self.pending.put(None)


class Thread:
    ident = 1
    def __init__(self, *, stuck=False): self.alive, self.stuck = True, stuck
    def is_alive(self): return self.alive
    def join(self, timeout):
        if not self.stuck: self.alive = False


def finished(warm=0):
    drain = entry.Drain('a' * 32, warm_requests=warm)
    for i in range(8 + warm):
        drain.handler_enter(i)
        drain.begin(i)
        drain.submitted(SimpleNamespace(rid='rid-' + str(i)))
        drain.done()
        drain.end(i)
        drain.handler_exit(i)
    return drain


def accepted():
    drain, worker = finished(), Worker()
    drain.handler_enter(100)
    drain.request('a' * 32, worker, 100)
    drain.handler_exit(100)
    return drain, worker


def donor():
    path = entry.HERE.parent / 'pareto_realistic/native_server.py'
    name = '_diagnostic_cpu_actual_native'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


class Tests(unittest.TestCase):
    def test_empty_cli_is_inert_default_off(self):
        with patch.object(entry, 'load', side_effect=AssertionError('must not load')), contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(entry.main([]), 0)
        self.assertEqual(json.loads(out.getvalue()), entry.contract())
        self.assertFalse(entry.contract()['performance_qualified'])

    def test_actual_arguments_keep_original_flags_and_reject_other_domains(self):
        combined = entry.load(entry.COMBINED, '_diagnostic_cpu_arguments')
        argv = ['--diagnostic-out','/tmp/not-created','--drain-token','a'*32,
            '--model','/tmp/model-not-opened','--port','18023','--package-root','/tmp/no-import',
            '--source-manifest','/tmp/not-read','--source-manifest-sha256','a'*64,
            '--max-len','2048','--max-seqs','1','--max-batched-tokens','2048','--cache-limit-gib','24',
            '--compile-decode','on','--fuse-decode','on','--gates-compile','on','--norm-reuse','off',
            '--tcp-nodelay','on','--io-timeout-seconds','300','--route-decode-graph','on','--gdn-prework','on']
        args, options = entry.arguments(argv, combined)
        self.assertEqual((options.profile, args.max_seqs, args.phase), ('off', 1, 'formal'))
        for mode in ('runner', 'full'):
            self.assertEqual(entry.arguments(argv + ['--profile',mode], combined)[1].profile, mode)
        bad = list(argv); bad[bad.index('--max-seqs')+1] = '8'
        with self.assertRaisesRegex(RuntimeError, 'C1'): entry.arguments(bad, combined)

    def test_all_eight_normal_done_then_control_then_actual_stop(self):
        drain, worker = accepted()
        drain.stop_worker(worker)
        self.assertIsNone(worker.failure)
        thread = Thread(); thread.join(1)
        drain.verify_stopped(worker, thread)
        result = drain.receipt()
        self.assertEqual((result['phase'], result['normal_done_flushes']), ('DRAINED',8))
        self.assertTrue(all(r['entered_ns'] <= r['done_flush_ns'] <= r['handler_exit_ns'] for r in result['requests']))

    def test_done_requires_exact_successful_write_and_flush(self):
        drain = entry.Drain('a'*32); drain.handler_enter(1); drain.begin(1)
        drain.submitted(SimpleNamespace(rid='one'))
        class Writer(io.BytesIO):
            def flush(self): raise BrokenPipeError('real flush failed')
        tap = entry.DoneWriter(Writer(), drain)
        self.assertEqual(tap.write(entry.DONE), len(entry.DONE))
        self.assertIsNone(drain.rows[0]['done_flush_ns'])
        with self.assertRaises(BrokenPipeError): tap.flush()
        self.assertIsNone(drain.rows[0]['done_flush_ns'])
        tap = entry.DoneWriter(SimpleNamespace(write=lambda b: len(b)-1), drain)
        with self.assertRaisesRegex(RuntimeError, 'incomplete'): tap.write(entry.DONE)
        self.assertIsNone(drain.rows[0]['done_flush_ns'])

    def test_done_writer_forwards_bytes_once_and_duplicate_done_rejected(self):
        drain = entry.Drain('a'*32); drain.handler_enter(1); drain.begin(1)
        drain.submitted(SimpleNamespace(rid='one'))
        raw = io.BytesIO(); tap = entry.DoneWriter(raw, drain)
        chunk = b'data: {"choices":[]}\n\n'
        self.assertEqual(tap.write(chunk), len(chunk)); tap.flush()
        self.assertIsNone(drain.rows[0]['done_flush_ns'])
        tap.write(entry.DONE); tap.flush(); tap.flush()
        self.assertEqual(raw.getvalue(), chunk + entry.DONE)
        tap.write(entry.DONE)
        with self.assertRaisesRegex(RuntimeError, 'unique'): tap.flush()

    def test_drain_rejects_missing_done_bad_nonce_and_live_http_handler(self):
        worker = Worker(); drain = finished(); drain.handler_enter(100)
        with self.assertRaisesRegex(RuntimeError, 'token'): drain.request('bad', worker, 100)
        drain.handler_enter(101)
        with self.assertRaisesRegex(RuntimeError, 'handlers'): drain.request('a'*32, worker, 100)
        drain.handler_exit(101)
        drain.rows[-1]['done_flush_ns'] = None
        with self.assertRaisesRegex(RuntimeError, 'normal DONE'): drain.request('a'*32, worker, 100)
        self.assertEqual(drain.phase, 'OPEN')

    def test_drain_rejects_all_work_and_failure_sources(self):
        cases = ('jobs','active','pending','engine','failure','not-ready')
        for case in cases:
            with self.subTest(case=case):
                drain, worker = finished(), Worker(); drain.handler_enter(100)
                if case in ('jobs','active'): getattr(worker,case)['rid'] = object()
                elif case == 'pending': worker.pending.put(object())
                elif case == 'engine': worker.engine.has_work = lambda: True
                elif case == 'failure': worker.failure = 'original runtime failure'
                else: worker.ready.clear()
                with self.assertRaises(RuntimeError): drain.request('a'*32, worker,100)
                self.assertFalse(drain.stop.is_set())

    def test_fixed_warm_scope_and_admission_closed_or_concurrent(self):
        drain = finished(8)
        self.assertEqual([r['phase'] for r in drain.rows], ['warm']*8 + ['diagnostic']*8)
        self.assertEqual(drain.receipt()['expected_requests'], 8)
        drain.handler_enter(100)
        with self.assertRaisesRegex(RuntimeError, 'count exceeded'): drain.begin(100)
        d = entry.Drain('a'*32); d.handler_enter(1); d.begin(1); d.handler_enter(2)
        with self.assertRaisesRegex(RuntimeError, 'C1'): d.begin(2)
        d, _ = accepted(); d.handler_enter(4)
        with self.assertRaisesRegex(RuntimeError, 'admission closed'): d.begin(4)

    def test_next_c1_post_after_done_before_prior_finally_is_allowed_but_not_drained(self):
        d=entry.Drain('a'*32); d.handler_enter(1); first=d.begin(1)
        d.submitted(SimpleNamespace(rid='first')); d.done()
        # Simulate a second handler while the prior thread is paused after the
        # final original flush; real HTTP intervals do not overlap in this case.
        d.handler_enter(2); second=d.begin(2)
        self.assertIs(d.active[1],first); self.assertIs(d.active[2],second)
        d.submitted(SimpleNamespace(rid='second'))
        d.handler_enter(3)
        with self.assertRaisesRegex(RuntimeError,'C1'): d.begin(3)
        d.handler_exit(3)
        d.done(); d.end(2); d.handler_exit(2)
        d.local.row=first; d.end(1); d.handler_exit(1)
        self.assertEqual(d.receipt()['normal_done_flushes'],2)
        d.handler_enter(100)
        with self.assertRaisesRegex(RuntimeError,'all fixed'): d.request('a'*32,Worker(),100)

    def test_worker_loop_keeps_original_tick_and_real_exceptions(self):
        worker, stop, calls = Worker(), threading.Event(), []
        def tick(*, block): calls.append(block); stop.set()
        worker.tick = tick
        entry.worker_loop(worker, stop)
        self.assertEqual(calls, [True]); self.assertIsNone(worker.failure)
        stop.clear()
        def broken(*, block): raise ValueError('actual compute failure')
        worker.tick = broken
        entry.worker_loop(worker, stop)
        self.assertIn('ValueError: actual compute failure', worker.failure)

    def test_worker_loop_ast_only_adds_explicit_stop_predicate(self):
        old = ast.parse(textwrap.dedent(inspect.getsource(donor().BatchWorker.run))).body[0]
        new = ast.parse(textwrap.dedent(inspect.getsource(entry.worker_loop))).body[0]
        new.body.pop(0)  # Docstring.
        new.name, new.args = old.name, copy.deepcopy(old.args)
        new.body[0].body[0].test = new.body[0].body[0].test.values[0]
        class Rename(ast.NodeTransformer):
            def visit_Name(self, node):
                if node.id == 'worker': node.id = 'self'
                return node
        new = Rename().visit(new)
        self.assertEqual(ast.dump(new), ast.dump(old))

    def test_restore_after_actual_join_and_profile_write_last(self):
        drain, worker = accepted(); order = []
        class Scope:
            def __init__(self, name): self.name=name
            def restore(self): order.append(self.name)
        class Recorder:
            def write_once(self, path, **kwargs):
                self.assert_drained = kwargs['drained']; order.append('write'); return {'status':'RECORDED'}
        recorder = Recorder()
        with tempfile.TemporaryDirectory() as tmp:
            result = entry.finalize(worker, Thread(), drain,
                [Scope(n) for n in ('submit','delivery','runner','combined')], recorder,{},Path(tmp),
                timeout=1, handlers_joined=True)
        self.assertEqual(order,['submit','delivery','runner','combined','write'])
        self.assertEqual(result['status'],'RECORDED'); self.assertIsNone(result['worker_failure'])
        self.assertTrue(recorder.assert_drained)

    def test_live_thread_or_unjoined_handlers_prevent_restore_and_write(self):
        for stuck, joined in ((True,True),(False,False)):
            with self.subTest(stuck=stuck,joined=joined):
                drain, worker = accepted(); calls=[]
                scope = SimpleNamespace(restore=lambda: calls.append('restore'))
                recorder = SimpleNamespace(write_once=lambda *a,**k: calls.append('write'))
                with tempfile.TemporaryDirectory() as tmp:
                    result = entry.finalize(worker,Thread(stuck=stuck),drain,[scope],recorder,{},Path(tmp),
                        timeout=0.01,handlers_joined=joined)
                self.assertEqual(calls,[]); self.assertEqual(result['status'],'INCOMPLETE')

    def test_restore_error_continues_cleanup_but_never_writes_success(self):
        drain, worker = accepted(); calls=[]
        def broken(): calls.append('delivery'); raise RuntimeError('restore failed')
        scopes=[SimpleNamespace(restore=broken),SimpleNamespace(restore=lambda: calls.append('runner'))]
        recorder=SimpleNamespace(write_once=lambda *a,**k: calls.append('write'))
        with tempfile.TemporaryDirectory() as tmp:
            result=entry.finalize(worker,Thread(),drain,scopes,recorder,{},Path(tmp),timeout=1,handlers_joined=True)
        self.assertEqual(calls,['delivery','runner']); self.assertEqual(result['status'],'INCOMPLETE')
        self.assertIn('restore failed', ''.join(result['errors']))

    def test_actual_native_sse_body_reused_and_failed_final_flush_cannot_drain(self):
        native=donor()
        for fail in (False,True):
            with self.subTest(fail=fail):
                worker=native.BatchWorker(); worker.initialize(SimpleNamespace(has_work=lambda:False),None,{'eos_ids':[99]})
                original=worker.submit
                def immediate(ids,cap):
                    job=original(ids,cap); worker.pending.get_nowait(); worker.jobs.pop(job.rid)
                    job.finished=True; job.events.put({'token':7,'done':True}); return job
                worker.submit=immediate
                drain=entry.Drain('a'*32); tap=entry.SubmitTap(worker,drain)
                tokenizer=SimpleNamespace(encode=lambda text,**kw:SimpleNamespace(ids=[1,2]),decode=lambda ids,**kw:'x')
                base=native.handler_class(worker,tokenizer,max_len=2048,tcp_nodelay=True,io_timeout=1)
                handler=entry.handler_class(base,worker,drain).__new__(entry.handler_class(base,worker,drain))
                handler.path='/v1/completions'
                body=json.dumps({'prompt':'p','stream':True,'max_tokens':1}).encode()
                handler.headers={'Content-Length':str(len(body))}; handler.rfile=io.BytesIO(body)
                class Writer(io.BytesIO):
                    def flush(self):
                        if fail and self.getvalue().endswith(entry.DONE): raise BrokenPipeError('terminal flush')
                raw=Writer(); handler.wfile=entry.DoneWriter(raw,drain)
                handler.send_response=lambda code: self.assertEqual(code,200)
                handler.send_header=lambda *args:None; handler.end_headers=lambda:None
                handler.send_error=lambda *args:self.fail(str(args))
                drain.handler_enter(id(handler)); handler.do_POST(); drain.handler_exit(id(handler))
                payload=raw.getvalue()
                self.assertTrue(payload.endswith(entry.DONE))
                terminal=json.loads(payload.split(b'data: ')[2].split(b'\n\n')[0])
                self.assertEqual(terminal['benchmark_output_ids'],[7])
                self.assertEqual(terminal['benchmark_prompt_ids'],[1,2])
                self.assertEqual(drain.receipt()['normal_done_flushes'],0 if fail else 1)
                self.assertEqual(bool(drain.errors),fail)
                tap.restore(); self.assertIs(worker.submit,immediate)

    def test_fixed_sources_and_no_runtime_imports_at_module_level(self):
        for path,digest in entry.FIXED.items(): self.assertEqual(entry.sha(path),digest)
        tree=ast.parse(Path(entry.__file__).read_text())
        imports=[node for node in tree.body if isinstance(node,(ast.Import,ast.ImportFrom))]
        for node in imports:
            text=ast.unparse(node)
            self.assertNotIn('mlx',text); self.assertNotIn('tokenizers',text); self.assertNotIn('numpy',text)


if __name__ == '__main__':
    unittest.main()
