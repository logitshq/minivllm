"""Fake CPU contracts, not numerical or device qualification.

The original ModelRunner class is AST-loaded with explicitly fake arrays/backend.
No tutorial_code/MLX imports, socket, service, subprocess or model payload.
"""
import __future__
import ast
import gc
import importlib.util
import io
import json
from pathlib import Path
import queue
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import weakref

from bench.m5_ultra_256gb.c1_pipeline_profile import instrumentation as m, delivery as d

ROOT = Path(__file__).resolve().parents[3]
RUNNER = ROOT/'tutorial_code/miniengine/engine/model_runner.py'
DONOR = ROOT/'bench/m5_ultra_256gb/pareto_realistic/native_server.py'


class Value:
    def __init__(self, values, backend): self.values, self.backend = list(values), backend; self.shape = (len(values),)
    def astype(self, dtype):
        self.backend.log.append(('astype', tuple(self.values), dtype)); return Value(self.values, self.backend)
    def tolist(self):
        self.backend.log.append(('tolist', tuple(self.values))); return list(self.values)
    def item(self): return self.tolist()[0]
    def __getitem__(self, index): return Value(self.values[index] if isinstance(index, slice) else [self.values[index]], self.backend)


class MX:
    int32 = 'int32'; bfloat16 = 'bf16'
    def __init__(self): self.log = []
    def array(self, values): return Value(values, self)
    def eval(self, *values): self.log.append(('eval', tuple(tuple(v.values) for v in values)))
    def async_eval(self, *values): self.log.append(('async_eval', tuple(tuple(v.values) for v in values)))
    def synchronize(self): raise AssertionError('new synchronize')


class State:
    def __init__(self, conv, S): self.conv, self.S = conv, S


class Context:
    def __init__(self, mode, caches, **fields): self.mode, self.caches = mode, caches; vars(self).update(fields)


class Batch:
    def __init__(self, model):
        self.req_ids, self.lens = ['rid'], [257]
        self.layers = [State(model.mx.array([0]), model.mx.array([0]))]
    @property
    def size(self): return len(self.req_ids)
    def arrays(self): return [v for s in self.layers for v in (s.conv, s.S)]
    def clear(self): self.req_ids.clear(); self.layers.clear()


class Model:
    def __init__(self, mx):
        self.mx = mx; self.cfg = NS(quantization={}); self.embed_tokens = NS(_output_dtype=mx.bfloat16)
    def __call__(self, tokens, ctx, **kwargs):
        self.mx.log.append(('model', ctx.mode, getattr(ctx, 'max_len', None), tuple(tokens.values)))
        for cache in ctx.caches:
            cache.S = self.mx.array([getattr(ctx, 'max_len', 0)])
        return self.mx.array([tokens.values[-1]+1])


class Graph:
    def __init__(self, model): self.model, self._functions = model, {}
    def __call__(self, tokens, ctx):
        key = 1023 if ctx.max_len < 1024 else 1536
        if key not in self._functions: self._functions[key] = lambda ids, context: self.model(ids, context)
        return self._functions[key](tokens, ctx)


def fixture(*, graph=False):
    mx = MX(); model = Model(mx)
    namespace = dict(mx=mx, BatchState=State, BatchKV=type('KV', (), {}), DecodeBatch=Batch,
        ForwardContext=Context, sample=lambda logits, params: logits,
        prefill_batch_backend_supported=lambda: False, make_cache=lambda _: [])
    tree = ast.parse(RUNNER.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'ModelRunner')
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(RUNNER), 'exec',
                 flags=__future__.annotations.compiler_flag), namespace)
    runner = namespace['ModelRunner'](model, 2048)
    if graph: runner.decode_model = Graph(model)
    req = NS(req_id='rid', num_computed_tokens=257, num_tokens=258, output_token_ids=[10],
             sampling=NS(temperature=0, top_p=1.0, top_k=0, max_tokens=8, ignore_eos=False, stop_token_ids=[]))
    out = NS(decode=[req], prefill=[], finished_ids=[])
    return runner, mx, req, out


def advance(runner, req, out, steps=4):
    outputs = []
    for _ in range(steps):
        result = runner.run(out); outputs.append(result)
        req.output_token_ids.append(result['rid']); req.num_computed_tokens += 1; req.num_tokens += 1
    return outputs


def donor():
    name = '_c1_profile_fake_http'
    spec = importlib.util.spec_from_file_location(name, DONOR)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class Tests(unittest.TestCase):
    def test_default_off_does_not_inspect_objects(self):
        class Poison:
            def __getattribute__(self, key): raise AssertionError(key)
        self.assertIsNone(m.install(Poison()))
        self.assertIsNone(d.install(*[Poison()]*5))
        with self.assertRaises(TypeError): m.install(Poison(), enabled=1)

    def test_actual_frozen_runner_outputs_eval_order_and_ahead_are_unchanged(self):
        plain, pmx, req, out = fixture(); expected = advance(plain, req, out)
        runner, mx, req, out = fixture(); original_globals = type(runner).run.__globals__
        session = m.Session(runner, type(runner).run, m.Recorder(), [], {})
        self.assertIsNot(session.cloned.__func__.__globals__, original_globals)
        self.assertIs(session.cloned.__func__.__code__, type(runner).run.__code__)
        self.assertIs(original_globals['mx'], mx)
        self.assertEqual(advance(runner, req, out), expected)
        self.assertEqual(mx.log, pmx.log)  # Includes existing eval/async/tolist and real call order.
        steps = [row for row in session.recorder.rows if row['kind']=='runner_step']
        self.assertEqual([r['actual_decode_calls'] for r in steps], [['current','future'],['future'],['future'],['future']])
        self.assertEqual([r['ahead_consumed'] for r in steps], [False,True,True,True])
        self.assertEqual(sum(r['kind']=='existing_eval' for r in session.recorder.rows),4)
        self.assertEqual(sum(r['kind']=='existing_async_eval' for r in session.recorder.rows),8)
        session.restore(); self.assertNotIn('run',vars(runner)); self.assertIs(runner.decode_model,runner.model)

    def test_actual_callable_invocations_distinguished_from_family_creation(self):
        runner, mx, req, out = fixture(graph=True); graph = runner.decode_model; old = graph._functions
        session = m.Session(runner,type(runner).run,m.Recorder(),[('long',graph)],{})
        advance(runner,req,out)
        events=session.recorder.rows
        self.assertEqual(sum(x['kind']=='callable_family_created' for x in events),1)
        calls=[x for x in events if x['kind']=='compiled_callable']
        self.assertEqual([x['invocation'] for x in calls],[1,2,3,4,5])
        session.restore(); self.assertIs(graph._functions,old)
        self.assertTrue(all(type(fn).__name__ != '_Invocation' for fn in old.values()))

    def test_changed_ahead_key_recomputes_current_instead_of_false_consumption(self):
        runner,mx,req,out=fixture(); session=m.Session(runner,type(runner).run,m.Recorder(),[],{})
        advance(runner,req,out,1); req.output_token_ids[-1]+=4
        advance(runner,req,out,1)
        last=[r for r in session.recorder.rows if r['kind']=='runner_step'][-1]
        self.assertTrue(last['prior_ahead_present']); self.assertFalse(last['ahead_consumed'])
        self.assertEqual(last['actual_decode_calls'],['current','future']);session.restore()

    def test_existing_eval_error_preserved_and_scope_releases_references(self):
        runner,mx,req,out=fixture(); mx.eval=lambda *a: (_ for _ in ()).throw(RuntimeError('eval failed'))
        decoder=runner.decode_model; session=m.Session(runner,type(runner).run,m.Recorder(),[],{})
        ref=weakref.ref(runner)
        with self.assertRaisesRegex(RuntimeError,'eval failed'):
            with session: runner.run(out)
        self.assertTrue(session.closed); self.assertIs(runner.decode_model,decoder)
        errors=[r for r in session.recorder.rows if r.get('error')]
        self.assertEqual([r['kind'] for r in errors],['existing_eval','runner_step'])
        del runner; gc.collect(); self.assertIsNone(ref())

    def test_overflow_retains_program_semantics_and_only_writes_incomplete_once(self):
        runner,mx,req,out=fixture(); recorder=m.Recorder(max_events=2)
        session=m.Session(runner,type(runner).run,recorder,[],{}); advance(runner,req,out)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'profile.json'
            with self.assertRaisesRegex(ValueError,'restore'): recorder.write_once(path,sources={},drained=True)
            self.assertFalse(path.exists()); session.restore()
            result=recorder.write_once(path,sources={},drained=True)
            self.assertEqual(result['status'],'INCOMPLETE'); self.assertFalse(result['performance_qualified'])
            self.assertEqual(len(json.loads(path.read_text())['events']),2)
            with self.assertRaisesRegex(ValueError,'already'):recorder.write_once(path,sources={},drained=True)

    def test_foreign_observer_is_not_overwritten(self):
        runner,_,_,_=fixture();session=m.Session(runner,type(runner).run,m.Recorder(),[],{})
        own=runner.run; runner.run=object()
        with self.assertRaisesRegex(ValueError,'foreign'): session.restore()
        runner.run=own;session.restore()

    def test_queue_and_delivery_wrappers_preserve_same_objects_and_existing_io(self):
        recorder=m.Recorder(); q=queue.Queue(); tap=d._Queue(q,recorder,'rid'); item={'token':42,'done':False}
        tap.put(item); self.assertIs(tap.get(timeout=.1),item)
        class Tokenizer:
            def decode(self, ids, **kwargs): self.got=(ids,kwargs);return 'piece'
        original=Tokenizer(); wrapped=d._Tokenizer(original,recorder); ids=[42]
        self.assertEqual(wrapped.decode(ids,skip_special_tokens=False),'piece'); self.assertIs(original.got[0],ids)
        stream=io.BytesIO(); writer=d._Writer(stream,recorder)
        self.assertEqual(writer.write(b'exact SSE'),9);writer.flush();self.assertEqual(stream.getvalue(),b'exact SSE')
        rows=recorder.rows
        self.assertEqual([r['kind'] for r in rows],['delivery_queue_put','delivery_queue_get','text_decode','http_write','http_flush'])
        self.assertTrue(all(r.get('rid')=='rid' for r in rows))

    def test_actual_worker_submit_gets_tap_before_queue_admission_and_restores(self):
        module=donor(); worker=module.BatchWorker(); engine=NS(step=lambda:[('rid',1,False)])
        original=engine.step; recorder=m.Recorder()
        scope=d.install(engine,worker,module,object(),recorder,enabled=True)
        worker.initialize(engine,lambda **kw:kw,{})
        job=worker.submit([1,2],5)
        self.assertIs(worker.pending.get_nowait(),job);self.assertIsInstance(job.events,d._Queue)
        self.assertEqual(engine.step(),[('rid',1,False)]);scope.restore()
        self.assertIs(engine.step,original);self.assertNotIn('submit',vars(worker))
        self.assertIs(module.BatchWorker.submit.__globals__['Job'],module.Job)

    def test_private_unregistered_route_class_has_real_source_binding(self):
        path=ROOT/'bench/m5_ultra_256gb/route_decode_graph/candidate.py'
        spec=importlib.util.spec_from_file_location('_profile_unregistered_route',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        self.assertNotIn(module.__name__,sys.modules)
        obj=module.RoutePreservingDecode(object(),enabled=False); pins={}
        self.assertIs(m._verify_class(obj,'candidate.py',m.CANDIDATE_SHA,pins),vars(module))
        self.assertEqual(pins,{str(path):m.CANDIDATE_SHA})

    def test_deferred_engine_attachment_and_handler_error_do_not_leak_activity(self):
        module=donor(); worker=module.BatchWorker(); recorder=m.Recorder()
        scope=d.install(None,worker,module,object(),recorder,enabled=True)
        class Handler:
            def do_POST(self): raise RuntimeError('wire failure')
        with patch.object(module,'handler_class',return_value=Handler): cls=scope.handler_class()
        with self.assertRaisesRegex(RuntimeError,'wire failure'): cls().do_POST()
        self.assertEqual(scope.active_handlers,0)
        engine=NS(step=lambda:[]);scope.attach_engine(engine)
        worker.initialize(engine,lambda **kw:kw,{})
        with self.assertRaisesRegex(ValueError,'precede'):scope.attach_engine(engine)
        scope.restore();self.assertNotIn('submit',vars(worker))


if __name__=='__main__': unittest.main()
