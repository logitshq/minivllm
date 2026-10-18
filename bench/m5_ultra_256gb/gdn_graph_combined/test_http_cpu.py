"""CPU fakes only; no model/token oracle, socket, process or device operation."""
import ast
import asyncio
import copy
from contextlib import redirect_stderr
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from bench.m5_ultra_256gb.gdn_graph_combined import http_server as s, http_capture as c, http_qualification as q
from bench.m5_ultra_256gb.gdn_graph_combined import qualify as combined
from bench.m5_ultra_256gb.gdn_graph_combined import test_qualify_cpu as f
from bench.m5_ultra_256gb.route_decode_graph import http_server_v2 as entry
from bench.m5_ultra_256gb.route_decode_graph.test_http_server_cpu import argv, context, Engine
from bench.m5_ultra_256gb.pareto_realistic.test_http_capture_v2_cpu import Backend as NativeBackend


class Backend(NativeBackend):
    def __init__(self, spec, *, wrong_start=False, **kwargs):
        super().__init__(spec, **kwargs);self.wrong_start=wrong_start
    def spawn(self, command, environment, stdout, stderr):
        self.started=True
        assert command[:-4]==self.spec['command'] and environment==self.spec['environment']
        assert command[-4]=='--qualification-trace' and command[-2]=='--route-events'
        Path(command[-3]).write_text(json.dumps(dict(event='startup',identity=self.spec['identity']))+'\n')
        Path(command[-1]).write_text(json.dumps(dict(event='route-startup',schema=c.ROUTE_SCHEMA,
            pid=self.child.pid+int(self.wrong_start),phase=f"HTTP-C{self.spec['concurrency']}",identity=self.spec['identity']))+'\n')
        stdout.write(b'FAKE server only\n');return self.child


def arguments():
    path=c.donor.donor.HERE/'prereg-qualification-v1/plan.json';numeric=json.loads(path.read_text())
    kw=dict(python=numeric['execution']['python'],environment=numeric['execution']['environment'],
        transport=dict(port=18181,tcp_nodelay=True,io_timeout_seconds=300.0),
        deadlines=dict(ready=1,request=1,group=1,close=.05,term=.05,kill=.05,overall=3,resource_interval=.01))
    return path,kw


def fake_scope(engine):
    engine.runner.model=object()
    return f.Scope(engine)


def observer(engine,scope):
    native=s.load_fixed(s.DONOR,'_combined_test_native')
    # load_fixed rejects accidental reuse of a module identity, as production does.
    import sys
    sys.modules.pop('_combined_test_native',None)
    trace,route=io.StringIO(),io.StringIO()
    ob=s.observer_type(entry,combined)(engine,scope,native,trace,route,'HTTP-C1',f.f.q,f.f.old_module())
    return ob,native,trace,route


class Tests(unittest.TestCase):
    def test_independent_formal_switches_and_observers_off(self):
        for gdn in ('off','on'):
            for graph in ('off','on'):
                args=s.arguments(argv('--gdn-prework',gdn,'--route-decode-graph',graph))
                self.assertEqual(args.max_len,2048);self.assertIsNone(args.qualification_trace);self.assertIsNone(args.route_events)
                fields=s.identity_fields(args,s.source_pins(),installed=(gdn=='on' or graph=='on'),route_observer=False)
                self.assertEqual(fields['combined_candidates']['gdn_layer_installs'],48 if gdn=='on' else 0)
                self.assertFalse(fields['combined_candidates']['actual_compiled_invocations_observed'])
        self.assertEqual(s.arguments(argv()).gdn_prework,'off')

    def test_qualification_requires_both_and_separate_observers(self):
        on=['--phase','qualification','--gdn-prework','on','--route-decode-graph','on',
            '--qualification-trace','/cpu/trace','--route-events','/cpu/routes']
        self.assertEqual(s.arguments(argv(*on)).phase,'qualification')
        for extra in (on[:-2],[*on,'--gdn-prework','off'],[*on,'--route-events','/cpu/trace'],
                      [*on,'--norm-reuse','on'],[*on,'--max-len','1536'],['--route-events','/cpu/routes']):
            with redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):s.arguments(argv(*extra))

    def test_real_native_worker_uuid_trace_and_compiled_reuse_no_repeat_eval(self):
        engine=Engine();scope=fake_scope(engine);ob,native,trace,route=observer(engine,scope)
        worker=native.BatchWorker(trace=ob.trace);worker.initialize(engine,lambda **kw:SimpleNamespace(**kw),{'cpu':True})
        ob.startup({'cpu':True});ids=[]
        for _ in range(2):
            job=worker.submit([11,12],1);ids.append(job.rid);self.assertTrue(worker.tick())
            self.assertEqual(job.events.get_nowait(),{'token':42,'done':True})
        self.assertNotEqual(*ids)
        rows=[json.loads(x) for x in route.getvalue().splitlines()]
        self.assertEqual([x['event'] for x in rows],['route-startup','combined-graph-call','route-call','combined-graph-call','route-call'])
        self.assertEqual([rows[i]['kind'] for i in (1,3)],['compiled-tracing','compiled-reuse'])
        self.assertEqual([rows[i]['per_layer_candidate_delta'] for i in (1,3)],[1,0])
        self.assertEqual([x['event'] for x in map(json.loads,trace.getvalue().splitlines())],
            ['startup',*(['before_step','actual_scheduler_output','actual_model_runner_batch','after_step']*2)])
        ob.restore();self.assertIs(scope.graph_handle.decoder.long,scope.graph);scope.restore();self.assertTrue(scope.closed)

    def test_observer_constructor_failure_rolls_back_proxy(self):
        engine=Engine();scope=fake_scope(engine);saved=scope.graph._functions
        broken=SimpleNamespace(observe_runner=lambda *a:(_ for _ in ()).throw(RuntimeError('broken observer')))
        cls=s.observer_type(entry,combined)
        with self.assertRaisesRegex(RuntimeError,'broken observer'):
            cls(engine,scope,broken,io.StringIO(),io.StringIO(),'HTTP-C1',f.f.q,f.f.old_module())
        self.assertIs(scope.graph_handle.decoder.long,scope.graph);self.assertIs(scope.graph._functions,saved)
        self.assertNotIn('run',vars(engine.runner));scope.restore()

    def test_real_prepare_keeps_original_jobs_and_binds_new_sources(self):
        path,kw=arguments()
        for C,count in ((1,8),(8,16)):
            spec=c.prepare(path,c.sha(path),C,**kw);c.validate(spec)
            baseline=c.donor.donor.prepare(path,c.sha(path),C,**kw)
            self.assertEqual(spec['jobs'],baseline['jobs']);self.assertEqual(len(spec['jobs']),count)
            self.assertEqual(spec['command'][3],str(c.SERVER));self.assertEqual(spec['command'][-6:],
                ['--phase','qualification','--gdn-prework','on','--route-decode-graph','on'])
            self.assertEqual(spec['identity']['configuration'],baseline['identity']['configuration'])
            self.assertTrue(spec['identity']['combined_candidates']['route_observer'])
            self.assertEqual(spec['sources'][str(c.DONOR)],c.DONOR_SHA)
            for change in ('body','switch','source'):
                bad=copy.deepcopy(spec)
                if change=='body':bad['jobs'][0]['body']['prompt']+='changed'
                elif change=='switch':bad['command'][-1]='off'
                else:bad['sources'][str(c.SERVER)]='0'*64
                with self.assertRaises(ValueError):c.validate(bad)

    def test_replay_actual_observer_install_order_and_recording_run_not_overwritten(self):
        base=f.f.base_module();integration=f.Integration();events=[];recording={'concurrency':1};fixtures=[]
        sent=[]
        def replay(*args,**kw):sent.append((args[0],kw['base']));return 'recorded-history'
        original=(base.Observer,base.workloads,base.run_case)
        with q.adapted(base,SimpleNamespace(run_recorded_case=replay),integration,f.f.q,f.f.old_module(),
                       recording,fixtures,SimpleNamespace(emit=events.append)) as coverage:
            ob=base.Observer(f.f.engine(),object(),'candidate')
            self.assertIs(ob.route_install.proxy.witness,ob.route_witness)
            self.assertEqual(base.run_case(dict(name='D-C1',concurrency=1),[],[],None,None,None,None,None),'recorded-history')
            self.assertIs(sent[0][0],recording);self.assertIs(sent[0][1],base)
            ob.restore();self.assertTrue(integration.scopes[0].closed)
            ref=base.Observer(f.f.engine(),object(),'reference');ref.restore()
            self.assertEqual(coverage['combined']['layer_restores'],48);self.assertEqual(coverage['restores'],2)
        self.assertEqual((base.Observer,base.workloads,base.run_case),original)

    def test_replay_constructor_failure_restores_observer_then_combined_install(self):
        base=f.f.base_module();inherited=base.Observer;engine=f.f.engine();integration=f.Integration()
        class Broken(inherited):
            def __init__(self,*args):super().__init__(*args);raise RuntimeError('partial construction')
        base.Observer=Broken
        with q.adapted(base,None,integration,f.f.q,f.f.old_module(),{'concurrency':1},[],SimpleNamespace(emit=lambda x:None)):
            with self.assertRaisesRegex(RuntimeError,'partial construction'):base.Observer(engine,object(),'candidate')
            self.assertTrue(integration.scopes[0].closed);self.assertNotIn('run',vars(engine.runner))

    def test_raw_sidecar_exact_order_extra_missing_and_changed_reject(self):
        _,_,witness,_,events=f.proxy_fixture();witness.phase='HTTP-C1';witness.step=1
        for _ in range(2):witness(SimpleNamespace(shape=(1,)),context())
        header=dict(event='route-startup',schema=s.ROUTE_SCHEMA,pid=23456,phase='HTTP-C1',identity={})
        def check(rows,actual=events):
            raw=''.join(json.dumps(x)+'\n' for x in [header,*rows]).encode();output=io.StringIO()
            cursor=q.CapturedRoutes(io.BytesIO(raw),header,output,{'path':'/raw','sha256':q.hashlib.sha256(raw).hexdigest()},'HTTP-C1')
            for row in actual:cursor.emit(row)
            return cursor.finish(q.hashlib.sha256(raw).hexdigest())
        result=check(events);self.assertEqual(result['calls'],2);self.assertEqual(result['combined_kinds']['compiled-reuse'],1)
        for rows in (events[:-1],events+[events[-1]],events[1:]+events[:1]):
            with self.assertRaises(ValueError):check(rows)
        changed=copy.deepcopy(events);changed[0]['after'][0]['candidate']+=1
        with self.assertRaises(ValueError):check(changed)

    def test_default_replay_description_does_not_open_evidence(self):
        with patch.object(q,'validate',side_effect=AssertionError('validation')),patch('sys.stdout',new=io.StringIO()):
            self.assertEqual(q.main([]),0)
        self.assertIs(q.prepare,q.donor.prepare)

    def test_immutable_serving_prerequisite_rejects_unqualified_or_mixed_terminal(self):
        with tempfile.TemporaryDirectory() as temporary:
            child=Path(temporary).resolve();coverage=dict(route=dict(parameter_direct_calls=42,parameter_checks_complete=True),
                combined=dict(layer_installs=96,layer_restores=96))
            sources={str(q.HERE/'qualify.py'):q.FIXED[q.HERE/'qualify.py']}
            plan=dict(schema='GDN-graph-combined-serving-plan-v1',qualifier_sha256=sources[str(q.HERE/'qualify.py')],
                qualification={'sha256':q.BASE_PLAN_SHA},deadline_s=1800,sources=sources)
            q.write(child/'plan.json',plan)
            complete=dict(schema='GDN-graph-combined-serving-result-v1',status='PASS',same_schedule_checks_passed=True,
                source_pins_rechecked=True,plan_sha256=q.sha(child/'plan.json'),total_requests=37,parameter_direct_calls=42,
                sources=sources,coverage=coverage,http_qualified=False,performance_qualified=False)
            q.write(child/'complete.json',complete)
            audit=dict(schema='GDN-graph-combined-independent-v1',status='PASS',qualified=True,same_schedule_qualified=True,
                child_directory=str(child),audit_source_sha256=q.FIXED[q.HERE/'audit.py'],plan_sha256=q.sha(child/'plan.json'),
                base_plan_sha256=q.BASE_PLAN_SHA,terminal_sha256={name:q.sha(child/name) for name in ('plan.json','complete.json')},
                source_pins=sources,coverage=coverage,ledger=dict(cases=[{'completed':[{}]*n} for n in (8,16,13)]),
                http_qualified=False,performance_qualified=False)
            def refs(value):
                q.write(child/'audit.json',value)
                return {k:dict(path=str(child/name),sha256=q.sha(child/name)) for k,name in
                        (('plan','plan.json'),('audit','audit.json'),('complete','complete.json'))}
            q.serving_gate(refs(audit))
            for key,value in (('qualified',False),('status','FAIL'),('audit_source_sha256','0'*64),
                              ('terminal_sha256',{'plan.json':'0'*64,'complete.json':q.sha(child/'complete.json')})):
                bad=copy.deepcopy(audit);bad[key]=value
                with self.assertRaises(ValueError):q.serving_gate(refs(bad))


class CaptureTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.out=Path(self.temp.name).resolve()/'capture';path,kw=arguments();self.spec=c.prepare(path,c.sha(path),1,**kw)
    async def check(self,*args):pass

    async def test_real_frozen_cleanup_new_metadata_and_no_global_mutation(self):
        before=dict(c.donor.capture.__globals__);backend=Backend(self.spec)
        result=await c.capture(self.spec,self.out,checkpoint=self.check,backend=backend)
        self.assertEqual(result['status'],'PASS');self.assertTrue(result['route_events_retained'] and result['server_reaped'])
        self.assertFalse(result['numerical_qualified'] or result['performance_qualified'])
        self.assertEqual(result['schema'],'GDN-graph-combined-http-capture-result-v1')
        self.assertEqual(backend.child.actions,['TERM']);self.assertEqual(before,c.donor.capture.__globals__)
        plan={'concurrency':1,'qualification':self.spec['base_plan'],
              'capture':{k:dict(path=str(self.out/n),sha256=q.sha(self.out/n)) for k,n in q.CAPTURE_NAMES.items()},
              'inputs':{k:dict(path=str(self.out/n),sha256=q.sha(self.out/n)) for k,n in q.INPUT_NAMES.items()}}
        q.capture_gate(plan)
        release=json.loads((self.out/'release.json').read_text());release['waited_reaped']=False;q.write(self.out/'release.json',release)
        plan['capture']['release']['sha256']=q.sha(self.out/'release.json')
        result['artifacts']['release.json']={'sha256':q.sha(self.out/'release.json'),'bytes':(self.out/'release.json').stat().st_size}
        q.write(self.out/'complete.json',result);plan['capture']['complete']['sha256']=q.sha(self.out/'complete.json')
        with self.assertRaises(ValueError):q.capture_gate(plan)

    async def test_wrong_sidecar_pid_or_live_child_never_passes(self):
        for label,kw in (('startup',{'wrong_start':True}),('alive',{'bad_reap':True})):
            backend=Backend(self.spec,**kw)
            result=await c.capture(self.spec,self.out.with_name(label),checkpoint=self.check,backend=backend)
            self.assertEqual(result['status'],'FAIL');self.assertTrue(backend.waited)
            self.assertFalse(result['numerical_qualified'])

    async def test_resource_failure_retains_partial_then_strict_reap(self):
        backend=Backend(self.spec,block=True,ignore_term=True)
        async def guard(label,pid):
            if label=='capturing':raise RuntimeError('resource failure')
        result=await c.capture(self.spec,self.out,checkpoint=guard,backend=backend)
        self.assertEqual(result['status'],'FAIL');self.assertTrue(result['server_reaped'])
        self.assertEqual(backend.child.actions,['TERM','KILL'])
        self.assertEqual(json.loads((self.out/'group.json').read_text())['group_error']['type'],'CancelledError')

    async def test_repeated_cancel_during_client_close_still_waits_reaps(self):
        backend=Backend(self.spec,block=True,ignore_term=True)
        task=asyncio.create_task(c.capture(self.spec,self.out,checkpoint=self.check,backend=backend))
        await backend.group_started.wait();task.cancel();await asyncio.sleep(0);task.cancel()
        result=await task
        self.assertEqual(result['status'],'FAIL');self.assertTrue(result['server_reaped'])
        self.assertEqual(backend.child.actions,['TERM','KILL']);self.assertTrue(backend.waited)


if __name__=='__main__':unittest.main()
