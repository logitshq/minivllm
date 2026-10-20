"""CPU metadata and fake-lifecycle contracts; no backend, HTTP, process or GPU."""
import asyncio
import copy
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from bench.m5_ultra_256gb.c8_batched_http_v1 import server as s, trial as t
from bench.m5_ultra_256gb.c8_schedule_diagnostic_v1.test_cpu import Fake as OldFake


def argv(*extra):
    return ['--model','/cpu/model','--port','18181','--package-root','/cpu/package',
        '--source-manifest','/cpu/manifest.json','--source-manifest-sha256','unused',
        '--max-len','2048','--max-seqs','8','--max-batched-tokens','2048','--cache-limit-gib','2',
        '--compile-decode','on','--fuse-decode','on','--gates-compile','on','--norm-reuse','off',
        '--tcp-nodelay','on','--io-timeout-seconds','300',*extra]


def fake_spec():
    # Explicit synthetic spec: validate is replaced only inside lifecycle tests.
    return dict(schema=t.SCHEMA,label='CPU-two-waves', command=['python','-I','-B','/cpu/server.py','--run'],
        environment={},transport=dict(port=18181),identity={'CPU':True},concurrency=8,
        jobs=[dict(request_id=f'job{i}') for i in range(16)],wave_names=['warm','screen'],
        deadlines=dict(ready=.01,request=.01,group=.01,close=.05,term=.05,kill=.05,overall=1,resource_interval=.001))


class Fake(OldFake):
    def spawn(self, command, environment, stdout, stderr):
        child = super().spawn(command,environment,stdout,stderr)
        Path(command[-1]).write_text(json.dumps(dict(event='route-startup',schema=s.ROUTE_SCHEMA,
            pid=child.pid,phase='HTTP-C8',identity=self.spec['identity']))+'\n')
        return child


class Source(unittest.TestCase):
    def test_default_off_and_formal_qualification_parser_contract(self):
        with redirect_stdout(io.StringIO()): self.assertEqual(s.main([]),0)
        with self.assertRaisesRegex(ValueError,'explicit --run'): s.main(['--model','unused'])
        formal=s.arguments(argv('--phase','formal','--gdn-prework','on','--route-decode-graph','on'))
        self.assertIsNone(formal.qualification_trace);self.assertIsNone(formal.route_events)
        qual=['--phase','qualification','--gdn-prework','on','--route-decode-graph','on',
              '--qualification-trace','/cpu/trace','--route-events','/cpu/routes']
        self.assertEqual(s.arguments(argv(*qual)).phase,'qualification')
        for extra in (qual[:-2],[*qual,'--gdn-prework','off'],['--phase','formal','--qualification-trace','/cpu/trace']):
            with redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):s.arguments(argv(*extra))
        with self.assertRaisesRegex(ValueError,'fixed C8'):s.arguments(argv('--max-seqs','1'))
    def test_identity_records_same_candidate_and_actual_phase_observers(self):
        pins=s.source_pins()
        for phase in ('formal','qualification'):
            args=NS(phase=phase,gdn_prework='on',route_decode_graph='on')
            identity=s.identity_fields(args,pins,installed=True,route_observer=phase=='qualification')
            c=identity['combined_candidates']
            self.assertEqual(identity['benchmark_phase'],phase)
            self.assertEqual(c['integration_path'],str(s.BATCHED));self.assertEqual(c['integration_sha256'],s.BATCHED_SHA)
            self.assertEqual(c['gdn_layer_installs'],48);self.assertEqual(c['route_observer'],phase=='qualification')
            self.assertEqual(c['trace_policy_source_sha256'],s.sha(s.HERE/'server.py'))
            self.assertFalse(identity['numerical_qualified'] or identity['performance_qualified'])
    def test_main_and_trace_code_reuse_has_private_global_substitutions(self):
        original=s.original();call=s.execution_main(original)
        self.assertIs(call.__code__,original.main.__code__)
        self.assertEqual(call.__globals__['INTEGRATION'],s.BATCHED)
        self.assertNotEqual(original.INTEGRATION,s.BATCHED)
        combined=s.load(original.COMBINED,original.FIXED[original.COMBINED],'_http_cpu_original_trace')
        cls=s.trace_type(combined)
        self.assertIs(cls.__call__.__code__,combined.TraceGraph.__call__.__code__)
        self.assertIs(cls.__call__.__globals__['delta'],s.batched_delta)
        self.assertIsNot(combined.TraceGraph.__call__.__globals__['delta'],s.batched_delta)
    def test_actual_trace_code_batched_candidate_then_zero_delta_reuse(self):
        old=s.original();module=s.load(old.COMBINED,old.FIXED[old.COMBINED],'_http_cpu_trace')
        handles=[NS(calls=dict(candidate=0,fallback=0)) for _ in range(48)]
        sentinel=object()
        class Graph:
            def __init__(self):self._functions=NS(invocations=0);self.seen=set()
            def __call__(self,tokens,ctx):
                self._functions.invocations+=1;key=(tokens.shape[0],ctx.max_len)
                if key not in self.seen:
                    self.seen.add(key)
                    for h in handles:h.calls['candidate']+=1
                return sentinel
        graph=Graph();rows=[];stats={'compiled-tracing':0,'compiled-reuse':0,'eager-fallback':0,'traced_batches':[]}
        trace=s.trace_type(module)(graph,handles,rows.append,stats,0)
        trace.witness=NS(phase='HTTP-C8',step=1,direct_label=None)
        for B,used in ((2,1023),(2,1023),(4,1023),(6,1023),(8,1023),(8,1536),(8,1536)):
            self.assertIs(trace(NS(shape=(B,)),NS(max_len=used)),sentinel)
        self.assertEqual([r['per_layer_candidate_delta'] for r in rows],[1,0,1,1,1,1,0])
        self.assertTrue(all(r['per_layer_fallback_delta']==0 for r in rows))
        self.assertEqual(stats['compiled-tracing'],5);self.assertEqual(stats['compiled-reuse'],2)
        self.assertTrue(all(len(r['before'])==len(r['after'])==48 for r in rows))
    def test_trace_rejects_mixed_fallback_negative_and_wrong_batch(self):
        before=[dict(candidate=0,fallback=0) for _ in range(48)]
        good=[dict(candidate=1,fallback=0) for _ in range(48)]
        self.assertEqual(s.batched_delta(before,good,8),(1,0));self.assertEqual(s.batched_delta(good,good,8),(0,0))
        bad=copy.deepcopy(good);bad[0]['candidate']=0
        for after,B in ((bad,8),([dict(candidate=0,fallback=1) for _ in range(48)],8),(before,9),(before,True)):
            with self.assertRaises(ValueError):s.batched_delta(before,after,B)
        with self.assertRaises(ValueError):s.batched_delta(good,before,2)
    def test_failed_incomplete_or_nonfull_predecessor_rejected(self):
        good=dict(schema='c8-batched-gdn-graph-full-result-v1',status='PASS',source_input_pins_rechecked=True,
            full_history_qualified=True,http_qualified=False,performance_qualified=False,steps=1030,total_requests=16,
            base_result=dict(status='PASS',returncode=0),route_comparison=dict(calls=1028,actual_compiled_invocations=1028,fallbacks=0))
        t.predecessor_ok(good)
        for field,value in (('status','INCOMPLETE'),('status','FAIL'),('steps',49),('total_requests',0),('performance_qualified',True)):
            bad=copy.deepcopy(good);bad[field]=value
            with self.assertRaises(ValueError):t.predecessor_ok(bad)
    def test_capture_code_and_route_schema_are_only_private_lifecycle_changes(self):
        old=t.old();capture=old.module(old.CAPTURE);call=t.capture_function(capture,old.System)
        self.assertIs(call.__code__,capture.capture.__code__)
        self.assertIs(call.__globals__['System'],old.System);self.assertIs(call.__globals__['validate'],t.validate)
        self.assertIs(call.__globals__['route_startup'].__code__,capture.route_startup.__code__)
        self.assertEqual(call.__globals__['route_startup'].__globals__['ROUTE_SCHEMA'],s.ROUTE_SCHEMA)
        self.assertEqual(capture.ROUTE_SCHEMA,'E024-route-http-events-v1')


class Lifecycle(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.out=Path(self.temp.name)/'child';self.spec=fake_spec();self.checks=[]
    async def checkpoint(self,label,pid):self.checks.append((label,pid))
    async def test_two_actual_group_slices_retain_raw_and_reap(self):
        fake=Fake(self.spec)
        with patch.object(t,'validate',lambda spec:None):result=await t.run(self.spec,self.out,checkpoint=self.checkpoint,backend=fake)
        self.assertEqual(result['status'],'PASS',result['errors'])
        self.assertEqual(result['schema'],'c8-batched-http-result-v1')
        self.assertEqual(fake.calls,[self.spec['jobs'][:8],self.spec['jobs'][8:]])
        self.assertEqual(fake.child.actions,['TERM']);self.assertTrue(result['server_reaped'] and result['cleanup_complete'])
        self.assertEqual(self.checks[-1],('after-reap',None));self.assertFalse(result['performance_qualified'])
        self.assertEqual(t.read(self.out/'group.json')['requests'],self.spec['jobs'])
    async def test_first_wave_failure_preserved_and_no_second_wave(self):
        fake=Fake(self.spec,fail_first=True)
        with patch.object(t,'validate',lambda spec:None):result=await t.run(self.spec,self.out,checkpoint=self.checkpoint,backend=fake)
        self.assertEqual(result['status'],'FAIL');self.assertEqual(len(fake.calls),1)
        self.assertTrue(result['server_reaped']);self.assertTrue((self.out/'client/warm/group.json').is_file())
        self.assertFalse((self.out/'client/screen').exists())
    async def test_cancellation_preserves_partial_wave_and_reaps(self):
        fake=Fake(self.spec,block_second=True)
        with patch.object(t,'validate',lambda spec:None):
            task=asyncio.create_task(t.run(self.spec,self.out,checkpoint=self.checkpoint,backend=fake))
            await fake.second_started.wait();task.cancel();result=await task
        self.assertEqual(result['status'],'FAIL');self.assertTrue(result['server_reaped'])
        self.assertTrue((self.out/'client/screen/group.json').is_file())
        self.assertEqual(len(t.read(self.out/'group.json')['requests']),16)


if __name__=='__main__':unittest.main()
