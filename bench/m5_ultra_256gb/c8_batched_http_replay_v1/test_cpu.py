"""CPU contract tests for source binding, unchanged full replay and raw input join."""
import copy
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from bench.m5_ultra_256gb.c8_batched_http_replay_v1 import qualify as q
from bench.m5_ultra_256gb.c8_batched_prefix_v1.test_cpu import Combined, BaseObserver, Witness, engine
from bench.m5_ultra_256gb.pareto_realistic import test_http_join_cpu as raw


def counts(n):return [dict(candidate=n,fallback=0) for _ in range(48)]
def pair(step,before,after):
    c=dict(event='combined-graph-call',status='SUBMITTED',installation=0,phase='HTTP-C8',step=step,
        direct_label=None,B=8,used=1024,before=counts(before),after=counts(after),actual_invocations=1,
        kind='compiled-tracing' if after>before else 'compiled-reuse',bound=1536,
        per_layer_candidate_delta=after-before,per_layer_fallback_delta=0)
    r=dict(event='route-call',status='SUBMITTED',phase='HTTP-C8',step=step,direct_label=None,
           B=8,used=1024,actual_invocations=1)
    return c,r

def wire_rows(rows):return b''.join((json.dumps(row)+'\n').encode() for row in rows)

def recording():
    return dict(schema='E023-recorded-http-v1',concurrency=8,steps=[dict(step=i) for i in range(1,8)],
        jobs={f'r{i}':dict(cap=512,raw_ids=[i,99],committed_ids=[i]) for i in range(16)},histogram={'0':2,'8':5})

def coverage():
    return dict(installs=1,restores=2,combined=dict(installs=1,restores=1,layer_installs=48,layer_restores=48,**{'compiled-tracing':2}))

def inner(data):
    return dict(status='PASS',source_model_pins_rechecked=True,total_requests=16,cases=[dict(status='PASS',
        name='HTTP-C8',steps=len(data['steps']),completed=[dict(rid=rid,raw_ids=j['raw_ids'],committed_ids=j['committed_ids'])
        for rid,j in data['jobs'].items()],raw_output_count=32,committed_output_count=16,decode_batch_histogram=data['histogram'])])


def tiny_capture(child,trial,http):
    """Six actual synthetic steps, two drained8 waves, early EOS; original join."""
    template,_,_,fixtures,identity,terminals=raw.fixture(8)
    for f in fixtures.values():f['max_output_tokens']=512
    groups=[];jobs=[];events=[dict(event='startup',identity=identity)];previous_finished=[]
    for wave in range(2):
        rows=[];rids=[];first=[];second=[];third=[];even=[];odd=[]
        for i in range(8):
            n=8*wave+i;rid=f'{n+1:032x}';rids.append(rid);slot=i%2
            row=copy.deepcopy(template['requests'][slot]);row['request_id']=f'client-{n}';row['body']['max_tokens']=512
            terminal=copy.deepcopy(terminals[slot]);terminal['id']='cmpl-'+rid;terminal['benchmark_request_id']=rid
            raw.replace_wire(row,terminal);rows.append(row)
            jobs.append({k:copy.deepcopy(row[k]) for k in ('request_id','fixture_id','path','body')})
            first.append([rid,terminal['benchmark_output_ids'][0],False])
            second.append([rid,99 if slot==0 else 21,slot==0])
            if slot==0:even.append(rid)
            else:odd.append(rid);third.append([rid,99,True])
        group=copy.deepcopy(template);group['requests']=rows
        group['cleanup']['adapter']='native-close-cancellation-v1';group['summary']['transport_complete']=True
        groups.append(group)
        def step(arrivals,running,waiting,decode,prefill,finished,results):
            retired=[rid for rid,_,done in results if done]
            events.extend([dict(event='before_step',newly_submitted_to_scheduler=arrivals,scheduler_running=running,scheduler_waiting=waiting),
                dict(event='actual_scheduler_output',decode=decode,prefill=[dict(rid=rid,tokens=2) for rid in prefill],finished_ids=finished),
                dict(event='actual_model_runner_batch',decode_rows=decode),
                dict(event='after_step',results=results,retired=retired,surviving_decode_rows=[r for r in decode if r not in retired])])
        step(rids,[],rids,[],rids,previous_finished,first)
        step([],rids,[],rids,[],[],second)
        step([],odd,[],odd,[],even,third);previous_finished=odd
    refs=[]
    for name,group in zip(('warm','screen'),groups):
        path=child/'client'/name/'group.json';path.parent.mkdir(parents=True);q.write(path,group)
        refs.append(dict(phase=name,**q.ref(path)))
    group=trial.old().join_view(groups,refs);q.write(child/'group.json',group)
    (child/'trace.jsonl').write_bytes(wire_rows(events))
    (child/'model').mkdir();tokenizer=child/'model/tokenizer.json';tokenizer.write_text('CPU fake tokenizer')
    fixture_path=child/'fixtures.json';q.write(fixture_path,dict(fixtures=list(fixtures.values())))
    model=child/'model-manifest.json';q.write(model,dict(files={'tokenizer.json':q.sha(tokenizer)}))
    numeric=dict(fixtures=q.ref(fixture_path),model_manifest=q.ref(model),model=str(child/'model'))
    spec=dict(fixtures=list(fixtures.values()),tokenizer=q.ref(tokenizer),jobs=jobs,identity=identity,sources={'CPU':'synthetic'})
    q.write(child/'spec.json',spec)
    joined=raw.h.join(group,raw.lines(events),jobs,fixtures,identity,raw.decode)
    (child/'joined-input').mkdir();q.write(child/'joined-input/recording.json',joined)
    receipt=dict(schema='c8-batched-http-joined-input-v2',inherited_join_schema='E047-joined-input-v1',
        replay_numerical_checked=False,performance_qualified=False,spec_sha256=q.sha(child/'spec.json'),source_groups=refs,
        trace_sha256=q.sha(child/'trace.jsonl'),recording_sha256=q.sha(child/'joined-input/recording.json'),sources=spec['sources'])
    q.write(child/'joined-input/receipt.json',receipt)
    class Tokenizer:
        @staticmethod
        def from_file(path):return Tokenizer()
        def encode(self,text,add_special_tokens=False):return NS(ids=next(f['prompt_ids'] for f in fixtures.values() if f['prompt_text']==text))
        def decode(self,ids,skip_special_tokens=False):return raw.decode(ids)
    return spec,numeric,joined,NS(Tokenizer=Tokenizer)


class Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.trial,cls.server,cls.http=q.helpers()
    def test_complete_source_closure_has_actual_batched_server_and_replay(self):
        pins=q.source_pins(self.trial,self.http)
        self.assertEqual(pins[str(q.SERVER)],q.SERVER_SHA)
        self.assertEqual(pins[str(q.HTTP)],q.HTTP_SHA)
        self.assertEqual(pins[str(self.server.BATCHED)],self.server.BATCHED_SHA)
        self.assertEqual(pins[str(q.HERE/"qualify.py")],q.sha(q.HERE/"qualify.py"))
    def test_default_off_and_incomplete_action_are_inert(self):
        with patch.object(q,'helpers',side_effect=AssertionError('must remain inert')),redirect_stdout(io.StringIO()):
            self.assertEqual(q.main([]),0)
            with self.assertRaises(ValueError):q.main(['--out','/unused'])
    def test_closed_capture_flags_and_release_are_required(self):
        top=dict(schema='c8-batched-http-result-v2',status='PASS',capture_complete=True,server_reaped=True,cleanup_complete=True,
            cleanup_task_resolved=True,client_task_resolved=True,small_pins_rechecked=True,route_events_retained=True,errors=[],
            numerical_qualified=False,full_state_qualified=False,performance_qualified=False,replay_input_checked=False,ended_at=4)
        child=dict(pid=123456789,time=1);release=dict(pid=123456789,returncode=0,waited_reaped=True,strict_absence=True,port_absent=True,time=3)
        q.terminal_ok(top,child,release)
        for key,value in [('schema','c8-batched-http-result-v1'),('status','FAIL'),('status','INCOMPLETE'),('server_reaped',False),('performance_qualified',True),('errors',['failed'])]:
            bad=copy.deepcopy(top);bad[key]=value
            with self.assertRaises(ValueError):q.terminal_ok(bad,child,release)
        for key,value in [('pid',999),('waited_reaped',False),('time',5)]:
            bad=dict(release);bad[key]=value
            with self.assertRaises(ValueError):q.terminal_ok(top,child,bad)
    def test_dynamic_steps_eos_ids_counts_and_complete_release(self):
        data=recording();expected=q.recording_ok(data)
        self.assertEqual(expected,dict(expected_steps=7,expected_requests=16,expected_raw_outputs=32,expected_committed_outputs=16))
        actual=inner(data);self.assertEqual(q.full_result_ok(actual,data,coverage())['steps'],7)
        for mutate in (lambda d:d['cases'][0].update(steps=6),lambda d:d['cases'][0]['completed'].pop(),
                       lambda d:d['cases'][0].update(raw_output_count=8192),lambda d:d['cases'][0]['completed'][0]['raw_ids'].append(0)):
            bad=copy.deepcopy(actual);mutate(bad)
            with self.assertRaises(ValueError):q.full_result_ok(bad,data,coverage())
        bad=coverage();bad['combined']['layer_restores']=47
        with self.assertRaises(ValueError):q.full_result_ok(actual,data,bad)
    def test_same_private_handle_constructor_trace_and_original_restore(self):
        adapted=q.adapted_function(self.http,self.server)
        self.assertIs(adapted.__wrapped__.__code__,self.http.adapted.__wrapped__.__code__)
        private=adapted.__wrapped__.__globals__['load'](self.http.HERE/'qualify.py','_cpu_private_handle')
        original=self.http.load(self.http.HERE/'qualify.py','_cpu_original_handle')
        self.assertIs(private.Handle.restore,private.Handle.__mro__[1].restore)
        self.assertIs(private.Handle.__init__.__code__,private.Handle.__mro__[1].__init__.__code__)
        trace=private.Handle.__init__.__globals__['TraceGraph']
        self.assertIs(trace.__call__.__globals__['delta'],self.server.batched_delta)
        self.assertIsNot(original.Handle.__init__.__globals__['TraceGraph'].__call__.__globals__['delta'],self.server.batched_delta)
        self.assertIs(self.http.adapted.__wrapped__.__globals__['load'],self.http.load)
    def test_actual_engine_install_before_observer_and_complete_original_restore(self):
        base=NS(Observer=BaseObserver,workloads=object(),run_case=object());saved=base.Observer,base.workloads,base.run_case
        installation=Combined();candidate=engine();reference=engine()
        with q.adapted_function(self.http,self.server)(base,None,installation,NS(RouteWitness=Witness),None,
                {'concurrency':8},[],NS(emit=lambda row:None)) as covered:
            ref=base.Observer(reference,None,'reference');obs=base.Observer(candidate,None,'candidate')
            self.assertIs(installation.engines[0],candidate);self.assertEqual(len(installation.scopes[0].gdn_handles),48)
            handle=obs.route_install;self.assertIs(handle.proxy.witness,obs.route_witness)
            ref.restore();obs.restore();self.assertTrue(handle.closed)
            self.assertIsNone(handle.scope);self.assertEqual(handle.proxy.handles,[])
            self.assertEqual(covered['restores'],2);self.assertEqual(covered['combined']['layer_restores'],48)
        self.assertEqual((base.Observer,base.workloads,base.run_case),saved)
    def test_constructor_failure_keeps_error_and_cleans_original_scope(self):
        base=NS(Observer=BaseObserver,workloads=object(),run_case=object());installation=Combined()
        with q.adapted_function(self.http,self.server)(base,None,installation,NS(RouteWitness=Witness),None,
                {'concurrency':8},[],NS(emit=lambda row:None)),patch.object(BaseObserver,'fail',True):
            with self.assertRaisesRegex(RuntimeError,'base constructor failure'):base.Observer(engine(),None,'candidate')
        self.assertTrue(installation.scopes[0].closed)
    def test_original_full_run_case_receives_unmodified_recording_and_restores_on_error(self):
        base=NS(Observer=BaseObserver,workloads='old workloads',run_case='old case');data=recording();before=copy.deepcopy(data)
        def recorded(actual,*args,**kwargs):
            self.assertIs(actual,data);self.assertIs(type(actual['steps']),list);self.assertIs(kwargs['base'],base)
            raise RuntimeError('full replay sentinel')
        with self.assertRaisesRegex(RuntimeError,'full replay sentinel'):
            with q.adapted_function(self.http,self.server)(base,NS(run_recorded_case=recorded),None,None,None,data,[],None):
                base.run_case(dict(name='D-C8',concurrency=8),*([None]*7))
        self.assertEqual(data,before);self.assertEqual(base.run_case,'old case');self.assertIs(base.Observer,BaseObserver)
    def test_complete_double_routes_reuse_counts_and_failure_preservation(self):
        header=dict(event='route-startup');rows=[header,*pair(2,0,1),*pair(3,1,1)];blob=wire_rows(rows)
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'routes';path.write_bytes(blob)
            self.assertEqual(q.route_expectations(path,header,self.server),dict(expected_route_calls=2,expected_compiled_calls=2,expected_route_fallbacks=0))
            output=io.StringIO();cursor=self.http.CapturedRoutes(io.BytesIO(blob),header,output,q.ref(path),'HTTP-C8')
            for row in rows[1:]:cursor.emit(row)
            self.assertEqual(cursor.finish(q.sha(path))['calls'],2)
            cursor=self.http.CapturedRoutes(io.BytesIO(blob),header,io.StringIO(),q.ref(path),'HTTP-C8')
            with self.assertRaisesRegex(ValueError,'unconsumed'):cursor.finish(q.sha(path))
            bad=copy.deepcopy(rows);bad[1]['after'][0]['fallback']=1;path.write_bytes(wire_rows(bad))
            with self.assertRaises(ValueError):q.route_expectations(path,header,self.server)
            path.write_bytes(wire_rows([header,*pair(2,0,0)]))
            with self.assertRaisesRegex(ValueError,'reuse'):q.route_expectations(path,header,self.server)
            output=io.StringIO();cursor=self.http.CapturedRoutes(io.BytesIO(blob),header,output,{},'HTTP-C8')
            changed=copy.deepcopy(rows[1]);changed['B']=2
            with self.assertRaises(ValueError):cursor.emit(changed)
            self.assertIn('"B": 2',output.getvalue())
    def test_original_raw_join_two_waves_uses_received_bytes_and_rejects_changed_evidence(self):
        with tempfile.TemporaryDirectory() as td:
            child=Path(td).resolve();spec,numeric,joined,tokenizers=tiny_capture(child,self.trial,self.http)
            with patch.dict(sys.modules,{'tokenizers':tokenizers}):
                actual,_=q.rejoin(child,spec,numeric,self.trial,self.http)
                self.assertEqual(actual,joined);self.assertEqual(q.recording_ok(actual)['expected_steps'],6)
                self.assertEqual(len(actual['jobs']),16)
                changed=copy.deepcopy(joined);next(iter(changed['jobs'].values()))['raw_ids'][0]=777
                q.write(child/'joined-input/recording.json',changed)
                with self.assertRaisesRegex(ValueError,'raw rejoin'):q.rejoin(child,spec,numeric,self.trial,self.http)
                q.write(child/'joined-input/recording.json',joined)
                group=q.read(child/'client/warm/group.json');group['requests'][0]['raw_body_bytes']-=1
                q.write(child/'client/warm/group.json',group)
                with self.assertRaises(ValueError):q.rejoin(child,spec,numeric,self.trial,self.http)
    def test_runtime_failure_writes_failed_receipt_and_keeps_flags_off(self):
        with tempfile.TemporaryDirectory() as td,patch.object(q,'validate',lambda plan:None),\
             patch.dict(q.os.environ,{},clear=True),patch.object(q,'helpers',side_effect=RuntimeError('retained failure')),redirect_stdout(io.StringIO()):
            target=Path(td)/'out';code=q.run(dict(environment={}),target);result=q.read(target/'complete.json')
            self.assertEqual(code,1);self.assertEqual(result['status'],'FAIL');self.assertIn('retained failure',result['error']['message'])
            self.assertFalse(result['replay_numerical_checks_passed'] or result['performance_qualified'])
            self.assertTrue((target/'plan.json').is_file())


if __name__=='__main__':unittest.main()
