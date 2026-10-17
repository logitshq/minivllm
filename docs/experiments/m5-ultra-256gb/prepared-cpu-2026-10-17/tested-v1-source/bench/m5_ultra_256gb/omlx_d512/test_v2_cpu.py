"""Explicit CPU mocks only; no actual oMLX imports, listeners, MLX or models.

New gates use frozen vendor Pydantic class AST, real SSE parser/cleanup and a
fake direct process. A passing CPU test does not qualify official execution.
"""
import ast
import asyncio
import base64
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace as NS

from bench.m5_ultra_256gb.omlx_d512 import serve_probe_v2 as p, join_v2 as j, trial as t
from bench.m5_ultra_256gb.omlx_d512.test_probe_cpu import fixture as old_fixture

PROBE='a'*64
MODEL='/tmp/omlx-cpu-only-model'
PID=79123


def identity():
    return dict(engine_id=44,class_module='omlx.engine.vlm',class_name='VLMBatchedEngine',
        class_file=str(p.ROOT/'engine/vlm.py'),class_sha256=p.PINS['engine/vlm.py'],
        model_name=MODEL,model_path=MODEL,core_class_module='omlx.engine_core',core_class_name='AsyncEngineCore',
        scheduler_class_module='omlx.scheduler',scheduler_class_name='Scheduler',scheduler_id=1,
        prefix_cache_enabled=False)


def fixture(C=1,length=False):
    group,events,_,fixtures=old_fixture(C,length)
    jobs=p.jobs_for(list(fixtures.values()),C)
    events[0].update(schema='E023-omlx-D512-observer-v2',sources=p.PINS,fixture_sha256=p.FIXTURES_SHA,
        model_path=MODEL,official_args=['serve','--cpu-mock'],runtime=dict(python_executable=str(p.ENV_ROOT/'bin/python'),
        python_resolved=str((p.ENV_ROOT/'bin/python').resolve()),prefix=str(p.ENV_ROOT),python_version='3.13.7',
        versions=p.VERSIONS,pid=PID,ppid=os.getpid()))
    for i,(row,job) in enumerate(zip(group['requests'],jobs)):
        row.update(copy.deepcopy(job));start=1.0+2*(i//C);done=start+1
        row.update(start_perf_counter_s=start,done_perf_counter_s=done)
        row['raw_sse_chunks'][0]['client_perf_counter_s']=done
    responses=[e for e in events if e['event']=='response']
    for response,job in zip(responses,jobs):
        response['body']=copy.deepcopy(job['body']);response['loaded_identity']=identity()
    for event in events:
        if event['event']=='batch':event['route']=p.expected_route(C)
        if event['event']=='wire':event.update(stream_error=None,close_error=None,iterator_exhausted=False)
    return group,events[:-1],jobs,fixtures


def sealed(data):
    group,events,jobs,fixtures=data
    stream=io.BytesIO();closure=p.Closure(list(fixtures.values()),group['concurrency'],MODEL)
    ledger=p.Ledger(stream,closure)
    with patch.object(p.os,'fsync'):
        # BytesIO has no fileno; durable writes are exercised with a real temp file below.
        original=ledger._write
        ledger._write=lambda event,durable=False:original(event,False)
        for event in events:ledger(event)
    return stream.getvalue().splitlines(keepends=True)


def check(data,lines=None):
    group,events,jobs,fixtures=data
    return j.join(group,sealed(data) if lines is None else lines,jobs,fixtures,lambda ids:'a'*len(ids),PROBE,
        expected_official_args=['serve','--cpu-mock'],expected_model_path=MODEL,expected_pid=PID)


class SealAndJoinTests(unittest.TestCase):
    def test_actual_complete_counts_close_optional_and_identity_count_rule(self):
        for C,length in ((1,False),(8,False),(1,True)):
            data=fixture(C,length);result=check(data);N=8 if C==1 else 16
            self.assertEqual(result['raw_output_count'],N*(512 if length else 2))
            self.assertEqual(result['committed_output_count'],N*(512 if length else 1))
            self.assertEqual(result['max_http_outstanding'],C)
            self.assertFalse(result['observer_closed'] or result['generator_eof_proven'] or result['performance_qualified'])
            self.assertTrue(result['request_sealed'])

    def test_seal_requires_actual_request_set_and_done_not_expected_count(self):
        data=fixture();events=data[1]
        for kind in ('binding','batch','wire'):
            broken=copy.deepcopy(data)
            index=next(i for i,e in enumerate(broken[1]) if e['event']==kind)
            broken[1].pop(index)
            with self.subTest(kind=kind),self.assertRaises((ValueError,KeyError)):
                check(broken)
        closure=p.Closure(list(data[3].values()),1,MODEL)
        with self.assertRaisesRegex(ValueError,'incomplete'):closure.receipt('0'*64,0)
        broken=copy.deepcopy(data);next(e for e in broken[1] if e['event']=='wire')['complete']=False
        with self.assertRaises(ValueError):check(broken)

    def test_changed_runtime_loadclass_cache_and_full_prompt_are_rejected(self):
        for key in ('runtime','class','model','cache','prefix','prompt','sampling'):
            data=fixture();response=next(e for e in data[1] if e['event']=='response')
            if key=='runtime':data[1][0]['runtime']['versions']={**p.VERSIONS,'mlx':'other'}
            if key=='class':response['loaded_identity']['class_name']='BatchedEngine'
            if key=='model':response['loaded_identity']['model_path']='/tmp/other'
            if key=='cache':next(e for e in data[1] if e['event']=='batch')['route']['hot_cache_max_size']=1
            if key=='prefix':response['loaded_identity']['prefix_cache_enabled']=True
            if key=='prompt':response['prompt_ids']=[2]
            if key=='sampling':next(e for e in data[1] if e['event']=='binding')['sampling']['temperature']=.01
            with self.subTest(key=key),self.assertRaises(ValueError):check(data)

    def test_unused_ssd_limit_is_not_confused_with_live_cache(self):
        data=fixture()
        for event in data[1]:
            if event['event']=='batch':event['route']['paged_ssd_cache_max_size']=100*(1<<30)
        self.assertEqual(check(data)['status'],'PASS')
        next(e for e in data[1] if e['event']=='batch')['route']['paged_ssd_cache_manager_present']=True
        with self.assertRaises(ValueError):check(data)

    def test_independent_join_rejects_prefix_tamper_and_events_after_seal(self):
        data=fixture();rows=sealed(data)
        seal=json.loads(rows[-1]);seal['prefix_sha256']='0'*64
        with self.assertRaises(ValueError):check(data,rows[:-1]+[(json.dumps(seal)+'\n').encode()])
        for extra in (data[1][1],dict(event='violation',message='late event')):
            with self.subTest(extra=extra['event']),self.assertRaises(ValueError):
                check(data,rows+[(json.dumps(extra)+'\n').encode()])
        result=check(data,rows+[b'{"event":"close","restored":true,"official_returned_normally":true}\n'])
        self.assertTrue(result['observer_closed'])

    def test_c8_actual_intervals_and_actual_batch_are_separate_gates(self):
        data=fixture(8)
        for i,row in enumerate(data[0]['requests']):
            row['start_perf_counter_s']=2*i+1.;row['done_perf_counter_s']=2*i+2.
            row['raw_sse_chunks'][0]['client_perf_counter_s']=2*i+2.
        with self.assertRaisesRegex(ValueError,'HTTP outstanding'):check(data)
        self.assertEqual(j.http_outstanding([dict(start_perf_counter_s=1.,done_perf_counter_s=2.),
            dict(start_perf_counter_s=2.,done_perf_counter_s=3.)],1),1)

    def test_vendor_completion_model_preserves_new_request_exactly(self):
        # Execute only these actual field definitions, never import the official package.
        from pydantic import BaseModel,Field,field_validator
        from typing import Optional,List,Union
        source=p.ROOT/'api/openai_models.py'
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(),p.PINS['api/openai_models.py'])
        tree=ast.parse(source.read_text());classes=[n for n in tree.body if isinstance(n,ast.ClassDef)
            and n.name in ('StreamOptions','CompletionRequest')]
        self.assertEqual(len(classes),2)
        namespace=dict(BaseModel=BaseModel,Field=Field,field_validator=field_validator,Optional=Optional,List=List,Union=Union)
        exec(compile(ast.Module(body=classes,type_ignores=[]),str(source),'exec'),namespace)
        request=namespace['CompletionRequest'];body=fixture()[2][0]['body']
        actual=request(**body).model_dump(mode='json',exclude_unset=True)
        self.assertEqual(json.dumps(actual,sort_keys=True),json.dumps(body,sort_keys=True))
        self.assertIs(type(body['temperature']),float);self.assertNotIn('n',body)
        old={**body,'temperature':0,'n':1}
        self.assertNotEqual(json.dumps(request(**old).model_dump(mode='json',exclude_unset=True),sort_keys=True),
                            json.dumps(old,sort_keys=True))


class StreamErrorTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_wrapper_records_original_failure_after_done_and_cancel_semantics(self):
        for failure in (None,RuntimeError('after DONE'),asyncio.CancelledError()):
            events=[]
            class Core:
                async def add_request(self,prompt,sampling_params=None,request_id=None,**kwargs):return 'unused'
            class Scheduler:
                def _process_batch_responses(self,responses):return responses
            async def original(engine,prompt,request,model_load_duration=0.,prompt_token_ids=None,
                resolved_model=None,response_id=None,inference_request_id=None):
                yield 'data: [DONE]\n\n'
                if failure is not None:raise failure
            server=NS(stream_completion=original);request=NS(model_dump=lambda **k:{})
            with p.Observer(server,Core,Scheduler,lambda _: {},events.append,1,identity_read=lambda _:identity()):
                stream=server.stream_completion(object(),'x',request,prompt_token_ids=[1],response_id='cmpl-00000001')
                self.assertEqual(await anext(stream),'data: [DONE]\n\n')
                try:await anext(stream)
                except (StopAsyncIteration,RuntimeError,asyncio.CancelledError):pass
            wire=next(e for e in events if e['event']=='wire')
            self.assertEqual(p.wire_errors_allowed(wire),not isinstance(failure,RuntimeError))
            if failure is not None:self.assertEqual(wire['stream_error']['type'],type(failure).__name__)
        data=fixture();next(e for e in data[1] if e['event']=='wire')['close_error']={'type':'RuntimeError','message':'close failed'}
        with self.assertRaises(ValueError):check(data)


class Process:
    pid=PID
    returncode=None
    def poll(self):return self.returncode
    def terminate(self):self.returncode=-15
    def kill(self):self.returncode=-9


class Backend:
    """Fake process/files, actual frozen SSE client over MockTransport."""
    def __init__(self,data,fail=None):self.data=data;self.child=Process();self.fail=fail;self.waited=0
    def spawn(self,command,environment,stdout,stderr,*,cwd):self.spawned=(command,environment,cwd);return self.child
    async def port(self,port,pid=None):
        if pid is None and (not hasattr(self,'spawned') or self.child.poll() is not None):return dict(returncode=1,stdout='',stderr='')
        return dict(returncode=0,stdout=f'p{PID}\nf4\nn127.0.0.1:{port}\n',stderr='')
    async def identity(self,pid):
        return dict(returncode=0,stdout=f'{pid} {os.getpid()} Sat Oct 17 00:00:00 2026 python\n',stderr='')
    async def readiness(self,spec):
        raw={'object':'list','data':[dict(id=spec['served_model'],object='model',owned_by='omlx')]}
        return dict(status=200,body_base64=base64.b64encode(json.dumps(raw).encode()).decode())
    async def group(self,spec,output):
        import httpx
        client=t.load('../pareto_realistic/native_cleanup.py')
        out=Path(spec['output_directory']);(out/'probe').mkdir()
        with (out/'probe/observer.jsonl').open('xb') as stream:
            ledger=p.Ledger(stream,p.Closure(list(self.data[3].values()),spec['concurrency'],MODEL))
            for event in self.data[1]:ledger(event)
        if self.fail=='seal':(out/'probe/observer.jsonl').write_bytes(b'')
        body_by_id={row['request_id']:base64.b64decode(row['raw_sse_chunks'][0]['body_bytes_base64']) for row in self.data[0]['requests']}
        class Bytes(httpx.AsyncByteStream):
            def __init__(self,raw):self.raw=raw
            async def __aiter__(self):
                await asyncio.sleep(0)
                yield self.raw
            async def aclose(self):pass
        async def handle(request):
            if self.fail=='cancel':await asyncio.Event().wait()
            return httpx.Response(200,headers={'content-type':'text/event-stream'},stream=Bytes(body_by_id[request.headers['X-Request-ID']]))
        return await client.run_group(f"http://127.0.0.1:{spec['port']}",spec['jobs'],spec['concurrency'],
            phase='qualification',qualification=None,headers={'Authorization':'Bearer '+spec['api_key']},
            request_timeout_s=spec['deadlines']['request'],group_timeout_s=spec['deadlines']['group'],
            cleanup_timeout_s=spec['deadlines']['close'],output_dir=output,transport=httpx.MockTransport(handle))
    async def seal(self,path):return t.request_seal(path)
    async def wait(self,child,seconds):self.waited+=1;return child.returncode
    async def absent(self,pid):return dict(returncode=1,stdout='',stderr='')
    def decoder(self,spec):return lambda ids:'a'*len(ids)


def fake_spec(out,C=1):
    data=fixture(C);jobs=data[2]
    spec=dict(output_directory=str(out),concurrency=C,port=18024,api_key='cpu',command=['cpu-only'],cwd='/tmp',environment={},
        sources={str(t.HERE/'serve_probe_v2.py'):PROBE},official_args=['serve','--cpu-mock'],model_directory=MODEL,
        served_model='Qwen3.8-27B-nvfp4',jobs=jobs,fixtures=list(data[3].values()),files={str(out/'state/settings.json'):{},
        str(out/'state/model_settings.json'):{}},symlinks={str(out/'models/Qwen3.8-27B-nvfp4'):MODEL},
        deadlines=dict(overall=4,ready=1,request=1,group=2,close=.1,term=15,kill=15,resource_interval=.01))
    return spec,data


class TrialTests(unittest.IsolatedAsyncioTestCase):
    async def test_direct_trial_sse_seal_and_explained_sigterm_reap(self):
        for C in (1,8):
            with tempfile.TemporaryDirectory() as directory:
                out=Path(directory)/'trial';spec,data=fake_spec(out,C);backend=Backend(data);calls=[]
                with patch.object(t,'validate'):
                    result=await t.run(spec,out,lambda label,pid:calls.append((label,pid)),backend=backend)
                self.assertEqual(result['status'],'PASS',result['errors'])
                self.assertTrue(result['route_count_text_qualified'] and result['server_reaped'])
                self.assertFalse(result['observer_closed'] or result['performance_qualified'])
                self.assertEqual(t.read(out/'release.json')['returncode'],-15)
                self.assertEqual(backend.spawned[2],'/tmp');self.assertGreaterEqual(backend.waited,1)
                self.assertEqual(t.read(out/'raw-join.json')['max_http_outstanding'],C)
                self.assertEqual(calls[-1],('after-reap',None))

    async def test_missing_seal_never_qualifies_but_reaps(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)/'trial';spec,data=fake_spec(out);backend=Backend(data,'seal')
            with patch.object(t,'validate'):result=await t.run(spec,out,lambda *_:None,backend=backend)
            self.assertEqual(result['status'],'FAIL');self.assertTrue(result['server_reaped'])
            self.assertFalse(result['route_count_text_qualified']);self.assertTrue((out/'group.json').exists())

    async def test_cancellation_keeps_client_partial_and_reaps(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)/'trial';spec,data=fake_spec(out);backend=Backend(data,'cancel')
            async def checkpoint(label,pid):
                if label=='qualification':raise asyncio.CancelledError('CPU stop')
            with patch.object(t,'validate'):result=await t.run(spec,out,checkpoint,backend=backend)
            self.assertEqual(result['status'],'FAIL');self.assertTrue(result['server_reaped'])
            self.assertTrue((out/'group.json').is_file())
            self.assertFalse(t.read(out/'group.json')['summary']['formal_point_eligible'])
            self.assertTrue(result['client_task_resolved'])

    async def test_kill_reaps_but_never_grants_request_qualification(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)/'trial';spec,data=fake_spec(out);backend=Backend(data)
            backend.child.terminate=lambda:None
            original_wait=backend.wait
            async def wait(child,seconds):
                if child.poll() is None:raise subprocess.TimeoutExpired('CPU only',seconds)
                return await original_wait(child,seconds)
            backend.wait=wait
            with patch.object(t,'validate'):result=await t.run(spec,out,lambda *_:None,backend=backend)
            self.assertEqual(result['status'],'FAIL');self.assertTrue(result['server_reaped'])
            self.assertFalse(result['route_count_text_qualified'])
            self.assertEqual(t.read(out/'release.json')['returncode'],-9)

    async def test_unknown_spawn_failure_keeps_intent_without_forged_reap(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)/'trial';spec,data=fake_spec(out);backend=Backend(data)
            def spawn(*args,**kwargs):raise OSError('CPU unknown Popen result')
            backend.spawn=spawn
            with patch.object(t,'validate'):result=await t.run(spec,out,lambda *_:None,backend=backend)
            self.assertEqual(result['status'],'FAIL');self.assertTrue(result['spawn_attempted'])
            self.assertFalse(result['popen_returned'] or result['server_reaped'] or result['route_count_text_qualified'])
            self.assertTrue((out/'spawn-intent.json').is_file());self.assertFalse((out/'release.json').exists())

    def test_materialization_cannot_escape_registered_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)/'trial';out.mkdir();spec,_=fake_spec(out)
            spec['files'][str(Path(directory)/'outside.json')]={}
            with self.assertRaises(ValueError):t.materialize(spec,out)
            self.assertFalse((Path(directory)/'outside.json').exists())


if __name__=='__main__':unittest.main()
