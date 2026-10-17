"""CPU-only protocol/ownership fakes; no real server, GPU, signal or service."""
import asyncio
import base64
import copy
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import httpx

from bench.m5_ultra_256gb.lithos_d512_retry import trial as m
from bench.m5_ultra_256gb.lithos_d512_retry import test_cpu as fixtures


def fake_spec(out,C):
    group,events,jobs,fs=fixtures.fixture(C)
    events[0]['probe_sha256']=m.FIXED['serve_probe.py']
    spec=dict(schema='E023-lithos-D512-trial-spec-v1',output_directory=str(out.resolve()),concurrency=C,port=18029,
        api_key='CPU-key',command=['CPU-no-process','--run'],cwd='/CPU-no-cwd',environment={'CPU':'only'},
        deadlines=dict(ready=.05,request=2,group=2,close=.05,term=15,kill=15,overall=3,resource_interval=.01),
        official_args=fixtures.ARGS,pack_dir=fixtures.PACK,served_model='CPU-model',jobs=jobs,fixtures=list(fs.values()))
    return spec,group,events


class Child:
    pid=33331
    def __init__(self):self.returncode=None;self.actions=[];self.stubborn=False;self.term_error=False
    def poll(self):return self.returncode
    def terminate(self):
        self.actions.append('TERM')
        if self.term_error:raise OSError('CPU TERM failed')
        if not self.stubborn:self.returncode=-15
    def kill(self):self.actions.append('KILL');self.returncode=-9


class Stream(httpx.AsyncByteStream):
    def __init__(self,raw):self.raw=raw
    async def __aiter__(self):
        await asyncio.sleep(0);yield self.raw[:11]
        await asyncio.sleep(0);yield self.raw[11:]


class Backend:
    def __init__(self,spec,group,events):
        self.spec,self.fixture,self.events=spec,group,events;self.child=Child();self.spawned=False
        self.block=False;self.unknown=False;self.bad_identity=False;self.bad_reap=False;self.wrong_ready=False
        self.group_started=asyncio.Event();self.cleanup_started=asyncio.Event();self.allow_reap=None
        self.bad_observer=False;self.incomplete_cleanup=False;self.headers=[]
    def decoder(self,spec):return lambda ids:''.join('a' if token==7 else '?' for token in ids)
    def spawn(self,command,environment,stdout,stderr,*,cwd):
        assert command==self.spec['command'] and environment==self.spec['environment'] and cwd==self.spec['cwd']
        self.spawned=True
        if self.unknown:raise RuntimeError('CPU unknown Popen outcome')
        self.stdout,self.stderr=stdout,stderr;return self.child
    async def port(self,port,pid=None):
        return dict(returncode=0 if pid else 1,stdout=f'p{pid}\nf4\nn127.0.0.1:{port}\n' if pid else '',stderr='')
    async def identity(self,pid):
        return dict(returncode=0,stdout=f'{pid} {1 if self.bad_identity else os.getpid()} CPU',stderr='')
    async def readiness(self,spec):
        document=dict(object='list',data=[dict(id='wrong' if self.wrong_ready else spec['served_model'],object='model',
                                             created=100,owned_by='lithos-metal',context_window=2048)])
        return dict(status=200,body_base64=base64.b64encode(json.dumps(document).encode()).decode())
    async def wait(self,child,seconds):
        self.cleanup_started.set()
        if self.allow_reap is not None:await self.allow_reap.wait()
        if child.poll() is None:raise subprocess.TimeoutExpired('CPU',seconds)
        return child.returncode
    async def absent(self,pid):
        return dict(returncode=0 if self.bad_reap else 1,stdout=str(pid) if self.bad_reap else '',stderr='')
    async def group(self,spec,output):
        self.group_started.set();client=m.load('../pareto_realistic/lithos_client.py')
        if self.block:
            try:await asyncio.Future()
            except asyncio.CancelledError:
                raise client.GroupCancelled(dict(group_error={'type':'CancelledError'},cleanup={'complete':False}))
        pending=list(self.fixture['requests'])
        async def handler(request):
            self.headers.append(request.headers['authorization']);body=json.loads(request.content)
            index=next(i for i,r in enumerate(pending) if r['body']==body);row=pending.pop(index)
            raw=b''.join(base64.b64decode(c['body_bytes_base64']) for c in row['raw_ndjson_chunks'])
            return httpx.Response(200,stream=Stream(raw),headers={'content-type':'application/x-ndjson'})
        value=await client.run_group('http://127.0.0.1:18029',spec['jobs'],spec['concurrency'],phase='qualification',
            qualification=None,headers={'Authorization':'Bearer '+spec['api_key']},transport=httpx.MockTransport(handler),
            request_timeout_s=2,group_timeout_s=2,cleanup_timeout_s=.05,output_dir=output)
        probe=Path(spec['output_directory'])/'probe';probe.mkdir()
        events=copy.deepcopy(self.events)
        if self.bad_observer:events.pop()
        (probe/'observer.jsonl').write_bytes(b''.join(fixtures.lines(events)))
        for _ in spec['jobs']:
            self.stderr.write(b"INFO:monolith.serve:Generation: {'cached_prompt_tokens': 0, 'setup_ms': 1.0, 'verify_tokens': 1}\n")
        self.stderr.flush()
        if self.incomplete_cleanup:value['cleanup']['complete']=False
        return value


class Tests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name).resolve()
        self.out=self.root/'trial';self.spec,group,events=fake_spec(self.out,8);self.backend=Backend(self.spec,group,events)
        self.checkpoints=[];self.validator=patch.object(m,'validate',return_value=None)
        self.validator.start();self.addCleanup(self.validator.stop)
    async def checkpoint(self,label,pid):self.checkpoints.append((label,pid))
    async def run_trial(self):return await m.run(self.spec,self.out,self.checkpoint,backend=self.backend)
    async def test_real_frozen_client_raw_join_c1_c8_fresh_owned_children(self):
        for C in (1,8):
            out=self.root/f'c{C}';spec,group,events=fake_spec(out,C);backend=Backend(spec,group,events)
            result=await m.run(spec,out,self.checkpoint,backend=backend)
            self.assertEqual(result['status'],'PASS',result);self.assertTrue(result['route_count_text_qualified'])
            self.assertFalse(result['performance_qualified'] or result['full_state_qualified'])
            self.assertEqual(m.read(out/'raw-join.json')['max_http_outstanding'],C)
            self.assertEqual(len(backend.headers),8 if C==1 else 16);self.assertEqual(set(backend.headers),{'Bearer CPU-key'})
            self.assertTrue(result['server_reaped'] and result['cleanup_complete']);self.assertEqual(backend.child.actions,['TERM'])
            self.assertFalse((out/'point.json').exists())
    async def test_readiness_mismatch_reaps_without_requests(self):
        self.backend.wrong_ready=True;result=await self.run_trial()
        self.assertEqual(result['status'],'FAIL');self.assertTrue(result['server_reaped'])
        self.assertFalse(self.backend.group_started.is_set());self.assertFalse((self.out/'raw-join.json').exists())
    async def test_postspawn_identity_failure_still_reaps(self):
        self.backend.bad_identity=True;result=await self.run_trial()
        self.assertEqual(result['status'],'FAIL');self.assertTrue(result['server_reaped'])
        self.assertEqual(self.backend.child.actions,['TERM'])
    async def test_unknown_spawn_never_fabricates_release(self):
        self.backend.unknown=True;result=await self.run_trial()
        self.assertTrue(result['spawn_attempted']);self.assertFalse(result['popen_returned'] or result['server_reaped'])
        self.assertTrue((self.out/'spawn-intent.json').exists());self.assertFalse((self.out/'release.json').exists())
        self.assertIn(('cleanup-incomplete',None),self.checkpoints)
    async def test_terminate_error_still_kills_and_keeps_failure(self):
        self.backend.child.term_error=True;result=await self.run_trial()
        self.assertEqual(result['status'],'FAIL');self.assertTrue(result['server_reaped'])
        self.assertEqual(self.backend.child.actions,['TERM','KILL'])
    async def test_missing_observer_seal_rejects_after_reap(self):
        self.backend.bad_observer=True;result=await self.run_trial()
        self.assertEqual(result['status'],'FAIL');self.assertTrue(result['server_reaped'])
        self.assertFalse(result['route_count_text_qualified']);self.assertIn('all real generations',result['errors'][-1]['message'])
    async def test_unresolved_client_or_surviving_child_not_qualified(self):
        self.backend.incomplete_cleanup=True;self.backend.bad_reap=True;result=await self.run_trial()
        self.assertEqual(result['status'],'FAIL');self.assertFalse(result['cleanup_complete'] or result['server_reaped'])
        self.assertFalse((self.out/'release.json').exists());self.assertIn(('cleanup-incomplete',self.backend.child.pid),self.checkpoints)
    async def test_checkpoint_failure_during_group_cancels_then_reaps(self):
        self.backend.block=True
        async def reject(label,pid):
            if label=='qualification':raise RuntimeError('CPU resource guard')
        result=await m.run(self.spec,self.out,reject,backend=self.backend)
        self.assertEqual(result['status'],'FAIL');self.assertTrue(result['server_reaped'] and result['client_task_resolved'])
        self.assertEqual(m.read(self.out/'group.json')['group_error']['type'],'CancelledError')
        self.assertFalse(result['cleanup_complete'])
    async def test_repeated_outer_cancel_cannot_cancel_owned_reap(self):
        self.backend.block=True;self.backend.allow_reap=asyncio.Event()
        task=asyncio.create_task(self.run_trial());await self.backend.group_started.wait();task.cancel()
        await self.backend.cleanup_started.wait();task.cancel();await asyncio.sleep(0)
        self.assertFalse(task.done());self.backend.allow_reap.set();result=await task
        self.assertEqual(result['status'],'FAIL');self.assertTrue(result['server_reaped'] and result['cleanup_task_resolved'])
        self.assertTrue(any(e['phase']=='cancel-during-cleanup' for e in result['errors']))
    def test_prepare_uses_frozen_original_pack_and_exact_qualification_command(self):
        self.validator.stop();p=m.load('serve_probe.py');out=self.root/'registered'
        environment=p.specification(out,8,18029,'CPU-key')['environment']
        spec=m.prepare(out,8,18029,'CPU-key',deadlines=self.spec['deadlines'],environment=environment)
        m.validate(spec);self.assertEqual(len(spec['jobs']),16);self.assertTrue(spec['observer_enabled'])
        self.assertIn('--no-draft',spec['official_args']);self.assertIn('--no-warmup',spec['official_args'])
        self.assertIn('--run',spec['command']);self.assertNotIn('--qualification-trace',spec['command'])
        for field,value in [('observer_enabled',False),('concurrency',4),('pack_dir','/CPU/repacked')]:
            bad=copy.deepcopy(spec);bad[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):m.validate(bad)
        with self.assertRaises(ValueError):m.prepare(out,8,18029,'CPU-key',deadlines=spec['deadlines'],
            environment={**environment,'MLX_UNREGISTERED':'on'})


if __name__=='__main__':unittest.main()
