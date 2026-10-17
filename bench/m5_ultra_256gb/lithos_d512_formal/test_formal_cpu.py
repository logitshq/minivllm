"""Focused CPU mock lifecycle/log windows with the real frozen NDJSON client.

The qualification binding in these tests is explicitly synthetic; no model,
real sockets, OS child, signal or resource/service operation is performed.
"""
import asyncio
import base64
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import httpx

from bench.m5_ultra_256gb.lithos_d512_formal import formal as m, contract
from bench.m5_ultra_256gb.lithos_d512_retry import test_cpu as fixtures
from bench.m5_ultra_256gb.lithos_d512_retry.test_trial_cpu import Backend as OriginalBackend,Stream


def fake_spec(out,C=8):
    group,events,original,fs=fixtures.fixture(C)
    jobs={phase:[{**copy.deepcopy(original[i%8]),'request_id':f'lithos-c{C}-r1-{phase}-{i:02d}'}
        for i in range(count)] for phase,count in (('warm',8),('formal',64))}
    binding={name:{'path':'/CPU/not-a-real-'+name,'sha256':'c'*64}
        for name in ('audit','resource_audit','restoration_audit','owner_plan')}
    binding['owner_directory']='/CPU/not-a-real-owner'
    spec=dict(schema='E023-lithos-D512-formal-spec-v1',output_directory=str(out.resolve()),concurrency=C,repeat=1,port=18029,
        api_key='CPU-key',command=['CPU-no-process','-B','-m','monolith.serve'],cwd='/CPU-no-cwd',environment={'CPU':'only'},
        deadlines=dict(ready=.05,request=2,warm=2,formal=2,close=.1,term=15,kill=15,overall=5,resource_interval=.01),
        observer_enabled=False,served_model='CPU-model',jobs=jobs,fixtures=list(fs.values()),qualification=binding,
        base_plan={'path':'/CPU/not-a-real-base','sha256':'b'*64},qualification_identity={'scope':'CPU-mock'},
        contracts={phase:contract._contract(rows,list(fs.values()),binding,C) for phase,rows in jobs.items()})
    return spec,group,events


class Backend(OriginalBackend):
    def __init__(self,spec,group,events):
        super().__init__(spec,group,events)
        self.phases=[];self.formal_setup=False;self.cached=False;self.extra_log=False;self.omit_log=False
        self.rotate=False;self.rewrite=False;self.tail_extra=False;self.warm_close_incomplete=False
        self.counts={};self.formal_began=asyncio.Event();self.block_phase='formal';self.first_term=False
    async def group(self,spec,phase,output):
        self.group_started.set();self.phases.append(phase)
        if phase=='formal':self.formal_began.set()
        client=m.load('../pareto_realistic/lithos_client.py')
        if self.block and phase==self.block_phase:
            try:await asyncio.Future()
            except asyncio.CancelledError:
                raise client.GroupCancelled(dict(group_error={'type':'CancelledError'},cleanup={'complete':False}))
        submitted=[]
        async def handler(request):
            self.headers.append(request.headers['authorization']);body=json.loads(request.content)
            i=len(submitted);submitted.append(body)
            f=next(f for f in spec['fixtures'] if f['messages']==body['messages'])
            stop=i%3!=2;n=1 if i%3==0 else 512;committed=n-int(stop)
            text='a'*committed;finish='stop' if stop else 'length'
            final=dict(model='CPU-model',created_at=f'CPU-{phase}-{i}',message=dict(role='assistant',content=''),
                done=True,done_reason=finish,total_duration=100,load_duration=0,prompt_eval_count=len(f['prompt_ids']),
                prompt_eval_duration=10,eval_count=n,eval_duration=20)
            raw=(json.dumps(dict(model='CPU-model',created_at=f'CPU-{i}',message=dict(role='assistant',content=text),done=False))+'\n'+json.dumps(final)+'\n').encode()
            setup=1.0 if (phase=='warm' and i==0) or (phase=='formal' and self.formal_setup) else 0.0
            log=f"INFO:__main__:Generation: {{'cached_prompt_tokens': {128 if self.cached else 0}, 'setup_ms': {setup}, 'verify_tokens': 1}}\n".encode()
            if not(self.omit_log and phase=='formal' and i==0):self.stderr.write(log)
            if self.extra_log and phase=='formal' and i==0:self.stderr.write(log)
            self.stderr.flush()
            if self.tail_extra and phase=='formal' and i==0:raw+=b'x'
            return httpx.Response(200,stream=Stream(raw),headers={'content-type':'application/x-ndjson'})
        value=await client.run_group('http://127.0.0.1:18029',spec['jobs'][phase],spec['concurrency'],phase=phase,
            qualification=contract.count_contract(spec,phase),headers={'Authorization':'Bearer '+spec['api_key']},
            transport=httpx.MockTransport(handler),request_timeout_s=2,group_timeout_s=2,cleanup_timeout_s=.1,output_dir=output)
        self.counts[phase]=len(submitted)
        if self.warm_close_incomplete and phase=='warm':value['cleanup']['complete']=False
        if phase=='formal' and self.rotate:
            path=Path(spec['output_directory'])/'server.stderr';raw=path.read_bytes();path.rename(path.with_suffix('.old'));path.write_bytes(raw)
        if phase=='formal' and self.rewrite:
            path=Path(spec['output_directory'])/'server.stderr'
            with path.open('r+b') as stream:stream.write(b'WARN')
        return value


class Tests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name).resolve()
        self.out=self.root/'trial';self.spec,group,events=fake_spec(self.out);self.backend=Backend(self.spec,group,events)
        self.checkpoints=[];self.real_load=m.load
        self.loader=patch.object(m,'load',side_effect=lambda name:contract if name=='contract.py' else self.real_load(name))
        self.loader.start();self.addCleanup(self.loader.stop)
        self.validator=patch.object(m,'validate',return_value=None);self.validator.start();self.addCleanup(self.validator.stop)
        self.binder=patch.object(contract,'bind',side_effect=lambda *args:({},list(fixtures.fixture()[3].values()),{}, {'scope':'CPU-mock'}))
        self.binder.start();self.addCleanup(self.binder.stop)
    async def checkpoint(self,label,pid):self.checkpoints.append((label,pid))
    async def run_trial(self):return await m.run(self.spec,self.out,self.checkpoint,backend=self.backend)

    async def test_c1_c8_original_client_two_groups_and_exact_durable_log_slices(self):
        for C in (1,8):
            out=self.root/f'c{C}';spec,g,e=fake_spec(out,C);b=Backend(spec,g,e)
            result=await m.run(spec,out,self.checkpoint,backend=b)
            self.assertEqual(result['status'],'PASS',result);self.assertEqual(b.counts,{'warm':8,'formal':64})
            self.assertEqual(b.phases,['warm','formal']);self.assertEqual(b.child.actions,['TERM'])
            self.assertEqual(set(b.headers),{'Bearer CPU-key'});self.assertFalse((out/'probe').exists())
            self.assertTrue(result['count_eligible'] and result['server_reaped'] and result['logs_rechecked'])
            self.assertFalse(result['performance_qualified'] or result['full_state_qualified'])
            raw=(out/'server.stderr').read_bytes()
            for phase,count in (('warm',8),('formal',64)):
                rec=m.read(out/(phase+'-log-check.json'));start,end=rec['start'],rec['end']
                window=raw[start['offset']:end['offset']]
                self.assertEqual(window,(out/rec['artifact']['path']).read_bytes())
                self.assertEqual(hashlib.sha256(window).hexdigest(),rec['artifact']['sha256'])
                self.assertEqual(rec['check']['count'],count);self.assertEqual(start['inode'],end['inode'])
            warm=m.read(out/'warm-group.json');formal=m.read(out/'formal-group.json')
            self.assertLessEqual(warm['cleanup']['end_perf_counter_s'],min(r['start_perf_counter_s'] for r in formal['requests']))
            point=m.read(out/'point.json');self.assertEqual(point['requests'],64);self.assertEqual(point['max_http_outstanding'],C)
            self.assertEqual(point['common_window_s'],max(r['done_perf_counter_s'] for r in formal['requests'])-min(r['start_perf_counter_s'] for r in formal['requests']))

    async def test_bad_formal_setup_cache_missing_extra_logs_fail_and_preserve_raw(self):
        for flag in ('formal_setup','cached','extra_log','omit_log'):
            out=self.root/flag;spec,g,e=fake_spec(out);b=Backend(spec,g,e);setattr(b,flag,True)
            result=await m.run(spec,out,self.checkpoint,backend=b)
            self.assertEqual(result['status'],'FAIL',flag);self.assertTrue(result['server_reaped'])
            self.assertFalse(result['count_eligible']);self.assertFalse((out/'point.json').exists())
            self.assertTrue((out/'server.stderr').is_file());self.assertTrue((out/'warm-group.json').is_file())
            if flag!='cached':self.assertTrue((out/'formal-group.json').is_file())

    async def test_rotated_or_rewritten_stderr_never_qualifies(self):
        for flag in ('rotate','rewrite'):
            out=self.root/flag;spec,g,e=fake_spec(out);b=Backend(spec,g,e);setattr(b,flag,True)
            result=await m.run(spec,out,self.checkpoint,backend=b)
            self.assertEqual(result['status'],'FAIL');self.assertTrue(result['server_reaped'])
            self.assertFalse(result['count_eligible']);self.assertTrue(any('stderr' in x['message'] for x in result['errors']))

    async def test_unclosed_warm_never_starts_formal(self):
        self.backend.warm_close_incomplete=True;result=await self.run_trial()
        self.assertEqual(result['status'],'FAIL');self.assertEqual(self.backend.phases,['warm'])
        self.assertFalse((self.out/'formal-start-log-boundary.json').exists());self.assertTrue(result['server_reaped'])

    async def test_bad_ndjson_tail_does_not_publish_point(self):
        self.backend.tail_extra=True;result=await self.run_trial()
        self.assertEqual(result['status'],'FAIL');self.assertTrue(result['server_reaped']);self.assertFalse(result['count_eligible'])
        self.assertTrue((self.out/'formal-group.json').is_file());self.assertFalse((self.out/'point.json').exists())

    async def test_resource_failure_during_formal_keeps_partial_and_reaps(self):
        self.backend.block=True
        async def reject(label,pid):
            if label=='formal':raise RuntimeError('CPU resource guard')
        result=await m.run(self.spec,self.out,reject,backend=self.backend)
        self.assertEqual(result['status'],'FAIL');self.assertTrue(result['server_reaped'] and result['client_task_resolved'])
        self.assertEqual(m.read(self.out/'formal-group.json')['group_error']['type'],'CancelledError')
        self.assertFalse(result['cleanup_complete']);self.assertEqual(self.backend.child.actions,['TERM'])

    async def test_repeated_outer_cancel_cannot_interrupt_reap(self):
        self.backend.block=True;self.backend.allow_reap=asyncio.Event()
        task=asyncio.create_task(self.run_trial());await asyncio.wait_for(self.backend.formal_began.wait(),3);task.cancel()
        await asyncio.wait_for(self.backend.cleanup_started.wait(),3);task.cancel();await asyncio.sleep(0)
        self.assertFalse(task.done());self.backend.allow_reap.set();result=await task
        self.assertEqual(result['status'],'FAIL');self.assertTrue(result['server_reaped'] and result['cleanup_task_resolved'])
        self.assertTrue(any(x['phase']=='cancel-during-cleanup' for x in result['errors']))

    async def test_unknown_spawn_or_bad_absence_never_fabricates_clean_point(self):
        for flag in ('unknown','bad_reap'):
            out=self.root/flag;spec,g,e=fake_spec(out);b=Backend(spec,g,e);setattr(b,flag,True)
            result=await m.run(spec,out,self.checkpoint,backend=b)
            self.assertEqual(result['status'],'FAIL');self.assertFalse(result['server_reaped'] or result['count_eligible'])
            self.assertFalse((out/'release.json').exists());self.assertFalse((out/'point.json').exists())

    def test_partial_line_log_boundary_rejected(self):
        self.out.mkdir();stream=(self.out/'server.stderr').open('xb');self.addCleanup(stream.close)
        stream.write(b'partial');stream.flush()
        with self.assertRaisesRegex(ValueError,'splits a line'):
            m.LogWindows(self.out,stream,lambda *a,**k:None).start('warm')

    def test_help_has_no_device_or_network_import(self):
        code="""import importlib.abc,runpy,sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,name,*args):
  if name.split('.')[0] in {'mlx','torch','numpy','monolith','transformers','httpx'}:raise AssertionError(name)
sys.meta_path.insert(0,Block());sys.argv=[sys.argv[1],'--help'];runpy.run_path(sys.argv[0],run_name='__main__')
"""
        result=subprocess.run([sys.executable,'-I','-B','-c',code,str(Path(m.__file__).resolve())],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)


class PreparationTests(unittest.TestCase):
    def test_actual_closed_qualification_prepare_validate_no_backend(self):
        from bench.m5_ultra_256gb.lithos_d512_formal.test_contract_cpu import binding
        budgets=dict(ready=180,request=600,warm=720,formal=2400,close=10,
                     term=15,kill=15,overall=3600,resource_interval=5)
        with tempfile.TemporaryDirectory() as tmp, patch.object(m,'System',side_effect=AssertionError('no backend during prepare')):
            for C in (1,8):
                out=Path(tmp)/f'c{C}';key='CPU-prepare-only'
                env=m.load('../lithos_d512_retry/serve_probe.py').specification(out,C,18029,key,observed=False)['environment']
                spec=m.prepare(out,C,1,18029,key,qualification=binding(),deadlines=budgets,environment=env)
                m.validate(spec)
                self.assertFalse(out.exists());self.assertFalse(spec['observer_enabled'])
                self.assertEqual(spec['command'][1:4],['-B','-m','monolith.serve'])
                self.assertEqual({p:len(rows) for p,rows in spec['jobs'].items()},{'warm':8,'formal':64})
                for phase,rows in spec['jobs'].items():
                    self.assertEqual(m.load('contract.py').count_contract(spec,phase),spec['contracts'][phase])
                    self.assertEqual(rows[0]['body']['messages'],spec['fixtures'][0]['messages'])
                bad=copy.deepcopy(spec);bad['observer_enabled']=True
                with self.assertRaisesRegex(ValueError,'formal spec changed'):m.validate(bad)


if __name__=='__main__':unittest.main()
