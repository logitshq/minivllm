"""Unexecuted fake-only tests; no sockets, model, service or device qualification.

The frozen client/queue/SSE/count checker runs with httpx.MockTransport. Only the
expensive preexisting formal qualification validator is replaced in fixtures.
Process, readiness, resource checkpoint and drain operations are synthetic.
"""
import asyncio
import base64
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import httpx

from bench.m5_ultra_256gb.c1_pipeline_profile import capture as m
from bench.m5_ultra_256gb.pareto_realistic import test_native_formal_cpu as old


def specification(root,mode='off',*,arm='on'):
    base=old.prepare(root/'metadata',C=1)
    factory=m.factory(); server=factory.module(factory.SERVER)
    base['command'],base['identity']=factory.select_entry(base,arm,server,server.source_pins())
    base['arm']=arm
    source=root/'qualified-spec.json'; m.write(source,base)
    with patch.object(m,'factory',return_value=SimpleNamespace(validate=lambda value:None)):
        spec=m.prepare(old.pin(source),mode,output_directory=root/'trial',drain_token='test-nonce-'+'x'*40,
            server_sha256=m.sha(m.SERVER),drain_seconds=.3,worker_join_seconds=.01)
    return spec


async def raw_group(spec,stage,output):
    fixtures={f['prompt_text']:f for f in spec['fixtures']}; submitted=[]
    async def handler(request):
        body=json.loads(request.content); fixture=fixtures[body['prompt']]
        index=len(submitted); submitted.append(body); eos=spec['identity']['eos_ids'][0]
        raw=[10,eos] if index%3==0 else [eos] if index%3==1 else [11]*512
        rid=f'{index+(1 if stage=="warm" else 1000):032x}'
        terminal=old.server.completion(old.server.Job(rid,fixture['prompt_ids'],512),raw,set(spec['identity']['eos_ids']))
        text=old.decode(terminal['benchmark_committed_output_ids'])
        delta=old.wire(dict(id='cmpl-'+rid,choices=[dict(index=0,text=text,finish_reason=None)])) if text else b''
        return httpx.Response(200,stream=old.Stream(delta+old.wire(terminal)+old.wire('[DONE]')),
                              headers={'content-type':'text/event-stream'})
    return await m.native().load('native_cleanup').run_group('http://cpu-mock.invalid',spec['jobs'][stage],1,
        phase=m.CLIENT_PHASE[stage],qualification=m.count_contract(spec,stage),request_timeout_s=2,
        group_timeout_s=2,cleanup_timeout_s=.1,transport=httpx.MockTransport(handler),output_dir=output)


def retain_server_closure(spec,groups,pid):
    """Synthetic *server* receipts; raw UUIDs come from the actual frozen parser."""
    out=Path(spec['output_directory'])/'server'; out.mkdir()
    m.write(out/'identity.json',spec['identity']); join=m.native().load('http_join'); rows=[]
    for stage in m.STAGES:
        for request in groups[stage]['requests']:
            rid,_,_=join.raw_completion(request)
            start=int(request['start_perf_counter_s']*1e9); done=int(request['done_perf_counter_s']*1e9)
            rows.append(dict(ordinal=len(rows)+1,phase=stage,rid=rid,entered_ns=start,
                             done_flush_ns=done,handler_exit_ns=done+1))
    accepted=max(row['handler_exit_ns'] for row in rows)+1
    drain=dict(phase='DRAINED',expected_requests=8,warm_requests=8,attempts=16,normal_done_flushes=16,
        active_handlers=0,active_completions=0,accepted_at_ns=accepted,stopped_at_ns=accepted+1,requests=rows,errors=[])
    closure=dict(schema='C1-pipeline-diagnostic-http-v1',status='RECORDED',pid=pid,drain=drain,
        worker_stopped=True,handlers_stopped=True,worker_failure=None,errors=[],sources=spec['server_sources'],
        profile_status=None if spec['mode']=='off' else 'RECORDED',numerical_qualified=False,performance_qualified=False)
    m.write(out/'closure.json',closure)
    if spec['mode']!='off':
        m.write(out/'profile.json',dict(schema='C1-pipeline-profile-v1',status='RECORDED',overflow=False,event_count=1,
            events=[dict(sequence=0,kind='fake-span',start_ns=1,end_ns=2)],sources=spec['server_sources'],
            numerical_qualified=False,performance_qualified=False))


class Backend(old.Backend):
    def __init__(self,spec,*,conflicts=0,natural_rc=0,missing_closure=False,close_complete=True,**kwargs):
        super().__init__(spec,**kwargs)
        self.groups={}; self.drains=[]; self.conflicts=conflicts; self.natural_rc=natural_rc
        self.missing_closure=missing_closure; self.close_complete=close_complete
    async def group(self,spec,stage,output):
        self.phases.append(stage); self.group_started.set()
        value=await raw_group(spec,stage,output); self.groups[stage]=value; return value
    async def drain(self,port,token,seconds,output):
        assert self.phases==['warm','diagnostic'] and all(len(g['requests'])==8 for g in self.groups.values())
        assert port==self.spec['transport']['port'] and token==self.spec['drain_token']
        self.drains.append((port,token,seconds))
        status=409 if len(self.drains)<=self.conflicts else 200
        raw=dict(status=status,body_base64=base64.b64encode(json.dumps(
            dict(status='DRAIN_ACCEPTED',performance_qualified=False) if status==200 else {'error':'handler still closing'}
            ).encode()).decode(),close_complete=self.close_complete)
        m.write(output,raw)
        if status==200:
            if not self.missing_closure: retain_server_closure(self.spec,self.groups,self.child.pid)
            self.child.returncode=self.natural_rc
        return raw


class Preparation(unittest.TestCase):
    def test_three_modes_share_actual_jobs_flags_and_explicit_nonformal_view(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve(); specs=[]
            for mode in m.MODES:
                directory=root/mode; directory.mkdir(); specs.append(specification(directory,mode))
            first=specs[0]
            for mode,spec in zip(m.MODES,specs):
                self.assertEqual(spec['mode'],mode)
                for field in ('jobs','environment','transport','deadlines','fixtures','tokenizer'):
                    self.assertEqual(spec[field],first[field])
                self.assertEqual(spec['command'][:spec['command'].index('--profile')],first['command'][:first['command'].index('--profile')])
                self.assertEqual(spec['command'][3],str(m.SERVER))
                self.assertEqual([len(spec['jobs'][s]) for s in m.STAGES],[8,8])
                self.assertFalse(spec['performance_qualified']); self.assertFalse(spec['numerical_qualified'])
                self.assertFalse(spec['identity']['step_observer']); self.assertFalse(spec['identity']['qualification_trace_enabled'])
                self.assertEqual(spec['identity']['combined_entry']['combined_candidates']['gdn_layer_installs'],48)
                self.assertNotIn('--route-events',spec['command'])
    def test_unqualified_base_wrong_entry_bad_nonce_and_modified_spec_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve(); spec=specification(root); path=Path(spec['formal_spec']['path'])
            original=path.read_bytes()
            for field,value in (('concurrency',8),('command',['x','-I','-B','/wrong'])):
                base=json.loads(original); base[field]=value; m.write(path,base)
                with patch.object(m,'factory',return_value=SimpleNamespace(validate=lambda value:None)),self.assertRaises(ValueError):
                    m.prepare(old.pin(path),'off',output_directory=root/'new',drain_token=spec['drain_token'],
                        server_sha256=m.sha(m.SERVER),drain_seconds=.3,worker_join_seconds=.01)
            path.write_bytes(original)
            with self.assertRaises(ValueError):
                m.prepare(spec['formal_spec'],'off',output_directory=root/'new',drain_token='short',server_sha256='not-read')
            with patch.object(m,'factory',return_value=SimpleNamespace(validate=lambda value:(_ for _ in ()).throw(ValueError('not qualified')))),self.assertRaisesRegex(ValueError,'not qualified'):
                m.prepare(spec['formal_spec'],'off',output_directory=root/'new',drain_token=spec['drain_token'],server_sha256=m.sha(m.SERVER))
            bad=copy.deepcopy(spec); bad['jobs']['diagnostic'][0]['body']['temperature']=1
            with patch.object(m,'factory',return_value=SimpleNamespace(validate=lambda value:None)),self.assertRaisesRegex(ValueError,'specification changed'):
                m.validate(bad)
    def test_default_description_never_prepares_or_launches(self):
        stream=io.StringIO()
        with patch.object(m,'factory',side_effect=AssertionError('unexpected import')),contextlib.redirect_stdout(stream):
            self.assertEqual(m.main([]),0)
        value=json.loads(stream.getvalue()); self.assertEqual(value['diagnostic'],8); self.assertFalse(value['performance_qualified'])


class Capture(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve(); self.spec=specification(self.root); self.out=Path(self.spec['output_directory'])
        self.validation=patch.object(m,'validate',return_value=None); self.validation.start(); self.addCleanup(self.validation.stop)
        self.labels=[]
    async def checkpoint(self,label,pid): self.labels.append((label,pid))
    async def test_normal_done_then_natural_exit_never_signals_and_keeps_all_raw(self):
        backend=Backend(self.spec); result=await m.run(self.spec,self.out,checkpoint=self.checkpoint,backend=backend)
        self.assertEqual(result['status'],'PASS',result['errors']); self.assertEqual(backend.child.actions,[])
        self.assertEqual(backend.phases,['warm','diagnostic']); self.assertEqual(len(backend.drains),1)
        self.assertTrue(result['natural_exit'] and result['server_reaped'] and result['cleanup_complete'])
        self.assertFalse(result['forced_stop'] or result['performance_qualified'] or result['numerical_qualified'])
        group=m.read(self.out/'diagnostic-group.json'); self.assertEqual(group['phase'],'admission_probe')
        self.assertFalse(group['summary']['formal_point_eligible']); self.assertEqual(len(group['requests']),8)
        check=m.read(self.out/'diagnostic-raw-check.json'); self.assertEqual(check['validation_view_phase'],'warm')
        self.assertEqual(check['raw_client_phase'],'admission_probe'); self.assertFalse((self.out/'point.json').exists())
        self.assertEqual(self.labels[-1],('after-reap',None)); self.assertTrue(backend.waited)
    async def test_handler_finally_conflict_retries_only_control_POST(self):
        backend=Backend(self.spec,conflicts=1)
        result=await m.run(self.spec,self.out,checkpoint=self.checkpoint,backend=backend)
        self.assertEqual(result['status'],'PASS',result['errors']); self.assertEqual(len(backend.drains),2)
        self.assertEqual(backend.phases,['warm','diagnostic']); self.assertEqual(backend.child.actions,[])
        self.assertEqual(m.read(self.out/'drain-000.json')['status'],409)
    async def test_409_retries_and_final_200_share_absolute_drain_budget(self):
        # Advance only capture's deadline clock, not the event loop's scheduling
        # clock. The fake closes the process on 200, reproducing the bypass in
        # which the subsequent while-child-alive loop has zero iterations.
        clock=[0.0]; backend=Backend(self.spec,conflicts=1); original=backend.drain
        async def delayed(*args):
            value=await original(*args); clock[0]+=.2; return value
        async def yield_only(seconds): await asyncio.sleep(0)
        facade=SimpleNamespace(**vars(asyncio)); facade.get_running_loop=lambda:SimpleNamespace(time=lambda:clock[0])
        facade.sleep=yield_only; backend.drain=delayed
        with patch.object(m,'asyncio',facade):
            result=await m.run(self.spec,self.out,checkpoint=self.checkpoint,backend=backend)
        self.assertEqual(result['status'],'FAIL'); self.assertTrue(result['server_reaped'])
        self.assertFalse(result['natural_exit']); self.assertEqual(backend.child.actions,[])
        self.assertEqual(len(backend.drains),2)
        self.assertAlmostEqual(backend.drains[0][2],.3); self.assertAlmostEqual(backend.drains[1][2],.1)
        self.assertEqual(m.read(self.out/'drain-001.json')['status'],200)
        self.assertTrue(any('absolute deadline' in e['message'] for e in result['errors']))
    async def test_exit_during_late_checkpoint_cannot_bypass_drain_deadline(self):
        clock=[0.0]; backend=Backend(self.spec)
        async def late_checkpoint(label,pid):
            await self.checkpoint(label,pid)
            if label=='draining' and backend.drains: clock[0]=.4
        facade=SimpleNamespace(**vars(asyncio)); facade.get_running_loop=lambda:SimpleNamespace(time=lambda:clock[0])
        with patch.object(m,'asyncio',facade):
            result=await m.run(self.spec,self.out,checkpoint=late_checkpoint,backend=backend)
        self.assertEqual(result['status'],'FAIL'); self.assertTrue(result['drain_accepted'])
        self.assertTrue(result['server_reaped']); self.assertFalse(result['natural_exit'])
        self.assertTrue(any('checkpoint exceeded absolute deadline' in e['message'] for e in result['errors']))
    async def test_repeated_409_checks_resources_without_refreshing_drain_budget(self):
        clock=[0.0]; backend=Backend(self.spec,conflicts=100); original=backend.drain
        checkpoints=[]
        async def conflict(*args):
            value=await original(*args); clock[0]+=.1; return value
        async def yield_only(seconds): await asyncio.sleep(0)
        async def bounded_checkpoint(label,pid):
            await self.checkpoint(label,pid)
            if label=='draining':
                checkpoints.append((len(backend.drains),pid,clock[0]))
                if len(backend.drains)==2: clock[0]+=.11
        facade=SimpleNamespace(**vars(asyncio)); facade.get_running_loop=lambda:SimpleNamespace(time=lambda:clock[0])
        facade.sleep=yield_only; backend.drain=conflict
        with patch.object(m,'asyncio',facade):
            result=await m.run(self.spec,self.out,checkpoint=bounded_checkpoint,backend=backend)
        self.assertEqual([row[0] for row in checkpoints],[0,1,2])
        self.assertTrue(all(row[1]==backend.child.pid for row in checkpoints))
        self.assertEqual(len(backend.drains),2)
        self.assertAlmostEqual(backend.drains[0][2],.3); self.assertAlmostEqual(backend.drains[1][2],.2)
        self.assertEqual(result['status'],'FAIL'); self.assertFalse(result['natural_exit'])
        self.assertTrue(result['server_reaped']); self.assertEqual(backend.child.actions,['TERM'])
        self.assertTrue(any('checkpoint exceeded absolute deadline' in e['message'] for e in result['errors']))
    async def test_bad_diagnostic_raw_prevents_drain_and_preserves_failure(self):
        backend=Backend(self.spec); original=backend.group
        async def bad(spec,stage,out):
            group=await original(spec,stage,out)
            if stage=='diagnostic': group['requests'][0]['answer']='corrupted'
            return group
        backend.group=bad
        result=await m.run(self.spec,self.out,checkpoint=self.checkpoint,backend=backend)
        self.assertEqual(result['status'],'FAIL'); self.assertEqual(backend.drains,[])
        self.assertEqual(backend.child.actions,['TERM']); self.assertTrue(result['server_reaped'])
        self.assertTrue((self.out/'diagnostic-group.json').exists()); self.assertFalse(result['count_verified'])
    async def test_accepted_control_without_natural_clean_closure_never_passes(self):
        for i,options in enumerate((dict(natural_rc=1),dict(missing_closure=True),dict(close_complete=False))):
            spec=copy.deepcopy(self.spec); spec['output_directory']=str(self.root/f'bad-{i}')
            backend=Backend(spec,**options)
            result=await m.run(spec,Path(spec['output_directory']),checkpoint=self.checkpoint,backend=backend)
            self.assertEqual(result['status'],'FAIL'); self.assertTrue(result['server_reaped'])
            self.assertFalse(result['performance_qualified']); self.assertTrue(result['errors'])
            if options.get('close_complete') is False: self.assertFalse(result['client_task_resolved'])
    async def test_closed_uuid_profile_and_done_receipts_must_match_actual_wire(self):
        self.spec['mode']='runner'; self.spec['identity']['diagnostic']['profile']='runner'
        self.spec['command'][self.spec['command'].index('--profile')+1]='runner'; backend=Backend(self.spec)
        result=await m.run(self.spec,self.out,checkpoint=self.checkpoint,backend=backend)
        self.assertEqual(result['status'],'PASS',result['errors'])
        closure=self.out/'server/closure.json'; profile=self.out/'server/profile.json'
        saved_closure=closure.read_bytes(); saved_profile=profile.read_bytes()
        for mutate in (lambda c:c['drain']['requests'][0].update(rid='wrong'),
                       lambda c:c['drain']['requests'][0].update(done_flush_ns=None),
                       lambda c:c.update(worker_stopped=False)):
            closure.write_bytes(saved_closure); value=m.read(closure); mutate(value); m.write(closure,value)
            with self.assertRaises(ValueError):m.closed_receipts(self.out,self.spec,backend.child.pid,backend.groups)
        closure.write_bytes(saved_closure); value=m.read(profile); value['events'][0]['error']='later async failure'; m.write(profile,value)
        with self.assertRaises(ValueError):m.closed_receipts(self.out,self.spec,backend.child.pid,backend.groups)
        profile.write_bytes(saved_profile)
    async def test_repeated_cancel_waits_client_cleanup_then_reaps_with_kill(self):
        backend=Backend(self.spec,ignore_term=True); closing,allow=asyncio.Event(),asyncio.Event()
        async def slow(spec,stage,out):
            backend.group_started.set()
            try: await asyncio.Future()
            except asyncio.CancelledError:
                closing.set(); await allow.wait()
                raise m.native().load('native_cleanup').GroupCancelled(dict(group_error={'type':'CancelledError'},cleanup={'complete':False}))
        backend.group=slow
        task=asyncio.create_task(m.run(self.spec,self.out,checkpoint=self.checkpoint,backend=backend))
        await backend.group_started.wait(); task.cancel(); await closing.wait(); task.cancel(); await asyncio.sleep(0); allow.set()
        result=await task
        self.assertEqual(result['status'],'FAIL'); self.assertEqual(backend.child.actions,['TERM','KILL'])
        self.assertTrue(result['server_reaped']); self.assertEqual(backend.drains,[])
        self.assertTrue((self.out/'warm-group.json').exists())
    async def test_surviving_pid_or_port_cannot_be_called_reaped(self):
        backend=Backend(self.spec,bad_reap=True)
        result=await m.run(self.spec,self.out,checkpoint=self.checkpoint,backend=backend)
        self.assertEqual(result['status'],'FAIL'); self.assertFalse(result['server_reaped'])
        self.assertFalse((self.out/'release.json').exists())
    async def test_unknown_spawn_retains_intent_without_claiming_absence_or_release(self):
        backend=Backend(self.spec)
        def unknown(*args): raise OSError('Popen return was not obtained')
        backend.spawn=unknown
        result=await m.run(self.spec,self.out,checkpoint=self.checkpoint,backend=backend)
        self.assertEqual(result['status'],'FAIL'); self.assertFalse(result['server_reaped'])
        self.assertFalse(result['cleanup_complete']); self.assertEqual(backend.waited,[])
        self.assertTrue((self.out/'spawn-intent.json').exists())
        for name in ('child.json','child-absence.json','release.json'):
            self.assertFalse((self.out/name).exists())
        self.assertEqual(self.labels[-1],('cleanup-incomplete',None))
        # The external owner must reconcile the retained unknown launch; None
        # here is not a NO_CHILD_STARTED proof or permission to restore service.
    async def test_returned_pid_with_wrong_direct_parent_is_still_reaped_and_fails(self):
        backend=Backend(self.spec)
        async def wrong_parent(pid): return dict(returncode=0,stdout=f'{pid} 1 CPU MOCK',stderr='')
        backend.identity=wrong_parent
        result=await m.run(self.spec,self.out,checkpoint=self.checkpoint,backend=backend)
        self.assertEqual(result['status'],'FAIL'); self.assertTrue(result['server_reaped'])
        self.assertEqual(backend.child.actions,['TERM']); self.assertEqual(backend.phases,[])
        self.assertEqual(m.read(self.out/'child.json')['pid'],backend.child.pid)
        self.assertEqual(m.read(self.out/'release.json')['pid'],backend.child.pid)


if __name__=='__main__':unittest.main()
