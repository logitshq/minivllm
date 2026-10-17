"""CPU fake route/wire/async identity tests; no official/runtime imports."""
import asyncio
import base64
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS
import unittest

from bench.m5_ultra_256gb.omlx_d512 import serve_probe as p, join as j

PROBE = 'a' * 64


def lines(events):
    return [(json.dumps(e) + '\n').encode() for e in events]


def fixture(C=1, length=False):
    count = 8 if C == 1 else 16
    fixtures = {f'f{i}': dict(fixture_id=f'f{i}', prompt_text=f'prompt{i}', prompt_ids=[1, i+10],
        max_output_tokens=512, normal_eos=True, eos_token_ids=[99]) for i in range(8)}
    jobs, rows, events = [], [], [dict(event='open', schema='E023-omlx-D512-observer-v1', concurrency=C,
        source_root=str(p.ROOT), sources=p.PINS, probe_sha256=PROBE, performance_qualified=False)]
    bodies, streams = {}, {}
    for i in range(count):
        f = fixtures[f'f{i%8}']; rid = f'cmpl-{i:08x}'; request_id = f'engine-{i}'
        body = dict(model='Qwen3.8-27B-nvfp4', prompt=f['prompt_text'], temperature=0, top_p=1,
                    repetition_penalty=1, max_tokens=512, n=1, stream=True, stream_options={'include_usage': True})
        job = dict(request_id=f'client-{i}', fixture_id=f['fixture_id'], path='/v1/completions', body=body)
        jobs.append(job); bodies[request_id] = f
        events.extend([dict(event='response', response_id=rid, prompt=f['prompt_text'], prompt_ids=f['prompt_ids'], body=body),
            dict(event='binding', response_id=rid, request_id=request_id, scheduler_id=1,
            sampling=dict(max_tokens=512, temperature=0.0, top_p=1.0, top_k=0, min_p=0.0,
                repetition_penalty=1.0, presence_penalty=0.0, frequency_penalty=0.0,
                xtc_probability=0.0, stop=[], stop_token_ids=[], thinking_budget=None))])
        tokens = [7] * 512 if length else [7, 99]
        committed = tokens if length else tokens[:-1]; finish = 'length' if length else 'stop'
        head = dict(id=rid, object='text_completion', created=1, model='Qwen3.8-27B-nvfp4')
        frames = [dict(**head, choices=[dict(index=0, text='a'*len(committed), finish_reason=finish)]),
                  dict(**head, choices=[], usage=dict(prompt_tokens=2, completion_tokens=len(committed),
                    total_tokens=2+len(committed), prompt_tokens_details={'cached_tokens': 0}))]
        raw = ''.join('data: ' + json.dumps(e) + '\n\n' for e in frames).encode() + b'data: [DONE]\n\n'
        row = dict(**job, status='done', http_status=200, done=True, error=None, validation_errors=[],
            qualified_output_tokens=None, raw_body_truncated=False, raw_body_bytes=len(raw),
            start_perf_counter_s=1.0, done_perf_counter_s=2.0,
            raw_sse_chunks=[dict(client_perf_counter_s=2.0, body_bytes_base64=base64.b64encode(raw).decode())])
        rows.append(row); streams[rid] = dict(event='wire', response_id=rid, complete=True,
            bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    for start in range(0, count, C):
        for n in range(512 if length else 2):
            last = n == (511 if length else 1); finish = ('length' if length else 'stop') if last else None
            token = 7 if length or not last else 99
            rids = [f'engine-{i}' for i in range(start, start+C)]
            events.append(dict(event='batch', scheduler_id=1,
                route=dict(max_num_seqs=C, vlm_mtp_drafter=False, vlm_mtp_active=False,
                    block_drafter=False, embedded_mtp=False, specprefill=False, turboquant=False),
                tokens=[dict(request_id=rid, token=token, finish_reason=finish,
                    prompt_ids=bodies[rid]['prompt_ids'], output_parser=False) for rid in rids],
                outputs=[dict(request_id=rid, finished=last, finish_reason=finish, error=None,
                    prompt_tokens=2, completion_tokens=n+1-(finish=='stop'), cached_tokens=0,
                    committed_ids=[7]*(n+1-(finish=='stop')) if last else None) for rid in rids]))
    events.extend(streams.values()); events.append(dict(event='close', restored=True, official_returned_normally=True))
    group = dict(schema='E023-http-group-v1', phase='qualification', concurrency=C, qualification=None,
        group_error=None, summary={'formal_point_eligible': False}, requests=rows,
        cleanup=dict(complete=True, deadline_exceeded=False, errors=[], pending_tasks=[], cancelled_again=False,
                     adapter='native-close-cancellation-v1'))
    return group, events, jobs, fixtures


class JoinTests(unittest.TestCase):
    def check(self, data):
        group, events, jobs, fixtures = data
        return j.join(group, lines(events), jobs, fixtures, lambda ids: 'a'*len(ids), PROBE)

    def test_stop_and_length_semantics_including_c8_repeated_prompts(self):
        for C, length in ((1, False), (8, False), (1, True)):
            result = self.check(fixture(C, length))
            self.assertTrue(result['protocol_route_count_qualified'])
            self.assertFalse(result['performance_qualified'] or result['full_state_numerical_qualified'])
            count = 8 if C == 1 else 16
            self.assertEqual(result['raw_output_count'], count*(512 if length else 2))
            self.assertEqual(result['committed_output_count'], count*(512 if length else 1))
            self.assertEqual(result['observed_terminal_eos'], 0 if length else count)

    def test_wrong_binding_tokens_cache_route_and_missing_termination_rejected(self):
        for name in ('prompt', 'binding', 'cache', 'route', 'sampling', 'count', 'committed', 'post-EOS', 'close', 'wire', 'error'):
            data = fixture(8); group, events, jobs, fixtures = data
            batch = next(e for e in events if e['event']=='batch')
            terminal = next(e for e in events if e['event']=='batch' and e['outputs'][0]['finished'])
            if name == 'prompt': batch['tokens'][0]['prompt_ids'] = [2, 3]
            if name == 'binding': next(e for e in events if e['event']=='binding')['response_id'] = 'cmpl-ffffffff'
            if name == 'cache': batch['outputs'][0]['cached_tokens'] = 1
            if name == 'route': batch['route']['embedded_mtp'] = True
            if name == 'sampling': next(e for e in events if e['event']=='binding')['sampling']['stop_token_ids'] = [99]
            if name == 'count': terminal['outputs'][0]['completion_tokens'] += 1
            if name == 'committed': terminal['outputs'][0]['committed_ids'] = [8]
            if name == 'post-EOS': events.insert(events.index(terminal)+1, copy.deepcopy(batch))
            if name == 'close': events.pop()
            if name == 'wire': next(e for e in events if e['event']=='wire')['sha256'] = '0'*64
            if name == 'error': batch['outputs'][0]['error'] = 'failure'
            with self.subTest(name=name), self.assertRaises(ValueError): self.check(data)

    def test_raw_sse_not_derived_summary_duplicate_done_and_extra_tail_rejected(self):
        for tail in (b'data: [DONE]\n\n', b'x', b'\n'):
            data=fixture(); row=data[0]['requests'][0]
            raw=base64.b64decode(row['raw_sse_chunks'][0]['body_bytes_base64']) + tail
            row['raw_sse_chunks'][0]['body_bytes_base64']=base64.b64encode(raw).decode(); row['raw_body_bytes']=len(raw)
            with self.subTest(tail=tail), self.assertRaises(ValueError): self.check(data)
        data=fixture();data[0]['requests'][0]['answer']='false derived answer'
        self.assertEqual(self.check(data)['status'],'PASS')

    def test_client_close_cancellation_never_qualifies(self):
        data=fixture(); data[0]['cleanup'].update(complete=False,errors=[{'type':'CancelledError'}])
        with self.assertRaises(ValueError):self.check(data)

    def test_wire_text_and_original_prompt_body_are_hard_gates(self):
        data=fixture();data[0]['requests'][0]=copy.deepcopy(data[0]['requests'][0])
        data[0]['requests'][0]['body']['prompt']='other'
        with self.assertRaises(ValueError):self.check(data)
        group, events, jobs, fixtures=fixture()
        with self.assertRaisesRegex(ValueError,'committed-ID decode'):
            j.join(group,lines(events),jobs,fixtures,lambda ids:'wrong',PROBE)


class ObserverTests(unittest.IsolatedAsyncioTestCase):
    async def test_output_tap_preserves_original_arguments_result_and_terminal_eos(self):
        events=[];calls=[]
        class Core:
            async def add_request(self,prompt,sampling_params=None,request_id=None,**kwargs):return 'unused'
        result=([NS(request_id='engine-1',finished=True,finish_reason='stop',error=None,
                    prompt_tokens=2,completion_tokens=1,cached_tokens=0,output_token_ids=[7])],{'engine-1'})
        class Scheduler:
            uid_to_request_id={4:'engine-1'}
            running={'engine-1':NS(prompt_token_ids=[1,2])}
            _output_parser_factory=None
            def _process_batch_responses(self,responses):
                calls.append(responses);return result
        async def original(engine,prompt,request,**kwargs):yield ''
        server=NS(stream_completion=original);scheduler=Scheduler()
        responses=[NS(uid=4,token=99,finish_reason='stop')]
        with p.Observer(server,Core,Scheduler,lambda _: {'captured':True},events.append,1):
            self.assertIs(scheduler._process_batch_responses(responses),result)
        self.assertIs(calls[0],responses)
        self.assertEqual(events[0]['tokens'][0]['token'],99)
        self.assertEqual(events[0]['outputs'][0]['committed_ids'],[7])
        self.assertEqual(responses[0].token,99)

    async def test_done_then_client_close_is_not_mistaken_for_missing_protocol_terminal(self):
        events=[]
        class Core:
            async def add_request(self,prompt,sampling_params=None,request_id=None,**kwargs):return 'unused'
        class Scheduler:
            def _process_batch_responses(self,responses):return responses
        async def original(engine,prompt,request,model_load_duration=0.0,prompt_token_ids=None,
                           resolved_model=None,response_id=None,inference_request_id=None):
            yield 'data: [DONE]\n\n'
        server=NS(stream_completion=original);request=NS(model_dump=lambda **kwargs:{})
        observer=p.Observer(server,Core,Scheduler,lambda _: {},events.append,1)
        with observer:
            stream=server.stream_completion(None,'prompt',request,prompt_token_ids=[1],response_id='cmpl-00000001')
            self.assertEqual(await anext(stream),'data: [DONE]\n\n')
            await stream.aclose()
        receipt=next(x for x in events if x['event']=='wire')
        self.assertTrue(receipt['complete']);self.assertFalse(receipt['iterator_exhausted'])
        self.assertIsNone(observer.current.get())

    async def test_real_contextvar_interleaving_preserves_ids_bytes_and_restores_seams(self):
        events=[];scheduler=NS()
        class Core:
            def __init__(self):self.engine=NS(scheduler=scheduler)
            async def add_request(self,prompt,sampling_params=None,request_id=None,**kwargs):
                await asyncio.sleep(0)
                return 'engine-'+prompt
        class Scheduler:
            def _process_batch_responses(self,responses):return responses
        core=Core();params=NS(**dict(max_tokens=512,temperature=0.0,top_p=1.0,top_k=0,min_p=0.0,
            repetition_penalty=1.0,presence_penalty=0.0,frequency_penalty=0.0,xtc_probability=0.0,
            stop=[],stop_token_ids=[],thinking_budget=None))
        async def original(engine,prompt,request,model_load_duration=0.0,prompt_token_ids=None,
                           resolved_model=None,response_id=None,inference_request_id=None):
            actual=await engine.add_request(prompt,sampling_params=params)
            yield actual+'\n'
            await asyncio.sleep(0)
            yield 'DONE '+response_id
        server=NS(stream_completion=original); originals=(original,Core.add_request,Scheduler._process_batch_responses)
        request=NS(model_dump=lambda **kwargs: {'max_tokens':512})
        observer=p.Observer(server,Core,Scheduler,lambda _: {},events.append,8)
        async def consume(prompt,rid):
            chunks=[]
            async for chunk in server.stream_completion(core,prompt,request,prompt_token_ids=[1],response_id=rid):
                self.assertIsNone(observer.current.get())
                chunks.append(chunk)
            return chunks
        with observer:
            left,right=await asyncio.gather(consume('left','cmpl-00000001'),consume('right','cmpl-00000002'))
        self.assertEqual(left,['engine-left\n','DONE cmpl-00000001'])
        self.assertEqual(right,['engine-right\n','DONE cmpl-00000002'])
        self.assertEqual({x['request_id']:x['response_id'] for x in events if x['event']=='binding'},
                         {'engine-left':'cmpl-00000001','engine-right':'cmpl-00000002'})
        self.assertEqual((server.stream_completion,Core.add_request,Scheduler._process_batch_responses),originals)
        self.assertIsNone(observer.current.get())

    async def test_exceptions_restore_global_seams_and_unbound_request_is_rejected(self):
        class Core:
            async def add_request(self,prompt,sampling_params=None,request_id=None,**kwargs):
                raise AssertionError('unbound call must not reach original')
        class Scheduler:
            def _process_batch_responses(self,responses):return responses
        async def original(engine,prompt,request,**kwargs):
            raise RuntimeError('original failure')
            yield ''
        server=NS(stream_completion=original);before=Core.add_request
        with self.assertRaisesRegex(ValueError,'unbound engine request'):
            with p.Observer(server,Core,Scheduler,lambda _: {},lambda _:None,1):
                await Core().add_request('bad')
        self.assertIs(Core.add_request,before);self.assertIs(server.stream_completion,original)


class ImportTests(unittest.TestCase):
    def test_pure_spec_keeps_original_settings_and_disables_observer_for_formal(self):
        left=p.specification('/tmp/omlx-probe-cpu-unused',8,18024,'cpu-key')
        right=p.specification('/tmp/omlx-probe-cpu-unused',8,18024,'cpu-key',observed=False)
        self.assertEqual(left['files'],right['files']);self.assertEqual(left['symlinks'],right['symlinks'])
        self.assertIn('--run',left['command']);self.assertNotIn('--run',right['command'])
        self.assertTrue(right['command'][0].endswith('/bin/omlx'))
        settings=next(v for k,v in left['files'].items() if k.endswith('/settings.json'))
        self.assertEqual(settings['scheduler']['max_concurrent_requests'],8)
        models=next(v for k,v in left['files'].items() if k.endswith('/model_settings.json'))
        self.assertFalse(any(models['models']['Qwen3.8-27B-nvfp4'][key]
            for key in ('mtp_enabled','vlm_mtp_enabled','dflash_enabled','specprefill_enabled','turboquant_kv_enabled')))

    def test_help_and_import_do_not_import_runtime(self):
        result=subprocess.run([sys.executable,'-B',str(Path(p.__file__)),'--help'],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertFalse(any(name=='mlx' or name.startswith(('mlx.','omlx.')) for name in sys.modules))


if __name__=='__main__':unittest.main()
