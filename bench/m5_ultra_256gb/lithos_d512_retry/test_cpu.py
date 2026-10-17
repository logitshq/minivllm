"""Explicit CPU fakes for tap ownership and raw joins, never device qualification."""
import base64
import ast
import copy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
import unittest
import uuid
from datetime import datetime, timezone

from bench.m5_ultra_256gb.lithos_d512_retry import serve_probe as p, join as j

PROBE = 'a'*64
PACK = '/CPU/pinned-existing-pack'
ARGS = ['CPU-explicit-registered-arguments']


def lines(events):
    return [(json.dumps(e)+'\n').encode() for e in events]


def route():
    return dict(device=dict(name='Apple M5 Ultra', gpu_cores=80, apple_family=10), backend='m5_ultra_80c',
        pack_dir=PACK, capacity=2048, has_drafter=False, verify='cost', verify_length=None,
        decoder_kernel_config=None, sampling=None, seed=0, accelerator='on', prefill_chunk_size=128,
        effective_prefill_exact=False, prefix_cache=True, eos=[248046,248044], config_eos=[248046,248044],
        programs=[dict(key='prefill.128',backend='m5_ultra_80c',capacity=2048)])


def fixture(C=1, mode='stop'):
    fixtures = {f'f{i}':dict(fixture_id=f'f{i}',messages=[dict(role='user',content=f'prompt{i}')],
        prompt_text=f'<|im_start|>user\nprompt{i}<|im_end|>\n<|im_start|>assistant\n',prompt_ids=[1,i+10],
        normal_eos=True,max_output_tokens=512,eos_token_ids=[248046,248044]) for i in range(8)}
    jobs = p.jobs_for(list(fixtures.values()), C, 'CPU-model'); count = len(jobs)
    events = [dict(event='open',schema=p.SCHEMA,concurrency=C,source_root=str(p.ROOT),sources=p.PINS,
                   probe_sha256=PROBE,official_args=ARGS,qualification_only=True)]
    rows, wire_events = [], []
    for i,job in enumerate(jobs,1):
        rawids = [7,248046] if mode=='stop' else [7]*512 if mode=='length' else [7]*511+[248046]
        committed = rawids if mode=='length' else rawids[:-1]; text = 'a'*len(committed)
        finish = 'length' if mode=='length' else 'stop'; wid = f'{i:032x}'
        gen = dict(tokens=rawids,cached_prompt_tokens=0,setup_ms=1.25 if i==1 else 0.0,
                   accepted=None,committed=None,verify_len=None,confidences=None)
        metrics = dict(cached_prompt_tokens=0,setup_ms=gen['setup_ms'],verify_tokens=1,
                       load_ms=2.0,prefill_wall_ms=3.0,decode_wall_ms=4.0)
        body = dict(model='CPU-model',created_at=f'CPU-time-{i}',message=dict(role='assistant',content=text),
            done=True,done_reason=finish,total_duration=i*100000000,load_duration=int((2+gen['setup_ms'])*1e6),
            prompt_eval_count=2,prompt_eval_duration=3000000,eval_count=len(rawids),eval_duration=4000000)
        final = {**body,'message':dict(role='assistant',content='')}; final_line = json.dumps(final)+'\n'
        events += [dict(event='generation_start',index=i,prompt_ids=fixtures[job['fixture_id']]['prompt_ids'],
                        cap=512,streaming=True,cache_prefix_tokens=3),
                   dict(event='generation_return',index=i,generation=gen,route=route()),
                   dict(event='body',index=i,wire_id=wid,metrics=metrics,body=body)]
        wire_events.append(dict(event='wire',wire_id=wid,line=final_line))
        raw = (json.dumps(dict(model='CPU-model',created_at=f'CPU-delta-{i}',
                message=dict(role='assistant',content=text),done=False))+'\n'+final_line).encode()
        request = json.dumps(job['body']).encode()
        start = 1.0 + 2*((i-1)//C); done = start+1.0
        rows.append(dict(**copy.deepcopy(job),status='done',http_status=200,error=None,drain_error=None,
            stream_eof=True,raw_body_truncated=False,raw_body_bytes=len(raw),start_perf_counter_s=start,
            done_perf_counter_s=done,eof_perf_counter_s=done+.1,
            request_body_bytes_base64=base64.b64encode(request).decode(),request_body_bytes_sha256=hashlib.sha256(request).hexdigest(),
            raw_ndjson_chunks=[dict(client_perf_counter_s=done,body_bytes_base64=base64.b64encode(raw).decode())]))
    events += list(reversed(wire_events))  # Finish delivery order is NOT generation order.
    events += [dict(event='sealed',status='COMPLETE',source_pins_rechecked=True,
                    started=count,returned=count,bodies=count,terminals=count,active=0)]
    group = dict(schema='E023-lithos-ndjson-group-v1',concurrency=C,phase='qualification',qualification=None,
        group_error=None,summary={'formal_point_eligible':False},requests=rows,
        cleanup=dict(complete=True,deadline_exceeded=False,errors=[],pending_tasks=[],cancelled_again=False,client_close_started=True))
    return group,events,jobs,fixtures


class JoinTests(unittest.TestCase):
    def check(self,data):
        group,events,jobs,fixtures = data
        return j.join(group,lines(events),jobs,fixtures,lambda ids:''.join('a' if t==7 else '?' for t in ids),PROBE,
                      expected_official_args=ARGS,expected_pack_dir=PACK)

    def test_c1_c8_and_terminal_eos_at_cap_with_reversed_finish_order(self):
        for C,mode in ((1,'stop'),(8,'stop'),(1,'length'),(8,'eos-at-cap')):
            with self.subTest(C=C,mode=mode):
                out=self.check(fixture(C,mode));self.assertEqual(out['status'],'PASS')
                self.assertFalse(out['performance_qualified'] or out['full_state_qualified'])
                self.assertEqual(len(out['requests']),8 if C==1 else 16)
                self.assertEqual(out['max_http_outstanding'],C)
                self.assertEqual(out['requests'][0]['generation_index'],1)
                self.assertEqual(out['requests'][0]['setup_ms'],1.25) # qualification may include first setup

    def test_original_messages_not_rendered_prompt_and_no_mutable_body_alias(self):
        for C in (1,8):
            group,events,jobs,fixtures=fixture(C)
            for job in jobs:
                f=fixtures[job['fixture_id']]
                self.assertEqual(job['body']['messages'],f['messages'])
                self.assertNotEqual(job['body']['messages'][0]['content'],f['prompt_text'])
                self.assertIsNot(job['body']['messages'],f['messages'])
                self.assertIsNot(job['body']['messages'][0],f['messages'][0])
            self.assertEqual(self.check((group,events,jobs,fixtures))['status'],'PASS')
            before=copy.deepcopy(fixtures)
            jobs[0]['body']['messages'][0]['content']='caller mutated body'
            self.assertEqual(fixtures,before)

    def test_double_template_body_rejected_even_when_sent_body_sha_matches(self):
        data=fixture();group,_,jobs,fixtures=data
        body=jobs[0]['body'];body['messages']=[dict(role='user',content=fixtures[jobs[0]['fixture_id']]['prompt_text'])]
        row=group['requests'][0];row['body']=copy.deepcopy(body)
        raw=json.dumps(body).encode();row['request_body_bytes_base64']=base64.b64encode(raw).decode()
        row['request_body_bytes_sha256']=hashlib.sha256(raw).hexdigest()
        with self.assertRaisesRegex(ValueError,'fixed original request body'):self.check(data)

    def test_original_messages_do_not_relax_full_prompt_id_oracle(self):
        data=fixture();fixed=copy.deepcopy(data[3]);data[1][1]['prompt_ids']=[1,10,999]
        with self.assertRaisesRegex(ValueError,'actual full prompt differs'):self.check(data)
        self.assertEqual(data[3],fixed)

    def test_wrong_prompt_route_cache_ids_eos_body_finish_and_sources_rejected(self):
        for name in ('prompt','route','cache','tokens','post-EOS','cap','body','finish','source','pack','spec','sampling','close'):
            data=fixture();group,events,jobs,_=data
            first=events[1];returned=events[2]
            if name=='prompt':first['prompt_ids']=[1,999]
            if name=='route':returned['route']['backend']='m4_max_40c'
            if name=='cache':returned['generation']['cached_prompt_tokens']=128
            if name=='tokens':returned['generation']['tokens']=[8,248046]
            if name=='post-EOS':returned['generation']['tokens']=[248046,7]
            if name=='cap':first['cap']=511
            if name=='body':group['requests'][0]['body']['options']['num_predict']=511
            if name=='finish':next(e for e in events if e['event']=='wire')['line']='{}\n'
            if name=='source':events[0]['probe_sha256']='b'*64
            if name=='pack':returned['route']['pack_dir']='/other'
            if name=='spec':returned['route']['has_drafter']=True
            if name=='sampling':returned['route']['sampling']=[1,0,1,0]
            if name=='close':events.pop()
            with self.subTest(name=name),self.assertRaises(ValueError):self.check(data)

    def test_no_terminal_line_order_guess_and_no_derived_raw_ids(self):
        data=fixture();data[0]['requests'][0]['answer']='ignored derived summary'
        self.assertEqual(self.check(data)['status'],'PASS')
        data=fixture();a,b=data[0]['requests'][:2]
        a['raw_ndjson_chunks']=copy.deepcopy(b['raw_ndjson_chunks']);a['raw_body_bytes']=b['raw_body_bytes']
        with self.assertRaises(ValueError):self.check(data)

    def test_original_sent_body_sha_and_tail_clock_cleanup_are_hard_gates(self):
        for name in ('sentbody','tail','clock','eof','close'):
            data=fixture();row=data[0]['requests'][0]
            if name=='sentbody':row['request_body_bytes_sha256']='0'*64
            if name=='tail':
                raw=base64.b64decode(row['raw_ndjson_chunks'][0]['body_bytes_base64'])+b'x'
                row['raw_body_bytes']=len(raw);row['raw_ndjson_chunks'][0]['body_bytes_base64']=base64.b64encode(raw).decode()
            if name=='clock':row['done_perf_counter_s']=1.9
            if name=='eof':row['stream_eof']=False
            if name=='close':data[0]['cleanup']['complete']=False
            with self.subTest(name=name),self.assertRaises(ValueError):self.check(data)

    def test_actual_http_intervals_reject_serial_mislabeled_c8_and_excess_overlap(self):
        for mode in ('serial-C8','sixteen-C8','overlap-C1'):
            C=1 if mode=='overlap-C1' else 8
            data=fixture(C)
            for i,row in enumerate(data[0]['requests']):
                start=1.0+2*i if mode=='serial-C8' else 1.0
                row.update(start_perf_counter_s=start,done_perf_counter_s=start+1,eof_perf_counter_s=start+1.1)
                for chunk in row['raw_ndjson_chunks']:chunk['client_perf_counter_s']=start+1
            with self.subTest(mode=mode),self.assertRaisesRegex(ValueError,'actual HTTP outstanding peak'):
                self.check(data)

    def test_equal_end_start_is_half_open_and_zero_duration_is_rejected(self):
        data=fixture(8)
        for i,row in enumerate(data[0]['requests']):
            start=1.0+i//8
            row.update(start_perf_counter_s=start,done_perf_counter_s=start+1,eof_perf_counter_s=start+1.1)
            for chunk in row['raw_ndjson_chunks']:chunk['client_perf_counter_s']=start+1
        self.assertEqual(self.check(data)['max_http_outstanding'],8)
        row=data[0]['requests'][0];row['start_perf_counter_s']=row['done_perf_counter_s']
        with self.assertRaisesRegex(ValueError,'positive finite POST-to-DONE interval'):self.check(data)

    def test_decode_from_actual_ids_not_retokenization(self):
        group,events,jobs,fixtures=fixture()
        with self.assertRaisesRegex(ValueError,'decoded text'):
            j.join(group,lines(events),jobs,fixtures,lambda ids:'wrong',PROBE,
                   expected_official_args=ARGS,expected_pack_dir=PACK)


@dataclass
class Gen:
    tokens:list
    cached_prompt_tokens:int=0
    setup_ms:float=0.0


class ObserverTests(unittest.TestCase):
    def test_actual_pinned_wire_body_finish_through_taps_and_join(self):
        # Execute only the actual official WireResponse/string helpers, never
        # the package imports, server, tokenizer or native runtime.
        path=p.ROOT/'monolith/serving/events.py';raw=path.read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(),p.PINS['monolith/serving/events.py'])
        tree=ast.parse(raw);names={'sse','ndjson','ollama_time','WireResponse'}
        tree.body=[n for n in tree.body if getattr(n,'name',None) in names]
        ns=dict(json=json,time=time,uuid=uuid,datetime=datetime,timezone=timezone,APIError=ValueError)
        exec(compile(tree,str(path),'exec'),ns)
        Wire=ns['WireResponse']
        group,_,jobs,fixtures=fixture()
        events=[dict(event='open',schema=p.SCHEMA,concurrency=1,source_root=str(p.ROOT),sources=p.PINS,
                     probe_sha256=PROBE,official_args=ARGS,qualification_only=True)]
        class Session:
            def generate(self,prompt_ids,max_new_tokens,**kwargs):return Gen([7,248046])
        checked=[]
        with p.Observer(Session,Wire,events.append,lambda _:route(),expected_count=8,
                        source_check=lambda:checked.append(True)) as observer:
            pending=[]
            for i,job in enumerate(jobs):
                Session().generate(fixtures[job['fixture_id']]['prompt_ids'],512,on_tokens=lambda _:None,cache_prefix_tokens=3)
                wire=Wire('ollama','CPU-model');wire.metrics=dict(cached_prompt_tokens=0,setup_ms=0.0,verify_tokens=1,
                    load_ms=0.0,prefill_wall_ms=0.0,decode_wall_ms=0.0)
                body=wire.body(dict(role='assistant',content='a'),'stop',2,2)
                pending.append((i,wire,body))
            for i,wire,body in reversed(pending):
                raw=(''.join(wire.delta('a'))+''.join(wire.finish(body,False))).encode()
                row=group['requests'][i];row['raw_body_bytes']=len(raw)
                row['raw_ndjson_chunks']=[dict(client_perf_counter_s=row['done_perf_counter_s'],body_bytes_base64=base64.b64encode(raw).decode())]
            self.assertTrue(observer.sealed);self.assertEqual(checked,[True])
        result=j.join(group,lines(events),jobs,fixtures,lambda ids:'a'*len(ids),PROBE,
                      expected_official_args=ARGS,expected_pack_dir=PACK)
        self.assertEqual(result['observed_eos'],8)

    def test_same_objects_arguments_and_reversed_wire_finish(self):
        events=[];calls=[];generations=[]
        class Session:
            def generate(self,prompt_ids,max_new_tokens,**kwargs):
                calls.append((prompt_ids,max_new_tokens,kwargs));gen=Gen([prompt_ids[0],248046]);generations.append(gen);return gen
        class Wire:
            protocol='ollama'
            def __init__(self,wid):self.id=wid;self.metrics={}
            def body(self,message,finish,prompt_tokens,completion_tokens):return dict(message=message,done=True)
            def finish(self,body,include_usage):yield json.dumps(body)+'\n'
        originals=Session.generate,Wire.body,Wire.finish
        with p.Observer(Session,Wire,events.append,lambda _:route()) as observer:
            outputs=[]
            for i in range(2):
                x=[i+1];callback=lambda _:None
                actual=Session().generate(x,512,on_tokens=callback)
                self.assertIs(actual,generations[-1]);self.assertIs(calls[-1][0],x)
                self.assertIs(calls[-1][2]['on_tokens'],callback)
                wire=Wire(f'{i+1:032x}');body=wire.body({'content':'a'},'stop',1,2);outputs.append((wire,body))
            for wire,body in reversed(outputs):
                self.assertEqual(list(wire.finish(body,False)),[json.dumps(body)+'\n'])
            self.assertEqual(observer.counts(),dict(started=2,returned=2,bodies=2,terminals=2,active=0))
        self.assertEqual((Session.generate,Wire.body,Wire.finish),originals)
        self.assertEqual([r['index'] for r in events if r['event']=='body'],[1,2])
        self.assertEqual([r['wire_id'] for r in events if r['event']=='wire'],[f'{2:032x}',f'{1:032x}'])

    def test_thread_local_pending_survives_another_worker_and_exception_restores(self):
        events=[];ready=threading.Event();resume=threading.Event();errors=[]
        class Session:
            def generate(self,prompt_ids,max_new_tokens,**kwargs):return Gen(prompt_ids)
        class Wire:
            protocol='ollama';metrics={}
            def __init__(self,wid):self.id=wid
            def body(self,*args):return {'done':True}
            def finish(self,*args):yield '{"done": true}\n'
        old=Session.generate
        def worker():
            try:
                Session().generate([1],512);ready.set();resume.wait(2)
                Wire('first').body({},'stop',1,1)
            except BaseException as exc:errors.append(exc)
        with self.assertRaisesRegex(RuntimeError,'CPU scope failure'):
            with p.Observer(Session,Wire,events.append,lambda _:route()):
                t=threading.Thread(target=worker);t.start();self.assertTrue(ready.wait(2))
                Session().generate([2],512);Wire('second').body({},'stop',1,1)
                resume.set();t.join(2);self.assertFalse(t.is_alive());self.assertFalse(errors)
                raise RuntimeError('CPU scope failure')
        self.assertIs(Session.generate,old)
        self.assertEqual({r['wire_id']:r['index'] for r in events if r['event']=='body'}, {'first':1,'second':2})


class LogAndCLITests(unittest.TestCase):
    def test_quiet_phase_logs_allow_warm_setup_but_require_formal_zero(self):
        row={'cached_prompt_tokens':0,'setup_ms':1.0,'verify_tokens':1}
        raw=('INFO:__main__:Generation: '+repr(row)+'\n').encode()
        self.assertFalse(j.generation_log_window(raw,1,require_setup_zero=False)['all_setup_ms_zero'])
        with self.assertRaises(ValueError):j.generation_log_window(raw,1,require_setup_zero=True)
        row['setup_ms']=0.0;raw=('INFO:__main__:Generation: '+repr(row)+'\n').encode()*64
        out=j.generation_log_window(raw,64,require_setup_zero=True)
        self.assertFalse(out['per_wire_request_id_bound']);self.assertEqual(out['count'],64)

    def test_bad_logs_missing_extra_cache_truncation_and_duplicate_keys_rejected(self):
        good="INFO:__main__:Generation: {'cached_prompt_tokens': 0, 'setup_ms': 0.0, 'verify_tokens': 1}\n"
        for raw in (good[:-1],good+good,good.replace("tokens': 0","tokens': 1"),
                    good.replace('INFO:','WARNING:'),good.replace("'setup_ms':", "'cached_prompt_tokens': 0, 'setup_ms':")):
            with self.subTest(raw=raw),self.assertRaises(ValueError):j.generation_log_window(raw.encode(),1,require_setup_zero=True)

    def test_saved_official_e020_main_logger_and_imported_probe_are_not_mixed(self):
        raw=(Path(__file__).resolve().parent.parent/'lithos_d512/e020-generation-fixture.log').read_bytes()
        self.assertEqual(j.generation_log_window(raw,1,require_setup_zero=False)['status'],'PASS')
        with self.assertRaises(ValueError):j.generation_log_window(raw,1,require_setup_zero=True)
        probe=raw.replace(b'INFO:__main__:',b'INFO:monolith.serve:',1)
        with self.assertRaises(ValueError):j.generation_log_window(probe,1,require_setup_zero=False)
        self.assertEqual(j.generation_log_window(probe,1,require_setup_zero=False,
                         expected_logger='monolith.serve')['status'],'PASS')

    def test_help_and_default_have_no_runtime_or_network_side_effects(self):
        script="""
import importlib.abc,runpy,sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,name,*args):
  if name.split('.')[0] in {'mlx','numpy','torch','monolith','transformers','uvicorn','httpx'}:raise AssertionError(name)
sys.meta_path.insert(0,Block())
sys.argv=[sys.argv[1],*sys.argv[2:]]
runpy.run_path(sys.argv[0],run_name='__main__')
"""
        for args in ([],['--help']):
            r=subprocess.run([sys.executable,'-I','-B','-c',script,str(Path(p.__file__).resolve()),*args],
                             capture_output=True,text=True,timeout=5)
            self.assertEqual(r.returncode,0,r.stderr)

    def test_formal_specification_has_original_cli_and_no_observer(self):
        spec=p.specification('/CPU/out',8,18024,'CPU-key',observed=False)
        self.assertEqual(spec['command'][1:4],['-B','-m','monolith.serve'])
        self.assertNotIn(str(Path(p.__file__).resolve()),spec['command'])
        self.assertIn('--no-draft',spec['command']);self.assertFalse(spec['observer_enabled'])


if __name__=='__main__':
    unittest.main()
