"""Synthetic raw protocol/trace corruption tests; no socket/model/tokenizer."""
import base64
import copy
import importlib.util
import json
from pathlib import Path
import unittest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('native_http_join_cpu',HERE/'http_join.py')
h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
A='123456789abc4def8123456789abcdef';B='23456789abcd4efa923456789abcdef0'


def decode(ids):return ''.join('答'+str(t) for t in ids)
def wire(value):return ('data: '+(value if isinstance(value,str) else json.dumps(value,ensure_ascii=False))+'\n\n').encode()
def lines(events):return ((json.dumps(e)+'\n').encode() for e in events)


def fixture(C=8):
    identity=dict(configuration=dict(max_seqs=C,max_len=2048,max_batched_tokens=2048),eos_ids=[99],
                  step_observer=True,qualification_trace_enabled=True,per_token_observer=False,drafter=None,normal_eos=True,prefix_cache=False)
    fixtures={};jobs=[];rows=[];terminals=[]
    for i,(rid,raw) in enumerate(((A,[10,99]),(B,[20,21,99]))):
        f=dict(fixture_id='f'+str(i),native_path='/v1/completions',prompt_text='prompt'+str(i),prompt_ids=[i+1,i+2],
               max_output_tokens=3,normal_eos=True,eos_token_ids=[99]);fixtures[f['fixture_id']]=f
        body=dict(model='native-target-only',prompt=f['prompt_text'],max_tokens=3,temperature=0,top_p=1.0,repetition_penalty=1.0,
                  stream=True,stream_options={'include_usage':True},n=1)
        job=dict(request_id='client-'+str(i),fixture_id=f['fixture_id'],path='/v1/completions',body=body);jobs.append(job)
        terminal=dict(id='cmpl-'+rid,choices=[dict(index=0,text='',finish_reason='stop')],
            usage=dict(prompt_tokens=2,completion_tokens=len(raw)-1,total_tokens=len(raw)+1,prompt_tokens_details={'cached_tokens':0}),
            benchmark_request_id=rid,benchmark_prompt_ids=f['prompt_ids'],benchmark_output_ids=raw,
            benchmark_committed_output_ids=raw[:-1],benchmark_terminal_eos_id=99)
        terminals.append(terminal)
        content=wire(dict(id='cmpl-'+rid,choices=[dict(index=0,text=decode(raw[:-1]),finish_reason=None)]))+wire(terminal)+wire('[DONE]')
        split=content.index('答'.encode())+1;chunks=[content[:split],content[split:-4],content[-4:]]
        rows.append({**copy.deepcopy(job),'status':'done','http_status':200,'error':None,'validation_errors':[],
                     'qualified_output_tokens':None,'raw_sse_chunks':[dict(client_perf_counter_s=2+k,body_bytes_base64=base64.b64encode(b).decode()) for k,b in enumerate(chunks)],
                     'raw_body_truncated':False,'raw_body_bytes':len(content),'start_perf_counter_s':1,'done_perf_counter_s':4,
                     'answer':'derived text deliberately wrong','native_fields':{'benchmark_output_ids':[777]},'usage':{'completion_tokens':999}})
    group=dict(schema='E023-http-group-v1',phase='qualification',qualification=None,group_error=None,summary={'formal_point_eligible':False},
               cleanup=dict(complete=True,deadline_exceeded=False,errors=[],pending_tasks=[],cancelled_again=False),concurrency=C,requests=rows)
    events=[dict(event='startup',identity=copy.deepcopy(identity))]
    def step(arrivals,running,waiting,decode_rows,prefill,finished,results):
        retired=[r for r,t,d in results if d]
        events.extend([dict(event='before_step',newly_submitted_to_scheduler=arrivals,scheduler_running=running,scheduler_waiting=waiting),
          dict(event='actual_scheduler_output',decode=decode_rows,prefill=[dict(rid=r,tokens=n) for r,n in prefill],finished_ids=finished),
          dict(event='actual_model_runner_batch',decode_rows=decode_rows),
          dict(event='after_step',results=results,retired=retired,surviving_decode_rows=[r for r in decode_rows if r not in retired])])
    if C==1:
        step([A],[],[A],[],[[A,2]],[],[[A,10,False]])
        step([],[A],[],[A],[],[],[[A,99,True]])
        step([B],[],[B],[],[[B,2]],[A],[[B,20,False]])
        step([],[B],[],[B],[],[],[[B,21,False]])
        step([],[B],[],[B],[],[],[[B,99,True]])
    else:
        step([A,B],[],[A,B],[],[[A,2],[B,2]],[],[[A,10,False],[B,20,False]])
        step([],[A,B],[],[A,B],[],[],[[A,99,True],[B,21,False]])
        step([],[B],[],[B],[],[A],[[B,99,True]])
    return group,events,jobs,fixtures,identity,terminals


def replace_wire(row,terminal,*,rid=None,suffix=None,events=None):
    rid=rid or terminal['benchmark_request_id']
    payload=(wire(dict(id='cmpl-'+rid,choices=[dict(index=0,text=decode(terminal['benchmark_committed_output_ids']),finish_reason=None)]))
             +wire(terminal)+wire('[DONE]')) if events is None else b''.join(wire(e) for e in events)
    if suffix is not None:payload+=suffix
    row['raw_sse_chunks']=[dict(client_perf_counter_s=4,body_bytes_base64=base64.b64encode(payload).decode())]
    row['raw_body_bytes']=len(payload)


class Tests(unittest.TestCase):
    def run_join(self,data):
        group,events,jobs,fixtures,identity,_=data
        return h.join(group,lines(events),jobs,fixtures,identity,decode)

    def test_c1_c8_records_use_raw_fields_and_do_not_modify_evidence(self):
        for C in (1,8):
            data=fixture(C);before=copy.deepcopy(data);result=self.run_join(data)
            self.assertEqual(data,before)
            self.assertEqual(result['schema'],'E023-recorded-http-v1')
            self.assertEqual(result['jobs'][A]['raw_ids'],[10,99])
            self.assertEqual(result['jobs'][A]['committed_ids'],[10])
            self.assertEqual(result['jobs'][B]['client_request_id'],'client-1')
            self.assertEqual(result['eos_token_ids'],[99])
            self.assertFalse(result['engine_qualified'] or result['performance_qualified'])
            self.assertEqual(result['histogram'],{'0':2,'1':3} if C==1 else {'0':1,'2':1,'1':1})
            self.assertEqual(result['trace_lines'],1+4*len(result['steps']))

    def test_group_phase_receipt_cleanup_and_identity_are_hard_gates(self):
        for key in ('formal','receipt','formalpoint','cleanup','cancel','identity','C','observer','prequalified'):
            data=fixture();g,e,_,_,identity,_=data
            if key=='formal':g['phase']='formal'
            if key=='receipt':g['qualification']={'status':'qualified'}
            if key=='formalpoint':g['summary']['formal_point_eligible']=True
            if key=='cleanup':g['cleanup']['complete']=False
            if key=='cancel':g['group_error']={'type':'CancelledError'}
            if key=='identity':e[0]['identity']['configuration']['max_len']=1024
            if key=='C':g['concurrency']=1
            if key=='observer':identity['step_observer']=False
            if key=='prequalified':g['requests'][0]['qualified_output_tokens']=1
            with self.subTest(key=key),self.assertRaises(ValueError):self.run_join(data)

    def test_full_same_length_prompt_body_and_uuid_bijection(self):
        for key in ('prompt-ids','prompt-text','path','body','client-id','duplicate-rid','wrong-sse-id'):
            data=fixture();g,e,j,f,identity,t=data
            if key=='prompt-ids':t[0]['benchmark_prompt_ids']=[9,8];replace_wire(g['requests'][0],t[0])
            if key=='prompt-text':g['requests'][0]['body']['prompt']='same-length-wrong'
            if key=='path':g['requests'][0]['path']='/v1/chat/completions'
            if key=='body':g['requests'][0]['body']['temperature']=1
            if key=='client-id':g['requests'][0]['request_id']='client-1'
            if key=='duplicate-rid':t[1]['id']='cmpl-'+A;t[1]['benchmark_request_id']=A;replace_wire(g['requests'][1],t[1])
            if key=='wrong-sse-id':t[0]['id']='cmpl-'+B;replace_wire(g['requests'][0],t[0])
            with self.subTest(key=key),self.assertRaises(ValueError):self.run_join(data)

    def test_terminal_counts_eos_cap_cache_and_exact_decoded_text(self):
        for key in ('committed','cache','usage','eos','early','finish','text','bool-token','total'):
            data=fixture();g,_,_,_,_,ts=data;t=ts[0]
            if key=='committed':t['benchmark_committed_output_ids']=[99]
            if key=='cache':t['usage']['prompt_tokens_details']['cached_tokens']=1
            if key=='usage':t['usage']['completion_tokens']=True
            if key=='eos':t['benchmark_output_ids']=[99,10]
            if key=='early':t['benchmark_output_ids']=[10,11]
            if key=='finish':t['choices'][0]['finish_reason']='length'
            if key=='bool-token':t['benchmark_output_ids']=[True,99]
            if key=='total':t['usage']['total_tokens']=999
            replace_wire(g['requests'][0],t)
            if key=='text':
                chunk=g['requests'][0]['raw_sse_chunks'][0];raw=base64.b64decode(chunk['body_bytes_base64']).replace('答10'.encode(),b'wrong')
                chunk['body_bytes_base64']=base64.b64encode(raw).decode();g['requests'][0]['raw_body_bytes']=len(raw)
            with self.subTest(key=key),self.assertRaises(ValueError):self.run_join(data)

    def test_immediate_eos_empty_text_and_exact_cap_are_both_preserved(self):
        data=fixture(1);g,e,_,_,_,ts=data
        first,last=ts
        first.update(benchmark_output_ids=[99],benchmark_committed_output_ids=[])
        first['usage'].update(completion_tokens=0,total_tokens=2)
        replace_wire(g['requests'][0],first,events=[first,'[DONE]'])
        e[4].update(results=[[A,99,True]],retired=[A]);del e[5:9]
        last.update(benchmark_output_ids=[20,21,22],benchmark_committed_output_ids=[20,21,22],benchmark_terminal_eos_id=None)
        last['choices'][0]['finish_reason']='length';last['usage'].update(completion_tokens=3,total_tokens=5)
        replace_wire(g['requests'][1],last);e[-1]['results']=[[B,22,True]]
        result=self.run_join(data)
        self.assertEqual(result['jobs'][A]['committed_ids'],[])
        self.assertEqual(result['jobs'][B]['committed_ids'],[20,21,22])

    def test_missing_duplicate_terminal_done_and_all_retained_tails(self):
        for key in ('duplicate-terminal','duplicate-DONE','no-DONE','DONE-first','partial','blank-tail','partial-tail','utf8','duplicate-json-key'):
            data=fixture();g,_,_,_,_,ts=data;row=g['requests'][0];t=ts[0]
            if key=='duplicate-terminal':replace_wire(row,t,events=[t,t,'[DONE]'])
            if key=='duplicate-DONE':replace_wire(row,t,events=[t,'[DONE]','[DONE]'])
            if key=='no-DONE':replace_wire(row,t,events=[t])
            if key=='DONE-first':replace_wire(row,t,events=['[DONE]',t])
            if key=='partial':replace_wire(row,t);raw=base64.b64decode(row['raw_sse_chunks'][0]['body_bytes_base64'])[:-1]
            if key=='blank-tail':replace_wire(row,t,suffix=b'\n')
            if key=='partial-tail':replace_wire(row,t,suffix=b'data: {')
            if key=='utf8':replace_wire(row,t,suffix=b'\xff')
            if key=='duplicate-json-key':
                replace_wire(row,t);raw=base64.b64decode(row['raw_sse_chunks'][0]['body_bytes_base64']).replace(b'"index": 0',b'"index": 1,"index": 0',1)
            if key in ('partial','duplicate-json-key'):
                row['raw_sse_chunks'][0]['body_bytes_base64']=base64.b64encode(raw).decode();row['raw_body_bytes']=len(raw)
            with self.subTest(key=key),self.assertRaises(ValueError):self.run_join(data)

    def test_trace_missing_unknown_duplicate_reorder_and_undrained_fail(self):
        for key in ('missing','unknown','startup-again','empty-runner','arrive-again','unknown-rid','before-order','decode-order','result-dup','token','done','retire','survive','tail'):
            data=fixture();g,e,_,_,_,_=data
            if key=='missing':e.pop(3)
            if key=='unknown':e[3]['event']='fabricated'
            if key=='startup-again':e.insert(1,copy.deepcopy(e[0]))
            if key=='empty-runner':e[2:4]=[]
            if key=='arrive-again':e[5]['newly_submitted_to_scheduler']=[A]
            if key=='unknown-rid':e[1]['newly_submitted_to_scheduler']=['0'*32]
            if key=='before-order':e[1]['scheduler_waiting'].reverse()
            if key=='decode-order':e[6]['decode'].reverse()
            if key=='result-dup':e[4]['results'].append(e[4]['results'][0])
            if key=='token':e[4]['results'][0][1]=777
            if key=='done':e[4]['results'][0][2]=True
            if key=='retire':e[8]['retired']=[]
            if key=='survive':e[8]['surviving_decode_rows']=[]
            if key=='tail':e.pop()
            with self.subTest(key=key),self.assertRaises(ValueError):self.run_join(data)

    def test_streaming_trace_and_real_multirow_requirement(self):
        data=fixture(1);g,e,j,f,i,_=data
        g['concurrency']=i['configuration']['max_seqs']=8;e[0]['identity']=copy.deepcopy(i)
        with self.assertRaisesRegex(ValueError,'multirow'):self.run_join(data)
        data=fixture();g,e,j,f,i,_=data
        class OnePass:
            def __init__(self):self.used=False
            def __iter__(self):
                if self.used:raise AssertionError('trace iterated twice')
                self.used=True;yield from lines(e)
        self.assertEqual(h.join(g,OnePass(),j,f,i,decode)['concurrency'],8)
        raw=list(lines(e));raw[-1]=raw[-1][:-1]
        with self.assertRaisesRegex(ValueError,'truncated'):h.join(g,iter(raw),j,f,i,decode)


if __name__=='__main__':unittest.main()
