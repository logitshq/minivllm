"""Join received native SSE bytes and ordered qualification JSONL; stdlib only.

join(group, trace, jobs, fixtures_by_id, identity, decode_text) raises ValueError
on any mismatch. trace is an iterator of newline-terminated JSONL bytes/strings;
decode_text(ids) must use the externally pinned tokenizer, skip_special_tokens=False.
Success is a bounded replay input, NOT engine, HTTP performance or resource
qualification. All retained tail bytes are checked; the frozen client stops at
DONE, so this cannot prove absence of unreceived wire bytes after DONE.
"""
import base64
import codecs
from collections import Counter
import copy
import hashlib
import json
import math
import re

SCHEMA='E023-recorded-http-v1'


def need(ok,message):
    if not ok: raise ValueError(message)


def same(a,b,message):
    need(json.dumps(a,sort_keys=True,allow_nan=False)==json.dumps(b,sort_keys=True,allow_nan=False),message)


def integer(x):return type(x) is int and x>=0
def ids(x):return isinstance(x,list) and all(integer(i) for i in x)
def finite(x):return type(x) in (int,float) and math.isfinite(x)


def object_pairs(pairs):
    result={}
    for k,v in pairs:
        need(k not in result,'duplicate JSON key: '+k);result[k]=v
    return result


def parse(text):
    def bad(value):raise ValueError('nonfinite JSON constant: '+value)
    return json.loads(text,object_pairs_hook=object_pairs,parse_constant=bad)


def raw_completion(row):
    """Parse original native frames, never derived native_fields/answer/usage."""
    decoder=codecs.getincrementaldecoder('utf-8')('strict')
    buffer='';pending=None;terminal=None;rid=None;text=[];done=False;size=0;last=None;done_time=None
    def consume(payload,received):
        nonlocal terminal,rid,done,done_time
        need(not done,'retained data after DONE')
        if payload=='[DONE]':
            need(terminal is not None,'DONE before unique terminal completion')
            done=True;done_time=received;return
        need(terminal is None,'extra/duplicate event after terminal completion')
        data=parse(payload)
        need(isinstance(data,dict) and 'error' not in data,'non-object or error SSE')
        match=re.fullmatch(r'cmpl-([0-9a-f]{32})',data.get('id',''))
        need(match is not None,'response ID must be cmpl-server UUID')
        actual=match[1];need(rid is None or rid==actual,'response ID changed within request');rid=actual
        choices=data.get('choices')
        need(isinstance(choices,list) and len(choices)==1 and isinstance(choices[0],dict),'one native choice required')
        choice=choices[0]
        need(set(choice)=={'index','text','finish_reason'} and type(choice['index']) is int and choice['index']==0
             and isinstance(choice['text'],str),'native choice shape')
        if choice['finish_reason'] is None:
            need(set(data)=={'id','choices'} and choice['text'],'native delta fields/text')
            text.append(choice['text'])
        else:
            need(set(data)=={'id','choices','usage','benchmark_request_id','benchmark_prompt_ids','benchmark_output_ids',
                             'benchmark_committed_output_ids','benchmark_terminal_eos_id'} and choice['text']=='',
                 'exact unique native terminal fields')
            need(choice['finish_reason'] in ('stop','length') and data['benchmark_request_id']==rid,'terminal finish/rid mismatch')
            terminal=data
    chunks=row.get('raw_sse_chunks')
    need(isinstance(chunks,list) and chunks and len(chunks)<=65536,'raw SSE chunks missing/bound')
    for chunk in chunks:
        need(set(chunk)=={'client_perf_counter_s','body_bytes_base64'},'raw chunk schema')
        received=chunk['client_perf_counter_s']
        need(finite(received) and (last is None or received>=last),'raw receive clock order')
        last=received
        raw=base64.b64decode(chunk['body_bytes_base64'],validate=True);size+=len(raw)
        need(size<=8<<20,'qualification response raw byte bound')
        need(not done or not raw,'retained tail bytes after DONE')
        buffer+=decoder.decode(raw)
        while '\n' in buffer:
            line,buffer=buffer.split('\n',1)
            need(not done,'retained extra frame/bytes after DONE')
            if line=='':
                need(pending is not None,'unexpected empty SSE frame')
                consume(pending,received);pending=None
            else:
                need(pending is None and line.startswith('data: '),'unsupported/multiline native SSE framing')
                pending=line[6:]
        need(not done or not buffer,'retained unterminated tail after DONE')
    buffer+=decoder.decode(b'',final=True)
    need(done and terminal is not None and pending is None and not buffer,'truncated or missing terminal/DONE')
    need(row.get('raw_body_truncated') is False and type(row.get('raw_body_bytes')) is int and row['raw_body_bytes']==size,'raw byte receipt mismatch')
    need(finite(row.get('start_perf_counter_s')) and row['start_perf_counter_s']<=chunks[0]['client_perf_counter_s']
         and row.get('done_perf_counter_s')==done_time,'request/DONE clock inconsistent with raw bytes')
    return rid,terminal,''.join(text)


def http_jobs(group,jobs,fixtures,identity,decode_text):
    need(group.get('schema')=='E023-http-group-v1' and group.get('phase')=='qualification'
         and group.get('qualification') is None,'unqualified qualification-phase native group required')
    need(group.get('group_error') is None and group.get('summary',{}).get('formal_point_eligible') is False,'group failure or formal-point claim')
    cleanup=group.get('cleanup',{})
    need(cleanup.get('complete') is True and cleanup.get('deadline_exceeded') is False and cleanup.get('errors')==[]
         and cleanup.get('pending_tasks')==[] and cleanup.get('cancelled_again') is False,'incomplete/cancelled client cleanup')
    C=group.get('concurrency');config=identity['configuration']
    need(type(C) is int and C>=1 and config['max_seqs']==C and type(config['max_seqs']) is int,'configured/concurrent domain mismatch')
    need(type(config['max_batched_tokens']) is int and config['max_batched_tokens']>=C
         and type(config['max_len']) is int and config['max_len']>0,'native capacity identity')
    need(identity['step_observer'] is True and identity['qualification_trace_enabled'] is True
         and identity['per_token_observer'] is False and identity['drafter'] is None
         and identity['normal_eos'] is True and identity['prefix_cache'] is False,'qualification target-only identity flags')
    eos=identity['eos_ids'];need(ids(eos) and eos and len(set(eos))==len(eos),'EOS identity')
    rows=group.get('requests');need(isinstance(rows,list) and len(rows)==len(jobs)>0,'complete fixed job queue required')
    clients=set();joined={}
    for job,row in zip(jobs,rows):
        need(set(job)=={'request_id','fixture_id','path','body'},'fixed job fields')
        client=job['request_id'];fid=job['fixture_id']
        need(isinstance(client,str) and client and all(32<=ord(c)<=126 for c in client) and client not in clients,'duplicate/invalid client request ID');clients.add(client)
        fixture=fixtures[fid];body=job['body']
        same({k:row.get(k) for k in job},job,'row does not match original job/body/path')
        need(job['path']=='/v1/completions' and fixture['native_path']==job['path'] and fixture['fixture_id']==fid,'fixture/native route')
        expected=dict(model='native-target-only',prompt=fixture['prompt_text'],max_tokens=fixture['max_output_tokens'],
                      temperature=0,top_p=1,repetition_penalty=1,stream=True,stream_options={'include_usage':True},n=1)
        # JSON number 1 and 1.0 are both the unmodified-logit native contract.
        need(set(body)==set(expected) and body['model']==expected['model'] and isinstance(body['prompt'],str) and body['prompt']==expected['prompt']
             and type(fixture['max_output_tokens']) is int
             and type(body['max_tokens']) is int and body['max_tokens']==expected['max_tokens']>0
             and body['stream'] is True and isinstance(body['stream_options'],dict) and set(body['stream_options'])=={'include_usage'}
             and body['stream_options']['include_usage'] is True
             and type(body['n']) is int and body['n']==1,'full native body/prompt/cap')
        for key,value in (('temperature',0),('top_p',1),('repetition_penalty',1)):
            need(type(body[key]) in (int,float) and body[key]==value,'greedy/unmodified-logit request')
        prompt=fixture['prompt_ids'];cap=body['max_tokens']
        need(ids(prompt) and prompt and len(prompt)+cap<=config['max_len'] and fixture['normal_eos'] is True,'fixture IDs/capacity/EOS')
        need(ids(fixture['eos_token_ids']) and set(fixture['eos_token_ids'])==set(eos),'fixture EOS identity')
        need(row.get('status')=='done' and row.get('http_status')==200 and row.get('error') is None
             and row.get('validation_errors')==[] and row.get('qualified_output_tokens') is None,'failed request or prequalified token count')
        rid,terminal,text=raw_completion(row);need(rid not in joined,'server UUID reused across requests')
        raw,committed=terminal['benchmark_output_ids'],terminal['benchmark_committed_output_ids']
        same(terminal['benchmark_prompt_ids'],prompt,'full prompt IDs differ from frozen fixture')
        need(ids(raw) and 1<=len(raw)<=cap and ids(committed) and not any(t in eos for t in raw[:-1]),'invalid output IDs/cap/nonterminal EOS')
        stopped=raw[-1] in eos
        need(stopped or len(raw)==cap,'early non-EOS termination')
        same(committed,raw[:-1] if stopped else raw,'committed output IDs')
        same(terminal['benchmark_terminal_eos_id'],raw[-1] if stopped else None,'terminal EOS ID')
        need(terminal['choices'][0]['finish_reason']==('stop' if stopped else 'length'),'finish versus EOS/cap')
        same(terminal['usage'],dict(prompt_tokens=len(prompt),completion_tokens=len(committed),total_tokens=len(prompt)+len(committed),
                                   prompt_tokens_details={'cached_tokens':0}),'exact usage/zero cached count')
        need(decode_text(committed)==text,'wire text differs from committed-ID exact decode')
        joined[rid]=dict(rid=rid,client_request_id=client,fixture_id=fid,prompt_ids=list(prompt),cap=cap,
                         raw_ids=raw,committed_ids=committed)
    return C,joined


def join(group,trace,jobs,fixtures_by_id,identity,decode_text):
    C,joined=http_jobs(group,list(jobs),fixtures_by_id,identity,decode_text)
    iterator=iter(trace);digest=hashlib.sha256();lines=0
    def next_event(optional=False):
        nonlocal lines
        try:raw=next(iterator)
        except StopIteration:
            need(optional,'incomplete trace event sequence');return None
        need(isinstance(raw,(bytes,str)),'trace must yield raw JSONL bytes/strings')
        raw=raw.encode('utf-8') if isinstance(raw,str) else raw
        need(raw.endswith(b'\n') and len(raw)<=8<<20,'truncated/oversized JSONL event')
        digest.update(raw);lines+=1
        event=parse(raw)
        need(isinstance(event,dict),'trace event must be object');return event
    startup=next_event();same(startup,{'event':'startup','identity':identity},'startup expected identity mismatch')
    seen=set();completed=set();running=[];waiting=[];computed={};outputs={};finished=[]
    records=[];hist=Counter();upper=sum(len(j['prompt_ids'])+j['cap'] for j in joined.values())
    while (before:=next_event(optional=True)) is not None:
        need(len(records)<upper,'trace exceeds finite request work bound')
        need(set(before)=={'event','newly_submitted_to_scheduler','scheduler_running','scheduler_waiting'}
             and before['event']=='before_step','unknown/duplicate/out-of-order trace event')
        arrivals=before['newly_submitted_to_scheduler']
        need(isinstance(arrivals,list) and all(isinstance(r,str) for r in arrivals) and len(arrivals)==len(set(arrivals)),'duplicate/invalid arrivals')
        for rid in arrivals:
            need(rid in joined and rid not in seen,'unknown or repeated admission')
            seen.add(rid);waiting.append(rid);computed[rid]=0;outputs[rid]=[]
        need(len(seen-completed)<=C,'more live server jobs than client concurrency')
        same(before['scheduler_running'],running,'before running order');same(before['scheduler_waiting'],waiting,'before waiting order')
        canonical_before=dict(running=list(running),waiting=list(waiting))
        budget=identity['configuration']['max_batched_tokens']
        decode=[r for r in running if computed[r]>=len(joined[r]['prompt_ids'])];budget-=len(decode)
        prefill=[]
        for rid in running:
            remaining=len(joined[rid]['prompt_ids'])-computed[rid]
            if remaining>0 and budget>0:
                n=min(remaining,budget);prefill.append([rid,n]);budget-=n
        while waiting and len(running)<C and budget>0:
            rid=waiting.pop(0);running.append(rid);n=min(len(joined[rid]['prompt_ids']),budget)
            prefill.append([rid,n]);budget-=n
        need(decode or prefill,'empty-runner trace is not supported')
        schedule=next_event()
        same(schedule,dict(event='actual_scheduler_output',decode=decode,prefill=[dict(rid=r,tokens=n) for r,n in prefill],finished_ids=finished),
             'actual scheduler order/budget/finished IDs (empty-runner sequences unsupported)')
        model=next_event();same(model,dict(event='actual_model_runner_batch',decode_rows=decode),'postrun decode rows')
        after=next_event()
        need(set(after)=={'event','results','retired','surviving_decode_rows'} and after['event']=='after_step','incomplete/unknown after-step event')
        expected_delivered=list(decode)
        for rid,n in prefill:
            computed[rid]+=n
            if computed[rid]==len(joined[rid]['prompt_ids']):expected_delivered.append(rid)
        for rid in decode:computed[rid]+=1
        results=after['results']
        need(isinstance(results,list) and all(isinstance(x,list) and len(x)==3 for x in results),'result tuple schema')
        same([x[0] for x in results],expected_delivered,'missing/duplicate/reordered delivered output')
        retired=[]
        for rid,token,done in results:
            need(integer(token) and type(done) is bool and rid not in completed,'invalid/post-retirement output')
            raw=joined[rid]['raw_ids'];position=len(outputs[rid])
            need(position<len(raw) and token==raw[position],'trace token differs from raw SSE source')
            outputs[rid].append(token)
            expected_done=token in identity['eos_ids'] or len(outputs[rid])==joined[rid]['cap']
            need(done==expected_done,'trace normal EOS/cap/done mismatch')
            if done:need(outputs[rid]==raw,'retirement before complete SSE output');retired.append(rid);completed.add(rid)
        same(after['retired'],retired,'retirement order/coverage')
        surviving=[r for r in decode if r not in completed]
        same(after['surviving_decode_rows'],surviving,'surviving decode rows')
        # Scheduler's pending finished list follows running order; the delivery
        # and retirement lists follow decode-then-prefill output order.
        finished=[r for r in running if r in retired]
        running=[r for r in running if r not in completed]
        record=dict(step=len(records)+1,arrivals=arrivals,before=canonical_before,
                    schedule=dict(decode=decode,prefill=prefill,finished_ids=schedule['finished_ids']),postrun_decode_rows=model['decode_rows'],
                    results=results,retired=retired,surviving_decode_rows=surviving)
        records.append(record);hist[str(len(decode))]+=1
    need(records and seen==completed==set(joined) and not running and not waiting,'all planned jobs must be admitted, retired and drained')
    if C>=8:need(any(int(b)>=2 for b in hist),'C8+ requires actual multirow decode coverage')
    return dict(schema=SCHEMA,concurrency=C,identity=copy.deepcopy(identity),eos_token_ids=list(identity['eos_ids']),
                jobs=joined,steps=records,histogram=dict(hist),trace_sha256=digest.hexdigest(),trace_lines=lines,
                engine_qualified=False,performance_qualified=False,
                scope='Strict received-SSE/trace join for numerical replay only; no model/runtime/resource qualification. Retained tails checked; unreceived wire tails are outside this evidence.')
