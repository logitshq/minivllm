"""V130 bounded parallel POST client; official adapters, no retry or custom queue."""
import concurrent.futures,json,statistics,threading,time
MESSAGE='用一句话解释什么是 KV cache。'

def body(model,protocol):
 messages=[{'role':'user','content':MESSAGE}]
 if protocol=='openai':return {'model':model,'messages':messages,'temperature':0,'top_p':1,'max_completion_tokens':32,'stream':True,'stream_options':{'include_usage':True}}
 if protocol=='ollama':return {'model':model,'messages':messages,'stream':True,'think':False,'options':{'temperature':0,'top_p':1,'top_k':0,'seed':0,'num_predict':32,'repeat_penalty':1}}
 raise ValueError('unknown wire protocol')

def run_group(base,model,key,concurrency,protocol,tokenizer,group_id,gold_ids):
 if type(concurrency)is not int or concurrency not in (1,2,4):raise ValueError('declared C1/C2/C4 only')
 import httpx
 barrier=threading.Barrier(concurrency)
 def request(i):
  pieces=[];events=[];terminal=None;first=None;status=None;usage=None;finish=None;error=None;headers_at=None;start=None;end=None
  endpoint='/v1/chat/completions' if protocol=='openai' else '/api/chat'
  with httpx.Client(headers={'Authorization':'Bearer '+key},trust_env=False,timeout=180) as c:
   barrier.wait();start=time.perf_counter()
   try:
    with c.stream('POST',base+endpoint,json=body(model,protocol)) as response:
     status=response.status_code;headers_at=time.perf_counter()
     if status!=200:
      content=response.read()
      try: rejected=json.loads(content)
      except (ValueError,UnicodeDecodeError):rejected={'non_JSON_error_bytes':len(content)}
      end=time.perf_counter();error={'kind':'HTTP_REJECTION','status':status,'body':rejected}
     else:
      for line in response.iter_lines():
       if not line:continue
       if protocol=='openai':
        if not line.startswith('data:'):continue
        payload=line[5:].strip()
        if payload=='[DONE]':terminal='SSE_DONE';end=time.perf_counter();break
        value=json.loads(payload)
       else:value=json.loads(line)
       now=time.perf_counter();events.append({'received_perf_counter_s':now,'value':value})
       if 'error' in value:raise RuntimeError('stream error: '+str(value['error']))
       if protocol=='openai':
        usage=value.get('usage') or usage
        for choice in value.get('choices',[]):
         finish=choice.get('finish_reason') or finish;piece=choice.get('delta',{}).get('content','') or ''
         if piece:first=first or now;pieces.append(piece)
       else:
        piece=value.get('message',{}).get('content','') or ''
        if piece:first=first or now;pieces.append(piece)
        if value.get('done') is True:
         terminal='NDJSON_DONE_TRUE';end=time.perf_counter();finish=value.get('done_reason')
         usage={'prompt_tokens':value.get('prompt_eval_count'),'completion_tokens':value.get('eval_count'),'cached_tokens_observed':None}
         break
      if end is None:end=time.perf_counter();error={'kind':'MISSING_TERMINAL','message':'stream ended without required terminal'}
   except Exception as exc:
    end=time.perf_counter();error={'kind':type(exc).__name__,'message':str(exc)}
  cleanup=time.perf_counter();text=''.join(pieces);ids=tokenizer.encode(text,add_special_tokens=False).ids
  correct=(status==200 and error is None and terminal is not None and finish=='length' and usage is not None and usage.get('prompt_tokens')==19 and usage.get('completion_tokens')==32 and len(ids)==32 and ids==gold_ids)
  return {'request_id':f'{group_id}-R{i}','protocol':protocol,'endpoint':endpoint,'HTTP_status':status,'start_perf_counter_s':start,'headers_perf_counter_s':headers_at,'first_text_perf_counter_s':first,'terminal_perf_counter_s':end,'cleanup_perf_counter_s':cleanup,'complete_latency_s':end-start,'terminal':terminal,'finish_reason':finish,'usage':usage,'text':text,'visible_roundtrip_ids':ids,'internal_raw_ids':None,'events':events,'error':error,'all_correctness_checks_passed':correct,'retry_count':0,'server_queue_wait_s_observed':None}
 with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:rows=list(pool.map(request,range(concurrency)))
 start=min(q['start_perf_counter_s'] for q in rows);end=max(q['terminal_perf_counter_s'] for q in rows)
 success=[q for q in rows if q['all_correctness_checks_passed']];all_ok=len(success)==concurrency
 return {'group_id':group_id,'protocol':protocol,'requested_concurrency':concurrency,'GPU_batch_observed':None,'requests':rows,'actual_request_count':len(rows),'successful_requests':len(success),'rejected_requests':sum(q['HTTP_status'] not in (None,200) for q in rows),'all_requested_POSTs_before_first_terminal':max(q['start_perf_counter_s'] for q in rows)<min(q['terminal_perf_counter_s'] for q in rows),'arrival_spread_s':max(q['start_perf_counter_s'] for q in rows)-start,'all_attempts_window_s':end-start,'latency_p50_ms':1000*statistics.median(q['complete_latency_s'] for q in rows) if all_ok else None,'total_throughput_tok_s':32*concurrency/(end-start) if all_ok else None,'completed_token_count':sum((q['usage'] or {}).get('completion_tokens') or 0 for q in rows),'all_correctness_checks_passed':all_ok,'retries':0,'queue_wait_in_complete_client_latency':True,'boundary':'client pre-POST through received NDJSON done:true (Ollama) or SSE [DONE] (OpenAI), cleanup excluded'}
