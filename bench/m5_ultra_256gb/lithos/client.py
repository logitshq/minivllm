"""Original concurrent SSE timing and canonical-token client, preserved verbatim."""
import concurrent.futures,json,statistics,threading,time,types
import httpx
def make_group(engine,base,headers,prompt,tok,model_name):
 a=types.SimpleNamespace(engine=engine)
 path="/v1/completions";body={"model":model_name,"prompt":prompt,"temperature":0,"repetition_penalty":1.0,"max_tokens":32}
 if engine=='lithos':
  path='/v1/chat/completions';body={'model':model_name,'messages':[{'role':'user','content':'用一句话解释什么是 KV cache。'}],'temperature':0,'top_p':1,'max_completion_tokens':32}
 def request(barrier,group_id,request_id):
    with httpx.Client(headers=headers,trust_env=False,timeout=180) as c:
     barrier.wait();start=time.perf_counter();first=None;pieces=[];events=[];usage=None;finish=None;done=False;done_time=None;output_ids=None
     with c.stream('POST',base+path,json={**body,'stream':True,'stream_options':{'include_usage':True}}) as resp:
      resp.raise_for_status()
      for line in resp.iter_lines():
       if not line.startswith('data:'):continue
       payload=line[5:].strip()
       if payload=='[DONE]':done_time=time.perf_counter();done=True;break
       d=json.loads(payload)
       if 'error' in d:raise RuntimeError(d['error'])
       usage=d.get('usage') or usage
       if 'benchmark_output_ids' in d:output_ids=d['benchmark_output_ids']
       for choice in d.get('choices',[]):
        finish=choice.get('finish_reason') or finish;piece=choice.get('text',choice.get('delta',{}).get('content',''))
        if piece:
         now=time.perf_counter();first=first or now;pieces.append(piece);events.append({'client_perf_counter_s':now,'text':piece,'engine_delivery':d.get('benchmark_delivery')})
     cleanup_complete=time.perf_counter();end=done_time;assert done and usage and finish=='length',(done,usage,finish);assert usage['prompt_tokens']==19 and usage['completion_tokens']==32,usage
     cached=usage.get('prompt_tokens_details',{}).get('cached_tokens',0);assert cached==0,cached
     text=''.join(pieces);visible_ids=tok.encode(text,add_special_tokens=False).ids;first_ids=tok.encode(pieces[0],add_special_tokens=False).ids
     prefix_ok=visible_ids[:len(first_ids)]==first_ids and text.startswith(pieces[0]);full_roundtrip_ok=len(visible_ids)==32
     assert len(visible_ids)==32,'visible canonical token count mismatch'
     if a.engine=='tutorial':assert output_ids is not None,'native raw token IDs missing'
     if output_ids is not None:assert isinstance(output_ids,list) and len(output_ids)==32 and output_ids==visible_ids,'actual engine token IDs mismatch/length/EOS'
     first_count=len(first_ids);remaining=32-first_count
     # HTTP APIs expose text chunks. Reconstructed counts are an explicitly labelled client estimate.
     interactivity=remaining/(end-first) if prefix_ok and full_roundtrip_ok and remaining>0 else None
     return {'group_id':group_id,'request_id':request_id,'cleanup_complete_perf_counter_s':cleanup_complete,'stream_cleanup_after_done_s':cleanup_complete-end,'start_perf_counter_s':start,'done_perf_counter_s':end,'first_text_perf_counter_s':first,'client_total_s':end-start,'client_ttft_text_s':first-start,'client_post_first_chunk_s':end-first,'usage':usage,'finish_reason':finish,'text':text,'output_ids_when_available':output_ids,'visible_roundtrip_ids':visible_ids,'first_chunk_text':pieces[0],'first_chunk_reconstructed_tokens':first_count,'first_chunk_prefix_verified':prefix_ok,'full_visible_roundtrip_count_matches_usage':full_roundtrip_ok,'x_client_estimated_interactivity_tok_s_user':interactivity,'events':events}
 def group(batch,group_id):
    barrier=threading.Barrier(batch)
    with concurrent.futures.ThreadPoolExecutor(max_workers=batch) as ex:runs=list(ex.map(lambda i:request(barrier,group_id,f"{group_id}-R{i}"),range(batch)))
    begin=min(r['start_perf_counter_s'] for r in runs);end=max(r['done_perf_counter_s'] for r in runs);window=end-begin;xs=[r['x_client_estimated_interactivity_tok_s_user'] for r in runs]
    return {'group_id':group_id,'actual_request_count':batch,'requests':runs,'arrival_spread_s':max(r['start_perf_counter_s'] for r in runs)-begin,'group_window_s':window,'total_completion_tokens':sum(r['usage']['completion_tokens'] for r in runs),'y_total_output_throughput_tok_s_mac':batch*32/window,'x_client_estimated_interactivity_tok_s_user':statistics.median(xs) if all(x is not None for x in xs) else None}
 return group
