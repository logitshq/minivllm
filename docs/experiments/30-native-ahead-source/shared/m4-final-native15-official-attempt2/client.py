"""Serial-arm client; mature SSE request/group functions copied exactly."""
import argparse,ast,concurrent.futures,hashlib,json,os,secrets,socket,statistics,subprocess,threading,time,traceback
from pathlib import Path
from common import *
def scheduler(engine,batch):
 if engine=='native':return {'max_seqs':4,'max_len':2048,'max_batched_tokens':512}
 v=copy.deepcopy(REGISTRY[engine]['scheduler'])
 if engine=='omlx':v.update(max_concurrent_requests=batch,prefill_batch_size=1,completion_batch_size=batch)
 return v

def validate_warm(v,batch):
 r=v['routes'];assert v['gates']['enabled'] is True and v['parameter_tree_unchanged'] is True and v['completed_requests']==3*batch
 assert r['hooks_restored_before_measure'] is True and r['setter_calls']==48
 counts={int(k):n for k,n in r['decode_batch_counts'].items()};assert sum(b*n for b,n in counts.items())==48*31*3*batch
 assert r['normal_decode_calls']==r['gates_route_calls']==sum(counts.values())
 assert r['compiled_eligible_calls']==counts.get(1,0) and r['fallback_calls']==sum(n for b,n in counts.items() if b!=1)
 if batch==1:assert v['gates']['compiled_callable_count']==48 and r['compiled_eligible_calls']==4464

def validate_native_requests(requests,batch,continuation):
 assert all(len(q['output_ids_when_available'])==32 and q['output_ids_when_available']==q['visible_roundtrip_ids'] for q in requests),'native raw32 own canonical mismatch'
 if batch==1:assert all(q['output_ids_when_available']==continuation for q in requests),'C1 pinned continuation mismatch'
 return [q['output_ids_when_available'] for q in requests]

def official_command(engine,batch,run_dir,port,key,model,env):
 rec=REGISTRY[engine];command=list(rec['reference_launch']);command[command.index('--port')+1]=str(port)
 if engine=='omlx':
  models=run_dir/'models';models.mkdir();(models/model.name).symlink_to(model,target_is_directory=True)
  state=run_dir/'state';state.mkdir()
  write_new(state/'settings.json',{'server':{'burst_decode_mode':'balanced'},'scheduler':{'max_concurrent_requests':batch,'chunked_prefill':False,'prefill_priority':'speed','decode_fairness':True},'cache':{'enabled':False}})
  for flag,value in [('--model-dir',models),('--base-path',state),('--api-key',key),('--max-concurrent-requests',batch)]:command[command.index(flag)+1]=str(value)
 else:
  assert batch==1,'Lithos pinned endpoint admits one request only'
  env.update({'LITHOS_METAL_API_KEY':key,'MONOLITH_ACCELERATOR':'on'})
 py=rec['actual_environment']['python_executable']
 return [py,'-B',str(D/'official_entry.py'),'--engine',engine,'--receipt',str(run_dir/'official-preimport.json'),'--',*command]
def run_one(item,attempt,power,shared,pins_sha256):
 label=item["label"];config=item["engine"];batch=item["batch"]
 enabled=True;rec=REGISTRY.get(config,{});engine=config;mode='balanced' if engine=='omlx' else None
 run_dir=attempt/label;run_dir.mkdir();samples=[];proc=None;report=None;lf=None;first_error=None;cleanup_errors=[];exit_record=None;port=None;birth=None
 def sample(label):return probe(samples,run_dir/'resource-snapshots.json',label,power,proc.pid if proc else None)
 try:
  import httpx
  from tokenizers import Tokenizer
  w=json.loads((P/'workload.json').read_text());model=Path(w['model_dir'])
  assert model.name=='5ff8ef173ad0d7c3aae92f0be43031a6ab8067c6' and len(w['prompt_ids'])==19
  tok=Tokenizer.from_file(str(model/'tokenizer.json'));prompt=tok.decode(w['prompt_ids'],skip_special_tokens=False);assert tok.encode(prompt,add_special_tokens=False).ids==w['prompt_ids']
  with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
  key='benchmark-'+secrets.token_urlsafe(24)
  env=child_environment(os.environ)
  if 'first_env' not in shared:shared['first_env']=dict(env)
  assert env==shared['first_env'],'arms must use identical environment'
  command=[PYTHON,'-B',str(CHILD),'--model',str(model),'--port',str(port),'--batch','4','--client-batch',str(batch),'--trace',str(run_dir/'engine-trace.json'),'--package-root',str(CANDIDATE.parent.parent),'--source-manifest',str(SOURCE),'--gates','on','--owner-pid',str(os.getpid()),'--run-dir',str(run_dir),'--pins-sha256',pins_sha256]
  if engine!='native':command=official_command(engine,batch,run_dir,port,key,model,env)
  write_new(run_dir/'launch.json',{'command':[('<api-key>' if x==key else x) for x in command],'environment':boot.environment_record(env),'engine':engine,'config':config,'selected_environment':{k:v for k,v in env.items() if k.startswith(('MLX_','OMLX_','MONOLITH_','HF_','TRANSFORMERS_')) or k in ('PYTHONDONTWRITEBYTECODE','TOKENIZERS_PARALLELISM','PYTHONPATH','PYTHONHOME')},'gates_compile_enabled':engine=='native','only_treatment':'seven fixed points; native15 gates ON; original batch eligibility','timed_instrumentation':False,'instrumentation_scope':('measurement has no added hooks; three native warm requests validate routes before restore' if engine=='native' else 'official source unchanged; no added measurement or warm hooks')})
  sample('before-model-launch');start=time.perf_counter()
  lf=(run_dir/'server.log').open('w')
  proc=subprocess.Popen(command,stdout=lf,stderr=subprocess.STDOUT,env=env)
  write_new(run_dir/'owned-pid.json',{'pid':proc.pid,'ppid':os.getpid()})
  birth=identity(proc.pid);assert birth is not None,'server exited before ownership capture'
  write_new(run_dir/'owned-birth.json',{'pid':proc.pid,'ppid':os.getpid(),'identity_words':birth,'birth_start_words':birth[:5],'birth_sha256':hashlib.sha256(json.dumps(birth).encode()).hexdigest()})
  base=f'http://127.0.0.1:{port}';headers={'Authorization':'Bearer '+key}
  with httpx.Client(headers=headers,trust_env=False,timeout=180) as c:
   while True:
    if proc.poll() is not None:raise RuntimeError('owned server exited before ready')
    try:
     resp=c.get(base+'/v1/models',timeout=1)
     if resp.status_code==200:break
    except httpx.HTTPError:pass
    if time.perf_counter()-start>180:raise TimeoutError('readiness')
    time.sleep(.2)
  ready_s=time.perf_counter()-start
  exact_listener(proc.pid,port,run_dir/'server-ready-exact-listener.json')
  birth=lock_owned_identity(proc,birth,engine,run_dir,'ready')
  runtime_identity=resp.json().get('benchmark_identity')
  if engine=='native':assert runtime_identity['manifest_all_sources_match'] and runtime_identity['gates_startup']['enabled'] is True
  name=resp.json()['data'][0]['id'];sample('server-ready-before-warmup')
  path='/v1/completions';body={'model':name,'prompt':prompt,'temperature':0,'repetition_penalty':1.0,'max_tokens':32}
  if engine=='lithos':path='/v1/chat/completions';body={'model':name,'messages':[{'role':'user','content':'用一句话解释什么是 KV cache。'}],'temperature':0,'top_p':1,'max_completion_tokens':32}
  a=argparse.Namespace(engine='tutorial' if engine=='native' else engine)
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
  groups={};metadata=None
  for phase in ['warm','measure']:
   groups[phase]=[]
   for i in range(3):
    gid=f'C{batch}-{phase}-{i}';sample('before-'+gid)
    result=group(batch,gid)
    write_new(run_dir/(gid+'.json'),result);groups[phase].append(result)
    sample('after-'+gid)
   if phase=='warm' and engine=='native':
    deadline=time.monotonic()+30
    while not (run_dir/'warm-metadata.json').exists():
     assert proc.poll() is None,'child exited before warm metadata'
     if time.monotonic()>deadline:raise TimeoutError('warm metadata')
     time.sleep(.01)
    metadata=json.loads((run_dir/'warm-metadata.json').read_text());assert metadata['gates']['enabled'] is True and metadata['routes']['hooks_restored_before_measure'] is True
    validate_warm(metadata,batch)
    assert metadata['parameter_tree_unchanged'] is True
   if phase=='warm':sample('after-three-warmup-groups-before-measurement')
  sample('after-measurement-before-unload')
  requests=[q for phase in ['warm','measure'] for g in groups[phase] for q in g['requests']]
  if engine=='native':
   validate_native_requests(requests,batch,w['continuation_ids'])
   shared.setdefault('native_output_variants',{}).setdefault(str(batch),[]).append([q['output_ids_when_available'] for q in requests])
  lithos_proof=None
  if engine=='lithos':
   text=(run_dir/'server.log').read_text();assert 'Serving backend=m4_max,' in text and 'draft=None' in text and 'backend=m5' not in text,'actual M4/noDraft backend log proof missing'
   native=[ast.literal_eval(line.split('Generation: ',1)[1]) for line in text.splitlines() if 'Generation: ' in line]
   assert len(native)>=6 and all(row['cached_prompt_tokens']==0 for row in native)
   lithos_proof={'backend':'m4_max','draft':None,'MONOLITH_ACCELERATOR':env['MONOLITH_ACCELERATOR'],'native_generation_metrics':native,'server_log_sha256':sha(run_dir/'server.log'),'scope':'actual backend log plus frozen source and accelerator launch option; no M5 profile override'}
  report={'schema':'m4-final-native15-official-arm-v1','config':config,'label':label,'engine':engine,'mode':mode,'gates_compile_enabled':engine=='native','lithos_proof':lithos_proof,'registry_entry_sha256':sha(D/'registry.json'),'concurrency':batch,'warmup_groups':3,'repeat':3,'warmups':groups['warm'],'runs':groups['measure'],'runtime_source_identity':runtime_identity,'post_warm_metadata':metadata,'ready_s':ready_s,'model_dir':str(model),'checkpoint_revision':model.name,'prompt_ids':w['prompt_ids'],'config_sha256':sha(model/'config.json'),'quantization':json.loads((model/'config.json').read_text())['quantization'],'activation_dtype':'bfloat16','output_cap':32,'sampling':{'temperature':0,'EOS':'normal','draft':None},'scheduler_config':scheduler(engine,batch),'prefix_cache':0,'boundary':'unchanged original client POST start through SSE DONE; cleanup excluded','x_definition':'median per request (32 - reconstructed first nonempty chunk token count)/(SSE DONE - first text); client delivery proxy','y_definition':'batch*32 actual completion tokens / group POST-through-DONE window; queue/prefill/HTTP included','median_x_client_estimated_interactivity_tok_s_user':statistics.median(g['x_client_estimated_interactivity_tok_s_user'] for g in groups['measure']),'median_y_total_output_throughput_tok_s_mac':statistics.median(g['y_total_output_throughput_tok_s_mac'] for g in groups['measure']),'expected_power':power,'environment':boot.environment_record(env),'source_identity':verify_source() if engine=='native' else {'source_and_native_tree_sha256':rec['tree_sha256'],'environment_manifest_sha256':rec['environment_manifest_sha256']},'controller_sha256':sha(__file__),'child_sha256':sha(CHILD),'timed_instrumentation':False,'instrumentation_scope':('measurement has no added hooks; three native warm requests validate routes before restore' if engine=='native' else 'official source unchanged; no added measurement or warm hooks'),'scope':'C1/C2/C4 seven fixed points; native raw IDs, official usage/canonical text only; all21 fresh processes retained; no default-on or promotion'}
 except BaseException as exc:
  first_error=exc
  write_new(run_dir/'first-failure.json',{'type':type(exc).__name__,'error':str(exc),'traceback':traceback.format_exc(),'raw_groups_preserved':True})
 finally:
  if proc is not None:
   try:
    birth=lock_owned_identity(proc,birth,engine,run_dir,'pre-cleanup')
    exit_record=cleanup_owned(proc,birth,run_dir/'cleanup-signal-identities.json');write_new(run_dir/'owned-process-exit.json',exit_record)
   except BaseException as exc:cleanup_errors.append({'phase':'wait-reap-absence','type':type(exc).__name__,'error':str(exc)})
  if port is not None:
   try:write_new(run_dir/'port-absence.json',wait_port_absence(port))
   except BaseException as exc:cleanup_errors.append({'phase':'port-absence','type':type(exc).__name__,'error':str(exc)})
  try:probe(samples,run_dir/'resource-snapshots.json','after-owned-server-unload-finally',power)
  except BaseException as exc:cleanup_errors.append({'phase':'finally-postflight','type':type(exc).__name__,'error':str(exc)})
  if lf is not None:lf.close()
  write_new(run_dir/'cleanup.json',{'first_error':None if first_error is None else {'type':type(first_error).__name__,'error':str(first_error)},'cleanup_errors':cleanup_errors,'exit_record':exit_record})
 if first_error is not None:raise first_error
 if cleanup_errors:raise RuntimeError('owned cleanup/postflight failed: '+str(cleanup_errors))
 if report is not None:
  report['resource_snapshots']=samples;report['owned_process_exit']=exit_record
  write_new(run_dir/'result.json',report)
  return {'label':label,'config':config,'engine':engine,'report':str(run_dir/'result.json'),'report_sha256':sha(run_dir/'result.json'),'x':report['median_x_client_estimated_interactivity_tok_s_user'],'y':report['median_y_total_output_throughput_tok_s_mac'],'actual_completed_requests':6*batch,'actual_completion_tokens':192*batch}
