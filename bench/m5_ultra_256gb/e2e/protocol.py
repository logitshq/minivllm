"""Historical E2E functions; sole edit is the target resource-module import."""
def derived_snapshot(raw,pid=None):
 s=copy.deepcopy(raw)
 if pid is not None and not any(x['pid']==pid for x in s['known_inference_processes']):s['known_inference_processes'].append({'pid':pid,'owned':True,'engine':'explicit-v19-e2e-child'})
 s['known_inference_process_count']=len(s['known_inference_processes']);s['only_owned_known_inference_processes']=all(x['owned'] for x in s['known_inference_processes']);return s

def cleanup_owned(proc):
 if proc.poll() is None:
  proc.terminate()
  try:proc.wait(timeout=10)
  except subprocess.TimeoutExpired:proc.kill();proc.wait(timeout=15)
 else:proc.wait()
 absence=pid_absence(proc.pid)
 assert absence['absent'] is True,'owned exact PID remains/unknown'
 return {'pid':proc.pid,'wait_completed':True,'returncode':proc.returncode,'exact_pid_absence':absence}

def probe(samples,path,label,power,pid=None):
 from resources import snapshot,deltas
 raw=snapshot(label,pid)
 raw_path=path.parent/(path.stem+'-%02d-raw.json'%len(samples))
 write_new(raw_path,raw) # immutable unmodified snapshot precedes derived guards
 samples.append(derived_snapshot(raw,pid));deltas(samples);write(path,samples)
 phase_guard(samples,power,pid)
 assert samples[-1]['known_inference_process_count']==int(pid is not None),'unexpected known inference inventory'
 assert samples[-1]['pressure_level_value']==1,'requires pressure1'
 return raw_path

def wait_port_absence(port):
 deadline=time.monotonic()+30;checks=[]
 while True:
  with socket.socket() as sock:sock.settimeout(.2);rc=sock.connect_ex(('127.0.0.1',port))
  checks.append({'monotonic_s':time.monotonic(),'connect_ex':rc})
  if rc==errno.ECONNREFUSED:return {'port':port,'localhost_only':True,'absent':True,'checks':checks}
  if rc!=0:raise OSError(rc,'unknown loopback port absence; only ECONNREFUSED proves absent')
  if time.monotonic()>deadline:raise TimeoutError('owned loopback port remains open')
  time.sleep(.05)

def run_one(label,enabled,attempt,power,shared,go_path):
 run_dir=attempt/label;run_dir.mkdir();samples=[];proc=None;report=None;lf=None;first_error=None;cleanup_errors=[];exit_record=None;port=None
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
  command=[PYTHON,'-B',str(CHILD),'--model',str(model),'--port',str(port),'--batch','4','--trace',str(run_dir/'engine-trace.json'),'--package-root',str(CANDIDATE.parent.parent),'--source-manifest',str(SOURCE),'--norm-reuse',('on' if enabled else 'off'),'--owner-pid',str(os.getpid()),'--run-dir',str(run_dir),'--go',str(go_path),'--attempt-dir',str(attempt),'--expected-power',power]
  write_new(run_dir/'launch.json',{'command':command,'environment':boot.environment_record(env),'gates_compile_enabled':True,'norm_reuse_enabled':enabled,'only_treatment':'48 norm reuse setters; both arms compiled gates ON; same V19 source','timed_instrumentation':False,'instrumentation_scope':'measurement has no added hooks; three warm requests have route validation hooks'})
  sample('before-model-launch');start=time.perf_counter()
  lf=(run_dir/'server.log').open('w')
  proc=subprocess.Popen(command,stdout=lf,stderr=subprocess.STDOUT,env=env)
  write_new(run_dir/'owned-pid.json',{'pid':proc.pid,'ppid':os.getpid()})
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
  runtime_identity=resp.json()['benchmark_identity'];assert runtime_identity['manifest_all_sources_match'] and runtime_identity['gates_startup']['enabled'] is True and runtime_identity['norm_reuse_startup']['enabled']==enabled
  name=resp.json()['data'][0]['id'];sample('server-ready-before-warmup')
  path='/v1/completions';body={'model':name,'prompt':prompt,'temperature':0,'repetition_penalty':1.0,'max_tokens':32}
  a=argparse.Namespace(engine='tutorial')
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
  groups={}
  for phase in ['warm','measure']:
   groups[phase]=[]
   for i in range(3):
    gid=f'C1-{phase}-{i}';sample('before-'+gid)
    result=group(1,gid)
    write_new(run_dir/(gid+'.json'),result);groups[phase].append(result)
    sample('after-'+gid)
   if phase=='warm':
    deadline=time.monotonic()+30
    while not (run_dir/'warm-metadata.json').exists():
     assert proc.poll() is None,'child exited before warm metadata'
     if time.monotonic()>deadline:raise TimeoutError('warm metadata')
     time.sleep(.01)
    metadata=json.loads((run_dir/'warm-metadata.json').read_text());assert metadata['gates']['enabled'] is True and metadata['gates']['compiled_callable_count']==48 and metadata['norm_reuse']['enabled']==enabled and metadata['routes']['hooks_restored_before_measure'] is True
    from warm_routes import validate_metadata
    validate_metadata(metadata['routes'],enabled)
    assert metadata['parameter_tree_unchanged'] is True
    sample('after-three-warmup-groups-before-measurement')
  sample('after-measurement-before-unload')
  requests=[g['requests'][0] for phase in ['warm','measure'] for g in groups[phase]]
  ids=requests[0]['output_ids_when_available']
  assert len(ids)==32 and ids==w['continuation_ids'] and all(r['output_ids_when_available']==ids==r['visible_roundtrip_ids'] for r in requests),'native raw32 own/canonical mismatch'
  if 'ids' in shared:assert shared['ids']==ids,'native raw32 cross-six settings mismatch'
  shared['ids']=ids
  report={'schema':'v19-e2e-incremental-native-result-build2','config':label,'gates_compile_enabled':True,'norm_reuse_enabled':enabled,'concurrency':1,'warmup_groups':3,'repeat':3,'warmups':groups['warm'],'runs':groups['measure'],'runtime_source_identity':runtime_identity,'post_warm_metadata':metadata,'ready_s':ready_s,'model_dir':str(model),'checkpoint_revision':model.name,'prompt_ids':w['prompt_ids'],'config_sha256':sha(model/'config.json'),'quantization':json.loads((model/'config.json').read_text())['quantization'],'activation_dtype':'bfloat16','output_cap':32,'sampling':{'temperature':0,'EOS':'normal','draft':None},'scheduler_config':{'max_seqs':4,'max_len':2048,'max_batched_tokens':512},'prefix_cache':0,'boundary':'unchanged original client POST start through SSE DONE; cleanup excluded','x_definition':'median per request (32 - reconstructed first nonempty chunk token count)/(SSE DONE - first text); client delivery proxy','y_definition':'32 actual completion tokens / group POST-through-DONE window; queue/prefill/HTTP included','median_x_client_estimated_interactivity_tok_s_user':statistics.median(g['x_client_estimated_interactivity_tok_s_user'] for g in groups['measure']),'median_y_total_output_throughput_tok_s_mac':statistics.median(g['y_total_output_throughput_tok_s_mac'] for g in groups['measure']),'expected_power':power,'environment':boot.environment_record(env),'source_identity':verify_source(),'controller_sha256':sha(__file__),'child_sha256':sha(CHILD),'timed_instrumentation':False,'instrumentation_scope':'measurement has no added hooks; three warm requests have route validation hooks','scope':'V19 norm_reuse internal off/on effect with compiled gates ON; no official comparison or promotion; warm path hooks removed before measure'}
 except BaseException as exc:
  first_error=exc
  write_new(run_dir/'first-failure.json',{'type':type(exc).__name__,'error':str(exc),'traceback':traceback.format_exc(),'raw_groups_preserved':True})
 finally:
  if proc is not None:
   try:exit_record=cleanup_owned(proc);write_new(run_dir/'owned-process-exit.json',exit_record)
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
  write_new(run_dir/'result.json',report);verify_frozen()
  return {'config':label,'enabled':enabled,'report':str(run_dir/'result.json'),'report_sha256':sha(run_dir/'result.json'),'x':report['median_x_client_estimated_interactivity_tok_s_user'],'y':report['median_y_total_output_throughput_tok_s_mac'],'actual_completed_requests':6,'actual_completion_tokens':192}
