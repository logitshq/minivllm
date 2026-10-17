"""Root-only new static C2/C4 entry qualification; no performance or fullcurve claim.

One fresh server per C;3warm groups prove real B>1 and remove route hooks, then
one fresh re-entry group. Old same-C output streams are assertions, never input.
"""
import argparse,ast,concurrent.futures,hashlib,importlib.util,json,os,secrets,signal,socket,statistics,subprocess,sys,threading,time,traceback
from pathlib import Path
D=Path(__file__).resolve().parent;P=D.parent;OLD=P/'m4-final-native15-official-attempt2'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,allow_nan=False)
def one_point(batch,output,controls,oldclient):
    import httpx
    from tokenizers import Tokenizer
    run_dir=output/f'C{batch}';run_dir.mkdir()
    w=json.loads((P/'workload.json').read_text());tok=Tokenizer.from_file(str(Path(w['model_dir'])/'tokenizer.json'))
    prompt=tok.decode(w['prompt_ids'],skip_special_tokens=False);assert tok.encode(prompt,add_special_tokens=False).ids==w['prompt_ids']
    allowed=json.loads((D/'old-same-C-ID-assertions.json').read_text())['variants'][str(batch)]
    with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    env=controls.child_environment(os.environ);assert not any(k.startswith(('DYLD_','METAL_')) for k in env)
    command=[sys.executable,'-B',str(D/'server.py'),'--enable-full-curve','--concurrency',str(batch),
        '--config',str(D/'profiles.json'),'--model',w['model_dir'],'--port',str(port),
        '--trace',str(run_dir/'engine-trace.json'),'--run-dir',str(run_dir),'--owner-pid',str(os.getpid())]
    samples=[];proc=None;birth=None;log=None;error=None;cleanup=None;result=None
    def sample(label):
        controls.probe(samples,run_dir/'resources.json',label,'AC',proc.pid if proc is not None and proc.poll() is None else None)
        assert samples[-1]['processinfo_thermal_metadata']['code']==0
    def check_ids(g):
        for q in g['requests']:
            ids=q['output_ids_when_available'];h=hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()
            assert allowed.get(h)==ids,('same-C strict output assertion failed',batch,h)
    base=f'http://127.0.0.1:{port}';headers={'Authorization':'Bearer '+secrets.token_urlsafe(24)}
    path='/v1/completions';a=argparse.Namespace(engine='tutorial');body={}
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

    def wait_trace(delivered):
        deadline=time.monotonic()+30
        while True:
            try:value=json.loads((run_dir/'engine-trace.json').read_text())
            except (FileNotFoundError,json.JSONDecodeError):value=None
            if value and sum(s['delivered_ids'] for s in value['steps'])==delivered:return value
            assert proc.poll() is None
            if time.monotonic()>deadline:raise TimeoutError('idle trace delivery frontier')
            time.sleep(.01)
    try:
        sample('before-launch');write(run_dir/'launch.json',{'command':command,'environment':{k:env[k] for k in ('MLX_MAX_OPS_PER_BUFFER','MLX_MAX_MB_PER_BUFFER','MLX_BFS_MAX_WIDTH','MLX_METAL_FAST_SYNCH')},'profile':'legacy_batched256','new_entry_qualification':True})
        log=(run_dir/'server.log').open('x');proc=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
        birth=controls.identity(proc.pid);assert birth is not None
        write(run_dir/'owned-pid.json',{'pid':proc.pid,'ppid':os.getpid(),'birth':birth})
        start=time.monotonic()
        with httpx.Client(headers=headers,trust_env=False,timeout=180) as client:
            while True:
                assert proc.poll() is None,'server exited before ready'
                try:
                    response=client.get(base+'/v1/models',timeout=1)
                    if response.status_code==200:break
                except httpx.HTTPError:pass
                if time.monotonic()-start>180:raise TimeoutError('batch server ready')
                time.sleep(.2)
        controls.exact_listener(proc.pid,port,run_dir/'server-ready-exact-listener.json')
        birth=controls.lock_owned_identity(proc,birth,'native',run_dir,'ready')
        identity=response.json()['benchmark_identity']
        assert identity['manifest_all_sources_match'] and identity['static_profile']=='legacy_batched256'
        assert identity['server_max_seqs']==4 and identity['required_client_concurrency']==batch
        assert identity['physical_capacity_steps']==[256,256] and identity['native_adapter_installed'] is False
        assert identity['dynamic_native_batch_migration'] is False and identity['parameter_tree_array_count']==1349
        body.update(model=response.json()['data'][0]['id'],prompt=prompt,temperature=0,repetition_penalty=1.0,max_tokens=32)
        sample('ready-before-warm')
        groups=[]
        for i in range(3):
            sample(f'before-warm-{i}');g=group(batch,f'C{batch}-warm-{i}');check_ids(g);groups.append(g)
            write(run_dir/f'warm-{i}.json',g);sample(f'after-warm-{i}')
        deadline=time.monotonic()+30
        while not(run_dir/'warm-metadata.json').exists():
            assert proc.poll() is None
            if time.monotonic()>deadline:raise TimeoutError('batch warm metadata')
            time.sleep(.01)
        metadata=json.loads((run_dir/'warm-metadata.json').read_text());oldclient.validate_warm(metadata,batch)
        counts={int(k):v for k,v in metadata['routes']['decode_batch_counts'].items()}
        assert counts.get(batch,0)>0,'HTTP concurrency without actual requested batch'
        warm_trace=wait_trace(3*batch*32);prefix_steps=len(warm_trace['steps'])
        sample('before-post-hook-reentry');g=group(batch,f'C{batch}-post-hook-reentry');check_ids(g);groups.append(g)
        write(run_dir/'post-hook-reentry.json',g);sample('after-post-hook-reentry')
        trace=wait_trace(4*batch*32);reentry_steps=trace['steps'][prefix_steps:]
        assert reentry_steps and max(s['active_decode_batch'] for s in reentry_steps)==batch
        result={'concurrency':batch,'profile':'legacy_batched256','HTTP_requests':4*batch,'completion_tokens':128*batch,
            'actual_requested_batch_GDN_calls':counts[batch],'weighted_warm_GDN_rows':sum(k*v for k,v in counts.items()),
            'warm_hooks_removed_before_reentry':True,'post_hook_reentry_actual_requested_batch':True,
            'reentry_trace_prefix_steps':prefix_steps,'reentry_trace_steps':len(reentry_steps),
            'raw32_same_C_reference_and_visible_usage_exact':True,
            'idle_source_assertions_completed':True,'parameter_tree_identity_unchanged':metadata['parameter_tree_unchanged'],
            'runtime_identity':identity,'groups':groups,'performance_point':False,'full_tensor_numerical_recompute':False}
    except BaseException as exc:
        error=exc;write(run_dir/'first-failure.json',{'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc(),'raw_preserved':True})
    finally:
        failures=[]
        if proc is not None:
            try:
                birth=controls.lock_owned_identity(proc,birth,'native',run_dir,'pre-cleanup')
                cleanup=controls.cleanup_owned(proc,birth,run_dir/'cleanup-signal-identities.json')
                write(run_dir/'owned-process-exit.json',cleanup)
            except BaseException as exc:failures.append({'phase':'owned-cleanup','message':str(exc)})
        try:write(run_dir/'port-absence.json',controls.wait_port_absence(port))
        except BaseException as exc:failures.append({'phase':'port','message':str(exc)})
        try:sample('after-unload')
        except BaseException as exc:failures.append({'phase':'resource','message':str(exc)})
        if log is not None:log.close()
        write(run_dir/'cleanup.json',{'error':None if error is None else str(error),'failures':failures,'owned_exit':cleanup})
    if error is not None:raise error
    assert not failures,failures
    result['owned_exit']=cleanup;result['resources']=samples;write(run_dir/'receipt.json',result)
    return result

def main():
    ap=argparse.ArgumentParser(description=__doc__);mode=ap.add_mutually_exclusive_group(required=True)
    mode.add_argument('--inert',action='store_true');mode.add_argument('--execute-gpu',action='store_true')
    ap.add_argument('--out',type=Path);ap.add_argument('--pins-sha256',required=True);args=ap.parse_args()
    assert sha(D/'source-pins.json')==args.pins_sha256
    for path,h in json.loads((D/'source-pins.json').read_text())['files'].items():assert sha(path)==h,path
    assert not any(n=='mlx' or n.startswith('mlx.') for n in sys.modules)
    if args.inert:
        print(json.dumps({'status':'PASS_INERT_V126_ENTRY_QUALIFICATION_GPU_UNRUN','points':[2,4],'fresh_loads':2,'HTTP_requests':24,'completion_tokens':768,'GPU_used':False}));return 0
    assert args.out is not None and args.out.is_absolute()
    sys.path.insert(0,str(OLD));import common as controls
    spec=importlib.util.spec_from_file_location('v126_old_client_validators',OLD/'client.py');oldclient=importlib.util.module_from_spec(spec);spec.loader.exec_module(oldclient)
    assert sys.executable==controls.PYTHON and not controls.selected(os.environ)
    args.out.mkdir();write(D/'qualification.attempt.consumed.json',{'pid':os.getpid(),'pins_sha256':args.pins_sha256,'no_retry':True})
    def stop(sig,frame):raise KeyboardInterrupt('owner SIGTERM')
    previous=signal.signal(signal.SIGTERM,stop)
    completed=[]
    try:
        for c in (2,4):completed.append(one_point(c,args.out,controls,oldclient))
        assert sum(x['HTTP_requests'] for x in completed)==24 and sum(x['completion_tokens'] for x in completed)==768
        for path,h in json.loads((D/'source-pins.json').read_text())['files'].items():assert sha(path)==h,path
        write(args.out/'receipt.json',{'status':'PASS_V126_STATIC_BATCH_ENTRY_C2_C4_GENUINE_BATCH_RAW_IDS_AND_IDLE',
            'points':completed,'HTTP_requests':24,'completion_tokens':768,'performance_point':False,
            'scope':'new entry route/old same-C IDs/ordinary lifecycle only; unchanged accepted25 math inherited, no new full128 tensor replay',
            'GPU_released':False,'external_owner_reap_and_absence_required':True})
    except BaseException as exc:
        write(args.out/'suite-failure.json',{'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc(),'completed_points':len(completed),'raw_preserved':True});raise
    finally:signal.signal(signal.SIGTERM,previous)
    return 0
if __name__=='__main__':raise SystemExit(main())
