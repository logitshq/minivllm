"""Unhooked official oMLX none vs qualified external MTP C1 paired screen."""
import argparse
import ast
import concurrent.futures
import importlib.util
import json
import os
from pathlib import Path
import secrets
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time
import traceback
D=Path(__file__).resolve().parent;P=D.parent;OLD=P/'m4-final-native15-official-attempt2'
sys.dont_write_bytecode=True
sys.path.insert(0,str(OLD))
import common as controls
from support import verify,sha
from checks import inspect_log,summarize,owned_reap_state

def write_new(path,value):controls.write_new(path,value)

def run_arm(item,output,config,previous_env):
    import httpx
    from tokenizers import Tokenizer
    spec=importlib.util.spec_from_file_location('v111_pinned_official_client',OLD/'client.py');official=importlib.util.module_from_spec(spec);spec.loader.exec_module(official)
    mode=item['mode'];run_dir=output/item['label'];run_dir.mkdir();model=Path(config['target_model_dir'])
    tok=Tokenizer.from_file(str(model/'tokenizer.json'));prompt=tok.decode(config['prompt_ids'],skip_special_tokens=False)
    assert tok.encode(prompt,add_special_tokens=False).ids==config['prompt_ids']
    with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    key='benchmark-'+secrets.token_urlsafe(24);env=controls.child_environment(os.environ)
    if previous_env:assert env==previous_env
    else:previous_env.update(env)
    assert not any(k.startswith(('DYLD_','OMLX_','METAL_')) for k in env)
    command=official.official_command('omlx',1,run_dir,port,key,model,env)
    assert str(OLD/'official_entry.py') in command and not any('v109' in x or 'observer' in x for x in command)
    settings={'mtp_enabled':False,'dflash_enabled':False,'specprefill_enabled':False,'vlm_mtp_enabled':mode=='mtp','vlm_mtp_draft_model':config['draft_model_dir'] if mode=='mtp' else None,'vlm_mtp_draft_block_size':None}
    write_new(run_dir/'state/model_settings.json',{'version':1,'models':{model.name:settings}})
    samples=[];proc=None;birth=None;log=None;error=None;cleanup=None;result=None;cleanup_errors=[]
    def sample(label):
        controls.probe(samples,run_dir/'resources.json',label,'AC',proc.pid if proc else None)
        assert samples[-1]['processinfo_thermal_metadata']['code']==0,'nominal thermal required'
    try:
        sample('before-model-launch')
        write_new(run_dir/'launch.json',{'pair':item['pair'],'mode':mode,'command':['<api-key>' if x==key else x for x in command],'environment':controls.boot.environment_record(env),'model_settings':settings,'official_cli_original_entry':True,'qualification_observers':False,'measurement_wrappers':False,'default_block_override':None})
        log=(run_dir/'server.log').open('x');proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env)
        write_new(run_dir/'owned-pid.json',{'pid':proc.pid,'ppid':os.getpid()})
        birth=controls.identity(proc.pid);assert birth is not None
        write_new(run_dir/'owned-birth.json',{'identity':birth})
        base=f'http://127.0.0.1:{port}';headers={'Authorization':'Bearer '+key};deadline=time.monotonic()+180
        with httpx.Client(headers=headers,trust_env=False,timeout=180) as client:
            while True:
                assert proc.poll() is None,'owned server exited before ready'
                try:
                    response=client.get(base+'/v1/models',timeout=1)
                    if response.status_code==200:break
                except httpx.HTTPError:pass
                if time.monotonic()>deadline:raise TimeoutError('server readiness')
                time.sleep(0.2)
        controls.exact_listener(proc.pid,port,run_dir/'server-ready-exact-listener.json');birth=controls.lock_owned_identity(proc,birth,'omlx',run_dir,'ready')
        pre=json.loads((run_dir/'official-preimport.json').read_text());assert pre['engine']=='omlx' and pre['pid']==proc.pid and pre['ppid']==os.getpid() and pre['mlx_not_imported']
        assert not pre.get('qualification_observers_added',False)
        name=response.json()['data'][0]['id'];assert name==model.name;sample('server-ready-before-warmup')
        path='/v1/completions';body={'model':name,'prompt':prompt,'temperature':0,'repetition_penalty':1.0,'max_tokens':32}
        a=argparse.Namespace(engine='omlx')
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
        groups={};log_proofs={}
        for phase in ('warm','measure'):
            groups[phase]=[]
            for index in range(3):
                gid=f'C1-{phase}-{index}';sample('before-'+gid)
                current=group(1,gid);write_new(run_dir/(gid+'.json'),current);groups[phase].append(current)
                assert current['requests'][0]['visible_roundtrip_ids']==config['qualified_control_raw32'],'visible32 differs from qualified fresh official control'
                sample('after-'+gid)
            count=3 if phase=='warm' else 6
            text=(run_dir/'server.log').read_text();proof=inspect_log(text,mode,count,config,str(run_dir/'models'/model.name))
            proof['server_log_sha256_at_check']=sha(run_dir/'server.log');log_proofs[phase]=proof;write_new(run_dir/('post-'+phase+'-log-proof.json'),proof)
            if phase=='warm':sample('after-three-warm-before-measure')
        sample('after-measure-before-unload')
        requests=[q for phase in ('warm','measure') for g in groups[phase] for q in g['requests']]
        assert len(requests)==6 and all(q['usage']['completion_tokens']==32 for q in requests)
        result={'label':item['label'],'pair':item['pair'],'mode':mode,'warmups':groups['warm'],'runs':groups['measure'],'x':statistics.median(g['x_client_estimated_interactivity_tok_s_user'] for g in groups['measure']),'y':statistics.median(g['y_total_output_throughput_tok_s_mac'] for g in groups['measure']),'ttft_s':statistics.median(q['client_ttft_text_s'] for g in groups['measure'] for q in g['requests']),'E2E_s':statistics.median(q['client_total_s'] for g in groups['measure'] for q in g['requests']),'log_participation':log_proofs,'official_runtime_identity':pre,'sampling':{'greedy':True,'EOS':'normal','max_tokens':32,'prefix_cache':False,'repetition_penalty':1.0},'source_config_sha256':sha(D/'config.json'),'measurement_observers':False,'raw_IDs_currently_available':all(q['output_ids_when_available'] is not None for q in requests),'visible32_matches_qualified_control':True,'numerical_qualification_receipt_sha256':config['qualification']['receipt_sha256'],'scope':'single official implementation none/MTP fixed C1 configuration screen, not native comparison/fullcurve/globalbest'}
    except BaseException as exc:
        error=exc;write_new(run_dir/'first-failure.json',{'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc(),'raw_preserved':True})
    finally:
        if proc is not None:
            try:
                birth=controls.lock_owned_identity(proc,birth,'omlx',run_dir,'pre-cleanup');cleanup=controls.cleanup_owned(proc,birth,run_dir/'cleanup-signal-identities.json');write_new(run_dir/'owned-process-exit.json',cleanup)
            except BaseException as exc:cleanup_errors.append({'phase':'owned-process','message':str(exc)})
        try:write_new(run_dir/'port-absence.json',controls.wait_port_absence(port))
        except BaseException as exc:cleanup_errors.append({'phase':'port','message':str(exc)})
        try:
            controls.probe(samples,run_dir/'resources.json','after-owned-unload','AC');assert samples[-1]['processinfo_thermal_metadata']['code']==0
        except BaseException as exc:cleanup_errors.append({'phase':'resource','message':str(exc)})
        if log is not None:log.close()
        write_new(run_dir/'cleanup.json',{'first_error':None if error is None else str(error),'cleanup_errors':cleanup_errors,'owned_exit':cleanup})
    if error is not None:raise error
    assert not cleanup_errors and result is not None,cleanup_errors
    result.update(resource_snapshots=samples,owned_exit=cleanup);write_new(run_dir/'result.json',result)
    return {k:result[k] for k in ('label','pair','mode','x','y','ttft_s','E2E_s')}

def main():
    ap=argparse.ArgumentParser(description=__doc__);mode=ap.add_mutually_exclusive_group(required=True);mode.add_argument('--inert',action='store_true');mode.add_argument('--run',action='store_true');ap.add_argument('--out',type=Path);ap.add_argument('--pins-sha256',required=True);args=ap.parse_args()
    config=verify(args.pins_sha256);assert not any(n=='mlx' or n.startswith(('mlx.','numpy')) for n in sys.modules)
    assert not controls.selected(os.environ) and sys.executable==controls.PYTHON
    if args.inert:
        print(json.dumps({'status':'PASS_CPU_INERT','plan':config['plan'],'fresh_servers':6,'HTTP_requests':36,'HTTP_tokens':1152,'measured_requests':18,'measured_tokens':576,'fixed_cooling_gaps':5,'fixed_cooling_seconds':20,'GPU_used':False}));return 0
    assert args.out and args.out.is_absolute();args.out.mkdir();write_new(D/'attempt.consumed.json',{'pid':os.getpid(),'pins_sha256':args.pins_sha256})
    completed=[];previous_env={};resources=[];coolings=[];error=None;completion=None
    def stop(sig,frame):raise KeyboardInterrupt('owned driver signal '+str(sig))
    previous=signal.signal(signal.SIGTERM,stop)
    try:
        controls.probe(resources,args.out/'owner-resources.json','before-campaign','AC');assert resources[-1]['processinfo_thermal_metadata']['code']==0
        for index,item in enumerate(config['plan']):
            print(json.dumps({'event':'ARM_STARTED',**item}),flush=True);completed.append(run_arm(item,args.out,config,previous_env));print(json.dumps({'event':'ARM_COMPLETED',**completed[-1]}),flush=True)
            if index<5:
                # Fixed20 seconds, only after run_arm has strictly reaped and unloaded.
                controls.probe(resources,args.out/'owner-resources.json',f'cooling-{index+1}-before','AC');assert resources[-1]['processinfo_thermal_metadata']['code']==0
                started=time.monotonic();time.sleep(20);ended=time.monotonic();coolings.append({'after_arm':item['label'],'fixed_seconds':20,'elapsed_seconds':ended-started,'outside_request_clocks':True,'adaptive':False})
                controls.probe(resources,args.out/'owner-resources.json',f'cooling-{index+1}-after','AC');assert resources[-1]['processinfo_thermal_metadata']['code']==0
                write_new(args.out/f'cooling-{index+1}.json',coolings[-1])
        assert len(completed)==6 and len(coolings)==5;verify(args.pins_sha256)
        completion={'status':'PASS_FIXED_C1_NONE_VS_QUALIFIED_EXTERNAL_MTP_SCREEN','arms':completed,**summarize(completed),'HTTP_requests':36,'HTTP_completion_tokens':1152,'measured_requests':18,'measured_tokens':576,'fixed_cooling':coolings,'fullfrontier_win':False,'global_best_claim':False,'adopted':False,'qualification_scope':'inheritsV109 actualraw32/participation; measurement uses unhooked originalCLI/visible32 plus nativeoperational logs, not perverifyrawcounter hooks'}
    except BaseException as exc:
        error=exc;write_new(args.out/'suite-failure.json',{'status':'FAIL_STOP','type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc(),'completed':completed,'coolings':coolings,'raw_preserved':True,'partial_pooling':False});traceback.print_exc()
    finally:
        signal.signal(signal.SIGTERM,previous)
        try:
            controls.probe(resources,args.out/'owner-resources.json','after-campaign','AC');assert resources[-1]['processinfo_thermal_metadata']['code']==0
        except BaseException as exc:
            if error is None:error=exc
            write_new(args.out/'postflight-failure.json',{'type':type(exc).__name__,'message':str(exc),'raw_preserved':True})
        ownership=owned_reap_state(args.out)
        write_new(args.out/'owner-cleanup.json',{'pid':os.getpid(),'completed_arms':len(completed),**ownership,'error':None if error is None else str(error),'driver_reaping_by_outerowner_required':True})
    if error is not None:return 1
    assert completion is not None;write_new(args.out/'suite-complete.json',completion);print(json.dumps({'event':'SUITE_COMPLETE','arm_medians':completion['arm_medians'],'paired_signed_medians':completion['paired_signed_medians'],'adopted':False}),flush=True);return 0

if __name__=='__main__':raise SystemExit(main())
