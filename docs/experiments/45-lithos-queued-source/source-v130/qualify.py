"""Root-only official Lithos qualification: OpenAI C1, Ollama C1/C2/C4. No retry."""
import argparse,ast,hashlib,importlib.util,json,os,secrets,signal,socket,subprocess,sys,time
from pathlib import Path
D=Path(__file__).resolve().parent;P=D.parent;OLD=P/'m4-final-native15-official-attempt2'
sys.path.insert(0,str(OLD));import common as control
spec=importlib.util.spec_from_file_location('v130_unchanged_official_client',OLD/'client.py');official=importlib.util.module_from_spec(spec);spec.loader.exec_module(official)
sys.path.insert(0,str(D));from client import run_group

def verify_pins(pins_sha256):
 assert control.sha(D/'pins.json')==pins_sha256
 pins=json.loads((D/'pins.json').read_text())
 for path,h in pins['files'].items():assert control.sha(path)==h,path
 return pins

def run_fresh(out,power,pins_sha256,screen_C=None):
 pins=verify_pins(pins_sha256)
 if screen_C is None:
  plan=[('calibrate-openai-C1','openai',1),('qualify-ollama-C1','ollama',1),('qualify-ollama-C2','ollama',2),('qualify-ollama-C4','ollama',4)]
 else:
  assert type(screen_C)is int and screen_C in (1,2,4)
  plan=[(f'C{screen_C}-{phase}-{i}','ollama',screen_C) for phase in ('warm','measure') for i in range(3)]
 assert out and out.is_absolute();O=out;O.mkdir()
 if screen_C is None:control.write_new(D/'qualification.attempt.consumed.json',{'out':str(O),'fixed_power':power,'time':time.time()})
 import httpx
 from tokenizers import Tokenizer
 w=json.loads((P/'workload.json').read_text());model=Path(w['model_dir']);tok=Tokenizer.from_file(str(model/'tokenizer.json'))
 ref=json.loads(Path(pins['visible_gold_source']).read_text())['requests'][0]['visible_roundtrip_ids'];assert len(ref)==32
 with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
 key='benchmark-'+secrets.token_urlsafe(24);env=control.child_environment(os.environ)
 command=official.official_command('lithos',1,O,port,key,model,env)
 samples=[];proc=None;birth=None;lf=None;failure=None;cleanup=None;groups=[];cleanup_errors=[]
 def sample(label):
  control.probe(samples,O/'resources.json',label,power,proc.pid if proc else None)
  assert samples[-1]['processinfo_thermal_metadata']['code']==0
 try:
  sample('before-launch');lf=(O/'server.log').open('x');proc=subprocess.Popen(command,env=env,stdout=lf,stderr=subprocess.STDOUT);birth=control.identity(proc.pid);assert birth is not None
  control.write_new(O/'owned-pid.json',{'pid':proc.pid,'ppid':os.getpid()});control.write_new(O/'owned-birth.json',{'identity':birth})
  base=f'http://127.0.0.1:{port}';deadline=time.monotonic()+180
  with httpx.Client(headers={'Authorization':'Bearer '+key},trust_env=False,timeout=1) as c:
   while True:
    assert proc.poll() is None
    try:
     response=c.get(base+'/v1/models')
     if response.status_code==200:break
    except httpx.HTTPError:pass
    if time.monotonic()>deadline:raise TimeoutError('official Lithos ready')
    time.sleep(.2)
  name=response.json()['data'][0]['id'];control.exact_listener(proc.pid,port,O/'server-ready-exact-listener.json');birth=control.lock_owned_identity(proc,birth,'lithos',O,'ready');sample('ready')
  for label,protocol,C in plan:
   sample('before-'+label);g=run_group(base,name,key,C,protocol,tok,label,ref);control.write_new(O/(label+'.json'),g);groups.append(g);sample('after-'+label)
   assert g['all_correctness_checks_passed'],label+' mismatch/rejection retained'
   assert g['all_requested_POSTs_before_first_terminal'],label+' requested concurrency not achieved'
  # Actual backend participation/cache is observed from original logs only outside all groups.
  text=(O/'server.log').read_text();assert 'Serving backend=m4_max,' in text and 'draft=None' in text and 'backend=m5' not in text
  metrics=[ast.literal_eval(line.split('Generation: ',1)[1]) for line in text.splitlines() if 'Generation: ' in line]
  expected=sum(C for _,_,C in plan);assert len(metrics)==expected and all(x['cached_prompt_tokens']==0 and x['prefix_source'] is None for x in metrics)
  control.write_new(O/'participation.json',{'backend':'m4_max','draft':None,'original_log_sha256':control.sha(O/'server.log'),'Generation_records':metrics,'generation_count':expected,'cached_prompt_tokens0_count':expected,'prefix_source_None_count':expected,'GPU_batch_observed':None,'queue_policy':'official Ollama FifoLock ticket queue, not GPU batching','HTTP_arrival_fairness_claim':False})
 except BaseException as exc:
  failure={'type':type(exc).__name__,'message':str(exc)};control.write_new(O/'failure.json',failure)
 finally:
  if proc is not None:
   try:
    birth=control.lock_owned_identity(proc,birth,'lithos',O,'pre-cleanup');cleanup=control.cleanup_owned(proc,birth,O/'cleanup-signal-identities.json')
   except BaseException as exc:cleanup_errors.append({'phase':'owned','type':type(exc).__name__,'message':str(exc)})
  if lf is not None:lf.close()
  try:control.write_new(O/'port-absence.json',control.wait_port_absence(port))
  except BaseException as exc:cleanup_errors.append({'phase':'port','type':type(exc).__name__,'message':str(exc)})
  try:
   control.probe(samples,O/'resources.json','after-owned-unload',power);assert samples[-1]['processinfo_thermal_metadata']['code']==0
  except BaseException as exc:cleanup_errors.append({'phase':'post-resource','type':type(exc).__name__,'message':str(exc)})
  control.write_new(O/'cleanup.json',{'failure':failure,'cleanup_errors':cleanup_errors,'owned_exit':cleanup})
 if failure or cleanup_errors:raise RuntimeError('qualification STOP: '+str(failure or cleanup_errors))
 status='PASS_V130_OFFICIAL_OLLAMA_QUEUE_C1_C2_C4_OUTPUT_AND_CALIBRATION' if screen_C is None else 'PASS_V130_FRESH_OLLAMA_QUEUED_ARM'
 receipt={'status':status,'fixed_power':power,'HTTP_requests':sum(C for _,_,C in plan),'completion_tokens':32*sum(C for _,_,C in plan),'groups':[{k:g[k] for k in ('group_id','protocol','requested_concurrency','successful_requests','all_requested_POSTs_before_first_terminal','latency_p50_ms','total_throughput_tok_s','all_correctness_checks_passed')} for g in groups],'source_pins_sha256':pins_sha256,'gold_is_saved_visible32_not_internal_raw':True,'no_automatic_retry':True,'no_new_performance_point':True,'custom_queue':False,'GPU_batch_claim':False,'HTTP_arrival_fairness_claim':False,'old_429_raw_observed':False,'owned_exit':cleanup}
 control.write_new(O/'receipt.json',receipt)
 return receipt

def main():
 ap=argparse.ArgumentParser(description=__doc__);m=ap.add_mutually_exclusive_group(required=True);m.add_argument('--cpu-dry-run',action='store_true');m.add_argument('--run',action='store_true');ap.add_argument('--out',type=Path);ap.add_argument('--pins-sha256',required=True);ap.add_argument('--power',choices=['AC','Battery'],required=True);a=ap.parse_args()
 verify_pins(a.pins_sha256)
 if a.cpu_dry_run:print(json.dumps({'status':'CPU_READY_GPU_UNRUN','HTTP_requests':8,'outputs_if_all32':256,'fixed_power':a.power,'groups':['OpenAI C1 calibration','Ollama C1','Ollama C2','Ollama C4'],'custom_queue':False,'retries':0}));return
 receipt=run_fresh(a.out,a.power,a.pins_sha256)
 print(json.dumps({'status':receipt['status'],'out':str(a.out)}))

if __name__=='__main__':
 signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(KeyboardInterrupt('owner SIGTERM cleanup')))
 main()
