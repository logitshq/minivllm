"""Root owned V120 C1 performance screen: two fresh services, frozen factory, no request-window probes."""
import argparse, hashlib, json, os, socket, subprocess, sys, time, traceback
from pathlib import Path
P=Path('/private/tmp/minivllm-engine-optimization-20261014')
D=P/'v120-spec-throughput-latency-prework-build1'
R=P/'v23-root-control-build1'
sys.path.insert(0,str(P/'m4-final-native15-official-attempt2'))
import common as c
sys.path.insert(0,str(D))
import adapter, client
ap=argparse.ArgumentParser();ap.add_argument('--out',required=True,type=Path);ap.add_argument('--modes',nargs='+',choices=('ordinary','speculative'),default=['ordinary','speculative']);a=ap.parse_args()
assert sys.executable=='/Users/ryanli/code/minivllm/.venv/bin/python'
assert not c.selected(os.environ)
assert a.out.is_absolute();a.out.mkdir()
binding=adapter.qualification(D/'qualification-binding.json')
pins={str(p):c.sha(p) for p in D.glob('*.py')}
c.write(a.out/'source-pins.json',pins)
c.write(a.out/'owner-receipt.json',{'owner_pid':os.getpid(),'stage':'running','modes':a.modes,'scope':'short C1 screening only; no frontend-wide or competing engine performance claim','sample_policy':'outside requests only; startup sampled every2seconds'})
results=[];owned=[];status='FAILED'
try:
 for index,mode in enumerate(a.modes):
  for p,h in pins.items():assert c.sha(p)==h,'frozen source changed'
  arm=a.out/('%02d-'%index+mode);arm.mkdir()
  samples=[];proc=None;birth=None;lf=None
  with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
  def sample(label):
   c.probe(samples,arm/'owner-resources.json',label,'AC',proc.pid if proc is not None and proc.poll() is None else None)
   assert samples[-1]['processinfo_thermal_metadata']['code']==0,'nominal thermal required'
  try:
   sample('before-service')
   command=[sys.executable,'-B',str(D/'server.py'),'--execute-gpu','--mode',mode,'--qualification-binding',str(D/'qualification-binding.json'),'--port',str(port),'--owner-pid',str(os.getpid()),'--run-dir',str(arm/'server')]
   env={**os.environ,**adapter.CONTROLS,'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false'}
   c.write(arm/'launch.json',{'command':command,'controls':adapter.CONTROLS,'source_pins':pins,'qualification_receipt_sha256':binding['receipt_sha256']})
   lf=(arm/'server.log').open('x');proc=subprocess.Popen(command,env=env,stdout=lf,stderr=subprocess.STDOUT)
   birth=c.identity(proc.pid);assert birth is not None and c.process_parent(proc.pid)['ppid']==os.getpid()
   c.write(arm/'owned-process.json',{'pid':proc.pid,'owner_pid':os.getpid(),'birth':birth})
   print(json.dumps({'phase':'loading','mode':mode,'owner_pid':os.getpid(),'server_pid':proc.pid,'output':str(arm)}),flush=True)
   import httpx
   started=time.monotonic();next_probe=0;base='http://127.0.0.1:'+str(port)
   with httpx.Client(trust_env=False,timeout=1) as h:
    while True:
     assert proc.poll() is None,'server exited'
     if time.monotonic()>=next_probe:sample('startup');next_probe=time.monotonic()+2
     try:
      resp=h.get(base+'/v1/models')
      if resp.status_code==200:break
     except httpx.HTTPError:pass
     failure=arm/'server/worker-failure.json'
     if failure.exists():raise RuntimeError(failure.read_text())
     assert time.monotonic()-started<240,'startup deadline'
     time.sleep(.2)
   c.exact_listener(proc.pid,port,arm/'exact-listener.json')
   sample('ready')
   print(json.dumps({'phase':'three-warm-three-measure','mode':mode}),flush=True)
   client.measure_arm(argparse.Namespace(qualification_binding=D/'qualification-binding.json',base_url=base,mode=mode,out=arm/'client',power_domain='AC Power',server_pid=proc.pid))
   result=json.loads((arm/'client/result.json').read_text());results.append({k:result[k] for k in ('status','mode','latency_p50_ms','total_output_tok_s','ttft_p50_ms')})
   sample('after-measure')
  finally:
   if proc is not None:
    cleanup=c.cleanup_owned(proc,birth,arm/'cleanup-signals.json');c.write(arm/'owned-process-exit.json',cleanup);owned.append(cleanup)
   if lf:lf.close()
   c.write(arm/'port-absence.json',c.wait_port_absence(port));sample('after-unload')
  if index+1<len(a.modes):time.sleep(20)
 status='COMPLETED_C1_SCREEN'
except BaseException as e:
 c.write(a.out/'failure.json',{'type':type(e).__name__,'error':str(e),'traceback':traceback.format_exc()})
 raise
finally:
 c.write(a.out/'owner-receipt.json',{'owner_pid':os.getpid(),'stage':'finished','status':status,'results':results,'owned_processes_reaped':owned,'scope':'internal C1 screen, same19→32 fixture; not external/full-frontier adoption'})
 print(json.dumps({'status':status,'results':results,'all_owned_reaped':len(owned)==len(a.modes)}),flush=True)
