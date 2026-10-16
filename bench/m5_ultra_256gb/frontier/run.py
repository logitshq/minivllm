"""Fresh paired C1/C2/C4 screen of V19-on and official oMLX, on M5 Ultra only."""
import argparse,hashlib,importlib.metadata,json,os,secrets,socket,statistics,subprocess,sys,time,traceback
from pathlib import Path
D=Path(__file__).resolve().parent;REPO=D.parents[2];BASE=REPO/'.artifacts/m5-ultra-256gb';QUAL=BASE/'fullqual-build2';INPUTS=BASE/'frontier-inputs-v1';OUT=BASE/'frontier-v1'
HARNESS=BASE/'build2-harness';sys.path.insert(0,str(HARNESS))
import resources
from runtime import digest
from client import make_group
import httpx
from tokenizers import Tokenizer
PYTHON=REPO/'.venv/bin/python';OMLX=REPO/'.artifacts/omlx-baseline/.venv/bin/omlx'
ENV={'MLX_MAX_OPS_PER_BUFFER':'50','MLX_MAX_MB_PER_BUFFER':'50','MLX_BFS_MAX_WIDTH':'20','MLX_METAL_FAST_SYNCH':'0'}
SOURCE=BASE/'source-m5-decode-v2/v19'
CONFIGS=[(c,pair,engine) for c in [1,2,4] for pair in [1,2,3] for engine in (['tutorial','omlx'] if pair!=2 else ['omlx','tutorial'])]
def write(path,value):path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
def absence(pid):
 r=subprocess.run(['/bin/ps','-p',str(pid),'-o','pid=,ppid=,rss=,comm='],capture_output=True,text=True)
 row={'pid':pid,'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr};assert r.returncode==1 and not r.stdout and not r.stderr;return row
def snapshot(out,samples,label,pid=None):
 raw=resources.snapshot(label,pid);write(out/f'resource-{len(samples):02d}-raw.json',raw);samples.append(raw);resources.deltas(samples);resources.guard(samples)
 assert raw['pressure_level_value']==1 and raw['power_metadata']['source']=='AC Power' and raw['processinfo_thermal_metadata']['code'] in (0,1)
 if pid:assert raw['exact_owned_pid_metadata']['status']=='known'
 write(out/'resource-derived.json',samples)
def verify():
 f=json.loads((INPUTS/'frozen.json').read_text())
 for p,h in f['files'].items():assert digest(p)==h,'changed input '+p
 assert {k:v for k,v in os.environ.items() if k.startswith(('MLX_','METAL_','DYLD_','OMLX_')) or k in ('PYTHONPATH','PYTHONHOME','PYTHONDONTWRITEBYTECODE')}==ENV
 return f
def prepare():
 INPUTS.mkdir();f=json.loads((QUAL/'frozen.json').read_text());write(INPUTS/'source.json',{'files':f['candidate_sources']})
 # Distribution paths and native assets come from the actual installed official wheel.
 command=[str(OMLX.parent/'python'),'-c',"import importlib.metadata as m,json; from pathlib import Path; names=['omlx','mlx','mlx-metal','mlx-lm','mlx-vlm']; print(json.dumps({'versions':{n:m.version(n) for n in names},'files':[str(Path(m.distribution(n).locate_file(p)).resolve()) for n in names for p in m.distribution(n).files or [] if str(p).endswith(('.py','.so','.dylib','.metallib','METADATA'))]}))"]
 info=json.loads(subprocess.check_output(command,text=True));assert info['versions']['mlx']==info['versions']['mlx-metal']=='0.32.3' and info['versions']['omlx']=='0.7.1.dev1';write(INPUTS/'official-runtime.json',info)
 files=list(D.glob('*.py'))+[INPUTS/'source.json',INPUTS/'official-runtime.json',QUAL/'frozen.json',QUAL/'replay.json',QUAL/'GPU-RELEASE.json',QUAL/'workload.json']+[Path(p) for p in info['files']]
 files+=list((SOURCE/'tutorial_code/miniengine').rglob('*.py'))
 write(INPUTS/'frozen.json',{'files':{str(p):digest(p) for p in files if p.is_file()},'configs':CONFIGS,'environment':ENV,'fullqual_sha256':digest(QUAL/'frozen.json'),'scope':'three fresh paired processes per C; V19-on versus official oMLX balanced; no Lithos result'})
 print(info['versions'])
def run_one(c,pair,engine):
 out=OUT/f'C{c}-pair{pair}-{engine}';out.mkdir();w=json.loads((QUAL/'workload.json').read_text());model=Path(w['model_dir']);tok=Tokenizer.from_file(str(model/'tokenizer.json'));prompt=tok.decode(w['prompt_ids'],skip_special_tokens=False);assert tok.encode(prompt,add_special_tokens=False).ids==w['prompt_ids']
 with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
 key='benchmark-'+secrets.token_urlsafe(24);headers={'Authorization':'Bearer '+key};base=f'http://127.0.0.1:{port}';samples=[];proc=None;first=None;cleanup=[];report=None
 if engine=='tutorial':command=[str(PYTHON),'-B',str(D/'native_child.py'),'--model',str(model),'--port',str(port),'--batch','4','--trace',str(out/'engine-trace.json'),'--package-root',str(SOURCE),'--source-manifest',str(INPUTS/'source.json')]
 else:
  models=out/'models';models.mkdir();(models/'Qwen3.8-27B-nvfp4').symlink_to(model);state=out/'state';state.mkdir()
  write(state/'settings.json',{'server':{'burst_decode_mode':'balanced'},'scheduler':{'max_concurrent_requests':c,'chunked_prefill':False,'prefill_priority':'speed','decode_fairness':True},'cache':{'enabled':False}})
  command=[str(OMLX),'serve','--model-dir',str(models),'--base-path',str(state),'--host','127.0.0.1','--port',str(port),'--api-key',key,'--headless','--no-hf-cache','--no-cache','--max-concurrent-requests',str(c),'--sse-keepalive-mode','off']
 public_command=list(command)
 if '--api-key' in public_command:public_command[public_command.index('--api-key')+1]='<local-ephemeral-key>'
 write(out/'launch.json',{'command':public_command,'environment':ENV,'model':str(model),'concurrency':c,'pair':pair,'engine':engine})
 log=(out/'server.log').open('w')
 try:
  snapshot(out,samples,'before-launch');proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT);write(out/'owned-pid.json',{'pid':proc.pid});start=time.monotonic()
  with httpx.Client(headers=headers,trust_env=False,timeout=180) as client:
   while True:
    if proc.poll() is not None:raise RuntimeError('server exited before ready')
    try:
     response=client.get(base+'/v1/models',timeout=1)
     if response.status_code==200 and response.json().get('data'):break
    except httpx.HTTPError:pass
    if time.monotonic()-start>180:raise TimeoutError('readiness')
    time.sleep(.2)
  snapshot(out,samples,'ready-before-warm',proc.pid);group=make_group(engine,base,headers,prompt,tok,response.json()['data'][0]['id']);groups={}
  for phase in ('warm','measure'):
   groups[phase]=[]
   for i in range(3):
    gid=f'C{c}-{phase}-{i}';snapshot(out,samples,'before-'+gid,proc.pid);row=group(c,gid);write(out/(gid+'.json'),row);groups[phase].append(row);snapshot(out,samples,'after-'+gid,proc.pid)
  requests=[r for phase in groups.values() for g in phase for r in g['requests']]
  assert len(requests)==6*c and all(r['first_chunk_prefix_verified'] and r['full_visible_roundtrip_count_matches_usage'] for r in requests)
  report={'engine':engine,'pair':pair,'concurrency':c,'groups':groups,'x':statistics.median(g['x_client_estimated_interactivity_tok_s_user'] for g in groups['measure']),'y':statistics.median(g['y_total_output_throughput_tok_s_mac'] for g in groups['measure']),'TTFT_s':statistics.median(statistics.median(r['client_ttft_text_s'] for r in g['requests']) for g in groups['measure']),'actual_gpu_batch_scope':'native trace records scheduler batch after each step; official GPU batch not inferred from HTTP concurrency','official_raw_ids_available':False,'runtime_identity':response.json().get('benchmark_identity'),'environment':ENV,'expected_power':'AC','checkpoint_revision':model.name,'normal_EOS':True,'prefix_cache':0,'draft':None,'usage_and_canonical_count':32,'gates_compile':True if engine=='tutorial' else 'official','norm_reuse':True if engine=='tutorial' else 'official'}
 except BaseException as e:first={'type':type(e).__name__,'text':str(e),'traceback':traceback.format_exc()};write(out/'first-failure.json',first)
 finally:
  if proc:
   try:
    if proc.poll() is None:
     proc.terminate()
     try:proc.wait(timeout=15)
     except subprocess.TimeoutExpired:proc.kill();proc.wait(timeout=15)
    else:proc.wait()
    exited={'pid':proc.pid,'returncode':proc.returncode,'waited_reaped':True,'absence':absence(proc.pid)};write(out/'process-exit.json',exited)
   except BaseException as e:cleanup.append(str(e))
  log.close()
  try:snapshot(out,samples,'after-unload')
  except BaseException as e:cleanup.append(str(e))
  write(out/'cleanup.json',{'first_failure':first,'cleanup_errors':cleanup})
 if first or cleanup:raise RuntimeError('frontier stopped: '+str(first or cleanup))
 report['resources']=samples;report['process_exit']=exited;write(out/'result.json',report);return {'path':str(out/'result.json'),'sha256':digest(out/'result.json'),'engine':engine,'pair':pair,'C':c,'x':report['x'],'y':report['y'],'TTFT_s':report['TTFT_s']}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');a=ap.parse_args()
 if a.prepare:return prepare()
 verify();OUT.mkdir();write(OUT/'attempt.json',{'owner_pid':os.getpid(),'configs':CONFIGS,'frozen_sha256':digest(INPUTS/'frozen.json'),'single_attempt':True});rows=[]
 try:
  for c,pair,engine in CONFIGS:
   print(f'START C{c} pair{pair} {engine}',flush=True);verify();rows.append(run_one(c,pair,engine));write(OUT/'progress.json',rows);print(json.dumps(rows[-1]),flush=True)
  verify();write(OUT/'complete.json',{'rows':rows,'owner_pid':os.getpid(),'scope':'M5 Ultra only, three fresh pairs at each C1/C2/C4, no official raw IDs or Lithos score'})
 except BaseException as e:write(OUT/'failure.json',{'error':str(e),'traceback':traceback.format_exc(),'completed':rows});raise
if __name__=='__main__':main()
