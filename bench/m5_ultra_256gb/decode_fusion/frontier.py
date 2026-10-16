"""Fresh-process validation of projection/conv fusion against the accepted graph and external baselines."""
import argparse,hashlib,importlib.metadata,json,os,secrets,signal,socket,statistics,subprocess,sys,time,traceback
from pathlib import Path
D=Path(__file__).resolve().parent;REPO=D.parents[2];BASE=REPO/'.artifacts/m5-ultra-256gb';QUAL=BASE/'fullqual-build2';INPUTS=BASE/'fusion-inputs-v1';OUT=BASE/'fusion-frontier-v1'
HARNESS=BASE/'build2-harness';sys.path.insert(0,str(HARNESS))
import resources
from runtime import digest
sys.path.insert(0,str(D.parent/"lithos"))
from client import make_group
import httpx
from tokenizers import Tokenizer
PYTHON=REPO/'.venv/bin/python';OMLX=REPO/'.artifacts/omlx-baseline/.venv/bin/omlx'
ENV={'MLX_MAX_OPS_PER_BUFFER':'50','MLX_MAX_MB_PER_BUFFER':'50','MLX_BFS_MAX_WIDTH':'20','MLX_METAL_FAST_SYNCH':'0'}
SOURCE=BASE/'fusion-source-v1'
BASE_SOURCE=BASE/'graph-source-v1'
BASE_MANIFEST=BASE/'graph-inputs-v1/source.json'
LITHOS=Path('/Users/ryanli/code/lithos-metal')
QUAL_LITHOS=BASE/'lithos-adaptation-v1/kernel-qualification.json'
CONFIGS=[(1,p,e) for p,order in enumerate([['base','fused','omlx','lithos'],['lithos','omlx','fused','base'],['fused','base','lithos','omlx']],1) for e in order]+[(c,p,e) for c in (2,4) for p,order in enumerate([['fused','omlx'],['omlx','fused'],['fused','omlx']],1) for e in order]
def write(path,value):path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
def absence(pid):
 r=subprocess.run(['/bin/ps','-p',str(pid),'-o','pid=,ppid=,rss=,comm='],capture_output=True,text=True)
 row={'pid':pid,'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr};assert r.returncode==1 and not r.stdout and not r.stderr;return row
def snapshot(out,samples,label,pid=None):
 raw=resources.snapshot(label,pid);write(out/f'resource-{len(samples):02d}-raw.json',raw);samples.append(raw);resources.deltas(samples);resources.guard(samples)
 assert raw['pressure_level_value']==1 and raw['power_metadata']['source']=='AC Power' and raw['processinfo_thermal_metadata']['code'] in (0,1)
 if pid:assert raw['exact_owned_pid_metadata']['status']=='known'
 state=Path('/Users/ryanli/Documents/Codex/2026-10-15/w/outputs/local-model-comparison.json')
 if state.exists():assert not json.loads(state.read_text())['status'].startswith('benchmarking'),'Another chat began GPU benchmarking'
 write(out/'resource-derived.json',samples)
def verify():
 f=json.loads((INPUTS/'frozen.json').read_text())
 for p,h in f['files'].items():assert digest(p)==h,'changed input '+p
 assert {k:v for k,v in os.environ.items() if k.startswith(('MLX_','METAL_','DYLD_','OMLX_')) or k in ('PYTHONPATH','PYTHONHOME','PYTHONDONTWRITEBYTECODE')}==ENV
 return f
def prepare():
 import shutil
 assert json.loads((BASE/'fusion-qual-v3/complete.json').read_text())['status']=='PASS'
 assert json.loads((BASE/'fusion-kernel-v1/complete.json').read_text())['status']=='PASS'
 INPUTS.mkdir();SOURCE.mkdir()
 shutil.copytree(REPO/'tutorial_code/miniengine',SOURCE/'tutorial_code/miniengine',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
 source_files={str(p.relative_to(SOURCE/'tutorial_code/miniengine')):digest(p) for p in (SOURCE/'tutorial_code/miniengine').rglob('*.py')}
 write(INPUTS/'source.json',{'files':source_files})
 # Reverify all prior official runtime/native assets; never repin modified baselines.
 old=json.loads((BASE/'lithos-inputs-v1/frozen.json').read_text())['files']
 for p,h in old.items():assert digest(p)==h,p
 files=list(D.glob('*.py'))+list(SOURCE.rglob('*.py'))+[INPUTS/'source.json',BASE/'fusion-qual-v3/complete.json',BASE/'fusion-qual-v3/checks.json',BASE/'fusion-qual-v3/scheduler.json']
 files += list((REPO/'tutorial_code/miniengine').rglob('*.py')) + list(BASE_SOURCE.rglob('*.py')) + [BASE_MANIFEST, BASE/'fusion-qual-v3/parameters.json',BASE/'fusion-kernel-v1/complete.json',BASE/'fusion-kernel-v1/checks.json',BASE/'fusion-free-generation-v1/complete.json']
 frozen={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),'files':dict(old,**{str(p):digest(p) for p in files}),'configs':CONFIGS,'environment':ENV,'scope':'projection packing + fused GDN preparation vs accepted whole decode graph; both pool2 and gates on; standalone V19 off; no prompt cache or draft; C1/C2/C4, first 256 positions only'}
 write(INPUTS/'frozen.json',frozen)
 print('PREPARED',len(frozen['files']))
def run_one(c,pair,engine):
 out=OUT/f'C{c}-pair{pair}-{engine}';out.mkdir();w=json.loads((QUAL/'workload.json').read_text());model=Path(w['model_dir']);tok=Tokenizer.from_file(str(model/'tokenizer.json'));prompt=tok.decode(w['prompt_ids'],skip_special_tokens=False);assert tok.encode(prompt,add_special_tokens=False).ids==w['prompt_ids']
 with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
 key='benchmark-'+secrets.token_urlsafe(24);headers={'Authorization':'Bearer '+key};base=f'http://127.0.0.1:{port}';samples=[];proc=None;first=None;cleanup=[];report=None
 if engine in ('base','fused'):command=[str(PYTHON),'-B',str(D/'native_child.py'),'--model',str(model),'--port',str(port),'--batch','4','--trace',str(out/'engine-trace.json'),'--package-root',str(SOURCE if engine=='fused' else BASE_SOURCE),'--source-manifest',str(INPUTS/'source.json' if engine=='fused' else BASE_MANIFEST),'--pool','2','--graph']+(['--fuse'] if engine=='fused' else [])
 elif engine=='omlx':
  models=out/'models';models.mkdir();(models/'Qwen3.8-27B-nvfp4').symlink_to(model);state=out/'state';state.mkdir()
  write(state/'settings.json',{'server':{'burst_decode_mode':'balanced'},'scheduler':{'max_concurrent_requests':c,'chunked_prefill':False,'prefill_priority':'speed','decode_fairness':True},'cache':{'enabled':False}})
  command=[str(OMLX),'serve','--model-dir',str(models),'--base-path',str(state),'--host','127.0.0.1','--port',str(port),'--api-key',key,'--headless','--no-hf-cache','--no-cache','--max-concurrent-requests',str(c),'--sse-keepalive-mode','off']
 else:
  assert engine=='lithos' and c==1,'Lithos OpenAI server rejects simultaneous requests with 429'
  command=[str(LITHOS/'.venv/bin/python'),'-B','-m','monolith.serve','--model',str(model),'--no-draft','--pack',str(BASE/'lithos-adaptation-v1/packs'),'--local-files-only','--host','127.0.0.1','--port',str(port),'--max-context','2048','--keep-warm','0','--no-warmup']
 child_env=os.environ.copy()
 if engine=='lithos':child_env.update({'LITHOS_METAL_API_KEY':key,'MONOLITH_ACCELERATOR':'on','HF_HUB_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false'})
 public_command=list(command)
 if '--api-key' in public_command:public_command[public_command.index('--api-key')+1]='<local-ephemeral-key>'
 write(out/'launch.json',{'command':public_command,'environment':dict(ENV,**({'MONOLITH_ACCELERATOR':'on'} if engine=='lithos' else {})),'model':str(model),'concurrency':c,'pair':pair,'engine':engine})
 log=(out/'server.log').open('w')
 try:
  snapshot(out,samples,'before-launch');proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,cwd=LITHOS if engine=='lithos' else REPO,env=child_env);write(out/'owned-pid.json',{'pid':proc.pid});start=time.monotonic();last_probe=start
  with httpx.Client(headers=headers,trust_env=False,timeout=180) as client:
   while True:
    if proc.poll() is not None:raise RuntimeError('server exited before ready')
    try:
     response=client.get(base+'/v1/models',timeout=1)
     if response.status_code==200 and response.json().get('data'):break
    except httpx.HTTPError:pass
    if time.monotonic()-start>900:raise TimeoutError('readiness')
    if time.monotonic()-last_probe>10:snapshot(out,samples,'readiness-wait',proc.pid);last_probe=time.monotonic()
    time.sleep(.2)
  snapshot(out,samples,'ready-before-warm',proc.pid);group=make_group('tutorial' if engine in ('base','fused') else engine,base,headers,prompt,tok,response.json()['data'][0]['id']);groups={}
  for phase in ('warm','measure'):
   groups[phase]=[]
   for i in range(3):
    gid=f'C{c}-{phase}-{i}';snapshot(out,samples,'before-'+gid,proc.pid);row=group(c,gid);write(out/(gid+'.json'),row);groups[phase].append(row);snapshot(out,samples,'after-'+gid,proc.pid)
  requests=[r for phase in groups.values() for g in phase for r in g['requests']]
  assert len(requests)==6*c and all(r['first_chunk_prefix_verified'] and r['full_visible_roundtrip_count_matches_usage'] for r in requests)
  lithos_metrics=[]
  if engine=='lithos':
   import ast
   lithos_metrics=[ast.literal_eval(line.split('Generation: ',1)[1]) for line in (out/'server.log').read_text().splitlines() if 'Generation: ' in line]
   assert len(lithos_metrics)==6,lithos_metrics
   assert all(m['cached_prompt_tokens']==0 and m['verify_tokens']==1 for m in lithos_metrics)
   assert all(m['setup_ms']==0 for m in lithos_metrics[3:])
  report={'lithos_native_metrics':lithos_metrics,'lithos_profile':'apple-m5-ultra-80c' if engine=='lithos' else None,'engine':engine,'pair':pair,'concurrency':c,'groups':groups,'x':statistics.median(g['x_client_estimated_interactivity_tok_s_user'] for g in groups['measure']),'y':statistics.median(g['y_total_output_throughput_tok_s_mac'] for g in groups['measure']),'TTFT_s':statistics.median(statistics.median(r['client_ttft_text_s'] for r in g['requests']) for g in groups['measure']),'actual_gpu_batch_scope':'native trace records scheduler batch after each step; official GPU batch not inferred from HTTP concurrency','official_raw_ids_available':False,'runtime_identity':response.json().get('benchmark_identity'),'environment':ENV,'expected_power':'AC','checkpoint_revision':model.name,'normal_EOS':True,'prefix_cache':0,'draft':None,'usage_and_canonical_count':32,'gates_compile':True if engine in ('base','fused') else 'official','norm_reuse':False if engine in ('base','fused') else 'official','compile_decode':engine in ('base','fused'),'fuse_decode':engine=='fused','allocator_pool_gib':2 if engine in ('base','fused') else 'official'}
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
 def stop(signum,frame):raise KeyboardInterrupt('owned benchmark signal '+str(signum))
 signal.signal(signal.SIGTERM,stop)
 ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');a=ap.parse_args()
 if a.prepare:return prepare()
 verify();OUT.mkdir();write(OUT/'attempt.json',{'owner_pid':os.getpid(),'configs':CONFIGS,'frozen_sha256':digest(INPUTS/'frozen.json'),'single_attempt':True});rows=[]
 try:
  for c,pair,engine in CONFIGS:
   print(f'START C{c} pair{pair} {engine}',flush=True);verify();rows.append(run_one(c,pair,engine));write(OUT/'progress.json',rows);print(json.dumps(rows[-1]),flush=True)
  verify();write(OUT/'complete.json',{'rows':rows,'owner_pid':os.getpid(),'scope':'M5 Ultra projection/conv fusion fresh comparison; C1 three processes each; C2/C4 candidate and official oMLX; no prompt reuse/draft; native raw IDs only'})
 except BaseException as e:write(OUT/'failure.json',{'error':str(e),'traceback':traceback.format_exc(),'completed':rows});raise
if __name__=='__main__':main()
